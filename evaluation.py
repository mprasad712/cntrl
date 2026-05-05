"""
Evaluation API Endpoints - Enhanced Version

Features:
- User isolation (only access own scores/traces)
- Integration with existing observability
- LLM-as-a-Judge with proper trace fetching
- Proper error handling
"""

import os
import json
import asyncio
import time
import re
import csv
import io
from threading import Lock, Thread
from datetime import datetime, timezone, timedelta
from typing import Annotated, Any, List, Optional, Dict, Union
from collections import defaultdict
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, UploadFile, File
from loguru import logger
from pydantic import BaseModel, Field
from sqlmodel import select
from sqlalchemy import and_, or_, true
from agentcore.services.deps import session_scope
# `agent` objects are stored as `Agent` in the DB; import AccessTypeEnum and
# alias `Agent` to `agent` so the rest of the module can keep using `agent`.
from agentcore.services.database.models.agent.model import AccessTypeEnum, Agent as agent
from agentcore.services.database.models.agent_deployment_prod.model import (
    AgentDeploymentProd,
    DeploymentPRODStatusEnum,
    ProdDeploymentVisibilityEnum,
)
from agentcore.services.database.models.agent_deployment_uat.model import (
    AgentDeploymentUAT,
    DeploymentUATStatusEnum,
)
from agentcore.services.database.models.agent_publish_recipient.model import (
    AgentPublishRecipient,
)

from agentcore.services.auth.utils import get_current_active_user
from agentcore.services.auth.permissions import normalize_role
from agentcore.services.database.models.user.model import User
from agentcore.services.database.models.department.model import Department
from agentcore.services.database.models.organization.model import Organization
from agentcore.services.database.models.user_department_membership.model import UserDepartmentMembership
from agentcore.services.database.models.user_organization_membership.model import UserOrganizationMembership
from agentcore.api.utils import DbSession
from agentcore.api.observability import fetch_traces_from_langfuse, fetch_scores_for_trace
from agentcore.api.observability.parsing import extract_trace_user_ids
from agentcore.services.observability.rbac import resolve_observability_scope
from agentcore.services.database.models.dataset.model import Dataset
from agentcore.services.database.models.dataset_item.model import DatasetItem
from agentcore.services.database.models.dataset_run.model import DatasetRun
from agentcore.services.database.models.dataset_run_item.model import DatasetRunItem
from sqlalchemy import func, delete as sa_delete

# Try importing litellm for the judge
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    LITELLM_AVAILABLE = False
    logger.debug("LiteLLM not installed. Judge will use Model Service or OpenAI SDK.")

# Try importing OpenAI as a fallback for the judge when LiteLLM isn't present
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_AVAILABLE = False

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])

_LITELLM_STD_LOGGING_PATCHED = False
_DATASET_EXPERIMENT_JOBS: dict[str, dict[str, Any]] = {}
_DATASET_EXPERIMENT_JOBS_LOCK = Lock()
_SCORE_LIST_CACHE: dict[str, dict[str, Any]] = {}
_SCORE_LIST_CACHE_STALE_SECONDS = 90.0
# Pending-reviews response cache (per user_id)
_PENDING_REVIEWS_CACHE: dict[str, dict[str, Any]] = {}
_PENDING_REVIEWS_CACHE_TTL_SECONDS = 30.0
# Dataset list response cache (per user_id)
_DATASETS_LIST_CACHE: dict[str, dict[str, Any]] = {}
_DATASETS_LIST_CACHE_TTL_SECONDS = 60.0

# Persistent evaluator configs stored in the database (see Evaluator model)
from agentcore.services.database.models.evaluator.model import Evaluator  # noqa: E402
from agentcore.services.model_registry_service import get_decrypted_config as get_model_decrypted_config  # noqa: E402


async def _resolve_model_from_registry(model_registry_id: str, session: Any = None) -> tuple[str, str | None, str | None]:
    """Resolve model_name, decrypted api_key, and api_base from the model registry.

    Returns (model_name, api_key, api_base) or raises HTTPException if not found.
    If *session* is provided it is reused; otherwise a fresh session_scope is opened.
    """
    if not model_registry_id or not str(model_registry_id).strip():
        raise HTTPException(status_code=400, detail="model_registry_id is required but was empty")

    try:
        model_uuid = UUID(str(model_registry_id).strip())
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid model_registry_id: {model_registry_id}")

    try:
        if session is not None:
            config = await get_model_decrypted_config(session, model_uuid)
        else:
            async with session_scope() as new_session:
                config = await get_model_decrypted_config(new_session, model_uuid)
        if not config:
            raise HTTPException(status_code=404, detail=f"Model registry entry not found: {model_registry_id}")
        provider = config.get("provider", "")
        model_name = config.get("model_name", "")
        api_key = config.get("api_key") or None
        api_base = config.get("base_url") or None
        # Build a provider-prefixed model string for LiteLLM (e.g. "openai/gpt-4o")
        if provider and model_name and not model_name.startswith(f"{provider}/"):
            resolved_model = f"{provider}/{model_name}"
        else:
            resolved_model = model_name
        return resolved_model, api_key, api_base
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=True).error("Failed to resolve model from registry id={}: {}", model_registry_id, str(e))
        raise HTTPException(status_code=400, detail=f"Failed to resolve model from registry: {type(e).__name__}: {str(e) or 'unknown error'}")


# ---------------------------------------------------------------------------
#  RBAC helpers for evaluator configs (mirrors guardrails_catalogue.py)
# ---------------------------------------------------------------------------

def _is_root_user(current_user) -> bool:
    return str(getattr(current_user, "role", "")).lower() == "root"


def _normalize_visibility(value: str | None) -> str:
    normalized = (value or "private").strip().lower()
    if normalized not in {"private", "public"}:
        raise HTTPException(status_code=400, detail=f"Unsupported visibility '{value}'")
    return normalized


def _normalize_public_scope(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in {"organization", "department"}:
        raise HTTPException(status_code=400, detail=f"Unsupported public_scope '{value}'")
    return normalized


def _string_ids(values: list | None) -> list[str]:
    return [str(v) for v in (values or [])]


def _first_eval_membership_scope(
    org_ids: set[UUID],
    dept_pairs: list[tuple[UUID, UUID]],
) -> tuple[UUID | None, UUID | None]:
    if dept_pairs:
        current_org_id, current_dept_id = sorted(dept_pairs, key=lambda x: (str(x[0]), str(x[1])))[0]
        return current_org_id, current_dept_id
    if org_ids:
        return sorted(org_ids, key=str)[0], None
    return None, None


async def _get_eval_scope_memberships(session, user_id: UUID) -> tuple[set[UUID], list[tuple[UUID, UUID]]]:
    """Return (org_ids, dept_pairs) for the given user."""
    org_rows = (
        await session.exec(
            select(UserOrganizationMembership.org_id).where(
                UserOrganizationMembership.user_id == user_id,
                UserOrganizationMembership.status.in_(["accepted", "active"]),
            )
        )
    ).all()

    dept_rows = (
        await session.exec(
            select(UserDepartmentMembership.org_id, UserDepartmentMembership.department_id).where(
                UserDepartmentMembership.user_id == user_id,
                UserDepartmentMembership.status == "active",
            )
        )
    ).all()
    org_ids = {r if isinstance(r, UUID) else r[0] for r in org_rows}
    return org_ids, [(row[0], row[1]) for row in dept_rows]


async def _validate_eval_scope_refs(session, org_id: UUID | None, dept_id: UUID | None) -> None:
    if dept_id and not org_id:
        raise HTTPException(status_code=400, detail="dept_id requires org_id")
    if org_id:
        org = await session.get(Organization, org_id)
        if not org:
            raise HTTPException(status_code=400, detail="Invalid org_id")
    if dept_id:
        dept = (
            await session.exec(
                select(Department).where(Department.id == dept_id, Department.org_id == org_id)
            )
        ).first()
        if not dept:
            raise HTTPException(status_code=400, detail="Invalid dept_id for org_id")


async def _validate_departments_exist_for_org(session, org_id: UUID, dept_ids: list[UUID]) -> None:
    if not dept_ids:
        return
    rows = (
        await session.exec(
            select(Department.id).where(Department.org_id == org_id, Department.id.in_(dept_ids))
        )
    ).all()
    if len({str(r if isinstance(r, UUID) else r[0]) for r in rows}) != len({str(d) for d in dept_ids}):
        raise HTTPException(status_code=400, detail="One or more public_dept_ids are invalid for org_id")


def _is_evaluator_owner(evaluator: Evaluator, current_user) -> bool:
    """Check if current user owns the evaluator."""
    return evaluator.user_id is not None and evaluator.user_id == current_user.id


async def _get_scoped_langfuse_for_evaluation(
    session,
    current_user,
    org_id: UUID | None = None,
    dept_id: UUID | None = None,
) -> tuple[set[str], Any]:
    """Resolve allowed_user_ids and a single langfuse client for evaluation.

    Returns (allowed_user_ids, langfuse_client). Falls back to env-var client
    when no bindings are configured. For super_admin the first binding is
    the org_admin binding (see resolve_observability_scope ordering), so
    write paths that need "one client for this user" land on the right
    project.
    """
    allowed_user_ids, clients = await _get_scoped_langfuse_clients_for_evaluation(
        session, current_user, org_id=org_id, dept_id=dept_id,
    )
    return allowed_user_ids, (clients[0] if clients else None)


async def _get_scoped_langfuse_clients_for_evaluation(
    session,
    current_user,
    org_id: UUID | None = None,
    dept_id: UUID | None = None,
) -> tuple[set[str], list[Any]]:
    """Resolve allowed_user_ids and ALL scoped langfuse clients for evaluation.

    Use this for read paths (list scores, pending reviews) so super_admin's
    reads cover both the org_admin project (where super_admin's own traces
    and scores live) and every department project in their org.
    """
    from agentcore.api.observability import get_langfuse_client, _get_langfuse_client_for_binding

    try:
        scope = await resolve_observability_scope(
            session,
            current_user=current_user,
            org_id=org_id,
            dept_id=dept_id,
            enforce_filter_for_admin=False,
        )
        allowed_user_ids = scope.allowed_user_ids

        clients: list[Any] = []
        for binding in scope.bindings:
            try:
                lf_client = _get_langfuse_client_for_binding(binding)
                if lf_client:
                    clients.append(lf_client)
            except Exception:
                continue

        if not clients:
            env_client = get_langfuse_client()
            if env_client:
                clients.append(env_client)

        return allowed_user_ids, clients
    except Exception:
        env_client = get_langfuse_client()
        return {str(current_user.id)}, ([env_client] if env_client else [])


# =============================================================================
# Response Models
# =============================================================================

class ScoreResponse(BaseModel):
    """Represents a single evaluation score."""
    id: str
    trace_id: str
    agent_name: str | None = None
    name: str
    value: float
    source: str  # "ANNOTATION" (human), "API" (llm judge)
    comment: str | None = None
    user_id: str | None = None
    created_at: datetime | None = None
    observation_id: str | None = None
    config_id: str | None = None


class CreateScoreRequest(BaseModel):
    """Request to create a manual score (annotation)."""
    trace_id: str
    name: str
    value: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1")
    comment: str | None = None
    observation_id: str | None = None


class JudgeConfig(BaseModel):
    """Saved judge configuration for reuse."""
    id: str | None = None
    name: str
    criteria: str
    model: str = "gpt-4o"
    # target: 'existing' -> evaluate matching existing traces now
    #         'new' -> save evaluator to apply to future traces (not executed immediately)
    target: str = Field("existing", description="'existing' or 'new'")
    # Filtering options to select traces
    trace_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    project_name: Optional[str] = None
    ts_from: Optional[str] = None  # ISO timestamp
    ts_to: Optional[str] = None
    user_id: str | None = None


class EvaluatorCreateRequest(BaseModel):
    name: str
    criteria: str
    model_registry_id: str  # ID from model registry (required)
    preset_id: Optional[str] = None
    target: Optional[Union[str, List[str]]] = Field(default="existing")
    ground_truth: Optional[str] = None
    trace_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_ids: Optional[List[str]] = None
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    project_name: Optional[str] = None
    ts_from: Optional[str] = None
    ts_to: Optional[str] = None


class EvaluatorResponse(BaseModel):
    id: str
    name: str
    criteria: str
    model: str
    model_registry_id: Optional[str] = None
    user_id: str | None = None
    preset_id: Optional[str] = None
    agent_ids: Optional[List[str]] = None
    target: Optional[List[str]] = None
    ground_truth: Optional[str] = None
    trace_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    project_name: Optional[str] = None
    ts_from: Optional[str] = None
    ts_to: Optional[str] = None
    created_at: Optional[str] = None


class TraceForReview(BaseModel):
    """Trace info for annotation queue."""
    id: str
    name: str | None
    timestamp: datetime | None
    input: Any | None
    output: Any | None
    session_id: str | None
    agent_name: str | None
    has_scores: bool = False
    score_count: int = 0


class DatasetResponse(BaseModel):
    """Represents a Langfuse dataset."""
    id: str
    name: str
    description: str | None = None
    metadata: Any | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    item_count: int | None = None
    visibility: str = "private"
    public_scope: str | None = None
    owner_user_id: str | None = None
    created_by: str | None = None
    created_by_id: str | None = None
    org_id: str | None = None
    dept_id: str | None = None
    public_dept_ids: list[str] | None = None


class CreateDatasetRequest(BaseModel):
    """Request to create a dataset."""
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    metadata: Any | None = None
    visibility: str = "private"
    public_scope: str | None = None
    org_id: UUID | None = None
    dept_id: UUID | None = None
    public_dept_ids: list[UUID] | None = None


class UpdateDatasetRequest(BaseModel):
    """Request to update a dataset."""
    description: str | None = None
    visibility: str | None = None
    public_scope: str | None = None
    org_id: UUID | None = None
    dept_id: UUID | None = None
    public_dept_ids: list[UUID] | None = None


class DatasetItemResponse(BaseModel):
    """Represents a single dataset item."""
    id: str
    dataset_name: str
    status: str | None = None
    input: Any | None = None
    expected_output: Any | None = None
    metadata: Any | None = None
    source_trace_id: str | None = None
    source_observation_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CreateDatasetItemRequest(BaseModel):
    """Request to create a dataset item."""
    input: Any | None = None
    expected_output: Any | None = None
    metadata: Any | None = None
    source_trace_id: str | None = None
    source_observation_id: str | None = None
    trace_id: str | None = None
    use_trace_output_as_expected: bool = True


class DatasetCsvImportError(BaseModel):
    """CSV import error for one row."""
    row: int
    message: str


class DatasetCsvImportResponse(BaseModel):
    """CSV import summary for dataset items."""
    dataset_name: str
    total_rows: int
    created_count: int
    failed_count: int
    skipped_count: int = 0
    errors: list[DatasetCsvImportError] = Field(default_factory=list)


class DatasetRunResponse(BaseModel):
    """Represents a dataset experiment run."""
    id: str
    name: str
    description: str | None = None
    metadata: Any | None = None
    dataset_id: str | None = None
    dataset_name: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DatasetRunItemScoreResponse(BaseModel):
    """Score snapshot linked to a dataset run item trace."""
    id: str
    name: str
    value: float
    source: str
    comment: str | None = None
    created_at: datetime | None = None


class DatasetRunItemDetailResponse(BaseModel):
    """Detailed dataset run item response."""
    id: str
    dataset_item_id: str | None = None
    trace_id: str | None = None
    observation_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    trace_name: str | None = None
    trace_input: Any | None = None
    trace_output: Any | None = None
    score_count: int = 0
    scores: list[DatasetRunItemScoreResponse] = Field(default_factory=list)


class DatasetRunDetailResponse(BaseModel):
    """Detailed run payload including run items and associated traces/scores."""
    run: DatasetRunResponse
    item_count: int
    items: list[DatasetRunItemDetailResponse]


class RunDatasetExperimentRequest(BaseModel):
    """Request to run a Langfuse dataset experiment."""
    experiment_name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    agent_id: str | None = None
    generation_model_registry_id: str | None = None  # Model registry ID for generation model
    evaluator_config_id: str | None = None
    preset_id: str | None = None
    evaluator_name: str | None = None
    criteria: str | None = None
    judge_model_registry_id: str | None = None  # Model registry ID for judge model


class DatasetExperimentEnqueueResponse(BaseModel):
    """Response when a dataset experiment is queued."""
    job_id: str
    dataset_name: str
    experiment_name: str
    status: str


class DatasetExperimentJobResponse(BaseModel):
    """Background dataset experiment job state."""
    job_id: str
    status: str
    dataset_name: str
    experiment_name: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: Dict[str, Any] | None = None


# =============================================================================
# Helper Functions
# =============================================================================

def get_langfuse_client():
    """Get a Langfuse client using environment variables."""
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")

    if not all([secret_key, public_key, base_url]):
        return None

    try:
        from langfuse import Langfuse
        if not os.getenv("LANGFUSE_BASE_URL") and os.getenv("LANGFUSE_HOST"):
            os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_HOST")

        client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=base_url
        )
        api_obj = getattr(client, "api", None)
        client._is_v3 = bool(
            hasattr(client, "auth_check")
            or (
                api_obj
                and (
                    hasattr(api_obj, "trace")
                    or hasattr(api_obj, "traces")
                )
            )
        )
        return client
    except ImportError:
        logger.warning("Langfuse package not installed")
        return None
    except Exception as e:
        logger.error("Failed to create Langfuse client: {}", str(e))
        return None


def get_attr(obj, *attrs, default=None):
    """Get attribute from object or dict, trying multiple attribute names."""
    for attr in attrs:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val is not None:
                return val
        if isinstance(obj, dict) and attr in obj:
            val = obj[attr]
            if val is not None:
                return val
    return default


def _ensure_litellm_logging_compatibility_patch() -> None:
    """Patch LiteLLM standard logging cold-storage hook to avoid proxy-only imports."""
    global _LITELLM_STD_LOGGING_PATCHED

    if not LITELLM_AVAILABLE or _LITELLM_STD_LOGGING_PATCHED:
        return

    try:
        from litellm.litellm_core_utils import litellm_logging as _litellm_logging

        setup_cls = getattr(_litellm_logging, "StandardLoggingPayloadSetup", None)
        if setup_cls is None or not hasattr(setup_cls, "_generate_cold_storage_object_key"):
            _LITELLM_STD_LOGGING_PATCHED = True
            return

        def _disabled_cold_storage_key(*args, **kwargs):
            return None

        setup_cls._generate_cold_storage_object_key = staticmethod(_disabled_cold_storage_key)
        _LITELLM_STD_LOGGING_PATCHED = True
        logger.debug("Applied LiteLLM logging compatibility patch: disabled cold-storage object key generation.")
    except Exception as exc:
        logger.debug("Could not apply LiteLLM logging compatibility patch: {}", str(exc))


def parse_trace_data(trace) -> Dict[str, Any]:
    """Extract and normalize trace data."""
    metadata = get_attr(trace, 'metadata', 'meta')
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            metadata = parsed if isinstance(parsed, dict) else metadata
        except Exception:
            pass

    top_level_user_id = get_attr(trace, 'user_id', 'userId', 'sender', 'user')
    metadata_user_id = None
    metadata_user_uuid = None
    if isinstance(metadata, dict):
        # Prefer the explicit app-user UUID (written by the tracer as
        # metadata.user_uuid) over the displayable user_id/userId, which is
        # now a username string after the langfuse username-display change.
        metadata_user_uuid = metadata.get("user_uuid")
        metadata_user_id = (
            metadata_user_uuid
            or metadata.get("app_user_id")
            or metadata.get("created_by_user_id")
            or metadata.get("owner_user_id")
            or metadata.get("user_id")
            or metadata.get("userId")
        )

    top_level_session_id = get_attr(trace, 'session_id', 'sessionId')
    metadata_session_id = None
    if isinstance(metadata, dict):
        metadata_session_id = metadata.get("session_id") or metadata.get("sessionId")

    return {
        "id": get_attr(trace, 'id', 'trace_id', 'traceId'),
        "name": get_attr(trace, 'name', 'display_name', 'trace_name'),
        "timestamp": get_attr(trace, 'timestamp', 'createdAt', 'created_at'),
        "input": get_attr(trace, 'input', 'inputs', 'input_data', 'generation', 'query'),
        "output": get_attr(trace, 'output', 'outputs', 'generation', 'text_output', 'response'),
        "session_id": top_level_session_id or metadata_session_id,
        # Prefer the app-user UUID (metadata) over the top-level user_id,
        # which after the langfuse username-display change is a username
        # string. Downstream consumers expect UUIDs here.
        "user_id": metadata_user_id or top_level_user_id,
        "user_uuid": metadata_user_uuid,
        "metadata": metadata,
        "tags": get_attr(trace, 'tags', 'labels'),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert object-like values into plain dictionaries."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}
    if hasattr(value, "__dict__"):
        try:
            raw = dict(vars(value))
            return {k: v for k, v in raw.items() if not k.startswith("_")}
        except Exception:
            return {}
    return {}


def _parse_paginated_response(response: Any) -> tuple[list[Any], int | None]:
    """Extract rows + total from Langfuse paginated payload variants."""
    if response is None:
        return [], None

    rows: list[Any] = []
    total: int | None = None

    if isinstance(response, list):
        return response, len(response)

    if isinstance(response, dict):
        rows = list(response.get("data") or response.get("items") or [])
        meta = response.get("meta") or {}
        if isinstance(meta, dict):
            total = meta.get("total_items") or meta.get("total")
        if total is None:
            total = response.get("total")
        return rows, int(total) if total is not None else None

    if hasattr(response, "data"):
        rows = list(getattr(response, "data", []) or [])
        meta = getattr(response, "meta", None)
        if isinstance(meta, dict):
            total = meta.get("total_items") or meta.get("total")
        elif meta is not None:
            total = getattr(meta, "total_items", None) or getattr(meta, "total", None)
        return rows, int(total) if total is not None else None

    return [], None


def _dataset_owned_by_user(dataset_obj: Any, user_id: str) -> bool:
    """Best-effort user scoping for datasets via metadata."""
    metadata = get_attr(dataset_obj, "metadata", default=None)
    if not isinstance(metadata, dict):
        return False

    owner = (
        metadata.get("app_user_id")
        or metadata.get("user_id")
        or metadata.get("owner_user_id")
        or metadata.get("created_by_user_id")
    )
    if owner is None:
        return False
    return str(owner) == str(user_id)


def _dataset_accessible_by_users(
    dataset_obj: Any,
    allowed_user_ids: set[str],
    *,
    current_user: Any | None = None,
    org_ids: set[UUID] | None = None,
    dept_pairs: list[tuple[UUID, UUID]] | None = None,
) -> bool:
    """Check if dataset is accessible by any of the allowed user IDs or visibility rules."""
    metadata = get_attr(dataset_obj, "metadata", default=None)
    if not isinstance(metadata, dict):
        return False
    owner = (
        metadata.get("app_user_id")
        or metadata.get("user_id")
        or metadata.get("owner_user_id")
        or metadata.get("created_by_user_id")
    )
    ds_org_id = metadata.get("org_id")
    ds_dept_id = metadata.get("dept_id")

    if current_user and _is_root_user(current_user):
        return bool(
            owner is not None
            and str(owner) == str(current_user.id)
            and ds_org_id is None
            and ds_dept_id is None
        )

    # Owner always has access
    if current_user and owner is not None and str(owner) == str(current_user.id):
        return True

    # Check visibility-based access
    visibility = metadata.get("visibility", "private")
    role = normalize_role(str(getattr(current_user, "role", ""))) if current_user else ""
    if visibility == "private":
        if role == "super_admin" and ds_org_id and org_ids and UUID(ds_org_id) in org_ids:
            return True
        if role == "department_admin" and ds_dept_id and dept_pairs:
            user_dept_ids = {str(d) for _, d in dept_pairs}
            if str(ds_dept_id) in user_dept_ids:
                return True

    if visibility == "public":
        public_scope = metadata.get("public_scope")
        ds_public_dept_ids = metadata.get("public_dept_ids") or []

        if public_scope == "organization" and ds_org_id and org_ids:
            if UUID(ds_org_id) in org_ids:
                return True
        elif public_scope == "department":
            if dept_pairs and ds_public_dept_ids:
                user_dept_ids = {str(d) for _, d in dept_pairs}
                if any(did in user_dept_ids for did in ds_public_dept_ids):
                    return True
            elif dept_pairs and ds_dept_id:
                user_dept_ids = {str(d) for _, d in dept_pairs}
                if ds_dept_id in user_dept_ids:
                    return True

    return False


def _dataset_item_owned_by_user(item_obj: Any, user_id: str) -> bool:
    """Best-effort user scoping for dataset items via metadata."""
    metadata = get_attr(item_obj, "metadata", default=None)
    if not isinstance(metadata, dict):
        return False
    owner = (
        metadata.get("app_user_id")
        or metadata.get("user_id")
        or metadata.get("owner_user_id")
        or metadata.get("created_by_user_id")
    )
    if owner is None:
        return False
    return str(owner) == str(user_id)


def _dataset_item_accessible_by_users(item_obj: Any, allowed_user_ids: set[str]) -> bool:
    """Check if dataset item is accessible by any of the allowed user IDs."""
    metadata = get_attr(item_obj, "metadata", default=None)
    if not isinstance(metadata, dict):
        return False
    owner = (
        metadata.get("app_user_id")
        or metadata.get("user_id")
        or metadata.get("owner_user_id")
        or metadata.get("created_by_user_id")
    )
    if owner is None:
        return False
    return str(owner) in allowed_user_ids


def _dataset_dept_candidates(metadata: dict[str, Any]) -> set[str]:
    dept_candidates = set(metadata.get("public_dept_ids") or [])
    if metadata.get("dept_id"):
        dept_candidates.add(str(metadata.get("dept_id")))
    return dept_candidates


def _is_multi_dept_dataset(metadata: dict[str, Any]) -> bool:
    return (
        (metadata.get("visibility") or "private") == "public"
        and metadata.get("public_scope") == "department"
        and len(_dataset_dept_candidates(metadata)) > 1
    )


def _dataset_owner_id(metadata: dict[str, Any]) -> str | None:
    return (
        metadata.get("app_user_id")
        or metadata.get("user_id")
        or metadata.get("owner_user_id")
        or metadata.get("created_by_user_id")
    )


def _can_manage_dataset(
    dataset_obj: Any,
    current_user,
    org_ids: set[UUID],
    dept_pairs: list[tuple[UUID, UUID]],
) -> bool:
    metadata = get_attr(dataset_obj, "metadata", default=None)
    if not isinstance(metadata, dict):
        return False

    owner = _dataset_owner_id(metadata)
    ds_org_id = metadata.get("org_id")
    ds_dept_id = metadata.get("dept_id")
    visibility = metadata.get("visibility", "private")
    public_scope = metadata.get("public_scope")

    if current_user and _is_root_user(current_user):
        return bool(
            owner is not None
            and str(owner) == str(current_user.id)
            and ds_org_id is None
            and ds_dept_id is None
        )

    role = normalize_role(str(getattr(current_user, "role", "")))
    if role == "super_admin":
        if visibility == "private" and owner is not None and ds_org_id is None and ds_dept_id is None:
            return str(owner) == str(current_user.id)
        if ds_org_id and org_ids:
            return UUID(str(ds_org_id)) in org_ids

    if role == "department_admin":
        if _is_multi_dept_dataset(metadata):
            return False
        if visibility == "public" and public_scope == "organization":
            return False
        dept_id_set = {str(d) for _, d in dept_pairs}
        dept_candidates = _dataset_dept_candidates(metadata)
        if visibility == "private":
            return bool(dept_candidates.intersection(dept_id_set))
        if public_scope == "department":
            return bool(dept_candidates.intersection(dept_id_set))
        return False

    if role in {"developer", "business_user"}:
        return visibility == "private" and owner is not None and str(owner) == str(current_user.id)

    return False


async def _check_dataset_access(
    dataset_obj: Any,
    allowed_user_ids: set[str],
    current_user: Any,
    session: Any,
) -> bool:
    """Async helper: checks dataset access using full visibility + scope."""
    org_ids, dept_pairs = await _get_eval_scope_memberships(session, current_user.id)
    return _dataset_accessible_by_users(
        dataset_obj, allowed_user_ids,
        current_user=current_user, org_ids=org_ids, dept_pairs=dept_pairs,
    )


async def _enforce_dataset_creation_scope(
    session,
    current_user,
    payload: CreateDatasetRequest,
) -> tuple[str, str | None, list[str], str | None, str | None]:
    user_role = normalize_role(str(current_user.role))
    visibility = _normalize_visibility(payload.visibility)
    public_scope = _normalize_public_scope(payload.public_scope) if visibility == "public" else None
    public_dept_ids = _string_ids(payload.public_dept_ids)
    org_ids, dept_pairs = await _get_eval_scope_memberships(session, current_user.id)

    p_org_id = payload.org_id
    p_dept_id = payload.dept_id

    if visibility == "private":
        public_scope = None
        public_dept_ids = []
        if user_role == "root":
            p_org_id = None
            p_dept_id = None
        elif user_role == "super_admin":
            current_org_id, _ = _first_eval_membership_scope(org_ids, dept_pairs)
            if not current_org_id:
                raise HTTPException(status_code=403, detail="No active organization scope found")
            p_org_id = current_org_id
            p_dept_id = None
        elif user_role in {"department_admin", "developer", "business_user"}:
            current_org_id, current_dept_id = _first_eval_membership_scope(org_ids, dept_pairs)
            if not current_org_id or not current_dept_id:
                raise HTTPException(status_code=403, detail="No active department scope found")
            p_org_id = current_org_id
            p_dept_id = current_dept_id
        else:
            p_org_id = None
            p_dept_id = None
    else:
        if public_scope is None:
            raise HTTPException(status_code=400, detail="public_scope is required when visibility is public")
        if public_scope == "organization":
            if not p_org_id:
                raise HTTPException(status_code=400, detail="org_id is required for public organization visibility")
            if user_role != "root" and p_org_id not in org_ids:
                raise HTTPException(status_code=403, detail="org_id must belong to your organization scope")
            p_dept_id = None
            public_dept_ids = []
        else:
            if user_role in {"super_admin", "root"}:
                if not p_org_id:
                    raise HTTPException(status_code=400, detail="org_id is required for department visibility")
                if user_role != "root" and p_org_id not in org_ids:
                    raise HTTPException(status_code=403, detail="org_id must belong to your organization scope")
                if not public_dept_ids and p_dept_id:
                    public_dept_ids = [str(p_dept_id)]
                if not public_dept_ids:
                    raise HTTPException(status_code=400, detail="Select at least one department")
                await _validate_departments_exist_for_org(session, p_org_id, [UUID(v) for v in public_dept_ids])
                p_dept_id = UUID(public_dept_ids[0]) if len(public_dept_ids) == 1 else None
            else:
                current_org_id, current_dept_id = _first_eval_membership_scope(org_ids, dept_pairs)
                if not current_org_id or not current_dept_id:
                    raise HTTPException(status_code=403, detail="No active department scope found")
                p_org_id = current_org_id
                p_dept_id = current_dept_id
                public_dept_ids = [str(current_dept_id)]

    await _validate_eval_scope_refs(session, p_org_id, p_dept_id)
    return (
        visibility,
        public_scope,
        public_dept_ids,
        str(p_org_id) if p_org_id else None,
        str(p_dept_id) if p_dept_id else None,
    )


def _merge_dataset_metadata(
    metadata: Any,
    *,
    user_id: str,
    visibility: str = "private",
    public_scope: str | None = None,
    org_id: str | None = None,
    dept_id: str | None = None,
    public_dept_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Attach app metadata while preserving user-provided fields."""
    base = _as_dict(metadata)
    base["app_user_id"] = str(user_id)
    base["created_by_user_id"] = str(user_id)
    base["owner_user_id"] = str(user_id)
    base["user_id"] = str(user_id)
    base.setdefault("created_via", "agentcore-evaluation")
    base["visibility"] = visibility or "private"
    if public_scope:
        base["public_scope"] = public_scope
    else:
        base.pop("public_scope", None)
    if org_id:
        base["org_id"] = str(org_id)
    else:
        base.pop("org_id", None)
    if dept_id:
        base["dept_id"] = str(dept_id)
    else:
        base.pop("dept_id", None)
    if public_dept_ids:
        base["public_dept_ids"] = [str(d) for d in public_dept_ids]
    else:
        base.pop("public_dept_ids", None)
    return base


def _parse_csv_json_cell(value: Any) -> Any | None:
    """Parse CSV cell into JSON when possible; keep plain text otherwise."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"null", "none"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def _parse_csv_bool_cell(value: Any, *, default: bool = True) -> bool:
    """Parse boolean CSV cells with safe defaults."""
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _get_row_value(row: dict[str, Any], *keys: str) -> Any | None:
    """Get first non-empty value from a CSV row by alias keys."""
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _create_dataset_item_for_user(
    *,
    client: Any,
    dataset_name: str,
    payload: CreateDatasetItemRequest,
    current_user_id: str,
    flush: bool = True,
) -> DatasetItemResponse:
    """Create one dataset item while enforcing user scoping and trace ownership."""
    item_input = payload.input
    expected_output = payload.expected_output
    source_trace_id = payload.source_trace_id

    if payload.trace_id:
        trace_raw = _fetch_trace_by_id(client, payload.trace_id)
        if not trace_raw:
            raise HTTPException(status_code=404, detail=f"Trace '{payload.trace_id}' not found")

        trace_dict = parse_trace_data(trace_raw)
        trace_user = _extract_trace_user_id(trace_dict)
        if trace_user and trace_user != str(current_user_id):
            raise HTTPException(status_code=403, detail="Trace does not belong to current user")

        if item_input is None:
            item_input = trace_dict.get("input")
        if expected_output is None and payload.use_trace_output_as_expected:
            expected_output = trace_dict.get("output")
        source_trace_id = source_trace_id or str(trace_dict.get("id") or payload.trace_id)

    if item_input is None and expected_output is None:
        raise HTTPException(
            status_code=400,
            detail="Provide input/expected_output or a trace_id to create a dataset item",
        )

    metadata = _merge_dataset_metadata(payload.metadata, user_id=str(current_user_id))
    if payload.trace_id:
        metadata.setdefault("trace_id", str(payload.trace_id))

    item = client.create_dataset_item(
        dataset_name=dataset_name,
        input=item_input,
        expected_output=expected_output,
        metadata=metadata,
        source_trace_id=source_trace_id,
        source_observation_id=payload.source_observation_id,
    )
    if flush and hasattr(client, "flush"):
        client.flush()
    return _dataset_item_to_response(item)


def _csv_row_to_dataset_item_request(row: dict[str, Any]) -> CreateDatasetItemRequest:
    """Map one CSV row to CreateDatasetItemRequest with flexible header aliases."""
    row_lower = {(str(k).strip().lower() if k is not None else ""): v for k, v in row.items()}
    item_input = _parse_csv_json_cell(
        _get_row_value(row_lower, "input", "query", "question", "prompt")
    )
    expected_output = _parse_csv_json_cell(
        _get_row_value(
            row_lower,
            "expected_output",
            "expected output",
            "ground_truth",
            "ground truth",
            "answer",
        )
    )
    metadata = _parse_csv_json_cell(_get_row_value(row_lower, "metadata"))
    source_trace_id = _get_row_value(
        row_lower,
        "source_trace_id",
        "source trace id",
        "source_trace",
    )
    source_observation_id = _get_row_value(
        row_lower,
        "source_observation_id",
        "source observation id",
        "source_observation",
    )
    trace_id = _get_row_value(row_lower, "trace_id", "trace id")
    use_trace_output_as_expected = _parse_csv_bool_cell(
        _get_row_value(row_lower, "use_trace_output_as_expected"),
        default=True,
    )
    return CreateDatasetItemRequest(
        input=item_input,
        expected_output=expected_output,
        metadata=metadata,
        source_trace_id=source_trace_id,
        source_observation_id=source_observation_id,
        trace_id=trace_id,
        use_trace_output_as_expected=use_trace_output_as_expected,
    )


def _dataset_to_response(
    dataset_obj: Any,
    *,
    item_count: int | None = None,
    created_by_lookup: dict[str, str] | None = None,
) -> DatasetResponse:
    """Serialize Langfuse dataset object to API response."""
    metadata = get_attr(dataset_obj, "metadata", default=None)
    meta_dict = metadata if isinstance(metadata, dict) else {}
    owner_user_id = meta_dict.get("app_user_id") or meta_dict.get("created_by_user_id")
    return DatasetResponse(
        id=str(get_attr(dataset_obj, "id", default="") or ""),
        name=str(get_attr(dataset_obj, "name", default="") or ""),
        description=get_attr(dataset_obj, "description", default=None),
        metadata=metadata,
        created_at=get_attr(dataset_obj, "created_at", "createdAt", default=None),
        updated_at=get_attr(dataset_obj, "updated_at", "updatedAt", default=None),
        item_count=item_count,
        visibility=meta_dict.get("visibility", "private"),
        public_scope=meta_dict.get("public_scope"),
        owner_user_id=owner_user_id,
        created_by=created_by_lookup.get(str(owner_user_id)) if created_by_lookup and owner_user_id else None,
        created_by_id=str(owner_user_id) if owner_user_id else None,
        org_id=meta_dict.get("org_id"),
        dept_id=meta_dict.get("dept_id"),
        public_dept_ids=meta_dict.get("public_dept_ids"),
    )


def _dataset_item_to_response(item_obj: Any) -> DatasetItemResponse:
    """Serialize Langfuse dataset item object to API response."""
    status = get_attr(item_obj, "status", default=None)
    if hasattr(status, "value"):
        status = status.value

    return DatasetItemResponse(
        id=str(get_attr(item_obj, "id", default="") or ""),
        dataset_name=str(get_attr(item_obj, "dataset_name", "datasetName", default="") or ""),
        status=str(status) if status is not None else None,
        input=get_attr(item_obj, "input", default=None),
        expected_output=get_attr(item_obj, "expected_output", "expectedOutput", default=None),
        metadata=get_attr(item_obj, "metadata", default=None),
        source_trace_id=get_attr(item_obj, "source_trace_id", "sourceTraceId", default=None),
        source_observation_id=get_attr(item_obj, "source_observation_id", "sourceObservationId", default=None),
        created_at=get_attr(item_obj, "created_at", "createdAt", default=None),
        updated_at=get_attr(item_obj, "updated_at", "updatedAt", default=None),
    )


def _dataset_run_to_response(run_obj: Any) -> DatasetRunResponse:
    """Serialize Langfuse dataset run object to API response."""
    return DatasetRunResponse(
        id=str(get_attr(run_obj, "id", default="") or ""),
        name=str(get_attr(run_obj, "name", default="") or ""),
        description=get_attr(run_obj, "description", default=None),
        metadata=get_attr(run_obj, "metadata", default=None),
        dataset_id=get_attr(run_obj, "dataset_id", "datasetId", default=None),
        dataset_name=get_attr(run_obj, "dataset_name", "datasetName", default=None),
        created_at=get_attr(run_obj, "created_at", "createdAt", default=None),
        updated_at=get_attr(run_obj, "updated_at", "updatedAt", default=None),
    )


def _dataset_run_item_to_detail_response(
    item_obj: Any,
    *,
    trace_dict: dict[str, Any] | None = None,
    scores: list[DatasetRunItemScoreResponse] | None = None,
) -> DatasetRunItemDetailResponse:
    """Serialize Langfuse dataset run item object to detailed response."""
    trace_dict = trace_dict or {}
    scores = scores or []
    trace_name = trace_dict.get("name")
    if trace_name is None:
        trace_name = get_attr(item_obj, "trace_name", "traceName", default=None)

    trace_input = trace_dict.get("input")
    if trace_input is None:
        trace_input = get_attr(item_obj, "input", default=None)

    trace_output = trace_dict.get("output")
    if trace_output is None:
        trace_output = get_attr(item_obj, "output", default=None)

    return DatasetRunItemDetailResponse(
        id=str(get_attr(item_obj, "id", default="") or ""),
        dataset_item_id=get_attr(item_obj, "dataset_item_id", "datasetItemId", default=None),
        trace_id=get_attr(item_obj, "trace_id", "traceId", default=None),
        observation_id=get_attr(item_obj, "observation_id", "observationId", default=None),
        created_at=get_attr(item_obj, "created_at", "createdAt", default=None),
        updated_at=get_attr(item_obj, "updated_at", "updatedAt", default=None),
        trace_name=str(trace_name) if trace_name is not None else None,
        trace_input=trace_input,
        trace_output=trace_output,
        score_count=len(scores),
        scores=scores,
    )


def _extract_run_item_evaluation_scores(item_obj: Any) -> list[DatasetRunItemScoreResponse]:
    """Extract evaluator scores directly from dataset run item payload."""
    rows: list[DatasetRunItemScoreResponse] = []
    evaluations = get_attr(item_obj, "evaluations", default=None) or []
    if not isinstance(evaluations, list):
        return rows

    for idx, evaluation in enumerate(evaluations, 1):
        value = get_attr(evaluation, "value", default=None)
        try:
            numeric_value = float(value)
        except Exception:
            continue
        rows.append(
            DatasetRunItemScoreResponse(
                id=str(get_attr(evaluation, "id", default=None) or f"run-eval-{idx}"),
                name=str(get_attr(evaluation, "name", default="Score") or "Score"),
                value=numeric_value,
                source="EXPERIMENT",
                comment=get_attr(evaluation, "comment", default=None),
                created_at=get_attr(evaluation, "created_at", "createdAt", default=None),
            )
        )
    return rows


def _set_dataset_experiment_job(job_id: str, **updates: Any) -> None:
    """Upsert in-memory dataset experiment job state."""
    with _DATASET_EXPERIMENT_JOBS_LOCK:
        current = _DATASET_EXPERIMENT_JOBS.get(job_id, {}).copy()
        current.update(updates)
        _DATASET_EXPERIMENT_JOBS[job_id] = current


def _get_dataset_experiment_job(job_id: str) -> dict[str, Any] | None:
    """Return a copy of in-memory dataset experiment job state."""
    with _DATASET_EXPERIMENT_JOBS_LOCK:
        current = _DATASET_EXPERIMENT_JOBS.get(job_id)
        return current.copy() if current else None


def _dataset_job_response(job_id: str, payload: dict[str, Any]) -> DatasetExperimentJobResponse:
    """Serialize internal dataset experiment job payload."""
    return DatasetExperimentJobResponse(
        job_id=job_id,
        status=str(payload.get("status") or "unknown"),
        dataset_name=str(payload.get("dataset_name") or ""),
        experiment_name=str(payload.get("experiment_name") or ""),
        started_at=payload.get("started_at"),
        finished_at=payload.get("finished_at"),
        error=payload.get("error"),
        result=payload.get("result"),
    )


def _to_text(value: Any) -> str:
    """Convert arbitrary payloads into compact text for prompts/inputs."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _build_experiment_evaluation(
    *,
    name: str,
    value: Any,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Create a Langfuse experiment Evaluation object when available."""
    try:
        from langfuse import Evaluation as LangfuseEvaluation  # type: ignore

        kwargs: dict[str, Any] = {
            "name": name,
            "value": value,
            "comment": comment,
        }
        if metadata is not None:
            kwargs["metadata"] = metadata
        return LangfuseEvaluation(**kwargs)
    except Exception:
        return SimpleNamespace(name=name, value=value, comment=comment, metadata=metadata)


DATASET_PROMPT_CONTEXT_TEMPLATE = (
    "Input:\n"
    "Query: {{query}}\n"
    "Generation: {{generation}}\n"
    "Ground Truth: {{ground_truth}}"
)


def _ensure_dataset_prompt_template(criteria: str | None) -> str:
    """Ensure dataset judge criteria contains Query/Generation/Ground Truth placeholders."""
    base = str(criteria or "").strip()
    if not base:
        return DATASET_PROMPT_CONTEXT_TEMPLATE

    normalized = " ".join(base.lower().split())
    if "query: {{query}}" in normalized and "generation: {{generation}}" in normalized and "ground truth: {{ground_truth}}" in normalized:
        return base
    return f"{base}\n\n{DATASET_PROMPT_CONTEXT_TEMPLATE}"


def _render_dataset_judge_criteria(
    *,
    criteria: str | None,
    query: Any,
    generation: Any,
    ground_truth: Any,
) -> str:
    """Render criteria template placeholders with current dataset item values."""
    rendered = _ensure_dataset_prompt_template(criteria)
    replacements = {
        "{{query}}": _to_text(query) or "[EMPTY]",
        "{{generation}}": _to_text(generation) or "[EMPTY]",
        "{{ground_truth}}": _to_text(ground_truth) or "[NOT PROVIDED]",
    }
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered


def _extract_agent_output_from_run_response(run_response: Any) -> Any:
    """Best-effort extraction of final agent output from RunResponse variants."""
    outputs = get_attr(run_response, "outputs", default=None)
    if outputs is None and isinstance(run_response, dict):
        outputs = run_response.get("outputs")
    if not isinstance(outputs, list):
        if hasattr(run_response, "model_dump"):
            try:
                return run_response.model_dump()
            except Exception:
                return run_response
        return run_response

    text_candidates: list[str] = []
    value_candidates: list[Any] = []
    for run_output in outputs:
        result_entries = get_attr(run_output, "outputs", default=None)
        if not isinstance(result_entries, list):
            continue
        for result_data in result_entries:
            if result_data is None:
                continue
            messages = get_attr(result_data, "messages", default=None)
            if isinstance(messages, list):
                for msg in messages:
                    msg_value = get_attr(msg, "message", default=None)
                    if msg_value is not None:
                        value_candidates.append(msg_value)
                        if isinstance(msg_value, str) and msg_value.strip():
                            text_candidates.append(msg_value)

            output_map = get_attr(result_data, "outputs", default=None)
            if isinstance(output_map, dict):
                for output_entry in output_map.values():
                    out_value = get_attr(output_entry, "message", default=None)
                    if out_value is not None:
                        value_candidates.append(out_value)
                        if isinstance(out_value, str) and out_value.strip():
                            text_candidates.append(out_value)

            raw_result = get_attr(result_data, "results", default=None)
            if raw_result not in (None, "", {}, []):
                value_candidates.append(raw_result)

    if text_candidates:
        return text_candidates[-1]
    if value_candidates:
        return value_candidates[-1]
    if hasattr(run_response, "model_dump"):
        try:
            return run_response.model_dump()
        except Exception:
            pass
    return run_response


def _normalize_for_exact_match(value: Any) -> str:
    """Normalize values for exact-match evaluator comparison."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False).strip()
    except Exception:
        return str(value).strip()


def _run_async(coro):
    """Run coroutine in sync contexts, even if current thread already has a loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if not loop.is_running():
        return loop.run_until_complete(coro)

    container: dict[str, Any] = {}
    error_holder: dict[str, Exception] = {}

    def _runner():
        try:
            container["value"] = asyncio.run(coro)
        except Exception as exc:  # noqa: BLE001
            error_holder["error"] = exc

    t = Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if "error" in error_holder:
        raise error_holder["error"]
    return container.get("value")


async def _run_dataset_item_with_agent(
    *,
    agent_payload: dict[str, Any],
    user_id: str,
    item_input: Any,
    session_id: str,
) -> tuple[Any, str | None]:
    """Execute one dataset item input against an agent and return (output, trace_id)."""
    from agentcore.api.v1_schemas import InputValueRequest
    from agentcore.api.utils import build_graph_from_data
    from agentcore.services.deps import get_chat_service
    from agentcore.processing.process import run_graph_internal

    agent_id_str = str(agent_payload["id"])
    graph_data = agent_payload["data"].copy()

    graph = await build_graph_from_data(
        agent_id=agent_id_str,
        payload=graph_data,
        user_id=str(user_id),
        agent_name=agent_payload["name"],
        chat_service=get_chat_service(),
    )

    inputs = [
        InputValueRequest(
            components=[],
            input_value=_to_text(item_input),
            type="chat",
        )
    ]
    outputs = [
        vertex.id
        for vertex in graph.vertices
        if vertex.is_output and "chat" in vertex.id.lower()
    ]

    task_result, _ = await run_graph_internal(
        graph=graph,
        agent_id=agent_id_str,
        session_id=session_id,
        inputs=inputs,
        outputs=outputs,
        stream=False,
    )

    # Extract the actual Langfuse/OTEL trace ID and the tracer's client
    trace_id = None
    tracer_client = None
    try:
        from agentcore.services.tracing.service import trace_context_var
        trace_ctx = trace_context_var.get(None)
        if trace_ctx:
            lf_tracer = (getattr(trace_ctx, "tracers", None) or {}).get("langfuse")
            if lf_tracer:
                if hasattr(lf_tracer, "langfuse_trace_id"):
                    trace_id = lf_tracer.langfuse_trace_id
                # Get the tracer's Langfuse client for score submission
                tracer_client = getattr(lf_tracer, "_client", None)
            # Flush to ensure trace is sent to Langfuse
            for tracer in (getattr(trace_ctx, "tracers", None) or {}).values():
                if hasattr(tracer, "flush"):
                    try:
                        tracer.flush()
                    except Exception:
                        pass
    except Exception as e:
        logger.debug(f"Failed to extract Langfuse trace ID: {e}")

    # Fallback to graph._run_id if OTEL trace ID not available
    if not trace_id:
        trace_id = getattr(graph, "_run_id", None)
    logger.info(f"Dataset item agent run completed: agent={agent_payload.get('name')}, trace_id={trace_id}, has_tracer_client={tracer_client is not None}")

    from agentcore.api.v1_schemas import RunResponse
    run_response = RunResponse(outputs=task_result, session_id=session_id)
    output = _extract_agent_output_from_run_response(run_response)
    return output, trace_id, tracer_client


def _get_dataset_experiment_concurrency() -> int:
    """Resolve safe background concurrency for dataset experiments."""
    raw = str(os.getenv("EVALUATION_DATASET_MAX_CONCURRENCY") or "").strip()
    try:
        value = int(raw) if raw else 5
    except Exception:
        value = 5
    return max(1, min(20, value))


def _build_generation_messages(item_input: Any) -> list[dict[str, str]]:
    """Normalize dataset item input into chat-completion messages."""
    if isinstance(item_input, dict):
        maybe_messages = item_input.get("messages")
        if isinstance(maybe_messages, list):
            messages: list[dict[str, str]] = []
            for message in maybe_messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "user")
                content = _to_text(message.get("content"))
                if content:
                    messages.append({"role": role, "content": content})
            if messages:
                return messages

        for key in ("query", "question", "prompt", "input", "message", "text"):
            if key in item_input and item_input.get(key) is not None:
                return [{"role": "user", "content": _to_text(item_input.get(key))}]

    return [{"role": "user", "content": _to_text(item_input) or ""}]


async def _call_openai_generation_completion(
    *,
    model_candidates: list[str],
    model_api_key: str | None,
    messages: list[dict[str, str]],
) -> tuple[str, str]:
    """Call OpenAI-compatible chat completion for generation with retries."""
    if openai is None:
        raise RuntimeError("OpenAI SDK not available")

    last_error: Exception | None = None
    for candidate_model in model_candidates:
        request_model = _model_name_for_openai_fallback(candidate_model)
        api_base = _resolve_api_base_for_model(candidate_model)
        api_key = _resolve_openai_fallback_api_key(candidate_model, explicit_api_key=model_api_key)
        if not api_key:
            env_names = ", ".join(_candidate_api_key_env_names(candidate_model))
            raise RuntimeError(
                f"No API key resolved for generation model '{candidate_model}'. "
                f"Provide model_api_key or set one of: {env_names}"
            )

        try:
            if hasattr(openai, "AsyncOpenAI"):
                client_kwargs: dict[str, Any] = {}
                if api_key:
                    client_kwargs["api_key"] = api_key
                if api_base:
                    client_kwargs["base_url"] = api_base
                async_client = openai.AsyncOpenAI(**client_kwargs)
                try:
                    resp = await async_client.chat.completions.create(
                        model=request_model,
                        messages=messages,
                    )
                finally:
                    close_func = getattr(async_client, "close", None)
                    if callable(close_func):
                        try:
                            await close_func()
                        except Exception:
                            pass
            elif hasattr(openai, "OpenAI"):
                client_kwargs = {}
                if api_key:
                    client_kwargs["api_key"] = api_key
                if api_base:
                    client_kwargs["base_url"] = api_base

                def _sync_call_v1():
                    sync_client = openai.OpenAI(**client_kwargs)
                    try:
                        return sync_client.chat.completions.create(
                            model=request_model,
                            messages=messages,
                        )
                    finally:
                        close_func = getattr(sync_client, "close", None)
                        if callable(close_func):
                            try:
                                close_func()
                            except Exception:
                                pass

                resp = await asyncio.to_thread(_sync_call_v1)
            else:
                if api_key:
                    openai.api_key = api_key
                if api_base:
                    openai.api_base = api_base
                chat_completion = getattr(openai, "ChatCompletion", None)
                if chat_completion and hasattr(chat_completion, "acreate"):
                    resp = await chat_completion.acreate(
                        model=request_model,
                        messages=messages,
                    )
                elif chat_completion and hasattr(chat_completion, "create"):

                    def _sync_call_legacy():
                        return chat_completion.create(
                            model=request_model,
                            messages=messages,
                        )

                    resp = await asyncio.to_thread(_sync_call_legacy)
                else:
                    raise RuntimeError("OpenAI SDK does not expose a supported chat completion API")

            content = _extract_openai_chat_content(resp)
            return content, request_model
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Generation call failed for model={}: {}", candidate_model, str(exc))
            if _is_openai_retryable_model_error(exc):
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Generation call failed without response")


async def _dataset_generate_with_model(
    *,
    model: str,
    model_api_key: str | None,
    item_input: Any,
    model_registry_id: str | None = None,
) -> tuple[Any, str]:
    """Generate output for one dataset item via Model Service."""
    if not model_registry_id:
        raise RuntimeError("model_registry_id is required for dataset generation")

    messages = _build_generation_messages(item_input)
    system_prompt = ""
    user_prompt = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
        elif msg.get("role") == "user":
            user_prompt = msg.get("content", "")

    logger.info("Dataset generation: using Model Service for registry_id={}", model_registry_id)
    content = await _call_model_service_completion(
        model_registry_id=model_registry_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    if not content:
        raise RuntimeError(
            f"Model Service returned no response for registry_id={model_registry_id}"
        )
    return content, f"registry:{model_registry_id}"


async def _dataset_llm_evaluate(
    *,
    criteria: str,
    model: str,
    model_api_key: str | None,
    item_input: Any,
    output: Any,
    expected_output: Any,
    model_registry_id: str | None = None,
) -> tuple[float, str, str]:
    """Run LLM-as-a-judge for one dataset item output and return normalized score."""
    query_text = _to_text(item_input) or "[EMPTY]"
    generation_text = _to_text(output) or "[EMPTY]"
    ground_truth_text = _to_text(expected_output) or "[NOT PROVIDED]"
    rendered_criteria = _render_dataset_judge_criteria(
        criteria=criteria,
        query=item_input,
        generation=output,
        ground_truth=expected_output,
    )

    system_prompt = (
        "You are an impartial AI judge evaluating an assistant output. "
        "Given criteria, query, generation, and ground truth, assign a score between 0 and 5 inclusive. "
        "Return JSON with keys score_0_5 and reason."
    )
    user_prompt = f"""### Criteria
{rendered_criteria}

### Input
Query: {query_text}
Generation: {generation_text}
Ground Truth: {ground_truth_text}

Respond ONLY with valid JSON:
{{
  "score_0_5": number,
  "reason": "short explanation"
}}
"""

    if not model_registry_id:
        raise RuntimeError("model_registry_id is required for dataset LLM judge evaluation")

    logger.info("Dataset LLM judge: using Model Service for registry_id={}", model_registry_id)
    content = await _call_model_service_completion(
        model_registry_id=model_registry_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    if not content:
        raise RuntimeError(
            f"Model Service returned no response for registry_id={model_registry_id}"
        )
    used_model = f"registry:{model_registry_id}"

    payload = str(content or "").strip()
    if payload.startswith("```json"):
        payload = payload[7:]
    if payload.startswith("```"):
        payload = payload[3:]
    if payload.endswith("```"):
        payload = payload[:-3]
    payload = payload.strip()
    decoded = json.loads(payload)
    raw_score = decoded.get("score_0_5", decoded.get("score", 0))
    score_0_5 = max(0.0, min(5.0, float(raw_score)))
    reason = str(decoded.get("reason") or "No reason provided")
    return score_0_5 / 5.0, reason, used_model


def _list_all_datasets_for_user(client: Any, user_id: str, *, max_rows: int = 500) -> list[Any]:
    """Fetch datasets and apply best-effort user scoping."""
    collected: list[Any] = []
    if hasattr(client, "api") and hasattr(client.api, "datasets") and hasattr(client.api.datasets, "list"):
        page = 1
        page_size = min(100, max_rows)
        while len(collected) < max_rows:
            try:
                response = client.api.datasets.list(page=page, limit=page_size)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dataset list via api.datasets.list failed at page {}: {}", page, str(exc))
                break

            rows, _ = _parse_paginated_response(response)
            if not rows:
                break
            collected.extend(rows)
            if len(rows) < page_size:
                break
            page += 1

    filtered = [dataset for dataset in collected if _dataset_owned_by_user(dataset, user_id)]
    filtered.sort(
        key=lambda dataset: _parse_trace_timestamp(get_attr(dataset, "created_at", "createdAt", default=None))
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return filtered[:max_rows]


def _list_all_datasets_for_scope(
    client: Any,
    allowed_user_ids: set[str],
    *,
    max_rows: int = 500,
    current_user: Any | None = None,
    org_ids: set[UUID] | None = None,
    dept_pairs: list[tuple[UUID, UUID]] | None = None,
) -> list[Any]:
    """Fetch datasets and apply scope-based user + visibility filtering."""
    collected: list[Any] = []
    if hasattr(client, "api") and hasattr(client.api, "datasets") and hasattr(client.api.datasets, "list"):
        page = 1
        page_size = min(100, max_rows)
        while len(collected) < max_rows:
            try:
                response = client.api.datasets.list(page=page, limit=page_size)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dataset list via api.datasets.list failed at page {}: {}", page, str(exc))
                break
            rows, _ = _parse_paginated_response(response)
            if not rows:
                break
            collected.extend(rows)
            if len(rows) < page_size:
                break
            page += 1

    filtered = [
        dataset for dataset in collected
        if _dataset_accessible_by_users(
            dataset, allowed_user_ids,
            current_user=current_user, org_ids=org_ids, dept_pairs=dept_pairs,
        )
    ]
    filtered.sort(
        key=lambda dataset: _parse_trace_timestamp(get_attr(dataset, "created_at", "createdAt", default=None))
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return filtered[:max_rows]


def _fetch_dataset_items_page(client: Any, dataset_name: str, page: int, limit: int) -> tuple[list[Any], int]:
    """Fetch one page of dataset items using SDK-compatible APIs."""
    if hasattr(client, "api") and hasattr(client.api, "dataset_items") and hasattr(client.api.dataset_items, "list"):
        response = client.api.dataset_items.list(dataset_name=dataset_name, page=page, limit=limit)
        rows, total = _parse_paginated_response(response)
        return rows, int(total) if total is not None else len(rows)

    dataset = client.get_dataset(dataset_name)
    rows = list(getattr(dataset, "items", []) or [])
    total_rows = len(rows)
    start = (page - 1) * limit
    return rows[start:start + limit], total_rows


def _fetch_dataset_runs_page(client: Any, dataset_name: str, page: int, limit: int) -> tuple[list[Any], int]:
    """Fetch one page of dataset runs using SDK-compatible APIs."""
    if hasattr(client, "get_dataset_runs"):
        response = client.get_dataset_runs(dataset_name=dataset_name, page=page, limit=limit)
        rows, total = _parse_paginated_response(response)
        return rows, int(total) if total is not None else len(rows)

    if hasattr(client, "api") and hasattr(client.api, "datasets") and hasattr(client.api.datasets, "get_runs"):
        response = client.api.datasets.get_runs(dataset_name=dataset_name, page=page, limit=limit)
        rows, total = _parse_paginated_response(response)
        return rows, int(total) if total is not None else len(rows)

    return [], 0


def _find_dataset_run_by_id(
    client: Any,
    *,
    dataset_name: str,
    run_id: str,
    max_scan: int = 1000,
) -> Any | None:
    """Find dataset run object by id using paginated scans."""
    page_size = min(100, max_scan)
    scanned = 0
    page = 1
    while scanned < max_scan:
        rows, _ = _fetch_dataset_runs_page(client, dataset_name, page, page_size)
        if not rows:
            break
        for row in rows:
            if str(get_attr(row, "id", default="") or "") == str(run_id):
                return row
        scanned += len(rows)
        if len(rows) < page_size:
            break
        page += 1
    return None


def _fetch_dataset_run_items(
    client: Any,
    *,
    dataset_name: str,
    dataset_id: str | None,
    run_name: str,
    item_limit: int,
) -> list[Any]:
    """Fetch run items for a dataset run."""
    if hasattr(client, "api") and hasattr(client.api, "datasets") and hasattr(client.api.datasets, "get_run"):
        try:
            run_with_items = client.api.datasets.get_run(dataset_name=dataset_name, run_name=run_name)
            run_items = get_attr(run_with_items, "dataset_run_items", "datasetRunItems", default=[]) or []
            return list(run_items)[:item_limit]
        except Exception as exc:  # noqa: BLE001
            logger.debug("api.datasets.get_run failed for dataset={} run={}: {}", dataset_name, run_name, str(exc))

    if (
        dataset_id
        and hasattr(client, "api")
        and hasattr(client.api, "dataset_run_items")
        and hasattr(client.api.dataset_run_items, "list")
    ):
        page = 1
        page_size = min(100, item_limit)
        collected: list[Any] = []
        while len(collected) < item_limit:
            response = client.api.dataset_run_items.list(
                dataset_id=dataset_id,
                run_name=run_name,
                page=page,
                limit=page_size,
            )
            rows, _ = _parse_paginated_response(response)
            if not rows:
                break
            collected.extend(rows)
            if len(rows) < page_size:
                break
            page += 1
        return collected[:item_limit]

    return []


def _fetch_all_dataset_items(
    client: Any,
    *,
    dataset_name: str,
    max_rows: int = 5000,
) -> list[Any]:
    """Fetch dataset items across pages with a hard safety cap."""
    collected: list[Any] = []
    page = 1
    page_size = min(100, max_rows)
    while len(collected) < max_rows:
        rows, _ = _fetch_dataset_items_page(client, dataset_name, page, page_size)
        if not rows:
            break
        collected.extend(rows)
        if len(rows) < page_size:
            break
        page += 1
    return collected[:max_rows]


def _fetch_all_dataset_runs(
    client: Any,
    *,
    dataset_name: str,
    max_rows: int = 2000,
) -> list[Any]:
    """Fetch dataset runs across pages with a hard safety cap."""
    collected: list[Any] = []
    page = 1
    page_size = min(100, max_rows)
    while len(collected) < max_rows:
        rows, _ = _fetch_dataset_runs_page(client, dataset_name, page, page_size)
        if not rows:
            break
        collected.extend(rows)
        if len(rows) < page_size:
            break
        page += 1
    return collected[:max_rows]


def _fetch_dataset_item_by_id(client: Any, item_id: str) -> Any | None:
    """Fetch one dataset item by id if SDK exposes a direct getter."""
    if hasattr(client, "api") and hasattr(client.api, "dataset_items") and hasattr(client.api.dataset_items, "get"):
        try:
            return client.api.dataset_items.get(id=item_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("dataset_items.get failed for item_id={}: {}", item_id, str(exc))
    return None


def _delete_dataset_item(client: Any, item_id: str) -> None:
    """Delete one dataset item via SDK-compatible APIs."""
    if hasattr(client, "api") and hasattr(client.api, "dataset_items") and hasattr(client.api.dataset_items, "delete"):
        client.api.dataset_items.delete(id=item_id)
        return
    raise RuntimeError("Dataset item deletion is not supported by current Langfuse SDK")


def _delete_dataset_run(client: Any, *, dataset_name: str, run_name: str) -> None:
    """Delete one dataset run via SDK-compatible APIs."""
    if hasattr(client, "delete_dataset_run"):
        client.delete_dataset_run(dataset_name=dataset_name, run_name=run_name)
        return
    if hasattr(client, "api") and hasattr(client.api, "datasets") and hasattr(client.api.datasets, "delete_run"):
        client.api.datasets.delete_run(dataset_name=dataset_name, run_name=run_name)
        return
    raise RuntimeError("Dataset run deletion is not supported by current Langfuse SDK")


def _delete_dataset_container(client: Any, dataset_name: str) -> bool:
    """Delete a dataset container across SDK variants."""
    attempts: list[Any] = []

    if hasattr(client, "delete_dataset"):
        attempts.append(lambda: client.delete_dataset(dataset_name=dataset_name))

    if hasattr(client, "api") and hasattr(client.api, "datasets"):
        datasets_api = client.api.datasets
        if hasattr(datasets_api, "delete"):
            attempts.append(lambda: datasets_api.delete(dataset_name=dataset_name))
        if hasattr(datasets_api, "delete_dataset"):
            attempts.append(lambda: datasets_api.delete_dataset(dataset_name=dataset_name))

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            attempt()
            return True
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if last_error is not None:
        logger.debug("Dataset container delete not available for '{}': {}", dataset_name, str(last_error))
    return False

_KNOWN_LITELLM_PROVIDERS = {
    "openai",
    "azure",
    "anthropic",
    "groq",
    "gemini",
    "google",
    "vertex_ai",
    "bedrock",
    "openrouter",
    "mistral",
    "cohere",
    "huggingface",
    "ollama",
    "togetherai",
    "fireworks_ai",
    "xai",
    "replicate",
    "perplexity",
    "sambanova",
    "deepseek",
    "watsonx",
}


def _split_known_provider_prefix(model: str) -> tuple[str | None, str]:
    """Split known LiteLLM provider prefix from model string if present."""
    value = str(model or "").strip()
    if not value or "/" not in value:
        return None, value
    prefix, tail = value.split("/", 1)
    if prefix.strip().lower() in _KNOWN_LITELLM_PROVIDERS:
        return prefix.strip().lower(), tail.strip()
    return None, value


def _infer_litellm_provider(model: str, api_key: str | None = None) -> str | None:
    """Infer LiteLLM provider from model/API key/environment."""
    explicit_provider = os.getenv("LITELLM_DEFAULT_PROVIDER")
    if explicit_provider:
        value = explicit_provider.strip().lower()
        if value:
            return value

    prefixed_provider, _ = _split_known_provider_prefix(model)
    if prefixed_provider:
        return prefixed_provider

    key = str(api_key or "").strip()
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("sk-or-"):
        return "openrouter"
    if key.startswith("hf_"):
        return "huggingface"
    if key.startswith("xai-"):
        return "xai"
    if key.startswith("AIza"):
        return "gemini"
    if key.startswith("sk-"):
        return "openai"

    model_lower = str(model or "").strip().lower()
    if not model_lower:
        return None
    if model_lower.startswith(("gpt-", "o1", "o3", "o4", "text-embedding-")):
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "gemini" in model_lower:
        return "gemini"
    return None


def _normalize_model_name_for_provider(model_name: str, provider_hint: str | None = None) -> str:
    """Normalize human-entered model names into provider-friendly ids."""
    value = str(model_name or "").strip()
    if not value:
        return value

    lowered = value.lower()
    if lowered.startswith("models/"):
        value = value.split("/", 1)[1].strip()

    provider = (provider_hint or "").strip().lower()
    if provider in {"gemini", "google", "vertex_ai"}:
        # Gemini/Google model ids are slug-like (e.g. gemini-2.5-flash-lite).
        value = value.replace("_", "-")
        value = re.sub(r"\s+", "-", value.strip())
        value = re.sub(r"[^A-Za-z0-9._:\-]", "-", value)
        value = re.sub(r"-{2,}", "-", value).strip("-").lower()
        return value

    # Generic cleanup for obvious display labels containing spaces.
    if " " in value:
        compact = re.sub(r"\s+", "-", value.strip())
        compact = re.sub(r"[^A-Za-z0-9._:/\-]", "-", compact)
        compact = re.sub(r"-{2,}", "-", compact).strip("-")
        if compact:
            return compact
    return value


def _build_model_name_variants(model_name: str, provider_hint: str | None = None) -> list[str]:
    """Build ordered model-name variants from a user-entered model field."""
    raw = str(model_name or "").strip()
    if not raw:
        return []

    variants: list[str] = []
    normalized = _normalize_model_name_for_provider(raw, provider_hint)
    if normalized and normalized != raw:
        variants.append(normalized)
    variants.append(raw)

    if "/" in raw:
        _, tail = raw.split("/", 1)
        tail = tail.strip()
        if tail:
            tail_normalized = _normalize_model_name_for_provider(tail, provider_hint)
            if tail_normalized and tail_normalized not in variants:
                variants.append(tail_normalized)
            if tail not in variants:
                variants.append(tail)

    deduped: list[str] = []
    seen: set[str] = set()
    for value in variants:
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _build_litellm_model_candidates(model: str, api_key: str | None = None) -> list[str]:
    """Build model candidates for LiteLLM retries with provider inference."""
    raw = str(model or "").strip()
    if not raw:
        return []

    provider, raw_tail = _split_known_provider_prefix(raw)
    inferred_provider = _infer_litellm_provider(raw, api_key)
    provider_hint = provider or inferred_provider
    candidates: list[str] = []

    def add_candidate(value: str) -> None:
        value = value.strip()
        if value and value not in candidates:
            candidates.append(value)

    if provider:
        tail_variants = _build_model_name_variants(raw_tail, provider_hint=provider)
        for tail in tail_variants:
            add_candidate(f"{provider}/{tail}")
        add_candidate(raw)
    else:
        name_variants = _build_model_name_variants(raw, provider_hint=provider_hint)
        if inferred_provider:
            for name in name_variants:
                add_candidate(f"{inferred_provider}/{name}")
        for name in name_variants:
            add_candidate(name)

    # OpenAI-compatible fallback for custom model names when base URL is configured.
    openai_base = os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not provider and openai_base:
        for name in _build_model_name_variants(raw, provider_hint="openai"):
            add_candidate(f"openai/{name}")

    return candidates


def _is_litellm_provider_error(exc: Exception) -> bool:
    """Return True when LiteLLM error indicates provider/model prefix mismatch."""
    message = str(exc).lower()
    return (
        "llm provider not provided" in message
        or "pass in the llm provider" in message
        or "provider not found" in message
        or "unknown provider" in message
    )


def _is_litellm_retryable_model_error(exc: Exception) -> bool:
    """Return True when retrying with alternate model candidates may succeed."""
    if _is_litellm_provider_error(exc):
        return True

    message = str(exc).lower()
    return (
        "unexpected model name format" in message
        or "generatecontentrequest.model" in message
        or "invalid_argument" in message
        or "invalid model" in message
        or "model not found" in message
        or "unknown model" in message
    )


def _resolve_api_base_for_model(model: str) -> str | None:
    """Resolve provider-specific API base from environment when available."""
    provider, _ = _split_known_provider_prefix(model)
    # Global override first.
    global_base = os.getenv("LITELLM_API_BASE")
    if global_base:
        return global_base

    if provider == "openai":
        return os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if provider in ("azure", "azure_openai"):
        return os.getenv("AZURE_OPENAI_API_BASE") or os.getenv("AZURE_API_BASE") or os.getenv("AZURE_OPENAI_ENDPOINT")
    if provider == "groq":
        return os.getenv("GROQ_API_BASE_URL") or os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_BASE_URL")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_BASE_URL") or "https://openrouter.ai/api/v1"
    if provider in {"gemini", "google", "vertex_ai"}:
        return (
            os.getenv("GEMINI_API_BASE_URL")
            or os.getenv("GOOGLE_API_BASE_URL")
            or os.getenv("VERTEX_API_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    return None


def _resolve_openai_fallback_api_key(model: str, explicit_api_key: str | None = None) -> str | None:
    """Resolve API key for OpenAI SDK fallback based on model/provider."""
    value = str(explicit_api_key or "").strip()
    if value:
        return value

    provider = _infer_litellm_provider(model)
    env_by_provider: dict[str, list[str]] = {
        "openai": ["OPENAI_API_KEY"],
        "azure": ["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "vertex_ai": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
    }

    env_names = env_by_provider.get(provider or "", [])
    if not env_names:
        env_names = ["OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"]
    for env_name in env_names:
        env_value = str(os.getenv(env_name) or "").strip()
        if env_value:
            return env_value
    return None


def _candidate_api_key_env_names(model: str) -> list[str]:
    """Return likely API key environment variables for the model provider."""
    provider = _infer_litellm_provider(model)
    mapping: dict[str, list[str]] = {
        "openai": ["OPENAI_API_KEY"],
        "azure": ["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "vertex_ai": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
    }
    return mapping.get(provider or "", ["OPENAI_API_KEY"])


def _model_name_for_openai_fallback(model: str) -> str:
    """Strip provider prefixes (e.g. 'groq/') for OpenAI-compatible SDK calls."""
    _, tail = _split_known_provider_prefix(model)
    value = tail.strip() if tail else str(model or "").strip()
    return value or str(model or "").strip()


def _is_openai_retryable_model_error(exc: Exception) -> bool:
    """Return True when retrying with another model candidate may succeed."""
    message = str(exc).lower()
    return (
        "model not found" in message
        or "unknown model" in message
        or "invalid model" in message
        or "does not exist" in message
        or "unexpected model name format" in message
        or "llm provider not provided" in message
        or "provider not found" in message
        or "generatecontentrequest.model" in message
    )


def _is_openai_response_format_error(exc: Exception) -> bool:
    """Return True when provider rejects JSON response_format."""
    message = str(exc).lower()
    return "response_format" in message or "json_object" in message


def _extract_openai_chat_content(resp: Any) -> str:
    """Extract message content from OpenAI chat completion response variants."""
    choices = resp.get("choices", []) if isinstance(resp, dict) else getattr(resp, "choices", [])
    if not choices:
        raise RuntimeError("OpenAI judge returned no choices")

    first_choice = choices[0]
    if isinstance(first_choice, dict):
        message = first_choice.get("message", {}) or {}
        content = message.get("content")
    else:
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None

    if isinstance(content, list):
        # Some OpenAI-compatible providers return structured content blocks.
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if text:
                text_parts.append(str(text))
        content = "".join(text_parts)

    if content is None:
        raise RuntimeError("OpenAI judge returned empty content")
    return str(content)


async def _call_openai_judge_completion(
    *,
    model_candidates: list[str],
    model_api_key: str | None,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, str]:
    """Call OpenAI SDK (v1/v2 or legacy) with retries across model candidates."""
    if openai is None:
        raise RuntimeError("OpenAI SDK not available")

    last_error: Exception | None = None
    for candidate_model in model_candidates:
        request_model = _model_name_for_openai_fallback(candidate_model)
        api_base = _resolve_api_base_for_model(candidate_model)
        api_key = _resolve_openai_fallback_api_key(candidate_model, explicit_api_key=model_api_key)
        if not api_key:
            env_names = ", ".join(_candidate_api_key_env_names(candidate_model))
            raise RuntimeError(
                f"No API key resolved for judge model '{candidate_model}'. "
                f"Provide model_api_key or set one of: {env_names}"
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            if hasattr(openai, "AsyncOpenAI"):
                client_kwargs: dict[str, Any] = {}
                if api_key:
                    client_kwargs["api_key"] = api_key
                if api_base:
                    client_kwargs["base_url"] = api_base
                async_client = openai.AsyncOpenAI(**client_kwargs)
                try:
                    resp = await async_client.chat.completions.create(
                        model=request_model,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                except Exception as response_format_exc:
                    if not _is_openai_response_format_error(response_format_exc):
                        raise
                    resp = await async_client.chat.completions.create(
                        model=request_model,
                        messages=messages,
                    )
                finally:
                    close_func = getattr(async_client, "close", None)
                    if callable(close_func):
                        try:
                            await close_func()
                        except Exception:
                            pass
            elif hasattr(openai, "OpenAI"):
                client_kwargs = {}
                if api_key:
                    client_kwargs["api_key"] = api_key
                if api_base:
                    client_kwargs["base_url"] = api_base

                def _sync_call_v1():
                    sync_client = openai.OpenAI(**client_kwargs)
                    try:
                        try:
                            return sync_client.chat.completions.create(
                                model=request_model,
                                messages=messages,
                                response_format={"type": "json_object"},
                            )
                        except Exception as response_format_exc:
                            if not _is_openai_response_format_error(response_format_exc):
                                raise
                            return sync_client.chat.completions.create(
                                model=request_model,
                                messages=messages,
                            )
                    finally:
                        close_func = getattr(sync_client, "close", None)
                        if callable(close_func):
                            try:
                                close_func()
                            except Exception:
                                pass

                resp = await asyncio.to_thread(_sync_call_v1)
            else:
                if api_key:
                    openai.api_key = api_key
                if api_base:
                    openai.api_base = api_base

                chat_completion = getattr(openai, "ChatCompletion", None)
                if chat_completion and hasattr(chat_completion, "acreate"):
                    resp = await chat_completion.acreate(
                        model=request_model,
                        messages=messages,
                    )
                elif chat_completion and hasattr(chat_completion, "create"):
                    def _sync_call_legacy():
                        return chat_completion.create(
                            model=request_model,
                            messages=messages,
                        )

                    resp = await asyncio.to_thread(_sync_call_legacy)
                else:
                    raise RuntimeError("OpenAI SDK does not expose a supported chat completion API")

            content = _extract_openai_chat_content(resp)
            return content, request_model
        except Exception as e:
            last_error = e
            logger.warning("OpenAI judge call failed for model={}: {}", candidate_model, str(e))
            if _is_openai_retryable_model_error(e):
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("OpenAI judge call failed without a response")


def _submit_score_to_langfuse(
    client: Any,
    *,
    trace_id: str,
    name: str,
    value: float,
    comment: str | None = None,
    observation_id: str | None = None,
    source: str | None = None,
    user_id: str | None = None,
) -> None:
    """Submit score using SDK-compatible method across Langfuse versions."""
    payload: Dict[str, Any] = {
        "trace_id": trace_id,
        "name": name,
        "value": value,
        "comment": comment,
    }
    if observation_id:
        payload["observation_id"] = observation_id
    if source:
        payload["source"] = source
    if user_id:
        payload["user_id"] = user_id

    # Build payload variants to handle different SDK versions that may not accept
    # all kwargs (e.g., older SDKs don't accept user_id, source, observation_id).
    optional_keys = {"source", "observation_id", "user_id"}
    present_optional = {k for k in optional_keys if k in payload}
    payload_variants = [payload]
    # Add a variant without optional keys for compatibility
    if present_optional:
        payload_variants.append({k: v for k, v in payload.items() if k not in present_optional})

    def _call_with_compatible_kwargs(func) -> bool:
        last_error: TypeError | None = None
        for kwargs in payload_variants:
            try:
                func(**kwargs)
                return True
            except TypeError as exc:
                last_error = exc
        if last_error:
            raise last_error
        return False

    # v2-style helper
    if hasattr(client, "score"):
        if _call_with_compatible_kwargs(client.score):
            logger.info(f"Score submitted via client.score(): trace_id={trace_id}, name={name}, value={value}")
            return

    # v3-style helper
    if hasattr(client, "create_score"):
        if _call_with_compatible_kwargs(client.create_score):
            logger.info(f"Score submitted via client.create_score(): trace_id={trace_id}, name={name}, value={value}")
            return

    # Direct API fallbacks (SDK internals)
    if hasattr(client, "api"):
        api_obj = getattr(client, "api")
        for attr in ("score", "scores"):
            score_api = getattr(api_obj, attr, None)
            if score_api and hasattr(score_api, "create"):
                if _call_with_compatible_kwargs(score_api.create):
                    logger.info(f"Score submitted via api.{attr}.create(): trace_id={trace_id}, name={name}, value={value}")
                    return

    if hasattr(client, "client") and hasattr(client.client, "scores"):
        scores_client = client.client.scores
        if hasattr(scores_client, "create"):
            if _call_with_compatible_kwargs(scores_client.create):
                logger.info(f"Score submitted via client.client.scores.create(): trace_id={trace_id}, name={name}, value={value}")
                return

    raise RuntimeError("No supported score submission method found on Langfuse client")


def _extract_trace_run_id(trace_dict: Dict[str, Any]) -> str | None:
    """Extract run identifier from trace metadata/tags."""
    metadata = trace_dict.get("metadata")
    if isinstance(metadata, dict):
        run_id = metadata.get("run_id") or metadata.get("runId")
        if run_id:
            return str(run_id)

    tags = trace_dict.get("tags") or []
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("run_id:"):
                value = tag.split(":", 1)[1].strip()
                if value:
                    return value
    return None


def _extract_trace_user_id(trace_dict: Dict[str, Any]) -> str | None:
    """Extract user id from top-level fields or metadata/tags."""
    user_id = trace_dict.get("user_id")
    if user_id:
        return str(user_id)

    metadata = trace_dict.get("metadata")
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            metadata = parsed if isinstance(parsed, dict) else metadata
        except Exception:
            pass
    if isinstance(metadata, dict):
        value = (
            metadata.get("user_id")
            or metadata.get("userId")
            or metadata.get("app_user_id")
            or metadata.get("created_by_user_id")
            or metadata.get("owner_user_id")
        )
        if value:
            return str(value)

    tags = trace_dict.get("tags") or []
    if isinstance(tags, list):
        for tag in tags:
            if not isinstance(tag, str):
                continue
            for prefix in ("user_id:", "app_user_id:", "created_by_user_id:"):
                if tag.startswith(prefix):
                    value = tag.split(":", 1)[1].strip()
                    if value:
                        return value
    return None


def _normalize_agent_id(agent_id: str | None) -> str | None:
    """Normalize agent identifiers from UI/API inputs."""
    if not agent_id:
        return None
    value = str(agent_id).strip()
    if value.startswith("lb:"):
        value = value.split("lb:", 1)[1]
    return value or None


def _normalize_targets(target: Union[str, List[str], None]) -> List[str]:
    """Normalize target input to a lowercase list."""
    if target is None:
        return ["existing"]
    if isinstance(target, str):
        values = [target]
    elif isinstance(target, list):
        values = target
    else:
        values = []
    normalized = [str(t).strip().lower() for t in values if str(t).strip()]
    return list(dict.fromkeys(normalized))


def _normalize_agent_ids(agent_ids: Optional[List[str]]) -> List[str]:
    """Normalize and de-duplicate agent ids."""
    if not agent_ids:
        return []
    normalized: List[str] = []
    for agent_id in agent_ids:
        value = _normalize_agent_id(agent_id)
        if value:
            normalized.append(value)
    return list(dict.fromkeys(normalized))


def _extract_trace_agent_id(trace_dict: Dict[str, Any]) -> str | None:
    """Extract agent id from trace metadata/tags."""
    metadata = trace_dict.get("metadata")
    if isinstance(metadata, dict):
        agent_id = _normalize_agent_id(metadata.get("agent_id") or metadata.get("agentId"))
        if agent_id:
            return agent_id

    tags = trace_dict.get("tags") or []
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("agent_id:"):
                agent_id = _normalize_agent_id(tag.split(":", 1)[1])
                if agent_id:
                    return agent_id
    return None


def _extract_trace_agent_name(trace_dict: Dict[str, Any]) -> str | None:
    """Extract agent name from trace metadata/tags/name."""
    metadata = trace_dict.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("agent_name") or metadata.get("agentName")
        if value:
            return str(value)

    tags = trace_dict.get("tags") or []
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("agent_name:"):
                return tag.split(":", 1)[1]

    name = trace_dict.get("name")
    return str(name) if name else None


def _extract_trace_project_name(trace_dict: Dict[str, Any]) -> str | None:
    """Extract project name from trace metadata/tags."""
    metadata = trace_dict.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("project_name") or metadata.get("projectName")
        if value:
            return str(value)

    tags = trace_dict.get("tags") or []
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("project_name:"):
                return tag.split(":", 1)[1]
    return None


def _parse_trace_timestamp(value: Any) -> datetime | None:
    """Parse trace timestamp from datetime/string/epoch variants."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        epoch = float(value)
        # Handle millisecond epoch values.
        if epoch > 10_000_000_000:
            epoch = epoch / 1000.0
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    try:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _trace_matches_evaluator_filters(
    trace_dict: Dict[str, Any],
    *,
    trace_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    agent_ids: Optional[List[str]] = None,
    agent_name: str | None = None,
    project_name: str | None = None,
    ts_from: datetime | None = None,
    ts_to: datetime | None = None,
) -> bool:
    """Check whether a trace satisfies evaluator filters."""
    trace_trace_id = str(trace_dict.get("id") or "")
    trace_run_id = _extract_trace_run_id(trace_dict)
    trace_session_id = str(trace_dict.get("session_id") or "")
    trace_agent_id = _extract_trace_agent_id(trace_dict)
    trace_agent_name = _extract_trace_agent_name(trace_dict) or ""
    trace_project_name = _extract_trace_project_name(trace_dict) or ""
    trace_ts = _parse_trace_timestamp(trace_dict.get("timestamp"))

    normalized_agent_id = _normalize_agent_id(agent_id)
    normalized_agent_ids = set(_normalize_agent_ids(agent_ids))

    if trace_id and str(trace_id) not in {trace_trace_id, str(trace_run_id or "")}:
        return False
    if session_id and str(session_id) != trace_session_id:
        return False

    # Strict agent filtering: if filters are present and trace does not expose a matching agent_id, reject.
    if normalized_agent_id:
        if not trace_agent_id or trace_agent_id != normalized_agent_id:
            return False
    if normalized_agent_ids:
        if not trace_agent_id or trace_agent_id not in normalized_agent_ids:
            return False

    if agent_name and agent_name.lower() not in trace_agent_name.lower():
        return False
    if project_name and project_name.lower() not in trace_project_name.lower():
        return False

    if ts_from and trace_ts and trace_ts < ts_from:
        return False
    if ts_to and trace_ts and trace_ts > ts_to:
        return False
    if (ts_from or ts_to) and trace_ts is None:
        return False

    return True


def _parse_iso_datetime_or_400(value: str | None, field_name: str) -> datetime | None:
    """Parse ISO datetime and raise HTTP 400 on invalid values."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}. Use ISO datetime format.")


def _fetch_trace_by_id(client, trace_id: str) -> Any | None:
    """Fetch a trace by id using available SDK methods."""
    if not trace_id:
        return None

    fetch_methods = [
        ("fetch_trace", lambda: client.fetch_trace(trace_id)),
        ("client.traces.get", lambda: client.client.traces.get(trace_id)),
        ("api.trace.get", lambda: client.api.trace.get(trace_id)),
    ]

    for method_name, method in fetch_methods:
        try:
            if method_name == "fetch_trace" and not hasattr(client, "fetch_trace"):
                continue
            if method_name == "client.traces.get":
                if not (hasattr(client, "client") and hasattr(client.client, "traces")):
                    continue
            if method_name == "api.trace.get":
                if not (hasattr(client, "api") and hasattr(client.api, "trace")):
                    continue

            response = method()
            if response is None:
                continue
            return response.data if hasattr(response, "data") else response
        except Exception as e:
            logger.debug(
                "Trace fetch via {} failed for trace_id={}: {}",
                method_name,
                trace_id,
                str(e),
            )

    return None


def _choose_trace_candidate(
    traces: List[Any],
    *,
    requested_trace_id: str,
    user_id: str,
    session_id: str | None = None,
    agent_id: str | None = None,
    agent_name: str | None = None,
    project_name: str | None = None,
    timestamp: datetime | None = None,
) -> Dict[str, Any] | None:
    """Choose the best matching trace from a list using contextual scoring."""
    normalized_agent_id = _normalize_agent_id(agent_id)
    requested_session = str(session_id) if session_id else None
    requested_agent_name = str(agent_name).lower() if agent_name else None
    requested_project_name = str(project_name).lower() if project_name else None
    requested_user_id = str(user_id) if user_id else None

    scored_candidates: List[tuple[float, Dict[str, Any]]] = []
    for raw_trace in traces or []:
        trace_dict = parse_trace_data(raw_trace)
        candidate_trace_id = str(trace_dict.get("id") or "")
        if not candidate_trace_id:
            continue

        candidate_user_id = str(_extract_trace_user_id(trace_dict) or "")
        candidate_session_id = str(trace_dict.get("session_id") or "")
        candidate_agent_id = _extract_trace_agent_id(trace_dict)
        candidate_agent_name = (_extract_trace_agent_name(trace_dict) or "").lower()
        candidate_project_name = (_extract_trace_project_name(trace_dict) or "").lower()
        candidate_run_id = _extract_trace_run_id(trace_dict)
        candidate_ts = _parse_trace_timestamp(trace_dict.get("timestamp"))

        # Keep user isolation strict if the trace payload includes user id.
        if requested_user_id and candidate_user_id and candidate_user_id != requested_user_id:
            continue

        # Exclude explicit conflicts for session/agent when candidate provides those fields.
        if requested_session and candidate_session_id and candidate_session_id != requested_session:
            continue
        if normalized_agent_id and candidate_agent_id and candidate_agent_id != normalized_agent_id:
            continue
        if requested_project_name and candidate_project_name and requested_project_name not in candidate_project_name:
            continue

        score = 0.0
        if requested_trace_id and candidate_trace_id == requested_trace_id:
            score += 500.0
        if requested_trace_id and candidate_run_id and str(candidate_run_id) == requested_trace_id:
            score += 450.0
        if requested_session and candidate_session_id == requested_session:
            score += 140.0
        if normalized_agent_id and candidate_agent_id == normalized_agent_id:
            score += 120.0
        if requested_agent_name and requested_agent_name in candidate_agent_name:
            score += 50.0
        if requested_project_name and requested_project_name in candidate_project_name:
            score += 35.0

        if timestamp and candidate_ts:
            delta_seconds = abs((candidate_ts - timestamp).total_seconds())
            if delta_seconds <= 2:
                score += 100.0
            elif delta_seconds <= 10:
                score += 80.0
            elif delta_seconds <= 30:
                score += 60.0
            elif delta_seconds <= 120:
                score += 40.0
            elif delta_seconds <= 600:
                score += 15.0
            else:
                score -= 20.0

        scored_candidates.append((score, trace_dict))

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_trace = scored_candidates[0]

    # Confidence gate to avoid evaluating the wrong trace.
    enough_context = bool(
        (session_id and str(best_trace.get("session_id") or "") == str(session_id))
        or (
            normalized_agent_id
            and _extract_trace_agent_id(best_trace)
            and _extract_trace_agent_id(best_trace) == normalized_agent_id
        )
        or (
            requested_trace_id
            and (
                str(best_trace.get("id") or "") == requested_trace_id
                or _extract_trace_run_id(best_trace) == requested_trace_id
            )
        )
    )
    if not enough_context and best_score < 180.0:
        return None

    return best_trace


async def _resolve_trace_for_judge(
    client,
    *,
    trace_id: str,
    user_id: str,
    session_id: str | None = None,
    agent_id: str | None = None,
    agent_name: str | None = None,
    project_name: str | None = None,
    timestamp: datetime | None = None,
    max_attempts: int = 8,
) -> tuple[str | None, Dict[str, Any] | None]:
    """Resolve the canonical Langfuse trace id and trace payload for judging."""
    trace_id = str(trace_id)
    resolved_timestamp = _parse_trace_timestamp(timestamp) or datetime.now(timezone.utc)

    for attempt in range(1, max_attempts + 1):
        # Step 1: direct fetch by id (fast-path for existing traces)
        fetched_trace = _fetch_trace_by_id(client, trace_id)
        if fetched_trace:
            trace_dict = parse_trace_data(fetched_trace)
            fetched_id = str(trace_dict.get("id") or trace_id)
            # Langfuse's trace.user_id is now a username string; the app UUID
            # lives in metadata.user_uuid. extract_trace_user_ids reads both.
            trace_user_ids = extract_trace_user_ids(fetched_trace)
            if not trace_user_ids or str(user_id) in trace_user_ids:
                return fetched_id, trace_dict
            logger.warning(
                f"Resolved trace {fetched_id} belongs to different user; expected={user_id}, got={trace_user_ids}"
            )

        # Step 2: fallback lookup by context among user traces.
        window_minutes = min(2 + (attempt * 3), 30)
        from_ts = resolved_timestamp - timedelta(minutes=window_minutes)
        to_ts = resolved_timestamp + timedelta(minutes=window_minutes)
        try:
            traces = fetch_traces_from_langfuse(
                client,
                user_id=str(user_id),
                limit=200,
                from_timestamp=from_ts,
                to_timestamp=to_ts,
            )
        except Exception as e:
            logger.debug(
                "Context trace lookup failed for trace_ref={} attempt={}/{}: {}",
                trace_id,
                attempt,
                max_attempts,
                str(e),
            )
            traces = []

        candidate = _choose_trace_candidate(
            traces,
            requested_trace_id=trace_id,
            user_id=str(user_id),
            session_id=session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            project_name=project_name,
            timestamp=resolved_timestamp,
        )
        if candidate:
            return str(candidate.get("id")), candidate

        # final wide search without strict time window in case ingestion lag is high
        if attempt == max_attempts:
            try:
                traces = fetch_traces_from_langfuse(
                    client,
                    user_id=str(user_id),
                    limit=400,
                )
                candidate = _choose_trace_candidate(
                    traces,
                    requested_trace_id=trace_id,
                    user_id=str(user_id),
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    project_name=project_name,
                    timestamp=resolved_timestamp,
                )
                if candidate:
                    return str(candidate.get("id")), candidate
            except Exception as e:
                logger.debug("Wide trace lookup failed for trace_ref={}: {}", trace_id, str(e))

        await asyncio.sleep(min(2.0 * attempt, 10.0))

    return None, None


async def _call_model_service_completion(
    model_registry_id: str,
    system_prompt: str,
    user_prompt: str,
) -> str | None:
    """Call the Model microservice using a registry model ID.

    This is the canonical way to invoke any registered model (OpenAI, Azure,
    Anthropic, Google, Groq, etc.) — the microservice handles all provider-
    specific logic (Azure deployment names, API versions, base URLs, etc.).

    Returns the response content string or None if the service is unavailable.
    """
    from agentcore.services.model_service_client import (
        is_service_configured,
        _get_model_service_settings,
        _headers,
    )

    if not is_service_configured():
        return None

    url, api_key = _get_model_service_settings()

    payload = {
        "provider": "openai",  # placeholder — overridden by registry resolution
        "model": "",  # resolved from registry by the service
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "provider_config": {"registry_model_id": model_registry_id},
        "response_format": {"type": "json_object"},
        "stream": False,
    }

    try:
        import httpx
        async with httpx.AsyncClient(timeout=300.0) as http_client:
            resp = await http_client.post(
                f"{url}/v1/chat/completions",
                headers=_headers(api_key),
                json=payload,
            )
            resp.raise_for_status()

        data = resp.json()
        if data.get("choices"):
            return data["choices"][0].get("message", {}).get("content", "")
        return None
    except Exception as e:
        logger.warning("Model service completion failed for registry_id={}: {}", model_registry_id, str(e))
        return None


async def run_llm_judge_task(
    client,
    trace_id: str,
    criteria: str,
    score_name: str,
    model: str,
    user_id: str,
    model_api_key: str | None = None,
    model_api_base: str | None = None,
    model_registry_id: str | None = None,
    preset_id: str | None = None,
    ground_truth: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    agent_name: str | None = None,
    project_name: str | None = None,
    timestamp: datetime | None = None,
    trace_input: Any | None = None,
    trace_output: Any | None = None,
):
    """Background task to run LLM judge via Model Service."""
    if not model_registry_id:
        logger.error("No model_registry_id provided, cannot run judge for trace_ref={}", trace_id)
        return

    try:
        resolved_trace_id = str(trace_id)

        # If trace input/output were passed directly (new-trace evaluations),
        # use them immediately — no need to fetch from Langfuse.
        if trace_input is not None or trace_output is not None:
            logger.info(f"Using directly-provided trace data for evaluation: trace_ref={trace_id}")
            trace_input = trace_input or ''
            trace_output = trace_output or ''
        else:
            # Fetch from Langfuse (for manual/existing-trace evaluations)
            await asyncio.sleep(5)
            logger.info(
                f"Resolving trace for evaluation: trace_ref={trace_id}, session_id={session_id}, agent_id={agent_id}"
            )
            _, trace_dict = await _resolve_trace_for_judge(
                client,
                trace_id=str(trace_id),
                user_id=str(user_id),
                session_id=session_id,
                agent_id=agent_id,
                agent_name=agent_name,
                project_name=project_name,
                timestamp=timestamp,
                max_attempts=6,
            )
            if not trace_dict:
                logger.error(
                    f"Judge failed: could not resolve trace for trace_ref={trace_id}, "
                    f"user_id={user_id}, session_id={session_id}, agent_id={agent_id}"
                )
                return
            resolved_trace_id = str(trace_dict.get("id") or trace_id)
            if resolved_trace_id != str(trace_id):
                logger.info(f"Resolved trace_ref={trace_id} to canonical trace_id={resolved_trace_id}")
            trace_input = trace_dict.get('input', '')
            trace_output = trace_dict.get('output', '')

        # Convert to string — trace data may be Message objects, dicts, lists, etc.
        def _to_str(val: Any) -> str:
            if val is None:
                return ''
            if isinstance(val, str):
                return val
            # Handle Message-like objects (LangChain, custom)
            if hasattr(val, 'content'):
                return str(val.content)
            if hasattr(val, 'text'):
                return str(val.text)
            # Handle dicts/lists
            try:
                return json.dumps(val, default=str)
            except Exception:
                return str(val)

        trace_input = _to_str(trace_input)
        trace_output = _to_str(trace_output)

        # 2. Construct Prompt
        if ground_truth:
            system_prompt = (
                "You are an impartial AI judge evaluating an AI assistant's interaction. "
                "You will be given the Input (User Query), the Output (AI Response), and the Ground Truth (Expected Answer). "
                "Your task is to evaluate the Output against the Ground Truth based strictly on the provided Criteria. "
                "Provide a score between 0.0 (worst) and 1.0 (perfect) and explain your reasoning."
            )
        else:
            system_prompt = (
                "You are an impartial AI judge evaluating an AI assistant's interaction. "
                "You will be given the Input (User Query) and the Output (AI Response). "
                "Your task is to evaluate the Output based strictly on the provided Criteria. "
                "Provide a score between 0.0 (worst) and 1.0 (perfect) and explain your reasoning."
            )
        
        if ground_truth:
            user_prompt = f"""### Criteria
{criteria}

### Input
{trace_input}

### Ground Truth (Expected Answer)
{ground_truth}

### Output (Actual Response)
{trace_output}

### Instructions
Evaluate the Output by comparing it to the Ground Truth based on the Criteria.
Provide a numeric score between 0 and 5 inclusive (0 worst, 5 best).
Respond with a JSON object containing:
- "score_0_5": A number between 0 and 5.
- "reason": A concise explanation of your scoring (1-3 sentences).

Respond ONLY with valid JSON, no markdown formatting."""
        else:
            user_prompt = f"""### Criteria
{criteria}

### Input
{trace_input}

### Output
{trace_output}

### Instructions
Evaluate the Output based on the Criteria.
Provide a numeric score between 0 and 5 inclusive (0 worst, 5 best).
Respond with a JSON object containing:
- "score_0_5": A number between 0 and 5.
- "reason": A concise explanation of your scoring (1-3 sentences).

Respond ONLY with valid JSON, no markdown formatting."""

        # 3. Call LLM via Model Service
        logger.info(f"Calling LLM judge via Model Service: registry_id={model_registry_id}")
        content = await _call_model_service_completion(
            model_registry_id=model_registry_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        used_model = f"registry:{model_registry_id}"

        # Clean up markdown code blocks if present
        if not content:
            logger.error(f"Judge failed: LLM returned empty response for trace_ref={trace_id}")
            return
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        result = json.loads(content)

        # Extract 0-5 score and reason
        raw_score = result.get("score_0_5") if result.get("score_0_5") is not None else result.get("score")
        try:
            score_0_5 = float(raw_score)
        except Exception:
            score_0_5 = 0.0
        reason = result.get("reason", "No reason provided")

        # Clamp to 0-5
        score_0_5 = max(0.0, min(5.0, score_0_5))

        # Normalize to 0-1 for Langfuse value field but include raw in comment
        normalized_value = score_0_5 / 5.0

        # 4. Submit Score to Langfuse
        logger.info(f"Submitting score to Langfuse: {score_name} raw={score_0_5} normalized={normalized_value}")
        score_comment_data = {
            "criteria": criteria,
            "model": used_model,
            "requested_model": model,
            "reason": reason,
            "source": "llm-judge",
            "raw_score_0_5": score_0_5,
            "preset_id": preset_id,
            "requested_trace_id": str(trace_id),
            "resolved_trace_id": str(resolved_trace_id),
        }
        if ground_truth:
            score_comment_data["ground_truth"] = ground_truth
        score_comment = json.dumps(score_comment_data)
        _submit_score_to_langfuse(
            client,
            trace_id=str(resolved_trace_id),
            name=score_name,
            value=normalized_value,
            comment=score_comment,
            user_id=str(user_id),
        )
        
        # Flush to ensure immediate send
        if hasattr(client, "flush"):
            client.flush()

        # Wait briefly for Langfuse to index the score, then verify
        await asyncio.sleep(3)
        try:
            from agentcore.api.observability.parsing import fetch_scores_for_trace
            verification_scores = fetch_scores_for_trace(client, str(resolved_trace_id), limit=50)
            found = any(
                getattr(s, 'name', '') == score_name or str(getattr(s, 'name', '')) == score_name
                for s in (verification_scores or [])
            )
            logger.info(
                f"Score verification for trace {resolved_trace_id}: "
                f"found={found}, total_scores={len(verification_scores or [])}, "
                f"score_names={[getattr(s, 'name', '') for s in (verification_scores or [])]}"
            )
        except Exception as verify_err:
            logger.debug(f"Score verification failed: {verify_err}")

        # Invalidate score cache so the UI picks up the new score immediately
        for cache_key in [k for k in list(_SCORE_LIST_CACHE) if k.startswith(f"{user_id}|")]:
            _SCORE_LIST_CACHE.pop(cache_key, None)
        for cache_key in [k for k in list(_PENDING_REVIEWS_CACHE) if k.startswith(f"{user_id}|")]:
            _PENDING_REVIEWS_CACHE.pop(cache_key, None)

        logger.info(
            f"Judge completed for trace_ref={trace_id}, trace_id={resolved_trace_id}: "
            f"{score_name}={normalized_value}"
        )

    except json.JSONDecodeError as e:
        logger.error("LLM Judge JSON parsing error for trace_ref={}: {}", trace_id, str(e))
        logger.error("Response content: {}", content)
    except Exception as e:
        logger.error("LLM Judge error for trace_ref={}: {}", trace_id, str(e))


async def _resolve_agent_payload_for_experiment(
    *,
    agent_id: str | None,
    current_user: User,
) -> dict[str, Any] | None:
    """Resolve agent payload for dataset experiment task execution."""
    normalized_agent_id = _normalize_agent_id(agent_id)
    if not normalized_agent_id:
        return None
    try:
        agent_uuid = UUID(normalized_agent_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid agent_id for experiment run")

    async with session_scope() as session:
        agent_obj = await session.get(agent, agent_uuid)
        if not agent_obj:
            raise HTTPException(status_code=404, detail=f"Agent {normalized_agent_id} not found")
        if str(agent_obj.user_id) != str(current_user.id) and agent_obj.access_type != AccessTypeEnum.PUBLIC:
            raise HTTPException(status_code=403, detail="You do not have access to this agent")
        if agent_obj.data is None:
            raise HTTPException(status_code=400, detail="Selected agent has no data payload")

        return {
            "id": str(agent_obj.id),
            "name": agent_obj.name or str(agent_obj.id),
            "data": agent_obj.data,
        }


async def _resolve_experiment_judge_config(
    *,
    current_user: User,
    evaluator_config_id: str | None,
    preset_id: str | None,
    evaluator_name: str | None,
    criteria: str | None,
    judge_model: str | None,
    judge_model_api_key: str | None,
    judge_model_registry_id: str | None = None,
    session: Any = None,
) -> dict[str, Any]:
    """Resolve dataset experiment judge settings from optional saved evaluator."""
    judge_name = (evaluator_name or "").strip() or "Dataset LLM Judge"
    resolved_criteria = (criteria or "").strip() or None
    resolved_model = (judge_model or "").strip() or None
    resolved_api_key = (judge_model_api_key or "").strip() or None
    resolved_preset_id = (preset_id or "").strip() or None

    # Resolve judge model from registry if provided
    if judge_model_registry_id:
        reg_model, reg_key, _reg_base = await _resolve_model_from_registry(judge_model_registry_id, session=session)
        if not resolved_model:
            resolved_model = reg_model
        if not resolved_api_key:
            resolved_api_key = reg_key

    if evaluator_config_id:
        try:
            config_uuid = UUID(evaluator_config_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid evaluator_config_id")

        if session is not None:
            evaluator = await session.get(Evaluator, config_uuid)
            if not evaluator or str(evaluator.user_id) != str(current_user.id):
                raise HTTPException(status_code=404, detail="Evaluator config not found")
        else:
            async with session_scope() as new_session:
                evaluator = await new_session.get(Evaluator, config_uuid)
                if not evaluator or str(evaluator.user_id) != str(current_user.id):
                    raise HTTPException(status_code=404, detail="Evaluator config not found")

        judge_name = evaluator.name or judge_name
        if not resolved_criteria:
            resolved_criteria = (evaluator.criteria or "").strip() or None
        if not resolved_model and evaluator.model_registry_id:
            reg_model, reg_key, _reg_base = await _resolve_model_from_registry(evaluator.model_registry_id, session=session)
            resolved_model = reg_model
            if not resolved_api_key:
                resolved_api_key = reg_key
        elif not resolved_model:
            resolved_model = (evaluator.model or "").strip() or None
        if not resolved_preset_id:
            resolved_preset_id = str(evaluator.preset_id) if evaluator.preset_id else None

    preset = get_preset_by_id(resolved_preset_id)
    if resolved_preset_id and resolved_preset_id != "__custom__" and preset is None:
        raise HTTPException(status_code=400, detail=f"Unknown preset_id '{resolved_preset_id}'")

    # Optional convenience: allow evaluator_name to match a preset id/name.
    if not preset and judge_name:
        normalized_name = judge_name.strip().lower()
        for candidate in EVALUATION_PRESETS:
            candidate_id = str(candidate.get("id") or "").strip().lower()
            candidate_name = str(candidate.get("name") or "").strip().lower()
            if normalized_name and normalized_name in {candidate_id, candidate_name}:
                preset = candidate
                resolved_preset_id = str(candidate.get("id"))
                break

    if preset:
        if not resolved_criteria:
            resolved_criteria = str(preset.get("criteria") or "").strip() or None
        if not (evaluator_name or "").strip():
            judge_name = str(preset.get("name") or judge_name)

    if resolved_criteria:
        resolved_criteria = _ensure_dataset_prompt_template(resolved_criteria)

    return {
        "judge_name": judge_name,
        "criteria": resolved_criteria,
        "model": resolved_model,
        "model_api_key": resolved_api_key,
        "preset_id": resolved_preset_id,
        "requires_ground_truth": bool(preset and preset.get("requires_ground_truth")),
    }


async def _run_dataset_experiment_async(
    *,
    client: Any,
    dataset_name: str,
    dataset_id: str,
    experiment_name: str,
    description: str | None,
    user_id: str,
    agent_payload: dict[str, Any] | None,
    generation_model: str | None,
    generation_model_api_key: str | None,
    generation_model_registry_id: str | None = None,
    judge_name: str | None,
    judge_preset_id: str | None,
    judge_criteria: str | None,
    judge_model: str | None,
    judge_model_api_key: str | None,
    judge_model_registry_id: str | None = None,
) -> dict[str, Any]:
    """Run dataset experiment: load items from local DB, run agent/model, evaluate, store results."""
    dataset_uuid = UUID(dataset_id)

    # 1. Load dataset items from local DB
    async with session_scope() as db:
        data_items = (await db.exec(
            select(DatasetItem)
            .where(DatasetItem.dataset_id == dataset_uuid, DatasetItem.status == "ACTIVE")
            .order_by(DatasetItem.created_at.asc())
        )).all()

    if not data_items:
        raise RuntimeError(f"Dataset '{dataset_name}' has no items")

    has_expected_outputs = any(item.expected_output is not None for item in data_items)
    llm_metric_name = f"llm_judge:{judge_name or 'judge'}"

    # 2. Create DatasetRun in local DB
    async with session_scope() as db:
        dataset_row = await db.get(Dataset, dataset_uuid)
        run = DatasetRun(
            dataset_id=dataset_uuid,
            name=experiment_name,
            description=description,
            metadata_={
                "source": "agentcore-evaluation-datasets",
                "user_id": user_id,
                "dataset_name": dataset_name,
                "dataset_visibility": dataset_row.visibility if dataset_row else "private",
                "dataset_public_scope": dataset_row.public_scope if dataset_row else None,
                "dataset_org_id": str(dataset_row.org_id) if dataset_row and dataset_row.org_id else None,
                "dataset_dept_id": str(dataset_row.dept_id) if dataset_row and dataset_row.dept_id else None,
                "dataset_public_dept_ids": [str(v) for v in (dataset_row.public_dept_ids or [])] if dataset_row else [],
            },
            user_id=UUID(user_id),
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)
        run_id = run.id

    # 3. Process each item
    metric_buckets: dict[str, list[float]] = defaultdict(list)
    run_items_to_save: list[DatasetRunItem] = []

    for item in data_items:
        item_input = item.input
        expected_output = item.expected_output
        output = None
        trace_id = None
        item_scores: list[dict[str, Any]] = []

        # 3a. Run task (agent or model generation)
        tracer_client = None
        try:
            if agent_payload:
                session_id = f"dataset:{dataset_name}:{item.id}"
                output, trace_id, tracer_client = await _run_dataset_item_with_agent(
                    agent_payload=agent_payload,
                    user_id=str(user_id),
                    item_input=item_input,
                    session_id=session_id,
                )
            elif generation_model or generation_model_registry_id:
                generated_output, _ = await _dataset_generate_with_model(
                    model=generation_model or "",
                    model_api_key=generation_model_api_key,
                    item_input=item_input,
                    model_registry_id=generation_model_registry_id,
                )
                output = generated_output
                # Create a Langfuse trace for model generation experiments
                if client:
                    try:
                        gen_trace_id = str(uuid4())
                        if hasattr(client, "trace"):
                            client.trace(
                                id=gen_trace_id,
                                name=f"experiment:{experiment_name}",
                                input=item_input if isinstance(item_input, (str, dict)) else str(item_input),
                                output=output if isinstance(output, (str, dict)) else str(output),
                                user_id=str(user_id),
                                session_id=f"dataset:{dataset_name}",
                                metadata={"source": "dataset-experiment", "dataset": dataset_name, "experiment": experiment_name},
                            )
                        trace_id = gen_trace_id
                    except Exception:
                        pass
            else:
                output = "[ERROR] No agent or generation model configured"
        except Exception as exc:
            logger.warning("Experiment task failed for item {}: {}", item.id, str(exc))
            output = f"[TASK_ERROR] {exc}"

        # Normalize output for evaluation
        output_str = output if isinstance(output, str) else json.dumps(output, default=str) if output else ""
        input_str = item_input if isinstance(item_input, str) else json.dumps(item_input, default=str) if item_input else ""
        expected_str = expected_output if isinstance(expected_output, str) else json.dumps(expected_output, default=str) if expected_output else None

        # 3b. Run evaluators
        # Exact match
        if has_expected_outputs:
            if expected_output is not None:
                expected_norm = _normalize_for_exact_match(expected_output)
                output_norm = _normalize_for_exact_match(output)
                is_match = expected_norm == output_norm
                score_val = 1.0 if is_match else 0.0
                comment = "Exact match" if is_match else "Output differs from expected output"
            else:
                score_val = 0.0
                comment = "No expected output configured for this dataset item."
            item_scores.append({"name": "exact_match", "value": score_val, "source": "evaluator", "comment": comment})
            metric_buckets["exact_match"].append(score_val)

        # LLM judge
        if judge_criteria and (judge_model or judge_model_registry_id):
            try:
                value, reason, used_model = await _dataset_llm_evaluate(
                    criteria=judge_criteria,
                    model=judge_model or "",
                    model_api_key=judge_model_api_key,
                    item_input=input_str,
                    output=output_str,
                    expected_output=expected_str,
                    model_registry_id=judge_model_registry_id,
                )
                judge_comment = json.dumps({"reason": reason, "model": used_model, "criteria": judge_criteria})
                item_scores.append({"name": llm_metric_name, "value": float(value), "source": "llm-judge", "comment": judge_comment})
                metric_buckets[llm_metric_name].append(float(value))
            except Exception as exc:
                logger.debug("LLM evaluator failed for experiment={}: {}", experiment_name, str(exc))
                item_scores.append({"name": llm_metric_name, "value": 0.0, "source": "llm-judge", "comment": f"LLM evaluator error: {exc}"})
                metric_buckets[llm_metric_name].append(0.0)

        # 3c. Submit scores to Langfuse if we have a trace_id
        # Use the tracer's client (same Langfuse project as the trace) if available
        score_client = tracer_client or client
        if trace_id and score_client:
            for s in item_scores:
                try:
                    _submit_score_to_langfuse(
                        score_client,
                        trace_id=trace_id,
                        name=s["name"],
                        value=s["value"],
                        comment=s.get("comment"),
                    )
                except Exception:
                    pass
            if hasattr(score_client, "flush"):
                try:
                    score_client.flush()
                except Exception:
                    pass

        # 3d. Store output as dict for JSON column
        output_dict = output if isinstance(output, dict) else {"text": str(output)} if output else None

        run_items_to_save.append(DatasetRunItem(
            run_id=run_id,
            dataset_item_id=item.id,
            trace_id=trace_id,
            output=output_dict,
            scores=item_scores,
        ))

    # 4. Final flush of Langfuse client to ensure all traces/scores are sent
    if client:
        try:
            if hasattr(client, "flush"):
                client.flush()
        except Exception:
            pass
    # Also flush the global Langfuse client (used by tracing service)
    try:
        from agentcore.services.tracing.service import TracingService
        global_lf = get_langfuse_client()
        if global_lf and hasattr(global_lf, "flush"):
            global_lf.flush()
    except Exception:
        pass

    # 5. Save all run items to DB
    async with session_scope() as db:
        db.add_all(run_items_to_save)
        await db.commit()

    # 6. Build metrics summary
    metrics_summary: dict[str, dict[str, Any]] = {}
    for metric_name, values in metric_buckets.items():
        if not values:
            continue
        metrics_summary[metric_name] = {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "dataset_run_id": str(run_id),
        "run_name": experiment_name,
        "item_count": len(data_items),
        "metrics": metrics_summary,
    }


async def _run_dataset_experiment_job(
    *,
    job_id: str,
    client: Any,
    dataset_name: str,
    dataset_id: str = "",
    experiment_name: str,
    description: str | None,
    user_id: str,
    agent_payload: dict[str, Any] | None,
    generation_model: str | None,
    generation_model_api_key: str | None,
    generation_model_registry_id: str | None = None,
    judge_name: str | None,
    judge_preset_id: str | None,
    judge_criteria: str | None,
    judge_model: str | None,
    judge_model_api_key: str | None,
    judge_model_registry_id: str | None = None,
) -> None:
    """Background task runner for dataset experiments."""
    _set_dataset_experiment_job(
        job_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    try:
        result_payload = await _run_dataset_experiment_async(
            client=client,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            experiment_name=experiment_name,
            description=description,
            user_id=user_id,
            agent_payload=agent_payload,
            generation_model=generation_model,
            generation_model_api_key=generation_model_api_key,
            generation_model_registry_id=generation_model_registry_id,
            judge_name=judge_name,
            judge_preset_id=judge_preset_id,
            judge_criteria=judge_criteria,
            judge_model=judge_model,
            judge_model_api_key=judge_model_api_key,
            judge_model_registry_id=judge_model_registry_id,
        )
        _set_dataset_experiment_job(
            job_id,
            status="completed",
            finished_at=datetime.now(timezone.utc),
            result=result_payload,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.opt(exception=True).error(
            "Dataset experiment job failed: job_id={}, dataset={}, error={}",
            job_id,
            dataset_name,
            str(exc),
        )
        _set_dataset_experiment_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc),
            error=str(exc),
        )


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/status")
async def get_status(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
) -> Dict[str, Any]:
    """Check if evaluation features are available."""
    from agentcore.api.observability import _get_langfuse_client_for_binding

    # Check env-var client first
    client = get_langfuse_client()
    langfuse_available = client is not None

    # If no env-var client, check if user has any active Langfuse bindings
    if not langfuse_available:
        try:
            scope = await resolve_observability_scope(
                session,
                current_user=current_user,
                enforce_filter_for_admin=False,
            )
            if scope.bindings:
                for binding in scope.bindings:
                    try:
                        binding_client = _get_langfuse_client_for_binding(binding)
                        if binding_client:
                            langfuse_available = True
                            break
                    except Exception:
                        continue
        except Exception:
            pass

    return {
        "langfuse_available": langfuse_available,
        "llm_judge_available": LITELLM_AVAILABLE or OPENAI_AVAILABLE,
        "user_id": str(current_user.id)
    }


@router.get("/scores")
async def get_scores(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    page: Annotated[int, Query(ge=1)] = 1,
    trace_id: Annotated[str | None, Query()] = None,
    name: Annotated[str | None, Query()] = None,
    org_id: Annotated[UUID | None, Query()] = None,
    dept_id: Annotated[UUID | None, Query()] = None,
    environment: Annotated[str | None, Query(description="'uat' or 'production'")] = None,
) -> Dict[str, Any]:
    """
    List evaluation scores visible to the current user (scope-aware).
    Uses allowed_user_ids from RBAC scope resolution.
    """
    # Resolve scope-aware langfuse clients and allowed user IDs. Super_admin
    # has both the org_admin project (where their own scores live) and every
    # department project in their org in scope, so we query all of them and
    # merge results.
    allowed_user_ids, clients = await _get_scoped_langfuse_clients_for_evaluation(
        session, current_user, org_id=org_id, dept_id=dept_id
    )
    if not clients:
        env_client = get_langfuse_client()
        if env_client:
            clients = [env_client]
    if not clients:
        raise HTTPException(status_code=503, detail="Langfuse not configured.")
    # Preserve `client` for back-compat with code paths below that may still
    # reference a single client (e.g. logging).
    client = clients[0]
    logger.info(
        f"get_scores: user={current_user.id}, env={environment}, page={page}, limit={limit}, "
        f"num_clients={len(clients)}"
    )

    try:
        user_id = str(current_user.id)
        trace_id = str(trace_id).strip() if trace_id and str(trace_id).strip() else None
        name = str(name).strip() if name and str(name).strip() else None
        scope_key = f"{user_id}|{str(org_id) if org_id else ''}|{str(dept_id) if dept_id else ''}"
        score_cache_key = f"{scope_key}|{page}|{limit}|{trace_id or ''}|{(name or '').lower()}|{environment or ''}"
        now_mono = time.monotonic()
        cached_score_payload: dict[str, Any] | None = None
        cached_score_entry = _SCORE_LIST_CACHE.get(score_cache_key)
        if cached_score_entry:
            cached_age = now_mono - float(cached_score_entry.get("ts", 0))
            cached_payload = cached_score_entry.get("payload")
            if isinstance(cached_payload, dict):
                # Fast path: serve any cached result (including empty) within TTL.
                # Empty results use a shorter TTL (30s) so we re-check Langfuse quickly.
                is_empty_result = not cached_payload.get("items") and cached_payload.get("total", 0) == 0
                ttl = 30.0 if is_empty_result else _SCORE_LIST_CACHE_STALE_SECONDS
                if cached_age <= ttl:
                    return cached_payload
                # Stale but non-empty: keep as fallback in case fresh fetch returns empty.
                if not is_empty_result:
                    cached_score_payload = cached_payload

        trace_lookup: Dict[str, Dict[str, Any]] = {}

        def _extract_scores_payload(response: Any) -> tuple[list[Any], int | None]:
            if response is None:
                return [], None

            rows: list[Any] = []
            total_items: int | None = None
            if hasattr(response, "data"):
                rows = list(response.data or [])
                meta = getattr(response, "meta", None)
                if isinstance(meta, dict):
                    total_items = meta.get("total_items") or meta.get("total")
                elif meta is not None:
                    total_items = (
                        getattr(meta, "total_items", None)
                        or getattr(meta, "total", None)
                    )
            elif isinstance(response, dict):
                rows = list(response.get("data") or [])
                meta = response.get("meta")
                if isinstance(meta, dict):
                    total_items = meta.get("total_items") or meta.get("total")
            elif isinstance(response, list):
                rows = response
                total_items = len(rows)
            return rows, total_items

        def _list_scores_page_for_client(
            lf_client: Any, page_num: int, page_limit: int, *, include_user_filter: bool,
        ) -> tuple[list[Any], int | None]:
            if not (hasattr(lf_client, "client") and hasattr(lf_client.client, "scores")):
                # Try alternative score list methods
                if hasattr(lf_client, "api"):
                    api_obj = lf_client.api
                    for attr in ("score_v_2", "scores", "score"):
                        score_api = getattr(api_obj, attr, None)
                        if score_api and hasattr(score_api, "get"):
                            try:
                                kwargs: Dict[str, Any] = {"page": page_num, "limit": page_limit}
                                if trace_id:
                                    kwargs["trace_id"] = trace_id
                                response = score_api.get(**kwargs)
                                rows, total = _extract_scores_payload(response)
                                return rows, total
                            except Exception as e:
                                logger.debug(f"api.{attr}.get() failed: {e}")
                                continue
                        if score_api and hasattr(score_api, "list"):
                            try:
                                kwargs = {"page": page_num, "limit": page_limit}
                                if trace_id:
                                    kwargs["trace_id"] = trace_id
                                response = score_api.list(**kwargs)
                                rows, total = _extract_scores_payload(response)
                                return rows, total
                            except Exception as e:
                                logger.debug(f"api.{attr}.list() failed: {e}")
                                continue
                return [], None

            kwargs = {
                "page": page_num,
                "limit": page_limit,
            }
            if trace_id:
                kwargs["trace_id"] = trace_id
            if name:
                kwargs["name"] = name
            if include_user_filter:
                kwargs["user_id"] = user_id

            try:
                response = lf_client.client.scores.list(**kwargs)
            except TypeError:
                kwargs.pop("user_id", None)
                response = lf_client.client.scores.list(**kwargs)
            rows, total = _extract_scores_payload(response)
            return rows, total

        def _list_scores_page(page_num: int, page_limit: int, *, include_user_filter: bool) -> tuple[list[Any], int | None]:
            """Fetch the same page from every scoped client and merge by score id."""
            merged: dict[str, Any] = {}
            ordered: list[Any] = []
            total_across: int = 0
            for lf_client in clients:
                try:
                    rows, subtotal = _list_scores_page_for_client(
                        lf_client, page_num, page_limit, include_user_filter=include_user_filter,
                    )
                except Exception as e:
                    logger.debug(f"Score list per-client fetch failed: {e}")
                    continue
                for row in rows:
                    sid = str(get_attr(row, "id", default="") or "")
                    key = sid or f"{get_attr(row, 'trace_id', 'traceId', default='')}::{get_attr(row, 'name', default='')}::{get_attr(row, 'created_at', 'createdAt', 'timestamp', default='')}"
                    if key in merged:
                        continue
                    merged[key] = row
                    ordered.append(row)
                if subtotal is not None:
                    total_across += int(subtotal)
            logger.info(
                f"Score list merged across {len(clients)} client(s) (user_filter={include_user_filter}) "
                f"returned {len(ordered)} rows, total_across={total_across}"
            )
            return ordered, (total_across or None)

        def _score_matches_name(score_row: Any) -> bool:
            if not name:
                return True
            score_name = str(get_attr(score_row, "name", default="") or "")
            return name.lower() in score_name.lower()

        raw_scores: list[Any] = []
        total = 0

        # Primary fetch: scores in Langfuse are project-scoped, so fetch without
        # user_id filter (scores typically don't carry user_id). This avoids
        # the expensive fallback chain that was causing high latency.
        primary_rows, primary_total = _list_scores_page(page, limit, include_user_filter=False)
        primary_rows = [
            row for row in primary_rows
            if _score_matches_name(row)
        ]

        # If no results without user filter, try with user filter as secondary
        if not primary_rows:
            primary_rows, primary_total = _list_scores_page(page, limit, include_user_filter=True)
            primary_rows = [
                row for row in primary_rows
                if _score_matches_name(row)
            ]

        if primary_rows:
            raw_scores = primary_rows
            total = (
                int(primary_total)
                if primary_total is not None and len(primary_rows) > 0
                else len(primary_rows)
            )

        # Also fetch scores from local DB (dataset experiment run items)
        try:
            from agentcore.services.database.models.dataset_run_item.model import DatasetRunItem
            from agentcore.services.database.models.dataset_run.model import DatasetRun
            from agentcore.services.database.models.dataset.model import Dataset as DatasetModel

            async with session_scope() as local_db:
                # Build query for run items that have scores
                query = (
                    select(DatasetRunItem, DatasetRun, DatasetModel)
                    .join(DatasetRun, DatasetRunItem.run_id == DatasetRun.id)
                    .join(DatasetModel, DatasetRun.dataset_id == DatasetModel.id)
                    .where(DatasetRunItem.scores.isnot(None))
                )
                if trace_id:
                    query = query.where(DatasetRunItem.trace_id == trace_id)
                query = query.order_by(DatasetRunItem.created_at.desc()).limit(limit)

                results = (await local_db.exec(query)).all()

                for run_item, run, dataset in results:
                    if not run_item.scores:
                        continue
                    for score_entry in run_item.scores:
                        score_name = score_entry.get("name", "score")
                        if name and name.lower() not in score_name.lower():
                            continue
                        raw_scores.append({
                            "id": f"local:{run_item.id}:{score_name}",
                            "trace_id": run_item.trace_id or str(run_item.id),
                            "name": score_name,
                            "value": float(score_entry.get("value", 0.0)),
                            "source": score_entry.get("source", "API"),
                            "comment": score_entry.get("comment"),
                            "timestamp": run_item.created_at.isoformat() if run_item.created_at else None,
                            "_agent_name": f"experiment-item-run",
                            "_dataset_name": dataset.name if dataset else None,
                            "_run_name": run.name if run else None,
                        })
                        total += 1
        except Exception as local_err:
            logger.debug("Failed to fetch local DB scores: {}", str(local_err))

        # Parse to response model (including agent/agent name).
        # NOTE: We do NOT fetch traces individually per score — that caused extreme latency.
        # Instead, we extract agent name from the score's own metadata or trace_name field.
        items: list[ScoreResponse] = []
        for s in raw_scores:
            score_trace_id = str(get_attr(s, "trace_id", "traceId", default="") or "")

            # Extract agent name from score metadata without per-score Langfuse calls
            local_agent_name = get_attr(s, "_agent_name") if isinstance(s, dict) else None
            # Langfuse scores may carry trace name directly
            agent_name = str(get_attr(s, "trace_name", "traceName", default="") or "") or None
            if not agent_name:
                agent_name = str(local_agent_name) if local_agent_name else None

            source = get_attr(s, "source")
            if hasattr(source, "value"):
                source = source.value

            score_id = str(get_attr(s, "id", default="") or "")
            items.append(ScoreResponse(
                id=score_id or f"{score_trace_id}:{get_attr(s, 'name', default='score')}",
                trace_id=score_trace_id,
                agent_name=agent_name,
                name=str(get_attr(s, "name", default="Score") or "Score"),
                value=float(get_attr(s, "value", default=0.0) or 0.0),
                source=str(source) if source is not None else "API",
                comment=get_attr(s, "comment"),
                user_id=str(get_attr(s, "user_id", "userId")) if get_attr(s, "user_id", "userId") else None,
                created_at=get_attr(s, "timestamp", "createdAt", "created_at"),
                observation_id=get_attr(s, "observation_id", "observationId"),
                config_id=get_attr(s, "config_id", "configId"),
            ))

        response_payload = {
            "items": items,
            "total": total,
            "page": page,
            "limit": limit
        }
        # Always write to cache, including empty results.
        # Empty results use a short TTL (30s) so they are re-validated quickly.
        cache_payload = {
            "items": [item.model_dump() for item in items],
            "total": total,
            "page": page,
            "limit": limit,
        }
        _SCORE_LIST_CACHE[score_cache_key] = {
            "ts": now_mono,
            "payload": cache_payload,
        }
        if len(_SCORE_LIST_CACHE) > 512:
            oldest_key = min(
                _SCORE_LIST_CACHE.items(),
                key=lambda kv: float(kv[1].get("ts", 0)),
            )[0]
            _SCORE_LIST_CACHE.pop(oldest_key, None)

        # If fresh fetch returned empty but we have stale non-empty data, prefer stale.
        if not items and total == 0 and cached_score_payload and page == 1 and not trace_id and not name:
            logger.warning(
                "Using stale cached score payload for user_id={} after transient empty response",
                user_id,
            )
            return cached_score_payload

        return response_payload

    except Exception as e:
        logger.opt(exception=True).error("Error fetching scores: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create")
async def create_score(
    payload: CreateScoreRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
) -> Dict[str, str]:
    """
    Create a manual score (Annotation).
    """
    _, client = await _get_scoped_langfuse_for_evaluation(session, current_user)
    if not client:
        raise HTTPException(status_code=503, detail="Langfuse not configured.")

    try:
        user_id = str(current_user.id)
        
        # Verify trace belongs to user (optional but recommended)
        # For now we trust the frontend only shows user's traces
        
        _submit_score_to_langfuse(
            client,
            trace_id=payload.trace_id,
            observation_id=payload.observation_id,
            name=payload.name,
            value=payload.value,
            comment=payload.comment,
            source="ANNOTATION",
        )
        
        # Flush to ensure it sends
        if hasattr(client, "flush"):
            client.flush()

        # Invalidate score list and pending-reviews caches so the next fetch is fresh.
        for _k in [k for k in list(_SCORE_LIST_CACHE) if k.startswith(user_id + "|")]:
            _SCORE_LIST_CACHE.pop(_k, None)
        for _k in [k for k in list(_PENDING_REVIEWS_CACHE) if k.startswith(user_id + "|")]:
            _PENDING_REVIEWS_CACHE.pop(_k, None)

        logger.info(f"User {user_id} created score for trace {payload.trace_id}")
        
        return {"status": "success", "message": "Score created successfully"}

    except Exception as e:
        logger.opt(exception=True).error("Error creating score: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces/pending")
async def get_pending_reviews(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    trace_id: Annotated[Optional[str], Query()] = None,
    agent_name: Annotated[Optional[str], Query()] = None,
    session_id: Annotated[Optional[str], Query()] = None,
    user_id_filter: Annotated[Optional[str], Query()] = None,
    ts_from: Annotated[Optional[str], Query()] = None,
    ts_to: Annotated[Optional[str], Query()] = None,
    limit: int = 20,
    org_id: Annotated[UUID | None, Query()] = None,
    dept_id: Annotated[UUID | None, Query()] = None,
    environment: Annotated[str | None, Query(description="'uat' or 'production'")] = None,
) -> List[TraceForReview]:
    """
    Get recent traces that might need review (Annotation Queue).
    Returns traces visible within the user's RBAC scope.
    """
    # Resolve scope-aware langfuse client and allowed user IDs
    allowed_user_ids, client = await _get_scoped_langfuse_for_evaluation(
        session, current_user, org_id=org_id, dept_id=dept_id
    )
    if not client:
        raise HTTPException(status_code=503, detail="Langfuse not configured.")

    try:
        user_id = str(current_user.id)

        # Fast-path: serve cached pending reviews within TTL.
        # Filters bust the cache so only unfiltered requests are cached.
        _use_pending_cache = not any([trace_id, agent_name, session_id, user_id_filter, ts_from, ts_to])
        scope_key = f"{user_id}|{str(org_id) if org_id else ''}|{str(dept_id) if dept_id else ''}"
        _pending_cache_key = f"{scope_key}|{limit}|{environment or ''}"
        _now_mono = time.monotonic()
        if _use_pending_cache:
            _pending_entry = _PENDING_REVIEWS_CACHE.get(_pending_cache_key)
            if _pending_entry:
                _age = _now_mono - float(_pending_entry.get("ts", 0))
                if _age <= _PENDING_REVIEWS_CACHE_TTL_SECONDS:
                    return _pending_entry["payload"]

        # Fetch recent traces for all allowed users in scope
        fetch_limit = max(limit * 5, 100)
        traces_data = []
        for uid in allowed_user_ids:
            try:
                uid_traces = fetch_traces_from_langfuse(client, user_id=uid, limit=fetch_limit, environment=environment)
                traces_data.extend(uid_traces or [])
            except Exception as e:
                logger.warning("fetch_traces_from_langfuse failed for user_id={}: {}", uid, str(e))
        logger.info(f"Fetched {len(traces_data)} traces for {len(allowed_user_ids)} allowed user(s) (limit={fetch_limit})")

        # Fetch scores and count per trace. Langfuse's score.user_id mirrors
        # trace.user_id (now a username string, not the app UUID), so scoping
        # by user_id on the server side is unreliable. Instead fetch scores
        # once and filter to the trace ids we already resolved.
        score_counts = defaultdict(int)
        scoped_trace_ids: set[str] = {
            str(parse_trace_data(t).get('id') or '') for t in traces_data
        }
        scoped_trace_ids.discard('')
        if scoped_trace_ids and hasattr(client, 'client') and hasattr(client.client, 'scores'):
            try:
                scores_response = client.client.scores.list(limit=1000)
                all_scores = []
                if hasattr(scores_response, 'data'):
                    all_scores = scores_response.data
                elif isinstance(scores_response, list):
                    all_scores = scores_response
                for score in all_scores:
                    s_trace_id = get_attr(score, 'trace_id', 'traceId')
                    if s_trace_id and str(s_trace_id) in scoped_trace_ids:
                        score_counts[s_trace_id] += 1
            except Exception:
                pass

        # Get agent names from database for better context
        agent_query = select(agent).where(agent.user_id == current_user.id)
        db_agents = (await session.execute(agent_query)).scalars().all()
        agent_names = {str(agent.id): agent.name for agent in db_agents}

        # Apply filtering and build response
        result = []
        for t in traces_data:
            trace_dict = parse_trace_data(t)
            tid = trace_dict.get('id')
            if not tid:
                continue

            # filter by trace id exact match
            if trace_id and str(tid) != str(trace_id):
                continue

            # filter by agent name substring match (uses metadata or agent_id)
            metadata = trace_dict.get('metadata') or {}
            inferred_agent_name = None
            if isinstance(metadata, dict):
                inferred_agent_name = metadata.get('agent_name') or metadata.get('agentId') or metadata.get('agent_id')
            if not inferred_agent_name:
                # fallback to DB lookup using agent id stored in metadata
                agent_id = metadata.get('agent_id') if isinstance(metadata, dict) else None
                if agent_id:
                    inferred_agent_name = agent_names.get(str(agent_id))

            if agent_name:
                if not inferred_agent_name or agent_name.lower() not in str(inferred_agent_name).lower():
                    continue

            # filter by session id
            if session_id and str(trace_dict.get('session_id') or '').lower().find(session_id.lower()) < 0:
                continue

            # filter by user id — match against the UUID, the user_uuid
            # metadata, or the display username from the original trace.
            if user_id_filter:
                filter_l = user_id_filter.lower()
                candidates = [
                    str(trace_dict.get('user_id') or ''),
                    str(trace_dict.get('user_uuid') or ''),
                ]
                tmeta = trace_dict.get('metadata')
                if isinstance(tmeta, dict):
                    candidates.append(str(tmeta.get('user_id') or ''))
                if not any(filter_l in c.lower() for c in candidates if c):
                    continue

            # filter by timestamp range if provided (ISO format)
            if ts_from or ts_to:
                try:
                    ts_val = None
                    ts = trace_dict.get('timestamp')
                    if isinstance(ts, (int, float)):
                        ts_val = datetime.fromtimestamp(float(ts) / 1000.0, timezone.utc)
                    else:
                        try:
                            ts_val = datetime.fromisoformat(str(ts))
                        except Exception:
                            ts_val = None

                    if ts_val:
                        if ts_from:
                            try:
                                tfrom = datetime.fromisoformat(ts_from)
                                if ts_val < tfrom:
                                    continue
                            except Exception:
                                pass
                        if ts_to:
                            try:
                                tto = datetime.fromisoformat(ts_to)
                                if ts_val > tto:
                                    continue
                            except Exception:
                                pass
                except Exception:
                    pass

            score_count = score_counts.get(str(tid), 0)

            result.append(TraceForReview(
                id=str(tid),
                name=trace_dict.get('name'),
                timestamp=trace_dict.get('timestamp'),
                input=trace_dict.get('input'),
                output=trace_dict.get('output'),
                session_id=trace_dict.get('session_id'),
                agent_name=inferred_agent_name,
                has_scores=score_count > 0,
                score_count=score_count
            ))

            if len(result) >= limit:
                break

        # Cache unfiltered results so subsequent calls are served instantly.
        if _use_pending_cache:
            _PENDING_REVIEWS_CACHE[_pending_cache_key] = {"ts": _now_mono, "payload": result}
            # Evict oldest entry if cache exceeds 128 entries.
            if len(_PENDING_REVIEWS_CACHE) > 128:
                _oldest = min(_PENDING_REVIEWS_CACHE.items(), key=lambda kv: float(kv[1].get("ts", 0)))[0]
                _PENDING_REVIEWS_CACHE.pop(_oldest, None)

        return result

    except Exception as e:
        logger.opt(exception=True).error("Error fetching pending queue: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


def _can_access_dataset(
    dataset: Dataset,
    current_user,
    org_ids: set[UUID],
    dept_pairs: list[tuple[UUID, UUID]],
) -> bool:
    """Check if user can see this dataset. Mirrors _can_access_guardrail exactly."""
    ds_org_id = dataset.org_id
    ds_dept_id = dataset.dept_id
    ds_user_id = dataset.user_id
    ds_visibility = (dataset.visibility or "private").strip().lower()
    ds_public_scope = dataset.public_scope
    ds_public_dept_ids = dataset.public_dept_ids or []

    # Root: only own global datasets (no org/dept)
    if _is_root_user(current_user):
        return (
            ds_user_id == current_user.id
            and ds_org_id is None
            and ds_dept_id is None
        )

    role = normalize_role(str(getattr(current_user, "role", "")))

    # Super admin: all datasets in their orgs
    if role == "super_admin" and ds_org_id and ds_org_id in org_ids:
        return True

    dept_id_set = {str(d) for _, d in dept_pairs}

    # Private: owner OR dept_admin with matching dept
    if ds_visibility == "private":
        if role == "department_admin":
            return bool(ds_dept_id and str(ds_dept_id) in dept_id_set)
        return ds_user_id == current_user.id

    # Public org-scope: user's org matches
    if ds_public_scope == "organization":
        return bool(ds_org_id and ds_org_id in org_ids)

    # Public dept-scope: user's dept in candidates (public_dept_ids + dept_id)
    if ds_public_scope == "department":
        dept_candidates = set(ds_public_dept_ids)
        if ds_dept_id:
            dept_candidates.add(str(ds_dept_id))
        return bool(dept_candidates.intersection(dept_id_set))

    return False


def _can_edit_dataset(
    dataset: Dataset,
    current_user,
    org_ids: set[UUID],
    dept_pairs: list[tuple[UUID, UUID]],
) -> bool:
    """Check if user can edit/delete this dataset. Mirrors _can_edit_guardrail."""
    ds_org_id = dataset.org_id
    ds_dept_id = dataset.dept_id
    ds_visibility = (dataset.visibility or "private").strip().lower()
    ds_public_scope = dataset.public_scope
    ds_public_dept_ids = dataset.public_dept_ids or []

    role = normalize_role(str(getattr(current_user, "role", "")))

    if _is_root_user(current_user):
        return (
            dataset.user_id == current_user.id
            and ds_org_id is None
            and ds_dept_id is None
        )

    if role == "super_admin":
        # Own global private datasets
        if ds_visibility == "private" and ds_org_id is None and ds_dept_id is None:
            return dataset.user_id == current_user.id
        # All datasets in their orgs
        if ds_org_id and org_ids:
            return ds_org_id in org_ids

    if role == "department_admin":
        # Cannot edit org-scoped public datasets
        if ds_visibility == "public" and ds_public_scope == "organization":
            return False
        # Cannot edit multi-dept datasets
        dept_candidates = set(ds_public_dept_ids)
        if ds_dept_id:
            dept_candidates.add(str(ds_dept_id))
        if len(dept_candidates) > 1:
            return False
        dept_id_set = {str(d) for _, d in dept_pairs}
        if ds_visibility == "private":
            return bool(dept_candidates.intersection(dept_id_set))
        if ds_public_scope == "department":
            return bool(dept_candidates.intersection(dept_id_set))
        return False

    if role in {"developer", "business_user"}:
        return ds_visibility == "private" and dataset.user_id == current_user.id

    return False


async def _assert_dataset_manage_access(db, dataset: Dataset, current_user) -> None:
    """Require dataset-manage access for write operations on dataset children."""
    manage_org_ids, manage_dept_pairs = await _get_eval_scope_memberships(db, current_user.id)
    if not _can_edit_dataset(dataset, current_user, manage_org_ids, manage_dept_pairs):
        raise HTTPException(status_code=403, detail="Not authorized to manage dataset")


def _merge_local_dataset_metadata(
    metadata: Any,
    *,
    owner_user_id: str,
    visibility: str,
    public_scope: str | None,
    org_id: str | None,
    dept_id: str | None,
    public_dept_ids: list[str] | None,
) -> dict[str, Any]:
    """Keep dataset metadata aligned with the authoritative dataset columns."""
    return _merge_dataset_metadata(
        metadata,
        user_id=owner_user_id,
        visibility=visibility,
        public_scope=public_scope,
        org_id=org_id,
        dept_id=dept_id,
        public_dept_ids=public_dept_ids,
    )


def _merge_local_dataset_item_metadata(
    metadata: Any,
    *,
    current_user_id: str,
    dataset: Dataset,
    source: str,
) -> dict[str, Any]:
    """Attach audit metadata to dataset items while inheriting the parent dataset scope."""
    base = _as_dict(metadata)
    base["app_user_id"] = str(current_user_id)
    base["created_by_user_id"] = str(current_user_id)
    base["owner_user_id"] = str(current_user_id)
    base["user_id"] = str(current_user_id)
    base["dataset_id"] = str(dataset.id)
    base["dataset_name"] = dataset.name
    base["dataset_visibility"] = dataset.visibility or "private"
    base["dataset_public_scope"] = dataset.public_scope
    if dataset.org_id:
        base["dataset_org_id"] = str(dataset.org_id)
    else:
        base.pop("dataset_org_id", None)
    if dataset.dept_id:
        base["dataset_dept_id"] = str(dataset.dept_id)
    else:
        base.pop("dataset_dept_id", None)
    if dataset.public_dept_ids:
        base["dataset_public_dept_ids"] = [str(v) for v in dataset.public_dept_ids]
    else:
        base.pop("dataset_public_dept_ids", None)
    base["created_via"] = source
    return base


@router.get("/datasets")
async def list_datasets(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    page: Annotated[int, Query(ge=1)] = 1,
    search: Annotated[str | None, Query()] = None,
    org_id: Annotated[UUID | None, Query()] = None,
    dept_id: Annotated[UUID | None, Query()] = None,
) -> Dict[str, Any]:
    """List datasets visible within the user's RBAC scope (local DB).

    Uses Python-based _can_access_dataset filter (mirrors guardrail pattern).
    """
    try:
        async with session_scope() as db:
            scope_org_ids, scope_dept_pairs = await _get_eval_scope_memberships(db, current_user.id)
            role = normalize_role(str(getattr(current_user, "role", "")))

            # Broad SQL pre-filter to avoid loading entire table
            pre_conditions = [Dataset.user_id == current_user.id]
            if _is_root_user(current_user):
                pass  # Owner condition is enough
            elif role == "super_admin":
                if scope_org_ids:
                    pre_conditions.append(Dataset.org_id.in_(list(scope_org_ids)))
            else:
                # Include datasets in user's orgs and depts
                if scope_org_ids:
                    pre_conditions.append(Dataset.org_id.in_(list(scope_org_ids)))
                if scope_dept_pairs:
                    dept_id_list = [d for _, d in scope_dept_pairs]
                    pre_conditions.append(Dataset.dept_id.in_(dept_id_list))

            stmt = select(Dataset).where(or_(*pre_conditions))
            if search:
                stmt = stmt.where(Dataset.name.ilike(f"%{search.strip()}%"))
            stmt = stmt.order_by(Dataset.updated_at.desc())
            all_candidates = (await db.exec(stmt)).all()

            # Apply exact RBAC filter in Python (mirrors _can_access_guardrail)
            visible = [d for d in all_candidates if _can_access_dataset(d, current_user, scope_org_ids, scope_dept_pairs)]

            total = len(visible)
            start = (page - 1) * limit
            page_datasets = visible[start:start + limit]

            # Get item counts in one query
            dataset_ids = [d.id for d in page_datasets]
            item_counts: dict[UUID, int] = {}
            if dataset_ids:
                count_rows = (await db.exec(
                    select(DatasetItem.dataset_id, func.count(DatasetItem.id))
                    .where(DatasetItem.dataset_id.in_(dataset_ids))
                    .group_by(DatasetItem.dataset_id)
                )).all()
                item_counts = {row[0]: row[1] for row in count_rows}

            # Get creator emails
            user_ids = list({d.user_id for d in page_datasets})
            created_by_lookup: dict[str, str] = {}
            if user_ids:
                user_rows = (await db.exec(
                    select(User.id, User.email).where(User.id.in_(user_ids))
                )).all()
                created_by_lookup = {str(r[0]): r[1] for r in user_rows if r[1]}

            items = [
                DatasetResponse(**d.to_response(
                    item_count=item_counts.get(d.id, 0),
                    created_by=created_by_lookup.get(str(d.user_id)),
                ))
                for d in page_datasets
            ]

        return {"items": items, "total": total, "page": page, "limit": limit}
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Error listing datasets: {}", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/datasets")
async def create_dataset(
    payload: CreateDatasetRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
) -> DatasetResponse:
    """Create a dataset in the local database."""
    dataset_name = payload.name.strip()
    if not dataset_name:
        raise HTTPException(status_code=400, detail="Dataset name is required")

    visibility, public_scope, resolved_public_dept_ids, resolved_org_id, resolved_dept_id = (
        await _enforce_dataset_creation_scope(session, current_user, payload)
    )

    try:
        async with session_scope() as db:
            # Check uniqueness
            existing = (await db.exec(
                select(Dataset).where(Dataset.user_id == current_user.id, Dataset.name == dataset_name)
            )).first()
            if existing:
                raise HTTPException(status_code=409, detail=f"Dataset '{dataset_name}' already exists")

            dataset = Dataset(
                name=dataset_name,
                description=payload.description,
                metadata_=_merge_local_dataset_metadata(
                    payload.metadata,
                    owner_user_id=str(current_user.id),
                    visibility=visibility,
                    public_scope=public_scope,
                    org_id=resolved_org_id,
                    dept_id=resolved_dept_id,
                    public_dept_ids=resolved_public_dept_ids,
                ),
                user_id=current_user.id,
                org_id=UUID(resolved_org_id) if resolved_org_id else None,
                dept_id=UUID(resolved_dept_id) if resolved_dept_id else None,
                visibility=visibility,
                public_scope=public_scope,
                public_dept_ids=[str(d) for d in resolved_public_dept_ids] if resolved_public_dept_ids else None,
            )
            db.add(dataset)
            await db.commit()
            await db.refresh(dataset)

        return DatasetResponse(**dataset.to_response(item_count=0, created_by=current_user.email))
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Error creating dataset '{}': {}", dataset_name, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/datasets/{dataset_name}")
async def update_dataset(
    dataset_name: str,
    payload: UpdateDatasetRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
) -> DatasetResponse:
    """Update dataset visibility/scope/description."""
    try:
        async with session_scope() as db:
            dataset = await _get_dataset_with_access(db, dataset_name, current_user)

            # Check edit authorization (mirrors _can_edit_guardrail)
            edit_org_ids, edit_dept_pairs = await _get_eval_scope_memberships(db, current_user.id)
            if not _can_edit_dataset(dataset, current_user, edit_org_ids, edit_dept_pairs):
                raise HTTPException(status_code=403, detail="Not authorized to edit dataset")

            # Update description if provided
            if payload.description is not None:
                dataset.description = payload.description

            # Update visibility/scope if provided
            if payload.visibility is not None:
                # Re-use the existing scope enforcement logic
                scope_payload = CreateDatasetRequest(
                    name=dataset.name,
                    visibility=payload.visibility,
                    public_scope=payload.public_scope,
                    org_id=payload.org_id,
                    dept_id=payload.dept_id,
                    public_dept_ids=payload.public_dept_ids,
                )
                visibility, public_scope, resolved_public_dept_ids, resolved_org_id, resolved_dept_id = (
                    await _enforce_dataset_creation_scope(db, current_user, scope_payload)
                )
                dataset.visibility = visibility
                dataset.public_scope = public_scope
                dataset.public_dept_ids = [str(d) for d in resolved_public_dept_ids] if resolved_public_dept_ids else None
                dataset.org_id = UUID(resolved_org_id) if resolved_org_id else None
                dataset.dept_id = UUID(resolved_dept_id) if resolved_dept_id else None
                if visibility == "private":
                    dataset.user_id = current_user.id

            dataset.metadata_ = _merge_local_dataset_metadata(
                dataset.metadata_,
                owner_user_id=str(dataset.user_id),
                visibility=dataset.visibility or "private",
                public_scope=dataset.public_scope,
                org_id=str(dataset.org_id) if dataset.org_id else None,
                dept_id=str(dataset.dept_id) if dataset.dept_id else None,
                public_dept_ids=[str(d) for d in (dataset.public_dept_ids or [])] or None,
            )

            dataset.updated_at = datetime.now(timezone.utc)
            db.add(dataset)
            await db.commit()
            await db.refresh(dataset)

            # Get item count
            item_count = (await db.exec(
                select(func.count()).select_from(DatasetItem).where(DatasetItem.dataset_id == dataset.id)
            )).one()

        return DatasetResponse(**dataset.to_response(item_count=item_count, created_by=current_user.email))
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Error updating dataset '{}': {}", dataset_name, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/datasets/{dataset_name}")
async def delete_dataset(
    dataset_name: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    org_id: Annotated[UUID | None, Query()] = None,
    dept_id: Annotated[UUID | None, Query()] = None,
) -> Dict[str, Any]:
    """Delete a dataset and all its items/runs (CASCADE)."""
    try:
        async with session_scope() as db:
            dataset = await _get_dataset_with_access(db, dataset_name, current_user)

            # Check edit/delete authorization (mirrors _can_edit_guardrail)
            del_org_ids, del_dept_pairs = await _get_eval_scope_memberships(db, current_user.id)
            if not _can_edit_dataset(dataset, current_user, del_org_ids, del_dept_pairs):
                raise HTTPException(status_code=403, detail="Not authorized to delete dataset")

            # Count items and runs before deletion
            items_count = (await db.exec(
                select(func.count()).select_from(DatasetItem).where(DatasetItem.dataset_id == dataset.id)
            )).one()
            runs_count = (await db.exec(
                select(func.count()).select_from(DatasetRun).where(DatasetRun.dataset_id == dataset.id)
            )).one()

            # Delete children first (no ON DELETE CASCADE on FK)
            # 1. Delete run_items for all runs of this dataset
            run_ids_stmt = select(DatasetRun.id).where(DatasetRun.dataset_id == dataset.id)
            await db.execute(sa_delete(DatasetRunItem).where(DatasetRunItem.run_id.in_(run_ids_stmt)))
            # 2. Delete runs
            await db.execute(sa_delete(DatasetRun).where(DatasetRun.dataset_id == dataset.id))
            # 3. Delete items
            await db.execute(sa_delete(DatasetItem).where(DatasetItem.dataset_id == dataset.id))
            # 4. Delete dataset
            await db.delete(dataset)
            await db.commit()

        return {
            "status": "deleted",
            "dataset_name": dataset_name,
            "dataset_deleted": True,
            "runs_deleted": runs_count,
            "items_deleted": items_count,
            "errors": [],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Error deleting dataset '{}': {}", dataset_name, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


async def _get_dataset_with_access(db, dataset_name: str, current_user) -> Dataset:
    """Lookup dataset by name with RBAC check. Raises 404 if not found."""
    datasets = (await db.exec(
        select(Dataset).where(Dataset.name == dataset_name)
    )).all()
    if not datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    
    org_ids, dept_pairs = await _get_eval_scope_memberships(db, current_user.id)
    
    accessible_datasets = [
        d for d in datasets 
        if _can_access_dataset(d, current_user, org_ids, dept_pairs)
    ]
    
    if not accessible_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
    for d in accessible_datasets:
        if d.user_id == current_user.id:
            return d
            
    return accessible_datasets[0]


@router.get("/datasets/{dataset_name}/items")
async def list_dataset_items(
    dataset_name: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    page: Annotated[int, Query(ge=1)] = 1,
    source_trace_id: Annotated[str | None, Query()] = None,
    org_id: Annotated[UUID | None, Query()] = None,
    dept_id: Annotated[UUID | None, Query()] = None,
) -> Dict[str, Any]:
    """List items in a dataset (local DB)."""
    try:
        async with session_scope() as db:
            dataset = await _get_dataset_with_access(db, dataset_name, current_user)

            stmt = select(DatasetItem).where(DatasetItem.dataset_id == dataset.id)
            if source_trace_id:
                stmt = stmt.where(DatasetItem.source_trace_id == source_trace_id)
            stmt = stmt.order_by(DatasetItem.created_at.desc())

            total = (await db.exec(
                select(func.count()).select_from(DatasetItem)
                .where(DatasetItem.dataset_id == dataset.id)
            )).one()

            offset = (page - 1) * limit
            items = (await db.exec(stmt.offset(offset).limit(limit))).all()

        return {
            "items": [DatasetItemResponse(**i.to_response(dataset_name=dataset_name)) for i in items],
            "total": total,
            "page": page,
            "limit": limit,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Error listing dataset items for '{}': {}", dataset_name, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/datasets/{dataset_name}/items")
async def create_dataset_item(
    dataset_name: str,
    payload: CreateDatasetItemRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    org_id: Annotated[UUID | None, Query()] = None,
    dept_id: Annotated[UUID | None, Query()] = None,
) -> DatasetItemResponse:
    """Create one dataset item (local DB)."""
    try:
        async with session_scope() as db:
            dataset = await _get_dataset_with_access(db, dataset_name, current_user)
            await _assert_dataset_manage_access(db, dataset, current_user)

            # Normalize input/expected_output to dict
            item_input = payload.input
            if isinstance(item_input, str):
                item_input = {"text": item_input}
            expected_output = payload.expected_output
            if isinstance(expected_output, str):
                expected_output = {"text": expected_output}

            item = DatasetItem(
                dataset_id=dataset.id,
                input=item_input,
                expected_output=expected_output,
                metadata_=_merge_local_dataset_item_metadata(
                    payload.metadata,
                    current_user_id=str(current_user.id),
                    dataset=dataset,
                    source="agentcore-evaluation-manual-item",
                ),
                source_trace_id=payload.source_trace_id or payload.trace_id,
                source_observation_id=payload.source_observation_id,
            )
            db.add(item)
            await db.commit()
            await db.refresh(item)

        return DatasetItemResponse(**item.to_response(dataset_name=dataset_name))
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Error creating dataset item for '{}': {}", dataset_name, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/datasets/{dataset_name}/items/upload-csv")
async def upload_dataset_items_csv(
    dataset_name: str,
    csv_file: Annotated[UploadFile, File(...)],
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    org_id: Annotated[UUID | None, Query()] = None,
    dept_id: Annotated[UUID | None, Query()] = None,
) -> DatasetCsvImportResponse:
    """Bulk-create dataset items from CSV rows (local DB)."""
    filename = (csv_file.filename or "").strip()
    if filename and not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    max_bytes = 10 * 1024 * 1024
    max_rows = 5000
    max_error_rows = 100

    raw = await csv_file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty")
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"CSV file too large. Maximum: {max_bytes // (1024 * 1024)} MB.")

    try:
        text = raw.decode("utf-8-sig")
    except Exception:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV header row is required")

    total_rows = 0
    created_count = 0
    failed_count = 0
    skipped_count = 0
    errors: list[DatasetCsvImportError] = []

    try:
        async with session_scope() as db:
            dataset = await _get_dataset_with_access(db, dataset_name, current_user)
            await _assert_dataset_manage_access(db, dataset, current_user)
            batch: list[DatasetItem] = []

            for row_number, row in enumerate(reader, start=2):
                row_values = list((row or {}).values())
                if not any(str(v).strip() for v in row_values if v is not None):
                    skipped_count += 1
                    continue

                total_rows += 1
                if total_rows > max_rows:
                    failed_count += 1
                    errors.append(DatasetCsvImportError(row=row_number, message=f"Row limit exceeded. Max: {max_rows}"))
                    break

                try:
                    req = _csv_row_to_dataset_item_request(row or {})
                    item_input = req.input
                    if isinstance(item_input, str):
                        item_input = {"text": item_input}
                    expected_output = req.expected_output
                    if isinstance(expected_output, str):
                        expected_output = {"text": expected_output}

                    batch.append(DatasetItem(
                        dataset_id=dataset.id,
                        input=item_input,
                        expected_output=expected_output,
                        metadata_=_merge_local_dataset_item_metadata(
                            req.metadata,
                            current_user_id=str(current_user.id),
                            dataset=dataset,
                            source="agentcore-evaluation-csv-import",
                        ),
                        source_trace_id=req.source_trace_id or req.trace_id,
                        source_observation_id=req.source_observation_id,
                    ))
                    created_count += 1
                except Exception as exc:
                    failed_count += 1
                    if len(errors) < max_error_rows:
                        errors.append(DatasetCsvImportError(row=row_number, message=str(exc)))

            if total_rows == 0:
                raise HTTPException(status_code=400, detail="CSV has no importable rows")

            if batch:
                db.add_all(batch)
                await db.commit()

    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Error importing CSV for '{}': {}", dataset_name, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    return DatasetCsvImportResponse(
        dataset_name=dataset_name,
        total_rows=total_rows,
        created_count=created_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        errors=errors,
    )


@router.delete("/datasets/{dataset_name}/items/{item_id}")
async def delete_dataset_item(
    dataset_name: str,
    item_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    org_id: Annotated[UUID | None, Query()] = None,
    dept_id: Annotated[UUID | None, Query()] = None,
) -> Dict[str, Any]:
    """Delete one dataset item (local DB)."""
    try:
        item_uuid = UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid item ID")

    try:
        async with session_scope() as db:
            dataset = await _get_dataset_with_access(db, dataset_name, current_user)
            await _assert_dataset_manage_access(db, dataset, current_user)
            item = (await db.exec(
                select(DatasetItem).where(DatasetItem.id == item_uuid, DatasetItem.dataset_id == dataset.id)
            )).first()
            if not item:
                raise HTTPException(status_code=404, detail=f"Dataset item '{item_id}' not found")

            await db.delete(item)
            await db.commit()

        return {"status": "deleted", "dataset_name": dataset_name, "item_id": item_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Error deleting dataset item '{}': {}", item_id, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/datasets/{dataset_name}/runs")
async def list_dataset_runs(
    dataset_name: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    page: Annotated[int, Query(ge=1)] = 1,
    org_id: Annotated[UUID | None, Query()] = None,
    dept_id: Annotated[UUID | None, Query()] = None,
) -> Dict[str, Any]:
    """List experiment runs for a dataset (local DB)."""
    try:
        async with session_scope() as db:
            dataset = await _get_dataset_with_access(db, dataset_name, current_user)

            total = (await db.exec(
                select(func.count()).select_from(DatasetRun).where(DatasetRun.dataset_id == dataset.id)
            )).one()

            offset = (page - 1) * limit
            runs = (await db.exec(
                select(DatasetRun)
                .where(DatasetRun.dataset_id == dataset.id)
                .order_by(DatasetRun.created_at.desc())
                .offset(offset).limit(limit)
            )).all()

        return {
            "items": [DatasetRunResponse(**r.to_response(dataset_name=dataset_name)) for r in runs],
            "total": total,
            "page": page,
            "limit": limit,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Error listing dataset runs for '{}': {}", dataset_name, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/datasets/{dataset_name}/runs/{run_id}")
async def get_dataset_run_detail(
    dataset_name: str,
    run_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    item_limit: Annotated[int, Query(ge=1, le=200)] = 50,
    score_limit: Annotated[int, Query(ge=1, le=100)] = 20,
    org_id: Annotated[UUID | None, Query()] = None,
    dept_id: Annotated[UUID | None, Query()] = None,
) -> DatasetRunDetailResponse:
    """Return a dataset run with item-level details (local DB + Langfuse scores)."""
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID")

    try:
        async with session_scope() as db:
            dataset = await _get_dataset_with_access(db, dataset_name, current_user)

            run = (await db.exec(
                select(DatasetRun).where(DatasetRun.id == run_uuid, DatasetRun.dataset_id == dataset.id)
            )).first()
            if not run:
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in dataset '{dataset_name}'")

            run_items = (await db.exec(
                select(DatasetRunItem)
                .where(DatasetRunItem.run_id == run.id)
                .order_by(DatasetRunItem.created_at.asc())
                .limit(item_limit)
            )).all()

            # Get associated dataset items for input/expected_output
            item_ids = [ri.dataset_item_id for ri in run_items if ri.dataset_item_id]
            dataset_items_lookup: dict[UUID, DatasetItem] = {}
            if item_ids:
                ds_items = (await db.exec(
                    select(DatasetItem).where(DatasetItem.id.in_(item_ids))
                )).all()
                dataset_items_lookup = {di.id: di for di in ds_items}

        detailed_items: list[DatasetRunItemDetailResponse] = []
        for ri in run_items:
            ds_item = dataset_items_lookup.get(ri.dataset_item_id) if ri.dataset_item_id else None
            score_rows = [
                DatasetRunItemScoreResponse(
                    id=str(s.get("id", "")),
                    name=str(s.get("name", "Score")),
                    value=float(s.get("value", 0.0)),
                    source=str(s.get("source", "API")),
                    comment=s.get("comment"),
                    created_at=s.get("created_at"),
                )
                for s in (ri.scores or [])
            ]

            detailed_items.append(DatasetRunItemDetailResponse(
                id=str(ri.id),
                dataset_item_id=str(ri.dataset_item_id) if ri.dataset_item_id else None,
                trace_id=ri.trace_id,
                observation_id=ri.observation_id,
                created_at=ri.created_at,
                updated_at=ri.created_at,
                trace_name=None,
                trace_input=ds_item.input if ds_item else None,
                trace_output=ri.output,
                score_count=len(score_rows),
                scores=score_rows,
            ))

        return DatasetRunDetailResponse(
            run=DatasetRunResponse(**run.to_response(dataset_name=dataset_name)),
            item_count=len(detailed_items),
            items=detailed_items,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Error getting run detail '{}': {}", run_id, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/datasets/{dataset_name}/runs/{run_id}")
async def delete_dataset_run(
    dataset_name: str,
    run_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    org_id: Annotated[UUID | None, Query()] = None,
    dept_id: Annotated[UUID | None, Query()] = None,
) -> Dict[str, Any]:
    """Delete one dataset run (local DB, CASCADE deletes run items)."""
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID")

    try:
        async with session_scope() as db:
            dataset = await _get_dataset_with_access(db, dataset_name, current_user)
            await _assert_dataset_manage_access(db, dataset, current_user)

            run = (await db.exec(
                select(DatasetRun).where(DatasetRun.id == run_uuid, DatasetRun.dataset_id == dataset.id)
            )).first()
            if not run:
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in dataset '{dataset_name}'")

            run_name = run.name
            await db.delete(run)
            await db.commit()

        return {"status": "deleted", "dataset_name": dataset_name, "run_id": run_id, "run_name": run_name}
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Error deleting dataset run '{}': {}", run_id, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/datasets/{dataset_name}/experiments")
async def run_dataset_experiment(
    dataset_name: str,
    payload: RunDatasetExperimentRequest,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    org_id: Annotated[UUID | None, Query()] = None,
    dept_id: Annotated[UUID | None, Query()] = None,
) -> DatasetExperimentEnqueueResponse:
    """Queue an experiment run against a dataset (local DB)."""
    # Verify dataset exists and user has access
    async with session_scope() as db:
        dataset = await _get_dataset_with_access(db, dataset_name, current_user)
        dataset_id = dataset.id

        # Check ground truth if needed
        if payload.preset_id:
            preset = get_preset_by_id(payload.preset_id)
            if preset and preset.get("requires_ground_truth"):
                has_gt = (await db.exec(
                    select(func.count()).select_from(DatasetItem)
                    .where(DatasetItem.dataset_id == dataset_id, DatasetItem.expected_output.isnot(None))
                )).one()
                if not has_gt:
                    raise HTTPException(
                        status_code=400,
                        detail="Selected evaluator preset requires ground truth, but dataset items do not have expected_output.",
                    )

    # Get Langfuse client for score submission
    _, client = await _get_scoped_langfuse_for_evaluation(session, current_user)
    if not client:
        client = get_langfuse_client()

    agent_payload = await _resolve_agent_payload_for_experiment(
        agent_id=payload.agent_id,
        current_user=current_user,
    )
    generation_model: str | None = None
    generation_model_api_key: str | None = None
    if payload.generation_model_registry_id:
        generation_model, generation_model_api_key, _ = await _resolve_model_from_registry(
            payload.generation_model_registry_id, session=session
        )

    if not agent_payload and not generation_model:
        raise HTTPException(status_code=400, detail="Select an agent or provide a generation model from the registry.")

    judge_cfg = await _resolve_experiment_judge_config(
        current_user=current_user,
        evaluator_config_id=payload.evaluator_config_id,
        preset_id=payload.preset_id,
        evaluator_name=payload.evaluator_name,
        criteria=payload.criteria,
        judge_model=None,
        judge_model_api_key=None,
        judge_model_registry_id=payload.judge_model_registry_id,
        session=session,
    )
    if judge_cfg["criteria"] and not judge_cfg["model"]:
        judge_cfg["model"] = "gpt-4o"

    job_id = str(uuid4())
    _set_dataset_experiment_job(
        job_id,
        status="queued",
        dataset_name=dataset_name,
        experiment_name=payload.experiment_name,
        started_at=None,
        finished_at=None,
        result=None,
        error=None,
        user_id=str(current_user.id),
    )

    background_tasks.add_task(
        _run_dataset_experiment_job,
        job_id=job_id,
        client=client,
        dataset_name=dataset_name,
        dataset_id=str(dataset_id),
        experiment_name=payload.experiment_name,
        description=payload.description,
        user_id=str(current_user.id),
        agent_payload=agent_payload,
        generation_model=generation_model,
        generation_model_api_key=generation_model_api_key,
        generation_model_registry_id=payload.generation_model_registry_id,
        judge_name=judge_cfg["judge_name"],
        judge_preset_id=judge_cfg.get("preset_id"),
        judge_criteria=judge_cfg["criteria"],
        judge_model=judge_cfg["model"],
        judge_model_api_key=judge_cfg["model_api_key"],
        judge_model_registry_id=payload.judge_model_registry_id,
    )

    return DatasetExperimentEnqueueResponse(
        job_id=job_id,
        dataset_name=dataset_name,
        experiment_name=payload.experiment_name,
        status="queued",
    )


@router.get("/datasets/experiments/{job_id}")
async def get_dataset_experiment_job(
    job_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> DatasetExperimentJobResponse:
    """Fetch status for a background dataset experiment job."""
    payload = _get_dataset_experiment_job(job_id)
    if not payload or str(payload.get("user_id")) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Dataset experiment job not found")
    return _dataset_job_response(job_id, payload)


# =============================================================================
# Presets Configuration
# =============================================================================

EVALUATION_PRESETS = [
    {
        "id": "correctness",
        "name": "Correctness",
        "description": "Evaluate if the output is factually correct compared to the ground truth.",
        "criteria": "Evaluate the correctness of the generation against the ground truth on a scale 0-1. Consider:\n- Factual accuracy: Does the output match the ground truth?\n- Completeness: Are all key points from the ground truth covered?\n- Precision: Is the information accurate without hallucinations?",
        "requires_ground_truth": True,
    },
    {
        "id": "helpfulness",
        "name": "Helpfulness",
        "description": "Evaluate how helpful the response is to the user's query.",
        "criteria": "Evaluate how helpful the output is in addressing the user's input on a scale 0-1. Consider:\n- Relevance: Does it directly address what was asked?\n- Clarity: Is the response easy to understand?\n- Actionability: Can the user act on this information?",
        "requires_ground_truth": False,
    },
    {
        "id": "conciseness",
        "name": "Conciseness",
        "description": "Evaluate if the response is appropriately concise without unnecessary verbosity.",
        "criteria": "Evaluate the conciseness of the output on a scale 0-1. Consider:\n- Brevity: Is it as short as possible while being complete?\n- Focus: Does it avoid tangential information?\n- Efficiency: Does it convey the message without redundancy?",
        "requires_ground_truth": False,
    },
    {
        "id": "coherence",
        "name": "Coherence",
        "description": "Evaluate if the response flows logically and makes sense.",
        "criteria": "Evaluate the coherence and logical flow of the output on a scale 0-1. Consider:\n- Logical structure: Do ideas connect naturally?\n- Internal consistency: Are there contradictions?\n- Clarity of thought: Is the reasoning easy to follow?",
        "requires_ground_truth": False,
    },
    {
        "id": "relevance",
        "name": "Relevance",
        "description": "Evaluate how relevant the response is to the input query.",
        "criteria": "Evaluate the relevance of the output to the input query on a scale 0-1. Consider:\n- Topic alignment: Does it stay on topic?\n- Query understanding: Does it address the user's intent?\n- Information pertinence: Is all information provided relevant?",
        "requires_ground_truth": False,
    },
]


def get_preset_by_id(preset_id: str | None) -> Dict[str, Any] | None:
    """Return preset configuration by id."""
    if not preset_id or preset_id == "__custom__":
        return None
    for preset in EVALUATION_PRESETS:
        if str(preset.get("id")) == str(preset_id):
            return preset
    return None


def validate_ground_truth_requirement(preset_id: str | None, ground_truth: str | None) -> None:
    """Validate whether ground truth is provided for presets that require it."""
    preset = get_preset_by_id(preset_id)
    if preset and preset.get("requires_ground_truth") and not (ground_truth or "").strip():
        raise HTTPException(
            status_code=400,
            detail=f"Ground truth is required for preset '{preset.get('name', preset_id)}'.",
        )


async def run_saved_evaluators_for_new_trace(
    *,
    trace_id: str,
    user_id: str,
    agent_id: str | None = None,
    agent_name: str | None = None,
    session_id: str | None = None,
    project_name: str | None = None,
    timestamp: datetime | None = None,
    trace_input: Any | None = None,
    trace_output: Any | None = None,
    langfuse_client: Any | None = None,
) -> int:
    """Run all saved evaluators targeting new traces for a just-finished trace."""
    logger.info(
        f"EVALUATOR FUNCTION CALLED: trace={trace_id}, user={user_id}, "
        f"agent_id={agent_id}, agent_name={agent_name}"
    )
    
    # Model Service is the primary LLM backend — don't require LiteLLM/OpenAI
    # to be installed. They're only needed as fallbacks.

    try:
        user_uuid = UUID(str(user_id))
    except Exception:
        logger.warning(f"Invalid user_id for new-trace evaluation: {user_id}")
        return 0

    # Use the tracer's Langfuse client if provided (same project as the trace).
    # This ensures scores are submitted to the same Langfuse project as the trace.
    client = langfuse_client
    if client:
        logger.info(f"Using tracer's Langfuse client for evaluator score submission (type={type(client).__name__})")
    else:
        # Fallback: resolve via DB bindings / env-var
        try:
            async with session_scope() as _scope_session:
                from agentcore.services.database.models.user.model import User as UserModel
                user_obj = await _scope_session.get(UserModel, user_uuid)
                if user_obj:
                    _, client = await _get_scoped_langfuse_for_evaluation(
                        _scope_session, user_obj,
                    )
        except Exception as scope_err:
            logger.debug("Scoped Langfuse client resolution failed: {}", str(scope_err))

    if not client:
        client = get_langfuse_client()
    if not client:
        logger.warning("Langfuse client not available, skipping evaluators")
        return 0

    requested_timestamp = _parse_trace_timestamp(timestamp) or datetime.now(timezone.utc)
    trace_ref_id = str(trace_id)
    normalized_caller_agent_id = _normalize_agent_id(agent_id)

    # Build trace_dict from caller-provided data (authoritative source).
    # No need to fetch from Langfuse — the tracing service provides all context.
    trace_dict: Dict[str, Any] = {
        "id": trace_ref_id,
        "session_id": session_id,
        "timestamp": requested_timestamp,
        "metadata": {
            "agent_id": normalized_caller_agent_id,
            "agent_name": agent_name,
            "project_name": project_name,
            "run_id": trace_ref_id,
        },
    }
    resolved_trace_id = trace_ref_id

    try:
        async with session_scope() as session:
            stmt = select(Evaluator).where(Evaluator.user_id == user_uuid)
            rows = await session.exec(stmt)
            evaluators = rows.all()
    except Exception as e:
        logger.warning("Failed loading evaluators for new trace {}: {}", trace_id, str(e))
        return 0

    logger.info(f" Found {len(evaluators)} evaluator(s) for user {user_id}")
    
    scheduled = 0
    for idx, evaluator in enumerate(evaluators, 1):
        logger.info(
            f" Evaluator {idx}/{len(evaluators)}: name='{evaluator.name}', "
            f"target={evaluator.target}, agent_id={evaluator.agent_id}, "
            f"agent_ids={evaluator.agent_ids}, agent_name={evaluator.agent_name}"
        )
        
        targets = _normalize_targets(evaluator.target)
        if "new" not in targets:
            logger.info(f"  Skipping: target={targets} (not 'new')")
            continue

        logger.info(
            f"   Checking filters: trace_dict_agent_id={trace_dict.get('metadata', {}).get('agent_id')}, "
            f"trace_dict_agent_name={trace_dict.get('metadata', {}).get('agent_name')}"
        )
        
        matches = _trace_matches_evaluator_filters(
            trace_dict,
            trace_id=evaluator.trace_id,
            session_id=evaluator.session_id,
            agent_id=evaluator.agent_id,
            agent_ids=evaluator.agent_ids,
            agent_name=evaluator.agent_name,
            project_name=evaluator.project_name,
            ts_from=evaluator.ts_from,
            ts_to=evaluator.ts_to,
        )
        
        if not matches:
            logger.info(f"   Skipping: trace does not match filters")
            continue
        
        logger.info(f"  MATCH! Scheduling evaluation for '{evaluator.name}'")

        # Skip invalid evaluator definitions instead of failing all.
        try:
            validate_ground_truth_requirement(evaluator.preset_id, evaluator.ground_truth)
        except HTTPException as e:
            logger.warning(
                f"Skipping evaluator {evaluator.id} for trace {trace_id}: {e.detail}"
            )
            continue

        # Resolve API key and base URL from model registry if available
        effective_model = evaluator.model or "gpt-4o"
        effective_api_key = None
        effective_api_base = None
        if evaluator.model_registry_id:
            try:
                effective_model, effective_api_key, effective_api_base = await _resolve_model_from_registry(
                    evaluator.model_registry_id
                )
            except Exception as reg_err:
                logger.warning(
                    "Failed to resolve model registry {} for evaluator {}: {}",
                    evaluator.model_registry_id, evaluator.name, str(reg_err),
                )

        asyncio.create_task(
            run_llm_judge_task(
                client=client,
                trace_id=str(resolved_trace_id or trace_ref_id),
                criteria=evaluator.criteria,
                score_name=f"Evaluator: {evaluator.name}",
                model=effective_model,
                user_id=str(user_id),
                model_api_key=effective_api_key,
                model_api_base=effective_api_base,
                model_registry_id=evaluator.model_registry_id,
                preset_id=evaluator.preset_id,
                ground_truth=evaluator.ground_truth,
                session_id=session_id,
                agent_id=agent_id,
                agent_name=agent_name,
                project_name=project_name,
                timestamp=requested_timestamp,
                trace_input=trace_input,
                trace_output=trace_output,
            )
        )
        scheduled += 1

    if scheduled:
        logger.info(
            f" Scheduled {scheduled} new-trace evaluator(s) for trace_ref={trace_ref_id}, "
            f"resolved_trace_id={resolved_trace_id}, user_id={user_id}"
        )
    else:
        logger.info(
            f"No evaluators scheduled for trace {trace_ref_id}. "
            f"Total evaluators checked: {len(evaluators)}"
        )
    return scheduled


@router.get("/presets")
async def list_evaluation_presets(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> List[Dict[str, Any]]:
    """List available evaluation presets with their requirements."""
    return EVALUATION_PRESETS


@router.get("/models")
async def list_evaluation_models(
    current_user: Annotated[User, Depends(get_current_active_user)],
    environment: Annotated[str | None, Query(description="'uat' or 'production'")] = None,
) -> Dict[str, Any]:
    """Return agents accessible to the current user as a normalized model list.

    When *environment* is provided, only agents deployed to that environment
    (with the same RBAC rules as the orchestration chat) are returned.
    Without an environment filter the legacy behaviour (all owned/public agents) is used.
    """
    try:
        async with session_scope() as session:
            # Always fetch legacy agents (owned / public from base Agent table)
            stmt = select(agent).where(
                or_(
                    agent.user_id == current_user.id,
                    agent.access_type == AccessTypeEnum.PUBLIC,
                )
            )
            is_component_col = getattr(agent, "is_component", None)
            if is_component_col is not None:
                stmt = stmt.where(
                    or_(
                        is_component_col == False,  # noqa: E712
                        is_component_col.is_(None),
                    )
                )
            _res = await session.exec(stmt)
            raw_agents = _res.all()
            legacy_data = [
                _agent_to_payload(a, environment=environment or None) for a in raw_agents
            ]
            logger.debug(
                "Legacy agent query returned {} agent(s) for user_id={}",
                len(legacy_data), current_user.id,
            )

            if environment and environment.lower() in ("uat", "production", "prod"):
                env = environment.lower()
                if env == "prod":
                    env = "production"
                deployed_data = await _list_deployed_agents(session, current_user, env)
                logger.debug(
                    "Deployed agent query ({}) returned {} agent(s) for user_id={}",
                    env, len(deployed_data), current_user.id,
                )
                if env == "production":
                    # Production: only deployed production agents
                    agents_data = deployed_data
                else:
                    # UAT: deployed UAT agents + legacy (base table) agents, de-duplicated
                    seen_agent_ids: set[str] = set()
                    agents_data = []
                    for item in deployed_data + legacy_data:
                        aid = item.get("metadata", {}).get("agent_id", "") or item.get("id", "")
                        if aid not in seen_agent_ids:
                            seen_agent_ids.add(aid)
                            agents_data.append(item)
            else:
                agents_data = legacy_data

        return {"object": "list", "data": agents_data}
    except Exception as e:
        logger.opt(exception=True).error("Error listing evaluation models: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


def _agent_to_payload(agent_obj, *, environment: str | None = None) -> dict:
    """Convert an Agent (or deployment record) into the normalised model payload."""
    updated = agent_obj.updated_at
    try:
        updated_dt = datetime.fromisoformat(updated) if isinstance(updated, str) else updated
    except Exception:
        updated_dt = None
    created_ts = int(updated_dt.timestamp()) if updated_dt else int(time.time())
    endpoint_name = getattr(agent_obj, "endpoint_name", None)
    model_id = endpoint_name or agent_obj.id
    access = (
        agent_obj.access_type.value
        if getattr(agent_obj, "access_type", None)
        else AccessTypeEnum.PRIVATE.value
    )
    return {
        "id": f"lb:{model_id}",
        "name": agent_obj.name,
        "object": "model",
        "created": created_ts,
        "owned_by": str(agent_obj.user_id) if getattr(agent_obj, "user_id", None) else None,
        "root": f"lb:{model_id}",
        "parent": None,
        "permission": [],
        "metadata": {
            "display_name": agent_obj.name,
            "description": getattr(agent_obj, "description", None),
            "endpoint_name": endpoint_name,
            "agent_id": str(agent_obj.id),
            "agent_ids": [str(agent_obj.id)],
            "access": access,
            **({"environment": environment} if environment else {}),
        },
    }


def _deploy_to_payload(deploy_rec, *, environment: str) -> dict:
    """Convert a deployment record into the normalised model payload."""
    created_ts = int(time.time())
    if getattr(deploy_rec, "updated_at", None):
        try:
            dt = (
                datetime.fromisoformat(deploy_rec.updated_at)
                if isinstance(deploy_rec.updated_at, str)
                else deploy_rec.updated_at
            )
            created_ts = int(dt.timestamp())
        except Exception:
            pass
    agent_id = str(deploy_rec.agent_id)
    return {
        "id": f"lb:{agent_id}",
        "name": deploy_rec.agent_name or agent_id,
        "object": "model",
        "created": created_ts,
        "owned_by": str(deploy_rec.deployed_by) if getattr(deploy_rec, "deployed_by", None) else None,
        "root": f"lb:{agent_id}",
        "parent": None,
        "permission": [],
        "metadata": {
            "display_name": deploy_rec.agent_name or agent_id,
            "description": getattr(deploy_rec, "agent_description", None),
            "endpoint_name": None,
            "agent_id": agent_id,
            "agent_ids": [agent_id],
            "access": "public",
            "environment": environment,
        },
    }


async def _list_deployed_agents(session, current_user: User, env: str) -> list[dict]:
    """Return deployed agents for the given environment using orchestrator RBAC logic."""
    from agentcore.services.auth.permissions import normalize_role as _nr

    current_role = str(getattr(current_user, "role", "")).lower()
    is_admin = current_role in {"super_admin", "department_admin", "root"}

    if env == "production":
        # ---- Production RBAC (mirrors orchestrator.py) ----
        prod_share_exists = (
            select(AgentPublishRecipient.id)
            .where(
                or_(
                    AgentPublishRecipient.deploy_id == AgentDeploymentProd.id,
                    and_(
                        AgentPublishRecipient.deploy_id.is_(None),
                        AgentPublishRecipient.agent_id == AgentDeploymentProd.agent_id,
                    ),
                ),
                AgentPublishRecipient.recipient_user_id == current_user.id,
                or_(
                    AgentDeploymentProd.dept_id.is_(None),
                    AgentPublishRecipient.dept_id == AgentDeploymentProd.dept_id,
                ),
            )
            .exists()
        )
        prod_dept_member_exists = (
            select(UserDepartmentMembership.id)
            .where(
                UserDepartmentMembership.user_id == current_user.id,
                UserDepartmentMembership.department_id == AgentDeploymentProd.dept_id,
                UserDepartmentMembership.status == "active",
            )
            .exists()
        )
        prod_private_access = (
            (AgentDeploymentProd.deployed_by == current_user.id)
            | prod_share_exists
        )
        prod_public_access = prod_private_access | prod_dept_member_exists
        if is_admin:
            prod_private_access = prod_private_access | true()
            prod_public_access = prod_public_access | true()

        stmt = (
            select(AgentDeploymentProd)
            .where(AgentDeploymentProd.status == DeploymentPRODStatusEnum.PUBLISHED)
            .where(AgentDeploymentProd.is_active == True)  # noqa: E712
            .where(AgentDeploymentProd.is_enabled == True)  # noqa: E712
            .where(
                (
                    (AgentDeploymentProd.visibility == ProdDeploymentVisibilityEnum.PUBLIC)
                    & prod_public_access
                )
                | (
                    (AgentDeploymentProd.visibility == ProdDeploymentVisibilityEnum.PRIVATE)
                    & prod_private_access
                )
            )
        )
        records = list((await session.exec(stmt)).all())
        return [_deploy_to_payload(r, environment="production") for r in records]

    else:
        # ---- UAT RBAC (mirrors orchestrator.py) ----
        uat_share_exists = (
            select(AgentPublishRecipient.id)
            .where(
                or_(
                    AgentPublishRecipient.deploy_id == AgentDeploymentUAT.id,
                    and_(
                        AgentPublishRecipient.deploy_id.is_(None),
                        AgentPublishRecipient.agent_id == AgentDeploymentUAT.agent_id,
                    ),
                ),
                AgentPublishRecipient.recipient_user_id == current_user.id,
                or_(
                    AgentDeploymentUAT.dept_id.is_(None),
                    AgentPublishRecipient.dept_id == AgentDeploymentUAT.dept_id,
                ),
            )
            .exists()
        )
        uat_access = (
            (AgentDeploymentUAT.deployed_by == current_user.id)
            | uat_share_exists
        )
        if is_admin:
            uat_access = uat_access | true()

        stmt = (
            select(AgentDeploymentUAT)
            .where(AgentDeploymentUAT.status == DeploymentUATStatusEnum.PUBLISHED)
            .where(AgentDeploymentUAT.is_active == True)  # noqa: E712
            .where(AgentDeploymentUAT.is_enabled == True)  # noqa: E712
            .where(uat_access)
        )
        records = list((await session.exec(stmt)).all())
        return [_deploy_to_payload(r, environment="uat") for r in records]


async def _enqueue_existing_trace_evaluations(
    *,
    background_tasks: BackgroundTasks,
    user_id: str,
    evaluator_name: str,
    criteria: str,
    model: str | None,
    trace_id: str | None = None,
    agent_id: str | None = None,
    agent_ids: Optional[List[str]] = None,
    agent_name: str | None = None,
    session_id: str | None = None,
    project_name: str | None = None,
    ts_from: datetime | None = None,
    ts_to: datetime | None = None,
    model_api_key: str | None = None,
    preset_id: str | None = None,
    ground_truth: str | None = None,
    environment: str | None = None,
    langfuse_client: Any = None,
) -> int:
    """Queue evaluator runs for all matching existing traces."""
    validate_ground_truth_requirement(preset_id, ground_truth)

    client = langfuse_client or get_langfuse_client()
    if not client:
        return 0

    normalized_agent_id = _normalize_agent_id(agent_id)
    normalized_agent_ids = _normalize_agent_ids(agent_ids)

    try:
        traces = fetch_traces_from_langfuse(
            client,
            user_id=user_id,
            limit=1000,
            from_timestamp=ts_from,
            to_timestamp=ts_to,
            environment=environment,
        )
    except Exception as exc:
        logger.warning("Failed to fetch traces for evaluator run: {}", str(exc))
        return 0

    enqueued = 0
    seen_trace_ids: set[str] = set()
    for trace in traces or []:
        trace_dict = parse_trace_data(trace)
        if not _trace_matches_evaluator_filters(
            trace_dict,
            trace_id=trace_id,
            session_id=session_id,
            agent_id=normalized_agent_id,
            agent_ids=normalized_agent_ids,
            agent_name=agent_name,
            project_name=project_name,
            ts_from=ts_from,
            ts_to=ts_to,
        ):
            continue

        matched_trace_id = str(trace_dict.get("id") or "")
        if not matched_trace_id:
            continue
        if matched_trace_id in seen_trace_ids:
            continue
        seen_trace_ids.add(matched_trace_id)

        background_tasks.add_task(
            run_llm_judge_task,
            client=client,
            trace_id=matched_trace_id,
            criteria=criteria,
            score_name=f"Evaluator: {evaluator_name}",
            model=model or "gpt-4o",
            user_id=user_id,
            model_api_key=model_api_key,
            preset_id=preset_id,
            ground_truth=ground_truth,
            session_id=str(trace_dict.get("session_id") or "") or None,
            agent_id=_extract_trace_agent_id(trace_dict),
            agent_name=_extract_trace_agent_name(trace_dict),
            project_name=_extract_trace_project_name(trace_dict),
            timestamp=_parse_trace_timestamp(trace_dict.get("timestamp")),
        )
        enqueued += 1

    return enqueued


@router.get("/visibility-options")
async def get_evaluation_visibility_options(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
) -> Dict[str, Any]:
    """Return organisations and departments for visibility selectors."""
    org_ids, dept_pairs = await _get_eval_scope_memberships(session, current_user.id)
    role = normalize_role(str(current_user.role))

    organizations: list[dict] = []
    if role == "root":
        org_rows = (await session.exec(select(Organization.id, Organization.name).where(Organization.status == "active"))).all()
        organizations = [{"id": str(r[0]), "name": r[1]} for r in org_rows]
    elif org_ids:
        org_rows = (
            await session.exec(select(Organization.id, Organization.name).where(Organization.id.in_(list(org_ids)), Organization.status == "active"))
        ).all()
        organizations = [{"id": str(r[0]), "name": r[1]} for r in org_rows]

    dept_ids = {dept_id for _, dept_id in dept_pairs}
    departments: list[dict] = []
    if role == "root":
        dept_rows = (await session.exec(select(Department.id, Department.name, Department.org_id).where(Department.status == "active"))).all()
        departments = [{"id": str(r[0]), "name": r[1], "org_id": str(r[2])} for r in dept_rows]
    elif role == "super_admin" and org_ids:
        dept_rows = (
            await session.exec(
                select(Department.id, Department.name, Department.org_id).where(Department.org_id.in_(list(org_ids)), Department.status == "active")
            )
        ).all()
        departments = [{"id": str(r[0]), "name": r[1], "org_id": str(r[2])} for r in dept_rows]
    elif dept_ids:
        dept_rows = (
            await session.exec(
                select(Department.id, Department.name, Department.org_id).where(Department.id.in_(list(dept_ids)), Department.status == "active")
            )
        ).all()
        departments = [{"id": str(r[0]), "name": r[1], "org_id": str(r[2])} for r in dept_rows]

    return {
        "organizations": organizations,
        "departments": departments,
        "role": role,
    }


@router.post("/configs")
async def create_evaluator_config(
    payload: EvaluatorCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    environment: Annotated[str | None, Query(description="'uat' or 'production'")] = None,
) -> EvaluatorResponse:
    """Create a reusable evaluator configuration and optionally run on existing traces."""
    try:
        normalized_target = _normalize_targets(payload.target)
        invalid_targets = [target for target in normalized_target if target not in {"existing", "new"}]
        if invalid_targets:
            raise HTTPException(status_code=400, detail=f"Invalid target value(s): {', '.join(invalid_targets)}")
        normalized_agent_id = _normalize_agent_id(payload.agent_id)
        normalized_agent_ids = _normalize_agent_ids(payload.agent_ids)
        validate_ground_truth_requirement(payload.preset_id, payload.ground_truth)

        from_ts = _parse_iso_datetime_or_400(payload.ts_from, "ts_from")
        to_ts = _parse_iso_datetime_or_400(payload.ts_to, "ts_to")

        # Resolve model from registry
        effective_model, effective_api_key, _ = await _resolve_model_from_registry(payload.model_registry_id, session=session)

        async with session_scope() as session:
            evaluator = Evaluator(
                name=payload.name,
                criteria=payload.criteria,
                model=effective_model,
                model_registry_id=payload.model_registry_id,
                preset_id=payload.preset_id,
                ground_truth=payload.ground_truth,
                target=normalized_target,
                trace_id=payload.trace_id,
                agent_id=normalized_agent_id,
                agent_ids=normalized_agent_ids or None,
                agent_name=payload.agent_name,
                session_id=payload.session_id,
                project_name=payload.project_name,
                ts_from=from_ts,
                ts_to=to_ts,
                user_id=current_user.id,
            )
            session.add(evaluator)
            await session.commit()
            await session.refresh(evaluator)

        eid = str(evaluator.id)
        logger.info(
            f"evaluation - Created evaluator config in DB: id={eid}, user={current_user.id}, target={normalized_target}"
        )

        # If target includes 'existing', fetch matching traces and enqueue judge tasks.
        if "existing" in normalized_target:
            _, lf_client = await _get_scoped_langfuse_for_evaluation(session, current_user)
            enqueued = await _enqueue_existing_trace_evaluations(
                background_tasks=background_tasks,
                user_id=str(current_user.id),
                evaluator_name=payload.name,
                criteria=payload.criteria,
                model=effective_model,
                trace_id=payload.trace_id,
                agent_id=normalized_agent_id,
                agent_ids=normalized_agent_ids,
                agent_name=payload.agent_name,
                session_id=payload.session_id,
                project_name=payload.project_name,
                ts_from=from_ts,
                ts_to=to_ts,
                model_api_key=effective_api_key,
                preset_id=payload.preset_id,
                ground_truth=payload.ground_truth,
                environment=environment,
                langfuse_client=lf_client,
            )
            logger.info(f"evaluation - Enqueued {enqueued} judge tasks for evaluator id={eid}")

        response_data = evaluator.to_response()
        return EvaluatorResponse(**response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=True).error("Error creating evaluator config: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configs/{config_id}/run")
async def run_evaluator_config(
    config_id: str,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: DbSession,
    environment: Annotated[str | None, Query(description="'uat' or 'production'")] = None,
) -> Dict[str, Any]:
    """Run an existing saved evaluator against matching existing traces."""
    try:
        async with session_scope() as session:
            try:
                eval_obj = await session.get(Evaluator, UUID(config_id))
            except Exception:
                raise HTTPException(status_code=404, detail="Evaluator not found")
            if not eval_obj:
                raise HTTPException(status_code=404, detail="Evaluator not found")
            if not _is_evaluator_owner(eval_obj, current_user):
                raise HTTPException(status_code=404, detail="Evaluator not found")

        normalized_target = _normalize_targets(eval_obj.target)
        if "existing" not in normalized_target:
            logger.info(
                "evaluation - Skipping manual run for evaluator id={} user={} target={}",
                config_id,
                current_user.id,
                normalized_target,
            )
            return {
                "status": "noop",
                "config_id": config_id,
                "enqueued": 0,
                "target": normalized_target,
                "message": "Evaluator is configured for new traces only. It will run automatically on new traces.",
            }

        # Resolve model from registry at runtime
        if eval_obj.model_registry_id:
            effective_model, effective_api_key, _ = await _resolve_model_from_registry(eval_obj.model_registry_id, session=session)
        else:
            effective_model = eval_obj.model or "gpt-4o"
            effective_api_key = None

        _, lf_client = await _get_scoped_langfuse_for_evaluation(session, current_user)
        enqueued = await _enqueue_existing_trace_evaluations(
            background_tasks=background_tasks,
            user_id=str(current_user.id),
            evaluator_name=eval_obj.name,
            criteria=eval_obj.criteria,
            model=effective_model,
            trace_id=eval_obj.trace_id,
            agent_id=eval_obj.agent_id,
            agent_ids=eval_obj.agent_ids,
            agent_name=eval_obj.agent_name,
            session_id=eval_obj.session_id,
            project_name=eval_obj.project_name,
            ts_from=eval_obj.ts_from,
            ts_to=eval_obj.ts_to,
            model_api_key=effective_api_key,
            preset_id=eval_obj.preset_id,
            ground_truth=eval_obj.ground_truth,
            environment=environment,
            langfuse_client=lf_client,
        )
        logger.info(
            "evaluation - Enqueued {} judge tasks for existing evaluator id={} user={}",
            enqueued,
            config_id,
            current_user.id,
        )
        return {
            "status": "queued",
            "config_id": config_id,
            "enqueued": enqueued,
            "target": normalized_target,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=True).error("Error running evaluator config: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs")
async def list_evaluator_configs(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> List[EvaluatorResponse]:
    """List evaluator configs owned by the current user."""
    try:
        async with session_scope() as session:
            evaluators = (await session.exec(
                select(Evaluator).where(Evaluator.user_id == current_user.id)
            )).all()
        return [EvaluatorResponse(**e.to_response()) for e in evaluators]
    except Exception as e:
        logger.opt(exception=True).error("Error listing evaluator configs: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/configs/{config_id}")
async def update_evaluator_config(
    config_id: str,
    payload: EvaluatorCreateRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> EvaluatorResponse:
    """Update an existing evaluator config."""
    try:
        validate_ground_truth_requirement(payload.preset_id, payload.ground_truth)
        normalized_target = _normalize_targets(payload.target)
        invalid_targets = [target for target in normalized_target if target not in {"existing", "new"}]
        if invalid_targets:
            raise HTTPException(status_code=400, detail=f"Invalid target value(s): {', '.join(invalid_targets)}")
        normalized_agent_id = _normalize_agent_id(payload.agent_id)
        normalized_agent_ids = _normalize_agent_ids(payload.agent_ids)
        from_ts = _parse_iso_datetime_or_400(payload.ts_from, "ts_from")
        to_ts = _parse_iso_datetime_or_400(payload.ts_to, "ts_to")

        # Fetch evaluator from DB
        async with session_scope() as session:
            try:
                eval_obj = await session.get(Evaluator, UUID(config_id))
            except Exception:
                raise HTTPException(status_code=404, detail="Evaluator not found")
            if not eval_obj:
                raise HTTPException(status_code=404, detail="Evaluator not found")

            if not _is_evaluator_owner(eval_obj, current_user):
                raise HTTPException(status_code=403, detail="Not authorized to edit evaluator")

            # Resolve model from registry
            effective_model, _, _ = await _resolve_model_from_registry(payload.model_registry_id, session=session)

            eval_obj.name = payload.name
            eval_obj.criteria = payload.criteria
            eval_obj.model = effective_model
            eval_obj.model_registry_id = payload.model_registry_id
            eval_obj.preset_id = payload.preset_id
            eval_obj.ground_truth = payload.ground_truth
            eval_obj.target = normalized_target
            eval_obj.trace_id = payload.trace_id
            eval_obj.agent_id = normalized_agent_id
            eval_obj.agent_ids = normalized_agent_ids or None
            eval_obj.agent_name = payload.agent_name
            eval_obj.session_id = payload.session_id
            eval_obj.project_name = payload.project_name
            eval_obj.ts_from = from_ts
            eval_obj.ts_to = to_ts

            session.add(eval_obj)
            await session.commit()
            await session.refresh(eval_obj)

        return EvaluatorResponse(**eval_obj.to_response())
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=True).error("Error updating evaluator config: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/configs/{config_id}")
async def delete_evaluator_config(
    config_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> Dict[str, str]:
    """Delete an evaluator config."""
    try:
        async with session_scope() as session:
            try:
                eval_obj = await session.get(Evaluator, UUID(config_id))
            except Exception:
                raise HTTPException(status_code=404, detail="Evaluator not found")
            if not eval_obj:
                raise HTTPException(status_code=404, detail="Evaluator not found")

            if not _is_evaluator_owner(eval_obj, current_user):
                raise HTTPException(status_code=403, detail="Not authorized to delete evaluator")

            await session.delete(eval_obj)
            await session.commit()
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=True).error("Error deleting evaluator config: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))
