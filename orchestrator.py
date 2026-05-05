
from __future__ import annotations

import asyncio
import json
import os
from io import BytesIO
from pathlib import Path
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

_request_base_url: ContextVar[str | None] = ContextVar("_request_base_url", default=None)

import httpx

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, status

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, or_, true
from sqlmodel import select

from fastapi.responses import StreamingResponse

from agentcore.api.utils import CurrentActiveUser, DbSession
from agentcore.services.auth.decorators import PermissionChecker
from agentcore.services.database.models.agent.model import Agent
from agentcore.events.event_manager import create_default_event_manager
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
from agentcore.services.database.models.department.model import Department
from agentcore.services.database.models.role.model import Role
from agentcore.services.database.models.user_organization_membership.model import (
    UserOrganizationMembership,
)
from agentcore.services.database.models.user.model import User
from agentcore.services.database.models.orch_conversation.model import OrchConversationTable
from agentcore.services.database.models.orch_conversation.crud import (
    orch_add_message,
    orch_archive_session,
    orch_delete_session,
    orch_get_active_agent,
    orch_get_messages,
    orch_get_sessions,
    orch_rename_session,
    orch_set_session_title,
)
from agentcore.services.database.models.orch_transaction.crud import (
    orch_delete_session_transactions,
)
from agentcore.services.deps import get_settings_service
from agentcore.services.job_queue.redis_build_events import RedisBuildEventStore, get_redis_job_event_store
router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])


async def _resolve_super_admin_user_id(
    *,
    session: DbSession,
    org_id: UUID | None,
) -> UUID | None:
    if not org_id:
        return None
    stmt = (
        select(User)
        .join(UserOrganizationMembership, UserOrganizationMembership.user_id == User.id)
        .join(Role, Role.id == UserOrganizationMembership.role_id)
        .where(
            UserOrganizationMembership.org_id == org_id,
            UserOrganizationMembership.status == "active",
            func.lower(Role.name) == "super_admin",
        )
        .order_by(User.create_at.asc())
    )
    rows = (await session.exec(stmt)).all()
    return rows[0].id if rows else None


async def _designated_super_admin_org_ids(
    session: DbSession,
    current_user: CurrentActiveUser,
) -> set[UUID]:
    role = str(getattr(current_user, "role", "")).lower()
    if role != "super_admin":
        return set()
    rows = (
        await session.exec(
            select(UserOrganizationMembership.org_id).where(
                UserOrganizationMembership.user_id == current_user.id,
                UserOrganizationMembership.status == "active",
            )
        )
    ).all()
    org_ids = {r if isinstance(r, UUID) else r[0] for r in rows}
    if not org_ids:
        return set()
    allowed: set[UUID] = set()
    for org_id in org_ids:
        super_admin_id = await _resolve_super_admin_user_id(session=session, org_id=org_id)
        if super_admin_id == current_user.id:
            allowed.add(org_id)
    return allowed


async def _department_admin_dept_ids(
    session: DbSession,
    current_user: CurrentActiveUser,
) -> set[UUID]:
    role = str(getattr(current_user, "role", "")).lower()
    if role != "department_admin":
        return set()
    rows = (
        await session.exec(
            select(Department.id).where(Department.admin_user_id == current_user.id)
        )
    ).all()
    return {r if isinstance(r, UUID) else r[0] for r in rows}

class OrchAgentSummary(BaseModel):
    deploy_id: UUID
    agent_id: UUID
    agent_name: str
    agent_description: str | None = None
    version_number: int
    version_label: str
    environment: str
    promoted_from_uat_id: UUID | None = None
    source_uat_version_number: int | None = None


class OrchChatRequest(BaseModel):
    session_id: str
    agent_id: UUID | None = None
    deployment_id: UUID | None = None
    model_id: UUID | None = None  # selected model from registry (for direct model chat)
    input_value: str
    version_number: int | None = None
    env: str | None = None  # "uat" or "prod"
    files: list[str] | None = None
    enable_reasoning: bool = False  # enable CoT reasoning if model supports it
    image_mode: bool = False  # explicit image-generation mode (fast path, skips intent classification)
    # MiBuddy-style canvas flag. Frontend sends `canvasEnabled`; pydantic
    # aliases handle the camelCase → snake_case mapping.
    canvas_enabled: bool = False


class OrchMessageResponse(BaseModel):
    id: UUID
    timestamp: str
    sender: str
    sender_name: str
    session_id: str
    text: str
    agent_id: UUID | None = None
    deployment_id: UUID | None = None
    model_id: UUID | None = None
    model_name: str | None = None
    reasoning_content: str | None = None
    category: str = "message"
    files: list[str] | None = None
    properties: dict | None = None
    content_blocks: list | None = None
    # Per-message thumbs up/down feedback — populated for assistant messages the
    # user has rated. NULL on user messages and on unrated assistant messages.
    feedback_rating: str | None = None
    feedback_reasons: list[str] | None = None
    feedback_comment: str | None = None
    feedback_at: str | None = None


class OrchChatResponse(BaseModel):
    session_id: str
    agent_name: str
    message: OrchMessageResponse
    context_reset: bool = False


class EditMessageRequest(BaseModel):
    """Payload for editing a past user message (MiBuddy-style in-place UPSERT)."""
    edited_text: str
    enable_reasoning: bool = False
    image_mode: bool = False
    # Currently-selected model from the UI dropdown. Carried so the edit
    # endpoint can seed the router with the user's current choice (same as a
    # fresh send) — intent classification still drives mode selection. The
    # edit endpoint deliberately ignores `image_mode` for routing purposes
    # because the edited text may imply a different intent than the original.
    model_id: UUID | None = None


class EditMessageResponse(BaseModel):
    """Response from the edit endpoint — contains updated user + agent messages."""
    user_message: OrchMessageResponse
    agent_message: OrchMessageResponse


class FeedbackRequest(BaseModel):
    """Thumbs up/down feedback payload for an assistant message.

    MiBuddy-parity: the popup lets the user pick a rating, optionally tag it
    with preset reason chips, and add a free-text comment (≤ 300 chars).
    """
    rating: Literal["up", "down"]
    reasons: list[str] = Field(default_factory=list, max_length=5)
    comment: str | None = Field(default=None, max_length=300)


class FeedbackResponse(BaseModel):
    """Current feedback state on the message after the operation."""
    message_id: UUID
    feedback_rating: str | None = None
    feedback_reasons: list[str] | None = None
    feedback_comment: str | None = None
    feedback_at: str | None = None


class OrchModelSummary(BaseModel):
    model_id: UUID
    display_name: str
    provider: str
    model_name: str
    model_type: str = "llm"
    capabilities: dict | None = None
    is_default: bool = False


class OrchSessionSummary(BaseModel):
    session_id: str
    last_timestamp: str | None = None
    preview: str = ""
    active_agent_id: UUID | None = None
    active_deployment_id: UUID | None = None
    active_agent_name: str | None = None
    is_archived: bool = False
    # User-chosen title (null = auto-derived from preview/agent name)
    session_title: str | None = None



def _best_from_message(msg: Any) -> str | None:
    """Try to pull a human-readable string from a message-like value."""
    if isinstance(msg, dict):
        candidates = [
            msg.get("message"),
            msg.get("text"),
            msg.get("data", {}).get("text") if isinstance(msg.get("data"), dict) else None,
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        # Recurse into nested dict/list values to handle wrappers like {"result": {message_dict}}
        for value in msg.values():
            if isinstance(value, (dict, list)):
                text = _best_from_message(value)
                if text:
                    return text
    elif isinstance(msg, list):
        for item in msg:
            text = _best_from_message(item)
            if text:
                return text
    if isinstance(msg, str) and msg.strip():
        return msg
    return None


def _extract_text(payload: Any) -> str:
    """Extract a human-readable response from a serialized RunResponse dict."""
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        outputs = payload.get("outputs") or []
        for run_output in outputs:
            if not isinstance(run_output, dict):
                continue
            for result_entry in run_output.get("outputs") or []:
                if not isinstance(result_entry, dict):
                    continue
                text = (
                    _best_from_message(result_entry.get("results"))
                    or _best_from_message(result_entry.get("outputs"))
                    or _best_from_message(result_entry.get("messages"))
                )
                if text:
                    return text
        # Fallback: search all top-level values (skip session_id which is a UUID, not a response)
        for key, value in payload.items():
            if key == "session_id":
                continue
            text = _best_from_message(value)
            if text:
                return text

    if isinstance(payload, list):
        for item in payload:
            text = _extract_text(item)
            if text:
                return text

    try:
        return json.dumps(payload, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(payload)


def _pick_best_text(candidates: list[str]) -> str:
    """Choose the most human-readable candidate text from multiple options."""
    cleaned: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        text = candidate.strip()
        if text:
            cleaned.append(text)

    if not cleaned:
        return ""

    def _score(text: str) -> tuple[int, int]:
        score = 0
        if any(ch.isalpha() for ch in text):
            score += 2
        if any(ch.isspace() for ch in text):
            score += 3
        if any(ch in ".!?" for ch in text):
            score += 1
        if text.startswith("{") or text.startswith("["):
            score -= 4
        if text.count("-") >= 4 and " " not in text and len(text) >= 32:
            score -= 3
        return score, len(text)

    return max(cleaned, key=_score)


def _normalize_content_blocks(content_blocks: Any) -> list[dict]:
    """Normalize content_blocks to a JSON-serializable list of dicts."""
    if not isinstance(content_blocks, list):
        return []

    normalized: list[dict] = []
    for block in content_blocks:
        if hasattr(block, "model_dump"):
            try:
                block = block.model_dump()
            except Exception:  # noqa: BLE001
                continue
        if isinstance(block, dict):
            normalized.append(block)
    return normalized


def _dedupe_content_blocks(content_blocks: list[dict]) -> list[dict]:
    """Deduplicate content blocks while preserving order."""
    unique: list[dict] = []
    seen: set[str] = set()

    for block in content_blocks:
        try:
            signature = json.dumps(block, sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            signature = str(block)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(block)
    return unique


def _score_content_blocks(content_blocks: list[dict]) -> tuple[int, int]:
    """Score block lists by total content items first, then block count."""
    content_items = 0
    for block in content_blocks:
        contents = block.get("contents")
        if isinstance(contents, list):
            content_items += len(contents)
    return content_items, len(content_blocks)


def _pick_best_content_blocks(candidates: list[list[dict]]) -> list[dict]:
    """Pick the richest content_blocks candidate from multiple sources."""
    cleaned: list[list[dict]] = []
    for candidate in candidates:
        normalized = _dedupe_content_blocks(_normalize_content_blocks(candidate))
        if normalized:
            cleaned.append(normalized)
    if not cleaned:
        return []
    return max(cleaned, key=_score_content_blocks)


def _extract_content_blocks(payload: Any) -> list[dict]:
    """Recursively extract all content_blocks lists from a run payload."""
    collected: list[dict] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if "content_blocks" in node:
                collected.extend(_normalize_content_blocks(node.get("content_blocks")))
            for value in node.values():
                if isinstance(value, dict | list):
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, dict | list):
                    _walk(item)

    _walk(payload)
    return _dedupe_content_blocks(collected)


def _is_interrupted_payload(payload: Any) -> bool:
    """Return True when a /run payload indicates a HITL interrupt."""
    if not isinstance(payload, dict):
        return False

    if payload.get("interrupted") is True:
        return True

    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        return False

    for run_output in outputs:
        if not isinstance(run_output, dict):
            continue

        # Canonical path for LangGraph RunOutputs metadata.
        run_meta = run_output.get("metadata")
        if isinstance(run_meta, dict) and str(run_meta.get("status", "")).lower() == "interrupted":
            return True

        # Legacy / alternative nested result shapes.
        for result_entry in run_output.get("outputs") or []:
            if not isinstance(result_entry, dict):
                continue
            result_meta = result_entry.get("metadata")
            if isinstance(result_meta, dict) and str(result_meta.get("status", "")).lower() == "interrupted":
                return True
            nested_results = result_entry.get("results")
            if isinstance(nested_results, dict):
                nested_meta = nested_results.get("metadata")
                if isinstance(nested_meta, dict) and str(nested_meta.get("status", "")).lower() == "interrupted":
                    return True

    return False


def _get_orchestrator_event_store() -> RedisBuildEventStore | None:
    try:
        settings_service = get_settings_service()
        return get_redis_job_event_store(settings_service, namespace="orchestrator_events")
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Orchestrator Redis event store unavailable: {exc}")
        return None


async def _create_orchestrator_redis_response(
    *,
    job_id: str,
    event_store: RedisBuildEventStore,
) -> StreamingResponse:
    async def consume_and_yield():
        def _has_end_event(payload: bytes | str) -> bool:
            try:
                text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
            except Exception:
                return False
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    if line.split(":", 1)[1].strip() == "end":
                        return True
                    continue
                if line.startswith("data:"):
                    line = line.split(":", 1)[1].strip()
                if "\"event\"" not in line:
                    continue
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict) and parsed.get("event") == "end":
                        return True
                except Exception:
                    if "\"event\":\"end\"" in line or "\"event\": \"end\"" in line:
                        return True
            return False

        cursor = 0
        saw_end_event = False
        terminal_idle_polls = 0
        last_total = -1
        while True:
            try:
                events = await event_store.get_events_from(job_id, cursor)
                for payload in events:
                    if not saw_end_event and _has_end_event(payload):
                        saw_end_event = True
                    yield payload
                cursor += len(events)

                status = await event_store.get_status(job_id)
                if status in RedisBuildEventStore.TERMINAL_STATUSES:
                    total = await event_store.get_events_count(job_id)
                    if total != last_total:
                        terminal_idle_polls = 0
                        last_total = total
                    elif not events:
                        terminal_idle_polls += 1

                    if cursor >= total and saw_end_event:
                        break
                    if cursor >= total and terminal_idle_polls >= 20:
                        logger.warning(
                            f"[ORCH-STREAM] Redis stream closed without end event for job {job_id} "
                            f"(status={status}, total={total}, cursor={cursor})"
                        )
                        break
                elif not events and status is None and not await event_store.job_exists(job_id):
                    break
                else:
                    terminal_idle_polls = 0
                    last_total = -1

                await asyncio.sleep(0.05)
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"[ORCH-STREAM] Error streaming Redis events for job {job_id}: {exc}")
                break

    return StreamingResponse(
        consume_and_yield(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _orch_call_run_api(
    *,
    agent_id: str,
    env: str,
    version: str,
    input_value: str,
    session_id: str,
    files: list[str] | None = None,
    stream: bool = False,
    event_manager=None,
    orch_deployment_id: str | None = None,
    orch_session_id: str | None = None,
    orch_org_id: str | None = None,
    orch_dept_id: str | None = None,
    orch_user_id: str | None = None,
    user_id: str | None = None,
) -> tuple[str, bool, list]:
    """Call POST /api/v1/run/{agent_id} internally with the AGENTCORE_INTERNAL_SECRET header.

    Returns (response_text, was_interrupted, content_blocks).
    For streaming, SSE token/add_message events are forwarded to event_manager;
    the function waits for the 'end' event to obtain the final text.
    """
    base_url = os.environ.get("ORCHESTRATOR_BASE_URL") or _request_base_url.get()
    if not base_url:
        raise RuntimeError(
            "Orchestrator base URL is not configured. "
            "Set ORCHESTRATOR_BASE_URL."
        )
    logger.info(f"[ORCH] base_url resolved to: {base_url}")
    secret = os.environ.get("AGENTCORE_INTERNAL_SECRET", "")
    url = (
        f"{base_url}/api/run/{agent_id}"
        f"?env={env}&version={version}&stream={str(stream).lower()}"
    )
    logger.info(f"[ORCH] calling run API: {url} | stream={stream}")
    body: dict = {"input_value": input_value, "session_id": session_id}
    if files:
        body["files"] = files
    headers = {"X-Internal-Secret": secret, "Content-Type": "application/json"}
    if orch_deployment_id:
        headers["X-Orch-Deployment-Id"] = orch_deployment_id
    if orch_session_id:
        headers["X-Orch-Session-Id"] = orch_session_id
    if orch_org_id:
        headers["X-Orch-Org-Id"] = orch_org_id
    if orch_dept_id:
        headers["X-Orch-Dept-Id"] = orch_dept_id
    if orch_user_id:
        headers["X-Orch-User-Id"] = orch_user_id
    if user_id:
        headers["X-Orch-User-Id"] = user_id

    if not stream:
        async with httpx.AsyncClient(timeout=300, verify=False) as client:
            resp = await client.post(url, json=body, headers=headers)
            logger.info(f"[ORCH] run API response: status={resp.status_code}")
            resp.raise_for_status()
        payload = resp.json()
        text = _extract_text(payload)
        content_blocks = _extract_content_blocks(payload)
        interrupted = _is_interrupted_payload(payload)
        logger.info(
            f"[ORCH] run API completed | interrupted={interrupted} | "
            f"response_length={len(text)} | content_blocks={len(content_blocks)}"
        )
        return text, interrupted, content_blocks

    # --- Streaming: forward SSE events to event_manager, collect final text ---
    final_text = ""
    token_chunks: list[str] = []
    latest_agent_add_message_text = ""
    message_text_by_id: dict[str, str] = {}
    active_message_id: str | None = None
    message_content_blocks_by_id: dict[str, list[dict]] = {}
    latest_agent_content_blocks: list[dict] = []
    end_content_blocks: list[dict] = []
    saw_end_event = False
    pending_sse_event_type = ""
    was_interrupted = False

    def _extract_stream_text(data: Any) -> str:
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            return ""
        candidates = [
            data.get("chunk"),
            data.get("text"),
            data.get("token"),
            data.get("message"),
            data.get("data", {}).get("text") if isinstance(data.get("data"), dict) else None,
            data.get("data", {}).get("chunk") if isinstance(data.get("data"), dict) else None,
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate:
                return candidate
        return ""

    def _reconstructed_stream_text() -> str:
        token_text = "".join(token_chunks).strip()
        message_candidates = [
            text.strip()
            for text in message_text_by_id.values()
            if isinstance(text, str) and text.strip()
        ]
        combined_text = ""
        if latest_agent_add_message_text and token_text:
            if token_text.startswith(latest_agent_add_message_text):
                combined_text = token_text
            elif latest_agent_add_message_text.endswith(token_text):
                combined_text = latest_agent_add_message_text
            else:
                combined_text = f"{latest_agent_add_message_text}{token_text}"
        return _pick_best_text(
            [*message_candidates, combined_text, token_text, latest_agent_add_message_text]
        )

    async with httpx.AsyncClient(timeout=300, verify=False) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            logger.info(f"[ORCH] stream started: status={resp.status_code}")
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line:
                    pending_sse_event_type = ""
                    continue

                if line.startswith("event:"):
                    pending_sse_event_type = line.split(":", 1)[1].strip()
                    continue

                payload_line = line.split(":", 1)[1].strip() if line.startswith("data:") else line

                try:
                    parsed = json.loads(payload_line)
                except Exception:  # noqa: BLE001
                    continue

                if isinstance(parsed, dict) and "event" in parsed:
                    evt = parsed
                elif isinstance(parsed, dict) and pending_sse_event_type:
                    evt = {"event": pending_sse_event_type, "data": parsed}
                else:
                    continue

                etype = evt.get("event", "")
                edata = evt.get("data", {})
                if etype == "token":
                    chunk = _extract_stream_text(edata)
                    if isinstance(chunk, str) and chunk:
                        token_chunks.append(chunk)
                        token_id = ""
                        if isinstance(edata, dict):
                            token_id = str(edata.get("id") or edata.get("message_id") or "").strip()
                        target_id = token_id or active_message_id
                        if target_id:
                            existing = message_text_by_id.get(target_id, "")
                            if not existing.endswith(chunk):
                                message_text_by_id[target_id] = f"{existing}{chunk}" if existing else chunk
                            active_message_id = target_id
                    if event_manager:
                        event_manager.on_token(data=edata)
                elif etype == "add_message":
                    msg_text = _extract_stream_text(edata)
                    sender = (
                        str(edata.get("sender") or edata.get("sender_name") or "").lower()
                        if isinstance(edata, dict)
                        else ""
                    )
                    msg_id = (
                        str(edata.get("id") or edata.get("message_id") or "").strip()
                        if isinstance(edata, dict)
                        else ""
                    )
                    if isinstance(msg_text, str) and msg_text.strip() and "user" not in sender:
                        latest_agent_add_message_text = msg_text
                        target_id = msg_id or active_message_id or "__orch_agent_msg__"
                        message_text_by_id[target_id] = msg_text
                        active_message_id = target_id
                    msg_blocks = _extract_content_blocks(edata)
                    if msg_blocks and "user" not in sender:
                        target_id = msg_id or active_message_id or "__orch_agent_msg__"
                        merged_blocks = _pick_best_content_blocks(
                            [message_content_blocks_by_id.get(target_id, []), msg_blocks]
                        )
                        message_content_blocks_by_id[target_id] = merged_blocks
                        latest_agent_content_blocks = merged_blocks
                        active_message_id = target_id
                    if event_manager:
                        event_manager.on_message(data=edata)
                elif etype == "end":
                    saw_end_event = True
                    result = edata.get("result", edata) if isinstance(edata, dict) else edata
                    end_content_blocks = _pick_best_content_blocks(
                        [_extract_content_blocks(result), _extract_content_blocks(edata)]
                    )
                    was_interrupted = _is_interrupted_payload(result)
                    if was_interrupted:
                        final_text = ""
                    else:
                        parsed_text = _extract_text(result)
                        parsed_text = parsed_text if isinstance(parsed_text, str) else str(parsed_text)
                        end_text = _extract_stream_text(edata) if isinstance(edata, dict) else ""
                        reconstructed_text = _reconstructed_stream_text()
                        token_text = "".join(token_chunks).strip()
                        final_text = _pick_best_text(
                            [parsed_text, end_text, reconstructed_text, token_text, latest_agent_add_message_text]
                        )
                    logger.info(
                        "[ORCH] stream ended | "
                        f"interrupted={was_interrupted} "
                        f"response_length={len(final_text)} "
                        f"token_chars={len(''.join(token_chunks))} "
                        f"content_blocks={len(end_content_blocks)}"
                    )
                elif etype == "error":
                    # The run-API emits two flavors of `error` events:
                    #   1. Component-level ErrorMessage sent via send_error()
                    #      — payload has `text="<msg>"` plus `error=True`
                    #        (a boolean flag from MessageEvent), and
                    #   2. run_agent_generator's terminal error
                    #      — payload has `error="<msg>"` (string).
                    # Picking the literal "error" key first turns the boolean
                    # into the user-visible string "False". Prefer `text`,
                    # fall back to `error` only if it is a non-empty string.
                    err_msg: Any = edata.get("text") if isinstance(edata, dict) else None
                    if not (isinstance(err_msg, str) and err_msg.strip()):
                        candidate = edata.get("error") if isinstance(edata, dict) else None
                        err_msg = candidate if isinstance(candidate, str) and candidate.strip() else None
                    if not (isinstance(err_msg, str) and err_msg.strip()):
                        err_msg = "Stream error from /run"
                    raise ValueError(err_msg)

    reconstructed_text = _reconstructed_stream_text()
    if not was_interrupted:
        final_text = _pick_best_text([final_text, reconstructed_text])

    if not saw_end_event:
        logger.warning(
            "[ORCH] stream closed without end event | "
            f"token_chars={len(''.join(token_chunks))} "
            f"add_message_chars={len(latest_agent_add_message_text)} "
            f"reconstructed_chars={len(reconstructed_text)} "
            f"final_chars={len(final_text)}"
        )

    streamed_blocks = list(message_content_blocks_by_id.values())
    final_content_blocks = _pick_best_content_blocks(
        [end_content_blocks, latest_agent_content_blocks, *streamed_blocks]
    )

    return final_text, was_interrupted, final_content_blocks


async def _lookup_agent_project(session: DbSession, agent_id: UUID) -> tuple[str | None, str | None]:
    """Look up the agent's project_id and project_name for observability metadata."""
    try:
        agent = await session.get(Agent, agent_id)
        if agent and agent.project_id:
            project_id = str(agent.project_id)
            project_name = None
            try:
                from agentcore.services.database.models.folder.model import Folder
                folder = await session.get(Folder, agent.project_id)
                if folder:
                    project_name = folder.name
            except Exception:
                pass
            return project_id, project_name
    except Exception:
        pass
    return None, None


def _serialize_content_blocks(content_blocks: list) -> list:
    """Serialize ContentBlock objects to dicts for JSON storage."""
    serialized = []
    for block in content_blocks:
        if hasattr(block, "model_dump"):
            serialized.append(block.model_dump())
        elif isinstance(block, dict):
            serialized.append(block)
    return serialized


async def _resolve_agent(
    session: DbSession,
    current_user: CurrentActiveUser,
    body: OrchChatRequest,
) -> tuple[UUID, UUID, AgentDeploymentProd | AgentDeploymentUAT]:
    """Resolve the target agent for a chat request (sticky routing).

    If agent_id/deployment_id are provided → use them (explicit @mention).
    Otherwise → look up the last active agent in the session.
    Raises 400 if no agent can be resolved (new session with no @mention).
    """
    agent_id = body.agent_id
    deployment_id = body.deployment_id

    if not agent_id or not deployment_id:
        active = await orch_get_active_agent(session, body.session_id)
        if active:
            agent_id = agent_id or active["agent_id"]
            deployment_id = deployment_id or active["deployment_id"]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No agent specified and no active agent in session. Mention an agent with @ to start.",
            )

    deployment = await session.get(AgentDeploymentProd, deployment_id)
    if not deployment:
        deployment = await session.get(AgentDeploymentUAT, deployment_id)
    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found",
        )

    if not await _user_can_access_deployment(session, current_user, deployment):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this deployment.",
        )

    return agent_id, deployment_id, deployment


async def _user_can_access_deployment(
    session: DbSession,
    current_user: CurrentActiveUser,
    deployment: AgentDeploymentProd | AgentDeploymentUAT,
) -> bool:
    role = str(getattr(current_user, "role", "")).lower()
    if role == "root":
        return True
    if role == "department_admin":
        dept_ids = await _department_admin_dept_ids(session, current_user)
        if dept_ids and deployment.dept_id in dept_ids:
            return True
    if role == "super_admin":
        org_ids = await _designated_super_admin_org_ids(session, current_user)
        if org_ids and deployment.org_id in org_ids:
            return True

    if deployment.deployed_by == current_user.id:
        return True

    recipient_exists = (
        await session.exec(
            select(AgentPublishRecipient.id)
            .where(
                or_(
                    AgentPublishRecipient.deploy_id == deployment.id,
                    and_(
                        AgentPublishRecipient.deploy_id.is_(None),
                        AgentPublishRecipient.agent_id == deployment.agent_id,
                    ),
                ),
                AgentPublishRecipient.recipient_user_id == current_user.id,
                or_(
                    deployment.dept_id is None,
                    AgentPublishRecipient.dept_id == deployment.dept_id,
                ),
            )
            .limit(1)
        )
    ).first()
    if recipient_exists:
        return True

    if isinstance(deployment, AgentDeploymentProd):
        visibility_value = (
            deployment.visibility.value
            if hasattr(deployment.visibility, "value")
            else str(deployment.visibility)
        )
        if str(visibility_value).upper() == "PUBLIC":
            # Keep Orchestration aligned with Registry behavior:
            # PUBLIC PROD agents are visible/usable by authenticated users.
            return True

    return False


async def _maybe_context_reset(
    session,
    *,
    session_id: str,
    new_agent_id: UUID,
    new_agent_name: str,
    user_id: UUID,
    new_deployment_id: UUID,
) -> bool:
    """Insert a context-reset system message if the active agent has changed.

    Returns True if a context reset occurred (agent switched).
    """
    active = await orch_get_active_agent(session, session_id)
    if not active or active["agent_id"] == new_agent_id:
        return False

    reset_ts = datetime.now(timezone.utc).replace(tzinfo=None)
    reset_msg = OrchConversationTable(
        id=uuid4(),
        sender="system",
        sender_name="System",
        session_id=session_id,
        text=f"Switched to {new_agent_name}",
        agent_id=new_agent_id,
        user_id=user_id,
        deployment_id=new_deployment_id,
        timestamp=reset_ts,
        files=[],
        properties={},
        category="context_reset",
        content_blocks=[],
    )
    await orch_add_message(reset_msg, session)
    logger.info(f"[ORCH] Context reset: switched to {new_agent_name} in session {session_id}")
    return True



@router.get("/agents", response_model=list[OrchAgentSummary], status_code=200)
async def list_orch_agents(
    *,
    session: DbSession,
    current_user: CurrentActiveUser,
):
    """Return accessible UAT/PROD deployed agents for orchestration chat."""
    try:
        current_role = str(getattr(current_user, "role", "")).lower()
        is_root = current_role == "root"
        dept_ids = await _department_admin_dept_ids(session, current_user)
        org_ids = await _designated_super_admin_org_ids(session, current_user)

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
        prod_private_access = (
            (AgentDeploymentProd.deployed_by == current_user.id)
            | prod_share_exists
        )
        # Keep Orchestration aligned with Registry behavior:
        # PUBLIC PROD agents are visible to authenticated users.
        prod_public_access = true()
        if is_root:
            prod_private_access = prod_private_access | true()
        elif current_role == "department_admin" and dept_ids:
            prod_private_access = prod_private_access | AgentDeploymentProd.dept_id.in_(list(dept_ids))
        elif current_role == "super_admin" and org_ids:
            prod_private_access = prod_private_access | AgentDeploymentProd.org_id.in_(list(org_ids))

        prod_stmt = (
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
        if is_root:
            uat_access = uat_access | true()
        elif current_role == "department_admin" and dept_ids:
            uat_access = uat_access | AgentDeploymentUAT.dept_id.in_(list(dept_ids))
        elif current_role == "super_admin" and org_ids:
            uat_access = uat_access | AgentDeploymentUAT.org_id.in_(list(org_ids))

        uat_stmt = (
            select(AgentDeploymentUAT)
            .where(AgentDeploymentUAT.status == DeploymentUATStatusEnum.PUBLISHED)
            .where(AgentDeploymentUAT.is_active == True)  # noqa: E712
            .where(AgentDeploymentUAT.is_enabled == True)  # noqa: E712
            .where(uat_access)
        )

        prod_records = list((await session.exec(prod_stmt)).all())
        uat_records = list((await session.exec(uat_stmt)).all())

        # Keep all PROD versions. Hide only UAT rows that were promoted to a
        # currently visible PROD deployment. Newer UAT versions for the same
        # agent must still appear (so UAT badge can be shown in orchestration).
        promoted_uat_ids_in_prod = {
            str(rec.promoted_from_uat_id)
            for rec in prod_records
            if rec.promoted_from_uat_id is not None
        }
        filtered_uat_records = [
            rec for rec in uat_records if str(rec.id) not in promoted_uat_ids_in_prod
        ]
        source_uat_version_map: dict[UUID, int] = {}
        promoted_from_uat_ids = [
            rec.promoted_from_uat_id
            for rec in prod_records
            if rec.promoted_from_uat_id is not None
        ]
        if promoted_from_uat_ids:
            source_rows = (
                await session.exec(
                    select(AgentDeploymentUAT.id, AgentDeploymentUAT.version_number).where(
                        AgentDeploymentUAT.id.in_(promoted_from_uat_ids)
                    )
                )
            ).all()
            source_uat_version_map = {
                dep_id: version_number for dep_id, version_number in source_rows
            }

        records_with_env: list[tuple[AgentDeploymentProd | AgentDeploymentUAT, str]] = (
            [(rec, "prod") for rec in prod_records]
            + [(rec, "uat") for rec in filtered_uat_records]
        )
        records_with_env.sort(
            key=lambda row: (
                str(row[0].agent_name or "").lower(),
                -int(getattr(row[0], "version_number", 0) or 0),
            )
        )

        return [
            OrchAgentSummary(
                deploy_id=r.id,
                agent_id=r.agent_id,
                agent_name=r.agent_name,
                agent_description=r.agent_description,
                version_number=r.version_number,
                version_label=f"v{r.version_number}",
                environment=env_name,
                promoted_from_uat_id=getattr(r, "promoted_from_uat_id", None),
                source_uat_version_number=(
                    source_uat_version_map.get(getattr(r, "promoted_from_uat_id", None))
                    if env_name == "prod"
                    else None
                ),
            )
            for r, env_name in records_with_env
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing orchestrator agents: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Intent-based routing helpers
# ---------------------------------------------------------------------------

async def _get_model_display_name(session: DbSession, model_id: UUID) -> str:
    """Look up the display_name of a registry model."""
    from agentcore.services.database.models.model_registry.model import ModelRegistry
    row = await session.get(ModelRegistry, model_id)
    if row:
        return row.display_name
    return "Model"


async def _model_has_capability(session: DbSession, model_id: UUID | None, cap: str) -> bool:
    """Return True when a registry model has a given capability (e.g. `web_search`).

    Used by the web_search / image_gen dispatch branches to decide whether to
    swap the response model to the configured specialist. If the user already
    selected a model that can handle the intent natively (e.g. Gemini 3 Pro
    with `web_search=true`), keep their selection — otherwise the dropdown
    would flip to a different model that may be registered with a provider
    (like `google` using Generative Language API) that can't serve the
    follow-up general-chat turn in the same session.
    """
    if not model_id:
        return False
    try:
        from agentcore.services.database.models.model_registry.model import ModelRegistry
        from agentcore.services.mibuddy.model_capabilities import detect_capabilities

        row = await session.get(ModelRegistry, model_id)
        if not row:
            return False
        caps = detect_capabilities(
            row.provider,
            row.model_name,
            row.capabilities,
            getattr(row, "provider_config", None),
        )
        return bool(caps.get(cap))
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[ORCH] Capability probe failed for {model_id} / {cap}: {exc}")
        return False


async def _collect_session_doc_filenames(
    session_id: str,
    user_id: UUID | None = None,
) -> list[str]:
    """Aggregate doc filenames uploaded across a whole chat session.

    Walks user messages in the session (by timestamp) and unions their `files`
    field, keeping only entries whose extension is a supported doc type.
    Returns just `Path(p).name` — the filenames are what Pinecone's
    `source_file` metadata is keyed on for per-file search.

    This is the authoritative "which files has the user uploaded in this
    session?" source. Earlier we relied on a semantic Pinecone probe, but
    that missed files whose chunks didn't happen to match the current
    question (e.g. a follow-up question about HR policies would pull chunks
    from the HR PDFs and leave a tech-spec .txt invisible — the file would
    then silently drop out of the per-file search).
    """
    from agentcore.services.deps import session_scope
    from agentcore.services.mibuddy.document_extractor import SUPPORTED_DOC_EXTENSIONS

    async with session_scope() as db:
        stmt = (
            select(OrchConversationTable)
            .where(
                OrchConversationTable.session_id == session_id,
                OrchConversationTable.sender == "user",
            )
            .order_by(OrchConversationTable.timestamp.asc())
        )
        if user_id is not None:
            stmt = stmt.where(OrchConversationTable.user_id == user_id)
        rows = (await db.exec(stmt)).all()

    seen: set[str] = set()
    names: list[str] = []
    for row in rows:
        for p in (row.files or []):
            name = Path(p).name
            if Path(p).suffix.lower() in SUPPORTED_DOC_EXTENSIONS and name not in seen:
                seen.add(name)
                names.append(name)
    return names


async def _route_request(
    session: DbSession,
    current_user: CurrentActiveUser,
    body: OrchChatRequest,
) -> dict:
    """Determine routing mode for a chat request.

    Returns dict with:
      mode: "agent" | "model_direct" | "web_search" | "image_gen" | "document_qa"
      agent_id, deployment_id, deployment: for agent mode
      model_id: for model_direct mode
      intent: classified intent string
      doc_files: list of document file paths (for document_qa mode)
      image_files: list of image file paths
    """
    from agentcore.services.mibuddy.document_extractor import IMAGE_EXTENSIONS, SUPPORTED_DOC_EXTENSIONS

    # Priority 0: Explicit @agent mention wins over everything else, including
    # attached files. The agent's own ChatInput → resolve_attachments path
    # handles file extraction, so we must NOT short-circuit into document_qa.
    # (Previously document_qa had Priority 0 and beat @agent, which made
    # @AgentName + file upload always land in MiBuddy's RAG flow.)
    if body.agent_id or body.deployment_id:
        agent_id, deployment_id, deployment = await _resolve_agent(session, current_user, body)
        return {
            "mode": "agent",
            "agent_id": agent_id,
            "deployment_id": deployment_id,
            "deployment": deployment,
            "model_id": None,
            "intent": None,
        }

    # Priority 0.5: Sticky session continuation. If this session was already
    # bound to an agent (i.e. a previous turn @-mentioned one and it ran),
    # follow-ups stay with that agent — including follow-ups that attach
    # files. Without this, message 2 in an agent session would silently fall
    # into document_qa just because the user re-attached a doc.
    # Fresh sessions (no prior agent run) are NOT caught here and continue
    # to document_qa / intent classification below — that's intended.
    sticky_active = await orch_get_active_agent(session, body.session_id)
    if sticky_active:
        sticky_deployment = await session.get(AgentDeploymentProd, sticky_active["deployment_id"])
        if not sticky_deployment:
            sticky_deployment = await session.get(AgentDeploymentUAT, sticky_active["deployment_id"])
        if sticky_deployment and await _user_can_access_deployment(
            session, current_user, sticky_deployment
        ):
            return {
                "mode": "agent",
                "agent_id": sticky_active["agent_id"],
                "deployment_id": sticky_active["deployment_id"],
                "deployment": sticky_deployment,
                "model_id": None,
                "intent": None,
            }

    # Priority 1: No @agent and no sticky agent, but document files attached
    # → document_qa mode (fresh-session RAG flow, unchanged).
    if body.files:
        doc_files = [
            f for f in body.files
            if Path(f).suffix.lower() in SUPPORTED_DOC_EXTENSIONS
        ]
        image_files = [
            f for f in body.files
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ]
        if doc_files:
            model_id = body.model_id
            if not model_id:
                settings = get_settings_service()
                default_id = settings.settings.default_chat_model_id
                if default_id:
                    model_id = UUID(default_id)
            return {
                "mode": "document_qa",
                "agent_id": None,
                "deployment_id": None,
                "deployment": None,
                "model_id": model_id,
                "intent": "document_processing",
                "doc_files": doc_files,
                "image_files": image_files,
            }

    # Priority 0.5 (removed): previously force-routed to web_search or image_gen
    # based on the selected model's capability flags. That was too aggressive
    # for multi-purpose models (e.g. Gemini 3 Pro is registered with
    # web_search=true but is also used for plain chat and image generation) —
    # it would push every query through web_search, including "create image"
    # requests. Routing now relies on:
    #   - the explicit `image_mode=True` flag (set by the frontend when a
    #     dedicated image model like Nano Banana is selected) — caught by the
    #     fast-path below.
    #   - the intent classifier, which inspects the prompt text.
    # The dispatch-side "settings override" (web_search_model_name /
    # image_gen_model_name) swaps the responding model to the configured
    # specialist when the intent doesn't match the selected model.

    # Fast path: explicit image_mode flag from frontend — skip intent classification.
    # When the user has picked a specific image model (body.model_id set), mark
    # the intent as "image_generation_explicit" so the dispatch branch uses THAT
    # model (e.g. DALL-E if the user selected DALL-E) instead of overriding to
    # settings.image_gen_model_name.
    if body.image_mode:
        logger.info(f"[ORCH] image_mode=true, routing directly to image_gen")
        return {
            "mode": "image_gen",
            "agent_id": None,
            "deployment_id": None,
            "deployment": None,
            "model_id": body.model_id,
            "intent": "image_generation_explicit" if body.model_id else "image_generation",
        }

    # Mode 2/3: No @agent — run intent classification
    from agentcore.services.mibuddy.intent_classifier import IntentClassifier, Intent

    classifier = IntentClassifier()
    intent = await classifier.classify(body.input_value)
    logger.info(f"[ORCH] Intent classified: {intent.value} for input: {body.input_value[:80]!r}")

    if intent == Intent.KNOWLEDGE_BASE_SEARCH:
        return {
            "mode": "kb_search",
            "agent_id": None,
            "deployment_id": None,
            "deployment": None,
            "model_id": body.model_id,
            "intent": intent.value,
        }

    if intent == Intent.WEB_SEARCH:
        return {
            "mode": "web_search",
            "agent_id": None,
            "deployment_id": None,
            "deployment": None,
            "model_id": body.model_id,
            "intent": intent.value,
        }

    if intent == Intent.IMAGE_GENERATION:
        return {
            "mode": "image_gen",
            "agent_id": None,
            "deployment_id": None,
            "deployment": None,
            "model_id": body.model_id,
            "intent": intent.value,
        }

    if intent == Intent.OUTLOOK_QUERY:
        return {
            "mode": "outlook_query",
            "agent_id": None,
            "deployment_id": None,
            "deployment": None,
            "model_id": body.model_id,
            "intent": intent.value,
        }

    # Intent is general_chat
    # Check if session has documents in Pinecone (follow-up question about uploaded docs)
    try:
        from agentcore.services.mibuddy.document_processor import session_has_documents
        if await session_has_documents(body.session_id):
            logger.info(f"[ORCH] Session has documents in Pinecone — routing to document_qa for follow-up")
            model_id = body.model_id
            if not model_id:
                settings = get_settings_service()
                default_id = settings.settings.default_chat_model_id
                if default_id:
                    model_id = UUID(default_id)
            return {
                "mode": "document_qa",
                "agent_id": None,
                "deployment_id": None,
                "deployment": None,
                "model_id": model_id,
                "intent": "document_followup",
                "doc_files": [],
                "image_files": [],
            }
    except Exception as e:
        logger.debug(f"[ORCH] Document session check failed (non-critical): {e}")

    # Check if user selected the default/smart-router model — auto-pick best model
    if body.model_id:
        settings_svc = get_settings_service().settings
        default_name = (settings_svc.default_orch_model_name or "").strip().lower()
        if default_name and settings_svc.smart_router_enabled:
            # Look up the selected model's display name from registry
            is_default_model = False
            try:
                from agentcore.services.model_service_client import fetch_registry_models_async
                all_models = await fetch_registry_models_async(model_type="llm", active_only=True)
                for m in (all_models or []):
                    if str(m.get("id", "")) == str(body.model_id):
                        if (m.get("display_name", "")).strip().lower() == default_name:
                            is_default_model = True
                        break
            except Exception:
                pass

            if is_default_model:
                from agentcore.services.mibuddy.smart_router import route_to_best_model
                logger.info(f"[ORCH] Default model (smart router): analyzing query to pick best model")

                # Detect attached files (image/document) for routing hints
                has_image = False
                has_document = False
                if body.files:
                    from agentcore.services.mibuddy.document_extractor import IMAGE_EXTENSIONS, SUPPORTED_DOC_EXTENSIONS
                    for f in body.files:
                        ext = Path(f).suffix.lower()
                        if ext in IMAGE_EXTENSIONS:
                            has_image = True
                        elif ext in SUPPORTED_DOC_EXTENSIONS:
                            has_document = True

                # Pull last model used in this session for follow-up routing
                last_model = None
                try:
                    prev_messages = await orch_get_messages(session, session_id=body.session_id)
                    for prev in reversed(prev_messages or []):
                        if getattr(prev, "sender", "") == "agent":
                            last_model = getattr(prev, "sender_name", None)
                            break
                except Exception:
                    pass

                routed = await route_to_best_model(
                    body.input_value,
                    last_model=last_model,
                    has_image=has_image,
                    has_document=has_document,
                )
                if routed:
                    routed_id, routed_name = routed
                    logger.info(f"[ORCH] Smart router selected: {routed_name}")
                    return {
                        "mode": "model_direct",
                        "agent_id": None,
                        "deployment_id": None,
                        "deployment": None,
                        "model_id": UUID(routed_id),
                        "intent": "smart_router",
                        "routed_model_name": routed_name,
                    }
                else:
                    logger.warning("[ORCH] Smart router failed, falling back to direct model chat")

    # Check if user selected a model
    if body.model_id:
        return {
            "mode": "model_direct",
            "agent_id": None,
            "deployment_id": None,
            "deployment": None,
            "model_id": body.model_id,
            "intent": intent.value,
        }

    # Check sticky session
    active = await orch_get_active_agent(session, body.session_id)
    if active:
        agent_id = active["agent_id"]
        deployment_id = active["deployment_id"]
        deployment = await session.get(AgentDeploymentProd, deployment_id)
        if not deployment:
            deployment = await session.get(AgentDeploymentUAT, deployment_id)
        if deployment:
            return {
                "mode": "agent",
                "agent_id": agent_id,
                "deployment_id": deployment_id,
                "deployment": deployment,
                "model_id": None,
                "intent": intent.value,
            }

    # Check default model
    settings = get_settings_service()
    default_model_id = settings.settings.default_chat_model_id
    if default_model_id:
        return {
            "mode": "model_direct",
            "agent_id": None,
            "deployment_id": None,
            "deployment": None,
            "model_id": UUID(default_model_id),
            "intent": intent.value,
        }

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="No agent specified, no model selected, and no default model configured. "
               "Mention an agent with @ or select a model to start.",
    )


@router.post(
    "/chat",
    response_model=OrchChatResponse,
    status_code=200,
    dependencies=[Depends(PermissionChecker(["interact_agents"]))],
)
async def orch_chat(
    *,
    request: Request,
    session: DbSession,
    current_user: CurrentActiveUser,
    body: OrchChatRequest,
):
    """Send a user message to a deployed agent or model and return the reply.

    Routing modes:
    1. @agent mention (agent_id/deployment_id) -> existing agent flow
    2. No @agent -> intent classification -> web_search / image_gen / general_chat
    3. general_chat + model_id -> direct model call
    4. general_chat + sticky session -> existing agent flow
    5. general_chat + default_chat_model_id -> direct model call
    """
    _request_base_url.set(str(request.base_url).rstrip("/"))
    try:
        # -- 1. Route request ------------------------------------------------
        routing = await _route_request(session, current_user, body)
        mode = routing["mode"]
        logger.info(f"[ORCH] Routing mode={mode} intent={routing.get('intent')} session={body.session_id}")

        # -- 2. Handle agent mode (existing flow) ----------------------------
        if mode == "agent":
            agent_id = routing["agent_id"]
            deployment_id = routing["deployment_id"]
            deployment = routing["deployment"]

            did_reset = await _maybe_context_reset(
                session,
                session_id=body.session_id,
                new_agent_id=agent_id,
                new_agent_name=deployment.agent_name,
                user_id=current_user.id,
                new_deployment_id=deployment_id,
            )

            msg_ts = datetime.now(timezone.utc).replace(tzinfo=None)
            user_msg = OrchConversationTable(
                id=uuid4(),
                sender="user",
                sender_name=current_user.username or "User",
                session_id=body.session_id,
                text=body.input_value,
                agent_id=agent_id,
                user_id=current_user.id,
                deployment_id=deployment_id,
                timestamp=msg_ts,
                files=body.files or [],
                properties={},
                category="message",
                content_blocks=[],
            )
            await orch_add_message(user_msg, session)

            logger.info(f"[ORCH] Agent={deployment.agent_name} | session={body.session_id}")
            _env_str = "2" if isinstance(deployment, AgentDeploymentProd) else "1"
            _version_str = f"v{deployment.version_number}"
            agent_text, _was_hitl, agent_content_blocks = await _orch_call_run_api(
                agent_id=str(agent_id),
                env=_env_str,
                version=_version_str,
                input_value=body.input_value,
                session_id=body.session_id,
                files=body.files,
                orch_deployment_id=str(deployment_id) if deployment_id else None,
                orch_session_id=body.session_id,
                orch_org_id=str(deployment.org_id) if deployment.org_id else None,
                orch_dept_id=str(deployment.dept_id) if deployment.dept_id else None,
                orch_user_id=str(current_user.id),
                user_id=str(current_user.id),
            )

            if not agent_text or not agent_text.strip():
                agent_text = "Agent did not produce a response."

            serialized_blocks = _serialize_content_blocks(agent_content_blocks)

            reply_ts = datetime.now(timezone.utc).replace(tzinfo=None)
            agent_msg = OrchConversationTable(
                id=uuid4(),
                sender="agent",
                sender_name=deployment.agent_name,
                session_id=body.session_id,
                text=agent_text,
                agent_id=agent_id,
                user_id=current_user.id,
                deployment_id=deployment_id,
                timestamp=reply_ts,
                files=[],
                properties={},
                category="message",
                content_blocks=serialized_blocks,
            )
            saved_agent_msg = await orch_add_message(agent_msg, session)

            return OrchChatResponse(
                session_id=body.session_id,
                agent_name=deployment.agent_name,
                context_reset=did_reset,
                message=OrchMessageResponse(
                    id=saved_agent_msg.id,
                    timestamp=saved_agent_msg.timestamp.isoformat() if saved_agent_msg.timestamp else "",
                    sender="agent",
                    sender_name=deployment.agent_name,
                    session_id=body.session_id,
                    text=agent_text,
                    agent_id=agent_id,
                    deployment_id=deployment_id,
                    content_blocks=serialized_blocks or None,
                ),
            )

        # -- 3. Handle non-agent modes (model_direct, web_search, image_gen) --
        # Persist user message (no agent_id)
        msg_ts = datetime.now(timezone.utc).replace(tzinfo=None)
        user_msg = OrchConversationTable(
            id=uuid4(),
            sender="user",
            sender_name=current_user.username or "User",
            session_id=body.session_id,
            text=body.input_value,
            user_id=current_user.id,
            model_id=routing.get("model_id"),
            timestamp=msg_ts,
            files=body.files or [],
            properties={},
            category="message",
            content_blocks=[],
        )
        await orch_add_message(user_msg, session)

        response_text = ""
        reasoning_content = None
        sender_name = "Assistant"
        resp_model_id = routing.get("model_id")
        resp_model_name = None

        if mode == "model_direct":
            from agentcore.services.mibuddy.direct_model_chat import direct_model_chat
            result = await direct_model_chat(
                model_id=str(resp_model_id),
                input_value=body.input_value,
                session_id=body.session_id,
                files=body.files,
                enable_reasoning=body.enable_reasoning,
            )
            response_text = result["response_text"]
            reasoning_content = result.get("reasoning_content")
            resp_model_name = result.get("model_name", "")
            sender_name = await _get_model_display_name(session, resp_model_id)

        elif mode == "kb_search":
            from agentcore.services.mibuddy.kb_search_handler import handle_kb_search
            result = await handle_kb_search(body.input_value)
            response_text = result["response_text"]
            resp_model_name = result.get("model_name", "knowledge-base")
            settings = get_settings_service()
            sender_name = settings.settings.company_kb_name or "Knowledge Base"

        elif mode == "web_search":
            from agentcore.services.mibuddy.web_search_handler import handle_web_search
            from agentcore.services.mibuddy.system_prompts import get_system_identity_prompt
            result = await handle_web_search(body.input_value, system_message=get_system_identity_prompt())
            response_text = result["response_text"]
            resp_model_name = result.get("model_name", "gemini")
            # Keep the user's selected model when it can handle web_search itself
            # (prevents the dropdown from flipping to a different provider that
            # may fail on follow-up general-chat turns in the same session).
            user_model_can_web_search = await _model_has_capability(session, resp_model_id, "web_search")
            if user_model_can_web_search or (routing.get("intent") == "web_search_explicit" and resp_model_id):
                sender_name = await _get_model_display_name(session, resp_model_id)
            else:
                settings_svc = get_settings_service().settings
                sender_name = settings_svc.web_search_model_name or "Web Search"
            # Snap dropdown back to the default chat model after this turn
            # (same intent as the image_gen flow): the web_search handler
            # doesn't consult resp_model_id, so overriding it here only
            # affects the `routed_model_id` the frontend uses to sync the
            # dropdown. Keeps follow-up general-chat turns on a provider
            # that actually works for plain chat.
            try:
                _default_chat_id = get_settings_service().settings.default_chat_model_id
                if _default_chat_id:
                    resp_model_id = UUID(str(_default_chat_id))
            except Exception:
                pass

        elif mode == "image_gen":
            from agentcore.services.mibuddy.image_gen_handler import handle_image_generation
            # Pass session_id so the handler can detect edit requests and
            # use the last generated image in the session as a reference.
            result = await handle_image_generation(
                body.input_value,
                model_id=str(resp_model_id) if resp_model_id else None,
                user_id=str(current_user.id),
                session_id=body.session_id,
            )
            response_text = result["response_text"]
            resp_model_name = result.get("model_name", "image-generation")
            # Keep the user's selected model when it can generate images itself.
            user_model_can_image_gen = await _model_has_capability(session, resp_model_id, "image_generation")
            if user_model_can_image_gen or (routing.get("intent") == "image_generation_explicit" and resp_model_id):
                sender_name = await _get_model_display_name(session, resp_model_id)
            else:
                settings_svc = get_settings_service().settings
                sender_name = settings_svc.image_gen_model_name or "Image Generator"

        elif mode == "outlook_query":
            # Port of MiBuddy's outlook agent. We delegate to the verbatim
            # copy at `agentcore.services.mibuddy.outlook_agent`. It needs
            # a LangGraph-style state dict with the current user id and
            # the user's message; it fills `state["final_response"]` with
            # the markdown reply. The agent may also flip
            # `is_canvas_enabled` to True for compose/reply intents —
            # we bubble that back to the frontend as `auto_canvas`.
            from agentcore.services.mibuddy.outlook_agent import outlook_agent_node
            from agentcore.api.outlook_orch import get_outlook_token_from_request
            state = {
                "messages": [{"role": "user", "content": body.input_value}],
                "user_id": str(current_user.id),
                "is_canvas_enabled": bool(body.canvas_enabled),
                # Pass the token from the cookie so any pod can serve the
                # request, not just the pod that handled the OAuth callback.
                "access_token": get_outlook_token_from_request(request),
            }
            state = await outlook_agent_node(state)
            response_text = state.get("final_response", "") or (
                "I couldn't process that Outlook request."
            )
            # auto_canvas = agent turned canvas ON even though the user
            # didn't ask. Store in reasoning_content as a side-channel
            # JSON blob the frontend can pick up. (Re-using an existing
            # field avoids schema churn.)
            if state.get("is_canvas_enabled") and not body.canvas_enabled:
                logger.info("[ORCH] Outlook agent auto-enabled canvas")
            resp_model_name = "outlook"
            # Label the reply with the underlying model that produced it,
            # not the literal "Outlook". Matches the pattern used by
            # model_direct / document_qa / web_search / image_gen so the UI
            # always shows a model name (e.g. "MiBuddy AI", "gpt-5.1").
            sender_name = (
                await _get_model_display_name(session, resp_model_id)
                if resp_model_id else "Outlook"
            )

        elif mode == "document_qa":
            from agentcore.services.mibuddy.document_processor import process_and_ingest, search_documents, build_doc_qa_prompt
            from agentcore.services.mibuddy.direct_model_chat import direct_model_chat

            # Ingest new documents if attached
            doc_files = routing.get("doc_files", [])
            if doc_files:
                count = await process_and_ingest(doc_files, body.session_id)
                logger.info(f"[ORCH] Ingested {count} chunks from {len(doc_files)} files")
                # Wait for Pinecone to index vectors (eventual consistency)
                if count > 0:
                    await asyncio.sleep(5)

            # Authoritative session-wide file list — union of files uploaded
            # across every user message in this session so per-file search
            # sees every document, including ones whose chunks don't match
            # the current query semantically. Merge this-turn's filenames
            # defensively in case the user message hasn't been flushed yet.
            file_names_for_search = await _collect_session_doc_filenames(
                body.session_id, current_user.id,
            )
            for p in doc_files:
                name = Path(p).name
                if name not in file_names_for_search:
                    file_names_for_search.append(name)
            chunks = await search_documents(
                body.input_value,
                body.session_id,
                file_list=file_names_for_search or None,
            )
            logger.info(
                f"[ORCH] Document search returned {len(chunks)} chunks "
                f"(session has {len(file_names_for_search)} file(s): {file_names_for_search})"
            )

            # Build enriched prompt and call model
            enriched_prompt = build_doc_qa_prompt(body.input_value, chunks)
            if not resp_model_id:
                raise HTTPException(status_code=400, detail="No model selected for document Q&A.")
            image_files = routing.get("image_files", [])
            result = await direct_model_chat(
                model_id=str(resp_model_id),
                input_value=enriched_prompt,
                session_id=body.session_id,
                files=image_files,
            )
            response_text = result["response_text"]
            reasoning_content = result.get("reasoning_content")
            resp_model_name = result.get("model_name", "")
            sender_name = await _get_model_display_name(session, resp_model_id) if resp_model_id else "Document Q&A"

        if not response_text or not response_text.strip():
            response_text = "No response was generated. Please try again."

        # Persist response — sender="model" here because this path handles
        # direct model chat (model_direct, web_search, image_gen, doc_qa, kb_search).
        # Agent-deployment responses use sender="agent" (earlier branch).
        reply_ts = datetime.now(timezone.utc).replace(tzinfo=None)
        agent_msg = OrchConversationTable(
            id=uuid4(),
            sender="model",
            sender_name=sender_name,
            session_id=body.session_id,
            text=response_text,
            user_id=current_user.id,
            model_id=resp_model_id,
            reasoning_content=reasoning_content,
            timestamp=reply_ts,
            files=[],
            properties={},
            category="message",
            content_blocks=[],
        )
        saved_msg = await orch_add_message(agent_msg, session)

        return OrchChatResponse(
            session_id=body.session_id,
            agent_name=sender_name,
            context_reset=False,
            message=OrchMessageResponse(
                id=saved_msg.id,
                timestamp=saved_msg.timestamp.isoformat() if saved_msg.timestamp else "",
                sender="model",
                sender_name=sender_name,
                session_id=body.session_id,
                text=response_text,
                model_id=resp_model_id,
                model_name=resp_model_name,
                reasoning_content=reasoning_content,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in orchestrator chat: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/chat/stream",
    status_code=200,
    dependencies=[Depends(PermissionChecker(["interact_agents"]))],
)
async def orch_chat_stream(
    *,
    request: Request,
    session: DbSession,
    current_user: CurrentActiveUser,
    body: OrchChatRequest,
):
    """Stream response token-by-token as NDJSON events.

    Supports both agent-based and model-direct/web-search/image-gen streaming.

    Event types emitted:
      - ``add_message``  – first chunk (UI creates the message bubble)
      - ``token``        – subsequent chunks ``{chunk, id}``
      - ``reasoning``    – CoT reasoning chunks (for models that support it)
      - ``end``          – signals stream is done, carries final ``{agent_text, message_id}``
    """
    _request_base_url.set(str(request.base_url).rstrip("/"))

    # -- 1. Route request ------------------------------------------------
    routing = await _route_request(session, current_user, body)
    mode = routing["mode"]
    logger.info(f"[ORCH-STREAM] Routing mode={mode} intent={routing.get('intent')} session={body.session_id} enable_reasoning={body.enable_reasoning} image_mode={body.image_mode}")

    # -- 2. For non-agent modes, use direct streaming --------------------
    if mode in ("model_direct", "web_search", "image_gen", "document_qa", "kb_search", "outlook_query"):
        # Persist user message
        stream_msg_ts = datetime.now(timezone.utc).replace(tzinfo=None)
        user_msg = OrchConversationTable(
            id=uuid4(),
            sender="user",
            sender_name=current_user.username or "User",
            session_id=body.session_id,
            text=body.input_value,
            user_id=current_user.id,
            model_id=routing.get("model_id"),
            timestamp=stream_msg_ts,
            files=body.files or [],
            properties={},
            category="message",
            content_blocks=[],
        )
        await orch_add_message(user_msg, session)

        resp_model_id = routing.get("model_id")
        intent = routing.get("intent")
        sender_name = "Assistant"
        if mode in ("model_direct", "document_qa") and resp_model_id:
            sender_name = await _get_model_display_name(session, resp_model_id)
        elif mode == "kb_search":
            settings_svc = get_settings_service()
            sender_name = settings_svc.settings.company_kb_name or "Knowledge Base"
        elif mode == "web_search":
            # If the user already picked a web-search-capable model (e.g. Gemini
            # 3 Pro with web_search=true), keep it — don't flip the dropdown to
            # the WEB_SEARCH_MODEL_NAME specialist. That flip otherwise strands
            # follow-up general-chat turns on a different provider that may
            # fail (e.g. google / Generative Language API).
            user_model_can_web_search = await _model_has_capability(session, resp_model_id, "web_search")
            if user_model_can_web_search or (intent == "web_search_explicit" and resp_model_id):
                sender_name = await _get_model_display_name(session, resp_model_id)
            else:
                settings_svc = get_settings_service().settings
                sender_name = settings_svc.web_search_model_name or "Web Search"
            # Mirror the image_gen auto-switch-back: the web_search handler
            # doesn't use resp_model_id at all (WEB_SEARCH_MODEL_NAME is
            # resolved internally), so point the dropdown at the default
            # chat model instead. Prevents follow-up general-chat turns
            # from being stranded on a web-search-only registry row whose
            # provider may fail on plain chat (e.g. the google / Generative
            # Language API 403 on Motherson's GCP project).
            try:
                _default_chat_id = get_settings_service().settings.default_chat_model_id
                if _default_chat_id:
                    resp_model_id = UUID(str(_default_chat_id))
            except Exception:
                pass
        elif mode == "image_gen":
            # Same pattern as web_search: honour the user's choice when the
            # selected model can generate images natively; only swap to the
            # IMAGE_GEN_MODEL_NAME specialist when it can't.
            user_model_can_image_gen = await _model_has_capability(session, resp_model_id, "image_generation")
            if user_model_can_image_gen or (intent == "image_generation_explicit" and resp_model_id):
                sender_name = await _get_model_display_name(session, resp_model_id)
            else:
                settings_svc = get_settings_service().settings
                sender_name = settings_svc.image_gen_model_name or "Image Generator"
                # Override resp_model_id to the IMAGE_GEN_MODEL_NAME registry entry
                try:
                    from agentcore.services.model_service_client import fetch_registry_models_async
                    ig_name = (settings_svc.image_gen_model_name or "").strip().lower()
                    if ig_name:
                        _all = await fetch_registry_models_async(active_only=True) or []
                        _match = next((m for m in _all if (m.get("display_name") or "").strip().lower() == ig_name), None)
                        if _match and _match.get("id"):
                            resp_model_id = UUID(str(_match["id"]))
                except Exception:
                    pass
        elif mode == "outlook_query":
            # Same treatment as model_direct / document_qa — reply appears
            # under the underlying model's display name rather than the
            # literal "Outlook" so the UI shows what actually answered.
            if resp_model_id:
                sender_name = await _get_model_display_name(session, resp_model_id)
            else:
                sender_name = "Outlook"

        queue: asyncio.Queue = asyncio.Queue()
        event_manager = create_default_event_manager(queue)

        # Fire the "routing" event IMMEDIATELY so the frontend can update the
        # model dropdown before the (potentially slow) response begins. This
        # gives instant visual feedback — e.g. user picks "MiBuddy AI", asks
        # "latest news", dropdown flips to "Web Search" right away instead of
        # waiting for the response to complete.
        try:
            event_manager.on_routing(data={
                "routed_model_id": str(resp_model_id) if resp_model_id else None,
                "routed_model_name": sender_name,
                "mode": mode,
            })
        except Exception as _rerr:
            logger.debug(f"[ORCH-STREAM] routing event emit failed (non-critical): {_rerr}")

        _input_value = body.input_value
        _session_id = body.session_id
        _user_id = current_user.id
        _enable_reasoning = body.enable_reasoning
        _sender_name = sender_name
        _resp_model_id = resp_model_id
        _mode = mode
        _files = body.files
        _doc_files = routing.get("doc_files", [])
        _image_files = routing.get("image_files", [])
        _canvas_enabled = bool(body.canvas_enabled)

        async def _run_direct_and_persist():
            try:
                result = {}
                if _mode == "kb_search":
                    from agentcore.services.mibuddy.kb_search_handler import handle_kb_search_stream
                    result = await handle_kb_search_stream(
                        _input_value,
                        event_manager=event_manager,
                    )
                elif _mode == "document_qa":
                    from agentcore.services.mibuddy.document_processor import process_and_ingest, search_documents, build_doc_qa_prompt
                    from agentcore.services.mibuddy.direct_model_chat import direct_model_chat_stream

                    # Ingest new documents — show progress to user
                    if _doc_files:
                        event_manager.on_token(data={"chunk": "📄 Processing document... "})
                        count = await process_and_ingest(_doc_files, _session_id)
                        logger.info(f"[ORCH-STREAM] Ingested {count} chunks from {len(_doc_files)} files")
                        if count > 0:
                            event_manager.on_token(data={"chunk": f"✅ Indexed {count} chunks. "})
                            event_manager.on_token(data={"chunk": "🔍 Searching... "})
                            await asyncio.sleep(5)
                        else:
                            event_manager.on_token(data={"chunk": "⚠️ No content extracted. "})
                    else:
                        event_manager.on_token(data={"chunk": "🔍 Searching documents... "})

                    # Authoritative session-wide file list (see non-stream
                    # branch for rationale). Merge this-turn's filenames
                    # defensively in case the user message hasn't flushed.
                    _file_names_for_search = await _collect_session_doc_filenames(
                        _session_id, _user_id,
                    )
                    for p in _doc_files:
                        name = Path(p).name
                        if name not in _file_names_for_search:
                            _file_names_for_search.append(name)
                    chunks = await search_documents(
                        _input_value,
                        _session_id,
                        file_list=_file_names_for_search or None,
                    )
                    logger.info(
                        f"[ORCH-STREAM] Document search returned {len(chunks)} chunks "
                        f"(session has {len(_file_names_for_search)} file(s): {_file_names_for_search})"
                    )

                    if chunks:
                        event_manager.on_token(data={"chunk": f"Found {len(chunks)} relevant sections.\n\n"})
                    else:
                        event_manager.on_token(data={"chunk": "No relevant sections found.\n\n"})

                    enriched_prompt = build_doc_qa_prompt(_input_value, chunks)

                    result = await direct_model_chat_stream(
                        model_id=str(_resp_model_id),
                        input_value=enriched_prompt,
                        session_id=_session_id,
                        files=_image_files,
                        event_manager=event_manager,
                    )
                elif _mode == "model_direct":
                    from agentcore.services.mibuddy.direct_model_chat import direct_model_chat_stream
                    # MiBuddy canvas parity: when canvas is on we prepend a
                    # short system-style instruction so the LLM produces a
                    # cleanly formatted draft document (the rendering side
                    # is handled by the frontend canvas panel).
                    prompt_for_model = _input_value
                    if _canvas_enabled:
                        prompt_for_model = (
                            "You are helping the user draft a clear, "
                            "well-structured document. Respond with the draft "
                            "in clean markdown (headings, short paragraphs, "
                            "bullet lists where useful). Do not add "
                            "meta-commentary or follow-up questions.\n\n"
                            f"User request: {_input_value}"
                        )
                    result = await direct_model_chat_stream(
                        model_id=str(_resp_model_id),
                        input_value=prompt_for_model,
                        session_id=_session_id,
                        files=_files,
                        enable_reasoning=_enable_reasoning,
                        event_manager=event_manager,
                    )
                elif _mode == "web_search":
                    from agentcore.services.mibuddy.web_search_handler import handle_web_search_stream
                    from agentcore.services.mibuddy.system_prompts import get_system_identity_prompt
                    logger.warning(f"[ORCH-STREAM] >>> ABOUT TO CALL handle_web_search_stream with enable_reasoning={_enable_reasoning}")
                    result = await handle_web_search_stream(
                        _input_value,
                        system_message=get_system_identity_prompt(),
                        event_manager=event_manager,
                        enable_reasoning=_enable_reasoning,
                    )
                    _rc = result.get("reasoning_content") or ""
                    _rt = result.get("response_text") or ""
                    logger.warning(f"[ORCH-STREAM] <<< handle_web_search_stream RETURNED, reasoning_len={len(_rc)}, response_len={len(_rt)}")
                    logger.warning(f"[ORCH-STREAM] REASONING preview: {_rc[:300]!r}")
                    logger.warning(f"[ORCH-STREAM] RESPONSE preview: {_rt[:300]!r}")
                elif _mode == "image_gen":
                    from agentcore.services.mibuddy.image_gen_handler import handle_image_generation_stream
                    # session_id → enables the image-edit flow when the user
                    # asks to modify a previously-generated image.
                    result = await handle_image_generation_stream(
                        _input_value,
                        model_id=str(_resp_model_id) if _resp_model_id else None,
                        user_id=str(_user_id),
                        event_manager=event_manager,
                        session_id=_session_id,
                    )
                elif _mode == "outlook_query":
                    # Port of MiBuddy's outlook_agent_node. Runs to completion
                    # (not token-stream) and emits the full markdown as one
                    # chunk + an end event, matching other non-streaming
                    # modes like kb_search.
                    from agentcore.services.mibuddy.outlook_agent import outlook_agent_node
                    from agentcore.api.outlook_orch import get_outlook_token_from_request
                    state = {
                        "messages": [{"role": "user", "content": _input_value}],
                        "user_id": str(_user_id),
                        "is_canvas_enabled": _canvas_enabled,
                        "access_token": get_outlook_token_from_request(request),
                    }
                    state = await outlook_agent_node(state)
                    outlook_text = state.get("final_response", "") or (
                        "I couldn't process that Outlook request."
                    )
                    event_manager.on_token(data={"chunk": outlook_text})
                    # Did the outlook agent flip canvas ON (reply/compose)?
                    agent_canvas = bool(state.get("is_canvas_enabled"))
                    result = {
                        "response_text": outlook_text,
                        "model_name": "outlook",
                        "auto_canvas": agent_canvas and not _canvas_enabled,
                    }

                response_text = result.get("response_text", "")
                reasoning_content = result.get("reasoning_content")

                if not response_text.strip():
                    response_text = "No response was generated."

                from agentcore.services.deps import session_scope
                reply_ts = datetime.now(timezone.utc).replace(tzinfo=None)
                async with session_scope() as db:
                    # sender="model" — streaming path for model-direct chat.
                    agent_msg = OrchConversationTable(
                        id=uuid4(),
                        sender="model",
                        sender_name=_sender_name,
                        session_id=_session_id,
                        text=response_text,
                        user_id=_user_id,
                        model_id=_resp_model_id,
                        reasoning_content=reasoning_content,
                        timestamp=reply_ts,
                        files=[],
                        # Persist canvas flag so a reload / refetch keeps
                        # the message rendered in the canvas editor.
                        properties={
                            "canvas_enabled": bool(
                                _canvas_enabled or result.get("auto_canvas"),
                            ),
                        } if (_canvas_enabled or result.get("auto_canvas")) else {},
                        category="message",
                        content_blocks=[],
                    )
                    await orch_add_message(agent_msg, db)

                logger.warning(f"[ORCH-STREAM] SENDING end event with reasoning_content len={len(reasoning_content or '')}, agent_text len={len(response_text or '')}")
                event_manager.on_end(data={
                    "agent_text": response_text,
                    "message_id": str(agent_msg.id),
                    "reasoning_content": reasoning_content,
                    # MiBuddy parity: tell the frontend when the backend
                    # auto-enabled canvas (compose/reply email).
                    "auto_canvas": bool(result.get("auto_canvas", False)),
                    # Tell the frontend which model actually produced the response.
                    # Used by the UI to sync the model dropdown when smart router
                    # / intent classifier routed to a different model than the user selected.
                    "routed_model_id": str(_resp_model_id) if _resp_model_id else None,
                    "routed_model_name": _sender_name,
                })
            except Exception as exc:
                logger.exception(f"[ORCH-STREAM] Direct mode error: {exc}")
                event_manager.on_error(data={"text": str(exc)})
                event_manager.on_end(data={})
            finally:
                queue.put_nowait((None, None, None))

        asyncio.create_task(_run_direct_and_persist())

        async def _direct_event_generator():
            while True:
                item = await queue.get()
                if item == (None, None, None):
                    break
                _event_id, raw_data, _timestamp = item
                # raw_data is already JSON-encoded bytes from event_manager
                if isinstance(raw_data, bytes):
                    yield raw_data
                else:
                    yield str(raw_data).encode("utf-8")

        return StreamingResponse(
            _direct_event_generator(),
            media_type="application/x-ndjson",
        )

    # -- 3. Agent mode streaming (existing flow) -------------------------
    agent_id = routing["agent_id"]
    deployment_id = routing["deployment_id"]
    deployment = routing["deployment"]

    await _maybe_context_reset(
        session,
        session_id=body.session_id,
        new_agent_id=agent_id,
        new_agent_name=deployment.agent_name,
        user_id=current_user.id,
        new_deployment_id=deployment_id,
    )

    stream_msg_ts = datetime.now(timezone.utc).replace(tzinfo=None)
    user_msg = OrchConversationTable(
        id=uuid4(),
        sender="user",
        sender_name=current_user.username or "User",
        session_id=body.session_id,
        text=body.input_value,
        agent_id=agent_id,
        user_id=current_user.id,
        deployment_id=deployment_id,
        timestamp=stream_msg_ts,
        files=body.files or [],
        properties={},
        category="message",
        content_blocks=[],
    )
    await orch_add_message(user_msg, session)

    queue: asyncio.Queue = asyncio.Queue()
    event_manager = create_default_event_manager(queue)

    agent_id_str = str(agent_id)
    agent_name = deployment.agent_name
    input_value = body.input_value
    chat_session_id = body.session_id
    user_id_str = str(current_user.id)
    dep_agent_id = agent_id
    dep_deployment_id = deployment_id
    dep_user_id = current_user.id
    dep_org_id = str(deployment.org_id) if deployment.org_id else None
    dep_dept_id = str(deployment.dept_id) if deployment.dept_id else None
    dep_is_prod = isinstance(deployment, AgentDeploymentProd)
    dep_files = body.files

    async def _run_and_persist():
        """Background coroutine: run the agent via /run API, persist reply, close the queue."""
        try:
            _env_str = "2" if dep_is_prod else "1"
            _version_str = f"v{deployment.version_number}"
            agent_text, was_interrupted, agent_content_blocks = await _orch_call_run_api(
                agent_id=agent_id_str,
                env=_env_str,
                version=_version_str,
                input_value=input_value,
                session_id=chat_session_id,
                files=dep_files,
                stream=True,
                event_manager=event_manager,
                orch_deployment_id=str(dep_deployment_id) if dep_deployment_id else None,
                orch_session_id=chat_session_id,
                orch_org_id=dep_org_id,
                orch_dept_id=dep_dept_id,
                orch_user_id=user_id_str,
                user_id=user_id_str,
            )

            # When interrupted (HITL pause), _emit_hitl_pause_event already
            # emitted the add_message with HITL metadata to the frontend.
            # Do NOT persist an agent reply or emit end with agent_text — that
            # would overwrite the HITL action buttons on the frontend.
            if was_interrupted:
                logger.info(f"[ORCH-STREAM] Run interrupted (HITL) — persisting pause message")
                # Persist the HITL pause message so it survives page navigation.
                # The SSE `add_message` event (from _emit_hitl_pause_event) only
                # lives in the active stream; when the user navigates away and
                # comes back, messages are reloaded from DB.  This row ensures
                # the HITL action buttons reappear for still-pending requests.
                from agentcore.services.deps import session_scope

                try:
                    # Fetch the pending HITL request to get interrupt_data
                    from agentcore.services.database.models.hitl_request.model import HITLRequest, HITLStatus
                    from sqlmodel import col, select as _sel

                    async with session_scope() as db:
                        stmt = (
                            _sel(HITLRequest)
                            .where(HITLRequest.session_id == chat_session_id)
                            .where(HITLRequest.status == HITLStatus.PENDING)
                            .order_by(col(HITLRequest.requested_at).desc())
                            .limit(1)
                        )
                        hitl_row = (await db.exec(stmt)).first()

                    actions = []
                    question = "Awaiting human review"
                    if hitl_row and hitl_row.interrupt_data:
                        idata = hitl_row.interrupt_data
                        actions = idata.get("actions", [])
                        question = idata.get("question", question)

                    actions_display = "\n".join(f"• {a}" for a in actions) if actions else "—"
                    hitl_text = (
                        f"⏸ **Waiting for human review**\n\n"
                        f"{question}\n\n"
                        f"**Available actions:**\n{actions_display}"
                    )

                    hitl_ts = datetime.now(timezone.utc).replace(tzinfo=None)
                    async with session_scope() as db:
                        hitl_msg = OrchConversationTable(
                            id=uuid4(),
                            sender="agent",
                            sender_name=agent_name,
                            session_id=chat_session_id,
                            text=hitl_text,
                            agent_id=dep_agent_id,
                            user_id=dep_user_id,
                            deployment_id=dep_deployment_id,
                            timestamp=hitl_ts,
                            files=[],
                            properties={
                                "hitl": True,
                                "thread_id": chat_session_id,
                                "actions": actions,
                                "is_deployed_run": True,
                            },
                            category="message",
                            content_blocks=[],
                        )
                        await orch_add_message(hitl_msg, db)
                except Exception as _err:
                    logger.warning(f"[ORCH-STREAM] Could not persist HITL message: {_err}")

                event_manager.on_end(data={})
                return

            if not agent_text or not agent_text.strip():
                agent_text = "Agent did not produce a response."

            # Serialize content_blocks for storage
            serialized_blocks = _serialize_content_blocks(agent_content_blocks)

            # Persist agent reply in orch tables (uses its own DB session)
            from agentcore.services.deps import session_scope

            stream_reply_ts = datetime.now(timezone.utc).replace(tzinfo=None)
            async with session_scope() as db:
                agent_msg = OrchConversationTable(
                    id=uuid4(),
                    sender="agent",
                    sender_name=agent_name,
                    session_id=chat_session_id,
                    text=agent_text,
                    agent_id=dep_agent_id,
                    user_id=dep_user_id,
                    deployment_id=dep_deployment_id,
                    timestamp=stream_reply_ts,
                    files=[],
                    properties={},
                    category="message",
                    content_blocks=serialized_blocks,
                )
                await orch_add_message(agent_msg, db)

            # Signal end with final data so the frontend can finalize
            event_manager.on_end(data={
                "agent_text": agent_text,
                "message_id": str(agent_msg.id),
                "content_blocks": serialized_blocks,
            })
        except Exception as exc:
            logger.exception(f"[ORCH-STREAM] Error: {exc}")
            # Surface the agent failure to the user the same way Playground
            # does: persist a visible agent message carrying the error text
            # and close the stream via `end` with `agent_text` populated.
            # Relying on the `error` SSE event alone leaves the placeholder
            # bubble empty — the frontend then falls back to rendering
            # EMPTY_OUTPUT_SEND_MESSAGE ("Message empty.").
            error_text = str(exc) or "Agent run failed."
            from agentcore.services.deps import session_scope

            err_msg_id = uuid4()
            try:
                err_ts = datetime.now(timezone.utc).replace(tzinfo=None)
                async with session_scope() as db:
                    err_agent_msg = OrchConversationTable(
                        id=err_msg_id,
                        sender="agent",
                        sender_name=agent_name,
                        session_id=chat_session_id,
                        text=error_text,
                        agent_id=dep_agent_id,
                        user_id=dep_user_id,
                        deployment_id=dep_deployment_id,
                        timestamp=err_ts,
                        files=[],
                        properties={"error": True},
                        category="error",
                        content_blocks=[],
                    )
                    await orch_add_message(err_agent_msg, db)
            except Exception as _persist_err:
                logger.warning(f"[ORCH-STREAM] Could not persist error message: {_persist_err}")

            event_manager.on_error(data={"text": error_text})
            event_manager.on_end(data={
                "agent_text": error_text,
                "message_id": str(err_msg_id),
                "content_blocks": [],
                "error": True,
            })
        finally:
            # Sentinel to stop the consumer
            queue.put_nowait((None, None, None))

    # -- 4. Start background task and return streaming response ----------
    from agentcore.services.deps import get_rabbitmq_service

    rabbitmq_service = get_rabbitmq_service()
    if rabbitmq_service.is_enabled():
        event_store = _get_orchestrator_event_store()
        if event_store is None:
            logger.warning("[ORCH-STREAM] Redis event store unavailable; falling back to in-process execution")
        else:
            job_id = str(uuid4())
            try:
                await event_store.init_job(job_id)
                job_data = {
                    "job_id": job_id,
                    "agent_id": agent_id_str,
                    "agent_name": agent_name,
                    "input_value": input_value,
                    "session_id": chat_session_id,
                    "user_id": user_id_str,
                    "files": dep_files,
                    "deployment_id": str(dep_deployment_id) if dep_deployment_id else None,
                    "env": "2" if dep_is_prod else "1",
                    "version": f"v{deployment.version_number}",
                    "orch_deployment_id": str(dep_deployment_id) if dep_deployment_id else None,
                    "orch_session_id": chat_session_id,
                    "orch_org_id": dep_org_id,
                    "orch_dept_id": dep_dept_id,
                }
                await rabbitmq_service.publish_orchestrator_job(job_data)
                logger.info(f"[ORCH-STREAM] Published RabbitMQ job {job_id}; streaming from Redis backplane")
                return await _create_orchestrator_redis_response(job_id=job_id, event_store=event_store)
            except Exception as exc:
                logger.exception(f"[ORCH-STREAM] Failed to start RabbitMQ+Redis stream job: {exc}")
                try:
                    await event_store.mark_status(job_id, status="failed", error=str(exc))
                except Exception:
                    pass
                raise HTTPException(status_code=500, detail="Failed to start orchestrator stream") from exc

    # --- Direct fallback path ---
    run_task = asyncio.create_task(_run_and_persist())

    async def _consume():
        while True:
            try:
                _event_id, value, _ = await queue.get()
                if value is None:
                    break
                yield value
            except Exception:
                break

    async def _on_disconnect():
        run_task.cancel()

    return StreamingResponse(
        _consume(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
        background=_on_disconnect,
    )



@router.get("/sessions", response_model=list[OrchSessionSummary], status_code=200)
async def list_orch_sessions(
    *,
    session: DbSession,
    current_user: CurrentActiveUser,
):
    """Return all orchestrator chat sessions owned by the current user.

    Each session includes the currently active agent (from the most recent message).
    """
    try:
        rows = await orch_get_sessions(session, user_id=current_user.id)
        summaries = []
        for row in rows:
            summary = OrchSessionSummary(**row)
            active = await orch_get_active_agent(session, row["session_id"])
            if active:
                summary.active_agent_id = active["agent_id"]
                summary.active_deployment_id = active["deployment_id"]
                dep = await session.get(AgentDeploymentProd, active["deployment_id"])
                if not dep:
                    dep = await session.get(AgentDeploymentUAT, active["deployment_id"])
                if dep:
                    summary.active_agent_name = dep.agent_name
            summaries.append(summary)
        return summaries
    except Exception as e:
        logger.error(f"Error listing orch sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e




@router.get("/sessions/{session_id}/messages", response_model=list[OrchMessageResponse], status_code=200)
async def get_orch_session_messages(
    *,
    session: DbSession,
    current_user: CurrentActiveUser,
    session_id: str,
):
    """Return all messages in an orchestrator session ordered by timestamp."""
    try:
        messages = await orch_get_messages(
            session,
            session_id=session_id,
            user_id=current_user.id,
        )
        return [
            OrchMessageResponse(
                id=m.id,
                timestamp=m.timestamp.isoformat() if m.timestamp else "",
                sender=m.sender,
                sender_name=m.sender_name,
                session_id=m.session_id,
                text=m.text,
                agent_id=m.agent_id,
                deployment_id=m.deployment_id,
                model_id=getattr(m, "model_id", None),
                reasoning_content=getattr(m, "reasoning_content", None),
                category=m.category or "message",
                files=m.files if m.files else None,
                properties=m.properties if isinstance(m.properties, dict) else None,
                content_blocks=m.content_blocks if m.content_blocks else None,
                feedback_rating=getattr(m, "feedback_rating", None),
                feedback_reasons=getattr(m, "feedback_reasons", None),
                feedback_comment=getattr(m, "feedback_comment", None),
                feedback_at=(
                    getattr(m, "feedback_at").isoformat()
                    if getattr(m, "feedback_at", None)
                    else None
                ),
            )
            for m in messages
            # Safety net: skip messages with empty text that were persisted by
            # intermediate graph nodes before the orch_skip_node_persist fix.
            if (m.text and m.text.strip()) or m.category == "context_reset"
        ]
    except Exception as e:
        logger.error(f"Error getting orch session messages: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put(
    "/messages/{message_id}/edit",
    response_model=EditMessageResponse,
    status_code=200,
    dependencies=[Depends(PermissionChecker(["interact_agents"]))],
)
async def edit_orch_message(
    *,
    request: Request,
    session: DbSession,
    current_user: CurrentActiveUser,
    message_id: str,
    body: EditMessageRequest,
):
    """Edit a past user message in-place (MiBuddy-style UPSERT).

    - Updates the user message's text to `edited_text`.
    - Finds the NEXT agent response in the same session (by timestamp).
    - Regenerates a fresh response via the same model the original used.
    - Updates the agent response's text in-place (keeps same message ID).
    - Returns both updated rows.

    Messages AFTER the edited pair are LEFT UNTOUCHED in the DB (MiBuddy behavior).
    The frontend is responsible for hiding them in the UI.
    """
    try:
        # 1. Fetch the user message being edited
        user_msg = await session.get(OrchConversationTable, UUID(message_id))
        if user_msg is None:
            raise HTTPException(status_code=404, detail="Message not found")
        if user_msg.sender != "user":
            raise HTTPException(status_code=400, detail="Only user messages can be edited")
        if user_msg.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Cannot edit another user's message")

        # 2. Find the next agent message in the same session (the response to this user msg)
        next_agent_stmt = (
            select(OrchConversationTable)
            .where(
                OrchConversationTable.session_id == user_msg.session_id,
                # Assistant messages are "agent" (agent deployments) or
                # "model" (direct model chat) — match either.
                OrchConversationTable.sender.in_(("agent", "model")),
                OrchConversationTable.timestamp > user_msg.timestamp,
            )
            .order_by(OrchConversationTable.timestamp.asc())
        )
        next_agent_rows = (await session.exec(next_agent_stmt)).all()
        agent_msg = next_agent_rows[0] if next_agent_rows else None

        # 3. Update the user message text first (in-place).
        # Note: OrchConversationTable doesn't have an `updated_at` column — the
        # original timestamp is preserved on edits (matches MiBuddy behavior).
        user_msg.text = body.edited_text
        session.add(user_msg)
        await session.flush()

        # 4. Re-run intent classification on the edited prompt so the regenerated
        # response goes through the correct mode handler (doc_qa, web_search,
        # outlook_query, image_gen, kb_search, model_direct) — not always plain
        # model chat. Previously the edit bypassed routing entirely and every
        # edit regenerated as model_direct, losing doc/web/outlook/image features.
        original_model_id = getattr(agent_msg, "model_id", None) if agent_msg else None
        if not original_model_id:
            original_model_id = getattr(user_msg, "model_id", None)

        # default_chat_id acts as the "neutral" seed for routing. We do NOT pass
        # original_model_id into the synthetic body because _route_request's
        # Priority 0.5 force-routes to image_gen / web_search based on the
        # model's capabilities — which would lock the edit into the original
        # mode. E.g. an image-gen message edited to "what are today top news"
        # would still hit Nano Banana instead of reaching the intent
        # classifier. Using a plain chat model as the seed makes the router
        # classify the edited TEXT and pick the right mode.
        settings_fallback = get_settings_service().settings
        default_chat_id: UUID | None = None
        if settings_fallback.default_chat_model_id:
            try:
                default_chat_id = UUID(settings_fallback.default_chat_model_id)
            except (ValueError, TypeError):
                default_chat_id = None

        if not original_model_id and not default_chat_id:
            raise HTTPException(
                status_code=400,
                detail="Cannot regenerate: no model_id associated with this conversation.",
            )

        # Synthesise a request that mirrors what would have arrived if the user
        # had typed this as a new message. Keep original files so doc_qa picks
        # them up again; carry the request's reasoning flag.
        #
        # Seed precedence for model_id: caller's current UI selection
        # (body.model_id) → default chat model → original response's model.
        # This matches the send flow — user's dropdown choice is the starting
        # point, and dispatch-side settings override may still swap to a
        # specialist model (Nano Banana / web search) based on intent.
        #
        # image_mode is deliberately FORCED False. Edit text may imply a
        # different intent than the original (e.g. edit "generate image" →
        # "what's the news"); respecting the frontend's current image_mode
        # would short-circuit intent classification via the fast-path,
        # defeating the re-routing we want on edits.
        synthetic_body = OrchChatRequest(
            session_id=user_msg.session_id,
            model_id=(body.model_id or default_chat_id or original_model_id),
            input_value=body.edited_text,
            files=user_msg.files if user_msg.files else None,
            enable_reasoning=body.enable_reasoning,
            image_mode=False,
        )
        routing = await _route_request(session, current_user, synthetic_body)
        mode = routing["mode"]
        # Fallback chain for the model that will actually answer:
        #   routing's explicit model_id → default chat → caller's original.
        _resp_model_candidate = routing.get("model_id") or default_chat_id or original_model_id
        if not _resp_model_candidate:
            # Guarded earlier (we raise if both original and default are
            # missing), so this is defensive. Re-raise to keep the type
            # checker happy about resp_model_id being non-None below.
            raise HTTPException(
                status_code=400,
                detail="Cannot regenerate: no model_id resolvable for this edit.",
            )
        resp_model_id: UUID = _resp_model_candidate if isinstance(_resp_model_candidate, UUID) else UUID(str(_resp_model_candidate))
        intent = routing.get("intent")

        new_response_text = ""
        new_reasoning: str | None = None
        new_model_name = ""
        regen_sender_name = "Assistant"

        if mode == "model_direct":
            from agentcore.services.mibuddy.direct_model_chat import direct_model_chat
            result = await direct_model_chat(
                model_id=str(resp_model_id),
                input_value=body.edited_text,
                session_id=user_msg.session_id,
                files=synthetic_body.files,
                enable_reasoning=body.enable_reasoning,
            )
            new_response_text = result.get("response_text", "")
            new_reasoning = result.get("reasoning_content")
            new_model_name = result.get("model_name", "")
            regen_sender_name = await _get_model_display_name(session, resp_model_id)

        elif mode == "kb_search":
            from agentcore.services.mibuddy.kb_search_handler import handle_kb_search
            result = await handle_kb_search(body.edited_text)
            new_response_text = result.get("response_text", "")
            new_model_name = result.get("model_name", "knowledge-base")
            settings_svc = get_settings_service().settings
            regen_sender_name = settings_svc.company_kb_name or "Knowledge Base"

        elif mode == "web_search":
            from agentcore.services.mibuddy.web_search_handler import handle_web_search
            from agentcore.services.mibuddy.system_prompts import get_system_identity_prompt
            # Non-explicit web_search: swap resp_model_id to the configured
            # WEB_SEARCH_MODEL_NAME registry entry so agent_msg.model_id
            # reflects the actual model that answered (mirrors the main
            # streaming flow — needed because the synthetic body uses
            # default_chat_id as a seed, not the web-search model).
            settings_svc = get_settings_service().settings
            if intent == "web_search_explicit":
                regen_sender_name = await _get_model_display_name(session, resp_model_id)
            else:
                regen_sender_name = settings_svc.web_search_model_name or "Web Search"
                try:
                    from agentcore.services.model_service_client import fetch_registry_models_async
                    ws_name = (settings_svc.web_search_model_name or "").strip().lower()
                    if ws_name:
                        _all = await fetch_registry_models_async(active_only=True) or []
                        _match = next(
                            (m for m in _all if (m.get("display_name") or "").strip().lower() == ws_name),
                            None,
                        )
                        if _match and _match.get("id"):
                            resp_model_id = UUID(str(_match["id"]))
                except Exception:  # noqa: BLE001
                    pass
            result = await handle_web_search(body.edited_text, system_message=get_system_identity_prompt())
            new_response_text = result.get("response_text", "")
            new_model_name = result.get("model_name", "gemini")

        elif mode == "image_gen":
            from agentcore.services.mibuddy.image_gen_handler import handle_image_generation
            # Non-explicit image_gen: swap resp_model_id to the configured
            # IMAGE_GEN_MODEL_NAME registry entry — otherwise we'd call
            # handle_image_generation with default_chat_id (a text model).
            settings_svc = get_settings_service().settings
            if intent == "image_generation_explicit":
                regen_sender_name = await _get_model_display_name(session, resp_model_id)
            else:
                regen_sender_name = settings_svc.image_gen_model_name or "Image Generator"
                try:
                    from agentcore.services.model_service_client import fetch_registry_models_async
                    ig_name = (settings_svc.image_gen_model_name or "").strip().lower()
                    if ig_name:
                        _all = await fetch_registry_models_async(active_only=True) or []
                        _match = next(
                            (m for m in _all if (m.get("display_name") or "").strip().lower() == ig_name),
                            None,
                        )
                        if _match and _match.get("id"):
                            resp_model_id = UUID(str(_match["id"]))
                except Exception:  # noqa: BLE001
                    pass
            result = await handle_image_generation(
                body.edited_text,
                model_id=str(resp_model_id),
                user_id=str(current_user.id),
                session_id=user_msg.session_id,
            )
            new_response_text = result.get("response_text", "")
            new_model_name = result.get("model_name", "image-generation")

        elif mode == "outlook_query":
            from agentcore.services.mibuddy.outlook_agent import outlook_agent_node
            from agentcore.api.outlook_orch import get_outlook_token_from_request
            state = {
                "messages": [{"role": "user", "content": body.edited_text}],
                "user_id": str(current_user.id),
                "is_canvas_enabled": False,
                "access_token": get_outlook_token_from_request(request),
            }
            state = await outlook_agent_node(state)
            new_response_text = state.get("final_response", "") or (
                "I couldn't process that Outlook request."
            )
            new_model_name = "outlook"
            regen_sender_name = (
                await _get_model_display_name(session, resp_model_id)
                if resp_model_id else "Outlook"
            )

        elif mode == "document_qa":
            from agentcore.services.mibuddy.document_processor import (
                process_and_ingest,
                search_documents,
                build_doc_qa_prompt,
            )
            from agentcore.services.mibuddy.direct_model_chat import direct_model_chat

            # Newly-attached files are rare on an edit (user usually edits text
            # only), but handle them anyway for parity with the main chat path.
            doc_files = routing.get("doc_files", [])
            if doc_files:
                count = await process_and_ingest(doc_files, user_msg.session_id)
                if count > 0:
                    await asyncio.sleep(5)

            # Session-wide file list so per-file search sees every uploaded doc.
            file_names_for_search = await _collect_session_doc_filenames(
                user_msg.session_id, current_user.id,
            )
            for p in doc_files:
                nm = Path(p).name
                if nm not in file_names_for_search:
                    file_names_for_search.append(nm)

            chunks = await search_documents(
                body.edited_text,
                user_msg.session_id,
                file_list=file_names_for_search or None,
            )
            enriched_prompt = build_doc_qa_prompt(body.edited_text, chunks)

            if not resp_model_id:
                raise HTTPException(status_code=400, detail="No model selected for document Q&A.")
            image_files = routing.get("image_files", [])
            result = await direct_model_chat(
                model_id=str(resp_model_id),
                input_value=enriched_prompt,
                session_id=user_msg.session_id,
                files=image_files,
            )
            new_response_text = result.get("response_text", "")
            new_reasoning = result.get("reasoning_content")
            new_model_name = result.get("model_name", "")
            regen_sender_name = (
                await _get_model_display_name(session, resp_model_id)
                if resp_model_id else "Document Q&A"
            )

        else:
            # Unknown or "agent" mode — edit doesn't support switching to an
            # agent deployment on the fly. Fall back to plain model chat.
            from agentcore.services.mibuddy.direct_model_chat import direct_model_chat
            result = await direct_model_chat(
                model_id=str(resp_model_id),
                input_value=body.edited_text,
                session_id=user_msg.session_id,
                files=None,
                enable_reasoning=body.enable_reasoning,
            )
            new_response_text = result.get("response_text", "")
            new_reasoning = result.get("reasoning_content")
            new_model_name = result.get("model_name", "")
            regen_sender_name = await _get_model_display_name(session, resp_model_id)

        if not new_response_text or not new_response_text.strip():
            new_response_text = "No response was generated. Please try again."

        logger.info(
            f"[ORCH] Edit {message_id}: mode={mode} intent={intent} "
            f"model_id={resp_model_id} sender_name={regen_sender_name}"
        )

        # 5. Update the agent message in-place (if one exists) or create a new one.
        # Also refresh sender_name + model_id so the UI shows the correct model
        # for the new mode (e.g. edit flipped from model_direct to image_gen).
        if agent_msg is not None:
            agent_msg.text = new_response_text
            agent_msg.sender_name = regen_sender_name
            if hasattr(agent_msg, "model_id") and resp_model_id:
                agent_msg.model_id = resp_model_id
            if hasattr(agent_msg, "reasoning_content"):
                agent_msg.reasoning_content = new_reasoning
            # Clear stale files/content_blocks from the original response.
            # Without this, a refetch (page reload) would still see the OLD
            # tool cards or attached files alongside the regenerated text —
            # e.g. a Nano Banana edit kept the original tool-call blocks
            # from a prior agent answer, and a mode flip from agent →
            # image_gen left orphan content_blocks pointing at stale data.
            # The non-streaming handlers (handle_image_generation,
            # handle_web_search, etc.) embed their output entirely in
            # response_text, so resetting these to empty matches what the
            # streaming-flow persistence does for the same modes.
            agent_msg.files = []
            agent_msg.content_blocks = []
            session.add(agent_msg)
        else:
            # Edge case: user edited a message that had no response yet.
            agent_msg = OrchConversationTable(
                id=uuid4(),
                sender="model",
                sender_name=regen_sender_name,
                session_id=user_msg.session_id,
                text=new_response_text,
                user_id=user_msg.user_id,
                model_id=resp_model_id,
                reasoning_content=new_reasoning,
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                files=[],
                properties={},
                category="message",
                content_blocks=[],
            )
            await orch_add_message(agent_msg, session)

        await session.commit()
        await session.refresh(user_msg)
        await session.refresh(agent_msg)

        logger.info(f"[ORCH] Edited message {message_id} in session {user_msg.session_id}")

        # 6. Return both updated rows
        return EditMessageResponse(
            user_message=OrchMessageResponse(
                id=user_msg.id,
                timestamp=user_msg.timestamp.isoformat() if user_msg.timestamp else "",
                sender=user_msg.sender,
                sender_name=user_msg.sender_name,
                session_id=user_msg.session_id,
                text=user_msg.text,
                agent_id=user_msg.agent_id,
                deployment_id=user_msg.deployment_id,
                model_id=getattr(user_msg, "model_id", None),
                reasoning_content=getattr(user_msg, "reasoning_content", None),
                category=user_msg.category or "message",
                files=user_msg.files if user_msg.files else None,
                properties=user_msg.properties if isinstance(user_msg.properties, dict) else None,
                content_blocks=user_msg.content_blocks if user_msg.content_blocks else None,
            ),
            agent_message=OrchMessageResponse(
                id=agent_msg.id,
                timestamp=agent_msg.timestamp.isoformat() if agent_msg.timestamp else "",
                sender=agent_msg.sender,
                sender_name=agent_msg.sender_name,
                session_id=agent_msg.session_id,
                text=agent_msg.text,
                agent_id=agent_msg.agent_id,
                deployment_id=agent_msg.deployment_id,
                model_id=getattr(agent_msg, "model_id", None),
                model_name=new_model_name,
                reasoning_content=getattr(agent_msg, "reasoning_content", None),
                category=agent_msg.category or "message",
                files=agent_msg.files if agent_msg.files else None,
                properties=agent_msg.properties if isinstance(agent_msg.properties, dict) else None,
                content_blocks=agent_msg.content_blocks if agent_msg.content_blocks else None,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error editing message {message_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Thumbs up/down feedback (MiBuddy-parity)
#
# Storage is 4 nullable columns on `orch_conversation` — a single feedback per
# message. Re-voting UPDATEs the same row (no history bloat); un-voting clears
# the columns to NULL. One row in the DB == current state == what the UI shows.
# ---------------------------------------------------------------------------


async def _load_user_feedback_target(
    session: "DbSession",
    message_id: str,
    current_user,
) -> OrchConversationTable:
    """Load a message and verify the caller owns it for feedback purposes.

    - Must be an assistant message (user messages cannot be rated).
    - Must belong to the caller's session (enforced via user_id match).
    """
    try:
        msg_uuid = UUID(message_id)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid message id") from exc

    msg = await session.get(OrchConversationTable, msg_uuid)
    if msg is None:
        raise HTTPException(status_code=404, detail="Message not found")
    # Accept both senders because assistant messages are stored as "agent"
    # for agent-deployment responses and "model" for direct model chat.
    if msg.sender not in ("agent", "model"):
        raise HTTPException(
            status_code=400,
            detail="Only assistant messages can receive feedback",
        )
    if msg.user_id is not None and msg.user_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Cannot rate a message that does not belong to your session",
        )
    return msg


def _feedback_row_to_response(msg: OrchConversationTable) -> FeedbackResponse:
    return FeedbackResponse(
        message_id=msg.id,
        feedback_rating=msg.feedback_rating,
        feedback_reasons=msg.feedback_reasons,
        feedback_comment=msg.feedback_comment,
        feedback_at=msg.feedback_at.isoformat() if msg.feedback_at else None,
    )


@router.post(
    "/messages/{message_id}/feedback",
    response_model=FeedbackResponse,
    status_code=200,
    dependencies=[Depends(PermissionChecker(["interact_agents"]))],
)
async def submit_message_feedback(
    *,
    session: DbSession,
    current_user: CurrentActiveUser,
    message_id: str,
    body: FeedbackRequest,
):
    """Upsert thumbs-up / thumbs-down feedback on an assistant message.

    Re-voting on the same message overwrites the existing row — no duplicates
    and no history records. Switching from up → down (or vice versa) is just
    another POST with the new rating.
    """
    try:
        msg = await _load_user_feedback_target(session, message_id, current_user)

        # Upsert: four columns on the same row. Re-vote / switch = UPDATE.
        msg.feedback_rating = body.rating
        msg.feedback_reasons = list(body.reasons) if body.reasons else None
        msg.feedback_comment = body.comment if body.comment else None
        msg.feedback_at = datetime.now(timezone.utc).replace(tzinfo=None)

        session.add(msg)
        await session.commit()
        await session.refresh(msg)

        logger.info(
            f"[ORCH] Feedback {body.rating!r} saved for message {message_id} "
            f"(reasons={len(body.reasons)}, comment={'yes' if body.comment else 'no'})"
        )
        return _feedback_row_to_response(msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving feedback for message {message_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete(
    "/messages/{message_id}/feedback",
    response_model=FeedbackResponse,
    status_code=200,
    dependencies=[Depends(PermissionChecker(["interact_agents"]))],
)
async def remove_message_feedback(
    *,
    session: DbSession,
    current_user: CurrentActiveUser,
    message_id: str,
):
    """Remove thumbs-up / thumbs-down feedback — clears all 4 columns to NULL.

    Called when the user clicks the currently-active thumb a second time
    (un-vote). Idempotent: calling on an already-cleared row is a no-op.
    """
    try:
        msg = await _load_user_feedback_target(session, message_id, current_user)

        msg.feedback_rating = None
        msg.feedback_reasons = None
        msg.feedback_comment = None
        msg.feedback_at = None

        session.add(msg)
        await session.commit()
        await session.refresh(msg)

        logger.info(f"[ORCH] Feedback cleared for message {message_id}")
        return _feedback_row_to_response(msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing feedback for message {message_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/sessions/{session_id}/shared-messages", status_code=200)
async def get_shared_orch_session_messages(
    *,
    session: DbSession,
    current_user: CurrentActiveUser,
    session_id: str,
):
    """MiBuddy-parity share view: authenticated read of a session's messages
    by UUID-as-token, without the owner filter. The recipient must be
    signed in (ProtectedRoute bounces unauthenticated users to login),
    but the session does NOT have to belong to them — matching MiBuddy's
    `/history/share/view/{share_token}` flow.

    Returns `is_owner` so the frontend can decide whether to render a
    "shared read-only" banner or treat it as a normal owned session.
    """
    try:
        messages = await orch_get_messages(session, session_id=session_id)
        if not messages:
            raise HTTPException(status_code=404, detail="Conversation not found")
        is_owner = any(m.user_id == current_user.id for m in messages)
        session_title: str | None = None
        for m in messages:
            if getattr(m, "session_title", None):
                session_title = m.session_title
                break
        return {
            "session_id": session_id,
            "is_owner": is_owner,
            "session_title": session_title,
            "messages": [
                OrchMessageResponse(
                    id=m.id,
                    timestamp=m.timestamp.isoformat() if m.timestamp else "",
                    sender=m.sender,
                    sender_name=m.sender_name,
                    session_id=m.session_id,
                    text=m.text,
                    agent_id=m.agent_id,
                    deployment_id=m.deployment_id,
                    model_id=getattr(m, "model_id", None),
                    reasoning_content=getattr(m, "reasoning_content", None),
                    category=m.category or "message",
                    files=m.files if m.files else None,
                    properties=m.properties if isinstance(m.properties, dict) else None,
                    content_blocks=m.content_blocks if m.content_blocks else None,
                    feedback_rating=getattr(m, "feedback_rating", None),
                    feedback_reasons=getattr(m, "feedback_reasons", None),
                    feedback_comment=getattr(m, "feedback_comment", None),
                    feedback_at=(
                        getattr(m, "feedback_at").isoformat()
                        if getattr(m, "feedback_at", None)
                        else None
                    ),
                )
                for m in messages
                if (m.text and m.text.strip()) or m.category == "context_reset"
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching shared orch session messages: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_orch_session(
    *,
    session: DbSession,
    current_user: CurrentActiveUser,
    session_id: str,
):
    """Delete all messages and transactions for an orchestrator session."""
    try:
        await orch_delete_session(session, session_id, user_id=current_user.id)
        await orch_delete_session_transactions(session, session_id)
        # Cleanup document Q&A vectors from Pinecone
        try:
            from agentcore.services.mibuddy.document_processor import cleanup_session_docs
            await cleanup_session_docs(session_id)
        except Exception as cleanup_err:
            logger.warning(f"[DocQA] Cleanup failed for session {session_id}: {cleanup_err}")
    except Exception as e:
        logger.error(f"Error deleting orch session: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ═══════════════════════════════════════════════════════════════════════════
# 5b. Archive / unarchive a session
# ═══════════════════════════════════════════════════════════════════════════


class ArchiveRequest(BaseModel):
    is_archived: bool = True


@router.post("/sessions/{session_id}/archive", status_code=200)
async def archive_orch_session(
    *,
    session: DbSession,
    current_user: CurrentActiveUser,
    session_id: str,
    body: ArchiveRequest,
):
    """Toggle archive/unarchive for an orchestrator session."""
    try:
        updated = await orch_archive_session(
            session, session_id, is_archived=body.is_archived, user_id=current_user.id,
        )
        if updated == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        action = "archived" if body.is_archived else "unarchived"
        return {
            "session_id": session_id,
            "is_archived": body.is_archived,
            "message": f"Session {session_id} has been {action} successfully.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error archiving orch session: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class SessionTitleRequest(BaseModel):
    title: str | None = None  # null to clear


@router.post("/sessions/{session_id}/title", status_code=200)
async def set_orch_session_title(
    *,
    session: DbSession,
    current_user: CurrentActiveUser,
    session_id: str,
    body: SessionTitleRequest,
):
    """Set (or clear) the user-chosen title for an orchestrator session.

    MiBuddy-parity rename — writes to the `session_title` column added
    in migration `95791a21c989_add_session_title_to_orch_conversation`.
    """
    try:
        title = (body.title or "").strip() or None
        updated = await orch_set_session_title(
            session, session_id, title=title, user_id=current_user.id,
        )
        if updated == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session_id": session_id, "title": title}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming orch session: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ═══════════════════════════════════════════════════════════════════════════
# 6. Rename a session
# ═══════════════════════════════════════════════════════════════════════════


@router.patch("/sessions/{session_id}", status_code=200)
async def rename_orch_session(
    *,
    session: DbSession,
    current_user: CurrentActiveUser,
    session_id: str,
    new_session_id: str = Query(..., description="New session identifier"),
):
    """Rename an orchestrator session (updates session_id on all messages)."""
    try:
        count = await orch_rename_session(session, session_id, new_session_id, user_id=current_user.id)
        return {"updated": count, "new_session_id": new_session_id}
    except Exception as e:
        logger.error(f"Error renaming orch session: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e




class ActiveAgentResponse(BaseModel):
    agent_id: UUID | None = None
    deployment_id: UUID | None = None
    agent_name: str | None = None


@router.get(
    "/sessions/{session_id}/active-agent",
    response_model=ActiveAgentResponse,
    status_code=200,
)
async def get_active_agent(
    *,
    session: DbSession,
    current_user: CurrentActiveUser,
    session_id: str,
):
    """Return the currently active (sticky) agent for a session.

    Returns null fields if the session has no messages yet.
    """
    try:
        active = await orch_get_active_agent(session, session_id)
        if not active:
            return ActiveAgentResponse()
        dep = await session.get(AgentDeploymentProd, active["deployment_id"])
        if not dep:
            dep = await session.get(AgentDeploymentUAT, active["deployment_id"])
        return ActiveAgentResponse(
            agent_id=active["agent_id"],
            deployment_id=active["deployment_id"],
            agent_name=dep.agent_name if dep else None,
        )
    except Exception as e:
        logger.error(f"Error getting active agent: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


from agentcore.services.mibuddy.model_capabilities import detect_capabilities


# ---------------------------------------------------------------------------
# Canvas edit (MiBuddy-style) — persist user edits of an agent message
# ---------------------------------------------------------------------------


class CanvasEditRequest(BaseModel):
    """Port of MiBuddy's /canvas/edit payload.

    Supported operations (matches MiBuddy exactly):
      - "manual":        just save whatever `content` the user typed
      - "reading_level": rewrite at `level` (kindergarten / middle school /
                         high school / college / graduate)
      - "emoji":         add playful emojis (emoji_action="words") OR
                         strip all emojis (emoji_action="remove")
    """
    message_id: UUID
    session_id: str | None = None
    content: str
    operation: str = "manual"
    level: str | None = None
    emoji_action: str | None = None  # "words" | "remove"


# Ported verbatim from MiBuddy app.py:3036-3042.
_LEVEL_DESCRIPTIONS = {
    "kindergarten": "Use very simple vocabulary, very short sentences, suitable for a 5-year-old child.",
    "middle school": "Use moderately simple vocabulary, clear explanations, suitable for students aged 10–14.",
    "high school": "Use moderate complexity, varied sentence structure, suitable for grade 9-12 students.",
    "college": "Use advanced vocabulary, complex ideas, academically structured sentences.",
    "graduate": "Use highly academic tone, domain-specific terminology, and research-level abstraction.",
}


_EMOJI_PROMPT = """
You are a creative writing assistant.
Task: Transform the text by adding **expressive, colorful emojis** throughout.
Guidelines:
- Add emojis to most major words (nouns, verbs, adjectives, and festive terms).
- Use emojis that match the emotion, meaning, or energy of each word.
- Keep the meaning and structure intact — do NOT remove words or punctuation.
- Return only the transformed text
Example:
Input: Wishing you a bright and joyful Diwali!
Output: ✨🙏 Wishing you a 🌟 bright & 😊 joyful Diwali! ✨🪔

Text:
{content}
"""


def _strip_emojis(text: str) -> str:
    """Port of `remove_emojis` — removes unicode emoji chars.

    Uses the same `emoji` python library MiBuddy uses. If unavailable,
    falls back to a regex that removes most common emoji ranges.
    """
    try:
        import emoji  # type: ignore
        return emoji.replace_emoji(text, "")
    except Exception:
        import re as _re
        pat = _re.compile(
            "[\U0001F300-\U0001FAFF"
            "\U0001F600-\U0001F64F"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U00002600-\U000026FF"
            "\U00002700-\U000027BF"
            "]+",
            flags=_re.UNICODE,
        )
        return pat.sub("", text)


def _canvas_llm_rewrite(prompt: str) -> str:
    """Reuse the same AzureAIFoundryLLM shim the Outlook agent uses."""
    from agentcore.services.mibuddy._outlook_agent_deps import AzureAIFoundryLLM
    llm = AzureAIFoundryLLM()
    return llm.complete(prompt).text or ""


@router.post(
    "/canvas/edit",
    status_code=200,
    dependencies=[Depends(PermissionChecker(["interact_agents"]))],
)
async def canvas_edit(
    *,
    body: CanvasEditRequest,
    session: DbSession,
    current_user: CurrentActiveUser,
):
    """Canvas edit endpoint — port of MiBuddy's /canvas/edit.

    Supports:
      - operation="manual"        → save user-edited text
      - operation="reading_level" → rewrite at the given reading level
      - operation="emoji"         → add or strip emojis
    Returns MiBuddy's exact response shape:
        {"status":"success","data":[{"id":..,"content":..,"canvas":true,...}]}
    so the frontend can drop-in reuse MiBuddy's client-side handler.
    """
    from sqlalchemy import update

    edited = body.content or ""

    # Reading-level rewrite — MiBuddy calls LLM with the level description.
    if body.operation == "reading_level" and body.level:
        if body.level == "reading level":
            # MiBuddy's "keep current reading level" sentinel — no-op.
            pass
        elif body.level in _LEVEL_DESCRIPTIONS:
            prompt = (
                "Rewrite the following content for this reading level.\n\n"
                f"Target Level: {body.level}\n"
                f"Description: {_LEVEL_DESCRIPTIONS[body.level]}\n\n"
                "Rules:\n"
                "- Preserve meaning\n"
                "- Do not summarize\n"
                "- Do not add new content\n\n"
                f"Content:\n{edited}"
            )
            try:
                rewritten = _canvas_llm_rewrite(prompt)
                if rewritten.strip():
                    edited = rewritten
            except Exception as e:
                logger.error(f"[canvas/edit] reading_level LLM failed: {e}")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid level. Valid: {list(_LEVEL_DESCRIPTIONS.keys())}",
            )

    # Emoji operation — can run alongside reading_level (as in MiBuddy).
    if body.emoji_action == "words":
        try:
            rewritten = _canvas_llm_rewrite(_EMOJI_PROMPT.format(content=edited))
            if rewritten.strip():
                edited = rewritten
        except Exception as e:
            logger.error(f"[canvas/edit] emoji-add LLM failed: {e}")
    elif body.emoji_action == "remove":
        edited = _strip_emojis(edited)

    # Persist the final content
    stmt = (
        update(OrchConversationTable)
        .where(OrchConversationTable.id == body.message_id)
        .where(OrchConversationTable.user_id == current_user.id)
        .values(text=edited)
    )
    result = await session.execute(stmt)
    await session.commit()
    if (result.rowcount or 0) == 0:
        raise HTTPException(
            status_code=404,
            detail="Message not found or not owned by this user",
        )

    # MiBuddy-compatible response shape
    return {
        "status": "success",
        "data": [
            {
                "role": "assistant",
                "id": str(body.message_id),
                "content": edited,
                "canvas": True,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }
        ],
    }


# ---------------------------------------------------------------------------
# MiBuddy file upload (dedicated container)
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    status_code=201,
    dependencies=[Depends(PermissionChecker(["interact_agents"]))],
)
async def mibuddy_upload_file(
    *,
    request: Request,
    session: DbSession,
    current_user: CurrentActiveUser,
    file: UploadFile,
):
    """Upload a file to the MiBuddy dedicated container.

    Used by orchestrator model chat for document uploads and chat images.
    Files are stored in: {mibuddy_container}/{user_id}/{category}/{filename}

    Returns the file_path for use in chat requests.
    """
    from agentcore.services.mibuddy.docqa_storage import save_file as mibuddy_save, FileCategory
    from agentcore.services.mibuddy.document_extractor import IMAGE_EXTENSIONS, SUPPORTED_DOC_EXTENSIONS

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    user_id = str(current_user.id)
    file_content = await file.read()
    file_name = file.filename
    ext = Path(file_name).suffix.lower()

    # Determine category based on file type
    if ext in IMAGE_EXTENSIONS:
        category = FileCategory.CHAT_IMAGES
    elif ext in SUPPORTED_DOC_EXTENSIONS:
        category = FileCategory.UPLOADS
    else:
        category = FileCategory.UPLOADS

    try:
        file_path = await mibuddy_save(user_id, file_name, file_content, category=category)
        logger.info(f"[MiBuddy Upload] {category.value}/{file_name} for user {user_id}")
        return {"file_path": file_path, "category": category.value}
    except Exception as e:
        logger.error(f"[MiBuddy Upload] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# List AI-generated images from MiBuddy container
# ---------------------------------------------------------------------------


@router.get(
    "/generated-images",
    status_code=200,
    dependencies=[Depends(PermissionChecker(["interact_agents"]))],
)
async def list_generated_images(
    *,
    current_user: CurrentActiveUser,
):
    """List AI-generated images for the current user from the MiBuddy container.

    Returns auth-proxied URLs that the frontend loads with JWT, so images
    display regardless of whether the blob container has public access.
    """
    try:
        from agentcore.services.mibuddy.docqa_storage import (
            list_files,
            FileCategory,
        )

        user_id = str(current_user.id)
        file_names = await list_files(user_id, category=FileCategory.GENERATED_IMAGES)

        images = []
        for name in sorted(file_names, reverse=True)[:20]:  # newest first, max 20
            images.append({
                "name": name,
                "src": f"/api/files/images/{user_id}/generated-images/{name}",
            })
        return images
    except Exception as e:
        logger.error(f"Error listing generated images: {e}")
        return []


# ---------------------------------------------------------------------------
# Serve MiBuddy images (generated-images, uploads, chat-images)
# ---------------------------------------------------------------------------


@router.get("/images/{user_id}/{subfolder}/{file_name}")
async def serve_mibuddy_image(user_id: str, subfolder: str, file_name: str):
    """Serve images from the MiBuddy container subfolders."""
    from agentcore.services.mibuddy.docqa_storage import get_file_by_path
    from agentcore.services.storage.constants import build_content_type_from_extension

    extension = file_name.split(".")[-1]
    try:
        content_type = build_content_type_from_extension(extension)
    except Exception:
        content_type = "image/png"

    try:
        path = f"{user_id}/{subfolder}/{file_name}"
        file_content = await get_file_by_path(path)
        logger.info(f"[MiBuddy] Served image: {path}")
        return StreamingResponse(BytesIO(file_content), media_type=content_type)
    except Exception as e:
        logger.warning(f"[MiBuddy] Image not found: {user_id}/{subfolder}/{file_name}")
        raise HTTPException(status_code=404, detail=f"Image not found: {file_name}") from e


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Autocomplete suggestions
# ---------------------------------------------------------------------------


@router.get(
    "/suggestions",
    status_code=200,
    dependencies=[Depends(PermissionChecker(["interact_agents"]))],
)
async def get_suggestions(
    q: str = "",
):
    """Generate autocomplete suggestions as user types."""
    try:
        from agentcore.services.mibuddy.suggestion_service import get_suggestions
        suggestions = await get_suggestions(q)
        return {"suggestions": suggestions}
    except Exception as e:
        logger.debug(f"Suggestions failed: {e}")
        return {"suggestions": []}


# ---------------------------------------------------------------------------
# List available models for orchestrator chat
# ---------------------------------------------------------------------------


@router.get(
    "/models",
    response_model=list[OrchModelSummary],
    status_code=200,
    dependencies=[Depends(PermissionChecker(["interact_agents"]))],
)
async def list_orch_models(
    *,
    session: DbSession,
    current_user: CurrentActiveUser,
):
    """List deployed LLM models available to the current user for direct chat.

    Includes:
    - Virtual entries for configured MiBuddy features (Web Search, Nano Banana, etc.)
    - Real models from the Model Registry
    """
    try:
        from agentcore.services.database.models.model_registry.model import (
            ModelApprovalStatus,
            ModelRegistry,
        )
        from agentcore.services.model_service_client import fetch_registry_models_async

        result: list[OrchModelSummary] = []
        settings = get_settings_service().settings
        default_model_name = (settings.default_orch_model_name or "").strip().lower()

        logger.info(f"[ORCH Models] image_gen_model_name='{settings.image_gen_model_name}', web_search_model_name='{settings.web_search_model_name}', default_orch_model_name='{settings.default_orch_model_name}'")

        # ── Real models from registry ──
        raw_rows = await fetch_registry_models_async(
            model_type="llm",
            active_only=True,
        )

        if raw_rows:
            # Model service returned data — append to result (which already has virtual entries)
            for row in raw_rows:
                try:
                    model_id = row.get("id")
                    if not model_id:
                        continue
                    approval = str(row.get("approval_status", "approved")).lower()
                    if approval != "approved":
                        continue
                    # Filter by show_in: only show models meant for orchestrator
                    show_in = row.get("show_in") or ["orchestrator", "agent"]
                    if "orchestrator" not in show_in:
                        continue
                    provider = row.get("provider", "")
                    model_name_val = row.get("model_name", "")
                    explicit_caps = row.get("capabilities")
                    merged_caps = detect_capabilities(provider, model_name_val, explicit_caps)
                    display = row.get("display_name", model_name_val)
                    result.append(
                        OrchModelSummary(
                            model_id=UUID(str(model_id)),
                            display_name=display,
                            provider=provider,
                            model_name=model_name_val,
                            model_type=row.get("model_type", "llm"),
                            capabilities=merged_caps,
                            is_default=bool(default_model_name and display.strip().lower() == default_model_name),
                        )
                    )
                except Exception:
                    continue
            # Put default model first
            result.sort(key=lambda m: (not m.is_default, m.display_name))
            return result

        # Strategy 2: Fallback to local DB if model service unavailable
        stmt = (
            select(ModelRegistry)
            .where(
                ModelRegistry.is_active.is_(True),
                ModelRegistry.model_type == "llm",
                ModelRegistry.approval_status == ModelApprovalStatus.APPROVED.value,
            )
            .order_by(ModelRegistry.provider, ModelRegistry.display_name)
        )
        rows = (await session.exec(stmt)).all()

        return [
            OrchModelSummary(
                model_id=row.id,
                display_name=row.display_name,
                provider=row.provider,
                model_name=row.model_name,
                model_type=row.model_type,
                capabilities=detect_capabilities(row.provider, row.model_name, row.capabilities),
                is_default=bool(default_model_name and row.display_name.strip().lower() == default_model_name),
            )
            for row in rows
            if "orchestrator" in (row.show_in or ["orchestrator", "agent"])
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing orchestrator models: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
