"""Add deploy_id to agent_publish_recipient for per-version sharing

Revision ID: 20260505_add_deploy_id_apr
Revises: 20260317_merge_all
Create Date: 2026-05-05

Adds a deploy_id column so shares are scoped to a specific deployment
version (UAT or PROD) rather than the whole agent. Existing rows keep
deploy_id = NULL and are treated as legacy agent-level shares.
The old unique constraint (agent_id, dept_id, recipient_email) is dropped
and replaced with (deploy_id, dept_id, recipient_email); Postgres treats
NULLs as distinct so legacy rows remain unaffected.

All steps are guarded with existence checks so the migration is safe to
run on any DB state (e.g. partial apply, stamp-without-upgrade, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260505_add_deploy_id_apr"
down_revision: Union[str, Sequence[str], None] = "20260317_merge_all"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(bind, table: str, column: str) -> bool:
    result = bind.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :column"
        ),
        {"table": table, "column": column},
    )
    return result.fetchone() is not None


def _index_exists(bind, index_name: str) -> bool:
    result = bind.execute(
        sa.text("SELECT 1 FROM pg_indexes WHERE indexname = :name"),
        {"name": index_name},
    )
    return result.fetchone() is not None


def _constraint_exists(bind, constraint_name: str) -> bool:
    result = bind.execute(
        sa.text("SELECT 1 FROM pg_constraint WHERE conname = :name"),
        {"name": constraint_name},
    )
    return result.fetchone() is not None


def upgrade() -> None:
    bind = op.get_bind()

    if not _column_exists(bind, "agent_publish_recipient", "deploy_id"):
        op.add_column(
            "agent_publish_recipient",
            sa.Column("deploy_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        )

    if not _index_exists(bind, "ix_agent_publish_recipient_deploy_id"):
        op.create_index(
            "ix_agent_publish_recipient_deploy_id",
            "agent_publish_recipient",
            ["deploy_id"],
        )

    if _constraint_exists(bind, "uq_agent_publish_recipient_agent_dept_email"):
        op.drop_constraint(
            "uq_agent_publish_recipient_agent_dept_email",
            "agent_publish_recipient",
            type_="unique",
        )

    if not _constraint_exists(bind, "uq_agent_publish_recipient_deploy_dept_email"):
        op.create_unique_constraint(
            "uq_agent_publish_recipient_deploy_dept_email",
            "agent_publish_recipient",
            ["deploy_id", "dept_id", "recipient_email"],
        )


def downgrade() -> None:
    bind = op.get_bind()

    if _constraint_exists(bind, "uq_agent_publish_recipient_deploy_dept_email"):
        op.drop_constraint(
            "uq_agent_publish_recipient_deploy_dept_email",
            "agent_publish_recipient",
            type_="unique",
        )

    if _index_exists(bind, "ix_agent_publish_recipient_deploy_id"):
        op.drop_index("ix_agent_publish_recipient_deploy_id", table_name="agent_publish_recipient")

    if _column_exists(bind, "agent_publish_recipient", "deploy_id"):
        op.drop_column("agent_publish_recipient", "deploy_id")

    if not _constraint_exists(bind, "uq_agent_publish_recipient_agent_dept_email"):
        op.create_unique_constraint(
            "uq_agent_publish_recipient_agent_dept_email",
            "agent_publish_recipient",
            ["agent_id", "dept_id", "recipient_email"],
        )
