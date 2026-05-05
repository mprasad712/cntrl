"""Add deploy_id to agent_publish_recipient for per-version sharing

Revision ID: 20260505_add_deploy_id_apr
Revises: 20260504_fix_prt_index
Create Date: 2026-05-05

Adds a deploy_id column so shares are scoped to a specific deployment
version (UAT or PROD) rather than the whole agent. Existing rows keep
deploy_id = NULL and are treated as legacy agent-level shares.
The old unique constraint (agent_id, dept_id, recipient_email) is dropped
and replaced with (deploy_id, dept_id, recipient_email); Postgres treats
NULLs as distinct so legacy rows remain unaffected.
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


def upgrade() -> None:
    op.add_column(
        "agent_publish_recipient",
        sa.Column("deploy_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        "ix_agent_publish_recipient_deploy_id",
        "agent_publish_recipient",
        ["deploy_id"],
    )
    op.drop_constraint(
        "uq_agent_publish_recipient_agent_dept_email",
        "agent_publish_recipient",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_agent_publish_recipient_deploy_dept_email",
        "agent_publish_recipient",
        ["deploy_id", "dept_id", "recipient_email"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_agent_publish_recipient_deploy_dept_email",
        "agent_publish_recipient",
        type_="unique",
    )
    op.drop_index("ix_agent_publish_recipient_deploy_id", table_name="agent_publish_recipient")
    op.drop_column("agent_publish_recipient", "deploy_id")
    op.create_unique_constraint(
        "uq_agent_publish_recipient_agent_dept_email",
        "agent_publish_recipient",
        ["agent_id", "dept_id", "recipient_email"],
    )
