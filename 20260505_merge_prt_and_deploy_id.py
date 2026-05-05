"""Merge password_reset_token index fix and deploy_id branches

Revision ID: 20260505_merge_prt_deploy
Revises: 20260504_fix_prt_index, 20260505_add_deploy_id_apr
Create Date: 2026-05-05
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union


revision: str = "20260505_merge_prt_deploy"
down_revision: Union[str, Sequence[str], None] = (
    "20260504_fix_prt_index",
    "20260505_add_deploy_id_apr",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
