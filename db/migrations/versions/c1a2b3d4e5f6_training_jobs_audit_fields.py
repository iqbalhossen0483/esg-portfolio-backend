"""training_jobs audit fields + portfolio_allocations index

Revision ID: c1a2b3d4e5f6
Revises: b714349a8468
Create Date: 2026-04-23 11:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "c1a2b3d4e5f6"
down_revision: Union[str, None] = "b714349a8468"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # training_jobs: new audit/idempotency columns
    op.add_column("training_jobs", sa.Column("file_sha256", sa.String(length=64), nullable=True))
    op.add_column("training_jobs", sa.Column("uploaded_by", sa.Integer(), nullable=True))
    op.add_column("training_jobs", sa.Column("error", sa.Text(), nullable=True))
    op.create_index("ix_training_jobs_file_sha256", "training_jobs", ["file_sha256"], unique=False)
    op.create_foreign_key(
        "fk_training_jobs_uploaded_by",
        "training_jobs",
        "users",
        ["uploaded_by"],
        ["id"],
        ondelete="SET NULL",
    )

    # portfolio_allocations: index portfolio_id (no FK target table yet)
    op.create_index(
        "ix_portfolio_allocations_portfolio_id",
        "portfolio_allocations",
        ["portfolio_id"],
        unique=False,
    )
    op.create_index(
        "ix_portfolio_allocations_portfolio_symbol",
        "portfolio_allocations",
        ["portfolio_id", "symbol"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_portfolio_allocations_portfolio_symbol", table_name="portfolio_allocations")
    op.drop_index("ix_portfolio_allocations_portfolio_id", table_name="portfolio_allocations")
    op.drop_constraint("fk_training_jobs_uploaded_by", "training_jobs", type_="foreignkey")
    op.drop_index("ix_training_jobs_file_sha256", table_name="training_jobs")
    op.drop_column("training_jobs", "error")
    op.drop_column("training_jobs", "uploaded_by")
    op.drop_column("training_jobs", "file_sha256")
