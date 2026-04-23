"""vector column made 3072 dim

Revision ID: 05604ffd2f37
Revises: 1be07f9ede1e
Create Date: 2026-04-23 16:26:10.147999
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = '05604ffd2f37'
down_revision: Union[str, None] = '1be07f9ede1e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.execute("DROP INDEX IF EXISTS idx_knowledge_base_embedding")
    op.execute("ALTER TABLE knowledge_base ALTER COLUMN embedding TYPE vector(3072)")

def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_knowledge_base_embedding")
    op.execute("ALTER TABLE knowledge_base ALTER COLUMN embedding TYPE vector(768)")
