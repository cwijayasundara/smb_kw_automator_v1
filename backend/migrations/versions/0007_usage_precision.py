"""Keep sub-microdollar token costs instead of rounding cheap calls to zero.

Revision ID: 0007
"""

from alembic import op

revision = "0007"
down_revision = "0006"


def upgrade() -> None:
    op.execute("ALTER TABLE usage_events ALTER COLUMN cost_usd TYPE numeric(18,12)")


def downgrade() -> None:
    op.execute("ALTER TABLE usage_events ALTER COLUMN cost_usd TYPE numeric(12,6)")
