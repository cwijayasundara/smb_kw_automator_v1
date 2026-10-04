"""Tenant-scoped in-flight AI budget reservations.

Revision ID: 0006
"""

from alembic import op

from migrations.rls import tenant_rls

revision = "0006"
down_revision = "0005"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE ai_reservations (
            id uuid PRIMARY KEY,
            org_id uuid NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
            amount numeric NOT NULL CHECK (amount >= 0),
            expires_at timestamptz NOT NULL
        );
        CREATE INDEX ai_reservations_org_expiry ON ai_reservations (org_id, expires_at);
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai_reservations TO keel_app;
        """
    )
    op.execute(tenant_rls("ai_reservations"))


def downgrade() -> None:
    op.execute("DROP TABLE ai_reservations")
