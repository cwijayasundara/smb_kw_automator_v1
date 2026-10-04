"""Cross-process reservations. Every query runs under verified tenant RLS context."""

import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from keel.audit.service import record_usage
from keel.platform.config import get_settings
from keel.platform.db import tenant_session


class BudgetExceeded(ValueError):
    """The tenant's configured AI budget cannot fund the next call."""


async def reserve(org_id: uuid.UUID, user_id: uuid.UUID, attempt_id: uuid.UUID, estimate: Decimal) -> None:
    s = get_settings()
    async with tenant_session(org_id, user_id) as db:
        await db.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:org, 0))"), {"org": str(org_id)})
        spent = await db.scalar(
            text("SELECT coalesce(sum(cost_usd), 0) FROM usage_events WHERE at >= date_trunc('month', now())")
        )
        # A crashed process may already have incurred provider spend. Expiry marks
        # stale reservations for reconciliation; it must not silently free budget.
        held = await db.scalar(text("SELECT coalesce(sum(amount), 0) FROM ai_reservations"))
        if Decimal(spent or 0) + Decimal(held or 0) + estimate > s.monthly_ai_budget_usd:
            raise BudgetExceeded("Monthly AI budget exhausted")
        await db.execute(
            text(
                "INSERT INTO ai_reservations (id, org_id, amount, expires_at) "
                "VALUES (:id, :org, :amount, now() + make_interval(secs => :ttl))"
            ),
            {"id": attempt_id, "org": org_id, "amount": estimate, "ttl": s.router_timeout_seconds + 60},
        )


async def settle(
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    attempt_id: uuid.UUID,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: Decimal,
    **ref: Any,
) -> None:
    async with tenant_session(org_id, user_id) as db:
        await db.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:org, 0))"), {"org": str(org_id)})
        # Deletion is the exactly-once settlement marker. Duplicate callbacks cannot bill twice.
        deleted = await db.scalar(text("DELETE FROM ai_reservations WHERE id = :id RETURNING id"), {"id": attempt_id})
        if deleted is None:
            return
        await record_usage(
            db,
            org_id,
            "ask",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            attempt_id=str(attempt_id),
            **ref,
        )
