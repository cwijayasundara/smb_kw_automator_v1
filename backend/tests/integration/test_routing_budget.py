import asyncio
import uuid
from decimal import Decimal

from sqlalchemy import text

from keel.platform.db import tenant_session
from keel.routing.budget import BudgetExceeded, reserve, settle


async def _ctx(client):
    me = (await client.get("/api/me")).json()
    return uuid.UUID(me["org_id"]), uuid.UUID(me["user_id"])


async def test_reservations_are_tenant_isolated_and_settle_once(owner, other_tenant):
    org, user = await _ctx(owner)
    other_org, other_user = await _ctx(other_tenant)
    attempt = uuid.uuid4()
    await reserve(org, user, attempt, Decimal("0.01"))
    async with tenant_session(other_org, other_user) as db:
        assert await db.scalar(text("SELECT count(*) FROM ai_reservations WHERE id=:id"), {"id": attempt}) == 0
    for _ in range(2):
        await settle(
            org, user, attempt, model="recorded-model", input_tokens=100, output_tokens=10, cost_usd=Decimal("0.005")
        )
    async with tenant_session(org, user) as db:
        count = await db.scalar(
            text("SELECT count(*) FROM usage_events WHERE ref->>'attempt_id'=:id"), {"id": str(attempt)}
        )
        assert count == 1


async def test_concurrent_calls_cannot_reserve_more_than_budget(owner):
    org, user = await _ctx(owner)
    attempts = [uuid.uuid4(), uuid.uuid4()]
    results = await asyncio.gather(*(reserve(org, user, a, Decimal("6")) for a in attempts), return_exceptions=True)
    assert sum(r is None for r in results) == 1
    assert sum(isinstance(r, BudgetExceeded) for r in results) == 1
    successful = attempts[results.index(None)]
    await settle(org, user, successful, model="recorded-model", input_tokens=0, output_tokens=0, cost_usd=Decimal(0))


async def test_expired_reservation_does_not_discard_possible_provider_spend(owner):
    org, user = await _ctx(owner)
    attempt = uuid.uuid4()
    await reserve(org, user, attempt, Decimal("6"))
    async with tenant_session(org, user) as db:
        await db.execute(
            text("UPDATE ai_reservations SET expires_at = now() - interval '1 day' WHERE id=:id"), {"id": attempt}
        )
    try:
        await reserve(org, user, uuid.uuid4(), Decimal("6"))
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("A stale reservation must retain budget until reconciled")
    await settle(org, user, attempt, model="recorded-model", input_tokens=0, output_tokens=0, cost_usd=Decimal("6"))
