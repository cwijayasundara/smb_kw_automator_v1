import uuid
from decimal import Decimal
from typing import Any

import pytest
from harness_model_router import Router, Usage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from sqlalchemy import text

from keel.agents import factory
from keel.platform.config import get_settings
from keel.platform.db import tenant_session
from keel.routing.profiles import profiles, route_request
from tests.integration.test_routing_budget import _ctx


class RecordedModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "recorded"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="Recorded answer",
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "total_tokens": 110,
                            "input_token_details": {"cache_read": 20},
                        },
                    )
                )
            ]
        )


@pytest.fixture
def live_recorded_models(monkeypatch):
    monkeypatch.setenv("KEEL_LLM_MODE", "live")
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-live")
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-not-live")
    get_settings.cache_clear()

    def build(profile):
        return RecordedModel(metadata={"router_model": profile.model, "router_profile": profile.id})

    monkeypatch.setattr(factory, "model_factory", build)
    yield
    get_settings.cache_clear()


async def test_stream_meters_the_actual_routed_model_and_cached_tokens(owner, live_recorded_models):
    org, user = await _ctx(owner)
    conversation = str(uuid.uuid4())
    response = await owner.post(
        "/api/agent/ask", json={"message": "Compare order totals", "conversation_id": conversation}
    )
    assert response.status_code == 200 and '"type": "error"' not in response.text
    decision = Router(profiles()).select(route_request("Compare order totals", 1000))
    async with tenant_session(org, user) as db:
        rows = (
            await db.execute(
                text(
                    "SELECT model, input_tokens, output_tokens, cost_usd, ref FROM usage_events "
                    "WHERE ref->>'conversation_id'=:conversation"
                ),
                {"conversation": conversation},
            )
        ).all()
        held = await db.scalar(text("SELECT count(*) FROM ai_reservations"))
    assert len(rows) == 1 and held == 0
    model, tin, tout, cost, ref = rows[0]
    assert model == decision.profile.model and (tin, tout) == (100, 10)
    expected = decision.profile.price.cost(Usage(input_tokens=100, output_tokens=10, cached_tokens=20))
    assert cost == expected and ref["cached_tokens"] == 20 and ref["cost_status"] == "known"
    assert ref["router_task_family"] == "reasoning"


async def test_zero_budget_prevents_the_provider_call(owner, live_recorded_models, monkeypatch):
    monkeypatch.setenv("KEEL_MONTHLY_AI_BUDGET_USD", "0")
    get_settings.cache_clear()
    calls: list[Any] = []
    original = RecordedModel._generate

    def generate(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RecordedModel, "_generate", generate)
    response = await owner.post("/api/agent/ask", json={"message": "List orders", "conversation_id": str(uuid.uuid4())})
    assert '"type": "error"' in response.text
    assert calls == []


async def test_unknown_usage_is_estimated_and_settled_once(owner, live_recorded_models):
    from keel.routing.usage import UsageRecorder

    org, user = await _ctx(owner)
    recorder = UsageRecorder(org, user, uuid.uuid4())
    attempt = uuid.uuid4()
    profile = profiles()[0]
    await recorder.on_chat_model_start({}, [[]], run_id=attempt, metadata={"router_model": profile.model})
    await recorder.on_llm_error(TimeoutError(), run_id=attempt)
    await recorder.close()
    async with tenant_session(org, user) as db:
        rows = (
            await db.execute(
                text("SELECT cost_usd, ref FROM usage_events WHERE ref->>'attempt_id'=:id"), {"id": str(attempt)}
            )
        ).all()
    assert len(rows) == 1 and rows[0][0] > Decimal(0) and rows[0][1]["cost_status"] == "estimated"
