from typing import Any

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from harness_model_router import ModelProfile, Price, Router, RouteRequest
from harness_model_router.adapters.langchain import ModelRouterMiddleware


def middleware(*, shadow=False, max_fallbacks=2):
    profiles = [
        ModelProfile(
            id=f"p{tier}",
            provider="test",
            model=f"model-{tier}",
            tier=tier,
            tasks=frozenset({"routine"}),
            context_window=100000,
            price=Price(input=str(tier), cached_input="0.1", output=str(tier)),
        )
        for tier in (1, 2, 3)
    ]
    return ModelRouterMiddleware(
        Router(profiles),
        lambda p: FakeListChatModel(responses=[p.model], metadata={"router_model": p.model}),
        lambda task, tokens: RouteRequest(task="routine", input_tokens=tokens, available_providers={"test"}),
        shadow=shadow,
        max_fallbacks=max_fallbacks,
    )


def request(messages=None, state=None):
    return ModelRequest(
        model=FakeListChatModel(responses=["baseline"], metadata={"router_model": "baseline"}),
        messages=messages or [HumanMessage("List orders")],
        tools=[],
        state=state or {"messages": []},
        runtime=None,
    )


async def success(req):
    return ModelResponse(result=[AIMessage(content=req.model.metadata["router_model"])])


@pytest.mark.asyncio
async def test_route_persists_and_task_change_reclassifies():
    mw = middleware()
    first = await mw.awrap_model_call(request(), success)
    assert first.command.update["router_profile"] == "p1"
    state = dict(first.command.update)
    state["router_profile"] = "p2"
    continued = await mw.awrap_model_call(request(state=state), success)
    assert continued.command.update["router_profile"] == "p2"
    new = await mw.awrap_model_call(request([HumanMessage("Find customers")], state), success)
    assert new.command.update["router_profile"] == "p1"


@pytest.mark.asyncio
async def test_transport_fallback_is_bounded_and_persists_actual_model():
    calls = []

    async def handler(req):
        calls.append(req.model.metadata["router_model"])
        if len(calls) == 1:
            raise TimeoutError()
        return await success(req)

    result = await middleware().awrap_model_call(request(), handler)
    assert calls == ["model-1", "model-2"]
    assert result.command.update["router_profile"] == "p2"


@pytest.mark.asyncio
async def test_never_retries_after_stream_delta():
    calls = []

    async def handler(req):
        calls.append(req.model)
        await req.model.callbacks[0].on_llm_new_token("partial")
        raise TimeoutError()

    with pytest.raises(TimeoutError):
        await middleware().awrap_model_call(request(), handler)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_budget_rejection_is_not_retried():
    calls = []

    async def handler(req):
        calls.append(req.model)
        raise ValueError("budget exhausted")

    with pytest.raises(ValueError):
        await middleware().awrap_model_call(request(), handler)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_tool_failure_upgrades_but_missing_data_does_not():
    mw = middleware()
    first = await mw.awrap_model_call(request(), success)
    state = first.command.update
    for status, expected in [("error", "p2"), ("success", "p1")]:
        messages = [
            HumanMessage("List orders"),
            ToolMessage("empty", tool_call_id="a", status=status),
            ToolMessage("empty", tool_call_id="b", status=status),
        ]
        result = await mw.awrap_model_call(request(messages, state), success)
        assert result.command.update["router_profile"] == expected


@pytest.mark.asyncio
async def test_shadow_records_candidate_but_executes_baseline():
    seen: list[Any] = []

    async def handler(req):
        seen.append(req.model.metadata)
        return await success(req)

    await middleware(shadow=True).awrap_model_call(request(), handler)
    assert seen[0]["router_model"] == "baseline"
    assert seen[0]["router_shadow_profile"] == "p1"


@pytest.mark.asyncio
async def test_identical_text_in_a_new_user_turn_resets_task_limit():
    mw = middleware()
    first = await mw.awrap_model_call(request([HumanMessage("List orders", id="turn-one")]), success)
    state = dict(first.command.update) | {"router_attempts": 8}
    with pytest.raises(ValueError, match="call limit"):
        await mw.awrap_model_call(request([HumanMessage("List orders", id="turn-one")], state), success)
    second = await mw.awrap_model_call(request([HumanMessage("List orders", id="turn-two")], state), success)
    assert second.command.update["router_attempts"] == 1


@pytest.mark.asyncio
async def test_fallbacks_count_toward_total_call_limit():
    mw = middleware()
    first = await mw.awrap_model_call(request(), success)
    state = dict(first.command.update) | {"router_attempts": 7}
    calls = []

    async def failing(req):
        calls.append(req.model)
        raise TimeoutError()

    with pytest.raises(ValueError, match="call limit"):
        await mw.awrap_model_call(request(state=state), failing)
    assert len(calls) == 1
