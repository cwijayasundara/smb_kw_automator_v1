"""Async LangChain middleware. Credentials and task policy belong to the host."""

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, NotRequired, cast

from langchain.agents.middleware import AgentMiddleware, AgentState, ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from harness_model_router import ModelProfile, Router, RouteRequest


class RoutingState(AgentState):
    router_profile: NotRequired[str]
    router_task: NotRequired[str]
    router_upgrades: NotRequired[int]
    router_floor: NotRequired[int]
    router_attempts: NotRequired[int]
    router_failure_marker: NotRequired[str]


class StreamBoundary(AsyncCallbackHandler):
    """Even a tool/reasoning delta commits this attempt: don't mix two streams."""

    def __init__(self) -> None:
        self.started = False

    async def on_llm_new_token(self, token: Any, **kwargs: Any) -> None:
        self.started = True


class ModelRouterMiddleware(AgentMiddleware[RoutingState, Any, Any]):
    state_schema = RoutingState

    def __init__(
        self,
        router: Router,
        factory: Callable[[ModelProfile], BaseChatModel],
        request_builder: Callable[[str, int], RouteRequest],
        *,
        max_fallbacks: int = 2,
        max_upgrades: int = 2,
        shadow: bool = False,
        max_calls: int = 8,
    ) -> None:
        self.router = router
        self.factory = factory
        self.request_builder = request_builder
        self.max_fallbacks = max_fallbacks
        self.max_upgrades = max_upgrades
        self.shadow = shadow
        self.max_calls = max_calls

    def _request(self, request: ModelRequest[Any]) -> tuple[RouteRequest, str]:
        human = next((m for m in reversed(request.messages) if isinstance(m, HumanMessage)), None)
        task = human.text if human else ""
        # Conservative text estimate includes tools and the dynamically assembled system prompt.
        content = json.dumps([m.model_dump(mode="json") for m in request.messages], default=str)
        tools = [getattr(t, "args", str(t)) for t in request.tools]
        content += str(request.system_message) + json.dumps(tools, default=str)
        requirements = self.request_builder(task, (len(content) + 1) // 2)
        image_types = {"image", "image_url", "input_image"}
        if any(
            isinstance(m.content, list)
            and any(isinstance(block, dict) and block.get("type") in image_types for block in m.content)
            for m in request.messages
        ):
            requirements = requirements.model_copy(update={"capabilities": requirements.capabilities | {"image"}})
        # A new user turn with identical text must still reset the per-task call limit.
        identity = task + (human.id or "" if human else "")
        return requirements, hashlib.sha256(identity.encode()).hexdigest()

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ExtendedModelResponse[Any]:
        requirements, task_hash = self._request(request)
        state = cast(RoutingState, request.state)
        same = state.get("router_task") == task_hash
        preferred = state.get("router_profile") if same else None
        upgrades = state.get("router_upgrades", 0) if same else 0
        calls = state.get("router_attempts", 0) if same else 0
        failure_marker = state.get("router_failure_marker", "") if same else ""
        floor = max(requirements.min_tier, state.get("router_floor", 1)) if same else requirements.min_tier
        # Two consecutive tool errors are concrete failure evidence. Empty retrieval isn't an error.
        recent = []
        for message in reversed(request.messages):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, ToolMessage):
                recent.append(message)
                if len(recent) == 2:
                    break
        marker = ":".join(str(m.id or m.tool_call_id) for m in recent)
        if (
            preferred in self.router.profiles
            and len(recent) == 2
            and all(m.status == "error" for m in recent)
            and upgrades < self.max_upgrades
            and marker != failure_marker
        ):
            current = self.router.profiles[preferred].tier
            if current < 3:
                floor = max(floor, current + 1)
                upgrades += 1
                preferred = None
                failure_marker = marker
        requirements = requirements.model_copy(update={"min_tier": floor})
        excluded: set[str] = set()
        for attempt in range(self.max_fallbacks + 1):
            if calls + attempt >= self.max_calls:
                raise ValueError("Model call limit exhausted, including fallback attempts")
            requirements = requirements.model_copy(update={"excluded": frozenset(excluded)})
            decision = self.router.select(requirements, preferred=preferred)
            profile = decision.profile
            boundary = StreamBoundary()
            model = request.model if self.shadow else self.factory(profile)
            metadata = dict(model.metadata or {}) | {
                "router_reason": decision.reason,
                "router_policy_version": decision.policy_version,
                "router_task_family": requirements.task,
            }
            if self.shadow:
                metadata["router_shadow_profile"] = profile.id
            model = model.model_copy(update={"callbacks": [boundary], "metadata": metadata})
            try:
                response = await handler(request.override(model=model))
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                transient = (
                    isinstance(exc, (TimeoutError, ConnectionError))
                    or type(exc).__name__ in {"APITimeoutError", "APIConnectionError", "ConnectError", "ReadTimeout"}
                    or status in {401, 403, 429, 500, 502, 503, 504}
                )
                if self.shadow or boundary.started or not transient or attempt >= self.max_fallbacks:
                    raise
                excluded.add(profile.id)
                if status in {401, 403}:
                    excluded.update(p.id for p in self.router.profiles.values() if p.provider == profile.provider)
                # Availability fallback keeps the task floor, not the failed model's tier.
                preferred = None
                continue
            return ExtendedModelResponse(
                model_response=response,
                command=Command(
                    update={
                        "router_profile": profile.id,
                        "router_task": task_hash,
                        "router_floor": floor,
                        "router_upgrades": upgrades,
                        "router_attempts": calls + attempt + 1,
                        "router_failure_marker": failure_marker,
                    }
                ),
            )
        raise RuntimeError("Routing attempt limit exhausted")
