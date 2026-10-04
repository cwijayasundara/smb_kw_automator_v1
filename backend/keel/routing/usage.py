"""Meter all agent model calls, including child agents and summarization, at source."""

import time
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from harness_model_router import Price, Usage
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from keel.platform.config import get_settings
from keel.routing.budget import reserve, settle


@dataclass
class Attempt:
    model: str
    profile: str
    price: Price
    estimate: Decimal
    started: float
    ref: dict[str, Any]


class UsageRecorder(AsyncCallbackHandler):
    raise_error = True  # A rejected reservation must prevent the paid provider call.

    def __init__(self, org_id: uuid.UUID, user_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        self.org_id, self.user_id, self.conversation_id = org_id, user_id, conversation_id
        self.attempts: dict[uuid.UUID, Attempt] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: uuid.UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        s = get_settings()
        if not s.live_llm:
            return
        meta = metadata or {}
        params = kwargs.get("invocation_params", {})
        model = str(meta.get("router_model") or params.get("model") or params.get("model_name") or "")
        price = s.model_prices.get(model)
        if price is None:
            raise ValueError("Model price is unknown; refusing an unmetered paid call")
        size = len(str(messages)) + len(str(params.get("tools", [])))
        output = int(params.get("max_tokens") or params.get("max_output_tokens") or s.router_max_output_tokens)
        estimate = price.cost(Usage(input_tokens=(size + 1) // 2, output_tokens=output))
        await reserve(self.org_id, self.user_id, run_id, estimate)
        self.attempts[run_id] = Attempt(
            model,
            str(meta.get("router_profile", "auxiliary")),
            price,
            estimate,
            time.monotonic(),
            {
                k: meta[k]
                for k in ("router_reason", "router_policy_version", "router_task_family", "router_shadow_profile")
                if k in meta
            },
        )

    async def _finish(self, run_id: uuid.UUID, response: LLMResult | None, error: bool) -> None:
        attempt = self.attempts.get(run_id)
        if attempt is None:
            return
        raw: dict[str, Any] = {}
        if response:
            for generations in response.generations:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    if message is not None and message.usage_metadata:
                        raw = dict(message.usage_metadata)
                        break
        known = bool(raw)
        usage = Usage(
            input_tokens=int(raw.get("input_tokens", 0)),
            output_tokens=int(raw.get("output_tokens", 0)),
            cached_tokens=min(
                int((raw.get("input_token_details") or {}).get("cache_read", 0)), int(raw.get("input_tokens", 0))
            ),
        )
        # Provider failures/cancelled streams can have incurred spend without returned usage.
        # Keep the reservation estimate as explicitly estimated spend, never report it as free.
        await settle(
            self.org_id,
            self.user_id,
            run_id,
            model=attempt.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=attempt.price.cost(usage) if known else attempt.estimate,
            profile=attempt.profile,
            conversation_id=str(self.conversation_id),
            cached_tokens=usage.cached_tokens,
            price_version=attempt.price.version,
            cost_status="known" if known else "estimated",
            outcome="error" if error else "completed",
            latency_ms=int((time.monotonic() - attempt.started) * 1000),
            **attempt.ref,
        )
        self.attempts.pop(run_id, None)

    async def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        await self._finish(run_id, response, False)

    async def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        await self._finish(run_id, kwargs.get("response"), True)

    async def close(self) -> None:
        for run_id in list(self.attempts):
            await self._finish(run_id, None, True)
