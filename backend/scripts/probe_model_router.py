"""Small, opt-in live provider contract probe using synthetic text only.

Run from backend: uv run python -m scripts.probe_model_router --live
No tenant data is sent or database records written. Estimated total spend is capped.
"""

import argparse
import asyncio
import json
import time
from decimal import Decimal
from typing import Literal

from harness_model_router import Usage
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from keel.platform.config import get_settings
from keel.routing.profiles import available_providers, model_factory, profiles


class TaskChoice(BaseModel):
    """Classify a synthetic request."""

    task: Literal["routine", "reasoning"]


async def probe(limit: Decimal) -> None:
    s = get_settings()
    s.router_max_output_tokens = 1024
    s.router_timeout_seconds = 25
    spent = Decimal(0)
    available = available_providers(s)
    for profile in profiles(s):
        if not profile.enabled or profile.price is None or profile.provider not in available:
            continue
        reserve = profile.price.cost(Usage(input_tokens=2000, output_tokens=1024))
        if spent + reserve > limit:
            print(json.dumps({"profile": profile.id, "status": "skipped_budget"}))
            continue
        started = time.monotonic()
        spent += reserve  # retain worst-case estimate even if an error gives no usage
        base = None
        try:
            base = model_factory(profile)
            model = base.bind_tools([TaskChoice], tool_choice=TaskChoice.__name__)
            combined = None
            async for chunk in model.astream(
                [
                    HumanMessage(
                        "Classify 'List the approved orders' as routine or reasoning. "
                        "Call TaskChoice with task='routine'. Do not do any other work."
                    )
                ]
            ):
                combined = chunk if combined is None else combined + chunk
            if combined is None or not combined.tool_calls:
                raise ValueError("No schema-conforming tool call")
            parsed = TaskChoice.model_validate(combined.tool_calls[0]["args"])
            raw = combined.usage_metadata or {}
            usage = Usage(
                input_tokens=int(raw.get("input_tokens", 0)),
                output_tokens=int(raw.get("output_tokens", 0)),
                cached_tokens=int((raw.get("input_token_details") or {}).get("cache_read", 0)),
            )
            print(
                json.dumps(
                    {
                        "profile": profile.id,
                        "model": profile.model,
                        "status": "passed" if parsed.task == "routine" and raw else "failed_contract",
                        "usage_returned": bool(raw),
                        "usd": str(profile.price.cost(usage)) if raw else None,
                        "latency_ms": int((time.monotonic() - started) * 1000),
                    }
                )
            )
        except Exception as exc:
            # Provider exception strings can contain request details; emit only type/status.
            print(
                json.dumps(
                    {
                        "profile": profile.id,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "http_status": getattr(exc, "status_code", None),
                    }
                )
            )
        finally:
            if base is not None:
                closer = getattr(base, "aclose", None)
                if closer is not None:
                    await closer()
                elif (client := getattr(base, "root_async_client", None)) is not None:
                    await client.close()
                sync_closer = getattr(base, "close", None)
                if sync_closer is not None:
                    sync_closer()
                elif (client := getattr(base, "root_client", None)) is not None:
                    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--budget-usd", type=Decimal, default=Decimal("0.10"))
    args = parser.parse_args()
    asyncio.run(probe(args.budget_usd))
