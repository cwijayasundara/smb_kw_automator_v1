"""Ask Keel streaming endpoint (Server-Sent Events): tokens, tool calls and tool results as they happen."""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from pydantic import BaseModel, Field

from keel.agents import factory
from keel.agents.factory import ModelPool, agent_input, build_agent, thread
from keel.api.deps import Viewer
from keel.audit.service import record_usage
from keel.identity.service import Ctx
from keel.platform.config import get_settings
from keel.platform.db import tenant_session
from keel.platform.logging import log
from keel.platform.tracing import trace_config
from keel.routing.usage import UsageRecorder

router = APIRouter(tags=["agent"])


class AskIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: uuid.UUID


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


async def _stream(ctx: Ctx, body: AskIn) -> AsyncIterator[str]:
    pool = ModelPool(lambda profile: factory.model_factory(profile))
    config = thread(ctx.org_id, ctx.user_id, body.conversation_id) | trace_config(
        "ask-keel", ctx.org_id, ctx.user_id, str(body.conversation_id)
    )
    recorder = UsageRecorder(ctx.org_id, ctx.user_id, body.conversation_id)
    config["callbacks"] = [*config.get("callbacks", []), recorder]
    streamed_text = False
    try:
        agent = await build_agent(ctx.org_id, ctx.user_id, pool)
        async for mode, chunk in agent.astream(
            await agent_input(ctx.org_id, ctx.user_id, body.message), config, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                msg, meta = chunk
                if isinstance(msg, AIMessageChunk) and meta.get("langgraph_node") == "model":
                    text = _text(msg.content)
                    if text:
                        streamed_text = True
                        yield _sse({"type": "token", "text": text})
                continue
            for update in (chunk or {}).values():
                for m in (update or {}).get("messages", []) if isinstance(update, dict) else []:
                    if isinstance(m, AIMessage):
                        for tc in m.tool_calls:
                            yield _sse({"type": "tool", "name": tc["name"], "args": tc["args"]})
                        if not streamed_text and _text(m.content):
                            yield _sse({"type": "token", "text": _text(m.content)})
                    elif isinstance(m, ToolMessage):
                        yield _sse({"type": "tool_result", "name": m.name, "preview": str(m.content)[:400]})
    except Exception as exc:  # surface a readable error to the chat instead of a broken stream
        log.error("agent.failed", error_type=type(exc).__name__, status_code=getattr(exc, "status_code", None))
        yield _sse({"type": "error", "message": "Keel could not answer that. Try rephrasing."})
    finally:
        try:
            await recorder.close()
        finally:
            await pool.close()
    if not get_settings().live_llm:
        async with tenant_session(ctx.org_id, ctx.user_id) as db:
            await record_usage(db, ctx.org_id, "ask", model="offline")
    yield _sse({"type": "done"})


@router.post("/agent/ask")
async def ask(body: AskIn, ctx: Ctx = Viewer) -> StreamingResponse:
    return StreamingResponse(
        _stream(ctx, body),
        media_type="text/event-stream",
        # no-transform: proxies (Next's gzip included) must not compress, which buffers the whole stream
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
