"""Ask Keel: a deepagents analyst with skills, tenant memory and read-only, tenant-bound tools.

- Skills (`/skills/`) and shared rules (`/shared/`) are mounted read-only from the repo.
- The tenant profile is rendered into `/tenant/AGENTS.md` and loaded as memory.
- Writes are denied on every path: Ask Keel explains and finds; it never changes records.
- Conversation state lives in the Postgres checkpointer under a thread id the server builds.
"""

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.utils import create_file_data
from deepagents.middleware.filesystem import FilesystemPermission
from harness_model_router import ModelProfile
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from sqlalchemy import select

from keel.agents.offline import OfflineModel
from keel.agents.tools import build_tools
from keel.domain.models import Customer, Product
from keel.identity.models import Org, TenantProfile
from keel.platform.config import get_settings
from keel.platform.db import tenant_session
from keel.routing.profiles import initial_profile, profiles, route_request
from keel.routing.profiles import model_factory as model_factory
from keel.workflows.runtime import checkpointer

ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = ROOT / "skills"
SHARED_DIR = ROOT / "shared"

SYSTEM = (
    "You are Keel, the back-office assistant for a small food producer. You answer the owner's questions "
    "from Keel's records using your tools. Read /shared/rules.md once if you are unsure about a rule.\n\n"
    + (SHARED_DIR / "rules.md").read_text()
)


class ModelPool:
    """Per-request clients: reused across calls and closed when the stream finishes."""

    def __init__(self, factory: Callable[[ModelProfile], BaseChatModel]) -> None:
        self.factory = factory
        self.models: dict[str, BaseChatModel] = {}

    def get(self, profile: ModelProfile) -> BaseChatModel:
        if profile.id not in self.models:
            self.models[profile.id] = self.factory(profile)
        return self.models[profile.id]

    async def close(self) -> None:
        for model in self.models.values():
            closer = getattr(model, "aclose", None)
            if closer is not None:
                await closer()
            elif (client := getattr(model, "root_async_client", None)) is not None:
                await client.close()
            sync_closer = getattr(model, "close", None)
            if sync_closer is not None:
                sync_closer()
            elif (client := getattr(model, "root_client", None)) is not None:
                client.close()


def chat_model(pool: ModelPool | None = None) -> Any:
    s = get_settings()
    if not s.live_llm:
        return OfflineModel()
    return pool.get(initial_profile()) if pool else model_factory(initial_profile())


def agent_middleware(pool: ModelPool | None = None) -> list[Any]:
    from harness_model_router import Router
    from harness_model_router.adapters.langchain import ModelRouterMiddleware

    s = get_settings()
    middleware: list[Any] = [ModelCallLimitMiddleware(run_limit=8), ToolCallLimitMiddleware(run_limit=10)]
    if s.live_llm and s.router_mode in {"enabled", "shadow"}:
        middleware.append(
            ModelRouterMiddleware(
                Router(profiles(s)),
                pool.get if pool else model_factory,
                route_request,
                max_fallbacks=s.router_max_fallbacks,
                max_upgrades=s.router_max_upgrades,
                shadow=s.router_mode == "shadow",
            )
        )
    return middleware


async def tenant_memory(org_id: uuid.UUID, user_id: uuid.UUID) -> str:
    async with tenant_session(org_id, user_id) as db:
        org = await db.get(Org, org_id)
        profile = await db.get(TenantProfile, org_id)
        customers = (await db.scalars(select(Customer.name).order_by(Customer.name).limit(40))).all()
        products = (
            await db.scalars(select(Product).where(Product.active.is_(True)).order_by(Product.name).limit(60))
        ).all()
    assert org is not None
    lines = [
        "## Business context",
        f"- Business: {org.name}",
        f"- Country: {org.country}; currency: {org.currency}; timezone: {org.timezone}",
        f"- Customers: {', '.join(customers) or 'none yet'}",
        f"- Products: {', '.join(f'{p.name} ({p.sku}, {p.unit})' for p in products) or 'none yet'}",
    ]
    for k, v in (profile.profile if profile else {}).items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


async def build_agent(org_id: uuid.UUID, user_id: uuid.UUID, pool: ModelPool | None = None) -> Any:
    tools = build_tools(org_id, user_id)
    model = chat_model(pool)
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True),
            "/shared/": FilesystemBackend(root_dir=SHARED_DIR, virtual_mode=True),
        },
    )
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM,
        skills=["/skills/"],
        memory=["/tenant/AGENTS.md"],
        permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
        backend=backend,
        middleware=cast(Any, agent_middleware(pool)),
        subagents=cast(
            Any,
            [
                {
                    "name": "general-purpose",
                    "description": "Read-only analysis of Keel records",
                    "model": model,
                    "tools": tools,
                    "system_prompt": SYSTEM,
                    "skills": ["/skills/"],
                    "memory": ["/tenant/AGENTS.md"],
                    "middleware": agent_middleware(pool),
                }
            ],
        ),
        checkpointer=await checkpointer(),
        name="ask-keel",
    )


async def agent_input(org_id: uuid.UUID, user_id: uuid.UUID, message: str) -> dict[str, Any]:
    memory = await tenant_memory(org_id, user_id)
    return {
        "messages": [{"role": "user", "content": message}],
        "files": {"/tenant/AGENTS.md": create_file_data(memory)},
    }


def thread(org_id: uuid.UUID, user_id: uuid.UUID, conversation_id: uuid.UUID) -> dict[str, Any]:
    # The server builds the thread id from the verified session, so a client can't open someone else's thread.
    return {"configurable": {"thread_id": f"ask:{org_id}:{user_id}:{conversation_id}"}, "recursion_limit": 40}
