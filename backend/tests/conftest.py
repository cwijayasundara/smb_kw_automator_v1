"""Tests run against a real Postgres (keel_test) as the RLS-bound app role, exactly like production."""

import os
import uuid
from collections.abc import AsyncIterator

os.environ.setdefault("KEEL_ENV", "test")
os.environ.setdefault("KEEL_DATABASE_URL", "postgresql+psycopg://keel_app:keel_app@localhost:5432/keel_test")
os.environ.setdefault("KEEL_DATABASE_OWNER_URL", "postgresql+psycopg://keel_owner:keel_owner@localhost:5432/keel_test")
os.environ.setdefault("KEEL_LLM_MODE", "fake")
os.environ.setdefault("KEEL_STORAGE_DIR", "/tmp/keel-test-storage")
# Hermetic: a developer's backend/.env (API keys, tracing) must not leak into tests.
os.environ["OPENAI_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
os.environ["FIREWORKS_API_KEY"] = ""
os.environ["QWEN_API_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_BASE_URL"] = ""

import httpx
import pytest
from asgi_lifespan import LifespanManager
from sqlalchemy import create_engine, text

from keel.__main__ import _migrate
from keel.api.app import create_app
from keel.platform.config import get_settings


@pytest.fixture(scope="session", autouse=True)
def database() -> None:
    _migrate()
    owner = create_engine(get_settings().database_owner_url)
    with owner.begin() as conn:
        tables = (
            conn.execute(
                text(
                    "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename <> 'alembic_version'"
                    " AND tablename NOT LIKE 'checkpoint_migrations'"
                )
            )
            .scalars()
            .all()
        )
        conn.execute(text("TRUNCATE " + ", ".join(tables) + " CASCADE"))
    owner.dispose()


@pytest.fixture(scope="session")
async def app() -> AsyncIterator[object]:
    application = create_app()
    async with LifespanManager(application):
        yield application


def _client(app: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")  # type: ignore[arg-type]


@pytest.fixture
async def anon(app: object) -> AsyncIterator[httpx.AsyncClient]:
    async with _client(app) as c:
        yield c


async def signup(app: object, org: str = "Acme Kulfi") -> httpx.AsyncClient:
    client = _client(app)
    email = f"{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post(
        "/api/auth/signup", json={"org_name": org, "name": "Owner", "email": email, "password": "correct-horse-battery"}
    )
    assert r.status_code == 200, r.text
    client.email = email  # type: ignore[attr-defined]
    return client


@pytest.fixture
async def owner(app: object) -> AsyncIterator[httpx.AsyncClient]:
    c = await signup(app)
    yield c
    await c.aclose()


@pytest.fixture
async def other_tenant(app: object) -> AsyncIterator[httpx.AsyncClient]:
    c = await signup(app, "Other Bakery")
    yield c
    await c.aclose()
