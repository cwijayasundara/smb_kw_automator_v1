"""Embeddings for search: OpenAI text-embedding-3-small at 512 dims when a key is set, none offline.

Every stored vector records the model that made it; search only compares vectors from the same model,
so changing models never mixes incomparable spaces. Offline, search uses words and spelling only.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from keel.platform.config import get_settings


@dataclass(frozen=True)
class Embedded:
    vectors: list[list[float]]
    model: str
    tokens: int
    cost_usd: Decimal


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> Embedded: ...


class OpenAIEmbedder:
    def __init__(self, model: str, dims: int, usd_per_mtok: Decimal) -> None:
        self.model, self.dims, self.price = model, dims, usd_per_mtok
        self.name = f"openai:{model}-{dims}"

    async def embed(self, texts: list[str]) -> Embedded:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=get_settings().api_key("openai"))
        r = await client.embeddings.create(model=self.model, input=texts, dimensions=self.dims)
        tokens = r.usage.total_tokens
        return Embedded([d.embedding for d in r.data], self.name, tokens, self.price * tokens / 1_000_000)


def embedder() -> Embedder | None:
    s = get_settings()
    if s.live_llm and s.openai_api_key:
        return OpenAIEmbedder(s.embedding_model, s.embedding_dims, s.embedding_usd_per_mtok)
    return None


def literal(vector: list[float]) -> str:
    """pgvector text form; avoids a driver adapter dependency."""
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"
