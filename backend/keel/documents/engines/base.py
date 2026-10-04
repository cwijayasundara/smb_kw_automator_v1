"""Provider-agnostic extraction interface. Every engine returns the same shape plus cost."""

from dataclasses import dataclass
from typing import Any, Protocol

from harness_model_router import Usage
from pydantic import BaseModel

from keel.platform.config import get_settings


@dataclass
class PageInput:
    png: bytes
    text: str  # reading-order text from the text layer or OCR
    words: list[dict[str, Any]]


@dataclass
class EngineResult:
    data: BaseModel
    engine: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Extractor(Protocol):
    name: str

    async def extract(self, pages: list[PageInput], schema: type[BaseModel], hint: str = "") -> EngineResult: ...


def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model == "offline":
        return 0.0
    price = get_settings().model_prices.get(model)
    if price is None:
        raise ValueError("Model price is unknown; configure it in the model registry")
    return float(price.cost(Usage(input_tokens=input_tokens, output_tokens=output_tokens)))
