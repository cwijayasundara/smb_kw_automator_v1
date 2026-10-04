"""Selection is deterministic; hosts supply task requirements and quality eligibility."""

from decimal import Decimal
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MILLION = Decimal(1_000_000)


class Usage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_cache(self) -> Self:
        if self.cached_tokens > self.input_tokens:
            raise ValueError("cached tokens cannot exceed input tokens")
        return self


class Price(BaseModel):
    """USD per million tokens. An absent price is unknown, never free."""

    input: Decimal = Field(ge=0)
    output: Decimal = Field(ge=0)
    cached_input: Decimal = Field(ge=0)
    version: str = "2026-10-04"
    long_context_threshold: int | None = Field(default=None, gt=0)
    long_input_multiplier: Decimal = Field(default=Decimal(1), ge=1)
    long_output_multiplier: Decimal = Field(default=Decimal(1), ge=1)

    def cost(self, usage: Usage) -> Decimal:
        long = self.long_context_threshold is not None and usage.input_tokens > self.long_context_threshold
        pin = self.long_input_multiplier if long else Decimal(1)
        pout = self.long_output_multiplier if long else Decimal(1)
        return (
            (usage.input_tokens - usage.cached_tokens) * self.input * pin
            + usage.cached_tokens * self.cached_input * pin
            + usage.output_tokens * self.output * pout
        ) / MILLION


class ModelProfile(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    tier: int = Field(default=1, ge=1, le=3)
    tasks: frozenset[str]
    capabilities: frozenset[str] = frozenset({"text", "tools"})
    context_window: int = Field(gt=0)
    max_output: int = Field(default=4096, gt=0)
    price: Price | None = None
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict, alias="kwargs")


class RouteRequest(BaseModel):
    task: str
    min_tier: int = Field(default=1, ge=1, le=3)
    capabilities: frozenset[str] = frozenset({"text", "tools"})
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=2048, gt=0)
    cached_tokens: int = Field(default=0, ge=0)
    cached_profile: str | None = None
    available_providers: frozenset[str]
    excluded: frozenset[str] = frozenset()
    budget_usd: Decimal | None = Field(default=None, ge=0)


class RouteDecision(BaseModel):
    profile: ModelProfile
    estimated_cost: Decimal
    reason: str = "cheapest_eligible"
    policy_version: str = "cost-first-v1"


class NoEligibleModel(ValueError):
    """No priced, available profile satisfies this request."""


class Router:
    def __init__(self, profiles: list[ModelProfile]) -> None:
        self.profiles = {p.id: p for p in profiles}
        if len(self.profiles) != len(profiles):
            raise ValueError("profile IDs must be unique")

    def decision(self, profile: ModelProfile, request: RouteRequest) -> RouteDecision | None:
        if (
            not profile.enabled
            or profile.id in request.excluded
            or profile.provider not in request.available_providers
            or profile.tier < request.min_tier
            or request.task not in profile.tasks
            or not request.capabilities <= profile.capabilities
            or request.input_tokens + request.output_tokens > profile.context_window
            or request.output_tokens > profile.max_output
            or profile.price is None
        ):
            return None
        cached = min(request.cached_tokens, request.input_tokens) if request.cached_profile == profile.id else 0
        estimate = profile.price.cost(
            Usage(input_tokens=request.input_tokens, output_tokens=request.output_tokens, cached_tokens=cached)
        )
        if request.budget_usd is not None and estimate > request.budget_usd:
            return None
        return RouteDecision(profile=profile, estimated_cost=estimate)

    def select(self, request: RouteRequest, *, preferred: str | None = None) -> RouteDecision:
        if preferred is not None and preferred in self.profiles:
            sticky = self.decision(self.profiles[preferred], request)
            if sticky is not None:
                return sticky.model_copy(update={"reason": "continue_task"})
        candidates = [d for p in self.profiles.values() if (d := self.decision(p, request)) is not None]
        if not candidates:
            raise NoEligibleModel(
                "No priced available model meets the task, capability, context and budget requirements"
            )
        return min(candidates, key=lambda d: (d.estimated_cost, d.profile.tier, d.profile.id))
