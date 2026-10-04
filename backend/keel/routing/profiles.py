"""Task eligibility and model construction, configured entirely by Keel settings."""

import re
from typing import Any, cast

import httpx
from harness_model_router import ModelProfile, Router, RouteRequest
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from keel.platform.config import Settings, get_settings

# Bounded, explicit requests can avoid an extra paid classifier call.
SIMPLE = re.compile(r"^(show|list|find|lookup|look up|which|what is|what are|trace lot|summarize|summarise)\b", re.I)
REASONING = re.compile(
    r"\b(why|compare|reconcile|investigate|root cause|analyse|analyze|recommend|plan|optimi[sz]e|explain|forecast)\b",
    re.I,
)


def task_requirements(message: str) -> tuple[str, int]:
    if REASONING.search(message) or not SIMPLE.search(message.strip()):
        return "reasoning", 2
    return "routine", 1


def profiles(settings: Settings | None = None) -> list[ModelProfile]:
    s = settings or get_settings()
    result = []
    for template in s.model_registry.profiles:
        model = template.model
        field = s.model_registry.bindings.get(template.id)
        if field:
            configured = getattr(s, field, None)
            if configured:
                provider, sep, model = str(configured).partition(":")
                if not sep or provider != template.provider or not model:
                    raise ValueError(f"Model override for {template.id} must use the configured YAML provider")
        if template.provider == "qwen" and not s.qwen_base_url:
            continue
        result.append(template.model_copy(update={"model": model, "price": s.model_prices.get(model)}))
    return result


def available_providers(s: Settings | None = None) -> frozenset[str]:
    settings = s or get_settings()
    return frozenset(p.provider for p in settings.model_registry.profiles if settings.api_key(p.provider))


def route_request(message: str, input_tokens: int) -> RouteRequest:
    s = get_settings()
    task, tier = task_requirements(message)
    return RouteRequest(
        task=task,
        min_tier=tier,
        input_tokens=input_tokens,
        output_tokens=s.router_max_output_tokens,
        available_providers=available_providers(s),
    )


def model_factory(profile: ModelProfile) -> BaseChatModel:
    s = get_settings()
    kwargs: dict[str, Any] = dict(profile.parameters)
    kwargs.update(timeout=s.router_timeout_seconds, max_retries=0)
    output_key = "max_output_tokens" if profile.provider == "google_genai" else "max_tokens"
    kwargs[output_key] = min(profile.max_output, s.router_max_output_tokens)
    provider = profile.provider
    if provider == "qwen":
        # Compatibility is explicit; do not send Qwen requests to OpenAI's endpoint.
        provider = "openai"
        kwargs.update(base_url=s.qwen_base_url, use_responses_api=False)
    if provider == "openai":
        # LangChain caches its default transports across model instances. Give
        # this request's pool ownership so closing one pool cannot break another.
        kwargs.update(
            http_client=httpx.Client(timeout=s.router_timeout_seconds),
            http_async_client=httpx.AsyncClient(timeout=s.router_timeout_seconds),
        )
    return cast(
        BaseChatModel,
        init_chat_model(
            profile.model,
            model_provider=provider,
            api_key=s.api_key(profile.provider),
            metadata={"router_profile": profile.id, "router_model": profile.model, "router_provider": profile.provider},
            **kwargs,
        ),
    )


def initial_profile() -> ModelProfile:
    s = get_settings()
    router = Router(profiles(s))
    if s.router_mode == "off" or s.router_mode == "shadow":
        provider, _, model = s.model_default.partition(":")
        existing = next((p for p in router.profiles.values() if p.model == model and p.provider == provider), None)
        if existing is None or not existing.enabled or not s.api_key(provider) or existing.price is None:
            raise ValueError("Configured default model is unavailable or not registered")
        return existing
    return router.select(
        RouteRequest(
            task="routine", available_providers=available_providers(s), output_tokens=s.router_max_output_tokens
        )
    ).profile
