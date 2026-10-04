"""Runtime configuration. Every tunable (model IDs, thresholds, limits) lives here, never in code."""

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from harness_model_router import Price
from pydantic import Field, PrivateAttr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from keel.platform.model_registry import ModelRegistry, load_registry


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KEEL_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"

    # Runtime connects as a role WITHOUT BYPASSRLS; migrations run as the owner role.
    database_url: str = "postgresql+psycopg://keel_app:keel_app@localhost:5432/keel"
    database_owner_url: str = "postgresql+psycopg://keel_owner:keel_owner@localhost:5432/keel"

    storage_backend: Literal["local", "gcs"] = "local"
    storage_dir: Path = Path("./var/storage")  # local backend
    gcs_bucket: str | None = None  # gcs backend; credentials come from the runtime service account
    gcs_endpoint: str = "https://storage.googleapis.com"
    frontend_url: str = "http://localhost:3000"

    session_cookie: str = "keel_session"
    session_ttl_days: int = 30
    cookie_secure: bool = False
    magic_link_ttl_minutes: int = 20
    invite_ttl_days: int = 7

    # "fake" runs a deterministic local extractor and agent (no API keys, used by tests/demo).
    # "live" calls providers. "auto" picks live when a supported provider key is set.
    llm_mode: Literal["auto", "fake", "live"] = "auto"
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    fireworks_api_key: str | None = Field(default=None, validation_alias="FIREWORKS_API_KEY")
    qwen_api_key: str | None = Field(default=None, validation_alias="QWEN_API_KEY")

    model_registry_path: Path = Path(__file__).resolve().parents[1] / "models.yml"
    model_default: str = ""
    model_parser: str = ""
    model_vision_handwriting: str = ""
    model_reasoning: str = ""
    model_fallback: str = ""
    model_luna: str = ""
    model_glm_flash: str = ""
    model_deepseek_flash: str = ""
    model_balanced: str = ""
    model_qwen: str | None = None  # exact hosted/self-hosted deployment; needs a configured price
    qwen_base_url: str | None = None
    router_mode: Literal["off", "shadow", "enabled"] = "enabled"
    router_max_output_tokens: int = Field(default=4096, ge=256, le=16384)
    router_max_fallbacks: int = Field(default=2, ge=0, le=2)
    router_max_upgrades: int = Field(default=2, ge=0, le=2)
    router_timeout_seconds: float = Field(default=60, gt=0, le=300)
    model_prices: dict[str, Price] = Field(default_factory=dict)
    _registry: ModelRegistry = PrivateAttr()

    # Tier 4: only for pages that still fail checks after the Gemini re-read. Off unless named here;
    # which one is decided per document type by `keel evals parsers`.
    tier4_engine: Literal["reducto", "ade", "sol"] | None = None
    tier4_confidence: float = 0.85  # services return no per-field confidence; this is what we assume
    model_tier4_sol: str = ""
    reducto_api_key: str | None = Field(default=None, validation_alias="REDUCTO_API_KEY")
    reducto_url: str = "https://platform.reducto.ai"
    reducto_usd_per_page: float = 0.02  # placeholder: set from your plan; bake-off cost figures use it
    landingai_api_key: str | None = Field(default=None, validation_alias="VISION_AGENT_API_KEY")
    ade_url: str = "https://api.va.landing.ai"
    ade_usd_per_page: float = 0.03  # placeholder: set from your plan

    # Search: full-text + pg_trgm spelling + (live only) OpenAI embeddings, fused by RRF.
    embedding_model: str = ""
    embedding_dims: int = 512
    embedding_usd_per_mtok: Decimal = Decimal("0.02")
    chunk_chars: int = 600
    rrf_k: int = 60
    search_pool: int = 20
    search_min_similarity: float = 0.25  # cosine floor for "similar meaning"
    search_fuzzy: float = 0.5  # pg_trgm word_similarity floor for "close spelling"

    # Optional tracing (self-hosted Langfuse recommended: traces contain document text).
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str | None = Field(default=None, validation_alias="LANGFUSE_BASE_URL")

    ocr_enabled: bool = True
    max_upload_mb: int = 25
    low_confidence: float = 0.75
    unusual_quantity_factor: float = 3.0
    worker_poll_seconds: float = 1.0
    worker_drain_max_seconds: float = 50.0  # `keel worker --drain` stops before the next scheduled run
    monthly_ai_budget_usd: Decimal = Field(default=Decimal("10.0"), ge=0)

    @model_validator(mode="after")
    def registry_defaults(self) -> Self:
        self._registry = load_registry(self.model_registry_path)
        for field, value in self._registry.defaults.items():
            if field not in type(self).model_fields:
                raise ValueError(f"Unknown model default setting: {field}")
            if not getattr(self, field):
                setattr(self, field, value)
        self.model_prices = self._registry.prices | self.model_prices
        for field in self._registry.bindings.values():
            if field not in type(self).model_fields:
                raise ValueError(f"Unknown model binding setting: {field}")
        return self

    @property
    def model_registry(self) -> ModelRegistry:
        return self._registry

    @property
    def live_llm(self) -> bool:
        if self.llm_mode == "fake":
            return False
        if self.llm_mode == "live":
            return True
        return bool(self.openai_api_key or self.google_api_key or self.fireworks_api_key or self.qwen_api_key)

    def api_key(self, provider: str) -> str | None:
        """Keys loaded from .env never reach os.environ, so clients must be handed them explicitly."""
        return {
            "openai": self.openai_api_key,
            "google_genai": self.google_api_key,
            "fireworks": self.fireworks_api_key,
            "qwen": self.qwen_api_key,
        }.get(provider)


@lru_cache
def get_settings() -> Settings:
    return Settings()
