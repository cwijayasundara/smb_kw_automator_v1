from pathlib import Path

import pytest
import yaml
from harness_model_router import NoEligibleModel, Router

from keel.documents.engines.base import cost
from keel.documents.pipeline import default_extractor
from keel.platform.config import Settings, get_settings
from keel.routing.profiles import available_providers, profiles, route_request


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    get_settings.cache_clear()
    yield
    monkeypatch.undo()
    get_settings.cache_clear()


def test_models_and_prices_are_loaded_from_yaml():
    s = Settings(_env_file=None, llm_mode="fake")
    yaml_data = yaml.safe_load(s.model_registry_path.read_text())
    assert s.model_parser == yaml_data["defaults"]["model_parser"]
    assert s.model_prices.keys() == yaml_data["prices"].keys()


def test_arbitrary_model_names_can_be_changed_without_python_edits(tmp_path: Path):
    data = yaml.safe_load(Settings(_env_file=None).model_registry_path.read_text())
    data["defaults"]["model_luna"] = "openai:a-different-model"
    data["prices"]["a-different-model"] = {"input": "0.01", "cached_input": "0.001", "output": "0.01"}
    config = tmp_path / "models.yml"
    config.write_text(yaml.safe_dump(data))
    s = Settings(_env_file=None, model_registry_path=config)
    assert profiles(s)[0].model == "a-different-model"
    assert profiles(s)[0].price.input == s.model_prices["a-different-model"].input


def test_env_override_precedes_yaml(monkeypatch):
    monkeypatch.setenv("KEEL_MODEL_LUNA", "openai:replacement")
    s = Settings(_env_file=None)
    assert profiles(s)[0].model == "replacement"
    assert profiles(s)[0].price is None  # unpriced replacements are excluded


def test_fireworks_only_is_live_and_routes_to_cheapest_eligible(monkeypatch):
    monkeypatch.setenv("KEEL_LLM_MODE", "auto")
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    s = get_settings()
    assert s.live_llm and available_providers(s) == {"fireworks"}
    router = Router(profiles(s))
    for message in ("List orders", "Reconcile these orders"):
        decision = router.select(route_request(message, 1000))
        eligible = [router.decision(p, route_request(message, 1000)) for p in profiles(s)]
        assert decision.estimated_cost == min(d.estimated_cost for d in eligible if d)


def test_fake_mode_ignores_provider_keys(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    assert not get_settings().live_llm


def test_unknown_prices_are_not_zero():
    with pytest.raises(ValueError, match="unknown"):
        cost("unpriced-model", 100, 10)


def test_parser_uses_configured_fixed_model(monkeypatch):
    monkeypatch.setenv("KEEL_LLM_MODE", "live")
    monkeypatch.setenv("KEEL_MODEL_PARSER", "google_genai:configured-parser")
    assert default_extractor().model_id == "google_genai:configured-parser"


def test_unavailable_provider_does_not_fall_back_to_unconfigured_model():
    with pytest.raises(NoEligibleModel):
        Router(profiles()).select(route_request("List orders", 100))


@pytest.mark.asyncio
async def test_request_pools_own_independent_openai_transports(monkeypatch):
    from keel.agents.factory import ModelPool
    from keel.routing.profiles import model_factory

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    profile = next(p for p in profiles() if p.provider == "openai")
    first, second = ModelPool(model_factory), ModelPool(model_factory)
    one, two = first.get(profile), second.get(profile)
    assert first.get(profile) is one
    assert one.root_async_client._client is not two.root_async_client._client
    await first.close()
    assert one.root_async_client.is_closed()
    assert not two.root_async_client.is_closed()
    assert not two.root_client.is_closed()
    await second.close()
