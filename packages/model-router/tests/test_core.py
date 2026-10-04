from decimal import Decimal

import pytest

from harness_model_router import ModelProfile, NoEligibleModel, Price, Router, RouteRequest, Usage


def profile(id="cheap", *, tier=1, price="0.1", **kwargs):
    return ModelProfile(
        id=id,
        provider="test",
        model=id,
        tier=tier,
        tasks=frozenset({"routine", "reasoning"}),
        context_window=10000,
        price=Price(input=price, cached_input="0.01", output=price),
        **kwargs,
    )


def request(**kwargs):
    return RouteRequest(task="routine", available_providers=frozenset({"test"}), input_tokens=1000, **kwargs)


def test_cheapest_model_not_list_order():
    router = Router([profile("expensive", price="2"), profile()])
    assert router.select(request()).profile.id == "cheap"


def test_quality_floor_overrides_cost():
    router = Router([profile(), profile("strong", tier=2, price="2")])
    assert router.select(request(min_tier=2)).profile.id == "strong"


@pytest.mark.parametrize(
    "update",
    [
        {"enabled": False},
        {"price": None},
        {"tasks": frozenset({"other"})},
        {"capabilities": frozenset({"text"})},
        {"context_window": 1500},
        {"max_output": 100},
    ],
)
def test_ineligible_models_are_never_selected(update):
    candidate = profile().model_copy(update=update)
    with pytest.raises(NoEligibleModel):
        Router([candidate]).select(request())


def test_access_budget_and_exclusions():
    router = Router([profile()])
    for update in [{"available_providers": frozenset()}, {"budget_usd": Decimal(0)}, {"excluded": {"cheap"}}]:
        with pytest.raises(NoEligibleModel):
            router.select(request().model_copy(update=update))


def test_sticky_profile_is_revalidated():
    router = Router([profile(), profile("strong", tier=2, price="2")])
    assert router.select(request(), preferred="strong").reason == "continue_task"
    assert router.select(request(min_tier=2), preferred="cheap").profile.id == "strong"


def test_cache_cannot_be_transferred_between_models():
    router = Router([profile(), profile("peer", price="0.2")])
    decision = router.select(request(output_tokens=1, cached_tokens=1000, cached_profile="peer"))
    assert decision.profile.id == "peer"
    assert decision.estimated_cost == Decimal("0.0000102")


def test_decimal_cost_and_long_context_rates():
    price = Price(
        input="2",
        cached_input="0.2",
        output="10",
        long_context_threshold=100,
        long_input_multiplier="2",
        long_output_multiplier="1.5",
    )
    assert price.cost(Usage(input_tokens=200, cached_tokens=100, output_tokens=10)) == Decimal("0.00059")
    with pytest.raises(ValueError):
        Usage(input_tokens=1, cached_tokens=2)


def test_duplicate_aliases_rejected():
    with pytest.raises(ValueError, match="unique"):
        Router([profile(), profile()])
