from decimal import Decimal

import pytest

from slmstrict import FallbackConfig, StrictConfig
from slmstrict.ladder import Ladder
from slmstrict.metrics import (
    MetricsCollector,
    ModelKey,
    ModelPrice,
    PriceTable,
    UsageTotals,
    estimate_cost,
)
from slmstrict.testing import FakeBackend
from tests._ladder_support import GARBAGE, VALID
from tests.metrics._scenario import (
    CLOUD_MODEL,
    CLOUD_PROVIDER,
    SCHEMA,
    cloud_handler,
    run_known_scenario,
)

KEY = ModelKey(CLOUD_PROVIDER, CLOUD_MODEL)


async def test_cost_is_none_without_a_price_table() -> None:
    collector = MetricsCollector()
    await run_known_scenario(collector)
    estimate = collector.cost()
    assert estimate.total is None
    assert estimate.known is False
    assert estimate.currency is None
    assert dict(estimate.by_model) == {KEY: None}


async def test_cost_matches_the_hand_computed_value() -> None:
    # The scenario's cloud call: 1200 input tokens (1000 of them cached) and 300 output tokens.
    # fresh input 200 * 3.00 + cached 1000 * 0.30 + output 300 * 15.00, per million tokens:
    # (600 + 300 + 4500) / 1_000_000 = 0.0054
    table = PriceTable({(CLOUD_PROVIDER, CLOUD_MODEL): ModelPrice(3, 15, "0.30")})
    collector = MetricsCollector(price_table=table)
    await run_known_scenario(collector)
    estimate = collector.cost()
    assert estimate.total == Decimal("0.0054")
    assert estimate.by_model[KEY] == Decimal("0.0054")
    assert estimate.currency == "USD"
    assert estimate.known


async def test_cached_tokens_are_billed_as_input_without_a_cache_price() -> None:
    # 1200 * 3.00 + 300 * 15.00 = 8100 per million tokens
    table = PriceTable({(CLOUD_PROVIDER, CLOUD_MODEL): ModelPrice(3, 15)}, currency="EUR")
    collector = MetricsCollector()
    await run_known_scenario(collector)
    estimate = collector.cost(table)
    assert estimate.total == Decimal("0.0081")
    assert estimate.currency == "EUR"


async def test_a_model_missing_from_the_table_makes_the_cost_unknown() -> None:
    table = PriceTable({("other", "other-model"): ModelPrice(1, 1)})
    collector = MetricsCollector(price_table=table)
    await run_known_scenario(collector)
    estimate = collector.cost()
    assert estimate.total is None
    assert estimate.by_model[KEY] is None
    assert estimate.currency == "USD"


async def test_a_fallback_without_usage_makes_the_cost_unknown() -> None:
    table = PriceTable({(CLOUD_PROVIDER, CLOUD_MODEL): ModelPrice(3, 15)})
    collector = MetricsCollector(price_table=table)
    config = StrictConfig(
        observer=collector, fallback=FallbackConfig(handler=cloud_handler(usage=None))
    )
    await Ladder(FakeBackend([GARBAGE] * 3), config).run(SCHEMA, "p")
    snapshot = collector.snapshot()
    assert snapshot.fallback_used == 1
    assert snapshot.calls_without_usage == 1
    assert collector.cost().total is None
    assert collector.cost().calls_without_usage == 1


async def test_cost_with_a_table_and_no_fallback_calls_is_zero() -> None:
    table = PriceTable({(CLOUD_PROVIDER, CLOUD_MODEL): ModelPrice(3, 15)})
    collector = MetricsCollector(price_table=table)
    await Ladder(FakeBackend([VALID]), StrictConfig(observer=collector)).run(SCHEMA, "p")
    assert collector.cost().total == Decimal(0)


def test_models_are_priced_independently_and_summed() -> None:
    table = PriceTable(
        {
            ("a", "small"): ModelPrice("1.5", "2"),
            ("b", "big"): ModelPrice(10, 30),
        }
    )
    usage = {
        ModelKey("a", "small"): UsageTotals(input_tokens=2_000_000, output_tokens=500_000),
        ModelKey("b", "big"): UsageTotals(input_tokens=100_000, output_tokens=10_000),
    }
    estimate = estimate_cost(usage, table)
    assert estimate.by_model[ModelKey("a", "small")] == Decimal("4.0")  # 3.0 + 1.0
    assert estimate.by_model[ModelKey("b", "big")] == Decimal("1.3")  # 1.0 + 0.3
    assert estimate.total == Decimal("5.3")


def test_unreported_provider_or_model_cannot_be_priced() -> None:
    table = PriceTable({("a", "m"): ModelPrice(1, 1)})
    assert table.lookup(ModelKey(None, None)) is None
    assert table.lookup(ModelKey("a", None)) is None
    estimate = estimate_cost({ModelKey(None, None): UsageTotals(input_tokens=5)}, table)
    assert estimate.total is None


def test_price_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ModelPrice(-1, 1)
    with pytest.raises(ValueError, match="finite"):
        ModelPrice(float("nan"), 1)
    with pytest.raises(ValueError, match="not a valid number"):
        ModelPrice("cheap", 1)
    with pytest.raises(TypeError, match="number"):
        ModelPrice(True, 1)
    with pytest.raises(TypeError, match="ModelPrice"):
        PriceTable({("a", "m"): 3})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="provider, model"):
        PriceTable({"a": ModelPrice(1, 1)})  # type: ignore[dict-item]


def test_price_table_is_detached_from_its_input() -> None:
    source = {("a", "m"): ModelPrice(1, 1)}
    table = PriceTable(source)
    source.clear()
    assert table.lookup(ModelKey("a", "m")) is not None
    with pytest.raises(TypeError):
        table.prices[("x", "y")] = ModelPrice(1, 1)  # type: ignore[index]
