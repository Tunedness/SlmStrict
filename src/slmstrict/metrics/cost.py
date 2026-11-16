"""Cloud fallback cost estimation (PRD section 5, cost calculation).

The package ships no prices. The user supplies a :class:`PriceTable`; without one, or for a model
the table does not list, the cost is reported as ``None``, meaning "unknown". Nothing is guessed.
Token counts come from the provider's own usage report
(:class:`~slmstrict.hooks.TokenUsage` on a :class:`~slmstrict.hooks.FallbackResult`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Final, NamedTuple

__all__ = [
    "CostEstimate",
    "ModelKey",
    "ModelPrice",
    "PriceTable",
    "UsageTotals",
    "estimate_cost",
]

_MILLION: Final = Decimal(1_000_000)

PriceValue = Decimal | int | float | str


class ModelKey(NamedTuple):
    """Identity of a cloud model as a fallback handler reports it."""

    provider: str | None
    model: str | None


def _to_price(name: str, value: PriceValue) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    try:
        price = value if isinstance(value, Decimal) else Decimal(str(value))
    except ArithmeticError as exc:  # decimal.InvalidOperation is an ArithmeticError
        raise ValueError(f"{name} is not a valid number: {value!r}") from exc
    if not price.is_finite() or price < 0:
        raise ValueError(f"{name} must be a finite, non-negative number")
    return price


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Price of one model, per **one million** tokens, in the table's currency.

    ``cached_input_per_million`` is the price of input tokens the provider served from a prompt
    cache. ``None`` means no discount is known, so cached tokens are billed at the input price.
    Values may be given as ``Decimal``, ``int``, ``float`` or ``str``; they are stored as
    ``Decimal`` so the arithmetic is exact.
    """

    input_per_million: Decimal
    output_per_million: Decimal
    cached_input_per_million: Decimal | None = None

    def __init__(
        self,
        input_per_million: PriceValue,
        output_per_million: PriceValue,
        cached_input_per_million: PriceValue | None = None,
    ) -> None:
        object.__setattr__(
            self, "input_per_million", _to_price("input_per_million", input_per_million)
        )
        object.__setattr__(
            self, "output_per_million", _to_price("output_per_million", output_per_million)
        )
        cached = (
            None
            if cached_input_per_million is None
            else _to_price("cached_input_per_million", cached_input_per_million)
        )
        object.__setattr__(self, "cached_input_per_million", cached)

    def cost_of(self, usage: UsageTotals) -> Decimal:
        """Cost of ``usage`` at this price."""
        cached = usage.cached_input_tokens
        if self.cached_input_per_million is None or cached == 0:
            input_cost = Decimal(usage.input_tokens) * self.input_per_million
        else:
            # The cache hit is a subset of the input tokens (see TokenUsage); bill it separately.
            fresh = max(usage.input_tokens - cached, 0)
            input_cost = (
                Decimal(fresh) * self.input_per_million
                + Decimal(cached) * self.cached_input_per_million
            )
        output_cost = Decimal(usage.output_tokens) * self.output_per_million
        return (input_cost + output_cost) / _MILLION


@dataclass(frozen=True, slots=True)
class PriceTable:
    """User-supplied prices, keyed by ``(provider, model)``.

    ``provider`` and ``model`` are matched exactly against what the fallback handler reports in
    its :class:`~slmstrict.hooks.FallbackResult`. ``currency`` is a free label (default ``"USD"``)
    that only travels into the estimate; no conversion takes place.
    """

    prices: Mapping[tuple[str, str], ModelPrice] = field(default_factory=dict)
    currency: str = "USD"

    def __post_init__(self) -> None:
        for key, price in self.prices.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not all(isinstance(part, str) for part in key)
            ):
                raise TypeError("price table keys must be (provider, model) string pairs")
            if not isinstance(price, ModelPrice):
                raise TypeError("price table values must be ModelPrice instances")
        object.__setattr__(self, "prices", MappingProxyType(dict(self.prices)))

    def lookup(self, key: ModelKey) -> ModelPrice | None:
        """The price of ``key``, or ``None`` when the table does not list it."""
        if key.provider is None or key.model is None:
            return None
        return self.prices.get((key.provider, key.model))


@dataclass(frozen=True, slots=True)
class UsageTotals:
    """Summed token counts of one model.

    ``cached_input_tokens`` counts only calls that reported a cache figure; it is a subset of
    ``input_tokens``. ``calls`` is the number of fallback calls that reported usage.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    calls: int = 0


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Estimated cloud cost of the recorded fallback calls.

    ``total`` is ``None`` ("unknown") when there is no price table, when any recorded model has no
    price in it, or when a fallback call came back without a token usage report. ``by_model`` holds
    the per-model figure, ``None`` where that model's price is missing. The estimate never fills
    a gap with a guess.

    ``calls_without_usage`` counts calls that did get an answer (usable or not) but reported no
    token counts. ``failed_calls_without_usage`` counts calls that raised without any usage report
    (provider outage, ...): most of them were never billed, so they do not make ``total`` unknown,
    but the number is exposed so a caller can judge the figure.
    """

    total: Decimal | None
    by_model: Mapping[ModelKey, Decimal | None]
    currency: str | None
    calls_without_usage: int = 0
    failed_calls_without_usage: int = 0

    @property
    def known(self) -> bool:
        """``True`` when ``total`` is a real figure."""
        return self.total is not None


def estimate_cost(
    usage: Mapping[ModelKey, UsageTotals],
    price_table: PriceTable | None,
    *,
    calls_without_usage: int = 0,
    failed_calls_without_usage: int = 0,
) -> CostEstimate:
    """Price ``usage`` with ``price_table``.

    ``calls_without_usage`` is the number of fallback calls whose provider reported no token
    counts; any such call makes the total unknown, because the missing part cannot be priced.
    ``failed_calls_without_usage`` is passed through to the estimate as information only.
    """
    if price_table is None:
        return CostEstimate(
            total=None,
            by_model=MappingProxyType(dict.fromkeys(usage)),
            currency=None,
            calls_without_usage=calls_without_usage,
            failed_calls_without_usage=failed_calls_without_usage,
        )
    by_model: dict[ModelKey, Decimal | None] = {}
    total = Decimal(0)
    complete = calls_without_usage == 0
    for key, totals in usage.items():
        price = price_table.lookup(key)
        if price is None:
            by_model[key] = None
            complete = False
            continue
        cost = price.cost_of(totals)
        by_model[key] = cost
        total += cost
    return CostEstimate(
        total=total if complete else None,
        by_model=MappingProxyType(by_model),
        currency=price_table.currency,
        calls_without_usage=calls_without_usage,
        failed_calls_without_usage=failed_calls_without_usage,
    )
