"""In-process metrics collector (PRD section 5, measurement).

:class:`MetricsCollector` is a :class:`~slmstrict.hooks.LadderObserver`: register it as
``StrictConfig(observer=collector)`` and read :meth:`MetricsCollector.snapshot` whenever the
numbers are needed. It keeps counters only; it never stores prompts, outputs or any other content.

Everything is counted when a run ends (``on_finish``), from the run's own ``path`` and
``attempts``, so the counters of one run always move together and the rates stay consistent.
Runs that end by cancellation, or by an infrastructure error (see :func:`is_errored_run`), say
nothing about schema conformance and are counted separately; they stay out of ``runs`` and out of
every rate.
"""

from __future__ import annotations

import asyncio
import bisect
import itertools
import math
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeVar

from slmstrict.errors import ValidationFailed
from slmstrict.hooks import BaseLadderObserver, FallbackEvent, FallbackStatus, LadderFinished
from slmstrict.metrics.cost import (
    CostEstimate,
    ModelKey,
    PriceTable,
    UsageTotals,
    estimate_cost,
)
from slmstrict.result import Step

__all__ = [
    "DEFAULT_LATENCY_BUCKETS_S",
    "LatencyHistogram",
    "MetricsCollector",
    "MetricsSnapshot",
    "final_rung",
    "is_errored_run",
]

DEFAULT_LATENCY_BUCKETS_S: Final[tuple[float, ...]] = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)


_K = TypeVar("_K")
_V = TypeVar("_V")


def _freeze(mapping: Mapping[_K, _V]) -> Mapping[_K, _V]:
    return MappingProxyType(dict(mapping))


# Failures of a fallback call that did get an answer from the provider, so the call was made (and
# very likely billed) even though no value came out of it. Matched by class name because the
# ladder reports ``FallbackEvent.error_type`` as a name. ``InvalidFallbackValue`` is the ladder's
# own rejection of a handler value; ``FallbackValidationFailed`` is the cloud fallback's.
# The match is exact: a subclass of one of these raises under its own name, so when the cloud
# fallback gains such a subclass its name must be added here too (a test pins the current names).
_ANSWERED_FAILURES: Final = frozenset({"InvalidFallbackValue", "FallbackValidationFailed"})


def is_errored_run(event: LadderFinished) -> bool:
    """``True`` when the run ended by an error that is not a verdict on the schema.

    That is any exception other than :class:`~slmstrict.errors.ValidationFailed` (a backend that
    failed, ...) in a run whose cloud fallback was not involved. A run that failed because the
    ladder or the fallback could not produce a valid value is a verdict and counts as failed.
    """
    return (
        event.error is not None
        and not isinstance(event.error, ValidationFailed)
        and event.fallback is None
    )


def final_rung(event: LadderFinished) -> Step | None:
    """The rung of the attempt that ended a succeeded run; ``None`` when no rung was needed.

    A used cloud fallback is always ``Step.FALLBACK``, even when the handler returned no attempt
    records. Otherwise this is the ``step`` of the last attempt, which follows the Faz 04
    contract: a grammar attempt that the repair pipeline fixed still reports ``GRAMMAR``.
    """
    if event.fallback is not None and event.fallback.status is FallbackStatus.USED:
        return Step.FALLBACK
    return event.attempts[-1].step if event.attempts else None


@dataclass(frozen=True, slots=True)
class LatencyHistogram:
    """Run duration histogram.

    ``bounds`` are the inclusive upper bounds of the buckets in seconds, ascending.
    ``counts[i]`` is the number of runs with ``bounds[i-1] < duration <= bounds[i]``; the last
    element of ``counts`` is the overflow bucket (``duration > bounds[-1]``), so ``counts`` has
    one more element than ``bounds``.
    """

    bounds: tuple[float, ...]
    counts: tuple[int, ...]
    total_s: float
    count: int

    @property
    def mean_s(self) -> float | None:
        """Mean duration, ``None`` without observations."""
        return self.total_s / self.count if self.count else None


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """Immutable point-in-time copy of the collector's counters.

    ``runs`` counts the runs that reached a verdict on the schema: succeeded, or failed with
    :class:`~slmstrict.errors.ValidationFailed`. Cancelled runs (``cancelled``) and runs that ended
    by an infrastructure error such as a backend failure (``errored``) are not part of ``runs``,
    so a crashing backend does not drag the schema pass rate down. Once the cloud fallback is
    involved, an error raised by the handler (provider outage, timeout) ends the run as a failed
    one: the local rungs had already failed to produce a valid value. Every rate divides by ``runs``
    and is ``None`` while there is no run.

    * ``path_counts``: runs per ladder path (``result.path`` of the run, ``()`` when the raw output
      was valid as it was). The rung distribution of PRD §5.
    * ``rung_runs``: runs in which a rung took effect (a run that used grammar and a re-ask
      counts for both).
    * ``resolved_by``: succeeded runs per rung of the final attempt (see :func:`final_rung`); the
      key ``None`` stands for "the raw output was valid, no rung was needed".
    * ``reask_counts``: runs per number of re-ask attempts they made.
    * ``repair_step_counts``: how often each deterministic repair step was applied.
    * ``fallback_statuses``: runs per :class:`~slmstrict.hooks.FallbackStatus` for runs that
      needed the fallback.
    * ``usage`` and ``calls_without_usage``: token totals per cloud model and the number of
      fallback calls that were answered but came back without a usage report;
      ``failed_calls_without_usage`` counts calls that raised without a usage report. Both feed
      :meth:`cost`.
    """

    runs: int
    succeeded: int
    failed: int
    cancelled: int
    errored: int
    first_attempt_passed: int
    path_counts: Mapping[tuple[Step, ...], int]
    rung_runs: Mapping[Step, int]
    resolved_by: Mapping[Step | None, int]
    reask_counts: Mapping[int, int]
    repair_step_counts: Mapping[str, int]
    fallback_needed: int
    fallback_used: int
    fallback_statuses: Mapping[FallbackStatus, int]
    usage: Mapping[ModelKey, UsageTotals]
    calls_without_usage: int
    failed_calls_without_usage: int
    latency: LatencyHistogram

    @property
    def schema_pass_rate(self) -> float | None:
        """Share of runs that ended in a schema-valid value."""
        return self.succeeded / self.runs if self.runs else None

    @property
    def first_attempt_pass_rate(self) -> float | None:
        """Share of runs whose first local attempt already ended valid."""
        return self.first_attempt_passed / self.runs if self.runs else None

    @property
    def fallback_rate(self) -> float | None:
        """Share of runs the cloud fallback produced the value for."""
        return self.fallback_used / self.runs if self.runs else None

    @property
    def fallback_needed_rate(self) -> float | None:
        """Share of runs that exhausted the local rungs, whether or not the fallback was allowed."""
        return self.fallback_needed / self.runs if self.runs else None

    def rung_rate(self, step: Step) -> float | None:
        """Share of runs in which ``step`` took effect."""
        return self.rung_runs.get(step, 0) / self.runs if self.runs else None

    def cost(self, price_table: PriceTable | None) -> CostEstimate:
        """Estimated cloud cost with the user's price table; unknown without prices."""
        return estimate_cost(
            self.usage,
            price_table,
            calls_without_usage=self.calls_without_usage,
            failed_calls_without_usage=self.failed_calls_without_usage,
        )


class MetricsCollector(BaseLadderObserver):
    """Thread-safe, in-process :class:`~slmstrict.hooks.LadderObserver`.

    ``price_table`` is optional and only used by :meth:`cost`; without it the cost is unknown.
    ``latency_buckets`` are the histogram's upper bounds in seconds (ascending, finite,
    positive).
    """

    def __init__(
        self,
        price_table: PriceTable | None = None,
        *,
        latency_buckets: Sequence[float] = DEFAULT_LATENCY_BUCKETS_S,
    ) -> None:
        bounds = tuple(float(bound) for bound in latency_buckets)
        if not bounds or any(not math.isfinite(b) or b <= 0 for b in bounds):
            raise ValueError("latency_buckets must be non-empty, finite and positive")
        if any(a >= b for a, b in itertools.pairwise(bounds)):
            raise ValueError("latency_buckets must be strictly ascending")
        self._price_table = price_table
        self._bounds = bounds
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self) -> None:
        self._runs = 0
        self._succeeded = 0
        self._cancelled = 0
        self._errored = 0
        self._first_attempt_passed = 0
        self._path_counts: Counter[tuple[Step, ...]] = Counter()
        self._rung_runs: Counter[Step] = Counter()
        self._resolved_by: Counter[Step | None] = Counter()
        self._reask_counts: Counter[int] = Counter()
        self._repair_steps: Counter[str] = Counter()
        self._fallback_needed = 0
        self._fallback_used = 0
        self._fallback_statuses: Counter[FallbackStatus] = Counter()
        self._usage: dict[ModelKey, UsageTotals] = {}
        self._calls_without_usage = 0
        self._failed_calls_without_usage = 0
        self._latency_counts = [0] * (len(self._bounds) + 1)
        self._latency_total = 0.0
        self._latency_n = 0

    def reset(self) -> None:
        """Zero every counter."""
        with self._lock:
            self._reset_locked()

    # -- LadderObserver ----------------------------------------------------------------------

    def on_finish(self, event: LadderFinished) -> None:
        with self._lock:
            if isinstance(event.error, asyncio.CancelledError):
                self._cancelled += 1
                return
            if is_errored_run(event):
                self._errored += 1
                return
            self._runs += 1
            self._latency_n += 1
            self._latency_total += event.duration_s
            self._latency_counts[bisect.bisect_left(self._bounds, event.duration_s)] += 1
            self._path_counts[event.path] += 1
            for step in event.path:
                self._rung_runs[step] += 1
            if event.attempts and event.attempts[0].succeeded:
                self._first_attempt_passed += 1
            self._reask_counts[sum(1 for a in event.attempts if a.step is Step.REASK)] += 1
            for attempt in event.attempts:
                self._repair_steps.update(attempt.repair_steps)
            if event.succeeded:
                self._succeeded += 1
                self._resolved_by[final_rung(event)] += 1
            if event.fallback is not None:
                self._record_fallback(event.fallback)

    def _record_fallback(self, fallback: FallbackEvent) -> None:
        self._fallback_needed += 1
        self._fallback_statuses[fallback.status] += 1
        if fallback.status is FallbackStatus.USED:
            self._fallback_used += 1
        usage = fallback.usage
        if usage is not None:
            key = ModelKey(fallback.provider, fallback.model)
            before = self._usage.get(key, UsageTotals())
            self._usage[key] = UsageTotals(
                input_tokens=before.input_tokens + usage.input_tokens,
                output_tokens=before.output_tokens + usage.output_tokens,
                cached_input_tokens=before.cached_input_tokens + (usage.cached_input_tokens or 0),
                calls=before.calls + 1,
            )
        elif fallback.status is FallbackStatus.USED or (
            fallback.status is FallbackStatus.FAILED and fallback.error_type in _ANSWERED_FAILURES
        ):
            # The provider answered (usable or not) but reported no tokens: the call was made and
            # its cost is unknown. Counting nothing here would report a known 0.
            self._calls_without_usage += 1
        elif fallback.status is FallbackStatus.FAILED and fallback.duration_s is not None:
            # The handler raised before any answer: usually unbilled, but not certain.
            self._failed_calls_without_usage += 1

    # -- reading -----------------------------------------------------------------------------

    def snapshot(self) -> MetricsSnapshot:
        """An immutable copy of the current counters."""
        with self._lock:
            return MetricsSnapshot(
                runs=self._runs,
                succeeded=self._succeeded,
                failed=self._runs - self._succeeded,
                cancelled=self._cancelled,
                errored=self._errored,
                first_attempt_passed=self._first_attempt_passed,
                path_counts=_freeze(self._path_counts),
                rung_runs=_freeze(self._rung_runs),
                resolved_by=_freeze(self._resolved_by),
                reask_counts=_freeze(self._reask_counts),
                repair_step_counts=_freeze(self._repair_steps),
                fallback_needed=self._fallback_needed,
                fallback_used=self._fallback_used,
                fallback_statuses=_freeze(self._fallback_statuses),
                usage=_freeze(self._usage),
                calls_without_usage=self._calls_without_usage,
                failed_calls_without_usage=self._failed_calls_without_usage,
                latency=LatencyHistogram(
                    bounds=self._bounds,
                    counts=tuple(self._latency_counts),
                    total_s=self._latency_total,
                    count=self._latency_n,
                ),
            )

    def cost(self, price_table: PriceTable | None = None) -> CostEstimate:
        """Estimated cloud cost so far.

        Uses ``price_table`` when given, else the one passed to the constructor; ``total`` is
        ``None`` when neither exists or a model's price is missing.
        """
        table = price_table if price_table is not None else self._price_table
        return self.snapshot().cost(table)
