import asyncio
import threading
from decimal import Decimal

import pytest

from slmstrict import (
    Attempt,
    BackendError,
    FallbackConfig,
    FallbackEvent,
    FallbackStatus,
    LadderFinished,
    LadderObserver,
    Result,
    StrictConfig,
    TokenUsage,
    ValidationFailed,
)
from slmstrict.ladder import Ladder
from slmstrict.metrics import MetricsCollector, ModelPrice, PriceTable
from slmstrict.result import Step
from slmstrict.testing import FakeBackend
from tests._ladder_support import GARBAGE, VALID, Item, SpyHandler
from tests.metrics._scenario import (
    SCHEMA,
    PaidValidationFailure,
    run_known_scenario,
)


def _finished(
    *,
    succeeded: bool = True,
    path: tuple[Step, ...] = (),
    attempts: tuple[Attempt, ...] = (),
    duration_s: float = 0.1,
    fallback: FallbackEvent | None = None,
    error: BaseException | None = None,
) -> LadderFinished:
    return LadderFinished(
        run_id="r",
        schema_hash="h",
        succeeded=succeeded,
        path=path,
        attempts=attempts,
        duration_s=duration_s,
        fallback=fallback,
        error=error,
    )


def test_collector_is_a_ladder_observer() -> None:
    assert isinstance(MetricsCollector(), LadderObserver)
    StrictConfig(observer=MetricsCollector())  # accepted by the config validation


def test_empty_snapshot_has_no_rates() -> None:
    snapshot = MetricsCollector().snapshot()
    assert snapshot.runs == 0
    assert snapshot.schema_pass_rate is None
    assert snapshot.first_attempt_pass_rate is None
    assert snapshot.fallback_rate is None
    assert snapshot.rung_rate(Step.GRAMMAR) is None
    assert snapshot.latency.mean_s is None


async def test_rates_of_a_known_scenario() -> None:
    collector = MetricsCollector()
    await run_known_scenario(collector)
    snapshot = collector.snapshot()

    assert snapshot.runs == 6
    assert snapshot.succeeded == 5
    assert snapshot.failed == 1
    assert snapshot.cancelled == 0
    assert snapshot.schema_pass_rate == pytest.approx(5 / 6)
    assert snapshot.first_attempt_passed == 3  # runs 1, 2 (after repair) and 6
    assert snapshot.first_attempt_pass_rate == pytest.approx(3 / 6)
    assert snapshot.fallback_used == 1
    assert snapshot.fallback_rate == pytest.approx(1 / 6)
    assert snapshot.fallback_needed == 2  # run 5 needed it but none was configured
    assert snapshot.fallback_needed_rate == pytest.approx(2 / 6)
    assert dict(snapshot.fallback_statuses) == {
        FallbackStatus.USED: 1,
        FallbackStatus.NOT_CONFIGURED: 1,
    }


async def test_rung_distribution_of_a_known_scenario() -> None:
    collector = MetricsCollector()
    await run_known_scenario(collector)
    snapshot = collector.snapshot()

    assert dict(snapshot.path_counts) == {
        (): 1,
        (Step.REPAIR,): 1,
        (Step.REASK,): 2,  # run 3 and the failed run 5
        (Step.REASK, Step.FALLBACK): 1,
        (Step.GRAMMAR,): 1,
    }
    assert dict(snapshot.rung_runs) == {
        Step.REPAIR: 1,
        Step.REASK: 3,
        Step.FALLBACK: 1,
        Step.GRAMMAR: 1,
    }
    assert snapshot.rung_rate(Step.REASK) == pytest.approx(3 / 6)
    assert snapshot.rung_rate(Step.FALLBACK) == pytest.approx(1 / 6)
    assert dict(snapshot.resolved_by) == {
        None: 1,
        Step.REPAIR: 1,
        Step.REASK: 1,
        Step.FALLBACK: 1,
        Step.GRAMMAR: 1,
    }
    # re-asks per run: runs 1, 2, 6 made none, run 3 made one, runs 4 and 5 made two.
    assert dict(snapshot.reask_counts) == {0: 3, 1: 1, 2: 2}
    assert dict(snapshot.repair_step_counts) == {"fences": 1}  # run 2 only


async def test_run_latency_is_recorded() -> None:
    collector = MetricsCollector()
    await run_known_scenario(collector)
    latency = collector.snapshot().latency
    assert latency.count == 6
    assert sum(latency.counts) == 6
    assert len(latency.counts) == len(latency.bounds) + 1
    assert latency.total_s >= 0
    assert latency.mean_s is not None


def test_histogram_buckets_are_upper_inclusive_with_an_overflow_bucket() -> None:
    collector = MetricsCollector(latency_buckets=(0.1, 1.0))
    for duration in (0.05, 0.1, 0.5, 1.0, 5.0):
        collector.on_finish(_finished(duration_s=duration))
    latency = collector.snapshot().latency
    assert latency.bounds == (0.1, 1.0)
    assert latency.counts == (2, 2, 1)
    assert latency.total_s == pytest.approx(6.65)


@pytest.mark.parametrize("buckets", [(), (0.0, 1.0), (2.0, 1.0), (1.0, 1.0), (float("inf"),)])
def test_invalid_latency_buckets_are_rejected(buckets: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="latency_buckets"):
        MetricsCollector(latency_buckets=buckets)


def test_cancelled_runs_are_counted_apart() -> None:
    collector = MetricsCollector()
    collector.on_finish(_finished(succeeded=False, error=asyncio.CancelledError()))
    collector.on_finish(_finished())
    snapshot = collector.snapshot()
    assert (snapshot.runs, snapshot.cancelled, snapshot.succeeded) == (1, 1, 1)


async def test_backend_errors_stay_out_of_the_schema_pass_rate() -> None:
    collector = MetricsCollector()
    config = StrictConfig(observer=collector)
    for _ in range(5):  # the backend crashes: the model never produced an output
        with pytest.raises(BackendError):
            await Ladder(FakeBackend([BackendError("boom")]), config).run(SCHEMA, "p")
    for _ in range(5):
        await Ladder(FakeBackend([VALID]), config).run(SCHEMA, "p")
    snapshot = collector.snapshot()
    assert (snapshot.runs, snapshot.succeeded, snapshot.failed) == (5, 5, 0)
    assert snapshot.errored == 5
    assert snapshot.schema_pass_rate == 1.0  # not 0.5
    assert snapshot.latency.count == 5


async def test_a_schema_failure_is_still_a_failed_run() -> None:
    collector = MetricsCollector()
    with pytest.raises(ValidationFailed):
        await Ladder(FakeBackend([GARBAGE] * 3), StrictConfig(observer=collector)).run(SCHEMA, "p")
    snapshot = collector.snapshot()
    assert (snapshot.runs, snapshot.failed, snapshot.errored) == (1, 1, 0)


async def test_a_run_that_fails_in_the_fallback_is_a_verdict_not_an_error() -> None:
    collector = MetricsCollector()
    config = StrictConfig(
        observer=collector, fallback=FallbackConfig(handler=SpyHandler(error=RuntimeError("down")))
    )
    with pytest.raises(RuntimeError):
        await Ladder(FakeBackend([GARBAGE] * 3), config).run(SCHEMA, "p")
    snapshot = collector.snapshot()
    assert (snapshot.runs, snapshot.failed, snapshot.errored) == (1, 1, 0)
    assert snapshot.fallback_needed == 1
    assert snapshot.fallback_used == 0


async def test_a_used_fallback_without_attempt_records_resolves_at_the_fallback() -> None:
    collector = MetricsCollector()
    bare = Result(value=Item(name="cloud", count=9))  # a third-party handler: no attempts
    config = StrictConfig(
        observer=collector, fallback=FallbackConfig(handler=SpyHandler(result=bare))
    )
    await Ladder(FakeBackend([GARBAGE] * 3), config).run(SCHEMA, "p")
    assert dict(collector.snapshot().resolved_by) == {Step.FALLBACK: 1}


async def test_an_unusable_fallback_value_without_usage_makes_the_cost_unknown() -> None:
    table = PriceTable({("acme", "m"): ModelPrice(3, 15)})
    collector = MetricsCollector(price_table=table)
    bad = Result(value={"bad": 1}, attempts=(Attempt(step=Step.FALLBACK, raw_output="{}"),))
    config = StrictConfig(
        observer=collector, fallback=FallbackConfig(handler=SpyHandler(result=bad))
    )
    with pytest.raises(ValidationFailed):
        await Ladder(FakeBackend([GARBAGE] * 3), config).run(SCHEMA, "p")
    snapshot = collector.snapshot()
    assert dict(snapshot.fallback_statuses) == {FallbackStatus.FAILED: 1}
    assert snapshot.calls_without_usage == 1
    estimate = collector.cost()
    assert estimate.total is None  # not a known 0
    assert estimate.known is False


async def test_usage_reported_by_a_failing_handler_is_priced() -> None:
    table = PriceTable({("acme", "m"): ModelPrice(10, 20)})
    collector = MetricsCollector(price_table=table)
    failure = PaidValidationFailure(TokenUsage(1000, 500), "acme", "m")
    config = StrictConfig(
        observer=collector, fallback=FallbackConfig(handler=SpyHandler(error=failure))
    )
    with pytest.raises(ValidationFailed):
        await Ladder(FakeBackend([GARBAGE] * 3), config).run(SCHEMA, "p")
    snapshot = collector.snapshot()
    assert snapshot.calls_without_usage == 0
    assert collector.cost().total == Decimal("0.02")  # (1000 * 10 + 500 * 20) / 1_000_000


def test_snapshot_is_immutable_and_detached() -> None:
    collector = MetricsCollector()
    collector.on_finish(_finished())
    snapshot = collector.snapshot()
    with pytest.raises(AttributeError):
        snapshot.runs = 5  # type: ignore[misc]
    with pytest.raises(TypeError):
        snapshot.path_counts[()] = 9  # type: ignore[index]
    collector.on_finish(_finished())
    assert snapshot.runs == 1  # later events do not leak into an earlier snapshot
    assert collector.snapshot().runs == 2


def test_reset_zeroes_the_counters() -> None:
    collector = MetricsCollector()
    collector.on_finish(_finished())
    collector.reset()
    assert collector.snapshot().runs == 0


def test_usage_is_summed_per_model_and_calls_without_usage_are_classified() -> None:
    collector = MetricsCollector()
    usage = TokenUsage(100, 10, cached_input_tokens=40)

    def event(
        status: FallbackStatus,
        tokens: TokenUsage | None,
        model: str | None,
        error_type: str | None = None,
    ) -> LadderFinished:
        return _finished(
            fallback=FallbackEvent(
                run_id="r",
                schema_hash="h",
                status=status,
                attempts_before=3,
                usage=tokens,
                duration_s=0.5,
                provider="acme" if model else None,
                model=model,
                error_type=error_type,
            )
        )

    collector.on_finish(event(FallbackStatus.USED, usage, "m1"))
    collector.on_finish(event(FallbackStatus.USED, TokenUsage(50, 5), "m1"))
    snapshot = collector.snapshot()
    totals = next(iter(snapshot.usage.values()))
    assert (totals.input_tokens, totals.output_tokens) == (150, 15)
    assert (totals.cached_input_tokens, totals.calls) == (40, 2)
    assert (snapshot.calls_without_usage, snapshot.failed_calls_without_usage) == (0, 0)

    # the handler raised before any answer: not certain to be unbilled, tracked on its own
    collector.on_finish(event(FallbackStatus.FAILED, None, None, error_type="FallbackError"))
    snapshot = collector.snapshot()
    assert (snapshot.calls_without_usage, snapshot.failed_calls_without_usage) == (0, 1)

    # answered, no usage reported: the cost of the call is unknown
    collector.on_finish(event(FallbackStatus.USED, None, "m1"))
    collector.on_finish(
        event(FallbackStatus.FAILED, None, None, error_type="FallbackValidationFailed")
    )
    snapshot = collector.snapshot()
    assert (snapshot.calls_without_usage, snapshot.failed_calls_without_usage) == (2, 1)
    assert snapshot.cost(None).failed_calls_without_usage == 1


def test_cost_is_unknown_without_a_price_table() -> None:
    collector = MetricsCollector()
    collector.on_finish(
        _finished(
            fallback=FallbackEvent(
                run_id="r",
                schema_hash="h",
                status=FallbackStatus.USED,
                attempts_before=3,
                usage=TokenUsage(10, 1),
                provider="acme",
                model="m",
            )
        )
    )
    assert collector.cost().total is None
    assert collector.snapshot().cost(None).known is False
    assert collector.cost(PriceTable()).total is None  # empty table: model has no price


async def test_counters_stay_consistent_across_parallel_runs() -> None:
    collector = MetricsCollector()
    config = StrictConfig(observer=collector)

    async def good() -> None:
        await Ladder(FakeBackend([VALID]), config).run(SCHEMA, "p")

    async def bad() -> None:
        with pytest.raises(ValidationFailed):
            await Ladder(FakeBackend([GARBAGE] * 3), config).run(SCHEMA, "p")

    await asyncio.gather(*[good() if i % 2 == 0 else bad() for i in range(200)])
    snapshot = collector.snapshot()
    assert (snapshot.runs, snapshot.succeeded, snapshot.failed) == (200, 100, 100)
    assert snapshot.first_attempt_passed == 100
    assert dict(snapshot.path_counts) == {(): 100, (Step.REASK,): 100}
    assert dict(snapshot.reask_counts) == {0: 100, 2: 100}
    assert sum(snapshot.latency.counts) == 200


def test_counters_stay_consistent_across_threads() -> None:
    collector = MetricsCollector()
    threads_n, per_thread = 16, 500
    usage = TokenUsage(10, 2)
    event = _finished(
        path=(Step.REASK, Step.FALLBACK),
        attempts=(Attempt(step=Step.FALLBACK, raw_output="x"),),
        fallback=FallbackEvent(
            run_id="r",
            schema_hash="h",
            status=FallbackStatus.USED,
            attempts_before=3,
            usage=usage,
            provider="acme",
            model="m",
        ),
    )
    barrier = threading.Barrier(threads_n)
    stop = threading.Event()
    seen: list[tuple[int, int]] = []

    def worker() -> None:
        barrier.wait()
        for _ in range(per_thread):
            collector.on_finish(event)

    def reader() -> None:
        while not stop.is_set():
            snap = collector.snapshot()
            seen.append((snap.runs, snap.fallback_used))

    threads = [threading.Thread(target=worker) for _ in range(threads_n)]
    watcher = threading.Thread(target=reader)
    watcher.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stop.set()
    watcher.join()

    total = threads_n * per_thread
    snapshot = collector.snapshot()
    assert snapshot.runs == total
    assert snapshot.fallback_used == total
    assert snapshot.rung_runs[Step.FALLBACK] == total
    assert next(iter(snapshot.usage.values())).input_tokens == 10 * total
    assert next(iter(snapshot.usage.values())).calls == total
    assert sum(snapshot.latency.counts) == total
    assert all(runs == used for runs, used in seen)  # a snapshot is never torn


def test_answered_failure_names_cover_the_real_exception_classes() -> None:
    # The collector matches FallbackEvent.error_type (a class name) against this set; renaming the
    # cloud fallback's exception must break this test instead of silently re-opening the "known 0
    # cost" gap.
    from slmstrict.fallback.base import FallbackValidationFailed
    from slmstrict.metrics.collector import _ANSWERED_FAILURES

    assert FallbackValidationFailed.__name__ in _ANSWERED_FAILURES
    assert "InvalidFallbackValue" in _ANSWERED_FAILURES


async def test_the_real_fallback_validation_failure_without_usage_makes_the_cost_unknown() -> None:
    from slmstrict.fallback.base import FallbackValidationFailed

    table = PriceTable({("acme", "m"): ModelPrice(3, 15)})
    collector = MetricsCollector(price_table=table)
    failure = FallbackValidationFailed("the provider answered with an unusable value")
    config = StrictConfig(
        observer=collector, fallback=FallbackConfig(handler=SpyHandler(error=failure))
    )
    with pytest.raises(FallbackValidationFailed):
        await Ladder(FakeBackend([GARBAGE] * 3), config).run(SCHEMA, "p")
    snapshot = collector.snapshot()
    assert (snapshot.calls_without_usage, snapshot.failed_calls_without_usage) == (1, 0)
    assert collector.cost().total is None
