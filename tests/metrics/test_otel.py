import asyncio
import importlib
import os
import subprocess
import sys
from typing import Any

import pytest

from slmstrict import (
    BackendError,
    FallbackConfig,
    LadderFinished,
    Result,
    StrictConfig,
    ValidationFailed,
)
from slmstrict.ladder import Ladder
from slmstrict.metrics import DEFAULT_LATENCY_BUCKETS_S, otel
from slmstrict.metrics.otel import OtelObserver
from slmstrict.testing import FakeBackend
from tests._ladder_support import GARBAGE, VALID, Item, SpyHandler
from tests.metrics._scenario import (
    CLOUD_PROVIDER,
    SCHEMA,
    cloud_handler,
    run_known_scenario,
)

ALLOWED_ATTRIBUTES = {
    "slmstrict.step",
    "slmstrict.backend",
    "slmstrict.schema_hash",
    "slmstrict.provider",
    "slmstrict.outcome",
}


def test_importing_the_package_loads_no_opentelemetry() -> None:
    code = (
        "import sys, slmstrict, slmstrict.metrics, slmstrict.metrics.otel;"
        "assert not [m for m in sys.modules if m.startswith('opentelemetry')], 'loaded'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_missing_extra_gives_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(otel.importlib, "import_module", fail)
    with pytest.raises(ImportError, match=r"slmstrict\[otel\]"):
        OtelObserver()


def test_meter_provider_must_provide_get_meter() -> None:
    with pytest.raises(TypeError, match="get_meter"):
        OtelObserver(object())


# The tests below need the SDK of the `otel` extra. CI installs the extra, so there a missing SDK
# is a failure (the ADR-007 content test must really run); elsewhere they are skipped.


def _sdk() -> tuple[Any, Any]:
    names = ("opentelemetry.sdk.metrics", "opentelemetry.sdk.metrics.export")
    if os.environ.get("CI"):
        return tuple(importlib.import_module(name) for name in names)  # type: ignore[return-value]
    return tuple(pytest.importorskip(name) for name in names)  # type: ignore[return-value]


def _setup(backend: str | None = "fake") -> tuple[OtelObserver, Any]:
    sdk_metrics, sdk_export = _sdk()
    reader = sdk_export.InMemoryMetricReader()
    provider = sdk_metrics.MeterProvider(metric_readers=[reader])
    return OtelObserver(provider, backend=backend), reader


def _points(reader: Any) -> dict[str, list[Any]]:
    """Metric name -> data points."""
    data = reader.get_metrics_data()
    found: dict[str, list[Any]] = {}
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                found[metric.name] = list(metric.data.data_points)
    return found


def _total(points: list[Any], **attributes: str) -> int:
    return sum(
        p.value
        for p in points
        if all(p.attributes.get(key) == value for key, value in attributes.items())
    )


async def test_events_land_on_the_right_names_and_attributes() -> None:
    observer, reader = _setup()
    await run_known_scenario(observer)
    metrics = _points(reader)

    assert set(metrics) == {
        "slmstrict.runs",
        "slmstrict.run.duration",
        "slmstrict.attempts",
        "slmstrict.attempt.duration",
        "slmstrict.fallback.events",
        "slmstrict.fallback.input_tokens",
        "slmstrict.fallback.output_tokens",
        "slmstrict.fallback.cached_input_tokens",
    }
    runs = metrics["slmstrict.runs"]
    assert _total(runs) == 6
    assert _total(runs, **{"slmstrict.outcome": "success"}) == 5
    assert _total(runs, **{"slmstrict.outcome": "failure"}) == 1
    assert _total(runs, **{"slmstrict.step": "grammar"}) == 1
    assert _total(runs, **{"slmstrict.step": "repair"}) == 1
    assert _total(runs, **{"slmstrict.step": "reask"}) == 1
    assert _total(runs, **{"slmstrict.step": "fallback"}) == 1
    assert _total(runs, **{"slmstrict.step": "none", "slmstrict.outcome": "success"}) == 1
    assert _total(runs, **{"slmstrict.provider": CLOUD_PROVIDER}) == 1
    assert {p.attributes["slmstrict.backend"] for p in runs} == {"fake"}
    assert {p.attributes["slmstrict.schema_hash"] for p in runs} == {SCHEMA.schema_hash}

    fallbacks = metrics["slmstrict.fallback.events"]
    assert _total(fallbacks, **{"slmstrict.outcome": "used"}) == 1
    assert _total(fallbacks, **{"slmstrict.outcome": "not_configured"}) == 1
    assert _total(metrics["slmstrict.fallback.input_tokens"]) == 1200
    assert _total(metrics["slmstrict.fallback.output_tokens"]) == 300
    assert _total(metrics["slmstrict.fallback.cached_input_tokens"]) == 1000

    attempts = metrics["slmstrict.attempts"]
    assert _total(attempts, **{"slmstrict.step": "grammar", "slmstrict.outcome": "valid"}) == 1
    assert _total(attempts, **{"slmstrict.step": "reask", "slmstrict.outcome": "invalid"}) == 4
    assert _total(attempts, **{"slmstrict.step": "reask", "slmstrict.outcome": "valid"}) == 1
    assert sum(p.count for p in metrics["slmstrict.run.duration"]) == 6
    assert sum(p.count for p in metrics["slmstrict.attempt.duration"]) == _total(attempts)


async def test_no_attribute_or_metric_carries_content() -> None:
    secret_prompt = "SECRET-PROMPT-PATIENT-4711"
    secret_output = "SECRET-OUTPUT-DIAGNOSIS"
    observer, reader = _setup()
    config = StrictConfig(observer=observer, fallback=FallbackConfig(handler=cloud_handler()))
    await Ladder(FakeBackend([secret_output] * 3), config).run(SCHEMA, secret_prompt)
    with pytest.raises(ValidationFailed):
        await Ladder(FakeBackend([secret_output] * 3), StrictConfig(observer=observer)).run(
            SCHEMA, secret_prompt
        )
    await Ladder(FakeBackend([VALID]), StrictConfig(observer=observer)).run(SCHEMA, secret_prompt)

    keys: set[str] = set()
    values: set[str] = set()
    for points in _points(reader).values():
        for point in points:
            keys.update(point.attributes)
            values.update(str(v) for v in point.attributes.values())
    assert keys <= ALLOWED_ATTRIBUTES
    assert "slmstrict.step" in keys
    for value in values:
        assert "SECRET" not in value
    # the serialized payload as a whole carries no content either
    assert "SECRET" not in reader.get_metrics_data().to_json()


async def test_backend_attribute_is_omitted_when_unset() -> None:
    observer, reader = _setup(backend=None)
    await Ladder(FakeBackend([VALID]), StrictConfig(observer=observer)).run(SCHEMA, "p")
    for points in _points(reader).values():
        for point in points:
            assert "slmstrict.backend" not in point.attributes


async def test_failed_runs_are_labelled() -> None:
    observer, reader = _setup()
    with pytest.raises(ValidationFailed):
        await Ladder(FakeBackend([GARBAGE] * 3), StrictConfig(observer=observer)).run(SCHEMA, "p")
    runs = _points(reader)["slmstrict.runs"]
    assert _total(runs, **{"slmstrict.outcome": "failure", "slmstrict.step": "none"}) == 1


async def test_backend_errors_are_labelled_as_errors() -> None:
    observer, reader = _setup()
    backend = FakeBackend([BackendError("boom")])
    with pytest.raises(BackendError):
        await Ladder(backend, StrictConfig(observer=observer)).run(SCHEMA, "p")
    runs = _points(reader)["slmstrict.runs"]
    assert _total(runs, **{"slmstrict.outcome": "error", "slmstrict.step": "none"}) == 1


def test_cancelled_runs_are_labelled() -> None:
    observer, reader = _setup()
    observer.on_finish(
        LadderFinished(
            run_id="r",
            schema_hash="h",
            succeeded=False,
            path=(),
            attempts=(),
            duration_s=0.2,
            error=asyncio.CancelledError(),
        )
    )
    runs = _points(reader)["slmstrict.runs"]
    assert _total(runs, **{"slmstrict.outcome": "cancelled", "slmstrict.step": "none"}) == 1


async def test_a_used_fallback_without_attempt_records_is_labelled_fallback() -> None:
    observer, reader = _setup()
    bare = Result(value=Item(name="cloud", count=9))
    config = StrictConfig(
        observer=observer, fallback=FallbackConfig(handler=SpyHandler(result=bare))
    )
    await Ladder(FakeBackend([GARBAGE] * 3), config).run(SCHEMA, "p")
    runs = _points(reader)["slmstrict.runs"]
    assert _total(runs, **{"slmstrict.outcome": "success", "slmstrict.step": "fallback"}) == 1


async def test_duration_histograms_use_second_based_bucket_bounds() -> None:
    observer, reader = _setup()
    await Ladder(FakeBackend([VALID]), StrictConfig(observer=observer)).run(SCHEMA, "p")
    metrics = _points(reader)
    for name in ("slmstrict.run.duration", "slmstrict.attempt.duration"):
        assert tuple(metrics[name][0].explicit_bounds) == DEFAULT_LATENCY_BUCKETS_S


async def test_uses_the_global_provider_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk_metrics, sdk_export = _sdk()
    reader = sdk_export.InMemoryMetricReader()
    provider = sdk_metrics.MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(otel, "_load_meter_provider", lambda: provider)
    observer = OtelObserver()
    await Ladder(FakeBackend([VALID]), StrictConfig(observer=observer)).run(SCHEMA, "p")
    assert _total(_points(reader)["slmstrict.runs"]) == 1
