"""OpenTelemetry metrics export (PRD section 5, optional OTLP export).

:class:`OtelObserver` turns ladder events into OpenTelemetry metrics. It comes with the ``otel``
extra and imports OpenTelemetry lazily: importing this module loads nothing, constructing the
observer without the extra raises :class:`ImportError` with an install hint.

Data minimisation (ADR-007): metric attributes are limited to the rung (``slmstrict.step``), the
backend kind (``slmstrict.backend``), the schema hash (``slmstrict.schema_hash``) and the cloud
provider name (``slmstrict.provider``). No prompt, model output, error message or model name is
ever an attribute.

Instruments (all prefixed ``slmstrict.``):

* ``runs`` (counter): finished runs; extra attribute ``slmstrict.outcome`` (``success`` /
  ``failure`` / ``error`` for an infrastructure error such as a backend failure / ``cancelled``).
  ``slmstrict.step`` is the rung of the final attempt of a succeeded run, ``none`` when no rung
  was needed or the run did not succeed.
* ``run.duration`` (histogram, s): duration of a finished run. Both duration histograms advertise
  the same second-based bucket bounds as the local collector, since the SDK's default bounds are
  chosen for milliseconds.
* ``attempts`` (counter) and ``attempt.duration`` (histogram, s): local attempts per rung;
  ``attempts`` also carries ``slmstrict.outcome`` (``valid`` / ``invalid``).
* ``fallback.events`` (counter): runs that exhausted the local rungs; ``slmstrict.outcome`` is the
  :class:`~slmstrict.hooks.FallbackStatus` value.
* ``fallback.input_tokens``, ``fallback.output_tokens``, ``fallback.cached_input_tokens``
  (counters): token usage the provider reported.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping, Sequence
from typing import Final, Protocol

from slmstrict.hooks import (
    BaseLadderObserver,
    FallbackEvent,
    LadderFinished,
    StepFinished,
)
from slmstrict.metrics.collector import DEFAULT_LATENCY_BUCKETS_S, final_rung, is_errored_run
from slmstrict.result import Step

__all__ = ["OtelObserver"]

_INSTALL_HINT: Final = (
    "OpenTelemetry is required for the OTLP metrics export. "
    "Install it with: pip install 'slmstrict[otel]'"
)

_Attributes = Mapping[str, str]


class _Counter(Protocol):
    def add(self, amount: int, attributes: _Attributes | None = None) -> None: ...


class _Histogram(Protocol):
    def record(self, amount: float, attributes: _Attributes | None = None) -> None: ...


class _Meter(Protocol):
    def create_counter(self, name: str, unit: str = "", description: str = "") -> _Counter: ...

    def create_histogram(
        self,
        name: str,
        unit: str = "",
        description: str = "",
        *,
        explicit_bucket_boundaries_advisory: Sequence[float] | None = None,
    ) -> _Histogram: ...


class _MeterProvider(Protocol):
    def get_meter(self, name: str, version: str = "") -> _Meter: ...


def _load_meter_provider() -> _MeterProvider:
    """The global OpenTelemetry meter provider, imported on first use."""
    try:
        api = importlib.import_module("opentelemetry.metrics")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    provider: _MeterProvider = api.get_meter_provider()
    return provider


class OtelObserver(BaseLadderObserver):
    """A :class:`~slmstrict.hooks.LadderObserver` that writes OpenTelemetry metrics.

    ``meter_provider`` is the SDK ``MeterProvider`` to use; when ``None`` the global provider is
    used, so an application that already configured OTLP needs no further wiring. ``backend`` is
    the backend kind to put on the ``slmstrict.backend`` attribute (``"llamacpp"``, ``"onnx"``,
    ...); the ladder events do not carry it, so the caller names it. It must be a short, fixed
    identifier, never user data.
    """

    def __init__(
        self,
        meter_provider: object | None = None,
        *,
        backend: str | None = None,
        meter_name: str = "slmstrict",
    ) -> None:
        provider: _MeterProvider
        if meter_provider is None:
            provider = _load_meter_provider()
        elif hasattr(meter_provider, "get_meter"):
            provider = meter_provider
        else:
            raise TypeError("meter_provider must provide get_meter()")
        meter = provider.get_meter(meter_name)
        self._backend = backend
        self._runs = meter.create_counter("slmstrict.runs", "{run}", "Finished ladder runs")
        self._run_duration = meter.create_histogram(
            "slmstrict.run.duration",
            "s",
            "Duration of a finished ladder run",
            explicit_bucket_boundaries_advisory=DEFAULT_LATENCY_BUCKETS_S,
        )
        self._attempts = meter.create_counter(
            "slmstrict.attempts", "{attempt}", "Local generation attempts"
        )
        self._attempt_duration = meter.create_histogram(
            "slmstrict.attempt.duration",
            "s",
            "Duration of a local generation attempt",
            explicit_bucket_boundaries_advisory=DEFAULT_LATENCY_BUCKETS_S,
        )
        self._fallbacks = meter.create_counter(
            "slmstrict.fallback.events", "{event}", "Runs that exhausted the local rungs"
        )
        self._input_tokens = meter.create_counter(
            "slmstrict.fallback.input_tokens", "{token}", "Cloud fallback input tokens"
        )
        self._output_tokens = meter.create_counter(
            "slmstrict.fallback.output_tokens", "{token}", "Cloud fallback output tokens"
        )
        self._cached_tokens = meter.create_counter(
            "slmstrict.fallback.cached_input_tokens",
            "{token}",
            "Cloud fallback input tokens served from the provider's prompt cache",
        )

    def _attributes(
        self,
        schema_hash: str,
        step: Step | str | None,
        *,
        outcome: str | None = None,
        provider: str | None = None,
    ) -> dict[str, str]:
        attributes = {
            "slmstrict.schema_hash": schema_hash,
            "slmstrict.step": step.value if isinstance(step, Step) else (step or "none"),
        }
        if self._backend is not None:
            attributes["slmstrict.backend"] = self._backend
        if outcome is not None:
            attributes["slmstrict.outcome"] = outcome
        if provider is not None:
            attributes["slmstrict.provider"] = provider
        return attributes

    def on_step_finished(self, event: StepFinished) -> None:
        attributes = self._attributes(
            event.schema_hash, event.step, outcome="valid" if event.succeeded else "invalid"
        )
        self._attempts.add(1, attributes)
        self._attempt_duration.record(event.duration_s, attributes)

    def on_fallback(self, event: FallbackEvent) -> None:
        attributes = self._attributes(
            event.schema_hash, Step.FALLBACK, outcome=event.status.value, provider=event.provider
        )
        self._fallbacks.add(1, attributes)
        usage = event.usage
        if usage is None:
            return
        token_attributes = self._attributes(
            event.schema_hash, Step.FALLBACK, provider=event.provider
        )
        self._input_tokens.add(usage.input_tokens, token_attributes)
        self._output_tokens.add(usage.output_tokens, token_attributes)
        if usage.cached_input_tokens is not None:
            self._cached_tokens.add(usage.cached_input_tokens, token_attributes)

    def on_finish(self, event: LadderFinished) -> None:
        if isinstance(event.error, asyncio.CancelledError):
            outcome, step = "cancelled", None
        elif is_errored_run(event):
            outcome, step = "error", None
        elif event.succeeded:
            outcome, step = "success", final_rung(event)
        else:
            outcome, step = "failure", None
        provider = event.fallback.provider if event.fallback is not None else None
        attributes = self._attributes(event.schema_hash, step, outcome=outcome, provider=provider)
        self._runs.add(1, attributes)
        self._run_duration.record(event.duration_s, attributes)
