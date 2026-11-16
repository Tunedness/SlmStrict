"""Metrics, cost estimation and optional OTLP export (PRD section 5).

``OtelObserver`` is not imported here: it lives in :mod:`slmstrict.metrics.otel` and needs the
``otel`` extra, while this package imports nothing outside the core dependencies.
"""

from slmstrict.metrics.collector import (
    DEFAULT_LATENCY_BUCKETS_S,
    LatencyHistogram,
    MetricsCollector,
    MetricsSnapshot,
)
from slmstrict.metrics.cost import (
    CostEstimate,
    ModelKey,
    ModelPrice,
    PriceTable,
    UsageTotals,
    estimate_cost,
)

__all__ = [
    "DEFAULT_LATENCY_BUCKETS_S",
    "CostEstimate",
    "LatencyHistogram",
    "MetricsCollector",
    "MetricsSnapshot",
    "ModelKey",
    "ModelPrice",
    "PriceTable",
    "UsageTotals",
    "estimate_cost",
]
