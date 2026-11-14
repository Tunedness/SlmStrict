"""Configuration for the enforcement ladder."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CostConfig", "FallbackConfig", "MetricsConfig", "StrictConfig"]


@dataclass(frozen=True, slots=True)
class FallbackConfig:
    """Cloud fallback settings.

    Intentionally has no fields yet: provider, masking hook and logging options arrive with the
    fallback phase, which adds fields here without changing :class:`StrictConfig`.
    """


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Metrics settings (local counters, optional OTLP export).

    Intentionally has no fields yet; the metrics phase fills it in.
    """


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Cost estimation settings (user-supplied price table).

    Intentionally has no fields yet; the metrics phase fills it in. The package ships no prices.
    """


@dataclass(frozen=True, slots=True)
class StrictConfig:
    """Settings for one SlmStrict engine.

    ``max_reask`` is the number of re-ask attempts after the first generation (ADR-002).
    ``local_only`` locks the cloud fallback off whatever else is configured (ADR-003).
    ``fallback``, ``metrics`` and ``cost`` default to ``None``: no cloud fallback, no metrics
    export, cost reported as unknown.
    """

    max_reask: int = 2
    local_only: bool = False
    fallback: FallbackConfig | None = None
    metrics: MetricsConfig | None = None
    cost: CostConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_reask, bool) or not isinstance(self.max_reask, int):
            raise TypeError("max_reask must be an int")
        if self.max_reask < 0:
            raise ValueError("max_reask must be >= 0")
        if not isinstance(self.local_only, bool):
            raise TypeError("local_only must be a bool")
