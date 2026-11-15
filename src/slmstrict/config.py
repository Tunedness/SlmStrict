"""Configuration for the enforcement ladder."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from slmstrict.hooks import FallbackHandler, LadderObserver

__all__ = ["CostConfig", "FallbackConfig", "MetricsConfig", "StrictConfig"]


@dataclass(frozen=True, slots=True)
class FallbackConfig:
    """Cloud fallback settings.

    ``handler`` is the rung-4 implementation the ladder calls once grammar, repair and re-ask
    failed. With ``handler=None`` the fallback counts as not configured. Provider, masking hook and
    logging options arrive with the fallback phase, which adds fields here without changing
    :class:`StrictConfig`.
    """

    handler: FallbackHandler | None = None

    def __post_init__(self) -> None:
        handler = self.handler
        if handler is None:
            return
        if not isinstance(handler, FallbackHandler) or not inspect.iscoroutinefunction(
            handler.handle
        ):
            raise TypeError("handler must implement FallbackHandler (an async handle() method)")


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
    export, cost reported as unknown. The cloud fallback is active only when ``fallback`` carries a
    handler and ``local_only`` is off. ``observer`` receives the ladder events of every run.
    """

    max_reask: int = 2
    local_only: bool = False
    fallback: FallbackConfig | None = None
    metrics: MetricsConfig | None = None
    cost: CostConfig | None = None
    observer: LadderObserver | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_reask, bool) or not isinstance(self.max_reask, int):
            raise TypeError("max_reask must be an int")
        if self.max_reask < 0:
            raise ValueError("max_reask must be >= 0")
        if not isinstance(self.local_only, bool):
            raise TypeError("local_only must be a bool")
        if self.observer is not None and not isinstance(self.observer, LadderObserver):
            raise TypeError("observer must implement LadderObserver")
