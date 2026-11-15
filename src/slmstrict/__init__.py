"""SlmStrict: reliable structured output from on-device small language models."""

from slmstrict.api import agenerate, generate
from slmstrict.backends.base import Backend, Constraint
from slmstrict.config import CostConfig, FallbackConfig, MetricsConfig, StrictConfig
from slmstrict.errors import BackendError, SchemaError, SlmStrictError, ValidationFailed
from slmstrict.hooks import (
    AttemptStarted,
    BaseLadderObserver,
    FallbackEvent,
    FallbackHandler,
    FallbackResult,
    FallbackStatus,
    LadderFinished,
    LadderObserver,
    StepFinished,
    TokenUsage,
)
from slmstrict.result import Attempt, Result, Step
from slmstrict.schema import Schema, ValidationIssue, ValidationReport

__version__ = "0.1.0.dev0"

__all__ = [
    "Attempt",
    "AttemptStarted",
    "Backend",
    "BackendError",
    "BaseLadderObserver",
    "Constraint",
    "CostConfig",
    "FallbackConfig",
    "FallbackEvent",
    "FallbackHandler",
    "FallbackResult",
    "FallbackStatus",
    "LadderFinished",
    "LadderObserver",
    "MetricsConfig",
    "Result",
    "Schema",
    "SchemaError",
    "SlmStrictError",
    "Step",
    "StepFinished",
    "StrictConfig",
    "TokenUsage",
    "ValidationFailed",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
    "agenerate",
    "generate",
]
