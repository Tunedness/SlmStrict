"""SlmStrict: reliable structured output from on-device small language models."""

from slmstrict.backends.base import Backend, Constraint
from slmstrict.config import CostConfig, FallbackConfig, MetricsConfig, StrictConfig
from slmstrict.errors import BackendError, SchemaError, SlmStrictError, ValidationFailed
from slmstrict.result import Attempt, Result, Step
from slmstrict.schema import Schema, ValidationIssue, ValidationReport

__version__ = "0.1.0.dev0"

__all__ = [
    "Attempt",
    "Backend",
    "BackendError",
    "Constraint",
    "CostConfig",
    "FallbackConfig",
    "MetricsConfig",
    "Result",
    "Schema",
    "SchemaError",
    "SlmStrictError",
    "Step",
    "StrictConfig",
    "ValidationFailed",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
]
