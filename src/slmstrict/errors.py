"""Exception hierarchy for SlmStrict.

Every error raised on purpose by the library derives from :class:`SlmStrictError`, so callers
can catch the whole family with a single ``except`` clause.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slmstrict.schema import ValidationIssue

__all__ = [
    "BackendError",
    "SchemaError",
    "SlmStrictError",
    "ValidationFailed",
]


class SlmStrictError(Exception):
    """Root of all errors raised by SlmStrict."""


class SchemaError(SlmStrictError):
    """The schema given to SlmStrict is unusable (not a model, not a valid JSON Schema, ...)."""


class ValidationFailed(SlmStrictError):  # noqa: N818 - public name fixed by the phase contract
    """The model output could not be brought into line with the schema.

    ``issues`` holds the structured validation problems of the last attempt and ``last_output``
    the raw text of that attempt, when available. Neither is part of ``str(error)``, so the
    model output never leaks into logs through the exception message.
    """

    def __init__(
        self,
        message: str,
        *,
        issues: tuple[ValidationIssue, ...] = (),
        last_output: str | None = None,
    ) -> None:
        super().__init__(message)
        self.issues = issues
        self.last_output = last_output


class BackendError(SlmStrictError):
    """A backend failed to produce output (missing model, runtime failure, exhausted fake, ...)."""
