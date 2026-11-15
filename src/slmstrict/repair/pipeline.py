"""The deterministic repair pipeline (rung 2 of the enforcement ladder, ADR-002).

:func:`repair` runs three pure steps in a fixed order and validates the outcome:

1. ``fences``: cut the JSON object out of fences and surrounding prose,
2. ``brackets``: close unclosed brackets and strings, drop trailing commas,
3. ``coercion``: lossless, schema-directed type conversions.

The pipeline fixes *form* only. It never fills a missing required field, never maps a value onto
an enum member and never makes a value up. When the output cannot be made valid that way,
:attr:`RepairResult.failure` says why and the caller (the ladder) moves on to the next rung.

The pipeline knows nothing about backends or the network; it is a plain function of text and
schema. It never raises for bad *data*, and it keeps model output out of its records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Generic, TypeVar

from slmstrict.repair.brackets import complete_brackets, has_truncated_number
from slmstrict.repair.coercion import Lexeme, coerce
from slmstrict.repair.fences import strip_fences
from slmstrict.schema import Schema, ValidationIssue, ValidationReport

__all__ = ["STEP_NAMES", "RepairFailure", "RepairResult", "RepairStep", "repair"]

T = TypeVar("T")

STEP_NAMES = ("fences", "brackets", "coercion")


class RepairFailure(StrEnum):
    """Why the pipeline could not produce a valid document."""

    NO_JSON = "no_json"
    """The text holds no JSON object at all."""

    INVALID_JSON = "invalid_json"
    """A JSON object was found but is still not parseable after the structural repairs."""

    MULTIPLE_BLOCKS = "multiple_blocks"
    """The text holds several different JSON blocks (an example and an answer, a cut-off one,
    ...). A script cannot tell which one is the answer, so none is picked."""

    TRUNCATED_NUMBER = "truncated_number"
    """The output was cut off in the middle of a number: digits may be missing, so the value
    cannot be trusted whatever the schema says. The ladder re-asks."""

    SCHEMA_MISMATCH = "schema_mismatch"
    """Parseable JSON that still violates the schema (missing field, bad enum value, ...)."""


@dataclass(frozen=True, slots=True)
class RepairStep:
    """Record of one pipeline step: its name, whether it changed anything, and a short note."""

    name: str
    changed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class RepairResult(Generic[T]):
    """Outcome of :func:`repair`.

    ``text`` is the repaired JSON text and ``value`` the parsed object (``None`` when the text is
    not parseable JSON); on failure they hold the best effort. ``steps`` always holds one record
    per step, in pipeline order. ``report`` is the schema validation of ``text``.
    """

    text: str
    value: dict[str, object] | None
    steps: tuple[RepairStep, ...]
    report: ValidationReport[T]
    failure: RepairFailure | None

    @property
    def ok(self) -> bool:
        """``True`` when ``text`` is valid against the schema."""
        return self.failure is None

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        """Validation problems that remain; empty when :attr:`ok`."""
        return self.report.issues

    @property
    def applied(self) -> tuple[str, ...]:
        """Names of the steps that changed something (fits ``Attempt.repair_steps``)."""
        return tuple(step.name for step in self.steps if step.changed)


@dataclass(frozen=True, slots=True)
class _Trial(Generic[T]):
    text: str
    value: dict[str, object] | None
    brackets: str | None
    coercion: tuple[str, ...]
    report: ValidationReport[T]
    truncated_number: bool = False


def repair(text: str, schema: Schema[T]) -> RepairResult[T]:
    """Repair ``text`` (raw model output) so that it validates against ``schema`` if possible."""
    schema_json = schema.json_schema

    cut = strip_fences(text)
    if not cut.found:
        report = schema.validate(text)
        skipped = "skipped: no JSON object to work on"
        steps = (
            RepairStep("fences", False, cut.detail or "no JSON object found"),
            RepairStep("brackets", False, skipped),
            RepairStep("coercion", False, skipped),
        )
        return RepairResult(text, None, steps, report, RepairFailure.NO_JSON)

    if cut.ambiguous:
        report = ValidationReport(None, (ValidationIssue((), "multiple_blocks", cut.detail or ""),))
        skipped = "skipped: no single JSON block to work on"
        steps = (
            RepairStep("fences", False, cut.detail or "several different JSON blocks"),
            RepairStep("brackets", False, skipped),
            RepairStep("coercion", False, skipped),
        )
        return RepairResult(text, None, steps, report, RepairFailure.MULTIPLE_BLOCKS)

    fence_step = RepairStep("fences", cut.detail is not None, cut.detail or "no change")
    trial = _after_fences(cut.text, schema, schema_json)
    brackets_step = RepairStep(
        "brackets", trial.brackets is not None, trial.brackets or "no change"
    )
    if trial.value is None:
        coercion_step = RepairStep("coercion", False, "skipped: document is not valid JSON")
    elif trial.coercion:
        coercion_step = RepairStep("coercion", True, "; ".join(trial.coercion))
    else:
        coercion_step = RepairStep("coercion", False, "no change")

    failure: RepairFailure | None = None
    report = trial.report
    if trial.value is None:
        failure = RepairFailure.INVALID_JSON
    elif trial.truncated_number:
        failure = RepairFailure.TRUNCATED_NUMBER
        issue = ValidationIssue((), "truncated_number", "the output ends in the middle of a number")
        report = ValidationReport(None, (issue, *report.issues))
    elif not report.ok:
        failure = RepairFailure.SCHEMA_MISMATCH
    return RepairResult(
        trial.text,
        trial.value,
        (fence_step, brackets_step, coercion_step),
        report,
        failure,
    )


def _after_fences(text: str, schema: Schema[T], schema_json: dict[str, object]) -> _Trial[T]:
    """Run steps 2 and 3 on one cut-out candidate and validate the result."""
    truncated = has_truncated_number(text)
    bracketed, bracket_detail = complete_brackets(text)
    value = _parse_object(bracketed)
    if value is None:
        report = schema.validate(bracketed)
        if not any(issue.kind == "json_invalid" for issue in report.issues):
            # e.g. an out-of-range float, which the schema validator may not object to
            issue = ValidationIssue((), "json_invalid", "the document is not valid JSON")
            report = ValidationReport(None, (issue, *report.issues))
        return _Trial(bracketed, None, bracket_detail, (), report)

    # Parse again with float spellings kept, so a float can become a string losslessly.
    spelled = _parse_object(bracketed, spelling=True)
    coerced, changes = coerce(spelled if spelled is not None else value, schema_json)
    if not changes or not isinstance(coerced, dict):
        report = schema.validate(bracketed)
        return _Trial(bracketed, value, bracket_detail, (), report, truncated)
    final = json.dumps(coerced, ensure_ascii=False, allow_nan=False)
    report = schema.validate(final)
    return _Trial(final, _parse_object(final), bracket_detail, changes, report, truncated)


def _parse_object(text: str, *, spelling: bool = False) -> dict[str, object] | None:
    parse_float = _parse_spelled_float if spelling else _parse_float
    try:
        parsed: object = json.loads(text, parse_constant=_reject_constant, parse_float=parse_float)
    except (ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_float(text: str) -> float:
    number = float(text)
    if not isfinite(number):
        raise ValueError("number is out of range")
    return number


def _parse_spelled_float(text: str) -> float:
    _parse_float(text)
    return Lexeme(text)


def _reject_constant(name: str) -> object:
    raise ValueError(f"{name} is not valid JSON")
