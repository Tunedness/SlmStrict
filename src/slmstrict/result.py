"""Result object returned by the enforcement ladder."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from slmstrict.schema import ValidationIssue

__all__ = ["Attempt", "Result", "Step"]

T = TypeVar("T")


class Step(StrEnum):
    """A rung of the enforcement ladder (ADR-002)."""

    GRAMMAR = "grammar"
    REPAIR = "repair"
    REASK = "reask"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class Attempt:
    """Record of one generation attempt.

    ``step`` is the rung that took effect on ``raw_output``: ``GRAMMAR`` for a constrained
    generation, ``REASK`` for a re-ask, ``FALLBACK`` for a cloud attempt, ``REPAIR`` for an
    unconstrained first generation that the repair pipeline changed, and ``None`` for an
    unconstrained generation that no rung touched. ``repair_steps`` names the deterministic
    repairs applied to it (in order) and ``errors`` holds the validation problems that remained
    after them; an empty ``errors`` means the attempt ended in a valid document.
    """

    step: Step | None
    raw_output: str
    repair_steps: tuple[str, ...] = ()
    errors: tuple[ValidationIssue, ...] = ()

    @property
    def succeeded(self) -> bool:
        """``True`` when the attempt ended in a valid document."""
        return not self.errors


@dataclass(frozen=True, slots=True)
class Result(Generic[T]):
    """A schema-valid value plus the story of how it was obtained.

    ``value`` is a model instance for Pydantic input and a ``dict`` for JSON Schema input.
    ``path`` lists the ladder rungs that actually took effect, each once, in order of first use;
    it is empty when an unconstrained first output was valid as it was. ``attempts`` records every
    attempt, including failed ones.
    """

    value: T
    path: tuple[Step, ...] = ()
    attempts: tuple[Attempt, ...] = ()

    @property
    def used_fallback(self) -> bool:
        """``True`` when the cloud fallback rung produced the value."""
        return Step.FALLBACK in self.path
