"""Shared fakes for the ladder tests."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from slmstrict import (
    Attempt,
    BackendError,
    Constraint,
    FallbackResult,
    Result,
    Schema,
    Step,
)
from slmstrict.hooks import (
    AttemptStarted,
    BaseLadderObserver,
    FallbackEvent,
    LadderFinished,
    StepFinished,
)
from slmstrict.testing import FakeBackend


class Item(BaseModel):
    name: str
    count: int


VALID = '{"name": "apple", "count": 3}'
FENCED = '```json\n{"name": "apple", "count": 3}\n```'
GARBAGE = "I am sorry, I cannot do that."
GRAMMAR = Constraint("gbnf", "root ::= object")

ITEM_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["name", "count"],
}


class GrammarBackend(FakeBackend):
    """FakeBackend that supports grammar and builds a fixed constraint."""

    def __init__(
        self,
        outputs: Sequence[str | BaseException] = (),
        *,
        fail_build: bool = False,
        build_error: BaseException | None = None,
    ) -> None:
        super().__init__(outputs, supports_grammar=True)
        self.fail_build = fail_build
        self.build_error = build_error
        self.built = 0
        self.build_thread: int | None = None

    def build_constraint(self, schema: Schema[Any]) -> Constraint:
        self.built += 1
        self.build_thread = threading.get_ident()
        if self.build_error is not None:
            raise self.build_error
        if self.fail_build:
            raise BackendError("cannot build grammar")
        return GRAMMAR


class SpyHandler:
    """FallbackHandler double: counts calls and returns a canned result."""

    def __init__(
        self,
        value: object | None = None,
        *,
        error: Exception | None = None,
        result: Result[Any] | None = None,
    ) -> None:
        self.calls = 0
        self.seen_attempts: tuple[Attempt, ...] = ()
        self.seen_prompt: str | None = None
        self._error = error
        self._result = result
        self._value = value if value is not None else Item(name="cloud", count=9)

    async def handle(
        self, schema: Schema[Any], prompt: str, attempts: Sequence[Attempt]
    ) -> Result[Any]:
        self.calls += 1
        self.seen_prompt = prompt
        self.seen_attempts = tuple(attempts)
        if self._error is not None:
            raise self._error
        if self._result is not None:
            return self._result
        return FallbackResult(
            value=self._value,
            attempts=(Attempt(step=Step.FALLBACK, raw_output='{"name": "cloud", "count": 9}'),),
        )


class RecordingObserver(BaseLadderObserver):
    def __init__(self) -> None:
        self.events: list[object] = []

    def on_attempt_start(self, event: AttemptStarted) -> None:
        self.events.append(event)

    def on_step_finished(self, event: StepFinished) -> None:
        self.events.append(event)

    def on_fallback(self, event: FallbackEvent) -> None:
        self.events.append(event)

    def on_finish(self, event: LadderFinished) -> None:
        self.events.append(event)

    @property
    def kinds(self) -> list[str]:
        return [type(event).__name__ for event in self.events]
