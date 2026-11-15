"""The enforcement ladder (ADR-002): grammar, repair, re-ask, fallback.

:class:`Ladder` runs one generation request through the four rungs and returns a schema-valid
:class:`~slmstrict.result.Result`, or raises :class:`~slmstrict.errors.ValidationFailed`.

How a run goes:

1. The first attempt is generated with a grammar constraint when the backend supports grammar and
   builds one (an optional ``build_constraint(schema)`` method, see
   :class:`~slmstrict.hooks.ConstraintBuilder`); unconstrained otherwise. The ladder knows no
   grammar format: a backend that claims grammar support but offers no builder, or whose builder
   fails, is logged and run unconstrained.
2. The output is validated. When it does not validate, the deterministic repair pipeline runs.
3. When repair does not help either, the model is re-asked up to ``max_reask`` times with the
   validation problems. A re-ask output goes through validation and repair as well.
4. When all of that fails, the configured fallback handler is called. It is never called when
   ``local_only`` is on or no handler is configured (ADR-003); the run then fails with
   :class:`ValidationFailed`, which carries every attempt.

``Result.path`` lists the rungs that actually took effect, each once, in order of first use:
``GRAMMAR`` when a grammar constraint was applied, ``REPAIR`` when the repair pipeline changed an
output, ``REASK`` when the model was re-asked, ``FALLBACK`` when the fallback produced the value.
An unconstrained first output that is valid as it is leaves the path empty. ``Attempt.step`` follows
the same logic: ``GRAMMAR`` for a constrained generation, ``REASK`` for a re-ask, ``REPAIR`` for an
unconstrained first generation that repair changed, ``None`` for an unconstrained generation that
no rung touched. A backend failure
(:class:`~slmstrict.errors.BackendError`) is not a validation failure: it ends the run and is
re-raised, so a broken backend never turns into a silent cloud call.

Logging goes through ``slmstrict.ladder`` and never contains prompts or outputs (ADR-007).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from slmstrict.backends.base import Backend, Constraint
from slmstrict.config import StrictConfig
from slmstrict.errors import BackendError, ValidationFailed
from slmstrict.hooks import (
    AttemptStarted,
    ConstraintBuilder,
    FallbackEvent,
    FallbackResult,
    FallbackStatus,
    LadderFinished,
    LadderObserver,
    StepFinished,
    TokenUsage,
)
from slmstrict.reask import build_reask_prompt
from slmstrict.repair import repair
from slmstrict.result import Attempt, Result, Step
from slmstrict.schema import Schema, ValidationIssue, ValidationReport

__all__ = ["Ladder"]

T = TypeVar("T")

_logger = logging.getLogger("slmstrict.ladder")

# Backend types already reported as "supports grammar but has no builder": a property of the
# backend class, not a per-call failure, so it is logged once per class.
_WARNED_WITHOUT_BUILDER: set[str] = set()


@dataclass(slots=True)
class _Run:
    """Mutable bookkeeping of one run, shared by the helpers and the final event."""

    run_id: str
    schema_hash: str
    started: float
    path: list[Step] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    fallback: FallbackEvent | None = None

    def add_rung(self, step: Step) -> None:
        if step not in self.path:
            self.path.append(step)


@dataclass(frozen=True, slots=True)
class _Outcome(Generic[T]):
    """What one local attempt came to: the record plus the text and issues the next rung needs."""

    attempt: Attempt
    report: ValidationReport[T]
    text: str

    @property
    def ok(self) -> bool:
        return self.attempt.succeeded


class Ladder:
    """Runs generation requests for one backend and one configuration.

    A ``Ladder`` keeps no state between runs, so one instance serves concurrent runs.
    """

    def __init__(self, backend: Backend, config: StrictConfig | None = None) -> None:
        self._backend = backend
        self._config = config if config is not None else StrictConfig()

    async def run(self, schema: Schema[T], prompt: str) -> Result[T]:
        """Generate a value for ``schema`` from ``prompt`` and return it with its story.

        Raises :class:`ValidationFailed` when no rung produced a valid value and
        :class:`~slmstrict.errors.BackendError` when the backend itself failed.
        """
        run = _Run(uuid.uuid4().hex, schema.schema_hash, time.perf_counter())
        try:
            result = await self._execute(run, schema, prompt)
        except BaseException as exc:  # also cancellation: every run ends with on_finish
            self._finish(run, exc)
            raise
        self._finish(run, None)
        return result

    # -- the four rungs -------------------------------------------------------------------

    async def _execute(self, run: _Run, schema: Schema[T], prompt: str) -> Result[T]:
        constraint = await self._build_constraint(schema)
        first_step = Step.GRAMMAR if constraint is not None else None

        outcome = await self._attempt(run, schema, first_step, prompt, constraint, reask_number=0)
        if outcome.ok:
            return self._success(run, outcome)

        for reask_number in range(1, self._config.max_reask + 1):
            reask_prompt = build_reask_prompt(
                prompt, outcome.text, outcome.report.issues, schema=schema
            )
            outcome = await self._attempt(
                run, schema, Step.REASK, reask_prompt, constraint, reask_number=reask_number
            )
            if outcome.ok:
                return self._success(run, outcome)

        return await self._fallback(run, schema, prompt, outcome)

    async def _attempt(
        self,
        run: _Run,
        schema: Schema[T],
        step: Step | None,
        prompt: str,
        constraint: Constraint | None,
        *,
        reask_number: int,
    ) -> _Outcome[T]:
        """One generation plus validation and, when needed, repair.

        ``step`` is the rung that generates (``None`` for an unconstrained first attempt).
        """
        generating = step
        if step is not None:
            run.add_rung(step)
        index = len(run.attempts)
        self._notify(
            lambda observer: observer.on_attempt_start(
                AttemptStarted(
                    run_id=run.run_id,
                    schema_hash=run.schema_hash,
                    index=index,
                    step=generating,
                    reask_number=reask_number,
                    constrained=constraint is not None,
                    prompt_chars=len(prompt),
                )
            )
        )
        started = time.perf_counter()
        raw = await self._backend.agenerate(prompt, constraint=constraint)

        report = schema.validate(raw)
        repair_ran = not report.ok
        if report.ok:
            outcome = _Outcome(Attempt(step=step, raw_output=raw), report, raw)
        else:
            repaired = repair(raw, schema)
            issues = repaired.issues
            if not repaired.ok and not issues:  # defensive: a failure always names a problem
                failure = repaired.failure.value if repaired.failure else "repair_failed"
                issues = (ValidationIssue((), failure, "the output could not be repaired"),)
            if repaired.applied:
                run.add_rung(Step.REPAIR)
                if step is None:
                    step = Step.REPAIR  # repair is what took effect on this attempt
            attempt = Attempt(
                step=step,
                raw_output=raw,
                repair_steps=repaired.applied,
                errors=() if repaired.ok else issues,
            )
            report = repaired.report if repaired.ok else ValidationReport(None, issues)
            outcome = _Outcome(attempt, report, repaired.text)

        run.attempts.append(outcome.attempt)
        self._notify(
            lambda observer: observer.on_step_finished(
                StepFinished(
                    run_id=run.run_id,
                    schema_hash=run.schema_hash,
                    index=index,
                    step=outcome.attempt.step,
                    attempt=outcome.attempt,
                    repair_ran=repair_ran,
                    succeeded=outcome.ok,
                    duration_s=time.perf_counter() - started,
                )
            )
        )
        return outcome

    async def _fallback(
        self,
        run: _Run,
        schema: Schema[T],
        prompt: str,
        last: _Outcome[T],
    ) -> Result[T]:
        """Rung 4, or the failure that results when the fallback is off (ADR-003)."""
        handler = self._config.fallback.handler if self._config.fallback is not None else None
        attempts_before = len(run.attempts)

        if self._config.local_only:
            self._fallback_event(run, FallbackStatus.DISABLED_LOCAL_ONLY, attempts_before)
            raise self._failure(run, last, "the cloud fallback is disabled by local_only=True")
        if handler is None:
            self._fallback_event(run, FallbackStatus.NOT_CONFIGURED, attempts_before)
            raise self._failure(run, last, "no cloud fallback is configured")

        run.add_rung(Step.FALLBACK)
        started = time.perf_counter()
        try:
            handled = await handler.handle(schema, prompt, tuple(run.attempts))
        except Exception as exc:
            self._fallback_event(
                run,
                FallbackStatus.FAILED,
                attempts_before,
                duration_s=time.perf_counter() - started,
                error_type=type(exc).__name__,
            )
            raise

        duration = time.perf_counter() - started
        usage: TokenUsage | None = None
        provider: str | None = None
        model: str | None = None
        if isinstance(handled, FallbackResult):
            usage, provider, model = handled.usage, handled.provider, handled.model
        run.attempts.extend(handled.attempts)

        problems = _value_issues(schema, handled.value)
        if problems:
            # The provider call was made and paid for: usage and attempts are reported anyway.
            self._fallback_event(
                run,
                FallbackStatus.FAILED,
                attempts_before,
                duration_s=duration,
                usage=usage,
                provider=provider,
                model=model,
                error_type="InvalidFallbackValue",
            )
            raise ValidationFailed(
                "the cloud fallback returned a value that does not match the schema",
                issues=problems,
                last_output=handled.attempts[-1].raw_output if handled.attempts else None,
                attempts=tuple(run.attempts),
            )

        self._fallback_event(
            run,
            FallbackStatus.USED,
            attempts_before,
            duration_s=duration,
            usage=usage,
            provider=provider,
            model=model,
        )
        return Result(value=handled.value, path=tuple(run.path), attempts=tuple(run.attempts))

    # -- helpers --------------------------------------------------------------------------

    async def _build_constraint(self, schema: Schema[T]) -> Constraint | None:
        """The grammar constraint for the generations, or ``None`` when there is none.

        The backend builds it itself; compiling can be slow, so it runs off the event loop.
        """
        if not self._backend.supports_grammar:
            return None
        if not isinstance(self._backend, ConstraintBuilder):
            kind = f"{type(self._backend).__module__}.{type(self._backend).__qualname__}"
            if kind not in _WARNED_WITHOUT_BUILDER:
                _WARNED_WITHOUT_BUILDER.add(kind)
                _logger.warning(
                    "%s supports grammar but has no build_constraint(); "
                    "running without a grammar (reported once per backend class)",
                    kind,
                )
            return None
        try:
            return await asyncio.to_thread(self._backend.build_constraint, schema)
        except (BackendError, ImportError) as exc:
            _logger.warning(
                "grammar could not be built, continuing without it (%s)", type(exc).__name__
            )
            return None

    @staticmethod
    def _success(run: _Run, outcome: _Outcome[T]) -> Result[T]:
        value = cast("T", outcome.report.value)
        return Result(value=value, path=tuple(run.path), attempts=tuple(run.attempts))

    @staticmethod
    def _failure(run: _Run, last: _Outcome[T], reason: str) -> ValidationFailed:
        """The error for a run that ends without a valid value; model output stays out of it."""
        local = sum(1 for attempt in run.attempts if attempt.step is not Step.FALLBACK)
        return ValidationFailed(
            f"no rung produced a valid value after {local} attempt(s): {reason}",
            issues=last.attempt.errors,
            last_output=last.attempt.raw_output,
            attempts=tuple(run.attempts),
        )

    def _fallback_event(
        self,
        run: _Run,
        status: FallbackStatus,
        attempts_before: int,
        *,
        duration_s: float | None = None,
        usage: TokenUsage | None = None,
        provider: str | None = None,
        model: str | None = None,
        error_type: str | None = None,
    ) -> None:
        event = FallbackEvent(
            run_id=run.run_id,
            schema_hash=run.schema_hash,
            status=status,
            attempts_before=attempts_before,
            duration_s=duration_s,
            usage=usage,
            provider=provider,
            model=model,
            error_type=error_type,
        )
        run.fallback = event
        self._notify(lambda observer: observer.on_fallback(event))

    def _finish(self, run: _Run, error: BaseException | None) -> None:
        event = LadderFinished(
            run_id=run.run_id,
            schema_hash=run.schema_hash,
            succeeded=error is None,
            path=tuple(run.path),
            attempts=tuple(run.attempts),
            duration_s=time.perf_counter() - run.started,
            fallback=run.fallback,
            error=error,
        )
        self._notify(lambda observer: observer.on_finish(event))

    def _notify(self, call: Callable[[LadderObserver], None]) -> None:
        """Deliver an event; an observer that raises is logged and never breaks the run."""
        observer = self._config.observer
        if observer is None:
            return
        try:
            call(observer)
        except Exception as exc:
            _logger.warning("a ladder observer failed (%s)", type(exc).__name__)


def _value_issues(schema: Schema[T], value: object) -> tuple[ValidationIssue, ...]:
    """Problems with a fallback handler's value; empty when it is what the schema promises.

    A Pydantic value is checked for its type and then validated again from its dump, so an
    instance built with ``model_construct()`` or changed after construction does not pass. The
    schema (like local generation) speaks in aliases, so the dump is by alias; a model that is
    also valid under its field names (``populate_by_name``) is accepted from that dump as well.
    """
    model = schema.model
    if model is None:
        if not isinstance(value, dict):
            return (ValidationIssue((), "type", "expected a dict for a JSON Schema input"),)
        return schema.validate(value).issues
    if not isinstance(value, model):
        return (ValidationIssue((), "type", f"expected an instance of {model.__name__}"),)
    try:
        by_alias = schema.validate(value.model_dump(mode="json", by_alias=True)).issues
        if not by_alias:
            return ()
        if not schema.validate(value.model_dump(mode="json", by_alias=False)).issues:
            return ()
    except Exception:
        return (ValidationIssue((), "not_json", "the value cannot be serialized"),)
    return by_alias
