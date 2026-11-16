"""Extension points of the enforcement ladder (ADR-002, ADR-003).

Two ``typing.Protocol`` types let later components plug into the ladder without touching
``ladder.py``:

* :class:`FallbackHandler` is rung 4. The cloud fallback implementation provides it and hands it
  to the ladder through :attr:`slmstrict.config.FallbackConfig.handler`.
* :class:`LadderObserver` receives the events of one run (attempt start, rung outcome, fallback
  decision, finish). Metrics and tracing implement it and register it through
  :attr:`slmstrict.config.StrictConfig.observer`.

Observer events are plain frozen dataclasses. They carry meta data (rung, counts, durations,
token usage) and, in :class:`StepFinished`, the :class:`~slmstrict.result.Attempt` record, which
holds model output. Observers that log must follow ADR-007 and keep content out of the logs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable

from slmstrict.backends.base import Constraint
from slmstrict.result import Attempt, Result, Step
from slmstrict.schema import Schema

__all__ = [
    "AttemptStarted",
    "BaseLadderObserver",
    "ConstraintBuilder",
    "FallbackEvent",
    "FallbackHandler",
    "FallbackResult",
    "FallbackStatus",
    "LadderFinished",
    "LadderObserver",
    "StepFinished",
    "TokenUsage",
]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts reported by a cloud provider for one fallback call.

    ``cached_input_tokens`` is the part of ``input_tokens`` the provider served from a prompt
    cache, ``None`` when the provider does not report it. When a handler makes several provider
    calls for one fallback (retries, ...) it reports the sum.
    """

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cached_input_tokens"):
            value = getattr(self, name)
            if value is None and name == "cached_input_tokens":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class FallbackResult(Result[T]):
    """A :class:`~slmstrict.result.Result` that also tells where the value came from.

    A :class:`FallbackHandler` may return this subclass instead of a plain ``Result`` to pass the
    provider's token usage (and its identity) on to observers. ``usage`` is ``None`` when the
    provider did not report any, which cost estimation must treat as "unknown".
    """

    usage: TokenUsage | None = None
    provider: str | None = None
    model: str | None = None


@runtime_checkable
class ConstraintBuilder(Protocol):
    """Optional capability of a grammar-capable backend: it compiles the constraint itself.

    The ladder knows no grammar format. A backend whose ``supports_grammar`` is true offers
    ``build_constraint`` and returns the opaque :class:`~slmstrict.backends.base.Constraint` its
    own ``agenerate`` understands (``LlamaCppBackend`` does). The call may be slow on a cache miss,
    so the ladder runs it in a worker thread. Without a builder the ladder runs unconstrained.
    """

    def build_constraint(self, schema: Schema[T]) -> Constraint: ...


@runtime_checkable
class FallbackHandler(Protocol):
    """Rung 4 of the ladder: produce a schema-valid value somewhere else (ADR-003).

    The ladder calls :meth:`handle` only after grammar, repair and every re-ask failed, only when
    a handler is configured and only when ``local_only`` is off; the handler never has to check
    those itself.

    Contract:

    * ``prompt`` is the original, unmodified prompt. ``attempts`` are the local attempts so far,
      in order, including their raw outputs; masking either before it leaves the device is the
      handler's job.
    * The returned ``value`` must be valid against ``schema``: a model instance for a Pydantic
      schema, a ``dict`` for a JSON Schema. The ladder rejects anything else.
    * ``Result.attempts`` holds the handler's *own* attempts (``Step.FALLBACK``); the ladder
      appends them to the local ones and builds the final ``path`` itself, so the handler leaves
      ``Result.path`` empty.
    * Return a :class:`FallbackResult` to report token usage. Failures are raised; the ladder
      lets them propagate after notifying observers.
    * ``generate()`` runs every call on a fresh event loop, so a handler must not keep loop-bound
      resources (an async HTTP client, ...) across calls; create them per call or per loop.
    """

    async def handle(
        self,
        schema: Schema[T],
        prompt: str,
        attempts: Sequence[Attempt],
    ) -> Result[T]: ...


class FallbackStatus(StrEnum):
    """What happened when the local rungs were exhausted."""

    USED = "used"
    """The handler was called and returned a value."""

    FAILED = "failed"
    """The handler was called and raised, or returned a value that does not match the schema
    (``usage``, ``provider`` and ``model`` are still reported in the second case, and in the first
    when the raised exception carries them as ``usage``, ``provider`` and ``model`` attributes)."""

    DISABLED_LOCAL_ONLY = "disabled_local_only"
    """The fallback was needed but ``local_only`` is on; the handler was not called."""

    NOT_CONFIGURED = "not_configured"
    """The fallback was needed but none is configured; the handler was not called."""


@dataclass(frozen=True, slots=True)
class AttemptStarted:
    """A local generation attempt is about to start.

    ``index`` counts attempts of this run from 0. ``step`` is the rung that generates:
    ``GRAMMAR`` for a constrained first attempt, ``REASK`` for a re-ask and ``None`` for an
    unconstrained first attempt (no rung is involved yet). ``reask_number`` is 0 for the
    first attempt and 1..``max_reask`` for re-asks. ``prompt_chars`` is the length of the prompt
    sent to the backend; the prompt itself is not part of the event.
    """

    run_id: str
    schema_hash: str
    index: int
    step: Step | None
    reask_number: int
    constrained: bool
    prompt_chars: int


@dataclass(frozen=True, slots=True)
class StepFinished:
    """A local attempt is over: generated, validated and, when needed, repaired.

    ``step`` is ``attempt.step``. ``attempt`` is the full record (it holds the raw output).
    ``repair_ran`` says whether the deterministic repair pipeline ran on the output, which is the
    case exactly when the raw output did not validate as it was (whether it changed anything shows
    in ``attempt.repair_steps``). ``succeeded`` equals ``attempt.succeeded``. ``duration_s`` covers
    generation, validation and repair.
    """

    run_id: str
    schema_hash: str
    index: int
    step: Step | None
    attempt: Attempt
    repair_ran: bool
    succeeded: bool
    duration_s: float


@dataclass(frozen=True, slots=True)
class FallbackEvent:
    """The local rungs were exhausted and the ladder decided what to do about the fallback.

    Emitted exactly once per run in which that happens, whatever the decision (see
    :class:`FallbackStatus`); a run that succeeds locally emits none. ``attempts_before`` is the
    number of local attempts made. ``duration_s``, ``usage``, ``provider`` and ``model`` are only
    set when the handler was called and returned a :class:`FallbackResult` or raised an exception
    that carries them (``duration_s`` is set whenever it was called). ``error_type`` is the class
    name of the failure for ``FAILED``.
    """

    run_id: str
    schema_hash: str
    status: FallbackStatus
    attempts_before: int
    duration_s: float | None = None
    usage: TokenUsage | None = None
    provider: str | None = None
    model: str | None = None
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class LadderFinished:
    """One run is over, successfully or not.

    ``path`` and ``attempts`` are those of the :class:`~slmstrict.result.Result` on success and
    what was done so far on failure. ``error`` is the exception that ends the run (``None`` on
    success); it is re-raised to the caller after this event. Cancellation ends a run as well, with
    ``CancelledError`` as ``error``. ``fallback`` is the
    :class:`FallbackEvent` of the run, when there was one.
    """

    run_id: str
    schema_hash: str
    succeeded: bool
    path: tuple[Step, ...]
    attempts: tuple[Attempt, ...]
    duration_s: float
    fallback: FallbackEvent | None = None
    error: BaseException | None = None


@runtime_checkable
class LadderObserver(Protocol):
    """Receives the events of every ladder run, in order.

    Calls are synchronous and happen on the event loop, so implementations stay light (increment
    a counter, append to a buffer). An exception raised by an observer is logged and swallowed:
    observing never breaks generation.

    For one run the order is: ``on_attempt_start`` / ``on_step_finished`` pairs, then at most one
    ``on_fallback``, then ``on_finish``.
    """

    def on_attempt_start(self, event: AttemptStarted) -> None: ...

    def on_step_finished(self, event: StepFinished) -> None: ...

    def on_fallback(self, event: FallbackEvent) -> None: ...

    def on_finish(self, event: LadderFinished) -> None: ...


class BaseLadderObserver:
    """No-op :class:`LadderObserver`: subclass it and override only the events you need."""

    def on_attempt_start(self, event: AttemptStarted) -> None:
        return None

    def on_step_finished(self, event: StepFinished) -> None:
        return None

    def on_fallback(self, event: FallbackEvent) -> None:
        return None

    def on_finish(self, event: LadderFinished) -> None:
        return None
