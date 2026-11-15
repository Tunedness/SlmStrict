import asyncio
import logging
import threading
from typing import Any

import pytest
from pydantic import BaseModel, Field

from slmstrict import (
    Attempt,
    BackendError,
    BaseLadderObserver,
    Constraint,
    FallbackConfig,
    FallbackResult,
    FallbackStatus,
    Result,
    Schema,
    Step,
    StrictConfig,
    TokenUsage,
    ValidationFailed,
)
from slmstrict.hooks import (
    AttemptStarted,
    FallbackEvent,
    LadderFinished,
    StepFinished,
)
from slmstrict.ladder import _WARNED_WITHOUT_BUILDER, Ladder
from slmstrict.testing import FakeBackend
from tests._ladder_support import (
    FENCED,
    GARBAGE,
    GRAMMAR,
    ITEM_JSON_SCHEMA,
    VALID,
    GrammarBackend,
    Item,
    RecordingObserver,
    SpyHandler,
)

SCHEMA = Schema.of(Item)


class Aliased(BaseModel):
    user_name: str = Field(alias="userName")
    visit_count: int = Field(alias="visitCount")


@pytest.fixture(autouse=True)
def _reset_builderless_warning() -> None:
    _WARNED_WITHOUT_BUILDER.clear()


def _with_fallback(handler: SpyHandler, **kwargs: Any) -> StrictConfig:
    return StrictConfig(fallback=FallbackConfig(handler=handler), **kwargs)


# -- path combinations ---------------------------------------------------------------------


async def test_grammar_success_on_first_attempt() -> None:
    backend = GrammarBackend([VALID])
    result = await Ladder(backend).run(SCHEMA, "make an item")
    assert result.value == Item(name="apple", count=3)
    assert result.path == (Step.GRAMMAR,)
    assert [a.step for a in result.attempts] == [Step.GRAMMAR]
    assert result.attempts[0].succeeded
    assert backend.built == 1
    assert [call.constraint for call in backend.calls] == [GRAMMAR]
    assert backend.calls[0].prompt == "make an item"


async def test_unconstrained_valid_output_leaves_the_path_empty() -> None:
    backend = FakeBackend([VALID])
    result = await Ladder(backend).run(SCHEMA, "p")
    assert result.path == ()  # no rung took effect: the raw output was already valid
    assert result.attempts[0].step is None
    assert result.attempts[0].repair_steps == ()
    assert result.attempts[0].succeeded
    assert backend.calls[0].constraint is None


async def test_unconstrained_output_changed_by_repair_reports_repair() -> None:
    result = await Ladder(FakeBackend([FENCED])).run(SCHEMA, "p")
    assert result.path == (Step.REPAIR,)
    assert result.attempts[0].step is Step.REPAIR
    assert result.attempts[0].repair_steps == ("fences",)


async def test_repair_that_changes_nothing_is_not_reported() -> None:
    backend = GrammarBackend([GARBAGE, VALID])
    result = await Ladder(backend).run(SCHEMA, "p")
    assert Step.REPAIR not in result.path  # the pipeline ran on GARBAGE but changed nothing
    assert result.attempts[0].repair_steps == ()


@pytest.mark.parametrize(
    ("output", "expected_step"),
    [
        (FENCED, "fences"),
        ('{"name": "apple", "count": 3,}', "brackets"),
        ('{"name": "apple", "count": "3"}', "coercion"),
    ],
)
async def test_repair_success(output: str, expected_step: str) -> None:
    backend = GrammarBackend([output])
    result = await Ladder(backend).run(SCHEMA, "p")
    assert result.value == Item(name="apple", count=3)
    assert result.path == (Step.GRAMMAR, Step.REPAIR)
    assert expected_step in result.attempts[0].repair_steps
    assert result.attempts[0].step is Step.GRAMMAR
    assert len(backend.calls) == 1


async def test_success_on_first_reask() -> None:
    backend = GrammarBackend([GARBAGE, VALID])
    result = await Ladder(backend).run(SCHEMA, "make an item")
    assert result.path == (Step.GRAMMAR, Step.REASK)
    assert [a.step for a in result.attempts] == [Step.GRAMMAR, Step.REASK]
    assert not result.attempts[0].succeeded
    assert result.attempts[1].succeeded
    reask_prompt = backend.calls[1].prompt
    assert reask_prompt.startswith("make an item")
    assert GARBAGE in reask_prompt  # the previous output is handed back
    assert "Problems found" in reask_prompt
    assert backend.calls[1].constraint == GRAMMAR  # re-asks stay constrained


async def test_success_on_second_reask() -> None:
    backend = FakeBackend([GARBAGE, '{"name": "apple"}', FENCED])
    result = await Ladder(backend).run(SCHEMA, "p")
    assert result.value == Item(name="apple", count=3)
    assert result.path == (Step.REASK, Step.REPAIR)  # repair first took effect on the 2nd re-ask
    assert [a.step for a in result.attempts] == [None, Step.REASK, Step.REASK]
    assert result.attempts[2].repair_steps == ("fences",)
    # the second re-ask is built from the first re-ask's output and problems
    assert '{"name": "apple"}' in backend.calls[2].prompt
    assert "count" in backend.calls[2].prompt


async def test_fallback_takes_over_after_local_failure() -> None:
    backend = GrammarBackend([GARBAGE, GARBAGE, GARBAGE])
    handler = SpyHandler()
    result = await Ladder(backend, _with_fallback(handler)).run(SCHEMA, "make an item")
    assert result.value == Item(name="cloud", count=9)
    assert result.path == (Step.GRAMMAR, Step.REASK, Step.FALLBACK)
    assert result.used_fallback
    assert [a.step for a in result.attempts] == [
        Step.GRAMMAR,
        Step.REASK,
        Step.REASK,
        Step.FALLBACK,
    ]
    assert handler.calls == 1
    assert handler.seen_prompt == "make an item"  # the original prompt, not a re-ask prompt
    assert [a.step for a in handler.seen_attempts] == [Step.GRAMMAR, Step.REASK, Step.REASK]


async def test_total_failure_carries_every_attempt() -> None:
    backend = FakeBackend([GARBAGE, "{}", '{"name": 1}'])
    with pytest.raises(ValidationFailed) as caught:
        await Ladder(backend).run(SCHEMA, "p")
    error = caught.value
    assert [a.raw_output for a in error.attempts] == [GARBAGE, "{}", '{"name": 1}']
    assert error.last_output == '{"name": 1}'
    assert error.issues == error.attempts[-1].errors
    assert error.issues
    assert "3 attempt" in str(error)
    assert GARBAGE not in str(error)  # model output stays out of the message


async def test_dict_schema_yields_a_dict() -> None:
    schema = Schema.of(ITEM_JSON_SCHEMA)
    result = await Ladder(FakeBackend([FENCED])).run(schema, "p")
    assert result.value == {"name": "apple", "count": 3}
    assert result.path == (Step.REPAIR,)
    assert result.attempts[0].step is Step.REPAIR


# -- re-ask bound --------------------------------------------------------------------------


@pytest.mark.parametrize("max_reask", [0, 1, 2, 3])
async def test_reask_count_never_exceeds_max_reask(max_reask: int) -> None:
    backend = FakeBackend([GARBAGE] * 10)
    with pytest.raises(ValidationFailed) as caught:
        await Ladder(backend, StrictConfig(max_reask=max_reask)).run(SCHEMA, "p")
    assert len(backend.calls) == 1 + max_reask
    assert len(caught.value.attempts) == 1 + max_reask
    assert sum(a.step is Step.REASK for a in caught.value.attempts) == max_reask


async def test_default_max_reask_is_two() -> None:
    backend = FakeBackend([GARBAGE] * 10)
    with pytest.raises(ValidationFailed):
        await Ladder(backend).run(SCHEMA, "p")
    assert len(backend.calls) == 3


async def test_max_reask_zero_goes_straight_to_fallback() -> None:
    backend = FakeBackend([GARBAGE])
    handler = SpyHandler()
    result = await Ladder(backend, _with_fallback(handler, max_reask=0)).run(SCHEMA, "p")
    assert result.path == (Step.FALLBACK,)
    assert len(backend.calls) == 1


# -- grammar construction ------------------------------------------------------------------


@pytest.mark.parametrize(
    "error", [BackendError("cannot build"), ImportError("extra not installed")]
)
async def test_failed_grammar_build_degrades_to_unconstrained(
    error: BaseException, caplog: pytest.LogCaptureFixture
) -> None:
    backend = GrammarBackend([VALID], build_error=error)
    with caplog.at_level(logging.WARNING, logger="slmstrict.ladder"):
        result = await Ladder(backend).run(SCHEMA, "p")
    assert result.path == ()
    assert result.attempts[0].step is None
    assert backend.calls[0].constraint is None
    assert "grammar could not be built" in caplog.text


async def test_grammar_backend_without_builder_runs_unconstrained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # no monkeypatching: the ladder must not reach for any grammar compiler of its own
    backend = FakeBackend([VALID], supports_grammar=True)
    with caplog.at_level(logging.WARNING, logger="slmstrict.ladder"):
        result = await Ladder(backend).run(SCHEMA, "p")
    assert result.value == Item(name="apple", count=3)
    assert result.path == ()
    assert backend.calls[0].constraint is None
    assert "no build_constraint" in caplog.text


async def test_missing_builder_is_logged_once_per_backend_class(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="slmstrict.ladder"):
        for _ in range(3):
            backend = FakeBackend([VALID], supports_grammar=True)
            await Ladder(backend).run(SCHEMA, "p")
    assert caplog.text.count("no build_constraint") == 1


async def test_grammar_is_built_off_the_event_loop_thread() -> None:
    backend = GrammarBackend([VALID])
    await Ladder(backend).run(SCHEMA, "p")
    assert backend.build_thread is not None
    assert backend.build_thread != threading.get_ident()


async def test_backend_without_grammar_support_is_never_asked_for_a_builder() -> None:
    class Probe(FakeBackend):
        def build_constraint(self, schema: Schema[Any]) -> Constraint:
            raise AssertionError("must not be called")

    result = await Ladder(Probe([VALID])).run(SCHEMA, "p")
    assert result.path == ()


# -- fallback guarantees -------------------------------------------------------------------


async def test_handler_value_that_does_not_match_the_schema_is_rejected() -> None:
    handler = SpyHandler(value={"name": "cloud"})  # a dict where a model is expected
    observer = RecordingObserver()
    backend = FakeBackend([GARBAGE] * 3)
    with pytest.raises(ValidationFailed, match="does not match the schema"):
        await Ladder(backend, _with_fallback(handler, observer=observer)).run(SCHEMA, "p")
    fallback = next(e for e in observer.events if isinstance(e, FallbackEvent))
    assert fallback.status is FallbackStatus.FAILED


async def test_rejected_fallback_value_still_reports_usage_and_attempts() -> None:
    usage = TokenUsage(input_tokens=50, output_tokens=7)
    bad = FallbackResult(
        value=Item.model_construct(name="cloud"),  # count missing: not a valid Item
        attempts=(Attempt(step=Step.FALLBACK, raw_output="cloud-output"),),
        usage=usage,
        provider="p",
        model="m",
    )
    observer = RecordingObserver()
    config = _with_fallback(SpyHandler(result=bad), observer=observer)
    with pytest.raises(ValidationFailed) as caught:
        await Ladder(FakeBackend([GARBAGE] * 3), config).run(SCHEMA, "p")
    error = caught.value
    assert [a.step for a in error.attempts][-1] is Step.FALLBACK
    assert len(error.attempts) == 4
    assert error.last_output == "cloud-output"  # the handler's, not the last local attempt's
    assert [issue.path for issue in error.issues] == [("count",)]
    assert GARBAGE not in str(error)
    fallback = next(e for e in observer.events if isinstance(e, FallbackEvent))
    assert fallback.status is FallbackStatus.FAILED
    assert fallback.usage == usage
    assert (fallback.provider, fallback.model) == ("p", "m")
    finished = observer.events[-1]
    assert isinstance(finished, LadderFinished)
    assert finished.fallback == fallback
    assert finished.attempts == error.attempts


async def test_aliased_pydantic_fallback_value_is_accepted() -> None:
    aliased = Schema.of(Aliased)
    value = Aliased.model_validate({"userName": "ada", "visitCount": 2})
    handler = SpyHandler(value=value)
    backend = FakeBackend(['{"userName": "x"}'] * 3)
    result = await Ladder(backend, _with_fallback(handler)).run(aliased, "p")
    assert result.value is value
    assert result.used_fallback
    bad = SpyHandler(value=Aliased.model_construct(user_name="ada"))
    with pytest.raises(ValidationFailed):
        await Ladder(FakeBackend(["{}"] * 3), _with_fallback(bad)).run(aliased, "p")


async def test_handler_dict_value_is_validated_for_json_schema_input() -> None:
    schema = Schema.of(ITEM_JSON_SCHEMA)
    good = SpyHandler(value={"name": "cloud", "count": 1})
    result = await Ladder(FakeBackend([GARBAGE] * 3), _with_fallback(good)).run(schema, "p")
    assert result.value == {"name": "cloud", "count": 1}
    bad = SpyHandler(value={"name": "cloud", "count": "x"})
    with pytest.raises(ValidationFailed):
        await Ladder(FakeBackend([GARBAGE] * 3), _with_fallback(bad)).run(schema, "p")


async def test_handler_error_propagates_after_the_event() -> None:
    handler = SpyHandler(error=BackendError("provider down"))
    observer = RecordingObserver()
    backend = FakeBackend([GARBAGE] * 3)
    with pytest.raises(BackendError, match="provider down"):
        await Ladder(backend, _with_fallback(handler, observer=observer)).run(SCHEMA, "p")
    fallback = next(e for e in observer.events if isinstance(e, FallbackEvent))
    assert fallback.status is FallbackStatus.FAILED
    assert fallback.error_type == "BackendError"
    finished = observer.events[-1]
    assert isinstance(finished, LadderFinished)
    assert not finished.succeeded
    assert isinstance(finished.error, BackendError)


async def test_backend_error_is_not_turned_into_a_fallback() -> None:
    handler = SpyHandler()
    backend = FakeBackend([BackendError("model missing")])
    with pytest.raises(BackendError, match="model missing"):
        await Ladder(backend, _with_fallback(handler)).run(SCHEMA, "p")
    assert handler.calls == 0


async def test_plain_result_from_handler_is_accepted() -> None:
    plain: Result[Any] = Result(
        value=Item(name="cloud", count=9),
        attempts=(Attempt(step=Step.FALLBACK, raw_output="{}"),),
    )
    observer = RecordingObserver()
    config = _with_fallback(SpyHandler(result=plain), observer=observer)
    result = await Ladder(FakeBackend([GARBAGE] * 3), config).run(SCHEMA, "p")
    assert result.used_fallback
    fallback = next(e for e in observer.events if isinstance(e, FallbackEvent))
    assert fallback.status is FallbackStatus.USED
    assert fallback.usage is None
    assert fallback.provider is None


# -- observer ------------------------------------------------------------------------------


async def test_observer_sees_every_event_in_order() -> None:
    observer = RecordingObserver()
    backend = GrammarBackend([GARBAGE, VALID])
    result = await Ladder(backend, StrictConfig(observer=observer)).run(SCHEMA, "p")
    assert observer.kinds == [
        "AttemptStarted",
        "StepFinished",
        "AttemptStarted",
        "StepFinished",
        "LadderFinished",
    ]
    first_start, first_end, second_start, second_end, finished = observer.events
    assert isinstance(first_start, AttemptStarted)
    assert (first_start.index, first_start.step, first_start.constrained) == (
        0,
        Step.GRAMMAR,
        True,
    )
    assert first_start.reask_number == 0
    assert first_start.prompt_chars == len("p")
    assert isinstance(first_end, StepFinished)
    assert first_end.repair_ran
    assert not first_end.succeeded
    assert first_end.attempt is result.attempts[0]
    assert isinstance(second_start, AttemptStarted)
    assert (second_start.index, second_start.step, second_start.reask_number) == (1, Step.REASK, 1)
    assert isinstance(second_end, StepFinished)
    assert second_end.succeeded
    assert not second_end.repair_ran
    assert second_end.duration_s >= 0
    assert isinstance(finished, LadderFinished)
    assert finished.succeeded
    assert finished.error is None
    assert finished.path == result.path
    assert finished.attempts == result.attempts
    assert finished.fallback is None
    run_ids = {e.run_id for e in observer.events if hasattr(e, "run_id")}  # type: ignore[attr-defined]
    assert len(run_ids) == 1
    assert {e.schema_hash for e in observer.events} == {SCHEMA.schema_hash}  # type: ignore[attr-defined]


async def test_fallback_event_carries_token_usage_and_provider() -> None:
    usage = TokenUsage(input_tokens=120, output_tokens=30, cached_input_tokens=100)
    cloud = FallbackResult(
        value=Item(name="cloud", count=9),
        attempts=(Attempt(step=Step.FALLBACK, raw_output="{}"),),
        usage=usage,
        provider="anthropic",
        model="some-model",
    )
    observer = RecordingObserver()
    config = _with_fallback(SpyHandler(result=cloud), observer=observer)
    result = await Ladder(FakeBackend([GARBAGE] * 3), config).run(SCHEMA, "p")
    assert result.path[-1] is Step.FALLBACK
    fallback = next(e for e in observer.events if isinstance(e, FallbackEvent))
    assert fallback.status is FallbackStatus.USED
    assert fallback.attempts_before == 3
    assert fallback.usage == usage
    assert fallback.usage.total_tokens == 150
    assert (fallback.provider, fallback.model) == ("anthropic", "some-model")
    assert fallback.duration_s is not None
    finished = observer.events[-1]
    assert isinstance(finished, LadderFinished)
    assert finished.fallback == fallback
    assert finished.succeeded


async def test_failed_run_still_emits_finish_with_the_error() -> None:
    observer = RecordingObserver()
    backend = FakeBackend([GARBAGE] * 3)
    with pytest.raises(ValidationFailed) as caught:
        await Ladder(backend, StrictConfig(observer=observer)).run(SCHEMA, "p")
    finished = observer.events[-1]
    assert isinstance(finished, LadderFinished)
    assert not finished.succeeded
    assert finished.error is caught.value
    assert finished.attempts == caught.value.attempts
    fallback = finished.fallback
    assert fallback is not None
    assert fallback.status is FallbackStatus.NOT_CONFIGURED
    assert fallback.attempts_before == 3


async def test_a_raising_observer_never_breaks_generation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Broken(BaseLadderObserver):
        def on_attempt_start(self, event: AttemptStarted) -> None:
            raise RuntimeError("secret-looking detail")

        def on_finish(self, event: LadderFinished) -> None:
            raise ValueError("boom")

    with caplog.at_level(logging.WARNING, logger="slmstrict.ladder"):
        result = await Ladder(FakeBackend([VALID]), StrictConfig(observer=Broken())).run(
            SCHEMA, "p"
        )
    assert result.value == Item(name="apple", count=3)
    assert "RuntimeError" in caplog.text
    assert "ValueError" in caplog.text
    assert "secret-looking detail" not in caplog.text


async def test_base_observer_is_a_noop() -> None:
    observer = BaseLadderObserver()
    result = await Ladder(FakeBackend([VALID]), StrictConfig(observer=observer)).run(SCHEMA, "p")
    assert result.path == ()


async def test_ladder_does_not_log_content(caplog: pytest.LogCaptureFixture) -> None:
    secret_prompt = "patient-record-7731"
    with caplog.at_level(logging.DEBUG, logger="slmstrict"):
        backend = GrammarBackend([GARBAGE, VALID], fail_build=True)
        await Ladder(backend).run(SCHEMA, secret_prompt)
    assert secret_prompt not in caplog.text
    assert GARBAGE not in caplog.text


# -- config binding ------------------------------------------------------------------------


def test_config_binds_handler_and_observer_types() -> None:
    handler = SpyHandler()
    observer = RecordingObserver()
    config = StrictConfig(fallback=FallbackConfig(handler=handler), observer=observer)
    assert config.fallback is not None
    assert config.fallback.handler is handler
    assert config.observer is observer
    assert FallbackConfig().handler is None
    assert StrictConfig().observer is None


def test_config_rejects_objects_that_do_not_fit_the_protocols() -> None:
    with pytest.raises(TypeError, match="FallbackHandler"):
        FallbackConfig(handler=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LadderObserver"):
        StrictConfig(observer=object())  # type: ignore[arg-type]


def test_token_usage_validates_its_counts() -> None:
    assert TokenUsage(1, 2).total_tokens == 3
    assert TokenUsage(1, 2).cached_input_tokens is None
    with pytest.raises(ValueError, match="input_tokens"):
        TokenUsage(-1, 0)
    with pytest.raises(ValueError, match="output_tokens"):
        TokenUsage(0, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cached_input_tokens"):
        TokenUsage(1, 1, cached_input_tokens=-5)


async def test_cancellation_still_emits_finish() -> None:
    class Hanging(FakeBackend):
        async def agenerate(self, prompt: str, *, constraint: Constraint | None = None) -> str:
            await asyncio.Event().wait()
            return VALID

    observer = RecordingObserver()
    ladder = Ladder(Hanging(), StrictConfig(observer=observer))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(ladder.run(SCHEMA, "p"), timeout=0.05)
    assert observer.kinds == ["AttemptStarted", "LadderFinished"]
    finished = observer.events[-1]
    assert isinstance(finished, LadderFinished)
    assert not finished.succeeded
    assert isinstance(finished.error, asyncio.CancelledError)


def test_config_rejects_a_synchronous_handle() -> None:
    class SyncHandler:
        def handle(self, schema: Schema[Any], prompt: str, attempts: Any) -> Result[Any]:
            raise AssertionError("never called")

    with pytest.raises(TypeError, match="async handle"):
        FallbackConfig(handler=SyncHandler())  # type: ignore[arg-type]
