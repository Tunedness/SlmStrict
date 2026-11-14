import dataclasses

import pytest

from slmstrict import Attempt, Result, Step, ValidationIssue


def test_step_values_and_order() -> None:
    assert [step.value for step in Step] == ["grammar", "repair", "reask", "fallback"]
    assert Step("repair") is Step.REPAIR
    assert str(Step.GRAMMAR) == "grammar"


def test_result_holds_value_path_and_attempts() -> None:
    issue = ValidationIssue(("a",), "type", "expected integer, got string")
    attempts = (
        Attempt(step=Step.GRAMMAR, raw_output='{"a": "1"}', errors=(issue,)),
        Attempt(step=Step.REPAIR, raw_output='{"a": 1}', repair_steps=("coerce",)),
    )
    result = Result(value={"a": 1}, path=(Step.GRAMMAR, Step.REPAIR), attempts=attempts)
    assert result.value == {"a": 1}
    assert result.path == (Step.GRAMMAR, Step.REPAIR)
    assert result.attempts[0].errors == (issue,)
    assert result.attempts[1].repair_steps == ("coerce",)
    assert not result.used_fallback


def test_attempt_succeeded_reflects_errors() -> None:
    assert Attempt(step=Step.GRAMMAR, raw_output="{}").succeeded
    issue = ValidationIssue((), "type", "m")
    assert not Attempt(step=Step.GRAMMAR, raw_output="x", errors=(issue,)).succeeded


def test_used_fallback() -> None:
    assert Result(value=1, path=(Step.REASK, Step.FALLBACK)).used_fallback


def test_result_defaults_are_empty() -> None:
    result = Result(value=1)
    assert result.path == ()
    assert result.attempts == ()


def test_result_is_immutable() -> None:
    result = Result(value=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.value = 2  # type: ignore[misc]


def test_validation_issue_location_and_str() -> None:
    issue = ValidationIssue(("items", 0, "name"), "required", "required property is missing")
    assert issue.location == "$.items[0].name"
    assert str(issue) == "$.items[0].name: required property is missing"
    assert ValidationIssue((), "k", "m").location == "$"
