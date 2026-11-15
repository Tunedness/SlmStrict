import pytest

from slmstrict import Schema, ValidationIssue
from slmstrict.reask import build_reask_prompt
from tests._ladder_support import ITEM_JSON_SCHEMA

ISSUES = (
    ValidationIssue(("count",), "required", "required property is missing"),
    ValidationIssue(("name",), "type", "expected string, got integer"),
)


def test_prompt_holds_original_prompt_previous_output_and_issues() -> None:
    prompt = build_reask_prompt("Extract the item.", '{"name": 1}', ISSUES)
    assert prompt.startswith("Extract the item.")
    assert '{"name": 1}' in prompt
    assert "- $.count: required property is missing" in prompt
    assert "- $.name: expected string, got integer" in prompt
    assert prompt.endswith("do not wrap it in markdown fences.")
    assert prompt.index("Extract the item.") < prompt.index('{"name": 1}') < prompt.index("$.count")


def test_schema_is_included_on_request() -> None:
    schema = Schema.of(ITEM_JSON_SCHEMA)
    with_schema = build_reask_prompt("p", "x", ISSUES, schema=schema)
    assert "Required JSON schema:" in with_schema
    assert '"properties":{' in with_schema
    assert "Required JSON schema:" not in build_reask_prompt("p", "x", ISSUES)


def test_previous_output_is_clipped() -> None:
    prompt = build_reask_prompt("p", "a" * 500, ISSUES, max_output_chars=100)
    assert "a" * 100 in prompt
    assert "a" * 101 not in prompt
    assert "[truncated]" in prompt


def test_issue_count_and_message_length_are_capped() -> None:
    many = tuple(ValidationIssue((i,), "x", "m" * 50) for i in range(15))
    prompt = build_reask_prompt("p", "x", many, max_issues=3, max_message_chars=20)
    assert prompt.count("\n- ") == 4  # three issues plus the "... and N more" line
    assert "... and 12 more" in prompt
    assert "m" * 21 not in prompt


def test_empty_issue_list_still_gives_a_reason() -> None:
    assert "not a valid JSON object" in build_reask_prompt("p", "x", ())


@pytest.mark.parametrize("name", ["max_issues", "max_output_chars", "max_message_chars"])
def test_limits_must_be_positive(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        build_reask_prompt("p", "x", ISSUES, **{name: 0})
