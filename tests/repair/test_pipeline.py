import json
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from slmstrict import Schema
from slmstrict.repair import STEP_NAMES, RepairFailure, RepairResult, repair

FIXTURES = Path(__file__).parent / "fixtures"


class Profile(BaseModel):
    name: str
    age: int
    active: bool
    tags: list[str]
    nickname: str | None = None


class Customer(BaseModel):
    name: str
    vip: bool = False


class Item(BaseModel):
    sku: str
    qty: int


class Order(BaseModel):
    customer: Customer
    items: list[Item]
    status: Literal["new", "paid"]


class Friend(BaseModel):
    name: str
    age: int


class Person(BaseModel):
    name: str
    age: int
    friend: Friend | None = None


class Snippet(BaseModel):
    title: str
    code: str


class Count(BaseModel):
    count: int
    label: str


class BigFloat(BaseModel):
    a: int
    big: float | None = None


class OptionalCount(BaseModel):
    label: str
    count: int | None = None


SCORES: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {"type": "object", "additionalProperties": {"type": "integer"}},
    },
    "required": ["scores"],
}

PRICED: dict[str, Any] = {
    "type": "object",
    "properties": {"sku": {"type": "string"}, "price": {"type": "string"}},
    "required": ["sku", "price"],
}

TICKET: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "priority": {"enum": ["low", "medium", "high"]},
        "estimate": {"type": "number"},
        "labels": {"type": "array", "items": {"type": "string"}},
        "assignee": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
            "required": ["name", "id"],
        },
    },
    "required": ["title", "priority"],
}

SCHEMAS: dict[str, Schema[Any]] = {
    "profile": Schema.of(Profile),
    "ticket": Schema.of(TICKET),
    "order": Schema.of(Order),
    "person": Schema.of(Person),
    "snippet": Schema.of(Snippet),
    "count": Schema.of(Count),
    "optcount": Schema.of(OptionalCount),
    "bigfloat": Schema.of(BigFloat),
    "scores": Schema.of(SCORES),
    "priced": Schema.of(PRICED),
}

CASES = sorted(path.stem for path in FIXTURES.glob("*.txt"))


def test_fixture_set_is_large_enough() -> None:
    assert len(CASES) >= 25
    for name in CASES:
        assert (FIXTURES / f"{name}.json").is_file(), f"{name} has no expectation file"


@pytest.mark.parametrize("name", CASES)
def test_fixture(name: str) -> None:
    raw = (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")
    expected = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    schema = SCHEMAS[expected["schema"]]

    result = repair(raw, schema)

    assert [step.name for step in result.steps] == list(STEP_NAMES)
    if expected["ok"]:
        assert result.ok, [str(issue) for issue in result.issues]
        assert result.failure is None
        assert result.issues == ()
        assert result.value is not None
        assert list(result.applied) == expected["applied"]
        # The repaired text itself is valid, and it carries the expected content.
        report = schema.validate(result.text)
        assert report.ok
        parsed = json.loads(result.text)
        for key, value in expected["value"].items():
            assert parsed[key] == value
        assert set(parsed) == set(expected["value"])
    else:
        assert not result.ok
        assert result.failure == RepairFailure(expected["failure"])
        assert not result.report.ok
        assert result.applied == tuple(s.name for s in result.steps if s.changed)
        for kind in expected.get("issue_kinds", []):
            kinds = {issue.kind for issue in result.issues}
            assert kind in kinds or any(kind in k for k in kinds), (kind, kinds)


# -- the three prohibitions, each with its own test ---------------------------------------


def test_never_fills_a_missing_required_field() -> None:
    result = repair('{"name": "Ada", "active": true, "tags": []}', SCHEMAS["profile"])
    assert not result.ok
    assert result.failure is RepairFailure.SCHEMA_MISMATCH
    assert result.value is not None
    assert "age" not in result.value
    assert "age" not in json.loads(result.text)
    assert any(issue.path == ("age",) for issue in result.issues)


@pytest.mark.parametrize("given", ["High", "HIGH", " high", "hi", "urgent", "3"])
def test_never_matches_enum_values_fuzzily(given: str) -> None:
    raw = json.dumps({"title": "T", "priority": given})
    result = repair(raw, SCHEMAS["ticket"])
    assert not result.ok
    assert result.failure is RepairFailure.SCHEMA_MISMATCH
    assert result.value is not None
    assert result.value["priority"] == given
    assert result.steps[2].changed is False


def test_never_makes_up_values() -> None:
    # null stays null, an unparseable string stays a string, absent optional fields stay absent.
    raw = '{"name": "Ada", "age": null, "active": "maybe", "tags": null}'
    result = repair(raw, SCHEMAS["profile"])
    assert not result.ok
    assert result.value == {"name": "Ada", "age": None, "active": "maybe", "tags": None}
    assert result.text == raw
    assert result.applied == ()


# -- pipeline behaviour -------------------------------------------------------------------


def test_clean_input_is_returned_untouched() -> None:
    raw = '{"name": "Ada", "age": 36, "active": true, "tags": []}'
    result = repair(raw, SCHEMAS["profile"])
    assert result.ok
    assert result.text == raw
    assert result.applied == ()
    assert all(step.detail == "no change" for step in result.steps)
    assert isinstance(result.report.value, Profile)


def test_dict_schema_yields_dict_value() -> None:
    result = repair('```json\n{"title": "T", "priority": "low",}\n```', SCHEMAS["ticket"])
    assert result.ok
    assert result.report.value == {"title": "T", "priority": "low"}
    assert result.applied == ("fences", "brackets")


def test_step_records_describe_each_step() -> None:
    raw = '```json\n{"name": "Ada", "age": "36", "active": true, "tags": ["a"'
    result = repair(raw, SCHEMAS["profile"])
    fences, brackets, coercion = result.steps
    assert (fences.name, fences.changed) == ("fences", True)
    assert "fence" in fences.detail
    assert brackets.changed
    assert "closing bracket" in brackets.detail
    assert coercion.changed
    assert coercion.detail == "$.age: string -> integer"
    assert result.applied == ("fences", "brackets", "coercion")


def test_records_do_not_contain_model_output() -> None:
    secret = "s3cr3t-patient-name"
    raw = f'```json\n{{"name": {secret!r}, "age": "36", "active": true, "tags": []'.replace(
        "'", '"'
    )
    result = repair(raw, SCHEMAS["profile"])
    assert result.ok
    assert all(secret not in step.detail for step in result.steps)


def test_no_json_reports_skipped_steps() -> None:
    result = repair("no braces here", SCHEMAS["profile"])
    assert result.failure is RepairFailure.NO_JSON
    assert result.value is None
    assert result.text == "no braces here"
    assert [step.changed for step in result.steps] == [False, False, False]
    assert result.steps[1].detail.startswith("skipped")
    assert result.steps[2].detail.startswith("skipped")


def test_invalid_json_skips_coercion() -> None:
    result = repair("{name: Ada}", SCHEMAS["profile"])
    assert result.failure is RepairFailure.INVALID_JSON
    assert result.value is None
    assert result.steps[2].detail == "skipped: document is not valid JSON"
    assert result.issues[0].kind == "json_invalid"


def test_nan_constant_is_rejected() -> None:
    result = repair('{"name": "Ada", "age": NaN, "active": true, "tags": []}', SCHEMAS["profile"])
    assert result.failure is RepairFailure.INVALID_JSON


def test_deeply_nested_input_does_not_raise() -> None:
    raw = '{"a":' * 5000 + "1" + "}" * 5000
    result = repair(raw, SCHEMAS["profile"])
    assert not result.ok


def test_repair_is_deterministic() -> None:
    raw = '```json\n{"name": "Ada", "age": "36", "active": "true", "tags": "a"'
    first = repair(raw, SCHEMAS["profile"])
    second = repair(raw, SCHEMAS["profile"])
    assert first == second


def test_result_type_is_exported() -> None:
    result: RepairResult[Profile] = repair(
        '{"name": "A", "age": 1, "active": false, "tags": []}', Schema.of(Profile)
    )
    assert result.ok


# -- review round 1 regressions -----------------------------------------------------------


def test_truncated_outer_object_never_yields_an_inner_object() -> None:
    raw = '{"name": "Ali", "friend": {"name": "Veli", "age": 40}, "ag'
    result = repair(raw, SCHEMAS["person"])
    assert result.failure is RepairFailure.SCHEMA_MISMATCH
    assert result.value is not None
    assert result.value["name"] == "Ali"
    assert any(issue.path == ("age",) for issue in result.issues)


@pytest.mark.parametrize("schema_name", ["count", "optcount"])
def test_truncated_number_fails_the_repair(schema_name: str) -> None:
    result = repair('{"label": "x", "count": 12', SCHEMAS[schema_name])
    assert not result.ok
    assert result.failure is RepairFailure.TRUNCATED_NUMBER
    assert not result.report.ok
    assert result.report.value is None
    assert result.issues[0].kind == "truncated_number"
    # The member is neither dropped nor trusted: best effort text keeps it as written.
    assert result.value == {"label": "x", "count": 12}


def test_truncated_number_takes_precedence_over_schema_mismatch() -> None:
    result = repair('{"count": 12', SCHEMAS["count"])  # label is missing as well
    assert result.failure is RepairFailure.TRUNCATED_NUMBER
    assert {issue.kind for issue in result.issues} >= {"truncated_number", "missing"}


def test_number_followed_by_whitespace_is_complete() -> None:
    assert repair('{"label": "x", "count": 12\n', SCHEMAS["count"]).ok


def test_different_blocks_fail_whatever_the_schema_says() -> None:
    raw = '{"label": "a", "count": 1}\n{"label": "b", "count": 2}'
    result = repair(raw, SCHEMAS["count"])
    assert not result.ok
    assert result.failure is RepairFailure.MULTIPLE_BLOCKS
    assert result.value is None
    assert result.text == raw
    assert result.issues[0].kind == "multiple_blocks"
    assert [step.changed for step in result.steps] == [False, False, False]
    assert result.steps[1].detail.startswith("skipped")


def test_stray_braces_in_prose_count_as_a_second_block() -> None:
    result = repair('Use {name} here: {"label": "a", "count": 1}', SCHEMAS["count"])
    assert result.failure is RepairFailure.MULTIPLE_BLOCKS


def test_example_then_truncated_answer_never_returns_the_example() -> None:
    raw = 'Example: {"name": "Veli", "age": 40}\nAnswer: {"name": "Ali", "ag'
    result = repair(raw, SCHEMAS["person"])
    assert result.failure is RepairFailure.MULTIPLE_BLOCKS
    assert result.value is None


@pytest.mark.parametrize("coerced", [True, False])
def test_non_finite_floats_are_a_failure_not_an_exception(coerced: bool) -> None:
    raw = '{"a": "5", "big": 1e400}' if coerced else '{"a": 5, "big": 1e400}'
    result = repair(raw, SCHEMAS["bigfloat"])
    assert result.failure is RepairFailure.INVALID_JSON
    assert result.value is None
    assert not result.report.ok
    assert result.steps[2].detail == "skipped: document is not valid JSON"


def test_non_finite_float_with_negative_exponent_and_big_ints() -> None:
    assert repair('{"a": 1, "big": -1e999}', SCHEMAS["bigfloat"]).failure is (
        RepairFailure.INVALID_JSON
    )
    assert repair('{"a": 1, "big": 1e-999}', SCHEMAS["bigfloat"]).ok  # underflows to 0.0: finite
    assert repair('{"a": ' + "9" * 5000 + "}", SCHEMAS["bigfloat"]).failure is (
        RepairFailure.INVALID_JSON
    )


def test_number_before_a_closing_fence_line_is_complete_but_glued_fence_is_not() -> None:
    assert repair('```json\n{"label": "x", "count": 12\n```', SCHEMAS["count"]).ok
    glued = repair('{"label": "x", "count": 12```', SCHEMAS["optcount"])
    assert glued.failure is RepairFailure.TRUNCATED_NUMBER
    assert glued.value == {"label": "x", "count": 12}


def test_json_with_fence_markers_in_a_string_is_untouched() -> None:
    raw = '{"title": "t", "code": "```py\\nprint(1)\\n```"}'
    result = repair(raw, SCHEMAS["snippet"])
    assert result.ok
    assert result.text == raw
    assert result.applied == ()


def test_coercion_record_keeps_model_keys_out() -> None:
    result = repair('{"scores": {"Ayse Yilmaz": "34"}}', SCHEMAS["scores"])
    assert result.ok
    assert result.steps[2].detail == "$.scores.*: string -> integer"
    assert "Ayse" not in " ".join(step.detail for step in result.steps)


def test_result_value_holds_plain_floats() -> None:
    result = repair('{"sku": "A1", "price": 1.50}', SCHEMAS["priced"])
    assert result.value == {"sku": "A1", "price": "1.50"}
    plain = repair('{"title": "T", "priority": "low", "estimate": "2.5"}', SCHEMAS["ticket"])
    assert plain.value is not None
    assert type(plain.value["estimate"]) is float
