import json
from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel, Field

from slmstrict import Schema, SchemaError


class Address(BaseModel):
    city: str
    zip_code: str | None = None


class User(BaseModel):
    """A user."""

    name: str = Field(description="Full name")
    age: int = 0
    tags: list[str]
    address: Address | None = None


USER_DICT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "address": {"anyOf": [{"$ref": "#/$defs/Address"}, {"type": "null"}]},
    },
    "required": ["tags", "name"],
    "$defs": {
        "Address": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "zip_code": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["city"],
        }
    },
}


# -- hashing ------------------------------------------------------------------------------


def test_model_and_equivalent_dict_share_hash() -> None:
    assert Schema.of(User).schema_hash == Schema.of(USER_DICT).schema_hash


def test_model_json_schema_as_dict_shares_hash() -> None:
    assert Schema.of(User.model_json_schema()).schema_hash == Schema.of(User).schema_hash


def test_hash_is_stable_sha256_hex() -> None:
    digest = Schema.of(User).schema_hash
    assert len(digest) == 64
    assert digest == Schema.of(User).schema_hash
    int(digest, 16)


def test_hash_ignores_annotations_and_key_order() -> None:
    a = Schema.of({"type": "object", "properties": {"x": {"type": "integer", "title": "X"}}})
    b = Schema.of({"properties": {"x": {"type": "integer"}}, "description": "d", "type": "object"})
    assert a.schema_hash == b.schema_hash


def test_hash_differs_when_validity_differs() -> None:
    a = Schema.of({"type": "object", "properties": {"x": {"type": "integer"}}})
    b = Schema.of({"type": "object", "properties": {"x": {"type": "string"}}})
    c = Schema.of({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]})
    assert len({a.schema_hash, b.schema_hash, c.schema_hash}) == 3


def test_property_named_like_an_annotation_is_kept() -> None:
    schema = Schema.of(
        {
            "type": "object",
            "properties": {"title": {"type": "string"}, "default": {"type": "integer"}},
            "required": ["title"],
        }
    )
    assert set(schema.json_schema["properties"]) == {"title", "default"}
    assert not schema.validate('{"default": 1}').ok


def test_enum_values_are_data_not_schema() -> None:
    schema = Schema.of({"type": "object", "properties": {"k": {"enum": [{"title": "a"}]}}})
    assert schema.json_schema["properties"]["k"]["enum"] == [{"title": "a"}]


def test_json_schema_property_returns_copy() -> None:
    schema = Schema.of(USER_DICT)
    schema.json_schema["required"].append("hacked")
    assert "hacked" not in schema.json_schema["required"]


def test_input_dict_is_not_aliased() -> None:
    source: dict[str, Any] = {"type": "object", "properties": {"x": {"type": "integer"}}}
    schema = Schema.of(source)
    before = schema.schema_hash
    source["properties"]["x"]["type"] = "string"
    assert schema.schema_hash == before
    assert schema.validate('{"x": 1}').ok


def test_canonical_json_matches_normalized_schema() -> None:
    schema = Schema.of(USER_DICT)
    assert json.loads(schema.canonical_json) == schema.json_schema


# -- factory ------------------------------------------------------------------------------


def test_of_rejects_unsupported_sources() -> None:
    with pytest.raises(SchemaError):
        Schema.of(42)  # type: ignore[call-overload]
    with pytest.raises(SchemaError):
        Schema.of(int)  # type: ignore[type-var]


def test_model_property() -> None:
    assert Schema.of(User).model is User
    assert Schema.of(USER_DICT).model is None


# -- pydantic validation ------------------------------------------------------------------


def test_model_validate_success_returns_instance() -> None:
    report = Schema.of(User).validate('{"name": "Ada", "tags": ["a"]}')
    assert report.ok
    assert isinstance(report.value, User)
    assert report.value.name == "Ada"


def test_model_validate_accepts_parsed_object() -> None:
    report = Schema.of(User).validate({"name": "Ada", "tags": [], "address": {"city": "X"}})
    assert report.ok
    assert report.value is not None
    assert report.value.address == Address(city="X")


def test_model_validate_reports_structured_issues() -> None:
    report = Schema.of(User).validate('{"age": "x", "tags": [1]}')
    assert not report.ok
    assert report.value is None
    paths = {issue.path for issue in report.issues}
    assert paths == {("name",), ("age",), ("tags", 0)}
    by_path = {issue.path: issue for issue in report.issues}
    assert by_path[("name",)].kind == "missing"
    assert by_path[("tags", 0)].location == "$.tags[0]"


def test_model_validation_is_strict_no_coercion() -> None:
    report = Schema.of(User).validate('{"name": "Ada", "age": "5", "tags": []}')
    assert not report.ok
    assert report.issues[0].path == ("age",)


def test_model_validate_invalid_json() -> None:
    report = Schema.of(User).validate("not json")
    assert not report.ok
    assert report.issues[0].kind == "json_invalid"
    assert report.issues[0].path == ()


def test_model_validate_non_json_object() -> None:
    report = Schema.of(User).validate({"name": {1, 2}})
    assert not report.ok
    assert report.issues[0].kind == "not_json"


# -- json schema validation ---------------------------------------------------------------


def test_dict_validate_success_returns_dict() -> None:
    report = Schema.of(USER_DICT).validate(
        '{"name": "Ada", "tags": ["a"], "age": 3, "address": {"city": "X", "zip_code": null}}'
    )
    assert report.ok
    assert report.value == {
        "name": "Ada",
        "tags": ["a"],
        "age": 3,
        "address": {"city": "X", "zip_code": None},
    }


def test_dict_validate_reports_all_issues_with_paths() -> None:
    report = Schema.of(USER_DICT).validate('{"age": "5", "tags": [1], "address": {}}')
    assert not report.ok
    found = {(issue.location, issue.kind) for issue in report.issues}
    assert ("$.name", "required") in found
    assert ("$.age", "type") in found
    assert ("$.tags[0]", "type") in found
    assert ("$.address", "any_of") in found
    assert str(report.issues[0]).startswith("$")


def test_dict_validate_invalid_json_and_non_object() -> None:
    schema = Schema.of(USER_DICT)
    assert schema.validate("{oops").issues[0].kind == "json_invalid"
    assert schema.validate("").issues[0].kind == "json_invalid"
    assert schema.validate("[1]").issues[0].kind == "type"
    assert schema.validate('{"name": NaN}').issues[0].kind == "json_invalid"


def test_dict_validate_accepts_parsed_object() -> None:
    assert Schema.of(USER_DICT).validate({"name": "a", "tags": []}).ok
    bad = Schema.of(USER_DICT).validate({"name": float("nan"), "tags": []})
    assert bad.issues[0].kind == "not_json"


@pytest.mark.parametrize(
    ("prop", "good", "bad"),
    [
        ({"type": "integer"}, "3", "3.5"),
        ({"type": "integer"}, "3", "3.0"),
        ({"type": "integer"}, "3", "true"),
        ({"type": "number"}, "3.5", '"3"'),
        ({"type": "boolean"}, "true", "1"),
        ({"type": "null"}, "null", "0"),
        ({"type": ["string", "null"]}, "null", "1"),
        ({"enum": ["a", 1]}, "1", "true"),
        ({"const": {"a": [1]}}, '{"a": [1]}', '{"a": [true]}'),
        ({"type": "string", "minLength": 2}, '"ab"', '"a"'),
        ({"type": "string", "maxLength": 2}, '"ab"', '"abc"'),
        ({"type": "string", "pattern": "^[a-z]+$"}, '"abc"', '"ab1"'),
        ({"type": "number", "minimum": 1}, "1", "0.5"),
        ({"type": "number", "maximum": 1}, "1", "1.5"),
        ({"type": "number", "exclusiveMinimum": 1}, "1.1", "1"),
        ({"type": "number", "exclusiveMaximum": 1}, "0.9", "1"),
        ({"type": "number", "multipleOf": 0.1}, "0.3", "0.35"),
        ({"type": "array", "minItems": 1}, "[1]", "[]"),
        ({"type": "array", "maxItems": 1}, "[1]", "[1, 2]"),
        ({"type": "array", "uniqueItems": True}, "[1, 2]", "[1, 1.0]"),
        ({"type": "array", "uniqueItems": True}, "[1, true]", "[[1], [1]]"),
        (
            {"type": "array", "prefixItems": [{"type": "string"}], "items": False},
            '["a"]',
            '["a", 1]',
        ),
        (
            {"type": "array", "items": [{"type": "string"}], "additionalItems": False},
            '["a"]',
            "[1]",
        ),
        (
            {"type": "object", "additionalProperties": False, "properties": {"a": {}}},
            '{"a": 1}',
            '{"b": 1}',
        ),
        ({"type": "object", "additionalProperties": {"type": "integer"}}, '{"b": 1}', '{"b": "x"}'),
        (
            {"type": "object", "patternProperties": {"^x_": {"type": "integer"}}},
            '{"x_a": 1}',
            '{"x_a": "s"}',
        ),
        ({"type": "object", "minProperties": 1}, '{"a": 1}', "{}"),
        ({"type": "object", "maxProperties": 1}, '{"a": 1}', '{"a": 1, "b": 2}'),
        ({"oneOf": [{"type": "integer"}, {"type": "string"}]}, "1", "1.5"),
        ({"oneOf": [{"type": "integer"}, {"type": "number"}]}, "1.5", "1"),
        ({"allOf": [{"type": "integer"}, {"minimum": 2}]}, "3", "1"),
        ({"not": {"type": "string"}}, "1", '"a"'),
        (False, "", "1"),
    ],
)
def test_dict_keyword_support(prop: Any, good: str, bad: str) -> None:
    schema = Schema.of({"type": "object", "properties": {"v": prop}})
    if good:
        assert schema.validate(f'{{"v": {good}}}').ok, good
    assert not schema.validate(f'{{"v": {bad}}}').ok, bad


def test_dict_validate_recursive_ref() -> None:
    schema = Schema.of(
        {
            "$ref": "#/$defs/Node",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}
                    },
                    "required": ["children"],
                }
            },
        }
    )
    assert schema.validate('{"children": [{"children": []}]}').ok
    bad = schema.validate('{"children": [{"children": [{}]}]}')
    assert bad.issues[0].location == "$.children[0].children[0].children"


def test_dict_validate_allof_cycle_hits_recursion_limit() -> None:
    schema = Schema.of({"allOf": [{"$ref": "#"}]})
    assert schema.validate("{}").issues[0].kind == "recursion_limit"


# -- schema errors ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "array"},
        {"type": "object", "properties": {"a": {"type": "strng"}}},
        {"type": "object", "properties": {"a": 5}},
        {"type": "object", "properties": []},
        {"type": "object", "required": "a"},
        {"type": "object", "patternProperties": [1]},
        {"type": "object", "patternProperties": "ab"},
        {"type": "object", "properties": {"a": {"pattern": "("}}},
        {"type": "object", "properties": {"a": {"minLength": -1}}},
        {"type": "object", "properties": {"a": {"minimum": "1"}}},
        {"type": "object", "properties": {"a": {"multipleOf": 0}}},
        {"type": "object", "properties": {"a": {"enum": "x"}}},
        {"type": "object", "properties": {"a": {"anyOf": []}}},
        {"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}},
        {"type": "object", "properties": {"a": {"$ref": "http://example.com/s.json"}}},
        {"type": "object", "properties": {"a": {"$ref": "#/type"}}},
        {"$ref": "#/$defs/A", "$defs": {"A": {"$ref": "#/$defs/A"}}},
        {"type": "object", "if": {"required": ["a"]}, "then": {"required": ["b"]}},
        {"type": "object", "properties": {"a": {"contains": {"type": "string"}}}},
        {"type": "object", "default": float("nan")},
        {"type": "object", "properties": {"a": {"enum": [object()]}}},
    ],
)
def test_invalid_schema_raises_schema_error(bad: Any) -> None:
    with pytest.raises(SchemaError):
        Schema.of(bad)


def test_model_with_unresolvable_annotation_raises_schema_error() -> None:
    class Broken(BaseModel):
        x: "Undefined"  # type: ignore[name-defined]  # noqa: F821

    with pytest.raises(SchemaError):
        Schema.of(Broken)


def test_repr_is_short() -> None:
    assert "User" in repr(Schema.of(User))
    assert "json-schema" in repr(Schema.of(USER_DICT))


def test_format_is_kept_in_schema_and_hash() -> None:
    class AsString(BaseModel):
        x: str

    class AsDatetime(BaseModel):
        x: datetime

    assert Schema.of(AsString).schema_hash != Schema.of(AsDatetime).schema_hash
    assert Schema.of(AsDatetime).json_schema["properties"]["x"]["format"] == "date-time"
    plain = Schema.of({"type": "object", "properties": {"x": {"type": "string"}}})
    dated = Schema.of(
        {
            "type": "object",
            "properties": {"x": {"type": "string", "format": "date-time"}},
            "required": ["x"],
        }
    )
    assert plain.schema_hash != dated.schema_hash
    assert dated.schema_hash == Schema.of(AsDatetime).schema_hash


def test_model_and_dict_paths_agree_on_integral_floats() -> None:
    from_model = Schema.of(User).validate('{"name": "A", "age": 5.0, "tags": []}')
    from_dict = Schema.of(USER_DICT).validate('{"name": "A", "age": 5.0, "tags": []}')
    assert not from_model.ok
    assert not from_dict.ok
