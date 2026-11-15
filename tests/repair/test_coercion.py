from typing import Any

import pytest

from slmstrict.repair.coercion import Lexeme, coerce


def run(value: object, schema: dict[str, Any]) -> object:
    return coerce(value, schema)[0]


def obj(**properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties}


@pytest.mark.parametrize(
    ("value", "schema", "expected"),
    [
        ("12", {"type": "integer"}, 12),
        (" -7 ", {"type": "integer"}, -7),
        ("+3", {"type": "integer"}, 3),
        ("12.0", {"type": "integer"}, 12),
        (12.0, {"type": "integer"}, 12),
        ("2.5", {"type": "number"}, 2.5),
        ("1e3", {"type": "number"}, 1000.0),
        (".5", {"type": "number"}, 0.5),
        ("7", {"type": "number"}, 7),
        ("true", {"type": "boolean"}, True),
        (" False ", {"type": "boolean"}, False),
        (42, {"type": "string"}, "42"),
        (Lexeme("1.50"), {"type": "string"}, "1.50"),
        (Lexeme("1e16"), {"type": "string"}, "1e16"),
    ],
)
def test_scalar_conversions(value: object, schema: dict[str, Any], expected: object) -> None:
    result, changes = coerce(value, schema)
    assert result == expected
    assert type(result) is type(expected)
    assert len(changes) == 1


@pytest.mark.parametrize(
    ("value", "schema"),
    [
        ("12.5", {"type": "integer"}),
        (12.5, {"type": "integer"}),
        (float("inf"), {"type": "integer"}),
        ("1,234", {"type": "integer"}),
        ("twelve", {"type": "integer"}),
        ("", {"type": "integer"}),
        ("9" * 5000, {"type": "integer"}),
        ("9" * 5000, {"type": "number"}),
        ("nan", {"type": "number"}),
        ("inf", {"type": "number"}),
        ("1e999", {"type": "number"}),
        ("1,5", {"type": "number"}),
        (5, {"type": "number"}),
        ("yes", {"type": "boolean"}),
        ("1", {"type": "boolean"}),
        (1, {"type": "boolean"}),
        (None, {"type": "integer"}),
        (None, {"type": "string"}),
        (True, {"type": "integer"}),
        (True, {"type": "string"}),
        (False, {"type": "number"}),
        (float("nan"), {"type": "string"}),
        (2.5, {"type": "string"}),  # a float without its spelling cannot be stringified losslessly
        ("12", {"type": "null"}),
        ("12", {"type": "string"}),
        ("12", {}),
        ("12", True),
    ],
)
def test_no_conversion(value: object, schema: Any) -> None:
    result, changes = coerce(value, schema)
    assert result is value or result == value
    assert changes == ()


def test_single_value_becomes_a_list_and_items_are_coerced() -> None:
    schema = {"type": "array", "items": {"type": "integer"}}
    assert coerce("5", schema) == (
        [5],
        ("$: string -> array (wrapped in a list)", "$[0]: string -> integer"),
    )
    assert run({"a": 1}, {"type": "array"}) == [{"a": 1}]
    assert run(["1", "2"], schema) == [1, 2]


def test_null_is_not_wrapped() -> None:
    assert coerce(None, {"type": "array", "items": {"type": "string"}}) == (None, ())


def test_prefix_items_and_items() -> None:
    schema = {
        "type": "array",
        "prefixItems": [{"type": "integer"}, {"type": "boolean"}],
        "items": {"type": "string"},
    }
    assert run(["1", "true", 3], schema) == [1, True, "3"]


def test_object_properties_and_additional_properties() -> None:
    schema = {
        "type": "object",
        "properties": {"age": {"type": "integer"}},
        "additionalProperties": {"type": "boolean"},
    }
    value = {"age": "3", "flag": "true", "other": 1}
    assert run(value, schema) == {"age": 3, "flag": True, "other": 1}
    assert value == {"age": "3", "flag": "true", "other": 1}  # the input is not mutated


def test_unknown_properties_are_kept_and_nothing_is_added() -> None:
    schema = obj(age={"type": "integer"})
    schema["required"] = ["age", "name"]
    assert run({"extra": "x"}, schema) == {"extra": "x"}


def test_type_is_inferred_from_keywords() -> None:
    assert run({"a": "1"}, {"properties": {"a": {"type": "integer"}}}) == {"a": 1}
    assert run({"a": "1"}, {"additionalProperties": {"type": "integer"}}) == {"a": 1}
    assert run(["1"], {"items": {"type": "integer"}}) == [1]
    assert run(["1"], {"prefixItems": [{"type": "integer"}]}) == [1]


def test_non_container_values_are_left_alone_for_object_schemas() -> None:
    assert run("x", obj(a={"type": "integer"})) == "x"


def test_refs_are_followed() -> None:
    schema = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}, "list": {"$ref": "#/$defs/List"}},
        "$defs": {
            "Item": {"type": "object", "properties": {"qty": {"type": "integer"}}},
            "List": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
            "Esc~a/b": {"type": "integer"},
        },
    }
    value = {"item": {"qty": "2"}, "list": [{"qty": "3"}]}
    assert run(value, schema) == {"item": {"qty": 2}, "list": [{"qty": 3}]}


def test_ref_with_sibling_keywords_and_escapes() -> None:
    schema = {
        "properties": {"x": {"$ref": "#/$defs/Esc~0a~1b", "description": "d"}},
        "$defs": {"Esc~a/b": {"type": "integer"}},
    }
    assert run({"x": "4"}, schema) == {"x": 4}


@pytest.mark.parametrize(
    "ref",
    ["#/$defs/Missing", "http://example.com/x.json", "#/$defs/List/5", "#/$defs/List/x", 5],
)
def test_unresolvable_refs_leave_the_value_alone(ref: object) -> None:
    schema = {"properties": {"x": {"$ref": ref}}, "$defs": {"List": [1]}}
    assert run({"x": "4"}, schema) == {"x": "4"}


def test_root_ref_and_ref_loops() -> None:
    schema = {"$ref": "#", "properties": {"a": {"type": "integer"}}}
    assert run({"a": "1"}, schema) == {"a": "1"}  # a self-referencing root is not followed
    loop = {"properties": {"x": {"$ref": "#/$defs/A"}}, "$defs": {"A": {"$ref": "#/$defs/A"}}}
    assert run({"x": "1"}, loop) == {"x": "1"}


def test_list_index_refs() -> None:
    schema = {
        "properties": {"x": {"$ref": "#/prefixItems/0"}},
        "prefixItems": [{"type": "integer"}],
    }
    assert run({"x": "9"}, schema) == {"x": 9}


def test_all_of_applies_each_part() -> None:
    schema = {"allOf": [obj(a={"type": "integer"}), obj(b={"type": "boolean"})]}
    assert run({"a": "1", "b": "true"}, schema) == {"a": 1, "b": True}


def test_optional_integer_union() -> None:
    schema = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    assert run("5", schema) == 5
    assert run(None, schema) is None
    assert run(5, schema) == 5
    assert run("five", schema) == "five"


def test_optional_list_union_wraps_and_coerces() -> None:
    schema = {"anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "null"}]}
    assert run("5", schema) == [5]


def test_union_that_already_matches_a_branch_is_untouched() -> None:
    # "12" is already a valid string, so the integer reading is not applied.
    assert run("12", {"anyOf": [{"type": "integer"}, {"type": "string"}]}) == "12"
    assert run("12", {"oneOf": [{"type": "integer"}, {"type": "string"}]}) == "12"


def test_ambiguous_union_is_left_alone() -> None:
    schema = {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}
    # "1" can only become an integer, but "true" only a boolean: each is unambiguous.
    assert run("1", schema) == 1
    assert run("true", schema) is True
    # Two readings with different results are refused.
    both = {"anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}]}
    assert run("5", both) == "5"
    # Two object branches both match a dict: untouched.
    objs = {"anyOf": [obj(a={"type": "integer"}), obj(a={"type": "string"})]}
    assert run({"a": "1"}, objs) == {"a": "1"}


def test_union_branches_with_equal_outcomes_are_accepted() -> None:
    schema = {"anyOf": [{"type": "integer"}, {"type": "number"}]}
    assert run("12", schema) == 12


def test_union_with_const_and_enum_branches() -> None:
    schema = {"anyOf": [{"const": "none"}, {"type": "integer"}]}
    assert run("none", schema) == "none"
    assert run("3", schema) == 3
    enum_schema = {"anyOf": [{"enum": ["a", "b"]}, {"type": "integer"}]}
    assert run("b", enum_schema) == "b"
    assert run("7", enum_schema) == 7
    assert run("c", enum_schema) == "c"


def test_nested_union_in_branch() -> None:
    schema = {"anyOf": [{"anyOf": [{"type": "integer"}, {"type": "null"}]}, {"type": "boolean"}]}
    assert run("5", schema) == 5


def test_type_list() -> None:
    schema = {"type": ["integer", "null"]}
    assert run("5", schema) == 5
    assert run(None, schema) is None
    assert run(5, schema) == 5
    assert run("five", schema) == "five"
    assert run("5", {"type": ["string", "integer"]}) == "5"
    assert run("5", {"type": ["integer", "array"]}) == "5"
    assert run(
        {"a": "1"}, {"type": ["object", "null"], "properties": {"a": {"type": "integer"}}}
    ) == {"a": 1}


def test_enum_values_are_never_remapped() -> None:
    schema = {"enum": ["low", "high"]}
    assert coerce("High", schema) == ("High", ())
    assert coerce("High", {"type": "string", "enum": ["low", "high"]}) == ("High", ())


def test_depth_limit_stops_runaway_recursion() -> None:
    schema: dict[str, Any] = {"properties": {"n": {"$ref": "#"}}}
    value: dict[str, Any] = {"n": "x"}
    for _ in range(200):
        value = {"n": value}
    coerce(value, schema)


def test_change_notes_name_paths_and_types_only() -> None:
    schema = obj(age={"type": "integer"}, tags={"type": "array", "items": {"type": "string"}})
    _, changes = coerce({"age": "123456", "tags": [987]}, schema)
    assert changes == ("$.age: string -> integer", "$.tags[0]: integer -> string")
    assert all("123456" not in note and "987" not in note for note in changes)


def test_multi_type_that_matches_several_types_is_untouched() -> None:
    assert run(5, {"type": ["integer", "number"]}) == 5


def test_union_with_type_list_and_unusable_branches() -> None:
    # An unresolvable branch could accept anything, so the union is treated as ambiguous.
    schema = {"anyOf": [{"type": ["integer", "null"]}, {"$ref": "#/$defs/Missing"}]}
    assert run("5", schema) == "5"
    assert run("5", {"anyOf": [{"type": ["integer", "null"]}, {"type": "boolean"}]}) == 5
    assert run(True, {"anyOf": [{"type": "boolean"}, {"type": "null"}]}) is True
    assert run([1], {"anyOf": [{"type": "array"}, {"type": "null"}]}) == [1]
    assert run({"a": 1}, {"anyOf": [{"type": "array"}, {"type": "object"}]}) == {"a": 1}


def test_lexeme_is_a_float_that_remembers_its_spelling() -> None:
    number = Lexeme("1.50")
    assert number == 1.5
    assert isinstance(number, float)
    assert number.text == "1.50"
    assert run(number, {"type": "number"}) == 1.5


def test_keys_from_the_model_output_stay_out_of_the_notes() -> None:
    schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    value, changes = coerce({"Ayse Yilmaz": "34"}, schema)
    assert value == {"Ayse Yilmaz": 34}
    assert changes == ("$.*: string -> integer",)
    named = {"type": "object", "properties": {"age": {"type": "integer"}}}
    assert coerce({"age": "3"}, named)[1] == ("$.age: string -> integer",)
