import json

import pytest

from slmstrict.repair.brackets import complete_brackets, has_truncated_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1', '{"a": 1}'),
        ('{"a": [1, 2', '{"a": [1, 2]}'),
        ('{"a": {"b": [1, {"c": 2', '{"a": {"b": [1, {"c": 2}]}}'),
        ('{"a": {"b": [1, {"c": "d', '{"a": {"b": [1, {"c": "d"}]}}'),
        ('{"a": "he', '{"a": "he"}'),
        ('{"a": ["x", "y', '{"a": ["x", "y"]}'),
        ('{"a": 1,', '{"a": 1}'),
        ('{"a": 1, ', '{"a": 1}'),
        ('{"a": 1,}', '{"a": 1}'),
        ('{"a": [1, 2,]}', '{"a": [1, 2]}'),
        ('{"a": [1, 2,\n]\n}', '{"a": [1, 2\n]\n}'),
        ('{"a": 1, "b":', '{"a": 1}'),
        ('{"a": 1, "b"', '{"a": 1}'),
        ('{"a": 1, "b', '{"a": 1}'),
        ('{"a":', "{}"),
        ("{", "{}"),
        ('{"a": [1, 2}', '{"a": [1, 2]}'),
        ('{"a": [{"b": 1}', '{"a": [{"b": 1}]}'),
        ('{"a": "x\\', '{"a": "x"}'),
        ('{"a": "x\\u12', '{"a": "x"}'),
        ('{"a": "x\\u', '{"a": "x"}'),
        ('{"a": "x\\n', '{"a": "x\\n"}'),
        ('{"a": "x\\u00e9', '{"a": "x\\u00e9"}'),
        ('{"a": "x\\"', '{"a": "x\\""}'),
        ('{"a": tru', "{}"),
        ('{"a": 12.', "{}"),
        ('{"a": -', "{}"),
        ('{"a": 12', '{"a": 12}'),
        ('{"a": 12 ', '{"a": 12}'),
        ('{"a": 12\n', '{"a": 12}'),
        ('{"a": 12,', '{"a": 12}'),
        ('{"a": 1.5e3', '{"a": 1.5e3}'),
        ('{"a": 0', '{"a": 0}'),
        ('{"a": "x", "b": 125', '{"a": "x", "b": 125}'),
        ('{"a": 1e', "{}"),
        ('{"a": 1e+', "{}"),
        ('{"a": true', '{"a": true}'),
        ('{"a": null, "b": [1', '{"a": null, "b": [1]}'),
        ('{"a": null, "b": [1,', '{"a": null, "b": [1]}'),
        ('{"a": null, "b": [true', '{"a": null, "b": [true]}'),
        ('{"a": false, "b": [null', '{"a": false, "b": [null]}'),
        ('{"a": 1}  trailing words', '{"a": 1}'),
        ('{"a": 1} {"b": 2}', '{"a": 1}'),
        ('  {"a": "1', '{"a": "1"}'),
    ],
)
def test_repairs(text: str, expected: str) -> None:
    repaired, detail = complete_brackets(text)
    assert repaired == expected
    assert detail is not None
    json.loads(repaired)


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '{"a": [1, {"b": "}"}]}',
        '{\n  "a": [\n    1,\n    2\n  ]\n}',
        '{"a": "x, y]"}',
        "",
        "   ",
        "[1, 2",
        "plain text",
    ],
)
def test_untouched(text: str) -> None:
    assert complete_brackets(text) == (text, None)


def test_a_truncated_string_is_closed_not_completed() -> None:
    repaired, detail = complete_brackets('{"story": "Once upon a ti')
    assert json.loads(repaired) == {"story": "Once upon a ti"}
    assert detail is not None
    assert "closed a truncated string" in detail


def test_a_dangling_key_is_removed_not_given_a_value() -> None:
    repaired, detail = complete_brackets('{"a": 1, "b":')
    assert json.loads(repaired) == {"a": 1}
    assert detail is not None
    assert "dropped 2 incomplete" in detail
    assert "removed trailing comma" in detail


def test_string_values_that_look_like_keys_are_kept() -> None:
    repaired, _ = complete_brackets('{"a": ["k", "v"')
    assert json.loads(repaired) == {"a": ["k", "v"]}


def test_stray_closer_is_left_for_validation_to_reject() -> None:
    repaired, detail = complete_brackets('{"a": 1}}')
    assert repaired == '{"a": 1}'
    assert detail == "removed text after the closing brace"
    repaired, _ = complete_brackets('{"a": 1, ]')
    with pytest.raises(json.JSONDecodeError):
        json.loads(repaired)


def test_details_name_what_was_done() -> None:
    _, detail = complete_brackets('{"a": [1,')
    assert detail == "added 2 missing closing bracket(s); removed trailing comma"
    _, detail = complete_brackets('{"a": [1, 2}')
    assert detail == "added 1 missing closing bracket(s)"


def test_a_number_cut_by_the_end_of_the_text_is_kept_but_reported() -> None:
    # The member is not dropped; `has_truncated_number` lets the pipeline refuse the result.
    text = '{"label": "x", "count": 12'
    repaired, _ = complete_brackets(text)
    assert json.loads(repaired) == {"label": "x", "count": 12}
    assert has_truncated_number(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 12',
        '{"a": [1, 2',
        '{"a": {"b": 1.5',
        '{"a": -',
        '{"a": 12.',
        '{"a": 1e',
        '{"a": 1e+',
        '{"a": "x", "b": 0',
        '{"a": [true, 7',
    ],
)
def test_truncated_numbers_are_detected(text: str) -> None:
    assert has_truncated_number(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 12 ',
        '{"a": 12\n',
        '{"a": 12,',
        '{"a": 12}',
        '{"a": "12',
        '{"a": tru',
        '{"a": true',
        '{"a": null',
        '{"a": "x"',
        "{",
        "",
        "[1, 2",
        "12",
    ],
)
def test_other_endings_are_not_truncated_numbers(text: str) -> None:
    assert not has_truncated_number(text)
