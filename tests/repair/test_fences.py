import pytest

from slmstrict.repair.fences import MAX_CANDIDATES, find_candidates, strip_fences

OBJ = '{"a": 1}'


@pytest.mark.parametrize(
    "text",
    [
        f"```json\n{OBJ}\n```",
        f"```JSON\n{OBJ}\n```",
        f"```javascript\n{OBJ}\n```",
        f"```js\n{OBJ}\n```",
        f"```\n{OBJ}\n```",
        f"```json {OBJ}```",
        f"```{OBJ}```",
        f"~~~json\n{OBJ}\n~~~",
        f"```json\r\n{OBJ}\r\n```",
        f"```json\n{OBJ}\n",
        f"   \n{OBJ}\n  ",
    ],
)
def test_fence_variants_yield_the_object(text: str) -> None:
    result = strip_fences(text)
    assert result.found
    assert result.text == OBJ


def test_clean_text_reports_no_change() -> None:
    result = strip_fences(OBJ)
    assert result.text == OBJ
    assert result.detail is None


def test_whitespace_only_difference_is_no_change() -> None:
    assert strip_fences(f"\n  {OBJ}  \n").detail is None


def test_surrounding_prose_is_removed_and_reported() -> None:
    result = strip_fences(f"Here you go: {OBJ} Enjoy!")
    assert result.text == OBJ
    assert result.detail == "removed text around the JSON"


def test_fence_and_prose_are_both_reported() -> None:
    result = strip_fences(f"Sure!\n```json\n{OBJ}\n```\nDone.")
    assert result.detail == "removed the markdown code fence; removed text around the JSON"


def test_fence_only_is_reported_without_prose() -> None:
    assert strip_fences(f"```json\n{OBJ}\n```").detail == "removed the markdown code fence"


def test_no_object_found() -> None:
    result = strip_fences("nothing to see")
    assert not result.found
    assert result.text == "nothing to see"
    assert result.detail is not None


def test_empty_text() -> None:
    assert not strip_fences("").found


def test_braces_inside_strings_do_not_confuse_the_scan() -> None:
    text = 'Result: {"a": "}{ \\" }", "b": {"c": 1}} bye'
    assert strip_fences(text).text == '{"a": "}{ \\" }", "b": {"c": 1}}'


def test_different_blocks_are_ambiguous() -> None:
    result = strip_fences('{"example": 1} and {"answer": 42}')
    assert result.found
    assert result.ambiguous
    assert result.text == '{"example": 1} and {"answer": 42}'
    assert result.detail == "found 2 different JSON blocks; refusing to pick one"


def test_a_cut_off_block_next_to_a_complete_one_is_ambiguous() -> None:
    result = strip_fences('Example: {"name": "Veli", "age": 40}\nAnswer: {"name": "Ali", "ag')
    assert result.ambiguous


def test_identical_blocks_are_one_answer() -> None:
    result = strip_fences('```json\n{"a": 1}\n```\nAgain:\n```json\n{"a": 1}\n```')
    assert not result.ambiguous
    assert result.text == '{"a": 1}'
    assert result.detail is not None
    assert "found 2 identical JSON blocks, used the first" in result.detail


def test_too_many_blocks_are_ambiguous() -> None:
    result = strip_fences("{} " * (MAX_CANDIDATES + 5))
    assert result.ambiguous
    assert result.detail == f"found more than {MAX_CANDIDATES} JSON blocks; refusing to pick one"


def test_too_many_different_blocks_do_not_claim_to_be_identical() -> None:
    text = " ".join(f'{{"n": {n}}}' for n in range(MAX_CANDIDATES + 5))
    detail = strip_fences(text).detail
    assert detail == f"found more than {MAX_CANDIDATES} JSON blocks; refusing to pick one"


def test_candidates_are_top_level_objects_in_order() -> None:
    text = '{"first": {"inner": 1}}\n```json\n{"second": 2}\n```\n{"third": 3}'
    found = find_candidates(text)
    assert [c.text for c in found] == ['{"first": {"inner": 1}}', '{"second": 2}', '{"third": 3}']


def test_truncated_object_runs_to_the_end() -> None:
    result = strip_fences('Output: {"a": [1, 2')
    assert result.text == '{"a": [1, 2'


def test_an_inner_object_of_a_truncated_object_is_not_a_candidate() -> None:
    text = '{"name": "Ali", "friend": {"name": "Veli", "age": 40}, "ag'
    found = find_candidates(text)
    assert [c.text for c in found] == [text]
    assert strip_fences(text).text == text


def test_an_unbalanced_brace_swallows_the_rest_instead_of_guessing() -> None:
    text = 'Use { for blocks. Answer: {"a": 1}'
    found = find_candidates(text)
    assert [c.text for c in found] == ['{ for blocks. Answer: {"a": 1}']


def test_candidate_count_is_capped() -> None:
    text = "{} " * (MAX_CANDIDATES * 4)
    assert len(find_candidates(text)) == MAX_CANDIDATES + 1


@pytest.mark.parametrize("marker", ["```", "~~~"])
def test_fence_markers_inside_json_strings_are_content(marker: str) -> None:
    obj = '{"title": "t", "code": "' + marker + "py\\nprint(1)\\n" + marker + '"}'
    for text in (obj, f"```json\n{obj}\n```", f"Here:\n{obj}\nBye", f"~~~\n{obj}\n~~~"):
        result = strip_fences(text)
        assert result.text == obj, text


def test_text_that_already_parses_skips_the_cleanup() -> None:
    result = strip_fences('  {"a": "```"}  ')
    assert result.text == '{"a": "```"}'
    assert result.detail is None


def test_closing_fence_line_is_cut_from_a_truncated_object() -> None:
    result = strip_fences('```json\n{"a": [1, 2\n```\n')
    assert result.text == '{"a": [1, 2\n'
    assert result.detail == "removed the markdown code fence"


def test_unbalanced_object_in_a_fence_without_closing() -> None:
    result = strip_fences('```json\n{"a": 1, "b": [')
    assert result.text == '{"a": 1, "b": ['


def test_newline_before_a_closing_fence_stays_with_the_object() -> None:
    result = strip_fences('```json\n{"a": [1, 2\n```')
    assert result.text == '{"a": [1, 2\n'


def test_inline_closing_fence_is_cut_outside_strings() -> None:
    assert strip_fences('{"a": 12```').text == '{"a": 12'
    assert strip_fences('```json\n{"a": 12```').text == '{"a": 12'
    assert strip_fences('{"a": 12~~~ ').text == '{"a": 12'


def test_inline_fence_inside_a_cut_off_string_is_content() -> None:
    assert strip_fences('{"a": "text```').text == '{"a": "text```'
    # ...but a fence on its own line is still a fence.
    assert strip_fences('{"a": "text\n```').text == '{"a": "text\n'
