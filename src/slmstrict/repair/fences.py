"""Repair step 1: strip markdown code fences and text around the JSON object.

Small models wrap JSON in `````json`` fences (sometimes with the wrong language label), prepend
"Here is the JSON:" or append a closing remark. This step cuts the JSON object out of such text.
It only ever *removes* characters; it never edits the JSON itself.

Objects are found by a string-aware brace scan, not by looking for fence markers first. A
```` ``` ```` inside a JSON string is therefore string content and is never mistaken for a fence.
Text that already parses as a JSON object is returned as it is.

When the text holds several *separate* JSON objects (an example followed by the real answer, say)
that are not identical, the step refuses to choose: a deterministic script cannot know which one
is the answer, and picking the wrong one would hand back another entity's data. The result is
flagged ``ambiguous`` and the ladder re-asks. Identical blocks are the same answer, so one is
used. Objects nested inside another object are never candidates: if the outer object is cut off,
an inner object must not stand in for the answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

__all__ = ["Candidate", "FenceResult", "find_candidates", "strip_fences"]

# Upper bound on the number of candidates, so hostile or degenerate input stays cheap. More blocks
# than this count as ambiguous.
MAX_CANDIDATES = 32

_FENCE_TOKEN_RE = re.compile(r"(?:```|~~~)[ \t]*(?:[A-Za-z][\w+.#-]*)?")
# A closing fence on its own line at the very end of a cut-off object. The newline before it stays
# with the object, so a number before it is seen as finished.
_TRAILING_FENCE_LINE_RE = re.compile(r"(?<=\n)[ \t]*(?:```|~~~)[ \t\r\n]*\Z")
# A closing fence right after the last token (``12```), only honoured outside a string.
_TRAILING_FENCE_INLINE_RE = re.compile(r"(?:```|~~~)[ \t\r\n]*\Z")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A JSON object found in free text: ``text`` is the slice, ``start`` its offset."""

    text: str
    start: int


@dataclass(frozen=True, slots=True)
class FenceResult:
    """Outcome of :func:`strip_fences`.

    ``found`` is ``False`` when the text holds no JSON object at all. ``ambiguous`` is ``True``
    when it holds several different ones; ``text`` is then the original text. ``detail`` is
    ``None`` when the text needed no cleanup.
    """

    text: str
    found: bool
    detail: str | None = None
    ambiguous: bool = False


def find_candidates(text: str) -> list[Candidate]:
    """Return the top-level JSON-object-looking slices of ``text`` in order of appearance.

    A slice runs from a ``{`` to its balanced ``}``. When no balanced end exists (truncated
    output) it runs to the end of the text, minus a closing fence, and the scan stops there:
    everything after an unbalanced ``{`` belongs to that object. At most ``MAX_CANDIDATES + 1``
    slices are returned.
    """
    return [Candidate(text[start:end], start) for start, end in _object_spans(text)]


def strip_fences(text: str) -> FenceResult:
    """Cut the JSON object out of ``text``.

    Several different candidate blocks (cut-off or invalid ones included) make the result
    ``ambiguous``; one block, or several identical ones, is used.
    """
    stripped = text.strip()
    if _is_json_object(stripped):
        return FenceResult(stripped, found=True)
    candidates = find_candidates(text)
    if not candidates:
        return FenceResult(text, found=False, detail="no JSON object found in the text")

    distinct = {candidate.text.strip() for candidate in candidates}
    if len(candidates) > MAX_CANDIDATES:
        detail = f"found more than {MAX_CANDIDATES} JSON blocks; refusing to pick one"
        return FenceResult(text, found=True, detail=detail, ambiguous=True)
    if len(distinct) > 1:
        detail = f"found {len(distinct)} different JSON blocks; refusing to pick one"
        return FenceResult(text, found=True, detail=detail, ambiguous=True)
    chosen = candidates[0]

    notes: list[str] = []
    around = text[: chosen.start] + text[chosen.start + len(chosen.text) :]
    if _FENCE_TOKEN_RE.search(around):
        notes.append("removed the markdown code fence")
    if _FENCE_TOKEN_RE.sub("", around).strip():
        notes.append("removed text around the JSON")
    if len(candidates) > 1:
        notes.append(f"found {len(candidates)} identical JSON blocks, used the first")
    if not notes:
        return FenceResult(chosen.text, found=True)
    return FenceResult(chosen.text, found=True, detail="; ".join(notes))


def _object_spans(text: str) -> list[tuple[int, int]]:
    """Spans of top-level ``{...}`` objects in ``text``; an unbalanced one ends the scan."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    while len(spans) <= MAX_CANDIDATES:
        start = text.find("{", cursor)
        if start == -1:
            break
        end, in_string = _scan_object(text, start)
        if end is None:
            spans.append((start, _truncated_end(text, start, in_string=in_string)))
            break
        spans.append((start, end))
        cursor = end
    return spans


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text, parse_constant=_reject_constant), dict)
    except (ValueError, RecursionError):
        return False


def _reject_constant(name: str) -> object:
    raise ValueError(f"{name} is not valid JSON")


def _truncated_end(text: str, start: int, *, in_string: bool) -> int:
    """End of a cut-off object: the end of the text without a closing fence after it."""
    line = _TRAILING_FENCE_LINE_RE.search(text, start)
    if line:
        return line.start()
    if not in_string:
        inline = _TRAILING_FENCE_INLINE_RE.search(text, start)
        if inline:
            return inline.start()
    return len(text)


def _scan_object(text: str, start: int) -> tuple[int | None, bool]:
    """Scan the object starting at ``start``.

    Returns ``(end, in_string)``: ``end`` is the index just past the matching ``}`` or ``None``
    when the object is unbalanced, and ``in_string`` tells whether the scan ended inside a string.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1, False
    return None, in_string
