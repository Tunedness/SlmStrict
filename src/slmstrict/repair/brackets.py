"""Repair step 2: close unclosed brackets and strings, drop trailing commas.

Truncated or sloppy output usually fails in a handful of ways: a missing ``}`` or ``]`` at the end,
a string cut off mid-word, a comma before a closing bracket. This step fixes exactly those.

It never invents content. A truncated string is *closed*, not completed. A trailing member that
has no complete value (``"key":``, a bare key, ``tru``, ``12.``) is removed instead of being given
one. A number that ends exactly where the text ends is a special case: the cut may have taken
digits off it (``12`` could have been ``125``). This step closes the brackets and keeps the number
as written, but :func:`has_truncated_number` reports the situation so the pipeline can refuse the
result instead of trusting it. Whether the result is acceptable is decided later by schema
validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = ["complete_brackets", "has_truncated_number"]

_TokenKind = Literal["open", "close", "comma", "colon", "string", "scalar"]

_HEX = frozenset("0123456789abcdefABCDEF")
_STRUCTURAL = frozenset('{}[],:"')
_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_WORD_RE = re.compile(r"true|false|null")
# A number, or the beginning of one (``-``, ``12.``, ``1e``, ``1e+``).
_NUMBER_PREFIX_RE = re.compile(r"-|-?\d+(?:\.\d*)?(?:[eE][+-]?\d*)?")


@dataclass(slots=True)
class _Token:
    kind: _TokenKind
    text: str
    ws: str = ""
    is_key: bool = False
    terminated: bool = True
    at_end: bool = False  # the token ends exactly at the end of the text


def complete_brackets(text: str) -> tuple[str, str | None]:
    """Return ``(repaired_text, detail)``; ``detail`` is ``None`` when nothing needed fixing.

    ``text`` is expected to start with ``{`` (the previous step cuts it out). Text that does not
    is returned unchanged.
    """
    tokens = _tokenize(text)
    if not tokens or tokens[0].kind != "open" or tokens[0].text != "{":
        return text, None

    kept: list[_Token] = []
    stack: list[str] = []
    notes: list[str] = []
    inserted = 0
    removed_commas = 0
    trailing_garbage = False

    for position, token in enumerate(tokens):
        if token.kind == "open":
            stack.append(token.text)
            kept.append(token)
        elif token.kind == "close":
            opener = "{" if token.text == "}" else "["
            if opener not in stack:
                kept.append(token)  # stray closer: leave it, validation will reject the result
                continue
            while stack[-1] != opener:
                removed_commas += _drop_trailing_commas(kept)
                kept.append(_Token("close", "}" if stack.pop() == "{" else "]"))
                inserted += 1
            stack.pop()
            removed_commas += _drop_trailing_commas(kept)
            kept.append(token)
            if not stack:
                trailing_garbage = position + 1 < len(tokens)
                break
        else:
            if token.kind == "string":
                previous = kept[-1] if kept else None
                token.is_key = (
                    bool(stack)
                    and stack[-1] == "{"
                    and previous is not None
                    and previous.kind in ("open", "comma")
                )
            kept.append(token)

    dropped = 0
    closed_string = False
    if stack:
        dropped, removed = _trim_incomplete_tail(kept)
        removed_commas += removed
        # Trimming stops at an opener, so ``stack`` still matches the kept tokens.
        last = kept[-1] if kept else None
        if last is not None and last.kind == "string" and not last.terminated:
            last.text += '"'
            last.terminated = True
            closed_string = True
        for opener in reversed(stack):
            kept.append(_Token("close", "}" if opener == "{" else "]"))
        inserted += len(stack)

    if trailing_garbage:
        notes.append("removed text after the closing brace")
    if closed_string:
        notes.append("closed a truncated string")
    if inserted:
        notes.append(f"added {inserted} missing closing bracket(s)")
    if dropped:
        notes.append(f"dropped {dropped} incomplete trailing token(s)")
    if removed_commas:
        notes.append("removed trailing comma")
    if not notes:
        return text, None
    rebuilt = "".join(token.ws + token.text for token in kept)
    return rebuilt.lstrip(), "; ".join(notes)


def _drop_trailing_commas(kept: list[_Token]) -> int:
    """Pop commas at the end of ``kept`` and return how many there were."""
    count = 0
    while kept and kept[-1].kind == "comma":
        kept.pop()
        count += 1
    return count


def _trim_incomplete_tail(kept: list[_Token]) -> tuple[int, int]:
    """Remove the dangling end of a truncated document.

    Returns ``(dropped_tokens, removed_commas)``.
    """
    dropped = 0
    commas = 0
    while kept:
        last = kept[-1]
        if last.kind == "comma":
            kept.pop()
            commas += 1
        elif last.kind == "colon":
            kept.pop()
            dropped += 1
            if kept and kept[-1].kind == "string" and kept[-1].is_key:
                kept.pop()
                dropped += 1
        elif (last.kind == "string" and last.is_key) or (
            last.kind == "scalar" and not _is_trusted_scalar(last)
        ):
            kept.pop()
            dropped += 1
        else:
            break
    return dropped, commas


def has_truncated_number(text: str) -> bool:
    """``True`` when ``text`` is a cut-off object whose last token is a number (or a number start)
    that ends exactly at the end of the text, so digits may be missing."""
    tokens = _tokenize(text)
    if not tokens or tokens[0].kind != "open" or tokens[0].text != "{":
        return False
    last = tokens[-1]
    return last.kind == "scalar" and last.at_end and bool(_NUMBER_PREFIX_RE.fullmatch(last.text))


def _is_trusted_scalar(token: _Token) -> bool:
    """A complete JSON scalar."""
    return bool(_WORD_RE.fullmatch(token.text) or _NUMBER_RE.fullmatch(token.text))


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    length = len(text)
    position = 0
    while position < length:
        start = position
        while position < length and text[position].isspace():
            position += 1
        if position >= length:
            break
        whitespace = text[start:position]
        char = text[position]
        if char in "{[":
            tokens.append(_Token("open", char, whitespace))
            position += 1
        elif char in "}]":
            tokens.append(_Token("close", char, whitespace))
            position += 1
        elif char == ",":
            tokens.append(_Token("comma", char, whitespace))
            position += 1
        elif char == ":":
            tokens.append(_Token("colon", char, whitespace))
            position += 1
        elif char == '"':
            token, position = _read_string(text, position, whitespace)
            tokens.append(token)
        else:
            end = position
            while end < length and not text[end].isspace() and text[end] not in _STRUCTURAL:
                end += 1
            tokens.append(_Token("scalar", text[position:end], whitespace, at_end=end == length))
            position = end
    return tokens


def _read_string(text: str, start: int, whitespace: str) -> tuple[_Token, int]:
    """Read a string token starting at the opening quote.

    An unterminated string loses a dangling escape (``\\`` or a partial ``\\uXX``) so that closing
    it yields valid JSON.
    """
    length = len(text)
    index = start + 1
    while index < length:
        char = text[index]
        if char == "\\":
            if index + 1 >= length:
                return _Token("string", text[start:index], whitespace, terminated=False), length
            if text[index + 1] == "u":
                digits = text[index + 2 : index + 6]
                if index + 6 > length and all(c in _HEX for c in digits):
                    return (
                        _Token("string", text[start:index], whitespace, terminated=False),
                        length,
                    )
                index += 2
            else:
                index += 2
        elif char == '"':
            return _Token("string", text[start : index + 1], whitespace), index + 1
        else:
            index += 1
    return _Token("string", text[start:], whitespace, terminated=False), length
