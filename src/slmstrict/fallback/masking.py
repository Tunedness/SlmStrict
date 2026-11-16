"""Masking hook of the cloud fallback (ADR-003, ADR-007).

:class:`Masker` is SlmStrict's own ``typing.Protocol``. Its shape (text in, masked text out)
matches what McpGuard's PII scan produces for the text it forwards, so a McpGuard recognizer set
can be wrapped in a few lines, but nothing here imports or depends on McpGuard.

The default is *no masker*: with ``masker=None`` the prompt leaves the device as it is. That is a
deliberate, visible choice (ADR-003); :class:`RegexMasker` is a small, dependency-free example for
users who want a baseline.

A masking mistake must never be a leak: where a value *looks like* sensitive data but cannot be
confirmed (a failed checksum, trailing digits glued to a card number), :class:`RegexMasker` masks
it anyway. Over-masking costs some context for the cloud model; under-masking sends PII.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Final, Protocol, runtime_checkable

__all__ = ["Masker", "RegexMasker"]


@runtime_checkable
class Masker(Protocol):
    """Removes sensitive data from a text before it is sent to a cloud provider.

    Implementations must be deterministic enough to be tested and must not raise for any ``str``.
    If one does raise, the fallback fails closed: nothing is sent.
    """

    def mask(self, text: str) -> str: ...


# -- validators ---------------------------------------------------------------------------


def _longest_valid_iban_prefix(compact: str) -> int:
    """Length of the longest prefix of ``compact`` that is an IBAN by checksum (mod 97), or 0.

    One pass: the account part is folded into a running remainder, so every candidate length costs
    O(1) instead of a fresh big-integer conversion.
    """
    if len(compact) < 15:
        return 0
    head = "".join(str(int(ch, 36)) for ch in compact[:4].upper())
    head_value = int(head)
    shift = 10 ** len(head)
    remainder = 0
    best = 0
    for count, ch in enumerate(compact[4:], start=5):
        value = int(ch, 36)
        remainder = (remainder * (100 if value > 9 else 10) + value) % 97
        if count >= 15 and (remainder * shift + head_value) % 97 == 1:
            best = count
    return best


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, ch in enumerate(reversed(digits)):
        digit = int(ch)
        if index % 2 == 1:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


# IBAN length per country (ISO 13616 registry, the commonly used part of it).
_IBAN_LENGTHS: Final[dict[str, int]] = {
    "AD": 24,
    "AE": 23,
    "AL": 28,
    "AT": 20,
    "AZ": 28,
    "BA": 20,
    "BE": 16,
    "BG": 22,
    "BH": 22,
    "BR": 29,
    "BY": 28,
    "CH": 21,
    "CR": 22,
    "CY": 28,
    "CZ": 24,
    "DE": 22,
    "DK": 18,
    "DO": 28,
    "EE": 20,
    "EG": 29,
    "ES": 24,
    "FI": 18,
    "FO": 18,
    "FR": 27,
    "GB": 22,
    "GE": 22,
    "GI": 23,
    "GL": 18,
    "GR": 27,
    "GT": 28,
    "HR": 21,
    "HU": 28,
    "IE": 22,
    "IL": 23,
    "IQ": 23,
    "IS": 26,
    "IT": 27,
    "JO": 30,
    "KW": 30,
    "KZ": 20,
    "LB": 28,
    "LC": 32,
    "LI": 21,
    "LT": 20,
    "LU": 20,
    "LV": 21,
    "MC": 27,
    "MD": 24,
    "ME": 22,
    "MK": 19,
    "MR": 27,
    "MT": 31,
    "MU": 30,
    "NL": 18,
    "NO": 15,
    "PK": 24,
    "PL": 28,
    "PS": 29,
    "PT": 25,
    "QA": 29,
    "RO": 24,
    "RS": 22,
    "SA": 24,
    "SC": 31,
    "SE": 24,
    "SI": 19,
    "SK": 24,
    "SM": 27,
    "ST": 25,
    "SV": 28,
    "TL": 23,
    "TN": 24,
    "TR": 26,
    "UA": 29,
    "VA": 22,
    "VG": 24,
    "XK": 20,
}

_IBAN_START: Final = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2})\d{2}")
_IBAN_MIN_CHARS: Final = 10  # a prefix this long that starts like an IBAN is masked, valid or not
_IBAN_MIN_DIGITS: Final = 3  # ... when its account part holds at least this many digits
_MAX_SEPARATOR: Final = 8  # longest run of non-alphanumerics read as a separator between groups
_IBAN_MAX_CHARS: Final = 34


def _iban_ends(text: str, start: int) -> list[int]:
    """End offsets after each alphanumeric character of the IBAN-like run starting at ``start``.

    A separator (a run of 1-3 characters that are not letters or digits: spaces, tabs, dashes,
    slashes, dots, ...) is part of the run only after a complete group of four characters (IBANs
    are printed in groups of four) and only when an alphanumeric character follows it. Anything
    else ends the run, so a following word does not join in.
    """
    ends: list[int] = []
    group = 0
    i = start
    while i < len(text) and len(ends) < _IBAN_MAX_CHARS:
        ch = text[i]
        if ch.isascii() and ch.isalnum():
            i += 1
            group += 1
            ends.append(i)
            continue
        if group != 4:
            break
        j = i
        while j < len(text) and not text[j].isalnum() and j - i < _MAX_SEPARATOR:
            j += 1
        if j == i or j >= len(text) or not (text[j].isascii() and text[j].isalnum()):
            break
        i = j
        group = 0
    return ends


def _iban_end(text: str, start: int, country: str) -> int | None:
    """Where the IBAN starting at ``start`` ends, or ``None`` when this is not one.

    A known country fixes the length, so an IBAN is cut exactly there and the next word stays.
    It is masked even when the checksum fails (a typo is still an account number), and when the run
    is shorter than the country's length but at least ``_IBAN_MIN_CHARS`` long with a few digits in
    its account part (a truncated one; "AT12 pieces" is not).
    An unknown country needs a valid checksum; the longest valid prefix is taken.
    """
    ends = _iban_ends(text, start)
    expected = _IBAN_LENGTHS.get(country)
    if expected is not None:
        length = expected
        if len(ends) >= length:
            return ends[length - 1]
        if len(ends) < _IBAN_MIN_CHARS:
            return None
        account = text[start + 4 : ends[-1]]
        digits = sum(ch.isdigit() for ch in account)
        return ends[-1] if digits >= _IBAN_MIN_DIGITS else None
    compact = "".join(ch for ch in text[start : ends[-1]] if ch.isascii() and ch.isalnum())
    best = _longest_valid_iban_prefix(compact)
    return ends[best - 1] if best else None


def _mask_ibans(text: str, tag: str) -> str:
    out: list[str] = []
    position = 0
    for match in _IBAN_START.finditer(text):
        if match.start() < position:
            continue
        end = _iban_end(text, match.start(), match.group(1).upper())
        if end is None:
            continue
        out.append(text[position : match.start()])
        out.append(tag)
        position = end
    out.append(text[position:])
    return "".join(out)


# -- patterns -----------------------------------------------------------------------------

# The lookbehind makes the local part start at the start of its run only; without it a long
# run of local-part characters is rescanned from every offset (quadratic time).
_EMAIL: Final = re.compile(
    r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}"
)
# A payment card number as printed: one run of 13+ digits, or groups of 4-6 digits (4-4-4-4,
# Amex 4-6-5) with an optional short tail. The tail is swallowed on purpose: a CVV or an expiry
# glued to the number is as sensitive as the number. There is no checksum veto.
_SEP = rf"[^0-9A-Za-z]{{1,{_MAX_SEPARATOR}}}"
_CARD: Final = re.compile(
    rf"(?<!\d)(?:\d{{13,}}(?:{_SEP}\d{{3,4}})?"
    rf"|\d{{4}}(?:{_SEP}\d{{4,6}}){{2,4}}(?:{_SEP}\d{{1,4}})?)(?!\d)"
)
_TCKN: Final = re.compile(r"(?<!\d)\d{11}(?!\d)")
_PHONE: Final = re.compile(
    r"(?<![\w+])(?:"
    r"(?:\+?90|0)?[\s.\-/]?\(?5\d{2}\)?[\s.\-/]?\d{3}[\s.\-/]?\d{2}[\s.\-/]?\d{2}"  # TR mobile
    r"|\+\d{1,3}[\s.\-/]?\(?\d{2,4}\)?[\s.\-/]?\d{3,4}[\s.\-/]?\d{3,4}"  # international
    r")(?!\d)"
)

ALL_KINDS: Final = ("iban", "email", "card", "phone", "tckn")
"""Kinds :class:`RegexMasker` knows, in the order they are applied."""


def _card_shaped(match: re.Match[str]) -> bool:
    """Whether a card-shaped match is masked.

    Every match with 13+ digits is, with one exception: a list written with commas or semicolons
    ("years 2020, 2021, 2022, 2023") that is not a card by checksum. Cards are never printed that
    way, and year or quantity lists are common in extraction prompts. The checksum is tried on all
    digits and on all but the last group (a CVV or expiry may follow the number).
    """
    text = match.group(0)
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 13:
        return False
    if "," not in text and ";" not in text:
        return True
    groups = [g for g in re.split(r"[^0-9]+", text) if g]
    without_tail = "".join(groups[:-1])
    return _luhn_ok(digits) or (len(without_tail) >= 13 and _luhn_ok(without_tail))


class RegexMasker:
    """Example :class:`Masker`: replaces e-mail addresses, IBANs, payment card numbers, Turkish
    national ids (TCKN) and phone numbers with a ``[KIND]`` tag.

    The rules are built to err on the side of masking:

    * IBAN: anything that starts like one of a known country is masked up to the country's length,
      checksum or not; other countries need a valid checksum.
    * Card: every run of 13+ digits and every 4-4-4-4 / 4-6-5 grouping, together with a short
      group right after it (CVV, expiry), written with or without separators. Long digit runs
      that are not cards are masked too; the one exception is a comma or semicolon separated list
      that fails the card checksum (a list of years, say). About one such list in ten passes the
      checksum by chance and is masked anyway.
    * TCKN: every standalone run of exactly 11 digits, valid checksum or not (a mistyped national
      id is still one).

    Between the groups of a card number or an IBAN, a run of 1-8 characters that are not letters
    or digits is read as the separator (spaces, tabs, ``-``, ``_``, ``/``, ``.``, ``|``, ...). A
    longer gap, or a separator made of letters, is not recognized and the number is not masked.

    This is a best-effort example implementation, not a compliance tool: names, addresses and
    free-text identifiers are *not* detected and a number written in an unforeseen way may slip
    through. In production use dedicated PII recognizers, for example a McpGuard-compatible
    :class:`Masker`.

    ``kinds`` restricts the recognizers; the default is all of :data:`ALL_KINDS`.
    """

    def __init__(self, kinds: Iterable[str] | None = None) -> None:
        wanted = tuple(ALL_KINDS if kinds is None else kinds)
        unknown = sorted(set(wanted) - set(ALL_KINDS))
        if unknown:
            raise ValueError(f"unknown masking kinds: {', '.join(unknown)}")
        self._kinds = tuple(kind for kind in ALL_KINDS if kind in wanted)

    def mask(self, text: str) -> str:
        """Return ``text`` with every recognized sensitive value replaced by its tag."""
        for kind in self._kinds:
            text = self._apply(kind, text)
        return text

    @staticmethod
    def _apply(kind: str, text: str) -> str:
        tag = f"[{kind.upper()}]"
        if kind == "iban":
            return _mask_ibans(text, tag)
        pattern, shaped = _RULES[kind]

        def replace(match: re.Match[str]) -> str:
            if shaped is not None and not shaped(match):
                return match.group(0)
            return tag

        return pattern.sub(replace, text)

    def __repr__(self) -> str:
        return f"RegexMasker(kinds={self._kinds!r})"


_RULES: Final[dict[str, tuple[re.Pattern[str], Callable[[re.Match[str]], bool] | None]]] = {
    "email": (_EMAIL, None),
    "card": (_CARD, _card_shaped),
    "phone": (_PHONE, None),
    "tckn": (_TCKN, None),
}
