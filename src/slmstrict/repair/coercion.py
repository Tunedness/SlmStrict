"""Repair step 3: schema-directed, lossless type coercion.

The schema says what type a value should have; this step converts a value that holds the right
*content* in the wrong *type*:

* ``"12"`` -> ``12`` (integer), ``"2.5"`` -> ``2.5`` (number), ``12.0`` -> ``12`` (integer)
* ``"true"`` / ``"false"`` -> boolean (case-insensitive)
* a single value -> a one-element list, where the schema wants an array
* ``42`` -> ``"42"`` where the schema wants a string (a float only when it still carries the
  text the model wrote, see :class:`Lexeme`, so ``1.50`` becomes ``"1.50"`` and never ``"1.5"``)

Nothing here may invent or guess content. It never adds a missing property, never replaces
``null`` with a default, never maps a value onto an enum member (``"High"`` stays ``"High"``
even if ``"high"`` is allowed), and leaves ambiguous unions alone. A value that cannot be converted
losslessly is left as it is; schema validation then reports it.

The change records name schema paths and types only, never values or model-chosen keys (keys
under ``additionalProperties`` show up as ``*``), so they are safe to log.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from math import isfinite
from typing import Any
from urllib.parse import unquote

__all__ = ["Lexeme", "coerce"]

_MAX_DEPTH = 64

_INT_RE = re.compile(r"[+-]?\d+")
_INTEGRAL_DECIMAL_RE = re.compile(r"([+-]?\d+)\.0+")
_NUMBER_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")

_Changes = list[str]


class Lexeme(float):
    """A float that remembers how it was written in the JSON text.

    Parse with ``json.loads(text, parse_float=Lexeme)`` to let a number become a string without
    changing its spelling. Without it, converting a float to a string is refused.
    """

    __slots__ = ("text",)
    text: str

    def __new__(cls, text: str) -> Lexeme:
        instance = super().__new__(cls, text)
        instance.text = text
        return instance


def coerce(value: object, schema: Mapping[str, Any]) -> tuple[object, tuple[str, ...]]:
    """Coerce ``value`` towards ``schema`` (a normalized JSON Schema).

    Returns the new value (containers are copied, the input is not mutated) and one short note
    per conversion, such as ``"$.age: string -> integer"``.
    """
    changes: _Changes = []
    result = _Coercer(schema).visit(value, schema, "$", 0, changes)
    return result, tuple(changes)


class _Coercer:
    def __init__(self, root: Mapping[str, Any]) -> None:
        self._root = root

    # -- traversal ----------------------------------------------------------------------

    def visit(self, value: object, node: object, path: str, depth: int, out: _Changes) -> object:
        resolved = self._resolve(node)
        if resolved is None or depth > _MAX_DEPTH:
            return value
        for sub in _as_list(resolved.get("allOf")):
            value = self.visit(value, sub, path, depth + 1, out)
        for keyword in ("anyOf", "oneOf"):
            branches = [b for b in _as_list(resolved.get(keyword)) if isinstance(b, dict)]
            if branches:
                value = self._union(value, branches, path, depth, out)

        declared = resolved.get("type")
        if isinstance(declared, str):
            return self._typed(value, declared, resolved, path, depth, out)
        if isinstance(declared, list):
            types = [t for t in declared if isinstance(t, str)]
            return self._multi_type(value, types, resolved, path, depth, out)
        if "properties" in resolved or "additionalProperties" in resolved:
            return self._typed(value, "object", resolved, path, depth, out)
        if "items" in resolved or "prefixItems" in resolved:
            return self._typed(value, "array", resolved, path, depth, out)
        return value

    def _typed(
        self,
        value: object,
        type_name: str,
        node: dict[str, Any],
        path: str,
        depth: int,
        out: _Changes,
    ) -> object:
        if type_name == "object":
            return self._object(value, node, path, depth, out)
        if type_name == "array":
            if not isinstance(value, list):
                if value is None:
                    return value
                out.append(f"{path}: {_type_name(value)} -> array (wrapped in a list)")
                value = [value]
            return self._array(value, node, path, depth, out)
        converted = _scalar(value, type_name)
        if converted is _UNCHANGED:
            return value
        out.append(f"{path}: {_type_name(value)} -> {type_name}")
        return converted

    def _object(
        self, value: object, node: dict[str, Any], path: str, depth: int, out: _Changes
    ) -> object:
        if not isinstance(value, dict):
            return value
        properties = node.get("properties")
        known = properties if isinstance(properties, dict) else {}
        extra = node.get("additionalProperties")
        result: dict[str, object] = {}
        for key, member in value.items():
            if key in known:
                sub: object = known[key]
                label = key
            elif isinstance(extra, dict):
                sub = extra
                label = "*"  # the key comes from the model output, not from the schema
            else:
                result[key] = member
                continue
            result[key] = self.visit(member, sub, f"{path}.{label}", depth + 1, out)
        return result

    def _array(
        self, value: list[object], node: dict[str, Any], path: str, depth: int, out: _Changes
    ) -> object:
        prefix = _as_list(node.get("prefixItems"))
        items = node.get("items")
        result: list[object] = []
        for index, element in enumerate(value):
            sub: object = prefix[index] if index < len(prefix) else items
            result.append(self.visit(element, sub, f"{path}[{index}]", depth + 1, out))
        return result

    # -- unions -------------------------------------------------------------------------

    def _union(
        self,
        value: object,
        branches: list[dict[str, Any]],
        path: str,
        depth: int,
        out: _Changes,
    ) -> object:
        """Coerce through ``anyOf`` / ``oneOf`` only when exactly one reading is possible."""
        matching = [b for b in branches if self._matches(value, b, 0)]
        if len(matching) == 1:
            return self.visit(value, matching[0], path, depth + 1, out)
        if matching:
            return value  # several branches already accept the value shape: ambiguous
        outcomes: list[tuple[object, _Changes]] = []
        for branch in branches:
            trial: _Changes = []
            new = self.visit(value, branch, path, depth + 1, trial)
            if trial:
                outcomes.append((new, trial))
        return self._single_outcome(value, outcomes, out)

    def _multi_type(
        self,
        value: object,
        types: list[str],
        node: dict[str, Any],
        path: str,
        depth: int,
        out: _Changes,
    ) -> object:
        """Same rule as :meth:`_union` for ``"type": ["integer", "null"]``."""
        matching = [t for t in types if _type_matches(value, t)]
        if len(matching) == 1:
            return self._typed(value, matching[0], node, path, depth, out)
        if matching:
            return value
        outcomes: list[tuple[object, _Changes]] = []
        for type_name in types:
            trial: _Changes = []
            new = self._typed(value, type_name, node, path, depth, trial)
            if trial:
                outcomes.append((new, trial))
        return self._single_outcome(value, outcomes, out)

    @staticmethod
    def _single_outcome(
        value: object, outcomes: list[tuple[object, _Changes]], out: _Changes
    ) -> object:
        if not outcomes:
            return value
        first = _dump(outcomes[0][0])
        if any(_dump(new) != first for new, _ in outcomes[1:]):
            return value  # different readings: refuse to guess
        out.extend(outcomes[0][1])
        return outcomes[0][0]

    def _matches(self, value: object, node: object, depth: int) -> bool:
        """Shallow check: does ``value`` already have the shape ``node`` asks for?"""
        resolved = self._resolve(node)
        if resolved is None or depth > _MAX_DEPTH:
            return True
        if "const" in resolved and not _same(value, resolved["const"]):
            return False
        enum = resolved.get("enum")
        if isinstance(enum, list) and not any(_same(value, member) for member in enum):
            return False
        declared = resolved.get("type")
        if isinstance(declared, str):
            return _type_matches(value, declared)
        if isinstance(declared, list):
            return any(isinstance(t, str) and _type_matches(value, t) for t in declared)
        for keyword in ("anyOf", "oneOf"):
            branches = _as_list(resolved.get(keyword))
            if branches:
                return any(self._matches(value, b, depth + 1) for b in branches)
        return True

    # -- $ref ---------------------------------------------------------------------------

    def _resolve(self, node: object) -> dict[str, Any] | None:
        """Follow local ``$ref`` chains; ``None`` for anything that is not a schema object."""
        seen = 0
        current = node
        while isinstance(current, dict) and "$ref" in current:
            ref = current["$ref"]
            target = self._pointer(ref) if isinstance(ref, str) else None
            seen += 1
            if not isinstance(target, dict) or seen > _MAX_DEPTH:
                return None
            siblings = {k: v for k, v in current.items() if k != "$ref"}
            current = {**target, **siblings}
        return current if isinstance(current, dict) else None

    def _pointer(self, ref: str) -> object:
        if not ref.startswith("#"):
            return None
        current: object = self._root
        for part in ref[1:].split("/")[1:] if ref != "#" else []:
            key = unquote(part).replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit() and int(key) < len(current):
                current = current[int(key)]
            else:
                return None
        return current


# -- scalar conversions -----------------------------------------------------------------


class _Unchanged:
    """Sentinel: the scalar needs (or allows) no conversion."""


_UNCHANGED = _Unchanged()


def _scalar(value: object, type_name: str) -> object:
    if isinstance(value, bool):
        return _UNCHANGED  # a boolean is never silently turned into a number or a string
    if type_name == "integer":
        return _to_integer(value)
    if type_name == "number":
        return _to_number(value)
    if type_name == "boolean":
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        return _UNCHANGED
    if type_name == "string":
        if isinstance(value, Lexeme):
            return value.text
        if isinstance(value, int):
            return str(value)
    return _UNCHANGED


def _to_integer(value: object) -> object:
    if isinstance(value, str):
        text = value.strip()
        integral = _INTEGRAL_DECIMAL_RE.fullmatch(text)
        if _INT_RE.fullmatch(text) or integral:
            digits = integral.group(1) if integral else text
            try:
                return int(digits)
            except ValueError:  # absurdly long digit strings
                return _UNCHANGED
    elif isinstance(value, float) and isfinite(value) and value.is_integer():
        return int(value)
    return _UNCHANGED


def _to_number(value: object) -> object:
    if not isinstance(value, str):
        return _UNCHANGED
    text = value.strip()
    if _INT_RE.fullmatch(text):
        try:
            return int(text)
        except ValueError:
            return _UNCHANGED
    if _NUMBER_RE.fullmatch(text):
        number = float(text)
        if isfinite(number):
            return number
    return _UNCHANGED


# -- small helpers ----------------------------------------------------------------------


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _type_matches(value: object, type_name: str) -> bool:
    actual = _type_name(value)
    if type_name == "number":
        return actual in ("integer", "number")
    return actual == type_name


def _same(a: object, b: object) -> bool:
    return isinstance(a, bool) == isinstance(b, bool) and a == b


def _dump(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)
