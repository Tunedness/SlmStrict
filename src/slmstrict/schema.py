"""Schema input: the internal representation SlmStrict works with.

A :class:`Schema` is built from a Pydantic v2 model class or from a JSON Schema ``dict``. In
both cases it keeps

* a *normalized* JSON Schema (annotations such as ``title`` and ``description`` removed, order
  insensitive lists sorted), which later phases compile into grammars,
* a validator that reports problems as structured :class:`ValidationIssue` objects (the re-ask
  step feeds them back to the model), and
* a stable SHA-256 hash of the normalized schema, used as the grammar cache key. The same
  schema yields the same hash whether it was given as a model or as a ``dict``.

Validation is deliberately strict: it does not coerce types. ``"5"`` is not an ``integer``.
Fixing such output is the job of the repair step, and it has to be visible in ``Result.path``.

JSON Schema ``dict`` inputs are validated by a small built-in validator (the core depends on
Pydantic only). It supports the common validation keywords; ``format`` is kept in the normalized
schema (and in the hash, so grammars can use it) but is not enforced by the validator. Other
keywords that change what is valid but are not implemented raise
:class:`~slmstrict.errors.SchemaError` instead of being silently ignored. ``pattern`` uses
Python's ``re`` dialect.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import inf
from typing import Any, Generic, TypeVar, cast, overload
from urllib.parse import unquote

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticUserError

from slmstrict.errors import SchemaError

__all__ = ["Schema", "ValidationIssue", "ValidationReport"]

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

PathItem = str | int

_JSON_TYPES = frozenset({"null", "boolean", "integer", "number", "string", "array", "object"})

# Keywords whose value is a schema, a map of schemas, or a list of schemas (used to walk a schema
# without touching data such as ``enum`` / ``const`` values or property *names*).
_SCHEMA_KEYS = frozenset({"items", "additionalProperties", "not", "additionalItems"})
_SCHEMA_MAP_KEYS = frozenset({"properties", "patternProperties", "$defs", "definitions"})
_SCHEMA_LIST_KEYS = frozenset({"anyOf", "allOf", "oneOf", "prefixItems"})

# Annotations: they do not change which documents are valid, so they are not part of the
# normalized form (and therefore not of the hash).
_ANNOTATION_KEYS = frozenset(
    {
        "title",
        "description",
        "examples",
        "default",
        "$comment",
        "$schema",
        "$id",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)

# Keywords that affect validity but that the built-in validator does not implement.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "if",
        "then",
        "else",
        "contains",
        "minContains",
        "maxContains",
        "propertyNames",
        "dependentSchemas",
        "dependentRequired",
        "dependencies",
        "unevaluatedItems",
        "unevaluatedProperties",
        "$anchor",
        "$dynamicRef",
        "$dynamicAnchor",
        "$recursiveRef",
    }
)

_MAX_DEPTH = 128


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One structured validation problem.

    ``path`` points into the document (property names and array indexes), ``kind`` is a short
    machine-readable code and ``message`` a human-readable explanation. For Pydantic models
    ``kind`` is the Pydantic error type (for example ``int_type``).
    """

    path: tuple[PathItem, ...]
    kind: str
    message: str

    @property
    def location(self) -> str:
        """JSONPath-like location such as ``$.items[0].name``; ``$`` is the document root."""
        out = "$"
        for item in self.path:
            out += f"[{item}]" if isinstance(item, int) else f".{item}"
        return out

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


@dataclass(frozen=True, slots=True)
class ValidationReport(Generic[T]):
    """Outcome of :meth:`Schema.validate`: the parsed value, or the issues that prevent it."""

    value: T | None
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """``True`` when the document is valid and ``value`` is set."""
        return not self.issues


class Schema(Generic[T]):
    """Schema built from a Pydantic model or a JSON Schema ``dict``.

    Build one with :meth:`Schema.of` (or :meth:`from_model` / :meth:`from_json_schema`).
    ``T`` is the model class for Pydantic input and ``dict[str, Any]`` for ``dict`` input.
    """

    __slots__ = ("_canonical", "_hash", "_json_schema", "_model", "_validator")

    def __init__(
        self,
        *,
        json_schema: dict[str, Any],
        model: type[BaseModel] | None,
        validator: _JsonSchemaValidator | None,
    ) -> None:
        # Internal constructor; the public entry points are the factory methods below.
        if (model is None) == (validator is None):
            raise SchemaError("a Schema needs exactly one of a model or a validator")
        try:
            canonical = _canonical_json(json_schema)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"schema is not JSON-serializable: {exc}") from exc
        self._json_schema = json_schema
        self._canonical = canonical
        self._hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self._model = model
        self._validator = validator

    # -- construction ---------------------------------------------------------------------

    @overload
    @staticmethod
    def of(source: type[M]) -> Schema[M]: ...

    @overload
    @staticmethod
    def of(source: Mapping[str, Any]) -> Schema[dict[str, Any]]: ...

    @staticmethod
    def of(source: type[BaseModel] | Mapping[str, Any]) -> Schema[Any]:
        """Build a schema from a Pydantic model class or a JSON Schema mapping."""
        if isinstance(source, type) and issubclass(source, BaseModel):
            return Schema.from_model(source)
        if isinstance(source, Mapping):
            return Schema.from_json_schema(source)
        raise SchemaError(
            f"expected a Pydantic model class or a JSON Schema mapping, got {type(source).__name__}"
        )

    @staticmethod
    def from_model(model: type[M]) -> Schema[M]:
        """Build a schema from a Pydantic v2 model class."""
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise SchemaError("from_model() expects a Pydantic BaseModel subclass")
        try:
            raw = model.model_json_schema()
        except (PydanticUserError, PydanticInvalidForJsonSchema, NameError) as exc:
            raise SchemaError(f"cannot derive a JSON Schema from {model.__name__}: {exc}") from exc
        return Schema(json_schema=_normalize(raw), model=model, validator=None)

    @staticmethod
    def from_json_schema(schema: Mapping[str, Any]) -> Schema[dict[str, Any]]:
        """Build a schema from a JSON Schema mapping describing a JSON object."""
        if not isinstance(schema, Mapping):
            raise SchemaError("a JSON Schema must be a mapping")
        try:
            # The round trip checks JSON compatibility and detaches us from the caller's object.
            raw = json.loads(json.dumps(dict(schema), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"schema is not JSON-serializable: {exc}") from exc
        root_type = raw.get("type")
        if root_type is not None and root_type != "object":
            raise SchemaError("the root of a schema must describe a JSON object")
        _check_schema(raw, raw, "#")
        normalized = _normalize(raw)
        return Schema(
            json_schema=normalized, model=None, validator=_JsonSchemaValidator(normalized)
        )

    # -- properties -----------------------------------------------------------------------

    @property
    def json_schema(self) -> dict[str, Any]:
        """A copy of the normalized JSON Schema."""
        return copy.deepcopy(self._json_schema)

    @property
    def canonical_json(self) -> str:
        """The normalized schema as canonical JSON (sorted keys, no whitespace)."""
        return self._canonical

    @property
    def schema_hash(self) -> str:
        """Stable SHA-256 hex digest of the normalized schema (the grammar cache key)."""
        return self._hash

    @property
    def model(self) -> type[BaseModel] | None:
        """The Pydantic model class, or ``None`` when built from a JSON Schema."""
        return self._model

    # -- validation -----------------------------------------------------------------------

    def validate(self, text_or_obj: object) -> ValidationReport[T]:
        """Validate a JSON document given as text or as an already parsed object.

        Never raises for bad *data*: problems come back as :class:`ValidationIssue` objects in
        the report. On success ``value`` is a model instance (Pydantic input) or a ``dict``.
        """
        if isinstance(text_or_obj, str):
            text = text_or_obj
        else:
            try:
                text = json.dumps(text_or_obj, allow_nan=False)
            except (TypeError, ValueError, RecursionError) as exc:
                issue = ValidationIssue((), "not_json", f"object is not JSON-compatible: {exc}")
                return ValidationReport(None, (issue,))
        if self._model is not None:
            return self._validate_model(self._model, text)
        return self._validate_dict(text)

    def _validate_model(self, model: type[BaseModel], text: str) -> ValidationReport[T]:
        try:
            instance = model.model_validate_json(text, strict=True)
        except PydanticValidationError as exc:
            issues = tuple(
                ValidationIssue(
                    path=tuple(err["loc"]),
                    kind=str(err["type"]),
                    message=str(err["msg"]),
                )
                for err in exc.errors(include_url=False, include_context=False, include_input=False)
            )
            return ValidationReport(None, issues)
        return ValidationReport(cast("T", instance))

    def _validate_dict(self, text: str) -> ValidationReport[T]:
        validator = self._validator
        if validator is None:  # pragma: no cover - guarded by __init__
            raise SchemaError("schema has no validator")
        try:
            parsed = json.loads(text, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            message = f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
            return ValidationReport(None, (ValidationIssue((), "json_invalid", message),))
        except (ValueError, RecursionError) as exc:
            return ValidationReport(
                None, (ValidationIssue((), "json_invalid", f"invalid JSON: {exc}"),)
            )
        if not isinstance(parsed, dict):
            issue = ValidationIssue((), "type", f"expected object, got {_json_type_name(parsed)}")
            return ValidationReport(None, (issue,))
        issues = validator.validate(parsed)
        if issues:
            return ValidationReport(None, tuple(issues))
        return ValidationReport(cast("T", parsed))

    def __repr__(self) -> str:
        source = self._model.__name__ if self._model is not None else "json-schema"
        return f"Schema({source}, hash={self._hash[:12]})"


# -- normalization and hashing ------------------------------------------------------------


def _canonical_json(schema: Mapping[str, Any]) -> str:
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalize(node: Any) -> Any:
    """Return a normalized deep copy of a schema node (see module docstring)."""
    if not isinstance(node, dict):
        return copy.deepcopy(node)
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _ANNOTATION_KEYS:
            continue
        if key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
            out[key] = {name: _normalize(sub) for name, sub in value.items()}
        elif key in _SCHEMA_LIST_KEYS and isinstance(value, list):
            out[key] = [_normalize(sub) for sub in value]
        elif key in _SCHEMA_KEYS:
            if isinstance(value, list):
                out[key] = [_normalize(sub) for sub in value]
            else:
                out[key] = _normalize(value)
        elif key in ("required", "type") and _is_str_list(value):
            out[key] = sorted(set(value))
        else:
            out[key] = copy.deepcopy(value)
    return out


# -- schema well-formedness ---------------------------------------------------------------


def _check_schema(node: Any, root: dict[str, Any], where: str) -> None:
    """Reject schemas the built-in validator cannot enforce faithfully."""
    if isinstance(node, bool):
        return
    if not isinstance(node, dict):
        raise SchemaError(f"{where}: a schema must be an object or a boolean")
    for key in node:
        if key in _UNSUPPORTED_KEYWORDS:
            raise SchemaError(f"{where}: keyword {key!r} is not supported")
    _check_scalar_keywords(node, where)
    ref = node.get("$ref")
    if ref is not None:
        _check_ref(ref, root, where)
    for key in _SCHEMA_KEYS:
        if key in node:
            sub = node[key]
            for index, item in enumerate(sub if isinstance(sub, list) else [sub]):
                _check_schema(item, root, f"{where}/{key}/{index}")
    for key in _SCHEMA_MAP_KEYS:
        if key in node:
            if not isinstance(node[key], dict):
                raise SchemaError(f"{where}/{key}: expected an object")
            for name, sub in node[key].items():
                _check_schema(sub, root, f"{where}/{key}/{name}")
    for key in _SCHEMA_LIST_KEYS:
        if key in node:
            subs = node[key]
            if not isinstance(subs, list) or (not subs and key != "prefixItems"):
                raise SchemaError(f"{where}/{key}: expected a non-empty list of schemas")
            for index, item in enumerate(subs):
                _check_schema(item, root, f"{where}/{key}/{index}")


def _check_scalar_keywords(node: dict[str, Any], where: str) -> None:
    declared = node.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        if not names or any(not isinstance(n, str) or n not in _JSON_TYPES for n in names):
            raise SchemaError(f"{where}: invalid 'type'")
    required = node.get("required")
    if required is not None and (
        not isinstance(required, list) or any(not isinstance(r, str) for r in required)
    ):
        raise SchemaError(f"{where}: 'required' must be a list of strings")
    if "enum" in node and not isinstance(node["enum"], list):
        raise SchemaError(f"{where}: 'enum' must be a list")
    pattern = node.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise SchemaError(f"{where}: 'pattern' must be a string")
        _compile(pattern, where)
    pattern_props = node.get("patternProperties", {})
    if not isinstance(pattern_props, dict):
        raise SchemaError(f"{where}: 'patternProperties' must be an object")
    for pattern in pattern_props:
        _compile(pattern, where)
    for key in ("minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"):
        if key in node and not _is_int(node[key], non_negative=True):
            raise SchemaError(f"{where}: {key!r} must be a non-negative integer")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
        if key in node and not _is_number(node[key]):
            raise SchemaError(f"{where}: {key!r} must be a number")
    if "multipleOf" in node and node["multipleOf"] <= 0:
        raise SchemaError(f"{where}: 'multipleOf' must be greater than zero")
    if "uniqueItems" in node and not isinstance(node["uniqueItems"], bool):
        raise SchemaError(f"{where}: 'uniqueItems' must be a boolean")


def _check_ref(ref: Any, root: dict[str, Any], where: str) -> None:
    if not isinstance(ref, str):
        raise SchemaError(f"{where}: '$ref' must be a string")
    seen: set[str] = set()
    current: Any = ref
    while isinstance(current, str):
        if current in seen:
            raise SchemaError(f"{where}: '$ref' cycle through {current!r}")
        seen.add(current)
        target = _resolve_ref(root, current, where)
        if not isinstance(target, (dict, bool)):
            raise SchemaError(f"{where}: '$ref' {current!r} does not point to a schema")
        current = target.get("$ref") if isinstance(target, dict) else None


def _resolve_ref(root: dict[str, Any], ref: str, where: str) -> Any:
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        raise SchemaError(f"{where}: only local '$ref' values are supported, got {ref!r}")
    current: Any = root
    for raw in ref[2:].split("/"):
        token = unquote(raw).replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise SchemaError(f"{where}: '$ref' {ref!r} cannot be resolved")
    return current


def _compile(pattern: str, where: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise SchemaError(f"{where}: invalid regular expression {pattern!r}: {exc}") from exc


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _is_int(value: object, *, non_negative: bool = False) -> bool:
    ok = isinstance(value, int) and not isinstance(value, bool)
    return ok and (not non_negative or cast("int", value) >= 0)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# -- built-in validator -------------------------------------------------------------------


def _reject_constant(name: str) -> object:
    raise ValueError(f"{name} is not valid JSON")


def _json_type_name(value: object) -> str:
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


def _type_matches(value: object, expected: str) -> bool:
    if expected == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int)  # strict, like Pydantic: 5.0 is not an integer
    if expected == "number":
        return _is_number(value)
    return _json_type_name(value) == expected


def _json_equal(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        return False
    return a == b


def _is_multiple(value: float, divisor: float) -> bool:
    try:
        return Decimal(str(value)) % Decimal(str(divisor)) == 0
    except (InvalidOperation, ValueError):
        return False


class _JsonSchemaValidator:
    """Validates a parsed JSON value against a (normalized) JSON Schema."""

    def __init__(self, root: dict[str, Any]) -> None:
        self._root = root

    def validate(self, value: object) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        self._check(value, self._root, (), issues, 0)
        return issues

    def _check(
        self,
        value: object,
        schema: Any,
        path: tuple[PathItem, ...],
        issues: list[ValidationIssue],
        depth: int,
    ) -> None:
        if schema is True:
            return
        if schema is False:
            issues.append(ValidationIssue(path, "false_schema", "no value is allowed here"))
            return
        if depth > _MAX_DEPTH:
            issues.append(ValidationIssue(path, "recursion_limit", "schema nesting is too deep"))
            return
        nxt = depth + 1
        ref = schema.get("$ref")
        if ref is not None:
            self._check(value, _resolve_ref(self._root, ref, "#"), path, issues, nxt)
        if not self._check_type(value, schema, path, issues):
            return
        if "enum" in schema and not any(_json_equal(value, option) for option in schema["enum"]):
            issues.append(ValidationIssue(path, "enum", "value is not one of the allowed values"))
        if "const" in schema and not _json_equal(value, schema["const"]):
            issues.append(
                ValidationIssue(path, "const", "value does not equal the required constant")
            )
        if isinstance(value, str):
            self._check_string(value, schema, path, issues)
        elif _is_number(value):
            self._check_number(cast("float", value), schema, path, issues)
        elif isinstance(value, list):
            self._check_array(value, schema, path, issues, nxt)
        elif isinstance(value, dict):
            self._check_object(value, schema, path, issues, nxt)
        self._check_combinators(value, schema, path, issues, nxt)

    def _check_type(
        self,
        value: object,
        schema: dict[str, Any],
        path: tuple[PathItem, ...],
        issues: list[ValidationIssue],
    ) -> bool:
        declared = schema.get("type")
        if declared is None:
            return True
        names = declared if isinstance(declared, list) else [declared]
        if any(_type_matches(value, name) for name in names):
            return True
        expected = " or ".join(names)
        message = f"expected {expected}, got {_json_type_name(value)}"
        issues.append(ValidationIssue(path, "type", message))
        return False

    def _check_string(
        self,
        value: str,
        schema: dict[str, Any],
        path: tuple[PathItem, ...],
        issues: list[ValidationIssue],
    ) -> None:
        if "minLength" in schema and len(value) < schema["minLength"]:
            message = f"string is shorter than {schema['minLength']} characters"
            issues.append(ValidationIssue(path, "min_length", message))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            message = f"string is longer than {schema['maxLength']} characters"
            issues.append(ValidationIssue(path, "max_length", message))
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            issues.append(ValidationIssue(path, "pattern", f"string does not match {pattern!r}"))

    def _check_number(
        self,
        value: float,
        schema: dict[str, Any],
        path: tuple[PathItem, ...],
        issues: list[ValidationIssue],
    ) -> None:
        violations = (
            ("minimum", "less than", value < schema.get("minimum", value)),
            ("maximum", "greater than", value > schema.get("maximum", value)),
            ("exclusiveMinimum", "not greater than", value <= schema.get("exclusiveMinimum", -inf)),
            ("exclusiveMaximum", "not less than", value >= schema.get("exclusiveMaximum", inf)),
        )
        for key, wording, violated in violations:
            if key in schema and violated:
                issues.append(ValidationIssue(path, key, f"number is {wording} {schema[key]}"))
        if "multipleOf" in schema and not _is_multiple(value, schema["multipleOf"]):
            message = f"number is not a multiple of {schema['multipleOf']}"
            issues.append(ValidationIssue(path, "multiple_of", message))

    def _check_array(
        self,
        value: list[Any],
        schema: dict[str, Any],
        path: tuple[PathItem, ...],
        issues: list[ValidationIssue],
        depth: int,
    ) -> None:
        if "minItems" in schema and len(value) < schema["minItems"]:
            message = f"array has fewer than {schema['minItems']} items"
            issues.append(ValidationIssue(path, "min_items", message))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            message = f"array has more than {schema['maxItems']} items"
            issues.append(ValidationIssue(path, "max_items", message))
        if schema.get("uniqueItems") is True and any(
            _json_equal(value[i], value[j]) for i in range(len(value)) for j in range(i)
        ):
            issues.append(ValidationIssue(path, "unique_items", "array items are not unique"))
        prefix: list[Any] = []
        rest: Any = True
        if isinstance(schema.get("prefixItems"), list):
            prefix = schema["prefixItems"]
            rest = schema.get("items", True)
        elif isinstance(schema.get("items"), list):  # draft-07 tuple form
            prefix = schema["items"]
            rest = schema.get("additionalItems", True)
        else:
            rest = schema.get("items", True)
        for index, item in enumerate(value):
            sub = prefix[index] if index < len(prefix) else rest
            self._check(item, sub, (*path, index), issues, depth)

    def _check_object(
        self,
        value: dict[str, Any],
        schema: dict[str, Any],
        path: tuple[PathItem, ...],
        issues: list[ValidationIssue],
        depth: int,
    ) -> None:
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            message = f"object has fewer than {schema['minProperties']} properties"
            issues.append(ValidationIssue(path, "min_properties", message))
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            message = f"object has more than {schema['maxProperties']} properties"
            issues.append(ValidationIssue(path, "max_properties", message))
        for name in schema.get("required", []):
            if name not in value:
                issues.append(
                    ValidationIssue((*path, name), "required", "required property is missing")
                )
        properties: dict[str, Any] = schema.get("properties", {})
        pattern_props: dict[str, Any] = schema.get("patternProperties", {})
        additional: Any = schema.get("additionalProperties", True)
        for name, item in value.items():
            matched = False
            if name in properties:
                matched = True
                self._check(item, properties[name], (*path, name), issues, depth)
            for pattern, sub in pattern_props.items():
                if re.search(pattern, name) is not None:
                    matched = True
                    self._check(item, sub, (*path, name), issues, depth)
            if matched:
                continue
            if additional is False:
                issues.append(
                    ValidationIssue(
                        (*path, name), "additional_properties", "property is not allowed"
                    )
                )
            elif additional is not True:
                self._check(item, additional, (*path, name), issues, depth)

    def _check_combinators(
        self,
        value: object,
        schema: dict[str, Any],
        path: tuple[PathItem, ...],
        issues: list[ValidationIssue],
        depth: int,
    ) -> None:
        for sub in schema.get("allOf", []):
            self._check(value, sub, path, issues, depth)
        any_of = schema.get("anyOf")
        if any_of is not None and not any(self._matches(value, sub, path, depth) for sub in any_of):
            message = "value does not match any of the allowed schemas"
            issues.append(ValidationIssue(path, "any_of", message))
        one_of = schema.get("oneOf")
        if one_of is not None:
            count = sum(1 for sub in one_of if self._matches(value, sub, path, depth))
            if count != 1:
                message = (
                    "value does not match any of the allowed schemas"
                    if count == 0
                    else "value matches more than one of the allowed schemas"
                )
                issues.append(ValidationIssue(path, "one_of", message))
        if "not" in schema and self._matches(value, schema["not"], path, depth):
            issues.append(ValidationIssue(path, "not", "value matches a forbidden schema"))

    def _matches(self, value: object, schema: Any, path: tuple[PathItem, ...], depth: int) -> bool:
        scratch: list[ValidationIssue] = []
        self._check(value, schema, path, scratch, depth)
        return not scratch
