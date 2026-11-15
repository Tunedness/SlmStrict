"""JSON Schema to GBNF grammar compilation with a per-process LRU cache (ADR-006).

The compilation itself is ``llama-cpp-python``'s built-in ``LlamaGrammar.from_json_schema``;
this module only adds the cache and wraps the result in the opaque
:class:`~slmstrict.backends.base.Constraint` type. Compiling a grammar for a complex schema is
slow, so a schema is compiled once per process and looked up by its stable hash afterwards.

``llama_cpp`` is imported lazily: importing this module never loads it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from types import ModuleType
from typing import Any, Final

from slmstrict.backends.base import Constraint
from slmstrict.errors import BackendError
from slmstrict.schema import Schema

__all__ = [
    "GBNF_KIND",
    "GrammarCache",
    "clear_grammar_cache",
    "compile_grammar",
    "load_llama_cpp",
]

GBNF_KIND: Final = "gbnf"
DEFAULT_CACHE_SIZE: Final = 128

_INSTALL_HINT: Final = (
    "llama-cpp-python is required for the llama.cpp backend. "
    "Install it with: pip install 'slmstrict[llamacpp]'"
)

_logger = logging.getLogger("slmstrict.backends.llamacpp")


def load_llama_cpp() -> ModuleType:
    """Import ``llama_cpp`` on first use.

    Raises :class:`ImportError` with an installation hint when the extra is not installed.
    """
    try:
        return importlib.import_module("llama_cpp")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _ordered_json(schema: Schema[Any]) -> str:
    """The normalized schema as JSON that keeps the schema's own property order.

    ``Schema.canonical_json`` sorts keys, which would force the grammar to emit properties in
    alphabetical order. GBNF fixes the emission order to the order of ``properties``, and the
    order matters to small models (for example a ``reasoning`` field before ``answer``).
    """
    return json.dumps(schema.json_schema, separators=(",", ":"))


def _cache_key(schema: Schema[Any]) -> str:
    """Schema hash plus a hash of the order-preserving JSON.

    ``schema_hash`` ignores property order, but two schemas that differ only in order need
    different grammars, so the order-sensitive digest is part of the key.
    """
    ordered = hashlib.sha256(_ordered_json(schema).encode("utf-8")).hexdigest()
    return f"{schema.schema_hash}:{ordered}"


def _compile_with_llama_cpp(schema: Schema[Any]) -> Constraint:
    llama_cpp = load_llama_cpp()
    try:
        grammar = llama_cpp.LlamaGrammar.from_json_schema(_ordered_json(schema))
    except Exception as exc:
        # Do not put the schema in the message: it may carry business-specific names.
        raise BackendError(
            f"could not compile a GBNF grammar from the schema ({type(exc).__name__})"
        ) from exc
    return Constraint(kind=GBNF_KIND, payload=grammar)


class GrammarCache:
    """Thread-safe LRU cache of compiled grammars.

    The key is :attr:`Schema.schema_hash` combined with the schema's property order.

    The compile step runs under the lock, so concurrent callers asking for the same schema
    trigger exactly one compilation.
    """

    def __init__(
        self,
        *,
        maxsize: int = DEFAULT_CACHE_SIZE,
        compiler: Callable[[Schema[Any]], Constraint] = _compile_with_llama_cpp,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self._maxsize = maxsize
        self._compiler = compiler
        self._entries: OrderedDict[str, Constraint] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_compile(self, schema: Schema[Any]) -> Constraint:
        """Return the cached grammar constraint for ``schema``, compiling it on a miss."""
        key = _cache_key(schema)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return cached
            constraint = self._compiler(schema)
            self._entries[key] = constraint
            if len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)
            _logger.debug("compiled grammar for schema %s", schema.schema_hash[:12])
            return constraint

    def clear(self) -> None:
        """Drop every cached grammar."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, schema: object) -> bool:
        with self._lock:
            return isinstance(schema, Schema) and _cache_key(schema) in self._entries


_default_cache = GrammarCache()


def compile_grammar(schema: Schema[Any]) -> Constraint:
    """Compile ``schema`` to a ``gbnf`` :class:`Constraint` (cached per process)."""
    return _default_cache.get_or_compile(schema)


def clear_grammar_cache() -> None:
    """Empty the process-wide grammar cache (mainly useful in tests)."""
    _default_cache.clear()
