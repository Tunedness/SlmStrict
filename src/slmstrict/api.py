"""Public generation API (ADR-008): :func:`agenerate` and :func:`generate`.

``agenerate`` is the core. ``generate`` is a thin synchronous wrapper around it, so both follow the
same ladder and return the same :class:`~slmstrict.result.Result`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from slmstrict.backends.base import Backend
from slmstrict.config import StrictConfig
from slmstrict.ladder import Ladder
from slmstrict.result import Result
from slmstrict.schema import Schema

__all__ = ["agenerate", "generate"]

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


@overload
async def agenerate(
    backend: Backend, schema: Schema[T], prompt: str, *, config: StrictConfig | None = None
) -> Result[T]: ...


@overload
async def agenerate(
    backend: Backend, schema: type[M], prompt: str, *, config: StrictConfig | None = None
) -> Result[M]: ...


@overload
async def agenerate(
    backend: Backend,
    schema: Mapping[str, Any],
    prompt: str,
    *,
    config: StrictConfig | None = None,
) -> Result[dict[str, Any]]: ...


async def agenerate(
    backend: Backend,
    schema: Schema[Any] | type[BaseModel] | Mapping[str, Any],
    prompt: str,
    *,
    config: StrictConfig | None = None,
) -> Result[Any]:
    """Generate a value that is valid against ``schema`` from ``prompt``.

    ``schema`` is a Pydantic model class, a JSON Schema mapping or a prepared
    :class:`~slmstrict.schema.Schema`. The value in the result is a model instance for a Pydantic
    model and a ``dict`` otherwise. ``config`` defaults to ``StrictConfig()``: two re-asks, no
    cloud fallback.

    Raises :class:`~slmstrict.errors.ValidationFailed` when the enforcement ladder cannot produce
    a valid value, :class:`~slmstrict.errors.SchemaError` for an unusable schema and
    :class:`~slmstrict.errors.BackendError` when the backend fails.
    """
    ladder = _prepare(backend, prompt, config)
    prepared = schema if isinstance(schema, Schema) else Schema.of(schema)
    return await ladder.run(prepared, prompt)


@overload
def generate(
    backend: Backend, schema: Schema[T], prompt: str, *, config: StrictConfig | None = None
) -> Result[T]: ...


@overload
def generate(
    backend: Backend, schema: type[M], prompt: str, *, config: StrictConfig | None = None
) -> Result[M]: ...


@overload
def generate(
    backend: Backend,
    schema: Mapping[str, Any],
    prompt: str,
    *,
    config: StrictConfig | None = None,
) -> Result[dict[str, Any]]: ...


def generate(
    backend: Backend,
    schema: Schema[Any] | type[BaseModel] | Mapping[str, Any],
    prompt: str,
    *,
    config: StrictConfig | None = None,
) -> Result[Any]:
    """Synchronous form of :func:`agenerate`; same arguments, same result, same errors.

    It runs :func:`agenerate` on a fresh event loop, so it cannot be used from code that already
    runs inside an event loop: there it raises :class:`RuntimeError` and the caller awaits
    :func:`agenerate` instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "generate() cannot be called from a running event loop; "
            "use 'await agenerate(...)' instead"
        )
    return asyncio.run(agenerate(backend, schema, prompt, config=config))


def _prepare(backend: Backend, prompt: str, config: StrictConfig | None) -> Ladder:
    if not isinstance(backend, Backend):
        raise TypeError("backend must implement the Backend protocol")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a str")
    if config is not None and not isinstance(config, StrictConfig):
        raise TypeError("config must be a StrictConfig")
    return Ladder(backend, config)
