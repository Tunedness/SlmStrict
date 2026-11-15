"""llama.cpp backend adapter built on ``llama-cpp-python`` (ADR-004, ADR-006).

This is the only backend that supports the grammar step of the enforcement ladder: the JSON
Schema is compiled to GBNF (see :mod:`slmstrict.backends._grammar`) and passed to llama.cpp's
``grammar=`` argument, so the decoder can only emit schema-shaped text.

``llama_cpp`` is imported lazily. Importing this module does not load it; constructing a
backend from a model path, or compiling a grammar, does.
"""

from __future__ import annotations

import asyncio
import os
import threading
import weakref
from typing import TYPE_CHECKING, Any, Protocol

from slmstrict.backends._grammar import GBNF_KIND, compile_grammar, load_llama_cpp
from slmstrict.errors import BackendError

if TYPE_CHECKING:
    from slmstrict.backends.base import Backend, Constraint
    from slmstrict.schema import Schema

__all__ = ["LlamaCppBackend", "LlamaModel"]


class LlamaModel(Protocol):
    """The slice of ``llama_cpp.Llama`` the adapter relies on.

    ``grammar`` is typed ``Any`` on purpose: parameters are contravariant, so naming a wider
    type such as ``object`` would make the real ``Llama`` (which takes ``LlamaGrammar | None``)
    fail the structural check.
    """

    def create_completion(
        self,
        prompt: str,
        *,
        max_tokens: int | None = ...,
        temperature: float = ...,
        grammar: Any = ...,
        seed: int | None = ...,
    ) -> object: ...


class LlamaCppBackend:
    """:class:`~slmstrict.backends.base.Backend` implementation for llama.cpp.

    ``model`` is either a path to a GGUF file (loaded right away, which blocks; build the
    backend before entering a latency-sensitive path) or an already constructed
    ``llama_cpp.Llama`` instance. Generation settings live here, not on the call.

    A ``Llama`` instance is not safe for concurrent use, so calls on one backend are serialized.
    Waiting callers queue on a per-event-loop ``asyncio.Lock`` and only the call that actually
    generates occupies a worker thread, so the default executor is not drained by idle waiters.
    A thread lock inside the worker is the safety net across event loops and for a call whose
    awaiting task was cancelled while its thread is still running.
    """

    supports_grammar = True

    def __init__(
        self,
        model: str | os.PathLike[str] | LlamaModel,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if temperature < 0:
            raise ValueError("temperature must not be negative")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._seed = seed
        self._lock = threading.Lock()
        self._loop_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )
        if isinstance(model, str | os.PathLike):
            self._llama: LlamaModel = self._load(
                os.fspath(model), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers
            )
        else:
            self._llama = model

    @staticmethod
    def _load(path: str, *, n_ctx: int, n_gpu_layers: int) -> LlamaModel:
        llama_cpp = load_llama_cpp()
        if not os.path.isfile(path):
            raise BackendError(f"GGUF model file not found: {path}")
        try:
            model: LlamaModel = llama_cpp.Llama(
                model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False
            )
        except Exception as exc:
            raise BackendError(f"could not load the GGUF model ({type(exc).__name__})") from exc
        return model

    def build_constraint(self, schema: Schema[Any]) -> Constraint:
        """Compile ``schema`` to a grammar :class:`Constraint` (cached per process by hash)."""
        return compile_grammar(schema)

    async def agenerate(
        self,
        prompt: str,
        *,
        constraint: Constraint | None = None,
    ) -> str:
        """Generate a completion, constrained by ``constraint`` when one is given."""
        grammar: object | None = None
        if constraint is not None:
            if constraint.kind != GBNF_KIND:
                raise BackendError(f"unsupported constraint kind {constraint.kind!r}")
            grammar = constraint.payload
        async with self._loop_lock():
            return await asyncio.to_thread(self._complete, prompt, grammar)

    def _loop_lock(self) -> asyncio.Lock:
        # An asyncio.Lock binds to one event loop; the sync wrapper may use a fresh loop per call.
        loop = asyncio.get_running_loop()
        lock = self._loop_locks.get(loop)
        if lock is None:
            lock = self._loop_locks[loop] = asyncio.Lock()
        return lock

    def _complete(self, prompt: str, grammar: object | None) -> str:
        # Error messages deliberately exclude the prompt and the output (ADR-007).
        with self._lock:
            try:
                response = self._llama.create_completion(
                    prompt,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    grammar=grammar,
                    seed=self._seed,
                )
            except Exception as exc:
                raise BackendError(f"llama.cpp generation failed ({type(exc).__name__})") from exc
        return _extract_text(response)


def _extract_text(response: object) -> str:
    try:
        text = response["choices"][0]["text"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise BackendError("unexpected llama.cpp response shape") from exc
    if not isinstance(text, str):
        raise BackendError("unexpected llama.cpp response shape")
    return text


if TYPE_CHECKING:
    # Static proofs, checked by `mypy --strict src` (the command CI runs).

    class _RealGrammar:
        """Stands in for ``llama_cpp.LlamaGrammar``."""

    class _RealLlamaSignature:
        """Mirrors ``llama_cpp.Llama.create_completion`` (0.3.x), where ``grammar`` is a
        dedicated grammar type. ``LlamaModel`` must accept it, otherwise a real ``Llama`` would
        fail ``LlamaCppBackend(llama)`` in users' mypy runs. Keep ``grammar`` in ``LlamaModel``
        wide (``Any``): parameters are contravariant.
        """

        def create_completion(
            self,
            prompt: str | list[int],
            suffix: str | None = None,
            max_tokens: int | None = 16,
            temperature: float = 0.8,
            seed: int | None = None,
            grammar: _RealGrammar | None = None,
        ) -> dict[str, Any]: ...

    def _llama_model_conformance(llama: _RealLlamaSignature) -> LlamaModel:
        return llama

    def _backend_conformance(backend: LlamaCppBackend) -> Backend:
        return backend
