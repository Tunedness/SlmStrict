"""ONNX Runtime GenAI backend adapter (ADR-004, ADR-006).

``onnxruntime-genai`` exposes no grammar hook that SlmStrict uses, so this adapter reports
``supports_grammar = False`` and the enforcement ladder starts at the repair step. The
``onnxruntime_genai`` package is imported lazily: ``import slmstrict`` never loads it, and a
missing ``onnx`` extra produces an actionable :class:`ImportError`.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import math
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from slmstrict.backends.base import Backend, Constraint
from slmstrict.errors import BackendError

__all__ = ["OnnxBackend"]

_logger = logging.getLogger("slmstrict.backends.onnx")

_INSTALL_HINT = (
    "The ONNX backend needs the 'onnx' extra: install it with `pip install 'slmstrict[onnx]'`."
)


class _GenaiTokenizer(Protocol):
    """The slice of ``onnxruntime_genai.Tokenizer`` this adapter uses."""

    def encode(self, text: str) -> Sequence[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


class _GenaiParams(Protocol):
    """The slice of ``onnxruntime_genai.GeneratorParams`` this adapter uses."""

    def set_search_options(self, **options: float | int | bool) -> None: ...


class _GenaiGenerator(Protocol):
    """The slice of ``onnxruntime_genai.Generator`` this adapter uses."""

    def append_tokens(self, tokens: Sequence[int]) -> None: ...

    def is_done(self) -> bool: ...

    def generate_next_token(self) -> None: ...

    def get_sequence(self, index: int) -> Sequence[int]: ...


class _GenaiModule(Protocol):
    """The slice of the ``onnxruntime_genai`` module this adapter uses."""

    def Model(self, config_path: str) -> object: ...  # noqa: N802 - mirrors the library API

    def Tokenizer(self, model: object) -> _GenaiTokenizer: ...  # noqa: N802

    def GeneratorParams(self, model: object) -> _GenaiParams: ...  # noqa: N802

    def Generator(  # noqa: N802
        self, model: object, params: _GenaiParams
    ) -> _GenaiGenerator: ...


def _import_genai() -> _GenaiModule:
    """Import ``onnxruntime_genai`` on first use, with an install hint when it is missing."""
    try:
        module = importlib.import_module("onnxruntime_genai")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return cast("_GenaiModule", module)


class OnnxBackend:
    """Backend adapter on top of ``onnxruntime-genai`` (CPU execution provider by default).

    ``model_dir`` is an ONNX GenAI model directory (the one holding ``genai_config.json``).
    ``max_length`` is passed to the runtime as-is and counts prompt plus generated tokens.
    ``temperature`` of ``0`` selects greedy decoding; a positive value enables sampling.

    The model and tokenizer are loaded once, in the constructor, so a bad directory fails
    early. Each :meth:`agenerate` call builds its own generator and runs it in a worker
    thread, which keeps the event loop free.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_length: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite, non-negative number")
        path = Path(model_dir)
        if not path.is_dir():
            raise BackendError(f"ONNX model directory does not exist: {path}")

        genai = _import_genai()
        try:
            model = genai.Model(str(path))
            tokenizer = genai.Tokenizer(model)
        except Exception as exc:
            raise BackendError(f"Failed to load the ONNX model from {path}") from exc

        self._genai = genai
        self._model = model
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._temperature = temperature

    @property
    def supports_grammar(self) -> bool:
        """Always ``False``: grammar enforcement is not available on this backend (ADR-006)."""
        return False

    async def agenerate(
        self,
        prompt: str,
        *,
        constraint: Constraint | None = None,
    ) -> str:
        """Generate a completion for ``prompt`` and return only the newly generated text."""
        if constraint is not None:
            raise BackendError(
                "OnnxBackend does not support grammar constraints "
                f"(got constraint kind {constraint.kind!r}); pass constraint=None"
            )
        # asyncio.to_thread cannot interrupt a running thread, so cancellation is signalled to
        # the token loop through this event; otherwise a timed-out call keeps burning CPU.
        cancel = threading.Event()
        try:
            return await asyncio.to_thread(self._generate_sync, prompt, cancel)
        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception as exc:
            # The message deliberately omits the prompt and the output (ADR-007).
            raise BackendError("ONNX generation failed") from exc

    def _generate_sync(self, prompt: str, cancel: threading.Event) -> str:
        params = self._genai.GeneratorParams(self._model)
        if self._temperature > 0:
            params.set_search_options(
                max_length=self._max_length, do_sample=True, temperature=self._temperature
            )
        else:
            params.set_search_options(max_length=self._max_length, do_sample=False)
        generator = self._genai.Generator(self._model, params)
        prompt_tokens = self._tokenizer.encode(prompt)
        generator.append_tokens(prompt_tokens)
        while not generator.is_done():
            if cancel.is_set():
                return ""  # the awaiting task is gone; the result is discarded
            generator.generate_next_token()
        # The runtime returns prompt and completion together; keep the completion only.
        completion = generator.get_sequence(0)[len(prompt_tokens) :]
        _logger.debug("onnx generation finished: %d new tokens", len(completion))
        return self._tokenizer.decode(completion)


if TYPE_CHECKING:
    # Static proof (checked by `mypy --strict src`) that OnnxBackend satisfies the Backend protocol.
    def _backend_conformance(backend: OnnxBackend) -> Backend:
        return backend
