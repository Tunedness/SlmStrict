"""Unit tests for OnnxBackend; ``onnxruntime_genai`` is replaced by a fake module."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from slmstrict.backends import Backend, Constraint
from slmstrict.backends import onnx as onnx_module
from slmstrict.backends.onnx import OnnxBackend
from slmstrict.errors import BackendError, SlmStrictError


class FakeTokenizer:
    def __init__(self) -> None:
        self.decoded: list[Sequence[int]] = []

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, tokens: Sequence[int]) -> str:
        self.decoded.append(list(tokens))
        return "".join(chr(ord("a") + t) for t in tokens)


class FakeParams:
    def __init__(self) -> None:
        self.options: dict[str, Any] = {}

    def set_search_options(self, **options: Any) -> None:
        self.options = options


class FakeGenerator:
    def __init__(self, module: FakeGenai, params: FakeParams) -> None:
        self.module = module
        self.params = params
        self.sequence: list[int] = []
        self.steps = 0

    def append_tokens(self, tokens: Sequence[int]) -> None:
        self.sequence.extend(tokens)

    def is_done(self) -> bool:
        return self.steps >= len(self.module.completion)

    def generate_next_token(self) -> None:
        self.module.generate_threads.append(threading.get_ident())
        self.module.generate_calls += 1
        if self.module.generate_calls == 2:
            self.module.second_step.set()
        if self.module.step_delay:
            time.sleep(self.module.step_delay)
        if self.module.fail_on_generate:
            raise RuntimeError("secret prompt text leaked here")
        self.sequence.append(self.module.completion[self.steps])
        self.steps += 1

    def get_sequence(self, index: int) -> list[int]:
        assert index == 0
        return self.sequence


class FakeGenai:
    """Stand-in for the ``onnxruntime_genai`` module."""

    def __init__(self) -> None:
        self.completion: list[int] = [4, 5, 6]
        self.fail_on_load = False
        self.fail_on_generate = False
        self.model_paths: list[str] = []
        self.tokenizer = FakeTokenizer()
        self.params: list[FakeParams] = []
        self.generate_threads: list[int] = []
        self.generate_calls = 0
        self.step_delay = 0.0
        self.second_step = threading.Event()

    def Model(self, config_path: str) -> object:  # noqa: N802
        if self.fail_on_load:
            raise RuntimeError("bad model")
        self.model_paths.append(config_path)
        return object()

    def Tokenizer(self, model: object) -> FakeTokenizer:  # noqa: N802
        return self.tokenizer

    def GeneratorParams(self, model: object) -> FakeParams:  # noqa: N802
        params = FakeParams()
        self.params.append(params)
        return params

    def Generator(self, model: object, params: FakeParams) -> FakeGenerator:  # noqa: N802
        return FakeGenerator(self, params)


@pytest.fixture
def genai(monkeypatch: pytest.MonkeyPatch) -> FakeGenai:
    fake = FakeGenai()
    monkeypatch.setattr(onnx_module, "_import_genai", lambda: fake)
    return fake


def test_conforms_to_backend_protocol(genai: FakeGenai, tmp_path: Path) -> None:
    backend: Backend = OnnxBackend(tmp_path)
    assert isinstance(backend, Backend)


def test_reports_no_grammar_support(genai: FakeGenai, tmp_path: Path) -> None:
    assert OnnxBackend(tmp_path).supports_grammar is False


async def test_constraint_is_rejected_not_ignored(genai: FakeGenai, tmp_path: Path) -> None:
    backend = OnnxBackend(tmp_path)
    with pytest.raises(BackendError, match="grammar constraints"):
        await backend.agenerate("hi", constraint=Constraint("gbnf", "root ::= x"))
    assert genai.params == []


async def test_returns_only_the_completion(genai: FakeGenai, tmp_path: Path) -> None:
    backend = OnnxBackend(tmp_path)
    assert await backend.agenerate("hello") == "efg"
    assert genai.tokenizer.decoded == [[4, 5, 6]]
    assert genai.model_paths == [str(tmp_path)]


async def test_greedy_by_default(genai: FakeGenai, tmp_path: Path) -> None:
    await OnnxBackend(tmp_path, max_length=64).agenerate("hello")
    assert genai.params[0].options == {"max_length": 64, "do_sample": False}


async def test_positive_temperature_enables_sampling(genai: FakeGenai, tmp_path: Path) -> None:
    await OnnxBackend(tmp_path, max_length=64, temperature=0.7).agenerate("hello")
    assert genai.params[0].options == {"max_length": 64, "do_sample": True, "temperature": 0.7}


async def test_generation_runs_off_the_event_loop(genai: FakeGenai, tmp_path: Path) -> None:
    await OnnxBackend(tmp_path).agenerate("hello")
    assert genai.generate_threads
    assert threading.get_ident() not in genai.generate_threads


async def test_cancelled_call_stops_the_token_loop(genai: FakeGenai, tmp_path: Path) -> None:
    total = 1000
    genai.completion = [4] * total
    genai.step_delay = 0.005
    backend = OnnxBackend(tmp_path)
    task = asyncio.create_task(backend.agenerate("hello"))
    assert await asyncio.to_thread(genai.second_step.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)  # let the worker thread notice the signal
    settled = genai.generate_calls
    await asyncio.sleep(0.1)
    assert genai.generate_calls == settled
    assert settled < total


async def test_concurrent_calls_do_not_share_state(genai: FakeGenai, tmp_path: Path) -> None:
    backend = OnnxBackend(tmp_path)
    results = await asyncio.gather(backend.agenerate("a"), backend.agenerate("b"))
    assert list(results) == ["efg", "efg"]
    assert len(genai.params) == 2


async def test_runtime_failure_becomes_backend_error_without_leaking(
    genai: FakeGenai, tmp_path: Path
) -> None:
    genai.fail_on_generate = True
    backend = OnnxBackend(tmp_path)
    with pytest.raises(BackendError) as info:
        await backend.agenerate("secret prompt text")
    assert "secret" not in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_missing_model_dir_raises_backend_error(genai: FakeGenai, tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="does not exist"):
        OnnxBackend(tmp_path / "nope")


def test_model_load_failure_raises_backend_error(genai: FakeGenai, tmp_path: Path) -> None:
    genai.fail_on_load = True
    with pytest.raises(BackendError, match="Failed to load"):
        OnnxBackend(tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_length": 0}, "max_length"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": float("inf")}, "temperature"),
    ],
)
def test_invalid_settings_are_rejected(
    genai: FakeGenai, tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        OnnxBackend(tmp_path, **kwargs)


def test_backend_error_is_part_of_the_library_hierarchy() -> None:
    assert issubclass(BackendError, SlmStrictError)


def test_missing_extra_gives_install_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A None entry in sys.modules makes `import onnxruntime_genai` raise ImportError.
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", None)
    with pytest.raises(ImportError, match=r"slmstrict\[onnx\]"):
        OnnxBackend(tmp_path)
