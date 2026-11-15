from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from slmstrict import BackendError, Constraint, Schema
from slmstrict.backends import Backend
from slmstrict.backends._grammar import clear_grammar_cache
from slmstrict.backends.llamacpp import LlamaCppBackend


class Person(BaseModel):
    name: str
    age: int


class FakeLlama:
    """Stands in for ``llama_cpp.Llama``; records every ``create_completion`` call."""

    def __init__(self, text: str = '{"name": "Ada", "age": 36}', *, delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.calls: list[dict[str, Any]] = []
        self.threads: list[int] = []
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def create_completion(self, prompt: str, **kwargs: Any) -> object:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.threads.append(threading.get_ident())
            self.calls.append({"prompt": prompt, **kwargs})
            if self.delay:
                time.sleep(self.delay)
            return {"choices": [{"text": self.text, "finish_reason": "stop"}]}
        finally:
            with self._guard:
                self.active -= 1


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    clear_grammar_cache()


def test_satisfies_backend_protocol_and_supports_grammar() -> None:
    backend = LlamaCppBackend(FakeLlama())
    assert isinstance(backend, Backend)
    assert backend.supports_grammar is True


def test_importing_the_module_does_not_load_llama_cpp() -> None:
    code = (
        "import sys, slmstrict, slmstrict.backends.llamacpp;"
        "sys.exit(1 if 'llama_cpp' in sys.modules else 0)"
    )
    completed = subprocess.run([sys.executable, "-I", "-c", code], check=False)
    assert completed.returncode == 0


async def test_agenerate_returns_the_completion_text() -> None:
    llama = FakeLlama("hello")
    backend = LlamaCppBackend(llama, max_tokens=64, temperature=0.2, seed=7)
    assert await backend.agenerate("say hi") == "hello"
    assert llama.calls == [
        {"prompt": "say hi", "max_tokens": 64, "temperature": 0.2, "grammar": None, "seed": 7}
    ]


async def test_gbnf_constraint_is_passed_as_grammar() -> None:
    llama = FakeLlama()
    backend = LlamaCppBackend(llama)
    grammar = object()
    await backend.agenerate("p", constraint=Constraint("gbnf", grammar))
    assert llama.calls[0]["grammar"] is grammar


async def test_unknown_constraint_kind_is_rejected() -> None:
    llama = FakeLlama()
    backend = LlamaCppBackend(llama)
    with pytest.raises(BackendError, match="unsupported constraint kind"):
        await backend.agenerate("p", constraint=Constraint("regex", "x"))
    assert llama.calls == []


async def test_generation_runs_off_the_event_loop_thread() -> None:
    llama = FakeLlama()
    await LlamaCppBackend(llama).agenerate("p")
    assert llama.threads
    assert llama.threads[0] != threading.get_ident()


async def test_event_loop_stays_responsive_during_generation() -> None:
    backend = LlamaCppBackend(FakeLlama(delay=0.3))
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await backend.agenerate("p")
    task.cancel()
    assert ticks >= 5  # a blocked loop would have produced ~0 ticks


async def test_concurrent_calls_on_one_backend_are_serialized() -> None:
    llama = FakeLlama(delay=0.05)
    backend = LlamaCppBackend(llama)
    await asyncio.gather(*(backend.agenerate(f"p{i}") for i in range(4)))
    assert len(llama.calls) == 4
    assert llama.max_active == 1


async def test_waiting_callers_do_not_occupy_worker_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    in_flight = 0
    max_in_flight = 0
    real_to_thread = asyncio.to_thread

    async def counting_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            return await real_to_thread(func, *args, **kwargs)
        finally:
            in_flight -= 1

    monkeypatch.setattr("slmstrict.backends.llamacpp.asyncio.to_thread", counting_to_thread)
    llama = FakeLlama(delay=0.01)
    backend = LlamaCppBackend(llama)
    await asyncio.gather(*(backend.agenerate(f"p{i}") for i in range(20)))
    assert len(llama.calls) == 20
    assert max_in_flight == 1


def test_backend_is_reusable_across_event_loops() -> None:
    backend = LlamaCppBackend(FakeLlama("ok"))
    assert asyncio.run(backend.agenerate("a")) == "ok"
    assert asyncio.run(backend.agenerate("b")) == "ok"


async def test_runtime_failure_becomes_backend_error_without_leaking_the_prompt() -> None:
    class Exploding(FakeLlama):
        def create_completion(self, prompt: str, **kwargs: Any) -> object:
            raise RuntimeError(f"decode failed for {prompt}")

    backend = LlamaCppBackend(Exploding())
    with pytest.raises(BackendError, match="RuntimeError") as info:
        await backend.agenerate("patient-secret")
    assert "patient-secret" not in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)


@pytest.mark.parametrize("response", [{}, {"choices": []}, {"choices": [{"text": 5}]}, "oops"])
async def test_malformed_response_becomes_backend_error(response: object) -> None:
    class Odd(FakeLlama):
        def create_completion(self, prompt: str, **kwargs: Any) -> object:
            return response

    with pytest.raises(BackendError, match="response shape"):
        await LlamaCppBackend(Odd()).agenerate("p")


def test_constructor_validates_settings() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        LlamaCppBackend(FakeLlama(), max_tokens=0)
    with pytest.raises(ValueError, match="temperature"):
        LlamaCppBackend(FakeLlama(), temperature=-1)


def test_build_constraint_compiles_once_per_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled: list[str] = []

    def from_json_schema(text: str) -> str:
        compiled.append(text)
        return "grammar"

    module = types.ModuleType("llama_cpp")
    module.LlamaGrammar = types.SimpleNamespace(from_json_schema=from_json_schema)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", module)

    backend = LlamaCppBackend(FakeLlama())
    first = backend.build_constraint(Schema.of(Person))
    second = backend.build_constraint(Schema.of(Person))
    assert first is second
    assert first.kind == "gbnf"
    assert len(compiled) == 1


def _install_fake_llama_cpp(monkeypatch: pytest.MonkeyPatch, created: list[dict[str, Any]]) -> None:
    class Llama(FakeLlama):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            created.append(kwargs)

    module = types.ModuleType("llama_cpp")
    module.Llama = Llama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", module)


async def test_model_path_loads_llama_with_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: list[dict[str, Any]] = []
    _install_fake_llama_cpp(monkeypatch, created)
    model_file = tmp_path / "tiny.gguf"
    model_file.write_bytes(b"")
    backend = LlamaCppBackend(model_file, n_ctx=2048, n_gpu_layers=3)
    assert created == [
        {"model_path": str(model_file), "n_ctx": 2048, "n_gpu_layers": 3, "verbose": False}
    ]
    assert await backend.agenerate("p") == '{"name": "Ada", "age": 36}'


def test_missing_model_file_raises_backend_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_llama_cpp(monkeypatch, [])
    with pytest.raises(BackendError, match="not found"):
        LlamaCppBackend(str(tmp_path / "absent.gguf"))


def test_model_load_failure_becomes_backend_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Llama:
        def __init__(self, **kwargs: Any) -> None:
            raise ValueError("bad magic")

    module = types.ModuleType("llama_cpp")
    module.Llama = Llama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    model_file = tmp_path / "broken.gguf"
    model_file.write_bytes(b"")
    with pytest.raises(BackendError, match="ValueError"):
        LlamaCppBackend(model_file)


def test_model_path_without_the_extra_raises_import_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(ImportError, match=r"slmstrict\[llamacpp\]"):
        LlamaCppBackend(tmp_path / "x.gguf")


def test_a_ready_made_llama_instance_needs_no_llama_cpp_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    LlamaCppBackend(FakeLlama())  # must not raise
