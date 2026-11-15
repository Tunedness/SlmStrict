"""Real-model test for the llama.cpp adapter.

Needs the ``llamacpp`` extra and a small GGUF model; the model path comes from the
``SLMSTRICT_GGUF_PATH`` environment variable and the test is skipped when it is unset.
Run with::

    SLMSTRICT_GGUF_PATH=/path/to/model.gguf \\
        uv run --extra llamacpp pytest -m integration tests/backends/test_llamacpp_integration.py
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from slmstrict import Schema
from slmstrict.backends.llamacpp import LlamaCppBackend

pytestmark = pytest.mark.integration


class Person(BaseModel):
    name: str
    age: int


@pytest.fixture(scope="module")
def backend() -> LlamaCppBackend:
    model_path = os.environ.get("SLMSTRICT_GGUF_PATH")
    if not model_path:
        pytest.skip("SLMSTRICT_GGUF_PATH is not set")
    # The variable is set, so a missing package or file is a misconfiguration, not a skip.
    try:
        importlib.import_module("llama_cpp")
    except ImportError:
        pytest.fail("SLMSTRICT_GGUF_PATH is set but llama-cpp-python is not installed")
    if not Path(model_path).is_file():
        pytest.fail(f"SLMSTRICT_GGUF_PATH does not point to a file: {model_path}")
    return LlamaCppBackend(model_path, max_tokens=128, temperature=0.0, seed=1)


async def test_grammar_constrained_output_passes_the_schema_without_repair(
    backend: LlamaCppBackend,
) -> None:
    schema = Schema.of(Person)
    constraint = backend.build_constraint(schema)
    text = await backend.agenerate(
        "Extract the person as JSON: Ada Lovelace is 36 years old.", constraint=constraint
    )
    report = schema.validate(text)
    assert report.ok, f"grammar-constrained output failed validation: {report.issues}"
