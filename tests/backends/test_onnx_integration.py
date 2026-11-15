"""Integration test for OnnxBackend against a real ONNX GenAI model.

Not run in CI. Point ``SLMSTRICT_ONNX_MODEL_DIR`` at an ONNX GenAI model directory and run:
``uv run --extra onnx pytest -m integration tests/backends/test_onnx_integration.py``
"""

from __future__ import annotations

import os

import pytest

from slmstrict.backends import Constraint
from slmstrict.backends.onnx import OnnxBackend
from slmstrict.errors import BackendError

pytestmark = pytest.mark.integration

_MODEL_DIR = os.environ.get("SLMSTRICT_ONNX_MODEL_DIR")
_MAX_LENGTH = 256


@pytest.fixture(scope="module")
def backend() -> OnnxBackend:
    if not _MODEL_DIR:
        pytest.skip("SLMSTRICT_ONNX_MODEL_DIR is not set")
    pytest.importorskip("onnxruntime_genai")
    return OnnxBackend(_MODEL_DIR, max_length=_MAX_LENGTH)


async def test_generates_only_the_completion(backend: OnnxBackend) -> None:
    prompt = "Reply with a JSON object holding one key, name."
    text = await backend.agenerate(prompt)
    assert isinstance(text, str)
    assert text.strip()
    # The runtime returns prompt and completion together; the adapter must keep only the latter.
    assert not text.lstrip().startswith(prompt[:20])

    # Reuse the backend's tokenizer: a second Model would load the weights into memory twice.
    tokenizer = backend._tokenizer
    prompt_len = len(tokenizer.encode(prompt))
    # Re-encoding decoded text can differ from the generated ids by a token or two.
    assert len(tokenizer.encode(text)) <= _MAX_LENGTH - prompt_len + 2


async def test_rejects_grammar_constraint(backend: OnnxBackend) -> None:
    assert backend.supports_grammar is False
    with pytest.raises(BackendError):
        await backend.agenerate("hi", constraint=Constraint("gbnf", "root ::= x"))
