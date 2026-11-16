"""Provider adapters, exercised against the real SDKs with a mocked HTTP transport.

The SDKs speak ``httpx2``, which ``respx`` does not patch, so each test hands the provider a client
whose transport is an ``httpx2.MockTransport``: the SDK builds and sends its real request, the
transport records it and answers with canned JSON. No socket is ever opened. Tests of a provider
are skipped when its extra is not installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest

from slmstrict import Schema, TokenUsage
from slmstrict.fallback import (
    DEFAULT_MODELS,
    AnthropicProvider,
    CloudFallback,
    CloudProvider,
    FallbackError,
    FallbackValidationFailed,
    GeminiProvider,
    OpenAIProvider,
)
from slmstrict.fallback.base import load_sdk
from tests._ladder_support import ITEM_JSON_SCHEMA, Item
from tests.fallback import SECRET_KEY

ANSWER = '{"name": "cloud", "count": 9}'
PROMPT = "Extract the item: three apples"

Handler = Callable[[Any], Any]


class Recorder:
    """Transport handler: records requests and answers with a canned status and body."""

    def __init__(self, httpx2: Any, body: dict[str, Any], status: int = 200) -> None:
        self._httpx2 = httpx2
        self._body = body
        self._status = status
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        return self._httpx2.Response(self._status, json=self._body)

    @property
    def payload(self) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(self.requests[-1].content)
        return loaded

    def client(self) -> Any:
        return self._httpx2.AsyncClient(transport=self._httpx2.MockTransport(self))


# -- Anthropic -----------------------------------------------------------------------------


def _anthropic_body(
    text: str = ANSWER, stop: str = "end_turn", usage: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 20, "output_tokens": 7},
    }


@pytest.fixture
def anthropic_env() -> tuple[Any, Any]:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    return anthropic, httpx2


def _anthropic_provider(anthropic: Any, recorder: Recorder, **kwargs: Any) -> AnthropicProvider:
    def factory() -> Any:
        return anthropic.AsyncAnthropic(
            api_key=SECRET_KEY, http_client=recorder.client(), max_retries=0
        )

    return AnthropicProvider(client_factory=factory, **kwargs)


async def test_anthropic_sends_the_schema_as_structured_output(
    anthropic_env: tuple[Any, Any],
) -> None:
    anthropic, httpx2 = anthropic_env
    recorder = Recorder(httpx2, _anthropic_body())
    provider = _anthropic_provider(anthropic, recorder)
    reply = await provider.complete(PROMPT, Schema.of(Item))

    payload = recorder.payload
    assert recorder.requests[0].url.path == "/v1/messages"
    assert payload["model"] == DEFAULT_MODELS["anthropic"]
    assert payload["messages"] == [{"role": "user", "content": PROMPT}]
    fmt = payload["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["type"] == "object"
    assert fmt["schema"]["additionalProperties"] is False
    assert set(fmt["schema"]["properties"]) == {"name", "count"}
    assert recorder.requests[0].headers["authorization"] == f"Bearer {SECRET_KEY}"
    assert reply.text == ANSWER
    assert reply.usage == TokenUsage(input_tokens=20, output_tokens=7, cached_input_tokens=None)


async def test_anthropic_usage_counts_cache_reads_and_writes_as_input(
    anthropic_env: tuple[Any, Any],
) -> None:
    anthropic, httpx2 = anthropic_env
    usage = {
        "input_tokens": 10,
        "cache_read_input_tokens": 300,
        "cache_creation_input_tokens": 40,
        "output_tokens": 5,
    }
    recorder = Recorder(httpx2, _anthropic_body(usage=usage))
    reply = await _anthropic_provider(anthropic, recorder).complete(PROMPT, Schema.of(Item))
    assert reply.usage == TokenUsage(input_tokens=350, output_tokens=5, cached_input_tokens=300)


async def test_anthropic_model_and_max_tokens_can_be_overridden(
    anthropic_env: tuple[Any, Any],
) -> None:
    anthropic, httpx2 = anthropic_env
    recorder = Recorder(httpx2, _anthropic_body())
    provider = _anthropic_provider(anthropic, recorder, model="claude-haiku-5-5", max_tokens=512)
    await provider.complete(PROMPT, Schema.of(ITEM_JSON_SCHEMA))
    assert recorder.payload["model"] == "claude-haiku-5-5"
    assert recorder.payload["max_tokens"] == 512
    assert provider.model == "claude-haiku-5-5"


async def test_anthropic_reports_refusal_and_truncation(anthropic_env: tuple[Any, Any]) -> None:
    anthropic, httpx2 = anthropic_env
    refusal = Recorder(
        httpx2, {**_anthropic_body(text=""), "stop_reason": "refusal", "content": []}
    )
    reply = await _anthropic_provider(anthropic, refusal).complete(PROMPT, Schema.of(Item))
    assert (reply.text, reply.finish_reason) == ("", "refusal")

    cut = Recorder(httpx2, _anthropic_body(text='{"name": "cl', stop="max_tokens"))
    reply = await _anthropic_provider(anthropic, cut).complete(PROMPT, Schema.of(Item))
    assert reply.finish_reason == "length"


async def test_anthropic_through_the_router_validates_the_answer(
    anthropic_env: tuple[Any, Any],
) -> None:
    anthropic, httpx2 = anthropic_env
    good = _anthropic_provider(anthropic, Recorder(httpx2, _anthropic_body()))
    result = await CloudFallback(good).handle(Schema.of(Item), PROMPT, ())
    assert result.value == Item(name="cloud", count=9)

    bad = _anthropic_provider(anthropic, Recorder(httpx2, _anthropic_body(text='{"name": 5}')))
    with pytest.raises(FallbackValidationFailed) as caught:
        await CloudFallback(bad).handle(Schema.of(Item), PROMPT, ())
    assert caught.value.usage == TokenUsage(input_tokens=20, output_tokens=7)
    assert caught.value.provider == "anthropic"


async def test_anthropic_auth_error_does_not_leak_the_key(
    anthropic_env: tuple[Any, Any],
) -> None:
    anthropic, httpx2 = anthropic_env
    body = {
        "type": "error",
        "error": {"type": "authentication_error", "message": f"invalid key {SECRET_KEY}"},
    }
    provider = _anthropic_provider(anthropic, Recorder(httpx2, body, status=401))

    # The SDK's own error does carry the text; the router must not pass it on.
    with pytest.raises(anthropic.AuthenticationError) as sdk_error:
        await provider.complete(PROMPT, Schema.of(Item))
    assert SECRET_KEY in str(sdk_error.value)

    with pytest.raises(FallbackError) as caught:
        await CloudFallback(provider).handle(Schema.of(Item), PROMPT, ())
    assert SECRET_KEY not in str(caught.value)
    assert "AuthenticationError" in str(caught.value)
    assert "401" in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_anthropic_repr_has_no_key() -> None:
    pytest.importorskip("anthropic")
    provider = AnthropicProvider(api_key=SECRET_KEY)
    assert SECRET_KEY not in repr(provider)
    assert SECRET_KEY not in str(vars(provider).get("_sdk", ""))
    assert repr(provider) == f"AnthropicProvider(model={DEFAULT_MODELS['anthropic']!r})"


def test_anthropic_rejects_a_bad_max_tokens() -> None:
    pytest.importorskip("anthropic")
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicProvider(max_tokens=0)


async def test_anthropic_builds_its_own_client_from_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    client = AnthropicProvider(api_key=SECRET_KEY)._new_client()
    try:
        assert client.api_key == SECRET_KEY
    finally:
        await client.close()


# -- OpenAI --------------------------------------------------------------------------------


def _openai_body(
    text: str = ANSWER, *, status: str = "completed", content: list[Any] | None = None
) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-test",
        "status": status,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": content
                if content is not None
                else [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 50,
            "input_tokens_details": {"cached_tokens": 32, "cache_write_tokens": 0},
            "output_tokens": 11,
            "output_tokens_details": {"reasoning_tokens": 4},
            "total_tokens": 61,
        },
    }


@pytest.fixture
def openai_env() -> tuple[Any, Any]:
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    return openai, httpx2


def _openai_provider(openai: Any, recorder: Recorder, **kwargs: Any) -> OpenAIProvider:
    def factory() -> Any:
        return openai.AsyncOpenAI(api_key=SECRET_KEY, http_client=recorder.client(), max_retries=0)

    return OpenAIProvider(client_factory=factory, **kwargs)


async def test_openai_sends_the_schema_as_structured_output(
    openai_env: tuple[Any, Any],
) -> None:
    openai, httpx2 = openai_env
    recorder = Recorder(httpx2, _openai_body())
    schema = Schema.of(Item)
    reply = await _openai_provider(openai, recorder).complete(PROMPT, schema)

    payload = recorder.payload
    assert recorder.requests[0].url.path == "/v1/responses"
    assert payload["model"] == DEFAULT_MODELS["openai"]
    assert payload["input"] == PROMPT
    assert payload["store"] is False
    fmt = payload["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["name"] == f"slmstrict_{schema.schema_hash[:16]}"
    assert fmt["schema"] == schema.json_schema
    assert fmt["strict"] is False
    assert recorder.requests[0].headers["authorization"] == f"Bearer {SECRET_KEY}"
    assert reply.text == ANSWER
    assert reply.usage == TokenUsage(input_tokens=50, output_tokens=11, cached_input_tokens=32)


async def test_openai_strict_mode_is_opt_in(openai_env: tuple[Any, Any]) -> None:
    openai, httpx2 = openai_env
    recorder = Recorder(httpx2, _openai_body())
    await _openai_provider(openai, recorder, strict=True, model="gpt-x").complete(
        PROMPT, Schema.of(ITEM_JSON_SCHEMA)
    )
    assert recorder.payload["text"]["format"]["strict"] is True
    assert recorder.payload["model"] == "gpt-x"


async def test_openai_reports_refusal_and_truncation(openai_env: tuple[Any, Any]) -> None:
    openai, httpx2 = openai_env
    refusal = Recorder(
        httpx2, _openai_body(content=[{"type": "refusal", "refusal": "I can't help."}])
    )
    reply = await _openai_provider(openai, refusal).complete(PROMPT, Schema.of(Item))
    assert (reply.text, reply.finish_reason) == ("", "refusal")

    cut_body = _openai_body(text='{"name": "cl', status="incomplete")
    cut_body["incomplete_details"] = {"reason": "max_output_tokens"}
    reply = await _openai_provider(openai, Recorder(httpx2, cut_body)).complete(
        PROMPT, Schema.of(Item)
    )
    assert reply.finish_reason == "length"


async def test_openai_error_does_not_leak_the_key(openai_env: tuple[Any, Any]) -> None:
    openai, httpx2 = openai_env
    body = {"error": {"message": f"Incorrect API key provided: {SECRET_KEY}", "type": "x"}}
    provider = _openai_provider(openai, Recorder(httpx2, body, status=401))
    with pytest.raises(FallbackError) as caught:
        await CloudFallback(provider).handle(Schema.of(Item), PROMPT, ())
    assert SECRET_KEY not in str(caught.value)
    assert "401" in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


async def test_openai_through_the_router_rejects_invalid_output(
    openai_env: tuple[Any, Any],
) -> None:
    openai, httpx2 = openai_env
    provider = _openai_provider(openai, Recorder(httpx2, _openai_body(text="not json")))
    with pytest.raises(FallbackValidationFailed) as caught:
        await CloudFallback(provider).handle(Schema.of(Item), PROMPT, ())
    assert caught.value.usage == TokenUsage(
        input_tokens=50, output_tokens=11, cached_input_tokens=32
    )


def test_openai_repr_has_no_key() -> None:
    pytest.importorskip("openai")
    provider = OpenAIProvider(api_key=SECRET_KEY)
    assert SECRET_KEY not in repr(provider)


# -- Gemini --------------------------------------------------------------------------------


def _gemini_body(text: str = ANSWER, finish: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}
        ],
        "usageMetadata": {
            "promptTokenCount": 40,
            "cachedContentTokenCount": 16,
            "candidatesTokenCount": 9,
            "thoughtsTokenCount": 6,
            "totalTokenCount": 55,
        },
    }


@pytest.fixture
def gemini_env() -> tuple[Any, Any]:
    genai = pytest.importorskip("google.genai")
    httpx2 = pytest.importorskip("httpx2")
    return genai, httpx2


def _gemini_provider(genai: Any, recorder: Recorder, **kwargs: Any) -> GeminiProvider:
    def factory() -> Any:
        options = genai.types.HttpOptions(httpx_async_client=recorder.client())
        return genai.Client(api_key=SECRET_KEY, http_options=options)

    return GeminiProvider(client_factory=factory, **kwargs)


async def test_gemini_sends_the_schema_as_structured_output(
    gemini_env: tuple[Any, Any],
) -> None:
    genai, httpx2 = gemini_env
    recorder = Recorder(httpx2, _gemini_body())
    schema = Schema.of(Item)
    reply = await _gemini_provider(genai, recorder).complete(PROMPT, schema)

    payload = recorder.payload
    assert f"/models/{DEFAULT_MODELS['gemini']}:generateContent" in recorder.requests[0].url.path
    assert payload["contents"][0]["parts"] == [{"text": PROMPT}]
    config = payload["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == schema.json_schema
    assert SECRET_KEY in {
        recorder.requests[0].headers.get("x-goog-api-key"),
        recorder.requests[0].url.params.get("key"),
    }
    assert reply.text == ANSWER
    assert reply.usage == TokenUsage(input_tokens=40, output_tokens=15, cached_input_tokens=16)


async def test_gemini_reports_truncation_and_blocking(gemini_env: tuple[Any, Any]) -> None:
    genai, httpx2 = gemini_env
    cut = Recorder(httpx2, _gemini_body(text='{"name": "cl', finish="MAX_TOKENS"))
    reply = await _gemini_provider(genai, cut).complete(PROMPT, Schema.of(Item))
    assert reply.finish_reason == "length"

    blocked = Recorder(httpx2, _gemini_body(text="", finish="SAFETY"))
    reply = await _gemini_provider(genai, blocked).complete(PROMPT, Schema.of(Item))
    assert reply.finish_reason == "refusal"


async def test_gemini_prompt_block_is_a_refusal(gemini_env: tuple[Any, Any]) -> None:
    genai, httpx2 = gemini_env
    body = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 3}}
    reply = await _gemini_provider(genai, Recorder(httpx2, body)).complete(PROMPT, Schema.of(Item))
    assert (reply.text, reply.finish_reason) == ("", "refusal")


async def test_gemini_closes_both_of_its_clients(gemini_env: tuple[Any, Any]) -> None:
    genai, httpx2 = gemini_env
    recorder = Recorder(httpx2, _gemini_body())
    built: list[Any] = []

    def factory() -> Any:
        options = genai.types.HttpOptions(httpx_async_client=recorder.client())
        client = genai.Client(api_key=SECRET_KEY, http_options=options)
        calls: list[str] = []
        original_close = client.close

        def close() -> None:
            calls.append("sync")
            original_close()

        client.close = close
        client.closed_calls = calls
        built.append(client)
        return client

    await GeminiProvider(client_factory=factory).complete(PROMPT, Schema.of(Item))
    assert built[0].closed_calls == ["sync"]


async def test_gemini_error_does_not_leak_the_key(gemini_env: tuple[Any, Any]) -> None:
    genai, httpx2 = gemini_env
    body = {"error": {"code": 400, "message": f"API key not valid: {SECRET_KEY}", "status": "X"}}
    provider = _gemini_provider(genai, Recorder(httpx2, body, status=400))
    with pytest.raises(FallbackError) as caught:
        await CloudFallback(provider).handle(Schema.of(Item), PROMPT, ())
    assert SECRET_KEY not in str(caught.value)
    assert "400" in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


async def test_gemini_through_the_router_rejects_invalid_output(
    gemini_env: tuple[Any, Any],
) -> None:
    genai, httpx2 = gemini_env
    provider = _gemini_provider(genai, Recorder(httpx2, _gemini_body(text='{"count": "x"}')))
    with pytest.raises(FallbackValidationFailed) as caught:
        await CloudFallback(provider).handle(Schema.of(Item), PROMPT, ())
    assert caught.value.provider == "gemini"
    assert caught.value.usage == TokenUsage(
        input_tokens=40, output_tokens=15, cached_input_tokens=16
    )


def test_gemini_repr_has_no_key() -> None:
    pytest.importorskip("google.genai")
    provider = GeminiProvider(api_key=SECRET_KEY)
    assert SECRET_KEY not in repr(provider)


# -- shared contract -----------------------------------------------------------------------


def test_every_provider_implements_the_provider_protocol() -> None:
    for module, provider in (
        ("anthropic", lambda: AnthropicProvider()),
        ("openai", lambda: OpenAIProvider()),
        ("google.genai", lambda: GeminiProvider()),
    ):
        pytest.importorskip(module)
        assert isinstance(provider(), CloudProvider)


def test_default_models_table_is_read_only_and_complete() -> None:
    assert set(DEFAULT_MODELS) == {"anthropic", "openai", "gemini"}
    with pytest.raises(TypeError):
        DEFAULT_MODELS["anthropic"] = "other"  # type: ignore[index]


def test_importing_the_package_loads_no_sdk() -> None:
    code = (
        "import sys\n"
        "import slmstrict, slmstrict.fallback\n"
        "loaded = [m for m in ('anthropic', 'openai', 'google.genai') if m in sys.modules]\n"
        "print(loaded)\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_a_missing_extra_gives_an_installation_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    real = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> Any:
        if name in {"anthropic", "openai", "google.genai"}:
            raise ImportError(f"No module named {name!r}")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match=r"pip install 'slmstrict\[anthropic\]'"):
        AnthropicProvider()
    with pytest.raises(ImportError, match=r"pip install 'slmstrict\[openai\]'"):
        OpenAIProvider()
    with pytest.raises(ImportError, match=r"pip install 'slmstrict\[gemini\]'"):
        GeminiProvider()
    with pytest.raises(ImportError, match="slmstrict"):
        load_sdk("definitely_missing_module_xyz", "extra")
