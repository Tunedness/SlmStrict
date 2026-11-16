"""Google Gemini provider, built on the ``google-genai`` SDK (extra ``slmstrict[gemini]``).

The schema goes out through Gemini's structured output
(``response_mime_type="application/json"`` plus ``response_json_schema``). The router validates
the answer against the schema afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

from slmstrict.fallback.base import (
    DEFAULT_MODELS,
    ProviderReply,
    int_or_none,
    load_sdk,
)
from slmstrict.hooks import TokenUsage
from slmstrict.schema import Schema

__all__ = ["GeminiProvider"]


class GeminiProvider:
    """:class:`~slmstrict.fallback.base.CloudProvider` for Gemini models.

    ``model`` defaults to ``DEFAULT_MODELS["gemini"]``. ``api_key`` defaults to the SDK's
    ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` environment variables. ``client_factory`` builds the
    ``google.genai.Client`` for one call (Vertex setups, custom HTTP settings, tests) and replaces
    ``api_key``.
    """

    name = "gemini"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._genai: ModuleType = load_sdk("google.genai", "gemini")
        self._types: ModuleType = load_sdk("google.genai.types", "gemini")
        self.model = model if model is not None else DEFAULT_MODELS["gemini"]
        self._api_key = api_key
        self._client_factory = client_factory

    def _new_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        if self._api_key is not None:
            return self._genai.Client(api_key=self._api_key)
        return self._genai.Client()

    async def complete(self, prompt: str, schema: Schema[Any]) -> ProviderReply:
        """Send ``prompt`` and return Gemini's JSON answer."""
        config = self._types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema.json_schema,
        )
        client = self._new_client()
        aio = client.aio
        try:
            response = await aio.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        finally:
            # `genai.Client` opens a sync and an async HTTP client; `aclose()` closes only the
            # async one.
            try:
                await aio.aclose()
            finally:
                client.close()
        text = response.text
        return ProviderReply(
            text=text if isinstance(text, str) else "",
            usage=_usage(getattr(response, "usage_metadata", None)),
            finish_reason=_finish_reason(response),
        )

    def __repr__(self) -> str:
        return f"GeminiProvider(model={self.model!r})"


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return "refusal" if getattr(response, "prompt_feedback", None) is not None else None
    reason = getattr(candidates[0], "finish_reason", None)
    name = getattr(reason, "name", reason)
    if not isinstance(name, str):
        return None
    if name == "MAX_TOKENS":
        return "length"
    if name in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}:
        return "refusal"
    return name.lower()


def _usage(raw: object) -> TokenUsage | None:
    """Gemini's prompt count includes cached content; thinking tokens are billed as output."""
    if raw is None:
        return None
    prompt_tokens = int_or_none(getattr(raw, "prompt_token_count", None))
    if prompt_tokens is None:
        return None
    candidates = int_or_none(getattr(raw, "candidates_token_count", None)) or 0
    thoughts = int_or_none(getattr(raw, "thoughts_token_count", None)) or 0
    cached = int_or_none(getattr(raw, "cached_content_token_count", None))
    if cached is not None and cached > prompt_tokens:
        cached = None
    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=candidates + thoughts,
        cached_input_tokens=cached,
    )
