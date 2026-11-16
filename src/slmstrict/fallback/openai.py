"""OpenAI provider, built on the ``openai`` SDK (extra ``slmstrict[openai]``).

The schema goes out through the Responses API's structured outputs
(``text.format`` of type ``json_schema``). ``strict`` defaults to ``False``: OpenAI's strict mode
only accepts a subset of JSON Schema (every property required, ``additionalProperties: false``)
that Pydantic-derived schemas usually do not meet. The router validates the answer against the
schema either way. Pass ``strict=True`` when the schema is written for strict mode.

Requests are sent with ``store=False`` so OpenAI does not keep the response for later retrieval.
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

__all__ = ["OpenAIProvider"]

_INSTRUCTIONS = "Answer with a single JSON document that matches the provided JSON schema."


class OpenAIProvider:
    """:class:`~slmstrict.fallback.base.CloudProvider` for OpenAI models.

    ``model`` defaults to ``DEFAULT_MODELS["openai"]``. ``api_key`` defaults to the SDK's
    ``OPENAI_API_KEY`` environment variable. ``client_factory`` builds the
    ``openai.AsyncOpenAI`` client for one call (proxies, custom HTTP settings, tests) and replaces
    ``api_key``.
    """

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        strict: bool = False,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._sdk: ModuleType = load_sdk("openai", "openai")
        self.model = model if model is not None else DEFAULT_MODELS["openai"]
        self._api_key = api_key
        self._strict = bool(strict)
        self._client_factory = client_factory

    def _new_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        if self._api_key is not None:
            return self._sdk.AsyncOpenAI(api_key=self._api_key)
        return self._sdk.AsyncOpenAI()

    async def complete(self, prompt: str, schema: Schema[Any]) -> ProviderReply:
        """Send ``prompt`` and return the model's JSON answer."""
        client = self._new_client()
        try:
            response = await client.responses.create(
                model=self.model,
                instructions=_INSTRUCTIONS,
                input=prompt,
                store=False,
                text={
                    "format": {
                        "type": "json_schema",
                        # The name is hash-based: it must match [A-Za-z0-9_-]{1,64} and a model
                        # title may carry business-specific words.
                        "name": f"slmstrict_{schema.schema_hash[:16]}",
                        "schema": schema.json_schema,
                        "strict": self._strict,
                    }
                },
            )
        finally:
            await client.close()
        return ProviderReply(
            text=_text_or_empty(response.output_text),
            usage=_usage(getattr(response, "usage", None)),
            finish_reason=_finish_reason(response),
        )

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model!r}, strict={self._strict})"


def _text_or_empty(value: object) -> str:
    return value if isinstance(value, str) else ""


def _finish_reason(response: Any) -> str | None:
    """``"refusal"`` when the model refused, ``"length"`` when it was cut off."""
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", "") == "refusal":
                return "refusal"
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    if reason == "max_output_tokens":
        return "length"
    status = getattr(response, "status", None)
    return status if isinstance(status, str) else None


def _usage(raw: object) -> TokenUsage | None:
    """OpenAI's input count already includes the cached part."""
    if raw is None:
        return None
    input_tokens = int_or_none(getattr(raw, "input_tokens", None))
    output_tokens = int_or_none(getattr(raw, "output_tokens", None))
    if input_tokens is None or output_tokens is None:
        return None
    cached = int_or_none(getattr(getattr(raw, "input_tokens_details", None), "cached_tokens", None))
    if cached is not None and cached > input_tokens:
        cached = None
    return TokenUsage(
        input_tokens=input_tokens, output_tokens=output_tokens, cached_input_tokens=cached
    )
