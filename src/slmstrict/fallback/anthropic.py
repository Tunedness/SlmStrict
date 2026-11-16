"""Anthropic (Claude) provider, built on the ``anthropic`` SDK (extra ``slmstrict[anthropic]``).

The schema goes out through the Messages API's structured outputs (``output_config.format`` of
type ``json_schema``). The SDK's own ``transform_schema`` helper adapts the schema to the subset
the API accepts (``additionalProperties: false``, unsupported constraints moved into the field
descriptions); the answer is validated against the *original* schema afterwards by the router.
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

__all__ = ["AnthropicProvider"]

_DEFAULT_MAX_TOKENS = 16000


class AnthropicProvider:
    """:class:`~slmstrict.fallback.base.CloudProvider` for Claude models.

    ``model`` defaults to ``DEFAULT_MODELS["anthropic"]``. ``api_key`` defaults to the SDK's
    ``ANTHROPIC_API_KEY`` environment variable (and its other standard credential sources).
    ``client_factory`` builds the ``anthropic.AsyncAnthropic`` client for one call; it exists for
    proxies, custom HTTP settings and tests, and replaces ``api_key``.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive int")
        self._sdk: ModuleType = load_sdk("anthropic", "anthropic")
        self.model = model if model is not None else DEFAULT_MODELS["anthropic"]
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client_factory = client_factory

    def _new_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        if self._api_key is not None:
            return self._sdk.AsyncAnthropic(api_key=self._api_key)
        return self._sdk.AsyncAnthropic()

    async def complete(self, prompt: str, schema: Schema[Any]) -> ProviderReply:
        """Send ``prompt`` and return Claude's JSON answer."""
        try:
            wire_schema = self._sdk.transform_schema(schema.json_schema)
        except ValueError:
            # A schema the helper cannot adapt (a property without a type, ...): send it as is.
            wire_schema = schema.json_schema
        client = self._new_client()
        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": wire_schema}},
            )
        finally:
            await client.close()
        text = "".join(
            str(block.text) for block in response.content if getattr(block, "type", "") == "text"
        )
        stop_reason = getattr(response, "stop_reason", None)
        finish = stop_reason if isinstance(stop_reason, str) else None
        if finish == "max_tokens":
            finish = "length"
        return ProviderReply(
            text=text,
            usage=_usage(getattr(response, "usage", None)),
            finish_reason=finish,
        )

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self.model!r})"


def _usage(raw: object) -> TokenUsage | None:
    """Anthropic reports uncached input apart from cache reads and writes; the sum is the input."""
    if raw is None:
        return None
    uncached = int_or_none(getattr(raw, "input_tokens", None))
    output = int_or_none(getattr(raw, "output_tokens", None))
    if uncached is None or output is None:
        return None
    cache_read = int_or_none(getattr(raw, "cache_read_input_tokens", None))
    cache_write = int_or_none(getattr(raw, "cache_creation_input_tokens", None))
    return TokenUsage(
        input_tokens=uncached + (cache_read or 0) + (cache_write or 0),
        output_tokens=output,
        cached_input_tokens=cache_read,
    )
