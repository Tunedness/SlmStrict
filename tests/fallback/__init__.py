"""Tests of the cloud fallback (Faz 07). No test here opens a real network connection."""

from __future__ import annotations

from typing import Any

from slmstrict import TokenUsage
from slmstrict.fallback import ProviderReply
from slmstrict.schema import Schema

SECRET_KEY = "sk-test-SECRET-0123456789"
VALID_ANSWER = '{"name": "cloud", "count": 9}'


class FakeProvider:
    """A :class:`~slmstrict.fallback.CloudProvider` that records what it was asked and answers
    with a canned reply (or raises a canned error)."""

    def __init__(
        self,
        text: str = VALID_ANSWER,
        *,
        usage: TokenUsage | None = None,
        finish_reason: str | None = "stop",
        error: BaseException | None = None,
        name: str = "fake",
        model: str = "fake-model-1",
    ) -> None:
        self.name = name
        self.model = model
        self.calls = 0
        self.prompts: list[str] = []
        self._reply = ProviderReply(text=text, usage=usage, finish_reason=finish_reason)
        self._error = error

    async def complete(self, prompt: str, schema: Schema[Any]) -> ProviderReply:
        self.calls += 1
        self.prompts.append(prompt)
        if self._error is not None:
            raise self._error
        return self._reply
