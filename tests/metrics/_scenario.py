"""A known sequence of ladder runs shared by the metrics tests."""

from __future__ import annotations

import contextlib

from slmstrict import (
    Attempt,
    FallbackConfig,
    FallbackResult,
    LadderObserver,
    Schema,
    Step,
    StrictConfig,
    TokenUsage,
    ValidationFailed,
)
from slmstrict.ladder import Ladder
from slmstrict.testing import FakeBackend
from tests._ladder_support import FENCED, GARBAGE, VALID, GrammarBackend, Item, SpyHandler

SCHEMA = Schema.of(Item)

CLOUD_USAGE = TokenUsage(input_tokens=1200, output_tokens=300, cached_input_tokens=1000)
CLOUD_PROVIDER = "acme"
CLOUD_MODEL = "acme-large"


class PaidValidationFailure(ValidationFailed):
    """A handler failure that reports what the rejected call cost (like the cloud fallback's)."""

    def __init__(self, usage: TokenUsage, provider: str, model: str) -> None:
        super().__init__("the provider answered with an unusable value")
        self.usage = usage
        self.provider = provider
        self.model = model


def cloud_handler(usage: TokenUsage | None = CLOUD_USAGE) -> SpyHandler:
    result = FallbackResult(
        value=Item(name="cloud", count=9),
        attempts=(Attempt(step=Step.FALLBACK, raw_output='{"name": "cloud", "count": 9}'),),
        usage=usage,
        provider=CLOUD_PROVIDER,
        model=CLOUD_MODEL,
    )
    return SpyHandler(result=result)


async def run_known_scenario(observer: LadderObserver, *, prompt: str = "make an item") -> None:
    """Six runs with these outcomes (default ``max_reask`` of 2):

    1. valid as it is                      -> path ()
    2. fenced JSON, repaired               -> path (REPAIR,)
    3. garbage, then a valid re-ask        -> path (REASK,)
    4. garbage x3, cloud fallback succeeds -> path (REASK, FALLBACK)
    5. garbage x3, no fallback configured  -> ValidationFailed
    6. grammar backend, valid              -> path (GRAMMAR,)
    """
    config = StrictConfig(observer=observer)
    await Ladder(FakeBackend([VALID]), config).run(SCHEMA, prompt)
    await Ladder(FakeBackend([FENCED]), config).run(SCHEMA, prompt)
    await Ladder(FakeBackend([GARBAGE, VALID]), config).run(SCHEMA, prompt)
    with_fallback = StrictConfig(
        observer=observer, fallback=FallbackConfig(handler=cloud_handler())
    )
    await Ladder(FakeBackend([GARBAGE] * 3), with_fallback).run(SCHEMA, prompt)
    with contextlib.suppress(ValidationFailed):
        await Ladder(FakeBackend([GARBAGE] * 3), config).run(SCHEMA, prompt)
    await Ladder(GrammarBackend([VALID]), config).run(SCHEMA, prompt)
