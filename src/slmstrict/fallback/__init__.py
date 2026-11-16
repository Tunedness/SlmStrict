"""Cloud fallback: the last rung of the enforcement ladder (ADR-003, ADR-007).

Off by default. It runs only when a handler is configured *and* ``local_only`` is off, masks the
prompt through an optional :class:`Masker` before anything leaves the device and logs meta data
only unless content logging is switched on explicitly.

Wire it in like this::

    from slmstrict import FallbackConfig, StrictConfig
    from slmstrict.fallback import AnthropicProvider, CloudFallback, RegexMasker

    handler = CloudFallback(AnthropicProvider(), masker=RegexMasker())
    config = StrictConfig(fallback=FallbackConfig(handler=handler))

The provider SDKs are optional extras and are imported lazily when a provider is constructed;
importing this package loads none of them.
"""

from slmstrict.fallback.anthropic import AnthropicProvider
from slmstrict.fallback.base import (
    DEFAULT_MODELS,
    CloudProvider,
    FallbackConfigError,
    FallbackDisabledError,
    FallbackError,
    FallbackValidationFailed,
    ProviderReply,
)
from slmstrict.fallback.gemini import GeminiProvider
from slmstrict.fallback.masking import Masker, RegexMasker
from slmstrict.fallback.openai import OpenAIProvider
from slmstrict.fallback.router import CloudFallback

__all__ = [
    "DEFAULT_MODELS",
    "AnthropicProvider",
    "CloudFallback",
    "CloudProvider",
    "FallbackConfigError",
    "FallbackDisabledError",
    "FallbackError",
    "FallbackValidationFailed",
    "GeminiProvider",
    "Masker",
    "OpenAIProvider",
    "ProviderReply",
    "RegexMasker",
]
