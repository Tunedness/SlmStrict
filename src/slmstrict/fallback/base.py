"""Provider contract and errors of the cloud fallback (ADR-003, ADR-007).

A *provider* is the thin adapter to one cloud SDK: it takes the (already masked) prompt and the
schema, asks the model for schema-shaped JSON through the provider's structured output feature and
returns the raw text plus token usage. Everything that concerns privacy (the ``local_only`` check,
masking, logging) and validation lives in :class:`slmstrict.fallback.router.CloudFallback`, so a
provider stays free of policy.

Rules every provider follows:

* The SDK is imported lazily, when the provider is constructed, never when this package is
  imported. A missing extra raises :class:`ImportError` with an installation hint.
* The API key comes from the provider's ``api_key`` argument or from the SDK's standard
  environment variable. It is never logged, never part of ``repr()`` and never part of an error
  message.
* SDK clients are created per call: ``generate()`` runs every call on a fresh event loop, so a
  client must not outlive one.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Any, Final, Protocol, runtime_checkable

from slmstrict.errors import SlmStrictError, ValidationFailed
from slmstrict.hooks import TokenUsage
from slmstrict.result import Attempt
from slmstrict.schema import Schema, ValidationIssue

__all__ = [
    "DEFAULT_MODELS",
    "CloudProvider",
    "FallbackDisabledError",
    "FallbackError",
    "FallbackValidationFailed",
    "ProviderReply",
    "int_or_none",
    "load_sdk",
    "provider_failure",
]

DEFAULT_MODELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "anthropic": "claude-opus-5-5",
        "openai": "gpt-5.5",
        "gemini": "gemini-3.8-flash",
    }
)
"""Default model id per provider name. Read-only: pass ``model=`` to a provider to override."""


class FallbackError(SlmStrictError):
    """The cloud fallback could not produce an answer (provider failure, masker failure, ...).

    The message never contains the prompt, the model output, an API key or the text of the SDK's
    own exception: only the provider name, the exception class and an HTTP status when there is
    one.
    """


class FallbackDisabledError(FallbackError):
    """The fallback was asked to send data while ``local_only`` is on. Nothing was sent."""


class FallbackValidationFailed(ValidationFailed):
    """The provider answered, but the answer does not match the schema.

    Besides everything :class:`~slmstrict.errors.ValidationFailed` carries, it reports what the
    call cost (``usage``) and who answered (``provider``, ``model``): the request was made and
    paid for even though the value is unusable.

    ``last_output`` and ``attempts`` hold the provider's raw, unmasked answer (the
    :class:`~slmstrict.errors.ValidationFailed` contract); they are not part of ``str(error)``.
    Callers should not log them.
    """

    def __init__(
        self,
        message: str,
        *,
        issues: tuple[ValidationIssue, ...] = (),
        last_output: str | None = None,
        attempts: tuple[Attempt, ...] = (),
        usage: TokenUsage | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message, issues=issues, last_output=last_output, attempts=attempts)
        self.usage = usage
        self.provider = provider
        self.model = model


@dataclass(frozen=True, slots=True)
class ProviderReply:
    """What a provider got back for one request.

    ``text`` is the model's raw answer (empty when it produced none), ``usage`` the token counts
    the provider reported (``None`` when it reported none) and ``finish_reason`` the provider's own
    stop reason (``"length"``, ``"refusal"``, ...), for the metadata log only.
    """

    text: str
    usage: TokenUsage | None = None
    finish_reason: str | None = None


@runtime_checkable
class CloudProvider(Protocol):
    """Adapter to one cloud model.

    ``name`` identifies the provider (``"anthropic"``, ...), ``model`` the model id it calls.
    :meth:`complete` sends ``prompt`` and asks for JSON that matches ``schema``; it does not
    validate the answer and does not log content.
    """

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def complete(self, prompt: str, schema: Schema[Any]) -> ProviderReply: ...


def load_sdk(module: str, extra: str) -> ModuleType:
    """Import a provider SDK, or raise :class:`ImportError` with an installation hint."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"the '{module}' package is required for this cloud provider. "
            f"Install it with: pip install 'slmstrict[{extra}]'"
        ) from exc


def int_or_none(value: object) -> int | None:
    """``value`` when it is a non-negative ``int`` (not a ``bool``), otherwise ``None``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def provider_failure(provider: str, exc: BaseException) -> FallbackError:
    """A :class:`FallbackError` for ``exc`` that is safe to log and to show.

    The SDK's message is dropped on purpose: provider errors may echo request data or (part of)
    the API key.
    """
    detail = type(exc).__name__
    for attr in ("status_code", "code"):
        status = int_or_none(getattr(exc, attr, None))
        if status is not None:
            detail += f", status {status}"
            break
    return FallbackError(f"the {provider} request failed ({detail})")
