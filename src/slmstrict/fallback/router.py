"""The cloud fallback router: rung 4 of the enforcement ladder (ADR-002, ADR-003, ADR-007).

:class:`CloudFallback` implements :class:`slmstrict.hooks.FallbackHandler`. Everything that leaves
the device goes through :meth:`CloudFallback.handle`, in this fixed order:

1. ``local_only`` check. The ladder never calls a handler while ``local_only`` is on; the router
   checks its own ``local_only`` flag as well, so a handler used outside the ladder, or a ladder
   bug, still cannot send anything (defence in depth).
2. Masking hook (:class:`~slmstrict.fallback.masking.Masker`), when one is configured.
3. Provider call.
4. Validation of the answer against the same schema.

What is sent: the original prompt (masked). The local attempts are *not* sent. Their raw outputs
are the model's failed guesses, they may echo the input, and the provider does not need them.

What is logged (logger ``slmstrict.fallback``): meta data only, namely provider, model, schema
hash, number of characters sent, duration and outcome. Content is logged only with
``log_content=True`` and then only the *masked* prompt and a masked copy of the answer.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TypeVar

from slmstrict.fallback.base import (
    CloudProvider,
    FallbackDisabledError,
    FallbackError,
    FallbackValidationFailed,
    ProviderReply,
    provider_failure,
)
from slmstrict.fallback.masking import Masker
from slmstrict.hooks import FallbackResult
from slmstrict.result import Attempt, Result, Step
from slmstrict.schema import Schema

__all__ = ["CloudFallback"]

T = TypeVar("T")

_logger = logging.getLogger("slmstrict.fallback")


class CloudFallback:
    """Cloud fallback handler: masks, sends through a provider and validates the answer.

    ``provider`` is the :class:`~slmstrict.fallback.base.CloudProvider` to call. ``masker`` is the
    masking hook; ``None`` (the default) sends the prompt unmasked. ``local_only=True`` makes the
    router refuse to send anything (:class:`~slmstrict.fallback.base.FallbackDisabledError`).
    ``log_content=True`` additionally logs the masked prompt and the answer (see module docs).
    """

    def __init__(
        self,
        provider: CloudProvider,
        *,
        masker: Masker | None = None,
        local_only: bool = False,
        log_content: bool = False,
    ) -> None:
        if not isinstance(provider, CloudProvider):
            raise TypeError("provider must implement CloudProvider")
        if masker is not None and not isinstance(masker, Masker):
            raise TypeError("masker must implement Masker (a mask(text) -> str method)")
        if not isinstance(local_only, bool) or not isinstance(log_content, bool):
            raise TypeError("local_only and log_content must be bool")
        self._provider = provider
        self._masker = masker
        self._local_only = local_only
        self._log_content = log_content

    async def handle(
        self,
        schema: Schema[T],
        prompt: str,
        attempts: Sequence[Attempt],
    ) -> Result[T]:
        """Ask the cloud model for a value that matches ``schema``.

        ``attempts`` (the local attempts) are deliberately not sent anywhere. Raises
        :class:`FallbackDisabledError` under ``local_only``, :class:`FallbackError` when masking or
        the provider fails and :class:`FallbackValidationFailed` when the answer is not valid.
        """
        provider = self._provider.name
        model = self._provider.model
        schema_hash = schema.schema_hash

        # 1. local_only: the second lock, before anything is touched.
        if self._local_only:
            _logger.warning(
                "fallback blocked provider=%s model=%s schema=%s result=disabled_local_only",
                provider,
                model,
                schema_hash,
            )
            raise FallbackDisabledError("the cloud fallback is disabled by local_only=True")

        # 2. Masking. Fail closed: a masker that breaks means nothing is sent.
        outgoing = self._mask(prompt, provider, model, schema_hash)

        _logger.info(
            "fallback request provider=%s model=%s schema=%s chars=%d masked=%s",
            provider,
            model,
            schema_hash,
            len(outgoing),
            self._masker is not None,
        )
        if self._log_content:
            _logger.info("fallback prompt (masked) provider=%s: %s", provider, outgoing)

        # 3. Provider call.
        started = time.perf_counter()
        failure: FallbackError | None = None
        try:
            reply = await self._provider.complete(outgoing, schema)
        except FallbackError:
            self._log_done(provider, model, schema_hash, started, "error")
            raise
        except Exception as exc:
            self._log_done(provider, model, schema_hash, started, "error", type(exc).__name__)
            failure = provider_failure(provider, exc)
        if failure is not None:
            # Raised outside the `except` block on purpose: inside it, the SDK exception (and its
            # message, which may echo the API key) would stay reachable as `__context__`.
            raise failure
        if not isinstance(reply, ProviderReply):
            self._log_done(provider, model, schema_hash, started, "error", "InvalidReply")
            raise FallbackError(f"the {provider} provider returned an unusable reply")

        # 4. Validation against the same schema.
        report = schema.validate(reply.text)
        if self._log_content:
            _logger.info(
                "fallback answer (masked) provider=%s: %s", provider, self._mask_for_log(reply.text)
            )
        if not report.ok or report.value is None:
            outcome = "refused" if reply.finish_reason == "refusal" else "invalid"
            self._log_done(provider, model, schema_hash, started, outcome, reply.finish_reason)
            raise FallbackValidationFailed(
                "the cloud fallback returned output that does not match the schema",
                issues=report.issues,
                last_output=reply.text,
                attempts=(Attempt(Step.FALLBACK, reply.text, errors=report.issues),),
                usage=reply.usage,
                provider=provider,
                model=model,
            )

        self._log_done(provider, model, schema_hash, started, "ok", reply.finish_reason)
        return FallbackResult(
            value=report.value,
            attempts=(Attempt(Step.FALLBACK, reply.text),),
            usage=reply.usage,
            provider=provider,
            model=model,
        )

    # -- helpers --------------------------------------------------------------------------

    def _mask(self, prompt: str, provider: str, model: str, schema_hash: str) -> str:
        if self._masker is None:
            return prompt
        masked: object = None
        failure: str | None = None
        try:
            masked = self._masker.mask(prompt)
        except Exception as exc:
            failure = type(exc).__name__
        if failure is not None:
            _logger.warning(
                "fallback blocked provider=%s model=%s schema=%s result=masker_failed error=%s",
                provider,
                model,
                schema_hash,
                failure,
            )
            # Raised outside the `except` block: the masker's exception may quote the prompt.
            raise FallbackError(f"the masker failed ({failure}); nothing was sent to {provider}")
        if not isinstance(masked, str):
            _logger.warning(
                "fallback blocked provider=%s model=%s schema=%s result=masker_failed error=%s",
                provider,
                model,
                schema_hash,
                "NotAString",
            )
            raise FallbackError(f"the masker returned a non-string; nothing was sent to {provider}")
        return masked

    def _mask_for_log(self, text: str) -> str:
        """The answer as it may appear in a log: through the masker when there is one."""
        if self._masker is None:
            return text
        try:
            masked = self._masker.mask(text)
        except Exception:
            return "<answer withheld: masker failed>"
        return masked if isinstance(masked, str) else "<answer withheld: masker failed>"

    @staticmethod
    def _log_done(
        provider: str,
        model: str,
        schema_hash: str,
        started: float,
        result: str,
        detail: str | None = None,
    ) -> None:
        _logger.info(
            "fallback done provider=%s model=%s schema=%s duration_s=%.3f result=%s detail=%s",
            provider,
            model,
            schema_hash,
            time.perf_counter() - started,
            result,
            detail,
        )

    def __repr__(self) -> str:
        return (
            f"CloudFallback(provider={self._provider.name!r}, model={self._provider.model!r}, "
            f"masker={self._masker is not None}, local_only={self._local_only}, "
            f"log_content={self._log_content})"
        )
