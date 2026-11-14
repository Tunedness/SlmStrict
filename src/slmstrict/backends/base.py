"""Backend adapter contract (ADR-004, ADR-008)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["Backend", "Constraint"]


@dataclass(frozen=True, slots=True)
class Constraint:
    """Opaque, backend-specific generation constraint (for example a compiled grammar).

    The core never looks inside: ``kind`` names the constraint family (``"gbnf"``, ...) and
    ``payload`` is whatever the producing adapter needs back in :meth:`Backend.agenerate`.
    """

    kind: str
    payload: object


@runtime_checkable
class Backend(Protocol):
    """Async-first interface every backend adapter implements.

    Adapters keep their blocking calls off the event loop (``asyncio.to_thread``). Generation
    settings such as ``max_tokens`` or ``temperature`` belong to the adapter's constructor, not
    to this call. ``isinstance``
    checks against this protocol only verify that the members exist, not their signatures; the
    signatures are checked by mypy.
    """

    @property
    def supports_grammar(self) -> bool:
        """Whether the backend can enforce a grammar :class:`Constraint` during decoding."""
        ...

    async def agenerate(
        self,
        prompt: str,
        *,
        constraint: Constraint | None = None,
    ) -> str:
        """Generate a completion for ``prompt`` and return the raw text.

        ``constraint`` is honored only when ``supports_grammar`` is true. Failures surface as
        :class:`~slmstrict.errors.BackendError`.
        """
        ...
