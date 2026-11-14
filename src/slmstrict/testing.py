"""Test helpers for code built on SlmStrict (public API)."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from slmstrict.backends.base import Backend, Constraint
from slmstrict.errors import BackendError

__all__ = ["BackendCall", "FakeBackend"]


@dataclass(frozen=True, slots=True)
class BackendCall:
    """One recorded :meth:`FakeBackend.agenerate` call."""

    prompt: str
    constraint: Constraint | None


class FakeBackend:
    """Scripted in-memory backend.

    Each call to :meth:`agenerate` consumes the next scripted item: a ``str`` is returned, an
    exception instance is raised. When the script runs out a :class:`BackendError` is raised, so
    a test that triggers more calls than it planned fails loudly. Every call is recorded in
    :attr:`calls`.
    """

    def __init__(
        self,
        outputs: Iterable[str | BaseException] = (),
        *,
        supports_grammar: bool = False,
    ) -> None:
        self._outputs: deque[str | BaseException] = deque(outputs)
        self.supports_grammar = supports_grammar
        self.calls: list[BackendCall] = []

    def enqueue(self, *outputs: str | BaseException) -> None:
        """Append more scripted outputs."""
        self._outputs.extend(outputs)

    @property
    def remaining(self) -> int:
        """Number of scripted outputs not consumed yet."""
        return len(self._outputs)

    async def agenerate(
        self,
        prompt: str,
        *,
        constraint: Constraint | None = None,
    ) -> str:
        self.calls.append(BackendCall(prompt=prompt, constraint=constraint))
        if not self._outputs:
            raise BackendError("FakeBackend has no scripted output left")
        item = self._outputs.popleft()
        if isinstance(item, BaseException):
            raise item
        return item


if TYPE_CHECKING:
    # Static proof (checked by `mypy --strict src`) that FakeBackend satisfies the Backend protocol.
    _backend_conformance: Backend = FakeBackend()
