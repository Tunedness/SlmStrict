import pytest

from slmstrict import (
    Backend,
    BackendError,
    SchemaError,
    SlmStrictError,
    ValidationFailed,
    ValidationIssue,
)
from slmstrict.backends import Constraint
from slmstrict.testing import FakeBackend


class MinimalBackend:
    """A hand-written adapter; mypy checks it against the protocol where it is passed in."""

    @property
    def supports_grammar(self) -> bool:
        return True

    async def agenerate(
        self,
        prompt: str,
        *,
        constraint: Constraint | None = None,
    ) -> str:
        return f"{prompt}|{constraint is not None}"


class NotABackend:
    async def agenerate(self, prompt: str) -> str:
        return prompt


def use_backend(backend: Backend) -> bool:
    return backend.supports_grammar


def test_fake_backend_satisfies_protocol_statically() -> None:
    # mypy proves the conformance at the call site (see `mypy --strict src tests`).
    assert use_backend(FakeBackend(supports_grammar=True)) is True
    assert use_backend(FakeBackend()) is False


def test_runtime_protocol_check() -> None:
    assert isinstance(FakeBackend(), Backend)
    assert isinstance(MinimalBackend(), Backend)
    assert not isinstance(NotABackend(), Backend)


async def test_minimal_backend_call_shape() -> None:
    backend: Backend = MinimalBackend()
    assert await backend.agenerate("p") == "p|False"
    assert await backend.agenerate("p", constraint=Constraint("gbnf", "x")) == "p|True"


def test_constraint_is_opaque_and_immutable() -> None:
    constraint = Constraint(kind="gbnf", payload=object())
    assert constraint.kind == "gbnf"
    with pytest.raises(AttributeError):
        constraint.kind = "other"  # type: ignore[misc]


def test_error_hierarchy() -> None:
    for error in (SchemaError, ValidationFailed, BackendError):
        assert issubclass(error, SlmStrictError)
    assert issubclass(SlmStrictError, Exception)


def test_validation_failed_carries_issues_but_not_in_message() -> None:
    issue = ValidationIssue(("a",), "type", "expected integer, got string")
    error = ValidationFailed(
        "output did not match the schema", issues=(issue,), last_output="secret"
    )
    assert error.issues == (issue,)
    assert error.last_output == "secret"
    assert "secret" not in str(error)
    assert ValidationFailed("x").issues == ()
