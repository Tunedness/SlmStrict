import pytest

from slmstrict import BackendError, Constraint
from slmstrict.testing import BackendCall, FakeBackend


async def test_returns_scripted_outputs_in_order() -> None:
    backend = FakeBackend(["one", "two"])
    assert await backend.agenerate("a") == "one"
    assert await backend.agenerate("b") == "two"
    assert backend.remaining == 0


async def test_records_calls() -> None:
    backend = FakeBackend(["x", "y"], supports_grammar=True)
    constraint = Constraint("gbnf", "root ::= x")
    await backend.agenerate("first")
    await backend.agenerate("second", constraint=constraint)
    assert backend.calls == [
        BackendCall(prompt="first", constraint=None),
        BackendCall(prompt="second", constraint=constraint),
    ]


async def test_exhausted_script_raises_backend_error() -> None:
    backend = FakeBackend(["only"])
    await backend.agenerate("a")
    with pytest.raises(BackendError, match="no scripted output"):
        await backend.agenerate("b")
    assert len(backend.calls) == 2  # the failing call is recorded too


async def test_exception_items_are_raised() -> None:
    backend = FakeBackend([BackendError("boom"), "ok"])
    with pytest.raises(BackendError, match="boom"):
        await backend.agenerate("a")
    assert await backend.agenerate("b") == "ok"


async def test_enqueue_appends() -> None:
    backend = FakeBackend()
    backend.enqueue("a", "b")
    assert backend.remaining == 2
    assert await backend.agenerate("p") == "a"


def test_supports_grammar_is_settable() -> None:
    backend = FakeBackend()
    assert backend.supports_grammar is False
    backend.supports_grammar = True
    assert backend.supports_grammar is True
    assert FakeBackend(supports_grammar=True).supports_grammar is True
