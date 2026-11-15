from __future__ import annotations

import json
import sys
import threading
import types
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from slmstrict import BackendError, Constraint, Schema
from slmstrict.backends._grammar import (
    GBNF_KIND,
    GrammarCache,
    clear_grammar_cache,
    compile_grammar,
    load_llama_cpp,
)


class Person(BaseModel):
    name: str
    age: int


class Pet(BaseModel):
    name: str


@pytest.fixture
def fake_llama_cpp(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """Install a fake ``llama_cpp`` module; ``from_json_schema`` is a counting mock."""
    from_json_schema = MagicMock(side_effect=lambda text: f"grammar<{text}>")
    module = types.ModuleType("llama_cpp")
    module.LlamaGrammar = types.SimpleNamespace(from_json_schema=from_json_schema)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    clear_grammar_cache()
    yield from_json_schema
    clear_grammar_cache()


def test_same_schema_is_compiled_once(fake_llama_cpp: MagicMock) -> None:
    first = compile_grammar(Schema.of(Person))
    second = compile_grammar(Schema.of(Person))
    assert fake_llama_cpp.call_count == 1
    assert first is second


def test_model_and_equivalent_dict_share_one_entry(fake_llama_cpp: MagicMock) -> None:
    compile_grammar(Schema.of(Person))
    compile_grammar(Schema.of(Person.model_json_schema()))
    assert fake_llama_cpp.call_count == 1


class Answer(BaseModel):
    reasoning: str
    answer: str


def test_grammar_is_compiled_from_the_schema_own_property_order(
    fake_llama_cpp: MagicMock,
) -> None:
    compile_grammar(Schema.of(Answer))
    (text,) = fake_llama_cpp.call_args.args
    assert list(json.loads(text)["properties"]) == ["reasoning", "answer"]


def test_schemas_differing_only_in_property_order_get_separate_grammars(
    fake_llama_cpp: MagicMock,
) -> None:
    forward = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
    backward = {"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "string"}}}
    first, second = Schema.of(forward), Schema.of(backward)
    assert first.schema_hash == second.schema_hash  # the hash alone cannot tell them apart
    compile_grammar(first)
    compile_grammar(second)
    assert fake_llama_cpp.call_count == 2
    orders = [
        list(json.loads(call.args[0])["properties"]) for call in fake_llama_cpp.call_args_list
    ]
    assert orders == [["a", "b"], ["b", "a"]]


def test_different_schemas_compile_separately(fake_llama_cpp: MagicMock) -> None:
    compile_grammar(Schema.of(Person))
    compile_grammar(Schema.of(Pet))
    assert fake_llama_cpp.call_count == 2


def test_result_is_a_gbnf_constraint_with_the_compiled_grammar(fake_llama_cpp: MagicMock) -> None:
    schema = Schema.of(Person)
    constraint = compile_grammar(schema)
    assert isinstance(constraint, Constraint)
    assert constraint.kind == GBNF_KIND
    assert constraint.payload == f"grammar<{json.dumps(schema.json_schema, separators=(',', ':'))}>"
    fake_llama_cpp.assert_called_once()


def test_compile_failure_becomes_backend_error(fake_llama_cpp: MagicMock) -> None:
    fake_llama_cpp.side_effect = ValueError("unsupported keyword secret-field")
    with pytest.raises(BackendError, match="ValueError") as info:
        compile_grammar(Schema.of(Person))
    assert "secret-field" not in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)


def test_failed_compile_is_not_cached(fake_llama_cpp: MagicMock) -> None:
    fake_llama_cpp.side_effect = [ValueError("boom"), "grammar"]
    schema = Schema.of(Person)
    with pytest.raises(BackendError):
        compile_grammar(schema)
    assert compile_grammar(schema).payload == "grammar"


def test_missing_extra_raises_import_error_with_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # makes `import llama_cpp` fail
    with pytest.raises(ImportError, match=r"pip install 'slmstrict\[llamacpp\]'"):
        load_llama_cpp()
    clear_grammar_cache()
    with pytest.raises(ImportError, match="llamacpp"):
        compile_grammar(Schema.of(Person))


def _counting_cache(maxsize: int) -> tuple[GrammarCache, list[str]]:
    compiled: list[str] = []

    def compiler(schema: Schema[Any]) -> Constraint:
        compiled.append(schema.schema_hash)
        return Constraint(GBNF_KIND, schema.schema_hash)

    return GrammarCache(maxsize=maxsize, compiler=compiler), compiled


def test_lru_evicts_the_least_recently_used_entry() -> None:
    cache, compiled = _counting_cache(2)
    schemas = [Schema.of({"type": "object", "minProperties": n}) for n in range(3)]
    cache.get_or_compile(schemas[0])
    cache.get_or_compile(schemas[1])
    cache.get_or_compile(schemas[0])  # refresh 0; 1 is now the oldest
    cache.get_or_compile(schemas[2])  # evicts 1
    assert len(cache) == 2
    assert schemas[0] in cache
    assert schemas[1] not in cache
    cache.get_or_compile(schemas[1])
    assert compiled.count(schemas[1].schema_hash) == 2


def test_clear_empties_the_cache() -> None:
    cache, compiled = _counting_cache(4)
    schema = Schema.of(Person)
    cache.get_or_compile(schema)
    cache.clear()
    assert len(cache) == 0
    cache.get_or_compile(schema)
    assert len(compiled) == 2


def test_maxsize_must_be_positive() -> None:
    with pytest.raises(ValueError, match="maxsize"):
        GrammarCache(maxsize=0)


def test_concurrent_callers_trigger_a_single_compilation() -> None:
    cache, compiled = _counting_cache(4)
    schema = Schema.of(Person)
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        cache.get_or_compile(schema)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(compiled) == 1
