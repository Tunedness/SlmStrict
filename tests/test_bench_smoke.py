"""Smoke tests for the reference benchmark (``bench/``) and the README examples.

Nothing here loads a model: the benchmark runs on scripted ``FakeBackend`` outputs. The real run
needs model files and is described in ``bench/README.md``.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from bench.__main__ import main
from bench.report import (
    compute_stats,
    length_cautions,
    missed_targets,
    render_report,
    write_report,
)
from bench.runner import (
    REFERENCE_MODELS,
    BenchError,
    BenchmarkRun,
    CellResult,
    ModelSpec,
    Settings,
    Task,
    TaskRecord,
    collect_environment,
    load_config,
    load_tasks,
    read_revision,
    run_benchmark,
    run_cell,
    tasks_digest,
)

from slmstrict import Backend, BackendError, Step
from slmstrict.testing import FakeBackend
from tests._ladder_support import GrammarBackend

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
NO_WARMUP = Settings(warmup=False)

CONTACT = (
    '{"name": "Dana Whitfield", "email": "dana.whitfield@example.com", "phone": "+1 555 0134"}'
)
PRODUCT = '{"name": "Aurora", "brand": "Lumeo", "price": 49.9, "in_stock": true}'
FENCED_PRODUCT = f"```json\n{PRODUCT}\n```"

OK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def _two_tasks() -> tuple[Task, Task]:
    by_id = {task.id: task for task in load_tasks(BENCH / "tasks")}
    return by_id["flat_object-01-contact"], by_id["flat_object-02-product"]


# -- the task set -----------------------------------------------------------------------------


def test_task_set_is_complete_and_loadable() -> None:
    tasks = load_tasks(BENCH / "tasks")
    assert len(tasks) == 36  # bench/README.md and README.md say 36
    assert len({t.id for t in tasks}) == len(tasks)
    categories = {t.category for t in tasks}
    assert categories == {"flat_object", "nested_object", "array", "enum", "numeric", "optional"}
    for category in categories:
        assert sum(1 for t in tasks if t.category == category) >= 5
    assert len(tasks_digest(tasks)) == 64


def test_task_prompt_carries_the_schema() -> None:
    task = _two_tasks()[0]
    prompt = task.full_prompt()
    assert prompt.startswith(task.prompt)
    assert '"properties"' in prompt


def test_load_tasks_rejects_a_task_without_a_schema(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text(json.dumps({"id": "x", "category": "c", "prompt": "p"}))
    with pytest.raises(BenchError, match="schema"):
        load_tasks(tmp_path)


def test_load_tasks_rejects_an_unusable_schema(tmp_path: Path) -> None:
    bad = {"id": "x", "category": "c", "prompt": "p", "schema": {"type": "object", "if": {}}}
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(BenchError, match="unusable schema"):
        load_tasks(tmp_path)


# -- configuration ----------------------------------------------------------------------------


def test_example_config_describes_the_reference_set() -> None:
    config = load_config(BENCH / "models.toml.example")
    assert tuple(spec.name for spec in config.models) == REFERENCE_MODELS
    for spec in config.models:
        assert spec.gguf_path is not None
        assert spec.onnx_dir is not None
        assert spec.prompt_template is not None
        assert "{prompt}" in spec.prompt_template


def test_config_rejects_unknown_keys_bad_templates_and_bad_settings(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text('[[models]]\nname = "A"\ngguf_pth = "x"\n')
    with pytest.raises(BenchError, match="unknown key"):
        load_config(path)
    path.write_text('[[models]]\nname = "A"\nprompt_template = "no placeholder"\n')
    with pytest.raises(BenchError, match="prompt_template"):
        load_config(path)
    path.write_text("[settings]\nmax_tokens = 0\n[[models]]\nname = 'A'\n")
    with pytest.raises(BenchError, match="max_tokens"):
        load_config(path)


# -- the runner on FakeBackend ----------------------------------------------------------------


async def test_two_tasks_raw_and_strict() -> None:
    contact, product = _two_tasks()
    # Backend call order for two tasks: raw 0, raw 1, strict 0, strict 1 (see run_cell).
    backend = FakeBackend([CONTACT, FENCED_PRODUCT, CONTACT, FENCED_PRODUCT])

    cell = await run_cell("Fake", "fake", None, backend, [contact, product], NO_WARMUP)

    assert cell.status == "ok"
    stats = compute_stats(cell.records)
    assert (stats.tasks, stats.raw_valid, stats.strict_valid) == (2, 1, 2)
    assert stats.fallback_reached == 0
    assert stats.errors == 0
    assert stats.grammar_decode_overhead is None  # this backend has no grammar
    assert cell.metrics is not None
    assert dict(cell.metrics.path_counts) == {(): 1, (Step.REPAIR,): 1}
    assert backend.remaining == 0


async def test_grammar_path_is_timed_separately() -> None:
    contact = _two_tasks()[0]
    backend = GrammarBackend([CONTACT, CONTACT])

    cell = await run_cell("Fake", "fake", None, backend, [contact], NO_WARMUP)

    (record,) = cell.records
    assert backend.built == 1
    assert record.grammar_decode_seconds is not None
    assert record.raw_chars == len(CONTACT)
    assert record.grammar_chars == len(CONTACT)
    stats = compute_stats(cell.records)
    assert stats.grammar_decode_overhead is not None
    assert stats.grammar_total_overhead is not None
    assert cell.metrics is not None
    assert dict(cell.metrics.path_counts) == {(Step.GRAMMAR,): 1}


async def test_a_task_that_reaches_the_fallback_is_counted_and_nothing_is_sent() -> None:
    contact = _two_tasks()[0]
    # One raw call, then the strict run's first attempt and its two re-asks all fail.
    backend = FakeBackend(["no", "no", "no", "no"])

    cell = await run_cell("Fake", "fake", None, backend, [contact], NO_WARMUP)

    (record,) = cell.records
    assert record.strict_status == "fallback"
    assert record.raw_status == "invalid"
    stats = compute_stats(cell.records)
    assert (stats.strict_valid, stats.fallback_reached) == (0, 1)
    assert stats.fallback_rate == 1.0
    assert cell.metrics is not None
    assert cell.metrics.fallback_needed == 1
    assert Step.FALLBACK in next(iter(cell.metrics.path_counts))
    assert len(backend.calls) == 4  # the stub added no further call of its own


async def test_a_failing_backend_is_an_error_not_a_crash() -> None:
    contact = _two_tasks()[0]
    backend = FakeBackend([BackendError("boom"), BackendError("boom")])

    cell = await run_cell("Fake", "fake", None, backend, [contact], NO_WARMUP)

    (record,) = cell.records
    assert (record.raw_status, record.strict_status) == ("error", "error")
    assert record.error_type == "BackendError"
    stats = compute_stats(cell.records)
    assert (stats.errors, stats.strict_valid) == (1, 0)


async def test_a_failed_warm_up_fails_the_cell() -> None:
    contact = _two_tasks()[0]
    cell = await run_cell("Fake", "fake", None, FakeBackend([]), [contact], Settings())
    assert cell.status == "failed"
    assert cell.detail is not None
    assert "warm-up" in cell.detail


async def test_prompt_template_wraps_every_prompt() -> None:
    contact = _two_tasks()[0]
    backend = FakeBackend([CONTACT, CONTACT])

    await run_cell(
        "Fake", "fake", None, backend, [contact], NO_WARMUP, prompt_template="<u>{prompt}</u>"
    )

    assert len(backend.calls) == 2
    for call in backend.calls:
        assert call.prompt.startswith("<u>")
        assert call.prompt.endswith("</u>")


class _RecordingBackend:
    """Backend that logs every prompt it receives and answers with a valid object."""

    supports_grammar = False

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def agenerate(self, prompt: str, *, constraint: object = None) -> str:
        self.prompts.append(prompt)
        return '{"ok": true}'


async def test_consecutive_backend_calls_never_share_a_prompt() -> None:
    # llama.cpp reuses the KV-cache prefix of the previous call; a raw call directly followed by
    # the strict call of the same task would give the strict call the prompt evaluation for free.
    tasks = [Task(f"t{n}", "c", f"Say ok, variant {n}.", OK_SCHEMA) for n in range(5)]
    backend = _RecordingBackend()

    cell = await run_cell("Fake", "fake", None, backend, tasks, NO_WARMUP)  # type: ignore[arg-type]

    assert len(backend.prompts) == 10
    assert len(set(backend.prompts)) == 5  # each prompt exactly twice
    for before, after in zip(backend.prompts, backend.prompts[1:], strict=False):
        assert before != after
    assert [r.task_id for r in cell.records] == [t.id for t in tasks]
    assert all(r.raw_status == "valid" and r.strict_status == "valid" for r in cell.records)


@pytest.mark.parametrize(("cold", "expected_clears"), [(True, 2), (False, 0)])
async def test_cold_grammar_clears_the_cache_before_every_strict_run(
    monkeypatch: pytest.MonkeyPatch, cold: bool, expected_clears: int
) -> None:
    clears: list[int] = []
    monkeypatch.setattr("bench.runner.clear_grammar_cache", lambda: clears.append(1))
    backend = GrammarBackend([CONTACT, FENCED_PRODUCT, CONTACT, FENCED_PRODUCT])

    await run_cell("Fake", "fake", None, backend, list(_two_tasks()), NO_WARMUP, cold_grammar=cold)

    assert len(clears) == expected_clears


# -- the arithmetic behind the verdicts -------------------------------------------------------


def _record(**overrides: Any) -> TaskRecord:
    values: dict[str, Any] = {
        "task_id": "t",
        "category": "c",
        "raw_status": "valid",
        "raw_seconds": 10.0,
        "strict_status": "valid",
        "strict_seconds": 12.0,
        "grammar_decode_seconds": 11.0,
        "grammar_build_seconds": 0.5,
        "raw_chars": 100,
        "grammar_chars": 100,
    }
    values.update(overrides)
    return TaskRecord(**values)


def test_compute_stats_ratios_from_known_times() -> None:
    records = [
        _record(),
        _record(
            raw_seconds=20.0,
            strict_seconds=26.0,
            grammar_decode_seconds=22.0,
            grammar_build_seconds=1.0,
            raw_chars=200,
            grammar_chars=100,
        ),
        _record(raw_status="invalid", strict_status="fallback", strict_seconds=50.0),
    ]

    stats = compute_stats(records)

    assert (stats.tasks, stats.raw_valid, stats.strict_valid, stats.fallback_reached) == (
        3,
        2,
        2,
        1,
    )
    # decode: (11 + 22 + 11) / (10 + 20 + 10) - 1
    assert stats.grammar_decode_overhead == pytest.approx(44 / 40 - 1)
    # with build: (44 + 0.5 + 1.0 + 0.5) / 40 - 1
    assert stats.grammar_total_overhead == pytest.approx(46 / 40 - 1)
    # per character: (44 / 300) / (40 / 400) - 1
    assert stats.grammar_per_char_overhead == pytest.approx((44 / 300) / (40 / 400) - 1)
    # whole ladder: (12 + 26 + 50) / 40 - 1
    assert stats.ladder_overhead == pytest.approx(88 / 40 - 1)
    assert stats.mean_raw_chars == pytest.approx(400 / 3)


def test_compute_stats_without_grammar_calls_has_no_grammar_figures() -> None:
    stats = compute_stats([_record(grammar_decode_seconds=None, grammar_chars=0)])
    assert stats.grammar_decode_overhead is None
    assert stats.grammar_total_overhead is None
    assert stats.grammar_per_char_overhead is None
    assert stats.ladder_overhead == pytest.approx(12 / 10 - 1)


def _run_of(*records: TaskRecord) -> BenchmarkRun:
    cell = CellResult("Fake", "llamacpp", None, "ok", records=tuple(records))

    return BenchmarkRun(
        datetime.now().astimezone(), {}, Settings(), len(records), "0" * 64, (cell,)
    )


def test_report_names_source_and_quantization_of_every_model() -> None:
    cell = CellResult(
        "Fake",
        "onnx",
        "int4 (rtn, block 32)",
        "ok",
        records=(_record(grammar_decode_seconds=None, grammar_chars=0),),
        source="vendor/fake-onnx",
        revision="commit 0123456789ab, model.onnx.data sha256 fedcba9876543210",
    )
    started = datetime.now().astimezone()
    run = BenchmarkRun(started, {}, Settings(), 1, "0" * 64, (cell,))

    text = render_report(run)

    assert (
        "| Fake | onnx | vendor/fake-onnx | int4 (rtn, block 32) | "
        "commit 0123456789ab, model.onnx.data sha256 fedcba9876543210 | ok |"
    ) in text
    assert "conversions made by third parties" in text


def test_read_revision_uses_the_downloaders_manifest(tmp_path: Path) -> None:
    model = tmp_path / "Tiny"
    (model / "gguf").mkdir(parents=True)
    (model / "onnx").mkdir()
    manifest = {
        "gguf": {"commit": "a" * 40, "files": [{"path": "m.gguf", "size": 9, "sha256": "b" * 64}]},
        "onnx": {
            "commit": "c" * 40,
            "files": [
                {"path": "genai_config.json", "size": 1, "sha256": "d" * 64},
                {"path": "model.onnx.data", "size": 99, "sha256": "e" * 64},
            ],
        },
    }
    (model / "manifest.json").write_text(json.dumps(manifest))

    gguf = read_revision(model / "gguf" / "m.gguf", "llamacpp")
    onnx = read_revision(model / "onnx", "onnx")

    assert gguf == f"commit {'a' * 12}, m.gguf sha256 {'b' * 16}"
    assert onnx == f"commit {'c' * 12}, model.onnx.data sha256 {'e' * 16}"


def test_read_revision_is_none_without_a_usable_manifest(tmp_path: Path) -> None:
    assert read_revision(None, "onnx") is None
    assert read_revision(tmp_path / "onnx", "onnx") is None  # no manifest
    (tmp_path / "manifest.json").write_text("{not json")
    assert read_revision(tmp_path / "onnx", "onnx") is None
    (tmp_path / "manifest.json").write_text('{"onnx": {"commit": 1}}')
    assert read_revision(tmp_path / "onnx", "onnx") is None


def test_environment_records_the_llama_build_flags_and_system_info(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    flags = tmp_path / "flags.txt"
    flags.write_text("-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod\n")
    monkeypatch.setenv("SLMSTRICT_LLAMA_BUILD_FILE", str(flags))

    class FakeLlamaCpp:
        @staticmethod
        def llama_print_system_info() -> bytes:
            return b"CPU : NEON = 1 | ARM_FMA = 1 |\n DOTPROD = 1 |"

    monkeypatch.setitem(sys.modules, "llama_cpp", FakeLlamaCpp)

    environment = collect_environment(Settings())

    assert environment["llama.cpp build flags"] == (
        "-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod"
    )
    assert environment["llama.cpp system info"] == "CPU : NEON = 1 | ARM_FMA = 1 | DOTPROD = 1 |"


def test_environment_says_so_when_the_build_is_not_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SLMSTRICT_LLAMA_BUILD_FILE", raising=False)
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # an import of it fails

    environment = collect_environment(Settings())

    assert environment["llama.cpp build flags"].startswith("not recorded")
    assert environment["llama.cpp system info"].startswith("not available")

    monkeypatch.setenv("SLMSTRICT_LLAMA_BUILD_FILE", str(tmp_path / "missing.txt"))
    assert collect_environment(Settings())["llama.cpp build flags"].startswith("not recorded")


def test_missed_targets_flags_grammar_overhead_above_the_limit() -> None:
    slow = _record(grammar_decode_seconds=12.0, grammar_build_seconds=0.0)  # +20 percent
    lines = missed_targets(_run_of(slow, slow))
    assert any("grammar overhead (decoding) +20.0%" in line for line in lines)
    assert any("including grammar build" in line for line in lines)


def test_missed_targets_accepts_overhead_at_the_limit_and_flags_nothing_else() -> None:
    at_limit = _record(grammar_decode_seconds=11.5, grammar_build_seconds=0.0)  # +15 percent
    assert missed_targets(_run_of(at_limit, at_limit)) == []


def test_missed_targets_flags_pass_rate_and_fallback_rate() -> None:
    bad = _record(strict_status="fallback", grammar_decode_seconds=None, grammar_chars=0)
    lines = missed_targets(_run_of(bad))
    assert any("strict pass rate 0.0%" in line for line in lines)
    assert any("fallback rate 100.0%" in line for line in lines)


def test_length_caution_when_only_the_per_character_figure_is_over_the_limit() -> None:
    # Total time looks fine (+10 percent) but the grammar output is half as long: +120 per char.
    short_output = _record(raw_seconds=10.0, grammar_decode_seconds=11.0, grammar_chars=45)
    run = _run_of(short_output)
    assert missed_targets(run) == []
    (caution,) = length_cautions(run)
    assert "per output character" in caution
    assert "Read with care" in render_report(run)


# -- orchestration, report and entry point ----------------------------------------------------


class _ConstantBackend:
    """Backend that always answers with the same JSON object."""

    supports_grammar = False

    async def agenerate(self, prompt: str, *, constraint: object = None) -> str:
        return '{"ok": true}'


def _constant_factory(spec: ModelSpec, backend: str, settings: Settings) -> Backend:
    return _ConstantBackend()  # type: ignore[return-value]


def _write_fixture(tmp_path: Path, *, models: tuple[str, ...] = REFERENCE_MODELS) -> Path:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for number, category in enumerate(("flat_object", "enum"), start=1):
        doc = {
            "id": f"t{number}",
            "category": category,
            "prompt": f"Say ok, variant {number}.",
            "schema": OK_SCHEMA,
        }
        (tasks / f"t{number}.json").write_text(json.dumps(doc))
    config = tmp_path / "models.toml"
    config.write_text(
        "\n".join(
            f'[[models]]\nname = "{name}"\ngguf_path = "/nowhere/{name}.gguf"\n'
            f'gguf_quantization = "Q4_K_M"\nonnx_dir = "/nowhere/{name}-onnx"\n'
            for name in models
        )
    )
    return config


def test_main_writes_a_report_for_three_models_and_two_backends(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    output = tmp_path / "results"
    argv = [
        "--models",
        str(config),
        "--tasks",
        str(tmp_path / "tasks"),
        "--output",
        str(output),
    ]
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        code = main(argv, factory=_constant_factory)

    assert code == 0
    (report,) = output.glob("*.md")
    text = report.read_text(encoding="utf-8")
    for model in REFERENCE_MODELS:
        assert f"| {model} | llamacpp |" in text
        assert f"| {model} | onnx |" in text
    assert "Reference set complete" in text
    assert "## Targets (PRD section 6)" in text
    assert "Fallback rate" in text
    assert "Every target was met" in text
    assert f"report: {report}" in stdout.getvalue()


def test_main_reports_exit_code_two_for_bad_input(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert main(["--models", str(tmp_path / "missing.toml")]) == 2
        assert (
            main(["--models", str(config), "--tasks", str(tmp_path / "tasks"), "--model", "X"]) == 2
        )
    assert "unknown model" in err.getvalue()


async def test_incomplete_reference_set_and_missed_targets_are_flagged(tmp_path: Path) -> None:
    config = load_config(_write_fixture(tmp_path, models=("Gemma-3-1B",)))
    tasks = load_tasks(tmp_path / "tasks")

    def failing_factory(spec: ModelSpec, backend: str, settings: Settings) -> Backend:
        return FakeBackend(["no"] * 20)

    run = await run_benchmark(config, tasks, backends=("onnx",), factory=failing_factory)
    text = render_report(run)

    assert "Reference set INCOMPLETE" in text
    assert "Qwen3-4B / llamacpp" in text
    assert "Targets MISSED" in text
    assert any("strict pass rate" in line for line in missed_targets(run))
    assert any("fallback rate" in line for line in missed_targets(run))


async def test_a_model_without_a_location_is_skipped_and_a_broken_backend_fails_its_cell(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)
    config_path.write_text('[[models]]\nname = "OnlyGguf"\ngguf_path = "/nowhere/m.gguf"\n')
    tasks = load_tasks(tmp_path / "tasks")

    def broken_factory(spec: ModelSpec, backend: str, settings: Settings) -> Backend:
        raise BackendError("model file not found")

    run = await run_benchmark(load_config(config_path), tasks, factory=broken_factory)

    by_backend = {cell.backend: cell for cell in run.cells}
    assert by_backend["onnx"].status == "skipped"
    assert by_backend["llamacpp"].status == "failed"
    assert "model file not found" in (by_backend["llamacpp"].detail or "")
    assert "not run (skipped)" in render_report(run)


async def test_reports_are_never_overwritten(tmp_path: Path) -> None:
    config = load_config(_write_fixture(tmp_path))
    tasks = load_tasks(tmp_path / "tasks")
    run = await run_benchmark(config, tasks, factory=_constant_factory)

    first = write_report(run, tmp_path / "results")
    second = write_report(run, tmp_path / "results")

    assert first != second
    assert second.name.endswith("-2.md")
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


async def test_a_latency_comparison_needs_at_least_two_tasks(tmp_path: Path) -> None:
    config = load_config(_write_fixture(tmp_path))
    tasks = load_tasks(tmp_path / "tasks", limit=1)
    with pytest.raises(BenchError, match="at least 2 tasks"):
        await run_benchmark(config, tasks, factory=_constant_factory)


def test_main_rejects_a_single_task_trial(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        argv = ["--models", str(config), "--tasks", str(tmp_path / "tasks"), "--limit", "1"]
        assert main(argv) == 2
    assert "at least 2 tasks" in err.getvalue()


def test_load_tasks_rejects_two_tasks_with_one_prompt(tmp_path: Path) -> None:
    for name in ("a", "b"):
        doc = {"id": name, "category": "c", "prompt": "same", "schema": OK_SCHEMA}
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))
    with pytest.raises(BenchError, match="same prompt"):
        load_tasks(tmp_path)


def test_results_directory_is_kept_in_the_repository() -> None:
    assert (BENCH / "results" / ".gitkeep").is_file()


# -- the README examples ----------------------------------------------------------------------

_EXAMPLE = re.compile(
    r"<!-- runnable -->\n```python\n(?P<code>.*?)```\n\n<!-- output -->\n```text\n(?P<out>.*?)```",
    re.DOTALL,
)


def _readme_examples() -> list[tuple[str, str]]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    return [(m["code"], m["out"]) for m in _EXAMPLE.finditer(text)]


def test_readme_has_runnable_examples() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    # Every marker must be matched by the pattern, so a malformed example cannot slip through.
    assert len(_readme_examples()) == text.count("<!-- runnable -->") >= 5


@pytest.mark.parametrize(("code", "expected"), _readme_examples())
def test_readme_example_runs_verbatim_and_prints_what_the_readme_shows(
    code: str, expected: str
) -> None:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(code, "<README.md>", "exec"), {"__name__": "__readme__"})
    assert captured.getvalue() == expected
