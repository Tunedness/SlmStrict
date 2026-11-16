"""Markdown report: the PRD section 6 targets against what was measured."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from slmstrict import Step

from .runner import BACKENDS, REFERENCE_MODELS, BenchmarkRun, TaskRecord

__all__ = [
    "FALLBACK_RATE_LIMIT",
    "GRAMMAR_OVERHEAD_LIMIT",
    "STRICT_PASS_TARGET",
    "CellStats",
    "compute_stats",
    "length_cautions",
    "missed_targets",
    "render_report",
    "write_report",
]

STRICT_PASS_TARGET: Final = 0.95
GRAMMAR_OVERHEAD_LIMIT: Final = 0.15
FALLBACK_RATE_LIMIT: Final = 0.10


@dataclass(frozen=True, slots=True)
class CellStats:
    """Aggregates of one cell.

    ``strict_valid`` counts tasks the local rungs resolved (the cloud fallback is a stub, so a
    run that reached it is not counted as valid). Overheads are relative: ``0.10`` is 10 percent
    slower than the raw run of the same tasks, and ``None`` when there is nothing to compare.
    ``grammar_per_char_overhead`` is the same comparison per output character (seconds per
    character of the first grammar call against the raw call), which removes the effect of the two
    calls producing outputs of different length. The mean lengths are over the grammar tasks.
    """

    tasks: int
    raw_valid: int
    strict_valid: int
    fallback_reached: int
    errors: int
    grammar_decode_overhead: float | None
    grammar_total_overhead: float | None
    ladder_overhead: float | None
    grammar_per_char_overhead: float | None = None
    mean_raw_chars: float | None = None
    mean_grammar_chars: float | None = None

    @property
    def raw_pass_rate(self) -> float:
        return self.raw_valid / self.tasks if self.tasks else 0.0

    @property
    def strict_pass_rate(self) -> float:
        return self.strict_valid / self.tasks if self.tasks else 0.0

    @property
    def fallback_rate(self) -> float:
        return self.fallback_reached / self.tasks if self.tasks else 0.0


def compute_stats(records: Sequence[TaskRecord]) -> CellStats:
    """Aggregate the task records of one cell."""
    comparable = [r for r in records if r.raw_status != "error" and r.strict_status != "error"]
    raw_total = sum(r.raw_seconds for r in comparable)

    grammar = [r for r in comparable if r.grammar_decode_seconds is not None]
    grammar_raw = sum(r.raw_seconds for r in grammar)
    decode = sum(r.grammar_decode_seconds or 0.0 for r in grammar)
    build = sum(r.grammar_build_seconds for r in grammar)

    sized = [r for r in grammar if r.raw_chars > 0 and r.grammar_chars > 0]
    sized_raw_seconds = sum(r.raw_seconds for r in sized)
    sized_decode_seconds = sum(r.grammar_decode_seconds or 0.0 for r in sized)
    raw_chars = sum(r.raw_chars for r in sized)
    grammar_chars = sum(r.grammar_chars for r in sized)
    per_char: float | None = None
    if sized and sized_raw_seconds > 0:
        per_char = (sized_decode_seconds / grammar_chars) / (sized_raw_seconds / raw_chars) - 1

    return CellStats(
        tasks=len(records),
        raw_valid=sum(1 for r in records if r.raw_status == "valid"),
        strict_valid=sum(1 for r in records if r.strict_status == "valid"),
        fallback_reached=sum(1 for r in records if r.strict_status == "fallback"),
        errors=sum(1 for r in records if "error" in (r.raw_status, r.strict_status)),
        grammar_decode_overhead=decode / grammar_raw - 1 if grammar_raw > 0 else None,
        grammar_total_overhead=(decode + build) / grammar_raw - 1 if grammar_raw > 0 else None,
        ladder_overhead=(
            sum(r.strict_seconds for r in comparable) / raw_total - 1 if raw_total > 0 else None
        ),
        grammar_per_char_overhead=per_char,
        mean_raw_chars=raw_chars / len(sized) if sized else None,
        mean_grammar_chars=grammar_chars / len(sized) if sized else None,
    )


def missed_targets(run: BenchmarkRun) -> list[str]:
    """One line per target a cell did not meet; empty when every measured target was met."""
    missed: list[str] = []
    for cell in run.cells:
        if cell.status != "ok":
            continue
        stats = compute_stats(cell.records)
        label = f"{cell.model} / {cell.backend}"
        if stats.strict_pass_rate < STRICT_PASS_TARGET:
            missed.append(
                f"{label}: strict pass rate {stats.strict_pass_rate:.1%} "
                f"is below {STRICT_PASS_TARGET:.0%}"
            )
        for name, value in (
            ("grammar overhead (decoding)", stats.grammar_decode_overhead),
            ("grammar overhead (including grammar build)", stats.grammar_total_overhead),
        ):
            if value is not None and value > GRAMMAR_OVERHEAD_LIMIT:
                missed.append(f"{label}: {name} {value:+.1%} is above {GRAMMAR_OVERHEAD_LIMIT:.0%}")
        if stats.fallback_rate >= FALLBACK_RATE_LIMIT:
            missed.append(
                f"{label}: fallback rate {stats.fallback_rate:.1%} "
                f"is not below {FALLBACK_RATE_LIMIT:.0%}"
            )
    return missed


def length_cautions(run: BenchmarkRun) -> list[str]:
    """Cells whose total-time grammar overhead is within the limit while the per-character one is
    not: the output lengths differ enough to hide a grammar cost. Not a miss, but worth a look."""
    cautions: list[str] = []
    for cell in run.cells:
        if cell.status != "ok":
            continue
        stats = compute_stats(cell.records)
        decode = stats.grammar_decode_overhead
        per_char = stats.grammar_per_char_overhead
        if (
            decode is not None
            and per_char is not None
            and decode <= GRAMMAR_OVERHEAD_LIMIT < per_char
        ):
            cautions.append(
                f"{cell.model} / {cell.backend}: grammar overhead {decode:+.1%} meets the limit "
                f"but is {per_char:+.1%} per output character (raw and grammar outputs differ in "
                "length)"
            )
    return cautions


# -- rendering --------------------------------------------------------------------------------


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return "\n".join(lines)


def _count_rate(count: int, total: int) -> str:
    return f"{count / total:.1%} ({count}/{total})" if total else "n/a"


def _verdict(ok: bool) -> str:
    return "met" if ok else "**MISSED**"


def _overhead(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%} ({_verdict(value <= GRAMMAR_OVERHEAD_LIMIT)})"


def _path_label(path: tuple[Step, ...]) -> str:
    return " + ".join(step.value for step in path) if path else "none (first output was valid)"


def _results_rows(run: BenchmarkRun) -> list[list[str]]:
    rows: list[list[str]] = []
    for cell in run.cells:
        head = [cell.model, cell.backend]
        if cell.status != "ok":
            reason = cell.detail or cell.status
            rows.append([*head, f"not run ({cell.status}): {reason}", *["-"] * 9])
            continue
        s = compute_stats(cell.records)
        rows.append(
            [
                *head,
                str(s.tasks),
                _count_rate(s.raw_valid, s.tasks),
                f"{_count_rate(s.strict_valid, s.tasks)} "
                f"({_verdict(s.strict_pass_rate >= STRICT_PASS_TARGET)})",
                _overhead(s.grammar_decode_overhead),
                _overhead(s.grammar_total_overhead),
                "n/a"
                if s.grammar_per_char_overhead is None
                else f"{s.grammar_per_char_overhead:+.1%}",
                "n/a"
                if s.mean_raw_chars is None or s.mean_grammar_chars is None
                else f"{s.mean_raw_chars:.0f} / {s.mean_grammar_chars:.0f}",
                f"{s.fallback_rate:.1%} ({s.fallback_reached}/{s.tasks}) "
                f"({_verdict(s.fallback_rate < FALLBACK_RATE_LIMIT)})",
                "n/a" if s.ladder_overhead is None else f"{s.ladder_overhead:+.1%}",
                str(s.errors),
            ]
        )
    return rows


def _path_rows(run: BenchmarkRun) -> list[list[str]]:
    rows: list[list[str]] = []
    for cell in run.cells:
        if cell.status != "ok" or cell.metrics is None:
            continue
        total = cell.metrics.runs
        ordered = sorted(cell.metrics.path_counts.items(), key=lambda item: (-item[1], item[0]))
        for path, count in ordered:
            rows.append([cell.model, cell.backend, _path_label(path), _count_rate(count, total)])
    return rows


def _category_rows(run: BenchmarkRun) -> tuple[list[str], list[list[str]]]:
    categories = sorted({r.category for c in run.cells if c.status == "ok" for r in c.records})
    rows: list[list[str]] = []
    for cell in run.cells:
        if cell.status != "ok":
            continue
        raw: Counter[str] = Counter()
        strict: Counter[str] = Counter()
        total: Counter[str] = Counter()
        for r in cell.records:
            total[r.category] += 1
            raw[r.category] += r.raw_status == "valid"
            strict[r.category] += r.strict_status == "valid"
        rows.append(
            [
                cell.model,
                cell.backend,
                *[f"{raw[c]}/{total[c]} -> {strict[c]}/{total[c]}" for c in categories],
            ]
        )
    return categories, rows


def _coverage(run: BenchmarkRun) -> list[str]:
    done = {(c.model, c.backend) for c in run.cells if c.status == "ok"}
    return [
        f"{model} / {backend}"
        for model in REFERENCE_MODELS
        for backend in BACKENDS
        if (model, backend) not in done
    ]


def _models_rows(run: BenchmarkRun) -> list[list[str]]:
    return [
        [
            c.model,
            c.backend,
            c.source or "not stated",
            c.quantization or "not stated",
            c.revision or "not recorded",
            c.status,
        ]
        for c in run.cells
    ]


def _duration(seconds: float) -> str:
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def _timing(run: BenchmarkRun) -> str:
    text = f"started {run.started_at.isoformat(timespec='seconds')}"
    if run.finished_at is not None:
        text += f", finished {run.finished_at.isoformat(timespec='seconds')}"
    if run.duration_s is not None:
        text += f" (took {_duration(run.duration_s)})"
    return text


def render_report(run: BenchmarkRun) -> str:
    """The whole report as Markdown."""
    date = run.started_at.date().isoformat()
    out: list[str] = [f"# SlmStrict benchmark - {date}", ""]
    out.append(
        f"Measured {run.task_count} tasks per cell (task set sha256 `{run.task_digest[:16]}`), "
        f"{_timing(run)}. "
        "Produced by `uv run python -m bench`; this is not a CI job and the numbers hold for the "
        "machine described below only."
    )

    out += ["", "## Summary", ""]
    missing = _coverage(run)
    if missing:
        out.append(
            "- Reference set INCOMPLETE. Not measured: " + ", ".join(missing) + ". "
            "A target that is not measured is not met."
        )
    else:
        out.append(
            "- Reference set complete: " + ", ".join(REFERENCE_MODELS) + " on both backends."
        )
    missed = missed_targets(run)
    if missed:
        out.append("- Targets MISSED in the measured cells:")
        out.extend(f"  - {line}" for line in missed)
    elif any(c.status == "ok" for c in run.cells):
        out.append("- Every target was met in the measured cells.")
    else:
        out.append("- No cell was measured.")
    cautions = length_cautions(run)
    if cautions:
        out.append("- Read with care (not a miss):")
        out.extend(f"  - {line}" for line in cautions)

    out += ["", "## Environment", "", _table(["Item", "Value"], run.environment.items())]
    out += [
        "",
        "Models:",
        "",
        _table(
            ["Model", "Backend", "Source", "Quantization", "Revision", "Status"], _models_rows(run)
        ),
        "",
        "The files are conversions made by third parties, not by the model publishers, and the "
        "quantization is not the same everywhere (the column above says what each file is). "
        "Source repositories change over time: the Revision column is the commit and the main "
        "file's SHA-256 of what was downloaded (from the downloader's manifest), so two reports "
        "can be told apart when a repository was updated in between. "
        "Differences between rows can come from the quantization or the converter as well as from "
        "the model, so read the ONNX rows of different models side by side with that in mind.",
    ]

    out += [
        "",
        "## Targets (PRD section 6)",
        "",
        _table(
            ["Metric", "Target", "How it is measured here"],
            [
                [
                    "Schema pass rate",
                    f">= {STRICT_PASS_TARGET:.0%} strict",
                    "Share of tasks whose output is valid. Raw: backend output, `json.loads`, "
                    "validation. Strict: the full ladder with local rungs only.",
                ],
                [
                    "Grammar path overhead",
                    f"<= {GRAMMAR_OVERHEAD_LIMIT:.0%}",
                    "llama.cpp only. First grammar-constrained call of a strict run against the "
                    "raw call of the same task, summed over tasks; a second figure adds the "
                    "grammar compile time, a third divides by the output length.",
                ],
                [
                    "Fallback rate",
                    f"< {FALLBACK_RATE_LIMIT:.0%}",
                    "Share of tasks whose strict run reached rung 4. The fallback is a stub; "
                    "nothing is sent to a cloud provider.",
                ],
            ],
        ),
    ]

    out += [
        "",
        "## Results",
        "",
        _table(
            [
                "Model",
                "Backend",
                "Tasks",
                "Raw pass",
                "Strict pass",
                "Grammar overhead (decoding)",
                "Grammar overhead (with build)",
                "Grammar overhead per output char",
                "Mean output chars (raw / grammar)",
                "Fallback rate",
                "Whole-ladder overhead",
                "Errors",
            ],
            _results_rows(run),
        ),
        "",
        "Whole-ladder overhead is the strict run's total time (grammar, repair, re-asks) against "
        "the raw run; it is informational and has no target. Errors are runs that ended in a "
        "backend or unexpected exception; they count as failed tasks.",
    ]

    paths = _path_rows(run)
    out += ["", "## Ladder path distribution (strict runs)", ""]
    out.append(_table(["Model", "Backend", "Path (rungs that took effect)", "Runs"], paths))

    categories, category_rows = _category_rows(run)
    out += ["", "## Valid outputs by category (raw -> strict)", ""]
    out.append(_table(["Model", "Backend", *categories], category_rows))

    out += [
        "",
        "## Integration tests",
        "",
        "The two real-model integration suites are not part of `python -m bench`. Run them in "
        "the same environment as the benchmark and record the result here (see "
        "`bench/README.md`):",
        "",
        "- `uv run --extra llamacpp pytest -m integration "
        "tests/backends/test_llamacpp_integration.py` (`SLMSTRICT_GGUF_PATH`)",
        "- `uv run --extra onnx pytest -m integration "
        "tests/backends/test_onnx_integration.py` (`SLMSTRICT_ONNX_MODEL_DIR`)",
        "",
        "Result: not recorded by `python -m bench`. `bench/docker/run.sh integration` runs the "
        "tests and replaces this section in the newest report of the same day with their results.",
        "",
        "## Notes",
        "",
        "- Only schema validity is measured. Values are not checked for correctness, so the "
        "calculation tasks do not show whether the arithmetic is right.",
        "- Each task runs once per mode, sequentially, after one warm-up generation. Timings are "
        "wall-clock seconds on a possibly busy machine.",
        "- Call order: llama.cpp reuses the KV-cache prefix of the previous call, so a raw call "
        "followed by the strict call of the same task would hand the strict call the prompt "
        "evaluation for free. The two runs of a task are staggered (raw 0, raw 1, strict 0, "
        "raw 2, strict 1, ...) so consecutive calls never share a prompt, and the grammar cache "
        "is emptied before every strict run, so each run pays for its own grammar compile.",
        "- A constrained and an unconstrained generation can differ in length, which moves the "
        "total-time comparison in either direction (a negative overhead is possible). The "
        "per-character column and the output lengths are there to judge that; the targets are "
        "checked against the total-time figures.",
        "- Both modes get the same prompt (task text plus the schema) wrapped in the model's "
        "`prompt_template` when one is configured.",
    ]
    return "\n".join(out) + "\n"


def write_report(run: BenchmarkRun, directory: Path) -> Path:
    """Write ``<date>.md`` into ``directory`` and return its path.

    An existing report is never overwritten: a second run on the same day goes to
    ``<date>-2.md``, then ``<date>-3.md`` and so on.
    """
    directory.mkdir(parents=True, exist_ok=True)
    date = run.started_at.date().isoformat()
    text = render_report(run)
    number = 1
    while True:
        name = f"{date}.md" if number == 1 else f"{date}-{number}.md"
        path = directory / name
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(text)
        except FileExistsError:
            number += 1
            continue
        return path
