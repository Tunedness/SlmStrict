"""Run the real-model integration tests once per model and format, then print a summary.

Meant for the benchmark container (``run.sh integration``): it reads the container's model
configuration, runs ``pytest -m integration`` for the llama.cpp and the ONNX test of every model
(each in its own process, so one model is in memory at a time) and writes the full output to
``bench/results/integration-<date>.log``. The exit code is 0 only when every run passed.

When a benchmark report of the same date exists (``bench/results/<date>.md``, or the newest
``<date>-N.md``), its ``Integration tests`` section is replaced with the results: exit code and
test counts per model and format. Without a report only the log is written. The report is a
Markdown file the benchmark wrote earlier in the same container run; only that one section is
changed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from bench.runner import BACKEND_LLAMACPP, BACKEND_ONNX, BenchConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("models.container.toml")

_TESTS = {
    BACKEND_LLAMACPP: ("SLMSTRICT_GGUF_PATH", "tests/backends/test_llamacpp_integration.py"),
    BACKEND_ONNX: ("SLMSTRICT_ONNX_MODEL_DIR", "tests/backends/test_onnx_integration.py"),
}


@dataclass(frozen=True, slots=True)
class IntegrationRun:
    model: str
    backend: str
    variable: str
    location: Path
    test_file: str

    def command(self) -> list[str]:
        # No cache directory: the repository is mounted read-only.
        return [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "integration",
            "-p",
            "no:cacheprovider",
            "-rA",
            self.test_file,
        ]


def plan_runs(config: BenchConfig) -> list[IntegrationRun]:
    runs: list[IntegrationRun] = []
    for spec in config.models:
        for backend, (variable, test_file) in _TESTS.items():
            location = spec.location(backend)
            if location is not None:
                runs.append(IntegrationRun(spec.name, backend, variable, location, test_file))
    return runs


def _log_path(directory: Path) -> Path:
    number = 1
    while True:
        suffix = "" if number == 1 else f"-{number}"
        path = directory / f"integration-{date.today().isoformat()}{suffix}.log"
        if not path.exists():
            return path
        number += 1


@dataclass(frozen=True, slots=True)
class RunResult:
    run: IntegrationRun
    returncode: int
    counts: str


_COUNT = re.compile(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|deselected)")


def parse_counts(output: str) -> str:
    """The test counts of pytest's closing line, for example ``2 passed``."""
    for line in reversed(output.strip().splitlines()):
        found = _COUNT.findall(line)
        if found and " in " in line:
            return ", ".join(f"{number} {word}" for number, word in found)
    return "no test summary"


def latest_report(directory: Path, day: date) -> Path | None:
    """The newest benchmark report of ``day``: ``<day>.md``, ``<day>-2.md``, ..."""
    pattern = re.compile(rf"{re.escape(day.isoformat())}(?:-(\d+))?\.md")
    found: list[tuple[int, Path]] = []
    for path in directory.glob(f"{day.isoformat()}*.md"):
        match = pattern.fullmatch(path.name)
        if match:
            found.append((int(match.group(1) or 1), path))
    return max(found)[1] if found else None


def render_section(results: Sequence[RunResult], log_name: str, finished: datetime) -> str:
    rows = [f"| {r.run.model} | {r.run.backend} | {r.returncode} | {r.counts} |" for r in results]
    return "\n".join(
        [
            "## Integration tests",
            "",
            "Run by `bench/docker/run.sh integration` in the benchmark container (the same model "
            f"files, network off), finished {finished.isoformat(timespec='seconds')}. The full "
            f"pytest output is in `{log_name}`.",
            "",
            "- `pytest -m integration tests/backends/test_llamacpp_integration.py` "
            "(`SLMSTRICT_GGUF_PATH`)",
            "- `pytest -m integration tests/backends/test_onnx_integration.py` "
            "(`SLMSTRICT_ONNX_MODEL_DIR`)",
            "",
            "| Model | Backend | Exit code | Tests |",
            "| --- | --- | --- | --- |",
            *rows,
            "",
        ]
    )


def update_report(report: Path, section: str) -> None:
    """Replace the ``Integration tests`` section of ``report`` (appended when it is missing)."""
    text = report.read_text(encoding="utf-8")
    pattern = re.compile(r"^## Integration tests\n.*?(?=^## |\Z)", re.DOTALL | re.MULTILINE)
    if pattern.search(text):
        updated = pattern.sub(lambda _match: section + "\n", text, count=1)
    else:
        updated = text.rstrip("\n") + "\n\n" + section
    temporary = report.with_name(report.name + ".part")
    temporary.write_text(updated, encoding="utf-8")
    os.replace(temporary, report)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    config_path = Path(args[0]) if args else DEFAULT_CONFIG
    runs = plan_runs(load_config(config_path))
    results_dir = ROOT / "bench" / "results"
    log = _log_path(results_dir)
    summary: list[str] = []
    results: list[RunResult] = []
    with log.open("x", encoding="utf-8") as handle:
        for run in runs:
            env = {**os.environ, run.variable: str(run.location)}
            print(f"== {run.model} / {run.backend}", flush=True)
            handle.write(f"== {run.model} / {run.backend}\n")
            done = subprocess.run(
                run.command(), env=env, cwd=ROOT, capture_output=True, text=True, check=False
            )
            handle.write(done.stdout + done.stderr + "\n")
            counts = parse_counts(done.stdout)
            results.append(RunResult(run, done.returncode, counts))
            summary.append(f"{run.model} / {run.backend}: exit {done.returncode}: {counts}")
    print("\n".join(summary))
    print(f"log: {log}")

    report = latest_report(results_dir, date.today())
    if report is None:
        print("report: none for today; only the log was written")
    else:
        section = render_section(results, log.name, datetime.now().astimezone())
        update_report(report, section)
        print(f"report: {report} (Integration tests section updated)")
    return 1 if any(r.returncode != 0 for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
