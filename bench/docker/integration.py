"""Run the real-model integration tests once per model and format, then print a summary.

Meant for the benchmark container (``run.sh integration``): it reads the container's model
configuration, runs ``pytest -m integration`` for the llama.cpp and the ONNX test of every model
(each in its own process, so one model is in memory at a time) and writes the full output to
``bench/results/integration-<date>.log``. The exit code is 0 only when every run passed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
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


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    config_path = Path(args[0]) if args else DEFAULT_CONFIG
    runs = plan_runs(load_config(config_path))
    log = _log_path(ROOT / "bench" / "results")
    summary: list[str] = []
    failed = False
    with log.open("x", encoding="utf-8") as handle:
        for run in runs:
            env = {**os.environ, run.variable: str(run.location)}
            print(f"== {run.model} / {run.backend}", flush=True)
            handle.write(f"== {run.model} / {run.backend}\n")
            done = subprocess.run(
                run.command(), env=env, cwd=ROOT, capture_output=True, text=True, check=False
            )
            handle.write(done.stdout + done.stderr + "\n")
            tail = (done.stdout.strip().splitlines() or ["(no output)"])[-1]
            summary.append(f"{run.model} / {run.backend}: exit {done.returncode}: {tail}")
            failed = failed or done.returncode != 0
    print("\n".join(summary))
    print(f"log: {log}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
