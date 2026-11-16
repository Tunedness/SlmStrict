"""Command line entry point: ``uv run python -m bench``."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from .report import compute_stats, missed_targets, write_report
from .runner import (
    BACKENDS,
    BackendFactory,
    BenchError,
    default_backend_factory,
    load_config,
    load_tasks,
    run_benchmark,
)

_HERE = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        description="Run the SlmStrict reference benchmark and write a Markdown report.",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=_HERE / "models.toml",
        help="model configuration (default: bench/models.toml; see bench/models.toml.example)",
    )
    parser.add_argument(
        "--tasks", type=Path, default=_HERE / "tasks", help="task directory (default: bench/tasks)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_HERE / "results",
        help="directory for the report (default: bench/results)",
    )
    parser.add_argument(
        "--model", action="append", metavar="NAME", help="run only this model (repeatable)"
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=BACKENDS,
        help="run only this backend (repeatable)",
    )
    parser.add_argument(
        "--limit", type=int, help="use only the first N tasks, at least 2 (a quick trial)"
    )
    return parser


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None, *, factory: BackendFactory | None = None) -> int:
    """Run the benchmark. Exit code: 0 on success, 1 when a cell failed, 2 on bad input."""
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.models)
        tasks = load_tasks(args.tasks, limit=args.limit)
        if len(tasks) < 2:
            raise BenchError(
                "at least 2 tasks are needed so consecutive calls never share a prompt"
            )
        if args.model:
            known = {spec.name for spec in config.models}
            unknown = sorted(set(args.model) - known)
            if unknown:
                raise BenchError(f"unknown model(s): {', '.join(unknown)}")
    except BenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    run = asyncio.run(
        run_benchmark(
            config,
            tasks,
            models=args.model,
            backends=args.backend or BACKENDS,
            factory=factory or default_backend_factory,
            progress=_progress,
        )
    )
    path = write_report(run, args.output)

    for cell in run.cells:
        if cell.status == "ok":
            stats = compute_stats(cell.records)
            summary = (
                f"raw {stats.raw_pass_rate:.1%}, strict {stats.strict_pass_rate:.1%}, "
                f"fallback {stats.fallback_rate:.1%}"
            )
        else:
            summary = f"{cell.status}: {cell.detail}"
        print(f"{cell.model} / {cell.backend}: {summary}")
    for line in missed_targets(run):
        print(f"target missed: {line}")
    print(f"report: {path}")
    return 1 if any(cell.status == "failed" for cell in run.cells) else 0


if __name__ == "__main__":
    raise SystemExit(main())
