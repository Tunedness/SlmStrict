"""Benchmark runner: every model x backend x task, once raw and once through the ladder.

For each task two runs are made on the same backend with the same prompt:

* **raw**: the backend output, ``json.loads`` and schema validation. SlmStrict is not involved.
* **strict**: :func:`slmstrict.agenerate` with the full enforcement ladder.

The cloud fallback is a counting stub (:class:`CountingFallback`): it records that rung 4 was
reached and raises, so the benchmark never sends a request anywhere. A task whose strict run
reaches the fallback therefore counts as *not* resolved locally.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

from slmstrict import (
    Attempt,
    Backend,
    BackendError,
    Constraint,
    FallbackConfig,
    Result,
    Schema,
    SchemaError,
    StrictConfig,
    ValidationFailed,
    __version__,
    agenerate,
)
from slmstrict.backends._grammar import clear_grammar_cache
from slmstrict.metrics import MetricsCollector, MetricsSnapshot

__all__ = [
    "BACKENDS",
    "BACKEND_LLAMACPP",
    "BACKEND_ONNX",
    "REFERENCE_MODELS",
    "BackendFactory",
    "BenchConfig",
    "BenchError",
    "BenchmarkRun",
    "CallTiming",
    "CellResult",
    "CountingFallback",
    "FallbackReached",
    "MeasuredBackend",
    "ModelSpec",
    "Settings",
    "Task",
    "TaskRecord",
    "collect_environment",
    "default_backend_factory",
    "load_config",
    "load_tasks",
    "read_revision",
    "run_benchmark",
    "run_cell",
    "tasks_digest",
]

T = TypeVar("T")

BACKEND_LLAMACPP: Final = "llamacpp"
BACKEND_ONNX: Final = "onnx"
BACKENDS: Final = (BACKEND_LLAMACPP, BACKEND_ONNX)

# The reference model set of PRD section 6.
REFERENCE_MODELS: Final = ("Qwen3-4B", "Llama-3.2-3B", "Gemma-3-1B")

_WARMUP_PROMPT: Final = "Reply with a JSON object holding one key, ok, set to true."

RawStatus = Literal["valid", "invalid", "error"]
StrictStatus = Literal["valid", "fallback", "failed", "error"]
CellStatus = Literal["ok", "skipped", "failed"]


class BenchError(Exception):
    """The benchmark cannot start: bad task files or a bad model configuration."""


# -- tasks ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Task:
    """One benchmark task: a JSON Schema, a prompt and a category label."""

    id: str
    category: str
    prompt: str
    schema: Mapping[str, Any]

    def full_prompt(self) -> str:
        """The prompt both runs send: the task text plus the schema it must follow."""
        return (
            f"{self.prompt}\n\n"
            "Respond with a single JSON object that matches this JSON Schema, "
            "and nothing else:\n"
            f"{json.dumps(self.schema, separators=(',', ':'))}"
        )


def load_tasks(directory: Path, *, limit: int | None = None) -> list[Task]:
    """Load ``*.json`` task files from ``directory``, sorted by file name.

    Every schema is built once with :meth:`slmstrict.Schema.of`, so a task SlmStrict cannot use
    fails here and not in the middle of a run.
    """
    if not directory.is_dir():
        raise BenchError(f"task directory does not exist: {directory}")
    files = sorted(directory.glob("*.json"))
    if not files:
        raise BenchError(f"no task files (*.json) in {directory}")
    tasks: list[Task] = []
    seen: set[str] = set()
    prompts: dict[str, str] = {}
    for path in files:
        task = _read_task(path)
        if task.id in seen:
            raise BenchError(f"duplicate task id {task.id!r}")
        seen.add(task.id)
        if task.prompt in prompts:
            # Two tasks with one prompt would share a KV-cache prefix in llama.cpp (see run_cell).
            raise BenchError(f"{task.id!r} has the same prompt as {prompts[task.prompt]!r}")
        prompts[task.prompt] = task.id
        tasks.append(task)
    if limit is not None:
        if limit < 1:
            raise BenchError("limit must be at least 1")
        tasks = tasks[:limit]
    return tasks


def _read_task(path: Path) -> Task:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchError(f"{path.name}: cannot read task file ({type(exc).__name__})") from exc
    if not isinstance(data, dict):
        raise BenchError(f"{path.name}: a task must be a JSON object")
    fields: dict[str, str] = {}
    for key in ("id", "category", "prompt"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise BenchError(f"{path.name}: {key!r} must be a non-empty string")
        fields[key] = value
    schema = data.get("schema")
    if not isinstance(schema, dict):
        raise BenchError(f"{path.name}: 'schema' must be a JSON object")
    try:
        Schema.of(schema)
    except SchemaError as exc:
        raise BenchError(f"{path.name}: unusable schema: {exc}") from exc
    return Task(fields["id"], fields["category"], fields["prompt"], schema)


def tasks_digest(tasks: Sequence[Task]) -> str:
    """SHA-256 over the task set, so a report says exactly which tasks it measured."""
    canonical = json.dumps(
        [[t.id, t.category, t.prompt, t.schema] for t in tasks],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -- configuration ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Settings:
    """Generation and ladder settings shared by every cell of a run."""

    max_tokens: int = 512
    max_length: int = 2048
    temperature: float = 0.0
    seed: int = 1
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    max_reask: int = 2
    warmup: bool = True


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model: where its GGUF file and its ONNX GenAI directory are."""

    name: str
    gguf_path: Path | None = None
    gguf_quantization: str | None = None
    onnx_dir: Path | None = None
    onnx_quantization: str | None = None
    prompt_template: str | None = None
    gguf_source: str | None = None
    onnx_source: str | None = None

    def location(self, backend: str) -> Path | None:
        return self.gguf_path if backend == BACKEND_LLAMACPP else self.onnx_dir

    def quantization(self, backend: str) -> str | None:
        return self.gguf_quantization if backend == BACKEND_LLAMACPP else self.onnx_quantization

    def source(self, backend: str) -> str | None:
        """Where the files come from (a repository id), as the config states it."""
        return self.gguf_source if backend == BACKEND_LLAMACPP else self.onnx_source


@dataclass(frozen=True, slots=True)
class BenchConfig:
    settings: Settings
    models: tuple[ModelSpec, ...]


_SETTINGS_FIELDS: Final = frozenset(Settings.__dataclass_fields__)
_MODEL_FIELDS: Final = frozenset(ModelSpec.__dataclass_fields__)


def load_config(path: Path) -> BenchConfig:
    """Read a ``models.toml`` file (see ``bench/models.toml.example``)."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchError(f"cannot read {path}: {exc.strerror or type(exc).__name__}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise BenchError(f"{path}: invalid TOML: {exc}") from exc
    unknown = set(data) - {"settings", "models"}
    if unknown:
        raise BenchError(f"{path}: unknown top-level key(s): {', '.join(sorted(unknown))}")
    settings = _parse_settings(data.get("settings", {}), path)
    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise BenchError(f"{path}: at least one [[models]] table is required")
    models: list[ModelSpec] = []
    for index, table in enumerate(raw_models):
        spec = _parse_model(table, index, path.parent)
        if any(spec.name == other.name for other in models):
            raise BenchError(f"{path}: duplicate model name {spec.name!r}")
        models.append(spec)
    return BenchConfig(settings, tuple(models))


def _parse_settings(table: object, path: Path) -> Settings:
    if not isinstance(table, dict):
        raise BenchError(f"{path}: [settings] must be a table")
    unknown = set(table) - _SETTINGS_FIELDS
    if unknown:
        raise BenchError(f"{path}: unknown [settings] key(s): {', '.join(sorted(unknown))}")
    defaults = Settings()
    values: dict[str, Any] = {}
    for name in _SETTINGS_FIELDS:
        default = getattr(defaults, name)
        value = table.get(name, default)
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, float):
            ok = isinstance(value, int | float) and not isinstance(value, bool)
            value = float(value) if ok else value
        else:
            ok = isinstance(value, int) and not isinstance(value, bool)
        if not ok:
            raise BenchError(f"{path}: [settings] {name} has the wrong type")
        values[name] = value
    settings = Settings(**values)
    for name in ("max_tokens", "max_length", "n_ctx"):
        if getattr(settings, name) < 1:
            raise BenchError(f"{path}: [settings] {name} must be at least 1")
    if settings.temperature < 0 or settings.n_gpu_layers < 0 or settings.max_reask < 0:
        raise BenchError(
            f"{path}: [settings] temperature, n_gpu_layers and max_reask must not be negative"
        )
    return settings


def _parse_model(table: object, index: int, base: Path) -> ModelSpec:
    where = f"[[models]] #{index + 1}"
    if not isinstance(table, dict):
        raise BenchError(f"{where} must be a table")
    unknown = set(table) - _MODEL_FIELDS
    if unknown:
        raise BenchError(f"{where}: unknown key(s): {', '.join(sorted(unknown))}")
    name = table.get("name")
    if not isinstance(name, str) or not name.strip():
        raise BenchError(f"{where}: 'name' must be a non-empty string")
    strings: dict[str, str | None] = {}
    for key in (
        "gguf_path",
        "gguf_quantization",
        "onnx_dir",
        "onnx_quantization",
        "prompt_template",
        "gguf_source",
        "onnx_source",
    ):
        value = table.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise BenchError(f"{where} ({name}): {key!r} must be a non-empty string")
        strings[key] = value
    template = strings["prompt_template"]
    if template is not None and "{prompt}" not in template:
        raise BenchError(f"{where} ({name}): 'prompt_template' must contain {{prompt}}")
    return ModelSpec(
        name=name,
        gguf_path=_resolve(strings["gguf_path"], base),
        gguf_quantization=strings["gguf_quantization"],
        onnx_dir=_resolve(strings["onnx_dir"], base),
        onnx_quantization=strings["onnx_quantization"],
        prompt_template=template,
        gguf_source=strings["gguf_source"],
        onnx_source=strings["onnx_source"],
    )


def _resolve(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


# -- backends ---------------------------------------------------------------------------------

BackendFactory = Callable[[ModelSpec, str, Settings], Backend]


def default_backend_factory(spec: ModelSpec, backend: str, settings: Settings) -> Backend:
    """Build the real backend for ``spec``; needs the matching extra and the model files."""
    location = spec.location(backend)
    if location is None:
        raise BackendError(f"no {backend} location configured for {spec.name}")
    if backend == BACKEND_LLAMACPP:
        from slmstrict.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend(
            location,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            seed=settings.seed,
            n_ctx=settings.n_ctx,
            n_gpu_layers=settings.n_gpu_layers,
        )
    if backend == BACKEND_ONNX:
        from slmstrict.backends.onnx import OnnxBackend

        return OnnxBackend(
            location, max_length=settings.max_length, temperature=settings.temperature
        )
    raise BackendError(f"unknown backend {backend!r}")


@dataclass(frozen=True, slots=True)
class CallTiming:
    """Duration of one backend call."""

    constrained: bool
    seconds: float
    chars: int = 0


class MeasuredBackend:
    """Wraps a backend: applies the model's prompt template and times every call.

    The wrapper offers ``build_constraint`` like a grammar-capable adapter does, and times it, so
    the cost of compiling a grammar is visible separately from the decoding.
    """

    def __init__(self, inner: Backend, template: str | None = None) -> None:
        self._inner = inner
        self._template = template
        self.calls: list[CallTiming] = []
        self.build_seconds = 0.0

    @property
    def supports_grammar(self) -> bool:
        return self._inner.supports_grammar

    def reset(self) -> None:
        self.calls = []
        self.build_seconds = 0.0

    def build_constraint(self, schema: Schema[Any]) -> Constraint:
        builder = getattr(self._inner, "build_constraint", None)
        if builder is None:
            raise BackendError("the backend cannot build a grammar constraint")
        started = time.perf_counter()
        try:
            constraint: Constraint = builder(schema)
        finally:
            self.build_seconds += time.perf_counter() - started
        return constraint

    async def agenerate(self, prompt: str, *, constraint: Constraint | None = None) -> str:
        text = prompt if self._template is None else self._template.replace("{prompt}", prompt)
        started = time.perf_counter()
        output = ""
        try:
            output = await self._inner.agenerate(text, constraint=constraint)
            return output
        finally:
            self.calls.append(
                CallTiming(constraint is not None, time.perf_counter() - started, len(output))
            )


class FallbackReached(Exception):  # noqa: N818 - a signal, not an error condition
    """Raised by :class:`CountingFallback`: the ladder got to rung 4."""


class CountingFallback:
    """Stand-in for the cloud fallback: counts the call and refuses. Nothing is sent anywhere."""

    def __init__(self) -> None:
        self.calls = 0

    async def handle(
        self, schema: Schema[T], prompt: str, attempts: Sequence[Attempt]
    ) -> Result[T]:
        self.calls += 1
        raise FallbackReached("the ladder reached the cloud fallback (stub: no request is made)")


# -- results ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """What happened for one task in one cell.

    ``grammar_decode_seconds`` is the duration of the first strict backend call when it was
    grammar-constrained (``None`` otherwise); ``grammar_build_seconds`` is the time spent
    compiling the grammar for that run. ``raw_chars`` and ``grammar_chars`` are the lengths of the
    raw call's output and of that first grammar-constrained call's output (0 when there is none),
    so the latency comparison can be read against how much each call produced. ``error_type`` is
    an exception class name, never content.
    """

    task_id: str
    category: str
    raw_status: RawStatus
    raw_seconds: float
    strict_status: StrictStatus
    strict_seconds: float
    grammar_decode_seconds: float | None = None
    grammar_build_seconds: float = 0.0
    error_type: str | None = None
    raw_chars: int = 0
    grammar_chars: int = 0


@dataclass(frozen=True, slots=True)
class CellResult:
    """One model x backend cell. ``detail`` explains a skipped or failed cell."""

    model: str
    backend: str
    quantization: str | None
    status: CellStatus
    detail: str | None = None
    records: tuple[TaskRecord, ...] = ()
    metrics: MetricsSnapshot | None = None
    source: str | None = None
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """A finished run. ``started_at`` is taken when the run begins, ``finished_at`` and
    ``duration_s`` (measured with a monotonic clock) when its last cell is done."""

    started_at: datetime
    environment: Mapping[str, str]
    settings: Settings
    task_count: int
    task_digest: str
    cells: tuple[CellResult, ...]
    finished_at: datetime | None = None
    duration_s: float | None = None


def _is_valid_raw(schema: Schema[Any], text: str) -> bool:
    try:
        parsed = json.loads(text)
    except ValueError:
        return False
    return schema.validate(parsed).ok


@dataclass(frozen=True, slots=True)
class _RawRun:
    status: RawStatus
    seconds: float
    chars: int
    error_type: str | None


@dataclass(frozen=True, slots=True)
class _StrictRun:
    status: StrictStatus
    seconds: float
    error_type: str | None
    decode_seconds: float | None
    decode_chars: int
    build_seconds: float


async def _run_raw(backend: MeasuredBackend, schema: Schema[Any], prompt: str) -> _RawRun:
    backend.reset()
    started = time.perf_counter()
    try:
        text = await backend.agenerate(prompt)
    except Exception as exc:
        return _RawRun("error", time.perf_counter() - started, 0, type(exc).__name__)
    seconds = time.perf_counter() - started
    status: RawStatus = "valid" if _is_valid_raw(schema, text) else "invalid"
    return _RawRun(status, seconds, len(text), None)


async def _run_strict(
    backend: MeasuredBackend, schema: Schema[Any], prompt: str, config: StrictConfig
) -> _StrictRun:
    backend.reset()
    started = time.perf_counter()
    status: StrictStatus
    error_type: str | None = None
    try:
        await agenerate(backend, schema, prompt, config=config)
        status = "valid"
    except FallbackReached:
        status = "fallback"
    except ValidationFailed:
        status = "failed"
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
    seconds = time.perf_counter() - started
    first = backend.calls[0] if backend.calls and backend.calls[0].constrained else None
    return _StrictRun(
        status,
        seconds,
        error_type,
        first.seconds if first is not None else None,
        first.chars if first is not None else 0,
        backend.build_seconds if first is not None else 0.0,
    )


async def run_cell(
    model: str,
    backend_name: str,
    quantization: str | None,
    backend: Backend,
    tasks: Sequence[Task],
    settings: Settings,
    *,
    prompt_template: str | None = None,
    cold_grammar: bool = False,
    source: str | None = None,
    revision: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> CellResult:
    """Run every task on ``backend``: once raw, once through the ladder.

    Tasks run one after the other so the timings are not distorted by concurrency.

    **Call order.** llama.cpp keeps its KV cache between calls and reuses the longest prefix a new
    prompt shares with it. A raw call followed directly by the strict call of the *same* task
    would give the strict call the whole prompt for free and make the grammar path look faster
    than it is. The two runs of a task are therefore staggered: the order is ``raw 0, raw 1,
    strict 0, raw 2, strict 1, ...``, so two consecutive backend calls never come from the same
    task and never share a prompt (task prompts are unique, see :func:`load_tasks`). Raw and
    strict runs alternate over the whole run, which also spreads thermal drift evenly. A single
    task cannot be staggered; use at least two for a latency comparison.

    ``cold_grammar`` empties the process-wide grammar cache before every strict run, so each run
    pays for compiling its grammar and the figure does not depend on which model ran first.
    """
    measured = MeasuredBackend(backend, prompt_template)
    collector = MetricsCollector()
    fallback = CountingFallback()
    config = StrictConfig(
        max_reask=settings.max_reask,
        fallback=FallbackConfig(handler=fallback),
        observer=collector,
    )
    if settings.warmup:
        try:
            await measured.agenerate(_WARMUP_PROMPT)
        except Exception as exc:
            detail = f"warm-up generation failed ({type(exc).__name__})"
            return CellResult(
                model,
                backend_name,
                quantization,
                "failed",
                detail,
                source=source,
                revision=revision,
            )

    schemas = [Schema.of(task.schema) for task in tasks]
    raw_runs: dict[int, _RawRun] = {}
    strict_runs: dict[int, _StrictRun] = {}
    total = len(tasks)
    for position in range(total + 1):
        if position < total:
            _announce(progress, model, backend_name, "raw", position, total, tasks[position])
            raw_runs[position] = await _run_raw(
                measured, schemas[position], tasks[position].full_prompt()
            )
        index = position - 1
        if index >= 0:
            _announce(progress, model, backend_name, "strict", index, total, tasks[index])
            if cold_grammar:
                clear_grammar_cache()
            strict_runs[index] = await _run_strict(
                measured, schemas[index], tasks[index].full_prompt(), config
            )

    records = tuple(
        TaskRecord(
            task_id=task.id,
            category=task.category,
            raw_status=raw_runs[i].status,
            raw_seconds=raw_runs[i].seconds,
            strict_status=strict_runs[i].status,
            strict_seconds=strict_runs[i].seconds,
            grammar_decode_seconds=strict_runs[i].decode_seconds,
            grammar_build_seconds=strict_runs[i].build_seconds,
            error_type=strict_runs[i].error_type or raw_runs[i].error_type,
            raw_chars=raw_runs[i].chars,
            grammar_chars=strict_runs[i].decode_chars,
        )
        for i, task in enumerate(tasks)
    )
    return CellResult(
        model,
        backend_name,
        quantization,
        "ok",
        None,
        records,
        collector.snapshot(),
        source,
        revision,
    )


def _announce(
    progress: Callable[[str], None] | None,
    model: str,
    backend_name: str,
    mode: str,
    index: int,
    total: int,
    task: Task,
) -> None:
    if progress is not None:
        progress(f"[{model} / {backend_name}] {mode} {index + 1}/{total}: {task.id}")


async def run_benchmark(
    config: BenchConfig,
    tasks: Sequence[Task],
    *,
    models: Sequence[str] | None = None,
    backends: Sequence[str] = BACKENDS,
    factory: BackendFactory = default_backend_factory,
    progress: Callable[[str], None] | None = None,
) -> BenchmarkRun:
    """Run the benchmark for every configured model on every requested backend.

    A model that has no location for a backend is recorded as skipped; a backend that cannot be
    built (missing extra, bad path) is recorded as failed. Neither stops the other cells.
    """
    if len(tasks) < 2:
        raise BenchError("at least 2 tasks are needed so consecutive calls never share a prompt")
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    selected = [spec for spec in config.models if models is None or spec.name in models]
    cells: list[CellResult] = []
    for spec in selected:
        for backend_name in backends:
            cell = await _run_model_on_backend(spec, backend_name, config, tasks, factory, progress)
            cells.append(cell)
    return BenchmarkRun(
        started_at=started_at,
        environment=collect_environment(config.settings),
        settings=config.settings,
        task_count=len(tasks),
        task_digest=tasks_digest(tasks),
        cells=tuple(cells),
        finished_at=datetime.now().astimezone(),
        duration_s=time.perf_counter() - started,
    )


async def _run_model_on_backend(
    spec: ModelSpec,
    backend_name: str,
    config: BenchConfig,
    tasks: Sequence[Task],
    factory: BackendFactory,
    progress: Callable[[str], None] | None,
) -> CellResult:
    quantization = spec.quantization(backend_name)
    source = spec.source(backend_name)
    revision = read_revision(spec.location(backend_name), backend_name)
    if spec.location(backend_name) is None:
        detail = f"no {backend_name} location configured for this model"
        return CellResult(
            spec.name,
            backend_name,
            quantization,
            "skipped",
            detail,
            source=source,
            revision=revision,
        )
    try:
        backend = factory(spec, backend_name, config.settings)
    except (BackendError, ImportError, ValueError) as exc:
        detail = f"could not start the backend: {type(exc).__name__}: {exc}"
        return CellResult(
            spec.name,
            backend_name,
            quantization,
            "failed",
            detail,
            source=source,
            revision=revision,
        )
    try:
        return await run_cell(
            spec.name,
            backend_name,
            quantization,
            backend,
            tasks,
            config.settings,
            prompt_template=spec.prompt_template,
            cold_grammar=backend_name == BACKEND_LLAMACPP,
            source=source,
            revision=revision,
            progress=progress,
        )
    finally:
        del backend
        gc.collect()  # release the model weights before the next cell loads its own


def read_revision(location: Path | None, backend: str) -> str | None:
    """The commit and the main file's checksum of the downloaded model, from its manifest.

    ``bench/docker/download_models.py`` writes ``<model>/manifest.json`` next to the ``gguf`` and
    ``onnx`` folders. A model that was put there another way has no manifest, and the report then
    says the revision was not recorded; anything unreadable counts as absent.
    """
    if location is None:
        return None
    kind, root = (
        ("gguf", location.parent.parent)
        if backend == BACKEND_LLAMACPP
        else (
            "onnx",
            location.parent,
        )
    )
    try:
        entry = json.loads((root / "manifest.json").read_text(encoding="utf-8"))[kind]
        commit = entry["commit"]
        largest = max(entry["files"], key=lambda item: item["size"])
        text = f"commit {commit[:12]}, {largest['path']} sha256 {largest['sha256'][:16]}"
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return text


# -- environment ------------------------------------------------------------------------------


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


BUILD_FILE_ENV: Final = "SLMSTRICT_LLAMA_BUILD_FILE"


def _llama_build_flags() -> str:
    """The CMake arguments llama.cpp was built with, when the image recorded them.

    ``bench/docker/Dockerfile`` writes them to a file and names it in the environment variable
    ``SLMSTRICT_LLAMA_BUILD_FILE``.
    """
    name = os.environ.get(BUILD_FILE_ENV)
    if not name:
        return "not recorded (llama.cpp was not built by bench/docker)"
    try:
        return Path(name).read_text(encoding="utf-8").strip() or "not recorded (empty file)"
    except OSError:
        return "not recorded (build file unreadable)"


def _llama_system_info() -> str:
    """What the loaded llama.cpp reports about itself (CPU features it was built with).

    ``llama_print_system_info`` works without loading a model. It is the evidence that a build
    flag took effect: a baseline that was silently ignored shows up as a missing feature.
    """
    try:
        module = importlib.import_module("llama_cpp")
        info = module.llama_print_system_info()
    except Exception:
        return "not available (llama-cpp-python is not usable here)"
    text = info.decode("utf-8", "replace") if isinstance(info, bytes) else str(info)
    return " ".join(text.split()) or "not available (empty)"


def collect_environment(settings: Settings) -> dict[str, str]:
    """Hardware, OS and library versions of this run (written at the top of the report)."""
    return {
        "OS": platform.platform(),
        "Machine": f"{platform.machine()} ({platform.processor() or 'unknown processor'})",
        "CPU count": str(os.cpu_count()),
        "Python": sys.version.split()[0],
        "slmstrict": __version__,
        "llama-cpp-python": _package_version("llama-cpp-python"),
        "llama.cpp build flags": _llama_build_flags(),
        "llama.cpp system info": _llama_system_info(),
        "onnxruntime-genai": _package_version("onnxruntime-genai"),
        "max_tokens / max_length": f"{settings.max_tokens} / {settings.max_length}",
        "temperature / seed": f"{settings.temperature} / {settings.seed}",
        "n_ctx / n_gpu_layers": f"{settings.n_ctx} / {settings.n_gpu_layers}",
        "max_reask": str(settings.max_reask),
    }
