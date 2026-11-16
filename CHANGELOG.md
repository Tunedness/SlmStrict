# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First MVP, not released yet.

### Added

- `generate()` and `agenerate()`: one API in a synchronous and an asynchronous form. Both follow the same enforcement ladder and return a `Result` whose `path` lists the rungs that took effect.
- `Schema`: a Pydantic model or a JSON Schema as input, with structured validation issues and a stable schema hash.
- Enforcement ladder: grammar, deterministic repair (fence stripping, bracket completion, lossless type coercion), re-ask and cloud fallback.
- Backends: llama.cpp (`slmstrict[llamacpp]`) with GBNF grammar compiled from the schema and cached per process, and ONNX Runtime GenAI (`slmstrict[onnx]`) without a grammar rung. Any object that follows the `Backend` protocol works.
- Cloud fallback, off by default: `local_only` lock, a `Masker` hook (with the best-effort `RegexMasker` example), Anthropic, OpenAI and Gemini providers (`slmstrict[anthropic]`, `slmstrict[openai]`, `slmstrict[gemini]`) and meta-data-only logging unless content logging is switched on. `log_content=True` needs a masker (without one the constructor raises a configuration error); content may be logged unmasked only with an explicit `allow_unmasked_content_log=True`.
- Measurement: `MetricsCollector` (schema pass rate, path and repair distribution, fallback rate, latency histogram), cost estimation from a user-supplied price table (cost is unknown without one) and optional OTLP export (`slmstrict[otel]`).
- `slmstrict.testing.FakeBackend` for tests of code built on SlmStrict.
- Reference benchmark in `bench/`: 36 schema tasks in six categories, a runner that measures raw and strict output on Qwen3-4B, Llama-3.2-3B and Gemma-3-1B for both backends, and a Markdown report against the PRD targets. It needs real model files and is not run in CI; no result has been recorded yet. A Docker setup (`bench/docker/`) downloads the models into a volume and runs the benchmark and the integration tests with the network disabled.
- README with installation, quick start, the ladder, the privacy contract and measurement.

[Unreleased]: https://github.com/Tunedness/SlmStrict/commits/main
