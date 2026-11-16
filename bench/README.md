# SlmStrict reference benchmark

The benchmark measures the success metrics of the PRD (section 6) on a reference set of small models, on both backends, and writes the result as a Markdown report. It needs real model files and is not part of CI. The supported way to run it is the Docker setup in `bench/docker/`.

No benchmark result has been recorded yet. `bench/results/` holds only a `.gitkeep` until someone runs it.

## What it measures

For every model, every backend (llama.cpp, ONNX Runtime GenAI) and every task, the runner makes two runs with the same prompt:

- **raw**: the backend output, then `json.loads` and schema validation. SlmStrict is not involved.
- **strict**: `slmstrict.agenerate` with the whole enforcement ladder (grammar where the backend has it, repair, re-ask, fallback).

The report then shows, per cell:

| Metric | PRD target | How it is computed |
| --- | --- | --- |
| Raw pass rate | none (the baseline) | Valid raw outputs / tasks |
| Strict pass rate | >= 95% | Tasks the local rungs resolved / tasks |
| Grammar overhead | <= 15% | llama.cpp only. The first grammar-constrained backend call of a strict run against the raw call of the same task, summed over all tasks. A second figure adds the grammar compile time. A third column divides by the output length (see below); it has no target. |
| Fallback rate | < 10% | Tasks whose strict run reached rung 4 / tasks |
| Path distribution | none | How many strict runs ended on which combination of rungs |

The cloud fallback is a stub that counts the call and refuses. **The benchmark never sends a request to a cloud provider**, so a task that reaches the fallback is counted as not resolved locally. Strict pass rate and fallback rate are therefore two views of the same split: a run that fails locally reaches the fallback.

### How the latency comparison is kept fair

llama.cpp keeps its KV cache between calls and reuses the longest prefix a new prompt shares with it. If the strict call of a task ran right after the raw call of the same task, it would get the prompt evaluation for free and the grammar path would look faster than it is. Two measures prevent that:

- **Staggered order.** The runs of the tasks are interleaved as `raw 0, raw 1, strict 0, raw 2, strict 1, ...`, so two consecutive backend calls never come from the same task and never share a prompt (task prompts must be unique; the loader enforces it). Raw and strict runs also alternate over the whole run, which spreads thermal drift evenly. At least two tasks are required for this; `--limit 1` is rejected.
- **Cold grammar.** The process-wide grammar cache is emptied before every strict run on llama.cpp, so every run pays for compiling its own grammar. Without this, the first model of a run would carry the compile cost and the later ones would not.

The ONNX backend builds a fresh generator per call and is not affected by the first measure.

A constrained and an unconstrained generation can also differ in length: a raw run may add prose or a fence and go on until `max_tokens`, while a grammar run stops at the closing brace. The report therefore adds the mean output length of both and a per-character overhead (seconds per output character, grammar call against raw call). The targets are checked against the total-time figures, as the PRD words them; when the total-time figure meets the limit but the per-character one does not, the summary says so under "Read with care".

Only schema validity is checked. The values are not compared against an expected answer, so the calculation tasks do not tell you whether the arithmetic was right.

## Task set

`bench/tasks/` holds 36 tasks as JSON files, six per category: `flat_object`, `nested_object`, `array`, `enum`, `numeric` (including calculations) and `optional` (fields the text does not provide must be left out). Each file has `id`, `category`, `prompt` and a JSON Schema in `schema`. The prompt both runs send is the task text followed by the schema.

The report carries the first 16 hex digits of a SHA-256 over the whole task set, so a report says which tasks it measured. To add a task, drop a file with those four keys into the directory; the loader rejects a schema SlmStrict cannot use.

## Run in a container (recommended)

The benchmark and the integration tests are meant to run in a Docker container, so the native llama.cpp build, the model files and the network access stay out of your host environment. `bench/docker/run.sh` drives it in two stages. Requirements: Docker with Linux containers (arm64 or amd64), about 14 GB of free disk for the models plus a few GB for the image, and a Docker VM with more than 6 GB of memory (the run stage is limited to 6 GB by default).

```bash
bench/docker/run.sh build        # once; compiles llama.cpp for this machine's CPU
bench/docker/run.sh download     # stage 1: models into the named volume (network on)
bench/docker/run.sh run          # stage 2: benchmark, then both integration tests (network off)
```

- **build.** `docker build -f bench/docker/Dockerfile`. Python 3.12, `uv sync --locked --extra llamacpp --extra onnx`, CPU-only llama.cpp built from source. The base image is pinned by digest and `uv` by version, the image runs as a non-root user by default, and it holds the repository and the dependencies, never model weights. Build it on the machine that runs the benchmark: llama.cpp is compiled for the CPU it is built on.
- **download (stage 1).** The only stage with network access, and it runs nothing but `bench/docker/download_models.py`. It writes the models to the named volume `slmstrict-models` (`<model>/gguf/`, `<model>/onnx/`, `<model>/manifest.json`). It is safe to repeat: finished files are kept, interrupted ones resume, every large file is checked against the SHA-256 the Hub publishes and every small one (configs, tokenizer files) against its git blob id. The commit that was resolved and the checksums are written to `<model>/manifest.json`, and the benchmark report's `Models` table shows the commit and the main file's checksum (`Revision`), because the source repositories float on `main` and can change between two runs. Useful arguments: `--only NAME`, `--verify` (re-hash existing files), `--list`.
- **run (stage 2).** `--network none`; the repository is mounted read-only, the models volume is read-only, and only `bench/results/` is writable (plus a 256 MB tmpfs). Memory is capped at 6 GB with swap disabled (`SLMSTRICT_BENCH_MEMORY` changes it), capabilities are dropped, and the container runs as your user so the reports belong to you. The benchmark loads the models one after another and releases each before the next. `bench` and `integration` run the two parts separately; `bench` accepts the options of `python -m bench` (for example `--limit 4 --model Gemma-3-1B`).
- **Results.** `bench/results/<date>.md` (the report) and `bench/results/integration-<date>.log` (full pytest output of the six integration runs: llama.cpp and ONNX for each model). Paste the integration summary into the report's `Integration tests` section.

### The models

`bench/docker/models.download.json` lists where each model comes from; edit it to use other files. `bench/docker/models.container.toml` holds the matching paths and the prompt templates (keep them in step with `bench/models.toml.example`; a test checks that).

| Model | GGUF (Q4_K_M) | ONNX Runtime GenAI (int4, CPU) |
| --- | --- | --- |
| Qwen3-4B | `Qwen/Qwen3-4B-GGUF` | `onnx-community/Qwen3-4B-ONNX`, directory `onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128` |
| Llama-3.2-3B | `bartowski/Llama-3.2-3B-Instruct-GGUF` | `onnx-community/Llama-3.2-3B-Instruct-GENAI-ONNX`, directory `cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4` |
| Gemma-3-1B | `ggml-org/gemma-3-1b-it-GGUF` | `Arm/gemma-3-1b-instruct-onnx-genai-int4-emb-int8` |

What to know about them (checked against the Hub's repository metadata, not by running the models):

- **Third-party conversions, different quantizations.** The official `meta-llama/Llama-3.2-3B-Instruct` and `google/gemma-3-1b-it` repositories are gated (manual approval) and hold only Transformers weights, and no official ONNX GenAI build of `Qwen/Qwen3-4B` was found. Every GGUF and ONNX file above is a community or vendor conversion (`onnx-community` for Qwen3 and Llama ONNX, `Arm` for Gemma ONNX, `Qwen`, `bartowski` and `ggml-org` for the GGUF files), so the benchmark measures those conversions and their quantizations, not the publishers' own builds. The GGUF files are all Q4_K_M. The ONNX files are not quantized alike: Qwen3 is int4 with `kld` block-128 quantization, Llama is int4 `rtn` block-32 (accuracy level 4), Gemma is int4 with int8 embeddings. Differences between the ONNX rows of different models can therefore come from the quantization or the converter and not only from the model. The report's `Models` table lists the source repository and the quantization of every cell (taken from `models.container.toml`) and says so in a note; keep those fields correct when you change the files.
- **Gating.** At the time of writing the Hub metadata reports none of the repositories above as gated, so the download works without a token. The Llama ONNX repository carries the Llama 3.2 community license and one of its pages showed an access form despite the metadata; if the download answers `401` or `403`, it stops with a message that names the repository and says that a token is needed: accept the license on the model page while logged in, create a read token and run `HF_TOKEN=hf_... bench/docker/run.sh download`. The Gemma files fall under the Gemma terms of use. The token is read from `HF_TOKEN` only, sent to the Hub host only (it is dropped on redirects to a CDN) and never printed or stored.
- **Qwen3-4B is the hybrid-thinking model.** The template in the config closes the thinking block (`<think>` and `</think>`), so it answers directly.
- **Library support.** The ONNX configs declare the model types `qwen3`, `llama` and `gemma3_text`. The locked `onnxruntime-genai` is 0.19.0; the first real run shows whether it loads all three, and a model it cannot load is reported as a failed cell rather than stopping the run.
- **If a ready-made ONNX build is missing or unsuitable**, `python -m onnxruntime_genai.models.builder -m <model> -o <dir> -p int4 -e cpu` converts the original weights. It downloads the full-precision weights (gated for Meta and Google, so it needs the token and an accepted license) and needs working memory on the order of the fp32 size of the model (roughly 16 GB for a 4B model, an estimate), more than an 8 GB VM offers; run it on a larger machine and put the output under `<model>/onnx/` in the volume.

## Set up on the host

Without Docker, set the benchmark up on your machine:

1. Get the model files yourself; weights never go into the repository (`*.gguf`, `*.onnx` and `/models/` are git-ignored):
   - Qwen3-4B, Llama-3.2-3B and Gemma-3-1B, instruction-tuned variants.
   - For llama.cpp a GGUF file per model; for ONNX Runtime GenAI a model directory that contains `genai_config.json`.
2. Copy the example config and fill in the paths:

   ```bash
   cp bench/models.toml.example bench/models.toml
   ```

   `bench/models.toml` is git-ignored. Besides the paths it holds the generation settings (`temperature`, `seed`, `max_tokens`, ...) and a `prompt_template` per model. The backends take plain text and apply no chat template, so each model's chat format is written into its `prompt_template`. The templates in the example follow the published formats of the three model families; check them against the model card of the exact files you use, because a wrong template lowers the raw pass rate and distorts the comparison. Name the quantization of each file (`gguf_quantization`, `onnx_quantization`); it is printed in the report.
3. Install the extras of the backends you run:

   ```bash
   uv sync --extra llamacpp --extra onnx
   ```

   `llama-cpp-python` compiles native code on install. For GPU offload build it with the flags for your platform and set `n_gpu_layers`.

A model that has no path for a backend is reported as "not run" for that cell. The report says so at the top when the reference set (three models, two backends) is incomplete.

## Run on the host

From the repository root:

```bash
uv run --extra llamacpp --extra onnx python -m bench
```

Options:

| Option | Meaning |
| --- | --- |
| `--models PATH` | Config file (default `bench/models.toml`) |
| `--tasks DIR` | Task directory (default `bench/tasks`) |
| `--output DIR` | Report directory (default `bench/results`) |
| `--model NAME` | Run only this model; repeatable |
| `--backend NAME` | `llamacpp` or `onnx`; repeatable |
| `--limit N` | Use only the first N tasks (at least 2), for a quick trial |

A trial that checks your paths and templates before the full run:

```bash
uv run --extra llamacpp python -m bench --model Gemma-3-1B --backend llamacpp --limit 4
```

Progress goes to stderr, one line per task. At the end the command prints one line per cell, every missed target and the report path. The exit code is `0` when all cells ran, `1` when a backend could not be started (missing extra, bad path, failed warm-up) and `2` for bad input (config, tasks, unknown model). A missed target does not change the exit code: the benchmark succeeds by measuring honestly.

The report is `bench/results/<date>.md`. A second run on the same day writes `<date>-2.md`; an existing report is never overwritten.

## Reading the report

The report starts with a summary: whether the reference set is complete and which targets were missed, then the environment (hardware, OS, library versions, settings, quantization). Do not strip that part when you share a result; a number without its machine means little. Targets that were not met are marked **MISSED** in the table and repeated in the summary. They are not a failure of the phase; they are the finding.

Latency caveats:

- One run per task and mode, sequentially, after one untimed warm-up generation per cell.
- Output lengths can differ between the two modes, which moves the total-time comparison in either direction. A negative overhead is possible; read it together with the per-character column.
- Close other heavy programs, keep the machine on power and do not compare reports from different machines.

To compare runs, keep `temperature = 0.0` and the `seed`, and record the library versions the report prints.

## Integration tests on the host

Two real-model integration tests belong to the same environment. They are excluded from the default `pytest` run and are skipped when their variable is unset. Run them with the same model files as the benchmark, and paste the result into the `Integration tests` section of the report:

```bash
SLMSTRICT_GGUF_PATH=/path/to/model.gguf \
  uv run --extra llamacpp pytest -m integration tests/backends/test_llamacpp_integration.py

SLMSTRICT_ONNX_MODEL_DIR=/path/to/onnx-model-dir \
  uv run --extra onnx pytest -m integration tests/backends/test_onnx_integration.py
```

Expected: `1 passed` for the llama.cpp file and `2 passed` for the ONNX file. A `skipped` result means the variable was not set. These tests are one model each; run them for the model whose paths you use.

## Smoke test

`tests/test_bench_smoke.py` runs the benchmark end to end with scripted `FakeBackend` outputs and two tasks, checks the task set and the example config, and executes the README examples. It is part of the normal test run and needs no model:

```bash
uv run pytest tests/test_bench_smoke.py tests/test_bench_docker.py
```

`tests/test_bench_docker.py` covers the container tooling without Docker: the downloader runs against a fake Hub on the loopback interface (layout, hash check, resume, token handling), the config files are checked against each other, and `run.sh` is syntax-checked and read for its isolation flags. Nothing in the test suite builds an image or starts a container.

## If the benchmark finds a bug

A bug in the library that the benchmark exposes is not fixed in the benchmark. Record it in the report notes and open an issue.
