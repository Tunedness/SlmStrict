# SlmStrict

A Python library for reliable structured output from on-device small language models, with opt-in cloud fallback.

[![CI](https://github.com/Tunedness/SlmStrict/actions/workflows/ci.yml/badge.svg)](https://github.com/Tunedness/SlmStrict/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/Tunedness/SlmStrict)](LICENSE)
[![Designed & Maintained with Tan](https://muhammetsafak.com/badges/designed-maintained-with-tan.svg)](https://muhammetsafak.com/tan/)

SlmStrict is for developers who embed local language models in applications and need their output to match a schema. Models under 3B parameters often return JSON that deviates from the schema, wrap it in markdown fences or break it in other ways. SlmStrict handles that in one place: schema-constrained decoding where the backend supports it, deterministic repair, re-asking, and an optional cloud fallback that stays off unless configured. Give it a schema and a backend; it returns a valid value and tells you which rungs it needed.

## Project Status

**Experimental.** The MVP is feature-complete, but no release has been published and the API can still change. The reference benchmark exists but has not been run yet; see [Benchmark](#benchmark).

## Installation

Python 3.11 or newer. The core depends on Pydantic only; everything else is an extra:

| Extra | Adds |
| --- | --- |
| `llamacpp` | llama.cpp backend (`llama-cpp-python`), the only backend with grammar-constrained decoding |
| `onnx` | ONNX Runtime GenAI backend (`onnxruntime-genai`) |
| `anthropic`, `openai`, `gemini` | cloud fallback providers |
| `otel` | OTLP metrics export |

No release is published yet, so install from a clone:

```bash
git clone https://github.com/Tunedness/SlmStrict.git
cd SlmStrict
uv sync --extra llamacpp
```

Extras are imported lazily. `import slmstrict` loads none of them, and using one that is not installed raises an `ImportError` that names the extra to install.

## Quick start

The examples below run as they are: `FakeBackend` (from `slmstrict.testing`) replays scripted outputs, so you can try the API without a model. The test suite executes every example in this README that is followed by an output block and compares the printed output.

With a Pydantic model:

<!-- runnable -->
```python
from pydantic import BaseModel

from slmstrict import generate
from slmstrict.testing import FakeBackend


class Invoice(BaseModel):
    customer: str
    total: float
    paid: bool


# FakeBackend replays scripted outputs, so this runs without a model. The script is a typical
# small-model mistake: valid JSON wrapped in a markdown fence.
backend = FakeBackend(['```json\n{"customer": "Acme", "total": 99.5, "paid": false}\n```'])

result = generate(backend, Invoice, "Extract the invoice: Acme owes 99.50 and has not paid.")

print(result.value)
print([step.value for step in result.path])
```

<!-- output -->
```text
customer='Acme' total=99.5 paid=False
['repair']
```

The model wrapped its JSON in a markdown fence, so the repair rung stripped the fence; `result.path` lists that rung. `result.value` is an `Invoice` instance.

With a JSON Schema instead of a model, `result.value` is a `dict`:

<!-- runnable -->
```python
from slmstrict import generate
from slmstrict.testing import FakeBackend

schema = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "temperature_c": {"type": "number"}},
    "required": ["city", "temperature_c"],
}

backend = FakeBackend(['{"city": "Oslo", "temperature_c": 4}'])

result = generate(backend, schema, "Oslo is 4 degrees Celsius today.")

print(result.value)
print(result.path)
```

<!-- output -->
```text
{'city': 'Oslo', 'temperature_c': 4}
()
```

The path is empty: the first output was already valid, so no rung had to act.

With a real model, only the backend changes:

```python
from pydantic import BaseModel

from slmstrict import generate
from slmstrict.backends.llamacpp import LlamaCppBackend


class Invoice(BaseModel):
    customer: str
    total: float
    paid: bool


backend = LlamaCppBackend("/path/to/model.gguf", max_tokens=256, temperature=0.0)
result = generate(backend, Invoice, "Extract the invoice: Acme owes 99.50 and has not paid.")
```

Generation settings (`max_tokens`, `temperature`, ...) belong to the backend's constructor, not to the call. The prompt is passed to the model as plain text; if your model needs a chat template, apply it to the prompt yourself.

## Sync and async

`agenerate()` is the core; `generate()` is a thin synchronous wrapper around it. Both follow the same ladder and return the same `Result`. `generate()` runs on a fresh event loop, so it raises `RuntimeError` when called from inside a running loop; there, `await agenerate(...)`.

<!-- runnable -->
```python
import asyncio

from slmstrict import agenerate
from slmstrict.testing import FakeBackend

schema = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "temperature_c": {"type": "number"}},
    "required": ["city", "temperature_c"],
}


async def main() -> None:
    # The first answer is unusable; the ladder re-asks and the second answer is valid.
    backend = FakeBackend(["I cannot answer that.", '{"city": "Oslo", "temperature_c": 4}'])
    result = await agenerate(backend, schema, "Oslo is 4 degrees Celsius today.")
    print(result.value)
    print([step.value for step in result.path])
    print(len(result.attempts))


asyncio.run(main())
```

<!-- output -->
```text
{'city': 'Oslo', 'temperature_c': 4}
['reask']
2
```

`result.attempts` records every attempt, failed ones included.

## The enforcement ladder

SlmStrict tries the rungs in order and stops at the first valid output:

| Rung | What it does | Needs |
| --- | --- | --- |
| `grammar` | The JSON Schema is compiled to a GBNF grammar and decoding is constrained to it. | A grammar-capable backend (llama.cpp) |
| `repair` | Deterministic fixes of form only: strip fences and prose, close brackets, lossless type coercion. Repair never invents content. | Nothing |
| `reask` | The validation problems go back to the model, up to `max_reask` times (default 2). | Nothing |
| `fallback` | The same schema goes to a cloud model. | A configured provider, `local_only` off |

`result.path` is a tuple of the rungs that took effect, each once, in order of first use. It is empty when the first output of an unconstrained backend was valid as it was. The ONNX backend has no grammar rung; its ladder starts at repair.

When no rung produces a valid value, `generate()` raises `ValidationFailed`. It carries the issues, the last raw output and every attempt; none of them is part of `str(error)`.

<!-- runnable -->
```python
from slmstrict import ValidationFailed, generate
from slmstrict.testing import FakeBackend

schema = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

# Three unusable answers: the first attempt and the two default re-asks.
backend = FakeBackend(["no", "still no", "never"])

try:
    generate(backend, schema, "Name a city.")
except ValidationFailed as error:
    print("failed after", len(error.attempts), "attempts")
    print("last output:", error.last_output)
```

<!-- output -->
```text
failed after 3 attempts
last output: never
```

A backend that itself fails raises `BackendError`; the ladder does not turn that into a cloud call.

## Backends

| Backend | Class | Grammar rung |
| --- | --- | --- |
| llama.cpp | `slmstrict.backends.llamacpp.LlamaCppBackend` | yes |
| ONNX Runtime GenAI | `slmstrict.backends.onnx.OnnxBackend` | no |

Any object with `supports_grammar` and `async agenerate(prompt, *, constraint=None) -> str` is a backend (the `slmstrict.Backend` protocol). Compiled grammars are cached per process by schema hash.

## Cloud fallback and privacy

The fallback is off by default. It runs only when all of these hold: a handler is configured, `local_only` is not set, and the local rungs failed.

```python
from slmstrict import FallbackConfig, StrictConfig
from slmstrict.fallback import AnthropicProvider, CloudFallback, RegexMasker

handler = CloudFallback(AnthropicProvider(), masker=RegexMasker())
config = StrictConfig(fallback=FallbackConfig(handler=handler))
# StrictConfig(..., local_only=True) locks the fallback off whatever else is configured.
```

The privacy contract. Read the defaults first: **`CloudFallback` has no masker unless you pass one (`masker=None`), so the prompt is sent unmasked, and it logs no content unless you pass `log_content=True` (which requires a masker).**

- **`local_only`.** `StrictConfig(local_only=True)` stops the ladder from calling the fallback handler at all, whatever else is configured. `CloudFallback(..., local_only=True)` is a second, separate lock: the router itself then refuses to send anything, which also covers a handler that is used outside the ladder. The two flags are independent; set both when you want both locks.
- **Masking is opt-in.** Without a `masker`, the prompt goes to the provider **unmasked**. With one, the prompt passes through it (a `mask(text) -> str` protocol, compatible with McpGuard recognizers but without depending on them) before it leaves the device, and if the masker fails nothing is sent. `RegexMasker` is a best-effort example, not a compliance tool: it recognizes e-mail addresses, IBANs, payment card numbers, Turkish national ids and phone numbers; it does not detect names, addresses or other free text; and it reads only a run of 1-8 non-alphanumeric characters as the separator inside a card number or IBAN, so a number written with a longer or lettered separator is not masked. For production use, supply a recognizer built for your data.
- **What is sent.** The original prompt (masked only if you configured a masker). The local attempts and their raw outputs are not sent.
- **Logging.** By default only meta data is logged: provider, model, schema hash, characters sent, duration and outcome (logger `slmstrict.fallback`). `CloudFallback(..., log_content=True)` also logs the prompt that is sent and the provider's answer, and it needs a masker: `log_content=True` without one is a configuration error, so content never reaches a log unmasked by accident. If you really want the content logged in the clear, say so with `allow_unmasked_content_log=True` in addition; the unmasked log then holds personal data and is your responsibility. The ladder and backend loggers do not log prompts or outputs.
- **Provider SDKs** are extras and are imported lazily. The default model per provider is `slmstrict.fallback.DEFAULT_MODELS`; pass `model=` to choose your own.

## Measurement and cost

`MetricsCollector` is a ladder observer that keeps in-process counters only, never prompts or outputs:

<!-- runnable -->
```python
from slmstrict import StrictConfig, generate
from slmstrict.metrics import MetricsCollector
from slmstrict.testing import FakeBackend

schema = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
}

collector = MetricsCollector()
config = StrictConfig(observer=collector)

generate(FakeBackend(['{"n": 1}']), schema, "Give a number.", config=config)
generate(FakeBackend(['```json\n{"n": 2}\n```']), schema, "Give a number.", config=config)

snapshot = collector.snapshot()
print(snapshot.runs, snapshot.schema_pass_rate, snapshot.fallback_rate)
print(dict(snapshot.repair_step_counts))
print(collector.cost().total)
```

<!-- output -->
```text
2 1.0 0.0
{'fences': 1}
None
```

The snapshot also holds the path distribution, the rung each run was resolved by, re-ask counts, fallback statuses, a latency histogram and the cloud token usage.

Cost is estimated from a price table that you supply. The package ships no prices: without a table, or for a model the table does not list, the cost is `None` (unknown), not a guess.

```python
from slmstrict.metrics import MetricsCollector, ModelPrice, PriceTable

# Your provider's prices per million tokens; the package does not know them.
prices = PriceTable({("anthropic", "claude-opus-5-5"): ModelPrice(INPUT_PRICE, OUTPUT_PRICE)})
collector = MetricsCollector(prices)
```

For OTLP export, install the `otel` extra and register `slmstrict.metrics.otel.OtelObserver` as the config's `observer`; it exports counters and histograms without any content in the attributes.

## Benchmark

The reference benchmark (`bench/`) measures the PRD targets on Qwen3-4B, Llama-3.2-3B and Gemma-3-1B, on both backends, with 36 schema tasks in six categories. It needs real model files and is not run in CI; `bench/docker/run.sh` runs it, and the real-model integration tests, in a network-isolated Docker container.

| Metric | Target | Result |
| --- | --- | --- |
| Schema pass rate, strict (raw model: typically 60-80%) | >= 95% | not run yet |
| Extra latency on the grammar path | <= 15% | not run yet |
| Fallback rate on the reference tasks | < 10% | not run yet |

Nothing has been measured, so the table makes no claim. How to run it, and what it measures, is in [bench/README.md](bench/README.md). Results land in `bench/results/`.

## Contributing

Contributions are welcome. Read the [contributing guide](CONTRIBUTING.md) before opening an issue or pull request.

## Security

Do not report security vulnerabilities through public issues; follow the [security policy](SECURITY.md) instead.

## License

Licensed under the [Apache License 2.0](LICENSE).
