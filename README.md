# SlmStrict

A Python library for reliable structured output from on-device small language models, with opt-in cloud fallback.

[![CI](https://github.com/Tunedness/SlmStrict/actions/workflows/ci.yml/badge.svg)](https://github.com/Tunedness/SlmStrict/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/Tunedness/SlmStrict)](LICENSE)
[![Designed & Maintained with Tan](https://muhammetsafak.com/badges/designed-maintained-with-tan.svg)](https://muhammetsafak.com/tan/)

SlmStrict is intended for developers who embed local language models in applications and need their output to match a schema. Models under 3B parameters often return JSON that deviates from the schema, wrap it in markdown fences or break it in other ways. SlmStrict is being built to handle that in one place: schema-constrained decoding where the backend supports it, deterministic repair, re-asking, and an optional cloud fallback that stays off unless configured.

## Project Status

**Experimental.** The repository contains the project skeleton only; the library does not provide any behavior yet, and its API will change without notice.

## Contributing

Contributions are welcome. Read the [contributing guide](CONTRIBUTING.md) before opening an issue or pull request.

## Security

Do not report security vulnerabilities through public issues; follow the [security policy](SECURITY.md) instead.

## License

Licensed under the [Apache License 2.0](LICENSE).
