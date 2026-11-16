# syntax=docker/dockerfile:1
#
# Benchmark image: the repository plus a CPU-only build of the llama.cpp and ONNX extras.
# Model weights are not part of the image; they live in a named volume (see run.sh).
#
# Build from the repository root, on the machine that will run the benchmark (llama.cpp is
# compiled for the CPU it is built on):
#
#   docker build -f bench/docker/Dockerfile -t slmstrict-bench .
# Pinned by digest (the multi-arch index of 3.12-slim-bookworm, read from Docker Hub on 2026-11-16).
# To move to a newer base, look up the new digest and change it here in one deliberate commit.
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_METAL=OFF"

# llama-cpp-python is built from source (the lock file has no wheel for it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv version: the one the lock file was produced with (uv.lock needs a recent uv).
RUN pip install uv==0.13.8

WORKDIR /app
COPY . .
RUN uv sync --locked --extra llamacpp --extra onnx

ENV PATH="/opt/venv/bin:${PATH}"

# Not root by default. run.sh passes --user explicitly (the caller's uid for the run stage, root
# for the download stage that writes the models volume).
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin bench
USER bench
CMD ["python", "-m", "bench", "--models", "bench/docker/models.container.toml"]
