# syntax=docker/dockerfile:1
#
# Benchmark image: the repository plus a CPU-only build of the llama.cpp and ONNX extras.
# Model weights are not part of the image; they live in a named volume (see run.sh).
#
# Build from the repository root. llama.cpp is compiled for a fixed CPU baseline, not for the CPU
# of the build machine: GGML_NATIVE detection on arm64 produced flags gcc 12 (bookworm) cannot
# compile ("target specific option mismatch" in ggml-cpu/vec.cpp). The baseline is armv8.2-a with
# fp16 and dot product on arm64 (Apple M1 and newer, Graviton 2 and newer) and AVX2/FMA/F16C on
# amd64; a CPU without them cannot run this image.
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
    UV_PYTHON_DOWNLOADS=never

# llama-cpp-python is built from source (the lock file has no wheel for it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv version: the one the lock file was produced with (uv.lock needs a recent uv).
RUN pip install uv==0.13.8

WORKDIR /app
COPY . .
# CPU-only build with a portable baseline (see the note at the top).
RUN case "$(dpkg --print-architecture)" in \
      arm64) cpu="-DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod" ;; \
      amd64) cpu="-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" ;; \
      *) cpu="" ;; \
    esac \
    && cmake_args="-DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF $cpu" \
    && CMAKE_ARGS="$cmake_args" uv sync --locked --extra llamacpp --extra onnx \
    && echo "$cmake_args" > /opt/llama-build-flags.txt \
    && test -s /opt/llama-build-flags.txt

# The benchmark report quotes this file, so a report says which CPU baseline it was measured on.
ENV PATH="/opt/venv/bin:${PATH}" \
    SLMSTRICT_LLAMA_BUILD_FILE=/opt/llama-build-flags.txt

# Evidence that the baseline reached the build: llama.cpp's own report of its CPU features.
RUN python -c "import llama_cpp; print(llama_cpp.llama_print_system_info().decode())"

# Not root by default. run.sh passes --user explicitly (the caller's uid for the run stage, root
# for the download stage that writes the models volume).
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin bench
USER bench
CMD ["python", "-m", "bench", "--models", "bench/docker/models.container.toml"]
