#!/usr/bin/env bash
# Runs the SlmStrict benchmark and the real-model integration tests in a Docker container.
#
#   bench/docker/run.sh build        build the image (from the repository root)
#   bench/docker/run.sh download     stage 1: download the models into a named volume (network on)
#   bench/docker/run.sh bench        stage 2: run the benchmark (network off)
#   bench/docker/run.sh integration  stage 2: run the two integration test files per model
#   bench/docker/run.sh run          stage 2: benchmark, then integration tests
#
# Extra arguments of `download` go to download_models.py (for example --only Gemma-3-1B or
# --verify); extra arguments of `bench` go to `python -m bench` (for example --limit 4).
#
# Environment (all optional):
#   HF_TOKEN                      Hugging Face token, read by the download stage only
#   SLMSTRICT_BENCH_IMAGE         image name          (default: slmstrict-bench)
#   SLMSTRICT_MODELS_VOLUME       models volume       (default: slmstrict-models)
#   SLMSTRICT_BENCH_MEMORY        container memory    (default: 6g; keep it below the Docker VM)
#   SLMSTRICT_BENCH_CPUS          CPU limit           (default: no limit)
#   SLMSTRICT_BENCH_PLATFORM      docker --platform   (default: the Docker host's own)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${SLMSTRICT_BENCH_IMAGE:-slmstrict-bench}"
VOLUME="${SLMSTRICT_MODELS_VOLUME:-slmstrict-models}"
MEMORY="${SLMSTRICT_BENCH_MEMORY:-6g}"
CPUS="${SLMSTRICT_BENCH_CPUS:-}"
PLATFORM="${SLMSTRICT_BENCH_PLATFORM:-}"
CONFIG="bench/docker/models.container.toml"

die() {
  echo "error: $*" >&2
  exit 1
}

usage() {
  sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"

cmd_build() {
  local platform_args=()
  [[ -n "$PLATFORM" ]] && platform_args=(--platform "$PLATFORM")
  docker build ${platform_args[@]+"${platform_args[@]}"} -f "$REPO_ROOT/bench/docker/Dockerfile" -t "$IMAGE" "$REPO_ROOT"
}

# Stage 1. The only stage with network access, and it runs nothing but the download script.
# HF_TOKEN is passed by name, so its value never appears in this script or in the process list.
# It runs as root (the image's default user is not) because the new volume is owned by root; all
# capabilities are dropped, so it can write only what it owns.
cmd_download() {
  docker volume create "$VOLUME" >/dev/null
  docker run --rm \
    --user 0:0 \
    --read-only --tmpfs /tmp:rw,size=256m \
    --cap-drop ALL --security-opt no-new-privileges \
    -e HF_TOKEN \
    -v "$REPO_ROOT":/app:ro \
    -v "$VOLUME":/models \
    "$IMAGE" \
    python /app/bench/docker/download_models.py --dest /models "$@"
}

# Stage 2. No network; the repository and the models are read-only; only bench/results is
# writable (plus a small tmpfs). The container runs as the calling user so that the reports it
# writes belong to you.
run_isolated() {
  docker volume inspect "$VOLUME" >/dev/null 2>&1 \
    || die "models volume '$VOLUME' not found; run: $0 download"
  mkdir -p "$REPO_ROOT/bench/results"
  local limits=(--memory "$MEMORY" --memory-swap "$MEMORY")
  [[ -n "$CPUS" ]] && limits+=(--cpus "$CPUS")
  docker run --rm \
    --network none \
    "${limits[@]}" \
    --read-only --tmpfs /tmp:rw,size=256m \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 512 \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$REPO_ROOT":/app:ro \
    -v "$REPO_ROOT/bench/results":/app/bench/results:rw \
    -v "$VOLUME":/models:ro \
    -w /app \
    "$IMAGE" "$@"
}

cmd_bench() {
  run_isolated python -m bench --models "$CONFIG" "$@"
}

cmd_integration() {
  run_isolated python -m bench.docker.integration "$CONFIG"
}

case "${1:-help}" in
  build) shift; cmd_build ;;
  download) shift; cmd_download "$@" ;;
  bench) shift; cmd_bench "$@" ;;
  integration) shift; cmd_integration ;;
  run) shift; cmd_bench "$@"; cmd_integration ;;
  help | -h | --help) usage ;;
  *) usage >&2; exit 2 ;;
esac
