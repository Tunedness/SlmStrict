"""Tests for the benchmark's container tooling (``bench/docker/``).

Nothing here needs Docker, a network or a model: the downloader runs against a fake Hub served on
the loopback interface, and the shell script is only syntax-checked and read.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from bench.docker import download_models as dl
from bench.docker.integration import plan_runs
from bench.runner import REFERENCE_MODELS, load_config

ROOT = Path(__file__).resolve().parent.parent
DOCKER = ROOT / "bench" / "docker"
TOKEN = "hf_secret_token_value"

GGUF_BYTES = b"GGUF" + bytes(range(256)) * 40
ONNX_FILES = {
    "genai_config.json": b'{"model": {"type": "llama"}}',
    "model.onnx": b"graph" * 10,
    "model.onnx.data": bytes(range(256)) * 64,
}


# -- configuration files ----------------------------------------------------------------------


def test_download_plan_and_container_config_agree() -> None:
    plans = {plan.name: plan for plan in dl.load_plan(DOCKER / "models.download.json")}
    config = load_config(DOCKER / "models.container.toml")
    assert tuple(plans) == REFERENCE_MODELS
    assert tuple(spec.name for spec in config.models) == REFERENCE_MODELS
    for spec in config.models:
        plan = plans[spec.name]
        (gguf_name,) = plan.gguf.include
        assert spec.gguf_path == Path("/models") / spec.name / "gguf" / gguf_name
        assert spec.onnx_dir == Path("/models") / spec.name / "onnx"
        assert "Q4" in gguf_name
        # The report's Source column comes from the container config: it must name the same
        # repositories the downloader fetches from.
        assert spec.gguf_source == plan.gguf.repo
        assert spec.onnx_source is not None
        assert spec.onnx_source.startswith(plan.onnx.repo)
        assert spec.gguf_quantization
        assert spec.onnx_quantization
        assert spec.prompt_template is not None


def _templates(path: Path) -> dict[str, str | None]:
    return {spec.name: spec.prompt_template for spec in load_config(path).models}


def test_container_prompt_templates_match_the_example_config() -> None:
    assert _templates(ROOT / "bench/models.toml.example") == _templates(
        DOCKER / "models.container.toml"
    )


def test_integration_runs_cover_every_model_and_format() -> None:
    runs = plan_runs(load_config(DOCKER / "models.container.toml"))
    assert len(runs) == 6
    assert {r.variable for r in runs} == {"SLMSTRICT_GGUF_PATH", "SLMSTRICT_ONNX_MODEL_DIR"}
    for run in runs:
        assert (ROOT / run.test_file).is_file()
        assert "-m" in run.command()
        assert "integration" in run.command()


def test_dockerfile_keeps_weights_out_and_installs_both_extras() -> None:
    dockerfile = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.12" in dockerfile
    assert "uv sync --locked --extra llamacpp --extra onnx" in dockerfile
    # The baseline is recorded for the report and its effect is printed during the build.
    assert "/opt/llama-build-flags.txt" in dockerfile
    # An assignment in front of a command lasts for that command only: the file must be written
    # from a shell variable, not from CMAKE_ARGS (which was empty there once), and checked.
    assert 'echo "$CMAKE_ARGS"' not in dockerfile
    assert 'echo "$cmake_args" > /opt/llama-build-flags.txt' in dockerfile
    assert "test -s /opt/llama-build-flags.txt" in dockerfile
    assert "SLMSTRICT_LLAMA_BUILD_FILE=" in dockerfile
    assert "llama_print_system_info" in dockerfile
    # CPU build with a fixed baseline: GGML_NATIVE detection breaks the arm64 build with gcc 12.
    assert "-DGGML_NATIVE=OFF" in dockerfile
    assert "-DGGML_CPU_ARM_ARCH=armv8.2-a+fp16" in dockerfile
    assert "-DGGML_AVX2=ON" in dockerfile
    assert "-DGGML_CUDA=OFF" in dockerfile
    # Inputs are pinned: the base image by digest, uv by version, and the image is not root.
    assert re.search(r"^FROM python:3\.12-slim-bookworm@sha256:[0-9a-f]{64}$", dockerfile, re.M)
    assert re.search(r"pip install uv==\d+\.\d+\.\d+", dockerfile)
    assert re.search(r"^USER bench$", dockerfile, re.M)
    ignore = (DOCKER / "Dockerfile.dockerignore").read_text(encoding="utf-8")
    for pattern in ("*.gguf", "*.onnx", "*.onnx.data", "bench/models.toml", ".env"):
        assert pattern in ignore


# -- run.sh -----------------------------------------------------------------------------------


def test_run_script_is_valid_bash_and_isolates_the_run_stage() -> None:
    script = DOCKER / "run.sh"
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    subprocess.run([bash, "-n", str(script)], check=True)  # syntax check only

    text = script.read_text(encoding="utf-8")
    download = text.split("cmd_download() {")[1].split("\n}\n")[0]
    isolated = text.split("run_isolated() {")[1].split("\n}\n")[0]
    # Stage 1 has the network and runs only the download script; stage 2 has neither.
    assert "--network none" not in download
    assert "download_models.py" in download
    assert "-e HF_TOKEN \\" in download
    assert "--network none" in isolated
    assert "-e HF_TOKEN" not in isolated
    assert '-v "$REPO_ROOT":/app:ro' in isolated
    assert '-v "$VOLUME":/models:ro' in isolated
    assert "/app/bench/results:rw" in isolated
    assert "--memory" in isolated
    assert "--memory-swap" in isolated
    assert "--read-only" in isolated
    assert "--cap-drop ALL" in isolated
    assert "--security-opt no-new-privileges" in isolated
    assert "--pids-limit" in isolated
    assert '--user "$(id -u):$(id -g)"' in isolated
    # The image does not default to root, so the download stage, which writes the models volume
    # (owned by root), asks for root explicitly and with every capability dropped.
    assert "--user 0:0" in download
    assert "--cap-drop ALL" in download
    # The token is passed by name only; its value is never spelled out in the script.
    assert "HF_TOKEN=" not in text


def test_run_script_help_does_not_need_a_container() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    done = subprocess.run(
        [bash, str(DOCKER / "run.sh"), "help"], capture_output=True, text=True, check=False
    )
    # Without docker on PATH the script stops early; with it, help prints. Never a container.
    assert done.returncode in (0, 1)


# -- the downloader against a fake Hub --------------------------------------------------------


class FakeHub:
    """A tiny Hub (API plus file server) and a separate 'CDN' host that files redirect to."""

    def __init__(
        self,
        *,
        wrong_gguf_oid: bool = False,
        require_token: bool = False,
        refuse_all: bool = False,
        wrong_small_oid: bool = False,
        truncate_gguf: bool = False,
    ) -> None:
        self.wrong_small_oid = wrong_small_oid
        self.truncate_gguf = truncate_gguf
        self.refuse_all = refuse_all
        self.wrong_gguf_oid = wrong_gguf_oid
        self.require_token = require_token
        self.hub_auth: list[str | None] = []
        self.cdn_auth: list[str | None] = []
        self.ranges: list[str | None] = []
        self.commit = "c" * 40
        self.repos = {
            "acme/gguf": {"m-Q4.gguf": GGUF_BYTES, "m-Q8.gguf": b"unwanted"},
            "acme/onnx": {f"sub/dir/{n}": c for n, c in ONNX_FILES.items()} | {"README.md": b"x"},
        }
        hub, cdn = self, None

        class Hub(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def do_GET(self) -> None:
                hub.hub_auth.append(self.headers.get("Authorization"))
                if hub.refuse_all or (hub.require_token and not self.headers.get("Authorization")):
                    self.send_response(401)
                    self.end_headers()
                    return
                path = self.path.split("?")[0]
                parts = path.strip("/").split("/")
                if parts[0] == "api":  # api/models/<owner>/<name>/(revision|tree)/<x>
                    repo = f"{parts[2]}/{parts[3]}"
                    if repo not in hub.repos:
                        self.send_response(404)
                        self.end_headers()
                        return
                    if parts[4] == "revision":
                        self._json({"sha": hub.commit})
                    else:
                        self._json(hub._tree(repo))
                    return
                # <owner>/<name>/resolve/<sha>/<file...>: redirect to the CDN host.
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{cdn.port}/blob{path}")
                self.end_headers()

            def _json(self, payload: Any) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class Cdn(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def do_GET(self) -> None:
                hub.cdn_auth.append(self.headers.get("Authorization"))
                hub.ranges.append(self.headers.get("Range"))
                parts = self.path.removeprefix("/blob/").split("/")
                repo = f"{parts[0]}/{parts[1]}"
                name = "/".join(parts[4:])
                data = hub.repos[repo][name]
                if hub.truncate_gguf and name.endswith(".gguf"):
                    data = data[:-5]  # the tree promises more bytes than the server has
                start = 0
                status = 200
                header = self.headers.get("Range")
                if header:
                    start = int(header.removeprefix("bytes=").rstrip("-"))
                    status = 206
                body = data[start:]
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._hub = ThreadingHTTPServer(("127.0.0.1", 0), Hub)
        self._cdn = ThreadingHTTPServer(("127.0.0.1", 0), Cdn)
        cdn = self._cdn_info = type("Port", (), {"port": self._cdn.server_address[1]})()
        self.endpoint = f"http://127.0.0.1:{self._hub.server_address[1]}"
        self._threads = [
            threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
            )
            for server in (self._hub, self._cdn)
        ]

    def _tree(self, repo: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = [{"type": "directory", "path": "sub"}]
        for name, data in self.repos[repo].items():
            oid = hashlib.sha256(data).hexdigest()
            if self.wrong_gguf_oid and name.endswith(".gguf"):
                oid = "0" * 64
            entry: dict[str, Any] = {"type": "file", "path": name, "size": len(data)}
            if len(data) > 100:  # larger files are "LFS" and publish a SHA-256
                entry["lfs"] = {"oid": oid, "size": len(data)}
            else:  # the others publish their git blob id
                blob = hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False)
                entry["oid"] = blob.hexdigest()
                if self.wrong_small_oid:
                    entry["oid"] = "0" * 40
            entries.append(entry)
        return entries

    def __enter__(self) -> FakeHub:
        for thread in self._threads:
            thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for server in (self._hub, self._cdn):
            server.shutdown()
            server.server_close()


@pytest.fixture
def plan_file(tmp_path: Path) -> Path:
    plan = {
        "models": [
            {
                "name": "Tiny",
                "gguf": {"repo": "acme/gguf", "include": ["m-Q4.gguf"], "quantization": "Q4"},
                "onnx": {"repo": "acme/onnx", "subdir": "sub/dir", "quantization": "int4"},
            }
        ]
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


@pytest.fixture
def hub() -> Iterator[FakeHub]:
    with FakeHub() as fake:
        yield fake


def _run(monkeypatch: pytest.MonkeyPatch, hub: FakeHub, plan: Path, dest: Path, *extra: str) -> int:
    monkeypatch.setenv("HF_ENDPOINT", hub.endpoint)
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    return dl.main(["--config", str(plan), "--dest", str(dest), *extra])


def test_download_writes_the_layout_manifest_and_verifies_hashes(
    monkeypatch: pytest.MonkeyPatch,
    hub: FakeHub,
    plan_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dest = tmp_path / "models"

    assert _run(monkeypatch, hub, plan_file, dest) == 0

    assert (dest / "Tiny/gguf/m-Q4.gguf").read_bytes() == GGUF_BYTES
    assert not (dest / "Tiny/gguf/m-Q8.gguf").exists()  # not in the include list
    for name, content in ONNX_FILES.items():
        assert (dest / "Tiny/onnx" / name).read_bytes() == content  # subdir flattened
    assert not (dest / "Tiny/onnx/README.md").exists()
    assert not list(dest.rglob("*.part"))

    manifest = json.loads((dest / "Tiny/manifest.json").read_text())
    assert manifest["gguf"]["commit"] == hub.commit
    (gguf_file,) = manifest["gguf"]["files"]
    assert gguf_file["sha256"] == hashlib.sha256(GGUF_BYTES).hexdigest()
    assert {f["path"] for f in manifest["onnx"]["files"]} == set(ONNX_FILES)

    # The token reached the Hub, never the redirect target, and is nowhere in the output.
    assert f"Bearer {TOKEN}" in hub.hub_auth
    assert hub.cdn_auth
    assert all(value is None for value in hub.cdn_auth)
    shown = capsys.readouterr()
    assert TOKEN not in shown.out + shown.err
    assert TOKEN not in (dest / "Tiny/manifest.json").read_text()


def test_download_is_idempotent_and_resumes_a_partial_file(
    monkeypatch: pytest.MonkeyPatch, hub: FakeHub, plan_file: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "models"
    part = dest / "Tiny/gguf/m-Q4.gguf.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(GGUF_BYTES[:100])

    assert _run(monkeypatch, hub, plan_file, dest) == 0
    assert "bytes=100-" in hub.ranges
    assert (dest / "Tiny/gguf/m-Q4.gguf").read_bytes() == GGUF_BYTES

    hub.ranges.clear()
    assert _run(monkeypatch, hub, plan_file, dest) == 0  # second run downloads nothing
    assert hub.ranges == []


def test_download_rejects_a_file_whose_hash_differs_from_the_hub(
    monkeypatch: pytest.MonkeyPatch,
    plan_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with FakeHub(wrong_gguf_oid=True) as bad_hub:
        code = _run(monkeypatch, bad_hub, plan_file, tmp_path / "models")
    assert code == 1
    assert "SHA-256 does not match" in capsys.readouterr().err
    assert not list((tmp_path / "models").rglob("m-Q4.gguf*"))


def test_download_explains_a_gated_repository_without_leaking_the_token(
    monkeypatch: pytest.MonkeyPatch,
    plan_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with FakeHub(require_token=True) as gated:
        monkeypatch.setenv("HF_ENDPOINT", gated.endpoint)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        code = dl.main(["--config", str(plan_file), "--dest", str(tmp_path / "models")])
    assert code == 1
    err = capsys.readouterr().err
    assert "gated" in err
    assert "needs a Hugging Face token" in err
    assert "HF_TOKEN=hf_... bench/docker/run.sh download" in err
    assert "https://huggingface.co/acme/gguf" in err  # names the repository to accept
    assert "no token was set" in err


def test_a_rejected_token_is_reported_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch,
    plan_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with FakeHub(refuse_all=True) as refusing:
        monkeypatch.setenv("HF_ENDPOINT", refusing.endpoint)
        monkeypatch.setenv("HF_TOKEN", TOKEN)
        assert dl.main(["--config", str(plan_file), "--dest", str(tmp_path / "m")]) == 1
    shown = capsys.readouterr()
    assert "the token was sent but acme/gguf still refused it" in shown.err
    assert TOKEN not in shown.out + shown.err


def test_download_stops_when_the_disk_is_too_small(
    monkeypatch: pytest.MonkeyPatch,
    hub: FakeHub,
    plan_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Usage:
        free = 1000

    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _path: Usage())
    assert _run(monkeypatch, hub, plan_file, tmp_path / "models") == 1
    assert "not enough disk space" in capsys.readouterr().err


def test_token_checks() -> None:
    assert dl.check_token(None, dl.DEFAULT_ENDPOINT) is None
    assert dl.check_token("hf_abc", dl.DEFAULT_ENDPOINT) == "hf_abc"
    with pytest.raises(dl.DownloadError, match="unexpected format") as bad:
        dl.check_token("hf_abc\nX-Evil: 1", dl.DEFAULT_ENDPOINT)
    assert "hf_abc" not in str(bad.value)
    with pytest.raises(dl.DownloadError, match="non-https"):
        dl.check_token("hf_abc", "http://example.com")
    assert dl.check_token("hf_abc", "http://127.0.0.1:9") == "hf_abc"


def test_select_files_requires_every_pattern_to_match() -> None:
    entries = [{"type": "file", "path": "a/b.bin", "size": 3}]
    source = dl.Source("o/r", ("*.bin",), subdir="a")
    assert [f.path for f in dl.select_files(entries, source)] == ["b.bin"]
    with pytest.raises(dl.DownloadError, match="nothing matches"):
        dl.select_files(entries, dl.Source("o/r", ("*.bin", "missing.json"), subdir="a"))


def test_unsafe_paths_from_the_hub_are_refused() -> None:
    entries = [{"type": "file", "path": "../escape.bin", "size": 3}]
    with pytest.raises(dl.DownloadError, match="unsafe"):
        dl.select_files(entries, dl.Source("o/r", ("*",)))


def test_plan_rejects_unsafe_names(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"models": [{"name": "../x", "gguf": {}, "onnx": {}}]}))
    with pytest.raises(dl.DownloadError):
        dl.load_plan(path)


def test_small_files_are_checked_against_their_git_blob_id(
    monkeypatch: pytest.MonkeyPatch,
    plan_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with FakeHub(wrong_small_oid=True) as bad_hub:
        code = _run(monkeypatch, bad_hub, plan_file, tmp_path / "models")
    assert code == 1
    assert "git blob id does not match" in capsys.readouterr().err
    assert not list((tmp_path / "models").rglob("genai_config.json"))


def test_a_corrupt_small_file_that_is_already_there_is_fetched_again(
    monkeypatch: pytest.MonkeyPatch, hub: FakeHub, plan_file: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "models"
    assert _run(monkeypatch, hub, plan_file, dest) == 0
    config = dest / "Tiny/onnx/genai_config.json"
    good = config.read_bytes()
    config.write_bytes(b"X" * len(good))  # same size, wrong content

    assert _run(monkeypatch, hub, plan_file, dest) == 0
    assert config.read_bytes() == good


def test_a_short_download_is_a_clean_error_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    plan_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with FakeHub(truncate_gguf=True) as short_hub:
        code = _run(monkeypatch, short_hub, plan_file, tmp_path / "models")
    assert code == 1
    err = capsys.readouterr().err
    assert f"received {len(GGUF_BYTES) - 5} bytes, expected {len(GGUF_BYTES)}" in err
    assert "Traceback" not in err
    assert not list((tmp_path / "models").rglob("*.part"))


@pytest.mark.parametrize(
    ("old", "new", "kept"),
    [
        ("https://hub.example/a", "https://hub.example/b", True),
        ("https://hub.example/a", "http://hub.example/a", False),  # downgrade, same host
        ("https://hub.example/a", "https://cdn.example/a", False),
        ("https://hub.example/a", "https://hub.example:8443/a", False),
    ],
)
def test_authorization_survives_a_redirect_only_to_the_same_origin(
    old: str, new: str, kept: bool
) -> None:
    request = dl.urllib.request.Request(old)
    request.add_header("Authorization", "Bearer x")
    handler = dl._DropAuthorizationOnNewHost()
    redirected = handler.redirect_request(request, None, 302, "Found", {}, new)  # type: ignore[arg-type]
    assert redirected is not None
    assert redirected.has_header("Authorization") is kept
