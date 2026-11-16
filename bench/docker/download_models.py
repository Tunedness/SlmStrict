#!/usr/bin/env python3
"""Download the benchmark's reference models from Hugging Face. Standard library only.

    python download_models.py --dest /models

The plan (repositories, files, quantization) is read from ``models.download.json``. Each model
lands in ``<dest>/<name>/gguf/`` and ``<dest>/<name>/onnx/`` (the layout ``models.container.toml``
expects), together with a ``manifest.json`` that records the commit and the SHA-256 of every file.

Behaviour worth knowing:

* The token is read from the ``HF_TOKEN`` environment variable only. It is sent in an
  ``Authorization`` header to the Hub host and to nobody else (it is dropped when a download is
  redirected to another host), and it is never printed, logged or written to disk.
* Files are verified against the SHA-256 the Hub publishes for large files. A download that is
  interrupted resumes from its ``.part`` file.
* ``HF_ENDPOINT`` overrides the Hub URL (the tests point it at a local server).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

DEFAULT_ENDPOINT = "https://huggingface.co"
CHUNK_SIZE = 1 << 20
TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 5
DISK_MARGIN_BYTES = 1 << 30
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA1 = re.compile(r"[0-9a-fA-F]{40}")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class DownloadError(Exception):
    """A download problem. Messages never contain the token or a header."""


@dataclass(frozen=True, slots=True)
class Source:
    """One repository (or a directory of it) and the files wanted from it."""

    repo: str
    include: tuple[str, ...]
    revision: str = "main"
    subdir: str = ""
    quantization: str = ""


@dataclass(frozen=True, slots=True)
class ModelPlan:
    name: str
    gguf: Source
    onnx: Source


@dataclass(frozen=True, slots=True)
class RemoteFile:
    path: str  # relative to the source's subdirectory
    remote_path: str  # path inside the repository
    size: int
    sha256: str | None  # published for LFS files only
    git_sha1: str | None = None  # git blob id, published for files that are not LFS


# -- plan -------------------------------------------------------------------------------------


def _safe_relative(value: str, what: str) -> str:
    parts = value.split("/")
    if not value or value.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise DownloadError(f"unsafe {what}: {value!r}")
    return value


def _source(table: object, where: str, default_include: Sequence[str] | None = None) -> Source:
    if not isinstance(table, dict):
        raise DownloadError(f"{where} must be an object")
    repo = table.get("repo")
    if not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise DownloadError(f"{where}: 'repo' must look like 'owner/name'")
    include = table.get("include", default_include)
    if (
        not isinstance(include, list)
        or not include
        or not all(isinstance(p, str) and p for p in include)
    ):
        raise DownloadError(f"{where}: 'include' must be a non-empty list of patterns")
    subdir = table.get("subdir", "")
    if not isinstance(subdir, str):
        raise DownloadError(f"{where}: 'subdir' must be a string")
    if subdir:
        _safe_relative(subdir, f"{where} subdir")
    revision = table.get("revision", "main")
    quantization = table.get("quantization", "")
    if not isinstance(revision, str) or not revision or not isinstance(quantization, str):
        raise DownloadError(f"{where}: 'revision' and 'quantization' must be strings")
    return Source(repo, tuple(include), revision, subdir, quantization)


def load_plan(path: Path) -> list[ModelPlan]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DownloadError(f"cannot read {path}: {type(exc).__name__}") from exc
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list) or not models:
        raise DownloadError(f"{path}: 'models' must be a non-empty list")
    plans: list[ModelPlan] = []
    for index, table in enumerate(models):
        where = f"models[{index}]"
        if not isinstance(table, dict):
            raise DownloadError(f"{where} must be an object")
        name = table.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[\w.-]+", name):
            raise DownloadError(f"{where}: 'name' must be a simple name")
        if any(plan.name == name for plan in plans):
            raise DownloadError(f"duplicate model name {name!r}")
        plans.append(
            ModelPlan(
                name,
                _source(table.get("gguf"), f"{where}.gguf"),
                _source(table.get("onnx"), f"{where}.onnx", ["*"]),
            )
        )
    return plans


def select_files(entries: Sequence[dict[str, Any]], source: Source) -> list[RemoteFile]:
    """Pick the files of ``source`` out of a repository tree listing.

    Every include pattern must match something, so a renamed or removed file fails loudly.
    """
    prefix = f"{source.subdir}/" if source.subdir else ""
    chosen: dict[str, RemoteFile] = {}
    matched = dict.fromkeys(source.include, False)
    for entry in entries:
        remote = entry.get("path")
        if entry.get("type") != "file" or not isinstance(remote, str):
            continue
        if not remote.startswith(prefix):
            continue
        rel = remote[len(prefix) :]
        for pattern in source.include:
            if fnmatch.fnmatchcase(rel, pattern):
                matched[pattern] = True
                size = entry.get("size")
                lfs = entry.get("lfs")
                oid = lfs.get("oid") if isinstance(lfs, dict) else None
                blob = entry.get("oid")
                if not isinstance(size, int) or size < 0:
                    raise DownloadError(f"{source.repo}: no size for {remote!r}")
                digest = oid.lower() if isinstance(oid, str) and _SHA256.fullmatch(oid) else None
                git_id = blob.lower() if isinstance(blob, str) and _SHA1.fullmatch(blob) else None
                chosen[rel] = RemoteFile(
                    _safe_relative(rel, "file path"),
                    remote,
                    size,
                    digest,
                    None if digest else git_id,
                )
                break
    missing = [pattern for pattern, hit in matched.items() if not hit]
    if missing:
        raise DownloadError(f"{source.repo}: nothing matches {', '.join(missing)}")
    return sorted(chosen.values(), key=lambda f: f.path)


# -- http -------------------------------------------------------------------------------------


class _DropAuthorizationOnNewHost(urllib.request.HTTPRedirectHandler):
    """urllib would copy the Authorization header to whatever host a redirect points at."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        old = urllib.parse.urlsplit(req.full_url)
        target = urllib.parse.urlsplit(newurl)
        # A change of host, port or scheme (https to http included) is a new party.
        if new is not None and (target.scheme, target.netloc) != (old.scheme, old.netloc):
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_DropAuthorizationOnNewHost)


def check_token(token: str | None, endpoint: str) -> str | None:
    """Validate the token's shape (never echoing it) and refuse to send it over plain http."""
    if token is None:
        return None
    if not token or not token.isascii() or any(c.isspace() for c in token):
        raise DownloadError("HF_TOKEN has an unexpected format (spaces or non-ASCII characters)")
    parts = urllib.parse.urlsplit(endpoint)
    if parts.scheme != "https" and parts.hostname not in _LOOPBACK:
        raise DownloadError("refusing to send HF_TOKEN to a non-https endpoint")
    return token


def _request(url: str, token: str | None, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        return _OPENER.open(request, timeout=TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        # The URL is safe to show (no secrets in it); headers and body are not.
        status = exc.code
        exc.close()
        if status in (401, 403):
            repo = _repo_of(url)
            if token:
                hint = (
                    f"the token was sent but {repo} still refused it: check that the token can "
                    f"read repositories and that you accepted the license on "
                    f"https://huggingface.co/{repo}"
                )
            else:
                hint = (
                    f"{repo} needs a Hugging Face token (gated or private repository). Accept its "
                    f"license on https://huggingface.co/{repo} while logged in, create a read "
                    "token and run: HF_TOKEN=hf_... bench/docker/run.sh download (no token was set)"
                )
        elif status == 404:
            hint = "not found (repository, revision or file name)"
        else:
            hint = "server error" if status >= 500 else "request refused"
        raise DownloadError(f"HTTP {status} for {_origin_and_path(url)}: {hint}") from None


def _repo_of(url: str) -> str:
    """The ``owner/name`` of a Hub API or file URL, for messages."""
    parts = [p for p in urllib.parse.urlsplit(url).path.split("/") if p]
    if parts[:2] == ["api", "models"]:
        parts = parts[2:]
    return "/".join(parts[:2]) if len(parts) >= 2 else "the repository"


def _origin_and_path(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.netloc}{parts.path}"


def _get_json(url: str, token: str | None) -> tuple[Any, str]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with _request(url, token) as response:
                return json.load(response), response.headers.get("Link", "")
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            if attempt == MAX_ATTEMPTS:
                raise DownloadError(f"network error for {_origin_and_path(url)}") from None
            time.sleep(min(2**attempt, 20))
        except ValueError:
            raise DownloadError(f"unexpected reply from {_origin_and_path(url)}") from None
    raise AssertionError("unreachable")  # pragma: no cover


def _next_link(link_header: str, endpoint: str) -> str | None:
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    if not match:
        return None
    url = match.group(1)
    if urllib.parse.urlsplit(url).netloc != urllib.parse.urlsplit(endpoint).netloc:
        raise DownloadError("the Hub pointed at another host for the next page; stopping")
    return url


def resolve_commit(endpoint: str, repo: str, revision: str, token: str | None) -> str:
    quoted = urllib.parse.quote(revision, safe="")
    data, _ = _get_json(f"{endpoint}/api/models/{repo}/revision/{quoted}", token)
    sha = data.get("sha") if isinstance(data, dict) else None
    if not isinstance(sha, str) or not _COMMIT.fullmatch(sha):
        raise DownloadError(f"{repo}: could not resolve revision {revision!r} to a commit")
    return sha


def list_tree(endpoint: str, repo: str, commit: str, token: str | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    url: str | None = f"{endpoint}/api/models/{repo}/tree/{commit}?recursive=true"
    while url:
        data, link = _get_json(url, token)
        if not isinstance(data, list):
            raise DownloadError(f"{repo}: unexpected tree listing")
        entries.extend(e for e in data if isinstance(e, dict))
        url = _next_link(link, endpoint)
    return entries


# -- files ------------------------------------------------------------------------------------


def hashes_of(path: Path) -> tuple[str, str]:
    """SHA-256 and git blob id (SHA-1 over ``blob <size>`` NUL plus the content) of a file."""
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest()


def _verdict(digests: tuple[str, str], sha256: str | None, git_sha1: str | None) -> str | None:
    """A message when the file does not match what the Hub published, else ``None``."""
    if sha256 is not None and digests[0] != sha256:
        return "SHA-256 does not match the one the Hub publishes"
    if git_sha1 is not None and digests[1] != git_sha1:
        return "git blob id does not match the one the Hub publishes"
    return None


def _human(size: int) -> str:
    return f"{size / 1e9:.2f} GB" if size >= 1e8 else f"{size / 1e6:.1f} MB"


def download_file(
    url: str,
    target: Path,
    *,
    size: int,
    sha256: str | None,
    token: str | None,
    label: str,
    git_sha1: str | None = None,
    verify_existing: bool = False,
) -> str:
    """Download ``url`` to ``target`` (resuming a ``.part`` file) and return its SHA-256.

    Large (LFS) files are checked against the SHA-256 the Hub publishes, the others against their
    git blob id. A file that is already there with the right size is kept when it matches;
    ``verify_existing`` re-hashes LFS files too (they are trusted by size otherwise, as hashing
    gigabytes on every run is slow).
    """
    if target.is_file() and target.stat().st_size == size:
        if sha256 is not None and not verify_existing:
            print(f"present      {label}", flush=True)
            return sha256
        digests = hashes_of(target)
        if _verdict(digests, sha256, git_sha1) is None:
            print(f"present      {label}", flush=True)
            return digests[0]
        target.unlink()
    elif target.exists():
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    print(f"downloading  {label} ({_human(size)})", flush=True)
    attempt = 0
    while True:
        offset = part.stat().st_size if part.exists() else 0
        if offset > size:
            part.unlink()
            offset = 0
        if offset == size:
            break
        attempt += 1
        try:
            _stream(url, part, offset, size, token, label)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            if attempt >= MAX_ATTEMPTS:
                raise DownloadError(f"network error while downloading {label}") from None
            print(f"retrying     {label} (attempt {attempt + 1})", flush=True)
            time.sleep(min(2**attempt, 20))

    received = part.stat().st_size  # read before the file is removed
    if received != size:
        part.unlink()
        raise DownloadError(f"{label}: received {received} bytes, expected {size}")
    digests = hashes_of(part)
    problem = _verdict(digests, sha256, git_sha1)
    if problem is not None:
        part.unlink()
        raise DownloadError(f"{label}: {problem}")
    os.replace(part, target)
    return digests[0]


def _stream(url: str, part: Path, offset: int, size: int, token: str | None, label: str) -> None:
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with _request(url, token, headers) as response:
        resumed = offset > 0 and response.status == 206
        mode = "ab" if resumed else "wb"
        done = offset if resumed else 0
        next_mark = (done * 10 // size + 1) if size else 11
        with part.open(mode) as out:
            while chunk := response.read(CHUNK_SIZE):
                out.write(chunk)
                done += len(chunk)
                if size and done * 10 // size >= next_mark and next_mark < 10:
                    print(f"             {label}: {next_mark * 10}%", flush=True)
                    next_mark += 1


def free_space_check(dest: Path, needed: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if free < needed + DISK_MARGIN_BYTES:
        raise DownloadError(
            f"not enough disk space in {dest}: need about {_human(needed + DISK_MARGIN_BYTES)}, "
            f"{_human(free)} free"
        )


# -- orchestration ----------------------------------------------------------------------------


def _pending_bytes(files: Sequence[RemoteFile], folder: Path) -> int:
    total = 0
    for item in files:
        target = folder / item.path
        if not (target.is_file() and target.stat().st_size == item.size):
            part = target.with_name(target.name + ".part")
            total += item.size - (part.stat().st_size if part.exists() else 0)
    return max(total, 0)


def download_model(
    plan: ModelPlan,
    dest: Path,
    *,
    endpoint: str,
    token: str | None,
    verify_existing: bool,
) -> None:
    """Fetch both formats of one model and write its manifest."""
    resolved: dict[str, tuple[str, list[RemoteFile]]] = {}
    for kind, source in (("gguf", plan.gguf), ("onnx", plan.onnx)):
        commit = resolve_commit(endpoint, source.repo, source.revision, token)
        files = select_files(list_tree(endpoint, source.repo, commit, token), source)
        resolved[kind] = (commit, files)

    root = dest / plan.name
    needed = sum(_pending_bytes(files, root / kind) for kind, (_, files) in resolved.items())
    free_space_check(dest, needed)

    manifest: dict[str, Any] = {"name": plan.name}
    for kind, source in (("gguf", plan.gguf), ("onnx", plan.onnx)):
        commit, files = resolved[kind]
        folder = root / kind
        recorded = []
        for item in files:
            quoted = urllib.parse.quote(item.remote_path)
            url = f"{endpoint}/{source.repo}/resolve/{commit}/{quoted}"
            digest = download_file(
                url,
                folder / item.path,
                size=item.size,
                sha256=item.sha256,
                git_sha1=item.git_sha1,
                token=token,
                label=f"{plan.name}/{kind}/{item.path}",
                verify_existing=verify_existing,
            )
            recorded.append({"path": item.path, "size": item.size, "sha256": digest})
        manifest[kind] = {
            "repo": source.repo,
            "commit": commit,
            "subdir": source.subdir,
            "quantization": source.quantization,
            "files": recorded,
        }
    manifest["downloaded_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, root / "manifest.json")
    print(f"done         {plan.name}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the benchmark models from Hugging Face.")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("models.download.json")
    )
    parser.add_argument("--dest", type=Path, default=Path("/models"))
    parser.add_argument("--only", action="append", metavar="NAME", help="download only this model")
    parser.add_argument(
        "--verify", action="store_true", help="re-hash files that are already there"
    )
    parser.add_argument("--list", action="store_true", help="print the plan and exit (no network)")
    args = parser.parse_args(argv)
    try:
        plans = load_plan(args.config)
        if args.only:
            unknown = sorted(set(args.only) - {p.name for p in plans})
            if unknown:
                raise DownloadError(f"unknown model(s): {', '.join(unknown)}")
            plans = [p for p in plans if p.name in args.only]
        if args.list:
            for plan in plans:
                print(f"{plan.name}: gguf {plan.gguf.repo}, onnx {plan.onnx.repo}")
            return 0
        endpoint = os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
        token = check_token(os.environ.get("HF_TOKEN") or None, endpoint)
        print(
            f"Hub: {endpoint}; token: {'set' if token else 'not set'}; destination: {args.dest}",
            flush=True,
        )
        for plan in plans:
            download_model(
                plan, args.dest, endpoint=endpoint, token=token, verify_existing=args.verify
            )
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
