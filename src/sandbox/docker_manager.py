from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import docker
import requests
from docker.errors import DockerException, ImageNotFound
from docker.models.containers import Container
from docker.types import LogConfig, Ulimit
from pydantic import Field, field_validator

from validators.schemas import FrozenModel, StrictModel

MAX_FILES = 200
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 5 * 1024 * 1024
MAX_PATH_CHARS = 255
MAX_SEGMENT_CHARS = 100
MAX_LOG_BYTES = 64 * 1024
_DIGEST_RE = re.compile(r"^[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}$")
_FORBIDDEN_PATH_CHARS_RE = re.compile(r'[\x00-\x1f\x7f\\:*?"<>|]')

# Executed with `python -I` so nothing in /workspace can shadow the stdlib (e.g. ast.py).
_CONTAINER_CHECK_SCRIPT = """\
import ast, pathlib, sys
root = pathlib.Path('/workspace')
errors = []
files = sorted(p for p in root.rglob('*.py') if p.is_file() and not p.is_symlink())
for path in files:
    rel = path.relative_to(root)
    try:
        ast.parse(path.read_bytes(), filename=str(rel))
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        errors.append(f'{rel}: {type(exc).__name__}: {exc}')
if errors:
    print('\\n'.join(errors), file=sys.stderr)
    sys.exit(1)
print(f'Parsed {len(files)} Python files')
"""


class SandboxLimits(FrozenModel):
    cpu_quota: int = Field(default=50_000, ge=1_000, le=4_000_000)
    cpu_period: int = Field(default=100_000, ge=1_000, le=1_000_000)
    mem_limit: str = Field(default="512m", pattern=r"^[1-9][0-9]*(k|m|g)$")
    pids_limit: int = Field(default=128, ge=1, le=4096)
    timeout_seconds: int = Field(default=180, ge=1, le=1800)


def _normalize_relative_path(raw: str) -> str:
    """Strict, platform-independent relative path validation (no translation, only rejection)."""
    if not raw or len(raw) > MAX_PATH_CHARS:
        raise ValueError(f"Path must be 1-{MAX_PATH_CHARS} characters")
    if _FORBIDDEN_PATH_CHARS_RE.search(raw):
        raise ValueError(f"Path contains forbidden characters: {raw!r}")
    segments = raw.split("/")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise ValueError(f"Absolute, empty, or traversal segment forbidden in {raw!r}")
        if len(segment) > MAX_SEGMENT_CHARS:
            raise ValueError(f"Path segment too long in {raw!r}")
    return "/".join(segments)


def _reject_file_directory_conflicts(paths: set[str]) -> None:
    for path in paths:
        parts = path.split("/")
        for index in range(1, len(parts)):
            if "/".join(parts[:index]) in paths:
                raise ValueError(f"Path {path!r} is nested under file {'/'.join(parts[:index])!r}")


class SandboxRequest(StrictModel):
    files: dict[str, str] = Field(min_length=1)
    limits: SandboxLimits = Field(default_factory=SandboxLimits)

    @field_validator("files")
    @classmethod
    def _validate_files(cls, files: dict[str, str]) -> dict[str, str]:
        if len(files) > MAX_FILES:
            raise ValueError(f"At most {MAX_FILES} files are allowed")
        normalized: dict[str, str] = {}
        total = 0
        for supplied_path, content in files.items():
            safe_path = _normalize_relative_path(supplied_path)
            if safe_path in normalized:
                raise ValueError(f"Duplicate path: {supplied_path!r}")
            size = len(content.encode("utf-8"))
            if size > MAX_FILE_BYTES:
                raise ValueError(f"File {supplied_path!r} exceeds {MAX_FILE_BYTES} bytes")
            total += size
            if total > MAX_TOTAL_BYTES:
                raise ValueError(f"Payload exceeds {MAX_TOTAL_BYTES} bytes")
            normalized[safe_path] = content
        _reject_file_directory_conflicts(set(normalized))
        return normalized


class SandboxCheckResult(StrictModel):
    name: Literal["ast", "container"]
    passed: bool
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0


class SandboxReport(StrictModel):
    success: bool
    checks: list[SandboxCheckResult] = Field(default_factory=list)
    error: str | None = None


def _safe_workspace_destination(root: Path, relative_path: str) -> Path:
    normalized = _normalize_relative_path(relative_path)
    root_resolved = root.resolve()
    destination = (root_resolved / normalized).resolve()
    try:
        destination.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {relative_path!r}") from exc
    if destination == root_resolved:
        raise ValueError("Path must identify a file")
    return destination


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _read_logs(container: Container, *, stdout: bool, stderr: bool) -> str:
    """Stream logs with a hard byte cap so a chatty container cannot exhaust host memory."""
    chunks: list[bytes] = []
    size = 0
    for chunk in container.logs(stdout=stdout, stderr=stderr, stream=True, follow=False):
        remaining = MAX_LOG_BYTES - size
        if remaining <= 0:
            break
        chunks.append(chunk[:remaining])
        size += min(len(chunk), remaining)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _best_effort(action: Callable[[], object]) -> None:
    with contextlib.suppress(DockerException, requests.RequestException):
        action()


class HardenedDockerSandbox:
    """Runs a static syntax check of untrusted files inside a locked-down container.

    The host-side AST pass is a cheap fast-fail; the in-container pass re-parses under the
    image's own interpreter version (which may differ from the host's).
    """

    def __init__(
        self,
        *,
        image: str,
        client: docker.DockerClient | None = None,
        require_digest: bool = True,
        seccomp_profile: Path | None = None,
        max_concurrent: int = 4,
    ) -> None:
        image = image.strip()
        if not image:
            raise ValueError("Image required")
        if require_digest and not _DIGEST_RE.fullmatch(image):
            raise ValueError("Images must be pinned as <repo>@sha256:<64 hex chars>")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self._image = image
        self._client = client
        self._owns_client = client is None
        self._client_lock = threading.Lock()
        self._hardening_verified = False
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._security_opt = ["no-new-privileges:true"]
        self._custom_seccomp = False
        if seccomp_profile is not None:
            profile = json.loads(seccomp_profile.read_text(encoding="utf-8"))
            if not isinstance(profile, dict) or "syscalls" not in profile:
                raise ValueError("seccomp profile must be a JSON object with 'syscalls'")
            self._security_opt.append("seccomp=" + json.dumps(profile))
            self._custom_seccomp = True

    # ----------------------------------------------------------------- API --

    def run(self, request: SandboxRequest) -> SandboxReport:
        ast_result = self._run_ast_check(request)
        if not ast_result.passed:
            return SandboxReport(success=False, checks=[ast_result], error="AST validation failed")
        with self._slots:
            return self._run_in_container(request, ast_result)

    async def run_async(self, request: SandboxRequest) -> SandboxReport:
        return await asyncio.to_thread(self.run, request)

    def close(self) -> None:
        with self._client_lock:
            if self._owns_client and self._client is not None:
                self._client.close()
                self._client = None
                self._hardening_verified = False

    def __enter__(self) -> HardenedDockerSandbox:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ----------------------------------------------------------- internals --

    def _get_client(self) -> docker.DockerClient:
        with self._client_lock:
            if self._client is None:
                self._client = docker.from_env()
            if not self._hardening_verified:
                if not self._custom_seccomp:
                    options = self._client.info().get("SecurityOptions", [])
                    if not any("name=seccomp" in str(option) for option in options):
                        raise RuntimeError("Docker daemon reports no seccomp support; refusing to run")
                self._hardening_verified = True
            return self._client

    def _run_in_container(
        self, request: SandboxRequest, ast_result: SandboxCheckResult
    ) -> SandboxReport:
        started = time.monotonic()
        error: str | None = None
        try:
            result = self._execute(request, started)
            if not result.passed:
                error = "Container check failed"
        except TimeoutError as exc:
            error = str(exc)
            result = self._infra_failure(124, error, started)
        except (DockerException, requests.RequestException, OSError, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = self._infra_failure(1, error, started)
        return SandboxReport(success=result.passed, checks=[ast_result, result], error=error)

    @staticmethod
    def _infra_failure(code: int, message: str, started: float) -> SandboxCheckResult:
        return SandboxCheckResult(
            name="container",
            passed=False,
            return_code=code,
            stderr=message,
            duration_ms=_elapsed_ms(started),
        )

    def _execute(self, request: SandboxRequest, started: float) -> SandboxCheckResult:
        workspace = self._materialize(request)
        container: Container | None = None
        try:
            client = self._get_client()
            try:
                client.images.get(self._image)
            except ImageNotFound as exc:
                raise RuntimeError(f"Image {self._image} is missing locally (no auto-pull).") from exc
            container = client.containers.create(**self._container_kwargs(request, workspace))
            container.start()
            try:
                wait_result = container.wait(timeout=request.limits.timeout_seconds)
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                _best_effort(container.kill)
                raise TimeoutError(
                    f"Execution exceeded {request.limits.timeout_seconds}s "
                    "(or the Docker daemon connection was lost)"
                ) from exc
            return_code = int(wait_result.get("StatusCode", 1))
            return SandboxCheckResult(
                name="container",
                passed=return_code == 0,
                return_code=return_code,
                stdout=_read_logs(container, stdout=True, stderr=False),
                stderr=_read_logs(container, stdout=False, stderr=True),
                duration_ms=_elapsed_ms(started),
            )
        finally:
            if container is not None:
                _best_effort(lambda: container.remove(force=True, v=True))
            shutil.rmtree(workspace, ignore_errors=True)

    def _container_kwargs(self, request: SandboxRequest, workspace: Path) -> dict[str, Any]:
        limits = request.limits
        return {
            "image": self._image,
            "command": ["python", "-I", "-B", "-c", _CONTAINER_CHECK_SCRIPT],
            "working_dir": "/",
            "user": "65534:65534",
            "network_mode": "none",
            "network_disabled": True,
            "ipc_mode": "none",
            "read_only": True,
            "privileged": False,
            "cap_drop": ["ALL"],
            "security_opt": list(self._security_opt),
            "mem_limit": limits.mem_limit,
            "memswap_limit": limits.mem_limit,  # equal to mem_limit => swap disabled
            "cpu_period": limits.cpu_period,
            "cpu_quota": limits.cpu_quota,
            "pids_limit": limits.pids_limit,
            "ulimits": [
                Ulimit(name="nofile", soft=256, hard=256),
                Ulimit(name="fsize", soft=16 * 1024 * 1024, hard=16 * 1024 * 1024),
                Ulimit(name="core", soft=0, hard=0),
            ],
            "volumes": {str(workspace): {"bind": "/workspace", "mode": "ro"}},
            "tmpfs": {"/tmp": "rw,noexec,nosuid,nodev,size=64m"},  # noqa: S108
            "log_config": LogConfig(
                type=LogConfig.types.JSON, config={"max-size": "1m", "max-file": "1"}
            ),
            "labels": {"zero-touch.sandbox": "1"},
            "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
            "stdin_open": False,
            "tty": False,
            "auto_remove": False,
        }

    @staticmethod
    def _materialize(request: SandboxRequest) -> Path:
        workspace = Path(tempfile.mkdtemp(prefix="zero_touch_sandbox_")).resolve()
        try:
            for relative_path, content in request.files.items():
                destination = _safe_workspace_destination(workspace, relative_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive create: never overwrite or follow anything pre-existing.
                with destination.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(content)
            # mkdtemp is 0700; the sandbox user (nobody) needs read/traverse on the ro mount.
            workspace.chmod(0o755)
            for path in workspace.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)
        except BaseException:
            shutil.rmtree(workspace, ignore_errors=True)
            raise
        return workspace

    @staticmethod
    def _run_ast_check(request: SandboxRequest) -> SandboxCheckResult:
        errors: list[str] = []
        for relative_path, content in request.files.items():
            if not relative_path.lower().endswith(".py"):
                continue
            try:
                ast.parse(content, filename=relative_path)
            except SyntaxError as exc:
                errors.append(f"{relative_path}:{exc.lineno or 0}: {exc.msg}")
            except (ValueError, RecursionError, MemoryError) as exc:
                errors.append(f"{relative_path}: {type(exc).__name__}")
        return SandboxCheckResult(
            name="ast",
            passed=not errors,
            return_code=1 if errors else 0,
            stderr="\n".join(errors),
        )
