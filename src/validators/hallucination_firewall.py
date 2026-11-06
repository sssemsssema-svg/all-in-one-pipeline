from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from pydantic import Field, field_validator

from validators.schemas import FrozenModel, StrictModel

_NAME_RE = re.compile(r"^([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9])$")
MAX_DEPS = 50
MAX_JSON_BYTES = 5 * 1024 * 1024
MIN_PACKAGE_AGE_DAYS = 14


class Verdict(StrEnum):
    VERIFIED = "verified"
    REJECTED = "rejected"  # positive evidence the dependency is bad / hallucinated
    UNVERIFIABLE = "unverifiable"  # could not establish either way (fail-closed)


class DependencySpec(FrozenModel):
    name: str = Field(min_length=1, max_length=200)
    version: str | None = Field(default=None, max_length=100)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not _NAME_RE.fullmatch(value):
            raise ValueError(f"Invalid Python package name: {value!r}")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                Version(value)
            except InvalidVersion:
                raise ValueError(f"Invalid PEP 440 version: {value!r}") from None
        return value


class PackageVerification(StrictModel):
    name: str
    verdict: Verdict
    reason: str | None = None
    resolved_version: str | None = None

    @property
    def verified(self) -> bool:
        return self.verdict is Verdict.VERIFIED


class FirewallReport(StrictModel):
    verified: list[PackageVerification] = Field(default_factory=list)
    rejected: list[PackageVerification] = Field(default_factory=list)
    unverifiable: list[PackageVerification] = Field(default_factory=list)

    @property
    def all_clear(self) -> bool:
        return not self.rejected and not self.unverifiable


class _Transient(Exception):
    """Retryable condition (timeout, 429, 5xx)."""


class _Definitive(Exception):
    """Final, non-retryable outcome for one dependency."""

    def __init__(self, verdict: Verdict, reason: str) -> None:
        super().__init__(reason)
        self.verdict = verdict
        self.reason = reason


def _parse_upload_time(file_info: Any) -> datetime | None:
    if not isinstance(file_info, dict):
        return None
    raw = file_info.get("upload_time_iso_8601") or file_info.get("upload_time")
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _earliest_upload(releases: dict[str, Any]) -> datetime | None:
    times = [
        stamp
        for files in releases.values()
        if isinstance(files, list)
        for file_info in files
        if (stamp := _parse_upload_time(file_info)) is not None
    ]
    return min(times, default=None)


def _same_version(candidate: str, target: Version) -> bool:
    try:
        return Version(candidate) == target
    except InvalidVersion:
        return False


class HallucinationFirewall:
    """Fail-closed PyPI existence / age / yank verification for LLM-proposed dependencies."""

    def __init__(
        self,
        *,
        base_url: str = "https://pypi.org/pypi",
        timeout: float = 10.0,
        concurrency: int = 5,
        retries: int = 2,
        backoff_seconds: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        parts = urlsplit(base_url)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError("base_url must be a plain absolute https:// URL")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._concurrency = max(1, concurrency)
        self._retries = max(0, retries)
        self._backoff = max(0.0, backoff_seconds)
        self._transport = transport

    async def verify(self, deps: Iterable[DependencySpec]) -> FirewallReport:
        unique: dict[tuple[str, str | None], DependencySpec] = {}
        for dep in deps:
            version_key = str(Version(dep.version)) if dep.version else None
            unique.setdefault((canonicalize_name(dep.name), version_key), dep)
        items = list(unique.values())
        accepted, overflow = items[:MAX_DEPS], items[MAX_DEPS:]

        report = FirewallReport()
        report.rejected.extend(
            PackageVerification(
                name=dep.name,
                verdict=Verdict.REJECTED,
                reason=f"Dependency limit of {MAX_DEPS} exceeded",
            )
            for dep in overflow
        )
        if not accepted:
            return report

        buckets = {
            Verdict.VERIFIED: report.verified,
            Verdict.REJECTED: report.rejected,
            Verdict.UNVERIFIABLE: report.unverifiable,
        }
        limits = httpx.Limits(
            max_connections=self._concurrency, max_keepalive_connections=self._concurrency
        )
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            limits=limits,
            follow_redirects=False,
            headers={"Accept": "application/json", "User-Agent": "zero-touch-firewall/0.2"},
            transport=self._transport,
        ) as client:
            semaphore = asyncio.Semaphore(self._concurrency)
            results = await asyncio.gather(
                *(self._verify_with_retry(client, semaphore, dep) for dep in accepted)
            )
        for result in results:
            buckets[result.verdict].append(result)
        return report

    async def _verify_with_retry(
        self, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, dep: DependencySpec
    ) -> PackageVerification:
        reason = "unknown error"
        for attempt in range(self._retries + 1):
            try:
                async with semaphore:
                    return await self._verify_once(client, dep)
            except _Definitive as exc:
                return PackageVerification(name=dep.name, verdict=exc.verdict, reason=exc.reason)
            except _Transient as exc:
                reason = str(exc)
            except httpx.TransportError as exc:
                reason = f"Network error: {type(exc).__name__}"
            except Exception as exc:
                return PackageVerification(
                    name=dep.name,
                    verdict=Verdict.UNVERIFIABLE,
                    reason=f"Unexpected {type(exc).__name__}",
                )
            if attempt < self._retries:
                await asyncio.sleep(self._backoff * 2**attempt)
        return PackageVerification(
            name=dep.name,
            verdict=Verdict.UNVERIFIABLE,
            reason=f"{reason} (after {self._retries + 1} attempts)",
        )

    async def _verify_once(
        self, client: httpx.AsyncClient, dep: DependencySpec
    ) -> PackageVerification:
        # PyPI 301-redirects non-canonical names; query with the canonical name instead.
        url = f"{self._base_url}/{quote(canonicalize_name(dep.name), safe='')}/json"
        payload = await self._fetch_json(client, url)
        return self._evaluate(dep, payload)

    @staticmethod
    async def _fetch_json(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
        async with client.stream("GET", url) as response:
            status = response.status_code
            if status == 404:
                raise _Definitive(Verdict.REJECTED, "Not found on PyPI")
            if status == 429 or status >= 500:
                raise _Transient(f"HTTP {status}")
            if 300 <= status < 400:
                raise _Definitive(Verdict.UNVERIFIABLE, "Redirects are blocked for security")
            if status != 200:
                raise _Definitive(Verdict.UNVERIFIABLE, f"HTTP {status}")
            declared = response.headers.get("Content-Length", "")
            if declared.isdigit() and int(declared) > MAX_JSON_BYTES:
                raise _Definitive(Verdict.UNVERIFIABLE, "PyPI response exceeds size limit")
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_JSON_BYTES:
                    raise _Definitive(Verdict.UNVERIFIABLE, "PyPI response exceeded size limit")
                chunks.append(chunk)
        try:
            payload = json.loads(b"".join(chunks))
        except ValueError:
            raise _Definitive(Verdict.UNVERIFIABLE, "Invalid JSON from PyPI") from None
        if not isinstance(payload, dict):
            raise _Definitive(Verdict.UNVERIFIABLE, "Unexpected PyPI metadata shape")
        return payload

    @staticmethod
    def _evaluate(dep: DependencySpec, payload: dict[str, Any]) -> PackageVerification:
        info, releases = payload.get("info"), payload.get("releases")
        if not isinstance(info, dict) or not isinstance(releases, dict):
            raise _Definitive(Verdict.UNVERIFIABLE, "Unexpected PyPI metadata shape")

        first_upload = _earliest_upload(releases)
        if first_upload is None:
            raise _Definitive(Verdict.REJECTED, "No uploaded release files found")
        if (datetime.now(UTC) - first_upload).days < MIN_PACKAGE_AGE_DAYS:
            raise _Definitive(
                Verdict.REJECTED,
                f"Package is younger than {MIN_PACKAGE_AGE_DAYS} days (slopsquatting protection)",
            )

        if dep.version is None:
            latest = info.get("version")
            if not isinstance(latest, str) or not latest:
                raise _Definitive(Verdict.UNVERIFIABLE, "No version in PyPI info")
            resolved = latest
        else:
            target = Version(dep.version)
            files = next(
                (f for key, f in releases.items() if _same_version(key, target)), None
            )
            if not isinstance(files, list) or not files:
                raise _Definitive(Verdict.REJECTED, f"Version {dep.version} is not published")
            if any(isinstance(f, dict) and f.get("yanked", False) for f in files):
                raise _Definitive(Verdict.REJECTED, "Release is yanked")
            resolved = dep.version
        return PackageVerification(
            name=dep.name, verdict=Verdict.VERIFIED, reason="Verified", resolved_version=resolved
        )
