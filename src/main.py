from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx

from agents.prompts import build_reducer_prompt, parse_agent_delta
from orchestrator.fsm import AgentCallable, AgentError, FSMError, PipelineFSM, make_blueprint
from orchestrator.quota_governor import QuotaGovernor, QuotaGovernorError
from validators.hallucination_firewall import DependencySpec, HallucinationFirewall
from validators.schemas import (
    ACCOUNT_ID_PLACEHOLDER,
    AgentDelta,
    ApiStyle,
    Blueprint,
    GatekeeperDecision,
    GatekeeperState,
    Phase,
    ProviderSpec,
    Stage,
)

log = logging.getLogger("zero_touch")

REQUEST_TIMEOUT_SECONDS = 120.0
MAX_PIPELINE_CALLS = 15
MAX_REVISIONS_PER_STAGE = 3
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
SYSTEM_PROMPT = "You are a precise autonomous agent. Follow the user's schema exactly."
_ACCOUNT_ID_RE = re.compile(r"^[A-Za-z0-9]{1,64}$")

STAGE_INSTRUCTIONS: dict[Stage, str] = {
    Stage.CREATIVE: (
        "Explore objective thoroughly. Extract all explicit user constraints (e.g. named tools like Bing, "
        "modular tabs, continuous voice streaming, typography preservation) into metadata.requirements. "
        "Propose a resilient plan adhering strictly to the Engineering Quality Contracts."
    ),
    Stage.ARCHITECT: (
        "Review blueprint as a principal systems architect. Verify 100% of user constraints in "
        "metadata.requirements have dedicated architectural steps. Forbid regex for NLP, "
        "enforce persistent media streams with silence recovery, and require modular UI tabs for multi-purpose features."
    ),
    Stage.CODER: (
        "Review blueprint as a principal implementation engineer. "
        "Turn requirements into hardened, production-grade code. Enforce zero-fail mobile fallbacks, "
        "LLM-backed intent extractors, prompt preservation layers, and verified interactive UI."
    ),
}


def _default_config_path() -> Path:
    if env := os.getenv("ZERO_TOUCH_CONFIG"):
        return Path(env)
    candidates = (Path.cwd() / "config.yaml", Path(__file__).resolve().parents[1] / "config.yaml")
    return next((path for path in candidates if path.is_file()), candidates[0])


# --------------------------------------------------------------------------- #
# Provider transport
# --------------------------------------------------------------------------- #


class ProviderRequestError(AgentError):
    """Recoverable provider failure; the FSM fails over to the next provider."""


def _resolve_endpoint(provider: ProviderSpec) -> str:
    endpoint = provider.endpoint
    if ACCOUNT_ID_PLACEHOLDER in endpoint:
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        if not _ACCOUNT_ID_RE.fullmatch(account_id):
            raise ProviderRequestError(
                "CLOUDFLARE_ACCOUNT_ID is missing or malformed", cooldown_seconds=3600.0
            )
        endpoint = endpoint.replace(ACCOUNT_ID_PLACEHOLDER, account_id)
    if provider.api_style is ApiStyle.CLOUDFLARE and not endpoint.rstrip("/").endswith(
        provider.model
    ):
        endpoint = f"{endpoint.rstrip('/')}/{provider.model}"
    return endpoint


def _build_request(
    provider: ProviderSpec, api_key: str, prompt: str
) -> tuple[dict[str, str], dict[str, Any]]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    match provider.api_style:
        case ApiStyle.GEMINI:
            headers["x-goog-api-key"] = api_key  # never in the URL (logs, proxies, referers)
            body: dict[str, Any] = {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
            }
        case ApiStyle.CLOUDFLARE:
            headers["Authorization"] = f"Bearer {api_key}"
            body = {"messages": chat, "max_tokens": 4096, "temperature": 0.2}
        case _:
            headers["Authorization"] = f"Bearer {api_key}"
            body = {"model": provider.model, "messages": chat, "temperature": 0.2}
    return headers, body


def _dig(data: Any, *path: str | int) -> Any:
    for key in path:
        try:
            data = data[key]
        except (KeyError, IndexError, TypeError):
            location = "/".join(map(str, path))
            raise ProviderRequestError(f"Unexpected response shape at {location}") from None
    return data


def _extract_text(style: ApiStyle, payload: dict[str, Any]) -> str:
    match style:
        case ApiStyle.GEMINI:
            parts = _dig(payload, "candidates", 0, "content", "parts")
            texts = [p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
            if not texts:
                raise ProviderRequestError("Gemini response contains no text")
            return "\n".join(texts)
        case ApiStyle.CLOUDFLARE:
            text = _dig(payload, "result", "response")
        case _:
            text = _dig(payload, "choices", 0, "message", "content")
            if isinstance(text, list):
                text = "\n".join(
                    item["text"] for item in text if isinstance(item, dict) and isinstance(item.get("text"), str)
                )
    if not isinstance(text, str) or not text.strip():
        raise ProviderRequestError("Provider response contains no textual message")
    return text


def _reported_tokens(style: ApiStyle, payload: dict[str, Any]) -> int:
    if style is ApiStyle.GEMINI:
        usage = payload.get("usageMetadata")
        total = usage.get("totalTokenCount") if isinstance(usage, dict) else None
        return total if isinstance(total, int) and total > 0 else 0
    usage = payload.get("usage")
    if not isinstance(usage, dict) and isinstance(payload.get("result"), dict):
        usage = payload["result"].get("usage")
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, int) and total > 0:
        return total
    prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        return prompt_tokens + completion_tokens
    return 0


def _cooldown_for(response: httpx.Response) -> float:
    status = response.status_code
    if status == 429:
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.replace(".", "", 1).isdigit():
            return min(float(retry_after), 900.0)
        return 60.0
    if status in (401, 403):
        return 3600.0  # credentials/permissions problem: do not hammer this provider
    return 30.0 if status >= 500 else 0.0


class ProviderClient:
    """Thin, size-capped, redirect-free HTTP client shared across one pipeline run."""

    def __init__(self, *, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=20.0), follow_redirects=False
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ProviderClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def complete(self, provider: ProviderSpec, prompt: str) -> tuple[str, int]:
        api_key = os.getenv(provider.api_key_env, "").strip()
        if not api_key:
            raise ProviderRequestError(
                f"Missing API key environment variable: {provider.api_key_env}",
                cooldown_seconds=3600.0,
            )
        url = _resolve_endpoint(provider)
        headers, body = _build_request(provider, api_key, prompt)
        try:
            with self._client.stream("POST", url, headers=headers, json=body) as response:
                raw = self._read_capped(response)
                if not response.is_success:
                    snippet = raw[:200].decode("utf-8", errors="replace").replace(api_key, "***")
                    raise ProviderRequestError(
                        f"HTTP {response.status_code}: {snippet}",
                        cooldown_seconds=_cooldown_for(response),
                    )
        except httpx.TransportError as exc:
            raise ProviderRequestError(
                f"Transport error: {type(exc).__name__}", cooldown_seconds=30.0
            ) from exc
        try:
            payload = json.loads(raw)
        except ValueError:
            raise ProviderRequestError("Provider returned invalid JSON") from None
        if not isinstance(payload, dict):
            raise ProviderRequestError("Provider returned a non-object JSON document")
        text = _extract_text(provider.api_style, payload)
        tokens = _reported_tokens(provider.api_style, payload) or (len(prompt) + len(text)) // 4 + 1
        return text, tokens

    @staticmethod
    def _read_capped(response: httpx.Response) -> bytes:
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise ProviderRequestError("Provider response exceeds the size limit")
            chunks.append(chunk)
        return b"".join(chunks)


# --------------------------------------------------------------------------- #
# Agent + gatekeeper + pipeline
# --------------------------------------------------------------------------- #


def make_agent_callable(objective: str, provider_client: ProviderClient) -> AgentCallable:
    def agent(stage: Stage, provider: ProviderSpec, blueprint: Blueprint) -> tuple[AgentDelta, int]:
        prompt = build_reducer_prompt(
            role=f"{stage.value} specialist",
            task=f"Objective: {objective}\n\n{STAGE_INSTRUCTIONS[stage]}",
            blueprint=blueprint,
            stage=stage,
        )
        raw_text, tokens = provider_client.complete(provider, prompt)
        try:
            delta = parse_agent_delta(raw_text, expected_stage=stage)
        except ValueError as exc:  # includes pydantic.ValidationError
            raise AgentError(
                f"Invalid agent output from {provider.name}: {str(exc)[:500]}", tokens=tokens
            ) from exc
        return delta, tokens

    return agent


def gatekeeper_decision(stage: Stage, delta: AgentDelta, attempt: int) -> GatekeeperState:
    for severity, label in (("blocker", "Blockers"), ("error", "Errors")):
        messages = [c.message for c in delta.critiques if c.severity == severity]
        if messages:
            return GatekeeperState(
                stage=stage,
                decision=GatekeeperDecision.REVISE,
                reason=f"{label}: {'; '.join(messages)}",
            )
    return GatekeeperState(
        stage=stage, decision=GatekeeperDecision.APPROVE, reason=f"Approved after attempt {attempt}."
    )


def run_pipeline(objective: str, config_path: Path) -> int:
    objective = objective.strip()
    if not objective:
        raise ValueError("Objective must not be blank")
    governor = QuotaGovernor.from_yaml(config_path)
    fsm = PipelineFSM(
        make_blueprint(uuid.uuid4().hex, objective),
        governor,
        max_revisions_per_stage=MAX_REVISIONS_PER_STAGE,
    )
    stage_attempts = dict.fromkeys(Stage, 0)
    log.info("PIPELINE START: %s", objective)

    with ProviderClient() as provider_client:
        agent = make_agent_callable(objective, provider_client)
        for call_number in range(1, MAX_PIPELINE_CALLS + 1):
            if fsm.phase is not Phase.RUNNING:
                break
            stage = fsm.stage
            stage_attempts[stage] += 1
            log.info(
                "[%d/%d] stage=%s attempt=%d",
                call_number,
                MAX_PIPELINE_CALLS,
                stage.value,
                stage_attempts[stage],
            )
            try:
                delta = fsm.run_stage(agent)
                gate = gatekeeper_decision(stage, delta, stage_attempts[stage])
                log.info("[gate] %s: %s", gate.decision.value, gate.reason)
                fsm.evaluate_gate(gate)
            except (FSMError, QuotaGovernorError) as exc:
                fsm.fail(f"{type(exc).__name__}: {exc}")  # no-op if already failed
                break
        else:
            fsm.fail("Maximum pipeline calls exceeded")  # no-op unless still running

    log.info("RESULT: %s", fsm.phase.value)
    if fsm.state.error:
        log.error("Error: %s", fsm.state.error)
    if fsm.phase is Phase.COMPLETED:
        print(fsm.state.blueprint.model_dump_json(indent=2))
        return 0
    return 1


# --------------------------------------------------------------------------- #
# Firewall CLI
# --------------------------------------------------------------------------- #


def _parse_requirement(text: str) -> DependencySpec:
    name, _, version = text.partition("==")
    return DependencySpec(name=name.strip(), version=version.strip() or None)


async def _run_firewall_check(requirements: list[str]) -> int:
    report = await HallucinationFirewall().verify(_parse_requirement(r) for r in requirements)
    for item in report.verified:
        print(f"[PASS] {item.name}=={item.resolved_version}")
    for item in report.unverifiable:
        print(f"[UNVERIFIED] {item.name}: {item.reason}")
    for item in report.rejected:
        print(f"[FAIL] {item.name}: {item.reason}")
    return 0 if report.all_clear else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zero-touch")
    sub = parser.add_subparsers(dest="command", required=True)
    firewall = sub.add_parser("firewall", help="verify packages against PyPI (live network call)")
    firewall.add_argument("packages", nargs="*", default=["pydantic"], metavar="NAME[==VERSION]")
    run = sub.add_parser("run", help="run the multi-agent pipeline")
    run.add_argument("objective")
    run.add_argument(
        "--config", type=Path, default=_default_config_path(), help="path to config.yaml"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return run_pipeline(args.objective, args.config.resolve())
        return asyncio.run(_run_firewall_check(args.packages))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        log.error("FATAL: %s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
