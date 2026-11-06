from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import jsonpatch
from jsonpointer import JsonPointerException

from orchestrator.quota_governor import QuotaGovernor, QuotaGovernorError
from validators.schemas import (
    MAX_PATCH_OPS,
    AgentDelta,
    Blueprint,
    GatekeeperDecision,
    GatekeeperState,
    JSONPatchOp,
    Phase,
    PipelineState,
    ProviderSpec,
    Stage,
)

_STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)
MAX_REVISIONS_PER_STAGE = 3
MAX_PROVIDER_ATTEMPTS = 3
DEFAULT_ESTIMATED_TOKENS = 4096
MAX_BLUEPRINT_BYTES = 256 * 1024
_IMMUTABLE_FIELDS = ("blueprint_id", "goal", "revision")
_PATCH_ERRORS: tuple[type[BaseException], ...] = (
    jsonpatch.JsonPatchException,
    JsonPointerException,
    LookupError,
    TypeError,
    ValueError,  # includes pydantic.ValidationError
    AssertionError,  # JsonPatchTestFailed
    RecursionError,
)


class FSMError(Exception):
    pass


class PatchRejectedError(FSMError):
    """A patch set was refused; blueprint state is guaranteed unchanged."""


class PatchAuthorizationError(PatchRejectedError):
    pass


class AgentError(FSMError):
    """Recoverable agent failure (transport, parsing, schema). Triggers provider failover."""

    def __init__(self, message: str, *, tokens: int = 0, cooldown_seconds: float = 0.0) -> None:
        super().__init__(message)
        self.tokens = max(0, tokens)
        self.cooldown_seconds = max(0.0, cooldown_seconds)


class AgentCallable(Protocol):
    def __call__(
        self, stage: Stage, provider: ProviderSpec, blueprint: Blueprint
    ) -> tuple[AgentDelta, int]: ...


# --------------------------------------------------------------------------- #
# Patch authorization
# --------------------------------------------------------------------------- #


def _decode_pointer(pointer: str) -> tuple[str, ...]:
    return tuple(
        segment.replace("~1", "/").replace("~0", "~") for segment in pointer[1:].split("/")
    )


def writable_prefixes(stage: Stage) -> tuple[str, ...]:
    """The ONLY blueprint subtrees a stage may modify (allowlist, not denylist)."""
    return ("/steps", f"/metadata/{stage.value}")


def _is_within(path: tuple[str, ...], prefixes: Sequence[tuple[str, ...]]) -> bool:
    return any(path[: len(prefix)] == prefix for prefix in prefixes)


def _authorize_patches(stage: Stage, ops: Sequence[JSONPatchOp]) -> None:
    if len(ops) > MAX_PATCH_OPS:
        raise PatchRejectedError(f"Too many patch operations (max {MAX_PATCH_OPS})")
    allowed = [_decode_pointer(prefix) for prefix in writable_prefixes(stage)]
    for op in ops:
        if not _is_within(_decode_pointer(op.path), allowed):
            raise PatchAuthorizationError(
                f"Stage {stage.value!r} may not modify {op.path!r}; "
                f"writable prefixes: {list(writable_prefixes(stage))}"
            )
        # 'move' deletes its source, so the source must be writable too.
        # 'copy' only reads its source, which may be anywhere in the blueprint.
        if op.op == "move" and op.from_ is not None:
            if not _is_within(_decode_pointer(op.from_), allowed):
                raise PatchAuthorizationError(
                    f"Stage {stage.value!r} may not move from {op.from_!r}"
                )


def _apply_ops(blueprint: Blueprint, ops: Sequence[JSONPatchOp], stage: Stage) -> Blueprint:
    """Authorize and atomically apply ops; returns a NEW blueprint with revision + 1."""
    _authorize_patches(stage, ops)
    current = blueprint.model_dump(mode="json")
    try:
        patched = jsonpatch.apply_patch(
            current, [op.to_jsonpatch() for op in ops], in_place=False
        )
        if not isinstance(patched, dict):
            raise PatchRejectedError("Patch produced a non-object blueprint")
        for field in _IMMUTABLE_FIELDS:
            if patched.get(field) != current[field]:
                raise PatchAuthorizationError(f"Patch attempted to change immutable field {field!r}")
        patched["revision"] = blueprint.revision + 1
        updated = Blueprint.model_validate(patched)
    except FSMError:
        raise
    except _PATCH_ERRORS as exc:
        raise PatchRejectedError(f"JSON patch rejected: {exc}") from exc
    if len(json.dumps(updated.model_dump(mode="json")).encode("utf-8")) > MAX_BLUEPRINT_BYTES:
        raise PatchRejectedError(f"Blueprint would exceed {MAX_BLUEPRINT_BYTES} bytes")
    return updated


# --------------------------------------------------------------------------- #
# FSM
# --------------------------------------------------------------------------- #


class PipelineFSM:
    """Deterministic creative -> architect -> coder state machine.

    Not thread-safe: a single orchestrator owns a given FSM instance.
    """

    def __init__(
        self,
        blueprint: Blueprint,
        governor: QuotaGovernor,
        *,
        max_revisions_per_stage: int = MAX_REVISIONS_PER_STAGE,
        max_provider_attempts: int = MAX_PROVIDER_ATTEMPTS,
        estimated_tokens: int = DEFAULT_ESTIMATED_TOKENS,
    ) -> None:
        if max_revisions_per_stage < 0:
            raise ValueError("max_revisions_per_stage must not be negative")
        if max_provider_attempts < 1:
            raise ValueError("max_provider_attempts must be at least 1")
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens must not be negative")
        self.governor = governor
        self.max_revisions_per_stage = max_revisions_per_stage
        self.max_provider_attempts = max_provider_attempts
        self.estimated_tokens = estimated_tokens
        self._revision_attempts: dict[Stage, int] = dict.fromkeys(_STAGE_ORDER, 0)
        self._last_delta: AgentDelta | None = None
        self.state = PipelineState(
            stage=_STAGE_ORDER[0], phase=Phase.RUNNING, blueprint=blueprint.model_copy(deep=True)
        )

    # ---------------------------------------------------------- properties --

    @property
    def stage(self) -> Stage:
        return self.state.stage

    @property
    def phase(self) -> Phase:
        return self.state.phase

    @property
    def last_delta(self) -> AgentDelta | None:
        return self._last_delta

    # ------------------------------------------------------------- helpers --

    def _require_running(self) -> None:
        if self.state.phase is not Phase.RUNNING:
            raise FSMError(f"Pipeline is not running (phase={self.state.phase.value})")

    def _log(self, event: str, **fields: Any) -> None:
        self.state.history.append({"event": event, "stage": self.state.stage.value, **fields})

    def fail(self, reason: str) -> None:
        """Terminally fail the pipeline. Idempotent; the first failure reason wins."""
        if self.state.phase is not Phase.RUNNING:
            return
        self.state.phase = Phase.FAILED
        self.state.error = reason
        self._log("pipeline_failed", error=reason)

    # --------------------------------------------------------------- deltas --

    def apply_delta(self, delta: AgentDelta) -> Blueprint:
        self._require_running()
        stage = self.state.stage
        if delta.stage != stage:
            raise FSMError(
                f"Stage mismatch: current stage is {stage.value!r}, delta stage is {delta.stage.value!r}"
            )
        if delta.has_blockers:
            self._last_delta = delta
            self._log(
                "delta_blocked",
                blockers=[c.message for c in delta.critiques if c.severity == "blocker"],
            )
            return self.state.blueprint
        if delta.patches:
            self.state.blueprint = _apply_ops(self.state.blueprint, delta.patches, stage)
        self._last_delta = delta
        self._log(
            "delta_applied", revision=self.state.blueprint.revision, patch_count=len(delta.patches)
        )
        return self.state.blueprint

    def run_stage(self, agent_fn: AgentCallable) -> AgentDelta:
        """Run the current stage, failing over across providers on recoverable errors."""
        self._require_running()
        stage = self.state.stage
        last_error: FSMError | None = None
        for attempt in range(1, self.max_provider_attempts + 1):
            try:
                reservation = self.governor.reserve_provider(
                    stage, estimated_tokens=self.estimated_tokens
                )
            except QuotaGovernorError as exc:
                self.fail(f"{type(exc).__name__}: {exc}")
                raise
            tokens, cooldown = 0, 0.0
            try:
                delta, tokens = agent_fn(
                    stage, reservation.provider, self.state.blueprint.model_copy(deep=True)
                )
                if tokens < 0:
                    raise AgentError("Agent reported a negative token count")
                if delta.stage != stage:
                    raise AgentError("Agent returned a delta for the wrong stage", tokens=tokens)
                self.apply_delta(delta)
            except AgentError as exc:
                tokens, cooldown, last_error = exc.tokens, exc.cooldown_seconds, exc
            except PatchRejectedError as exc:
                last_error = exc
            except BaseException as exc:
                self.fail(f"{type(exc).__name__}: {exc}")
                raise
            else:
                return delta
            finally:
                self.governor.settle(reservation, tokens=tokens, cooldown_seconds=cooldown)
            self._log(
                "provider_attempt_failed",
                provider=reservation.provider.name,
                attempt=attempt,
                error=f"{type(last_error).__name__}: {last_error}",
            )
        message = f"All {self.max_provider_attempts} provider attempts failed: {last_error}"
        self.fail(message)
        raise FSMError(message) from last_error

    # ---------------------------------------------------------------- gate --

    def evaluate_gate(self, decision: GatekeeperState) -> Phase:
        self._require_running()
        stage = self.state.stage
        if decision.stage != stage:
            raise FSMError("Gatekeeper stage does not match current stage")
        match decision.decision:
            case GatekeeperDecision.APPROVE:
                self._approve(decision)
            case GatekeeperDecision.REVISE:
                self._request_revision(decision)
            case GatekeeperDecision.REJECT:
                self.state.gatekeeper = decision
                self.fail(decision.reason)
        return self.state.phase

    def _approve(self, decision: GatekeeperState) -> None:
        stage = self.state.stage
        if self._last_delta is None:
            raise FSMError("Cannot approve a stage that has not produced a delta")
        if self._last_delta.has_blockers:
            raise FSMError("Gatekeeper cannot approve a delta containing blocker critiques")
        if decision.required_patches:
            self.state.blueprint = _apply_ops(
                self.state.blueprint, decision.required_patches, stage
            )
        self.state.gatekeeper = decision
        self._revision_attempts[stage] = 0
        self._log("stage_approved", revision=self.state.blueprint.revision)
        self._advance()

    def _request_revision(self, decision: GatekeeperState) -> None:
        stage = self.state.stage
        attempts = self._revision_attempts[stage] + 1
        self._revision_attempts[stage] = attempts
        self.state.gatekeeper = decision
        self._last_delta = None  # force a fresh agent run before any further approval
        self._log("revision_requested", attempt=attempts, reason=decision.reason)
        if attempts > self.max_revisions_per_stage:
            self.fail(f"Maximum revisions exceeded for stage {stage.value}")

    def _advance(self) -> None:
        next_stage = self._next_stage(self.state.stage)
        if next_stage is None:
            self.state.phase = Phase.COMPLETED
            self._log("pipeline_completed")
            return
        self.state.stage = next_stage
        self.state.gatekeeper = None
        self._last_delta = None
        self._log("stage_advanced")

    @staticmethod
    def _next_stage(stage: Stage) -> Stage | None:
        index = _STAGE_ORDER.index(stage) + 1
        return _STAGE_ORDER[index] if index < len(_STAGE_ORDER) else None


def make_blueprint(
    blueprint_id: str, goal: str, steps: Sequence[Mapping[str, Any]] | None = None
) -> Blueprint:
    return Blueprint.model_validate(
        {
            "blueprint_id": blueprint_id,
            "goal": goal,
            "steps": [dict(step) for step in steps or ()],
            "revision": 0,
            "metadata": {"creative": {}, "architect": {}, "coder": {}},
        }
    )
