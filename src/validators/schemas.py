from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

MAX_STEPS = 200
MAX_PATCH_OPS = 64
MAX_POINTER_CHARS = 512
ACCOUNT_ID_PLACEHOLDER = "{account_id}"


class StrictModel(BaseModel):
    """Base for every mutable model: unknown fields are always an error."""

    model_config = ConfigDict(extra="forbid")


class FrozenModel(StrictModel):
    """Base for immutable value objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Stage(StrEnum):
    CREATIVE = "creative"
    ARCHITECT = "architect"
    CODER = "coder"


class Phase(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class GatekeeperDecision(StrEnum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class ApiStyle(StrEnum):
    OPENAI = "openai"
    GEMINI = "gemini"
    CLOUDFLARE = "cloudflare"


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

class ProviderSpec(FrozenModel):
    name: Annotated[
        str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    ]
    model: NonEmptyStr
    endpoint: NonEmptyStr
    api_style: ApiStyle = ApiStyle.OPENAI
    rpm: int = Field(gt=0)
    tpd: int = Field(gt=0)
    api_key_env: Annotated[
        str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Z][A-Z0-9_]*$")
    ]
    tags: tuple[NonEmptyStr, ...] = ()

    @field_validator("endpoint")
    @classmethod
    def _endpoint_is_safe(cls, value: str) -> str:
        probe = urlsplit(value.replace(ACCOUNT_ID_PLACEHOLDER, "account"))
        if probe.scheme != "https" or not probe.hostname:
            raise ValueError("endpoint must be an absolute https:// URL")
        if probe.username or probe.password:
            raise ValueError("endpoint must not embed credentials")
        if probe.query or probe.fragment:
            raise ValueError("endpoint must not carry a query string or fragment")
        if "{" in probe.geturl() or "}" in probe.geturl():
            raise ValueError(f"endpoint contains an unsupported placeholder: {value!r}")
        return value


class RoutingConfig(FrozenModel):
    stages: tuple[Stage, ...]
    tag_to_stage: dict[str, Stage]

    @model_validator(mode="after")
    def _stages_complete(self) -> RoutingConfig:
        if len(set(self.stages)) != len(self.stages) or set(self.stages) != set(Stage):
            raise ValueError("routing.stages must list every Stage exactly once")
        return self


class ProviderInventory(FrozenModel):
    version: NonEmptyStr
    providers: dict[Stage, tuple[ProviderSpec, ...]]
    routing: RoutingConfig

    @model_validator(mode="after")
    def _validate_inventory(self) -> ProviderInventory:
        if set(self.providers) != set(Stage):
            raise ValueError(f"providers must define exactly these stages: {sorted(Stage)}")
        seen: set[str] = set()
        for stage, specs in self.providers.items():
            if not specs:
                raise ValueError(f"At least one provider is required for {stage.value}")
            for spec in specs:
                if spec.name in seen:
                    raise ValueError(f"Duplicate provider name: {spec.name}")
                seen.add(spec.name)
        return self


class ProviderUsage(StrictModel):
    provider: NonEmptyStr
    tokens_today: int = Field(default=0, ge=0)
    last_reset: float = Field(default=0.0, ge=0.0)


class Ledger(StrictModel):
    usage: dict[str, ProviderUsage] = Field(default_factory=dict)

PatchOperation = Literal["add", "remove", "replace", "move", "copy", "test"]
_VALUE_OPS = frozenset({"add", "replace", "test"})
_FROM_OPS = frozenset({"move", "copy"})
_BAD_ESCAPE_RE = re.compile(r"~(?![01])")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _check_json_pointer(value: str) -> str:
    if not value.startswith("/"):
        raise ValueError("JSON Pointer must start with '/'")
    if _BAD_ESCAPE_RE.search(value):
        raise ValueError("JSON Pointer contains an invalid '~' escape")
    if _CONTROL_RE.search(value):
        raise ValueError("JSON Pointer contains control characters")
    return value


JsonPointer = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_POINTER_CHARS),
    AfterValidator(_check_json_pointer),
]


class JSONPatchOp(FrozenModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    op: PatchOperation
    path: JsonPointer
    value: Any = None
    from_: JsonPointer | None = Field(default=None, alias="from")

    @model_validator(mode="after")
    def _validate_operation_fields(self) -> JSONPatchOp:
        has_value = "value" in self.model_fields_set
        needs_value = self.op in _VALUE_OPS
        if needs_value != has_value:
            verb = "requires" if needs_value else "does not accept"
            raise ValueError(f"'{self.op}' {verb} 'value'")
        needs_from = self.op in _FROM_OPS
        if needs_from != (self.from_ is not None):
            verb = "requires" if needs_from else "does not accept"
            raise ValueError(f"'{self.op}' {verb} 'from'")
        if self.op == "move" and self.from_ == self.path:
            raise ValueError("'move' source and destination must differ")
        return self

    def to_jsonpatch(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op, "path": self.path}
        if "value" in self.model_fields_set:
            result["value"] = self.value
        if self.from_ is not None:
            result["from"] = self.from_
        return result

class PlanStep(StrictModel):
    id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    stage: Stage
    instruction: NonEmptyStr
    status: StepStatus = StepStatus.PENDING
    artifacts: dict[str, Any] = Field(default_factory=dict)


class Blueprint(StrictModel):
    blueprint_id: NonEmptyStr
    goal: NonEmptyStr
    steps: list[PlanStep] = Field(default_factory=list, max_length=MAX_STEPS)
    revision: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_step_ids(self) -> Blueprint:
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Step ids must be unique")
        return self


class Critique(StrictModel):
    stage: Stage
    severity: Literal["info", "warn", "error", "blocker"]
    message: NonEmptyStr
    target_path: JsonPointer | None = None


class AgentDelta(StrictModel):
    stage: Stage
    critiques: list[Critique] = Field(default_factory=list)
    patches: list[JSONPatchOp] = Field(default_factory=list, max_length=MAX_PATCH_OPS)

    @model_validator(mode="after")
    def _critique_stage_matches(self) -> AgentDelta:
        if any(critique.stage != self.stage for critique in self.critiques):
            raise ValueError("Every critique must belong to the delta stage")
        return self

    @property
    def has_blockers(self) -> bool:
        return any(critique.severity == "blocker" for critique in self.critiques)


class GatekeeperState(StrictModel):
    stage: Stage
    decision: GatekeeperDecision
    reason: NonEmptyStr
    required_patches: list[JSONPatchOp] = Field(default_factory=list, max_length=MAX_PATCH_OPS)

    @property
    def approved(self) -> bool:
        return self.decision is GatekeeperDecision.APPROVE


class PipelineState(StrictModel):
    stage: Stage
    phase: Phase = Phase.RUNNING
    blueprint: Blueprint
    gatekeeper: GatekeeperState | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None

    def is_terminal(self) -> bool:
        return self.phase in (Phase.COMPLETED, Phase.FAILED)
