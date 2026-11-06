from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from validators.schemas import Ledger, ProviderInventory, ProviderSpec, ProviderUsage, Stage

_SECONDS_PER_DAY = 86_400


class QuotaGovernorError(Exception):
    pass


class NoAvailableProviderError(QuotaGovernorError):
    pass


class UnknownReservationError(QuotaGovernorError):
    pass


@dataclass(frozen=True, slots=True)
class Reservation:
    """Handle for one in-flight provider call. Must be settled exactly once."""

    reservation_id: int
    stage: Stage
    provider: ProviderSpec
    estimated_tokens: int


class QuotaGovernor:
    """Thread-safe, event-loop-safe provider rate/quota governor.

    All state transitions happen under a single non-reentrant lock and never block
    or await inside it, so it is safe to call from threads and from coroutines.

    Lifecycle: ``reserve_provider`` -> (perform the call) -> ``settle``.
    A reservation consumes an RPM slot immediately (conservatively, even if the call
    later fails) and holds ``estimated_tokens`` against the daily budget until settled,
    which prevents concurrent callers from overshooting the token budget.
    """

    def __init__(
        self,
        inventory: ProviderInventory,
        ledger: Ledger | None = None,
        *,
        rpm_window_seconds: float = 60.0,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if rpm_window_seconds <= 0:
            raise ValueError("rpm_window_seconds must be positive")
        self._inventory = inventory
        self._rpm_window = rpm_window_seconds
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

        day_start = self._day_start(wall_clock())
        supplied = ledger.usage if ledger is not None else {}
        specs = [spec for group in inventory.providers.values() for spec in group]
        self._ledger = Ledger(
            usage={
                spec.name: (
                    supplied[spec.name].model_copy(deep=True)
                    if spec.name in supplied
                    else ProviderUsage(provider=spec.name, last_reset=day_start)
                )
                for spec in specs
            }
        )
        self._rpm_events: dict[str, deque[float]] = {spec.name: deque() for spec in specs}
        self._inflight_tokens: dict[str, int] = dict.fromkeys(self._rpm_events, 0)
        self._cooldown_until: dict[str, float] = {}
        self._cursor: dict[Stage, int] = dict.fromkeys(inventory.providers, 0)
        self._open: dict[int, Reservation] = {}

    @classmethod
    def from_yaml(
        cls, path: str | Path, ledger: Ledger | None = None, *, rpm_window_seconds: float = 60.0
    ) -> QuotaGovernor:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Configuration root must be a YAML mapping")
        inventory = ProviderInventory.model_validate(raw)
        return cls(inventory, ledger, rpm_window_seconds=rpm_window_seconds)

    # ------------------------------------------------------------------ API --

    def reserve_provider(self, stage: Stage, *, estimated_tokens: int = 0) -> Reservation:
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens must not be negative")
        with self._lock:
            specs = self._inventory.providers[stage]
            now, wall = self._monotonic(), self._wall_clock()
            start = self._cursor[stage]
            for offset in range(len(specs)):
                index = (start + offset) % len(specs)
                spec = specs[index]
                if not self._is_available(spec, estimated_tokens, now, wall):
                    continue
                self._rpm_events[spec.name].append(now)
                self._inflight_tokens[spec.name] += estimated_tokens
                self._cursor[stage] = (index + 1) % len(specs)
                reservation = Reservation(next(self._ids), stage, spec, estimated_tokens)
                self._open[reservation.reservation_id] = reservation
                return reservation
        raise NoAvailableProviderError(
            f"All providers for stage {stage.value!r} are rate-limited, cooling down, "
            "or have exhausted their daily token budgets"
        )

    def settle(
        self, reservation: Reservation, *, tokens: int = 0, cooldown_seconds: float = 0.0
    ) -> None:
        """Release the in-flight hold, record actual usage, optionally cool the provider down."""
        if tokens < 0 or cooldown_seconds < 0:
            raise ValueError("tokens and cooldown_seconds must not be negative")
        with self._lock:
            if self._open.pop(reservation.reservation_id, None) is None:
                raise UnknownReservationError(
                    f"Reservation {reservation.reservation_id} is unknown or already settled"
                )
            name = reservation.provider.name
            self._inflight_tokens[name] -= reservation.estimated_tokens
            usage = self._ledger.usage[name]
            self._rollover(usage, self._wall_clock())
            usage.tokens_today += tokens
            if cooldown_seconds > 0:
                until = self._monotonic() + cooldown_seconds
                self._cooldown_until[name] = max(self._cooldown_until.get(name, 0.0), until)

    def snapshot(self) -> Ledger:
        """Return an isolated copy of the ledger (safe to persist or inspect)."""
        with self._lock:
            return self._ledger.model_copy(deep=True)

    # ------------------------------------------------------------ internals --

    def _is_available(self, spec: ProviderSpec, estimated: int, now: float, wall: float) -> bool:
        events = self._rpm_events[spec.name]
        cutoff = now - self._rpm_window
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= spec.rpm:
            return False
        if self._cooldown_until.get(spec.name, 0.0) > now:
            return False
        usage = self._ledger.usage[spec.name]
        self._rollover(usage, wall)
        projected = usage.tokens_today + self._inflight_tokens[spec.name] + max(estimated, 1)
        return projected <= spec.tpd

    @staticmethod
    def _day_start(wall: float) -> float:
        return wall - (wall % _SECONDS_PER_DAY)

    def _rollover(self, usage: ProviderUsage, wall: float) -> None:
        day_start = self._day_start(wall)
        if usage.last_reset < day_start:
            usage.tokens_today = 0
            usage.last_reset = day_start
