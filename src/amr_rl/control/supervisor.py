"""Motion-authority supervisor: the single gate between intentions and the drive.

Design references from BB8-RL ``interactive_runtime.py`` (Stop priority, command
generations, heartbeat cancellation, stale-command rejection, no automatic
resumption) re-implemented for the AMR. Rules:

* ``stop`` is always accepted, wins over anything else in the same batch, bumps
  the generation and clears queued motion.
* Every motion-granting request (manual, goal, enable_autonomy) must carry the
  current generation; anything older is rejected as stale.
* Autonomy is revoked (generation bumped, drive stopped) on: Stop, operator
  heartbeat loss, stale camera observations, localization loss, manual
  override, or an internal error. Nothing re-enables it automatically;
  ``enable_autonomy`` needs a fresh frame, tracking localization and the current
  generation.
* Optional bounded recovery: after localization loss, a pre-authorised recovery
  budget allows in-place rotation only (v = 0, |w| <= limit) for a bounded angle
  and duration. Its end (success or exhaustion) leaves the robot stopped and
  autonomy disabled.
* Knowledge may survive restart; authority never does (constructed disabled).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .contract import DriveCommand

REASONS = (
    "stop", "heartbeat_lost", "stale_observation", "localization_lost",
    "manual_override", "error", "restart", "disabled", "recovery_exhausted",
    "recovery_succeeded",
)


@dataclass
class SupervisorConfig:
    heartbeat_seconds: float = 3.0
    require_heartbeat: bool = True
    max_observation_age: float = 0.35  # sim seconds (camera at 10 Hz)
    command_lifetime: float = 0.2  # sim seconds
    manual_lifetime: float = 0.25
    allow_recovery: bool = True
    recovery_max_angle: float = 2 * math.pi
    recovery_max_seconds: float = 25.0
    recovery_max_rate: float = 0.45
    recovery_max_reverse: float = 0.45  # m of straight reversing (retracing a verified approach)
    recovery_max_reverse_speed: float = 0.10


@dataclass
class Ack:
    accepted: bool
    reason: str
    generation: int

    def to_dict(self):
        return {"accepted": self.accepted, "reason": self.reason, "generation": self.generation}


@dataclass
class _Recovery:
    started: float
    generation: int
    angle: float = 0.0
    reverse: float = 0.0
    last_time: float | None = None


@dataclass
class Supervisor:
    backend: object
    config: SupervisorConfig = field(default_factory=SupervisorConfig)
    wall_clock: object = None  # callable returning wall seconds (heartbeat)

    def __post_init__(self):
        import time

        self.wall_clock = self.wall_clock or time.monotonic
        self.generation = 0
        self.autonomy_enabled = False
        self.manual_until = -math.inf
        self.stopped = True
        self.revoked_reason = "restart"
        self.last_heartbeat = None
        self.last_frame_time = None
        self.localization_status = "initializing"
        self.recovery: _Recovery | None = None
        self.log: list[dict] = []
        self._last_now = 0.0

    # ------------------------------------------------------------------ inputs
    def heartbeat(self) -> Ack:
        self.last_heartbeat = self.wall_clock()
        return Ack(True, "heartbeat", self.generation)

    def observe_frame(self, timestamp: float) -> None:
        if self.last_frame_time is not None and timestamp <= self.last_frame_time:
            raise ValueError("Camera frames must have increasing timestamps")
        self.last_frame_time = float(timestamp)

    def observe_localization(self, status: str) -> None:
        self.localization_status = status

    # --------------------------------------------------------------- commands
    def _bump(self, reason: str) -> None:
        self.generation += 1
        self.revoked_reason = reason
        self.autonomy_enabled = False
        self.manual_until = -math.inf
        self.stopped = True
        self.backend.stop()
        self._record("revoke", reason=reason)

    def _record(self, kind, **detail):
        entry = {"kind": kind, "generation": self.generation, "sim_time": self._last_now, **detail}
        self.log.append(entry)
        if len(self.log) > 500:
            del self.log[:100]

    def stop(self, reason: str = "stop") -> Ack:
        self.recovery = None
        self._bump(reason)
        return Ack(True, reason, self.generation)

    def handle_batch(self, commands: list[dict], *, now: float) -> list[Ack]:
        """Process operator commands received together; Stop wins the batch."""
        if any(c.get("action") == "stop" for c in commands):
            acks = []
            for command in commands:
                if command.get("action") == "stop":
                    acks.append(self.stop())
                elif command.get("action") == "heartbeat":
                    acks.append(self.heartbeat())
                else:
                    acks.append(Ack(False, "superseded_by_stop", self.generation))
            return acks
        return [self.handle(command, now=now) for command in commands]

    def _fresh(self, now: float) -> bool:
        return (
            self.last_frame_time is not None
            and -1e-9 <= now - self.last_frame_time <= self.config.max_observation_age
        )

    def _heartbeat_ok(self) -> bool:
        if not self.config.require_heartbeat:
            return True
        return (
            self.last_heartbeat is not None
            and self.wall_clock() - self.last_heartbeat <= self.config.heartbeat_seconds
        )

    def handle(self, command: dict, *, now: float) -> Ack:
        self._last_now = now
        action = command.get("action")
        if action == "stop":
            return self.stop()
        if action == "heartbeat":
            return self.heartbeat()
        if action == "disable_autonomy":
            return self.stop("disabled")
        generation = command.get("generation")
        if type(generation) is not int or generation != self.generation:
            self._record("reject", action=action, reason="stale_generation")
            return Ack(False, "stale_generation", self.generation)
        if not self._heartbeat_ok():
            return Ack(False, "no_operator_heartbeat", self.generation)
        if action == "enable_autonomy":
            if not self._fresh(now):
                return Ack(False, "camera_observation_not_fresh", self.generation)
            if self.localization_status != "tracking":
                return Ack(False, "localization_not_tracking", self.generation)
            self.recovery = None
            self.generation += 1
            self.autonomy_enabled = True
            self.stopped = False
            self.revoked_reason = None
            self._record("grant", authority="autonomy")
            return Ack(True, "autonomy_enabled", self.generation)
        if action == "manual":
            v, w = command.get("v", 0.0), command.get("w", 0.0)
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (v, w)):
                return Ack(False, "invalid_manual_command", self.generation)
            if self.autonomy_enabled or self.recovery is not None:
                self.recovery = None
                self._bump("manual_override")
                return Ack(False, "autonomy_revoked_resend_manual", self.generation)
            self.manual_until = now + self.config.manual_lifetime
            self.stopped = False
            self.revoked_reason = None
            self.backend.command(
                DriveCommand(float(v), float(w), now, now + self.config.manual_lifetime,
                             self.generation, "manual"),
                now=now,
            )
            return Ack(True, "manual_accepted", self.generation)
        return Ack(False, "unknown_action", self.generation)

    # ------------------------------------------------------------ autonomy
    def check(self, *, now: float) -> str | None:
        """Per-tick revocation checks. Returns the revocation reason, if any."""
        self._last_now = now
        active = self.autonomy_enabled or now < self.manual_until or self.recovery is not None
        if not active:
            return None
        if not self._heartbeat_ok():
            self.recovery = None
            self._bump("heartbeat_lost")
            return "heartbeat_lost"
        manual = now < self.manual_until
        if not manual and not self._fresh(now):
            self.recovery = None
            self._bump("stale_observation")
            return "stale_observation"
        if self.autonomy_enabled and self.localization_status not in ("tracking", "predicted"):
            self._bump("localization_lost")
            if self.config.allow_recovery:
                self.recovery = _Recovery(started=now, generation=self.generation)
                self.stopped = False
                self._record("grant", authority="bounded_recovery")
            return "localization_lost"
        if self.recovery is not None:
            if self.localization_status == "tracking":
                self.recovery = None
                self._bump("recovery_succeeded")
                return "recovery_succeeded"
            if (
                now - self.recovery.started > self.config.recovery_max_seconds
                or self.recovery.angle >= self.config.recovery_max_angle
            ):
                self.recovery = None
                self._bump("recovery_exhausted")
                return "recovery_exhausted"
        return None

    def drive(self, v: float, w: float, *, now: float, generation: int, source="autonomy") -> Ack:
        """Autonomy/recovery motion request; rejected unless currently authorised."""
        self._last_now = now
        if type(generation) is not int or generation != self.generation:
            return Ack(False, "stale_generation", self.generation)
        if source == "autonomy":
            if not self.autonomy_enabled:
                return Ack(False, "autonomy_not_enabled", self.generation)
            if not self._fresh(now):
                return Ack(False, "camera_observation_not_fresh", self.generation)
        elif source == "recovery":
            r = self.recovery
            if r is None or r.generation != generation:
                return Ack(False, "recovery_not_authorised", self.generation)
            # Either rotate in place, or reverse straight (never forward, never both).
            rotating = abs(v) <= 1e-9 and abs(w) <= self.config.recovery_max_rate + 1e-9
            reversing = (abs(w) <= 1e-9 and -self.config.recovery_max_reverse_speed - 1e-9 <= v < 0
                         and r.reverse < self.config.recovery_max_reverse)
            if not (rotating or reversing):
                return Ack(False, "recovery_bounds_exceeded", self.generation)
            if r.last_time is not None:
                dt = max(0.0, now - r.last_time)
                r.angle += abs(w) * dt
                r.reverse += abs(v) * dt
            r.last_time = now
        else:
            return Ack(False, "unknown_source", self.generation)
        self.backend.command(
            DriveCommand(float(v), float(w), now, now + self.config.command_lifetime,
                         self.generation, source),
            now=now,
        )
        return Ack(True, "drive_accepted", self.generation)

    def snapshot(self) -> dict:
        age = None
        if self.last_heartbeat is not None:
            age = self.wall_clock() - self.last_heartbeat
        return {
            "autonomy_enabled": self.autonomy_enabled,
            "manual_active": self._last_now < self.manual_until,
            "recovery_active": self.recovery is not None,
            "stopped": self.stopped,
            "generation": self.generation,
            "revoked_reason": self.revoked_reason,
            "heartbeat_age": age,
            "heartbeat_required": self.config.require_heartbeat,
        }
