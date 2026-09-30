"""Differential-drive command contract, independent of any simulator or hardware.

Adapted from BB8-RL ``control/contract.py`` (planar unit-disk force requests for
a rolling sphere) into body-frame (v, omega) requests converted to wheel
angular-velocity targets for a two-wheel differential drive. Every command has a
monotonic timestamp and an expiry; a backend must reject future, expired or
non-increasing commands and must fall back to zero wheel speed on expiry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DriveCapabilities:
    backend_id: str
    display_name: str
    approximate: bool
    hardware_capable: bool = False
    requires_arming: bool = False


@dataclass(frozen=True)
class DriveLimits:
    """Engineered command limits (SI units). Not measured from hardware."""

    max_linear: float = 0.30  # m/s
    max_angular: float = 1.2  # rad/s
    max_wheel_speed: float = 12.0  # rad/s
    max_command_age: float = 0.25  # s; commands expire within this lifetime

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class DriveCommand:
    """Body-frame velocity request. Times use the backend's monotonic clock."""

    v: float
    w: float
    timestamp: float
    expires_at: float
    generation: int
    source: str = "autonomy"  # autonomy | manual | recovery

    def __post_init__(self) -> None:
        values = (self.v, self.w, self.timestamp, self.expires_at)
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
            raise ValueError("Drive command values must be finite numbers")
        if self.timestamp < 0 or self.expires_at <= self.timestamp:
            raise ValueError("Command expiry must follow a nonnegative timestamp")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("Command generation must be a nonnegative integer")
        if self.source not in ("autonomy", "manual", "recovery"):
            raise ValueError("Unknown command source")


@dataclass(frozen=True)
class WheelTargets:
    left: float
    right: float


def body_to_wheels(v, w, *, wheel_radius, track, limits: DriveLimits) -> WheelTargets:
    """Clamp (v, w) then convert to wheel speeds, scaling both to respect the wheel limit
    while preserving curvature."""
    v = max(-limits.max_linear, min(limits.max_linear, float(v)))
    w = max(-limits.max_angular, min(limits.max_angular, float(w)))
    left = (v - w * track / 2) / wheel_radius
    right = (v + w * track / 2) / wheel_radius
    scale = max(1.0, abs(left) / limits.max_wheel_speed, abs(right) / limits.max_wheel_speed)
    return WheelTargets(left / scale, right / scale)


@runtime_checkable
class DriveBackend(Protocol):
    @property
    def capabilities(self) -> DriveCapabilities: ...

    def command(self, command: DriveCommand, *, now: float) -> None: ...

    def apply(self, *, now: float) -> WheelTargets: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...
