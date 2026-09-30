"""The ONLY module that translates drive commands into Genesis actuator calls.

Wheel joints are driven with Genesis velocity control (``control_dofs_velocity``)
under a torque limit; the body moves only through wheel/ground contact. Body pose
and velocity setters appear only in ``initialize_pose`` for explicit resets.
Adapted from the structure of BB8-RL ``control/genesis_backend.py`` (command
queue, monotonic-time checks, expiry, stop semantics); the free-body force model
was replaced with physical wheel actuation.
"""

from __future__ import annotations

import math
from collections import deque

from .contract import DriveCapabilities, DriveCommand, DriveLimits, WheelTargets, body_to_wheels


class GenesisWheelBackend:
    capabilities = DriveCapabilities(
        backend_id="genesis-diff-drive",
        display_name="Genesis wheel-velocity differential drive (simulated)",
        approximate=True,
    )

    def __init__(self, robot, spec, limits: DriveLimits | None = None) -> None:
        self._robot = robot
        self.spec = spec
        self.limits = limits or DriveLimits(max_wheel_speed=float(spec.drive["max_wheel_speed"]))
        self.wheel_radius = float(spec.drive["wheel_radius"])
        self.track = float(spec.drive["track"])
        self._dofs = [
            robot.get_joint("left_wheel_joint").dofs_idx_local[0],
            robot.get_joint("right_wheel_joint").dofs_idx_local[0],
        ]
        torque = float(spec.drive["max_wheel_torque"])
        robot.set_dofs_kv([float(spec.drive["velocity_gain"])] * 2, self._dofs)
        robot.set_dofs_force_range([-torque, -torque], [torque, torque], self._dofs)
        robot.get_link("caster_link").set_friction(float(spec.caster["friction"]))
        for side in ("left_wheel", "right_wheel"):
            robot.get_link(side).set_friction(float(spec.drive["wheel_friction"]))
        self._pending: deque[DriveCommand] = deque()
        self._active: DriveCommand | None = None
        self._last_time = 0.0
        self._last_issued = -math.inf
        self.last_targets = WheelTargets(0.0, 0.0)
        self.stop()

    @property
    def connected(self) -> bool:
        return self._robot is not None

    def _check_time(self, now: float) -> None:
        if not math.isfinite(now) or now < self._last_time - 1e-12:
            raise ValueError("Simulation time must be finite and monotonic")
        self._last_time = now

    def command(self, command: DriveCommand, *, now: float) -> None:
        if not self.connected:
            raise RuntimeError("Drive backend is closed")
        self._check_time(now)
        if command.timestamp > now + 1e-9 or command.expires_at <= now:
            raise ValueError("Rejecting future or expired drive command")
        if command.timestamp <= self._last_issued:
            raise ValueError("Drive command timestamps must increase")
        if command.expires_at - command.timestamp > self.limits.max_command_age + 1e-9:
            raise ValueError("Command lifetime exceeds the configured maximum age")
        if len(self._pending) >= 64:
            raise RuntimeError("Drive command queue full")
        self._last_issued = command.timestamp
        self._pending.append(command)

    def apply(self, *, now: float) -> WheelTargets:
        """Call once per physics step, before ``scene.step()``."""
        if not self.connected:
            raise RuntimeError("Drive backend is closed")
        self._check_time(now)
        while self._pending and self._pending[0].timestamp <= now + 1e-9:
            self._active = self._pending.popleft()
        if self._active is not None and now < self._active.expires_at:
            targets = body_to_wheels(
                self._active.v, self._active.w,
                wheel_radius=self.wheel_radius, track=self.track, limits=self.limits,
            )
        else:
            self._active = None
            targets = WheelTargets(0.0, 0.0)
        self._robot.control_dofs_velocity([targets.left, targets.right], self._dofs)
        self.last_targets = targets
        return targets

    def stop(self) -> None:
        """Drop queued/active commands; wheels are commanded to zero immediately."""
        self._pending.clear()
        self._active = None
        self.last_targets = WheelTargets(0.0, 0.0)
        if self._robot is not None:
            self._robot.control_dofs_velocity([0.0, 0.0], self._dofs)

    def reset_clock(self) -> None:
        self.stop()
        self._last_time = 0.0
        self._last_issued = -math.inf

    def initialize_pose(self, x: float, y: float, theta: float) -> None:
        """Explicit reset only; never called by the stepping controller."""
        if not all(math.isfinite(v) for v in (x, y, theta)):
            raise ValueError("Initial pose must be finite")
        self.stop()
        z = float(self.spec.base_height) + 0.002
        half = theta / 2
        self._robot.set_pos([x, y, z], zero_velocity=True)
        self._robot.set_quat([math.cos(half), 0.0, 0.0, math.sin(half)], zero_velocity=True)
        self._robot.set_dofs_velocity([0.0] * self._robot.n_dofs)

    def close(self) -> None:
        if self._robot is not None:
            self.stop()
            self._robot = None
