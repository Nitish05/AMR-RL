"""Wheel response to drive commands ("Model C") and a command-only odometry from it.

ENGINEERED model with parameters FITTED OFFLINE from simulator ground truth of
live runs (round-9 research, docs/results/turn-drift-options.md; 30 live map
builds). At runtime the parameters are constants; nothing reads the simulator.

* Each wheel's speed follows its target (the drive contract's wheel conversion)
  with rate limits: speeding up at most ``accel`` (8 wheel-rad/s^2), slowing down at
  most ``decel`` (15 wheel-rad/s^2). No dead time (first-order-plus-dead-time fits
  worse).
* Body motion from the wheel speeds, times measured gains: in-place yaw 0.83 at
  >= 0.35 rad/s and 0.71-0.79 below; yaw while driving 0.77; forward speed 0.89
  (0.83 above curvature 4 1/m). Per-turn model noise: sd 1.26 % of the angle.

Uses: (1) the motion-consistency check of the camera-only robot compares vision with
this modelled motion instead of the raw command (false "angular" losses came from
a left sweep followed by a sharp right arc: the signed commands cancelled and the
arc's real yaw is ~0.6 of the command); (2) ``CommandModelOdometry`` gives an
``OdometryStep`` for the fusion when the IMU and encoders are unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..control.contract import DriveLimits, body_to_wheels
from .fusion import OdometryStep


@dataclass
class WheelModelConfig:
    accel: float = 8.0      # wheel rad/s^2 (speeding up)
    decel: float = 15.0     # wheel rad/s^2 (slowing down)
    k_turn_fast: float = 0.83
    k_turn_slow: float = 0.75
    turn_fast: float = 0.35  # rad/s
    k_turn_drive: float = 0.77
    k_v: float = 0.89
    k_v_tight: float = 0.83
    tight_curvature: float = 4.0
    turn_rel_sigma: float = 0.0126  # per-turn model noise (fraction of the angle)
    rot_floor: float = 0.0005       # rad per frame
    trans_rel_sigma: float = 0.05
    trans_floor: float = 0.003      # m per frame
    substep: float = 0.01


class WheelResponse:
    def __init__(self, wheel_radius=0.05, track=0.27, limits: DriveLimits | None = None,
                 config: WheelModelConfig | None = None):
        self.r, self.track = float(wheel_radius), float(track)
        self.limits = limits or DriveLimits()
        self.cfg = config or WheelModelConfig()
        self.wl = self.wr = 0.0  # modelled wheel speeds (rad/s)

    @classmethod
    def from_spec(cls, spec, config=None):
        return cls(spec.drive["wheel_radius"], spec.drive["track"],
                   DriveLimits(max_wheel_speed=float(spec.drive["max_wheel_speed"])), config)

    def reset(self):
        self.wl = self.wr = 0.0

    def _approach(self, cur, target, h):
        c = self.cfg
        up = abs(target) > abs(cur) and cur * target >= 0
        lim = (c.accel if up else c.decel) * h
        return cur + max(-lim, min(lim, target - cur))

    def step(self, command, dt):
        """Advance by ``dt`` under ``command`` (v, w). Returns the modelled body motion
        (dx, dy, dth) in the frame at the start of the step, and the mean (v, w)."""
        c = self.cfg
        v_cmd, w_cmd = (0.0, 0.0) if command is None else (float(command[0]), float(command[1]))
        tgt = body_to_wheels(v_cmd, w_cmd, wheel_radius=self.r, track=self.track, limits=self.limits)
        n = max(1, int(round(dt / c.substep)))
        h = dt / n
        x = y = th = 0.0
        for _ in range(n):
            self.wl = self._approach(self.wl, tgt.left, h)
            self.wr = self._approach(self.wr, tgt.right, h)
            v = self.r * (self.wl + self.wr) / 2
            w = self.r * (self.wr - self.wl) / self.track
            driving = abs(v) >= 0.02
            if driving:
                curv = abs(w) / max(abs(v), 1e-6)
                v *= c.k_v if curv <= c.tight_curvature else c.k_v_tight
                w *= c.k_turn_drive
            else:
                w *= c.k_turn_fast if abs(w_cmd) >= c.turn_fast else c.k_turn_slow
            x += v * h * math.cos(th + 0.5 * w * h)
            y += v * h * math.sin(th + 0.5 * w * h)
            th += w * h
        return np.array([x, y, th]), (math.hypot(x, y) * (1 if x >= 0 else -1) / dt, th / dt)

    def sigma(self, delta):
        c = self.cfg
        rot = math.hypot(c.rot_floor, c.turn_rel_sigma * abs(delta[2]))
        trans = c.trans_floor + c.trans_rel_sigma * math.hypot(delta[0], delta[1])
        return trans, rot


class CommandModelOdometry:
    """``OdometryStep``s from the robot's own commands through ``WheelResponse``: the
    fallback when no IMU/encoder samples arrive. Rotation is not trusted over vision
    (``trust_rotation`` False): the fusion then robustifies this prior, not vision."""

    def __init__(self, spec, config: WheelModelConfig | None = None):
        self.model = WheelResponse.from_spec(spec, config)
        self.t = None

    def step(self, command, t):
        t0, self.t = self.t, t
        if t0 is None:
            return None
        dt = max(t - t0, 1e-6)
        delta, (v, w) = self.model.step(command, dt)
        s_t, s_r = self.model.sigma(delta)
        cov = np.diag([s_t ** 2, (s_t * 0.5) ** 2 + (math.hypot(*delta[:2]) * s_r) ** 2, s_r ** 2])
        return OdometryStep(dt, float(delta[0]), float(delta[1]), float(delta[2]), cov, float(v), float(w),
                            float(math.hypot(*delta[:2])), rotation_source="command_model",
                            translation_source="command_model", trust_rotation=False)
