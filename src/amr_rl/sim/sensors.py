"""Simulated proprioceptive sensors: a 6-axis MEMS IMU and quadrature wheel encoders.

World side only (simulation). Each model starts from the physical quantity Genesis
computes (the IMU link's angular velocity and specific force from Genesis' IMU
sensor with its own noise disabled; the wheel joint angles) and applies the error
model of a real, purchasable part from its datasheet (``assets/robot/amr_spec.yaml``
sections ``imu`` and ``encoders`` name the parts and cite the numbers):

* IMU: per-unit turn-on bias, sensitivity (scale-factor) error and cross-axis
  coupling drawn once per robot from the datasheet tolerances; white noise from
  the noise spectral density and output data rate; bias random walk; quantisation
  to the LSB of the configured full-scale range; saturation at full scale.
* Encoders: integer quadrature counts of the motor shaft at the encoder's
  resolution, gearbox backlash as a hysteresis band between motor and wheel.

What the robot receives are the resulting samples (``amr_rl.odometry.samples``):
counts and quantised readings, never poses or velocities from the simulator.
Wheel slip is not modelled here: the encoders measure wheel rotation, and slip is
whatever the physics makes of wheel/floor contact (see ``SimWorld`` slip patches).
"""

from __future__ import annotations

import math

import numpy as np

from ..odometry.samples import EncoderSample, ImuSample, ProprioBatch

G = 9.80665
DPS = math.pi / 180.0


def _draw(rng, sigma, limit=None, size=3):
    x = rng.normal(0.0, sigma, size)
    return np.clip(x, -limit, limit) if limit is not None else x


class ImuErrorModel:
    """Datasheet error model of one IMU unit (engineered from published specs)."""

    def __init__(self, cfg: dict, rng: np.random.Generator, dt: float, unit_rng: np.random.Generator | None = None):
        """``unit_rng`` draws the unit's fixed errors (sensitivity, cross-axis: the same
        physical part every run); ``rng`` the run's turn-on offset and noise."""
        self.rng = rng
        self.dt = dt
        unit_rng = unit_rng if unit_rng is not None else rng
        g, a = cfg["gyro"], cfg["accel"]
        self.g_range = float(g["full_scale_dps"]) * DPS
        self.g_lsb = float(g["lsb_mdps"]) * 1e-3 * DPS
        odr = float(cfg["output_rate_hz"])
        bw = float(cfg.get("bandwidth_hz", odr / 2))
        # white noise per sample from the spectral density over the filter bandwidth
        self.g_noise = float(g["noise_density_dps_rthz"]) * DPS * math.sqrt(bw)
        self.g_rw = float(g.get("bias_random_walk_dps_rts", 0.0)) * DPS
        self.g_bias = _draw(rng, float(g["offset_sigma_dps"]) * DPS, float(g["offset_max_dps"]) * DPS)
        scale = _draw(unit_rng, float(g["sensitivity_sigma"]), float(g["sensitivity_max"]))
        cross = unit_rng.normal(0.0, float(g["cross_axis_sigma"]), (3, 3))
        self.g_M = np.eye(3) + np.diag(scale) + (cross - np.diag(np.diag(cross)))
        self.a_range = float(a["full_scale_g"]) * G
        self.a_lsb = float(a["lsb_mg"]) * 1e-3 * G
        self.a_noise = float(a["noise_density_ug_rthz"]) * 1e-6 * G * math.sqrt(bw)
        self.a_bias = _draw(rng, float(a["offset_sigma_mg"]) * 1e-3 * G, float(a["offset_max_mg"]) * 1e-3 * G)
        a_scale = _draw(unit_rng, float(a["sensitivity_sigma"]), float(a["sensitivity_max"]))
        a_cross = unit_rng.normal(0.0, float(a["cross_axis_sigma"]), (3, 3))
        self.a_M = np.eye(3) + np.diag(a_scale) + (a_cross - np.diag(np.diag(a_cross)))

    def gyro(self, omega):
        self.g_bias = self.g_bias + self.g_rw * math.sqrt(self.dt) * self.rng.normal(0.0, 1.0, 3)
        x = self.g_M @ omega + self.g_bias + self.rng.normal(0.0, self.g_noise, 3)
        return np.clip(np.round(x / self.g_lsb) * self.g_lsb, -self.g_range, self.g_range - self.g_lsb)

    def accel(self, f):
        x = self.a_M @ f + self.a_bias + self.rng.normal(0.0, self.a_noise, 3)
        return np.clip(np.round(x / self.a_lsb) * self.a_lsb, -self.a_range, self.a_range - self.a_lsb)


class EncoderModel:
    """Quadrature encoder on the motor shaft of a geared wheel motor."""

    def __init__(self, cfg: dict):
        self.cpr = float(cfg["counts_per_wheel_rev"])  # all quadrature edges, at the wheel
        self.backlash = math.radians(float(cfg.get("backlash_deg", 0.0)))  # at the wheel
        self.motor = None  # motor-side angle referred to the wheel (rad)

    def count(self, wheel_angle: float) -> int:
        if self.motor is None:
            self.motor = wheel_angle
        # Gearbox play: the motor can move within +-backlash/2 of the wheel without
        # turning it (a dead band), so the motor angle drags the wheel's.
        half = self.backlash / 2
        if wheel_angle - self.motor > half:
            self.motor = wheel_angle - half
        elif self.motor - wheel_angle > half:
            self.motor = wheel_angle + half
        return int(math.floor(self.motor / (2 * math.pi) * self.cpr))


class ProprioSensors:
    """The robot's IMU and wheel encoders in a SimWorld.

    ``attach`` must be called before ``scene.build()`` (Genesis sensors are part of
    the scene); ``sample`` after every physics step; ``drain`` hands the samples
    since the last call to the harness (as driver FIFOs would)."""

    def __init__(self, spec, *, seed: int, dt: float, faults: dict | None = None):
        self.spec = spec
        self.dt = dt
        self.faults = faults or {}
        rng = np.random.default_rng([int(seed), 0x1A3])  # own stream: physics/world RNG untouched
        imu_cfg, enc_cfg = spec.raw["imu"], spec.raw["encoders"]
        # The robot's one physical IMU: fixed errors from its serial (overridable for
        # evaluation with another unit: sensor_faults {"imu_unit": n}).
        unit = int(self.faults.get("imu_unit", imu_cfg.get("unit_serial", 1)))
        self.unit = unit
        self.imu_model = ImuErrorModel(imu_cfg, rng, dt, unit_rng=np.random.default_rng([unit, 0x5E1]))
        self.imu_every = max(1, int(round(1.0 / (float(imu_cfg["output_rate_hz"]) * dt))))
        self.enc_every = max(1, int(round(1.0 / (float(enc_cfg["sample_rate_hz"]) * dt))))
        self.encoders = (EncoderModel(enc_cfg), EncoderModel(enc_cfg))
        self.mount = np.asarray(imu_cfg["mount_xyz"], float)
        self._imu_sensor = None
        self._robot = None
        self._dofs = None
        self._tick = 0
        self._imu: list[ImuSample] = []
        self._enc: list[EncoderSample] = []

    def attach(self, scene, robot):
        import genesis as gs

        link = robot.get_link("base_link")
        self._imu_sensor = scene.add_sensor(gs.sensors.IMU(
            entity_idx=robot.idx, link_idx_local=link.idx_local, pos_offset=tuple(self.mount)))
        self._robot = robot

    def bind(self):
        """After ``scene.build()``: resolve the wheel joints."""
        self._dofs = [self._robot.get_joint(n).dofs_idx_local[0] for n in ("left_wheel_joint", "right_wheel_joint")]

    def sample(self, t: float):
        self._tick += 1
        if self._tick % self.imu_every == 0 and not self.faults.get("imu_off"):
            r = self._imu_sensor.read()
            omega = np.asarray(r.ang_vel, float).reshape(3)
            f = np.asarray(r.lin_acc, float).reshape(3)
            self._imu.append(ImuSample(t, self.imu_model.gyro(omega), self.imu_model.accel(f)))
        if self._tick % self.enc_every == 0 and not self.faults.get("encoders_off"):
            q = np.asarray(self._robot.get_dofs_position(self._dofs), float).reshape(2)
            self._enc.append(EncoderSample(t, self.encoders[0].count(float(q[0])), self.encoders[1].count(float(q[1]))))

    def drain(self) -> ProprioBatch:
        batch = ProprioBatch(self._imu, self._enc)
        self._imu, self._enc = [], []
        return batch
