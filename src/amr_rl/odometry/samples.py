"""Raw proprioceptive sensor samples as the robot's drivers would deliver them.

These are the only proprioceptive inputs the runtime receives: integer encoder
counts and quantised IMU readings in the sensor's own frame and units. Nothing in
them is simulator ground truth (the simulated parts are modelled from datasheets
in ``amr_rl.sim.sensors``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImuSample:
    """One IMU output sample. ``gyro`` in rad/s, ``accel`` (specific force) in m/s^2,
    both in the IMU body frame (x forward, y left, z up when mounted level)."""

    t: float
    gyro: np.ndarray
    accel: np.ndarray


@dataclass(frozen=True)
class EncoderSample:
    """Quadrature counter values (signed, cumulative) of the left and right wheel
    motors, latched at time ``t``. Positive counts = wheel rolling forward."""

    t: float
    left: int
    right: int


@dataclass
class ProprioBatch:
    """Samples received between two camera frames (time ordered)."""

    imu: list[ImuSample]
    encoders: list[EncoderSample]

    def __len__(self):
        return len(self.imu) + len(self.encoders)
