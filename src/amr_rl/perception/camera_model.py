"""Calibrated onboard camera model and planar-robot geometry.

Conventions
-----------
* World/map frame: x, y on the floor, z up (metres). Origin = robot base_link at
  the start of the map (not the simulator origin).
* base_link: x forward, y left, z up; planar pose (x, y, theta) is world<-base.
* Camera optical frame: OpenCV (x right, y down, z forward), pixel centres at
  integer coordinates, principal point ((w-1)/2, (h-1)/2).

Calibration assumptions (documented in docs/VSLAM.md): pinhole with no
distortion (the simulated lens has none), intrinsics derived from the specified
vertical field of view, and the camera->base mounting from the robot spec. The
floor is flat (z = 0) and the camera optical centre is ``height`` metres above
it. That height is the *only* source of metric scale in the RGB-only pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np


def planar_T(x: float, y: float, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[0, 3], T[1, 3] = x, y
    return T


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class CameraModel:
    K: np.ndarray
    T_base_cam: np.ndarray  # optical frame expressed in base_link
    width: int
    height: int
    base_z: float = 0.05  # base_link height above the floor (wheel radius)

    @classmethod
    def from_spec(cls, spec, *, height_error: float = 0.0, pitch_error_deg: float = 0.0):
        """Build from the robot spec. The error arguments exist only for the
        calibration-sensitivity experiment; operational code uses zeros."""
        T = spec.camera_to_base().copy()
        if pitch_error_deg:
            a = math.radians(pitch_error_deg)
            Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            T[:3, :3] = Ry @ T[:3, :3]
        T[2, 3] += height_error
        return cls(spec.camera_intrinsics(), T, int(spec.camera["width"]), int(spec.camera["height"]),
                   float(spec.base_height))

    @property
    def T_cam_base(self) -> np.ndarray:
        return np.linalg.inv(self.T_base_cam)

    @property
    def camera_height(self) -> float:
        """Optical centre height above the floor (base_link z is wheel radius above floor)."""
        return float(self.T_base_cam[2, 3] + self.base_z)

    @property
    def version(self) -> str:
        payload = json.dumps({
            "K": np.round(self.K, 6).tolist(), "T": np.round(self.T_base_cam, 6).tolist(),
            "size": [self.width, self.height], "base_z": round(self.base_z, 6),
        }, sort_keys=True)
        return "cal-" + hashlib.sha256(payload.encode()).hexdigest()[:12]

    # ---------------------------------------------------------------- geometry
    def T_world_cam(self, pose) -> np.ndarray:
        """Camera optical frame in world, for planar pose (x, y, theta). World z=0 is the floor."""
        T = planar_T(*pose)
        T[2, 3] = self.base_z
        return T @ self.T_base_cam

    def horizon_row(self) -> float:
        """Image row of the horizon (rays parallel to the floor)."""
        R = self.T_base_cam[:3, :3]
        # direction in camera frame of a horizontal forward ray
        d = R.T @ np.array([1.0, 0.0, 0.0])
        return float(self.K[1, 1] * d[1] / d[2] + self.K[1, 2])

    def rays(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(uv).astype(float)
        x = (uv[:, 0] - self.K[0, 2]) / self.K[0, 0]
        y = (uv[:, 1] - self.K[1, 2]) / self.K[1, 1]
        return np.stack([x, y, np.ones_like(x)], axis=1)

    def ground_points_world(self, uv: np.ndarray, pose, max_range: float = 3.0):
        """Intersect pixel rays with the floor. Returns (N,3) world points and a validity mask."""
        T = self.T_world_cam(pose)
        R, t = T[:3, :3], T[:3, 3]
        d = self.rays(uv) @ R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -t[2] / d[:, 2]
        valid = (d[:, 2] < -1e-3) & (s > 0)
        points = t + d * s[:, None]
        horizontal = np.linalg.norm(points[:, :2] - t[:2], axis=1)
        valid &= horizontal <= max_range
        return points, valid

    def floor_homography(self, pose) -> np.ndarray:
        """3x3 map from floor coordinates (x, y, 1) to image pixels."""
        T = np.linalg.inv(self.T_world_cam(pose))
        R, t = T[:3, :3], T[:3, 3]
        return self.K @ np.column_stack([R[:, 0], R[:, 1], t])

    def project_world(self, points: np.ndarray, pose):
        """Project world points; returns pixels (N,2) and depth (N,)."""
        T = np.linalg.inv(self.T_world_cam(pose))
        pc = points @ T[:3, :3].T + T[:3, 3]
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.K[0, 0] * pc[:, 0] / z + self.K[0, 2]
            v = self.K[1, 1] * pc[:, 1] / z + self.K[1, 2]
        return np.stack([u, v], axis=1), z
