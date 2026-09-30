"""Typed robot specification loaded from the editable YAML source asset.

The spec is the single source of truth for geometry, mass properties, sensor
mounting and actuator limits. Generated URDF/meshes are derived artifacts.
Values are simulation design choices, not physical measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = PROJECT_ROOT / "assets" / "robot" / "amr_spec.yaml"
GENERATED_DIR = PROJECT_ROOT / "assets" / "robot" / "generated"


def _positive(values: dict, *names: str) -> None:
    for name in names:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class RobotSpec:
    raw: dict

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SPEC) -> "RobotSpec":
        data = yaml.safe_load(Path(path).read_text())
        spec = cls(data)
        spec.validate()
        return spec

    # Convenience accessors -------------------------------------------------
    def __getattr__(self, name):  # section access: spec.chassis["length"]
        try:
            return self.raw[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def validate(self) -> None:
        c, d, k, cam, s = (
            self.raw["chassis"],
            self.raw["drive"],
            self.raw["caster"],
            self.raw["camera"],
            self.raw["screen"],
        )
        _positive(c, "length", "width", "height", "ground_clearance", "mass")
        _positive(d, "wheel_radius", "wheel_width", "track", "wheel_mass")
        _positive(d, "max_wheel_speed", "max_wheel_torque", "velocity_gain")
        _positive(k, "radius", "mass")
        _positive(cam, "height_above_ground", "vertical_fov", "width", "height")
        _positive(s, "width", "height", "depth", "mast_height", "mass")
        inner_gap = d["track"] - d["wheel_width"]
        if c["width"] >= inner_gap:
            raise ValueError("Chassis must fit between the drive wheels")
        if c["ground_clearance"] >= d["wheel_radius"]:
            raise ValueError("Chassis bottom must sit above the wheel axle bottom")
        if not 0 < cam["pitch_down"] < 45:
            raise ValueError("Camera pitch must look forward and down")
        top = c["ground_clearance"] + c["height"]
        if cam["height_above_ground"] <= top:
            raise ValueError("Camera optical centre must clear the chassis top")
        # Caster must reach the floor from the chassis underside.
        if k["radius"] * 2 < c["ground_clearance"] - 1e-6:
            raise ValueError("Caster cannot reach the floor")

    # Derived frames (base_link = axle midpoint, z = wheel radius) ------------
    @property
    def base_height(self) -> float:
        return float(self.drive["wheel_radius"])

    @property
    def chassis_center(self) -> np.ndarray:
        c = self.chassis
        z = c["ground_clearance"] + c["height"] / 2 - self.base_height
        return np.array([-c["axle_offset_x"], 0.0, z])

    @property
    def footprint_radius(self) -> float:
        """Circumscribed radius about base_link used for conservative planning."""
        c, d = self.chassis, self.drive
        front = c["length"] / 2 - c["axle_offset_x"]
        rear = c["length"] / 2 + c["axle_offset_x"]
        half_w = max(c["width"] / 2, d["track"] / 2 + d["wheel_width"] / 2)
        return float(math.hypot(max(front, rear), half_w))

    @property
    def overall_width(self) -> float:
        return float(self.drive["track"] + self.drive["wheel_width"])

    @property
    def camera_offset(self) -> np.ndarray:
        cam = self.camera
        return np.array(
            [cam["forward_from_axle"], 0.0, cam["height_above_ground"] - self.base_height]
        )

    @property
    def screen_center(self) -> np.ndarray:
        c, s = self.chassis, self.screen
        top = c["ground_clearance"] + c["height"] - self.base_height
        return np.array([-s["back_from_axle"], 0.0, top + s["mast_height"] + s["height"] / 2])

    @property
    def caster_center(self) -> np.ndarray:
        c, k = self.chassis, self.caster
        x = -c["axle_offset_x"] - c["length"] / 2 + k["inset_from_rear"]
        return np.array([x, 0.0, k["radius"] - self.base_height])

    def camera_intrinsics(self) -> np.ndarray:
        """Pinhole K (OpenCV pixel convention, principal point at image centre)."""
        cam = self.camera
        w, h = cam["width"], cam["height"]
        f = (h / 2) / math.tan(math.radians(cam["vertical_fov"]) / 2)
        return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])

    def camera_to_base(self) -> np.ndarray:
        """4x4 transform of the OpenCV optical frame (x right, y down, z forward)
        expressed in base_link (x forward, y left, z up)."""
        pitch = math.radians(self.camera["pitch_down"])
        forward = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
        right = np.array([0.0, -1.0, 0.0])
        down = np.cross(forward, right)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = right, down, forward
        T[:3, 3] = self.camera_offset
        return T
