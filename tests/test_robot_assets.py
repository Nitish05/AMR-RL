"""Robot geometry generator and spec validation (no simulator)."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from amr_rl.robot.generate import generate
from amr_rl.robot.spec import DEFAULT_SPEC, RobotSpec


def test_generated_assets_are_reproducible_and_complete(tmp_path):
    a = generate(DEFAULT_SPEC, tmp_path / "a")
    b = generate(DEFAULT_SPEC, tmp_path / "b")
    assert a["files"] == b["files"]
    root = ET.parse(tmp_path / "a" / "amr.urdf").getroot()
    links = {link.get("name") for link in root.findall("link")}
    assert {"base_link", "left_wheel", "right_wheel", "caster_link", "camera_link", "screen_link"} <= links
    joints = {j.get("name"): j.get("type") for j in root.findall("joint")}
    assert joints["left_wheel_joint"] == "continuous" and joints["right_wheel_joint"] == "continuous"
    for link in root.findall("link"):
        assert link.find("inertial/mass") is not None
    dims = a["dimensions_m"]
    assert dims["length"] == pytest.approx(0.35) and dims["overall_width"] == pytest.approx(0.30)
    assert 0.2 < a["total_mass_kg"] < 10


def test_committed_generated_assets_match_spec(tmp_path):
    committed = json.loads((DEFAULT_SPEC.parent / "generated" / "manifest.json").read_text())
    fresh = generate(DEFAULT_SPEC, tmp_path / "g")
    assert committed["spec_sha256"] == fresh["spec_sha256"]
    assert committed["files"]["amr.urdf"] == fresh["files"]["amr.urdf"]


def test_camera_is_unobstructed_and_mounted_forward():
    spec = RobotSpec.load()
    T = spec.camera_to_base()
    forward = T[:3, 2]
    assert forward[0] > 0.9 and forward[2] < 0  # looks forward and slightly down
    assert spec.camera_offset[0] > spec.screen_center[0]  # screen is behind the camera
    top = spec.chassis["ground_clearance"] + spec.chassis["height"]
    assert spec.camera["height_above_ground"] > top


@pytest.mark.parametrize("patch", [
    {"chassis": {"width": 0.26}},
    {"camera": {"pitch_down": -5}},
    {"camera": {"height_above_ground": 0.1}},
])
def test_invalid_specs_rejected(tmp_path, patch):
    data = yaml.safe_load(DEFAULT_SPEC.read_text())
    for section, values in patch.items():
        data[section].update(values)
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        RobotSpec.load(path)


def test_intrinsics_follow_fov():
    spec = RobotSpec.load()
    K = spec.camera_intrinsics()
    h = spec.camera["height"]
    fov = 2 * np.degrees(np.arctan((h / 2) / K[1, 1]))
    assert fov == pytest.approx(spec.camera["vertical_fov"])
