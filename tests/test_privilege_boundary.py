"""Runtime modules must not import simulator ground truth or Genesis.

The operational pipeline (perception, mapping, navigation, learning, behavior,
runtime, control supervisor, expression) consumes onboard RGB frames only. This
static check fails if any of those packages imports the evaluator, the world,
the harness, or Genesis itself.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "amr_rl"
RUNTIME_PACKAGES = ["perception", "mapping", "navigation", "learning", "behavior", "runtime", "expression",
                    "odometry"]
RUNTIME_FILES = ["control/supervisor.py", "control/contract.py"]
FORBIDDEN = ("amr_rl.sim", "genesis", "..sim", ".sim")


def _imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield ("." * node.level) + (node.module or "")


def _files():
    for package in RUNTIME_PACKAGES:
        yield from sorted((SRC / package).rglob("*.py"))
    for name in RUNTIME_FILES:
        yield SRC / name


@pytest.mark.parametrize("path", list(_files()), ids=lambda p: str(p.relative_to(SRC)))
def test_runtime_module_has_no_privileged_imports(path):
    bad = [name for name in _imports(path)
           if name == "genesis" or name.startswith("genesis.") or name.startswith("amr_rl.sim")
           or name.startswith("..sim") or (name.startswith(".sim") and path.parent.name == "amr_rl")]
    assert not bad, f"{path} imports privileged modules: {bad}"


def test_only_backend_calls_actuators():
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        if "control_dofs_velocity" in text and path.name != "genesis_backend.py":
            offenders.append(path.name)
        if "robot.set_pos(" in text or "robot.set_quat(" in text or "_robot.set_pos(" in text:
            if path.name != "genesis_backend.py":
                offenders.append(path.name)
        if ("set_pos(" in text or "set_quat(" in text) and path.name not in ("genesis_backend.py", "world.py"):
            offenders.append(path.name)  # world.py may move environment objects only
    assert offenders == []


def test_harness_passes_only_frames_backend_and_commands_to_runtime():
    text = (SRC / "sim" / "harness.py").read_text()
    assert "RobotRuntime(spec, self.world.backend" in text
    assert "runtime.on_frame(frame)" in text
    # proprioception: only the raw sample batches of the sensor models
    assert "self.runtime.on_proprio(proprio)" in text
    assert "proprio = self.world.sensors.drain()" in text


def test_sensor_samples_carry_no_ground_truth():
    """The proprioceptive samples are counts and quantised IMU readings only."""
    from dataclasses import fields

    from amr_rl.odometry.samples import EncoderSample, ImuSample

    assert [f.name for f in fields(ImuSample)] == ["t", "gyro", "accel"]
    assert [f.name for f in fields(EncoderSample)] == ["t", "left", "right"]
