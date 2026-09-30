"""Real Genesis World checks for the robot body, wheel actuation, camera and screen.

Opt-in: ``RUN_GENESIS=1 scripts/amr.sh test-sim`` (CPU backend, headless). These
use ground truth to *verify* the simulation contract; the runtime never does.
"""

import math
import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.genesis,
    pytest.mark.skipif(os.environ.get("RUN_GENESIS") != "1", reason="set RUN_GENESIS=1 for Genesis tests"),
]


@pytest.fixture(scope="module")
def world():
    from amr_rl.sim.world import SimWorld, load_world_config

    w = SimWorld(load_world_config("dev_interact"), inspection=True)
    for _ in range(30):
        w.step()
    return w


def _cmd(world, v, w, seconds):
    from amr_rl.control.contract import DriveCommand

    steps = int(seconds / world.dt)
    for k in range(steps):
        if k % 5 == 0:
            world.backend.command(DriveCommand(v, w, world.time, world.time + 0.2, 0, "manual"), now=world.time)
        world.step()


def test_named_frames_and_mass(world):
    names = {link.name for link in world.robot.links}
    assert {"base_link", "left_wheel", "right_wheel", "caster_link", "camera_link", "screen_link"} <= names
    assert world.robot.get_mass() == pytest.approx(5.08, abs=0.01)


def test_wheels_drive_body_forward_and_turn(world):
    from amr_rl.sim.evaluator import true_pose

    world.backend.initialize_pose(-1.0, -0.2, 0.0)
    for _ in range(30):
        world.step()
    p0 = true_pose(world)
    _cmd(world, 0.2, 0.0, 2.0)
    p1 = true_pose(world)
    assert 0.25 < p1[0] - p0[0] < 0.45 and abs(p1[1] - p0[1]) < 0.03
    _cmd(world, 0.0, 0.8, 2.0)
    p2 = true_pose(world)
    turned = math.atan2(math.sin(p2[2] - p1[2]), math.cos(p2[2] - p1[2]))
    assert 0.6 < turned < 1.8  # wheel slip: less than the commanded 1.6 rad
    assert np.hypot(*(p2[:2] - p1[:2])) < 0.05  # turning in place


def test_command_expiry_stops_wheels(world):
    from amr_rl.control.contract import DriveCommand
    from amr_rl.sim.evaluator import true_pose

    world.backend.command(DriveCommand(0.2, 0.0, world.time, world.time + 0.2, 0, "manual"), now=world.time)
    for _ in range(100):  # 1 s with no new commands
        world.step()
    a = true_pose(world)
    for _ in range(50):
        world.step()
    b = true_pose(world)
    assert np.hypot(*(b[:2] - a[:2])) < 0.005


def test_camera_is_attached_to_the_moving_body(world):
    world.backend.initialize_pose(-1.0, -0.2, 0.0)
    for _ in range(30):
        world.step()
    f0 = world.capture("t")
    T0 = np.array(world.camera.transform)
    _cmd(world, 0.2, 0.0, 1.0)
    f1 = world.capture("t")
    T1 = np.array(world.camera.transform)
    link = world.robot.get_link("camera_link")
    cam_pos = np.asarray(link.get_pos()).reshape(3)
    assert np.linalg.norm(T1[:3, 3] - cam_pos) < 0.01  # optical centre at the camera link
    assert T1[0, 3] - T0[0, 3] > 0.1  # moved forward with the body
    assert f1.timestamp > f0.timestamp and f1.index == f0.index + 1
    assert np.abs(f1.rgb.astype(int) - f0.rgb.astype(int)).mean() > 2


def test_screen_is_visible_to_inspection_not_to_onboard_camera(world):
    from amr_rl.expression.policy import expression_from
    from amr_rl.expression.screen import render

    cam = np.array(world.inspection.transform)[:3, 3]
    world.backend.initialize_pose(0.0, 0.0, math.atan2(cam[1], cam[0]))  # face the inspection camera
    for _ in range(30):
        world.step()
    world.set_screen(render(expression_from({"activity": {"name": "idle"}})), signal_pattern=False)
    onboard_a = world.capture("t").rgb
    inspect_a = world.render_inspection()
    face = render(expression_from({"activity": {"name": "engage"}, "interaction": {"action": "signal",
                                                                                  "status": "acting"}}))
    world.set_screen(face, signal_pattern=True)
    onboard_b = world.capture("t").rgb
    inspect_b = world.render_inspection()
    assert world.signal_active
    assert np.abs(onboard_b.astype(int) - onboard_a.astype(int)).max() == 0
    # the inspection camera is far away; the screen occupies few pixels but changes them
    assert (np.abs(inspect_b.astype(int) - inspect_a.astype(int)).sum(axis=2) > 30).sum() > 5


def test_fixture_signal_rule_raises_panel_visible_onboard(world):
    from amr_rl.perception.camera_model import CameraModel
    from amr_rl.perception.entities import FixtureDetector

    item, entity = world.fixtures["bloom"]
    fx, fy = item["pos"]
    world.backend.initialize_pose(fx - 0.75, fy, 0.0)
    for _ in range(40):
        world.step()
    det = FixtureDetector(CameraModel.from_spec(world.spec))
    before = [d.state_token for d in det.detect(world.capture("t").rgb) if d.color == "cyan"]
    world.set_screen(np.zeros((132, 192, 3), np.uint8), signal_pattern=True)
    for _ in range(150):
        world.step()
    world.set_screen(np.zeros((132, 192, 3), np.uint8), signal_pattern=False)
    for _ in range(60):
        world.step()
    after = [d.state_token for d in det.detect(world.capture("t").rgb) if d.color == "cyan"]
    assert before == ["attach:none"] and after == ["attach:yellow"]
    assert world.events[-1].fixture == "bloom" and world.events[-1].response == "yellow"
