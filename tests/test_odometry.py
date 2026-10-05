"""Round 9: IMU + wheel-encoder sensor models, wheel-inertial odometry and its fusion
with the VSLAM (no simulator; docs/results/heading-r9.md)."""

import math

import numpy as np
from turn_scene import run_turn

from amr_rl.odometry.fusion import WheelInertialOdometry
from amr_rl.odometry.samples import EncoderSample, ImuSample, ProprioBatch
from amr_rl.perception.vslam import VSLAMConfig
from amr_rl.robot.spec import RobotSpec
from amr_rl.sim.sensors import DPS, EncoderModel, ImuErrorModel

SPEC = RobotSpec.load()


def _samples(t0, n, rate, bias, wl, wr, counts, dt=0.01, rng=None, cpr=4480.0):
    """n IMU + encoder samples of a body turning at ``rate`` (rad/s) with wheel speeds
    wl, wr (rad/s); ``counts``: running [left, right] wheel angles (rad), updated."""
    rng = rng or np.random.default_rng(0)
    imu, enc = [], []
    for k in range(1, n + 1):
        t = round(t0 + k * dt, 9)
        counts[0] += wl * dt
        counts[1] += wr * dt
        imu.append(ImuSample(t, np.array([0.0, 0.0, rate + bias + rng.normal(0, 4e-4)]), np.array([0.0, 0.0, 9.81])))
        enc.append(EncoderSample(t, int(math.floor(counts[0] / (2 * math.pi) * cpr)),
                                 int(math.floor(counts[1] / (2 * math.pi) * cpr))))
    return ProprioBatch(imu, enc)


def test_still_period_measures_the_gyro_offset_and_turns_integrate():
    odo = WheelInertialOdometry(SPEC)
    bias, counts, t = 0.017, [0.0, 0.0], 0.0
    odo.step(_samples(t, 10, 0.0, bias, 0, 0, counts), t + 0.1)
    for k in range(1, 15):  # 1.4 s still
        odo.step(_samples(k * 0.1, 10, 0.0, bias, 0, 0, counts), (k + 1) * 0.1)
    assert odo.bias_calibrated and abs(odo.bias - bias) < 2e-4
    r, track = SPEC.drive["wheel_radius"], SPEC.drive["track"]
    w = 0.45
    wheel = w * track / 2 / r
    total = 0.0
    for k in range(15, 15 + 70):  # 7 s turn
        st = odo.step(_samples(k * 0.1, 10, w, bias, -wheel, wheel, counts), (k + 1) * 0.1)
        total += st.dth
        assert not st.slip
    assert abs(total - w * 7.0) < 0.01  # < 0.6 deg over 180 deg
    assert st.cov[2, 2] < (0.002) ** 2


def test_spinning_wheels_and_disagreeing_gyro_are_flagged_as_slip():
    odo = WheelInertialOdometry(SPEC)
    counts = [0.0, 0.0]
    odo.step(_samples(0, 10, 0, 0, 0, 0, counts), 0.1)
    for k in range(1, 10):
        odo.step(_samples(k * 0.1, 10, 0, 0, 0, 0, counts), (k + 1) * 0.1)
    # one wheel spins (encoders say turning), the body does not turn (gyro 0)
    st = odo.step(_samples(1.0, 10, 0.0, 0.0, 0.0, 6.0, counts), 1.1, command=(0.0, 0.0))
    assert st.slip and "yaw" in st.info["slip_reasons"]
    assert st.cov[0, 0] > 0.01 ** 2  # translation no longer trusted
    # both wheels spin far faster than commanded (blocked robot, spinning wheels)
    odo2 = WheelInertialOdometry(SPEC)
    counts = [0.0, 0.0]
    odo2.step(_samples(0, 10, 0, 0, 0, 0, counts), 0.1)
    st = odo2.step(_samples(0.1, 10, 0.0, 0.0, 8.0, 8.0, counts), 0.2, command=(0.05, 0.0))
    assert st.slip and "command" in st.info["slip_reasons"]


def test_missing_sensors_degrade_gracefully():
    odo = WheelInertialOdometry(SPEC)
    counts = [0.0, 0.0]
    full = _samples(0, 10, 0, 0, 0, 0, counts)
    odo.step(full, 0.1)
    enc_only = _samples(0.1, 10, 0.3, 0, -1.0, 1.0, counts)
    st = odo.step(ProprioBatch([], enc_only.encoders), 0.2)
    assert st.rotation_source == "encoders" and st.cov[2, 2] > 1e-4
    imu_only = _samples(0.2, 10, 0.3, 0, 0, 0, counts)
    st = odo.step(ProprioBatch(imu_only.imu, []), 0.3, command=(0.1, 0.3))
    assert st.translation_source == "command" and st.rotation_source == "gyro"
    assert odo.step(ProprioBatch([], []), 0.4) is None


def test_gyro_scale_samples_and_calibration_prior():
    odo = WheelInertialOdometry(SPEC, calibration={"gyro_scale": 1.0025, "gyro_scale_sigma": 0.0002})
    assert odo.gyro_scale == 1.0025
    for _ in range(5):  # in-run samples consistent with the calibration only monitor it
        odo.add_scale_sample(1.0027, 0.0005)
    assert odo.gyro_scale == 1.0025
    for _ in range(10):  # a clear, repeated disagreement (e.g. temperature) replaces it
        odo.add_scale_sample(1.0060, 0.0005)
    assert abs(odo.gyro_scale - 1.0060) < 1e-9 and odo.log["scale_recalibrated"] >= 1
    assert not odo.add_scale_sample(1.2, 0.001)  # implausible: ignored
    free = WheelInertialOdometry(SPEC)  # no calibration: samples refine the datasheet prior
    for _ in range(5):
        free.add_scale_sample(1.0027, 0.0005)
    assert 1.002 < free.gyro_scale < 1.0028


def test_imu_model_quantises_bounds_and_keeps_the_unit_fixed():
    cfg = SPEC.imu
    a = ImuErrorModel(cfg, np.random.default_rng(1), 0.01, unit_rng=np.random.default_rng([1, 0x5E1]))
    b = ImuErrorModel(cfg, np.random.default_rng(2), 0.01, unit_rng=np.random.default_rng([1, 0x5E1]))
    assert np.allclose(a.g_M, b.g_M)  # same physical unit: same sensitivity errors
    assert abs(a.g_M[2, 2] - 1) <= cfg["gyro"]["sensitivity_max"]
    assert np.all(np.abs(a.g_bias) <= cfg["gyro"]["offset_max_dps"] * DPS)
    g = a.gyro(np.array([0.0, 0.0, 0.3]))
    lsb = cfg["gyro"]["lsb_mdps"] * 1e-3 * DPS
    assert np.allclose(g / lsb, np.round(g / lsb))
    big = a.gyro(np.array([0.0, 0.0, 100.0]))
    assert big[2] <= cfg["gyro"]["full_scale_dps"] * DPS


def test_encoder_counts_and_backlash():
    enc = EncoderModel({"counts_per_wheel_rev": 4480, "backlash_deg": 1.0})
    assert enc.count(0.0) == 0
    assert enc.count(2 * math.pi) == 4480 - 1 or enc.count(2 * math.pi) >= 4470
    # reversing by less than the backlash does not move the counter
    before = enc.count(2 * math.pi + 0.001)
    assert enc.count(2 * math.pi + 0.001 - math.radians(0.9)) == before


def test_fusion_keeps_a_near_box_turn_on_the_gyro():
    """Box face 0.30 m from the axis: vision alone over-rotates by > 8 deg per turn; with
    the gyro (scale error 0.05 %) fused the turn stays within 1.5 deg (measured 1.04),
    including ~40 frames dead-reckoned on the gyro where vision found no fix."""
    _, rows = run_turn(VSLAMConfig(), box_dist=0.30, turn_deg=360, start_heading=math.pi)
    assert abs([r[2] for r in rows if r[2] is not None][-1]) > 8
    slam, rows = run_turn(VSLAMConfig(), box_dist=0.30, turn_deg=360, start_heading=math.pi,
                          odometry={"scale": 1.0005, "sigma": 2e-4})
    errs = [abs(r[2]) for r in rows if r[2] is not None]
    assert max(errs) < 1.5
    assert slam.fusion_log["frames"] > 50


def test_fusion_is_inert_without_odometry():
    slam, rows = run_turn(VSLAMConfig(), box_dist=None, turn_deg=90)
    assert slam.fusion_log["frames"] == 0 and slam._cov is None


def test_wheel_model_reproduces_the_measured_turn_responses():
    """Live ratios of true to commanded yaw per 0.1 s frame (round-9 analysis):
    from rest 0.27 / 0.76 / 0.83; after a reversal -0.36 / 0.36 / 0.79."""
    from amr_rl.odometry.wheel_model import WheelResponse

    m = WheelResponse.from_spec(SPEC)
    ratios = [m.step((0.0, 0.45), 0.1)[0][2] / 0.045 for _ in range(4)]
    assert abs(ratios[0] - 0.27) < 0.05 and abs(ratios[1] - 0.76) < 0.06 and abs(ratios[3] - 0.83) < 0.01
    rev = [m.step((0.0, -0.45), 0.1)[0][2] / -0.045 for _ in range(3)]
    assert abs(rev[0] + 0.36) < 0.1 and abs(rev[1] - 0.36) < 0.12 and abs(rev[2] - 0.79) < 0.05


def test_wheel_model_does_not_cancel_a_sweep_against_a_sharp_arc():
    """The false 'angular' losses: a left survey sweep, then a sharp right arc. Signed
    commands nearly cancel over the 3 s window; the modelled motion follows the real
    (lagged, slipped) yaw, so vision agrees with it."""
    from amr_rl.odometry.wheel_model import WheelResponse

    truth, model = WheelResponse.from_spec(SPEC), WheelResponse.from_spec(SPEC)
    cmds = [(0.0, 0.45)] * 20 + [(0.2, -0.87)] * 10
    true_rot = sum(truth.step(c, 0.1)[0][2] for c in cmds)
    mod_rot = sum(model.step(c, 0.1)[0][2] for c in cmds)
    cmd_rot = sum(c[1] * 0.1 for c in cmds)
    assert abs(true_rot - mod_rot) < 1e-9
    assert abs(true_rot - cmd_rot) > 0.25  # the raw command is far off


def test_command_model_odometry_is_not_trusted_over_vision():
    from amr_rl.odometry.wheel_model import CommandModelOdometry

    co = CommandModelOdometry(SPEC)
    assert co.step((0.0, 0.45), 0.0) is None
    st = co.step((0.0, 0.45), 0.1)
    assert not st.trust_rotation and st.rotation_source == "command_model" and st.dth > 0


def test_wheel_model_consistency_keeps_a_normal_turn_tracking():
    slam, rows = run_turn(VSLAMConfig(wheel_model="consistency"), box_dist=None, turn_deg=180)
    assert all(r[1] == "tracking" for r in rows[1:])


def test_wheel_radius_scale_is_learned_from_vision_distances():
    """In simulation the wheels roll ~10 % short of the spec radius (true / encoder
    distance 0.89-0.91 on straight frames of live runs): learned online."""
    odo = WheelInertialOdometry(SPEC)
    assert odo.radius_scale == 1.0 and math.sqrt(odo.radius_var) >= 0.1
    for r in (0.90, 0.89, 0.91, 0.90, 1.6):  # the last: a slip or vision fault, ignored
        odo.add_radius_sample(r)
    assert abs(odo.radius_scale - 0.90) < 0.006 and math.sqrt(odo.radius_var) < 0.02
    counts = [0.0, 0.0]
    odo.step(_samples(0, 10, 0, 0, 0, 0, counts), 0.1)
    st = odo.step(_samples(0.1, 10, 0.0, 0.0, 4.0, 4.0, counts), 0.2)
    assert abs(st.distance - 0.9 * 0.05 * 4.0 * 0.1) < 0.002


def test_runtime_holds_the_wheels_for_imu_zero_velocity_updates():
    from amr_rl.runtime.robot import RobotRuntime, RuntimeConfig

    rt = RobotRuntime.__new__(RobotRuntime)
    rt.cfg = RuntimeConfig(odometry="imu_encoders")
    rt.odo = WheelInertialOdometry(SPEC)
    rt._zupt_hold_until, rt.zupt_holds = -math.inf, 0
    assert rt._imu_booting(0.5)  # boot: offset measurement first
    rt.odo.last_zupt_t = 0.9
    assert not rt._imu_booting(1.0)
    assert not rt._imu_booting(30.0)
    assert rt._imu_booting(31.0) and rt.zupt_holds == 1  # 30 s without one: a short stop
    assert rt._imu_booting(31.5) and not rt._imu_booting(31.9)
    rt.odo.last_zupt_t = 31.6
    assert not rt._imu_booting(50.0)
    rt.odo = None  # camera-only robot: never held
    assert not rt._imu_booting(0.1)
