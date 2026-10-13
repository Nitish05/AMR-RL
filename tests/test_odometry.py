"""Round 9: IMU + wheel-encoder sensor models, wheel-inertial odometry and its fusion
with the VSLAM (no simulator; docs/results/heading-r9.md)."""

import math

import numpy as np
import pytest
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
    assert st.rotation_source == "encoders" and st.cov[2, 2] > 1e-4 and not st.trust_rotation
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
    rt._zupt_hold_until, rt._zupt_hold_start, rt.zupt_holds, rt.zupt_hold_failures = -math.inf, None, 0, 0
    rt._zupt_frames_at_hold = None
    rt._boot_t0 = 0.0  # first control tick at t = 0
    rt.zupt_hold_log = []
    assert rt._imu_booting(0.5)  # boot: offset measurement first
    rt.cfg.imu_zupt_interval_s = 30.0
    assert rt._imu_booting(1.5)  # 2 s boot stand-still
    rt.odo.last_bias_update_t = 1.9
    rt.odo.bias_var = math.radians(0.05) ** 2  # offset uncertain: a stop is necessary
    assert not rt._imu_booting(2.1)
    assert not rt._imu_booting(30.0)
    assert rt._imu_booting(32.5) and rt.zupt_holds == 1  # no offset update for 30 s: a short stop
    assert rt._imu_booting(33.0)  # held until the offset is measured (10 still frames)...
    rt.odo.last_bias_update_t, rt.odo.zupt_frames = 33.1, rt.odo.zupt_frames + 10
    assert not rt._imu_booting(33.2)  # ...then driving resumes at once
    assert not rt._imu_booting(50.0)
    assert rt._imu_booting(64.0) and rt._imu_booting(66.4)  # never measured: at most 2.5 s
    assert not rt._imu_booting(66.6) and rt.zupt_hold_failures == 1
    assert [h["trigger"] for h in rt.zupt_hold_log] == ["interval", "interval"]
    assert rt.zupt_hold_log[0]["success"] and not rt.zupt_hold_log[1]["success"]
    assert abs(rt.zupt_hold_log[1]["end"] - rt.zupt_hold_log[1]["start"] - 2.6) < 1e-9
    rt.odo = None  # camera-only robot: never held
    assert not rt._imu_booting(0.1)


def test_straight_driving_measures_the_gyro_offset_without_stopping():
    from amr_rl.odometry.fusion import OdometryConfig

    odo = WheelInertialOdometry(SPEC, OdometryConfig(zaru=True))  # off by default (simulated skid)
    counts, bias = [0.0, 0.0], 0.0175
    odo.step(_samples(0, 10, 0, bias, 0, 0, counts), 0.1)
    odo.bias, odo.bias_var = 0.0175 + 0.0004, (0.0006) ** 2  # offset drifted since the last stop
    for k in range(1, 61):  # 6 s straight at 0.2 m/s (4 rad/s per wheel)
        odo.step(_samples(k * 0.1, 10, 0.0, bias, 4.0, 4.0, counts), (k + 1) * 0.1)
    assert odo.log.get("zaru", 0) >= 2 and odo.log["zupt"] == 0
    assert abs(odo.bias - bias) < 0.00015 and odo.last_bias_update_t > 1.0


def test_no_recalibration_stop_when_the_offset_is_still_known():
    from amr_rl.runtime.robot import RobotRuntime, RuntimeConfig

    rt = RobotRuntime.__new__(RobotRuntime)
    rt.cfg = RuntimeConfig(odometry="imu_encoders")
    rt.odo = WheelInertialOdometry(SPEC)
    rt._zupt_hold_until, rt._zupt_hold_start, rt.zupt_holds, rt.zupt_hold_failures = -math.inf, None, 0, 0
    rt._zupt_frames_at_hold = None
    rt._boot_t0 = 0.0  # first control tick at t = 0
    rt.zupt_hold_log = []
    rt.odo.last_bias_update_t = 1.0
    rt.odo.bias_var = math.radians(0.01) ** 2  # vision kept it known
    assert not rt._imu_booting(200.0) and rt.zupt_holds == 0
    rt.odo.bias_var = math.radians(0.03) ** 2
    assert rt._imu_booting(200.0) and rt.zupt_holds == 1


def test_offset_is_measured_from_a_zero_command_when_the_encoders_are_out():
    odo = WheelInertialOdometry(SPEC)
    counts, bias = [0.0, 0.0], 0.0175
    for k in range(15):
        b = _samples(k * 0.1, 10, 0.0, bias, 0, 0, counts)
        odo.step(ProprioBatch(b.imu, []), (k + 1) * 0.1, command=(0.0, 0.0))
    assert odo.bias_calibrated and abs(odo.bias - bias) < 3e-4


def test_vision_window_offset_measurements_update_with_their_own_uncertainty():
    odo = WheelInertialOdometry(SPEC)
    odo.bias, odo.bias_var = 0.0, math.radians(0.05) ** 2
    true = math.radians(0.03)  # the offset is 0.03 dps more than estimated
    for _ in range(6):  # six 20 s windows, each measuring the residual +-0.016 dps
        odo.add_bias_feedback(true - odo.bias, math.radians(0.016))
    assert abs(math.degrees(odo.bias) - 0.03) < 0.012
    assert math.degrees(math.sqrt(odo.bias_var)) > 0.005  # not falsely certain
    before = odo.bias
    odo.add_bias_feedback(math.radians(1.0), math.radians(0.016))  # implausible: rejected
    assert odo.bias == before and odo.log["bias_feedback_rejected"] == 1


def test_temperature_compensation_tracks_the_warm_up_drift():
    """Offset = b0 + k (T - T0): learned from stops at different chip temperatures,
    then followed between stops while the robot warms up (no stop needed)."""
    odo = WheelInertialOdometry(SPEC)
    b0, k = 0.017, math.radians(-0.02)  # -0.02 dps/degC (datasheet +-0.010 typ)
    t, counts = 0.0, [0.0, 0.0]

    def still(temp, seconds):
        nonlocal t
        for _ in range(int(seconds * 10)):
            b = _samples(t, 10, 0.0, b0 + k * (temp - 22.0), 0, 0, counts)
            imu = [ImuSample(s.t, s.gyro, s.accel, temp) for s in b.imu]
            odo.step(ProprioBatch(imu, b.encoders), round(t + 0.1, 9))
            t = round(t + 0.1, 9)

    still(22.0, 1.5)   # boot
    still(26.0, 1.5)   # a stop after warming 4 degC
    still(29.0, 1.5)   # another at +7 degC
    assert abs(odo.tempco[0] - k) < math.radians(0.004)
    odo.temp = 33.0    # warmer still, no stop: the offset follows the temperature
    assert abs(odo.bias - (b0 + k * 11.0)) < math.radians(0.03)


def _runtime_for_holds():
    from amr_rl.runtime.robot import RobotRuntime, RuntimeConfig

    rt = RobotRuntime.__new__(RobotRuntime)
    rt.cfg = RuntimeConfig(odometry="imu_encoders")
    rt.odo = WheelInertialOdometry(SPEC)
    rt._zupt_hold_until, rt._zupt_hold_start, rt.zupt_holds, rt.zupt_hold_failures = -math.inf, None, 0, 0
    rt._zupt_frames_at_hold, rt._boot_t0 = None, None
    rt.zupt_hold_log = []
    return rt


def test_boot_hold_is_measured_from_the_first_tick_not_absolute_time():
    """A runtime started at a large clock value (hardware monotonic time) still stands
    still for imu_boot_still_s, even with periodic recalibration disabled."""
    rt = _runtime_for_holds()
    rt.cfg.imu_zupt_interval_s = 0.0
    assert rt._imu_booting(5000.0) and rt._imu_booting(5001.5)
    assert not rt._imu_booting(5002.1)


def test_encoder_heading_without_the_imu_keeps_a_straight_drive_straight():
    """IMU missing: positions within the frame follow the encoders' differential yaw
    (a typo used the right wheel's travel alone, bending straight drives)."""
    odo = WheelInertialOdometry(SPEC)
    counts = [0.0, 0.0]
    odo.step(_samples(0, 10, 0, 0, 0, 0, counts), 0.1)
    for k in range(1, 4):
        b = _samples(k * 0.1, 10, 0.0, 0.0, 6.0, 6.0, counts)
        st = odo.step(ProprioBatch([], b.encoders), (k + 1) * 0.1)
        assert st.rotation_source == "encoders" and abs(st.dth) < 1e-9
        assert st.dx > 0.025 and abs(st.dy) < 1e-6
    # an arc: the within-frame chord follows half the frame's encoder yaw
    b = _samples(0.4, 10, 0.0, 0.0, 4.0, 8.0, counts)
    st = odo.step(ProprioBatch([], b.encoders), 0.5)
    assert st.dth > 0 and abs(math.atan2(st.dy, st.dx) - st.dth / 2) < 0.1 * st.dth


def test_gyro_dropout_is_not_integrated_twice():
    """IMU silent for 0.5 s during a turn (frames dead-reckoned on encoder yaw): the
    first frame after it integrates its own 0.1 s only, not the whole gap again
    (measured before the fix: ~0.15 rad extra here)."""
    odo = WheelInertialOdometry(SPEC)
    r, track = SPEC.drive["wheel_radius"], SPEC.drive["track"]
    w = 0.3
    wheel = w * track / 2 / r
    counts = [0.0, 0.0]
    odo.step(_samples(0, 10, 0, 0, 0, 0, counts), 0.1)
    odo.step(_samples(0.1, 10, w, 0, -wheel, wheel, counts), 0.2)
    for k in range(2, 7):
        b = _samples(k * 0.1, 10, w, 0, -wheel, wheel, counts)
        assert odo.step(ProprioBatch([], b.encoders), (k + 1) * 0.1).rotation_source == "encoders"
    st = odo.step(_samples(0.7, 10, w, 0, -wheel, wheel, counts), 0.8)
    assert st.rotation_source == "gyro" and abs(st.dth - w * 0.1) < 0.003
    assert odo.log.get("imu_gaps") == 1


def test_encoder_dropout_travel_is_not_counted_twice():
    """Encoders silent for 0.5 s while driving (distance from the command meanwhile):
    the cumulative counts on their return add only this frame's share."""
    odo = WheelInertialOdometry(SPEC)
    counts = [0.0, 0.0]
    wheel = 0.2 / SPEC.drive["wheel_radius"]  # 0.2 m/s
    odo.step(_samples(0, 10, 0, 0, wheel, wheel, counts), 0.1)
    odo.step(_samples(0.1, 10, 0, 0, wheel, wheel, counts), 0.2, command=(0.2, 0.0))
    for k in range(2, 7):
        b = _samples(k * 0.1, 10, 0, 0, wheel, wheel, counts)
        st = odo.step(ProprioBatch(b.imu, []), (k + 1) * 0.1, command=(0.2, 0.0))
        assert st.translation_source == "command"
    st = odo.step(_samples(0.7, 10, 0, 0, wheel, wheel, counts), 0.8, command=(0.2, 0.0))
    assert st.translation_source == "encoders" and abs(st.distance - 0.02) < 0.002
    assert odo.log.get("enc_gaps") == 1


def test_out_of_order_samples_are_dropped():
    odo = WheelInertialOdometry(SPEC)
    counts = [0.0, 0.0]
    odo.step(_samples(0, 10, 0, 0, 0, 0, counts), 0.1)
    b = _samples(0.1, 10, 0.2, 0, 0, 0, counts)
    stale = ImuSample(0.05, np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 9.81]))
    st = odo.step(ProprioBatch([stale] + b.imu, b.encoders), 0.2)
    assert abs(st.dth - 0.02) < 0.002 and odo.log["imu_out_of_order"] == 1


class _FakeStep:
    def __init__(self, dth, dt, updates, trust=True):
        self.dth, self.w, self.trust_rotation = dth, dth / dt, trust
        self.info = {"offset_updates": updates, "scale_sigma": 0.0002}


def _window_slam(**cfg):
    from amr_rl.perception.vslam import PlanarVSLAM

    slam = PlanarVSLAM.__new__(PlanarVSLAM)
    slam.cfg = VSLAMConfig(**cfg)
    slam._probation, slam._bias_win, slam.bias_feedback = 0, None, []
    slam._scale_anchor = slam._radius_acc = slam._vision_pose = None
    slam.fusion_log = {}
    return slam


def test_vision_offset_window_completes_while_the_temperature_model_moves():
    """The window measures (gyro rotation - vision rotation) / T over 15-90 s. It used to
    restart whenever the temperature-compensated offset value changed (every frame), so
    it never completed; now only a real re-measurement of the offset restarts it."""
    err = math.radians(0.02)  # the gyro reads 0.02 dps high after correction
    slam = _window_slam(odo_bias_feedback=True)
    th = 0.0
    for k in range(200):  # 20 s, slow drive, established landmarks
        true = 0.01 * math.sin(0.3 * k * 0.1)
        dth = (true - th) + err * 0.1
        th = true
        slam._bias_window(np.array([0.0, 0.0, true]), _FakeStep(dth, 0.1, updates=7), 0.9, 1.0, k * 0.1)
    assert len(slam.bias_feedback) == 1
    rate, sigma = slam.bias_feedback[0]
    assert abs(rate - err) < 1e-6 and 0 < sigma < math.radians(0.05)  # gyro - vision: offset up
    # a re-measurement of the offset (a stop) mid-window restarts it
    slam = _window_slam(odo_bias_feedback=True)
    for k in range(200):
        slam._bias_window(np.zeros(3), _FakeStep(0.0, 0.1, updates=7 if k < 100 else 8), 0.9, 1.0, k * 0.1)
    assert slam.bias_feedback == [] and slam._bias_win is not None
    # off by default (validation 4 ran without this feedback)
    slam = _window_slam()
    for k in range(200):
        slam._bias_window(np.zeros(3), _FakeStep(0.0, 0.1, updates=7), 0.9, 1.0, k * 0.1)
    assert slam.bias_feedback == [] and slam._bias_win is None


def test_map_corrections_and_tracking_gaps_restart_vision_windows():
    slam = _window_slam(odo_bias_feedback=True)
    for k in range(100):
        slam._bias_window(np.zeros(3), _FakeStep(0.0, 0.1, updates=1), 0.9, 1.0, k * 0.1)
    slam._scale_anchor, slam._radius_acc, slam._vision_pose = {"th": 0.0}, [0.1, 0.1], np.zeros(3)
    assert slam._bias_win is not None
    slam._reset_vision_refs()
    assert slam._bias_win is None and slam._scale_anchor is None and slam._radius_acc is None
    assert slam._vision_pose is None


def _chain_slam(n=40, drift_per_kf=math.radians(0.002), trusted=True):
    """Keyframes 0..n driving a 1 m radius circle (0.1 rad per keyframe, 1 s apart)
    with odometry links between consecutive keyframes."""
    from amr_rl.perception.vslam import Keyframe, PlanarVSLAM, compose

    slam = PlanarVSLAM.__new__(PlanarVSLAM)
    slam.cfg = VSLAMConfig()
    step = np.array([math.sin(0.1), 1 - math.cos(0.1), 0.1])
    pose, kfs = np.zeros(3), []
    for i in range(n + 1):
        kf = Keyframe(i, float(i), pose.copy(), np.zeros((0, 2)), np.zeros((0, 32), np.uint8), np.zeros(0, int))
        if i > 0:
            kf.odo_rel = step.copy()
            kf.odo_cov = np.diag([1e-6, 1e-6, 1e-8])
            kf.odo_drift = drift_per_kf if trusted else math.inf
        kfs.append(kf)
        pose = compose(pose, step)
    slam.keyframes = kfs
    return slam, kfs


def test_loop_odometry_check_rejects_a_closure_the_gyro_contradicts():
    """Validation 4's worst run: a verified, confirmed loop 0.21 m and 12 deg off was
    accepted while the gyro knew the relative heading to ~0.1 deg."""
    from amr_rl.perception.vslam import between, compose

    slam, kfs = _chain_slam()
    cand, kf = kfs[2], kfs[38]
    true_z = between(cand.pose, kf.pose)
    ok = slam._odometry_loop_check(cand, kf, true_z + np.array([0.01, -0.01, math.radians(0.3)]))
    assert ok["chain"] == "ok" and ok["pass_heading"] and ok["pass_full"]
    assert ok["sigma_th_deg"] < 0.6
    bad = slam._odometry_loop_check(cand, kf, compose(true_z, (0.15, 0.15, math.radians(12))))
    assert not bad["pass_heading"] and not bad["pass_full"] and abs(bad["res_th_deg"]) > 11
    shifted = slam._odometry_loop_check(cand, kf, true_z + np.array([0.5, 0.0, 0.0]))  # heading right, 0.5 m off
    assert shifted["pass_heading"] and not shifted["pass_full"]
    # a broken chain (a loss of tracking between the two) cannot be checked
    kfs[20].odo_rel = None
    assert slam._odometry_loop_check(cand, kf, true_z) is None
    # rotation not from a trusted gyro: no heading test
    slam, kfs = _chain_slam(trusted=False)
    res = slam._odometry_loop_check(kfs[2], kfs[38], compose(between(kfs[2].pose, kfs[38].pose), (0, 0, 0.2)))
    assert res["chain"] == "untrusted_rotation" and res["pass_heading"]


def test_loop_odometry_tolerance_grows_with_the_gyro_offset_uncertainty():
    slam, kfs = _chain_slam(drift_per_kf=math.radians(0.02))  # 0.02 dps offset sigma, 1 s per keyframe
    from amr_rl.perception.vslam import between, compose

    z = compose(between(kfs[2].pose, kfs[38].pose), (0, 0, math.radians(1.5)))
    res = slam._odometry_loop_check(kfs[2], kfs[38], z)
    assert res["sigma_th_deg"] > 0.7 and res["pass_heading"]  # 36 s x 0.02 dps = 0.72 deg


def _motion_slam(entries):
    from amr_rl.perception.vslam import PlanarVSLAM

    slam = PlanarVSLAM.__new__(PlanarVSLAM)
    slam.cfg = VSLAMConfig()
    slam._motion_log = entries
    slam.inconsistency = None
    return slam


def test_blocked_robot_is_noticed_although_the_wheels_report_no_travel_or_slip():
    """Pushing on an obstacle the camera does not see: commanded forward 0.2 m/s for
    1.2 s, vision (correctly) static. Stalled wheels give no encoder travel and spinning
    wheels are slip-flagged, so the odometry frozen check alone never fired."""
    still = np.zeros(3)
    for travel, slip in ((0.0, False), (0.02, True)):  # stalled; spinning (flagged)
        log = [(k * 0.1, still.copy(), travel, 0.0, slip, 0.02) for k in range(13)]
        slam = _motion_slam(log)
        assert not slam._odometry_consistent(1.2) and slam.inconsistency == "blocked"
        slam = _motion_slam(log)
        slam.cfg.odo_blocked_check = False
        assert slam._odometry_consistent(1.2)  # the round-9 behaviour
    # driving normally: vision follows the command
    log = [(k * 0.1, np.array([0.02 * k, 0.0, 0.0]), 0.02, 0.0, False, 0.02) for k in range(13)]
    assert _motion_slam(log)._odometry_consistent(1.2)
    # turning in place / standing still for a recalibration: nothing commanded forward
    log = [(k * 0.1, still.copy(), 0.0, 0.05, False, 0.0) for k in range(13)]
    assert _motion_slam(log)._odometry_consistent(1.2)


def test_blocked_without_vision_from_command_and_encoders():
    slam = _motion_slam([])
    slam._proprio_win = [(k * 0.1, 0.02, 0.0) for k in range(13)]  # commanded 0.2 m/s, wheels stalled or slipping
    assert slam.blocked_by_odometry(1.2)
    slam._proprio_win = [(k * 0.1, 0.02, 0.019) for k in range(13)]  # moving as commanded
    assert not slam.blocked_by_odometry(1.2)
    slam._proprio_win = [(k * 0.1, 0.0, 0.0) for k in range(13)]  # standing still on purpose
    assert not slam.blocked_by_odometry(1.2)
    slam._proprio_win = [(1.0 + k * 0.1, 0.02, 0.0) for k in range(6)]  # 0.5 s window at loss time
    assert slam.blocked_by_odometry(1.5, window=0.5) and not slam.blocked_by_odometry(1.5)


def test_recovery_backs_off_when_the_wheels_show_a_blocked_robot():
    from amr_rl.runtime.robot import choose_recovery

    common = dict(nudge_retrace=None, turning_clearance=0.5, required_clearance=0.3, certified=True)
    assert choose_recovery(loss_reason="tracking_failed", last_command=(0.2, 0.0), **common)["kind"] == "rotate"
    assert choose_recovery(loss_reason="tracking_failed", last_command=(0.2, 0.0), blocked=True,
                           **common)["kind"] == "back_off"


def test_rotation_recovery_stops_when_the_gyro_shows_no_turn():
    from types import SimpleNamespace

    from amr_rl.runtime.robot import RobotRuntime

    rt = RobotRuntime.__new__(RobotRuntime)
    rt.odo = WheelInertialOdometry(SPEC)
    rt.recovery_plan = {"kind": "rotate"}
    rt.last_odometry = SimpleNamespace(rotation_source="gyro", dt=0.1, dth=0.002)  # pressed against something
    cmds = [rt._recovery_step(k * 0.1) for k in range(14)]
    assert cmds[0] == (0.0, 0.35) and cmds[-1] == (0.0, 0.0)
    assert rt.recovery_plan["kind"] == "wait" and rt.recovery_plan["reason"] == "rotation_blocked"
    rt.recovery_plan = {"kind": "rotate"}
    rt.last_odometry = SimpleNamespace(rotation_source="gyro", dt=0.1, dth=0.034)  # turning as commanded
    assert all(rt._recovery_step(k * 0.1) == (0.0, 0.35) for k in range(30))


def _scoring_module(name):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_eval_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def test_turn_live_report_scores_drift_turns_strata_and_losses():
    tlr = _scoring_module("turn_live_report")
    rows, t = [], 0.0
    def row(status, est_th, gt_th, cmd, err=0.01):
        nonlocal t
        t += 0.1
        rows.append({"t": round(t, 1), "status": status, "est": [0.0, 0.0, est_th], "gt": [0.0, 0.0, gt_th],
                     "err": err, "herr": wrap_angle(est_th - gt_th), "cmd": cmd})
    def wrap_angle(a):
        return (a + math.pi) % (2 * math.pi) - math.pi
    for _ in range(10):
        row("tracking", 0.0, 0.0, [0.0, 0.0])
    for k in range(20):  # in-place turn, the estimate over-rotates 10 %: drift 0.9 rad * 0.1 = 5.2 deg
        row("tracking", 1.1 * 0.045 * (k + 1), 0.045 * (k + 1), [0.0, 0.45])
    for k in range(30):  # then 4 deg more within 10 s: an 8 deg drift event
        row("tracking", 0.99 + math.radians(4) * k / 29, 0.9, [0.2, 0.0])
    row("lost", 0.0, 0.9, [0.2, 0.0])  # given up while 9 deg off: a genuine loss
    out = tlr.analyse(rows)
    assert out["turns"] == 1 and out["drift_events"] == 1
    assert abs(out["turn_scale"]["median"] - 1.1) < 1e-6 and out["turn_scale"]["turns"] == 1
    assert out["losses"] == {"false": 0, "genuine": 1}
    assert out["strata"]["motion:turn"]["frames"] == 20 and out["strata"]["motion:straight"]["frames"] == 30
    found = {"L1-a-w-s0": {"max_heading_err_deg": 10.0, "ate_cm": 5.0, "turns": 10, "drift_events": 2,
                           "losses_false": 1, "losses_genuine": 0, "strata": out["strata"], "turn_scale": None},
             "L1-b-w-s0": {"max_heading_err_deg": 2.0, "ate_cm": 3.0, "turns": 10, "drift_events": 0,
                           "losses_false": 0, "losses_genuine": 0, "strata": None, "turn_scale": None}}
    cmp = tlr.compare(found, "a", "b")
    assert cmp["pairs"] == 1 and cmp["max_heading_err_deg"]["median_diff"] == -8.0
    assert cmp["drift_a"]["per_100"] == 20.0 and cmp["losses_a"]["false"] == 1


def test_reversal_scorer_uses_true_panel_events():
    """Round 6 seed 0 late: a misperceived 'moved' on stone before the swap (valence 0.6)
    hid the adaptation when the scorer used perceived outcomes."""
    report = _scoring_module("report")
    phase = {"switches": [{"t": 360.0, "outcomes_before": 11}],
             "outcomes": [{"t": 126.0, "fixture": "stone", "action": "signal", "valence": 0.6},
                          {"t": 451.0, "fixture": "stone", "action": "signal", "valence": 1.0}],
             "world_events": [{"time": 120.0, "fixture": "stone", "trigger": "signal", "response": "none"},
                              {"time": 200.0, "fixture": "bloom", "trigger": "signal", "response": "yellow"},
                              {"time": 447.0, "fixture": "stone", "trigger": "signal", "response": "yellow"}]}
    assert report.first_new_rewarding(phase, 360.0) == (447.0, "stone", "signal")
    del phase["world_events"]  # perceived outcomes only: the old miscount
    assert report.first_new_rewarding(phase, 360.0) is None
    table = report.reversal_table([("reversal_late", 0, {"phases": [{**phase, "world_events": [
        {"time": 447.0, "fixture": "stone", "trigger": "signal", "response": "yellow"}]}]}, None)])
    assert "| yes |" in table


def test_contact_split_counts_a_nudge_unfinished_at_phase_end_as_intended():
    """Suite inert s4 (2026-10-11): the last decision (479.2 s) chose a nudge on the
    roller; the phase ended before it finished, so its contacts were scored unintended."""
    report = _scoring_module("report")
    contacts = [{"t": 470.0}, {"t": 485.1}, {"t": 486.5}, {"t": 487.0}]
    interactions = [{"started": 463.0, "t": 479.1, "action": "nudge"}]
    nudge_last = [{"t": 479.2, "chosen": {"activity": "engage", "action": "nudge"}}]
    assert report.contact_split(contacts, interactions) == "1/2"
    assert report.contact_split(contacts, interactions, nudge_last) == "3/0"
    signal_last = [{"t": 479.2, "chosen": {"activity": "engage", "action": "signal"}}]
    assert report.contact_split(contacts, interactions, signal_last) == "1/2"
    # a nudge decision whose interaction was recorded opens no extra window
    done = interactions + [{"started": 479.2, "t": 484.0, "action": "nudge"}]
    assert report.contact_split(contacts, done, nudge_last) == "1/2"


def test_report_labels_the_sensors_each_run_used():
    report = _scoring_module("report")
    assert report.sensors_label([{"odometry": {"source": "imu_encoders"}}]) == "onboard RGB + IMU + wheel encoders"
    assert report.sensors_label([{}]) == "onboard RGB only"  # before round 9: no odometry record
    assert report.sensors_label([{"odometry": {"source": "imu_encoders"}}, {}]) == (
        "onboard RGB only; onboard RGB + IMU + wheel encoders")


def test_time_windowed_and_periodic_sensor_faults():
    from amr_rl.sim.sensors import ProprioSensors

    s = ProprioSensors.__new__(ProprioSensors)
    s.faults = {"imu_off_windows": [[10.0, 10.5]], "encoders_dropouts": {"start": 30.0, "period": 60.0, "duration": 2.0},
                "left_encoder_off_windows": [[5.0, 6.0]]}
    assert not s._off("imu", 9.99) and s._off("imu", 10.0) and s._off("imu", 10.49) and not s._off("imu", 10.5)
    assert not s._off("encoders", 29.0) and s._off("encoders", 31.0) and not s._off("encoders", 33.0)
    assert s._off("encoders", 91.5) and s._off("left_encoder", 5.5) and not s._off("right_encoder", 5.5)
    s.faults = {"imu_off": True}
    assert s._off("imu", 0.0) and not s._off("encoders", 0.0)


def test_interfaces_doc_lists_every_odometry_status_key():
    from pathlib import Path
    from types import SimpleNamespace

    from amr_rl.runtime.robot import RobotRuntime, RuntimeConfig

    rt = RobotRuntime.__new__(RobotRuntime)
    rt.cfg, rt.spec, rt.odo, rt.last_odometry = RuntimeConfig(), SPEC, WheelInertialOdometry(SPEC), None
    rt.slam = SimpleNamespace(fusion_log={}, blocked_kind=None)
    rt.imu_calibration_status, rt.odo_fallbacks, rt.zupt_holds, rt.zupt_hold_failures = "loaded", 0, 0, 0
    rt._zupt_hold_start, rt.now, rt._blocked_plan, rt.blocked_events = None, 0.0, None, []
    rt._sample_t, rt.sensor_state = {"imu": None, "encoders": None}, {"imu": "waiting", "encoders": "waiting"}
    doc = (Path(__file__).resolve().parents[1] / "docs" / "INTERFACES.md").read_text()
    missing = [k for k in rt.odometry_status() if f'"{k}"' not in doc]
    assert not missing, f"document these odometry status keys in docs/INTERFACES.md: {missing}"



def test_sensor_liveness_is_tracked_and_state_changes_logged():
    from types import SimpleNamespace

    from amr_rl.runtime.robot import RobotRuntime

    notes = []
    rt = RobotRuntime.__new__(RobotRuntime)
    rt.odo = WheelInertialOdometry(SPEC)
    rt.supervisor = SimpleNamespace(note=lambda kind, **d: notes.append((kind, d)))
    rt._proprio = ProprioBatch([], [])
    rt._sample_t, rt.sensor_state = {"imu": None, "encoders": None}, {"imu": "waiting", "encoders": "waiting"}
    counts = [0.0, 0.0]
    rt.on_proprio(_samples(0.0, 10, 0, 0, 0, 0, counts))
    rt._check_sensors(0.1)
    assert rt.sensor_state == {"imu": "ok", "encoders": "ok"}
    b = _samples(0.1, 10, 0, 0, 0, 0, counts)
    rt.on_proprio(ProprioBatch(b.imu, []))  # the encoders stop delivering
    rt._check_sensors(0.5)
    assert rt.sensor_state == {"imu": "ok", "encoders": "lost"}
    rt.on_proprio(_samples(0.5, 10, 0, 0, 0, 0, counts))
    rt._check_sensors(0.6)
    assert [(n[1]["source"], n[1]["state"]) for n in notes] == [("imu", "ok"), ("encoders", "ok"),
                                                                ("encoders", "lost"), ("encoders", "ok")]


def test_imu_calibration_states(tmp_path):
    from amr_rl.runtime.robot import RobotRuntime, RuntimeConfig

    rt = RobotRuntime.__new__(RobotRuntime)
    rt.spec = SPEC
    rt.cfg = RuntimeConfig(imu_calibration=str(tmp_path / "missing.yaml"))
    assert rt._load_imu_calibration() is None and rt.imu_calibration_status.startswith("missing")
    cal = tmp_path / "imu.yaml"
    cal.write_text(f"unit_serial: {SPEC.imu.get('unit_serial', 1)}\ngyro_scale: 1.0025\ngyro_scale_sigma: 0.0002\n")
    rt.cfg = RuntimeConfig(imu_calibration=str(cal))
    assert rt._load_imu_calibration()["gyro_scale"] == 1.0025 and rt.imu_calibration_status.startswith("loaded")
    rt.cfg = RuntimeConfig(imu_calibration=str(cal), imu_unit_serial=4)  # another unit installed
    assert rt._load_imu_calibration() is None and "mismatch" in rt.imu_calibration_status
    rt.cfg = RuntimeConfig(imu_calibration=None)
    assert rt._load_imu_calibration() is None


def test_unknown_world_overrides_are_rejected():
    import pytest

    from amr_rl.sim.harness import apply_world_overrides

    wc = {"name": "arena", "room": {"size": [4, 4]}}
    assert apply_world_overrides(wc, {"floor_friction": 0.06})["floor_friction"] == 0.06
    assert apply_world_overrides(wc, {"room": {"size": [5, 5]}})["room"]["size"] == [5, 5]
    with pytest.raises(ValueError, match="wheel_friction"):
        apply_world_overrides(wc, {"wheel_friction": 0.4})


def test_live_replay_uses_the_recorded_settings_and_unit_calibration(tmp_path):
    import json as _json

    lr = _scoring_module_dev("live_replay")
    run = tmp_path / "run"
    (run / "provenance").mkdir(parents=True)
    cal = tmp_path / "u2.yaml"
    cal.write_text("unit_serial: 2\ngyro_scale: 1.003\ngyro_scale_sigma: 0.0002\n")
    (run / "provenance" / "identity.json").write_text(_json.dumps({"extra": {"args": {
        "odometry": "imu_encoders", "vslam": ['loop_odo_guard="heading"'], "odo": ["zaru=true"],
        "runtime": [f'imu_calibration="{cal}"', "imu_unit_serial=2"], "world_set": ['sensor_faults={"imu_unit":2}']}}}))
    rec = lr.recorded_settings(run)
    assert rec["odometry"] == "imu_encoders" and rec["vslam"] == {"loop_odo_guard": "heading"}
    assert rec["odo"] == {"zaru": True} and rec["imu_unit"] == 2
    assert lr.load_calibration(rec["imu_calibration"], 2, SPEC)["gyro_scale"] == 1.003
    assert lr.load_calibration(rec["imu_calibration"], 1, SPEC) is None  # another unit: datasheet prior
    assert lr.recorded_settings(tmp_path / "none")["odometry"] == "command"


def _scoring_module_dev(name):
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "dev" / f"{name}.py"
    sys.path.insert(0, str(path.parents[1] / "eval"))
    spec = importlib.util.spec_from_file_location(f"_dev_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_blocked_keeps_the_pose_and_restarts_the_window():
    """Owner decision 2026-10-12: a blocked drive is not a loss of localisation. The
    VSLAM records the event (stalled against something, or spinning on the floor), keeps
    tracking, and restarts its consistency window for the back-off that follows."""
    still = np.zeros(3)
    for travel, slip, kind in ((0.0, False, "stalled"), (0.02, True, "spinning")):
        slam = _motion_slam([(k * 0.1, still.copy(), travel, 0.0, slip, 0.02) for k in range(13)])
        slam._proprio_win, slam.blocked, slam.blocked_events = [], False, 0
        assert not slam._odometry_consistent(1.2) and slam.inconsistency == "blocked"
        slam._note_blocked(1.2)
        assert slam.blocked and slam.blocked_kind == kind and slam.blocked_events == 1
        assert slam._motion_log == [] and slam.inconsistency is None
    # without vision the wheels alone cannot tell the two apart
    slam = _motion_slam([])
    slam._proprio_win, slam.blocked, slam.blocked_events = [(k * 0.1, 0.02, 0.0) for k in range(13)], False, 0
    assert slam.blocked_by_odometry(1.2)
    slam._note_blocked(1.2, kind="unknown")
    assert slam.blocked and slam.blocked_kind == "unknown" and slam._proprio_win == []
    assert VSLAMConfig().odo_blocked_mode == "lost"  # default since validation 6; keep_pose stays switchable
    # a slip flag older than the 1.2 s verdict window does not relabel a stall as spinning
    log = [(k * 0.1, still.copy(), 0.0, 0.0, k < 5, 0.02) for k in range(30)]  # slip only at t < 0.5 s
    slam = _motion_slam(log)
    slam._proprio_win, slam.blocked, slam.blocked_events = [], False, 0
    slam._note_blocked(2.9)
    assert slam.blocked_kind == "stalled"


def _blocked_runtime(pose=(1.0, 2.0, 0.0), kind="stalled"):
    from types import SimpleNamespace

    from amr_rl.runtime.robot import RobotRuntime, RuntimeConfig

    rt = RobotRuntime.__new__(RobotRuntime)
    rt.cfg = RuntimeConfig()
    rt.blocked_events, rt._blocked_plan, rt.triggers, rt.activity, rt.loc_status = [], None, [], None, "tracking"
    rt.pose = None if pose is None else np.array(pose, float)
    rt.now, rt.last_command = 0.0, (0.2, 0.0)
    rt.slam = SimpleNamespace(blocked_kind=kind)
    rt.planner = SimpleNamespace(cfg=SimpleNamespace(footprint_radius=0.16),
                                 obstacle_clearance_xy=lambda grid, xy: np.array([1.0] * len(xy)))
    rt.grid = SimpleNamespace(raised=[], raise_to=lambda xy, level: rt.grid.raised.append((np.asarray(xy), level)) or 5)
    rt.supervisor = SimpleNamespace(autonomy_enabled=True, recovery=None, stopped=[],
                                    stop=lambda reason: rt.supervisor.stopped.append(reason))
    rt.nav = SimpleNamespace(cancel=lambda reason: None)
    return rt


def test_blocked_runtime_backs_off_with_the_pose_kept_then_resumes():
    rt = _blocked_runtime()
    rt._on_blocked(10.0)
    ev = rt.blocked_events[-1]
    assert rt._blocked_plan is not None and ev["kind"] == "stalled" and ev["cells_marked"] == 5
    xy, level = rt.grid.raised[0]
    assert level == 1.5 and np.allclose(xy[:, 0], 1.20) and abs(xy[:, 1].mean() - 2.0) < 1e-9  # just ahead of the front
    assert rt._blocked_step(10.1) == (-0.06, 0.0)  # straight back, slowly
    rt.pose = np.array([0.95, 2.0, 0.0])  # 5 cm back: not yet
    assert rt._blocked_step(11.0) == (-0.06, 0.0)
    rt.pose = np.array([0.89, 2.0, 0.0])  # 11 cm back: done
    assert rt._blocked_step(12.0) == (0.0, 0.0)
    assert rt._blocked_plan is None and rt.triggers == ["blocked"] and ev["backoff_complete"] is True
    assert rt.supervisor.stopped == []
    # spinning wheels (a floor, not an object): back off, mark nothing the first time
    rt = _blocked_runtime(kind="spinning")
    rt._on_blocked(20.0)
    assert rt.grid.raised == [] and "cells_marked" not in rt.blocked_events[-1]
    # the back-off itself blocked: time limit, incomplete
    assert rt._blocked_step(24.1) == (0.0, 0.0) and rt.blocked_events[-1]["backoff_complete"] is False
    # stopped again within 0.3 m in the same minute: the spot is marked, whatever the kind
    rt.pose = np.array([1.1, 2.05, 0.0])
    rt._on_blocked(30.0)
    assert rt.blocked_events[-1]["cells_marked"] == 5 and len(rt.grid.raised) == 1
    rt._blocked_plan = None
    rt.pose = np.array([2.5, 2.0, 0.0])  # far away: a new place, not marked
    rt._on_blocked(40.0)
    assert "cells_marked" not in rt.blocked_events[-1]


def test_repeated_blocked_events_stop_autonomy_for_the_operator():
    rt = _blocked_runtime(kind="spinning")
    for t in (10.0, 20.0, 30.0):
        rt._on_blocked(t)
        rt._blocked_plan = None
    assert rt.supervisor.stopped == []
    rt._on_blocked(40.0)  # the 4th within a minute
    assert rt.supervisor.stopped == ["blocked_repeatedly"] and rt._blocked_plan is None
    assert rt.blocked_events[-1]["stopped"] is True
    rt = _blocked_runtime(kind="spinning")
    for t in (10.0, 20.0, 30.0, 100.0):  # the 4th is a minute later: allowed
        rt._on_blocked(t)
        rt._blocked_plan = None
    assert rt.supervisor.stopped == []


def test_first_temperature_stop_at_half_a_degree_until_the_tempco_is_known():
    """Owner decision 2026-10-12: before the tempco has been measured, a 0.5 degC change
    stops the robot even though the offset's own uncertainty is still small; after it,
    the 1 degC rule and the uncertainty gate apply as before."""
    from types import SimpleNamespace

    from amr_rl.runtime.robot import RobotRuntime, RuntimeConfig

    def runtime(temp, known, sigma_dps=0.008):
        rt = RobotRuntime.__new__(RobotRuntime)
        rt.cfg = RuntimeConfig(imu_zupt_temp_first_c=0.5)  # the switch on (default 0 since validation 6)
        rt.odo = SimpleNamespace(t=0.0, zupt_frames=12, last_bias_update_t=0.0, bias_var=math.radians(sigma_dps) ** 2,
                                 temp=temp, temp_at_update=25.0, tempco_known=known)
        rt._boot_t0, rt._zupt_hold_start, rt._zupt_hold_until, rt._zupt_frames_at_hold = -10.0, None, -math.inf, None
        rt.zupt_hold_log, rt.zupt_holds, rt.zupt_hold_failures = [], 0, 0
        return rt

    rt = runtime(25.6, known=False)
    assert rt._imu_booting(40.0) and rt.zupt_hold_log[-1]["trigger"] == "temperature_first"
    assert rt.zupt_hold_log[-1]["tempco_known"] is False
    assert not runtime(25.6, known=True)._imu_booting(40.0)  # 0.6 degC, tempco known, offset certain: no stop
    assert not runtime(25.3, known=False)._imu_booting(40.0)  # below 0.5 degC
    assert not runtime(25.6, known=False)._imu_booting(15.0)  # within imu_zupt_temp_min_s of the last update
    rt = runtime(26.2, known=True, sigma_dps=0.03)
    assert rt._imu_booting(40.0) and rt.zupt_hold_log[-1]["trigger"] == "temperature"
    rt = runtime(25.7, known=False, sigma_dps=0.03)  # uncertain too, but only the first rule reaches 0.7 degC
    assert rt._imu_booting(40.0) and rt.zupt_hold_log[-1]["trigger"] == "temperature_first"
    rt = runtime(26.2, known=False, sigma_dps=0.03)  # the 1 degC rule would have fired by itself
    assert rt._imu_booting(40.0) and rt.zupt_hold_log[-1]["trigger"] == "temperature"
    rt = runtime(25.0, known=True, sigma_dps=0.03)
    assert rt._imu_booting(130.0) and rt.zupt_hold_log[-1]["trigger"] == "interval"
    rt = runtime(25.6, known=False)
    rt.cfg.imu_zupt_temp_first_c = 0.0  # the switch off: the 2026-10-11 behaviour
    assert not rt._imu_booting(40.0)


def test_tempco_is_known_only_after_an_offset_measurement_at_another_temperature():
    odo = WheelInertialOdometry(SPEC)
    if odo._tempco_prior_sigma <= 0:
        assert odo.tempco_known
        return
    assert not odo.tempco_known
    odo.temp0 = odo.temp = 25.0
    odo._offset_update(odo.bias, (0.002 * DPS) ** 2, temp=25.0)  # same temperature: learns nothing about the tempco
    assert not odo.tempco_known
    odo._offset_update(odo.bias, (0.002 * DPS) ** 2, temp=25.5)  # one stop 0.5 degC from boot: measured once
    assert odo.tempco_known and odo._tempco_measured
    odo = WheelInertialOdometry(SPEC)
    odo.temp0 = odo.temp = 25.0
    odo._offset_update(odo.bias, (0.002 * DPS) ** 2, temp=25.0)  # the boot measurement pins the offset
    odo._offset_update(odo.bias, (0.002 * DPS) ** 2, temp=25.2)  # too close to boot to count as measured
    assert not odo.tempco_known
    for _ in range(3):
        odo._offset_update(odo.bias, (0.002 * DPS) ** 2, temp=25.3)  # but the covariance test can still pass
    assert odo.tempco_known


def _predict_slam(mode="keep_pose"):
    from amr_rl.perception.vslam import PlanarVSLAM

    slam = PlanarVSLAM.__new__(PlanarVSLAM)
    slam.cfg = VSLAMConfig(odo_blocked_mode=mode)
    slam._motion_log, slam._proprio_win = [], [(k * 0.1, 0.02, 0.0) for k in range(13)]  # driven, wheels not moving
    slam.inconsistency, slam.blocked, slam.blocked_kind, slam.blocked_events = None, False, None, 0
    slam.pose, slam._cov = np.array([1.0, 2.0, 0.3]), np.diag([1e-4, 1e-4, 1e-5])
    slam.position_sigma, slam.heading_sigma, slam.velocity = 0.01, 0.003, np.zeros(3)
    slam.predicted_time, slam.failures, slam._turn, slam._turn_count = 0.3, 1, None, 0
    slam.fusion_log = {"predicted_odo": 0}
    slam._vision_pose = slam._bias_win = slam._scale_anchor = slam._radius_acc = None
    slam._on_frozen = lambda t: None  # the landmark bookkeeping of a real loss is not under test
    return slam


def test_blocked_without_vision_keeps_dead_reckoning_and_reports_it():
    from amr_rl.odometry.fusion import OdometryStep
    from amr_rl.perception.vslam import LOST, PREDICTED

    spinning = OdometryStep(dt=0.1, dx=0.02, dy=0.0, dth=0.001, cov=np.diag([1e-5, 1e-5, 1e-6]), v=0.2, w=0.01,
                            distance=0.02, slip=True)
    slam = _predict_slam()
    r = slam._predict_odometry(spinning, 1.2)
    assert r.status == PREDICTED and r.info.get("blocked") is True and slam.blocked_kind == "unknown"
    assert np.allclose(slam.pose[:2], [1.0, 2.0])  # slip-flagged travel does not walk the kept pose
    assert abs(slam.pose[2] - 0.301) < 1e-9 and slam._proprio_win == []
    # a slip flag without a blocked verdict (the wheels do travel) dead-reckons as "lost" mode does:
    # the rule is gated on the blocked state since validation 6
    moving = _predict_slam()
    moving._proprio_win = [(k * 0.1, 0.02, 0.02) for k in range(13)]
    r = moving._predict_odometry(spinning, 1.2)
    assert r.status == PREDICTED and not r.info.get("blocked") and not moving.blocked
    assert not np.allclose(moving.pose[:2], [1.0, 2.0]) and abs(moving.pose[0] - (1.0 + 0.02 * np.cos(0.3))) < 1e-6
    # the 2026-10-11 behaviour is still available for the paired baseline
    slam = _predict_slam("lost")
    slam._reset_vision_refs = lambda **k: None
    r = slam._predict_odometry(spinning, 1.2)
    assert r.status == LOST and r.reason == "visual_motion_inconsistent_with_commands"
    with pytest.raises(ValueError):
        from amr_rl.perception.camera_model import CameraModel
        from amr_rl.perception.vslam import PlanarVSLAM
        PlanarVSLAM(CameraModel.from_spec(SPEC), VSLAMConfig(odo_blocked_mode="sometimes"))


def test_blocked_backoff_follows_the_driven_side_and_needs_clearance():
    # a blocked REVERSE drive: drive forward, mark the cells behind
    rt = _blocked_runtime()
    rt.last_command = (-0.1, 0.0)
    rt._on_blocked(10.0)
    assert rt.blocked_events[-1]["drive"] == "reverse" and rt._blocked_plan["dir"] == -1
    assert rt._blocked_step(10.1) == (0.06, 0.0)
    xy, _ = rt.grid.raised[0]
    assert np.allclose(xy[:, 0], 0.80)  # behind the pose at x = 1.0
    # blocked while turning in place: nothing to back away from; choose again
    rt = _blocked_runtime()
    rt.last_command = (0.0, 0.4)
    rt._on_blocked(10.0)
    assert rt._blocked_plan is None and rt.blocked_events[-1]["backoff_skipped"] == "turning"
    assert rt.triggers == ["blocked"] and rt.grid.raised == []
    # no clearance behind: no blind drive
    rt = _blocked_runtime()
    rt.planner.obstacle_clearance_xy = lambda grid, xy: np.array([0.05])
    rt._on_blocked(10.0)
    assert rt._blocked_plan is None and rt.blocked_events[-1]["backoff_skipped"].startswith("no clearance")


def test_cancelling_the_activity_closes_an_open_backoff():
    rt = _blocked_runtime()
    rt._on_blocked(10.0)
    rt.now = 11.0
    rt._cancel_activity("heartbeat_lost")  # a revocation while backing off
    ev = rt.blocked_events[-1]
    assert rt._blocked_plan is None and ev["interrupted"] == "heartbeat_lost" and ev["backoff_complete"] is False
    assert rt.triggers == []  # no stale trigger for the next session
    rt = _blocked_runtime()
    rt._on_blocked(10.0)
    rt._cancel_activity("blocked")  # the cancel issued by the blocked handler itself keeps the plan
    assert rt._blocked_plan is not None
