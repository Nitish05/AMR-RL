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
    for _ in range(5):
        odo.add_scale_sample(1.0027, 0.0005)
    assert 1.0025 < odo.gyro_scale < 1.0027 and math.sqrt(odo.scale_var) < 0.0002
    assert not odo.add_scale_sample(1.2, 0.001)  # implausible: ignored


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
