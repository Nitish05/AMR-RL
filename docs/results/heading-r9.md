# Round 9: IMU + wheel encoders, fused with the camera (2026-10-05)

**Decision and plan.** The owner decided on 2026-10-05 to add an IMU and wheel encoders that can be bought off the shelf, to keep wheel slip in mind, and to implement the round-9 plan ([plans/heading-robustness-plan.md](../plans/heading-robustness-plan.md)). The validation criteria were pre-registered in [heading-r9-prereg.md](heading-r9-prereg.md) before any validation run. The sensor parts are in [ROBOT.md](../ROBOT.md#proprioceptive-sensors-round-9-imu-and-wheel-encoders).

## What was built

| Item | Where | Default |
|---|---|---|
| IMU and encoder models from datasheets: LSM6DSOX on Adafruit 4438; Pololu 4754 encoders | `sim/sensors.py`, `assets/robot/amr_spec.yaml` | always simulated; the runtime uses them only with `odometry="imu_encoders"` |
| Raw-sample interface `RobotRuntime.on_proprio` | `runtime/robot.py`, `sim/harness.py` | — |
| Wheel-inertial odometry (details below) | `odometry/fusion.py` | — |
| Fusion in the VSLAM (details below) | `perception/vslam.py` (`odo_*`) | active when odometry steps are passed in |
| One-time gyro calibration by the robot (camera-closed turns) | `scripts/calibrate_imu.py`, `configs/calibration/imu.yaml` | used with the IMU |
| WS0: frame and sensor recording, live replay, paired report with BCa intervals | `--record-frames`, `scripts/dev/live_replay.py`, `scripts/eval/turn_live_report.py --arms` | evaluation |
| WS1: wheel response model (rate limits plus gains fitted offline) and a signed motion check | `odometry/wheel_model.py`, `VSLAMConfig.wheel_model` | off |
| WS2: command-model odometry as a soft prior. Also the fallback when no IMU or encoder samples arrive | `CommandModelOdometry`, `odometry="command_model"` | off (fallback on with the IMU) |
| WS3: depth-free planar epipolar turn estimator (used by the calibration) | `perception/turn_epipolar.py` | not in tracking |
| WS4: depth-model floor mask on floor lifting | `near_depth.floor_mask`, `VSLAMConfig.depth_floor_mask`, `RuntimeConfig.depth_device` | off |
| WS5: view-aware turning | `navigation/view_check.py`, `RuntimeConfig.view_aware_turns` | off |
| Evaluation faults: slip patches (low-friction plates), slippery floor, IMU or encoder dropout, other IMU units | `--world-set slip_patches=… floor_friction=… sensor_faults=…` | — |

**Wheel-inertial odometry** (`odometry/fusion.py`):
- rotation from the gyro z axis;
- the zero-rate offset is measured whenever the robot stands still, including a 1 s stand-still at boot;
- the gyro scale is calibrated once and refined from vision on full turns;
- distance from the encoders times a wheel-radius scale learned from vision;
- slip flags.

**Fusion in the VSLAM:**
- an information-weighted pose update;
- robust per-axis weights: on heading, vision is down-weighted when it disagrees; on translation, the odometry is;
- vision's heading information comes only from established landmarks;
- odometry factors in the local bundle adjustment;
- dead reckoning on odometry when vision has no fix;
- a signed motion-consistency check.

The camera-only default is byte-identical to before: 400 of 400 frames match (arena seed 1, 40 s).

## Development findings

Development used only arena seed 0, arena_textured seed 2 and synthetic tests. Each item is a fault found and fixed, in order.

1. **Gyro offset never measured.** The robot starts its survey turn at once, so it never stood still. The unit's offset of about 1 dps (datasheet ±1 dps typ) went uncorrected, and the gyro alone drifted 1.1°/s. **Fix:** a 1 s stand-still at boot (standard IMU practice), then zero-velocity updates whenever the robot is still.
2. **Gyro scale.** The ±1 % sensitivity tolerance costs up to 3.6° per full turn, and the datasheet prior made the fusion trust vision. **Fix:** a one-time calibration in which the robot turns 5 times each way and closes each set of turns with the planar epipolar residual between the start and end images.
   - For unit 1 the calibration gives 1.00249 ± 0.0002. The same turns scored against simulator truth (evaluation only) give 1.00248.
   - Two bugs were found on the way:
     - **Tilt projection.** Projecting the gyro onto the vertical with an accelerometer tilt estimate added +0.13 %, because accelerometer offsets bias the tilt by 1–2°. It was removed; on a flat floor the body z rate is the yaw rate.
     - **Online scale learning during calibration.** The VSLAM's online scale learning changed the scale during the calibration turns. It is now disabled there.
3. **Vision pulled the fused heading.** While exploring new ground, vision's heading is measured against landmarks the robot has only just made. That is visual odometry with the lever-arm bias, about +0.1°/s while turning near the start. **Fix:** vision's heading variance scales with 1 / (share of inliers older than 8 s).
4. **Local bundle adjustment re-fitted keyframes from vision alone** and set the pose from that, which brought the turn bias back. **Fix:** odometry relative-pose factors between consecutive keyframes in the local bundle adjustment (visual-inertial BA). On the two 60 s development runs the maximum heading error went from 2.2° to 0.44° (arena) and from 2.1° to 0.41° (arena_textured). Camera-only gives 4.1° and 3.2°.
5. **Encoders over-read distance by about 10 %.** On straight frames of live runs the true distance is 0.89–0.91 of the encoder distance in every world: the simulated wheels roll short of the 0.05 m spec radius. **Fix:** an online wheel-radius scale from vision distance over driving stretches of 0.3 m or more; the encoder distance uncertainty stays wide until about 1 m has been driven.
6. **Slip in the simulator.** Genesis takes the higher of two surfaces' friction values. Lowering the wheel friction therefore did nothing, and the first slip-patch runs had no slip. **Fix:** slip patches are 1 mm plates with their own low friction (the wheel's friction is lowered too), and `floor_friction` makes the whole floor slippery. With low friction the in-place turns actually scrub less.

## Validation 1: frozen candidate ce39b72 (pre-registered L1)

The run stopped at the first failed required check, per the pre-registered stop rule. 13 of the 25 pairs had run by then; all 13 are reported.

- **Evidence:** `work/evidence/r9-L1-20261005/`, report `report.json`.
- **Baseline check:** the camera-only arm reproduces the round-8 baseline exactly (arena seed 0: 7.755°).

| Metric (13 pairs) | Camera only | IMU + encoders | Paired median Δ (95 % CI) |
|---|---|---|---|
| Max heading error per run, median (max) | 7.76° (23.7°) | **2.18° (5.2°)** | −6.9° (−8.9, −1.4) |
| Drift events (≥ 8° in 10 s) per 100 in-place turns | 3.78 (29 / 767) | **0 (0 / 800)** | — |
| Lost fraction, mean | 0.084 | **0.002** | — |
| Map ATE, median (max) | 4.6 cm (33.3 cm) | 2.3 cm (11.8 cm) | −0.9 cm (−6.5, +0.3) |
| Keyframe map RMSE, median | 4.6 cm | 2.3 cm | −0.8 cm |

Against the pre-registered criteria:
- **P1 passes:** median 2.2° ≤ 5° and ≤ 50 % of baseline.
- **P2 passes:** no run is above 15°.
- **P3 passes:** 0 drift events.
- **P5 passes:** no increase in losses.
- **P4 fails:** two pairs are more than 3 cm worse. home_a s0 is 2.0 → 6.0 cm and heldout_c s1 is 6.8 → 11.8 cm.

**Cause.** Position error in the IMU arm grows from the start (home_a s0: 2.4 cm at 30 s against 0.7 cm camera-only). On straight frames the true distance is 0.89–0.91 of the encoder distance, and the fusion trusted the encoders at ±3.6 %; it down-weighted them on only 14 % of driving frames. This led to candidate R9b (addendum A of the pre-registration): an online wheel-radius scale.

PENDING
