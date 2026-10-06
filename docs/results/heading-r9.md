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

## Validation 2: R9b (6e89e60), stopped at the owner's request

R9b had a forced 0.8 s pause every 30 s. It was stopped after 47 L1 runs, because the owner judged the pause not smooth enough. On the 23 pairs that ran, **every L1 criterion passed**:

| Metric (23 pairs) | Camera only | R9b |
|---|---|---|
| Max heading error, median (worst) | 13.15° (53.8°) | 1.17° (2.1°) |
| Paired median Δ (95 % CI) | — | −11.6° (−18.5, −9.5) |
| Drift events per 100 turns | 6.5 | 0 |
| ATE median, paired Δ (CI), worst pair | — | −4.1 cm (−7.3, −1.8), worst +2.9 cm |
| Lost fraction, mean | 16.9 % | 0.3 % |

Its slip, fault, unit and navigation checks were never run.

## How the stop policy was chosen (development only, spent seeds 0–4)

**The constraint.** An exploring robot never stands still, so the gyro offset random-walks; on development data the gyro alone drifted 6° in 390 s.

**Rejected options:**
- **Updating the offset while driving straight** fails in this simulator: the chassis skids 0.1–0.3 dps with equal wheel travel, and one accepted window cost 14°.
- **Single-frame updates at stops** took the chassis's settling creep (about 0.06 dps) as the offset.

**Comparison.** Mean over six worlds of the per-run maximum heading error and of the ATE:

| Stop policy | Max heading error | ATE |
|---|---|---|
| Every 30 s | 1.33° | 2.20 cm |
| Every 60 s | 1.51° | 3.16 cm |
| Every 120 s with the offset learned from vision on established landmarks | 1.76° | 3.05 cm |
| Every 120 s | 2.51° | 3.37 cm |
| Camera only | 16.3° | 8.4 cm |

**Owner decision:** calibrate whenever the robot stops; if it has not stopped for 2 minutes, stop and recalibrate if necessary. This became R9c (pre-registration addendum B).

## Validation 3: R9c (890646c), fresh starts (seeds 5–9, heading offset 36°)

The IMU arm was run fresh. The camera-only arm is reused from validation 2; it is byte-identical. Evidence is in `work/evidence/r9c-val-20261005/`, with the reports `report-*.json`.

**L1, 25 pairs:**

| Metric | Camera only | R9c | Paired median Δ (95 % CI) |
|---|---|---|---|
| Max heading error per run, median (worst) | 13.15° (53.8°) | **1.70° (5.0°)** | −10.8° (−19.1, −7.5) |
| Drift events (≥ 8° in 10 s) per 100 in-place turns | 6.5 (87 / 1329) | **0 (0 / 1591)** | — |
| ATE median (worst) | 5.6 cm (54.3 cm) | 2.7 cm (7.9 cm) | −2.7 cm (−7.3, −0.8) |
| Keyframe map RMSE median | 6.5 cm | 2.6 cm | −3.3 cm (−7.0, −0.9) |
| Lost fraction, mean | 16.2 % | **0.4 %** | — |

- P1, P2, P3 and P5 pass.
- **P4 fails:** two pairs are more than 3 cm worse.
  - heldout_c s8: 1.6 → 7.3 cm.
  - home_a s5: 0.5 → 3.9 cm.
  - In both, the IMU arm's heading drifted slowly over long non-stop driving (up to 4.5°), and its position error grew with it (to 14 cm at the end of heldout_c s8).
  - In both, the camera-only arm was lost for long stretches: 998 frames between 106 s and 210 s, and 455 frames between 38 s and 90 s. Its ATE covers only the frames it tracked.
  - This explanation is a post-hoc observation, not a change of the criterion.

**Wheel slip.** Pass: R9c holds P1 and P5 against the paired baseline, and the slip detector fires.

| Condition | Camera only, max heading median (worst) | R9c, max heading median (worst) | Lost fraction (camera / R9c) |
|---|---|---|---|
| Whole floor `floor_friction=0.45`, 6 pairs | 11.9° (38.2°) | 2.1° (3.5°) | 21.7 % / 0 % |
| Two slip patches (friction 0.05), 3 pairs | 4.5° (37.0°) | 1.9° (2.6°) | 35.3 % / 0.7 % |

On the patch runs the slip flag is on for 20–64 % of frames, mostly from the yaw check; translation then relies on vision.

**Navigation (L3), 4 worlds.** Pass.
- Supported goals reached: 15/15 against 9/14.
- Reach goals reached: 5 against 2.
- Contact episodes: 0 in both arms.
- Mapping ATE: 0.5–3.7 cm against 0.7–4.4 cm.

**Sensor faults (F), arena s5.** Camera only on this start: 10.6° and 5.4 cm.
- **IMU off:** 6.8° and 11.6 cm, no losses. This passes (no crash, P5 holds, heading not worse). The fallback rotates on encoders, so the map error is higher.
- **Encoders off:** 64.1° and 58.7 cm. **This failed.** Without encoder counts, stillness was never detected, so the offset was never measured.
  - Fixed in d237559: a zero command plus a quiet gyro now counts as stillness; behaviour with encoders is unchanged.
  - Re-check from a pinned worktree (`work/evidence/r9d-faults-20261005/`): encoders off gives **2.2° and 0.9 cm**. IMU off is unchanged, as expected (6.8°, 11.6 cm). F now passes.

**Other IMU units (U), arena s5,** camera only 10.6°:

| Unit | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| Max heading error, calibrated | 2.5° | 2.3° | 2.3° | 3.0° |
| Max heading error, datasheet prior only | 2.6° | 2.1° | **10.1°** | 1.6° |
| Map ATE, calibrated | 2.4 cm | 1.6 cm | 7.3 cm | 4.9 cm |

The unit calibrations agree with simulator truth to 0.01–0.03 %. Unit 4 has a −1 % sensitivity error, which is why it needs the calibration.

**Verdict against the pre-registered rule.** Adoption needs L1, L1-slip, L3 and F to pass. L1 (P4) failed, and F failed before the dropout fix (it passes since d237559). R9c is therefore not adopted by the rule. The owner's decision is recorded below.


## Fixing the slow drift (owner: fix it before adoption)

**Diagnosis** (evaluation-only reconstruction of the simulated offset, development seeds): between recalibrations the gyro offset was off by 0.009–0.018 dps. Three causes:
1. **Vision offset learning was over-confident.** Correlated per-frame corrections were used as independent measurements, so the uncertainty shrank and the "if necessary" stop never fired. **Fix:** one measurement per 15–90 s window between frames where vision is an absolute reference, with its own uncertainty.
2. **The drift model was unrealistic.** A cited research step replaced the assumed random walk (0.002 dps/√s) with these terms:
   - Allan analogues (BMI160 and BMI055: 1.5e-4 dps/√s);
   - a per-unit offset plus 0.1 dps per power-on;
   - **thermal warm-up**: datasheet tempco ±0.010 dps/°C times 2–12 °C with τ 150–900 s (assumed ranges);
   - the chip's temperature sensor.

   The warm-up is the dominant slow term. **Estimator:** offset(T) = b0 + k(T − T0), a two-state Kalman filter with a tempco prior of 2× typical (one unit was at 2.5× typical).
3. **Stops measured only about 0.1 s of samples**, which leaves about 0.01 dps of noise (the true yaw rate during stops was exactly 0, so this is not creep). **Fix:** a recalibration stop lasts until 1 s of still samples is collected; the boot stand-still is 2 s.

**Owner-approved addition:** also recalibrate when the chip temperature has changed by ≥ 1 °C since the last calibration (only if the offset is uncertain).

**Development** (6 worlds, mean of the per-run maximum heading error and mean ATE):

| Policy | Max heading error | ATE | Recalibration stops in 7 min |
|---|---|---|---|
| With the temperature trigger | 1.28° | 1.94 cm | 1–3 |
| 2-minute rule only | 2.11° | 4.26 cm | — |

Pre-registration addendum C was written before validation 4.

## Validation 4: R9e (f0748fc), plus the IMU-off fix 811adbc, fresh starts

The IMU arm was run fresh. The camera-only arm is byte-identical and reused.
- **Display sleep:** 12 jobs failed when the Mac's display slept at night (the renderer found no screen). They were re-run with the display kept awake; the failed logs are kept as `*.failed-no-display.log`.
- **Evidence:** `work/evidence/r9e-val-20261006/` and `work/evidence/r9f-recheck-20261006/`.

**L1, 25 pairs, all criteria pass:**

| Metric | Camera only | R9e | Paired median Δ (95 % CI) |
|---|---|---|---|
| Max heading error per run, median (worst) | 13.15° (53.8°) | **1.25° (11.6°)** | -11.3° (-19.2, -9.5) |
| Drift events per 100 in-place turns | 6.55 (87 / 1329) | **0.06 (1 / 1594)** | — |
| ATE median (worst) | 5.6 cm (54.3) | **2.0 cm (23.9)** | -4.2 cm (-6.7, -2.5); worst pair +2.5 cm |
| Keyframe map RMSE median | 6.5 cm | 2.0 cm | -4.6 cm |
| Lost fraction, median (max) | 0.082 (0.55) | **0.000 (0.03)** | — |

**The one outlier** (arena_textured s7, 11.6° and 23.9 cm; its camera-only pair had 38.3° and 54.3 cm) is a **false loop closure** at t = 103.4 s, off by 0.21 m and 12°. Loop closure is a camera-side component from rounds 5–6. Next improvement: reject loop corrections whose heading change exceeds the gyro's own uncertainty (about 0.1° there).

**Other checks:**
- **Slip:** pass.
  - Slippery floor: median 1.2° (worst 1.8°) against 11.9° (38.2°), no losses.
  - Slip patches: 0.9° (1.1°) against 4.5° (37.0°), no losses.
  - The slip flag is on for 1260–3045 frames in the patch runs.
- **Navigation (L3):** pass. Supported goals reached 12/14 against 9/14, reach goals 6 against 2, 0 contacts. Mapping ATE 0.7–4.6 against 0.7–4.4 cm.
- **Sensor faults (F):** pass after one fix.
  - Encoders off: 1.1°.
  - IMU off failed at first (15.3° against 10.6° camera-only): encoder-only rotation was still trusted over vision. Fixed in 811adbc; re-check **4.0°**, no losses.
- **Other units (U, reported, not required):**
  - Calibrated units 2–5: 0.7–2.1°.
  - Uncalibrated (datasheet prior): 0.8–2.9°, except unit 4 (−1 % sensitivity) at 19.2°. **A unit must be calibrated** with `scripts/calibrate_imu.py`.

## Decision

Every required pre-registered check passes: L1, L1-slip, L3 and F. **`RuntimeConfig.odometry = "imu_encoders"` is the default from 2026-10-06.** `"command"` reproduces the camera-only robot exactly.

**Not yet evaluated in closed loop** (switches off):
- WS1 wheel model;
- WS2 command-model odometry (used as the fallback when no samples arrive);
- WS4 depth floor mask;
- WS5 view-aware turns;
- the loop-closure heading guard above.
