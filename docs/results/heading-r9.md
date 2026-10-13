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
- the zero-rate offset is measured whenever the robot stands still, including a stand-still at boot (1 s at first, 2 s since R9e);
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

The camera-only mode (`odometry="command"`, the default until 2026-10-06) is byte-identical to before: 400 of 400 frames match (arena seed 1, 40 s).

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

## Validation 3: R9c (890646c), the starts of validation 2 (seeds 5–9, heading offset 36°)

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
   - **Correction (2026-10-11):** the windows never completed. A defect restarted each window whenever the temperature-compensated offset changed, which is every frame. R9e's development numbers and validation 4 therefore had no vision offset feedback; the stops and the temperature trigger did the work. The defect is fixed in 6ddf562, and the feedback is off (`odo_bias_feedback`) until it is evaluated.
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

## Validation 4: R9e (f0748fc), plus the IMU-off fix 811adbc, on the starts of validations 2–3

The IMU arm was run fresh. The camera-only arm is byte-identical and reused.
- **Starts:** seeds 5–9 with heading offset 36°, the starts of validations 2 and 3, so no start was untouched. The two worst P4 pairs below (heldout_c s8, home_a s5) are the pairs inspected after validation 3 (pre-registration addendum D).
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

**The one outlier** (arena_textured s7, 11.6° and 23.9 cm; its camera-only pair had 38.3° and 54.3 cm) is a **false loop closure** at t = 103.4 s, off by 0.21 m and 12°. It is the only false one of 144 closures in the IMU arm's 36 builds; the camera-only arm had 2 false of 110 in its 33 builds (validation 2). Loop closure is a camera-side component from rounds 5–6. Next improvement: reject loop corrections whose heading change exceeds the gyro's own uncertainty (about 0.1° there).

**Other checks:**
- **Slip:** pass.
  - Slippery floor: median 1.2° (worst 1.8°) against 11.9° (38.2°), no losses.
  - Slip patches: 0.9° (1.1°) against 4.5° (37.0°), no losses.
  - The slip flag is on for 1260–3045 frames in the patch runs.
- **Navigation (L3):** pass. Supported goals reached 12/14 against 9/14, reach goals 6 against 2, 0 contacts. Mapping ATE 0.7–4.6 against 0.7–4.4 cm. These are 4 runs, one per world; `--seeds 5` starts at seed 0's pose.
- **Sensor faults (F):** pass after one fix.
  - Encoders off: 1.1°.
  - IMU off failed at first (15.3° against 10.6° camera-only): encoder-only rotation was still trusted over vision. Fixed in 811adbc; re-check **4.0°**, no losses.
- **Other units (U, reported, not required):**
  - Calibrated units 2–5: 0.7–2.1°.
  - Uncalibrated (datasheet prior): 0.8–2.9°, except unit 4 (−1 % sensitivity) at 19.2°, which fails the reported U rule (camera only 10.6°). Its in-run scale learning did not correct it (estimate 0.9994 against a true 0.990). **A unit must be calibrated** with `scripts/calibrate_imu.py`.
- **Learning (L5, reported, not required):** not run.

## Decision

Every required pre-registered check passes: L1, L1-slip, L3 and F. **`RuntimeConfig.odometry = "imu_encoders"` is the default from 2026-10-06.** `"command"` reproduces the camera-only robot exactly.

**Scope of this evidence** (pre-registration addendum D):
- starts reused from validations 2–3;
- IMU unit 1 only in L1, slip and L3;
- runs of at most 7 minutes;
- whole-run sensor faults on one start;
- no vision offset feedback (inactive by a defect);
- L5 not run.

A re-validation on untouched starts follows the 2026-10-11 fixes (validation 5 below).

**Not yet evaluated in closed loop** (switches off):
- WS1 wheel model;
- WS2 command-model odometry (used as the fallback when no samples arrive);
- WS4 depth floor mask;
- WS5 view-aware turns.

The loop-closure heading guard is on in R9g and was part of validation 5.

## Validation 5: R9g (aa143e3) on untouched starts, corrected physics (2026-10-11)

Pre-registration addendum E. What R9g is: the default above, plus the 2026-10-11 fixes, the loop-closure odometry guard and the blocked-robot check ([r9-followup.md](r9-followup.md)). The simulator's caster contact was corrected first, so both arms ran under the new physics.
- **Starts:** seeds 10–14 with heading offset 18°, never used before.
- **Runs:** 124 jobs from the pinned worktree `work/wt/r9g-val-aa143e3`, 8 at a time, none failed.
- **Evidence:** `work/evidence/r9g-val-20261011/`, scored by `analysis/val5.json` and `analysis/{L1,S1,S2,long}.json`.

**A property of the baseline that matters for P4.** `build_map.py` ends a build after the operator has re-enabled autonomy 6 times; this rule has been there since the first commit (2026-09-30). Camera-only builds hit it often:
- L1: 5 of 25 runs ended early (95–372 s of 420 s);
- slip: 4 of 9;
- 30-minute builds: all 3 (140–466 s).

The IMU arm hit it once (slip patch, arena s10, 178 s). An early-ended run's ATE covers only the path it drove and the frames it tracked.

**L1, 25 pairs:**

| Metric | Camera only | R9g | Paired median Δ (95 % CI) |
|---|---|---|---|
| Max heading error per run, median (worst) | 17.3° (59.4°) | **1.38° (2.41°)** | −15.6° (−19.4, −6.2) |
| Drift events per 100 in-place turns | 6.72 (96 / 1428) | **0 (0 / 1764)** | — |
| ATE median (worst) | 6.70 cm (53.0) | **2.13 cm (6.35)** | −3.97 cm (−9.26, −1.17); worst pair **+5.03 cm** |
| Keyframe map RMSE median | 6.96 cm | 2.00 cm | −5.46 cm (−10.65, −1.32) |
| Lost fraction, mean (max) | 20.9 % (73.1 %) | **0.25 % (6.2 %)** | — |
| Losses: false / genuine | 24 / 50 | 1 / 0 | — |
| In-place turn scale (estimated / true), median of run medians | 1.0032 | 1.0004 | — |
| Recalibration stops per run | — | 2.0 (max 3), 3.9 s in total | — |

**Strata** (mean change of heading error per tracked frame, pooled):

| Stratum | Camera only | R9g |
|---|---|---|
| Still | 0.118° | 0.0043° |
| Straight | 0.172° | 0.0025° |
| Arc | 0.199° | 0.0027° |
| In-place turn | 0.169° | 0.0024° |
| Turn rate < 0.15 rad/s | 0.158° | 0.0031° |
| Turn rate 0.15–0.5 rad/s | 0.164° | 0.0023° |
| Turn rate ≥ 0.5 rad/s | 0.212° | 0.0029° |

Against the criteria:
- **P1, P2, P3 pass.**
- **P5 passes** at the arm level, as judged in validations 1–4: mean 0.25 % against 20.9 %. One pair is above its baseline + 1 pp: arena s14, 0 % → 6.2 %, one 26 s stop after the blocked-robot check fired against a wall (below).
- **P4 fails on one pair.** heldout_c s11: 0.71 → 5.74 cm.
  - The camera-only run ended at 174 s, after 6 re-enables. It drove 1.7 m and tracked 516 frames.
  - The R9g run drove 12.7 m in 420 s without a loss. Its heading error stayed under 2°, and its position error grew to 10.9 cm during long drives into unmapped space between 150 and 210 s, before falling back to 3 cm.
  - Over the same 174 s, R9g's ATE is 1.98 cm (+1.27 cm). This is a post-hoc observation, not a change of the criterion.
- The same pattern is behind every P4 failure since validation 2. The two worst pairs of validations 3 and 4 are heldout_c s8 and home_a s5. Their camera-only runs, shared since validation 2, ended at 211 s and 90 s.

**Slip (L1-slip), 9 pairs: pass.**

| Condition | Camera only, max heading median (worst) | R9g | Lost, mean (camera / R9g) |
|---|---|---|---|
| Floor friction 0.06, 6 pairs | 5.29° (8.03°) | 1.76° (2.02°) | 29.2 % / 7.2 % |
| Two patches of friction 0.03, 3 pairs | 4.26° (4.71°) | 1.18° (1.39°) | 73.5 % / 23.9 % |

- P1: pooled median 1.39° against 4.71°.
- P5: arm-level mean 12.7 % against 43.9 %. One pair is worse: patches, arena s10, 31.9 % → 59.4 %. That IMU run ended at 178 s.
- The slip detector fires in all patch runs: the yaw check flagged 214, 416 and 638 frames.
- Every IMU-arm loss here started with the blocked-robot check, while the pose was still within 5 cm of truth: the wheels spun without moving the robot (below).

**Navigation (L3), 12 pairs: pass.**

| | Camera only | R9g |
|---|---|---|
| Supported goals reached | 38 / 44 | **45 / 46** |
| Reach goals reached | 9 / 48 | 19 / 48 |
| Contact episodes | 0 | 0 |
| Lens blackout goal: arrived / attempted | 7 / 8 | 12 / 12 |
| Moved obstacle goal: arrived / attempted | 1 / 7 | 3 / 10 |
| Mapping ATE | 0.75–11.0 cm | 0.48–5.08 cm |

**Sensor faults (F), arena and home_a, seed 10: pass, all 8 runs.** The camera-only runs on these starts reached 17.3° and 6.3°.

| Fault | arena | home_a |
|---|---|---|
| IMU dropouts, 0.5 s every 30 s | 0.67°, 1.3 cm | 1.29°, 0.8 cm |
| Encoder dropouts, 0.5 s every 30 s | 1.35°, 2.2 cm | 1.36°, 2.8 cm |
| Left encoder stuck 5 s, twice | 1.21°, 3.3 cm, 0.5 % lost | 1.50°, 3.0 cm |
| Both sensors silent 2 s, twice | 1.10°, 1.9 cm | 2.65°, 7.5 cm |

With both sensors silent, the command-model fallback ran for 38 frames.

**Reported checks**

**30-minute builds (L1-long), 3 worlds:**
- **R9g:** worst heading error per 5-minute window:
  - arena 1.05, 1.20, 1.22, 1.73, 0.89, 0.63°;
  - home_a 1.46, 1.63, 1.92, 2.49, 2.56, 1.73°;
  - heldout_c 1.61, 1.56, 0.61, 0.78, 1.68, 1.06°.
- **ATE:** 3.8, 6.0 and 2.5 cm over 62–74 m of path; no loss.
- **Stops:** 4–5 recalibration stops per 30 minutes, 7.4–9.9 s in total. The first was triggered by temperature, the rest by the 2-minute rule.
- **Camera only:** none of the three runs lasted. They ended at 466, 334 and 140 s, reaching 12.8°, 19.2° and 1.5°.
- **P2 and P5 hold.**

**Blind box (B), near-field guard off, 4 worlds.**
- **Camera only:** one contact in the moved-obstacle test, home_a, 1.9 s. It backed off at the end of the contact.
- **R9g:** two contacts, home_a 0.8 s and home_a_dim 1.1 s. Both backed off.
- **R9g reached the moved obstacle more often:** it attempted the test in all 4 rooms against 2, and arrived in 2.
- The blocked check shortens a blind contact; it does not prevent one.

**Other IMU units (U):**
- **Calibrated units 2–5:** 0.63–1.05°. The camera-only runs on these starts reached 11.1° and 19.7°. Pass.
- **Unit 4 with the datasheet prior only:** 22.5° (arena) and 14.9° (home_a). This fails the reported rule, as in validation 4.
- Its in-run scale learning reached 0.9977 and 0.9922; the true scale is 0.990. **Units must be calibrated.**

**Learning (L5, reported):** run after this validation ([r9-followup.md](r9-followup.md#9-learning-on-the-imu-default-2026-10-11)). Drift events per 100 in-place turns: 0.08 with the IMU against 33.9 camera only (pass); outcomes 384 against 118 (not worse). The camera-only arm relocalised in only 9 of 27 phases on its own map, whose build ended at 94.8 s; on the IMU arm's map (post hoc) it relocalised in all 27 and drifted 18.8 events per 100 turns.

**Loop closures** (scored against truth, all runs of this validation):

| | Camera only, 37 builds | R9g, 63 builds |
|---|---|---|
| Closures applied | 120 | 212 |
| False (> 10 cm or > 5.7°) | 3 | **0** |
| Rejected by the odometry guard | — | 15 (5 episodes) |

**What the guard rejected:**
- **13 rejections, 4 episodes, all in arena_textured** (repetitive texture), seeds 11–13. Against truth they were 0.15–0.57 m and 7–32° wrong; every one would have been a false closure.
- **2 rejections, 1 episode, in the uncalibrated unit-4 run:** a true loop. That gyro's −1 % scale error made the odometry chain disagree with it (3.7°, 3.7σ).

The guard therefore depends on a calibrated gyro, like the rest of the fusion.

**The blocked-robot check stops the robot by declaring the pose lost.** In this validation the IMU arm went from tracking to lost 9 times; 8 of these came from the motion-consistency check.
- In 6 of the 8, the pose was within 5 cm of truth: the wheels were turning but the robot was not moving. Against a wall in arena s14; on the slippery floor and the patches in the slip runs.
- By design, the loss starts a 10 cm back-off and then a wait for the operator (round 5: never rotate after a suspected contact). Each such stop lasted about 26 s.
- On a patch, back-off and retry repeat. S2 arena s10 ended after 6 re-enables.
- The pose is right in these cases, so declaring it lost is not needed. With encoders that show no slip-free travel agreeing with vision, the robot is blocked, not mislocalised.
- **Next candidate:** keep the pose, raise "blocked" to the runtime, and back off without relocalising. Like any fix, it needs development on spent seeds and a new pre-registered validation.

### Verdict

- **L1 fails** the pre-registered rule on P4: one pair of 25 is 5.0 cm worse.
- **L1-slip, L3 and F pass.**

By the decision rule of addendum E:
- the failure is documented here;
- the claims are scoped to what passed: heading (P1–P3), losses (P5), slip, navigation and sensor faults on untouched starts;
- "no pair worse than 3 cm" is **not** claimed; the median map error is lower (−4.0 cm, 95 % CI −9.3 to −1.2).
- R9g remains the code default; the owner decides whether it stays (as after validation 3).

**For future pre-registrations:** P4 should compare both arms over the span both runs lasted, or count a run that ends early as its own outcome. This is a recommendation for the next pre-registration, not applied here.

## Validation 6: R9h (3059289) on untouched starts, addendum F (2026-10-12)

Pre-registration [addendum F](heading-r9-prereg.md#addendum-f-keep-pose-blocked-handling-and-the-first-temperature-stop-written-2026-10-12-before-any-run). Candidate R9h = R9g plus keep-pose blocked handling and the first recalibration stop at 0.5 °C; baseline = the same commit with `--vslam odo_blocked_mode="lost" --runtime imu_zupt_temp_first_c=0`. Seeds 20–24, heading offset 27°, never used before. Evidence `work/evidence/r9h-val-20261012/` (scored by `analysis/valF.json`); 76 jobs from the pinned worktree of 3059289, plus the four base-arm navigation jobs from 4f0224f (3059289 + the `--vslam` option on `navigation.py`; recorded in the addendum before scoring). No run failed; every pair is in its table.

### F1: first stop (25 L1 pairs, required)

| measure | candidate median | baseline median | paired difference median | worst pair | 95 % BCa CI | rule | result |
|---|---|---|---|---|---|---|---|
| maximum heading error | 0.88° | 1.30° | −0.19° | **+4.70°** (heldout_b s22) | −0.40 to +0.05 | median ≤ 0°, no pair worse by > 1.0° | median passes; **one pair fails** |
| ATE | 1.34 cm | 1.87 cm | −0.36 cm | +7.30 cm (the same pair) | −1.01 to −0.05 | median ≤ +0.5 cm | passes (better) |
| lost frames | 0 | 0 | 0 | +3.0 pp | — | ≤ +2 pp (median) | passes |
| recalibration stops per run | 2 | 2 | 0 | +1 | — | ≤ baseline + 1 | passes |

Reported: the first stop is `temperature_first` in 24 of 25 candidate runs (one run had no stop in either arm); it comes at 67 s (median) against 122 s, and the heading error built before it is 0.33° against 0.95° (medians). Stop time per run is unchanged (3.8 s against 3.7 s). No blocked event occurred in any L1 run, so F1 measures the first-stop change alone.

**The failing pair.** heldout_b s22: the candidate ran at 0.2–0.3° heading error until t = 320 s; at t = 326.4 s a loop closure (keyframe 673 against candidate 313) with a true error of 10.4 cm and 8.15° was accepted, the odometry consistency check passing it (`false: true` in the evaluator's record). The heading error jumped to 5.6° and stayed; the baseline's five closures were all right. The first stop had run 236 s earlier. The cause is a false loop closure that the odometry guard let through, not the first-stop mechanism; by the pre-registered rule F1 still fails on the per-pair bound.

### F2: blocked handling (9 slip pairs + 4 blind-box navigation pairs, required)

Slip set (friction 0.06 on arena and home_a × seeds 20–22; two patches of friction 0.03 on arena × seeds 20–22):

| rule | candidate | baseline | result |
|---|---|---|---|
| no loss of tracking with reason `visual_motion_inconsistent_with_commands` | 0 | 12 | passes |
| median time from a blocked event to the next autonomous drive ≤ 8 s | 3.3 s (max 3.7 s; 19 events) | no events; 26 s waits | passes |
| false blocked events ≤ 10 % | 0 of 19 (true speed before each event ≤ 0.05 m/s) | — | passes |
| contact episodes and duration not higher, per pair | 0 obstacle contacts in all 9 pairs | 0 in all 9 pairs | passes (see the scoring note) |
| ATE median not worse by > 0.5 cm | 1.93 cm | 1.67 cm (+0.25 cm) | passes |
| maximum heading error median not worse by > 0.5° | 1.32° | 1.50° (−0.19°) | passes |
| no `blocked_repeatedly` stop unless the baseline lost tracking at the same place | 2 stops (patches, arena s21 at 129.6 s, s22 at 99.3 s) | lost tracking 5 and 4 times on the same patch: nearest baseline loss 0.24 m and 0.06 m from the stop, both inside the 0.8 m patch | passes (judgement: the same patch) |

Reported: 19 blocked events (18 spinning, 1 stalled), all on the two-patch runs (none on the uniformly slippery floors, none in L1); back-off completed in 15 (the other 4 ended in a repeat-stop or a cancellation); 27 cells marked; operator re-enables 0/0/0 on friction runs, 3 and 1 on the patch runs that stopped (baseline 3, 5, 5).

**Scoring note (correction of a measurement, recorded before adoption).** The harness's contact log counted the slip-patch plates themselves (invisible 1 mm floor plates, created as unnamed entities) as "contacts", for whole runs in both arms: raw episodes 5/4/18 (candidate) against 9/1/4 (baseline) on the patch runs, 100 % of them with `entity1`/`entity2`, the plates. No wall or fixture was touched in any slip run. Contacts are read as obstacle contacts, as in every earlier round; the evaluator now names the plates and excludes them (30d08be). The raw numbers are kept in `analysis/valF.json`.

Blind-box navigation (home_a, heldout_b, heldout_c, home_a_dim × seed 20, `near_field_guard=false`; baseline from the 4f0224f worktree):

| pair | blocked events c / b | contacts c / b (episodes / s) | ATE c / b (cm) | max heading c / b | supported goals c / b |
|---|---|---|---|---|---|
| heldout_b s20 | 0 / 0 | 0 / 0 | 0.69 / 1.01 | 1.03° / 1.02° | 4/4 / 3/3 |
| heldout_c s20 | 0 / 0 | **2 (4.1 s) / 0**, both with the blind box at 372–383 s | 1.68 / 0.88 | 0.23° / 0.47° | 4/4 / 4/4 |
| home_a s20 | 0 / 0 | 1 (5.1 s) / 1 (5.8 s) | 1.05 / 0.66 | 0.53° / 0.35° | 4/4 / 2/4 |
| home_a_dim s20 | 0 / 0 | 0 / 0 | 1.01 / 0.72 | 0.61° / 0.57° | 4/4 / 4/4 |

Medians over the 13 F2 pairs stay within the bounds (ATE +0.25 cm on the slip set, +0.23 cm here; heading −0.19° / +0.05°). No blocked event occurred in any navigation run of either arm. **The contact rule fails on heldout_c s20**: the candidate touched the blind box twice (4.1 s), the baseline never. With no blocked event, the only part of the keep-pose code that acted is its dead-reckoning rule while the slip flag is on (a slip-flagged translation is not composed into the predicted pose); that, or run-to-run divergence, put the robot on a different path towards the unseen box.

**Second deviation, found while scoring.** The four base-arm navigation jobs carried `--runtime` twice (`imu_zupt_temp_first_c=0` and `near_field_guard=false`); argparse keeps the last, so the base navigation arm ran with the first stop at 0.5 °C like the candidate (both arms' first stop: 24.3 s, `temperature_first`). The navigation pairs therefore isolate `odo_blocked_mode` alone, which is what F2 is about; the L1, S1 and S2 base jobs carried one `--runtime` and ran as pre-registered (their first stops: `interval` or `temperature`). `scripts/eval/batch.py` should refuse a job line with a repeated option (recommendation below).

### Verdict

By the decision rule of addendum F:
- **F1 fails** on its per-pair bound (one pair of 25, +4.7°, a false loop closure 236 s after the first stop); its medians pass and the mechanism did what it was built for (first stop at 67 s instead of 122 s; 0.33° instead of 0.95° built before it).
- **F2 fails** on its per-pair contact rule (one navigation pair of 13, two touches of the blind box, 4.1 s, in a run with no blocked event); every other F2 rule passes, and on the nine slip pairs the mechanism did what it was built for (19 events handled, none false, the next drive 3.3 s later instead of 26 s, no blocked loss against 12, 4 operator re-enables against 13).
- "If either fails, `odo_blocked_mode` and `imu_zupt_temp_first_c` are set back to the 2026-10-11 values": both defaults are reverted (`"lost"`, `0`); both mechanisms stay switchable, tested and documented. The owner may re-adopt either by decision, as with earlier defaults; the evidence above is what that decision would rest on.
- New open items: (1) the odometry consistency check accepted a loop closure 10 cm and 8° wrong (heldout_b s22, t = 326 s): heading residual −7.9° against σθ 3.0° over a 165 s / 5.2 m chain, 2.6 σ, under the `heading` guard's 3 σ; the full 3-DoF test would also have passed (d² 10.0 against the 99.9 % bound 16.3) — a 95 % bound (7.8) or k = 2.5 would have refused it, at a cost in true closures that a pre-registered check must weigh; (2) the keep-pose dead-reckoning rule for slip-flagged frames acted outside blocked events; it is now gated on the blocked verdict (`vslam._predict_odometry`, after e9ecffa), to be re-validated.

**For future pre-registrations:** (a) a contact rule for a blocked-handling change should count contacts in runs where a blocked event occurred, or be stated as an arm-level total, since a per-pair rule on chaotic trajectories fails on divergence alone; (b) `batch.py` should refuse job lines with a repeated option; (c) the slip-patch plates are now excluded from contacts by the evaluator itself. None of this is applied to this validation.
