# Round 9 follow-up: fixes after adoption (2026-10-11)

A review of the adopted IMU + wheel-encoder default ([heading-r9.md](heading-r9.md)) listed 76 open items. This document covers the first part of the agreed order:
1. small fusion defects;
2. wrong claims;
3. the loop-closure guard;
4. blocked-robot detection;
5. the caster-friction effect in the simulator;
6. evaluation tooling.

The re-validation on untouched starts follows under its own pre-registration ([heading-r9-prereg.md](heading-r9-prereg.md), addendum E).

Sources consulted are cited where they shaped a design. Library behaviour was checked against the installed source and, for Genesis, its documentation via Context7.

## 1. Fusion defects (6ddf562)

| Defect | Effect | Fix | Test |
|---|---|---|---|
| A sensor returning after a dropout re-integrated the gap: the IMU at one sample's rate, the cumulative encoder counts on top of the command dead reckoning | about 14° from a 0.5 s IMU gap during a turn; distance counted twice | integrate only from the current frame's start; scale the first encoder delta to the frame's share | `test_gyro_dropout_is_not_integrated_twice`, `test_encoder_dropout_travel_is_not_counted_twice` |
| IMU missing: the in-frame heading used the right wheel's travel alone (`dl_tot * 0`) | straight drives bent within each frame | the encoders' differential yaw | `test_encoder_heading_without_the_imu_keeps_a_straight_drive_straight` |
| Out-of-order samples | negative time steps | dropped and counted | `test_out_of_order_samples_are_dropped` |
| The vision gyro-offset window restarted whenever the temperature-compensated offset value changed (every frame) | no window ever completed; validation 4 ran without this feedback | restart only on a real re-measurement (an `offset_updates` epoch), a map correction or a tracking gap; behind `odo_bias_feedback` (off until evaluated) | `test_vision_offset_window_completes_while_the_temperature_model_moves` |
| Vision-referenced accumulators (offset window, scale anchor, radius stretch) survived loop closures, relocalisations, resyncs and dead-reckoned frames | samples mixing two map frames | reset on each | `test_map_corrections_and_tracking_gaps_restart_vision_windows` |
| The boot stand-still was keyed to absolute time 0 | no boot hold on a clock that does not start at 0 | measured from the first control tick | `test_boot_hold_is_measured_from_the_first_tick_not_absolute_time` |

Each new test fails on the previous code.

## 2. Corrections (fc49240)

Pre-registration addendum D records the deviations found after adoption:
- validations 3–4 reused validation 2's starts;
- the slip patches used friction 0.05, not 0.08;
- L3's seed 5 started at seed 0's pose;
- L5 was not run;
- U failed for uncalibrated unit 4;
- the vision offset windows were inactive;
- strata were not computed;
- faults were tested only as whole-run switches.

The ledger and README rows were scoped to what was measured. The round-6 reversal count is **7/10, not 6/10**. The report's scorer used perceived outcomes, and a misperceived `moved` before the swap hid seed 0's late adaptation. The scorer now uses the world's own panel events (`report.first_new_rewarding`).

## 3. Loop-closure odometry guard (ce5cf69)

**Problem.** Validation 4's worst run (arena_textured s7: 11.6°, 23.9 cm) came from a verified, confirmed loop closure that was 0.21 m and 12° off. It was accepted with robust weight 0.95 while the gyro knew the relative heading to about 1°.

**Mechanism.** Every verified loop is compared with the odometry chain between its two keyframes (gyro heading, encoder distance), independent of the map. The heading uncertainty is built from:
- the chain's propagated covariance;
- the correlated terms that a per-frame covariance treats as independent: gyro offset σ × time and scale σ × rotation, accumulated per keyframe as `odo_drift`;
- the loop pose's own error (0.5°).

`loop_odo_guard="heading"` rejects a loop whose heading residual exceeds 3.29σ (χ²₁ at 99.9 %). `"full"` adds a 3-DoF Mahalanobis test.

**Basis in the literature.** Production visual-inertial systems also gate loops on yaw consistency, but with fixed limits:
- ORB-SLAM3 `LoopClosing.cc`: 20° yaw, plus roll and pitch;
- VINS-Mono / VINS-Fusion `keyframe.cpp`: 30° and 20 m.

A covariance-based test with odometry-chain compounding is Kimera-RPGO's odometry check (Rosinol et al. 2020) and the per-link test of RRR (Latif et al. 2013). The research agent also noted that fixed chain sigmas let one false loop bend the chain cheaply, which is why the robust kernel kept a weight near 1.

**Evidence.**
- **15 IMU shadow builds** (seeds 15–17, heading offset 54°, five worlds; old physics) produced 32 loop constraints, none false.
  - The guard rejects none of them: heading residual at most 2.0°, at most 1.25σ; d² at most 1.6.
  - Its σ is conservative: |z| has a median of 0.27, against 0.67 if calibrated.
  - Loop closure itself, applied offline with the live rules, improved the keyframe map in 5 of 15 runs, worsened none, and left the median unchanged. It is kept on.
- **Replay of validation 4's worst run** at the validated commit, with only this logging added (`work/wt/r9f-logcheck`):
  - The run is reproduced exactly: 11.59°, 23.91 cm, the same 4 closures.
  - The false closure has a heading residual of 12.66° against σ = 1.00° (12.6σ) and d² = 166, so it is **rejected**.
  - The three true closures pass (at most 0.14σ).
- **Synthetic test:** `test_loop_odometry_check_rejects_a_closure_the_gyro_contradicts`.

**Limits.**
- A broken odometry chain disables the check: a loss of tracking, or a candidate from a loaded map.
- Long intervals without a stop widen σ. At 0.02 dps offset σ, 300 s gives about 6°.
- Aliasing that differs only in position passes the heading test.

## 4. Blocked-robot detection (8533459)

**Problem.**
- With odometry, the VSLAM replaces the command by the measured motion. Its frozen-estimate check therefore compared vision with encoder travel.
- Against an obstacle the camera does not see, the wheels either stall (no counts) or spin; spinning is slip-flagged, and slip frames were excluded. So the check never fired, and the contact back-off never started.
- The wheel-slip yaw check cannot see a straight push at all: both wheels spin equally.

**Probe** (`scripts/dev/blocked_probe.py`: drive at 0.2 m/s into a 35 cm, 86 kg box, near-field guard off; physics identical across arms):

| Box ahead | Camera only | IMU, before | IMU, after |
|---|---|---|---|
| 0.6 m | noticed 1.1 s after contact | **never** (the wheels spin at full speed, slip-flagged) | 1.1 s after contact |
| 0.25 m | vision lost before contact (the box fills the view); no motion-consistency loss | same | vision-free check below |
| none | no false alarm | — | no false alarm |

**Fix**, following the MER slip check (Maimone, Cheng, Matthies, *J. Field Robotics* 2007): compare vision with the robot's own commanded travel, not with slip-filtered odometry.
- **Vision tracking:** commanded forward travel of at least 0.12 m in 1.2 s, with vision showing less than 25 % of it, means "blocked". That is a motion-consistency loss, so the robot backs off 10 cm.
- **No visual fix:** commanded travel that the wheels did not make without slip ends dead reckoning the same way.
- **At loss time:** the recovery choice backs off if the last 0.5 s show a blocked robot.
- **Recovery turns:** a turn the gyro shows is not happening is stopped (round 5 rotated against a box for 18 s).

**Not yet tested:** the closed-loop navigation box test with the guard off. It is part of the re-validation.

## 5. Simulator: the caster dragged at the floor's friction (5cd6db9)

**Cause.** Genesis gives a contact the larger of the two surfaces' friction values:
- `contact.py:395`, `qd.max(qd.max(friction_a, friction_b), 1e-2)`;
- documented under `RigidEntity.set_friction`.

The floor plane kept the default 1.0, so the ball caster (spec friction 0.02, carrying about 10.6 N) dragged at 1.0. The default pyramidal friction cone is aligned with the world axes, so the resulting skid depends on heading.

**Drive physics probe** (`scripts/dev/drive_physics_probe.py`: empty arena, open loop through the wheel backend; phases that touched a wall excluded):

| Model | True / wheel distance (straight) | Yaw rate with equal wheels | In-place turn: true / encoder yaw | True / commanded |
|---|---|---|---|---|
| Floor 1.0 (until now) | 0.89–0.95 | 1–6 dps, sign depends on heading | 0.90 | 0.84 |
| Floor at the caster's 0.02 (now) | 0.997 | 0.02–0.04 dps | 0.98 | 0.965 |
| Elliptic cone, floor 1.0 | 0.998 | 0 | 0.90 | 0.82 |
| Wheel armature 1e-4 | 0.90 | about 5 dps | 0.91 | 0.84 |
| 4 substeps | 0.94–0.95 | 1.4–2.4 dps | 0.90 | 0.83 |

**What it explains.**
- Round 9's "wheels roll about 10 % short of the spec radius".
- "The chassis skids 0.1–0.3 dps while driving straight", which kept the straight-driving offset update off.

**The new model.**
- The floor carries the caster's friction, and loose objects bring their own: 1.0, which was their effective floor friction before, so object behaviour is unchanged.
- `physics.legacy_floor` reproduces the old model exactly.
- **Every earlier result was produced under the old model.**

**Consequences.**
- The camera-only baseline has to be re-run.
- The wheel-response model (WS1, fitted on old-model logs) is labelled old-physics.
- A slippery floor must be much more slippery to make the wheels slip. At floor friction 0.06 the wheels lose about 40 % of the travel while speeding up and 1–2 % in steady driving. 0.45, the old slip check, causes no slip.

**Not changed (recorded as an assumption).** The wheel joints carry Genesis's default armature of 0.1 kg·m². That is about 40 kg of equivalent mass per wheel, and it limits acceleration to about 0.5 m/s², much as a motor controller's ramp would. A realistic reflected rotor inertia (Pololu gives no figure) would also need an explicit acceleration limit, so it belongs to the motor-model item.

## 6. Evaluation tooling (ce5cf69, 8533459, dd638a4, 8583a4b)

**Batch runner and environment**
- `scripts/eval/batch.py` (`amr.sh batch`):
  - concurrency cap;
  - caffeinate on macOS;
  - retries, with failed logs kept;
  - resumable.
- The world fails early with a clear message when macOS has no display.
- `requirements/lock-macos-arm64-py3.12.txt` is used by setup; there is now a single OpenCV wheel.

**What runs now record**
- Map, navigation and learning results now save the odometry status and a log of each recalibration stop (start, end, trigger, frames measured, success).
- Scoring rows carry the simulated gyro offset, for scoring only.

**Report and tests**
- `turn_live_report` adds strata (motion class, turn-rate band), per-turn scale error, and a false/genuine loss split.
- Unit tests for the scoring code.
- A Genesis test of the simulated gyro, encoders and accelerometer against truth.

## 7. Development: which switches go into the re-validated candidate

Six spent scenarios (arena s0, s3; arena_textured s2, s4; home_a s1; heldout_b s2), 420 s coverage builds, corrected physics. Evidence: `work/evidence/r9g-dev-20261011/`, commit dd638a4. The offset error is scored against the simulated offset (evaluation only).

| Arm | Max heading error, mean (worst) | ATE, mean | Lost | Stops per run | Offset error, median over time |
|---|---|---|---|---|---|
| **Base: guard + blocked check + fixes** | **1.02° (1.31°)** | **1.34 cm** | **0 %** | 1.7 | 0.001–0.007 dps |
| + straight-driving update (ZARU) | 1.25° (1.94°) | 1.52 cm | 0.2 % | 1.5 | 0.001–0.010 dps |
| + vision offset windows | 3.63° (10.4°) | 4.50 cm | 5.1 % | 0.2 | 0.003–0.043 dps |
| + both | 2.90° (9.39°) | 3.80 cm | 0.2 % | 0.2 | 0.003–0.036 dps |
| Camera only | 13.6° (37.0°) | 9.15 cm | 19 % | — | — |

**The vision offset windows are harmful,** now that they actually complete. Over a 15–90 s window, vision's heading against established landmarks changes with the map's own heading inconsistency between regions. The windows measured that inconsistency (up to about 0.04 dps), not the gyro offset. Their confident updates also kept the offset uncertainty below the recalibration threshold, so the stops that would have corrected it never came. They stay off.

**The straight-driving update gives no gain** even with the caster fixed. It stays off; the stops are short (1–2 per 7 minutes, about 2 s each).

**One loop rejected by the guard** (ZF arena_textured s2, t = 92 s):
- Its heading disagreed with the gyro by 2.31° (3.5σ).
- Against truth, the loop was 5.8 cm and 2.2° wrong. That is below the "false" thresholds (10 cm, 5.7°), but it would have put a 2.2° error into the map.

The re-validation (pre-registration addendum E) uses the base arm.
