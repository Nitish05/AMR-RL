# Round 9 follow-up: fixes after adoption (2026-10-11)

A review of the adopted IMU + wheel-encoder default ([heading-r9.md](heading-r9.md)) listed 76 open items. This document covers the first part of the agreed order:
1. small fusion defects;
2. wrong claims;
3. the loop-closure guard;
4. blocked-robot detection;
5. the caster-friction effect in the simulator;
6. evaluation tooling.

The re-validation on untouched starts follows under its own pre-registration ([heading-r9-prereg.md](heading-r9-prereg.md), addendum E, §8).
Then learning on the IMU default (§9). Relocalisation range and the coverage-pass A/B are in [relocalisation.md](relocalisation.md#round-9-range-on-the-imu-robots-maps-2026-10-11-development-not-pre-registered), and the round-9 navigation check in [NAVIGATION_RESULTS.md](../NAVIGATION_RESULTS.md).

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

## 8. Re-validation on untouched starts (addendum E)

The full results are in [heading-r9.md](heading-r9.md#validation-5-r9g-aa143e3-on-untouched-starts-corrected-physics-2026-10-11). In brief:
- **L1 fails on P4, one pair of 25:** heldout_c s11 is +5.0 cm. Its camera-only run ended at 174 s, at the 6-re-enable limit; over the same span the IMU run was +1.3 cm (post hoc).
- **Every other required criterion passes,** and so do slip, navigation and the mid-run faults.
- **The loop-closure guard** rejected 4 false episodes (7–32°, arena_textured); 0 false closures were applied in 63 IMU builds.
- **The blocked-robot check** shortened blind contacts (back-off 0.8–1.1 s after contact). It also turned every wheel-spin event into a loss with a correct pose, about 26 s each.

**Where the remaining heading error comes from** (diagnosed on the validation-5 runs, scored against the simulated offset; nothing was changed):
- The robot measures the gyro offset at boot. Until the first recalibration stop it does not know how the offset moves with temperature: the tempco prior has mean 0.
- **Slow warm-up:** in seeds 10, 12 and 14 the chip warmed by less than 1 °C in the first 2 minutes. The first stop came at 115–122 s (in seed 14, at 0.98 °C, by the 2-minute rule).
  - Meanwhile unit 1's offset moved by up to 0.031 °/s; its tempco is −0.025 °/s per °C, 2.5× the datasheet's typical value.
  - The filter's own σ was 0.023 °/s at that stop, so it was honest about this.
- **Heading error before the first stop is 72 % of the run's maximum** (median over the 25 runs). In heldout_c s14 it was 1.84° of 1.92°.
- After the first stop the tempco is known, and the offset error stays at 0.002–0.005 °/s.
- **Faster warm-up:** with a stop at 42–48 s (seeds 11 and 13), the error before the first stop is only 0.26–0.66°. Those runs' maxima come later, from other sources.

**Options for the owner** (each would need development on spent seeds and its own pre-registration; not done):
- trigger the first temperature stop at 0.5 °C, until the tempco has been measured. This costs about one more 2-s stop early in a run;
- or stop when the heading uncertainty that the offset uncertainty has added since the last stop exceeds a budget (for example 0.5°), instead of fixed 2 minutes and 1 °C.

## 9. Learning on the IMU default (2026-10-11)

Evidence: `work/evidence/r9g-learn-20261011/` (maps chosen before any run, `MAP_CHOICE.md`;
scored in `analysis/learning.json` and `analysis/report-*.md`).

### The round-6 suite on the IMU default

7 experiments × 5 starts in the plain arena, on the round-6 map, code 8946f9e: restart 9/9, reversal **10/10** (round 6, camera only: 7/10), 0 of 131 relocalisations false, 0 drift events in 1295 in-place turns, nudges reached their target 24/24. Full table: [LEARNING_RESULTS.md](../LEARNING_RESULTS.md#round-9-2026-10-11-the-suite-on-the-imu-default).

### L5 as pre-registered (addendum E; reported, not required)

arena_textured, starts 0–2, 7 experiments, each arm on its own validation-5 map of seed 10,
code aa143e3. 27 phases per arm.

| | camera only | IMU + encoders |
|---|---|---|
| phases relocalised at start | **9 / 27** (start 0 only) | **27 / 27** |
| drift events (≥ 8° in 10 s) per 100 in-place turns | 33.9 (140 / 413) | **0.08** (1 / 1214) |
| per-run max heading error, median (worst) | 30.1° (31.4°) | 1.5° (6.1°) |
| relocalisations during the runs: all / false | 38 / **25** | 38 / **0** |
| attempts / outcomes / useful / aversive | 591 / 118 / 25 / 0 | 1828 / 384 / 119 / 3 |
| forced recalibration stops | — | 0 (natural stops kept the offset current) |

- **Drift: pass** (halved many times over). **Outcomes: not worse** (384 against 118).
- **The comparison is confounded by the camera-only map.** Its seed-10 build ended at 94.8 s
  after 6 re-enables (1.1 m of path). Starts 1 and 2 never relocalised on it, and start 0
  ran with heading errors up to 31°. The supplementary L5x arm below separates map from
  odometry.
- **Learning in arena_textured is limited by the colour detector, not by localisation.**
  - IMU arm: 61 % of attempts (1113 / 1828) were abandoned because the target was not
    confirmed in fresh images; 715 were aimed at no real fixture (texture read as objects).
  - Stone and bloom, the fixtures whose rules the history and reversal experiments change,
    were engaged twice in 27 phases.
  - So history_a and history_b, and reversal_early and reversal_late, ran identical
    trajectories until stone was first touched (deterministic simulator):
    - restart: the first decision targeted the learned option in 2 / 4;
    - reversal: adapted in 2 / 6 (both seed 2).
  - This matches the round-4 testbed prediction for the colour detector in textured worlds
    (restart 6 / 20, reversal 14 / 20; [textured-worlds.md](textured-worlds.md#5d-measured-errors-in-the-learner-testbed)).
  - The open-vocabulary arm below tests the detector swap.

### L5x: the camera-only robot on the IMU arm's map (post hoc, labelled)

Added after the camera-only L5 phases failed to relocalise and before any IMU-arm L5 result
was read (`MAP_CHOICE.md`). Same experiments, starts and code; only the map differs from L5.

| on one shared map (L1-odo-arena_textured-s10) | camera only (L5x) | IMU + encoders (L5) |
|---|---|---|
| phases relocalised at start | 27 / 27 | 27 / 27 |
| drift events per 100 in-place turns | 18.8 (225 / 1195) | **0.08** (1 / 1214) |
| per-run max heading error, median (worst) | 15.8° (34.6°) | **1.5° (6.1°)** |
| relocalisations during the runs: all / false | 110 / 14 | 38 / **0** |
| attempts / outcomes / useful / aversive | 1926 / 443 / 105 / 5 | 1828 / 384 / 119 / 3 |
| restart: first decision = learned option | 3 / 5 | 2 / 4 |
| reversal adapted | 2 / 6 | 2 / 6 |

- **The camera-only L5 failure was its map:** on the IMU arm's map every start relocalised.
- **Localisation still differs sharply:** the camera-only robot drifts (16° median worst heading)
  and makes false relocalisations; the IMU robot does neither.
- **Learning counts do not differ much.** In this world they are set by the colour detector
  (see L5 above), not by localisation. The drifting robot wandered more widely and touched bloom and
  stone more often (113 attempts against 2), but adapted no more often.

### L5ov: the IMU arm with the open-vocabulary detector (development)

Same map, experiments, starts and code as L5's IMU arm; `--detector open_vocab` (chosen before
the run, `MAP_CHOICE.md`). All 27 phases relocalised; 0 of 67 relocalisations false; 0 drift
events in 853 turns.
- **Restart 6 / 6 and reversal 6 / 6**, against 2 / 4 and 2 / 6 with the colour detector.
- Attempts abandoned for an unconfirmed target fell from 61 % to 28 %; useful outcomes per
  attempt rose from 0.07 to 0.20.
- Two phases got a second red panel from grump: one duplicate identity chosen 1.7 s before
  its merge, and one re-probe while grump's flag was already raised (a visual context the
  robot had no experience of).
- Full table: [textured-worlds.md](textured-worlds.md#5d-measured-errors-in-the-learner-testbed).

### Frustration renewal A/B (development; switch `ChooserConfig.frustration_renewal_s`, off by default)

The switch renews one probe on long-untried options when the robot has idled for that long
with an unsatisfied need. It targets the round-6 start-4 idle trap (idle from the rule swap to
the end of the run). Off (0) against 60 s; reversal_early and reversal_late; code 2ba9f93.

| runs | map | relocalised | adapted, off | adapted, 60 s | renewal fired |
|---|---|---|---|---|---|
| IMU, starts 0–4 | L1-odo-arena-s10 | 8 / 10 (start 4 never) | 8 / 8 | 8 / 8 | 1 run |
| start 4, IMU and camera only | round-6 map `i3-nocov-arena-s4` | 8 / 8 | 4 / 4 | 4 / 4 | never |

- **The round-6 trap no longer occurs.** On the round-6 map, start 4 adapted in every run, with
  either odometry, 35–244 s after the swap (stone/signal). The code and the simulator's caster
  physics have changed since round 6, and its path with them; which change removed the trap was
  not analysed.
- **Renewal fired in one run of 20** (reversal_early s3 on the first map: 4 renewals from 321 s).
  There the robot found the new option 520 s after the swap instead of 705 s (4 s before the
  end), with 8 useful outcomes instead of 3. The other runs were identical with and without it.
- **Kept off.** It helped in its one firing and changed nothing elsewhere, but n = 1 is not
  evidence of a general gain. Enabling it is a small owner decision.

## 10. Owner decisions implemented: keep the pose when blocked; an earlier first stop (2026-10-12)

Both decisions from §8 were taken and built; their live validation is pre-registered as
[addendum F](heading-r9-prereg.md#addendum-f-keep-pose-blocked-handling-and-the-first-temperature-stop-written-2026-10-12-before-any-run).

**Blocked wheels keep the pose** (`VSLAMConfig.odo_blocked_mode = "keep_pose"`; `"lost"` is the
2026-10-11 behaviour and, since validation 6, the default again — see the result below).
- The VSLAM's blocked check is unchanged: commanded forward travel over the 1.2 s window that
  vision did not see, after the frozen-estimate check (wheel travel without slip that vision did
  not see) has passed. Vision and the wheels agree the robot did not move, so the pose is right
  (the rover slip check; Maimone, Cheng and Matthies 2007; Kilic, Gu and Gross 2022 keep the
  localisation and report the slip).
- Instead of a loss: `TrackResult.info["blocked"]`, a kind (`stalled`: no encoder travel;
  `spinning`: slip-flagged; `unknown`: no vision, wheels only), and a restart of the consistency
  windows so the back-off is judged afresh. Tracking, or dead reckoning while vision is out,
  continues.
- The runtime (`_on_blocked`, `_blocked_step`, `_end_backoff`): cancel the activity with reason
  `blocked`, then back straight off 10 cm at 0.06 m/s under normal autonomy, away from the side
  that was being driven (a blocked reverse drive goes forward; a blocked turn backs off nowhere),
  only into space the map knows to be clear, for at most 4 s, measured by the kept pose; then
  hand back to the chooser with a `blocked` trigger. Stalled wheels raise the cells just beyond
  the driven side of the footprint to the depth guard's obstacle level, journalled, so planning
  routes around the spot; a second event within 0.3 m in a minute marks the spot whatever the
  kind (a floor the robot cannot drive on: the first smoke run backed off and drove into the
  same slip patch four times). More than three events in a minute stop autonomy
  (`blocked_repeatedly`) for the operator. Any revocation, operator command or loss of tracking
  closes an open back-off and records it as interrupted.
- During probation after a relocalisation a blocked verdict no longer rejects the
  relocalisation (the wheels did not move and vision agrees). Without vision, slip-flagged
  encoder travel is not composed into the dead-reckoned pose, so spinning wheels do not walk it
  (since validation 6 only while the blocked verdict holds; before, on every slip-flagged frame).
- Nudging cannot trigger the check: the nudge creep (0.06 m/s) commands less than the
  0.12 m the verdict needs in its 1.2 s window.
- Face and console: activity `blocked`, caption "Blocked - backing off"; `odometry_status()`
  reports `blocked_active`, `blocked_events`, `blocked_kind`; results carry `blocked_events`
  with each event's kind, pose, activity and back-off outcome.
- Tests: `test_blocked_keeps_the_pose_and_restarts_the_window`,
  `test_blocked_runtime_backs_off_with_the_pose_kept_then_resumes`,
  `test_repeated_blocked_events_stop_autonomy_for_the_operator`,
  `test_blocked_backoff_has_its_own_face_and_caption`.

**The first recalibration stop at 0.5 °C** (`RuntimeConfig.imu_zupt_temp_first_c = 0.5`; 0 is
the 2026-10-11 behaviour).
- `WheelInertialOdometry.tempco_known`: an offset measurement has been taken at least 0.4 °C
  from the boot temperature (`OdometryConfig.tempco_measured_dt_c`), or the tempco's
  uncertainty is below half its datasheet prior. One stop at 0.5 °C therefore ends the rule; the
  first smoke run, with a covariance-only criterion, stopped twice (34 s and 70 s).
- Until then the temperature trigger fires at 0.5 °C, whatever the offset's own uncertainty
  (the stop is what measures the tempco; the 2-minute rule and the 1 °C rule still require an
  uncertain offset). A stop is labelled `temperature_first` unless the 1 °C rule would have fired
  by itself; `tempco_known` is logged with every stop.
- Tests: `test_first_temperature_stop_at_half_a_degree_until_the_tempco_is_known`,
  `test_tempco_is_known_only_after_an_offset_measurement_at_another_temperature`.

An adversarial review of the diff (three reviewers; 15 findings, 2 verified before the session
limit, the rest judged by hand) led to the direction, clearance, interruption, probation,
dead-reckoning and labelling rules above, plus the per-frame `blocked` flag in the scoring rows
and the blocked probe's detection of the kept-pose verdict.

Nothing was tuned on any validation start.

**Result (validation 6, [heading-r9.md](heading-r9.md#validation-6-r9h-3059289-on-untouched-starts-addendum-f-2026-10-12)).**
Both mechanisms did what they were built for, and both failed one pre-registered per-pair bound,
so by the rule of addendum F both defaults are back at the 2026-10-11 values and both stay
switchable. F2: 19 blocked events on the slip patches, none false, the next drive 3.3 s after
each (26 s waits before), no `visual_motion_inconsistent` loss (12 in the baseline), ATE and
heading medians within bounds, the two `blocked_repeatedly` stops where the baseline lost
tracking on the same patch; but one blind-box navigation pair of 13 had two touches of the
unseen box (4.1 s) that the baseline did not, in a run with no blocked event. F1: the first stop
came at 67 s instead of 122 s with 0.33° instead of 0.95° built before it, medians better
(heading −0.19°, ATE −0.36 cm), but one pair of 25 was 4.7° worse through a false loop closure
236 s later. Two scoring findings: the slip-patch plates were counted as contacts (fixed in the
evaluator, 30d08be) and the base navigation arm carried `--runtime` twice, so it ran with the
first stop at 0.5 °C (the navigation pairs isolate the blocked change alone). Next: gate the
keep-pose slip-frame dead reckoning on the blocked state, tighten the loop-closure odometry
guard, and re-validate on new seeds; or the owner re-adopts by decision.

## 11. The open-vocabulary detector becomes the default (2026-10-12)

`RuntimeConfig.detector = "open_vocab"` (owner decision). The model is loaded from the local cache
only (`scripts/amr.sh fetch-detector-model`); without it the engineered colour detector runs and
`detector_status` says so, as the depth guard does. Basis: in the textured arena the IMU robot
learned only with this detector (restart 6/6, reversal 6/6 against 2/4 and 2/6; §9), and its
held-out benchmark is recall 1.0, precision 0.97 ([textured-worlds.md](textured-worlds.md)).

**Check, written before the run.** The plain-arena learning suite (7 experiments × 5 starts on the
round-6 map, as in [LEARNING_RESULTS.md](../LEARNING_RESULTS.md) round 9) is repeated with the new
default. Not worse means: starts relocalised 5/5; restart first decision = learned option in at
least 8 of the runs that learned one; reversal adapted ≥ 8/10; at most one red panel per phase in
≥ 43 of 45 phases; 0 unintended contact episodes; 0 false relocalisations. If it is worse, the
default goes back to the colour detector and the result is documented.

