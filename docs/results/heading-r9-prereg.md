# Pre-registration: round 9, IMU + wheel encoders (written 2026-10-05, before any validation run)

**Owner decision (2026-10-05):**
- Add an IMU and wheel encoders that can be bought off the shelf, and keep wheel slip in mind.
- Implement the round-9 plan ([plans/heading-robustness-plan.md](../plans/heading-robustness-plan.md)).

**Parts:**
- ST LSM6DSOX on the Adafruit 4438 breakout.
- Pololu 4754 gearmotors (70:1, 64 CPR encoders, so 4480 counts per wheel revolution).
- The camera stays as it is; it matches the Arducam B0394.
- Every number is in `assets/robot/amr_spec.yaml`. The calibration is in `configs/calibration/imu.yaml`, from `scripts/calibrate_imu.py`.

**Candidate.**
- **R9-odo:** `RuntimeConfig.odometry = "imu_encoders"`, with the defaults of `OdometryConfig` and `VSLAMConfig.odo_*` frozen at the commit that adds this file.
- **Baseline:** `odometry = "command"`, which is byte-identical to before (checked: 400 of 400 frames identical on arena seed 1, 40 s).

Development runs used only arena seed 0 and arena_textured seed 2, both 60 s builds without the coverage pass. The development runs and their numbers are recorded in [heading-r9.md](heading-r9.md).

## Checks (run once each, from a pinned worktree, paired: same commit, worlds and seeds)

| check | runs | pass if |
|---|---|---|
| L1 | coverage-pass builds as in round 8 (`build_map.py --seconds 420 --coverage-seconds 150`): arena, arena_textured, home_a, heldout_b, heldout_c × seeds 0–4. That is 25 pairs. heldout_b and heldout_c were never used in development | **P1:** median of the per-run maximum heading error at most 50 % of baseline and at most 5°.<br>**P2:** no run above 15°.<br>**P3:** drift events (≥ 8° within 10 s) per 100 in-place turns at most 25 % of baseline.<br>**P4:** paired ATE median Δ ≤ 0 cm, and no pair worse by more than 3 cm.<br>**P5:** lost fraction at most baseline + 1 pp. |
| L1-slip | arena and home_a × seeds 0–2 with `wheel_friction=0.45` everywhere, plus arena seeds 0–2 with two slip patches (friction 0.08) on the robot's early path. 9 pairs | R9-odo P1 and P5 hold against the paired baseline. The slip detector fires in the patch runs |
| L3 | navigation: home_a, heldout_b, heldout_c, home_a_dim × seed 0 (`navigation.py`) | arrivals at least baseline − 1; contact episodes not up |
| L5 | learning: arena_textured × seeds 0–2 on a map built by each arm | drift events per 100 turns at least halved; outcomes not worse (same counts ± 1) |
| F | sensor faults, arena seed 0: `imu_off`, `encoders_off` (whole run) | no crash; P5 holds; heading not worse than baseline |
| U | other IMU units (`imu_unit` 2–5), arena seed 0, datasheet prior only (no calibration for that unit) | not worse than baseline (P1 per run). With calibration, units 2–5 recalibrated by `calibrate_imu.py` |

**Statistics.** Medians use paired differences with BCa bootstrap 95 % confidence intervals (`scipy.stats.bootstrap(paired=True)`). With 25 pairs, an effect of d ≈ 0.6 is detectable. The development runs showed a much larger effect (max heading error 4.1 → 0.44°, 3.2 → 0.41°).

**Adoption.** R9-odo becomes the default only if L1, L1-slip, L3 and F pass. L5 and U are reported but not required. If L1 fails, the candidate stays switchable and off, and the result is documented. The other round-9 switches are evaluated separately and stay off unless they pass their own checks:
- `view_aware_turns`;
- the depth floor mask;
- the command-model fallback.

**Stop rule.** Stop at the first failed required check, document it, and do not tune on held-out results.

## Addendum A: candidate R9b (written after validation 1 stopped and before any R9b validation run)

**Validation 1** (frozen candidate ce39b72) stopped at its first failed required check:
- P4 failed on pair home_a seed 0: ATE 2.0 cm camera-only against 6.0 cm with the IMU, worse by more than 3 cm.
- **Cause, found on development data:** the encoders over-read distance by about 10 % (simulated rolling radius below the spec), and the fusion trusted them at a fixed ±3.6 %.

**R9b** is R9 plus these changes. All were found and fixed on development runs only (arena seed 0 and arena_textured seed 2), with the reasons in [heading-r9.md](heading-r9.md):
1. An online wheel-radius scale learned from vision. The encoder distance uncertainty stays wide (±10 %) until that scale is learned.
2. A loaded unit calibration is only monitored by in-run gyro-scale samples; it is replaced only when at least 5 samples disagree beyond 3σ. Each sample now carries vision's own uncertainty, not the fused one.
3. **Periodic zero-velocity stops:** the robot pauses 0.8 s when its last stand-still is older than 30 s, and the vision bias feedback is weak. On development data, an exploring robot never stood still, so the gyro offset random-walked (gyro alone: 6° in 390 s), and a strong vision feedback learned the map's own drift instead.
4. `calibrate_imu.py` gains a 1-turn coarse pass, needed for units with about 1 % sensitivity error.

Development results (arena seed 0 / arena_textured seed 2, 420 s builds with the coverage pass):

| | Max heading error | Map ATE |
|---|---|---|
| R9b | 1.27° / 0.57° | 0.58 / 1.07 cm |
| R9 | 0.75° / 5.23° | 1.15 / 3.03 cm |
| Camera only | 7.76° / 14.17° | 7.61 / 9.75 cm |

The R9b commit is named in [heading-r9.md](heading-r9.md) before its runs start: **6e89e60**.

**Fresh starts:** the same checks as above with seeds 5–9 and `--heading-offset 36`, so no start pose of validation 1 repeats. Worlds, durations and criteria are unchanged.

**Correction to L1-slip** (a mechanism fix, not a criterion change): `wheel_friction` had no physical effect, because Genesis takes the higher friction of the two surfaces. It is replaced by `floor_friction=0.45`, and the slip patches are now low-friction plates.

## Addendum B: candidate R9c (written before any R9c validation run)

**What happened to R9b.** Validation 2 of R9b (6e89e60) was not finished. The owner judged its forced pause every 30 s not smooth enough. Results so far:
- L1 passed every criterion on the 23 pairs that ran: max heading error median 1.17° against 13.15°, worst 2.1° against 53.8°; 0 against 87 drift events; paired ATE median −4.1 cm, worst pair +2.9 cm; lost fraction 0.3 % against 16.9 %.
- The slip, F, U and L3 checks of the IMU arm never ran.

**Owner decision (2026-10-05):** calibrate when the robot stops; if it has not stopped for 2 minutes, stop and recalibrate if necessary.

**R9c** is R9b with this stop policy:
- the offset is re-measured at every natural stop, after the wheels have stopped for 0.6 s plus 3 quiet frames;
- it is learned from vision on frames where at least 50 % of inliers are established landmarks;
- a recalibration stop (until the offset is measured, at most 2 s) happens only when there has been no update for 120 s **and** the offset uncertainty exceeds 0.02 dps;
- the straight-driving update stays off, because the simulated chassis skids.

**Development** used seeds 0–4 only, already spent in validation 1. Over 6 worlds the mean of the maximum heading error and mean ATE were:

| Stop policy | Max heading error | ATE |
|---|---|---|
| Stops every 30 s | 1.33° | 2.20 cm |
| Stops every 60 s | 1.51° | 3.16 cm |
| Every 120 s plus vision-learned offset (R9c's settings, without the "if necessary" condition) | 1.76° | 3.05 cm |
| Stops every 120 s | 2.51° | 3.37 cm |
| Camera only | 16.3° | 8.4 cm |

**Validation 3.** R9c is frozen as the commit that adds this addendum. All checks and criteria are those pre-registered above, on seeds 5–9 with `--heading-offset 36`. The IMU arm is run fresh. The camera-only arm is reused from validation 2: it is byte-identical across these commits (0 of 400 frames differ against the pre-round-9 code, arena seed 1, 40 s). Missing baseline runs are added.

## Addendum C: candidate R9e (written before any R9e validation run)

**Owner decision after validation 3:** fix the slow drift first.

**Diagnosis** (evaluation-only reconstruction of the simulated offset): in long non-stop runs the gyro offset was off by 0.009–0.018 dps between recalibrations. Three causes:
1. The vision offset learning was over-confident: correlated per-frame corrections were used as independent measurements.
2. The assumed offset random walk (0.002 dps/√s) was not realistic.
3. Each stop measured only about 0.1 s of samples (about 0.01 dps of noise).

**R9e** is R9c plus these changes:
- **Sensor model (simulation).** The new values come from a cited research step:
  - offset fixed per unit (±1 dps), plus 0.1 dps per power-on (assumed);
  - rate random walk 1.5e-4 dps/√s (Allan measurements of BMI160 and BMI055 as analogues);
  - **thermal warm-up**: datasheet tempco ±0.010 dps/°C times ΔT = 2–12 °C with τ = 150–900 s (assumed ranges);
  - the chip's temperature sensor (256 LSB/°C).
- **Estimator:**
  - temperature-compensated offset, offset(T) = b0 + k(T − T0), with a two-state Kalman filter and a tempco prior of 2× the typical value;
  - vision offset learning as one measurement per 15–90 s window, with honest uncertainty;
  - a recalibration stop holds until 1 s of still samples is collected (10 frames, at most 2.5 s); boot stand-still 2 s;
  - **owner-approved temperature trigger:** recalibrate when the chip temperature has changed by ≥ 1 °C since the last calibration and the offset is uncertain, at most every 20 s, in addition to the 2-minute rule.

**Development** (spent seeds 0–4, 6 worlds; mean of the maximum heading error and mean ATE): R9e 1.28° / 1.94 cm with 1–3 recalibration stops in 7 minutes; the 2-minute rule alone 2.11° / 4.26 cm.

**Validation 4.** All checks and criteria are unchanged (seeds 5–9, `--heading-offset 36`). The IMU arm is run fresh. The camera-only arm is reused, after a byte-identity check (it does not use the IMU).
