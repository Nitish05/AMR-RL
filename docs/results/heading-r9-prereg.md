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
