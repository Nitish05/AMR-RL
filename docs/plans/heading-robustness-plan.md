# Plan: heading robustness, round 9 (written 2026-10-04, work starts 2026-10-05)

**Scope.**
- Software only: no hardware or robot-spec changes (owner decision).
- Sensing for localisation is restricted to the onboard RGB camera, its calibration and the robot's own commanded motion (AGENTS.md).
- Every change ships as a switch, off by default, until it passes the validation below.

**Background.** [results/turn-drift.md](../results/turn-drift.md) (rounds 7–8 and the failed live validation of FAm) and [results/turn-drift-options.md](../results/turn-drift-options.md) (where the live error comes from).

## 1. What we now know (evidence, 30 live map builds plus research)

### Where the heading error comes from (baseline VSLAM)

| Source | Share of heading-error growth | Notes |
|---|---|---|
| Steady in-place turns | 52 % | mostly **unbiased per-turn scale error**: sd 2.3 % of the angle, independent from turn to turn (ICC ≈ 0) |
| Turns where the camera faces a surface within 0.5 m | — | over-rotate (+1.75 % mean, sd 4.3 %); they hold most of the worst jumps (the two worst runs: +16° within one survey turn) |
| Driving | 20 % | partly corrects turn error |
| Relocalisation jumps | — | 4.3° each |

**Tracking losses (16 % of run time).**
- 9 of 14 baseline losses are **false**. In each, a left survey sweep is followed by a sharp right arc within the 3 s window. The check sums the commanded turn with its sign, so the commands cancel. Vision is right (within 0.005 rad), but the arc's real yaw is only about 0.6 of the command and starts late.
- The other 5 losses are genuine wrong locks: 4 frozen, 1 linear.

### The wheels: a rate-limited response (no dead time)

Measured from ground truth; the first-order-plus-dead-time model fits worse.
- Each wheel ramps at 8 rad/s² when speeding up and 15 rad/s² when slowing down.
- Ratio of true to commanded motion:
  - in-place turns ≥ 0.35 rad/s: 0.83;
  - in-place turns 0.15–0.35 rad/s: 0.71–0.79;
  - yaw while driving: about 0.77;
  - forward speed: 0.89.
- Fit error with this model: 0.009–0.04 rad/s per frame.
- Per-turn noise of the model: sd 1.26 % of the angle, independent per turn.

**Implication:** the calibrated command model (sd about 1.3 %) is a **better per-turn rotation measurement than vision** (sd 2.3 %, and 4.3 % near surfaces). The two are independent, so fusing them properly should roughly halve the per-turn error. FAm failed because it used a hard gate and a crude, unmodelled command, not because fusion is wrong.

### Physics limits

- A differential drive cannot turn about the camera.
- Within about 0.25 m of a wall almost no floor is visible: about 2 % of image rows at 0.18 m.

### A depth-independent measurement exists

During an in-place turn the camera moves on a known circle of radius 0.125 m. The two-view epipolar geometry E(θ) therefore has a single unknown, the turn angle θ, and can be estimated from frame-to-frame matches **without any landmark depth**. References: Scaramuzza IJCV 2011, 1-point RANSAC; Choi & Kim IVC 2018, planar minimal solvers. This removes the wall-lifting bias by construction.

### Tools checked on this machine (Apple M5 Pro)

**Depth Anything V2 Small on MPS** (CPU in brackets):

| Input size | MPS | CPU |
|---|---|---|
| 252×336 | 10 ms | 47 ms |
| 364×476 | 18 ms | 95 ms |

**GTSAM 4.3.0** (BSD-2, wheel for macOS arm64):
- It has `PlanarProjectionFactor2`: a Pose2 robot, Point3 landmarks and an offset camera, which is exactly our setup.
- A 10-keyframe window with 446 landmarks solves in 29 ms with Levenberg–Marquardt; an iSAM2 update takes 3.5 ms.
- It throws `CheiralityException` on bad points; add only well-observed landmarks.

**Licences:**
- **Use:**
  - DA-V2 Small (Apache-2.0);
  - DA-V2 Metric-Hypersim-Small (Apache-2.0; the training data is CC-BY-SA);
  - OpenCV LSD (≥ 4.5.4) and FLD;
  - ELSED;
  - DeepLSD (MIT code and weights);
  - GTSAM;
  - symforce;
  - DRIVE (BSD-3, reference for the wheel model);
  - rpg_trajectory_evaluation (MIT).
- **Don't use:**
  - GPL: SVO, SVO Pro, REMODE, DSO, ORB-SLAM3, PAMPC, `evo`;
  - non-commercial: Depth Pro, UniDepth, DA-V2 Base/Large, SegFormer;
  - AGPL: pylsd;
  - unclear licence: Metric3D weights.

### Statistics

- 15 paired runs only detect large effects. Paired comparisons need about 34 pairs for an effect of d = 0.5 and 15 for d = 0.8.
- Comparing drift-event rates needs about 50 baseline events.
- **Lesson from FAm:** an offline benchmark of steady 0.45 rad/s turns from rest did not represent live turns, which run at 0.2–1.0 rad/s, are short, accelerating, and often follow driving.

## 2. Principles for this round

1. **Measure on live inputs first.** Record the camera frames of live runs and replay *identical* frames through estimator variants offline. That gives exact pairing, real command profiles and the highest statistical power. Closed-loop live A/B tests only confirm the result, and they are required only for changes that alter behaviour.
2. **Fuse, don't override.** Vision and the command model are independent measurements with known noise. Combine them by their uncertainty, with chi-square gating; no hard switches.
3. **One switch per idea,** default off, byte-identical when off (the existing `tests/test_turn_drift.py` baseline).
4. **Pre-register** thresholds, splits and stop rules before results are seen. Keep failures in the record.
5. **Pinned git worktrees for every benchmark run.** Scripts must be executable; check that runs actually start.

## 3. Workstreams, in order

### WS0. Evaluation infrastructure (day 1, first)

**0a. Record live frames.**
- Add a `--record-frames` option to `scripts/eval/build_map.py` and `scripts/eval/navigation.py`. It sets `RuntimeConfig.audit_images=True`. That option exists already (`evidence/images.py` writes every PNG in audit mode).
- Log `runtime.last_frame_record["raw_rgb_sha256"]` per frame in `sim/harness.py` `_score`, so each frame maps to its PNG.
- Disk cost is about 4,200 PNGs per 420 s build, roughly 0.5 GB per build. Keep the files under `work/`.

**0b. Live-replay bench `scripts/dev/live_replay.py`.**
- Inputs: a recorded run (frames, `cmd`, `gt`, `origin_world`, the map built in that run or a saved map).
- It replays `track(rgb, t, commanded=cmd)` frame by frame through VSLAM variants (`--variant name k=v`, the `turn_bench` conventions), starting from the run's own initialisation.
- Scoring uses the same metrics as `scripts/eval/turn_live_report.py`:
  - max heading error;
  - drift events per 100 in-place turns;
  - per-turn scale error;
  - losses, split into false and genuine (vision correct versus wrong at frame k−1, as in the liveparams analysis);
  - ATE.
- Stratify by motion class (turn start, steady turn, arc, straight), turn rate band, and facing distance from true geometry.
- **Limitation:** open loop. The estimator changes, but the robot's motion is the logged one. That is exact for estimator-only changes; behaviour changes need closed-loop runs.

**0c. Generalise `turn_live_report.py`.**
- Variant names from the command line instead of hard-coded `base` / `fam`.
- Paired bootstrap CIs with `scipy.stats.bootstrap(paired=True, method='BCa')`.
- A power note in the report.

**0d. Record the corpus (day 1, overnight).**
- Baseline map builds with `--record-frames`: arena, arena_textured, home_a, heldout_b, heldout_c × seeds 0–4. About 25 builds, roughly 3 h at 7 in parallel.
- Split: arena, arena_textured and home_a are **design**; heldout_b and heldout_c are **held out**. Also record 6 navigation runs as held-out data with goal-driven turns.

**0e.** Optional: `turn_capture --from-log`, which re-simulates logged command windows. Only needed for closed-loop behaviour checks.

**Tests:** frame manifest round-trip; replay determinism (same hash twice).

### WS1. Wheel response model and corrected consistency check (day 1–2)

Expected effect: removes about 9 of the 14 false losses, and every false angular loss.

**1a. New pure module `src/amr_rl/perception/wheel_model.py`** (no sim imports): `WheelResponse` ("Model C").
- Command → wheel targets through `robot/spec` geometry (track, wheel radius; reuse `control/contract.body_to_wheels`).
- Each wheel is rate-limited: speeding up 8 rad/s², slowing down 15 rad/s².
- Body motion back from the wheels, with gains:
  - in-place yaw 0.83 (≥ 0.35 rad/s), 0.71–0.79 below;
  - yaw while driving 0.77;
  - forward speed 0.89, down to 0.83 at curvature above 4 /m.
- Output: modelled (Δx, Δy, Δθ) per frame, plus σ per frame: in place `sqrt(0.010² + (0.013·|Δθ|)²)` rad per turn segment; wider for arcs.
- These parameters are **engineered and fitted from simulation ground truth offline**. Label them so in code and docs. At runtime they are constants plus the online adaptation of WS2.

**1b. `PlanarVSLAM._motion_consistent` (vslam.py ~928–958), switch `VSLAMConfig.wheel_model = "consistency"`.**
- Step the model once per `track()`. Log `(t, pose, v·dt, w·dt, Δθ_mod, Δxy_mod)`, keeping indices 2–3 so `_arm_dead_reckoning` is unchanged.
- Drop-in version: the current thresholds, compared against modelled motion instead of the raw command. The angular test becomes **sign-robust**: `|Δθ_vis − Δθ_mod| > 0.30 + 0.2·Σ|Δθ_mod|`.
- Optional tightened version (from the liveparams analysis):
  - freeze when the modelled 1.2 s chord ≥ 0.10 m and the vision chord < 0.5× it;
  - xy when the error exceeds 0.16 + 0.15·path;
  - in-place translation when the vision chord exceeds 0.10 m.
  - It caught 23 of 24 genuine wrong locks a median 0.6 s earlier, and no firing with perfect vision.

**1c. Tests** (`tests/test_perception_units.py`):
- `WheelResponse` step and reversal responses match the measured shapes:
  - from rest: 0.27 / 0.76 / 0.83;
  - reversal: −0.36 / 0.36 / 0.79.
- The live regression (sweep +0.45 for 2 s, then a (0.2, −0.87) arc with lagged, slipped truth) stays consistent.
- A genuine rotational lock (vision +1 rad, command 0) is still flagged.
- The existing freeze tests (lines 338, 363, 389) still pass.

**1d. Validation.**
- Live replay on the design corpus, then held out.
- Pass:
  - false losses −70 % or better;
  - no increase in heading error;
  - genuine locks still caught: on frames with injected wrong locks, recall ≥ the current check.
- Then 15 paired live L1 builds.

### WS2. Calibrated soft rotation prior, replacing the hard gate (days 2–3)

Expected effect: per-turn scale sd 2.3 % → about 1.2–1.5 %, which addresses the largest error share.

**2a. Prior mean and variance from `WheelResponse`.**
- Gain k adapted online by RLS per band (in place < 0.35, in place ≥ 0.35, driving).
- Forgetting factor 0.99. Update only from vision-trusted frames: high inliers, low degeneracy (2b), NIS gate passed.
- Freeze k while |ω_cmd| is small, where it is unobservable.
- Reference: DRIVE (BSD-3) Bayesian slip regression.

**2b. Vision degeneracy score.**
- Fraction of current inliers that are unvalidated floor-lifted points closer than 0.6 m.
- The lever-arm bias factor `1 + 0.125/d` of those inliers.
- Optionally a vision-only `H[2,2]`.

**2c. Fusion in `optimize`.**
- The motion prior becomes the calibrated model with σ_rot from 1a. It is whitened and Huber-robust as today, but **tight** (per-frame σ about 0.13 % of Δθ plus 0.3 mrad).
- Weighting so that:
  - far, healthy views let vision dominate where its information is high;
  - near-surface or degenerate views shift weight to the model.
- **No hard gate:** `turn_heading_gate` stays off.

**2d. NIS check.**
- Per-frame normalised innovation of the visual Δθ against the model, χ²₁(0.99) = 6.63.
- Mark "degraded" only after 3 of 5 frames fail. Do not drop the frame; inflate its noise.

**Switches:** `VSLAMConfig.motion_prior_mode = "legacy" | "calibrated"`, `prior_deg_lambda`.

**2e. Tests.**
- Synthetic `tests/turn_scene.py` with a lag hook (a `WheelResponse`-driven truth).
- Near-box scene (0.30 m): the error improves.
- Open floor: no worse than 0.2°.
- Turn starts from rest and reversals: no bias (FAm's failure case).

**2f. Validation.**
- Live replay, design then held out. Pass:
  - per-turn scale sd −30 % or better;
  - drift events per 100 turns −40 % or better;
  - no stratum (start, reversal, arc, rate band, far view) worse by more than 10 %;
  - losses not up.
- Then live L1 + L3 paired runs. **34 pairs** for L1, or event-rate criteria with ≥ 50 baseline events.

### WS3. Depth-independent turn rotation, planar epipolar estimator (days 3–4)

Expected effect: removes the near-surface over-rotation by construction.

**3a. `src/amr_rl/perception/turn_epipolar.py`.**
- Inputs: matched normalised bearings between the current frame and the turn's reference frame, i.e. the last keyframe or the turn start.
- Camera motion for base rotation θ about the axis: R(θ) plus translation t(θ) on the 0.125 m lever-arm circle, so E(θ) = [t(θ)]× R(θ).
- 1-point RANSAC: each correspondence gives a closed-form or 1-D root for θ.
- Then a robust 1-D minimisation of Sampson error over the inliers.
- Output: θ̂, σ_θ (from curvature), inlier count.
- About 60–100 lines of numpy. ORB matches come from `features` plus `self.matcher` (BFMatcher, Hamming).

**3b. Use during in-place turns** (`_turn` state):
- as an extra measurement on the relative heading in the fused estimate (WS2 framework), or
- as a robust check: flag map-based Δθ that disagrees with the epipolar Δθ beyond 3σ.

Start with frame-to-keyframe; frame-to-frame accumulates random-walk error. Extend to a 2-DoF planar version (θ plus arc radius) for arcs later.

**3c. Tests.**
- Synthetic two-view scenes with near walls at 0.18–0.4 m and random depths: θ̂ unbiased within 0.1°, where the IPM-lifted map estimate is biased.
- Degenerate cases (too few matches, pure noise) return "no estimate".

**3d. Validation.**
- Live replay. Pass: per-turn bias in the facing < 0.5 m stratum within ±0.5 % (baseline +1.75 %, sd 4.3 %), with overall drift events down.

### WS4. Depth-model floor mask for floor lifting (day 4)

Expected effect: stops wrong points entering the map at 0.3–0.6 m. It complements WS3, which protects the pose but not the map.

**4a. `perception/near_depth.py`:**
- Device option (MPS, at 252×336, 10 ms).
- `infer(rgb)` with a per-frame cache, shared with the guard.
- `floor_mask(rgb) -> not_floor`, from a fit in **disparity space**:
  - RANSAC on the bottom floor band, with inlier threshold about 5 % and at least 300 inliers;
  - EMA on (a, b) with α = 0.2;
  - hold the last fit and inflate its variance when too little floor is visible;
  - not-floor when model depth < 0.85 × the IPM depth.

**4b. VSLAM hook.**
- In `_maybe_keyframe` after `world, ok = self._ipm(...)` (around vslam.py:1019): `ok &= ~not_floor[...]`.
- `rgb` is passed lazily, so inference runs only when a keyframe is actually made.
- Attach like `slam.place`: `slam.depth = runtime depth detector`.
- Switches: `VSLAMConfig.depth_floor_mask = "off" | "turn" | "always"`; `RuntimeConfig.depth_device = "cpu" | "mps" | "auto"`. CPU stays deterministic for tests and benches.

**4c. Tests.**
- The synthetic disparity fit recovers the floor.
- Box-face pixels are masked (extend `tests/test_mapping_navigation.py:352`).
- The VSLAM stub-mask test on `turn_scene` with a box at 0.30 m.

**4d. Validation.**
- Live replay. Pass: fraction of not-floor floor-lifted landmarks (scored with rendered depth where available) −70 %, and no loss of tracking.

### WS5. Perception-aware turning (day 5, behaviour; closed-loop only)

Expected effect is modest: about 10–20 % of turn-error growth and 21–37 % of > 2° turn windows, while touching 2–7 % of turning.

**5a. New `src/amr_rl/navigation/view_check.py`.**
- `facing_distances(grid, planner, pose, cam_x, headings)` ray-casts from the camera point on the robot's own occupancy grid, reusing `Planner.obstacle_clearance_xy`.
- Report UNKNOWN cells separately.
- Threshold X = 0.6 m on the grid (true-geometry equivalent 0.5 m; grid recall 68 % at 0.5, 88 % at 0.6).

**5b. Hooks**, behind switch `RuntimeConfig.view_aware_turns`:
- **Explore look-around** (`activities.py` ~207–238): drop or shorten the sweep side that faces < X; skip the panorama when more than 30 % of it faces < X.
- **Navigator rotate branch and final align** (`navigator.py` ~106–147, `_turn_or_back_off` ~200–216): when the short-way sector faces < X, back off 0.2–0.3 m within the existing back-off budget, then turn. Turn the long way only when |err| is close to π.
- **Initial survey:** the grid is empty at t = 0, so use the depth guard's live points. This needs `DepthGuard` to keep `last_world`.

**5c. Tests.**
- New `tests/test_view_check.py`:
  - a wall 0.4 m to one side shortens or reverses the sweep;
  - open floor leaves it unchanged;
  - the navigator backs off before turning.

**5d. Validation.** Closed-loop paired live runs only, at least 34 pairs. Pass:
- drift events down;
- map coverage and navigation arrivals not worse (L2 / L3).

### WS6. Turn-scoped delayed landmarks (only if WS2–WS4 leave a gap)

- Inverse-depth candidate store (`TurnCandidates`) using the SVO depth-filter equations.
  - Write it ourselves from the papers; SVO code is GPL.
  - Start values: a = b = 10; σ² = z_range²/36; converged when σ < z_range/200; drop when a/(a+b) < 0.1.
- Candidates are promoted to landmarks only after convergence.
- During a turn, track only pre-turn landmarks.
- Risk: fresh maps are built entirely during turns. Needs a fallback.

### WS7. Later: global consistency

- **GTSAM sliding-window and global BA** with `PlanarProjectionFactor2` (Pose2 + Point3 + offset camera), robust kernels and iSAM2. Keep the hand-written SE(2) graph as a fallback and cross-check. Solves the pose-graph-only limitation of the loop-closure work.
- **Absolute heading factors from floor–wall junction lines.**
  - Detection: OpenCV LSD below the horizon.
  - Use: lift the lines to the floor (exact for junctions), take directions modulo 90°, add a unary heading factor to `perception/pose_graph.py` with σ 2–3° and a Cauchy kernel.
  - Gates: segments longer than 0.4 m and at least 2 agreeing segments.
- Both are separate rounds with their own pre-registration.

## 4. Validation protocol (applies to WS1–WS5)

1. **Pre-registration:** `docs/results/heading-r9-prereg.md`, committed before any candidate is replayed on held-out data. It fixes metrics, strata, thresholds and stop rules.
2. **Design.** Develop on live replay of the design corpus (arena, arena_textured, home_a), even seeds for tuning and odd seeds to confirm.
3. **Held out.** Live replay of heldout_b and heldout_c, run once.
4. **Live closed loop.** Paired runs with ≥ 34 pairs, or event-rate criteria with ≥ 50 baseline events:
   - L1, coverage builds;
   - L3, navigation;
   - L5, learning.

   Stop at the first failure.
5. **Adoption.** Make a change default only after it passes held-out replay **and** live. Update `README`, the capability ledger and `HANDOFF.md`.
6. **Metrics in every report.** Every metric above, with the confidence interval of the paired difference, per stratum. A change must not be worse than baseline by more than the stated margin in any stratum: turn start, reversal, steady turn by rate band, arc, straight, near versus far facing.

## 5. Schedule (estimate)

| Day | Work |
|---|---|
| 1 | WS0a–c (frame recording, live replay, report); start corpus recording (WS0d) overnight; WS1a–c |
| 2 | WS1 validation (replay, then live L1 pairs); WS2a–b |
| 3 | WS2c–f; WS3a–c prototype on replay |
| 4 | WS3 validation; WS4 |
| 5 | Pre-registered held-out replay of the best combination; live validation; WS5 if time allows |
| later | WS6, WS7 as separate rounds |

## 6. Risks

- **Live-replay realism.** Frames are real, but the motion is fixed to the logged one. Estimator changes that would alter behaviour (degraded flags → activity decisions) still need closed-loop runs.
- **Model parameters are simulator-specific** (fitted from simulator ground truth). The online k adaptation (WS2a) is what generalises; the fixed acceleration limits come from the robot spec. Label them as engineered.
- **Tight priors can drag correct vision** (FAm's failure). Fusion by measured noise and per-stratum non-inferiority checks guard against this.
- **MPS non-determinism.** Keep CPU for tests and benches; MPS only at runtime.
- **Disk:** about 0.5 GB per recorded build. Prune after analysis; never commit.
