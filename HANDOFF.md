# Handoff (2026-10-11): where AMR-RL stands

Project rules are in `AGENTS.md` (read it first). Results with denominators are in
`README.md`, `docs/NAVIGATION_RESULTS.md`, `docs/LEARNING_RESULTS.md`,
`docs/CAPABILITY_LEDGER.md` and `docs/results/`. The history of every round is below,
newest first.

## State of the repository

- **Branch `main`**, local and ahead of `origin/main` (`7b3d425`); pushing is the owner's
  decision.
- **Default robot:** onboard RGB camera + IMU (ST LSM6DSOX) + wheel encoders (Pololu 4754),
  `RuntimeConfig.odometry = "imu_encoders"`. Re-validated on untouched starts on 2026-10-11
  (validation 5, R9g `aa143e3`): every required check passed except L1's P4 on one pair
  (`docs/results/heading-r9.md`). On by default:
  - loop closure, with the gyro/encoder consistency guard;
  - the blocked-robot check, which since 2026-10-12 keeps the pose, backs off and chooses
    again (`odo_blocked_mode="keep_pose"`; validation pending, addendum F);
  - gyro offset recalibration at natural stops, plus a forced stop after 2 min without a
    measurement or a 1 °C temperature change, and since 2026-10-12 a first stop at 0.5 °C
    until the temperature coefficient is measured (validation pending, addendum F).
- **Camera-only robot:** `--odometry command` (`RuntimeConfig.odometry = "command"`).
- **Each IMU unit needs `scripts/calibrate_imu.py` once** (`configs/calibration/imu.yaml`).
  Unit 4 uncalibrated reached 22.5° of heading error.

**Running things:**
- Setup: `bash scripts/amr.sh setup` (installs against `requirements/lock-macos-arm64-py3.12.txt`
  on macOS arm64). Models: `fetch-place-model`, `fetch-depth-model`, `fetch-detector-model`
  (the open-vocabulary detector also needs the `perception` extra).
- Tests: `scripts/amr.sh test` (non-simulation suite, ruff, UI checks; also the CI job in
  `.github/workflows/ci.yml`) and `scripts/amr.sh test-sim`.
- Evaluations: `scripts/amr.sh batch <jobs.txt> --jobs 8 --cwd work/wt/<pinned worktree>`.
  - The batch runs under `caffeinate -dims`: with the display asleep, Genesis' renderer
    finds no screen.
  - About 2–2.5 GB per simulation: at most 8 in parallel on this 48 GB machine.
- Evidence goes in fresh `work/evidence/` directories, from pinned worktrees under `work/wt/`
  (retention rule in `docs/PROVENANCE.md`).

## Open items, in order

1. **Blocked or spinning wheels: decided and implemented (2026-10-12).** The pose is kept,
   the robot backs off 10 cm and chooses again (`VSLAMConfig.odo_blocked_mode="keep_pose"`,
   `RuntimeConfig.blocked_*`); the 2026-10-11 behaviour is `odo_blocked_mode="lost"`. Live
   validation under addendum F is the open step.
2. **First recalibration stop: decided and implemented (2026-10-12).** A 0.5 °C change
   triggers the first stop until the tempco has been measured
   (`RuntimeConfig.imu_zupt_temp_first_c`). Validated with item 1 under addendum F.
3. **Relocalisation range:** beyond 0.5 m from any keyframe, 2/12 on the IMU robot; no way
   to relocalise while stationary once recovery is used up; no autonomous retry after
   `recovery_exhausted`.
4. **Learning:** plain arena restart 9/9, reversal 10/10 (round-9 suite, colour detector).
   Textured arena 6/6 and 6/6 only with the open-vocabulary detector (2/4 and 2/6 with the
   colour detector), which is therefore the default since 2026-10-12 when its model is cached
   (`RuntimeConfig.detector`); the plain-arena suite is being re-run under it. The character
   store (traits with bounds, provenance, change log, reset levels; `behavior/character.py`)
   exists, and all six traits are mapped (`character.effective`): the four temperament traits
   onto the chooser, navigator and probe budgets, the two learning-style traits onto the
   memory's evidence weighting and curiosity bonus, behind `RuntimeConfig.character_active`,
   OFF until the pre-registered admission gates pass (`docs/results/character-gate-prereg.md`,
   `scripts/eval/character_gate.py`; the 27-run temperament batch and three 9-run
   learning-style batches are queued after the detector suite). The face's style (tempo,
   blink) comes from one latent state (the logged need) rescaled by the tempo trait, and a
   15 s boot cue names the session, remembered objects and any character change
   (`docs/EXPRESSION.md`, "Style"). At `close()` the character consolidates by engineered
   rules on the session's logged statistics (≤ 0.05 per trait per session, no mean
   reversion; `character.EVOLVES`); approach tempo and persistence have no rule yet. The
   speaker exists as a logged event bus (`expression/sound.py`, eight cues + transit layer,
   `render_wav` for evaluation); no listener study yet. Open: a duplicate identity
   of a disliked fixture can be chosen before it is merged, and aversion learned in one
   visual context does not carry to another (3 second red panels in 99 round-9 phases);
   16 % of interactions with the tall cylinder still end in a tracking loss.
5. **CI flake:** `tests/test_turn_drift.py::test_turn_closure_removes_the_heading_error_a_near_box_causes`
   failed in 2 of 8 Linux CI runs with no VSLAM change (synthetic ORB numerics); it now runs
   only on macOS arm64 like the turn baseline. Open: make the synthetic closure test platform-stable.
6. **Unevaluated switches:** WS1 `VSLAMConfig.wheel_model`, WS2 `odometry="command_model"`,
   WS4 `VSLAMConfig.depth_floor_mask`, WS5 `RuntimeConfig.view_aware_turns`.
   `floor_feature_share` is not needed with the IMU (textured builds lost 0 frames without
   it); keep it for the camera-only robot or retire it. Retiring rejected switches
   (`static_hypothesis_when_moving`, `command_consistency_score`, `reloc_local_first`,
   `suppress_free_above_bases`, `pnp`, `--legacy-loop-candidates`) needs the owner's agreement.
7. **Map quality:** parallax-floor free evidence inside obstacles (up to 26 deep
   false-free cells in one of 35 round-9 IMU builds).
8. **Statistical learning evaluation:** randomised fixture placements, 10+ seeds, baselines
   on every seed.
9. **Simulation realism:** camera degradation (blur, exposure), a motor/armature model,
   global bundle adjustment; real-time operation.
10. **Repository:** licence (owner), pushing (owner), removing the merged worktrees
   `work/wt/r9f-logcheck` (its staged changes are saved as a patch in
   `work/evidence/r9g-lcreplay-20261011/`), `r9g-ce5cf69` and `r9g-dev-dd638a4`.

## History, newest first

### Round 9 (2026-10-05): IMU + wheel encoders

- **Owner decisions:**
  - Add an IMU and wheel encoders that can be bought (ST LSM6DSOX on Adafruit 4438; Pololu 4754 encoders) and keep wheel slip in mind.
  - Implement the round-9 plan.
  - Calibrate the gyro offset at natural stops; if the robot has not stopped for 2 minutes, stop and recalibrate if necessary.
- **Where things are:**
  - Results: `docs/results/heading-r9.md`.
  - Pre-registration and its addenda (A–E by 2026-10-11): `docs/results/heading-r9-prereg.md`.
  - Parts: `docs/ROBOT.md`.
- **Switch:** `RuntimeConfig.odometry = "imu_encoders"` (at the time the default was `"command"`, byte-identical to before; `"imu_encoders"` became the default on 2026-10-06). It needs `configs/calibration/imu.yaml`, produced by `scripts/calibrate_imu.py`.
- **Validation 3 (R9c, 890646c, plus the encoder-dropout fix d237559):**
  - Large gains in heading, drift, losses, slip, navigation and sensor faults.
  - P4 fails on 2 of 25 pairs: the camera-only arm was lost for long stretches there, and the IMU arm's heading drifts slowly in long non-stop runs. Adoption was then the owner's decision (made 2026-10-06, after validation 4).
- **Plan items not yet evaluated in closed loop:**
  - WS1 `VSLAMConfig.wheel_model`;
  - WS2 `odometry="command_model"`;
  - WS4 `VSLAMConfig.depth_floor_mask` plus `RuntimeConfig.depth_device`;
  - WS5 `RuntimeConfig.view_aware_turns`.
  - WS0 live replay (`scripts/dev/live_replay.py`, `--record-frames`) is ready for them.
- **Machine limits:** about 2–2.5 GB per simulation; run at most 8 in parallel on this 48 GB machine. More triggered memory-pressure kills.

#### Round 9 outcome (2026-10-06)

- IMU + encoder odometry is the **default** (R9e f0748fc plus 811adbc). Validation 4 passed every required check. `docs/results/heading-r9.md` has the full story, including three failed or stopped validations.
- **Recalibration policy** (owner decisions): the gyro offset is measured at natural stops; if needed, the robot stops after 2 minutes without a measurement or after a 1 °C chip temperature change.
- **Each IMU unit needs `scripts/calibrate_imu.py` once.** Unit 4 uncalibrated reached 19° of heading error.
- **Next** (2026-10-06): a loop-closure heading guard using the gyro (done 2026-10-11); closed-loop evaluation of WS1, WS2, WS4 and WS5.

#### Round 9 re-validation (2026-10-11)

- R9g (aa143e3) = the default plus the 2026-10-11 fixes, the loop-closure odometry guard and the blocked-robot check, under corrected caster physics. Validation 5 on untouched starts (seeds 10–14, offset 18°): `docs/results/heading-r9.md`, evidence `work/evidence/r9g-val-20261011/`.
- **L1 fails P4 on one pair** (heldout_c s11, +5.0 cm; the camera-only run ended at 174 s). Every other required check passes. The claims are scoped accordingly; whether R9g stays the default is the owner's call (it is the code default).
- **Next candidate:** when the blocked check fires and vision agrees with the encoders, keep the pose and back off, instead of declaring it lost (about 26 s per event now). Develop on spent seeds; pre-register before validating.
- **For the next pre-registration:** compare P4 over the span both runs lasted (camera-only builds often end at `build_map.py`'s 6-re-enable limit).
- **Long runs: keep the Mac's display awake** (`caffeinate -dims`). When the display slept, Genesis' renderer found no screen and 12 jobs failed.

#### Round 9 learning, relocalisation and hygiene (2026-10-11, after validation 5)

- **Learning on the IMU default** (`docs/LEARNING_RESULTS.md` round 9; `docs/results/r9-followup.md` §9):
  - plain arena, round-6 suite: restart 9/9, reversal 10/10, 0 false relocalisations;
  - L5 (pre-registered, reported): drift 0.08 vs 33.9 events per 100 turns; the camera-only
    arm's own map was too short to relocalise on (L5x on the IMU map: post hoc);
  - textured arena: restart 6/6 and reversal 6/6 with the open-vocabulary detector, 2/4 and 2/6
    with the colour detector (`docs/results/textured-worlds.md`);
  - frustration renewal: the round-6 idle trap no longer occurs; the switch stays off.
- **Relocalisation range and coverage pass** (`docs/results/relocalisation.md` round 9): IMU
  maps 22/24 within 0.5 m of a keyframe, 0 wrong; the coverage pass no longer harms the map
  but does not extend the range (stays off); explained-share floor not built.
- **Navigation, round-9 section** in `docs/NAVIGATION_RESULTS.md`.
- **Hygiene:** CI workflow (not yet pushed), report labels the sensors used, camera-only labels
  on pre-round-9 benchmarks, scorer fix for nudges unfinished at phase end.

### Round 8 (2026-10-04): near-surface turn tail

- The tail is wrong depth (wall/box faces lifted to the floor, times the lever arm), not edge aliasing.
- Finalist **FA** passed every held-out criterion (`docs/results/turn-drift.md`, round 8). It is a combination of switches, all off by default:
  - `turn_heading_gate` 0.65° / 0.3;
  - `turn_gate_keyframes`;
  - `turn_max_prediction_s` 4;
  - `turn_gate_settle` 3;
  - `c9_parallax_check="turn"`;
  - `floor_validation="turn"`;
  - `turn_closure="rotate"`.
- It **failed the slip sweep on saved maps**, probably because the slip factor is learned slowly from a 0.85 start.
- Next: robust slip estimation, then a new held-out capture, a new slip sweep, and live L1–L5.
- New runtime switch `respect_degraded_heading` (off).
- Slip fix: FAm = FA + `turn_slip_mode="median"`, `turn_slip_min_inliers=100`.
  - It passes most held-out criteria (e360 p90 about 3°).
  - It fails two: confidently-wrong frames on fresh maps are 1.5 % (limit 0.5 %), and on saved maps at slip 0.75 the error is 7.7° (near-surface bin 16.6°).
  - Not adopted.
  - The seed-3 held-out and slip captures are now used up.
- **Live validation of FAm failed.** In L1 (15 paired coverage builds) heading and maps were worse: drift events 13.5 vs 5.1 per 100 turns, ATE +9 cm median.
  - Cause: the benchmark was not representative. Live turns are 0.2–1.0 rad/s, short and accelerating, and the gate overrides vision at turn starts.
  - Next: build the benchmark from logged live command sequences, settle at turn start, estimate slip per rate band.
  - FAm stays off; the `fam` preset is kept for comparison.

### Round 7 (2026-10-04): heading drift in in-place turns

- Research: code map, 354-run analysis, literature.
- Benchmark: `scripts/dev/turn_capture.py`, `turn_bench.py`.
- Eight candidates as `VSLAMConfig` switches, all off (`docs/results/turn-drift.md`, pre-registration `turn-drift-prereg.md`).
- None passed held out:
  - floor validation (C2a) fixes the typical near-object drift but not the tail;
  - the tail is heading locks near featureless walls (skirting edge);
  - the commanded-turn gate (C8) removes the wrong headings but loses tracking.
- Next: a degeneracy-aware gate that keeps the lock (spread of matches along the turn, slip-scaled dead reckoning, longer prediction while turning).
- Harness now logs the command per frame.

### Round 6 (2026-10-03): the five open items worked in order

1. **Safety** (`7ec324e`): fixed (0 contacts in 12 navigation runs).
2. **Loop closure:** uncertainty slots, revisits and covisibility edges were
   evaluated and not adopted. The pose-graph-only back-end ignores re-anchoring, so
   global BA is next (`docs/results/loop-closure.md`).
3. **Relocalisation range:** turn pooling, threshold 50 and the coverage pass were
   evaluated; none adopted. Next: an "explained share" floor and heading drift in
   in-place turns (`docs/results/relocalisation.md`).
4. **Learning suite re-run** (`docs/LEARNING_RESULTS.md`, round 6): 5/5 starts
   relocalised; restart 10/10; reversal 7/10 (re-scored 2026-10-11); inert does not settle; 0 unintended
   contacts.
5. **Perception swap:**
   - VLM pilot: Qwen3-VL-2B held out 0.935, but red misses 4 %, so not in the
     runtime (`docs/results/vlm-outcome-pilot.md`).
   - Textured worlds and the open-vocabulary detector (OmDet-Turbo): held out
     recall 1.0 / precision 0.97 / state 0.982 (`docs/results/textured-worlds.md`).
   - Measured error presets in the learner testbed.
   - Open: the VSLAM loses tracking in textured worlds unless
     `floor_feature_share=0.7` (off by default; needs navigation validation).
     Runtime A/B n = 2.

Models (all pinned, local cache): `scripts/amr.sh fetch-detector-model`, `fetch-vlm-model`;
the `perception` extra adds `timm`.

### Round 5 (2026-10-03): loop closure

Done (`73da7e2`, `211d626`, `c9a7694` and the docs commit; `docs/results/loop-closure.md`):
* **Research choice:**
  - MegaLoc place descriptors (2025, MIT; pinned, local cache,
    `scripts/amr.sh fetch-place-model`). In our arena they reach recall@1 0.91,
    against 0.14 for ORB bag-of-words.
  - Verification: planar two-point RANSAC against the candidate keyframe's own
    landmarks, plus two-keyframe consistency.
  - Back-end: SE(2) pose graph with GNC-Cauchy loop edges, and a ROVER (2026) style
    distortion check.
  - Correction of keyframes, landmarks, occupancy (evidence journal replay) and
    entities.
* **Validation:**
  - With the rules fixed beforehand: 20 closures, 0 false.
  - Big map-error drops where an early region is revisited; neutral otherwise.
  - Navigation: own-map goals 35/44 → 38/46, drift misses 3 → 1.
  - The first version without confirmation had 5 false closures in 44; it is kept in
    the record.
* **Map schema v2** (anchors + keyframe descriptors). Retrieval-first relocalisation
  gave a small gain (0.15–0.3 m bin 8/11 → 10/11).

Open, in order:
1. **Depth-guard creep contacted the evaluation box** in one navigation run (home_a
   s1, t = 425 s). FIXED in round 6 (`7ec324e`): unseen faces, loop-closure replay erasing guard marks, rotate recovery.
2. **Loop closure** only helps when the robot revisits a region mapped before the
   drift. Round 6 evaluated uncertainty slots, active revisits and covisibility edges
   (live A/B + paired shadow evaluation, `docs/results/loop-closure.md`): not adopted,
   defaults stay round 5. Correction helps drifted maps and hurts accurate ones;
   the pose-graph-only back-end misses re-anchoring, so global BA is the next step.
3. **Relocalisation** more than ~0.3 m from keyframes still mostly fails (one start
   per map). Round 6 (`docs/results/relocalisation.md`): pooling, threshold 50 and
   the coverage pass evaluated, none adopted. Next: an "explained share" floor
   against wall-landmark aliases, and fixing heading drift in in-place turns.
4. **The learning suite** (restart, reversal, inert, noisy) has not been re-run on
   round-4/5 code.
5. **Perception swap** in simulation.

### Round 4 (2026-10-03): open items worked through

Done (`7c301c5` .. `0bf21dc` and this commit; results in `docs/results/`):
* **Relocalisation:** planar two-point RANSAC with verification, view-change
  confirmation and probation (`vslam.py`, `reloc_method="planar2pt"`; legacy as
  `"pnp"`). The round-3 seed-4 start was wrong (0.82 m), so that seed's results are
  invalid. Probe tools: `scripts/dev/reloc_capture.py`, `scripts/dev/reloc_probe.py`.
* **Operator turns** reach the VSLAM as commanded motion. `build_map.py` and
  `slam_bench.py` get a counted operator turn after recovery gives up.
* **Avoid give-up:** after 3 failed retreats it holds with a backoff. A retreat must
  actually leave the radius to count as success.
* **Navigation:** the planner/corridor shared samples are back at the original
  spacing. Same-machine A/B shows no regression.
* **Pre-registered policy comparison:** the decision rule was met against random,
  nearest and fixed (`docs/results/policy-comparison.md`).
* **Frontier panoramas:** measured and left off (they drifted the map).

Open, in order:
1. **Relocalisation range.** Positions more than ~0.3 m from any keyframe mostly do
   not relocalise; one learning start per map fails. Next step: keyframe retrieval
   (map schema v2).
2. **Tracking drift.** Map builds vary from 2.5 to 12 cm ATE, and the most common
   missed navigation goal is "says arrived, 0.15 m+ off". There is no loop closure.
3. **The rest of the learning suite** (restart, reversal, inert, noisy) has not been
   re-run on round-4 code.
4. **The agreed plan:** swap in real perception inside the simulation (a detector,
   and a VLM for outcomes).

### First local session (macOS arm64), 2026-10-02

Done (commits `c416ccd` .. this one):
* Learner testbed committed and run (20 seeds x 4 noise levels). Main failure at high
  noise was position noise read as motion; `classify` now needs a displacement beyond
  the measured scatter (`0b8f9ea`). Results: `docs/results/learner-bench.md`.
* Full Genesis env (`.venv`) set up; `test` and `test-sim` green.
* Two real bugs fixed: SciPy cKDTree segfault under Genesis' flush-to-zero
  (`07f8ae0`), and planner/corridor sampling mismatch that froze the robot next to
  the aversive fixture (`5714aac`).
* Round-3 learning suite, 5 start poses: `docs/LEARNING_RESULTS.md` (round 3).

Open, in order:
1. Start poses 2 and 3 never relocalise against `map-arena-20261002-mac-s1`;
   2 of 3 mapping seeds lost tracking for good on this machine.
2. Policy vs baselines on total valence is not shown (baselines repeat the nearby
   rewarding fixture; learned idles when satiated). Compare at equal interaction budgets.
3. Forced `avoid` has no give-up rule if retreat is truly impossible.
4. Navigation evaluation not re-run after `5714aac`.
5. Then the agreed plan: perception swap in simulation (outcome label confusion is the
   largest remaining testbed error; keeping 6 pre-action frames is measured but only
   matters with a noisier detector).

### Original cloud handoff (2026-10-02)

Its repository state (last commit `3577fe5`, an uncommitted learner testbed) is long superseded; the task and plan are kept for the record.

#### Immediate task (original, now done): learner testbed under perception noise

Goal (agreed with the owner): the learning layer must survive the errors real
perception (an open-vocabulary detector + a VLM describing outcomes) will make,
before perception is swapped. Nothing learned in simulation transfers to a real
home; what transfers is the mechanism, so the mechanism is what gets tested.

The testbed runs the REAL `ExperienceMemory`, `EntityTracker`, `classify` and
`ActivityChooser` with no physics: a point robot, 8 objects with randomly assigned
hidden rules (1 reliable + 1 50 %-reliable rewarding, 1 aversive, 1 mover, the rest
inert, one pair of look-alike twins), and a perception error model with presets
`clean / low / medium / high` (position noise and outliers, appearance jitter and
bad views → identity splits, missed detections, per-frame state misreads,
consistently mislabelled outcomes, hallucinated changes). Experiments: `standard`
(learned vs random / nearest / fixed), `reversal`, `inert`, `restart`.

```bash
bash scripts/amr.sh bench-setup          # .venv-bench: numpy, scipy, opencv-python-headless, pyyaml, pytest, ruff
bash scripts/amr.sh bench-test           # testbed + learning + perception unit tests
bash scripts/amr.sh learner-bench --seeds 3 --quick   # smoke test
bash scripts/amr.sh learner-bench        # full grid, 20 seeds per cell
```

What to look for: how fast "true valence per attempt", "top belief is truly best",
aversive repeats, reversal adaptation and restart choice degrade from `clean` to
`high`; how many duplicate/impure identities appear. Then harden the learner where it
breaks (likely: outcome-label fragmentation, identity splits, noise-triggered change
detection), with each fix measured on the same grid.

#### Agreed plan after that

1. Swap perception inside simulation first (textured objects; open-vocabulary detector
   + re-identification; VLM before/after outcome descriptions mapped to categories).
2. Appearance-based priors so experience transfers between similar objects (the one
   part that can be trained in sim and fine-tuned on site).
3. Real robot: empty memory in one room, value from a real task or operator feedback.

Other open items:
* Depth guard: ~23 % of the cells it marks are open floor → write only cells near the
  planned path; also A/B Depth Anything 3 Small (Apache-2.0) against DA2-Small offline.
  DA3 pose output is too coarse at open-licence sizes to fix drift; the larger DA3
  models are non-commercial.
* Localisation drift in held-out rooms is now the main cause of missed goals (10/17).
* The Genesis learning suite has not been re-run since the nudge fix and the depth guard.

#### Notes

* Arrival tolerance is 0.15 m (the docs said 0.25 m until round 3; corrected).
* The full Genesis environment is `bash scripts/amr.sh setup` (+ `fetch-depth-model`);
  one Genesis process uses ~3–5 GB RAM, so run simulations one at a time.
* Commits: author Nitish <rrnitish@gmail.com>.
