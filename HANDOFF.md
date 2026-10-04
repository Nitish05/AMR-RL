# Handoff (2026-10-02, updated after the first local session): where AMR-RL stands

Written at the end of a cloud session so a local Claude Code session can continue.
Project rules are in `AGENTS.md` (read it first). Results with denominators are in
`README.md`, `docs/NAVIGATION_RESULTS.md`, `docs/LEARNING_RESULTS.md` and
`docs/CAPABILITY_LEDGER.md`.

## State of the repository

* Last commit on `main`: `3577fe5` (monocular depth near-field guard). Pushed by the owner
  if `git status` says "up to date with origin/main".
* **Uncommitted, never executed** (written in the cloud session, copied here):
  - `src/amr_rl/sim/learner_bench.py` — learner testbed (see below)
  - `scripts/eval/learner_bench.py` — runs the experiment grid in parallel, writes
    `work/evidence/learner-bench-<stamp>/rows.jsonl` and `report.md`
  - `tests/test_learner_bench.py`
  - `scripts/amr.sh` — new commands `bench-setup`, `bench-test`, `learner-bench`
  Expect first-run bugs. Fix them, keep `scripts/amr.sh test` green, then commit.


## Update: round 5 (2026-10-03): loop closure

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

## Update: round 4 (2026-10-03): open items worked through

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

## Update: first local session (macOS arm64), 2026-10-02

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

## Immediate task (original, now done): learner testbed under perception noise

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

## Agreed plan after that

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

## Notes

* Arrival tolerance is 0.15 m (the docs said 0.25 m until round 3; corrected).
* The full Genesis environment is `bash scripts/amr.sh setup` (+ `fetch-depth-model`);
  one Genesis process uses ~3–5 GB RAM, so run simulations one at a time.
* Commits: author Nitish <rrnitish@gmail.com>.
