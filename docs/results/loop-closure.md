# Loop closure (round 5): learned place recognition, planar verification, robust SE(2) pose graph

**Status:** on by default when the place model is cached locally
(`scripts/amr.sh fetch-place-model`). Without the model, `RuntimeConfig.place_descriptor`
is `None` and the runtime reports "unavailable"; nothing else changes.

**Code:**
- `perception/place_recognition.py`
- `perception/pose_graph.py`
- `perception/vslam.py` (`_loop_check`, `_verify_loop`, `_close_loop`)
- `mapping/occupancy.py` (evidence journal, `replay`)
- `runtime/robot.py` (`_on_loop_closure`)

**Tests:**
- `tests/test_pose_graph.py`
- `tests/test_mapping_navigation.py::test_loop_closure_replay_moves_evidence_with_its_keyframe`
- `tests/test_relocalisation.py::test_map_schema_v2_round_trips_anchors_and_place_descriptors`

## Why

The VSLAM had no loop closure, so drift accumulated. Arena maps reached 12–18 cm
RMSE, with up to 41 cm at single keyframes. Drift was also the most common reason for
missed navigation goals: "navigator says arrived, robot more than 0.15 m off".

## What the literature offers (2025–26), and what was chosen

- **Place recognition.** Foundation-model global descriptors lead the
  benchmarks: SALAD (DINOv2 + optimal-transport aggregation), BoQ, and MegaLoc, which
  extends SALAD with multi-dataset training ([Berton & Masone 2025](https://github.com/gmberton/MegaLoc),
  MIT). A 2026 evaluation of place-recognition methods for image-pair retrieval in
  robotics ([arXiv 2603.13917](https://arxiv.org/abs/2603.13917)) found MegaLoc and SALAD
  best on indoor ScanNet scenes, with EigenPlaces close and much faster. The classic
  bag of binary words (DBoW2, or the incremental iBoW-LCD) needs no learned weights.
- **Verification in repetitive scenes.** ROVER
  ([T-ASE 2026, arXiv 2508.13488](https://arxiv.org/abs/2508.13488)) runs pose-graph
  optimisation with each candidate loop and rejects the loop if the trajectory distorts.
  It reports AP 99.3 / max recall 85.1 vs 94.8 / 69.9 for the best image matcher
  (eLoFTR), at about 2 ms per check.
- **Robust back-end.** Graduated non-convexity with annealed Cauchy weights
  (Yang et al. 2020; GNC pose-graph schedules 2024–25) drives false loops to near-zero
  weight.
- **Two views are enough.** 2GO ([arXiv 2503.16275](https://arxiv.org/abs/2503.16275))
  shows that a sparse keyframe pose graph with two-view loop constraints works without
  bundle adjustment.

**Chosen:**
1. MegaLoc descriptors per keyframe.
2. Verification with the round-4 planar two-point RANSAC and guided verification,
   against the candidate keyframe's *own* landmarks, so the current, drifted map cannot
   vote.
3. Plausibility: the correction must be ≤ 0.15 m + 0.08 × the path length since the
   candidate, and the candidate must be at least 0.6 m of path back (a real revisit).
4. ROVER-style distortion check after optimisation.
5. An SE(2) keyframe pose graph with GNC-Cauchy loop edges.
6. Correction of everything placed in the map frame:
   - keyframes;
   - landmarks, which move with the keyframe that created them;
   - the current pose;
   - occupancy evidence, replayed from a journal with each keyframe's correction;
   - the trajectory, look spots and remembered entity positions.

## Place recognition in our simulated rooms (`scripts/dev/vpr_eval.py`)

Arena probe capture: 37 positions × full turns. Database = even positions, queries =
odd. A retrieval is correct within 0.5 m and 30°.

| descriptor | recall@1 | recall@5 | time per image |
|---|---|---|---|
| MegaLoc (8448-d) | **0.91** | **0.97** | 88 ms CPU / 20–26 ms MPS |
| ORB bag of binary words (vocabulary from the robot's own map) | 0.14 | 0.34 | 11 ms |

The repetitive floor texture that broke the old relocalisation also defeats the bag
of binary words.

## Calibration

The verification threshold was set from six 420 s builds with loop closure on: arena
seeds 0–2, home_a, heldout_b and heldout_c. That gave 3,349 logged candidates whose
relative pose was scored against ground truth. Correct candidates have a median of
only 26–27 guided inliers, because verification sees only the candidate's own
landmarks. False candidates reach up to 88.

- **The pre-stated rule:** 1.5× the strongest false candidate that passes all the
  other gates, in the arena runs only. That gives 41.
- **What happened at 41:** on the held-out rooms, one false candidate passed at
  exactly 41 (heldout_b, 0.23 m off).
- **What was chosen:** **50**, after seeing that. Those six builds therefore no longer
  count as independent; the validation below uses fresh seeds.

One accepted closure in those builds was "false" in a harmless way. Keyframe 651
matched keyframe 621 after only 0.42 m of path. The correction was 0.0 m, but the
constraint carried the current tracking's 6.5° heading drift. That led to the
minimum-travel rule (0.6 m).

## First A/B (threshold 70, before the minimum-travel rule)

Final keyframe-map error: the keyframe poses at the end of the run against truth at
each keyframe's time. On and off runs diverge after the first closure, so the pairing
is by scenario, not by goal.

| scenario | off: RMSE (max) | on: RMSE (max) | closures (false) |
|---|---|---|---|
| arena s0 | 13.0 cm (36.8) | **4.5 cm (8.8)** | 11 (0) |
| arena s1 | 17.5 cm (41.2) | **2.8 cm (7.0)** | 3 (1, the harmless one above) |
| arena s2 | 11.8 cm (22.3) | **4.9 cm (12.5)** | 1 (0) |
| home_a s0 | 2.2 cm (22.1) | 1.3 cm (3.5) | 4 (0) |
| heldout_b s0 | 1.1 cm (2.6) | 1.3 cm (4.9) | 11 (0) |
| heldout_c s0 | 1.8 cm (3.8) | 1.8 cm (3.8) | 0, identical run |

## Second set: fresh seeds, first version (threshold 50, minimum travel, no confirmation)

Evidence: `work/evidence/lcval-{on,off}-*`. The result was **net harmful**.

| scenario | off: RMSE (max) | on: RMSE (max) | closures (false) |
|---|---|---|---|
| arena s3 | 5.3 cm (11.0) | 8.4 cm (23.5) | 2 (0) |
| arena s4 | 9.0 cm (16.8) | 11.0 cm (28.9) | 8 (3) |
| home_a s1 | 5.1 cm (16.4) | 1.8 cm (6.6) | 4 (0) |
| heldout_b s1 | 8.3 cm (18.8) | 13.1 cm (41.2) | 11 (2; worst 46.6 cm) |
| heldout_c s1 | 3.6 cm (16.0) | 12.8 cm (24.8) | 3 (0) |
| home_a_dim s0 | 2.1 cm (3.5) | 2.3 cm (5.8) | 16 (0) |

**What went wrong.** In the harmful false closures (heldout_b), the robot really was
0.2–0.5 m from the candidate, and the candidate keyframes were accurate to 1–3 cm.
So these were not appearance aliases. The *pose* estimated from the candidate's own
landmarks was 25–47 cm wrong, with 57–66 verified inliers and no rival place to beat:
repetitive floor texture inside a small landmark subset.

The arena s4 "false" closures were a different kind. They closed against a region
built 25 s earlier, while the robot was already 23 cm off, so they restated the drift
(harmless but useless).

heldout_c s1 got worse without any false closure. Its 3 correct closures helped
(26 → 5.7 cm at t = 225 s), but fresh drift after t = 240 s was never corrected.

**Fix: temporal consistency, as in ORB-SLAM.** A verified loop closes only when
another keyframe within 6 implies the same correction of the current pose, within
5 cm and 2°, against the same old region. A pose that is wrong because of the texture
does not repeat.

Replaying the logged candidates at threshold 50: false closures 4 → 0 on these runs,
keeping 28 of 39 correct candidates; also 0 on the first six. The rule was designed
after seeing these runs, so a third set was needed.

## Third set: confirmation rule fixed beforehand

Evidence: `work/evidence/lcv3-{on,off}-*`.

| scenario | off: RMSE (max) | on: RMSE (max) | closures (false) | fresh? |
|---|---|---|---|---|
| arena s5 | 13.0 cm (36.8) | **7.2 cm (12.0)** | 2 (0) | no: same start as s0 |
| arena s6 | 17.5 cm (41.2) | **2.0 cm (4.3)** | 3 (0) | no: same start as s1 |
| home_a s2 | 3.5 cm (6.1) | 3.8 cm (9.6) | 1 (0) | yes |
| heldout_b s2 | 1.2 cm (3.0) | 1.0 cm (3.0) | 0 | yes |
| heldout_c s2 | 10.8 cm (20.2) | 12.3 cm (22.1) | 10 (0) | yes |
| home_a_dim s1 | 1.7 cm (3.8) | **0.9 cm (2.1)** | 4 (0) | yes |

Arena seeds 5 and 6 repeat seeds 0 and 1. `build_map` turns the start by seed × 72°,
so seed k ≡ k mod 5, and the off runs reproduce those exactly. They are not
independent of the calibration.

**Totals:** 20 closures, **0 false** (worst relative-pose error 0.9 cm).

In heldout_c s2, all 10 closures were against regions built *after* the drift had
set in (candidates from t ≈ 250–330 s). The corrections were 1–3 mm: correct, with
nothing to fix. The on and off runs are identical until t = 300 s, then diverge.

**Reading.** Loop closure is now safe (no false closure in 20). It removes drift when
the robot revisits a region mapped before the drift: the arena runs, home_a_dim, and
home_a in the first set. It cannot help when the only revisits are of already-drifted
regions. Preferring the oldest matching keyframe is a possible next step.

## Relocalisation with keyframe retrieval (open item 1)

v2 maps save the keyframe place descriptors and landmark anchors. Relocalisation then
tries the landmarks of the 5 most similar keyframes before the whole map.

Probe on the v2 map `lcv3-on-arena-s6/map`, 37 positions × full turns
(`reloc-capture-v2-arena-20261003`):

| distance to nearest keyframe | whole map: good / false / never | retrieval first: good / false / never |
|---|---|---|
| 0–0.15 m | 9 / 0 / 1 | 9 / 0 / 1 |
| 0.15–0.3 m | 8 / 0 / 3 | **10** / 0 / 1 |
| 0.3–0.5 m | 2 / 0 / 6 | 2 / 0 / 6 |
| > 0.5 m | 0 / 0 / 8 | 0 / 0 / 8 |

A modest gain at 0.15–0.3 m and no false acceptances. Positions more than 0.3 m from
any keyframe still mostly fail, and start 2 still never relocalises on this map.
Retrieval is not the bottleneck there; the floor around such viewpoints was not
mapped well enough to verify against.

## Navigation with loop closure

Setup: 4 rooms × seeds 0–2, `--map-seconds 300`, code `c9a7694`.
Evidence: `work/evidence/navlc-20261003-{on,off}-*`, reports `navlc-20261003-{on,off}-report.md`.

| | loop closure off | loop closure on |
|---|---|---|
| own-map goals arrived | 35/44 (80 %) | **38/46 (83 %)** |
| missed because "navigator says arrived, robot > 0.15 m off" (drift) | 3 | **1** |
| fault tests passed | 12/15 | 16/18 |
| impossible goals rejected | 36/36 | 36/36 |
| contact episodes | 0 | **4 (one run)** |
| loop closures (scored false) | – | 131 (2) |

Per room, own-map goals off → on: home_a 10/12 → 10/12, heldout_b 6/8 → 8/10,
heldout_c 7/12 → 8/12, home_a_dim 12/12 → 12/12. With n = 12 runs per variant the
gain in arrivals is within run-to-run variation; the drop in drift misses points the
same way.

**The two "false" closures (heldout_b s1).** Their relative poses were 10.1 cm and
4.1 cm off. The scorer flags anything above 10 cm or 0.1 rad. The corrections they
applied were 8 and 19 mm.

**The contacts (home_a s1, box-on-route fault test).**
- Contact began at t = 425.7 s, while the robot was still tracking on its way to the
  operator goal. The near-field guard had stopped once and was creeping, 5 creep steps.
- Pushing the box broke tracking at 426.9 s ("visual motion inconsistent with
  commands"), and the bounded recovery touched it again.
- The loop closures just before were 0–1 mm corrections. In the off run the box and
  goal were placed differently (the maps differ), and the planner rejected that goal.

This is attributed to the guard's creep behaviour, not to loop closure, and is
recorded as an open safety item.
