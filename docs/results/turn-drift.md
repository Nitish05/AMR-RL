# Heading drift during in-place turns (round 7): research, benchmark, eight candidates, none adopted

**Status:** every candidate is implemented as a `VSLAMConfig` switch, and all are **off by default**.
With the switches off, tracking is byte-identical to before
(`tests/test_turn_drift.py::test_switches_off_leave_tracking_unchanged`).

Nothing passed the pre-registered held-out criteria ([turn-drift-prereg.md](turn-drift-prereg.md)).
- The best candidate, floor validation by parallax (C2a), fixes the typical near-object drift. The median error after a full turn on a fresh map falls from 4.8° to 0.1° in heldout_b.
- It does not fix the worst 10 % of turns: heading locks near almost featureless walls, at 13–27°.

## What was found first (research)

Three analyses (code, 354 stored runs, literature) agreed.

- **No global bias.** Over 15.8 h of tracked rotation, estimated rotation was 1.0019 × true rotation. The heading error grows by about 2.5 % of the angle turned, with random sign.
- **Full turns do not close.** Back at the start view the error is median 2.1°, p90 14.6°, max 28.7°.
- **The one consistent bias is near objects.** Within 0.4 m of an object the robot over-rotated by +26° per turn.
  - Cause: `_ipm` lifts features on object faces below the camera onto the floor, too far away.
  - The camera sits 0.125 m ahead of the turning axis, so a turn produces real parallax that those wrongly placed points do not predict. The solver absorbs the mismatch as extra yaw.
- **Errors above ~12° lock in.** Old landmarks fall outside the matching windows, and keyframes every 10° add new landmarks from the drifted pose.
- **Nothing checks the rotation.** The angular consistency check only trips at more than 2× the command plus 20°, and loop closure cannot fire for a turn in place.

## Benchmark

`scripts/dev/turn_capture.py` and `scripts/dev/turn_bench.py`. Evidence: `work/evidence/turncap-*`, `work/evidence/turnbench-20261004/`.

- **Capture.** Real commanded turns in Genesis; the wheels slip to about 0.83 × the command.
- **Positions.** 18 per world, binned by the nearest object (0–0.4, 0.4–0.7, 0.7+ m), each within 0.3 m of a keyframe of a saved map.
- **Patterns:**
  - `full2`: at least 720° of true rotation;
  - `survey`: 360° in 45° steps;
  - `sweep`: ±50°;
  - `back`: +180° then −180°.
- **Replay modes:**
  - on the saved map, starting at the true pose;
  - on a fresh map built from the first frame.
- **Determinism:** replays are deterministic (hash-checked).

**Validity gate.** The benchmark reproduced the drift magnitude:
- baseline p90 after a full turn is 7–10°;
- in the worst bins it reaches 19° after one turn and 35° after two.

It did not reproduce the systematic over-rotation: the median ratio was 1.008, not 1.07 or more. The near-object link held in home_a only.

**A caveat on the on-map numbers.** Error measured on a saved map includes distortion already in that map. Turning 180° away and back still leaves 7° of error on the arena map, which no turn fix can remove. Fresh-map replays are the cleaner measure.

## Candidates (all in `src/amr_rl/perception/vslam.py`)

| id | switch | idea |
|---|---|---|
| C1 | `turn_pin_xy` = hard / prior | estimate only the heading during a turn; position pinned to the axis, with a fallback when the base really moves |
| C2a | `floor_validation` = turn / all | triangulate floor points against the keyframe that made them; compare the triangulated distance with the floor-lifted one; on the floor → confirmed; much nearer → converted to a 3-D point |
| C3 | `turn_landmarks` = gaps / tentative | make turn landmarks only where the map has gaps, or keep them unconfirmed until the base moves |
| C4 | `turn_cmd_check` = flag | flag a turn whose measured rotation exceeds the commanded one, or stalls |
| C5 | `turn_closure` = rotate / purge | on returning to the start heading, re-find it against pre-turn landmarks and spread the correction over the turn |
| C7 | `depth_weighting` | down-weight matches by their parallax uncertainty |
| C8 | `turn_heading_gate` | keep the commanded rotation × a learned slip factor when the visual rotation disagrees in a frame (added after the round-1 failure analysis; disclosed) |
| — | `visible_once_per_frame` | count a landmark as seen once per frame, not once per search pass |

C2a was first built with a fixed 3 cm height threshold. In the synthetic test that misread noisy floor points as objects, so it was changed to a distance ratio (`floor_val_ratio` 0.12) before any benchmark run.

## Results

All values are p90 heading error in degrees, on map / fresh map, pooled over worlds.
- **Design split:** arena, home_a and arena_textured; 108 sequences per mode.
- **Held-out split:** heldout_b, heldout_c, home_a_dim and home_a_textured; 216 on-map and 288 fresh-map sequences.

### Round 1: each candidate alone (design, even-numbered positions)

| variant | after 360° | near objects | after 720° | lost frames | confidently wrong |
|---|---|---|---|---|---|
| baseline | 10.1 / 8.3 | 10.1 / 10.1 | 12.3 / 11.4 | 6.3 % / 0.1 % | 5.3 % / 5.3 % |
| **C2a, turn scope** | **7.8 / 6.5** | 7.6 / 7.0 | 9.8 / 7.7 | 6.1 % / 0.2 % | 5.1 % / 3.9 % |
| C2a, all floor points | 13.3 / 6.6 | 7.4 / 7.9 | 13.4 / 7.9 | 5.5 % / 0.1 % | 7.3 % / 3.9 % |
| C1 hard | 8.5 / 8.3 (fresh median 2.7 → 7.3) | 9.3 / 9.4 | 14.1 / 16.9 | 6.4 % / 0 % | 5.3 % / 20.4 % |
| C1 prior | 10.1 / 8.4 | 9.5 / 8.0 | 10.7 / 13.8 | 6.0 % / 0 % | 4.5 % / 6.9 % |
| C3 gaps | 13.5 / 9.1 | 11.9 / 12.1 | 22.7 / 19.9 | 5.9 % / 0.9 % | 11.6 % / 8.0 % |
| C3 tentative | 12.2 / 10.3 | 12.7 / 10.6 | 12.4 / 21.1 | 8.5 % / 0.1 % | 7.0 % / 8.7 % |
| C4 flag | 10.1 / 7.7 | 10.1 / 10.1 | 12.3 / 8.2 | 6.3 % / 2.9 % | 5.3 % / 2.1 % |
| C5 rotate | 10.1 / 7.7 | 10.1 / 10.1 | 12.3 / 10.6 | 6.3 % / 0.1 % | 5.3 % / 5.2 % |
| C7 | 8.0 / **22.5** | 6.1 / 24.7 | 20.3 / 44.1 | **22.8 %** / 3.4 % | 6.2 % / 18.7 % |
| C8 gate 1.2° | 8.4 / 6.9 | 9.7 / 8.3 | 9.4 / 13.3 | 9.2 % / **14.9 %** | 3.2 % / 2.7 % |
| visible once | 9.7 / 8.2 | 9.3 / 10.2 | 17.0 / 16.9 | 6.3 % / 0.4 % | 6.3 % / 7.4 % |

**C7 fails by design.** It weights each point by its *estimated* depth, but the misplaced face points are estimated too far away, so they keep high weight while correct near floor points are suppressed. The synthetic tests showed the same.

### Round 2: combinations from C2a, turn scope (design, even-numbered positions)

| variant | after 360° | after 720° | lost frames | outcome |
|---|---|---|---|---|
| C2a ratio 0.08 / 0.18 | 11.2 / 6.3 and 10.7 / 7.0 | 15.4 / 6.3 and 14.2 / 6.9 | ≈ | worse on map |
| + C5 | 7.8 / 6.3 | 9.8 / 7.7 | ≈ | gain under 0.5°: stop rule |
| + C4 | 7.8 / 6.1 | 9.8 / 7.0 | 6.1 % / **2.9 %** | fails the loss limit |
| + visible once | 11.3 / 5.3 | 17.1 / 5.3 | ≈ | worse on map |
| + C1 prior | 9.5 / 7.4 | 9.1 / 7.4 | ≈ | worse |
| **+ C8 gate 1.2°** | **5.6 / 4.1** | **6.1 / 4.9** | **8.9 % / 13.9 %** | confidently-wrong frames ≈ 0, but it loses tracking: fails the loss limit |
| + C8 gate 2° / 3° | 7.2 / 6.0 and 7.2 / 6.5 | 8.5 / 8.7 and 8.5 / 9.1 | 7.1 % / 0.3 % and 6.7 % / 0.2 % | at most 0.6° better; dropped |

**Finalists:** C2a (turn scope), and C2a + C5.

### Confirmation on the design split's odd-numbered positions

| variant | after 360° | near objects | after 720° | lost frames |
|---|---|---|---|---|
| baseline | 13.2 / 6.0 | 22.1 / 11.3 | 15.2 / 7.0 | 8.8 % / 2.9 % |
| C2a | 11.1 / 2.8 | 7.8 / 5.6 | 11.8 / 5.6 | 9.9 % / 2.4 % |
| C2a + C5 | 11.1 / **0.8** | 7.8 / 5.6 | 11.8 / **2.5** | 9.9 % / 2.4 % |

### Held out, run once

| variant | after 360° (≤ 5) | near objects (≤ 8) | after 720° (≤ 7) | back to start (≤ 3) | lost frames | far median |
|---|---|---|---|---|---|---|
| baseline | 4.8 / 14.3 | 14.0 / 19.0 | 18.5 / 29.7 | 2.2 / 0.2 | 13.9 % / 2.2 % | 0.96 / 0.07 |
| C2a | 4.4 / 14.0 | 14.0 / 22.2 | **7.2** / 27.7 | 2.2 / 0.2 | 13.8 % / 2.0 % | 1.10 / 0.09 |
| C2a + C5 | the same as C2a (28 closures on fresh maps, none of which changed these figures) | | | | | |

**Not met:** the near-object and fresh-map criteria. The tail is set by a few sequences per world.

| world, fresh map | median after 360°, baseline → C2a | p90 after 360°, baseline → C2a |
|---|---|---|
| heldout_b | 4.8 → 0.1 | 17.6 → 15.0 |
| home_a_textured | 3.4 → 0.4 | 19.1 → 24.4 |

On heldout_b's saved map, C2a made the p90 worse (10.0 → 15.6).

**Not run:** the slip sweep and the live validation (L1–L5). Under the pre-registration they apply only to a finalist that passes offline. The partial slip captures were stopped.

## What the remaining tail is

The worst sequence on the design split (home_a p06) shows the second failure mode. The robot turns 0.2 m from an almost featureless wall whose only structure is the dark skirting band.
- ORB features along that horizontal edge look alike, so matches slide along it.
- The heading then locks onto a wrong rotation, gaining 10–27° per revolution in about a second at the same bearing each time.
- Inliers drop to about 70, but the tracker still reports "tracking".

The nearest-object bins count objects, not walls, so these views sit in the middle bin. C2a, C5 and C1 cannot help here.

The commanded-turn gate (C8) removes the confidently wrong headings (5 % → about 0 % of tracking frames), but in such views it turns them into lost tracking. A good fix would detect the degeneracy and keep the commanded rotation without dropping the lock. That is the next step, with:
- a degeneracy measure from the spread of matched features along the turn direction;
- dead reckoning scaled by the learned slip factor (already implemented with C8);
- a longer prediction window while turning in place.

## Engineering notes

- Each candidate is a documented engineered rule. The C8 slip factor is an online running average, an engineered estimate rather than a learned model.
- **Tests** (`tests/test_turn_drift.py`, 7 tests) use a synthetic turn scene (`tests/turn_scene.py`) in which a box face 0.30 m from the axis drifts 12° per turn:
  - switches off leave tracking unchanged;
  - turn state follows the command;
  - pinning and its fallback;
  - turn closure fixes the box drift and leaves a correct turn alone;
  - floor validation on an empty floor;
  - command-check flags.
- **Harness logging.** The harness now logs, per frame, the command passed to the VSLAM, keyframes, heading sigma and turn flags, so live runs can be analysed for turns.
- **An aborted run, kept in the record.** The first round-1 run used the working tree while `vslam.py` was being edited. It was stopped after the baseline variant and re-run from a pinned worktree (`work/evidence/turnbench-20261004/aborted-r1/`).

---

# Round 8: the near-surface tail

Evidence: `work/evidence/turnbench-r8-20261004/`; pre-registration: [turn-drift-r8-prereg.md](turn-drift-r8-prereg.md).

## Corrected diagnosis

The round-7 account of the tail ("matches slide along the skirting edge") was wrong. Replaying 4,482 turning frames from 7 bad and 7 good sequences showed the following.

- **The heading was well constrained.** Heading information in the pose Hessian (AUC 0.60) and the image spread of the inliers (AUC 0.43–0.57) do not separate the frames where the error jumps from the normal ones.
- **Most of the points used were on a surface, not on the floor.** In jump frames, 55 % of the inliers lie on a wall or box face 0.15–0.3 m away that was lifted onto the floor; in normal frames, 0 %.
- **The over-rotation follows that share and the lever-arm model.**

  | share of lifted face points | estimated / true rotation |
  |---|---|
  | under 5 % | 1.00 |
  | 40–60 % | 1.34 |
  | over 80 % | 1.70 |

  The model 1 + 0.125/d predicts 1.73 for a wall at 0.18 m.
- **The points were created during the same turn.** 92 % had been made in the previous second, so later floor validation (C2a) comes too late.
- **Close walls were hidden in the bins.** The round-7 bins counted objects only, so turns next to a wall fell into the middle bin. Round 8 bins by the nearest surface, walls included.

## Candidates added

All are switches in `VSLAMConfig`, off by default.
- **C9** `c9_parallax_check`: before lifting a new point onto the floor, triangulate it against the previous keyframe.
- **C10** `c10_weight`: give lower weight to unvalidated points made during the turn.
- **C8b**: a tighter gate (0.65°, `turn_gate_frac` 0.3, or a 3-frame window), longer prediction during turns (`turn_max_prediction_s`), keyframes on gated frames (`turn_gate_keyframes`), and later a step-based comparison and 3 settle frames after command changes (`turn_gate_visual_step`, `turn_gate_settle`).
- **C11** `floor_patch_size`: smaller ORB patches in the floor band.

## Design rounds

All values are p90 heading error after 360°, on map / fresh map. 108 sequences per replay mode.

| variant | after 360° | near-surface bin | lost frames | note |
|---|---|---|---|---|
| baseline | 10.1 / 8.3 | 10.9 / 20.9 | 6.3 % / 0.1 % | |
| C9 strict / lenient | 9.7 / 7.8; 10.2 / 7.5 | | strict loses more | little gain alone |
| C10 | 17.8 / 12.6 | | | worse |
| C8b per frame / window | 7.8 / 6.4; 7.0 / 4.6 | 8.5 / 7.1; 6.9 / 4.8 | 8.9 / 10.3 %; 13.7 / 16.6 % | accurate but loses tracking |
| C11 | 12.1 / 4.5 | | **58.9 %** / 0 % | old maps use other descriptors; on maps rebuilt with C11: 5.9 on map but 24 % lost |
| gate + keyframes on gated frames | 8.3 / 6.3 | 9.9 / 7.1 | 5.4 % / 0 % | the loss problem is solved |
| + C9 + C2a (F1) | 7.6 / 4.0 | 6.2 / 3.1 | 5.4 % / 0 % | |
| F1 + C5 | 6.3 / 3.6 | | | |
| F1 + settle | 7.6 / 4.0 | | | removes a 4–6° offset left by one gated frame at a turn reversal |

The finalists, frozen at commit `946491a`:
- **FA** = gate (0.65° / 0.3) + keyframes on gated frames + 4 s turn prediction + settle 3 + C9 (turn) + C2a (turn) + C5 (rotate).
- **FB** = FA without C5.

On the odd-numbered positions, FA scored 10.7 / 2.9 against the baseline's 13.2 / 6.0. On saved maps the remainder is mostly distortion already in the maps.

## Held out, run once

Fresh seed-2 capture, including 4 positions next to a wall in each world.
- On map: 232 sequences (3 worlds).
- Fresh map: 320 sequences (4 worlds).

| | baseline | FA | FB | limit |
|---|---|---|---|---|
| e360 p90 | 16.2 / 15.6 | **2.3 / 2.2** | 2.8 / 3.2 | ≤ 5 |
| near-surface p90 | 19.8 / 19.2 | **2.8 / 4.2** | 2.8 / 4.8 | ≤ 8 |
| e720 p90 | 31.9 / 30.5 | **2.4 / 1.0** | 2.9 / 3.1 | ≤ 7 |
| back to start, p90 | 2.6 / 0.2 | 2.7 / 0.3 | 2.7 / 0.3 | ≤ 3 |
| lost frames | 12.1 % / 5.5 % | 10.5 % / 4.9 % | 10.5 % / 4.9 % | ≤ +1 pp |
| predicted frames | 2.5 % / 0.5 % | 3.2 % / 1.1 % | 3.2 % / 1.1 % | ≤ +2 pp |
| confidently wrong frames | 6.9 % / 12.5 % | **0.0 % / 0.1 %** | 0.0 % / 0.1 % | ≤ 0.5 % |
| jump frames | 2.8 % / 3.0 % | 0.3 % / 0.2 % | 0.3 % / 0.2 % | ≤ 30 % of baseline |
| ms per frame, p95 | 218 / 94 | 183 / 81 | 126 / 69 | ≤ 1.25× |

Both finalists pass all held-out criteria. Under the choice rule FA wins: FB is 1.03° worse on fresh maps, just outside the 1° margin.

## Slip robustness: FA fails

These are reset-mode captures on arena, 24 sequences each. The pose is set every frame to slip × the commanded rotation.

| true slip | replay | baseline e360 p90 | FA e360 p90 | baseline → FA, confidently wrong frames | baseline → FA, lost frames |
|---|---|---|---|---|---|
| 0.75 | fresh map | 3.2 | **0.2** | 0.9 % → 0 % | 8.0 % → 7.0 % |
| 0.75 | saved map | 5.2 | **17.6** | 1.2 % → **19 %** | 10.5 % → 9.6 % |
| 1.0 | fresh map | 0.2 | 0.2 | 4.7 % → 4.7 % | 7.2 % → 7.6 % |
| 1.0 | saved map | 5.1 | **18.7** (e720 41) | 1.2 % → **36 %** | 8.7 % → **18.1 %** |

The robustness criterion is not met, so FA is **not adopted** and the live validation (L1–L5) was not started.

**Suspected cause (not yet confirmed).** When FA overrides vision, it uses the slip-scaled command. The slip factor starts at an engineered 0.85, is learned slowly (2 % per frame, and only on frames with at least 150 inliers), and is reset for every new session. When the true slip differs, every overridden frame adds error. Saved maps give fewer high-inlier frames, so the factor may never adapt there.

The reset captures are idealised: perfectly constant slip and no wheel lag. The criterion still stands as registered.

**Next:** make the slip factor robust (fast learning from well-tracked turn frames, and a gate that widens while the factor is uncertain), then re-test on a new held-out capture and a new slip sweep.

## Slip fix and final test (run once)

**The cause, confirmed with an oracle.** Given the true slip factor, FA's error on the worst slip-1.0 sequences fell from 16–23° to 0.8–4.9°.

**The fix: `turn_slip_mode="median"`.** The slip factor is the median of the raw visual/commanded rotation over the last 60 frames with at least 100 inliers. Gated frames are included, and the gate stays off until 15 samples exist.

This gives finalist **FAm** = FA + median slip. It was frozen at `993459f`, and the addendum to [turn-drift-r8-prereg.md](turn-drift-r8-prereg.md) was committed before this run.

The final test used new seed-3 data: held-out captures from 4 worlds, and slip captures at 0.75 and 1.0.

### Held-out split (seed 3)

| | baseline | FA | **FAm** | limit |
|---|---|---|---|---|
| e360 p90 | 8.2 / 13.3 | 2.5 / 2.6 | **2.9 / 3.0** ✓ | ≤ 5 |
| near-surface p90 | 16.1 / 18.0 | 2.3 / 3.8 | **3.8 / 4.2** ✓ | ≤ 8 |
| e720 p90 | 12.0 / 25.2 | 2.3 / 1.9 | **2.6 / 3.4** ✓ | ≤ 7 |
| back to start, p90 | 2.0 / 0.2 | 2.2 / 0.2 | **2.2 / 0.2** ✓ | ≤ 3 |
| lost frames | 18.7 % / 6.4 % | 15.8 % / 3.7 % | **17.2 % / 4.7 %** ✓ | ≤ +1 pp |
| predicted frames | 2.9 % / 0.6 % | 3.9 % / 1.1 % | **4.0 % / 1.0 %** ✓ | ≤ +2 pp |
| confidently wrong | 4.9 % / 6.8 % | 0.3 % / 0.8 % | 0.3 % / **1.5 %** ✗ | ≤ 0.5 % |
| jump frames | 2.6 % / 2.9 % | 0.4 % / 0.3 % | **0.6 % / 0.7 %** ✓ | ≤ 30 % of baseline |
| far median | 1.16 / 0.10 | 1.07 / 0.06 | **1.09 / 0.06** ✓ | ≤ +0.5 |
| ms per frame, p95 | 228 / 105 | 155 / 86 | **130 / 66** ✓ | ≤ 1.25× |

Values are on map / fresh map: 232 on-map sequences and 320 fresh-map sequences.

### Slip robustness (seed 3)

| slip | | baseline | FA | **FAm** |
|---|---|---|---|---|
| 0.75 | e360 p90, saved / fresh map | 9.8 / 3.5 | 17.7 / 0.2 | **7.7** / 0.2 |
| 0.75 | near-surface p90, saved / fresh map | 11.2 / 2.4 | 18.0 / 0.8 | **16.6** / 0.7 |
| 1.0 | e360 p90, saved / fresh map | 5.8 / 2.9 | **35.0** / 0.4 | **5.1** / 0.1 |
| 1.0 | e720 p90, saved / fresh map | 7.1 / 4.1 | 76.4 / 0.6 | 5.1 / 0.2 |
| 1.0 | confidently wrong, saved map | 5.7 % | 63 % | 3.9 % |

## Verdict

**FAm is not adopted**, because the pre-registered criteria are not all met:
1. **Confidently wrong frames on fresh maps: 1.5 %**, against a 0.5 % limit. The baseline has 6.8 %.
2. **Slip robustness on saved maps.**
   - At slip 0.75: e360 p90 7.7° (limit 5°) and near-surface 16.6° (limit 8°).
   - At slip 1.0: e360 p90 5.1°, just over the 5° limit.

   The baseline also fails these limits (9.8° and 5.8°).

**On everything else FAm is a large improvement over the baseline:**
- held-out p90 after 360° is about 3°, against 8–13°;
- after 720° it is 2.6–3.4°, against 12–25°;
- confidently wrong frames are 3–16× fewer;
- it loses tracking less and is faster.

**FAm also removes FA's slip failure.** At slip 1.0 on a saved map FA reached 35°; FAm stays at 5°.

Everything stays switchable and off. The live validation (L1–L5) was not started.

**What remains:**
- **Saved maps at a slip far from the learned value.** At 0.75, the near-surface bin is worse than the baseline: 16.6° vs 11.2°.
- **Fresh maps.** 1.5 % of tracking frames are still confidently wrong.

The reset-mode slip captures are idealised (perfectly constant slip, no wheel lag), so live runs would show more.
