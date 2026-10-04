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
