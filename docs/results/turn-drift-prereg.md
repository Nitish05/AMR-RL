# Pre-registration: heading drift in in-place turns (round 7)

Written after the benchmark and the baseline existed, and before any candidate was
replayed on the captured images. The plan is in
`/Users/rrnitish/.claude/plans/tingly-painting-panda.md`. Results will go in
[turn-drift.md](turn-drift.md).

**Disclosed beforehand.** The candidates were first run on a synthetic scene
(`tests/turn_scene.py`: floor, walls, and a box face 0.30–0.45 m from the turning
axis) to check the implementations. Those runs informed the candidate settings
listed below, but none of the thresholds here.

## Benchmark

**Capture: `scripts/dev/turn_capture.py`, driven mode.**
- Real wheel commands; slip comes from physics: about 0.83 × commanded at 0.45 rad/s.
- Per frame it stores the image, the command in force (what the runtime passes),
  the true pose, and the rendered depth (scoring only).
- 18 positions per world, 6 per nearest-object bin (< 0.4, 0.4–0.7, > 0.7 m). Each
  position is within 0.3 m of a keyframe of a saved map.
- Four patterns per position:
  - `full2`: at least 720° of true rotation;
  - `survey`: 8 × 45° with pauses;
  - `sweep`: ±50°;
  - `back`: +180° then −180°.

**Replay: `scripts/dev/turn_bench.py`.**
- `onmap`: the saved map, starting at the true pose.
- `fresh`: a new map from the first frame.
- Replays are deterministic: repeated runs give identical hashes.

**Design split:** arena (map `i3-nocov-arena-s2`), home_a (`navlc-20261003-off-home_a/home_a`),
arena_textured (`tex-arena_textured-s2`). Positions: `turncap-*-driven-20261004`.

**Held-out split:** heldout_b, heldout_c and home_a_dim on their navigation maps, plus
home_a_textured in fresh mode. They are captured with seed 1 and replayed only for
the frozen finalists.

## Baseline and validity gate

Baseline, all bins (onmap / fresh):

| world | e360 median | e360 p90 | e720 p90 | emax p90 | lost |
|---|---|---|---|---|---|
| arena | 3.6° / 2.3° | 6.8° / 3.7° | 5.4° / 3.8° | 8.9° / 4.0° | 3.5 % / 0 % |
| home_a | 1.1° / 0.1° | 7.7° / 8.6° | 11.1° / 10.2° | 5.5° / 8.9° | 6.6 % / 0.2 % |

The worst bin was home_a fresh, 0.4–0.7 m: e360 p90 19.4°, e720 p90 34.5°.

**Gate outcome.**
- The near-bin p90 is at least 2× the far-bin p90 in home_a (onmap 6.4° vs 2.1°;
  fresh 9.1° vs 0.1°), but not in arena onmap (7.0° vs 5.6°).
- e360 p90 ≥ 6°: **met**.
- Near-bin over-rotation ≥ 1.03: **not met**. The median est/true rotation ratio is
  1.008 in every bin.

So the benchmark reproduces the drift magnitude and, in home_a, its link to nearby
objects, but not the systematic over-rotation seen in the live data. The mechanism
in these captures is therefore partly different. The benchmark is used anyway,
because the live data also show random-sign drift (2.5 % of the angle, p90 14.6°
per full turn). L1–L5 below check the result in closed-loop runs.

## Candidates

Each is a `VSLAMConfig` switch, off by default.
- **C1** `turn_pin_xy` = hard | prior
- **C2a** `floor_validation` = turn | all
- **C3** `turn_landmarks` = gaps | tentative
- **C4** `turn_cmd_check` = flag
- **C5** `turn_closure` = rotate | purge
- **C7** `depth_weighting` = turn
- **side fix** `visible_once_per_frame`

## Protocol

1. **Round 1.** Every candidate alone, on the even-numbered positions of the three
   design worlds, both replay modes.
2. **Round 2.** Greedy combination, starting from the best single candidate. Stop
   when e360 p90 improves by less than 0.5° or a non-inferiority metric fails.
   At most 12 variant runs.
3. **Round 3.** The odd-numbered positions confirm the order. At most 2 frozen
   finalists are then run once on the held-out split. The reset-mode slip sweep
   (0.75 / 1.0) is run on arena positions.

The primary metric is pooled over worlds, bins and patterns, separately for onmap
and fresh.

## Pass criteria (held-out, finalists)

**Primary:**
- e360 p90 ≤ 5°
- near-bin (< 0.4 m) e360 p90 ≤ 8°
- e720 p90 ≤ 7°
- e_end p90 ≤ 3° for `back`, which returns to its start heading

**Non-inferiority against the paired baseline:**
- lost frames ≤ +1 pp
- loss events ≤ +10 %
- far-bin e360 median ≤ +0.5°
- confidently wrong tracking frames ≤ 0.5 % and not above baseline
- position slide p90 ≤ 5 cm
- ms/frame p95 ≤ 1.25×

**Robustness:** the primary criteria also hold at slip 0.75 and 1.0.

**Choice:** among passing finalists, the one with the fewest switches whose
primary metric is within 1° of the best.

## Live validation before adoption

Paired runs, same seeds and commit:

| check | runs | pass if |
|---|---|---|
| L1 | coverage-pass builds: arena, arena_textured, heldout_b × seeds 0–4 | median per-run max heading error ≤ 10° and none > 20°; ATE median Δ ≤ +0.5 cm; none worse by > 3 cm; loss ≤ +2 pp |
| L2 | normal builds: arena and home_a × 3 seeds | not worse |
| L3 | navigation: 4 rooms × 3 seeds | arrivals ≥ baseline − 1; contacts not up |
| L4 | relocalisation probe | recall not lower; 0 false |
| L5 | learning runs in arena_textured × 3 seeds | drift events ≥ 8° per 100 turns at least halved; outcomes not worse |

If any of these fails, the change stays switchable and off by default, and the
failure is reported.

## Stop rules

- A candidate is dropped if, within 3 settings, it fails to improve design e360 p90
  by at least 20 % or costs more than 1 pp of loss.
- The held-out split is run once; there is no retuning after it.
