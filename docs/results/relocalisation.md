# Global relocalisation: planar two-point RANSAC vs PnP (round 4, macOS arm64)

**Status: default** (`VSLAMConfig.reloc_method = "planar2pt"`). The old method stays
selectable as `"pnp"` (with `reloc_confirmations=2`, `reloc_min_view_change=0`,
`reloc_probation_frames=0` it reproduces the old behaviour). Method:
[../VSLAM.md](../VSLAM.md) §1. Code: `src/amr_rl/perception/vslam.py`. Tests:
`tests/test_relocalisation.py`.

## The failure

**Round-3 learning suite:**
- 65 of 143 lost→tracking transitions were false (> 0.2 m or > 8.6°).
- Learning start 4 relocalised 0.82 m / 22° off, and the whole session ran in that
  wrong frame (scored retroactively by `scripts/eval/common.reloc_transitions`).
- Starts 2 and 3 never relocalised.

**Why the old method failed:**
- **Precision.** Whole-map ORB matching is at best about 38 % precise. The map
  holds a median of 26 near-duplicate landmarks per point, and floor descriptors
  are self-similar.
- **The solver.** PnP-RANSAC on 95 %-coplanar floor landmarks collapses below
  about 30 % correct matches.
- **Confirmation.** Two consecutive frames of an operator turn are nearly
  identical, so they confirm an alias as readily as a true pose.

## Probe

`scripts/dev/reloc_capture.py` puts the robot at sampled poses (evaluation-only pose
resets) and renders a full in-place turn in 4° steps. Poses are the 5 learning
starts plus free positions binned by distance to the nearest map keyframe.
`scripts/dev/reloc_probe.py` replays the turn through `track()` with the turn
command, the way a phase start sees it.

A sequence is scored on the first pose the VSLAM accepts:
- **good**: within 0.10 m and 5°;
- **false**: more than 0.2 m or 8.6° off.

The legacy run reproduces the closed loop exactly: start 1 accepted after 60° (the
real run took 58°), starts 2 and 3 never, start 4 a wrong pose.

### Arena, `map-arena-20261002-mac-s1`

37 positions. Thresholds were calibrated on the even positions and checked on the
odd ones.

| distance to nearest keyframe | positions | legacy: good / false / never | planar: good / false / never |
|---|---|---|---|
| 0–0.15 m | 10 | 8 / 0 / 1 (+1 imprecise) | 9 / **1** / 0 |
| 0.15–0.3 m | 10 | 1 / **3** / 6 | 10 / 0 / 0 |
| 0.3–0.5 m | 9 | 0 / 0 / 9 | 4 / 0 / 5 |
| > 0.5 m | 8 | 0 / 0 / 8 | 0 / 0 / 8 |
| learning starts 0–4 | 5 | 0, 1 good; 2, 3 never; **4 false** | 0, 1, 3, 4 good; 2 never |

The one planar "false" result is an even (calibration) position. There the
relocalised pose is 1.5 cm and −10.6° from truth. The map itself is locally
rotated by −6° (median of the build's estimated-minus-true heading near that spot;
p10 −8.2°), so this is the map's own error showing through, not an alias. The
held-out odd positions have 0 false results.

### Held out: home_a, `map-home_a-20261003-nopano-s0`

32 positions, thresholds frozen.

| distance to nearest keyframe | legacy: good / false / never | planar: good / false / never |
|---|---|---|
| 0–0.15 m | 3 / **4** / 1 | 8 / 0 / 0 |
| 0.15–0.3 m | 0 / **4** / 4 | 6 / 0 / 2 |
| 0.3–0.5 m | 0 / **2** / 5 (+1 imprecise) | 3 / 0 / 5 |
| > 0.5 m | 0 / 0 / 8 | 0 / 0 / 8 |

The legacy method would have put the robot in a wrong frame at 10 of 32 positions.
The planar method put it in a wrong frame at none.

**Cost:** 26 ms per `global_localize` call on one core (map of 55k landmarks).

## Closed loop

**Learning starts, `no_memory`, map `map-arena-20261002-mac-s1`:**
- Starts 0, 1, 3 and 4 relocalised correctly, between 11.0 and 13.8 s, with errors
  of 0.2–4.0 cm and 1.4–2.9°.
- Start 2 never relocalised.
- 0 false transitions.

**The three round-4 arena maps** (`map-arena-20261003-nopano-s{0,1,2}`), every start
on every map:

| map | start 0 | start 1 | start 2 | start 3 | start 4 |
|---|---|---|---|---|---|
| s0 | ok | ok | never | ok | ok |
| s1 | ok | ok | never | ok | ok (14 cm / 8°) |
| s2 | ok | never | ok | ok | ok |

That is 12/15 correct and 0 wrong. On each map, one start is too far from that map's
keyframes.

**Map builds** (`build_map.py`, 420 s, frontier panoramas off, counted operator turns):

| | arena s0 | arena s1 | arena s2 | home_a s0 |
|---|---|---|---|---|
| frames lost (first macOS builds) | 467 / 4200 (2911) | 90 (116) | 115 (2203) | 374 / 3499 |
| ATE | 5.3 cm | 11.9 cm | 2.5 cm | 1.4 cm |
| free coverage | 0.58 | 0.58 | 0.57 | 0.51 |
| relocalisations, false | 4, 0 | 1, 0 | 2, 0 | 6, 0 |

## Not done

**Keyframe retrieval** (saving keyframe descriptors and matching a frame against
keyframes first) would extend recall beyond about 0.3 m from keyframes. It needs a
new map schema and rebuilt maps.
