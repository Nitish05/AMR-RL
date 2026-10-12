# Global relocalisation: planar two-point RANSAC vs PnP (round 4, macOS arm64)

**Status: default** (`VSLAMConfig.reloc_method = "planar2pt"`). The old method stays
selectable as `"pnp"` (with `reloc_confirmations=2`, `reloc_min_view_change=0`,
`reloc_probation_frames=0` it reproduces the old behaviour). Method:
[../VSLAM.md](../VSLAM.md) §1. Code: `src/amr_rl/perception/vslam.py`. Tests:
`tests/test_relocalisation.py`.

Rounds 4–6 below are camera-only. The default robot (IMU + encoders) is measured in
[Round 9](#round-9-range-on-the-imu-robots-maps-2026-10-11-development-not-pre-registered):
0 wrong acceptances in 36 probe positions; recall 22/24 within 0.5 m of a keyframe, 2/12
beyond.

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

## Round 6: extending the range (evaluated, nothing adopted)

Code: `34a8fc2`. Evidence: `work/evidence/item3-offline-20261003/`,
`work/evidence/i3-{nocov,cov}-*`, `work/evidence/i3-capture-arena-s{2,3}`.

The round-5 diagnosis had two parts:
- **0.3–0.5 m from keyframes, matching is the bottleneck.** Near-true hypotheses
  exist in only 14 % of frames.
- **Beyond 0.5 m, coverage is the bottleneck.** 34–67 % of drivable space is more
  than 0.5 m from any keyframe.

The pass rule was fixed beforehand:
- 0 false relocalisations on held-out captures;
- 0.3–0.5 m recall at least doubled;
- all 5 learning starts relocalise on at least one new map;
- map ATE not worse.

### 1. Pooling floor seeds across the relocalisation turn (`reloc_pool_frames`)

Floor correspondences from the last 5 frames, moved into the current frame by the
commanded rotation, go into one two-point RANSAC.

| capture (turn sequences) | bin | pooling off: good / false / never | pooling on |
|---|---|---|---|
| arena v2 map (design) | 0.15–0.3 m | 10 / 0 / 1 | 11 / 0 / 0 |
| | 0.3–0.5 m | 2 / 0 / 6 | **4** / 0 / 4 |
| | learning starts | 4 of 5 (start 2 never) | **5 of 5** |
| home_a (held out) | 0.15–0.3 m | 6 / 0 / 2 | 7 / 0 / 1 |
| | 0.3–0.5 m | 3 / 0 / 5 | 3 / **1** / 4 |

The held-out false acceptance was 1.08 m and 19° off. It rested on 70–88 inliers,
mostly wall (triangulated) landmarks, with only 12–16 % of the predicted-visible
map cells re-found.

**Stop rule applied: off by default.** The same signature (low "explained" share,
wall inliers) appears in the one false relocalisation during a live build (arena s4,
t = 97.8 s, 26 cm, rejected in probation after 1.4 s). A floor on the explained
share is the next candidate. It needs its own calibration and a fresh held-out
capture.

### 2. Verification threshold 50 instead of 70

On the arena v2 capture, threshold 50:
- **single frames:** good acceptances go from 573 to 811, but **8 are false** (7 at
  0.15–0.3 m);
- **turn sequences:** 0 false, because the 3-candidate confirmation caught them.

The existing calibration rule needs the threshold at 1.5× the strongest false
acceptance, so 50 is **rejected**, and the threshold stays at 70.

### 3. Coverage pass at the end of map building (`RuntimeConfig.coverage_lattice`)

In the last 150 s of a 420 s build, the robot visits free lattice points more than
0.3 m from every keyframe and makes a full survey turn at each. Paired builds,
otherwise identical (`--no-coverage` against `--coverage-seconds 150`):

| scenario | ATE without coverage pass | ATE with coverage pass |
|---|---|---|
| arena s2 | 4.9 cm | 8.3 cm |
| arena s3 | 10.1 cm | 5.3 cm |
| arena s4 | 4.0 cm | **29.1 cm** |
| heldout_b s2 (held out) | 2.7 cm | **19.3 cm** (1 false loop closure) |

The error grows only once the coverage pass starts. In arena s4 the heading error
jumped 14° within one survey turn at t = 290 s and reached 28°, while the VSLAM
still reported "tracking"; the position error then grew to 72 cm. This is the same
in-place-turn drift that made frontier panoramas harmful in round 4.

A drifted map also relocalises the robot into wrong places. On the arena s2 coverage
map, the same captured frames gave 4 positions and 1 start more than 20 cm off
(5 in all; 0 on the no-coverage map).

**Off by default.** Before more keyframes can be bought with turning, rotation
tracking during in-place turns needs fixing. Possible fixes:
- a gyro-free rotation check from the commanded turn;
- slower turns;
- no keyframes while rotating.

The IMU (round 9) fixed the turn drift. The pass was re-tested on it (round 9 below):
it no longer harms the map, but it does not extend the range, so it stays off.

### What does reach all 5 starts

With the default configuration, the new no-coverage arena s4 map relocalises all
5 learning starts correctly (`starts-s4-base70.json`; start 1 after 260°). It has
the best ATE of the new maps (4.0 cm). This comes from the map itself, not from a
round-6 change. On the s2 map, 3 starts never relocalise. On s3, all 5 relocalise,
but start 3 is imprecise. The s4 map was chosen for the round-6 learning suite
before that suite ran.

## Round 9: range on the IMU robot's maps (2026-10-11, development, not pre-registered)

The sections above are camera-only. This one measures the default robot (IMU + wheel
encoders, R9g at `1380537`; the probe script is new in that commit).

**Probe.** `scripts/dev/reloc_live_probe.py` loads a saved map and starts the robot,
live in Genesis, at free positions binned by distance to the nearest keyframe (4 per bin,
seeded headings). The robot turns in place at 0.4 rad/s until the VSLAM reports tracking,
and the first accepted pose is scored against truth:
- good: ≤ 10 cm and ≤ 5°;
- false: > 20 cm or > 8.6°;
- imprecise: in between.

Maps: each arm's own validation-5 build of seed 10 (arena, home_a held out, heldout_b held
out); each arm probes with its own odometry. Evidence:
`work/evidence/r9g-step9-20261011/reloc-*`, scored by the analysis in `analysis/step9.json`.

| distance to nearest keyframe | camera only: good / imprecise / false / none | IMU + encoders: good / imprecise / false / none |
|---|---|---|
| < 0.3 m | 8 / 3 / 0 / 1 | **12 / 0 / 0 / 0** |
| 0.3–0.5 m | 4 / 2 / 0 / 6 | **10 / 0 / 0 / 2** |
| ≥ 0.5 m | 1 / 1 / 1 / 9 | 2 / 0 / 0 / 10 |

- **On the IMU maps, nothing accepted was wrong** (24 of 24 within 10 cm and 5°; worst
  6.9 cm, 2.1°). Recall within 0.5 m is 22/24; beyond 0.5 m it is still 2/12.
- **The camera-only errors are the maps' own.** The camera-only arena map's keyframes are
  up to 11.2° off (its build reached 17.3° of heading error). Its "false" acceptance
  (17 cm, 11.1°) and the "imprecise" ones (5–8.5°) match that rotation: the robot
  relocalised correctly into a wrong map. The IMU maps' keyframes are within 0.9–1.7°.
- **In the validation-5 builds**, the camera-only robot made 122 lost → tracking
  transitions, 24 of them false; the IMU robot made 17, none false.
- n = 12 positions per bin and arm on three maps; the two arms also differ in the map,
  so this measures the whole robot, not the relocaliser alone.

**Coverage pass A/B on the IMU robot.** The pass was rejected in round 6 because its
survey turns drifted the camera-only map (above). Ten paired 420 s builds, five worlds ×
seeds 10–11, offset 18°:
- with the pass: the validation-5 L1 builds (`--coverage-seconds 150`);
- without: the same seeds at `1380537` (`--no-coverage` default; the mapping code is
  unchanged since `aa143e3`), `work/evidence/r9g-step9-20261011/nocov-odo-*`.

| over the 10 pairs | median with the pass | median without | median Δ (pass − none), range | pairs better with the pass |
|---|---|---|---|---|
| ATE | 1.6 cm | 2.2 cm | −0.06 cm (−1.3 to +0.7) | 5 / 10 |
| max heading error | 1.42° | 1.32° | −0.06° (−0.7 to +0.5) | 5 / 10 |
| free coverage | 58.7 % | 62.9 % | −1.7 pp (−21 to +3) | 4 / 10 |
| false-free cells (deep) | 23 (1.5) | 36.5 (5) | −17 (−2) | 7 / 10 (6 / 10) |
| free floor within 0.3 m of a keyframe | 35 % | 40 % | −3 pp (−10 to +4) | 4 / 10 |
| expected relocalisation recall over free floor | 0.54 | 0.61 | −0.03 (−0.14 to +0.03) | 4 / 10 |

The last row weights each map's share of free floor in each distance bin by the IMU
probe's recall in that bin (1.0, 10/12, 2/12); keyframes from each map's `map.json`.
- **The drift harm is gone:** the pass no longer moves ATE or heading.
- **It does not extend the range either.** The 150 s of frontier exploration it replaces
  reach slightly more floor (longer paths) than its survey turns.
- **It stays off by default.** The pre-registered L1 protocol keeps using it (both arms).

**Explained-share floor: not built.** It was proposed (round 6, above) against the alias
signature of the camera-only false acceptances. On the IMU robot there is no false
acceptance to calibrate it against (0 of 36 probe positions, 0 of 17 transitions in
validation 5). Building it needs such cases first, on a fresh capture.

## Keyframe retrieval (done in round 5)

Saving keyframe descriptors and trying the most similar keyframes' landmarks first was
built with loop closure (map schema v2, [loop-closure.md](loop-closure.md#relocalisation-with-keyframe-retrieval-open-item-1)).
It gave a small gain at 0.15–0.3 m on the camera-only robot. Beyond 0.5 m, recall stays
low on the IMU robot too (2/12 above): the floor seen from such viewpoints is not mapped
well enough to verify against.
