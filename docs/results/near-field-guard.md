# Near-field guard for objects placed on the route

**Status: on by default when the depth model is installed**
(`scripts/amr.sh fetch-depth-model`; without it the runtime reports
`near_field_guard: unavailable` and drives as before). Code:
`src/amr_rl/perception/near_depth.py` (detector) and
`src/amr_rl/navigation/near_field.py` (guard).

## The failure

Round 2: 4 of the 5 box-on-route tests ended in contact. The occupancy grid needs
about three keyframes of obstacle evidence (one keyframe per 10 cm of travel) to
overturn free evidence saturated at −4. An in-place turn gives no parallax at all.
So after turning toward its route the robot reached a box 0.3–0.45 m ahead before
the map changed.

`scripts/dev/turn_then_box_probe.py` reproduces it. It loads the map saved by a
round-2 navigation run, relocalises, and drives to a certified point. It then sends
a goal 1.5 m away in a direction 180° or 135° from the robot's heading, with a
35 cm box 0.6 m along that line (both inside certified free space). Ground truth
places the box and scores contacts only.

## What works: a monocular depth model, scaled by the floor

Depth Anything V2 Small (24.8 M parameters, Apache-2.0, pinned revision
`5426e4f0`) predicts relative inverse depth from one onboard frame. Each frame, a
robust affine fit against the floor's known inverse depth (camera height and
pitch) makes it metric. Pixels that come out ≥ 5 cm above the floor within 1.2 m
are obstacle points. No motion is needed, so it works right after an in-place
turn. The model input is 336 px wide, which takes about 0.2 s on 2 CPU cores.

**Offline accuracy** (`scripts/dev/depth_obstacle_probe.py`): random true poses,
with and without the box straight ahead, scored against true geometry in the
corridor ahead (x 0.15–1.0 m, |y| ≤ 0.2 m):

| Room | Obstacle within 0.6 m detected | Range error (median) | False alarm (corridor clear to 1.0 m, alarm ≤ 0.6 m) |
|---|---|---|---|
| home_a | 65/69 | −2.3 cm | 1/40 |
| heldout_b | 73/75 | −3.6 cm | 2/36 |
| heldout_c | 65/69 | −3.1 cm | 1/40 |
| home_a_dim | 65/69 | −2.5 cm | 1/40 |
| **all** | **268/282 (95 %)** | | **5/156 (3 %)** |

(home_a, heldout_c and home_a_dim share a layout, so the same seed gives the same
poses there.)

**The guard.** Every 0.3 s while following a path, including while turning
toward it:

- The free run is the along-path distance to the first path point whose
  footprint would overlap ≥ 6 obstacle points.
- **Stop:** one frame suffices when the free run is ≤ 0.20 m.
- **Creep:** at 0.07 m/s when the free run is ≤ 0.50 m.
- **Write to the map:** obstacle cells seen in two frames within 1.5 s are written
  into the grid as occupied, so the planner routes around or reports blocked.
- **Give up:** after 4 s stopped, the goal ends blocked (`obstacle_ahead`).

**Closed loop** (box 0.6 m along the new route; guard off = current code without
the model):

| Room | 180° turn: off | 180° turn: on | 135° turn: off | 135° turn: on |
|---|---|---|---|---|
| home_a | contact after 4.5 s | **no contact**, stopped 0.32 m from the box | contact after 3.7 s | skipped* |
| heldout_b | contact after 13.9 s | **no contact**, 0.36 m | skipped* | skipped* |
| heldout_c | contact after 4.5 s | **no contact**, 0.31 m | contact after 3.4 s | skipped* |
| home_a_dim | contact after 13.6 s | **no contact**, 0.31 m | no contact | skipped* |

Off: **6 contacts in 7 scenarios**. On: **0 contacts in 4**. Distances are robot
centre to box edge; touching is about 0.18–0.23 m.

\* Skipped: no straight certified 1.5 m line from any start point. With the guard
on, the cells it writes into the map during the trips to the start point change
which lines are certified. That is a real side effect, measured next.

**Cells written on open floor.** Scored against true geometry with a 10 cm margin
(`amr_rl.sim.evaluator.score_guard`), 2–31 cells per scenario were written on open
floor (about 10–25 % of the cells written). The rest were furniture, walls and the
box. These reduce certified free space.

## Full navigation evaluation (4 rooms × 3 seeds)

`work/evidence/navigation-20261001-depthguard`, compared with round 2 (same rooms,
seeds and protocol; details in [NAVIGATION_RESULTS.md](../NAVIGATION_RESULTS.md)):

| | round 2 | with guard |
|---|---|---|
| Box-on-route tests that ran | 5 | 5 |
| …ended in contact | 4 | **0** (4 stopped and reported blocked, 1 went around and arrived) |
| Contact steps, all phases | 533 | **0** |
| Own-map goals arrived | 22/32 | 20/37 |
| Misses plausibly caused by the guard's map marks | — | 3 of 17 (2 replan budget, 1 lost clearance) |
| Cells written on open floor | — | 1,107 of 4,712 (23 %) |

Most of the drop in arrival comes from localisation (10 drift misses, 4 from one
VSLAM failure). The guard's false marks are the cost to reduce next: write only
cells near the planned path instead of everything within 1 m.

## Rejected camera-only variants (plane-parallax floor test)

Both ran the map's plane-parallax floor test every 0.3 s between the current frame
and a keyframe with ≥ 4 cm baseline. The code is in git history (commit `bda4e0f`).

* **Write obstacle bases that two successive probes agree on:** 83 cells written
  on open floor in one run (24 events in 50 probes); later goals became
  unreachable. Offline, the rule "two probes agree" gave 13–63 false cells per
  scenario. The strictest rule tried (clusters of ≥ 8 points confirmed by 3
  probes) still gave 4 false cells in each 135° scenario and missed the box in all
  four.
* **Drive only where the floor ahead was just verified:** 4 of 5 ordinary
  open-floor trips ended `path_ahead_not_verified`; with the box ahead the band
  still verified at 0.45–0.60, and one of two scenarios ended in contact.

At 0.3–0.5 m the parallax test is not discriminative enough for a per-frame
decision. The depth model is.

## Limits

- About 23 % of the cells written into the map are on open floor.
- Simulation renders only; the model has not seen this robot's real camera.
- About 0.2 s per frame on 2 CPU cores is fine in lockstep but would need
  acceleration on hardware.
- The scale fit assumes a flat floor is visible below the obstacle. A wide object
  filling the lower image can corrupt the fit.
- Objects lower than 5 cm are not detected.
