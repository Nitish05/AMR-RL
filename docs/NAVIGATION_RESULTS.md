# Navigation results (simulation, lockstep)

Navigation and learning are evaluated separately; learning results are in
[LEARNING_RESULTS.md](LEARNING_RESULTS.md).

**Protocol** (`scripts/eval/navigation.py`; one process per room and seed): the
robot starts with autonomy disabled; the evaluator enables it and it explores for
300 s of simulated time (`explore_only`, onboard RGB only). Seed *k* turns the
start heading by *k* × 72°. Then the evaluator acts as the operator: fixed goals
(some impossible: inside furniture or outside the room), four goals sampled from
the interior of the robot's own certified map, a camera blackout during a goal,
and a 35 cm box placed on the robot's own planned route. When the robot is lost
and its bounded recovery has given up, the evaluator turns it slowly by hand
(manual commands, counted) so it can relocalise. Arrival = within 0.15 m of the
true goal **and** the navigator reports arrival. `home_a` is the development room;
the other three were never used for tuning.

Evidence: `work/evidence/navigation-20261001-depthguard` (round 3, current code,
12 runs, 0 crashed) and `work/evidence/navigation-20260930-154854` (round 2).
Regenerate: `python scripts/eval/report.py --nav <dir> --out <file>`.

**Correction (round 3):** earlier versions of this page and the README said
arrival meant within 0.25 m. The evaluator has always used **0.15 m**
(`ARRIVAL_TOLERANCE` in `scripts/eval/navigation.py`), so every published arrival
count was scored at 0.15 m. Only the description was wrong.

## Round 4 (macOS arm64): planner/corridor A/B

All runs on this machine, 4 rooms × seeds 0–2, `--map-seconds 300`, one process per
room and variant. Evidence: `work/evidence/navigation-20261002-2237-<variant>-<room>`;
reports `navigation-20261002-2237-<variant>-report.md`. The Linux round-3 numbers
below are not directly comparable: different platform, and the map builds differ.

| variant | own-map goals arrived | impossible goals rejected | contacts |
|---|---|---|---|
| `07f8ae0`: before the planner/corridor fix | 28/37 | 36/36 | 0 |
| `85898e6`: fix with denser (quarter-cell) sampling | 22/41 | 36/36 | 0 |
| `07f8ae0` + fix at the original half-cell sampling (current) | 23/37 | 36/36 | 0 |

**What 5714aac did.** It made the planner's shortcut test and the navigator's
corridor check use the same sample points. That fixed the round-3 learning bug where
the robot held for the rest of the run next to the aversive fixture. It also made
the shortcut test denser, which changed path shapes. The denser variant missed more
goals, mostly "navigator says arrived, robot more than 0.15 m off": 14 vs 7.

**The current variant.** It keeps the shared samples at the original spacing, so
planned paths are identical to `07f8ae0`. 33 of its 37 sampled goals are at the
same positions as `07f8ae0`'s.

**On those 33 paired goals the two variants are even: 24 vs 23 arrivals.** The
total difference comes almost entirely from one run, home_a seed 2. Its
exploration diverged, so it sampled different goals, and 4 of them were missed
through localisation drift.

**Not yet measured:** navigation with the round-4 relocalisation
([results/relocalisation.md](results/relocalisation.md)).

## Round 3: near-field depth guard (current code)

Round 3 is round 2 plus the monocular-depth near-field guard
([results/near-field-guard.md](results/near-field-guard.md)) and the nudge-reach
fix (which does not affect navigation). Same rooms, seeds and protocol.

| | round 2 | round 3 (depth guard) |
|---|---|---|
| Box-on-route tests that ran | 5 | 5 |
| …ended in contact | **4** | **0** (4 stopped and reported blocked; 1 went around the box and arrived) |
| Contact steps, all phases, 12 runs | 533 | **0** |
| Own-map goals arrived | 22/32 (69 %) | 20/37 (54 %) |
| Runs with no own-map goals (not localised at goal time) | 3 | 2 |
| Impossible goals rejected | 36/36 | 36/36 |
| Camera-blackout tests that ran / reached the goal | 7 / 5 | 5 / 4 |
| Deep false-free cells per run | 0–103 | 0–44 |
| Free coverage after 300 s | 29–73 % | 27–72 % |
| Cells the guard wrote on open floor | — | 1,107 of 4,712 written (23 %) |

**Reading these results:**

* **Contacts are gone in these 12 runs.** All 5 box tests that ran ended without
  touching. Round 2's 2 bench contacts during recovery did not recur either, but
  those runs took different paths, so this is not evidence of a fix for them.
* **Arrival fell from 69 % to 54 %, mostly from localisation.** Of the 17 missed
  own-map goals:
  - 10 were "arrived" by the robot's own estimate but 0.16–0.76 m off in truth (drift);
  - 4 were lost to a VSLAM failure mid-drive (heldout_c seed 2: tracking lost
    while driving across open floor, then never recovered);
  - **3 could come from the guard's marks**: 2 ran out of replan budget and 1 goal
    had lost clearance. Round 2 had 2 replan-budget failures.

  The guard stops and slows the robot, which changes its routes and so its VSLAM
  history. With 3 seeds per room, run-to-run variation of this size is expected,
  and the drop cannot be attributed to the guard with confidence.
* **Cost of the guard:** about 23 % of the cells it writes into the map are on
  open floor (scored with a 10 cm margin around true geometry). That reduces
  certified free space and is the next thing to improve: write only cells on or
  near the planned path instead of everything within 1 m.
* **Compute:** one depth frame every 0.3 s while following a path, about 0.2 s
  each on 2 CPU cores (lockstep, so it does not change simulated timing).

## Round 2 (before the depth guard): summary

* **Reaching goals in its own map: 22/32 (69 %)**, up from 5/16 (31 %) in round 1.
  By room: home_a 8/8, heldout_c 7/8, home_a_dim 4/8, heldout_b 3/8.
  **Denominator caveat:** in 3 of the 12 runs (home_a_dim seed 0, home_a
  seed 1, heldout_c seed 1) the robot was not localised when the goal phase
  began, so no own-map goals could be sampled and every goal was refused with
  `autonomy_not_enabled_or_not_localized`. Counting each of those runs as 4
  failed goals gives **22/44 (50 %)**.
* **Impossible goals rejected: 36/36** (9 of them in the 3 runs above, where
  every goal was refused because the robot was not localised). Unknown space is
  never treated as free.
* **Localisation:** ATE ≤ 5 cm in 8 of 12 runs (≤ 3 cm in 7); 8–18 cm (worst 45 cm) in the
  other 4, all in the held-out layout and appearance rooms (heldout_b, heldout_c
  seeds 1–2). One run (home_a seed 1) stayed lost after 170 s even with operator
  turns, so its mapping phase ended early.
* **Contacts: none while exploring (0 in 12 runs × 300 s).** 8 contact episodes
  in the fault phase: **6 with the box placed on the route** (4 of the 5
  obstacle tests that ran ended in contact; the box then filled the camera view
  and tracking was lost) and 2 with the bench in heldout_c seed 1, both during
  bounded recovery. Recovery rotates in place only if the last good pose had
  footprint clearance, and it did on the map, but that pose was already 48 cm
  wrong: VSLAM was confidently off before it lost tracking. (An earlier version
  of this page blamed a gap in the turning check; the trace shows otherwise.)
* **Camera blackout:** 7 tests ran (5 skipped: no reachable start/end pair);
  in every one tracking loss revoked motion, bounded recovery relocalised and the
  operator re-enabled; 5 reached the goal.
* **Pre-chosen "reachable" goals:** 9/48 arrived; most lie in space 300 s of
  exploration never certified (coverage 29–73 %), so rejecting them is expected.
* **Monocular scale:** similarity scale 0.91–1.07.

## Round 1 → round 2

| | round 1 (`navigation-20260930-024431`, 4 runs) | round 2 (12 runs) |
|---|---|---|
| own-map goals arrived | 5/16 | 22/32 |
| impossible goals rejected | 12/12 | 36/36 |
| contacts while exploring | 0 | 0 |
| obstacle-on-route tests with contact | 0/1 | 4/5 |
| worst mapping error | 28.5 cm | 45 cm |

Round-2 changes that bear on this: goal snapping (12 cm) when live evidence
removes a goal's clearance, a fix to periodic path validation, per-step
corridor checks, hard keep-out around remembered objects, faster detection of a
frozen visual estimate, removal of landmarks created just before a tracking
loss, and relocalisation gated by dead reckoning. VSLAM changes were also
benchmarked separately on seven scenarios ([results/vslam-benchmark.md](results/vslam-benchmark.md));
mapping settings were A/B-tested on identical trajectories ([results/mapper-ab.md](results/mapper-ab.md)).

## Round 3 generated tables

Round-2 generated tables are in git history (commit `bda4e0f`).

### Summary by room (all seeds)

| room | split | seeds | own-map goals arrived | impossible goals rejected | ATE cm (range) | worst error cm | free coverage | deep false-free | contact episodes | fault tests passed / run |
|---|---|---|---|---|---|---|---|---|---|---|
| home_a | development | 3 | 7/8 | 9/9 | 1.9–2.8 | 5.9 | 46%–53% | 0–4 | 0 | 3/4 (2 skipped) |
| heldout_b | heldout_layout | 3 | 4/9 | 9/9 | 4.9–14.6 | 44.6 | 27%–28% | 6–13 | 0 | 2/2 (4 skipped) |
| heldout_c | heldout_appearance | 3 | 1/8 | 9/9 | 1.3–8.3 | 14.6 | 61%–72% | 12–44 | 0 | 0/0 (6 skipped) |
| home_a_dim | heldout_lighting | 3 | 8/12 | 9/9 | 1.0–4.7 | 7.3 | 35%–40% | 1–12 | 0 | 4/4 (2 skipped) |

**All rooms and seeds:** own-map goals arrived 20/37; impossible goals rejected 36/36; contact episodes 0.


### Mapping by exploration (onboard RGB only)

| world | split | sim s | path m | ATE m | max err m | scale | tracking frames | lost frames | free coverage | free cells | false-free cells | deep false-free | contact episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| home_a | development | 300.000 | 9.009 | 0.020 | 0.045 | 1.009 | 3000/3000 | 0 | 0.526 | 4047 | 16 | 0 | 0 |
| heldout_b | heldout_layout | 300.000 | 4.842 | 0.049 | 0.124 | 1.045 | 3000/3000 | 0 | 0.282 | 2091 | 34 | 8 | 0 |
| heldout_c | heldout_appearance | 300.000 | 8.031 | 0.039 | 0.068 | 1.001 | 3000/3000 | 0 | 0.715 | 5548 | 71 | 30 | 0 |
| home_a_dim | heldout_lighting | 300.000 | 9.240 | 0.010 | 0.034 | 1.000 | 3000/3000 | 0 | 0.353 | 2765 | 58 | 8 | 0 |
| home_a s1 | development | 294.900 | 6.138 | 0.019 | 0.048 | 1.005 | 2367/2949 | 582 | 0.462 | 3540 | 5 | 1 | 0 |
| heldout_b s1 | heldout_layout | 300.000 | 5.634 | 0.146 | 0.446 | 1.180 | 2853/3000 | 147 | 0.267 | 1979 | 28 | 6 | 0 |
| heldout_c s1 | heldout_appearance | 225.800 | 5.938 | 0.013 | 0.024 | 0.997 | 1676/2258 | 582 | 0.610 | 4718 | 48 | 12 | 0 |
| home_a_dim s1 | heldout_lighting | 300.000 | 6.261 | 0.011 | 0.022 | 1.006 | 2873/3000 | 127 | 0.401 | 3079 | 8 | 1 | 0 |
| home_a s2 | development | 300.000 | 8.649 | 0.028 | 0.059 | 1.000 | 3000/3000 | 0 | 0.461 | 3560 | 31 | 4 | 0 |
| heldout_b s2 | heldout_layout | 300.000 | 5.410 | 0.138 | 0.259 | 1.068 | 3000/3000 | 0 | 0.285 | 2115 | 38 | 13 | 0 |
| heldout_c s2 | heldout_appearance | 300.000 | 10.305 | 0.083 | 0.146 | 0.979 | 2948/3000 | 52 | 0.705 | 5567 | 174 | 44 | 0 |
| home_a_dim s2 | heldout_lighting | 300.000 | 5.452 | 0.047 | 0.073 | 1.001 | 2949/3000 | 51 | 0.370 | 2879 | 46 | 12 | 0 |

False-free = estimated FREE where the true room is occupied or outside it; deep = more than 10 cm inside an obstacle/wall. Coverage = estimated-free ∩ true-free / true free floor.

### Goals

| world | config goals arrived | config-goal rejection reasons | unsupported goals rejected | supported accepted | supported arrived | supported final true distance m | contact episodes (whole run) |
|---|---|---|---|---|---|---|---|
| home_a | 1/4 | goal_in_unknown_space | 3/3 | 4/4 | 3/4 | 1.64, 0.03, 0.12, 0.01 | 0 |
| heldout_b | 1/4 | goal_in_unknown_space | 3/3 | 4/4 | 4/4 | 0.08, 0.06, 0.06, 0.06 | 0 |
| heldout_c | 2/4 | goal_in_unknown_space | 3/3 | 3/4 | 1/4 | 0.13, 0.16, 0.76, nan | 0 |
| home_a_dim | 0/4 | goal_in_unknown_space, goal_lacks_footprint_clearance | 3/3 | 4/4 | 4/4 | 0.06, 0.07, 0.05, 0.06 | 0 |
| home_a s1 | 0/4 | autonomy_not_enabled_or_not_localized | 3/3 | 0/0 | 0/0 | — | 0 |
| heldout_b s1 | 0/4 | goal_in_unknown_space, goal_lacks_footprint_clearance | 3/3 | 2/2 | 0/2 | 0.40, 0.46 | 0 |
| heldout_c s1 | 0/4 | autonomy_not_enabled_or_not_localized | 3/3 | 0/0 | 0/0 | — | 0 |
| home_a_dim s1 | 1/4 | goal_in_unknown_space | 3/3 | 4/4 | 4/4 | 0.06, 0.06, 0.07, 0.06 | 0 |
| home_a s2 | 1/4 | goal_in_unknown_space, goal_lacks_footprint_clearance | 3/3 | 4/4 | 4/4 | 0.06, 0.06, 0.08, 0.06 | 0 |
| heldout_b s2 | 0/4 | goal_in_unknown_space, goal_lacks_footprint_clearance | 3/3 | 3/3 | 0/3 | 0.16, 0.22, 0.17 | 0 |
| heldout_c s2 | 2/4 | goal_in_unknown_space, goal_lacks_footprint_clearance | 3/3 | 1/4 | 0/4 | 1.56, nan, nan, nan | 0 |
| home_a_dim s2 | 1/4 | goal_in_unknown_space | 3/3 | 4/4 | 0/4 | 0.24, 0.35, 0.29, 0.60 | 0 |

Config goals are fixed world points chosen before the run (some lie in space the robot never certified; rejecting those is the conservative outcome). Supported goals are seeded samples of the robot's own certified-traversable map (what an operator would click); arrival is scored in the true world (≤ 0.15 m).

### Fault tests

| world | fault | result | status/reason | revocations | recovery / operator | contact episodes |
|---|---|---|---|---|---|---|
| home_a | lens blackout | not_arrived | failed | recovery_succeeded | resent: goal_accepted | 0 |
| home_a | obstacle on route | not_arrived | failed | — | — | 0 |
| heldout_b | lens blackout | arrived | arrived | recovery_succeeded | resent: goal_accepted | 0 |
| heldout_b | obstacle on route | not_arrived | failed | — | — | 0 |
| heldout_c | lens blackout | skipped | did_not_reach_start | — | — | 0 |
| heldout_c | obstacle on route | skipped | did_not_reach_start | — | — | 0 |
| home_a_dim | lens blackout | arrived | arrived | recovery_succeeded | resent: goal_accepted | 0 |
| home_a_dim | obstacle on route | not_arrived | failed | — | — | 0 |
| home_a s1 | lens blackout | skipped | no_supported_pair_in_current_map | — | — | 0 |
| home_a s1 | obstacle on route | skipped | no_supported_pair_in_current_map | — | — | 0 |
| heldout_b s1 | lens blackout | skipped | no_supported_pair_in_current_map | — | — | 0 |
| heldout_b s1 | obstacle on route | skipped | no_supported_pair_in_current_map | — | — | 0 |
| heldout_c s1 | lens blackout | skipped | no_supported_pair_in_current_map | — | — | 0 |
| heldout_c s1 | obstacle on route | skipped | no_supported_pair_in_current_map | — | — | 0 |
| home_a_dim s1 | lens blackout | arrived | arrived | recovery_succeeded | resent: goal_accepted | 0 |
| home_a_dim s1 | obstacle on route | not_arrived | failed | — | — | 0 |
| home_a s2 | lens blackout | arrived | arrived | recovery_succeeded | resent: goal_accepted | 0 |
| home_a s2 | obstacle on route | arrived | arrived | — | — | 0 |
| heldout_b s2 | lens blackout | skipped | no_supported_pair_in_current_map | — | — | 0 |
| heldout_b s2 | obstacle on route | skipped | no_supported_pair_in_current_map | — | — | 0 |
| heldout_c s2 | lens blackout | skipped | no_supported_pair_in_current_map | — | — | 0 |
| heldout_c s2 | obstacle on route | skipped | no_supported_pair_in_current_map | — | — | 0 |
| home_a_dim s2 | lens blackout | skipped | did_not_reach_start | — | — | 0 |
| home_a_dim s2 | obstacle on route | skipped | did_not_reach_start | — | — | 0 |

### Interventions

* home_a: 2 operator enables (start mapping, after recovery_succeeded); mapping revocations: 0
* heldout_b: 2 operator enables (start mapping, after recovery_succeeded); mapping revocations: 0
* heldout_c: 1 operator enables (start mapping); mapping revocations: 0
* home_a_dim: 2 operator enables (start mapping, after recovery_succeeded); mapping revocations: 0
* home_a s1: 10 operator enables (start mapping, not tracking after recovery, not tracking after recovery, before goal, before goal, before goal, before goal, before goal, before goal, before goal); mapping revocations: 1
* heldout_b s1: 2 operator enables (start mapping, continue mapping); mapping revocations: 1
* heldout_c s1: 10 operator enables (start mapping, not tracking after recovery, not tracking after recovery, before goal, before goal, before goal, before goal, before goal, before goal, before goal); mapping revocations: 1
* home_a_dim s1: 4 operator enables (start mapping, continue mapping, continue mapping, after recovery_succeeded); mapping revocations: 2
* home_a s2: 2 operator enables (start mapping, after recovery_succeeded); mapping revocations: 0
* heldout_b s2: 1 operator enables (start mapping); mapping revocations: 0
* heldout_c s2: 5 operator enables (start mapping, continue mapping, before goal, before goal, before goal); mapping revocations: 1
* home_a_dim s2: 2 operator enables (start mapping, continue mapping); mapping revocations: 1
