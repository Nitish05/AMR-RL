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
(manual commands, counted) so it can relocalise. Arrival = within 0.25 m of the
true goal **and** the navigator reports arrival. `home_a` is the development room;
the other three were never used for tuning.

Evidence: `work/evidence/navigation-20260930-154854` (12 runs, 0 crashed). Regenerate:
`python scripts/eval/report.py --nav work/evidence/navigation-20260930-154854 --out <file>`.

## Summary (round 2: 4 rooms × 3 seeds)

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

Evidence: `work/evidence/navigation-20260930-154854`

Runs: 12; crashed runs: 0 

### Summary by room (all seeds)

| room | split | seeds | own-map goals arrived | impossible goals rejected | ATE cm (range) | worst error cm | free coverage | deep false-free | contact episodes | fault tests passed / run |
|---|---|---|---|---|---|---|---|---|---|---|
| home_a | development | 3 | 8/8 | 9/9 | 0.4–2.9 | 6.9 | 31%–41% | 0–16 | 3 | 2/4 (2 skipped) |
| heldout_b | heldout_layout | 3 | 3/8 | 9/9 | 2.0–16.4 | 27.7 | 29%–33% | 8–13 | 0 | 2/3 (3 skipped) |
| heldout_c | heldout_appearance | 3 | 7/8 | 9/9 | 3.0–17.5 | 45.0 | 71%–73% | 11–103 | 4 | 1/3 (3 skipped) |
| home_a_dim | heldout_lighting | 3 | 4/8 | 9/9 | 0.9–4.9 | 9.3 | 34%–43% | 5–13 | 1 | 1/2 (4 skipped) |

**All rooms and seeds:** own-map goals arrived 22/32; impossible goals rejected 36/36; contact episodes 8.


### Mapping by exploration (onboard RGB only)

| world | split | sim s | path m | ATE m | max err m | scale | tracking frames | lost frames | free coverage | free cells | false-free cells | deep false-free | contact episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| home_a | development | 300.000 | 9.495 | 0.023 | 0.069 | 1.005 | 3000/3000 | 0 | 0.380 | 3006 | 97 | 16 | 0 |
| heldout_b | heldout_layout | 300.000 | 5.726 | 0.020 | 0.040 | 0.999 | 3000/3000 | 0 | 0.332 | 2455 | 35 | 8 | 0 |
| heldout_c | heldout_appearance | 300.000 | 9.552 | 0.030 | 0.057 | 0.995 | 3000/3000 | 0 | 0.712 | 5519 | 65 | 11 | 0 |
| home_a_dim | heldout_lighting | 300.000 | 8.875 | 0.009 | 0.093 | 0.999 | 2883/3000 | 117 | 0.343 | 2697 | 67 | 5 | 0 |
| home_a s1 | development | 169.500 | 3.118 | 0.004 | 0.012 | 0.997 | 1113/1695 | 582 | 0.306 | 2344 | 2 | 0 | 0 |
| heldout_b s1 | heldout_layout | 300.000 | 5.748 | 0.164 | 0.277 | 0.907 | 2968/3000 | 32 | 0.299 | 2213 | 30 | 8 | 0 |
| heldout_c s1 | heldout_appearance | 300.000 | 10.852 | 0.175 | 0.450 | 0.991 | 3000/3000 | 0 | 0.732 | 5767 | 159 | 79 | 0 |
| home_a_dim s1 | heldout_lighting | 300.000 | 7.522 | 0.014 | 0.036 | 1.008 | 2873/3000 | 127 | 0.434 | 3350 | 26 | 9 | 0 |
| home_a s2 | development | 300.000 | 7.428 | 0.029 | 0.049 | 1.002 | 2931/3000 | 69 | 0.410 | 3229 | 95 | 15 | 0 |
| heldout_b s2 | heldout_layout | 300.000 | 5.410 | 0.138 | 0.259 | 1.068 | 3000/3000 | 0 | 0.287 | 2134 | 38 | 13 | 0 |
| heldout_c s2 | heldout_appearance | 300.000 | 8.727 | 0.082 | 0.142 | 0.979 | 2948/3000 | 52 | 0.713 | 5675 | 222 | 103 | 0 |
| home_a_dim s2 | heldout_lighting | 300.000 | 6.220 | 0.049 | 0.079 | 1.003 | 3000/3000 | 0 | 0.415 | 3253 | 78 | 13 | 0 |

False-free = estimated FREE where the true room is occupied or outside it; deep = more than 10 cm inside an obstacle/wall. Coverage = estimated-free ∩ true-free / true free floor.

### Goals

| world | config goals arrived | config-goal rejection reasons | unsupported goals rejected | supported accepted | supported arrived | supported final true distance m | contact episodes (whole run) |
|---|---|---|---|---|---|---|---|
| home_a | 1/4 | goal_in_unknown_space | 3/3 | 4/4 | 4/4 | 0.05, 0.06, 0.06, 0.07 | 1 |
| heldout_b | 0/4 | goal_in_unknown_space, goal_lacks_footprint_clearance | 3/3 | 2/2 | 2/2 | 0.05, 0.05 | 0 |
| heldout_c | 2/4 | goal_lacks_footprint_clearance | 3/3 | 4/4 | 4/4 | 0.14, 0.09, 0.03, 0.08 | 0 |
| home_a_dim | 0/4 | autonomy_not_enabled_or_not_localized | 3/3 | 0/0 | 0/0 | — | 0 |
| home_a s1 | 0/4 | autonomy_not_enabled_or_not_localized | 3/3 | 0/0 | 0/0 | — | 0 |
| heldout_b s1 | 0/4 | goal_in_unknown_space, goal_lacks_footprint_clearance | 3/3 | 3/3 | 1/3 | 0.16, 0.17, 0.13 | 0 |
| heldout_c s1 | 0/4 | autonomy_not_enabled_or_not_localized, goal_lacks_footprint_clearance | 3/3 | 0/0 | 0/0 | — | 2 |
| home_a_dim s1 | 1/4 | goal_in_unknown_space | 3/3 | 4/4 | 2/4 | 0.05, 0.07, 1.94, 2.21 | 1 |
| home_a s2 | 1/4 | goal_in_unknown_space | 3/3 | 4/4 | 4/4 | 0.07, 0.06, 0.06, 0.04 | 2 |
| heldout_b s2 | 0/4 | goal_in_unknown_space, goal_lacks_footprint_clearance | 3/3 | 3/3 | 0/3 | 0.34, 0.31, 0.21 | 0 |
| heldout_c s2 | 3/4 | goal_in_unknown_space | 3/3 | 4/4 | 3/4 | 0.05, 0.21, 0.09, 0.03 | 2 |
| home_a_dim s2 | 1/4 | goal_in_unknown_space | 3/3 | 4/4 | 2/4 | 0.06, 0.06, 0.22, 0.22 | 0 |

Config goals are fixed world points chosen before the run (some lie in space the robot never certified; rejecting those is the conservative outcome). Supported goals are seeded samples of the robot's own certified-traversable map (what an operator would click); arrival is scored in the true world (≤ 0.25 m).

### Fault tests

| world | fault | result | status/reason | revocations | recovery / operator | contact episodes |
|---|---|---|---|---|---|---|
| home_a | lens blackout | arrived | arrived | recovery_succeeded | resent: goal_accepted | 0 |
| home_a | obstacle on route | not_arrived | — | recovery_exhausted | — | 1 |
| heldout_b | lens blackout | arrived | arrived | recovery_succeeded | resent: goal_accepted | 0 |
| heldout_b | obstacle on route | not_arrived | — | recovery_succeeded | — | 0 |
| heldout_c | lens blackout | not_arrived | failed | recovery_succeeded | resent: goal_accepted | 0 |
| heldout_c | obstacle on route | skipped | did_not_reach_start | — | — | 0 |
| home_a_dim | lens blackout | skipped | no_supported_pair_in_current_map | — | — | 0 |
| home_a_dim | obstacle on route | skipped | no_supported_pair_in_current_map | — | — | 0 |
| home_a s1 | lens blackout | skipped | no_supported_pair_in_current_map | — | — | 0 |
| home_a s1 | obstacle on route | skipped | no_supported_pair_in_current_map | — | — | 0 |
| heldout_b s1 | lens blackout | not_arrived | arrived | recovery_succeeded | resent: goal_accepted | 0 |
| heldout_b s1 | obstacle on route | skipped | did_not_reach_start | — | — | 0 |
| heldout_c s1 | lens blackout | skipped | no_supported_pair_in_current_map | — | — | 0 |
| heldout_c s1 | obstacle on route | skipped | no_supported_pair_in_current_map | — | — | 0 |
| home_a_dim s1 | lens blackout | arrived | arrived | recovery_succeeded | resent: goal_accepted | 0 |
| home_a_dim s1 | obstacle on route | not_arrived | — | recovery_succeeded | — | 1 |
| home_a s2 | lens blackout | arrived | arrived | recovery_succeeded | resent: goal_accepted | 0 |
| home_a s2 | obstacle on route | not_arrived | — | recovery_exhausted | — | 2 |
| heldout_b s2 | lens blackout | skipped | did_not_reach_start | — | — | 0 |
| heldout_b s2 | obstacle on route | skipped | no_supported_pair_in_current_map | — | — | 0 |
| heldout_c s2 | lens blackout | arrived | arrived | recovery_succeeded | resent: goal_accepted | 0 |
| heldout_c s2 | obstacle on route | not_arrived | — | recovery_exhausted | — | 2 |
| home_a_dim s2 | lens blackout | skipped | did_not_reach_start | — | — | 0 |
| home_a_dim s2 | obstacle on route | skipped | did_not_reach_start | — | — | 0 |

### Interventions

* home_a: 2 operator enables (start mapping, after recovery_succeeded); mapping revocations: 0
* heldout_b: 2 operator enables (start mapping, after recovery_succeeded); mapping revocations: 0
* heldout_c: 2 operator enables (start mapping, after recovery_succeeded); mapping revocations: 0
* home_a_dim: 9 operator enables (start mapping, not tracking after recovery, before goal, before goal, before goal, before goal, before goal, before goal, before goal); mapping revocations: 0
* home_a s1: 10 operator enables (start mapping, not tracking after recovery, not tracking after recovery, before goal, before goal, before goal, before goal, before goal, before goal, before goal); mapping revocations: 1
* heldout_b s1: 4 operator enables (start mapping, continue mapping, continue mapping, after recovery_succeeded); mapping revocations: 2
* heldout_c s1: 6 operator enables (start mapping, before goal, before goal, before goal, before goal, before goal); mapping revocations: 0
* home_a_dim s1: 4 operator enables (start mapping, continue mapping, continue mapping, after recovery_succeeded); mapping revocations: 2
* home_a s2: 3 operator enables (start mapping, continue mapping, after recovery_succeeded); mapping revocations: 1
* heldout_b s2: 1 operator enables (start mapping); mapping revocations: 0
* heldout_c s2: 3 operator enables (start mapping, continue mapping, after recovery_succeeded); mapping revocations: 1
* home_a_dim s2: 1 operator enables (start mapping); mapping revocations: 0
