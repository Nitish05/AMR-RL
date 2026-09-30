# Navigation results (simulation, lockstep)

Navigation and learning are evaluated separately; learning results are in
[LEARNING_RESULTS.md](LEARNING_RESULTS.md).

**Protocol** (`scripts/eval/navigation.py`, one seed per world, fresh process and
throwaway memory per world): the robot starts with autonomy disabled; the
evaluator enables it once and it explores for 300 s of simulated time
(`explore_only` policy, onboard RGB only). Then, with activity selection off,
the evaluator acts as an operator: (1) fixed goals chosen before the run, some
deliberately unsupported (inside furniture or outside the room); (2) four
seeded "supported" goals sampled from interior cells of the robot's own
certified-traversable map; (3) a lens blackout during a goal (fault injection);
(4) a 35 cm box placed on the robot's own planned route. Ground truth is used
only to score. Arrival = true distance ≤ 0.25 m **and** the navigator reports
arrival. Development world: `home_a`. Held out (never used for tuning):
`heldout_b` (layout), `heldout_c` (appearance), `home_a_dim` (lighting).

Regenerate the tables: `python scripts/eval/report.py --nav work/evidence/navigation-20260930-024431 --out <file>`.

## Summary

* **Safety:** 0 contact episodes in all four worlds, including the
  obstacle-on-route test (home_a_dim: the robot stopped, then refused to turn in
  place next to the box — `no_certified_room_to_turn`).
* **Conservatism:** 12/12 unsupported goals were rejected (10 as
  `goal_in_unknown_space` because the robot never certified that space, 2 as
  `goal_occupied`). None of the 16 pre-chosen "reachable" goals counted as
  arrived: 12 were rejected up front, 4 were accepted and later blocked (one of
  those, in heldout_c, ended within 0.25 m but the navigator had stopped as
  blocked, so it does not count), because 300 s of exploration certified only 31–77 % of each
  room's free floor; rejecting them is the intended behaviour, but it means the
  robot cannot yet go everywhere an operator might expect.
* **Reaching its own map:** supported goals arrived **5/16** (home_a 1/4,
  heldout_b 0/4, heldout_c 0/4, home_a_dim 4/4). Two failure modes:
  (a) **VSLAM drift** — in heldout_b and heldout_c the robot "arrived" by its own
  estimate but was 0.20–0.41 m away in truth (mapping ATE 16.8 cm and 9.3 cm,
  max 28.5 cm and 20.9 cm); (b) **goal clearance lost to live evidence** — while
  driving, new obstacle evidence near the goal removed its footprint clearance
  and the navigator stopped (`blocked: goal_lacks_footprint_clearance`).
* **Tracking loss handling:** the two lens-blackout tests that ran both did
  localization_lost → bounded recovery → recovery_succeeded → explicit operator
  re-enable → goal resent (home_a_dim arrived; home_a then failed on goal
  clearance). Of 8 fault tests, 3 ran and 5 were **skipped** because no
  suitable start/end pair existed in the current map or the robot could not
  reach the start point (counted, not dropped).
* **Monocular scale:** the camera-height prior kept the similarity scale within
  0.967–1.003 in all worlds.

A superseded run of an earlier code state (`navigation-20260929-234925`, before the
corridor/turning safeguards) reached supported goals more often (home_a 3/4,
heldout_b 3/3) but drove into the evaluation box and clipped it while turning in
place. The current code trades arrival rate for not touching things; better
drift control and goal snapping are needed to recover the arrival rate
(see [CAPABILITY_LEDGER.md](CAPABILITY_LEDGER.md)).

Other retained, superseded runs and why they were superseded are listed in
each `work/evidence/*/SUPERSEDED.md`.

Evidence: `work/evidence/navigation-20260930-024431`

Worlds run: 4; crashed runs: 0 

### Mapping by exploration (onboard RGB only)

| world | split | sim s | path m | ATE m | max err m | scale | tracking frames | lost frames | free coverage | free cells | false-free cells | deep false-free | contact episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| home_a | development | 300.000 | 9.198 | 0.018 | 0.034 | 0.999 | 3000/3000 | 0 | 0.431 | 3357 | 60 | 2 | 0 |
| heldout_b | heldout_layout | 300.000 | 6.212 | 0.168 | 0.285 | 0.967 | 2996/3000 | 4 | 0.309 | 2300 | 46 | 16 | 0 |
| heldout_c | heldout_appearance | 300.000 | 10.432 | 0.093 | 0.209 | 1.003 | 2987/3000 | 13 | 0.773 | 6036 | 117 | 46 | 0 |
| home_a_dim | heldout_lighting | 300.000 | 8.008 | 0.014 | 0.031 | 0.998 | 2996/3000 | 4 | 0.380 | 2962 | 50 | 5 | 0 |

False-free = estimated FREE where the true room is occupied or outside it; deep = more than 10 cm inside an obstacle/wall. Coverage = estimated-free ∩ true-free / true free floor.

### Goals

| world | config goals arrived | config-goal rejection reasons | unsupported goals rejected | supported accepted | supported arrived | supported final true distance m | contact episodes (whole run) |
|---|---|---|---|---|---|---|---|
| home_a | 0/4 | goal_in_unknown_space, goal_lacks_footprint_clearance | 3/3 | 3/4 | 1/4 | 0.56, 0.05, nan, 0.30 | 0 |
| heldout_b | 0/4 | goal_in_unknown_space, goal_lacks_footprint_clearance | 3/3 | 4/4 | 0/4 | 0.26, 0.41, 0.30, 1.23 | 0 |
| heldout_c | 0/4 | goal_lacks_footprint_clearance | 3/3 | 4/4 | 0/4 | 0.57, 0.25, 0.34, 0.20 | 0 |
| home_a_dim | 0/4 | goal_in_unknown_space | 3/3 | 4/4 | 4/4 | 0.06, 0.06, 0.06, 0.07 | 0 |

Config goals are fixed world points chosen before the run (some lie in space the robot never certified; rejecting those is the conservative outcome). Supported goals are seeded samples of the robot's own certified-traversable map (what an operator would click); arrival is scored in the true world (≤ 0.25 m).

### Fault tests

| world | fault | result | status/reason | revocations | recovery / operator | contact episodes |
|---|---|---|---|---|---|---|
| home_a | lens blackout | not_arrived | failed | recovery_succeeded | resent: goal_accepted | 0 |
| home_a | obstacle on route | skipped | did_not_reach_start | — | — | 0 |
| heldout_b | lens blackout | skipped | no_supported_pair_in_current_map | — | — | 0 |
| heldout_b | obstacle on route | skipped | no_supported_pair_in_current_map | — | — | 0 |
| heldout_c | lens blackout | skipped | did_not_reach_start | — | — | 0 |
| heldout_c | obstacle on route | skipped | did_not_reach_start | — | — | 0 |
| home_a_dim | lens blackout | arrived | arrived | recovery_succeeded | resent: goal_accepted | 0 |
| home_a_dim | obstacle on route | not_arrived | failed | — | — | 0 |

### Interventions

* home_a: 2 operator enables (start mapping, after recovery_succeeded); mapping revocations: 0
* heldout_b: 3 operator enables (start mapping, continue mapping, continue mapping); mapping revocations: 2
* heldout_c: 3 operator enables (start mapping, continue mapping, continue mapping); mapping revocations: 2
* home_a_dim: 4 operator enables (start mapping, continue mapping, continue mapping, after recovery_succeeded); mapping revocations: 2
