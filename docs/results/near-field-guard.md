# Near-field guard for objects placed on the route (negative result)

**Status: not enabled.** `RuntimeConfig.near_field_guard` defaults to `False`; the
code is kept in `src/amr_rl/navigation/near_field.py` so the measurements below can
be reproduced. The round-2 failure (4 of 5 box-on-route tests ended in contact) is
still open.

## Reproducing the failure

`scripts/dev/turn_then_box_probe.py` loads the map a navigation run saved
(`navigation-20260930-154854/home_a`), relocalises, drives to a certified point,
then sets a goal 1.5 m away in a direction `turn` degrees from the robot's heading
with a 35 cm box `dist` metres along that line (both inside certified free space).
This is the round-2 geometry: turn in place first (no parallax), then drive.
Ground truth places the box and scores contacts only.

| Scenario (turn, box distance) | Guard off (current code) |
|---|---|
| 180°, 0.60 m | contact 4.5 s after the goal; tracking lost; recovery exhausted |
| 180°, 0.75 m | contact after 7.7 s |
| 135°, 0.60 m | contact after 3.7 s |
| 135°, 0.75 m | contact after 19.9 s |
| 90°, either | skipped: no straight certified 1.5 m line |

**4/4 scenarios that ran ended in contact.** With the guard in observe-only mode
the runs are identical (the probe changes nothing).

## Why the map reacts too late

The occupancy grid needs about three keyframes of obstacle evidence (one keyframe
per 10 cm of travel, two agreeing keyframe pairs each) to overturn free evidence
saturated at −4. After an in-place turn there is no translation and so no
parallax. The box face starts 0.3–0.45 m ahead of the bumper, and the robot
reaches it first.

## Variant A: assert obstacle bases from per-frame probes (rejected)

Every 0.3 s, run the plane-parallax floor test between the current frame and a
keyframe with ≥ 4 cm baseline. Write obstacle-base cells that two successive probes
agree on into the grid at once.

* In one run, before the box was even placed, it asserted **83 cells on open floor**
  (24 events in 50 probes). Every later trip to a start point failed as
  unreachable, so the scenario could not run.
* Observe-mode recordings scored against ground truth (`scripts/dev/guard_rules.py`):
  the rule "two probes agree" produced 13–63 false cells per scenario. The
  strictest rule tried (clusters of ≥ 8 points confirmed by 3 probes) still
  produced 4 false cells in each 135° scenario and detected the box in none of the
  four. The box, when detected, was first confirmed 0.31–0.39 m from the robot
  centre.

## Variant B: drive only where the floor ahead was just verified (rejected)

Using the same probes, compute the fraction of the path band 0.30–0.50 m ahead
(±0.10 m) that was verified as floor within 1 s. Full speed above 0.75; creep at
0.07 m/s below; stop after two probes below 0.50; end the goal as blocked after
2 s stopped.

| | Result |
|---|---|
| 180°, 0.60 m | **no contact**: stopped 0.28 m (centre to box edge), goal blocked |
| 180°, 0.75 m | **contact** after 13.3 s: with the box ahead the band still verified at 0.45–0.60, so it crept into it |
| Ordinary trips on open floor | **4 of 5 failed** as `path_ahead_not_verified` in one run; open-floor bands verify at 0.45–1.0 |

## Conclusion

At 0.3–0.5 m the plane-parallax floor test is not discriminative enough for a
per-frame stop decision. On open floor it often fails to verify floor, and with
a box ahead it still verifies some floor beside the box. A dedicated near-field
cue is needed: a monocular depth or floor-segmentation model behind the same
camera-only contract, a wider baseline (a short sideways arc before driving), or a
physical bumper on hardware. The reproducible scenarios above are the acceptance
test for whichever is tried next.
