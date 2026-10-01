# Nudge reach: planned creep vs true gap

Probe: `scripts/dev/nudge_reach_probe.py arena` (evaluation only; the true pose is
used to measure geometry, never by the robot). The robot is placed at the three
nudge standoff ranges (0.47, 0.52, 0.58 m) and nine approach angles around each
arena fixture, facing it. For every view in which the fixture is fully detected,
the planned forward creep is compared with the true gap between the robot's front
and the fixture surface along its heading. A nudge reaches the fixture when
creep ≥ gap.

| Fixture | Views | Reaches: old creep (centre − radius) | Reaches: new creep (contact edge) | New: depth past surface, mean / max |
|---|---|---|---|---|
| bloom (cylinder) | 27 | 27 | 27 | 3.4 / 3.5 cm |
| grump (box) | 27 | **21** | 27 | 3.2 / 5.0 cm |
| stone (box) | 27 | **23** | 27 | 3.2 / 5.0 cm |
| roller (ball) | 27 | 27 | 27 | 5.9 / 7.9 cm |

All misses of the old rule are boxes seen corner-on (45° and 315°): the silhouette
fill drops below 0.83, the detector treats the object as round and places its
centre at its front edge, and subtracting a radius again stops the creep about
7 cm short. The new rule uses the range to the nearest visible floor contact,
which does not depend on guessing the shape. It pushes a ball up to one radius
deeper (the ball rolls). The intended push depth is 3 cm.

In round 2 the fixed-order baseline nudged grump 53 times and touched it 0
times (its closest approach left a ~10 cm gap), which is how this was found.
The fix is unit-tested (`test_nudge_creep_uses_the_measured_contact_edge`) and
has not yet been through a full learning evaluation.
