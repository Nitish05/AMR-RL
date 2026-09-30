# Onboard visual SLAM, mapping and navigation

Everything in this document consumes only the robot-mounted RGB image, the
camera calibration (intrinsics + mounting) and the robot's own commanded body
motion. No depth, segmentation, simulator pose, room geometry or external camera
enters the pipeline (`tests/test_privilege_boundary.py`).

## 1. Pose estimation and relocalization (`perception/vslam.py`)

* **Features:** ORB (900 per 320×240 frame) on CLAHE-normalised grayscale.
* **State:** planar pose (x, y, θ) of `base_link` in the map frame. The map frame
  origin is the robot's pose when the map was started.
* **Tracking:** three prediction hypotheses (constant velocity, static, and the
  robot's own last commanded (v, ω) — proprioceptive knowledge of its action, not
  a sensor). For each: guided descriptor matching of projected map landmarks
  (radius 22 px, then 50 px), robust Gauss–Newton on the planar pose with an
  analytic Jacobian and Huber weights, established landmarks weighted higher
  than new ones, then a tight re-match (8 px) and re-optimisation. The
  hypothesis with most inliers wins; a confident first hypothesis ends the search.
* **Keyframes and landmarks:** on 0.10 m / 10° motion (or weak tracking after
  some motion), never from a weakly constrained pose. New landmarks come from
  two-view triangulation between keyframes (parallax ≥ 1.5°, reprojection
  < 2 px) and from inverse perspective mapping (IPM) of unmatched features below
  the horizon (≤ 2.2 m). IPM landmarks are tentative until re-observed from a
  camera position ≥ 4 cm away.
* **Windowed bundle adjustment:** the newest 6 keyframes (2 held fixed) and
  their shared landmarks are refined with sparse robust least squares after each
  keyframe; the result is rejected if the cost does not fall or a pose jumps.
* **Brief visual loss:** up to 1.5 s (and position σ ≤ 8 cm) the pose is
  dead-reckoned from the robot's own commands and reported as `predicted`.
  Ordinary navigation holds still on predicted poses; only the bounded nudge
  primitive may continue. Beyond those bounds the status is `lost`.
* **Relocalization:** global descriptor matching against the whole map +
  PnP-RANSAC (iterative, SQPnP fallback — EPnP is degenerate on the mostly
  coplanar floor landmarks), accepted only if the solution is upright at the
  known base height and repeats on two consecutive frames.
* **Persistence:** `save()`/`load()` write landmarks + keyframe poses with the
  map version and calibration id; a map built with a different calibration is
  refused. A loaded map starts in `relocalizing`.

### Monocular metric scale

A single camera cannot observe metric scale. AMR-RL resolves it with an
explicit, documented calibration assumption: **the floor is flat and the camera
optical centre is 0.155 m above it** (mechanical mounting from the robot spec).
Floor landmarks are metric through IPM; triangulated landmarks inherit that
scale through keyframe poses. The simulator's scale is never consulted.

Validation (`tests/test_perception_units.py::test_scale_prior_error_scales_ground_distances`):
a +10 % height error scales IPM ranges by +10 %; the evaluation reports the
similarity-alignment scale of every run (see NAVIGATION_RESULTS.md). A physical
robot would need the height/pitch measured (±5 mm / ±0.5°) or a calibration
target; wheel odometry fusion is a proposed extension (ledger), not used here.

### Failure handling (added after the first evaluation round)

* **Frozen-estimate check:** if at least 12 cm of motion was commanded in the
  last 1.2 s but vision reports less than a quarter of it, tracking is declared
  lost at once (the 3 s consistency window alone let a wrong lock run for about
  0.25 m).
* **Poisoned landmarks:** on any loss of tracking, landmarks created since the
  start of the suspect window are removed; otherwise relocalisation tends to
  lock back onto the wrong pose they were built from.
* **Dead-reckoning gate:** after a loss, relocalisation must agree with the
  commanded motion integrated from the last trusted pose (position and
  heading; the gates widen with commanded travel and expire after 20 s).
* **Measured effect** ([results/vslam-benchmark.md](results/vslam-benchmark.md), seven
  300 s exploration scenarios): mean ATE 9.7 → 4.9 cm, entirely from preventing
  one catastrophic failure (arena, seed 0: 43.9 → 9.2 cm, worst error 158 → 31 cm);
  the other six scenarios are unchanged within noise. One slow heading drift
  (arena seed 2, 14.9 cm) and one permanent loss after bounded recovery gave up
  (heldout_c, 175 s lost, waiting for the operator) occur with or without it.
* Stricter relocalisation (60 inliers, 3 confirmations) was **rejected**: it left
  the robot lost for 160 s in one scenario.
* Evaluated and **rejected**: dropping the "static" prediction hypothesis
  while the robot is driven (mean ATE 9.2 cm vs 3.8 cm over four scenarios in
  `scripts/dev/slam_bench.py`), and a score penalty for hypotheses far from the
  commanded motion (no measurable effect); both remain as configuration
  switches, off by default.

### Known limitations

No loop closure or pose-graph optimisation: drift accumulates (≈1–2 % of path in
most runs; a slow heading bias can reach 10–15 cm in 300 s). A wrong
relocalisation (accepted with 30–60 inliers but 10–20 cm off) is the dominant
large-error mode, and the robot cannot relocalise while standing still facing an
unmatched view after bounded recovery gives up (it then waits for the operator). Landmarks are not removed when
the world changes. Fast rotations close to textureless surfaces degrade tracking.

## 2. Map reconstruction and traversability (`mapping/occupancy.py`)

A sparse landmark map is **not** an obstacle map. Free and occupied evidence come
from dense two-view **plane-induced parallax**:

1. For a new keyframe and up to 3 recent + 1 wide-baseline earlier keyframe, the
   floor plane induces a homography between the images. It is estimated from
   matched features and accepted only if it agrees with the pose-predicted floor
   homography (or, for keyframes < 8 s apart, the pose-predicted one is used).
2. **Detectability gate:** for every pixel, compute where a just-detectable
   obstacle (3 cm tall, standing just in front of the floor point on the same
   ray) would appear in the other image. If that displacement is < 3.5 px the
   pixel can provide no evidence (UNKNOWN), no matter how well it matches.
3. **Discriminative floor test:** textured pixels whose NCC at the floor
   alignment is ≥ 0.80 **and** at least 0.20 higher than at the obstacle
   alignments become floor evidence; low-NCC textured pixels are non-floor.
4. Floor pixels' IPM points (≤ 2.6 m) add free evidence; the lowest non-floor run
   above floor in each column marks an obstacle base, applied only where at least
   two keyframe pairs agree within one cell (single-pair bases were 91 % on open
   floor, `scripts/dev/obstacle_hits_probe.py`); triangulated landmarks
   between 4 and 45 cm high and detected fixtures add occupied evidence; the
   robot's own body footprint along its path adds free evidence; an explicit
   operator attestation may mark a 0.32 m start disc free (recorded in state).
5. Log-odds cells (5 cm; free hit −0.45, obstacle hit +0.9, clamped to ±4): FREE ≤ −1.3, OCCUPIED ≥ +1.0, else UNKNOWN.

Textureless, far, never-seen or contradictory regions stay UNKNOWN.

## 3. Planning and wheel control

* `navigation/planner.py`: a robot-centre cell is traversable only if every
  cell within the circumscribed footprint radius (0.262 m + 0.03 m margin) is
  FREE. A* with a clearance cost and optional soft keep-out regions (learned
  aversions add cost; they never create free space). Goal rejections carry a
  reason: `goal_outside_map`, `goal_in_unknown_space`, `goal_occupied`,
  `goal_lacks_footprint_clearance`, `goal_unreachable_through_certified_space`.
* `navigation/navigator.py`: pure pursuit (≤ 0.22 m/s, ≤ 1 rad/s), rotate in
  place when the heading error exceeds 0.7 rad; stops when position σ exceeds
  6 cm. Every control step the next 0.6 m of path must still be certified
  traversable (else stop and replan; the whole path is re-validated every
  1.5 s, at most 6 replans). The robot turns in place only if no cell with
  obstacle evidence lies within its turning circle (0.262 m + 2 cm); otherwise
  it reverses straight, at most 0.4 m, or reports `no_certified_room_to_turn`.
  If an accepted goal loses its certification while driving (new evidence near
  it), a replan moves it to the nearest certified cell within 12 cm; new goals
  are still checked strictly.
* Remembered entities are hard keep-out discs for planning (half their measured
  size plus their position uncertainty; the object being approached keeps only
  its body so standoffs stay reachable).
* A newly placed object on well-mapped floor needs 3 obstacle observations to
  take a saturated free cell out of FREE (free evidence saturates at −4). Lower
  free saturation (−2, −3) was tried and rejected: stray obstacle hits then eroded
  certified space and goal clearance during live operation.

## 4. Exploration

Frontier-based: FREE cells next to UNKNOWN, split into 0.6 m tiles per connected
frontier (a frontier that rings the known area would otherwise have its centroid
in free space); the goal is a traversable
cell with ≥ 0.45 m clearance (turning in place close to a wall is a degenerate
view for monocular tracking) facing the frontier; on arrival the robot sweeps
±50°. Exploration value is scaled by recent map growth per trip (engineered
progress signal), so it fades when trips stop producing new free space.
