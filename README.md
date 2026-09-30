# AMR-RL

**A wheeled indoor robot whose later choices depend on what it has seen happen.**

AMR-RL simulates "Pip", a compact differential-drive robot with one forward RGB
camera and an expressive screen. Using only its own camera, Pip maps a room,
navigates conservatively, and interacts with objects. It records what each
action actually did, as seen in its own images, and uses that remembered
experience to decide what to do next — including after a restart.

Everything here runs in simulation (Genesis World 1.3.2, CPU backend,
lockstep). Nothing has run on hardware.

<p align="center">
  <img src="docs/media/robot-inspection.png" width="46%" alt="Pip beside a fixture, third-person inspection view">
  <img src="docs/media/onboard-frame.png" width="40%" alt="What Pip's own camera sees">
</p>
<p align="center"><em>Left: Pip (inspection camera, never used by the robot), showing its "signal" pattern on the screen.
Right: the same moment from Pip's onboard camera — the only sensor it perceives with.</em></p>

> **What this is not.** It is not a claim of consciousness, emotion or
> self-created desire. What counts as a good or bad consequence is
> **engineered** (a fixed valence table and a "stimulation need"). What the
> robot **learns** is which of its actions, on which objects, in which visible
> state, produce which consequences — and that changes what it chooses later.

---

## Contents

1. [Results summary](#results-summary)
2. [How it works](#how-it-works)
3. [Navigation results](#navigation-results)
4. [Learning results](#learning-results)
5. [Limitations and next gates](#limitations-and-next-gates)
6. [Quick start](#quick-start)
7. [Reproducing the evaluations](#reproducing-the-evaluations)
8. [Repository layout](#repository-layout)
9. [Honesty notes](#honesty-notes)

---

## Results summary

All numbers come from one seed per experiment, so they show what happened,
not statistics. Navigation and learning were evaluated separately. Full tables
with every denominator are in
[docs/NAVIGATION_RESULTS.md](docs/NAVIGATION_RESULTS.md) and
[docs/LEARNING_RESULTS.md](docs/LEARNING_RESULTS.md).

| Question | Result |
|---|---|
| Does remembered experience change later choices? | **Yes.** Trained in opposite worlds, then restarted: one robot's first two choices were bloom/signal, the other's stone/signal. A robot with no memory chose roller/signal and grump/signal. |
| Does it survive a restart without keeping motion permission? | **Yes.** Knowledge persisted; autonomy started disabled every time and was re-enabled only after relocalizing from fresh images (0.2 s). |
| Does it adapt when the rules change? | **Partly — 1 of 2.** After a late rule swap it found the new rewarding option 48 s later. After an early swap it never did in 590 s. |
| Is it better than simple baselines? | **Per interaction, yes.** Valence per observed outcome: learned 0.45, fixed 0.38, random 0.18, nearest 0.02. **In total, no:** the fixed-order baseline collected 12.6 vs 5.0, because it never rests and happened to repeat the rewarding option. |
| Does it avoid what hurt it? | **Partly.** The fixture that showed a red panel was never engaged again, but it was also registered as a duplicate identity, and the duplicate was re-targeted (every such attempt failed at navigation). |
| Does it touch things while navigating? | **No contact** in any of the 4 navigation rooms, including when a box was placed on its route. In long learning runs there were unintended contacts (7 episodes across 2 of 11 runs). |
| Does it refuse goals it cannot justify? | **Yes.** 12/12 goals inside furniture or outside the room were rejected; unknown space is never treated as free. |
| Can it reach goals in its own map? | **Not reliably: 5/16.** 4/4 in one held-out room; 0/4 in two others, because of visual-odometry drift (up to 29 cm) and goals losing clearance to new evidence. |

![Policy comparison](docs/media/policy-comparison.png)

---

## How it works

```
Genesis World (physics + rendering) ── sim/world.py: rooms, fixtures + hidden response rules
      │ onboard RGB (camera_link, 320×240, 10 Hz)      ▲ wheel velocity targets
      ▼                                                │
perception/vslam.py ──pose──► mapping/occupancy.py ──► navigation/planner + navigator
perception/entities.py ──► learning/identity.py ──► learning/memory.py (SQLite + image receipts)
perception/semantic.py (async; labels only, stale/revoked results rejected)
behavior/chooser.py + activities.py ──► control/supervisor.py ──► control/genesis_backend.py
expression/policy.py + screen.py ──► robot screen        ui/server.py + static/ ──► operator console
sim/evaluator.py (ground truth) ── scoring only; never imported by runtime code (tested)
```

### The robot

A generated URDF and GLB meshes, built from an editable spec
([`assets/robot/amr_spec.yaml`](assets/robot/amr_spec.yaml)):

| Item | Value |
|---|---|
| Footprint | 0.35 m × 0.30 m, 0.339 m tall; planning radius 0.262 m (+0.03 m margin) |
| Drive | two driven wheels (r = 0.05 m, track 0.27 m) + passive ball caster; 5.08 kg |
| Actuation | wheel-joint velocity control only; body pose setters used only for explicit resets |
| Camera | pinhole 320×240, 60° vertical FOV, 0.155 m above the floor, pitched 12° down |
| Screen | rendered face on a mast behind the camera; the `signal` action is a real screen pattern that fixtures respond to |

Details: [docs/ROBOT.md](docs/ROBOT.md).

### Perception and mapping (camera only)

* **Monocular visual SLAM on a planar pose:** ORB features, several motion
  hypotheses, guided matching, robust Gauss–Newton pose estimation, keyframes,
  triangulated and floor landmarks, windowed bundle adjustment, relocalization
  against a saved map. **Metric scale** comes from the calibrated camera height
  over a flat floor. Nothing uses simulator poses, depth or labels.
* **Occupancy:** free and obstacle evidence from plane-induced parallax between
  keyframes, with a *detectability gate*: a pixel only gives evidence if a 3 cm
  obstacle there would move by ≥ 3.5 px. Textureless, far or never-seen space
  stays **unknown**, and unknown is never free.
* **Fixtures:** saturated-colour regions; a smaller region resting on a larger
  one is an *attachment* (e.g. a raised panel), which is the visible state.

Details: [docs/VSLAM.md](docs/VSLAM.md).

### Navigation and safety

* A cell is traversable only if every cell within the robot's footprint radius
  is certified free. A* with clearance costs; every rejected goal carries a
  reason (`goal_in_unknown_space`, `goal_occupied`,
  `goal_lacks_footprint_clearance`, …).
* Every control step, the next 0.6 m of path must still be certified, or the
  robot stops at once and replans (at most every 1.5 s). It turns in place only
  if no obstacle evidence lies inside its turning circle; otherwise it backs
  out along its path.
* A **supervisor** owns authority: Stop always wins, commands carry
  generations and expire in 0.2 s, and a missing operator heartbeat, stale
  camera frames or lost localization revoke motion. Lost tracking gets a
  bounded recovery (rotate ≤ 2π or reverse ≤ 0.45 m); **autonomy never resumes
  on its own**.

### Interaction and learning

Four engineered fixtures with hidden rules (in [`configs/consequences/`](configs/consequences)):

| Fixture | Looks like | Standard rule |
|---|---|---|
| bloom | cyan cylinder | `signal` → a yellow panel rises for 8 s |
| grump | green box | `signal` or `nudge` → a red panel rises for 8 s |
| stone | blue box | nothing |
| roller | magenta ball | rolls when nudged (physics, no rule) |

Each interaction runs approach → confirm the target in ≥ 3 fresh frames
(recording the visible context and the **prediction**) → act (`signal`: show a
pattern for 2.2 s from 0.75 m; `nudge`: creep into contact) → observe for 2.5 s →
classify the **observed outcome from pixels** (`moved`, `attach:<colour>`,
`none`, or ambiguous → nothing learned). The receipt (frame hashes, identity
evidence, decision basis, prediction, context, outcome, map/calibration
versions, authority generation) is written with its images **before** the
memory update. Replays are no-ops.

The memory keeps a windowed, time-decayed Dirichlet posterior over outcomes for
each (entity, action, visible context), with change detection. The chooser
scores options as

```
value = need × E[valence] × novelty + curiosity × uncertainty × (0.3 + 0.7·need) − costs
```

and picks among **explore, investigate, engage, revisit, avoid and idle**, with
bounded reconsideration: at most every 3 s, at most 6 switches a minute, and
never in the middle of an interaction. Details: [docs/LEARNING.md](docs/LEARNING.md).

### Operator console

A loopback-only web console showing the onboard camera, the estimated map and
trajectory, localization confidence, the current activity and why it was
chosen (with competing options), entities and learned attitudes, predicted vs
observed outcomes, manual drive, memory reset and **Stop**. Below: a restarted
robot with a trained memory revisiting the fixture it learned to like
(expected +0.71) while listing the green fixture as disliked.

![Operator console with a trained memory](docs/media/console.png)

---

## Navigation results

**Protocol:** fresh process per room. The robot explores for 300 s of
simulated time using only its camera. Then an evaluator acts as the operator:
fixed goals (some deliberately impossible), four goals sampled from the
*interior* of the robot's own certified map, a camera blackout during a goal,
and a 35 cm box placed on the robot's own planned route. Arrival means ≤ 0.25 m
from the true goal **and** the navigator reporting arrival. `home_a` was used
for development; the other three rooms were never used for tuning.

![Estimated maps after 300 s of exploration](docs/media/nav-maps.png)

*Light = certified free, dark = unknown or occupied, blue = estimated
trajectory, rings = detected fixtures.*

### Mapping by exploration

| Room | Split | Path (m) | ATE (cm) | Max error (cm) | Scale | Frames tracked | Free coverage | False-free cells (deep) | Contacts |
|---|---|---|---|---|---|---|---|---|---|
| home_a | development | 9.2 | 1.8 | 3.4 | 0.999 | 3000/3000 | 43 % | 60 (2) | 0 |
| heldout_b | held-out layout | 6.2 | 16.8 | 28.5 | 0.967 | 2996/3000 | 31 % | 46 (16) | 0 |
| heldout_c | held-out appearance | 10.4 | 9.3 | 20.9 | 1.003 | 2987/3000 | 77 % | 117 (46) | 0 |
| home_a_dim | held-out lighting | 8.0 | 1.4 | 3.1 | 0.998 | 2996/3000 | 38 % | 50 (5) | 0 |

*ATE: absolute trajectory error (RMS) against ground truth. False-free:
estimated free where the true room is occupied; "deep" means more than 10 cm
inside an obstacle.*

### Goals and faults

| Room | Impossible goals rejected | Pre-chosen goals arrived | Own-map goals arrived | True distance at end (m) | Blackout test | Obstacle-on-route test |
|---|---|---|---|---|---|---|
| home_a | 3/3 | 0/4 | 1/4 | 0.56, 0.05, rejected, 0.30 | recovered, then blocked on goal clearance | skipped (start not reached) |
| heldout_b | 3/3 | 0/4 | 0/4 | 0.26, 0.41, 0.30, 1.23 | skipped | skipped |
| heldout_c | 3/3 | 0/4 | 0/4 | 0.57, 0.25, 0.34, 0.20 | skipped | skipped |
| home_a_dim | 3/3 | 0/4 | **4/4** | 0.06, 0.06, 0.06, 0.07 | **arrived** after recovery + re-enable | stopped, **no contact** |

**Reading these results:**

* **Safety held:** zero contact episodes in all four rooms. In the one
  obstacle test that ran, the robot stopped and then refused to turn in place
  beside the box.
* **Conservatism held:** every impossible goal was rejected. The cost is
  coverage: 300 s certified only 31–77 % of each room, so most pre-chosen
  goals were rejected too.
* **Two failure modes limit arrival.** (a) *Drift:* in heldout_b and heldout_c
  the robot believed it had arrived but was 0.20–0.41 m away. (b) *Clearance
  loss:* new obstacle evidence near a goal removed its footprint clearance
  mid-route.
* **Tracking loss handling works:** blackout → revoke → bounded recovery →
  explicit operator re-enable → goal resent.
* Of 8 fault tests, 3 ran and 5 were skipped because no valid start/end pair
  existed or the start could not be reached. Skipped tests stay in the
  denominator.
* An earlier code version (retained evidence) reached more own-map goals but
  drove into the test box. The current code gives up arrival rate to avoid
  touching things.

---

## Learning results

**Protocol:** every experiment starts from a fresh memory in the arena, with
the four fixtures and a map the robot built itself. Every phase is a new
process: autonomy starts disabled and is enabled only after relocalization.
Tests after a restart use *inert* rules, so what is measured is the choice
made from memory, not a fresh reward.

### Opposite histories and restart persistence

| Experiment | Training world | What it learned | First two choices after restart |
|---|---|---|---|
| history_a | standard (bloom rewards signal) | P(yellow \| bloom, signal) = 0.90 from 8 weighted outcomes | **bloom/signal, bloom/signal** |
| history_b | swapped (stone rewards signal) | P(yellow \| stone, signal) = 0.92 from 9 weighted outcomes | **stone/signal, stone/signal** |
| no_memory | — | nothing | roller/signal, grump/signal |

**Context matters.** Signalling bloom while its panel is still raised gives
`none`. Those outcomes are stored under the visible context `attach:yellow`,
so they do not erase the learned response in context `attach:none`. In
history_a, all 8 bloom outcomes follow this pattern.

### Policies under the same rules

| Policy | Attempts | Outcomes | Useful | Aversive (red) | Total valence | Valence / outcome | Idle |
|---|---|---|---|---|---|---|---|
| **Learned** | 19 | 11 | 6 | 1 | 5.0 | **0.45** | 27 % |
| Fixed order | 50 | 33 | 13 | 0 | **12.6** | 0.38 | 0 % |
| Random | 43 | 17 | 5 | 2 | 3.0 | 0.18 | 0 % |
| Nearest first | 41 | 34 | 1 | 0 | 0.6 | 0.02 | 0 % |

The learned policy chooses better per interaction but is not a reward-rate
maximizer: its engineered need drops after useful outcomes, so it rests. The
fixed-order baseline's advantage in total valence is luck of ordering: it
repeats whichever option sorts first, and here that was the rewarding one.

### Changing and unreliable consequences

| Experiment | What happened | Verdict |
|---|---|---|
| Late rule swap (after 10 outcomes, t = 360 s) | 3 failed bloom/signal tries, then stone/signal → yellow **48 s after the swap** (2 yellows) | adapted |
| Early rule swap (after 6 outcomes, t = 189 s) | 3 failed bloom/signal tries, then no new useful option found in 590 s (8 navigation failures, 5 unintended contacts) | did not adapt |
| Inert world (nothing responds) | 7 outcomes in the first half, 2 in the second; nothing useful | stops probing |
| Noisy world (bloom rewards 50 % of the time) | 2 yellows from 6 bloom signals, then only 1 outcome in the second half | backs off too early |

### All learning runs (denominators)

| Experiment | Phase | Policy | Sim s | Attempts | Outcomes | Useful | Aversive | Nav failures | Interrupted | Contacts intended / unintended | Operator re-enables |
|---|---|---|---|---|---|---|---|---|---|---|---|
| history_a | train | learned | 480 | 19 | 11 | 6 | 1 | 2 | 3 | 2 / 0 | 5 |
| history_a | after restart | learned | 66 | 4 | 2 | 0 | 0 | 1 | 0 | 0 / 0 | 0 |
| history_b | train | learned | 480 | 19 | 13 | 7 | 1 | 2 | 0 | 1 / 0 | 0 |
| history_b | after restart | learned | 21 | 2 | 2 | 0 | 0 | 0 | 0 | 0 / 0 | 0 |
| no_memory | fresh memory | learned | 34 | 2 | 2 | 0 | 0 | 0 | 0 | 0 / 0 | 0 |
| reversal_early | train | learned | 780 | 26 | 10 | 2 | 1 | 8 | 5 | 3 / 5 | 11 |
| reversal_late | train | learned | 780 | 30 | 15 | 7 | 1 | 9 | 2 | 2 / 2 | 5 |
| inert | train | learned | 480 | 17 | 9 | 0 | 0 | 7 | 0 | 2 / 0 | 2 |
| noisy | train | learned | 480 | 19 | 10 | 2 | 1 | 4 | 2 | 3 / 0 | 4 |
| learned_standard | train | learned | 480 | 19 | 11 | 6 | 1 | 2 | 3 | 2 / 0 | 5 |
| baseline_fixed | train | fixed | 480 | 50 | 33 | 13 | 0 | 9 | 3 | 5 / 0 | 2 |
| baseline_random | train | random | 480 | 43 | 17 | 5 | 2 | 10 | 11 | 3 / 0 | 6 |
| baseline_nearest | train | nearest | 480 | 41 | 34 | 1 | 0 | 5 | 1 | 2 / 0 | 1 |

*Intended contacts are those inside a nudge interaction. Most operator
re-enables follow tracking loss when nudging the tall cylinder fills the
camera view. `learned_standard` and history_a's training phase are the same
configuration and match exactly, because the simulator is deterministic.*

---

## Limitations and next gates

The full status of every capability is in
[docs/CAPABILITY_LEDGER.md](docs/CAPABILITY_LEDGER.md).

**Known limitations**

* **Navigation arrival rate** (5/16), limited by visual drift in held-out rooms
  and by goals losing clearance to live evidence.
* **Map quality:** partial coverage in 300 s, and 46–117 false-free cells per room.
* **Identity fragmentation:** one fixture was recorded twice, and the duplicate
  carried none of its bad history.
* **Unintended contacts in long learning runs:** one run pressed against a
  fixture for about 60 s late in the run.
* **Nudging tall objects** blinds the camera and causes tracking loss.
* **Noise handling:** backs off a 50 %-reliable option too early.
* **Perception generality:** the detector relies on uniformly painted objects in
  desaturated rooms, and the console's shape words can be wrong (the cyan
  cylinder is labelled "block").
* **Evidence strength:** one seed and one start pose per experiment.
* **Map provenance:** the learning runs use an arena map built with an earlier
  mapping-code state. A map rebuilt with current code was too fragmented to use.

**Next gates**

1. **Navigation reliability:** goal snapping to nearby certified cells, drift
   control (loop closure, better keyframes), suppressing stray obstacle hits,
   merging duplicate identities, and no unintended contacts in long runs.
2. **Statistical learning evaluation:** 10 or more seeds with randomized start
   poses and fixture placements, reporting distributions.
3. **Perception:** a learned detector behind the same contract (pixels in,
   labelled regions out; never coordinates or motor commands).
4. **Hardware:** a drive adapter behind the same supervisor, a physical
   emergency stop, real camera calibration, and perception running off the
   control thread in real time.

---

## Quick start

Requires Python 3.12. Linux x86_64 with the CPU backend and headless EGL was
used for all evidence; on macOS, Genesis uses its Metal/CPU backends.

```bash
git clone https://github.com/Nitish05/AMR-RL.git && cd AMR-RL
scripts/amr.sh setup                    # isolated .venv: genesis-world 1.3.2, torch, opencv, ...
scripts/amr.sh test                     # 236 tests + ruff + UI script checks (fast, no simulation)
RUN_GENESIS=1 scripts/amr.sh test-sim   # real Genesis checks: body, wheels, camera, screen, fixtures

# Operator console (loopback only). Autonomy starts DISABLED; press "Enable autonomy".
scripts/amr.sh app --world arena
open http://127.0.0.1:8770/
```

Use `AMR_PYTHON=/path/to/python scripts/amr.sh …` to select another
interpreter. The app takes `--map <saved map dir>`, `--memory <sqlite path>`
and `--consequences <standard|swapped|inert|noisy>`.

## Reproducing the evaluations

Each command writes a fresh evidence directory under `work/evidence/` (git
ignores it) with provenance: git commit, config and asset hashes, and
environment.

```bash
scripts/amr.sh map --world arena --seconds 420                       # build + save an arena map by exploration
scripts/amr.sh eval-nav --worlds home_a heldout_b heldout_c home_a_dim --map-seconds 300
scripts/amr.sh eval-learning --map work/evidence/<map-arena-…>/map   # all learning experiments
python scripts/eval/report.py --nav work/evidence/<navigation-…> \
    --learning work/evidence/<learning-…> --out report.md             # tables with denominators
PYTHONPATH=src python scripts/dev/results_figures.py --learning … --nav … --out docs/media
```

The results above come from `navigation-20260930-024431`,
`learning-20260930-022340` + `learning-20260930-042009` and the arena map
`map-arena-20260929-225035`. Runs that exposed bugs were kept with a
`SUPERSEDED.md` note naming the bug and its fix. The evidence itself (maps,
receipts, logs, trained memories) is not committed. The simulation runs at
about 0.6–0.8× real time on 2 CPU cores, so the full learning suite takes
several hours.

## Repository layout

| Path | Contents |
|---|---|
| `src/amr_rl/robot/` | spec → URDF/GLB generator |
| `src/amr_rl/sim/` | Genesis world, fixtures and rules, evaluator (ground truth, scoring only), Studio export |
| `src/amr_rl/control/` | command contract, supervisor, Genesis wheel backend |
| `src/amr_rl/perception/` | camera model, visual SLAM, fixture detector, async semantic worker |
| `src/amr_rl/mapping/`, `navigation/` | occupancy evidence, planner, navigator |
| `src/amr_rl/learning/` | identity tracking, outcome classification, SQLite experience memory |
| `src/amr_rl/behavior/` | activities, frontier exploration, chooser |
| `src/amr_rl/expression/`, `ui/` | screen face, operator console |
| `src/amr_rl/evidence/`, `runtime/`, `app.py` | receipts and provenance, runtime loop, operator app |
| `assets/robot/` | editable spec and generated robot assets |
| `configs/worlds/`, `configs/consequences/` | development and held-out rooms; engineered fixture rules |
| `projects/` | Genesis Studio project exports (schema 4) |
| `scripts/` | task runner, evaluation scripts, dev probes |
| `tests/` | unit, privilege-boundary, UI and opt-in Genesis tests |
| `docs/` | design notes, results, capability ledger, provenance |

## Honesty notes

* **Simulation vs hardware:** nothing has run on hardware, and there is no
  hardware adapter. Importing modules or running the simulation never connects
  to hardware.
* **Lockstep vs real time:** physics pauses while perception and decisions
  run. The measured rate (about 0.6–0.8× real time) is not a real-time result.
* **Engineered vs learned:** valences, need dynamics, costs, thresholds and
  fixture rules are engineered. Outcome expectations, attitudes and the
  resulting choices are learned from the robot's own image-grounded outcomes.
* **Development vs held-out:** `home_a`, `arena` and `dev_interact` were used
  for tuning; `heldout_b`, `heldout_c` and `home_a_dim` were not.
* **Provenance:** parts of the evidence, recording and supervisor patterns
  were adapted from an earlier project (BB8-RL). The source commit, file
  hashes and changes are recorded in [docs/PROVENANCE.md](docs/PROVENANCE.md).
  There are no runtime imports from it.
