# AMR-RL

**A wheeled indoor robot whose later choices depend on what it has seen happen.**
Simulation only (Genesis World 1.3.2, lockstep). Onboard RGB camera only.

AMR-RL is a standalone successor-in-spirit to BB8-RL with the same long-term goal
— an embodied robot that perceives entities, remembers interactions, learns
consequences and preferences, and chooses what to do — built for a different
body: a compact differential-drive AMR ("Pip") with one forward RGB camera and
an expressive screen. It is independent of the BB8-RL checkout (no runtime
imports; adapted code is recorded in [docs/PROVENANCE.md](docs/PROVENANCE.md)).

> What this is not: a claim of consciousness, emotion or self-created desires.
> Motivations (what counts as a good or bad consequence) are **engineered**;
> what the robot **learns** is which of its actions on which entities, in which
> visible context, produce which observed consequences — and that changes what
> it chooses later. See [docs/LEARNING.md](docs/LEARNING.md).

![Operator console](docs/media/console.png)

## What it does today (simulation)

| Capability | How | Evidence |
|---|---|---|
| Physical robot model | Generated URDF + GLB meshes from an editable spec; wheel-velocity actuation only | [docs/ROBOT.md](docs/ROBOT.md), `tests/test_robot_assets.py`, `tests/test_sim_genesis.py` |
| Onboard monocular visual SLAM | ORB tracking on a planar pose, triangulated + floor landmarks, windowed BA, relocalization; metric scale from the calibrated camera height | [docs/VSLAM.md](docs/VSLAM.md), [docs/NAVIGATION_RESULTS.md](docs/NAVIGATION_RESULTS.md) |
| Conservative mapping & navigation | Plane-parallax free/obstacle evidence with a detectability gate; unknown ≠ free; footprint-certified A* + pure pursuit | same |
| Grounded interaction & learning | Pixel-grounded before/after outcome classification; SQLite receipts with immutable images; windowed, time-decayed Dirichlet expectations with change detection | [docs/LEARNING.md](docs/LEARNING.md), [docs/LEARNING_RESULTS.md](docs/LEARNING_RESULTS.md) |
| Activity selection | explore / investigate / engage / revisit / avoid / idle, bounded reconsideration, appropriate idle | same |
| Screen expression | Pure function of real state (activity, uncertainty, learned attitude); the `signal` action is a real screen output | [docs/EXPRESSION.md](docs/EXPRESSION.md) |
| Operator console | Camera, map+trajectory, confidence, activity/decision, entities, predicted vs observed, learning updates, manual drive, Stop | `python -m amr_rl.app` |
| Authority & safety contracts | Stop wins, generations, heartbeat, stale-frame and localization-loss revocation, bounded recovery, no auto-resume | `tests/test_supervisor.py` |

Current limits and next gates: [docs/CAPABILITY_LEDGER.md](docs/CAPABILITY_LEDGER.md).

## Results at a glance (simulation, one seed per experiment)

* **Learning:** after training in opposite worlds and a process restart, the
  robot's first choices were opposite (bloom/signal ×2 vs stone/signal ×2); a
  no-memory control chose differently again. Adapted to a consequence reversal
  in 1 of 2 runs. Valence per outcome: learned 0.46, fixed 0.38, random 0.18,
  nearest 0.02 (the fixed baseline collects more total valence because it never
  rests and happened to repeat the rewarding option). Details and failures:
  [docs/LEARNING_RESULTS.md](docs/LEARNING_RESULTS.md).
* **Navigation:** 0 contact episodes in 4 rooms; 12/12 unsupported goals
  rejected; 5/16 goals sampled from the robot's own map reached (4/4 in the
  dimmed room, 0/4 in two held-out rooms because of VSLAM drift and clearance
  loss). Details: [docs/NAVIGATION_RESULTS.md](docs/NAVIGATION_RESULTS.md).

## Quick start

```bash
cd AMR-RL
scripts/amr.sh setup            # isolated .venv (Python 3.12): genesis-world 1.3.2, torch, opencv, ...
scripts/amr.sh test             # non-simulation suite + ruff + UI script checks (fast)
RUN_GENESIS=1 scripts/amr.sh test-sim   # real Genesis checks (body, wheels, camera, screen, fixtures)

# Operator console (loopback only). Autonomy starts DISABLED; press "Enable autonomy".
scripts/amr.sh app --world arena
open http://127.0.0.1:8770/

# Evaluations (write fresh evidence directories under work/evidence/)
scripts/amr.sh map --world arena --seconds 420            # build + save a map by exploration
scripts/amr.sh eval-nav --worlds home_a heldout_b heldout_c home_a_dim
scripts/amr.sh eval-learning --map work/evidence/<map-arena-…>/map
```

`AMR_PYTHON=/path/to/python scripts/amr.sh …` selects another interpreter. On
macOS, Genesis uses its Metal/CPU backends; this delivery's evidence was
produced on Linux x86_64 with the CPU backend and headless EGL rendering
(see provenance files in each evidence directory).

## Architecture

```
Genesis World (physics + rendering) ── sim/world.py: room, fixtures + hidden response rules
      │ onboard RGB (camera_link)            ▲ wheel velocity targets
      ▼                                      │
perception/vslam.py ──pose──► mapping/occupancy.py ──► navigation/planner+navigator
perception/entities.py ──► learning/identity.py ──► learning/memory.py (SQLite)
perception/semantic.py (async, stale/authority-checked labels only)
behavior/chooser.py + activities.py ──► control/supervisor.py ──► control/genesis_backend.py
expression/policy.py + screen.py ──► robot screen          ui/server.py + static/ ──► operator
sim/evaluator.py (ground truth) ── scoring only; never imported by runtime code
```

## Repository layout

`src/amr_rl/` package · `assets/robot/` spec + generated URDF/meshes ·
`configs/worlds/` development and held-out rooms · `configs/consequences/`
engineered fixture rules · `projects/` Genesis Studio exports · `scripts/`
runner, evaluation scripts, dev probes · `tests/` · `docs/` · `work/` (ignored:
evidence, memories, caches).

## Honesty notes

* **Simulation vs hardware:** nothing has run on hardware; there is no hardware adapter.
* **Lockstep vs real time:** physics pauses while perception and decisions run; the
  measured wall-clock rate (≈0.6–0.8× real time on 2 CPU cores) is not a real-time result.
* **Engineered vs general:** the fixture detector finds uniformly painted objects in
  deliberately desaturated rooms; it is not general object recognition. Fixture
  responses are engineered rules.
* **Development vs held-out:** `home_a`, `arena`, `dev_interact` were used for
  tuning; `heldout_b`, `heldout_c`, `home_a_dim` were not.
