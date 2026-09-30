# AMR-RL staged implementation plan

AMR-RL builds a wheeled indoor robot ("Pip") whose later choices depend on its
own remembered, image-grounded experience. Mapping and navigation support that
goal; they are not the goal. This plan fixes the stages, their acceptance gates
and file ownership before implementation. Status lives in
[CAPABILITY_LEDGER.md](CAPABILITY_LEDGER.md), not here.

## Ground rules (apply to every stage)

- Operational perception, mapping, localization and navigation consume only the
  robot-mounted RGB camera image, its calibration (intrinsics + mounting) and the
  robot's own commanded actions/state machine. Simulator ground truth (poses,
  labels, depth, segmentation, room geometry) is available **only** to
  `amr_rl.sim.evaluator` and evaluation scripts, never to runtime modules.
  A test (`tests/test_privilege_boundary.py`) enforces this import boundary.
- Only `amr_rl.control.genesis_backend` calls Genesis actuators. Body pose and
  velocity setters are used only in `initialize_pose()` for explicit resets.
- Importing any module or running simulation never touches hardware; there is no
  hardware adapter.
- Engineered quantities (motivations, valences, costs, thresholds, fixture
  response rules) are labelled as such. Learned quantities are listed in
  `docs/LEARNING.md` with their exact update points.
- Evidence goes into fresh named directories under ignored `work/`. Failed runs
  are retained and counted.
- No runtime import from the BB8-RL checkout. Adapted code is copied, modified
  and recorded in `docs/PROVENANCE.md` with source commit and file hashes.

## Stages and acceptance gates

| # | Stage | Acceptance gate |
|---|---|---|
| 1 | Independent project + reuse assessment | Own git repo (no remote), package `amr_rl`, isolated env recipe, fresh memory/evidence paths; provenance table for every adapted BB8 file; BB8/Studio checkouts untouched |
| 2 | 3D robot, physical control, camera, screen | Spec→URDF/GLB generator reproducible; Genesis loads robot with named frames; wheel-velocity actuation moves/turns the body; camera renders from `camera_link` and moves with it; screen framebuffer shown on the screen link (inspection view); supervisor tests for Stop priority, generations, heartbeat and stale-command rejection pass |
| 3 | Onboard VSLAM + conservative navigation | RGB-only monocular tracking with camera-height scale prior; evaluator-measured ATE and scale error reported; relocalization against saved map; occupancy with explicit unknown; planner never routes through unknown; rejects unsupported goals; tracking loss revokes motion; bounded recovery; results on held-out layouts/appearance/lighting with full denominators |
| 4 | Visually grounded interaction + persistent learning | ≥4 distinguishable fixtures with action-dependent consequences; each interaction records supporting frames, identity + confidence, action + basis, prediction, observed consequence (from pixels) and memory update; receipts deduplicated; async semantic results rejected when stale or under revoked authority |
| 5 | Experience-dependent activity selection + expression | Chooser over explore/investigate/engage/revisit/avoid/idle with bounded reconsideration and appropriate idle; screen expression is a documented function of real activity/uncertainty/learned attitude; operator UI with camera, map, trajectory, confidence, memories, predicted vs observed, learning updates, manual drive, Stop |
| 6 | Held-out evaluation + documentation | Navigation and learning evaluated separately: opposite histories, restart persistence, early/late reversal, noisy/ineffective settling, memory ablation, random/nearest/fixed baselines; README, launch/test commands, ledger with remaining work |

## Architecture

```
            Genesis World (physics + rendering)        <- sim only
   ┌────────────────────┴────────────────────────┐
   │ amr_rl.sim.world        room/fixtures/lights │  fixture response rules (engineered, world-side)
   │ amr_rl.sim.evaluator    ground truth scoring │  NEVER imported by runtime
   └───────┬──────────────────────────┬───────────┘
   onboard RGB (camera_link)     wheel joint velocity (genesis_backend only)
           │                          ▲
   perception.vslam  ──pose/map──> mapping.occupancy ──> navigation.planner/follower
   perception.entities (fixture detector) ──> learning.identity (entity memory)
   perception.semantic_worker (async, stale-rejecting annotations only)
           │                                   │
   learning.memory (SQLite) <── learning.outcomes (pixel before/after windows)
           │
   behavior.chooser (activities, bounded reconsideration, idle)
           │
   control.supervisor (authority, Stop, generations, heartbeat, freshness)
           │
   expression.screen (face)      ui.server (operator console)
```

## Parallel ownership

- Integration owner (main agent): sim, perception, mapping, navigation,
  behavior, runtime integration, all native Genesis runs, shared docs.
- Agent A: `src/amr_rl/evidence/*` (adapted provenance/recording/image
  receipts) and their tests.
- Agent B: `src/amr_rl/expression/*` and `src/amr_rl/ui/static/*` against the
  JSON state contract in `docs/INTERFACES.md`.

Agents do not run native simulations, commit, or edit shared files.
