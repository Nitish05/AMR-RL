# Capability ledger

Status of every capability the project brief asks for. **All results are from
simulation** (Genesis World 1.3.2, CPU backend, lockstep: physics pauses while
perception and decisions run). Nothing has run on hardware. "Engineered" means
configured by a person; "learned" means changed only by the robot's own
recorded, image-grounded outcomes.

Legend: **done** = implemented and exercised with evidence in this delivery ·
**partial** = implemented but with a named gap · **not done** = absent.

## Stage 1 — independent project

| Capability | Status | Evidence / note |
|---|---|---|
| Own git repository and package `amr_rl` | done | its own GitHub repository (`Nitish05/AMR-RL`); never pointed at the BB8 or Studio repositories |
| Isolated environment recipe | done | `scripts/amr.sh setup` creates `.venv` (evidence here was produced in an equivalent isolated venv outside the repo) |
| Selective adaptation from BB8-RL with provenance | done | [PROVENANCE.md](PROVENANCE.md): source commit, file hashes, what changed; `tests/test_privilege_boundary.py` forbids runtime imports of BB8 |
| Fresh agent identity and memory | done | new agent id per memory file; no BB8 experiences imported |
| BB8-RL / Genesis Studio checkouts untouched | done | read-only clones used for reading; nothing written into the user's checkouts |

## Stage 2 — body, control, camera, screen

| Capability | Status | Evidence / note |
|---|---|---|
| Editable spec → URDF + GLB meshes (35×30 cm chassis, 2 driven wheels + caster, camera, screen) | done | [ROBOT.md](ROBOT.md), `tests/test_robot_assets.py` |
| Physical wheel actuation (velocity targets on wheel joints); pose setters only for resets | done | `control/genesis_backend.py`; `tests/test_sim_genesis.py::test_wheels_drive_body_forward_and_turn` |
| Rigidly mounted onboard RGB camera moving with the body | done | `test_camera_is_attached_to_the_moving_body` |
| Screen: rendered face on the screen link; `signal` is a real screen pattern | done | `test_screen_is_visible_to_inspection_not_to_onboard_camera`, `test_fixture_signal_rule_raises_panel_visible_onboard` |
| Supervisor: Stop wins, generations, heartbeat, stale frames, localization-loss revocation, bounded recovery, no auto-resume | done | `tests/test_supervisor.py`; nav fault tests |
| No hardware connection on import or in simulation | done | there is no hardware adapter at all |
| Genesis Studio as external, version-recorded dependency | partial | Studio-schema project export validated against Studio's models ([STUDIO.md](STUDIO.md)); opening it in Studio's native macOS viewer was not exercised |

## Stage 3 — onboard VSLAM and conservative navigation

| Capability | Status | Evidence / note |
|---|---|---|
| RGB-only monocular planar VSLAM (ORB, keyframes, windowed BA, relocalization, save/load) | done | [VSLAM.md](VSLAM.md), [NAVIGATION_RESULTS.md](NAVIGATION_RESULTS.md) |
| Explicit metric scale handling | done (with an assumption) | scale from the calibrated camera height over a flat floor; sensitivity is documented and unit-tested (a +10 % height error gives +10 % ranges). Not valid on ramps/uneven floors |
| Unknown ≠ free; footprint-certified planning; rejected goals carry reasons | done | `tests/test_mapping_navigation.py`; held-out results |
| Tracking loss revokes motion; bounded recovery; explicit re-enable | done | nav lens-blackout fault tests |
| Dynamic obstacles | partial | Monocular depth guard (Depth Anything V2 Small, Apache-2.0, scaled to metres by the visible floor; [results/near-field-guard.md](results/near-field-guard.md)) on top of the per-step corridor and turning-circle checks. Round 3: **0 of 5 box-on-route tests in contact** (round 2: 4 of 5); 0 contact steps in 12 runs. Cost: ~23 % of the cells it marks are on open floor. Simulation images only; optional dependency (without the model the guard is off and the runtime says so) |
| Reaching goals in its own map | partial | round 3 (with depth guard, 4 rooms × 3 seeds): **20/37** own-map goals (home_a 7/8, home_a_dim 8/12, heldout_b 4/9, heldout_c 1/8); round 2: 22/32. Misses: 10 localisation drift (robot believes it arrived, 16–76 cm off), 4 from one VSLAM failure, 3 possibly from guard marks. In 2 of 12 runs the robot was not localised at goal time. Arrival = within 0.15 m (the docs said 0.25 m until round 3; the code always used 0.15 m) |
| Map quality | partial | round 3: free coverage 27–72 % after 300 s; deep false-free cells 0–44 per run (round 2: 0–103; the plane-parallax floor test admits a few hits inside obstacles). Contacts while exploring: 0 in 24 runs over rounds 2–3 |
| Real-time operation | not done | lockstep; ≈0.6–0.8× real time on 2 CPU cores |

## Stage 4 — visually grounded interaction and persistent learning

| Capability | Status | Evidence / note |
|---|---|---|
| ≥ 4 distinguishable fixtures with action-dependent consequences | done | [LEARNING.md](LEARNING.md) (engineered world-side rules) |
| Entity identity from appearance + map position, ambiguity → no learning; duplicate merging | done | `learning/identity.py`, unit tests (twins, merges, never-co-visible rule). Round 2: 15 merges across runs; no return to the aversive fixture or a duplicate of it in 14/14 learned runs |
| Display labels from the semantic worker | done | labels are display-only; the shape word is now only "ball" or "object" (the fill heuristic could not tell boxes from cylinders) |
| Outcome from before/after pixels (moved / new attachment / none / ambiguous) | done | `learning/outcomes.py`; receipts store frame hashes and images |
| Learning layer robust to perception errors | partial | synthetic testbed only (no physics, engineered noise model; [results/learner-bench.md](results/learner-bench.md)), 20 seeds × 4 noise levels. Learned beats random/nearest/fixed at every level (0.58 → 0.18 true valence per attempt clean → high; baselines ≤ 0.03). Fixed: noise read as motion (false `moved` on inert objects 104 of 443 → 4 at high; inert world settles again). Open: outcome label confusion; real moves missed under 10 cm position noise |
| Receipts deduplicated; conflicting replays rejected; images retained immutably | done | `tests/test_learning_memory.py`, `tests/test_evidence_images.py` |
| Stale/cancelled async semantic results rejected; semantics never produce coordinates or motor commands | done | `perception/semantic.py` schema + authority/stale checks, tests |
| Knowledge survives restart; authority does not | done | learning restart phases: autonomy disabled at start, enabled only after relocalization (all 9 restart phases, 3 seeds) |
| General object recognition | not done | the detector finds uniformly painted objects in desaturated rooms |

## Stage 5 — activity selection, expression, operator console

| Capability | Status | Evidence / note |
|---|---|---|
| explore / investigate / engage / revisit / avoid / idle with bounded reconsideration | done | `behavior/chooser.py`; decision logs in every learning run |
| Experience changes later choices | done (3 seeds) | first decision after restart targeted the learned option in 5/5 runs that learned one; carried out on that fixture in 3/5 (twice the remembered fixture was not confirmed in fresh images and the robot moved on). No-memory control differs on all seeds ([LEARNING_RESULTS.md](LEARNING_RESULTS.md)) |
| Adapting when consequences change | done (2 seeds) | reversal adapted in 4/4 informative runs: 3 failed tries of the old option, then the new one found 100–300 s after the swap. Seed-0 reversals uninformative (old option never learned) |
| Settling in ineffective / noisy worlds | partial | inert (seed 0): outcomes 10 → 5 per half. Noisy: untested in round 2 (seed 0 never reached the noisy fixture); adaptive change detection is unit-tested only. Round 1 backed off too early |
| Policy better than simple baselines | not shown | baselines ran on seed 0 only, where the learned policy never reached the rewarding fixture: total valence nearest 2.0, random 0.2, learned 0.2, fixed 0.0. Learned seeds 1–2: 0.57 and 0.54 valence per outcome, without baselines |
| Idle as a real choice (bounded rest, then reconsider) | done | `test_idle_is_a_bounded_rest_so_the_robot_reconsiders` |
| Screen expression as a documented function of real state | done | [EXPRESSION.md](EXPRESSION.md), `tests/test_expression.py` |
| Operator console: camera, map + trajectory, confidence, activity + decision, entities + attitudes, predicted vs observed, learning updates, manual drive, Stop, reset memory | done | `python -m amr_rl.app`, `tests/test_ui_server.py`, `tests/test_ui_static.js` |

## Stage 6 — evaluation

See [NAVIGATION_RESULTS.md](NAVIGATION_RESULTS.md) and
[LEARNING_RESULTS.md](LEARNING_RESULTS.md) for numbers with denominators. The
learning runs use an arena map the robot built with the current code
(`map-arena-20260930-122444`). Runs
that were aborted because they exposed a bug are retained under `work/evidence/`
with a `SUPERSEDED.md` note naming the bug and the fix.

## Known limitations and remaining work (next gates)

0. **Navigation and interaction reliability (next software gate).** Done in
   round 2: goal snapping, frozen-estimate detection, landmark purge on loss,
   dead-reckoning-gated relocalisation, base confirmation, identity merging,
   keep-out discs. Still open: wrong relocalisation (10–20 cm) and heading
   errors during in-place rotation; a robot that cannot relocalise while
   stationary once recovery is exhausted; localisation drift in held-out rooms
   (now the main cause of missed goals); false-free from the
   plane-parallax floor test; recovery turns that start from a confidently
   wrong pose (2 bench contacts, pose 48 cm off); tracking loss near tall
   fixtures (seed 0 learning). Newly placed obstacles: handled in simulation by
   the monocular depth guard (round 3: 0/5 box tests in contact); its false
   obstacle marks (~23 % on open floor) are the next cost to cut
   ([results/near-field-guard.md](results/near-field-guard.md)). Fixed after
   round 2, not yet re-evaluated end to end: nudge creep from the measured contact
   edge (boxes seen corner-on: 21/27 → 27/27 reach,
   [results/nudge-reach.md](results/nudge-reach.md)).

1. **Hardware gate (not started).** Needs: a real drive adapter behind the same
   supervisor contract (and a physical E-stop independent of software), camera
   intrinsics/extrinsics calibration on the real camera, real-time pipeline
   (perception off the control thread), and a staged test plan (wheels off the
   ground → tethered → free).
2. **Perception generality.** Replace the saturated-colour fixture detector
   with a learned detector/embedding while keeping the same contract (pixels in,
   labelled regions out, no coordinates/motor commands from semantics).
3. **Mapping completeness.** Plane-parallax evidence leaves textureless floor
   UNKNOWN by design; coverage in 300 s is partial. Next: longer active
   exploration with view planning, loop closure across sessions, map merging.
4. **Statistical power of learning results.** The simulator is deterministic;
   round 2 ran 3 seeds (start poses) for the main experiments and 1 for the
   baselines and inert/noisy worlds. Next gate: ≥ 10 seeds with randomized start
   poses and fixture placements, baselines on every seed, reporting
   distributions.
5. **Motivations are engineered.** The valence table and the stimulation-need
   dynamics are configured. What is learned is which action on which entity in
   which visible context produces which consequence.
6. **Genesis Studio viewer.** Opening the exported projects in Studio's native
   viewer on macOS remains to be exercised.
