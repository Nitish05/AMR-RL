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
| Own git repository, no remote, own package `amr_rl` | done | `git remote -v` is empty; no BB8/Studio origin |
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
| Dynamic obstacles | partial | per-step 0.6 m corridor check + no in-place turns with obstacle evidence in the turning circle: the one obstacle-on-route test that ran ended stopped without contact; earlier code clipped the box. A new object needs 3 observations to de-certify saturated floor |
| Reaching goals in its own map | partial | supported goals arrived 5/16 (4/4 in home_a_dim, 0/4 in heldout_b and heldout_c): VSLAM drift up to 0.29 m in held-out rooms, and goals losing footprint clearance to live evidence while driving |
| Map quality | partial | free coverage 31–77 % after 300 s; 46–117 false-free cells per room (2–46 more than 10 cm inside obstacles) |
| Real-time operation | not done | lockstep; ≈0.6–0.8× real time on 2 CPU cores |

## Stage 4 — visually grounded interaction and persistent learning

| Capability | Status | Evidence / note |
|---|---|---|
| ≥ 4 distinguishable fixtures with action-dependent consequences | done | [LEARNING.md](LEARNING.md) (engineered world-side rules) |
| Entity identity from appearance + map position, ambiguity → no learning | partial | `learning/identity.py`, unit tests (twins). In every learned run one fixture (grump) was also registered as a duplicate identity; the duplicate carried no bad history and was re-targeted (all such attempts failed at navigation) |
| Display labels from the semantic worker | partial | labels are display-only; the shape word is from a silhouette-fill heuristic and is wrong for the arena's cylinder/box ("cyan block" is a cylinder) |
| Outcome from before/after pixels (moved / new attachment / none / ambiguous) | done | `learning/outcomes.py`; receipts store frame hashes and images |
| Receipts deduplicated; conflicting replays rejected; images retained immutably | done | `tests/test_learning_memory.py`, `tests/test_evidence_images.py` |
| Stale/cancelled async semantic results rejected; semantics never produce coordinates or motor commands | done | `perception/semantic.py` schema + authority/stale checks, tests |
| Knowledge survives restart; authority does not | done | learning restart phases: autonomy disabled at start, enabled only after relocalization |
| General object recognition | not done | the detector finds uniformly painted objects in desaturated rooms |

## Stage 5 — activity selection, expression, operator console

| Capability | Status | Evidence / note |
|---|---|---|
| explore / investigate / engage / revisit / avoid / idle with bounded reconsideration | done | `behavior/chooser.py`; decision logs in every learning run |
| Experience changes later choices | done (1 seed) | opposite histories → opposite first choices after restart; no-memory control differs ([LEARNING_RESULTS.md](LEARNING_RESULTS.md)) |
| Adapting when consequences change | partial | reversal adapted in 1 of 2 runs (late switch: new useful option found 48 s after the switch; early switch: none found in 590 s) |
| Settling in ineffective / noisy worlds | partial | inert: probing stops (7 → 2 outcomes per half); noisy: backs off a 50 %-reliable option too early |
| Idle as a real choice (bounded rest, then reconsider) | done | `test_idle_is_a_bounded_rest_so_the_robot_reconsiders` |
| Screen expression as a documented function of real state | done | [EXPRESSION.md](EXPRESSION.md), `tests/test_expression.py` |
| Operator console: camera, map + trajectory, confidence, activity + decision, entities + attitudes, predicted vs observed, learning updates, manual drive, Stop, reset memory | done | `python -m amr_rl.app`, `tests/test_ui_server.py`, `tests/test_ui_static.js` |

## Stage 6 — evaluation

See [NAVIGATION_RESULTS.md](NAVIGATION_RESULTS.md) and
[LEARNING_RESULTS.md](LEARNING_RESULTS.md) for numbers with denominators. The
learning runs use an arena map the robot built with an earlier mapping-code
state (documented in LEARNING_RESULTS.md). Runs
that were aborted because they exposed a bug are retained under `work/evidence/`
with a `SUPERSEDED.md` note naming the bug and the fix.

## Known limitations and remaining work (next gates)

0. **Navigation reliability (next software gate).** Goal snapping to nearby
   certified cells when live evidence removes a goal's clearance; drift control
   in held-out rooms (loop closure, better keyframe selection); a mapping pass
   that suppresses stray obstacle hits so free space is not fragmented (a map
   rebuilt with current code, `map-arena-20260930-013357`, was too fragmented to
   use); duplicate-identity merging; no unintended contacts in long learning runs
   (reversal_early pressed against a fixture for ~60 s).

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
   each learning experiment here is one seed from one start pose. Next gate:
   ≥ 10 seeds with randomized start poses and fixture placements, reporting
   distributions.
5. **Motivations are engineered.** The valence table and the stimulation-need
   dynamics are configured. What is learned is which action on which entity in
   which visible context produces which consequence.
6. **Genesis Studio viewer.** Opening the exported projects in Studio's native
   viewer on macOS remains to be exercised.
