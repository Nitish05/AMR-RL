# Development and diagnostic scripts

Probes, benchmarks and diagnostics used during development. They are not part of
the evaluation suite (`scripts/eval/`). Each one writes under the ignored `work/`
directory unless told otherwise. Scripts that score against the simulator use the
truth for evaluation only. Run them with `PYTHONPATH=src .venv/bin/python
scripts/dev/<name>.py ...`; see each file's docstring for its arguments.

| Script | Purpose |
|---|---|
| `blocked_probe.py` | Blocked-robot probe (evaluation only): drive straight into a heavy box the robot does not avoid (near-field guard off) and record whether, and how fast, the robot notices that it is not moving: the VSLAM's motion-consistency loss (``visual_motion_inconsistent_with_commands``) that starts the contact back-off. |
| `box_detour_probe.py` | Replay of the round-5 box contact (evaluation only): a saved map, a start pose, a box put down near the route, an operator goal past it. Reports the true minimum footprint-to-box gap, contacts and the goal result for several lateral box offsets. |
| `box_probe.py` | Detector localisation error for fixtures seen from known poses (evaluation probe: uses the true robot pose so only the detector's geometry is measured). |
| `capture_common.py` | Scoring geometry shared by the evaluation capture scripts (simulator truth; never imported by the runtime). |
| `closed_loop.py` | Development closed-loop run (not an evaluation). |
| `depth_obstacle_probe.py` | Offline accuracy of the monocular-depth near-field detector (evaluation probe). |
| `detector_capture.py` | Capture onboard frames with ground-truth object boxes for detector benchmarks (evaluation only; perception swap, steps 5a/5b). |
| `drive_physics_probe.py` | Drive physics probe (evaluation / diagnostics only): open-loop drive phases through the wheel backend, scored against simulator truth, to explain the round-9 effects (wheels rolling ~10 % short of the spec radius; the chassis yawing 0.1-0.3 dps with equal wheel speeds; in-place turns smaller than the encoders say). |
| `engage_probe.py` | Dev probe: force one engage on the nearest visible entity and trace world/robot. |
| `false_free_probe.py` | Evaluation-only diagnostic: which mapping step puts FREE evidence into cells that are truly inside obstacles (false free)? Tags free hits by source: parallax floor evidence ('floor'), the robot's own footprint ('footprint'), start attestation ('attest'). |
| `live_replay.py` | Live replay (round 9, WS0): run recorded live frames and raw sensor samples through VSLAM variants offline, frame by frame, and score them. |
| `loop_shadow_eval.py` | Paired loop-closure evaluation from shadow-mode builds (evaluation only). |
| `mapper_ab.py` | Controlled A/B of mapper settings on ONE trajectory (evaluation only). |
| `media.py` | Render documentation images: a close third-person view of Pip beside a fixture (inspection camera only; never used by the robot) and the matching onboard frame. |
| `moved_box_probe.py` | Timeline of the obstacle-on-route fault (evaluation probe). |
| `nav_probe.py` | Diagnostic: run the learned policy on a saved map and print every navigator status/reason change (the evidence recorder keeps only status and goal). |
| `nudge_reach_probe.py` | Nudge reach: does the planned creep reach the fixture? (evaluation probe) |
| `obstacle_hits_probe.py` | Evaluation-only diagnostic: where does obstacle evidence come from, and how much of it lands on true free floor? Wraps the mapper to tag each obstacle hit by source (parallax 'bases' or triangulated 'landmarks') and scores the hit cells against the true obstacle mask (ground truth used for scoring only). |
| `reloc_capture.py` | Capture onboard frames for the offline relocalisation probe (evaluation only). |
| `reloc_probe.py` | Offline relocalisation probe: replay captured onboard frames through a freshly loaded map (evaluation only; ground truth is used for scoring). |
| `results_figures.py` | Render README result figures from evaluation evidence. |
| `slam_bench.py` | Localisation benchmark for VSLAM variants: exploration runs scored against ground truth (evaluation only). Usage: |
| `slam_trace.py` | Replay an exploration run and print VSLAM status/reason changes in a time window, with the evaluator's true error (scoring only) and relocalisation gate state. |
| `turn_bench.py` | Offline turn-drift benchmark: replay captured in-place turns through PlanarVSLAM variants (evaluation only; ground truth is used for initialisation and scoring). |
| `turn_capture.py` | Capture in-place turns for the offline turn-drift benchmark (evaluation only). |
| `turn_compare.py` | Pool turn_bench.py reports over worlds: the pre-registered turn-drift metrics per variant and replay mode (evaluation only). |
| `turn_then_box_probe.py` | Reproducible obstacle-on-route scenarios (evaluation probe). |
| `vlm_outcome_pilot.py` | Offline pilot: can a small vision-language model tell what an interaction did? (evaluation only; perception-swap stage, step 5.0) |
| `vpr_eval.py` | Place-recognition quality in the simulated rooms (evaluation only). |
| `vslam_openloop.py` | Development check: open-loop scripted drive, RGB-only VSLAM vs ground truth. |
