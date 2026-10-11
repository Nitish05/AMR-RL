# Provenance of adapted code and dependencies

AMR-RL has no runtime dependency on the BB8-RL checkout. The files below were
read at BB8-RL commit `1cc03a170aebe0515a7e43649183aee02ae088da` (branch `main`,
identical to `origin/main`, working tree clean when read on 2026-09-29), copied
into AMR-RL's own package and modified. The hash column is the first 16 hex
digits of the SHA-256 of the BB8 source file at that commit.

## Adapted from BB8-RL

| BB8-RL source | sha256 (prefix) | AMR-RL destination | What was kept | What changed |
|---|---|---|---|---|
| `src/bb8_rl/purpose.py` | `0ff81660262512fa` | `src/amr_rl/learning/memory.py` | SQLite store with `BEGIN IMMEDIATE` atomic event insertion; exact-replay returns no update and conflicting event ids raise; windowed evidence; finite probe budgets; early (2 useful) and stable (6 useful) change hypotheses that re-open suppressed alternatives on the 3rd consecutive failure; capped one-step value of information | Stations×scalar resource generalised to entity×action×visible-context → discrete observed outcome tokens (Dirichlet); explicit engineered valence table; time-decayed evidence (half-life) with at most one renewed probe per half-life; evidence before a detected change is discounted; persisted "armed" change state (fixes the early-reversal class of defect by construction); context backoff; entity table scoped by map version; learning-update log; explicit reset; agent identity check. Station coordinates, resource target and telemetry sources removed. |
| `src/bb8_rl/purpose_runtime.py` | `2bc0476357f001d5` | `src/amr_rl/behavior/chooser.py`, `src/amr_rl/behavior/activities.py`, `src/amr_rl/runtime/robot.py` | Separation of navigation failure (never learned) from interaction outcomes; route-failure backoff that is session-scoped; quiet idle; reconsideration only on evidence; cancellation on stale observations/localization loss; arrival alone earns nothing | Rewritten for six activities (explore, investigate, engage, revisit, avoid, idle) with bounded reconsideration (minimum interval, per-minute switch budget, hysteresis margin, non-preemptible interaction phases); engineered stimulation need + habituation; AMR navigator interface instead of BB8 `ControlSession`. |
| `src/bb8_rl/visual_interaction.py` | `f62e001e6293614b` | `src/amr_rl/perception/entities.py`, `src/amr_rl/learning/identity.py`, `src/amr_rl/learning/outcomes.py` | Idea of pixel-only observation records, multi-frame windows before/after an action, rejection of moved/uncertain/ambiguous evidence, learning only from complete windows | Fixed-camera marker/gauge decoder replaced: a moving onboard camera, saturated-colour fixture detection with stacked "attachments" as visible state, floor-contact localisation through the calibrated camera height + VSLAM pose, appearance+position identity with explicit ambiguity, relocation handling, twin handling; outcome classes (`moved`, `attach:<colour>`, `none`). Still an engineered-fixture detector, not object recognition. |
| `src/bb8_rl/visual_purpose_runtime.py` | `922a8f7d67426522` | `src/amr_rl/runtime/robot.py` (`_on_activity_done`), `src/amr_rl/behavior/activities.py` (`Engage`) | Learning happens only after the evidence has been retained; world completions are never numeric telemetry; interruption cancels without learning | The interaction is an explicit robot primitive (signal on the screen, or a bounded nudge) with approach → confirm → act → observe phases under the supervisor. |
| `src/bb8_rl/visual_evidence.py` | `8ce17c34c57f5e7c` | `src/amr_rl/evidence/images.py` | Exclusive hard-link publishing, fsync of file and directory, immutable receipts, exact replay verification (including PNG decode after the live buffer evicted the frame) | Generic `retain(receipt, frame_hashes)`, schema `amr_rl.image-receipt.v1`, configurable buffer, hash validation (path-traversal proof), `load_receipt`. |
| `src/bb8_rl/session_recording.py` | `a0b42809a79cb831` | `src/amr_rl/evidence/recording.py` | Rotating bounded telemetry vs complete audit mode, transition logging, heartbeat summarisation, atomic summary | AMR row schema; sim-time resets; `row_index`; finished flag. |
| `src/bb8_rl/run_identity.py` | `ae717c773032a6c7` | `src/amr_rl/evidence/provenance.py` | Source/config/runtime identity capture and verification; fresh snapshot directory | Lock-free git reads (`--no-optional-locks`); BB8 task/world graph removed; explicit config list; asset manifest re-hash; optional Studio revision; metadata-only package versions. |
| `src/bb8_rl/control/contract.py` | `8765020a319968fd` | `src/amr_rl/control/contract.py` | Timestamped commands with expiry, capability record, backend protocol | Planar unit-disk force requests → body-frame (v, ω) with generation and source; wheel-speed conversion that preserves curvature. |
| `src/bb8_rl/control/genesis_backend.py` | `975951b11adfe5ee` | `src/amr_rl/control/genesis_backend.py` | Single actuator-translation module; monotonic time checks; rejection of future/expired/non-increasing commands; stop semantics; reset-only pose setter | Free-body force drive replaced by Genesis wheel-joint velocity control under a torque limit. |
| `src/bb8_rl/interactive_runtime.py` (design reference only) | `6bd073cf6411642c` | `src/amr_rl/control/supervisor.py` | Stop wins a batch; generations; heartbeat expiry cancels motion; stale-command rejection; no automatic resumption | Re-implemented from scratch as a small testable supervisor, plus bounded recovery (in-place rotation or retracing a straight nudge approach). |
| `tests/test_purpose_adaptation.py` | `e34febb49a669807` | `tests/test_learning_memory.py` | Early/late reversal recovery across histories, replay idempotence during reversal, static ineffective world settles, isolated success does not re-open, restart keeps knowledge | Adapted to the new model; value-driven stopping may idle before the probe budget is exhausted, so exact probe counts became upper bounds. |
| `tests/test_visual_evidence.py` | `94c80e4843b5a234` | `tests/test_evidence_images.py` | All nine behaviours (immutability, bounded buffer, audit, partial write, corrupt PNG/receipt, replay after eviction and restart, concurrent conflicting publish, directory fsync failure) | New API. |
| `tests/test_visual_purpose_audit.py` | `31870a736042fe4a` | `tests/test_perception_units.py`, `tests/test_learning_memory.py` | Negative controls: missing/unstable evidence teaches nothing; unauthorised receipts rejected | Fixture-specific marker assertions dropped. |

Not reused: BB8's scan-once/fixed-camera mapping, SAC/Dreamer controllers,
marker/gauge decoder geometry, stations, BB8 memories and evidence. The AMR
starts with a new agent identity and an empty memory; no BB8 interaction was
imported as AMR experience.

## Created new for AMR-RL

Robot generator (`robot/`), wheel backend, supervisor, Genesis world and fixture
mechanics (`sim/`), planar monocular VSLAM (`perception/vslam.py`), plane-parallax
occupancy mapping (`mapping/`), conservative planner/navigator, frontier
exploration, asynchronous semantic worker, activity chooser, runtime, operator
console server and UI, screen expression, evaluation harness and scripts.
Rounds 4–9 added the planar two-point relocalisation, loop closure and its pose
graph (`perception/pose_graph.py`, `place_recognition.py`), the monocular depth
guard (`perception/near_depth.py`, `navigation/near_field.py`), the open-vocabulary
detector rules and re-ID (`perception/open_vocab.py`, `reid.py`), the IMU and
encoder models (`sim/sensors.py`), wheel-inertial odometry (`odometry/`), the
planar epipolar turn estimator, view-aware turning and the gyro calibration
(`scripts/calibrate_imu.py`). All are new code; where a published method shaped
a design, the results documents cite it.

## External dependencies (recorded per run by `evidence/provenance.py`)

| Dependency | Version used for the recorded evidence | Notes |
|---|---|---|
| Python | 3.12.13 on macOS arm64 (evidence from 2026-10-02 on, rounds 4–9); 3.12.3 on Linux x86_64 for the five surviving directories from 2026-09-29/30 | full environment: `requirements/lock-macos-arm64-py3.12.txt`; each run records its own in `provenance/identity.json` |
| genesis-world | 1.3.2 | Same version as the local Genesis Studio environment (`Genesis-Studio/.venv-genesis`); CPU backend |
| torch / torchvision | 2.14.1 / 0.29.1 (macOS); 2.13.0 on Linux | Genesis dependency, and the runtime's optional learned models (depth guard, MegaLoc place recognition, open-vocabulary detector) |
| transformers | 5.18.0 | the optional learned models above |
| numpy / scipy | 2.5.3 / 1.18.1 | |
| opencv-python | 5.0.0.93 | ORB, homographies, PnP; the GUI wheel, which genesis-world requires (the headless wheel was also installed until 2026-10-11 and shared `cv2`) |
| trimesh | 5.1.1 | asset generation; the committed robot meshes are kept (5.1.1 would write them slightly differently) |
| Genesis Studio | not imported | `docs/STUDIO.md` explains the optional project export; Studio revision `cd20c8c685f2b51263814fd0a6296bfd849bdce6` was read, never modified |

No third-party model weights are committed. Optional models are fetched on request
into the user's Hugging Face / torch-hub caches outside the repository, pinned by
revision and loaded with `local_files_only` at run time:

| Model | Licence | Pinned revision | Used by | Status |
|---|---|---|---|---|
| Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) | Apache-2.0 | `5426e4f0f36572d16453bbda7a8389317b1bef99` | `perception/near_depth.py` (near-field guard) | runtime, optional (`scripts/amr.sh fetch-depth-model`) |
| MegaLoc (code `gmberton/MegaLoc`, weights on HF) | MIT | code `5fe0dd697c4a70ba3e23607f6716ab3c606b16db`, weights `a0f34722c4297ff787e022433799250180860af7` | `perception/place_recognition.py` (loop closure, retrieval) | runtime, optional (`scripts/amr.sh fetch-place-model`) |
| Qwen3-VL-2B-Instruct (`Qwen/Qwen3-VL-2B-Instruct`) | Apache-2.0 | `89644892e4d85e24eaac8bacfd4f463576704203` | `scripts/dev/vlm_outcome_pilot.py` | evaluation only ([results/vlm-outcome-pilot.md](results/vlm-outcome-pilot.md)) |
| OmDet-Turbo Swin-T (`omlab/omdet-turbo-swin-tiny-hf`) | Apache-2.0 | `7fe93cecfb770c4d76cf71163956221249cab566` | `perception/open_vocab.py` (`RuntimeConfig.detector="open_vocab"`) | runtime, optional, off by default (`scripts/amr.sh fetch-detector-model`); needs `timm` (Apache-2.0) |
| DINOv2-S (`facebook/dinov2-small`) | Apache-2.0 | `ed25f3a31f01632728cabb09d1542f84ab7b0056` | `perception/reid.py` | evaluation (`scripts/eval/reid_bench.py`); not used by identity |
| LLMDet-tiny (`iSEE-Laboratory/llmdet_tiny`) | Apache-2.0 | `d05199165a19320a9236396e20fca0a5e065189c` | evaluated in `perception/open_vocab.py` (`backend="llmdet"`) | rejected (results/textured-worlds.md) |
| SmolVLM2-500M-Video-Instruct (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) | Apache-2.0 | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` | `scripts/dev/vlm_outcome_pilot.py` | evaluation only; its processor needs `num2words` (LGPL, unmodified, evaluation environment only) |

The default semantic backend is still an engineered describer, and outcomes come from
the engineered colour detector.

## Evidence retention

Evidence lives in fresh directories under the ignored `work/evidence/`. Runs that failed or were superseded are kept, with a note naming the reason.

Validation runs come from pinned git worktrees under `work/wt/`:
- A worktree is removed once the results document names its commit, it has no uncommitted changes, and the commit is in `main`'s history. The commit reproduces the code; the world-asset caches inside the worktree are regenerated on demand.
- A worktree with local changes keeps them as a patch next to its evidence before removal.
- Round-9 removals (2026-10-11): ce39b72, 552d630, 6e89e60, 890646c, d237559, f0748fc and 811adbc.

The large turn-capture corpora (`work/evidence/turncap-*`, about 29 GB) are kept until the owner decides to archive them elsewhere.
