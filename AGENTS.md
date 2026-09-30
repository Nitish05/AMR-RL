# AMR-RL working instructions

- AMR-RL is an independent project. It shares no runtime code with BB8-RL; code
  adapted from BB8-RL is copied, modified and recorded in `docs/PROVENANCE.md`.
  Never import from, write to, or run migrations against the BB8-RL or Genesis
  Studio checkouts.
- Genesis World owns physics and rendering. Only `src/amr_rl/control/genesis_backend.py`
  commands the robot's actuators. Body pose/velocity setters are for explicit
  resets (`initialize_pose`) only. World fixture mechanisms live in `sim/world.py`.
- Operational perception uses ONLY the robot-mounted RGB camera + its calibration
  + the robot's own commanded motion. Simulator ground truth is for
  `amr_rl.sim.evaluator` and evaluation scripts only (enforced by
  `tests/test_privilege_boundary.py`).
- Importing modules or running simulation must never connect to hardware.
- Label engineered vs learned quantities honestly; never claim consciousness,
  emotion or self-created desires. Keep the capability ledger current.
- Evidence goes into fresh directories under ignored `work/`; retain failures.
- Run `scripts/amr.sh test` (non-simulation suite + ruff + JS checks) for every
  change and `scripts/amr.sh test-sim` for simulation/control/camera changes.
- Do not commit environments, memories (`*.sqlite`), generated runs, credentials
  or third-party weights.
