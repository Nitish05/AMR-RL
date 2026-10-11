"""Drive physics probe (evaluation / diagnostics only): open-loop drive phases through
the wheel backend, scored against simulator truth, to explain the round-9 effects
(wheels rolling ~10 % short of the spec radius; the chassis yawing 0.1-0.3 dps with
equal wheel speeds; in-place turns smaller than the encoders say).

Phases from a standstill: straight at 0.2 m/s, stop, in-place turn at 0.45 rad/s,
stop, straight at 0.1 m/s, then the same straight drives at three other headings
(the pyramidal friction cone is anisotropic: world-axis aligned). Per phase:
  roll   = true distance / wheel distance (wheel angle x spec radius)
  yaw    = true yaw rate while both wheels turn equally (dps)
  turn   = true yaw / encoder yaw (track scale of an in-place turn)

    PYTHONPATH=src python scripts/dev/drive_physics_probe.py out_dir --world arena \\
        --physics '{"plane_friction": 0.02}'
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from common import provenance  # noqa: E402

from amr_rl.control.contract import DriveCommand  # noqa: E402
from amr_rl.robot.spec import RobotSpec  # noqa: E402
from amr_rl.sim import evaluator  # noqa: E402
from amr_rl.sim.world import SimWorld, load_world_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--world", default="arena")
    ap.add_argument("--physics", default="{}", help="world 'physics' block (JSON)")
    ap.add_argument("--world-set", default="{}", help="other world overrides (JSON object)")
    args = ap.parse_args()
    spec = RobotSpec.load()
    cfg = load_world_config(args.world)
    # an empty room (no obstacles or fixtures to bump into), starting in a corner
    cfg = {**cfg, "obstacles": [], "fixtures": [], "robot_start": [-1.4, -1.2, 0.6],
           **json.loads(args.world_set), "physics": json.loads(args.physics)}
    world = SimWorld(cfg, spec=spec, seed=0, inspection=False)
    r, track = float(spec.drive["wheel_radius"]), float(spec.drive["track"])
    dofs = world.backend._dofs

    def wheels():
        return np.asarray(world.robot.get_dofs_position(dofs), float).reshape(2)

    per_control = int(round(0.1 / world.dt))

    def run(v, w, seconds):
        for _ in range(int(round(seconds / 0.1))):  # one command per 0.1 s control step, as the runtime
            world.backend.command(DriveCommand(v, w, world.time, world.time + 0.2, 0, "manual"), now=world.time)
            for _ in range(per_control):
                world.step()

    def phase(name, v, w, seconds, settle=1.0):
        run(0.0, 0.0, settle)
        p0, q0 = evaluator.true_pose(world), wheels()
        run(v, w, 0.5)  # spin-up excluded from the steady-state numbers
        p1, q1, t1 = evaluator.true_pose(world), wheels(), world.time
        touched = set()
        for _ in range(int(round(seconds / 0.1))):
            run(v, w, 0.1)
            touched.update(evaluator.robot_contacts(world))
        p2, q2, t2 = evaluator.true_pose(world), wheels(), world.time
        dq, dq_all = q2 - q1, q2 - q0
        true_d = float(np.hypot(*(p2[:2] - p1[:2])))
        wheel_d = r * float(dq.mean())
        dyaw = math.remainder(p2[2] - p1[2], 2 * math.pi)
        enc_yaw = r * float(dq[1] - dq[0]) / track
        out = {"phase": name, "v": v, "w": w, "heading_deg": math.degrees(p1[2]),
               "true_distance_m": true_d, "wheel_distance_m": wheel_d,
               "roll": true_d / wheel_d if abs(wheel_d) > 1e-6 else None,
               "true_yaw_deg": math.degrees(dyaw), "encoder_yaw_deg": math.degrees(enc_yaw),
               "yaw_rate_dps": math.degrees(dyaw) / (t2 - t1),
               "turn": dyaw / enc_yaw if abs(enc_yaw) > 1e-3 else None,
               "turn_vs_command": dyaw / (w * (t2 - t1)) if abs(w) > 0 else None,
               "contacts": sorted(touched),
               "spinup_true_m": float(np.hypot(*(p1[:2] - p0[:2]))),
               "spinup_wheel_m": r * float((q1 - q0).mean()),
               "total_roll": float(np.hypot(*(p2[:2] - p0[:2]))) / (r * float(dq_all.mean()))
               if abs(dq_all.mean()) > 1e-6 else None}
        print(json.dumps({k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in out.items()}), flush=True)
        return out

    rows = [phase("straight_0.2", 0.2, 0.0, 3.0), phase("turn_0.45", 0.0, 0.45, 6.0),
            phase("straight_0.1", 0.1, 0.0, 3.0)]
    for k in range(3):  # other headings: turn ~60 deg, then drive
        run(0.0, 0.45, 2.3)
        rows.append(phase(f"straight_0.2_h{k + 1}", 0.2, 0.0, 2.5))
    rows.append(phase("turn_-0.45", 0.0, -0.45, 6.0))
    res = {"world": args.world, "physics": world.physics, "overrides": json.loads(args.world_set), "phases": rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    provenance(out, extra={"args": vars(args)})


if __name__ == "__main__":
    main()
