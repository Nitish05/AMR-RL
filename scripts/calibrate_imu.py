"""One-time gyro scale calibration of this robot's IMU, by the robot itself.

Procedure (engineered; uses only the robot's own camera, IMU and wheels):
1. stand still ``--still`` s: the odometry measures the gyro's zero-rate offset;
2. take a reference image, turn in place ``--turns`` full turns counter-clockwise
   (by the integrated, offset-corrected gyro at nominal sensitivity), stop, stand
   still, take an end image;
3. the true total angle = whole turns + the residual rotation between the reference
   and end images, measured by the depth-free planar epipolar estimator
   (perception/turn_epipolar.py; 0.1-0.2 deg per image pair);
4. repeat clockwise (the mean of both directions cancels a residual offset error);
5. gyro_scale = true angle / gyro angle; written to configs/calibration/imu.yaml
   with the unit serial, its uncertainty and the evidence directory.

Runs in simulation here (scripts/eval/common.py session); on hardware the same
steps apply. The simulator's true sensitivity is printed for evaluation only and is
not used for the result.

    PYTHONPATH=src .venv/bin/python scripts/calibrate_imu.py --world arena
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))
from common import dump, fresh_dir, provenance  # noqa: E402

from amr_rl.control.supervisor import SupervisorConfig  # noqa: E402
from amr_rl.perception.turn_epipolar import TurnEpipolar, match_orb  # noqa: E402
from amr_rl.robot.spec import PROJECT_ROOT  # noqa: E402
from amr_rl.runtime.robot import RuntimeConfig  # noqa: E402
from amr_rl.sim.harness import Session  # noqa: E402
from amr_rl.sim.world import WORLD_DIR  # noqa: E402


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def still(s, seconds):
    rt = s.runtime
    end = s.now + seconds
    while s.now < end - 1e-9:
        rt.command({"action": "manual", "v": 0.0, "w": 0.0, "generation": rt.supervisor.generation})
        s.control_step()


def one_direction(s, turns, w, est, scale_hint=1.0):
    rt = s.runtime
    still(s, 1.0)
    ref = rt.slam.features(rt.last_frame.rgb)
    i_ref = len(s.truth) - 1  # EVALUATION ONLY (scoring log index of the reference frame)
    gyro = 0.0
    target = turns * 2 * math.pi
    while abs(gyro) < target - 0.02:
        rem = target - abs(gyro)
        speed = abs(w) if rem > 0.3 else max(0.12, abs(w) * rem / 0.3)
        rt.command({"action": "manual", "v": 0.0, "w": math.copysign(speed, w), "generation": rt.supervisor.generation})
        s.control_step()
        if rt.last_odometry is not None:
            gyro += rt.last_odometry.dth
    for _ in range(15):  # stop and settle, still integrating
        rt.command({"action": "manual", "v": 0.0, "w": 0.0, "generation": rt.supervisor.generation})
        s.control_step()
        if rt.last_odometry is not None:
            gyro += rt.last_odometry.dth
    end = rt.slam.features(rt.last_frame.rgb)
    yaws = np.unwrap([row["gt"][2] for row in s.truth[i_ref:]])
    evaluation_true = float(yaws[-1] - yaws[0])  # EVALUATION ONLY, not used for the result
    pairs = match_orb(ref[2], end[2])
    hint = wrap(scale_hint * gyro)  # predicted residual after whole turns
    r = est.estimate(ref[1][pairs[:, 0]], end[1][pairs[:, 1]], theta_hint=hint)
    if not r.ok:
        return {"ok": False, "reason": r.reason, "gyro": gyro, "evaluation_true": evaluation_true}
    true = gyro + wrap(r.theta - gyro)
    return {"ok": True, "gyro": gyro, "residual": float(r.theta), "residual_sigma": float(r.sigma),
            "true": true, "ratio": true / gyro, "inliers": int(r.n_inliers),
            "translation_detected": bool(r.translation_detected), "evaluation_true": evaluation_true,
            "evaluation_residual": wrap(evaluation_true)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="arena")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--w", type=float, default=0.45)
    ap.add_argument("--still", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--unit", type=int, default=None, help="EVALUATION: calibrate another simulated IMU unit")
    ap.add_argument("--write", default=str(PROJECT_ROOT / "configs" / "calibration" / "imu.yaml"))
    args = ap.parse_args()
    out = Path(args.out).resolve() if args.out else fresh_dir("imu-calibration")
    provenance(out, configs=[WORLD_DIR / f"{args.world}.yaml"], extra={"args": vars(args)})
    cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), odometry="imu_encoders",
                        imu_calibration=None, place_descriptor=None, initial_survey=False, semantic=False,
                        near_field_guard=False)
    cfg.vslam.odo_scale_calibration = False  # the gyro must stay at nominal sensitivity here
    overrides = None if args.unit is None else {"sensor_faults": {"imu_unit": args.unit}}
    s = Session(args.world, run_dir=out / "session", memory_path=out / "throwaway-memory.sqlite", config=cfg,
                seed=args.seed, inspection=False, world_overrides=overrides)
    rt = s.runtime
    still(s, args.still)
    est = TurnEpipolar(rt.model)
    # Coarse pass: one turn each way (a 3 % sensitivity error leaves <= 11 deg, inside
    # the estimator's window), then the long turns with the residual predicted from it.
    coarse = [one_direction(s, 1, w, est) for w in (args.w, -args.w)]
    print("coarse", coarse, flush=True)
    good = [c["ratio"] for c in coarse if c["ok"]]
    scale_hint = float(np.mean(good)) if good else 1.0
    runs = []
    for _ in range(args.repeats):
        for w in (args.w, -args.w):
            r = one_direction(s, args.turns, w, est, scale_hint=scale_hint)
            runs.append({"w": w, **r})
            print(runs[-1], flush=True)
    ok = [r for r in runs if r["ok"]]
    ccw = [r["ratio"] for r in ok if r["w"] > 0]
    cw = [r["ratio"] for r in ok if r["w"] < 0]
    if not ccw or not cw:
        raise SystemExit("calibration failed: need both directions")
    assert rt.odo.gyro_scale == 1.0
    ratios = [0.5 * (a + b) for a, b in zip(ccw, cw)]
    scale = float(np.mean(ratios))
    per = [r["residual_sigma"] / abs(r["gyro"]) for r in ok]
    sigma = float(max(np.std(ratios) / math.sqrt(len(ratios)) if len(ratios) > 1 else 0.0,
                      math.sqrt(np.mean(np.square(per)) / len(ok)), 2e-4))
    cal = {"unit_serial": int(args.unit if args.unit is not None else rt.spec.imu["unit_serial"]),
           "part": rt.spec.imu["part"],
           "gyro_scale": scale, "gyro_scale_sigma": sigma,
           "gyro_offset_dps_at_calibration": math.degrees(rt.odo.bias),
           "method": f"{args.turns} in-place turns each way x {args.repeats}, closed by planar epipolar "
                     "residual between start and end images (scripts/calibrate_imu.py)",
           "date": time.strftime("%Y-%m-%d"), "evidence": str(out.relative_to(PROJECT_ROOT))}
    dump(out / "calibration.json", {"calibration": cal, "runs": runs,
                                    # EVALUATION ONLY: the simulated unit's true z sensitivity
                                    "evaluation_true_scale": float(1.0 / s.world.sensors.imu_model.g_M[2, 2])})
    print(cal, "evaluation: true scale", 1.0 / s.world.sensors.imu_model.g_M[2, 2])
    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text("# Gyro calibration of this robot's IMU unit (generated by scripts/calibrate_imu.py)\n"
                                    + yaml.safe_dump(cal, sort_keys=False))
    s.close()


if __name__ == "__main__":
    main()
