"""Live replay (round 9, WS0): run recorded live frames and raw sensor samples
through VSLAM variants offline, frame by frame, and score them.

A run recorded with ``build_map.py --record-frames`` (or navigation/learning) keeps
every onboard frame (``session/image-evidence/<sha>.png``), the per-frame scoring
log with the frame hash and the command (``session/trajectory_scoring.jsonl``) and
the raw IMU/encoder samples (``session/proprio.npz``). Replaying identical inputs
pairs variants exactly. Limitation: open loop. The robot's motion is the logged
one, which is exact for estimator changes; changes that would alter behaviour
need closed-loop runs.

    PYTHONPATH=src .venv/bin/python scripts/dev/live_replay.py RUN_DIR \
        --variant base odometry=command --variant odo odometry=imu_encoders \
        --variant odo_mask odometry=imu_encoders depth_floor_mask='"always"' --out replay.json

Scoring uses the logged ground truth (evaluation only).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from turn_live_report import analyse  # noqa: E402

from amr_rl.odometry.fusion import WheelInertialOdometry  # noqa: E402
from amr_rl.odometry.samples import EncoderSample, ImuSample, ProprioBatch  # noqa: E402
from amr_rl.odometry.wheel_model import CommandModelOdometry  # noqa: E402
from amr_rl.perception.camera_model import CameraModel  # noqa: E402
from amr_rl.perception.vslam import PlanarVSLAM, VSLAMConfig  # noqa: E402
from amr_rl.robot.spec import PROJECT_ROOT, RobotSpec  # noqa: E402


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_run(run_dir):
    session = Path(run_dir) / "session"
    rows = [json.loads(line) for line in open(session / "trajectory_scoring.jsonl")]
    if not rows or "frame_sha" not in rows[0]:
        raise SystemExit(f"{run_dir}: not recorded with --record-frames")
    prop = np.load(session / "proprio.npz") if (session / "proprio.npz").exists() else None
    return session, rows, prop


def batches(prop, times):
    """Raw samples between consecutive frame times (t_prev, t]."""
    if prop is None:
        return [ProprioBatch([], []) for _ in times]
    imu, enc = prop["imu"], prop["enc"]
    out, prev = [], -math.inf
    for t in times:
        mi = (imu[:, 0] > prev + 1e-9) & (imu[:, 0] <= t + 1e-9)
        me = (enc[:, 0] > prev + 1e-9) & (enc[:, 0] <= t + 1e-9)
        out.append(ProprioBatch([ImuSample(r[0], r[1:4], r[4:7]) for r in imu[mi]],
                                [EncoderSample(r[0], int(r[1]), int(r[2])) for r in enc[me]]))
        prev = t
    return out


def replay(session, rows, prop, settings, spec, model):
    settings = dict(settings)
    odometry = settings.pop("odometry", "command")
    cfg = VSLAMConfig()
    for k, v in settings.items():
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown VSLAMConfig field {k}")
        setattr(cfg, k, v)
    slam = PlanarVSLAM(model, cfg)
    if cfg.depth_floor_mask != "off":
        from amr_rl.perception.near_depth import LazyBackend, MonoDepthObstacles

        slam.floor_mask = MonoDepthObstacles(model, backend=LazyBackend(device="cpu"))
    odo = cmd_odo = None
    if odometry == "imu_encoders":
        cal_path = PROJECT_ROOT / "configs" / "calibration" / "imu.yaml"
        cal = yaml.safe_load(cal_path.read_text()) if cal_path.exists() else None
        odo = WheelInertialOdometry(spec, calibration=cal)
    if odometry in ("imu_encoders", "command_model"):
        cmd_odo = CommandModelOdometry(spec)
    out = []
    for row, batch in zip(rows, batches(prop, [r["t"] for r in rows])):
        rgb = cv2.imread(str(session / "image-evidence" / f"{row['frame_sha']}.png"))[..., ::-1].copy()
        cmd = tuple(row["cmd"]) if row.get("cmd") is not None else None
        step = None
        fallback = cmd_odo.step(cmd, row["t"]) if cmd_odo is not None else None
        if odo is not None:
            first = odo.t is None
            step = odo.step(batch, row["t"], command=cmd)
            if step is None and not first:
                step = fallback
        elif odometry == "command_model":
            step = fallback
        r = slam.track(rgb, row["t"], commanded=cmd, odometry=step)
        if odo is not None:
            for ratio, sigma in slam.pop_scale_samples():
                odo.add_scale_sample(ratio * odo.gyro_scale, sigma)
            for rate in slam.pop_bias_feedback():
                odo.add_bias_feedback(rate)
            for ratio in slam.pop_radius_samples():
                odo.add_radius_sample(ratio)
        est = None if r.pose is None else [float(v) for v in r.pose]
        herr = None if est is None else wrap(est[2] - row["gt"][2])
        out.append({"t": row["t"], "gt": row["gt"], "est": est, "herr": herr, "status": r.status, "cmd": row.get("cmd")})
    if odo is not None:
        print(f"  odometry: gyro_scale {odo.gyro_scale:.5f} radius_scale {odo.radius_scale:.4f} "
              f"({len(odo.radius_samples)} samples) track_scale {odo.track_scale:.3f} fusion {slam.fusion_log}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--variant", nargs="+", action="append", required=True,
                    help="name key=value ... (VSLAMConfig fields, JSON values; odometry=command|imu_encoders|command_model)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    spec = RobotSpec.load()
    model = CameraModel.from_spec(spec)
    results = {}
    for run in args.runs:
        session, rows, prop = load_run(run)
        for v in args.variant:
            name, kv = v[0], dict(item.split("=", 1) for item in v[1:])
            settings = {k: (val if k == "odometry" else json.loads(val)) for k, val in kv.items()}
            traj = replay(session, rows, prop, settings, spec, model)
            res = analyse(traj)
            results.setdefault(Path(run).name, {})[name] = res
            print(Path(run).name, name, json.dumps(res), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
