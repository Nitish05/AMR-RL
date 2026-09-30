"""Development check: open-loop scripted drive, RGB-only VSLAM vs ground truth."""
import json
import math
import sys
import time

import cv2
import numpy as np

from amr_rl.control.contract import DriveCommand
from amr_rl.mapping.occupancy import FloorEvidenceMapper, OccupancyGrid
from amr_rl.mapping.render import render_map
from amr_rl.perception.camera_model import CameraModel
from amr_rl.perception.vslam import PlanarVSLAM
from amr_rl.sim import evaluator
from amr_rl.sim.world import SimWorld, load_world_config

world_name = sys.argv[1] if len(sys.argv) > 1 else "home_a"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/claude-0/scr/vslam_ol"
import os; os.makedirs(out, exist_ok=True)
w = SimWorld(load_world_config(world_name), inspection=False)
model = CameraModel.from_spec(w.spec)
slam = PlanarVSLAM(model)
grid = OccupancyGrid(slam.map_version)
mapper = FloorEvidenceMapper(model, grid)
grid.mark_free_disc((0,0), 0.30)
traj=[]
for _ in range(30): w.step()
origin = evaluator.true_pose(w)
script = [(0.2, 0, 5), (0, 0.6, 11), (0.15, 0.25, 6), (0, -0.6, 5), (0.2, 0.0, 3), (0, 0.6, 6), (0.15, -0.2, 6)]
rows = []; t0 = time.time(); gen = 0; seq = 0
for v, wz, dur in script:
    steps = int(dur / w.dt)
    for k in range(steps):
        if k % 5 == 0:
            now = w.time
            w.backend.command(DriveCommand(v, wz, now, now + 0.2, gen, "manual"), now=now)
        w.step()
        if k % 10 == 0:
            f = w.capture(model.version)
            r = slam.track(f.rgb, f.timestamp, commanded=(v, wz))
            if r.pose is not None:
                traj.append(r.pose[:2].tolist())
                grid.mark_footprint(r.pose, 0.35, 0.30, 0.215)
            if r.keyframe and len(slam.keyframes) >= 2:
                res = mapper.update_keyframe(slam.keyframes)
            gt = evaluator.to_map_frame(origin, evaluator.true_pose(w))
            e = evaluator.pose_error(r.pose, gt) if r.pose is not None else (None, None)
            rows.append({"t": f.timestamp, "status": r.status, "est": None if r.pose is None else r.pose.tolist(),
                         "gt": gt.tolist(), "err": e[0], "herr": e[1], "sigma": r.position_sigma, "inl": r.inliers, "kf": r.keyframe, "reason": r.reason})
            if len(rows) % 50 == 0:
                print(f"t={f.timestamp:.1f} {r.status} inl={r.inliers} err={e[0]} sigma={r.position_sigma} lm={len(slam.lm)} kfs={len(slam.keyframes)}", flush=True)
errs = [r["err"] for r in rows if r["err"] is not None]
track = sum(r["status"] == "tracking" for r in rows)
gt_len = sum(math.hypot(rows[i]["gt"][0]-rows[i-1]["gt"][0], rows[i]["gt"][1]-rows[i-1]["gt"][1]) for i in range(1, len(rows)))
summary = {"ba_runs": getattr(slam, "ba_runs", 0), "frames": len(rows), "tracking_frames": track, "ate_rmse": float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
           "max_err": max(errs) if errs else None, "final_err": errs[-1] if errs else None, "path_length": gt_len,
           "max_heading_err_deg": math.degrees(max(r["herr"] for r in rows if r["herr"] is not None)),
           "wall_s": time.time() - t0, "landmarks": len(slam.lm), "keyframes": len(slam.keyframes)}
summary["map"] = evaluator.score_map(w, grid, origin)
summary["mapper"] = mapper.stats
img, meta = render_map(grid, trajectory=traj, pose=slam.pose)
cv2.imwrite(f"{out}/{world_name}_map.png", img[..., ::-1])
truth = evaluator.true_obstacle_mask(w, grid, origin)
print(json.dumps(summary, indent=1))
json.dump({"summary": summary, "rows": rows}, open(f"{out}/{world_name}.json", "w"))
