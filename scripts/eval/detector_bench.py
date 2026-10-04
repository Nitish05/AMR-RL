"""Score object detectors on frames from scripts/dev/detector_capture.py (evaluation only).

Per frame, detections are matched to ground-truth fixture boxes (greedy, IoU >= 0.5).
Reported, binned by true range:
  recall (matched / ground-truth objects; truncated objects counted separately),
  precision and false positives per frame,
  position error of matched detections (map position with the TRUE robot pose,
  so localisation is scored independently of the VSLAM),
  state-token accuracy on matched objects whose raised flag is visible.

    PYTHONPATH=src python scripts/eval/detector_bench.py work/evidence/detcap-arena out.json --detector fixture
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from amr_rl.perception.camera_model import CameraModel
from amr_rl.robot.spec import RobotSpec

BINS = [(0.0, 1.0), (1.0, 1.5), (1.5, 9.0)]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1)


def make_detector(name, model):
    if name == "fixture":
        from amr_rl.perception.entities import FixtureDetector

        return FixtureDetector(model)
    raise ValueError(name)


def to_map_pose(pose_world):
    return np.asarray(pose_world, float)  # scored in the world frame with the true pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--detector", default="fixture")
    ap.add_argument("--min-flag-pixels", type=int, default=40, help="a raised flag counts as visible above this")
    args = ap.parse_args()
    cap = Path(args.capture)
    meta = json.loads((cap / "frames.json").read_text())
    model = CameraModel.from_spec(RobotSpec.load())
    det = make_detector(args.detector, model)
    rows, times = [], []
    for fr in meta["frames"]:
        rgb = cv2.imread(str(cap / fr["file"]))[..., ::-1].copy()
        pose = to_map_pose(fr["pose_world"])
        t0 = time.time()
        dets = det.detect(rgb, pose)
        times.append(time.time() - t0)
        gts = fr["objects"]
        pairs = sorted(((iou(d.bbox, g["box"]), i, j) for i, d in enumerate(dets) for j, g in enumerate(gts)), reverse=True)
        used_d, used_g, matches = set(), set(), {}
        for v, i, j in pairs:
            if v < 0.5 or i in used_d or j in used_g:
                continue
            used_d.add(i)
            used_g.add(j)
            matches[j] = i
        for j, g in enumerate(gts):
            rng = float(np.hypot(g["xy_world"][0] - pose[0], g["xy_world"][1] - pose[1]))
            row = {"frame": fr["file"], "name": g["name"], "range": rng, "truncated": g["truncated"], "matched": j in matches}
            if j in matches:
                d = dets[matches[j]]
                if d.position is not None:
                    row["pos_err"] = float(np.hypot(d.position[0] - g["xy_world"][0], d.position[1] - g["xy_world"][1]))
                if g["flag"] is None or g["flag_pixels"] >= args.min_flag_pixels:
                    truth = f"attach:{g['flag'] or 'none'}"
                    row["state_truth"], row["state_pred"] = truth, d.state_token
            rows.append(row)
        rows.append({"frame": fr["file"], "false_positives": len(dets) - len(used_d), "detections": len(dets)})
    objs = [r for r in rows if "name" in r]
    fpf = [r for r in rows if "false_positives" in r]
    report = {"capture": str(cap), "world": meta["world"], "detector": args.detector, "frames": len(fpf), "bins": []}
    for lo, hi in BINS:
        rs = [r for r in objs if lo <= r["range"] < hi and not r["truncated"]]
        errs = [r["pos_err"] for r in rs if "pos_err" in r]
        st = [r for r in rs if "state_truth" in r]
        report["bins"].append({
            "range": f"{lo}-{hi} m", "objects": len(rs), "recall": sum(r["matched"] for r in rs) / max(1, len(rs)),
            "pos_err_median": float(np.median(errs)) if errs else None,
            "pos_err_p90": float(np.percentile(errs, 90)) if errs else None,
            "state_accuracy": sum(r["state_truth"] == r["state_pred"] for r in st) / max(1, len(st)), "state_n": len(st),
        })
    trunc = [r for r in objs if r["truncated"]]
    n_det = sum(r["detections"] for r in fpf)
    n_fp = sum(r["false_positives"] for r in fpf)
    report.update({
        "recall_all_untruncated": sum(r["matched"] for r in objs if not r["truncated"]) / max(1, sum(not r["truncated"] for r in objs)),
        "recall_truncated": sum(r["matched"] for r in trunc) / max(1, len(trunc)),
        "precision": (n_det - n_fp) / max(1, n_det), "false_positives_per_frame": n_fp / max(1, len(fpf)),
        "state_confusion": {},
        "seconds_per_frame": {"median": float(np.median(times)), "p95": float(np.percentile(times, 95))},
    })
    for r in objs:
        if "state_truth" in r:
            c = report["state_confusion"].setdefault(r["state_truth"], {})
            c[r["state_pred"]] = c.get(r["state_pred"], 0) + 1
    report["rows"] = rows
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
