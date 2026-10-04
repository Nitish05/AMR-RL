"""Offline relocalisation probe: replay captured onboard frames through a freshly
loaded map (evaluation only; ground truth is used for scoring).

Two measurements per VSLAM configuration ("variant"):

* single frame: ``global_localize`` on every captured frame -> recall (pose within
  0.10 m and 5 deg), imprecise (accepted but between that and the false threshold),
  false (> 0.2 m or > 8.6 deg), and time per call, binned by distance to the
  nearest map keyframe;
* sequence: each position's turn replayed through ``track`` the way a phase start
  sees it (a few stationary frames, then an in-place turn at 0.4 rad/s with that
  command passed to the VSLAM) -> degrees turned until the VSLAM accepts a pose and
  whether the accepted pose is right.

    PYTHONPATH=src python scripts/dev/reloc_probe.py work/evidence/reloc-capture-<stamp> out.json \\
        --variant legacy reloc_method=pnp --variant planar reloc_method=planar2pt
"""

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np

from amr_rl.perception.camera_model import CameraModel
from amr_rl.perception.vslam import TRACKING, PlanarVSLAM, VSLAMConfig
from amr_rl.robot.spec import RobotSpec

GOOD_M, GOOD_RAD = 0.10, math.radians(5)
FALSE_M, FALSE_RAD = 0.2, 0.15
W_TURN = 0.4


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def judge(est, truth):
    if est is None:
        return "none", None, None
    e = float(np.hypot(est[0] - truth[0], est[1] - truth[1]))
    h = abs(wrap(est[2] - truth[2]))
    if e <= GOOD_M and h <= GOOD_RAD:
        return "good", e, h
    if e > FALSE_M or h > FALSE_RAD:
        return "false", e, h
    return "imprecise", e, h


def parse_overrides(items):
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


_PLACE = {}


def _place(name):
    if not name:
        return None
    if name not in _PLACE:  # one model per worker process
        from amr_rl.perception.place_recognition import make_descriptor

        _PLACE[name] = make_descriptor(name)
    return _PLACE[name]


def run_position(job):
    capture, map_dir, pos, overrides, single, step_deg = job
    model = CameraModel.from_spec(RobotSpec.load())
    overrides = dict(overrides)
    place_name = overrides.pop("place", None)  # e.g. place=megaloc: retrieval-first relocalisation
    place = _place(place_name)
    cfg = VSLAMConfig(**overrides)
    frames = [cv2.imread(str(Path(capture) / f["file"]))[..., ::-1].copy() for f in pos["frames"]]
    truth = [f["map"] for f in pos["frames"]]
    res = {"id": pos["id"], "kind": pos["kind"], "start_index": pos.get("start_index"),
           "kf_distance": pos["kf_distance"], "single": [], "sequence": None}
    if single:
        slam = PlanarVSLAM.load(map_dir, model, cfg)
        slam.place = place
        for rgb, tr in zip(frames, truth):
            _, pts, desc = slam.features(rgb)
            t0 = time.time()
            out = slam.global_localize(pts, desc, rgb=rgb)
            ms = 1000 * (time.time() - t0)
            verdict, e, h = judge(None if out is None else out[0], tr)
            res["single"].append({"verdict": verdict, "err": e, "herr": h, "ms": ms, "info": dict(slam._reloc_info)})
    # Sequence: 5 stationary frames, then the turn (command passed to the VSLAM).
    slam = PlanarVSLAM.load(map_dir, model, cfg)
    slam.place = place
    dt_turn = math.radians(step_deg) / W_TURN
    t, seq = 0.0, None
    order = [(0, (0.0, 0.0), 0.1)] * 5 + [(k, (0.0, W_TURN), dt_turn) for k in range(1, len(frames))]
    for k, cmd, dt in order:
        t += dt
        r = slam.track(frames[k], t, commanded=cmd)
        if r.status == TRACKING:
            verdict, e, h = judge(r.pose, truth[k])
            seq = {"accepted": True, "frame": k, "deg_turned": k * step_deg, "verdict": verdict, "err": e, "herr": h,
                   "log": slam.reloc_log[-4:]}
            break
    res["sequence"] = seq or {"accepted": False, "deg_turned": (len(frames) - 1) * step_deg}
    return res


def summarise(results):
    bins = [(0.0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 9.0)]
    rows = []
    for lo, hi in bins:
        rs = [r for r in results if lo <= r["kf_distance"] < hi]
        single = [s for r in rs for s in r["single"]]
        seq = [r["sequence"] for r in rs]
        rows.append({
            "bin": f"{lo:.2f}-{hi:.2f} m", "positions": len(rs), "frames": len(single),
            "single_good": sum(s["verdict"] == "good" for s in single),
            "single_imprecise": sum(s["verdict"] == "imprecise" for s in single),
            "single_false": sum(s["verdict"] == "false" for s in single),
            "seq_good": sum(s.get("verdict") == "good" for s in seq),
            "seq_imprecise": sum(s.get("verdict") == "imprecise" for s in seq),
            "seq_false": sum(s.get("verdict") == "false" for s in seq),
            "seq_never": sum(not s["accepted"] for s in seq),
            "median_deg_to_accept": float(np.median([s["deg_turned"] for s in seq if s["accepted"]]))
            if any(s["accepted"] for s in seq) else None,
        })
    ms = [s["ms"] for r in results for s in r["single"]]
    starts = {r["start_index"]: r["sequence"] for r in results if r["kind"] == "start"}
    return {"bins": rows, "ms_median": float(np.median(ms)) if ms else None,
            "ms_p95": float(np.percentile(ms, 95)) if ms else None, "starts": starts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--variant", nargs="+", action="append", required=True,
                    help="name [key=value ...] (VSLAMConfig overrides)")
    ap.add_argument("--positions", default="all", help="all | even | odd | comma list of ids")
    ap.add_argument("--no-single", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--map", default=None, help="probe another map of the same scenario (same map origin); "
                    "positions keep the capture map's distance bins")
    args = ap.parse_args()
    meta = json.loads((Path(args.capture) / "poses.json").read_text())
    positions = meta["positions"]
    if args.positions == "even":
        positions = [p for p in positions if p["id"] % 2 == 0]
    elif args.positions == "odd":
        positions = [p for p in positions if p["id"] % 2 == 1]
    elif args.positions != "all":
        keep = {int(x) for x in args.positions.split(",")}
        positions = [p for p in positions if p["id"] in keep]
    map_dir = args.map or meta["map_dir"]
    report = {"capture": args.capture, "map_dir": map_dir, "capture_map_dir": meta["map_dir"],
              "positions": args.positions, "variants": {}}
    for spec in args.variant:
        name, overrides = spec[0], parse_overrides(spec[1:])
        jobs = [(args.capture, map_dir, p, overrides, not args.no_single, meta["step_deg"]) for p in positions]
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(run_position, jobs)
        summary = summarise(results)
        report["variants"][name] = {"overrides": overrides, "summary": summary, "results": results,
                                    "wall_seconds": time.time() - t0}
        print(f"== {name} {overrides} ({time.time() - t0:.0f} s)")
        for row in summary["bins"]:
            print("  ", row)
        print("   starts:", {k: (v.get("verdict"), v.get("deg_turned")) for k, v in summary["starts"].items()})
        print(f"   ms/frame median {summary['ms_median']}, p95 {summary['ms_p95']}")
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
