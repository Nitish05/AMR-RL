"""Offline turn-drift benchmark: replay captured in-place turns through PlanarVSLAM
variants (evaluation only; ground truth is used for initialisation and scoring).

Replay modes:
  onmap  load the saved map and start TRACKING at the true map pose (as an accepted
         relocalisation would), then track the turn on the map
  fresh  new PlanarVSLAM initialised from the first frame (as in map building);
         errors are relative to the true pose at frame 0
Every frame goes through ``track(rgb, t, commanded=cmd)`` with the command that was
in force during the preceding period (what the runtime passes).

Per sequence: signed heading error per frame, e360 / e720 (error when the true
cumulative rotation first reaches 360 / 720 deg), e_end, emax over tracking frames,
confidently-wrong tracking frames (|e| > 10 deg, not flagged degraded), loss frames
and events, estimated vs true rotation, position slide, ms per frame, and (with
captured depth) the share of IPM landmarks created during the turn that are not on
the floor.

    PYTHONPATH=src python scripts/dev/turn_bench.py work/evidence/turncap-arena-driven out.json \\
        --variant base --variant c7 depth_weighting=\\"turn\\" --replay both
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np

from amr_rl.perception.camera_model import CameraModel
from amr_rl.perception.vslam import LOST, PREDICTED, RELOCALIZING, TRACKING, PlanarVSLAM, VSLAMConfig
from amr_rl.robot.spec import RobotSpec
from amr_rl.sim import evaluator


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


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


def start_on_map(slam, pose, t0):
    """EVALUATION ONLY: begin tracking at the true pose, as an accepted relocalisation does."""
    slam._dr = None
    slam._probation = 0
    slam.pose, slam.status, slam.failures = np.asarray(pose, float).copy(), TRACKING, 0
    slam.velocity[:] = 0
    slam._motion_log.clear()
    slam.position_sigma, slam.heading_sigma = slam.cfg.sigma_floor, 0.002
    slam.last_time = t0


def false_floor(slam, created_from, frames_by_t, capture, model, t_start):
    """Scoring: IPM landmarks created during the sequence whose rendered depth along
    their pixel is clearly shorter than the floor depth they were lifted to."""
    lm = slam.lm
    new = np.flatnonzero((lm.kind == 0) & (lm.created >= t_start))
    out = {"created": int(len(new)), "not_floor": 0, "not_floor_confirmed": 0, "confirmed": int(lm.confirmed[new].sum())}
    for i in new:
        rec = frames_by_t.get(round(float(lm.created[i]), 3))
        pose = created_from.get(round(float(lm.created[i]), 3))
        if rec is None or pose is None or "depth" not in rec:
            continue
        uv, z = model.project_world(lm.pos[i:i + 1], pose)
        u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
        if not (0 <= u < model.width and 0 <= v < model.height) or z[0] <= 0:
            continue
        depth = cv2.imread(str(Path(capture) / rec["depth"]), cv2.IMREAD_UNCHANGED)
        true_z = float(depth[v, u]) / 1000.0
        if 0 < true_z < 0.85 * float(z[0]):
            out["not_floor"] += 1
            out["not_floor_confirmed"] += int(lm.confirmed[i])
    return out


def run_sequence(job):
    capture, map_dir, seq, overrides, replay = job
    model = CameraModel.from_spec(RobotSpec.load())
    cfg = VSLAMConfig(**overrides)
    frames = seq["frames"]
    if replay == "onmap":
        slam = PlanarVSLAM.load(map_dir, model, cfg)
        truth = [np.asarray(f["map"], float) for f in frames]
        start_on_map(slam, truth[0], frames[0]["t"] - 0.1)
    else:
        slam = PlanarVSLAM(model, cfg)
        w0 = np.asarray(frames[0]["world"], float)
        truth = [evaluator.to_map_frame(w0, np.asarray(f["world"], float)) for f in frames]
    rows, created_from, ms = [], {}, []
    for f, tr in zip(frames, truth):
        rgb = cv2.imread(str(Path(capture) / f["file"]))[..., ::-1].copy()
        t0 = time.time()
        r = slam.track(rgb, f["t"], commanded=tuple(f["cmd"]))
        ms.append(1000 * (time.time() - t0))
        est = None if r.pose is None else [float(v) for v in r.pose]
        if est is not None:
            created_from[round(f["t"], 3)] = np.asarray(est)
        rows.append({"t": f["t"], "status": r.status, "est": est, "truth": [float(v) for v in tr],
                     "e": None if est is None else wrap(est[2] - tr[2]), "inliers": int(r.inliers),
                     "kf": bool(r.keyframe), "degraded": bool(r.info.get("degraded", False)),
                     "turn": bool(r.info.get("turn", False)), "closure": r.info.get("turn_closure")})
    true_cum = np.concatenate([[0.0], np.cumsum([abs(wrap(b["truth"][2] - a["truth"][2])) for a, b in zip(rows[:-1], rows[1:])])])

    def e_at(deg):
        k = np.flatnonzero(true_cum >= math.radians(deg))
        if not len(k):
            return None
        e = rows[k[0]]["e"]
        return None if e is None else math.degrees(abs(e))

    trk = [r for r in rows if r["status"] == TRACKING and r["e"] is not None]
    est_rot = sum(abs(wrap(b["est"][2] - a["est"][2])) for a, b in zip(rows[:-1], rows[1:])
                  if a["est"] is not None and b["est"] is not None and b["status"] == TRACKING)
    tru_rot = sum(abs(wrap(b["truth"][2] - a["truth"][2])) for a, b in zip(rows[:-1], rows[1:])
                  if a["est"] is not None and b["est"] is not None and b["status"] == TRACKING)
    first_est = next((np.asarray(r["est"][:2]) for r in rows if r["est"] is not None), None)
    slide = max((float(np.linalg.norm(np.asarray(r["est"][:2]) - first_est)) for r in rows if r["est"] is not None),
                default=None)
    true_slide = max(float(np.linalg.norm(np.asarray(r["truth"][:2]) - np.asarray(rows[0]["truth"][:2]))) for r in rows)
    lost = [r["status"] in (LOST, RELOCALIZING) for r in rows]
    res = {"id": seq["id"], "pattern": seq["pattern"], "bin": seq["bin"], "object_m": seq["object_m"],
           "position": seq["position"], "replay": replay, "frames": len(rows),
           "e360": e_at(360), "e720": e_at(720),
           "e_end": None if rows[-1]["e"] is None else math.degrees(abs(rows[-1]["e"])),
           "emax": max((math.degrees(abs(r["e"])) for r in trk), default=None),
           "conf_wrong": sum(abs(r["e"]) > math.radians(10) and not r["degraded"] for r in trk),
           "flagged_wrong": sum(abs(r["e"]) > math.radians(10) and r["degraded"] for r in trk),
           "tracking": len(trk), "lost": int(sum(lost)), "predicted": sum(r["status"] == PREDICTED for r in rows),
           "loss_events": sum(1 for a, b in zip(lost[:-1], lost[1:]) if b and not a),
           "ratio": est_rot / tru_rot if tru_rot > 0.3 else None, "true_rot_deg": math.degrees(true_cum[-1]),
           "slide": slide, "true_slide": true_slide, "closures": sum(r["closure"] is not None for r in rows),
           "ms_median": float(np.median(ms)), "ms_p95": float(np.percentile(ms, 95)),
           "hash": hashlib.sha1(json.dumps([r["est"] for r in rows]).encode()).hexdigest()[:12],
           "rows": rows}
    if any("depth" in f for f in frames):
        res["false_floor"] = false_floor(slam, created_from, {round(f["t"], 3): f for f in frames}, capture, model,
                                         frames[0]["t"])
    return res


def pct(values, q):
    v = [x for x in values if x is not None]
    return float(np.percentile(v, q)) if v else None


def boot_p90(values, n=1000, seed=0):
    v = np.array([x for x in values if x is not None])
    if len(v) < 3:
        return None
    rng = np.random.default_rng(seed)
    s = [np.percentile(rng.choice(v, len(v)), 90) for _ in range(n)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def summarise(results):
    groups = {}
    for r in results:
        for key in (("all", r["replay"], "all"), (r["bin"], r["replay"], "all"), ("all", r["replay"], r["pattern"])):
            groups.setdefault(key, []).append(r)
    out = []
    for (b, replay, pattern), rs in sorted(groups.items()):
        frames = sum(r["frames"] for r in rs)
        trk = sum(r["tracking"] for r in rs)
        out.append({
            "bin": b, "replay": replay, "pattern": pattern, "n": len(rs),
            "e360_median": pct([r["e360"] for r in rs], 50), "e360_p90": pct([r["e360"] for r in rs], 90),
            "e360_p90_ci": boot_p90([r["e360"] for r in rs]),
            "e720_p90": pct([r["e720"] for r in rs], 90), "e_end_p90": pct([r["e_end"] for r in rs], 90),
            "emax_p90": pct([r["emax"] for r in rs], 90),
            "turns_over_5": sum((r["emax"] or 0) > 5 for r in rs) / len(rs),
            "turns_over_10": sum((r["emax"] or 0) > 10 for r in rs) / len(rs),
            "conf_wrong_frac": sum(r["conf_wrong"] for r in rs) / max(1, trk),
            "lost_frac": sum(r["lost"] for r in rs) / max(1, frames),
            "loss_events_per_seq": sum(r["loss_events"] for r in rs) / len(rs),
            "ratio_median": pct([r["ratio"] for r in rs], 50),
            "slide_p90": pct([r["slide"] for r in rs], 90), "true_slide_p90": pct([r["true_slide"] for r in rs], 90),
            "ms_p95": pct([r["ms_p95"] for r in rs], 95),
            "not_floor_frac": (sum(r["false_floor"]["not_floor"] for r in rs if "false_floor" in r)
                               / max(1, sum(r["false_floor"]["created"] for r in rs if "false_floor" in r)))
            if any("false_floor" in r for r in rs) else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--variant", nargs="+", action="append", required=True, help="name [key=value ...]")
    ap.add_argument("--replay", default="both", choices=["onmap", "fresh", "both"])
    ap.add_argument("--positions", default="all", help="all | even | odd | comma list of position ids")
    ap.add_argument("--patterns", nargs="*", default=None)
    ap.add_argument("--map", default=None, help="map for onmap replay (default: the capture's map)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--repeat", type=int, default=1, help="2 = determinism check (hashes must match)")
    ap.add_argument("--keep-rows", action="store_true")
    args = ap.parse_args()
    meta = json.loads((Path(args.capture) / "sequences.json").read_text())
    seqs = meta["sequences"]
    if args.positions in ("even", "odd"):
        seqs = [s for s in seqs if s["position"] % 2 == (0 if args.positions == "even" else 1)]
    elif args.positions != "all":
        keep = {int(x) for x in args.positions.split(",")}
        seqs = [s for s in seqs if s["position"] in keep]
    if args.patterns:
        seqs = [s for s in seqs if s["pattern"] in args.patterns]
    map_dir = args.map or meta["map_dir"]
    replays = ["onmap", "fresh"] if args.replay == "both" else [args.replay]
    if map_dir is None:
        replays = [r for r in replays if r != "onmap"]
    report = {"capture": args.capture, "map_dir": map_dir, "world": meta["world"], "mode": meta["mode"],
              "positions": args.positions, "variants": {}}
    for spec in args.variant:
        name, overrides = spec[0], parse_overrides(spec[1:])
        jobs = [(args.capture, map_dir, s, overrides, rp) for rp in replays for s in seqs] * args.repeat
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(run_sequence, jobs)
        if args.repeat > 1:
            n = len(jobs) // args.repeat
            same = all(a["hash"] == b["hash"] for a, b in zip(results[:n], results[n:2 * n]))
            print(f"   determinism: {'identical' if same else 'DIFFERENT'}")
            results = results[:n]
        if not args.keep_rows:
            for r in results:
                r.pop("rows")
        summary = summarise(results)
        report["variants"][name] = {"overrides": overrides, "summary": summary, "results": results,
                                    "wall_seconds": time.time() - t0}
        print(f"== {name} {overrides} ({time.time() - t0:.0f} s)")
        for row in summary:
            if row["pattern"] == "all":
                print(f"   {row['replay']:5s} {row['bin']:8s} n{row['n']:3d} e360 med {row['e360_median']} p90 "
                      f"{row['e360_p90']} e720 p90 {row['e720_p90']} emax p90 {row['emax_p90']} "
                      f">10 {row['turns_over_10']:.2f} lost {row['lost_frac']:.3f} confwrong {row['conf_wrong_frac']:.3f} "
                      f"ratio {row['ratio_median']} notfloor {row['not_floor_frac']}")
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
