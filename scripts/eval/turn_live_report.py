"""Paired live-validation report for turn handling (evaluation only; scoring truth).

For every run directory with ``session/trajectory_scoring.jsonl`` (map builds) or
``*/phase*/trajectory_scoring.jsonl`` (learning phases):
  - map ATE and keyframe RMSE (map builds), lost-frame fraction
  - max heading error while tracking (deg)
  - in-place turns (commanded |v| < 0.02 and |w| >= 0.15 for >= 0.5 s; needs the
    harness's per-frame "cmd")
  - drift events: heading-error change >= 8 deg within 10 s of continuous tracking
    (non-overlapping), and events per 100 in-place turns

    python scripts/eval/turn_live_report.py work/evidence/live-20261004 --prefix L1
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def analyse(rows):
    out = {"frames": len(rows)}
    trk = [r for r in rows if r["status"] == "tracking" and r.get("est") is not None]
    out["lost_frac"] = sum(r["status"] in ("lost", "relocalizing") for r in rows) / max(1, len(rows))
    out["max_heading_err_deg"] = max((math.degrees(abs(r["herr"])) for r in trk if r.get("herr") is not None), default=None)
    # in-place turns from the logged command
    turns, run = 0, 0.0
    for r in rows:
        c = r.get("cmd")
        if c is not None and abs(c[0]) < 0.02 and abs(c[1]) >= 0.15:
            run += 0.1
        else:
            turns += run >= 0.5
            run = 0.0
    turns += run >= 0.5
    out["turns"] = int(turns)
    # signed heading error while tracking
    events, i = 0, 0
    seq = [(r["t"], wrap(r["est"][2] - r["gt"][2]), r["status"]) for r in rows if r.get("est") is not None]
    while i < len(seq):
        if seq[i][2] != "tracking":
            i += 1
            continue
        j, hit = i + 1, None
        while j < len(seq) and seq[j][0] - seq[i][0] <= 10.0 and seq[j][2] == "tracking":
            if abs(wrap(seq[j][1] - seq[i][1])) >= math.radians(8):
                hit = j
                break
            j += 1
        if hit is not None:
            events += 1
            i = hit + 1
        else:
            i += 1
    out["drift_events"] = events
    out["events_per_100_turns"] = 100.0 * events / turns if turns else None
    return out


def runs(root, prefix):
    found = {}
    for d in sorted(glob.glob(os.path.join(root, f"{prefix}-*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        files = [os.path.join(d, "session", "trajectory_scoring.jsonl")]
        files += sorted(glob.glob(os.path.join(d, "**", "trajectory_scoring.jsonl"), recursive=True))
        files = [f for f in dict.fromkeys(files) if os.path.exists(f)]
        if not files:
            continue
        rows_all = []
        res = {"run": name, "parts": len(files)}
        agg = []
        for f in files:
            rows = [json.loads(line) for line in open(f)]
            agg.append(analyse(rows))
            rows_all.extend(rows)
        res.update({
            "lost_frac": float(np.mean([a["lost_frac"] for a in agg])),
            "max_heading_err_deg": max((a["max_heading_err_deg"] for a in agg if a["max_heading_err_deg"] is not None),
                                       default=None),
            "turns": sum(a["turns"] for a in agg), "drift_events": sum(a["drift_events"] for a in agg)})
        res["events_per_100_turns"] = 100.0 * res["drift_events"] / res["turns"] if res["turns"] else None
        result = Path(d) / "result.json"
        if result.exists():
            r = json.loads(result.read_text())
            res["ate_cm"] = 100 * r.get("trajectory", {}).get("ate_rmse_m", float("nan"))
            res["kf_rmse_cm"] = 100 * (r.get("keyframe_map_error") or {}).get("rmse_m", float("nan"))
        found[name] = res
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--prefix", default="L1")
    args = ap.parse_args()
    found = runs(args.root, args.prefix)
    keys = ["max_heading_err_deg", "ate_cm", "kf_rmse_cm", "lost_frac", "turns", "drift_events", "events_per_100_turns"]
    print(f"{'run':34s} " + " ".join(f"{k[:14]:>14s}" for k in keys))
    for name, r in found.items():
        print(f"{name:34s} " + " ".join(f"{r.get(k):>14.3f}" if isinstance(r.get(k), float) else f"{str(r.get(k)):>14s}"
                                         for k in keys))
    for v in ("base", "fam"):
        sel = [r for n, r in found.items() if f"-{v}-" in n]
        if not sel:
            continue
        mh = [r["max_heading_err_deg"] for r in sel if r["max_heading_err_deg"] is not None]
        print(f"== {v}: runs {len(sel)}, max heading err median {np.median(mh):.1f} max {max(mh):.1f}; "
              f"lost {np.mean([r['lost_frac'] for r in sel]):.3f}; drift events {sum(r['drift_events'] for r in sel)} "
              f"in {sum(r['turns'] for r in sel)} turns")
    pairs = [(n, n.replace("-base-", "-fam-")) for n in found if "-base-" in n and n.replace("-base-", "-fam-") in found]
    if pairs and all("ate_cm" in found[a] for a, _ in pairs):
        d = [found[b]["ate_cm"] - found[a]["ate_cm"] for a, b in pairs]
        print(f"== paired ATE change fam - base: median {np.median(d):+.2f} cm, worst {max(d):+.2f} cm (n {len(d)})")


if __name__ == "__main__":
    main()
