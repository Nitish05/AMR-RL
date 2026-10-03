"""Paired loop-closure evaluation from shadow-mode builds (evaluation only).

A shadow build (``build_map.py --loop-shadow``) detects, verifies and confirms loops
exactly as live but only records the accepted constraints. Here the final keyframe
poses of that same run are optimised with and without (subsets of) those
constraints in the same pose graph as live, and each result is scored against the
true pose at every keyframe's time. Live on/off runs diverge after the first
closure; this comparison does not.

Rules (subsets of the recorded constraints):
  none   no loop closure
  r5     round-5 candidates: the 3 most similar
  new    round-6 candidates: the 2 most similar, or any with pose-graph sigma >= 3 cm
  all    every recorded constraint

    PYTHONPATH=src python scripts/dev/loop_shadow_eval.py work/evidence/shadow-* --out report.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from common import _truth_at  # noqa: E402

from amr_rl.perception.pose_graph import PoseGraph, between  # noqa: E402

RULES = {
    "none": lambda e: False,
    "r5": lambda e: e.get("rank", 99) < 3,
    "new": lambda e: e.get("rank", 99) < 2 or (e.get("sigma_m") or 0.0) >= 0.03,
    "all": lambda e: True,
}


def optimise(kfs, edges, covis=()):
    g = PoseGraph()
    ordered = sorted(kfs, key=lambda k: k["id"])
    for k in ordered:
        g.add_node(k["id"], np.array(k["pose"]), fixed=(k["id"] == ordered[0]["id"]))
    for a, b in zip(ordered[:-1], ordered[1:]):
        rel = between(np.array(a["pose"]), np.array(b["pose"]))
        sx = 0.01 + 0.03 * float(np.hypot(rel[0], rel[1]))
        g.add_edge(a["id"], b["id"], rel, (sx, sx, 0.01 + 0.02 * abs(rel[2])))
    for e in edges:
        g.add_edge(e["candidate"], e["kf"], np.array(e["z"]), (0.03, 0.03, 0.02), loop=True)
    for a, b, z, _ in covis:
        if a in g.poses and b in g.poses:
            g.add_edge(a, b, np.array(z), (0.02, 0.02, 0.01))
    return g.optimize() if edges else {k["id"]: np.array(k["pose"]) for k in ordered}


def score(kfs, poses, truth):
    errs = [math.dist(poses[k["id"]][:2], _truth_at(truth, k["t"])[:2]) for k in kfs]
    return float(np.sqrt(np.mean(np.square(errs)))), float(max(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for d in args.runs:
        d = Path(d)
        kfs = json.loads((d / "map" / "map.json").read_text())["keyframes"]
        truth = [json.loads(line) for line in open(d / "session" / "trajectory_scoring.jsonl")]
        log = json.loads((d / "result.json").read_text())["loop_closure"]["log"]
        shadow = [e for e in log if e.get("outcome") == "shadow_closed"]
        covis = json.loads((d / "result.json").read_text())["loop_closure"].get("covis_edges", [])
        row = {"run": d.name, "keyframes": len(kfs), "constraints": len(shadow)}
        for name, rule in RULES.items():
            edges = [e for e in shadow if rule(e)]
            rmse, mx = score(kfs, optimise(kfs, edges, covis), truth)
            row[name] = {"edges": len(edges), "rmse_m": rmse, "max_m": mx}
        rows.append(row)
        print(f"{d.name:28s} constraints {len(shadow):3d} | " + " | ".join(
            f"{n} {100 * row[n]['rmse_m']:5.1f} cm ({row[n]['edges']})" for n in RULES), flush=True)
    for n in ("r5", "new", "all"):
        better = sum(r[n]["rmse_m"] < r["none"]["rmse_m"] - 0.002 for r in rows)
        worse = sum(r[n]["rmse_m"] > r["none"]["rmse_m"] + 0.002 for r in rows)
        print(f"{n}: better than none in {better}/{len(rows)}, worse in {worse}/{len(rows)}; "
              f"median change {100 * np.median([r[n]['rmse_m'] - r['none']['rmse_m'] for r in rows]):+.1f} cm")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
