"""Place-recognition quality in the simulated rooms (evaluation only).

Uses a relocalisation probe capture (``reloc_capture.py``: full turns at known
poses). Even positions form the database, odd positions are the queries. A
retrieved database frame is correct if it was taken within ``--max-m`` metres and
``--max-deg`` degrees of the query (ground truth used for scoring only). Reports
recall@1 / recall@5 over queries that have at least one correct database frame,
and the descriptor time per image.

    PYTHONPATH=src python scripts/dev/vpr_eval.py work/evidence/reloc-capture-<stamp> --methods megaloc orb_bow
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from amr_rl.perception.place_recognition import make_descriptor


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--methods", nargs="+", default=["megaloc", "orb_bow"])
    ap.add_argument("--max-m", type=float, default=0.5)
    ap.add_argument("--max-deg", type=float, default=30.0)
    ap.add_argument("--step", type=int, default=2, help="use every n-th heading (speed)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    meta = json.loads((Path(args.capture) / "poses.json").read_text())
    frames = []
    for pos in meta["positions"]:
        for k, f in enumerate(pos["frames"]):
            if k % args.step == 0:
                frames.append({"pos": pos["id"], "file": f["file"], "world": f["world"]})
    db = [f for f in frames if f["pos"] % 2 == 0]
    q = [f for f in frames if f["pos"] % 2 == 1]
    W = np.array([f["world"] for f in db])
    report = {"capture": args.capture, "db": len(db), "queries": len(q), "max_m": args.max_m,
              "max_deg": args.max_deg, "methods": {}}
    images = {f["file"]: cv2.imread(str(Path(args.capture) / f["file"]))[..., ::-1].copy() for f in frames}
    for name in args.methods:
        desc = make_descriptor(name, map_dir=meta["map_dir"])
        t0 = time.time()
        D = np.stack([desc.describe(images[f["file"]]) for f in db])
        Q = np.stack([desc.describe(images[f["file"]]) for f in q])
        ms = 1000 * (time.time() - t0) / (len(db) + len(q))
        sims = Q @ D.T
        r1 = r5 = n = 0
        for i, f in enumerate(q):
            d = np.hypot(W[:, 0] - f["world"][0], W[:, 1] - f["world"][1])
            h = np.abs([wrap(w - f["world"][2]) for w in W[:, 2]])
            good = (d <= args.max_m) & (h <= math.radians(args.max_deg))
            if not good.any():
                continue
            n += 1
            order = np.argsort(-sims[i])
            r1 += bool(good[order[0]])
            r5 += bool(good[order[:5]].any())
        report["methods"][name] = {"queries_with_match": n, "recall@1": r1 / max(1, n), "recall@5": r5 / max(1, n),
                                   "ms_per_image": ms, "dim": int(D.shape[1])}
        print(name, report["methods"][name], flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
