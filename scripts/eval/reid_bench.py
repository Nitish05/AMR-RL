"""Re-identification benchmark on detector captures (evaluation only; perception swap 5b).

Crops of the same fixture seen from different poses should be more similar than
crops of different fixtures. For a sample of ground-truth body boxes (untruncated,
>= --min-pixels), every pair is labelled same/different (scoring only) and scored
by each descriptor's similarity; reported as ROC AUC and the best balanced
threshold.

Descriptors:
  dinov2   cosine similarity of the DINOv2-S CLS embedding of the crop (learned)
  hue      negative circular hue distance of the crop's saturated pixels (the colour
           detector's identity cue, engineered)

    PYTHONPATH=src python scripts/eval/reid_bench.py work/evidence/detcap-home_a_textured-20261003 out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from amr_rl.perception.entities import circular_mean_deg, hue_distance
from amr_rl.perception.reid import CropEmbedder


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.sum()
    neg = len(labels) - pos
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / max(1, pos * neg))


def crop_hue(rgb, box):
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sat = hsv[hsv[:, 1] > 80]
    return circular_mean_deg(sat[:, 0] * 2.0) if len(sat) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--per-object", type=int, default=60)
    ap.add_argument("--min-pixels", type=int, default=400)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cap = Path(args.capture)
    meta = json.loads((cap / "frames.json").read_text())
    by = {}
    for fr in meta["frames"]:
        for o in fr["objects"]:
            if not o["truncated"] and o["pixels"] >= args.min_pixels:
                by.setdefault(o["name"], []).append((fr["file"], o["box"]))
    rng = np.random.default_rng(args.seed)
    items = []
    for name, lst in sorted(by.items()):
        for i in rng.permutation(len(lst))[:args.per_object]:
            items.append((name, *lst[i]))
    emb = CropEmbedder(device=args.device)
    E, H = [], []
    for _, file, box in items:
        rgb = cv2.imread(str(cap / file))[..., ::-1].copy()
        E.append(emb.embed(rgb, box))
        H.append(crop_hue(rgb, box))
    E = np.stack(E)
    names = [it[0] for it in items]
    iu = np.triu_indices(len(items), 1)
    same = np.array([names[a] == names[b] for a, b in zip(*iu)])
    cos = (E @ E.T)[iu]
    hue = -np.array([hue_distance(H[a], H[b]) for a, b in zip(*iu)])
    report = {"capture": str(cap), "world": meta["world"], "crops": {n: sum(x == n for x in names) for n in set(names)},
              "pairs": int(len(same)), "same_pairs": int(same.sum())}
    for key, s in (("dinov2", cos), ("hue", hue)):
        thr = np.quantile(s, np.linspace(0.01, 0.99, 99))
        bal = [(0.5 * ((s[same] >= t).mean() + (s[~same] < t).mean()), float(t)) for t in thr]
        best = max(bal)
        report[key] = {"auroc": auroc(s, same), "best_balanced_accuracy": best[0], "threshold": best[1],
                       "same_median": float(np.median(s[same])), "different_p99": float(np.percentile(s[~same], 99))}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
