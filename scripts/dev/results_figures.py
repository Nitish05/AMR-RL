"""Render README result figures from evaluation evidence.

    PYTHONPATH=src python scripts/dev/results_figures.py \
        --learning work/evidence/learning-20260930-022340 work/evidence/learning-20260930-042009 \
        --nav work/evidence/navigation-20260930-024431 --out docs/media
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
RUNS = [("learned_standard", "Learned (this project)"), ("baseline_fixed", "Fixed order"),
        ("baseline_random", "Random"), ("baseline_nearest", "Nearest first")]


def load(dirs, name):
    for d in dirs:
        f = Path(d) / name / "result.json"
        if f.exists():
            r = json.loads(f.read_text())
            if r.get("wall_seconds") is not None:
                return r["phases"][0]
    raise FileNotFoundError(name)


def policy_chart(dirs, out):
    per, total = [], []
    for key, _ in RUNS:
        s = load(dirs, key)["score"]
        total.append(s["total_valence"])
        per.append(s["total_valence"] / max(1, s["outcomes_learned"]))
    labels = [label for _, label in RUNS]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0), dpi=160, facecolor=SURFACE)
    for ax, vals, title, fmt in ((axes[0], per, "Valence per observed outcome", "{:.2f}"),
                                 (axes[1], total, "Total valence in 480 s", "{:.1f}")):
        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals, height=0.56, color=SERIES, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.02, yi, fmt.format(v), va="center", ha="left", fontsize=9, color=INK)
        ax.set_yticks(y, labels, fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
        ax.set_facecolor(SURFACE)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=INK2, labelsize=8)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[1].set_yticks([])
    fig.text(0.01, 0.015, "Standard rules, same map and start, one seed each. The fixed-order baseline never rests and "
             "happened to repeat the rewarding option;\nthe learned policy idles 27 % of the time by design "
             "(engineered stimulation need).", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def map_grid(nav_dir, out):
    worlds = [("home_a", "home_a (development)"), ("heldout_b", "heldout_b (held-out layout)"),
              ("heldout_c", "heldout_c (held-out appearance)"), ("home_a_dim", "home_a_dim (held-out lighting)")]
    tiles = []
    for w, label in worlds:
        img = cv2.imread(str(Path(nav_dir) / w / "map.png"))
        r = json.loads((Path(nav_dir) / w / "result.json").read_text())
        m, t = r["mapping"]["map"], r["mapping"]["trajectory"]
        img = cv2.resize(img, (420, int(img.shape[0] * 420 / img.shape[1])))
        pad = np.full((56, img.shape[1], 3), 252, np.uint8)
        cv2.putText(pad, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (11, 11, 11), 1, cv2.LINE_AA)
        cv2.putText(pad, f"coverage {m['free_coverage']:.0%}  ATE {t['ate_rmse_m'] * 100:.1f} cm  "
                         f"false-free {m['false_free_cells']}", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (78, 81, 82), 1, cv2.LINE_AA)
        tiles.append(np.vstack([pad, img]))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 4, 4, cv2.BORDER_CONSTANT, value=(252, 252, 252)) for t in tiles]
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(str(out), grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learning", nargs="+", required=True)
    ap.add_argument("--nav", required=True)
    ap.add_argument("--out", default="docs/media")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    policy_chart(a.learning, out / "policy-comparison.png")
    map_grid(a.nav, out / "nav-maps.png")
    print("wrote", out / "policy-comparison.png", out / "nav-maps.png")


if __name__ == "__main__":
    main()
