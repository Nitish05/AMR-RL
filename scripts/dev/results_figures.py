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
POLICIES = [("history_a", "Learned (this project)"), ("baseline_fixed", "Fixed order"),
            ("baseline_random", "Random"), ("baseline_nearest", "Nearest first")]


def load_runs(dirs):
    """{(experiment, seed): first phase} from learning evidence directories."""
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*/result.json")):
            r = json.loads(f.read_text())
            if r.get("wall_seconds") is None:
                continue
            out.setdefault((r.get("experiment", f.parent.name), r.get("seed", 0)), r["phases"][0])
    return out


def policy_chart(dirs, out):
    runs = load_runs(dirs)
    labels = [label for _, label in POLICIES]
    fig, ax = plt.subplots(figsize=(8.6, 3.0), dpi=160, facecolor=SURFACE)
    for row, (key, _) in enumerate(POLICIES):
        y = len(POLICIES) - 1 - row
        pts = sorted((seed, p["score"]) for (exp, seed), p in runs.items() if exp == key and p.get("score"))
        vals = sorted(((sc["total_valence"] / max(1, sc["outcomes_learned"]), seed) for seed, sc in pts))
        for k, (v, seed) in enumerate(vals):
            ax.plot([v], [y], "o", ms=8, color=SERIES, mec=SURFACE, mew=2, zorder=3)
            above = k % 2 == 0
            ax.text(v, y + (0.2 if above else -0.2), f"s{seed}: {v:.2f}", ha="center",
                    va="bottom" if above else "top", fontsize=8, color=INK2)
    ax.set_yticks(range(len(POLICIES))[::-1], labels, fontsize=9, color=INK)
    ax.set_ylim(-0.6, len(POLICIES) - 0.3)
    ax.set_xlim(-0.05, 0.7)
    ax.set_title("Valence per observed outcome (one dot per run; s = seed)", loc="left", fontsize=10.5,
                 color=INK, pad=8)
    ax.set_facecolor(SURFACE)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.text(0.01, 0.015, "Standard rules, same arena map. Baselines were run on seed 0 only. Learned seed 0 never "
             "completed an interaction with the rewarding fixture\n(4 approaches lost tracking, 1 failed to plan).",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def map_grid(nav_dir, out):
    worlds = [("home_a", "home_a (development)"), ("heldout_b", "heldout_b (held-out layout)"),
              ("heldout_c", "heldout_c (held-out appearance)"), ("home_a_dim", "home_a_dim (held-out lighting)")]
    tiles = []
    for w, label in worlds:
        img = cv2.imread(str(Path(nav_dir) / w / "map.png"))
        r = json.loads((Path(nav_dir) / w / "result.json").read_text())  # seed 0 of each room
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
