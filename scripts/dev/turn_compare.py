"""Pool turn_bench.py reports over worlds: the pre-registered turn-drift metrics per
variant and replay mode (evaluation only).

Bins use the nearest surface (object OR wall, ``sbin``) when the reports carry it,
else the capture's object bins.

    python scripts/dev/turn_compare.py work/evidence/turnbench-r8-*/r1-*.json
"""

import json
import sys

import numpy as np


def pct(v, q):
    v = [x for x in v if x is not None]
    return float(np.percentile(v, q)) if v else float("nan")


def pooled(reports):
    variants = []
    for rep in reports:
        variants += [v for v in rep["variants"] if v not in variants]
    rows = []
    for v in variants:
        res = [r for rep in reports if v in rep["variants"] for r in rep["variants"][v]["results"]]
        for mode in ("onmap", "fresh"):
            rs = [r for r in res if r["replay"] == mode]
            if not rs:
                continue
            b = [r.get("sbin", r["bin"]) for r in rs]
            near = [r for r, k in zip(rs, b) if k.startswith("0.0")]
            far = [r for r, k in zip(rs, b) if k.startswith("0.7")]
            frames = sum(r["frames"] for r in rs)
            trk = sum(r["tracking"] for r in rs)
            rows.append({
                "v": v, "mode": mode, "n": len(rs),
                "e360_p90": pct([r["e360"] for r in rs], 90), "e360_med": pct([r["e360"] for r in rs], 50),
                "near_p90": pct([r["e360"] for r in near], 90), "far_med": pct([r["e360"] for r in far], 50),
                "e720_p90": pct([r["e720"] for r in rs], 90),
                "back_end_p90": pct([r["e_end"] for r in rs if r["pattern"] == "back"], 90),
                "emax_p90": pct([r["emax"] for r in rs], 90),
                "lost": sum(r["lost"] for r in rs) / max(1, frames),
                "pred": sum(r["predicted"] for r in rs) / max(1, frames),
                "loss_ev": sum(r["loss_events"] for r in rs),
                "confw": sum(r["conf_wrong"] for r in rs) / max(1, trk),
                "jumps": sum(r.get("jump_frames", 0) for r in rs) / max(1, trk),
                "flag_ok": sum(r.get("flagged_ok", 0) for r in rs) / max(1, trk),
                "slide_p90": pct([r["slide"] for r in rs], 90),
                "ms_p95": pct([r["ms_p95"] for r in rs], 95),
                "closures": sum(r["closures"] for r in rs),
            })
    return rows


def main():
    rows = pooled([json.load(open(f)) for f in sys.argv[1:]])
    hdr = ["v", "mode", "n", "e360_p90", "e360_med", "near_p90", "far_med", "e720_p90", "back_end_p90", "emax_p90",
           "lost", "pred", "loss_ev", "confw", "jumps", "flag_ok", "slide_p90", "ms_p95", "closures"]
    print(" ".join(f"{h:>11s}" for h in hdr))
    for r in rows:
        print(" ".join(f"{r[h]:>11.3f}" if isinstance(r[h], float) else f"{str(r[h]):>11s}" for h in hdr))


if __name__ == "__main__":
    main()
