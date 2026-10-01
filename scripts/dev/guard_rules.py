"""Compare near-field guard confirmation rules on recorded probes (evaluation only).

Input: guard_debug.json files written by turn_then_box_probe.py --guard-observe
(every probe's obstacle-base points, labelled against ground truth with a 10 cm
margin). For each rule: cells that would have been asserted on open floor (false),
and how far the robot was from the box (estimated range to the nearest true base
point) when the box was first asserted.

    python scripts/dev/guard_rules.py <dir with t*/guard_debug.json> ...
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

RES = 0.05


def cells_of(xy):
    xy = np.asarray(xy, float).reshape(-1, 2)
    return {(int(np.floor(x / RES)), int(np.floor(y / RES))) for x, y in xy}


def grow(cells):
    return {(x + dx, y + dy) for x, y in cells for dx in (-1, 0, 1) for dy in (-1, 0, 1)}


def clusters(xy, link=0.06):
    xy = np.asarray(xy, float).reshape(-1, 2)
    n = len(xy)
    label = -np.ones(n, int)
    k = 0
    for i in range(n):
        if label[i] >= 0:
            continue
        stack, label[i] = [i], k
        while stack:
            j = stack.pop()
            near = np.flatnonzero((np.linalg.norm(xy - xy[j], axis=1) <= link) & (label < 0))
            label[near] = k
            stack.extend(near.tolist())
        k += 1
    return [xy[label == c] for c in range(k)]


def evaluate(probes, near, window, min_cluster, distinct_refs, min_probes):
    false_cells, true_cells, first_true = set(), set(), None
    history = []
    for p in probes:
        xy = np.asarray(p["base_xy"], float).reshape(-1, 2)
        truth = np.asarray(p["true"], bool)
        here = np.asarray(p["pose"][:2])
        keep = np.linalg.norm(xy - here, axis=1) <= near if len(xy) else np.zeros(0, bool)
        xy, truth = xy[keep], truth[keep]
        kept = np.zeros(len(xy), bool)
        for c in clusters(xy):
            if len(c) >= min_cluster:
                for pt in c:
                    kept |= np.all(np.isclose(xy, pt), axis=1)
        cells = cells_of(xy[kept])
        history = [h for h in history if p["t"] - h[0] <= window]
        support = {}
        for _, ref, old in history:
            if distinct_refs and ref == p["ref"]:
                continue
            for c in cells & grow(old):
                support[c] = support.get(c, 0) + 1
        confirmed = {c for c, n in support.items() if n >= min_probes - 1}
        history.append((p["t"], p["ref"], cells))
        tcells = cells_of(xy[truth & kept])
        for c in confirmed:
            if c in grow(tcells):
                true_cells.add(c)
                if p.get("after_box") and first_true is None:
                    d = np.linalg.norm(xy[truth & kept] - here, axis=1)
                    first_true = round(float(d.min()), 2) if len(d) else None
            else:
                false_cells.add(c)
    return len(false_cells), len(true_cells), first_true


files = [f for d in sys.argv[1:] for f in sorted(Path(d).glob("*/guard_debug.json"))]
runs = {f.parent.name: json.loads(f.read_text()) for f in files}
print("scenarios:", list(runs))
for name, probes in runs.items():
    pts = sum(len(p["base_xy"]) for p in probes)
    tru = sum(sum(p["true"]) for p in probes)
    print(f"  {name}: probes {len(probes)}, base points {pts}, on true geometry {tru}")
print("rule: near window min_cluster distinct_refs min_probes -> per scenario (false cells, true cells, "
      "range to box at first true assertion)")
for near, window, mc, dr, mp in itertools.product((1.0,), (1.5, 3.0), (1, 3, 5, 8), (False, True), (2, 3)):
    res = {n: evaluate(p, near, window, mc, dr, mp) for n, p in runs.items()}
    tot_false = sum(r[0] for r in res.values())
    print(f"near {near} win {window} cluster>={mc} distinct_refs {dr} probes>={mp}: false {tot_false:3d} | "
          + "  ".join(f"{n}:{r}" for n, r in res.items()))
