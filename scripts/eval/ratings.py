"""Human ratings of Pip's character: protocol export and pre-registered scoring.

Evaluation tooling (docs/research/PERSONALITY_RESEARCH_2.md section 4; docs/results/
character-ratings-prereg.md). Nothing here touches the robot: ratings never update traits
(they could only enter through the shadow adviser as a bounded, logged input, and no such path
is wired).

    protocol  PYTHONPATH=src python scripts/eval/ratings.py protocol --out <dir>
    score     PYTHONPATH=src python scripts/eval/ratings.py score <ratings.csv> --out <dir>

``protocol`` writes the rating form as JSON (items grouped by instrument, each with its source;
RoSAS is three separate scales and is never summed into one score), the clip plan for the
between-subjects trait-recovery study (four configurations x seeds, identical opening), and a
CSV template for responses. ``score`` applies the pre-registered exclusions, computes per-scale
means and Cronbach's alpha per sample, the manipulation-check contrasts (Cohen's d with a
bootstrap CI) between each trait's high and low configuration, the Discomfort / eeriness and
Experience checks, the null-control pair, and writes ratings.json / ratings.md with pass/fail
per criterion. Item wordings carry a ``wording_check`` flag until verified against the source.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- the instruments
MANIPULATION = [  # bespoke manipulation checks, one bipolar item per trait (1..7), high pole = high trait
    {"id": "mc_tempo", "trait": "approach_tempo", "low": "hesitant", "high": "bold"},
    {"id": "mc_caution", "trait": "caution", "low": "reckless", "high": "careful"},
    {"id": "mc_novelty", "trait": "novelty_bias", "low": "sticks to the familiar", "high": "drawn to new things"},
    {"id": "mc_persist", "trait": "persistence", "low": "gives up quickly", "high": "keeps trying"},
]
ROSAS = {  # Carpinella et al. 2017, 1..9 "definitely not associated" .. "definitely associated"; three SEPARATE scales
    "warmth": ["happy", "feeling", "social", "organic", "compassionate", "emotional"],
    "competence": ["capable", "responsive", "interactive", "reliable", "competent", "knowledgeable"],
    "discomfort": ["scary", "strange", "awkward", "dangerous", "awful", "aggressive"],
}
EERINESS = [  # Ho & MacDorman 2010 eeriness index, semantic differentials 1..7 (wording to be verified)
    ("reassuring", "eerie"), ("numbing", "freaky"), ("ordinary", "supernatural"), ("boring", "shocking"),
    ("uninspiring", "spine-tingling"), ("predictable", "thrilling"), ("bland", "uncanny"), ("unemotional", "hair-raising"),
]
MIND = {  # Gray, Gray & Wegner 2007 capacities, 1..7 "not at all" .. "very much"; Experience must stay below the midpoint
    "agency": ["self-control", "morality", "memory", "emotion recognition", "planning", "communication", "thought"],
    "experience": ["hunger", "fear", "pain", "pleasure", "rage", "desire", "personality", "consciousness", "pride",
                   "embarrassment", "joy"],
}
WEISMAN = {  # Weisman, Dweck & Markman 2017: body / heart / mind; pre-registered profile high mind, low heart, low body
    "body": ["getting hungry", "feeling pain", "feeling tired", "experiencing fear"],
    "heart": ["feeling love", "feeling embarrassed", "feeling pride", "having free will"],
    "mind": ["remembering things", "making choices", "detecting sounds", "sensing temperatures"],
}
ATTENTION = [{"id": "att_1", "text": "Select 'definitely associated' (9) for this item.", "expected": 9},
             {"id": "att_2", "text": "Select 'not at all' (1) for this item.", "expected": 1}]
CLIP_MEMORY = {"id": "clip_memory", "text": "Which object did the robot approach first in the clip?",
               "scored_against": "the clip's log"}
CONFIGURATIONS = {  # four configurations rendered from identical seeds, camera path and opening 10 s
    "default": {}, "bold": {"approach_tempo": 0.9, "caution": 0.1, "novelty_bias": 0.9, "persistence": 0.9},
    "careful": {"approach_tempo": 0.1, "caution": 0.9, "novelty_bias": 0.1, "persistence": 0.1},
    "null_control": {},  # the default clip shown to a second rater group: must show no effect
}
HIGH_LOW = {"approach_tempo": ("bold", "careful"), "caution": ("careful", "bold"),
            "novelty_bias": ("bold", "careful"), "persistence": ("bold", "careful")}
MIN_SECONDS, RATERS_PER_CELL, EXCLUSION_MARGIN = 180.0, 64, 0.15
D_MIN, SD_SLACK, ICC_MIN = 0.5, 0.3, 0.6
DISCLOSURE = ("The robot in the clip is simulated. Its character is a set of engineered parameters that change "
              "only between sessions by logged rules; it has no emotions and no inner states beyond the logged "
              "quantities shown in its console. Please rate what you observed.")


def protocol(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    form = {
        "schema": "amr_rl.ratings.protocol.v1", "disclosure": DISCLOSURE,
        "manipulation_checks": [{**m, "scale": "1..7 bipolar", "source": "bespoke (Esterwood 2021: no observer scale exists)"}
                                for m in MANIPULATION],
        "rosas": {k: {"items": v, "scale": "1..9", "source": "Carpinella 2017; scored as a separate scale "
                      "(Pithayarungsarit 2025); reliability reported per sample"} for k, v in ROSAS.items()},
        "eeriness": {"items": [{"low": a, "high": b} for a, b in EERINESS], "scale": "1..7",
                     "source": "Ho & MacDorman 2010", "wording_check": "verify item wording against the source before use"},
        "mind_perception": {k: {"items": v, "scale": "1..7", "source": "Gray 2007"} for k, v in MIND.items()},
        "body_heart_mind": {k: {"items": v, "scale": "1..7", "source": "Weisman 2017"} for k, v in WEISMAN.items()},
        "attention_checks": ATTENTION, "clip_memory_check": CLIP_MEMORY,
        "free_text": {"id": "free_text", "text": "In a few words, describe the robot's behaviour in the clip.",
                      "analysis": "emotion-word rate per cell; treated as untrusted text"},
        "exclusions": {"attention_checks": "both must be correct", "clip_memory": "must be correct",
                       "min_seconds": MIN_SECONDS, "unique_ip": True, "free_text": "non-empty and on topic"},
        "sample": {"raters_per_cell": RATERS_PER_CELL, "exclusion_margin": EXCLUSION_MARGIN,
                   "cells": list(CONFIGURATIONS)},
    }
    (out / "form.json").write_text(json.dumps(form, indent=1))
    plan = {"configurations": CONFIGURATIONS, "high_low": HIGH_LOW,
            "clips": [{"configuration": c, "seed": s, "opening_s": 10.0, "camera": "fixed path, identical across cells",
                       "render": f"scripts/eval/learning.py --map <map> --experiments history_a --seeds {s} "
                                 f"--record-frames --runtime character_active=true "
                                 + " ".join(f"--trait {k}={v}" for k, v in CONFIGURATIONS[c].items())}
                      for c in CONFIGURATIONS for s in args.seeds]}
    (out / "clips.json").write_text(json.dumps(plan, indent=1))
    with (out / "responses_template.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater_id", "cell", "clip_seed", "seconds", "ip_hash", "item", "value", "free_text"])
    print(f"protocol -> {out} (form.json, clips.json, responses_template.csv)")


# ---------------------------------------------------------------- scoring
def load_responses(path):
    raters = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            r = raters.setdefault(row["rater_id"], {"cell": row["cell"], "seconds": float(row.get("seconds") or 0),
                                                    "ip": row.get("ip_hash", ""), "items": {}, "free_text": ""})
            if row["item"] == "free_text":
                r["free_text"] = row.get("free_text") or row.get("value") or ""
            else:
                try:
                    r["items"][row["item"]] = float(row["value"])
                except (TypeError, ValueError):
                    r["items"][row["item"]] = None
    return raters


def exclude(raters):
    kept, reasons, seen_ip = {}, {}, set()
    for rid, r in raters.items():
        why = []
        for a in ATTENTION:
            if r["items"].get(a["id"]) != a["expected"]:
                why.append(f"{a['id']} failed")
        if r["items"].get("clip_memory_correct", 1) != 1:
            why.append("clip memory failed")
        if r["seconds"] < MIN_SECONDS:
            why.append("too fast")
        if r["ip"] and r["ip"] in seen_ip:
            why.append("duplicate ip")
        if not r["free_text"].strip():
            why.append("empty free text")
        if why:
            reasons[rid] = why
        else:
            kept[rid] = r
            if r["ip"]:
                seen_ip.add(r["ip"])
    return kept, reasons


def alpha(matrix):
    """Cronbach's alpha of an items x raters matrix (rows = raters)."""
    x = np.asarray(matrix, float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return None
    k = x.shape[1]
    item_var = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    return None if total_var == 0 else float(k / (k - 1) * (1 - item_var / total_var))


def scale_scores(raters, items):
    rows, scores = [], {}
    for rid, r in raters.items():
        vals = [r["items"].get(i) for i in items]
        if all(v is not None for v in vals):
            rows.append(vals)
            scores[rid] = float(np.mean(vals))
    return scores, alpha(rows)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return None if sp == 0 else float((a.mean() - b.mean()) / sp)


def bootstrap_ci(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    ds = []
    for _ in range(n):
        d = cohens_d(rng.choice(a, len(a)), rng.choice(b, len(b)))
        if d is not None:
            ds.append(d)
    return [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))] if ds else None


def score(args):
    out = Path(args.out) if args.out else Path(args.csv).parent
    out.mkdir(parents=True, exist_ok=True)
    raters = load_responses(args.csv)
    kept, excluded = exclude(raters)
    by_cell = {}
    for rid, r in kept.items():
        by_cell.setdefault(r["cell"], {})[rid] = r
    report = {"raters": len(raters), "kept": len(kept), "excluded": excluded, "cells": {c: len(v) for c, v in by_cell.items()},
              "scales": {}, "manipulation": {}, "checks": {}}
    # scales per cell, reliability per sample
    scale_items = {f"rosas_{k}": [f"rosas_{i}" for i in v] for k, v in ROSAS.items()}
    scale_items["eeriness"] = [f"eer_{k}" for k in range(len(EERINESS))]
    scale_items.update({f"mind_{k}": [f"mind_{i}" for i in v] for k, v in MIND.items()})
    scale_items.update({f"whm_{k}": [f"whm_{i}" for i in v] for k, v in WEISMAN.items()})
    scores = {}
    for name, items in scale_items.items():
        report["scales"][name] = {}
        for cell, rs in by_cell.items():
            s, a = scale_scores(rs, items)
            scores[(name, cell)] = list(s.values())
            report["scales"][name][cell] = {"n": len(s), "mean": float(np.mean(list(s.values()))) if s else None,
                                            "sd": float(np.std(list(s.values()), ddof=1)) if len(s) > 1 else None,
                                            "alpha": a}
    # manipulation checks: high against low configuration per trait
    passed = []
    for m in MANIPULATION:
        hi_cell, lo_cell = HIGH_LOW[m["trait"]]
        hi = [r["items"][m["id"]] for r in by_cell.get(hi_cell, {}).values() if r["items"].get(m["id"]) is not None]
        lo = [r["items"][m["id"]] for r in by_cell.get(lo_cell, {}).values() if r["items"].get(m["id"]) is not None]
        d, ci = cohens_d(hi, lo), bootstrap_ci(hi, lo)
        ok = d is not None and d >= D_MIN and ci is not None and ci[0] > 0
        report["manipulation"][m["trait"]] = {"item": m["id"], "high": hi_cell, "low": lo_cell, "n": [len(hi), len(lo)],
                                              "d": d, "ci95": ci, "pass": bool(ok)}
        passed.append(ok)
    # discomfort and eeriness within 0.3 SD of the default in every configuration; Experience below the midpoint
    base_d, base_e = scores.get(("rosas_discomfort", "default"), []), scores.get(("eeriness", "default"), [])
    comfort_ok, comfort = True, {}
    for cell in by_cell:
        if cell in ("default", "null_control"):
            continue
        for name, base in (("rosas_discomfort", base_d), ("eeriness", base_e)):
            vals = scores.get((name, cell), [])
            if len(base) > 1 and vals:
                sd = float(np.std(base, ddof=1)) or 1.0
                delta = (float(np.mean(vals)) - float(np.mean(base))) / sd
                comfort[f"{name}:{cell}"] = delta
                comfort_ok &= delta <= SD_SLACK
    exp = [v for (name, cell), vals in scores.items() if name == "mind_experience" for v in vals]
    experience_ok = bool(exp) and float(np.mean(exp)) < 4.0
    # null control: the same clip, two rater groups, must show no effect on the manipulation items
    null = {}
    for m in MANIPULATION:
        a = [r["items"][m["id"]] for r in by_cell.get("default", {}).values() if r["items"].get(m["id"]) is not None]
        b = [r["items"][m["id"]] for r in by_cell.get("null_control", {}).values() if r["items"].get(m["id"]) is not None]
        ci = bootstrap_ci(a, b)
        null[m["id"]] = {"d": cohens_d(a, b), "ci95": ci, "no_effect": ci is not None and ci[0] <= 0 <= ci[1]}
    null_ok = bool(null) and all(v["no_effect"] for v in null.values()) if any(
        scores.get((n, "null_control")) for n in scale_items) or "null_control" in by_cell else None
    report["checks"] = {"comfort_within_0.3_sd": comfort_ok, "comfort_deltas_sd": comfort,
                        "experience_below_midpoint": experience_ok, "experience_mean": float(np.mean(exp)) if exp else None,
                        "null_control": null, "null_control_no_effect": null_ok,
                        "sample_size_met": all(n >= RATERS_PER_CELL for c, n in report["cells"].items())}
    report["pass"] = bool(passed) and all(passed) and comfort_ok and experience_ok and (null_ok is not False)
    (out / "ratings.json").write_text(json.dumps(report, indent=1, default=float))
    lines = [f"# Ratings: {Path(args.csv).name}", "", f"{len(raters)} raters, {len(kept)} kept, "
             f"{len(excluded)} excluded; cells: {report['cells']}", "",
             "| trait | item | high vs low | n | d | 95% CI | pass |", "|---|---|---|---|---|---|---|"]
    for t, r in report["manipulation"].items():
        d = "n/a" if r["d"] is None else f"{r['d']:.2f}"
        ci = "n/a" if r["ci95"] is None else f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"
        lines.append(f"| {t} | {r['item']} | {r['high']} vs {r['low']} | {r['n']} | {d} | {ci} | "
                     f"{'yes' if r['pass'] else 'no'} |")
    lines += ["", f"Comfort within {SD_SLACK} SD: {comfort_ok}; Experience below midpoint: {experience_ok}; "
              f"null control no effect: {null_ok}; sample size met: {report['checks']['sample_size_met']}.", "",
              f"**{'PASS' if report['pass'] else 'FAIL'}** (pre-registered criteria, docs/results/character-ratings-prereg.md)."]
    (out / "ratings.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("protocol")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.set_defaults(func=protocol)
    s = sub.add_parser("score")
    s.add_argument("csv")
    s.add_argument("--out", default=None)
    s.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
