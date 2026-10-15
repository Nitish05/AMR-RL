"""Paired comparison of two plain-arena learning suites (docs/results/r9-followup.md §12).

Evaluation tooling (simulator ground truth allowed). ``A`` is the arm under test and ``B`` the
reference arm from the same commit. For each arm it computes §11's six absolute bounds and the
suite's metric table; for A against B the pre-registered non-inferiority bounds of §12, and the
re-targeting events the runs logged. Writes ``<out>/compare.json`` and ``<out>/compare.md``.

    PYTHONPATH=src python scripts/eval/suite_compare.py <A dir> <B dir> --out <dir>

A suite directory holds one ``S-<experiment>-s<seed>`` run directory per job, each with
``<experiment>/result.json`` as scripts/eval/learning.py writes it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import EXPECTED_AFTER_RESTART, contact_split, first_new_rewarding, repair_fixtures  # noqa: E402

RULE_FIXTURES = ("bloom", "stone", "grump")
COMPLETED = ("outcome", "ambiguous_outcome")


def load_runs(root: Path):
    """(experiment, seed, result) per run directory, with the fixture map repaired as report.py does."""
    runs = []
    for d in sorted(root.glob("S-*")):
        if not d.is_dir():
            continue
        f = next(iter(sorted(d.glob("*/result.json"))), None) or (d / "result.json")
        if not f.exists():
            continue
        result = json.loads(f.read_text())
        if "phases" not in result:
            continue
        result["_fixtures"] = repair_fixtures(result)
        runs.append((result.get("experiment", d.name[2:].split("-s")[0]), int(result.get("seed", 0)), result))
    runs.sort(key=lambda r: (r[1], r[0]))
    return runs


def restart_hits(runs):
    hits = total = 0
    for base, _seed, result in runs:
        if base not in EXPECTED_AFTER_RESTART or len(result["phases"]) < 2:
            continue
        test = result["phases"][-1]
        if test.get("failed"):
            continue
        fx = result["_fixtures"]
        att = result["phases"][0].get("attitudes") or {}
        liked = [(a.get("expected_value") or 0, eid, a) for eid, a in att.items() if a.get("attitude") == "liked"]
        if not liked:
            continue
        _, eid, a = max(liked, key=lambda x: x[0])
        learned = (fx.get(eid), a.get("best_action"))
        first = next((d["chosen"] for d in test.get("decisions", [])
                      if (d.get("chosen") or {}).get("activity") in ("engage", "revisit")), None)
        dec = None if first is None else (fx.get(first.get("entity_id")), first.get("action"))
        total += 1
        hits += dec == learned
    return hits, total


def reversal_adapted(runs):
    adapted, times = 0, []
    for base, _seed, result in runs:
        if base not in ("reversal_early", "reversal_late"):
            continue
        p = result["phases"][0]
        sw = (p.get("switches") or [None])[0]
        if not sw or p.get("failed"):
            times.append(None)
            continue
        hit = first_new_rewarding(p, sw["t"])
        times.append(None if hit is None else hit[0] - sw["t"])
        adapted += hit is not None
    return adapted, times


def arm_metrics(root: Path):
    runs = load_runs(root)
    m = {"runs": len(runs), "phases": 0, "failed": 0, "sim_seconds": 0.0, "starts": {"relocalised": 0, "worst_err_m": 0.0,
         "worst_herr_deg": 0.0}, "red_le1_phases": 0, "contacts": [0, 0], "reloc": [0, 0], "attempts": 0,
         "outcomes": 0, "useful": 0, "red": 0, "ambiguous": 0, "rule_outcomes": {f: 0 for f in RULE_FIXTURES},
         "nav_failures": 0, "lost_frames": 0, "frames": 0, "nudge": {"completed": 0, "reached": 0},
         "nofix": {"approaches": 0, "seconds": 0.0, "by_reason": {}}, "nofix_entities": 0,
         "targeting": {}, "targeting_by_fixture": {}}
    nofix_entities = set()
    for base, seed, result in runs:
        fx = result["_fixtures"]
        for p in result["phases"]:
            m["phases"] += 1
            if p.get("failed"):
                m["failed"] += 1
                continue
            m["sim_seconds"] += float(p.get("sim_seconds") or 0)
            e = p.get("reloc_error_at_start") or {}
            if p.get("relocalized_at") is not None and p.get("frame_valid") is not False:
                m["starts"]["relocalised"] += 1
            if e:
                m["starts"]["worst_err_m"] = max(m["starts"]["worst_err_m"], float(e.get("err_m") or 0))
                m["starts"]["worst_herr_deg"] = max(m["starts"]["worst_herr_deg"], math.degrees(abs(e.get("herr_rad") or 0)))
            events = p.get("world_events") or []
            m["red_le1_phases"] += sum(ev.get("response") == "red" for ev in events) <= 1
            inter = p.get("interactions") or []
            split = contact_split(p.get("contacts", []), inter, p.get("decisions", [])).split("/")
            m["contacts"][0] += int(split[0])
            m["contacts"][1] += int(split[1])
            t = p.get("trajectory") or {}
            m["reloc"][0] += int(t.get("reloc_transitions") or 0)
            m["reloc"][1] += int(t.get("false_reloc_transitions") or 0)
            m["lost_frames"] += int(t.get("lost_frames") or 0)
            m["frames"] += int(t.get("frames") or 0)
            for i in inter:
                if i.get("activity") not in ("engage", "revisit"):
                    continue
                m["attempts"] += 1
                if i["status"] == "navigation_failed":
                    m["nav_failures"] += 1
                if i["status"] == "ambiguous_outcome":
                    m["ambiguous"] += 1
                if i.get("fixture") is None and i["status"] not in COMPLETED:
                    m["nofix"]["approaches"] += 1
                    m["nofix"]["seconds"] += float(i["t"]) - float(i.get("started") or i["t"])
                    key = f"{i['status']}: {i.get('reason') or ''}"[:48]
                    m["nofix"]["by_reason"][key] = m["nofix"]["by_reason"].get(key, 0) + 1
                    nofix_entities.add((base, seed, i.get("entity_id")))
                if i["action"] == "nudge" and i["status"] == "outcome" and i.get("fixture") in RULE_FIXTURES:
                    m["nudge"]["completed"] += 1
                    m["nudge"]["reached"] += any(ev.get("trigger") == "nudge" and ev.get("fixture") == i.get("fixture")
                                                 and i["started"] - 0.5 <= ev["time"] <= i["t"] + 1.0 for ev in events)
            for o in p.get("outcomes") or []:
                m["outcomes"] += 1
                v = o.get("valence") or 0
                m["useful"] += v >= 0.3
                m["red"] += v < 0
                if o.get("fixture") in RULE_FIXTURES:
                    m["rule_outcomes"][o["fixture"]] += 1
            for ev in p.get("targeting") or []:
                kind = ev.get("event") + ("+held" if ev.get("held") else "")
                m["targeting"][kind] = m["targeting"].get(kind, 0) + 1
                cls = fx.get(ev.get("entity_id")) or "none"
                d = m["targeting_by_fixture"].setdefault(cls, {})
                d[kind] = d.get(kind, 0) + 1
    m["nofix_entities"] = len(nofix_entities)
    m["nofix"]["share"] = m["nofix"]["seconds"] / max(1.0, m["sim_seconds"])
    m["lost_frac"] = m["lost_frames"] / max(1, m["frames"])
    m["nudge"]["share"] = m["nudge"]["reached"] / m["nudge"]["completed"] if m["nudge"]["completed"] else None
    m["restart"] = list(restart_hits(runs))
    m["reversal"] = reversal_adapted(runs)
    m["rule_outcomes_total"] = sum(m["rule_outcomes"].values())
    return m


def absolute_bounds(m):
    """§11's six bounds, as (name, value, pass)."""
    adapted, times = m["reversal"]
    shown = [f"{t:.0f}" for t in times if t is not None]
    return [
        ("starts relocalised in every phase", f"{m['starts']['relocalised']}/{m['phases'] - m['failed']} "
         f"(worst {100 * m['starts']['worst_err_m']:.1f} cm / {m['starts']['worst_herr_deg']:.1f}°)",
         m["starts"]["relocalised"] == m["phases"] - m["failed"] and m["failed"] == 0),
        ("restart: first decision = learned option in ≥ 8", f"{m['restart'][0]}/{m['restart'][1]}", m["restart"][0] >= 8),
        ("reversal adapted ≥ 8/10", f"{adapted}/{len(times)}" + (f" ({min(shown, key=float)}–{max(shown, key=float)} s)" if shown else ""),
         adapted >= 8),
        ("≤ 1 red panel per phase in ≥ 43/45", f"{m['red_le1_phases']}/{m['phases']}", m["red_le1_phases"] >= 43),
        ("unintended contact episodes 0", f"{m['contacts'][1]} (intended {m['contacts'][0]})", m["contacts"][1] == 0),
        ("false relocalisations 0", f"{m['reloc'][1]} of {m['reloc'][0]}", m["reloc"][1] == 0),
    ]


def paired_bounds(a, b):
    """§12's bounds for arm A against arm B."""
    def pct(x):
        return f"{100 * x:.1f} %"

    rows = [
        ("non-fixture approaches not completed ≤ 2.0 % of time and ≤ half of B",
         f"A {pct(a['nofix']['share'])} ({a['nofix']['approaches']}, {a['nofix']['seconds']:.0f} s); "
         f"B {pct(b['nofix']['share'])} ({b['nofix']['approaches']}, {b['nofix']['seconds']:.0f} s)",
         a["nofix"]["share"] <= 0.02 and a["nofix"]["share"] <= 0.5 * b["nofix"]["share"]),
        ("completed outcomes ≥ 0.95 × B", f"A {a['outcomes']}; B {b['outcomes']}", a["outcomes"] >= 0.95 * b["outcomes"]),
        ("useful outcomes ≥ 0.95 × B", f"A {a['useful']}; B {b['useful']}", a["useful"] >= 0.95 * b["useful"]),
        ("outcomes on bloom, stone, grump ≥ 0.95 × B",
         f"A {a['rule_outcomes_total']} {a['rule_outcomes']}; B {b['rule_outcomes_total']} {b['rule_outcomes']}",
         a["rule_outcomes_total"] >= 0.95 * b["rule_outcomes_total"]),
        ("navigation failures ≤ B", f"A {a['nav_failures']}; B {b['nav_failures']}", a["nav_failures"] <= b["nav_failures"]),
        ("lost-frame fraction ≤ B + 1 pp", f"A {pct(a['lost_frac'])}; B {pct(b['lost_frac'])}",
         a["lost_frac"] <= b["lost_frac"] + 0.01),
        ("nudges reaching rule fixtures ≥ B's share − 5 pp",
         f"A {a['nudge']['reached']}/{a['nudge']['completed']}; B {b['nudge']['reached']}/{b['nudge']['completed']}",
         (a["nudge"]["share"] or 0.0) >= (b["nudge"]["share"] or 0.0) - 0.05),
    ]
    return rows


def metric_table(a, b):
    def row(name, fa, fb):
        return f"| {name} | {fa} | {fb} |"

    lines = ["| | A (holds on) | B (defaults) |", "|---|---|---|",
             row("runs / phases / failed", f"{a['runs']} / {a['phases']} / {a['failed']}", f"{b['runs']} / {b['phases']} / {b['failed']}"),
             row("simulated seconds", f"{a['sim_seconds']:.0f}", f"{b['sim_seconds']:.0f}"),
             row("attempts / completed outcomes / useful / red", f"{a['attempts']} / {a['outcomes']} / {a['useful']} / {a['red']}",
                 f"{b['attempts']} / {b['outcomes']} / {b['useful']} / {b['red']}"),
             row("outcomes on bloom / stone / grump", " / ".join(str(a["rule_outcomes"][f]) for f in RULE_FIXTURES),
                 " / ".join(str(b["rule_outcomes"][f]) for f in RULE_FIXTURES)),
             row("non-fixture approaches not completed (time, share)",
                 f"{a['nofix']['approaches']} ({a['nofix']['seconds']:.0f} s, {100 * a['nofix']['share']:.1f} %)",
                 f"{b['nofix']['approaches']} ({b['nofix']['seconds']:.0f} s, {100 * b['nofix']['share']:.1f} %)"),
             row("distinct non-fixture entities approached", a["nofix_entities"], b["nofix_entities"]),
             row("navigation failures / lost frames", f"{a['nav_failures']} / {100 * a['lost_frac']:.1f} %",
                 f"{b['nav_failures']} / {100 * b['lost_frac']:.1f} %"),
             row("nudges that reached rule fixtures", f"{a['nudge']['reached']} / {a['nudge']['completed']}",
                 f"{b['nudge']['reached']} / {b['nudge']['completed']}"),
             row("contact episodes unintended (intended)", f"{a['contacts'][1]} ({a['contacts'][0]})", f"{b['contacts'][1]} ({b['contacts'][0]})"),
             row("relocalisations all / false", f"{a['reloc'][0]} / {a['reloc'][1]}", f"{b['reloc'][0]} / {b['reloc'][1]}"),
             row("targeting events", json.dumps(a["targeting"], sort_keys=True), json.dumps(b["targeting"], sort_keys=True)),
             row("targeting events by fixture", json.dumps(a["targeting_by_fixture"], sort_keys=True),
                 json.dumps(b["targeting_by_fixture"], sort_keys=True))]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a, b = arm_metrics(Path(args.a)), arm_metrics(Path(args.b))
    abs_a, abs_b, paired = absolute_bounds(a), absolute_bounds(b), paired_bounds(a, b)
    adopt = all(ok for _, _, ok in abs_a) and all(ok for _, _, ok in paired)
    report = {"A": a, "B": b, "absolute_A": abs_a, "absolute_B": abs_b, "paired": paired, "adopt": adopt}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(report, indent=1, default=float))
    lines = [f"# Re-targeting holds: arm A ({args.a}) against arm B ({args.b})", "",
             f"Verdict: **{'ADOPT' if adopt else 'NOT ADOPTED'}** (all §11 bounds on A and all §12 paired bounds).", "",
             "## §11 absolute bounds", "", "| bound | A | pass | B | pass |", "|---|---|---|---|---|"]
    for (name, va, oka), (_, vb, okb) in zip(abs_a, abs_b):
        lines.append(f"| {name} | {va} | {'yes' if oka else 'no'} | {vb} | {'yes' if okb else 'no'} |")
    lines += ["", "## §12 paired bounds (A against B)", "", "| bound | values | pass |", "|---|---|---|"]
    for name, v, ok in paired:
        lines.append(f"| {name} | {v} | {'yes' if ok else 'no'} |")
    lines += ["", "## Metrics", "", metric_table(a, b), "",
              "Non-fixture approaches by outcome: A " + json.dumps(a["nofix"]["by_reason"], sort_keys=True)
              + "; B " + json.dumps(b["nofix"]["by_reason"], sort_keys=True)]
    (out / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
