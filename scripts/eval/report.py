"""Render evaluation evidence (navigation and learning, kept separate) as Markdown tables.

    PYTHONPATH=src python scripts/eval/report.py --nav work/evidence/navigation-… \
        --learning work/evidence/learning-… --out work/evidence/report.md

Every number comes from the result JSON files; failed/skipped runs stay in the
denominators and are listed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VALENCE = {"attach:yellow": 1.0, "moved": 0.6, "none": 0.0, "attach:red": -1.0}


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------- navigation
def navigation(nav_dir: Path) -> str:
    results = json.loads((nav_dir / "partial.json").read_text())
    lines = [f"Evidence: `{nav_dir.relative_to(nav_dir.parents[2])}`", ""]
    failed = [r["world"] for r in results if r.get("failed")]
    ok = [r for r in results if not r.get("failed")]
    lines.append(f"Worlds run: {len(results)}; crashed runs: {len(failed)} {failed or ''}")
    lines += ["", "### Mapping by exploration (onboard RGB only)", ""]
    rows = []
    for r in ok:
        m, t = r["mapping"], r["mapping"]["trajectory"]
        rows.append([r["world"], r["split"], m["sim_seconds"], t["true_path_length_m"], t["ate_rmse_m"],
                     t["max_position_error_m"], (t.get("similarity") or {}).get("scale"),
                     f'{t["tracking_frames"]}/{t["frames"]}', t["lost_frames"], m["map"]["free_coverage"],
                     m["map"]["free_cells"], m["map"]["false_free_cells"], m["map"]["false_free_deep_cells"],
                     episodes(m["contacts"])])
    lines.append(table(["world", "split", "sim s", "path m", "ATE m", "max err m", "scale", "tracking frames",
                        "lost frames", "free coverage", "free cells", "false-free cells", "deep false-free",
                        "contact episodes"], rows))
    lines += ["", "False-free = estimated FREE where the true room is occupied or outside it; "
              "deep = more than 10 cm inside an obstacle/wall. Coverage = estimated-free ∩ true-free "
              "/ true free floor.", ""]
    lines += ["### Goals", ""]
    rows = []
    for r in ok:
        g = r["goals"]
        reach = [x for x in g if x["expect"] == "reach"]
        rej = [x for x in g if x["expect"] == "reject"]
        sup = r.get("supported_goals", [])
        rows.append([r["world"],
                     f'{sum(bool(x.get("arrived")) for x in reach)}/{len(reach)}',
                     ", ".join(sorted({x.get("reason") for x in reach if x["result"] == "rejected"})) or "—",
                     f'{sum(x["result"] == "rejected" for x in rej)}/{len(rej)}',
                     f'{sum(x["ack"].get("accepted", False) for x in sup)}/{len(sup)}',
                     f'{sum(bool(x.get("arrived")) for x in sup)}/{len(sup)}',
                     ", ".join(f'{x.get("true_final_distance", float("nan")):.2f}' for x in sup) or "—",
                     episodes(r["all_contacts"])])
    lines.append(table(["world", "config goals arrived", "config-goal rejection reasons",
                        "unsupported goals rejected", "supported accepted", "supported arrived",
                        "supported final true distance m", "contact episodes (whole run)"], rows))
    lines += ["", "Config goals are fixed world points chosen before the run (some lie in space the robot "
              "never certified; rejecting those is the conservative outcome). Supported goals are "
              "seeded samples of the robot's own certified-traversable map (what an operator would "
              "click); arrival is scored in the true world (≤ 0.25 m).", ""]
    lines += ["### Fault tests", ""]
    rows = []
    for r in ok:
        for key, label in (("blackout_goal", "lens blackout"), ("moved_obstacle_goal", "obstacle on route")):
            f = r.get(key)
            if f is None:
                rows.append([r["world"], label, "not configured", "—", "—", "—", "—"])
                continue
            rows.append([r["world"], label, f.get("result"),
                         f.get("navigation_status") or f.get("reason"),
                         ", ".join(v["reason"] for v in f.get("revocations", [])) or "—",
                         f.get("recovery") or ("resent: " + ",".join(f["resent"]) if f.get("resent") else "—"),
                         episodes(f.get("contacts", []))])
    lines.append(table(["world", "fault", "result", "status/reason", "revocations", "recovery / operator",
                        "contact episodes"], rows))
    lines += ["", "### Interventions", ""]
    for r in ok:
        iv = [e for e in r["interventions"] if "intervention" in e]
        lines.append(f"* {r['world']}: {len(iv)} operator enables "
                     f"({', '.join(e.get('reason', '') for e in iv)}); "
                     f"mapping revocations: {sum(1 for e in r['interventions'] if e.get('event'))}")
    return "\n".join(lines)


def episodes(contacts, gap=0.5):
    n, last = 0, None
    for c in contacts or []:
        if last is None or c["t"] - last > gap:
            n += 1
        last = c["t"]
    return n


# ----------------------------------------------------------------------- learning
def load_exps(ldir: Path):
    out = {}
    for d in sorted(ldir.iterdir()):
        f = d / "result.json"
        if f.exists():
            out[d.name] = json.loads(f.read_text())
    return out


def phase_row(name, p):
    s = p.get("score") or {}
    if p.get("failed"):
        return [name, p["phase"]["label"], "FAILED: " + p["failed"][:60]] + ["—"] * 8
    outs = p.get("outcomes", [])
    return [name, p["phase"]["label"], p.get("policy"), p.get("sim_seconds"), s.get("interactions_attempted"),
            s.get("outcomes_learned"), s.get("useful_outcomes"), s.get("aversive_outcomes"),
            s.get("total_valence"), (s.get("total_valence") or 0) / max(1, len(outs)),
            s.get("ambiguous"), s.get("navigation_failures"), s.get("interrupted"),
            s.get("idle_fraction"), contact_split(p.get("contacts", [])), s.get("enable_interventions")]


def contact_split(contacts, gap=0.5):
    """'intended/unintended' contact episodes: intended = during an interaction's act phase."""
    eps, last = [], None
    for c in contacts or []:
        if last is None or c["t"] - last > gap:
            eps.append([])
        eps[-1].append(c)
        last = c["t"]
    intended = sum(any(":acting" in (c.get("activity") or "") for c in e) for e in eps)
    return f"{intended}/{len(eps) - intended}"


def learning(ldir: Path) -> str:
    exps = load_exps(ldir)
    lines = [f"Evidence: `{ldir.relative_to(ldir.parents[2])}`", ""]
    headers = ["experiment", "phase", "policy", "sim s", "attempts", "outcomes", "useful", "aversive",
               "total valence", "valence/outcome", "ambiguous", "nav failures", "interrupted", "idle frac",
               "contact episodes intended/unintended", "operator re-enables"]
    rows = [phase_row(n, p) for n, e in exps.items() for p in e["phases"]]
    lines += ["### All phases (denominators)", "", table(headers, rows), ""]
    lines += ["`useful` = valence ≥ 0.3 (yellow panel or moved); `aversive` = red panel. "
              "`attempts` counts engage/revisit interactions including cancelled/failed ones.", ""]
    lines += ["### Outcomes by true fixture / action / observed", ""]
    for n, e in exps.items():
        for p in e["phases"]:
            s = p.get("score") or {}
            if s.get("by_fixture_action"):
                items = ", ".join(f"{k} ×{v}" for k, v in sorted(s["by_fixture_action"].items()))
                lines.append(f"* **{n} / {p['phase']['label']}**: {items}")
    lines += ["", "### Restart persistence and opposite histories (first interactions after restart)", ""]
    rows = []
    for n in ("history_a", "history_b", "no_memory"):
        e = exps.get(n)
        if not e:
            continue
        test = e["phases"][-1]
        first = (test.get("score") or {}).get("first_interactions")
        rows.append([n, test["phase"]["label"], test.get("memory_sessions_before"), test.get("relocalized_at"),
                     (test.get("authority_at_start") or {}).get("autonomy_enabled"),
                     "; ".join(f"{f}/{a}" for f, a in (first or [])) or "—",
                     "; ".join(decision_basis(test)[:2]) or "—"])
    lines.append(table(["experiment", "phase", "prior sessions in memory", "relocalized at s",
                        "autonomy at start", "first interactions (true fixture/action)",
                        "first decisions (basis)"], rows))
    lines += ["", "### Policy comparison under the standard rules (same map, same start)", ""]
    rows = []
    for n in ("learned_standard", "history_a", "baseline_random", "baseline_nearest", "baseline_fixed"):
        e = exps.get(n)
        if not e:
            continue
        p = e["phases"][0]
        s = p.get("score") or {}
        rows.append([n, p.get("policy"), s.get("outcomes_learned"), s.get("useful_outcomes"),
                     s.get("aversive_outcomes"), s.get("total_valence"),
                     (s.get("total_valence") or 0) / max(1e-9, (p.get("sim_seconds") or 1)) * 100,
                     s.get("idle_fraction")])
    lines.append(table(["run", "policy", "outcomes", "useful", "aversive", "total valence", "valence / 100 s",
                        "idle frac"], rows))
    lines += ["", "### Consequence changes (reversal) and settling", ""]
    for n in ("reversal_early", "reversal_late"):
        e = exps.get(n)
        if not e:
            continue
        lines.append(reversal(n, e["phases"][0]))
    for n in ("inert", "noisy"):
        e = exps.get(n)
        if not e:
            continue
        lines.append(settling(n, e["phases"][0]))
    return "\n".join(lines)


def decision_basis(phase):
    out = []
    for d in phase.get("decisions", []):
        ch = d.get("chosen") or {}
        if ch.get("activity") in ("engage", "revisit"):
            out.append(f'{ch.get("activity")} {ch.get("action")}: {ch.get("basis", "")}')
    return out


def reversal(name, p):
    sw = (p.get("switches") or [None])[0]
    if not sw:
        return f"* **{name}**: no switch happened (phase {p.get('failed') or 'completed'})."
    t = sw["t"]
    outs = p.get("outcomes", [])
    before = [o for o in outs if o["t"] < t]
    after = [o for o in outs if o["t"] >= t]

    def summary(items):
        c = {}
        for o in items:
            k = f'{o["fixture"]}/{o["action"]}/{o["observed"]}'
            c[k] = c.get(k, 0) + 1
        return ", ".join(f"{k} ×{v}" for k, v in sorted(c.items())) or "none"

    old_useful = {(o["fixture"], o["action"]) for o in before if (o["valence"] or 0) >= 0.3}
    persist = 0
    for o in after:
        if (o["fixture"], o["action"]) in old_useful and (o["valence"] or 0) < 0.3:
            persist += 1
        elif (o["fixture"], o["action"]) not in old_useful:
            break
    first_new = next((o for o in after if (o["valence"] or 0) >= 0.3 and (o["fixture"], o["action"]) not in
                      old_useful), None)
    last_quarter = [o for o in after if o["t"] >= t + 0.5 * (p["sim_seconds"] - t)]
    return (f"* **{name}**: switch to `{sw['to']}` at t = {t:.0f} s after {sw['outcomes_before']} outcomes. "
            f"Before: {summary(before)}. After: {summary(after)}. Failed tries of the formerly useful option "
            f"before first trying something else: {persist}. First useful outcome from a different option: "
            + (f"t = {first_new['t']:.0f} s ({first_new['fixture']}/{first_new['action']}, "
               f"{first_new['t'] - t:.0f} s after the switch)" if first_new else "never")
            + f". Second half after switch: {summary(last_quarter)}.")


def settling(name, p):
    outs = p.get("outcomes", [])
    T = p.get("sim_seconds") or 1
    halves = [[o for o in outs if o["t"] < T / 2], [o for o in outs if o["t"] >= T / 2]]
    s = p.get("score") or {}
    return (f"* **{name}**: outcomes first half {len(halves[0])}, second half {len(halves[1])}; "
            f"useful {s.get('useful_outcomes')}; idle fraction {fmt(s.get('idle_fraction'))}; "
            f"by fixture/action: " + ", ".join(f"{k} ×{v}" for k, v in sorted((s.get('by_fixture_action') or {})
                                                                          .items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav")
    ap.add_argument("--learning")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = []
    if a.nav:
        parts += ["## Navigation", "", navigation(Path(a.nav).resolve())]
    if a.learning:
        parts += ["", "## Learning", "", learning(Path(a.learning).resolve())]
    Path(a.out).write_text("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
