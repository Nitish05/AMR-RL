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
def run_label(r):
    return r["world"] if not r.get("seed") else f'{r["world"]} s{r["seed"]}'


def navigation(nav_dir: Path) -> str:
    results = json.loads((nav_dir / "partial.json").read_text())
    for r in results:
        r["world_label"] = run_label(r)
    lines = [f"Evidence: `{nav_dir.relative_to(nav_dir.parents[2])}`", ""]
    failed = [run_label(r) for r in results if r.get("failed")]
    ok = [r for r in results if not r.get("failed")]
    lines.append(f"Runs: {len(results)}; crashed runs: {len(failed)} {failed or ''}")
    lines += ["", "### Summary by room (all seeds)", "", nav_summary(ok), ""]
    lines += ["", "### Mapping by exploration (onboard RGB only)", ""]
    rows = []
    for r in ok:
        m, t = r["mapping"], r["mapping"]["trajectory"]
        rows.append([r["world_label"], r["split"], m["sim_seconds"], t["true_path_length_m"], t["ate_rmse_m"],
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
        rows.append([r["world_label"],
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
              "click); arrival is scored in the true world (≤ 0.15 m).", ""]
    lines += ["### Fault tests", ""]
    rows = []
    for r in ok:
        for key, label in (("blackout_goal", "lens blackout"), ("moved_obstacle_goal", "obstacle on route")):
            f = r.get(key)
            if f is None:
                rows.append([r["world_label"], label, "not configured", "—", "—", "—", "—"])
                continue
            rows.append([r["world_label"], label, f.get("result"),
                         f.get("navigation_status") or f.get("reason"),
                         ", ".join(v["reason"] for v in f.get("revocations", [])) or "—",
                         f.get("recovery") or ("resent: " + ",".join(f["resent"]) if f.get("resent") else "—"),
                         episodes(f.get("contacts", []))])
    lines.append(table(["world", "fault", "result", "status/reason", "revocations", "recovery / operator",
                        "contact episodes"], rows))
    lines += ["", "### Interventions", ""]
    for r in ok:
        iv = [e for e in r["interventions"] if "intervention" in e]
        lines.append(f"* {r['world_label']}: {len(iv)} operator enables "
                     f"({', '.join(e.get('reason', '') for e in iv)}); "
                     f"mapping revocations: {sum(1 for e in r['interventions'] if e.get('event'))}")
    return "\n".join(lines)


def nav_summary(ok):
    rows = []
    worlds = sorted({r["world"] for r in ok}, key=lambda w: [x["world"] for x in ok].index(w))
    tot = {"sup": 0, "arr": 0, "rej": 0, "imp": 0, "contacts": 0, "faults_run": 0, "faults_ok": 0, "faults": 0}
    for w in worlds:
        rs = [r for r in ok if r["world"] == w]
        sup = [x for r in rs for x in r.get("supported_goals", [])]
        imp = [x for r in rs for x in r["goals"] if x["expect"] == "reject"]
        ates = [r["mapping"]["trajectory"]["ate_rmse_m"] for r in rs]
        maxe = [r["mapping"]["trajectory"]["max_position_error_m"] for r in rs]
        cov = [r["mapping"]["map"]["free_coverage"] for r in rs]
        deep = [r["mapping"]["map"]["false_free_deep_cells"] for r in rs]
        contacts = sum(episodes(r["all_contacts"]) for r in rs)
        faults = [r.get(k) for r in rs for k in ("blackout_goal", "moved_obstacle_goal") if r.get(k) is not None]
        run = [f for f in faults if f.get("result") != "skipped"]
        good = [f for f in run if f.get("result") == "arrived" or (f.get("fault") == "obstacle_placed_on_mapped_route"
                                                                    and not f.get("obstacle_contact"))]
        arr = sum(bool(x.get("arrived")) for x in sup)
        rej = sum(x["result"] == "rejected" for x in imp)
        rows.append([w, rs[0]["split"], len(rs), f"{arr}/{len(sup)}", f"{rej}/{len(imp)}",
                     f"{min(ates) * 100:.1f}–{max(ates) * 100:.1f}", f"{max(maxe) * 100:.1f}",
                     f"{min(cov):.0%}–{max(cov):.0%}", f"{min(deep)}–{max(deep)}", contacts,
                     f"{len(good)}/{len(run)} ({len(faults) - len(run)} skipped)"])
        tot["sup"] += len(sup)
        tot["arr"] += arr
        tot["rej"] += rej
        tot["imp"] += len(imp)
        tot["contacts"] += contacts
    header = ["room", "split", "seeds", "own-map goals arrived", "impossible goals rejected", "ATE cm (range)",
              "worst error cm", "free coverage", "deep false-free", "contact episodes", "fault tests passed / run"]
    out = table(header, rows)
    out += (f"\n\n**All rooms and seeds:** own-map goals arrived {tot['arr']}/{tot['sup']}; impossible goals "
            f"rejected {tot['rej']}/{tot['imp']}; contact episodes {tot['contacts']}.")
    return out


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
        return [name, p["phase"]["label"], "FAILED: " + p["failed"][:60]] + ["—"] * 14
    outs = p.get("outcomes", [])
    return [name, p["phase"]["label"], p.get("policy"), p.get("sim_seconds"), s.get("interactions_attempted"),
            s.get("outcomes_learned"), s.get("useful_outcomes"), s.get("aversive_outcomes"),
            s.get("total_valence"), (s.get("total_valence") or 0) / max(1, len(outs)),
            s.get("ambiguous"), s.get("navigation_failures"), s.get("interrupted"),
            s.get("idle_fraction"), contact_split(p.get("contacts", []), p.get("interactions", [])),
            s.get("enable_interventions"), s.get("operator_turns")]


def contact_split(contacts, interactions=(), gap=0.5):
    """'intended/unintended' contact episodes. Intended = inside the time window of a
    nudge interaction (touching is that action); anything else is unintended."""
    eps, last = [], None
    for c in contacts or []:
        if last is None or c["t"] - last > gap:
            eps.append([])
        eps[-1].append(c)
        last = c["t"]
    windows = [(i.get("started") or i["t"], i["t"] + 0.5) for i in interactions or ()
               if i.get("action") == "nudge"]
    intended = sum(all(any(a <= c["t"] <= b for a, b in windows) for c in e) for e in eps)
    return f"{intended}/{len(eps) - intended}"


EXPECTED_AFTER_RESTART = {"history_a": ("bloom", "signal"), "history_b": ("stone", "signal")}


def entity_fixtures(phase):
    """Scoring only: entity id -> true fixture, from the entity's remembered position
    at the end of the phase (robust when a fixture was moved, e.g. the rolling ball)."""
    return {eid: a.get("fixture") for eid, a in (phase.get("attitudes") or {}).items() if a.get("fixture")}


def repair_fixtures(result):
    """Fill interaction fixture labels that position matching left empty, using the
    entity's own fixture mapping from this or any phase of the same experiment."""
    mapping, votes = {}, {}
    for p in result["phases"]:
        for i in p.get("interactions", []) or []:
            if i.get("fixture") and i.get("entity_id"):
                votes.setdefault(i["entity_id"], {}).setdefault(i["fixture"], 0)
                votes[i["entity_id"]][i["fixture"]] += 1
    for eid, v in votes.items():  # fallback: the fixture its interaction targets matched most often
        mapping[eid] = max(v, key=v.get)
    for p in result["phases"]:
        mapping.update(entity_fixtures(p))
    for p in result["phases"]:
        for key in ("interactions", "outcomes"):
            for i in p.get(key, []) or []:
                if i.get("fixture") is None and i.get("entity_id") in mapping:
                    i["fixture"] = mapping[i["entity_id"]]
        outs = p.get("outcomes", []) or []
        if p.get("score") is not None:
            c = {}
            for o in outs:
                k = f'{o["fixture"]}/{o["action"]}/{o["observed"]}'
                c[k] = c.get(k, 0) + 1
            p["score"]["by_fixture_action"] = c
    return mapping


def learning(ldirs) -> str:
    runs = []  # (base name, seed, result, dir)
    seen = set()
    for ldir in ldirs:
        for dname, result in load_exps(ldir).items():
            base, seed = result.get("experiment", dname.split("-s")[0]), result.get("seed", 0)
            if (base, seed) in seen or result.get("wall_seconds") is None:
                continue
            seen.add((base, seed))
            result["_fixtures"] = repair_fixtures(result)
            runs.append((base, seed, result, str(ldir.relative_to(ldir.parents[2]))))
    runs.sort(key=lambda r: (r[1], r[0]))
    lines = ["Evidence: " + ", ".join(f"`{d.relative_to(d.parents[2])}`" for d in ldirs), ""]
    seeds = sorted({r[1] for r in runs})
    lines.append(f"Seeds: {seeds}. Seed k starts at a different pose (scripts/eval/learning.py ARENA_STARTS) and "
                 "must relocalise against the saved map before any authority is granted.")
    lines += ["", "### Restart persistence and opposite histories", "", restart_table(runs), ""]
    lines += ["### Policy comparison (standard rules)", "", policy_table(runs), ""]
    lines += ["### Consequence changes and settling", "", reversal_table(runs), ""]
    for base, seed, result, _ in runs:
        if base in ("inert", "noisy"):
            lines.append(settling(f"{base} s{seed}", result["phases"][0]))
    lines += ["", "### All phases (denominators)", ""]
    headers = ["experiment", "seed", "phase", "policy", "sim s", "attempts", "outcomes", "useful", "aversive",
               "total valence", "valence/outcome", "ambiguous", "nav failures", "interrupted", "idle frac",
               "contact episodes intended/unintended", "operator re-enables", "operator turns", "identity merges"]
    rows = []
    for base, seed, result, _ in runs:
        for p in result["phases"]:
            row = phase_row(base, p)
            rows.append(row[:1] + [seed] + row[1:] + [len(p.get("identity_merges") or [])])
    lines += [table(headers, rows), ""]
    lines += ["`useful` = valence ≥ 0.3 (yellow panel or moved); `aversive` = red panel. `attempts` counts "
              "engage/revisit interactions including cancelled/failed ones. Contact episodes are *intended* only "
              "when they fall inside a nudge interaction.", ""]
    lines += ["### Outcomes by true fixture / action / observed", ""]
    for base, seed, result, _ in runs:
        for p in result["phases"]:
            sc = p.get("score") or {}
            if sc.get("by_fixture_action"):
                items = ", ".join(f"{k} ×{v}" for k, v in sorted(sc["by_fixture_action"].items()))
                lines.append(f"* **{base} s{seed} / {p['phase']['label']}**: {items}")
    return "\n".join(lines)


def restart_table(runs):
    rows, hits, total = [], 0, 0
    for base, seed, result, _ in runs:
        if base not in ("history_a", "history_b", "no_memory"):
            continue
        test = result["phases"][-1]
        if test.get("failed"):
            rows.append([base, seed, "FAILED: " + str(test["failed"])[:50], "—", "—", "—", "—"])
            continue
        fx = result.get("_fixtures", {})
        first_dec = next((d["chosen"] for d in test.get("decisions", [])
                          if (d.get("chosen") or {}).get("activity") in ("engage", "revisit")), None)
        dec = None if first_dec is None else (fx.get(first_dec.get("entity_id")), first_dec.get("action"))
        outs = [(i["fixture"], i["action"]) for i in test.get("interactions", [])
                if i["status"] in ("outcome", "ambiguous_outcome")]
        # What did THIS robot learn as best in training? (an option it ended up liking)
        learned = None
        if len(result["phases"]) > 1:
            att = result["phases"][0].get("attitudes") or {}
            liked = [(a.get("expected_value") or 0, eid, a) for eid, a in att.items() if a.get("attitude") == "liked"]
            if liked:
                _, eid, a = max(liked, key=lambda x: x[0])
                learned = (fx.get(eid), a.get("best_action"))
        verdict = "—"
        if base in EXPECTED_AFTER_RESTART:
            if learned is None:
                verdict = "n/a (nothing liked after training)"
            else:
                total += 1
                ok = dec == learned
                hits += ok
                verdict = "yes" if ok else "no"
        rows.append([base, seed, "—" if learned is None else f"{learned[0]}/{learned[1]}",
                     "—" if dec is None else f"{dec[0]}/{dec[1]} ({first_dec.get('basis', '')})",
                     "; ".join(f"{f}/{a}" for f, a in outs[:2]) or "—", verdict,
                     len(test.get("operator_turns") or [])])
    out = table(["experiment", "seed", "best option learned in training", "first decision after restart (basis)",
                 "first completed interactions", "decision = learned option", "operator turns"], rows)
    return out + (f"\n\n**Where training produced a liked option, the first decision after restart targeted it "
                  f"in {hits}/{total} runs.** Designed opposite histories: history_a rewards bloom/signal, "
                  "history_b rewards stone/signal.")


def policy_table(runs):
    rows = []
    for base, seed, result, _ in runs:
        if base not in ("history_a", "baseline_random", "baseline_nearest", "baseline_fixed"):
            continue
        p = result["phases"][0]
        sc = p.get("score") or {}
        n = sc.get("outcomes_learned") or 0
        rows.append(["learned" if base == "history_a" else base.replace("baseline_", ""), seed, n,
                     sc.get("useful_outcomes"), sc.get("aversive_outcomes"), sc.get("total_valence"),
                     (sc.get("total_valence") or 0) / max(1, n), sc.get("idle_fraction")])
    return table(["policy", "seed", "outcomes", "useful", "aversive", "total valence", "valence / outcome",
                  "idle frac"], rows)


def reversal_table(runs):
    rows = []
    for base, seed, result, _ in runs:
        if base not in ("reversal_early", "reversal_late"):
            continue
        p = result["phases"][0]
        sw = (p.get("switches") or [None])[0]
        if not sw:
            rows.append([base, seed, "no switch", "—", "—", "—"])
            continue
        after = [o for o in p.get("outcomes", []) if o["t"] >= sw["t"]]
        before = [o for o in p.get("outcomes", []) if o["t"] < sw["t"]]
        old = {(o["fixture"], o["action"]) for o in before if (o["valence"] or 0) >= 0.3}
        first_new = next((o for o in after if (o["valence"] or 0) >= 0.3 and (o["fixture"], o["action"]) not in old),
                         None)
        rows.append([base, seed, f"{sw['t']:.0f}", sw["outcomes_before"],
                     "never" if first_new is None else f"{first_new['t'] - sw['t']:.0f} s "
                     f"({first_new['fixture']}/{first_new['action']})",
                     "yes" if first_new else "no"])
    return table(["experiment", "seed", "switch at s", "outcomes before", "first useful new option", "adapted"],
                 rows)


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
    ap.add_argument("--learning", nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = []
    if a.nav:
        parts += ["## Navigation", "", navigation(Path(a.nav).resolve())]
    if a.learning:
        parts += ["", "## Learning", "", learning([Path(d).resolve() for d in a.learning])]
    Path(a.out).write_text("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
