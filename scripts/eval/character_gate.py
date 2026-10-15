"""Character admission gate: are the temperament traits identifiable from behaviour?

Evaluation tooling (simulator ground truth allowed). A trait is admitted only if setting it
changes what an observer can measure, reliably across seeds, and the set values can be
recovered from the measurements (docs/results/character-gate-prereg.md, pre-registered
before the batch ran):

* identifiability: ICC(1) >= 0.4 of the trait's primary signature across its levels
  (0.2 / 0.5 / 0.8, other traits at 0.5; three seeds each in the first pre-registration, five
  in the second of 2026-10-15, which also changed the primaries, see PRIMARY);
* recovery: leave-one-run-out ridge regression from all signatures to the four trait
  values, Pearson r >= 0.7 between predicted and set values per trait;
* no harm: no level's median unintended-contact episodes above the default's + 1, no
  level's lost-frame fraction above the default's + 2 pp.

    plan   PYTHONPATH=src python scripts/eval/character_gate.py plan --map <map> --out <dir>
    score  PYTHONPATH=src python scripts/eval/character_gate.py score <dir>

``plan`` writes ``<dir>/jobs.txt`` for scripts/eval/batch.py: one learning.py run (the
experiment's training phase is scored) per trait x level x seed, plus the default character
per seed, all with ``--runtime character_active=true`` so the whole code path is exercised
(the defaults are the identity). The learning-style traits get their own batches and worlds
(``--traits surprise_weighting --experiment style_reversal`` etc.). ``score`` judges the
traits a batch varied and writes ``gate.json`` and ``gate.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import contact_split  # noqa: E402

from amr_rl.behavior.character import TRAITS as TRAIT_SPECS  # noqa: E402

# Second pre-registration (2026-10-15): three temperament traits remain (persistence retired).
TEMPERAMENT = ("approach_tempo", "caution", "novelty_bias")
INERT_AFTER_SWITCH = "bloom"  # style_reversal: the fixture whose reward the switch removes
TRAITS = tuple(s.key for s in TRAIT_SPECS)
DEFAULTS = {s.key: s.default for s in TRAIT_SPECS}
LEVELS = (0.2, 0.8)
# The signature pre-registered as each trait's own (ICC); every signature enters recovery.
# A batch may override one with ``score --primary trait=signature``. These are the second
# pre-registration's primaries (2026-10-15); the first gate's were speed_mps, chosen_uncertainty,
# info_frac, repeat_frac, post_switch_aversive and frac_stone_late (its batches re-score with
# ``--primary`` overrides).
PRIMARY = {"approach_tempo": "drive_speed_mps", "caution": "engage_frac", "novelty_bias": "info_frac",
           "persistence": "repeat_frac", "surprise_weighting": "post_switch_inert_attempts",
           "curiosity_type": "frac_stone_late"}
FIXTURES = ("bloom", "grump", "stone", "roller")
HARM = ("unintended_contacts", "lost_frac")
ICC_MIN, RECOVERY_MIN, CONTACT_SLACK, LOST_SLACK = 0.4, 0.7, 1, 0.02
NAME = re.compile(r"^T-(?P<trait>[a-z_]+)-(?P<level>[0-9.]+)-s(?P<seed>\d+)$|^T-default-s(?P<dseed>\d+)$")


def plan(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def job(name, trait=None, level=None, seed=0):
        cmd = ["scripts/eval/learning.py", "--map", str(Path(args.map).resolve()), "--world", args.world,
               "--experiments", args.experiment, "--seeds", str(seed), "--odometry", "imu_encoders",
               "--runtime", "character_active=true", "--out", str(out / name)]
        if trait is not None:
            cmd += ["--trait", f"{trait}={level}"]
        lines.append(f"{out / name}.log " + " ".join(shlex.quote(c) for c in cmd))

    for seed in args.seeds:
        job(f"T-default-s{seed}", seed=seed)
        for trait in args.traits:
            if trait not in TRAITS:
                raise SystemExit(f"unknown trait {trait}")
            for level in args.levels:
                job(f"T-{trait}-{level}-s{seed}", trait, level, seed)
    (out / "jobs.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {out / 'jobs.txt'}")


def signatures(rec):
    """Observer-measurable behaviour of one phase (no inner state)."""
    af = rec.get("activity_frames") or {}
    frames = max(1, sum(af.values()))
    active = max(1.0, float(rec.get("sim_seconds") or 0) - float(rec.get("relocalized_at") or 0))
    decisions = rec.get("decisions") or []
    inter = rec.get("interactions") or []
    outs = rec.get("outcomes") or []
    chosen_unc = [d["chosen"].get("uncertainty") for d in decisions
                  if (d.get("chosen") or {}).get("activity") in ("engage", "revisit")
                  and (d.get("chosen") or {}).get("uncertainty") is not None]
    pairs = [(i.get("entity_id"), i.get("action")) for i in inter]
    traj = rec.get("trajectory") or {}
    t0 = float(rec.get("relocalized_at") or 0)
    half = t0 + active / 2.0
    late = [i for i in inter if float(i.get("t") or 0) >= half]
    by_fixture = {f: sum(i.get("fixture") == f for i in late) / len(late) if late else float("nan")
                  for f in FIXTURES}
    bloom_stone = [i for i in inter if i.get("fixture") in ("bloom", "stone")]
    switches = rec.get("switches") or []
    t_switch = float(switches[0]["t"]) if switches else None
    post_aversive = sum((o.get("valence") or 0) < 0 and float(o["t"]) > t_switch for o in outs) if t_switch else float("nan")
    post_useful = [float(o["t"]) - t_switch for o in outs
                   if t_switch is not None and float(o["t"]) > t_switch and (o.get("valence") or 0) >= 0.3]
    # second pre-registration: attempts on the object the switch made inert; driving speed; revisits
    # of an entity whose last outcome was aversive (among interactions with any earlier outcome)
    inert_after = [i for i in inter if t_switch is not None and float(i.get("t") or 0) > t_switch
                   and i.get("fixture") == INERT_AFTER_SWITCH]
    driving = float(traj.get("driving_seconds") or 0.0)
    with_history = aversive_revisits = 0
    for i in inter:
        prior = [o for o in outs if o.get("entity_id") == i.get("entity_id") and float(o["t"]) < float(i.get("t") or 0)]
        if prior:
            with_history += 1
            aversive_revisits += (max(prior, key=lambda o: float(o["t"])).get("valence") or 0) < 0
    return {
        "speed_mps": float(traj.get("true_path_length_m") or 0.0) / active,
        "drive_speed_mps": float(traj.get("driving_path_length_m") or 0.0) / driving if driving > 0 else float("nan"),
        "aversive_revisit_frac": aversive_revisits / with_history if with_history else float("nan"),
        "post_switch_inert_attempts": len(inert_after) if t_switch is not None else float("nan"),
        "post_switch_inert_last_s": (max(float(i["t"]) for i in inert_after) - t_switch) if inert_after else float("nan"),
        "decisions_per_min": 60.0 * len(decisions) / active,
        "bout_s": active / max(1, len(decisions)),
        "info_frac": (af.get("explore", 0) + af.get("investigate", 0)) / frames,
        "engage_frac": (af.get("engage", 0) + af.get("revisit", 0)) / frames,
        "avoid_frac": af.get("avoid", 0) / frames,
        "chosen_uncertainty": float(np.mean(chosen_unc)) if chosen_unc else float("nan"),
        "repeat_frac": (1.0 - len(set(pairs)) / len(pairs)) if pairs else 0.0,
        "aversive_per_min": 60.0 * sum((o.get("valence") or 0) < 0 for o in outs) / active,
        # which objects it keeps returning to (second half of the phase), and the reversal response
        **{f"frac_{f}_late": v for f, v in by_fixture.items()},
        "frac_bloom_vs_stone": (sum(i.get("fixture") == "bloom" for i in bloom_stone) / len(bloom_stone)
                                if bloom_stone else float("nan")),
        "post_switch_aversive": post_aversive,
        "post_switch_first_useful_s": min(post_useful) if post_useful else float("nan"),
        # harm checks (ground truth)
        "unintended_contacts": int(contact_split(rec.get("contacts", []), inter, decisions).split("/")[1]),
        "lost_frac": float(traj.get("lost_frames") or 0) / max(1, float(traj.get("frames") or 1)),
    }


def run_results(root: Path):
    """``(run directory, result.json)`` for every ``T-*`` run. learning.py writes its result one
    level down, ``<out>/<experiment>/result.json`` (the 2026-10-12 temperament batch scored as
    "0 runs" before this was read); a result directly in the run directory is accepted too."""
    found = {}
    for result in sorted(root.glob("T-*/result.json")) + sorted(root.glob("T-*/*/result.json")):
        run = result.parent if result.parent.parent == root else result.parent.parent
        found.setdefault(run, result)
    return sorted(found.items())


def load_runs(root: Path):
    runs = []
    for run, result in run_results(root):
        m = NAME.match(run.name)
        if not m:
            continue
        data = json.loads(result.read_text())
        phase = next((p for p in data["phases"] if p.get("phase", {}).get("label") == "train"), None)
        row = {"name": run.name, "seed": int(m.group("seed") or m.group("dseed")),
               "values": dict(DEFAULTS)}
        if m.group("trait"):
            row["values"][m.group("trait")] = float(m.group("level"))
        if phase is None or phase.get("failed") or phase.get("frame_valid") is False:
            row["failed"] = (phase or {}).get("failed") or "no valid train phase"
        else:
            row["signatures"] = signatures(phase)
            row["idle"] = not phase.get("interactions")  # a valid run that never interacted (reported)
        runs.append(row)
    return runs


def icc1(groups):
    """One-way random-effects ICC(1) over groups of repeated measurements."""
    groups = [[x for x in g if np.isfinite(x)] for g in groups]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None
    k = statistics.mean(len(g) for g in groups)
    grand = np.mean([x for g in groups for x in g])
    msb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups) / (len(groups) - 1)
    msw = sum((x - np.mean(g)) ** 2 for g in groups for x in g) / max(1, sum(len(g) - 1 for g in groups))
    denom = msb + (k - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else None


def recovery(runs, keys, traits, alpha=1.0):
    """Leave-one-run-out ridge regression signatures -> trait values; r per trait."""
    ok = [r for r in runs if "signatures" in r and all(np.isfinite(r["signatures"][k]) for k in keys)]
    if len(ok) < 6:
        return {t: None for t in traits}, len(ok)
    X = np.array([[r["signatures"][k] for k in keys] for r in ok])
    Y = np.array([[r["values"][t] for t in traits] for r in ok])
    pred = np.zeros_like(Y)
    for i in range(len(ok)):
        keep = np.arange(len(ok)) != i
        mu, sd = X[keep].mean(0), X[keep].std(0) + 1e-9
        Xs = (X[keep] - mu) / sd
        A = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        reg = alpha * np.eye(A.shape[1])
        reg[-1, -1] = 0.0
        W = np.linalg.solve(A.T @ A + reg, A.T @ Y[keep])
        pred[i] = np.hstack([(X[i] - mu) / sd, 1.0]) @ W
    out = {}
    for j, t in enumerate(traits):
        if np.std(Y[:, j]) < 1e-9 or np.std(pred[:, j]) < 1e-9:
            out[t] = None
        else:
            out[t] = float(np.corrcoef(Y[:, j], pred[:, j])[0, 1])
    return out, len(ok)


def score(args):
    root = Path(args.dir)
    runs = load_runs(root)
    valid = [r for r in runs if "signatures" in r]
    primary = {**PRIMARY, **dict(item.split("=", 1) for item in args.primary)}
    # the traits this batch varied, and the signatures finite in every valid run
    judged = [t for t in TRAITS if len({r["values"][t] for r in valid}) >= 2]
    keys = [k for k in (valid[0]["signatures"] if valid else {}) if k not in HARM
            and all(np.isfinite(r["signatures"][k]) for r in valid)]
    report = {"runs": len(runs), "failed": [r["name"] for r in runs if "failed" in r], "traits": {},
              "idle": [r["name"] for r in valid if r.get("idle")],
              "judged": judged, "signatures_used": keys, "primary": {t: primary[t] for t in judged}}
    defaults = [r for r in valid if r["values"] == DEFAULTS]
    rec, n_rec = recovery(runs, keys, judged)
    report["recovery_runs"] = n_rec

    def med(rows, key):
        vals = [r["signatures"].get(key, float("nan")) for r in rows if "signatures" in r]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else None

    for trait in judged:
        base = DEFAULTS[trait]
        levels = {base: defaults}
        for level in sorted({r["values"][trait] for r in valid} - {base}):
            levels[level] = [r for r in valid if r["values"][trait] == level
                             and all(v == DEFAULTS[k] for k, v in r["values"].items() if k != trait)]
        sig = primary[trait]
        groups = [[r["signatures"].get(sig, float("nan")) for r in rows] for rows in levels.values()]
        icc = icc1(groups)
        # levels with at least two finite values of the primary: the ICC is over these only, and
        # the report says so when a level has none (the 2026-10-13 variance batch: the careful
        # level never interacted, so its primary was undefined and the column had vanished)
        icc_levels = sum(sum(bool(np.isfinite(x)) for x in g) >= 2 for g in groups)
        cols = keys + ([sig] if sig not in keys else []) + list(HARM)
        by_level = {str(lv): {"n": len(rows), **{k: med(rows, k) for k in cols}}
                    for lv, rows in levels.items()}
        base_c, base_l = med(defaults, "unintended_contacts"), med(defaults, "lost_frac")
        harm = []
        for lv, rows in levels.items():
            if lv == base or not rows:
                continue
            c, lost = med(rows, "unintended_contacts"), med(rows, "lost_frac")
            if base_c is not None and c is not None and c > base_c + CONTACT_SLACK:
                harm.append(f"level {lv}: unintended contacts {c} > default {base_c} + {CONTACT_SLACK}")
            if base_l is not None and lost is not None and lost > base_l + LOST_SLACK:
                harm.append(f"level {lv}: lost fraction {lost:.3f} > default {base_l:.3f} + {LOST_SLACK}")
        r = rec.get(trait)
        passed = icc is not None and icc >= ICC_MIN and r is not None and r >= RECOVERY_MIN and not harm
        report["traits"][trait] = {"primary": sig, "icc1": icc, "icc_levels": icc_levels, "levels": len(groups),
                                   "recovery_r": r, "harm": harm, "columns": cols, "by_level": by_level,
                                   "pass": bool(passed)}
    report["pass"] = bool(report["traits"]) and all(t["pass"] for t in report["traits"].values())
    (root / "gate.json").write_text(json.dumps(report, indent=1, default=float))
    lines = [f"# Character gate: {root.name}", "", f"{len(runs)} runs, {len(report['failed'])} failed "
             f"({', '.join(report['failed']) or 'none'}); {len(report['idle'])} valid without any interaction "
             f"({', '.join(report['idle']) or 'none'}); recovery fitted on {n_rec} runs.", "",
             "| trait | primary signature | ICC(1) | recovery r | harm | pass |", "|---|---|---|---|---|---|"]
    for t, row in report["traits"].items():
        f = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        icc_txt = f(row["icc1"])
        if row["icc_levels"] < row["levels"]:
            icc_txt += f" ({row['icc_levels']} of {row['levels']} levels have the primary)"
        lines.append(f"| {t} | {row['primary']} | {icc_txt} | {f(row['recovery_r'])} | "
                     f"{'; '.join(row['harm']) or 'none'} | {'yes' if row['pass'] else 'no'} |")
    lines += ["", f"Gate: **{'PASS' if report['pass'] else 'FAIL'}** (ICC >= {ICC_MIN}, r >= {RECOVERY_MIN}, "
              f"contacts <= default + {CONTACT_SLACK}, lost <= default + {LOST_SLACK:.0%}).", "",
              "## Medians by level", ""]
    for t, row in report["traits"].items():
        cols = row["columns"]
        lines += [f"### {t}", "", "| level | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for lv, m in row["by_level"].items():
            lines.append(f"| {lv} | {m['n']} | " + " | ".join(
                "n/a" if m[k] is None else f"{m[k]:.3f}" for k in cols) + " |")
        lines.append("")
    (root / "gate.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--map", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--world", default="arena")
    p.add_argument("--experiment", default="history_a")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--levels", nargs="+", type=float, default=list(LEVELS))
    p.add_argument("--traits", nargs="+", default=list(TEMPERAMENT))
    p.set_defaults(func=plan)
    s = sub.add_parser("score")
    s.add_argument("dir")
    s.add_argument("--primary", nargs="*", default=[], help="override trait=signature (pre-registered only)")
    s.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
