"""Character admission gate: are the temperament traits identifiable from behaviour?

Evaluation tooling (simulator ground truth allowed). A trait is admitted only if setting it
changes what an observer can measure, reliably across seeds, and the set values can be
recovered from the measurements (docs/results/character-gate-prereg.md, pre-registered
before the batch ran):

* identifiability: ICC(1) >= 0.4 of the trait's primary signature across its levels
  (0.2 / 0.5 / 0.8, other traits at 0.5, three seeds each);
* recovery: leave-one-run-out ridge regression from all signatures to the four trait
  values, Pearson r >= 0.7 between predicted and set values per trait;
* no harm: no level's median unintended-contact episodes above the default's + 1, no
  level's lost-frame fraction above the default's + 2 pp.

    plan   PYTHONPATH=src python scripts/eval/character_gate.py plan --map <map> --out <dir>
    score  PYTHONPATH=src python scripts/eval/character_gate.py score <dir>

``plan`` writes ``<dir>/jobs.txt`` for scripts/eval/batch.py: one learning.py run
(history_a, the 480 s training phase is scored) per trait x level x seed, plus the default
character per seed, all with ``--runtime character_active=true`` so the whole code path is
exercised (0.5 everywhere is the identity). ``score`` writes ``gate.json`` and ``gate.md``.
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

TRAITS = ("approach_tempo", "caution", "novelty_bias", "persistence")
LEVELS = (0.2, 0.8)
# The signature pre-registered as each trait's own (ICC); every signature enters recovery.
PRIMARY = {"approach_tempo": "speed_mps", "caution": "chosen_uncertainty", "novelty_bias": "info_frac",
           "persistence": "repeat_frac"}
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
        for trait in TRAITS:
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
    return {
        "speed_mps": float(traj.get("true_path_length_m") or 0.0) / active,
        "decisions_per_min": 60.0 * len(decisions) / active,
        "bout_s": active / max(1, len(decisions)),
        "info_frac": (af.get("explore", 0) + af.get("investigate", 0)) / frames,
        "engage_frac": (af.get("engage", 0) + af.get("revisit", 0)) / frames,
        "avoid_frac": af.get("avoid", 0) / frames,
        "chosen_uncertainty": float(np.mean(chosen_unc)) if chosen_unc else float("nan"),
        "repeat_frac": (1.0 - len(set(pairs)) / len(pairs)) if pairs else 0.0,
        "aversive_per_min": 60.0 * sum((o.get("valence") or 0) < 0 for o in outs) / active,
        # harm checks (ground truth)
        "unintended_contacts": int(contact_split(rec.get("contacts", []), inter, decisions).split("/")[1]),
        "lost_frac": float(traj.get("lost_frames") or 0) / max(1, float(traj.get("frames") or 1)),
    }


def load_runs(root: Path):
    runs = []
    for result in sorted(root.glob("T-*/result.json")):
        m = NAME.match(result.parent.name)
        if not m:
            continue
        data = json.loads(result.read_text())
        phase = next((p for p in data["phases"] if p.get("phase", {}).get("label") == "train"), None)
        row = {"name": result.parent.name, "seed": int(m.group("seed") or m.group("dseed")),
               "values": {t: 0.5 for t in TRAITS}}
        if m.group("trait"):
            row["values"][m.group("trait")] = float(m.group("level"))
        if phase is None or phase.get("failed") or phase.get("frame_valid") is False:
            row["failed"] = (phase or {}).get("failed") or "no valid train phase"
        else:
            row["signatures"] = signatures(phase)
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


def recovery(runs, keys, alpha=1.0):
    """Leave-one-run-out ridge regression signatures -> trait values; r per trait."""
    ok = [r for r in runs if "signatures" in r and all(np.isfinite(r["signatures"][k]) for k in keys)]
    if len(ok) < 6:
        return {t: None for t in TRAITS}, len(ok)
    X = np.array([[r["signatures"][k] for k in keys] for r in ok])
    Y = np.array([[r["values"][t] for t in TRAITS] for r in ok])
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
    for j, t in enumerate(TRAITS):
        if np.std(Y[:, j]) < 1e-9 or np.std(pred[:, j]) < 1e-9:
            out[t] = None
        else:
            out[t] = float(np.corrcoef(Y[:, j], pred[:, j])[0, 1])
    return out, len(ok)


def score(args):
    root = Path(args.dir)
    runs = load_runs(root)
    keys = ["speed_mps", "decisions_per_min", "bout_s", "info_frac", "engage_frac", "avoid_frac",
            "chosen_uncertainty", "repeat_frac", "aversive_per_min"]
    report = {"runs": len(runs), "failed": [r["name"] for r in runs if "failed" in r], "traits": {}}
    defaults = [r for r in runs if "signatures" in r and all(v == 0.5 for v in r["values"].values())]
    rec, n_rec = recovery(runs, keys)
    report["recovery_runs"] = n_rec

    def med(rows, key):
        vals = [r["signatures"][key] for r in rows if "signatures" in r and np.isfinite(r["signatures"][key])]
        return float(np.median(vals)) if vals else None

    for trait in TRAITS:
        levels = {0.5: defaults}
        for level in sorted({r["values"][trait] for r in runs} - {0.5}):
            levels[level] = [r for r in runs if "signatures" in r and r["values"][trait] == level
                             and all(v == 0.5 for k, v in r["values"].items() if k != trait)]
        primary = PRIMARY[trait]
        icc = icc1([[r["signatures"][primary] for r in rows] for rows in levels.values()])
        by_level = {str(lv): {"n": len(rows), **{k: med(rows, k) for k in keys + ["unintended_contacts", "lost_frac"]}}
                    for lv, rows in levels.items()}
        base_c, base_l = med(defaults, "unintended_contacts"), med(defaults, "lost_frac")
        harm = []
        for lv, rows in levels.items():
            if lv == 0.5 or not rows:
                continue
            c, lost = med(rows, "unintended_contacts"), med(rows, "lost_frac")
            if base_c is not None and c is not None and c > base_c + CONTACT_SLACK:
                harm.append(f"level {lv}: unintended contacts {c} > default {base_c} + {CONTACT_SLACK}")
            if base_l is not None and lost is not None and lost > base_l + LOST_SLACK:
                harm.append(f"level {lv}: lost fraction {lost:.3f} > default {base_l:.3f} + {LOST_SLACK}")
        r = rec.get(trait)
        passed = icc is not None and icc >= ICC_MIN and r is not None and r >= RECOVERY_MIN and not harm
        report["traits"][trait] = {"primary": primary, "icc1": icc, "recovery_r": r, "harm": harm,
                                   "by_level": by_level, "pass": bool(passed)}
    report["pass"] = bool(report["traits"]) and all(t["pass"] for t in report["traits"].values())
    (root / "gate.json").write_text(json.dumps(report, indent=1, default=float))
    lines = [f"# Character gate: {root.name}", "", f"{len(runs)} runs, {len(report['failed'])} failed "
             f"({', '.join(report['failed']) or 'none'}); recovery fitted on {n_rec} runs.", "",
             "| trait | primary signature | ICC(1) | recovery r | harm | pass |", "|---|---|---|---|---|---|"]
    for t, row in report["traits"].items():
        f = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        lines.append(f"| {t} | {row['primary']} | {f(row['icc1'])} | {f(row['recovery_r'])} | "
                     f"{'; '.join(row['harm']) or 'none'} | {'yes' if row['pass'] else 'no'} |")
    lines += ["", f"Gate: **{'PASS' if report['pass'] else 'FAIL'}** (ICC >= {ICC_MIN}, r >= {RECOVERY_MIN}, "
              f"contacts <= default + {CONTACT_SLACK}, lost <= default + {LOST_SLACK:.0%}).", "",
              "## Medians by level", ""]
    for t, row in report["traits"].items():
        lines += [f"### {t}", "", "| level | n | " + " | ".join(keys + ["unintended_contacts", "lost_frac"]) + " |",
                  "|---|---|" + "---|" * (len(keys) + 2)]
        for lv, m in row["by_level"].items():
            lines.append(f"| {lv} | {m['n']} | " + " | ".join(
                "n/a" if m[k] is None else f"{m[k]:.3f}" for k in keys + ["unintended_contacts", "lost_frac"]) + " |")
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
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--levels", nargs="+", type=float, default=list(LEVELS))
    p.set_defaults(func=plan)
    s = sub.add_parser("score")
    s.add_argument("dir")
    s.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
