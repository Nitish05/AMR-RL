"""Learner testbed: the real learning layer under controlled perception errors.

No physics simulator: see ``amr_rl.sim.learner_bench`` for what is real and what is
synthetic. Runs a grid of experiments x perception-noise levels x policies x seeds
in parallel and writes per-episode rows plus a report with denominators.

    python scripts/eval/learner_bench.py                      # full grid (default 20 seeds)
    python scripts/eval/learner_bench.py --seeds 3 --quick    # smoke test (~1 min)

Experiments
  standard  random rules (1 reliable + 1 unreliable rewarding object, 1 aversive,
            1 mover, the rest inert; one pair of look-alike twins), 900 s.
            Policies: learned, random, nearest, fixed.
  reversal  as standard; at 500 s every rewarding rule moves to an inert object. 1200 s.
  inert     nothing responds to anything, 900 s.
  restart   train 600 s (standard), restart with the same memory file in a world where
            nothing responds; is the first interaction the truly best trained option?
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from amr_rl.sim.learner_bench import (  # noqa: E402
    NOISE_PRESETS,
    Episode,
    EpisodeConfig,
    RobotConfig,
    SyntheticWorld,
    WorldConfig,
)

EXPERIMENTS = ("standard", "reversal", "inert", "restart")
BASELINES = ("random", "nearest", "fixed")


def run_job(job):
    exp, noise_name, policy, seed, quick, before_frames = job
    t0 = time.time()
    noise = NOISE_PRESETS[noise_name]
    robot = RobotConfig(before_frames=before_frames)
    scale = 0.4 if quick else 1.0
    try:
        if exp == "standard":
            cfg = EpisodeConfig(seconds=900 * scale, policy=policy, seed=seed, noise=noise, robot=robot)
            ep = Episode(cfg).run()
            row = ep.summary()
        elif exp == "inert":
            cfg = EpisodeConfig(seconds=900 * scale, policy=policy, seed=seed, noise=noise, robot=robot,
                                world=WorldConfig(world="inert"))
            ep = Episode(cfg).run()
            row = ep.summary()
            half = ep.t0 + cfg.seconds / 2
            row["outcomes_first_half"] = sum(e["status"] == "outcome" and e["t"] < half for e in ep.events)
            row["outcomes_second_half"] = sum(e["status"] == "outcome" and e["t"] >= half for e in ep.events)
        elif exp == "reversal":
            switch = 500 * scale
            cfg = EpisodeConfig(seconds=1200 * scale, policy=policy, seed=seed, noise=noise, robot=robot,
                                world=WorldConfig(switch_at=switch))
            world = SyntheticWorld(cfg.world, np.random.default_rng(seed + 1))
            old = {o.index for o in world.objects if o.rules["signal"][0] == "attach:yellow"}
            ep = Episode(cfg, world=world).run()
            row = ep.summary()
            new = {o.index for o in world.objects if o.rules["signal"][0] == "attach:yellow"}
            sw = ep.switch_time
            post = [e for e in ep.events if sw is not None and e["t"] >= sw]
            first_new = next((e["t"] - sw for e in post
                              if e.get("true_object") in new and e.get("true_outcome") == "attach:yellow"), None)
            row.update({
                "switch_time": sw,
                "rewards_before_switch": sum(e.get("true_outcome") == "attach:yellow" for e in ep.events
                                             if sw is None or e["t"] < sw),
                "old_option_tries_after_switch": sum(e.get("true_object") in old and e["action"] == "signal"
                                                     for e in post),
                "seconds_to_first_new_reward": first_new,
                "rewards_after_switch": sum(e.get("true_outcome") == "attach:yellow" for e in post),
                "learned_before_switch": row["rewards"] > 0 and any(
                    e.get("true_outcome") == "attach:yellow" for e in ep.events if sw is None or e["t"] < sw),
            })
        elif exp == "restart":
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "memory.sqlite")
                cfg = EpisodeConfig(seconds=600 * scale, policy=policy, seed=seed, noise=noise, robot=robot)
                train = Episode(cfg, path).run()
                row = {f"train_{k}": v for k, v in train.summary().items()}
                world = train.world
                valence = cfg.learning.valence
                best_idx, best_action, best_value = world.best_option(valence)
                emap = train.entity_map()
                train.memory.close()
                train_rules = {o.index: dict(o.rules) for o in world.objects}
                for o in world.objects:  # test world: nothing responds (choice from memory only)
                    o.rules = {a: ("none", 0.0) for a in o.rules}
                    o.state, o.confused_as = "attach:none", None
                test = Episode(replace(cfg, seconds=120 * scale), path, world=world, start_time=train.now + 60.0)
                test.rt.pose = train.rt.pose.copy()
                first = None
                end = test.t0 + test.cfg.seconds
                while test.now < end and first is None:
                    choice = test.decide()
                    if choice["activity"] in ("engage", "revisit"):
                        first = choice
                        break
                    if choice["activity"] == "explore":
                        test._explore(choice)
                    elif choice["activity"] == "investigate":
                        test._investigate(choice)
                    elif choice["activity"] == "avoid":
                        test._avoid(choice)
                    else:
                        test._idle(choice)
                chosen_obj = None
                if first is not None:
                    eid = test.memory.resolve(first["entity_id"])
                    chosen_obj = emap.get(eid, (None, 0))[0]
                first_value = None
                if chosen_obj is not None:
                    outcome, p = train_rules[chosen_obj][first["action"]]
                    first_value = p * valence.get(outcome, 0.0)
                row.update({
                    "best_true_option": [int(best_idx), best_action, best_value],
                    "first_choice": None if first is None else [chosen_obj, first["action"]],
                    "first_choice_seconds": None if first is None else test.now - test.t0,
                    "first_choice_is_best": None if first is None else
                    bool(chosen_obj == best_idx and first["action"] == best_action),
                    "first_choice_true_value": first_value,
                })
                test.memory.close()
        else:
            raise ValueError(exp)
        status = "ok"
    except Exception as error:  # keep failures in the denominator
        import traceback

        row, status = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}, "failed"
    return {"experiment": exp, "noise": noise_name, "policy": policy, "seed": seed, "status": status,
            "wall_seconds": round(time.time() - t0, 2), **row}


# ---------------------------------------------------------------- report
def _mean_ci(values):
    v = np.array([x for x in values if x is not None], float)
    if len(v) == 0:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.2f}"
    se = v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():.2f} ± {1.96 * se:.2f}"


def _frac(values):
    v = [x for x in values if x is not None]
    return f"{sum(bool(x) for x in v)}/{len(v)}" if v else "—"


def report(rows, out: Path, args):
    noises = [n for n in NOISE_PRESETS if any(r["noise"] == n for r in rows)]
    lines = ["# Learner testbed results", "",
             f"Seeds per cell: {args.seeds}{' (quick: durations x0.4)' if args.quick else ''}. "
             f"Pre-action frames kept: {args.before_frames}. "
             "Mean ± 95 % CI over seeds. Failed episodes: "
             f"{sum(r['status'] != 'ok' for r in rows)} of {len(rows)}.", "",
             "Perception noise presets (per frame unless noted):", "",
             "| preset | pos σ m | outliers | appearance jitter | bad view | missed | state misread | "
             "outcome label confused (per response) | hallucinated change (per interaction) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n in noises:
        c = NOISE_PRESETS[n]
        lines.append(f"| {n} | {c.pos_sigma} | {c.pos_outlier} | {c.appearance_jitter} | {c.bad_view} | {c.miss} | "
                     f"{c.token_flip} | {c.label_confusion} | {c.spurious_response} |")
    ok = [r for r in rows if r["status"] == "ok"]

    def cell(exp, noise, policy):
        return [r for r in ok if r["experiment"] == exp and r["noise"] == noise and r["policy"] == policy]

    lines += ["", "## Standard world: learned policy vs baselines", "",
              "True valence is scored from what really happened (not from what the robot perceived).", "",
              "| noise | policy | n | true valence / attempt | total true valence | rewards | aversive | "
              "aversive repeats | label accuracy | false 'moved' | real moves seen | top belief is truly best | "
              "entities / objects | impure entities |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in noises:
        for p in ("learned",) + BASELINES:
            c = cell("standard", n, p)
            if not c:
                continue
            lines.append(
                f"| {n} | {p} | {len(c)} | {_mean_ci([r['true_valence_per_attempt'] for r in c])} | "
                f"{_mean_ci([r['true_valence'] for r in c])} | {_mean_ci([r['rewards'] for r in c])} | "
                f"{_mean_ci([r['aversive'] for r in c])} | {_mean_ci([r['aversive_repeats'] for r in c])} | "
                f"{_mean_ci([r['label_accuracy'] for r in c])} | {_mean_ci([r.get('false_moved') for r in c])} | "
                f"{_mean_ci([r.get('moved_recall') for r in c])} | {_frac([r['top_belief_is_best'] for r in c])} | "
                f"{_mean_ci([r['entities'] for r in c])} / {c[0]['true_objects']} | "
                f"{_mean_ci([r['impure_entities'] for r in c])} |")
    lines += ["", "## Rule reversal (learned)", "",
              "| noise | n | learned the old option first | tries of old option after switch | "
              "found new option | seconds to first new reward | rewards after switch |",
              "|---|---|---|---|---|---|---|"]
    for n in noises:
        c = cell("reversal", n, "learned")
        if c:
            lines.append(f"| {n} | {len(c)} | {_frac([r['learned_before_switch'] for r in c])} | "
                         f"{_mean_ci([r['old_option_tries_after_switch'] for r in c])} | "
                         f"{_frac([r['seconds_to_first_new_reward'] is not None for r in c])} | "
                         f"{_mean_ci([r['seconds_to_first_new_reward'] for r in c])} | "
                         f"{_mean_ci([r['rewards_after_switch'] for r in c])} |")
    lines += ["", "## Inert world (learned): does it settle?", "",
              "| noise | n | outcomes, first half | outcomes, second half | idle fraction | entities / objects |",
              "|---|---|---|---|---|---|"]
    for n in noises:
        c = cell("inert", n, "learned")
        if c:
            lines.append(f"| {n} | {len(c)} | {_mean_ci([r['outcomes_first_half'] for r in c])} | "
                         f"{_mean_ci([r['outcomes_second_half'] for r in c])} | "
                         f"{_mean_ci([r['idle_fraction'] for r in c])} | "
                         f"{_mean_ci([r['entities'] for r in c])} / {c[0]['true_objects']} |")
    lines += ["", "## Restart persistence (learned)", "",
              "After a restart in a world where nothing responds, is the first interaction the truly best "
              "option from training?", "",
              "| noise | n | made a choice | first choice = truly best option | training rewards |",
              "|---|---|---|---|---|"]
    for n in noises:
        c = cell("restart", n, "learned")
        if c:
            lines.append(f"| {n} | {len(c)} | {_frac([r['first_choice'] is not None for r in c])} | "
                         f"{_frac([r['first_choice_is_best'] for r in c])} | "
                         f"{_mean_ci([r['train_rewards'] for r in c])} |")
    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        lines += ["", "## Failed episodes", ""] + [f"* {r['experiment']}/{r['noise']}/{r['policy']}/seed {r['seed']}: "
                                                   f"{r['error']}" for r in failed[:20]]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--noise", nargs="+", default=list(NOISE_PRESETS))
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true", help="shorter episodes (smoke test)")
    ap.add_argument("--before-frames", type=int, default=3,
                    help="pre-action frames kept for the outcome classifier (runtime default 3)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else ROOT / "work" / "evidence" / f"learner-bench-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    try:
        from common import provenance

        provenance(out, extra={"args": vars(args)})
    except Exception as error:  # provenance is best effort here (light environment)
        (out / "provenance-error.txt").write_text(f"{type(error).__name__}: {error}\n")
    jobs = []
    for exp in args.experiments:
        policies = ("learned",) + BASELINES if exp == "standard" else ("learned",)
        for noise in args.noise:
            for policy in policies:
                for seed in range(args.seeds):
                    jobs.append((exp, noise, policy, seed, args.quick, args.before_frames))
    print(f"{len(jobs)} episodes on {args.workers} workers -> {out}", flush=True)
    rows = []
    t0 = time.time()
    with open(out / "rows.jsonl", "w") as fh, mp.get_context("spawn").Pool(args.workers) as pool:
        for k, row in enumerate(pool.imap_unordered(run_job, jobs), 1):
            rows.append(row)
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} done ({time.time() - t0:.0f} s)", flush=True)
    print(report(rows, out, args))
    print(f"\nwrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
