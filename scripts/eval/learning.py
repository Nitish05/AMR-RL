"""Learning evaluation (separate from navigation).

All runs use the same pre-built arena map (built by exploration from onboard RGB
in a separate session with its own throwaway memory) and FRESH agent memories
unless a phase explicitly continues a previous memory. Every phase is a new
runtime ("process restart"): autonomy starts disabled and the evaluator enables
it only after the robot has relocalized from fresh images.

    PYTHONPATH=src python scripts/eval/learning.py --map work/evidence/<maps>/arena/map --experiments history_a ...
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump, fresh_dir, provenance, trajectory_metrics  # noqa: E402

from amr_rl.control.supervisor import SupervisorConfig  # noqa: E402
from amr_rl.learning.memory import LearningConfig  # noqa: E402
from amr_rl.robot.spec import PROJECT_ROOT  # noqa: E402
from amr_rl.runtime.robot import RuntimeConfig  # noqa: E402
from amr_rl.sim import evaluator  # noqa: E402
from amr_rl.sim.harness import Session  # noqa: E402

CONSEQUENCES = PROJECT_ROOT / "configs" / "consequences"
VALENCE = LearningConfig().valence

# name -> list of phases; a phase continues the experiment's memory unless memory="fresh"
EXPERIMENTS = {
    "history_a": [
        {"label": "train", "consequences": "standard", "seconds": 480},
        {"label": "test_after_restart", "consequences": "inert", "seconds": 120, "stop_after_interactions": 2},
    ],
    "history_b": [
        {"label": "train", "consequences": "swapped", "seconds": 480},
        {"label": "test_after_restart", "consequences": "inert", "seconds": 120, "stop_after_interactions": 2},
    ],
    "no_memory": [
        {"label": "test_fresh_memory", "consequences": "inert", "seconds": 120, "stop_after_interactions": 2},
    ],
    "reversal_late": [
        {"label": "train", "consequences": "standard", "seconds": 780, "switch": {"at": 360, "to": "swapped"}},
    ],
    "reversal_early": [
        {"label": "train", "consequences": "standard", "seconds": 780,
         "switch": {"after_useful": 2, "to": "swapped"}},
    ],
    "inert": [
        {"label": "train", "consequences": "inert", "seconds": 480},
    ],
    "noisy": [
        {"label": "train", "consequences": "noisy", "seconds": 480},
    ],
    "baseline_random": [{"label": "train", "consequences": "standard", "seconds": 480, "policy": "random"}],
    "baseline_nearest": [{"label": "train", "consequences": "standard", "seconds": 480, "policy": "nearest"}],
    "baseline_fixed": [{"label": "train", "consequences": "standard", "seconds": 480, "policy": "fixed"}],
    "learned_standard": [{"label": "train", "consequences": "standard", "seconds": 480}],
}


def load_consequences(name):
    return yaml.safe_load((CONSEQUENCES / f"{name}.yaml").read_text()) or {}


def fixture_for(session, map_xy):
    """Scoring only: which true fixture was the target (nearest true centre)."""
    if map_xy is None:
        return None
    world_xy = evaluator.map_to_world(session.origin, map_xy)
    best = None
    for name in session.world.fixtures:
        d = float(np.linalg.norm(evaluator.fixture_true_xy(session.world, name) - world_xy))
        if best is None or d < best[1]:
            best = (name, d)
    return None if best is None or best[1] > 0.5 else best[0]


def run_phase(phase, *, map_dir, memory_path, run_dir, seed, world="arena"):
    policy = phase.get("policy", "learned")
    config = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy=policy, seed=seed,
                           initial_survey=False)
    consequences = load_consequences(phase["consequences"])
    session = Session(world, run_dir=run_dir, memory_path=memory_path, config=config, consequences=consequences,
                      seed=seed, inspection=False, map_dir=map_dir)
    rt = session.runtime
    record = {"phase": phase, "policy": policy, "memory_sessions_before": rt.memory.counts()["sessions"] - 1,
              "authority_at_start": rt.supervisor.snapshot(), "enable_attempts": [], "switches": []}
    # Relocalize from fresh onboard images before any authority is granted.
    t_reloc = None
    while session.now < 30.0:
        session.control_step()
        if rt.slam.status == "tracking":
            t_reloc = session.now
            break
    record["relocalized_at"] = t_reloc
    if t_reloc is None:
        record["failed"] = "not_relocalized"
        session.save_summary()
        session.close()
        return record
    # Knowledge survived restart, authority did not: explicit enable now.
    rt.command({"action": "enable_autonomy", "generation": rt.supervisor.generation})
    session.control_step()
    record["enable_attempts"].append({"t": session.now, "ack": {k: v for k, v in rt.acks[-1].items()
                                                                if not k.startswith("_")}})
    end = session.now + phase["seconds"]
    switch = phase.get("switch")
    switched = False
    last_seen = 0
    while session.now < end:
        session.control_step()
        if not rt.supervisor.autonomy_enabled and rt.supervisor.recovery is None:
            if rt.slam.status == "tracking" and len(record["enable_attempts"]) < 12:
                rt.command({"action": "enable_autonomy", "generation": rt.supervisor.generation})
                session.control_step()
                record["enable_attempts"].append({"t": session.now, "after": rt.supervisor.log[-2:] if
                                                  rt.supervisor.log else None,
                                                  "ack": {k: v for k, v in rt.acks[-1].items()
                                                          if not k.startswith("_")}})
        outcomes = [e for e in rt.interaction_log if e.get("status") == "outcome"]
        if switch and not switched:
            useful = sum(VALENCE.get(e.get("observed"), 0) >= 0.3 for e in outcomes)
            if ("at" in switch and session.now >= switch["at"]) or (
                    "after_useful" in switch and useful >= switch["after_useful"]):
                session.world.consequences = load_consequences(switch["to"])
                switched = True
                record["switches"].append({"t": session.now, "to": switch["to"], "outcomes_before": len(outcomes)})
        if phase.get("stop_after_interactions"):
            engaged = [e for e in rt.interaction_log if e.get("activity") in ("engage", "revisit")
                       and e.get("status") in ("outcome", "ambiguous_outcome")]
            if len(engaged) >= phase["stop_after_interactions"]:
                break
        if len(rt.interaction_log) != last_seen:
            last_seen = len(rt.interaction_log)
            e = rt.interaction_log[-1]
            print(f"   t={session.now:6.1f} {e.get('activity')} {e.get('action', '')} "
                  f"{fixture_for(session, e.get('target_xy'))} -> {e.get('status')} {e.get('observed', '')} "
                  f"need={rt.motivation.need:.2f}", flush=True)
    interactions = []
    for e in rt.interaction_log:
        if e.get("activity") in ("engage", "revisit"):
            interactions.append({**{k: e.get(k) for k in ("t", "started", "activity", "action", "status", "observed",
                                                          "reason", "context", "entity_id")},
                                 "fixture": fixture_for(session, e.get("target_xy")),
                                 "valence": VALENCE.get(e.get("observed")) if e.get("status") == "outcome" else None,
                                 "decision": e.get("decision")})
    activity_time = {}
    for row in session.truth:
        activity_time[row["activity"] or "none"] = activity_time.get(row["activity"] or "none", 0) + 1
    record.update({
        "sim_seconds": session.now, "interactions": interactions,
        "outcomes": [i for i in interactions if i["status"] == "outcome"],
        "decisions": rt.decision_log, "world_events": [e.__dict__ for e in session.world.events],
        "activity_frames": activity_time, "contacts": session.contacts,
        "trajectory": trajectory_metrics(session.truth), "memory_counts": rt.memory.counts(),
        "attitudes": {e["entity_id"]: {**rt.memory.attitude(e["entity_id"], session.now),
                                       "fixture": fixture_for(session, [e["x"], e["y"]] if e.get("x") is not None
                                                              else None)}
                      for e in rt.memory.entities()},
        "identity_merges": rt.tracker.merge_log,
        "semantic": rt.semantic.stats if rt.semantic else None,
        "supervisor_log": rt.supervisor.log[-60:],
    })
    session.save_summary()
    session.close()
    return record


def score(record):
    outs = record.get("outcomes", [])
    engaged = record.get("interactions", [])
    first = [(i["fixture"], i["action"]) for i in engaged if i["status"] in ("outcome", "ambiguous_outcome")]
    return {
        "interactions_attempted": len(engaged),
        "outcomes_learned": len(outs),
        "total_valence": float(sum(i["valence"] or 0 for i in outs)),
        "aversive_outcomes": sum((i["valence"] or 0) < 0 for i in outs),
        "useful_outcomes": sum((i["valence"] or 0) >= 0.3 for i in outs),
        "navigation_failures": sum(i["status"] == "navigation_failed" for i in engaged),
        "interrupted": sum(i["status"] == "cancelled" for i in engaged),
        "ambiguous": sum(i["status"] == "ambiguous_outcome" for i in engaged),
        "first_interactions": first[:3],
        "by_fixture_action": _count([(i["fixture"], i["action"], i["observed"]) for i in outs]),
        "idle_fraction": record.get("activity_frames", {}).get("idle", 0) / max(1, sum(
            record.get("activity_frames", {}).values())),
        "contact_episodes": len(record.get("contacts", [])),
        "enable_interventions": len(record.get("enable_attempts", [])) - 1,
    }


def _count(items):
    out = {}
    for item in items:
        key = "/".join(str(x) for x in item)
        out[key] = out.get(key, 0) + 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, help="saved arena map directory (from build_map.py)")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else fresh_dir("learning")
    out.mkdir(parents=True, exist_ok=True)
    provenance(out, configs=[PROJECT_ROOT / "configs/worlds/arena.yaml", *sorted(CONSEQUENCES.glob("*.yaml"))],
               extra={"args": vars(args), "map_meta": json.loads((Path(args.map) / "map.json").read_text())
                      ["map_version"]})
    for name in args.experiments:
        exp_dir = out / name
        exp_dir.mkdir(exist_ok=True)
        memory = exp_dir / "memory.sqlite"
        results = {"experiment": name, "phases": []}
        t0 = time.time()
        print(f"== {name}", flush=True)
        for index, phase in enumerate(EXPERIMENTS[name]):
            mem = memory
            if phase.get("memory") == "fresh":
                mem = exp_dir / f"memory-fresh-{index}.sqlite"
            try:
                record = run_phase(phase, map_dir=args.map, memory_path=mem, run_dir=exp_dir / f"phase{index}",
                                   seed=args.seed + index)
                record["score"] = score(record)
            except Exception as error:  # retained in the denominator
                import traceback

                record = {"phase": phase, "failed": f"{type(error).__name__}: {error}",
                          "traceback": traceback.format_exc()}
                print(record["traceback"], flush=True)
            results["phases"].append(record)
            dump(exp_dir / "result.json", results)
        results["wall_seconds"] = time.time() - t0
        dump(exp_dir / "result.json", results)
        for p in results["phases"]:
            print(name, p["phase"]["label"], json.dumps(p.get("score") or p.get("failed")), flush=True)
    # keep a copy of the map identity used
    shutil.copy(Path(args.map) / "map.json", out / "map.json")


if __name__ == "__main__":
    main()
