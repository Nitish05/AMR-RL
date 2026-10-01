"""Navigation evaluation (separate from learning).

Per world: (1) map from scratch by autonomous frontier exploration using only
onboard RGB (interactions disabled); (2) operator goal requests - supported and
unsupported - converted from world coordinates into the robot's own map frame;
(3) lens-blackout fault during a goal (tracking loss -> revocation -> bounded
recovery -> explicit re-enable); (4) an obstacle placed on a previously free
route. Ground truth is used only for scoring. All attempts are retained.

    PYTHONPATH=src python scripts/eval/navigation.py --worlds home_a heldout_b --map-seconds 240
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump, fresh_dir, provenance, trajectory_metrics  # noqa: E402

from amr_rl.control.supervisor import SupervisorConfig  # noqa: E402
from amr_rl.runtime.robot import RuntimeConfig  # noqa: E402
from amr_rl.sim import evaluator  # noqa: E402
from amr_rl.sim.harness import Session  # noqa: E402
from amr_rl.sim.world import WORLD_DIR, load_world_config  # noqa: E402

ARRIVAL_TOLERANCE = 0.15  # true distance for a successful arrival


def wait_tracking(session, timeout=20.0, log=None):
    """Wait for tracking; after 5 s without it, act as the operator and turn the robot
    slowly in place with manual commands (counted as an intervention) so that it can
    see mapped structure again. The robot itself never resumes motion after bounded
    recovery gave up."""
    end = session.now + timeout
    rt = session.runtime
    turned = False
    while session.now < end:
        if rt.slam.status == "tracking":
            if turned:
                rt.command({"action": "manual", "v": 0.0, "w": 0.0, "generation": rt.supervisor.generation})
                session.control_step()
            return True
        if session.now > end - timeout + 5.0 and rt.supervisor.recovery is None and not rt.supervisor.autonomy_enabled:
            if not turned and log is not None:
                log.append({"t": session.now, "intervention": "manual_relocalisation_turn",
                            "reason": "not tracking after recovery"})
            turned = True
            rt.command({"action": "manual", "v": 0.0, "w": 0.4, "generation": rt.supervisor.generation})
        session.control_step()
    return rt.slam.status == "tracking"


def enable(session, log, reason):
    session.runtime.command({"action": "enable_autonomy", "generation": session.runtime.supervisor.generation})
    session.control_step()
    ack = session.runtime.acks[-1] if session.runtime.acks else {}
    log.append({"t": session.now, "intervention": "enable_autonomy", "reason": reason, "ack": ack})
    return bool(ack.get("accepted"))


def send_goal(session, world_xy, map_xy=None):
    rt = session.runtime
    if map_xy is None:
        map_xy = evaluator.world_to_map(session.origin, world_xy)
    rt.command({"action": "goal", "x": float(map_xy[0]), "y": float(map_xy[1]), "generation": rt.supervisor.generation})
    session.control_step()
    return dict(rt.acks[-1]), map_xy


def run_goal(session, goal, log, timeout=75.0, blackout=None):
    rt = session.runtime
    start_contacts = len(session.contacts)
    t0 = session.now
    if goal.get("map_xy") is not None:  # operator clicked a point on the robot's own map
        goal = dict(goal, xy=[float(v) for v in evaluator.map_to_world(session.origin, goal["map_xy"])])
    record = {"goal": goal["name"], "expect": goal["expect"], "world_xy": goal["xy"], "t_start": t0}
    if not rt.supervisor.autonomy_enabled:
        enable(session, log, "before goal")
    ack, map_xy = send_goal(session, goal["xy"], goal.get("map_xy"))
    record.update({"map_xy": [float(v) for v in map_xy], "ack": {k: v for k, v in ack.items() if not k.startswith("_")}})
    if not ack.get("accepted"):
        record.update({"result": "rejected", "reason": ack.get("reason"), "contacts": [], "revocations": [],
                       "arrived": False})
        return record
    if blackout:
        session.inject_blackout(session.now + blackout["delay"], session.now + blackout["delay"] + blackout["duration"])
    revocations = []
    end = session.now + timeout
    while session.now < end:
        session.control_step()
        act = rt.activity
        if rt.supervisor.recovery is not None:
            record.setdefault("recovery_seen", True)
            continue  # bounded recovery owns the wheels; wait for it to end
        if rt.supervisor.revoked_reason and not rt.supervisor.autonomy_enabled:
            reason = rt.supervisor.revoked_reason
            revocations.append({"t": session.now, "reason": reason})
            if blackout is None or len(revocations) > 3:
                break
            # Operator intervention after a fault: wait for fresh tracking, re-enable, resend.
            if not wait_tracking(session, 30.0, log):
                record["recovery"] = "not_relocalized"
                break
            enable(session, log, f"after {reason}")
            ack, _ = send_goal(session, goal["xy"], goal.get("map_xy"))
            record.setdefault("resent", []).append(ack.get("reason"))
            if not ack.get("accepted"):
                break
            continue
        if act is None or act.done:
            last = rt.interaction_log[-1] if rt.interaction_log else {}
            record["navigation_status"] = last.get("status")
            record["navigation_reason"] = last.get("reason")
            break
    truth = evaluator.true_pose(session.world)
    dist = float(np.hypot(truth[0] - goal["xy"][0], truth[1] - goal["xy"][1]))
    record.update({
        "t_end": session.now, "duration": session.now - t0, "true_final_distance": dist,
        "arrived": dist <= ARRIVAL_TOLERANCE and record.get("navigation_status") == "arrived",
        "revocations": revocations, "contacts": session.contacts[start_contacts:],
        "estimated_final_error": None if rt.pose is None else
        float(np.hypot(*(evaluator.to_map_frame(session.origin, truth)[:2] - rt.pose[:2]))),
        "result": "arrived" if dist <= ARRIVAL_TOLERANCE else "not_arrived",
    })
    return record


def sample_supported_goals(session, seed, count=4):
    import cv2

    rt = session.runtime
    rt.planner.prepare(rt.grid)
    trav = rt.planner._trav.astype(np.uint8)
    n, comp = cv2.connectedComponents(trav, connectivity=8)
    if rt.pose is None or n <= 1:
        return []
    rx, ry = rt.grid.to_cell(np.asarray(rt.pose[:2]))
    window = comp[max(0, ry[0] - 3):ry[0] + 4, max(0, rx[0] - 3):rx[0] + 4]
    labels = window[window > 0]
    if len(labels):
        mine = np.bincount(labels).argmax()
    else:  # robot slightly outside certified space (e.g. after recovery): largest component
        mine = np.bincount(comp[comp > 0]).argmax()
    # Interior certified cells (clearance >= footprint + margin + 0.10 m): what an operator
    # would click. Goals right at the edge of certified space lose clearance when live
    # evidence shifts by a cell (seen in navigation-20260930-021235/home_a).
    interior = (comp == mine) & (rt.planner._clear >= rt.planner.cfg.footprint_radius + rt.planner.cfg.margin + 0.10)
    ys, xs = np.nonzero(interior if interior.any() else comp == mine)
    cells = rt.grid.to_xy(xs, ys)
    rng = np.random.default_rng(seed)
    rng.shuffle(cells)
    chosen = []
    for xy in cells:
        if np.linalg.norm(xy - rt.pose[:2]) < 0.8:
            continue
        if all(np.linalg.norm(xy - c) >= 0.6 for c in chosen):
            chosen.append(xy)
        if len(chosen) >= count:
            break
    return chosen


def fault_test(session, log, seed, fault, *, blackout=None, obstacle=False, min_sep=1.2):
    """Drive to a start point, then to an end point with a fault injected.

    Obstacle placement: on the robot's own planned route (the path its planner
    produces on its map) near the middle of the route, at least 0.6 m from the
    robot's true position. The test is recorded as skipped (with a reason) when
    no valid pair/placement exists; skipped tests stay in the denominator."""
    rt = session.runtime
    points = sample_supported_goals(session, seed, count=6)
    pair = next(((a, b) for i, a in enumerate(points) for b in points[i + 1:]
                 if np.linalg.norm(a - b) >= min_sep), None)
    if pair is None:
        return {"fault": fault, "result": "skipped", "reason": "no_supported_pair_in_current_map"}
    start = run_goal(session, {"name": f"{fault}_from", "expect": "reach", "map_xy": list(pair[0]), "xy": None}, log)
    if not start.get("arrived"):
        return {"fault": fault, "result": "skipped", "reason": "did_not_reach_start", "start": start}
    b_goal = {"name": f"{fault}_to", "expect": "reach", "map_xy": list(pair[1]), "xy": None}
    placed = None
    if obstacle:
        try:
            path = rt.planner.plan(rt.grid, rt.pose[:2], pair[1])
        except Exception as error:  # GoalRejected
            return {"fault": fault, "result": "skipped", "reason": f"no_route:{getattr(error, 'reason', error)}"}
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s_along = np.r_[0, np.cumsum(seg)]
        truth = evaluator.true_pose(session.world)
        for frac in (0.5, 0.6, 0.4, 0.7):
            target = s_along[-1] * frac
            k = min(np.searchsorted(s_along, target), len(path) - 1)
            k0 = max(k - 1, 0)
            u = 0.0 if s_along[k] == s_along[k0] else (target - s_along[k0]) / (s_along[k] - s_along[k0])
            map_pt = path[k0] + u * (path[k] - path[k0])
            world_pt = np.asarray(evaluator.map_to_world(session.origin, map_pt), float)
            end_world = np.asarray(evaluator.map_to_world(session.origin, pair[1]), float)
            if np.linalg.norm(world_pt - truth[:2]) >= 0.6 and np.linalg.norm(world_pt - end_world) >= 0.5:
                placed = world_pt
                break
        if placed is None:
            return {"fault": fault, "result": "skipped", "reason": "route_too_short_for_obstacle"}
        session.world.place_evaluation_obstacle(placed)
        for _ in range(3):
            session.control_step()
    record = run_goal(session, b_goal, log, blackout=blackout)
    record["fault"] = fault
    record["start"] = {k: start.get(k) for k in ("result", "true_final_distance", "duration")}
    if obstacle:
        record["obstacle_world_xy"] = [float(v) for v in placed]
        record["obstacle_contact"] = any("evaluation_obstacle" in c["with"] for c in record.get("contacts", []))
    return record


def evaluate_world(name, out_root, map_seconds, seed):
    """Seed 0 uses the configured start; seed k turns the start heading by k x 72 deg
    (same position, so the start stays valid in every room)."""
    run_dir = out_root / (name if seed == 0 else f"{name}-s{seed}")
    cfg = load_world_config(name)
    start = list(cfg["robot_start"])
    start[2] = float(start[2] + seed * 2 * math.pi / 5)
    config = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="explore_only", seed=seed)
    session = Session(name, run_dir=run_dir, memory_path=run_dir / "throwaway-memory.sqlite", config=config,
                      seed=seed, inspection=False, world_overrides={"robot_start": start})
    log = []
    session.control_step()
    session.control_step()
    enable(session, log, "start mapping")
    t_map = time.time()
    # Mapping phase: re-enable once if a fault revokes authority (counted as an intervention).
    while session.now < map_seconds:
        session.control_step()
        if not session.runtime.supervisor.autonomy_enabled and session.runtime.supervisor.recovery is None:
            log.append({"t": session.now, "event": "mapping_authority_revoked",
                        "reason": session.runtime.supervisor.revoked_reason})
            if session.runtime.slam.status == "tracking" and sum("mapping" in (e.get("event") or "") for e in log) < 4:
                enable(session, log, "continue mapping")
            elif session.runtime.slam.status != "tracking" and not wait_tracking(session, 40, log):
                break
    mapping = {"sim_seconds": session.now, "wall_seconds": time.time() - t_map,
               "trajectory": trajectory_metrics(session.truth),
               "map": evaluator.score_map(session.world, session.runtime.grid, session.origin),
               "contacts": session.contacts[:], "map_counts": session.runtime.grid.counts()}
    session.runtime.save_map(run_dir / "map")
    import cv2

    img, _ = session.runtime.map_image()
    cv2.imwrite(str(run_dir / "map.png"), img[..., ::-1])
    # Goal phase: the operator directs the robot; no autonomous activity selection.
    if session.runtime.slam.status != "tracking":
        wait_tracking(session, 40, log)
    session.runtime.chooser.policy = "operator_only"
    session.runtime._cancel_activity("evaluation goals")
    goals = []
    for goal in cfg["nav_eval"]["goals"]:
        goals.append(run_goal(session, goal, log))
    # Supported goals: points an operator would click inside the robot's own certified
    # map (seeded sample of traversable cells >= 0.8 m apart). Arrival is scored in the
    # true world through the map origin, so map drift counts against arrival.
    supported = []
    for index, map_xy in enumerate(sample_supported_goals(session, seed)):
        goal = {"name": f"supported_{index}", "expect": "reach", "map_xy": list(map_xy), "xy": None}
        supported.append(run_goal(session, goal, log))
    nav = cfg["nav_eval"]
    # Fault tests: start/end points are re-sampled from the CURRENT certified map (the
    # map keeps changing, so earlier supported goals may have lost clearance).
    fault = None
    if nav.get("blackout"):
        fault = fault_test(session, log, seed + 101, "lens_blackout", blackout=nav["blackout"])
    moved = None
    if nav.get("moved_obstacle"):
        moved = fault_test(session, log, seed + 202, "obstacle_placed_on_mapped_route", obstacle=True)
    result = {
        "world": name, "split": cfg.get("split"), "seed": seed, "start_world": start, "mapping": mapping, "goals": goals,
        "supported_goals": supported,
        "blackout_goal": fault, "moved_obstacle_goal": moved, "interventions": log,
        "supervisor_log": session.runtime.supervisor.log[-80:], "all_contacts": session.contacts,
        "final_trajectory": trajectory_metrics(session.truth),
        "timing": "lockstep simulation (physics paused during perception)",
        "guard_status": session.runtime.guard_status,
        "guard": None if session.runtime.guard is None else {
            **session.runtime.guard.stats,
            **evaluator.score_guard(session.world, session.runtime.grid, session.origin,
                                    session.runtime.guard.asserted_log)},
    }
    dump(run_dir / "result.json", result)
    session.save_summary()
    session.close()
    return result


def contact_episodes(contacts, gap=0.5):
    episodes, last = 0, None
    for c in contacts:
        if last is None or c["t"] - last > gap:
            episodes += 1
        last = c["t"]
    return episodes


def summarise(results):
    rows = []
    for r in results:
        g = r["goals"]
        reach = [x for x in g if x["expect"] == "reach"]
        rej = [x for x in g if x["expect"] == "reject"]
        rows.append({
            "world": r["world"], "split": r["split"],
            "mapping_ate_m": r["mapping"]["trajectory"]["ate_rmse_m"],
            "mapping_scale": (r["mapping"]["trajectory"]["similarity"] or {}).get("scale"),
            "free_coverage": r["mapping"]["map"]["free_coverage"],
            "false_free_cells": r["mapping"]["map"]["false_free_cells"],
            "false_free_deep_cells": r["mapping"]["map"]["false_free_deep_cells"],
            "reach_requests": len(reach),
            "reach_accepted": sum(x["ack"].get("accepted", False) for x in reach),
            "reach_arrived": sum(bool(x.get("arrived")) for x in reach),
            "reach_rejected_reasons": [x.get("reason") for x in reach if x["result"] == "rejected"],
            "supported_requests": len(r.get("supported_goals", [])),
            "supported_accepted": sum(x["ack"].get("accepted", False) for x in r.get("supported_goals", [])),
            "supported_arrived": sum(bool(x.get("arrived")) for x in r.get("supported_goals", [])),
            "supported_final_errors_m": [round(x.get("true_final_distance", -1), 3) for x in r.get("supported_goals", [])],
            "unsupported_requests": len(rej),
            "unsupported_rejected": sum(x["result"] == "rejected" for x in rej),
            "contact_frames": len(r["all_contacts"]),
            "contact_episodes": contact_episodes(r["all_contacts"]),
            "blackout": None if r["blackout_goal"] is None else {
                "result": r["blackout_goal"].get("result"), "reason": r["blackout_goal"].get("reason"),
                "revocations": [v["reason"] for v in r["blackout_goal"].get("revocations", [])],
                "arrived": r["blackout_goal"].get("arrived"), "recovery": r["blackout_goal"].get("recovery")},
            "moved_obstacle": None if r["moved_obstacle_goal"] is None else {
                "result": r["moved_obstacle_goal"]["result"], "status": r["moved_obstacle_goal"].get("navigation_status"),
                "reason": r["moved_obstacle_goal"].get("navigation_reason") or r["moved_obstacle_goal"].get("reason"),
                "obstacle_contact": r["moved_obstacle_goal"].get("obstacle_contact")},
            "interventions": sum("intervention" in e for e in r["interventions"]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", nargs="+", default=["home_a", "heldout_b", "heldout_c", "home_a_dim"])
    parser.add_argument("--map-seconds", type=float, default=240.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else fresh_dir("navigation")
    out.mkdir(parents=True, exist_ok=True)
    provenance(out, configs=[WORLD_DIR / f"{w}.yaml" for w in args.worlds], extra={"args": vars(args)})
    # Several processes may write into one --out directory (one world/seed each, to
    # keep memory bounded): merge with results already there.
    partial = out / "partial.json"
    results = json.loads(partial.read_text()) if partial.exists() else []
    for seed in args.seeds:
        for name in args.worlds:
            print(f"== {name} seed {seed}", flush=True)
            results = [r for r in results if not (r.get("world") == name and r.get("seed", 0) == seed)]
            try:
                results.append(evaluate_world(name, out, args.map_seconds, seed))
            except Exception as error:  # retain failed runs in the denominator
                import traceback

                results.append({"world": name, "seed": seed, "failed": True, "error": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc()})
                print(results[-1]["traceback"], flush=True)
            dump(out / "partial.json", results)
    ok = [r for r in results if not r.get("failed")]
    summary = {"worlds": sorted({r["world"] for r in results}), "failed_runs": [r["world"] for r in results if r.get("failed")],
               "rows": summarise(ok)}
    dump(out / "summary.json", summary)
    for row in summary["rows"]:
        print(row, flush=True)


if __name__ == "__main__":
    main()
