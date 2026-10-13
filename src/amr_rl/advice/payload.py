"""What an adviser may be shown: robot-side records only. The trait vector with anchors and
bounds, per-object outcome counts and attitudes, a summary of the session's activity, the
triggers fired and the identity record. Never simulator ground truth, camera frames or the
fixtures' rule tables (tests/test_privilege_boundary.py covers this package). Numbers are sent
neutrally; the payload never says how Pip "feels"."""

from __future__ import annotations

import hashlib
import json
import re

SCHEMA = "amr_rl.advice.payload.v1"
FORBIDDEN_KEYS = ("truth", "ground_truth", "world", "fixture", "rule", "consequence", "frame", "image", "pose_gt")


def build_payload(memory, triggers, decision_log=(), now=0.0):
    traits = {k: {"value": r["value"], "anchor": r["anchor"], "lo": r["lo"], "hi": r["hi"]}
              for k, r in memory.character().items()}
    objects = {}
    with memory.lock:
        counts = memory.db.execute("SELECT entity_id, action, observed, COUNT(*) AS n FROM events "
                                   "GROUP BY entity_id, action, observed").fetchall()
        budgets = {(r["entity_id"], r["action"]): r["probes"] for r in
                   memory.db.execute("SELECT entity_id, action, probes FROM budgets").fetchall()}
    for ent in memory.entities():
        eid = ent["entity_id"]
        att = memory.attitude(eid, now)
        objects[eid] = {"appearance": {k: ent["appearance"].get(k) for k in ("hue", "fill") if k in ent["appearance"]},
                        "label": ent.get("label"), "attitude": att["attitude"], "expected_value": att["expected_value"],
                        "uncertainty": att["uncertainty"], "interactions": att["interactions"],
                        "outcomes": {}, "probes_left": {}, "learning_progress": {}}
    for r in counts:
        obj = objects.setdefault(r["entity_id"], {"outcomes": {}, "probes_left": {}, "learning_progress": {}})
        obj["outcomes"][f"{r['action']}:{r['observed']}"] = r["n"]
    for (eid, action), probes in budgets.items():
        if eid in objects:
            objects[eid]["probes_left"][action] = probes
            objects[eid]["learning_progress"][action] = round(memory.learning_progress(eid, action), 4)
    chosen = {}
    for d in decision_log:
        name = (d.get("chosen") or {}).get("activity") or "none"
        chosen[name] = chosen.get(name, 0) + 1
    payload = {
        "schema": SCHEMA,
        "identity": {"agent_id": memory.agent_id, "sessions": memory.sessions, "boots": len(memory.boots(limit=1000))},
        "traits": traits,
        "objects": objects,
        "activity": {"decisions": len(decision_log), "chosen": chosen, "session_statistics": memory.session_statistics()},
        "triggers": [{k: v for k, v in t.items()} for t in triggers],
    }
    _assert_robot_side(payload)
    return payload


def _assert_robot_side(payload):
    text = json.dumps(payload, sort_keys=True, default=str).lower()
    bad = [k for k in FORBIDDEN_KEYS if f'"{k}' in text]
    if bad:
        raise ValueError(f"payload carries non-robot-side keys: {bad}")


def signature(payload):
    """State signature for caching: trait values, each object's outcome counts and probe
    budgets, and the triggers (without their timestamps). Time-decayed attitudes and the
    activity counters are left out so the same impasse hashes the same."""
    core = {"traits": {k: v.get("value") for k, v in payload.get("traits", {}).items()},
            "objects": {k: {"outcomes": v.get("outcomes", {}), "probes_left": v.get("probes_left", {})}
                        for k, v in payload.get("objects", {}).items()},
            "triggers": [(t.get("trigger"), t.get("entity_id"), t.get("action")) for t in payload.get("triggers", [])]}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _numbers(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 6))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers(v, out)


def grounded(proposal, payload):
    """Faithfulness check: every object id and every number in the rationale must exist in the
    payload; the trait named must exist. Returns (ok, problems)."""
    problems = []
    rationale = str(proposal.get("rationale", ""))
    ids = set(payload.get("objects", {}))
    for token in re.findall(r"\bent-[0-9a-f]+\b|\b[a-z]{1,2}\b(?= \(object\))", rationale):
        if token not in ids:
            problems.append(f"unknown object {token}")
    for match in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", rationale):
        value = round(float(match), 6)
        known = set()
        _numbers(payload, known)
        if value not in known and float(match) not in (0.0, 1.0):
            problems.append(f"number {match} not in the payload")
    trait = proposal.get("trait")
    if trait is not None and trait not in payload.get("traits", {}):
        problems.append(f"unknown trait {trait}")
    for eid in proposal.get("objects", ()):
        if eid not in ids:
            problems.append(f"unknown object {eid}")
    return not problems, problems
