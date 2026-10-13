"""Stage 1 of the advice gate: a logged impasse. Deterministic rules on the experience
memory and the decision log; nothing here reads a simulator or a clock."""

from __future__ import annotations

import json
from dataclasses import dataclass

TRIGGERS = ("frustration_two_sessions", "attitude_flips", "zero_progress_with_probes", "new_object_low_margin",
            "trait_at_bound")


@dataclass
class ImpasseConfig:
    flips_min: int = 3           # sign changes of the valence among an object's last 10 outcomes
    flips_window: int = 10
    zero_progress_outcomes: int = 6   # outcomes on an option with no learning progress while probes remain
    margin_threshold: float = 0.05    # chooser top-two value margin for a never-tried object
    progress_eps: float = 0.01


def _sign_changes(valences):
    signs = [1 if v > 0 else -1 for v in valences if v != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def detect_impasses(memory, decision_log=(), *, renewals_now=0, renewals_prev=0, cfg: ImpasseConfig | None = None):
    """Return the impasses present now, each ``{"trigger", "entity_id", "action", "detail"}``."""
    cfg = cfg or ImpasseConfig()
    found = []
    if renewals_now > 0 and renewals_prev > 0:
        found.append({"trigger": "frustration_two_sessions", "entity_id": None, "action": None,
                      "detail": {"renewals_now": renewals_now, "renewals_prev": renewals_prev}})
    with memory.lock:
        rows = memory.db.execute("SELECT entity_id, action, valence, seq FROM events ORDER BY seq").fetchall()
        budgets = {(r["entity_id"], r["action"]): r["probes"] for r in
                   memory.db.execute("SELECT entity_id, action, probes FROM budgets").fetchall()}
    per_entity, per_option = {}, {}
    for r in rows:
        per_entity.setdefault(r["entity_id"], []).append(r["valence"])
        per_option.setdefault((r["entity_id"], r["action"]), []).append(r["valence"])
    for eid, vals in per_entity.items():
        flips = _sign_changes(vals[-cfg.flips_window:])
        if flips >= cfg.flips_min:
            found.append({"trigger": "attitude_flips", "entity_id": eid, "action": None,
                          "detail": {"sign_changes": flips, "window": min(len(vals), cfg.flips_window)}})
    for (eid, action), vals in per_option.items():
        if len(vals) >= cfg.zero_progress_outcomes and budgets.get((eid, action), 0) > 0 \
                and memory.learning_progress(eid, action) < cfg.progress_eps:
            found.append({"trigger": "zero_progress_with_probes", "entity_id": eid, "action": action,
                          "detail": {"outcomes": len(vals), "probes_left": budgets[(eid, action)],
                                     "learning_progress": memory.learning_progress(eid, action)}})
    for d in reversed(list(decision_log)):
        chosen = d.get("chosen") or {}
        top = d.get("top") or []
        if chosen.get("activity") in ("engage", "revisit") and len(top) >= 2:
            margin = float(top[0].get("value", 0.0)) - float(top[1].get("value", 0.0))
            eid = chosen.get("entity_id")
            if eid is not None and eid not in per_entity and margin < cfg.margin_threshold:
                found.append({"trigger": "new_object_low_margin", "entity_id": eid, "action": chosen.get("action"),
                              "detail": {"margin": margin, "t": d.get("t")}})
            break
    # a consolidation rule wants a trait at a bound it has already reached (a bound that is the
    # trait's engineered default, such as curiosity_type at 0, is resting, not an impasse)
    rows = memory.character()
    cfg_char = getattr(memory, "character_cfg", None)
    for e in memory.character_log(limit=50):
        if e.get("cause") != "session_end" or e.get("session") != memory.sessions:
            continue
        row = rows.get(e["key"])
        detail = e.get("detail")
        detail = json.loads(detail) if isinstance(detail, str) else (detail or {})
        target = detail.get("target")
        if row is None or target is None or cfg_char is None or e["after"] == cfg_char.spec(e["key"]).default:
            continue
        at_hi = e["after"] >= row["hi"] and target >= row["hi"]
        at_lo = e["after"] <= row["lo"] and target <= row["lo"]
        if at_hi or at_lo:
            found.append({"trigger": "trait_at_bound", "entity_id": None, "action": None,
                          "detail": {"trait": e["key"], "value": e["after"], "target": target}})
    return found
