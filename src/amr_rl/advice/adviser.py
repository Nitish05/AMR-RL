"""Stage 2 of the advice gate and the shadow-mode adviser.

A consultation happens at most ``budget_per_session`` times per session and never in a benign
session (no impasse). The backend is asked twice with differently worded prompts; the two
answers must agree (paraphrase consistency), every id and number must exist in the payload
(grounding), and the proposal must fit one of three fixed forms within the per-write cap. The
result is stored as a *shadow* record with its checks; at the next session's end deterministic
code compares its falsifiable prediction with the logged statistics and marks it
shadow_confirmed / shadow_refuted. Nothing is applied in this mode, and the honesty line of every
record says so. Repeated impasses with the same state signature reuse the stored record
instead of asking again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .impasse import ImpasseConfig, detect_impasses
from .payload import build_payload, grounded, signature

PROPOSAL_KINDS = ("trait_delta", "probe_ranking", "hypothesis")
STATISTICS = ("outcomes", "useful", "aversive", "change_hypotheses", "probes", "probes_useful",
              "options_engaged_3plus", "options_zero_progress", "distinct_options")
HONESTY = ("shadow mode: the proposal is stored and later scored against logged statistics; nothing is "
           "applied. An accepted change would be recorded as 'LLM-suggested bounded change to engineered "
           "parameter X, accepted by rule R after test T', never as a decision of Pip's.")


@dataclass
class AdviserConfig:
    enabled: bool = True
    mode: str = "shadow"  # the only mode that exists
    budget_per_session: int = 1
    backend: str = "template"
    impasse: ImpasseConfig = field(default_factory=ImpasseConfig)


class TemplateBackend:
    """First stage of the cascade: deterministic templates per trigger. It may abstain (None).
    Rationales quote only payload numbers and ids."""

    name = "template"

    def propose(self, payload, variant=0):
        triggers = payload.get("triggers") or []
        traits = payload.get("traits", {})
        objects = payload.get("objects", {})
        for t in triggers:
            kind = t["trigger"]
            if kind == "zero_progress_with_probes":
                eid, action = t["entity_id"], t["action"]
                untried = [(o, a) for o, obj in objects.items() for a, n in obj.get("probes_left", {}).items()
                           if o != eid and n > 0 and not any(k.startswith(f"{a}:") for k in obj.get("outcomes", {}))]
                if untried:
                    return {"kind": "probe_ranking", "objects": [o for o, _ in untried[:3]],
                            "probes": [{"entity_id": o, "action": a} for o, a in untried[:3]],
                            "rationale": f"{eid} (object) shows no learning progress after "
                                         f"{t['detail']['outcomes']} outcomes; untried options remain elsewhere",
                            "prediction": {"statistic": "probes_useful", "direction": "up"}}
                if "curiosity_type" in traits:
                    cap = 0.05
                    return {"kind": "trait_delta", "trait": "curiosity_type", "delta": cap,
                            "rationale": f"{eid} (object) has learning progress "
                                         f"{objects.get(eid, {}).get('learning_progress', {}).get(action, 0.0)} after "
                                         f"{t['detail']['outcomes']} outcomes",
                            "prediction": {"statistic": "options_zero_progress", "direction": "down"}}
            if kind == "attitude_flips":
                eid = t["entity_id"]
                return {"kind": "hypothesis", "objects": [eid],
                        "text": f"{eid} (object) alternates between good and bad outcomes; its response may be "
                                f"probabilistic rather than changing",
                        "rationale": f"{t['detail']['sign_changes']} sign changes in the last {t['detail']['window']} "
                                     f"outcomes of {eid} (object)",
                        "prediction": {"statistic": "change_hypotheses", "direction": "down"}}
            if kind == "frustration_two_sessions" and "novelty_bias" in traits:
                return {"kind": "trait_delta", "trait": "novelty_bias", "delta": 0.05,
                        "rationale": f"{t['detail']['renewals_now']} frustration renewals this session and "
                                     f"{t['detail']['renewals_prev']} in the previous one",
                        "prediction": {"statistic": "useful", "direction": "up"}}
            if kind == "new_object_low_margin":
                eid = t["entity_id"]
                return {"kind": "probe_ranking", "objects": [eid],
                        "probes": [{"entity_id": eid, "action": a} for a in objects.get(eid, {}).get("probes_left", {})],
                        "rationale": f"{eid} (object) has no outcomes yet and the top-two margin was "
                                     f"{round(t['detail']['margin'], 4)}",
                        "prediction": {"statistic": "probes_useful", "direction": "up"}}
        return None  # "don't know" is allowed without penalty


def schema_ok(proposal, payload, cap):
    problems = []
    kind = proposal.get("kind")
    if kind not in PROPOSAL_KINDS:
        problems.append(f"unknown kind {kind!r}")
    if kind == "trait_delta":
        if proposal.get("trait") not in payload.get("traits", {}):
            problems.append("trait missing or unknown")
        delta = proposal.get("delta")
        if not isinstance(delta, (int, float)) or abs(float(delta)) > cap + 1e-12:
            problems.append(f"delta outside the per-write cap {cap}")
    if kind == "probe_ranking":
        probes = proposal.get("probes") or []
        if not probes:
            problems.append("empty probe ranking")
        for p in probes:
            obj = payload.get("objects", {}).get(p.get("entity_id"), {})
            if p.get("action") not in obj.get("probes_left", {}):
                problems.append(f"probe {p} is not an option in the payload")
    if kind == "hypothesis" and not proposal.get("text"):
        problems.append("hypothesis without text")
    pred = proposal.get("prediction") or {}
    if pred.get("statistic") not in STATISTICS or pred.get("direction") not in ("up", "down"):
        problems.append("prediction must name a logged statistic and a direction")
    return not problems, problems


def _strip(proposal):
    return {k: v for k, v in (proposal or {}).items() if k != "rationale"}


class Adviser:
    def __init__(self, config: AdviserConfig | None = None, backend=None):
        self.cfg = config or AdviserConfig()
        self.backend = backend or TemplateBackend()
        if self.cfg.mode != "shadow":
            raise ValueError("only shadow mode exists; nothing is ever applied")
        self.last = None
        self.queries = 0

    # ------------------------------------------------------------ the gate
    def consult(self, memory, decision_log=(), *, renewals=0, now=0.0):
        """Session-end consultation. Returns the stored record, or None (benign session,
        budget spent, abstention)."""
        if not self.cfg.enabled or getattr(memory, "character_cfg", None) is None:
            return None
        self.verify_pending(memory)
        prev = int(memory.noted("advice:renewals_prev") or 0)
        memory.note("advice:renewals_prev", str(int(renewals)))
        triggers = detect_impasses(memory, decision_log, renewals_now=int(renewals), renewals_prev=prev,
                                   cfg=self.cfg.impasse)
        if not triggers:
            return None  # benign session: no query
        session_key = f"advice:queries:{memory.sessions}"
        used = int(memory.noted(session_key) or 0)
        if used >= self.cfg.budget_per_session:
            return None
        payload = build_payload(memory, triggers, decision_log, now)
        sig = signature(payload)
        cached = memory.advice_by_signature(sig)
        if cached and cached["status"] not in ("shadow_refuted", "rejected"):
            record = {**cached, "status": "cached", "t": float(now)}
            memory.record_advice(now, triggers[0]["trigger"], sig, payload, json.loads(cached["proposal"] or "null"),
                                 {"cached_from": cached["seq"]}, "cached", self.backend.name,
                                 json.loads(cached["baseline"] or "null"))
            self.last = record
            return record
        memory.note(session_key, str(used + 1))
        self.queries += 1
        first = self.backend.propose(payload, variant=0)
        second = self.backend.propose(payload, variant=1)
        checks = {"consistent": _strip(first) == _strip(second), "abstained": first is None}
        status = "shadow"
        if first is None:
            status = "abstained"
        else:
            ok_schema, schema_problems = schema_ok(first, payload, memory.character_cfg.advised_cap)
            ok_ground, ground_problems = grounded(first, payload)
            checks.update({"schema": ok_schema, "schema_problems": schema_problems, "grounded": ok_ground,
                           "grounding_problems": ground_problems})
            if not (ok_schema and ok_ground and checks["consistent"]):
                status = "rejected"
        baseline = memory.session_statistics()
        seq = memory.record_advice(now, triggers[0]["trigger"], sig, payload, first, checks, status,
                                   self.backend.name, baseline)
        self.last = {"seq": seq, "t": float(now), "trigger": triggers[0]["trigger"], "status": status,
                     "proposal": first, "checks": checks, "applied": False, "honesty": HONESTY}
        return self.last

    def verify_pending(self, memory):
        """Deterministic verifier: shadow records from earlier sessions are judged by this
        session's statistics against their baseline."""
        stats = memory.session_statistics()
        verdicts = []
        for row in memory.advice_rows(limit=200):
            if row["status"] != "shadow" or row["session"] == memory.sessions:
                continue
            proposal = json.loads(row["proposal"] or "null") or {}
            pred = proposal.get("prediction") or {}
            baseline = json.loads(row["baseline"] or "{}") or {}
            key = pred.get("statistic")
            if key not in stats or key not in baseline:
                verdict = "unverifiable"
            elif stats.get("outcomes", 0) == 0:
                verdict = "unverifiable"
            else:
                moved_up = stats[key] > baseline[key]
                moved_down = stats[key] < baseline[key]
                verdict = ("shadow_confirmed" if (pred.get("direction") == "up" and moved_up)
                           or (pred.get("direction") == "down" and moved_down) else "shadow_refuted")
            memory.set_advice_status(row["seq"], verdict, {"statistic": key, "baseline": baseline.get(key),
                                                           "observed": stats.get(key)})
            verdicts.append((row["seq"], verdict))
        return verdicts

    def snapshot(self, memory=None):
        out = {"enabled": self.cfg.enabled, "mode": self.cfg.mode, "backend": self.backend.name,
               "queries_this_process": self.queries, "last": self.last, "applied": 0, "honesty": HONESTY}
        if memory is not None and getattr(memory, "character_cfg", None) is not None:
            rows = memory.advice_rows(limit=500)
            by = {}
            for r in rows:
                by[r["status"]] = by.get(r["status"], 0) + 1
            out.update({"records": len(rows), "by_status": by})
        return out
