"""The shadow-mode adviser: impasse gate, robot-side payload, checks, budget, cache,
deterministic verification, and nothing applied (no simulator)."""

import copy
import json
from types import SimpleNamespace

import pytest

from amr_rl.advice import Adviser, AdviserConfig, TemplateBackend, build_payload, detect_impasses, grounded
from amr_rl.advice.payload import FORBIDDEN_KEYS, signature
from amr_rl.behavior.character import CharacterConfig
from amr_rl.learning.memory import ExperienceMemory

GOOD, BAD, NONE = "attach:yellow", "attach:red", "none"
counter = {"n": 0}


def receipt(entity, observed, now, action="signal"):
    counter["n"] += 1
    return {"event_id": f"ev-{counter['n']}", "entity_id": entity, "action": action, "context": "attach:none",
            "observed": observed, "timestamp": now, "authorised": True, "images_retained": True,
            "predicted": {}, "decision": {"policy": "learned"}}


def memory(path, outcomes=(), cfg=None):
    m = ExperienceMemory(path, agent_id="adv")
    m.ensure_character(cfg or CharacterConfig(), t=0.0)
    for eid in ("a", "b", "c"):
        m.upsert_entity(eid, appearance={"hue": 10.0 * len(eid), "fill": 0.9}, now=0)
    for k, (eid, obs) in enumerate(outcomes):
        m.record_outcome(receipt(eid, obs, 1.0 + k))
    return m


def decision(eid, margin, t=50.0):
    return {"t": t, "trigger": None, "chosen": {"activity": "engage", "entity_id": eid, "action": "signal", "value": 0.3},
            "top": [{"value": 0.3}, {"value": 0.3 - margin}]}


def test_impasse_triggers_fire_on_their_logged_conditions(tmp_path):
    flips = [("a", GOOD), ("a", BAD), ("a", GOOD), ("a", BAD), ("a", GOOD)]
    m = memory(tmp_path / "f.sqlite", flips)
    found = {t["trigger"] for t in detect_impasses(m)}
    assert "attitude_flips" in found
    assert not detect_impasses(memory(tmp_path / "quiet.sqlite", [("a", GOOD)] * 3))  # benign: nothing
    assert any(t["trigger"] == "frustration_two_sessions" for t in
               detect_impasses(m, renewals_now=2, renewals_prev=1))
    assert not any(t["trigger"] == "frustration_two_sessions" for t in detect_impasses(m, renewals_now=2, renewals_prev=0))
    low = detect_impasses(m, [decision("c", 0.01)])
    assert any(t["trigger"] == "new_object_low_margin" and t["entity_id"] == "c" for t in low)
    assert not any(t["trigger"] == "new_object_low_margin" for t in detect_impasses(m, [decision("c", 0.2)]))
    assert not any(t["trigger"] == "new_object_low_margin" for t in detect_impasses(m, [decision("a", 0.01)]))  # known object
    m.close()


def test_zero_progress_with_probes_and_trait_at_bound(tmp_path):
    noisy = [("b", GOOD), ("b", BAD)] * 4  # alternating: the predictions never improve, probes remain
    m = memory(tmp_path / "z.sqlite", noisy)
    zp = [t for t in detect_impasses(m) if t["trigger"] == "zero_progress_with_probes"]
    assert zp and zp[0]["entity_id"] == "b" and zp[0]["detail"]["probes_left"] > 0
    m.close()
    # every probe paid off, so the rule wants novelty bias at 1.0, where the owner already put it
    m = memory(tmp_path / "bound.sqlite", [("a", GOOD)] * 3, cfg=CharacterConfig(values={"novelty_bias": 1.0}))
    m.consolidate_character(100.0)
    found = [t for t in detect_impasses(m) if t["trigger"] == "trait_at_bound"]
    assert found and found[0]["detail"] == {"trait": "novelty_bias", "value": 1.0, "target": 1.0}
    m.close()
    # a trait resting at its anchor bound (curiosity_type at 0) is not an impasse
    m = memory(tmp_path / "anchor.sqlite", [("a", GOOD)] * 3)
    m.consolidate_character(100.0)
    assert not [t for t in detect_impasses(m) if t["trigger"] == "trait_at_bound"]
    m.close()


def test_payload_is_robot_side_only_and_signed_by_state(tmp_path):
    m = memory(tmp_path / "p.sqlite", [("a", GOOD), ("b", BAD)])
    triggers = [{"trigger": "attitude_flips", "entity_id": "a", "action": None, "detail": {"sign_changes": 3}}]
    p = build_payload(m, triggers, [decision("c", 0.01)], now=10.0)
    text = json.dumps(p).lower()
    for key in FORBIDDEN_KEYS:
        assert f'"{key}' not in text
    assert set(p) == {"schema", "identity", "traits", "objects", "activity", "triggers"}
    assert p["objects"]["a"]["outcomes"] == {"signal:attach:yellow": 1} and p["objects"]["a"]["attitude"] == "liked"
    assert "position" not in p["objects"]["a"] and "x" not in p["objects"]["a"]
    assert p["activity"]["chosen"] == {"engage": 1} and p["traits"]["caution"]["anchor"] == 0.5
    sig = signature(p)
    assert sig == signature(build_payload(m, triggers, [], now=99.0))  # activity counters are not part of the state
    m.record_outcome(receipt("c", GOOD, 20.0))
    assert sig != signature(build_payload(m, triggers, [], now=99.0))
    m.close()


def test_grounding_rejects_unknown_objects_numbers_and_traits(tmp_path):
    m = memory(tmp_path / "g.sqlite", [("a", GOOD)] * 2)
    p = build_payload(m, [], [], now=5.0)
    ok, problems = grounded({"kind": "hypothesis", "rationale": "a (object) gave 2 outcomes", "trait": None}, p)
    assert ok, problems
    ok, problems = grounded({"kind": "hypothesis", "rationale": "z (object) gave 17 outcomes"}, p)
    assert not ok and len(problems) == 2
    ok, problems = grounded({"kind": "trait_delta", "trait": "bravery", "rationale": ""}, p)
    assert not ok and problems == ["unknown trait bravery"]
    m.close()


def test_consult_only_after_an_impasse_once_per_session_and_never_applies_anything(tmp_path):
    adviser = Adviser(AdviserConfig())
    benign = memory(tmp_path / "b.sqlite", [("a", GOOD)] * 3)
    assert adviser.consult(benign, [], renewals=0, now=100.0) is None and adviser.queries == 0
    assert benign.advice_rows() == []
    benign.close()
    flips = [("a", GOOD), ("a", BAD), ("a", GOOD), ("a", BAD), ("a", GOOD)]
    m = memory(tmp_path / "f.sqlite", flips)
    before = copy.deepcopy(m.character())
    record = adviser.consult(m, [], renewals=0, now=100.0)
    assert record["status"] == "shadow" and record["proposal"]["kind"] == "hypothesis" and record["applied"] is False
    assert record["checks"]["consistent"] and record["checks"]["grounded"] and record["checks"]["schema"]
    assert "nothing is applied" in record["honesty"]
    assert m.character() == before  # nothing written to the traits
    assert all(e["provenance"] != "llm_advised" for e in m.character_log())
    assert adviser.consult(m, [], renewals=0, now=101.0) is None  # the session's budget is spent
    assert adviser.queries == 1 and len(m.advice_rows()) == 1
    insp = m.character_inspector()["advice"]
    assert insp["mode"] == "shadow" and insp["records"] == 1 and insp["applied"] == 0
    m.close()


def test_cache_by_state_signature_reuses_a_record_instead_of_asking_again(tmp_path):
    path = tmp_path / "c.sqlite"
    flips = [("a", GOOD), ("a", BAD), ("a", GOOD), ("a", BAD), ("a", GOOD)]
    m = memory(path, flips)
    adviser = Adviser()
    adviser.consult(m, [], renewals=0, now=100.0)
    m.close()
    m = ExperienceMemory(path, agent_id="adv")  # next session, same state
    m.ensure_character(CharacterConfig(), t=0.0)
    record = adviser.consult(m, [], renewals=0, now=200.0)
    assert record["status"] == "cached" and adviser.queries == 1
    assert [r["status"] for r in m.advice_rows()] == ["cached", "unverifiable"]  # no outcomes this session: unverifiable
    m.close()


def test_deterministic_verifier_confirms_or_refutes_by_the_next_sessions_statistics(tmp_path):
    path = tmp_path / "v.sqlite"
    noisy = [("b", GOOD), ("b", BAD), ("b", GOOD), ("b", BAD), ("b", GOOD), ("b", BAD), ("b", GOOD), ("b", BAD)]
    m = memory(path, noisy)
    adviser = Adviser()
    record = adviser.consult(m, [], renewals=0, now=100.0)
    assert record["status"] == "shadow" and record["proposal"]["prediction"]["direction"] in ("up", "down")
    pred = record["proposal"]["prediction"]
    m.close()
    # next session: outcomes that move the predicted statistic
    m = ExperienceMemory(path, agent_id="adv")
    m.ensure_character(CharacterConfig(), t=0.0)
    if pred["statistic"] == "change_hypotheses":
        for k in range(3):
            m.record_outcome(receipt("a", GOOD, 1.0 + k))
    else:
        for k in range(4):
            m.record_outcome(receipt("c", GOOD, 1.0 + k))  # probes on an untried object pay off
    verdicts = adviser.verify_pending(m)
    assert verdicts and verdicts[0][1] in ("shadow_confirmed", "shadow_refuted", "unverifiable")
    row = next(r for r in m.advice_rows() if r["seq"] == record["seq"])
    assert row["status"] == verdicts[0][1] and json.loads(row["verdict"])["statistic"] == pred["statistic"]
    m.close()


def test_paraphrase_disagreement_or_an_oversized_delta_is_rejected(tmp_path):
    class Flaky(TemplateBackend):
        name = "flaky"

        def propose(self, payload, variant=0):
            return {"kind": "trait_delta", "trait": "caution", "delta": 0.05 if variant == 0 else -0.05,
                    "rationale": "", "prediction": {"statistic": "aversive", "direction": "down"}}

    class Greedy(TemplateBackend):
        name = "greedy"

        def propose(self, payload, variant=0):
            return {"kind": "trait_delta", "trait": "caution", "delta": 0.4, "rationale": "",
                    "prediction": {"statistic": "aversive", "direction": "down"}}

    flips = [("a", GOOD), ("a", BAD), ("a", GOOD), ("a", BAD), ("a", GOOD)]
    for backend in (Flaky(), Greedy()):
        m = memory(tmp_path / f"{backend.name}.sqlite", flips)
        record = Adviser(backend=backend).consult(m, [], renewals=0, now=100.0)
        assert record["status"] == "rejected" and record["applied"] is False
        assert m.character()["caution"]["value"] == 0.5
        m.close()


def test_only_shadow_mode_exists_and_the_runtime_consults_at_close(tmp_path):
    with pytest.raises(ValueError):
        Adviser(AdviserConfig(mode="apply"))
    from amr_rl.runtime.robot import RobotRuntime, RuntimeConfig

    rt = RobotRuntime.__new__(RobotRuntime)
    rt.cfg = RuntimeConfig()
    rt.memory = memory(tmp_path / "rt.sqlite", [("a", GOOD), ("a", BAD), ("a", GOOD), ("a", BAD), ("a", GOOD)])
    rt.adviser = Adviser(rt.cfg.adviser)
    rt.decision_log, rt.now, rt.semantic = [], 100.0, None
    rt.chooser = SimpleNamespace(renewals=[])
    rt.supervisor = SimpleNamespace(stop=lambda reason: None)
    rt.close()
    assert rt.advice is not None and rt.advice["status"] == "shadow" and rt.advice["applied"] is False
