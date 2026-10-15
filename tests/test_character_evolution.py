"""The character evolves only at session end, by engineered rules on logged statistics,
bounded per session and in range, with no mean reversion (docs/research/
PERSONALITY_RESEARCH_2.md section 2; the twins, lock-in and drift tests it names). No simulator."""

from types import SimpleNamespace

import pytest

from amr_rl.advice import Adviser
from amr_rl.behavior.character import EVOLVES, TRAITS, CharacterConfig
from amr_rl.learning.memory import ExperienceMemory

GOOD, BAD, NONE = "attach:yellow", "attach:red", "none"
counter = {"n": 0}


def receipt(entity, observed, now, action="signal"):
    counter["n"] += 1
    return {"event_id": f"ev-{counter['n']}", "entity_id": entity, "action": action, "context": "attach:none",
            "observed": observed, "timestamp": now, "authorised": True, "images_retained": True,
            "predicted": {}, "decision": {"policy": "learned"}}


def session(path, outcomes, *, t_end=600.0, cfg=None):
    """One session: open (a boot), live through ``outcomes`` [(entity, observed), ...], consolidate, close."""
    memory = ExperienceMemory(path, agent_id="twin")
    memory.ensure_character(cfg or CharacterConfig(), t=0.0)
    for eid in ("a", "b", "c"):
        memory.upsert_entity(eid, appearance={"hue": 0, "fill": 1}, now=0)
    for k, (eid, observed) in enumerate(outcomes):
        memory.record_outcome(receipt(eid, observed, 1.0 + k))
    writes = memory.consolidate_character(t_end)
    values = {k: r["value"] for k, r in memory.character().items()}
    log = memory.character_log(limit=200)
    memory.close()
    return values, writes, log


def test_no_session_statistics_means_no_change_and_no_reversion(tmp_path):
    cfg = CharacterConfig(values={"caution": 0.8, "novelty_bias": 0.2, "curiosity_type": 0.7})
    values, writes, _ = session(tmp_path / "m.sqlite", [], cfg=cfg)
    # an empty session: the reserve decays toward 0, so caution moves toward 0.5 by at most one cap;
    # the rules with no evidence (probes, progress, stability) leave their traits where they are
    assert values["novelty_bias"] == 0.2 and values["curiosity_type"] == 0.7
    assert 0.75 <= values["caution"] < 0.8
    assert {w["key"] for w in writes} == {"caution"}


def test_raised_apart_twins_differ_on_character_in_the_documented_direction(tmp_path):
    aversive = [("a", BAD)] * 6 + [("b", BAD)] * 4 + [("c", GOOD)]
    benign = [("a", GOOD)] * 6 + [("b", GOOD)] * 4 + [("c", NONE)]
    hard, hw, _ = session(tmp_path / "hard.sqlite", aversive)
    soft, sw, _ = session(tmp_path / "soft.sqlite", benign)
    # asset protection: the twin that collected rewards protects them (more caution); the one
    # that met red panels has nothing to protect
    assert soft["caution"] > 0.5 > hard["caution"]
    assert soft["caution"] - 0.5 <= 0.05 + 1e-9 and 0.5 - hard["caution"] <= 0.05 + 1e-9  # one session's cap
    for w in hw + sw:
        assert w["key"] in EVOLVES and "statistic" in w
    # every write is logged with its cause, provenance and the rule
    _, _, log = session(tmp_path / "soft.sqlite", benign)
    ends = [e for e in log if e["cause"] == "session_end"]
    assert ends and all(e["provenance"] == "consolidation" for e in ends)


def test_same_world_twins_stay_identical(tmp_path):
    world = [("a", GOOD), ("b", BAD), ("a", GOOD), ("c", NONE), ("b", BAD), ("a", GOOD)]
    for _ in range(3):
        x, _, _ = session(tmp_path / "x.sqlite", world)
        y, _, _ = session(tmp_path / "y.sqlite", world)
    assert x == y and x["caution"] != 0.5


def test_lock_in_recovery_caution_comes_back_within_a_stated_number_of_sessions(tmp_path):
    rich = [("a", GOOD)] * 8
    path = tmp_path / "lock.sqlite"
    for _ in range(10):
        values, _, _ = session(path, rich)
    peak = values["caution"]
    assert peak > 0.7  # ten rich sessions: a cautious, asset-protecting robot (eta 0.1: ~20 sessions full range)
    sessions_back = None
    for k in range(20):
        values, _, _ = session(path, [])  # benign, empty sessions: the reserve decays
        if values["caution"] <= 0.5 + 0.5 * (peak - 0.5):
            sessions_back = k + 1
            break
    assert sessions_back is not None and sessions_back <= 12  # half-way back within twelve sessions


def test_drift_audit_every_step_within_the_cap_and_the_bounds(tmp_path):
    path = tmp_path / "drift.sqlite"
    cfg = CharacterConfig(session_cap=0.05)
    worlds = [[("a", GOOD)] * 6, [("a", BAD)] * 6, [("a", GOOD), ("b", BAD)] * 3, []]
    before = None
    for k in range(16):
        values, writes, log = session(path, worlds[k % 4], cfg=cfg)
        if before is not None:
            for key in values:
                assert abs(values[key] - before[key]) <= cfg.session_cap + 1e-9
        for key, value in values.items():
            spec = cfg.spec(key)
            assert spec.lo <= value <= spec.hi
        before = values
    assert sum(1 for e in log if e["cause"] == "session_end") >= 16


def test_rule_change_pulls_surprise_weighting_to_symmetric_and_a_stable_paying_world_lifts_it(tmp_path):
    stable = [("a", GOOD)] * 8
    values, writes, _ = session(tmp_path / "stable.sqlite", stable, cfg=CharacterConfig(values={"surprise_weighting": 0.3}))
    assert values["surprise_weighting"] > 0.3  # toward 0.6
    # an established effect that then fails three times is a detected change
    reversal = [("a", GOOD)] * 6 + [("a", NONE)] * 4
    memory = ExperienceMemory(tmp_path / "rev.sqlite", agent_id="twin")
    memory.ensure_character(CharacterConfig(values={"surprise_weighting": 0.9}), t=0.0)
    memory.upsert_entity("a", appearance={"hue": 0, "fill": 1}, now=0)
    for k, (eid, observed) in enumerate(reversal):
        memory.record_outcome(receipt(eid, observed, 1.0 + k))
    stats = memory.session_statistics()
    assert stats["change_hypotheses"] >= 1
    writes = memory.consolidate_character(600.0)
    w = next(w for w in writes if w["key"] == "surprise_weighting")
    assert w["target"] == 0.5 and w["after"] < 0.9
    memory.close()


def test_noisy_objects_move_curiosity_toward_learning_progress_and_paying_probes_lift_novelty(tmp_path):
    import numpy as np

    rng = np.random.default_rng(1)
    noisy = [("a", GOOD)] * 6 + [("b", GOOD if rng.random() < 0.5 else BAD) for _ in range(8)]
    values, writes, _ = session(tmp_path / "noisy.sqlite", noisy)
    w = {w["key"]: w for w in writes}
    # curiosity type is retired (gate 2026-10-13): its rule is documented but no longer applied
    assert "curiosity_type" not in w and values["curiosity_type"] == 0.0
    assert w["novelty_bias"]["target"] == pytest.approx(w["novelty_bias"]["statistic"]["probes_useful_fraction"])
    assert values["approach_tempo"] == 0.5 and values["persistence"] == 0.5  # no rule; persistence retired


def test_runtime_close_consolidates_unless_switched_off(tmp_path):
    from amr_rl.runtime.robot import RobotRuntime, RuntimeConfig

    for evolve in (True, False):
        rt = RobotRuntime.__new__(RobotRuntime)
        rt.cfg = RuntimeConfig(character_evolve=evolve)
        rt.memory = ExperienceMemory(tmp_path / f"rt-{evolve}.sqlite", agent_id="rt")
        rt.memory.ensure_character(rt.cfg.character, t=0.0)
        rt.memory.upsert_entity("a", appearance={"hue": 0, "fill": 1}, now=0)
        rt.memory.record_outcome(receipt("a", GOOD, 1.0))
        rt.now, rt.semantic, rt.decision_log = 100.0, None, []
        rt.supervisor = SimpleNamespace(stop=lambda reason: None)
        rt.chooser = SimpleNamespace(renewals=[])
        rt.adviser = Adviser(rt.cfg.adviser)
        rt.close()
        assert bool(rt.character_changes) is evolve


def test_every_trait_has_an_evolution_statement_for_the_inspector(tmp_path):
    assert set(EVOLVES) == {t.key for t in TRAITS}
    memory = ExperienceMemory(tmp_path / "i.sqlite", agent_id="i")
    memory.ensure_character(CharacterConfig(), t=0.0)
    assert memory.character_inspector()["evolution"] == EVOLVES
    assert memory.consolidate_character(1.0) and ExperienceMemory(tmp_path / "bare.sqlite").consolidate_character(1.0) == []
