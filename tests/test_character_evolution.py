"""The character evolves only at session end, by engineered rules on logged statistics,
bounded per session and in range, with no mean reversion (docs/research/
PERSONALITY_RESEARCH_2.md section 2; the twins, lock-in and drift tests it names). After the
second admission gate (2026-10-16) one trait evolves, novelty bias; the retired traits keep
their rows and never move. No simulator."""

from types import SimpleNamespace

import pytest

from amr_rl.advice import Adviser
from amr_rl.behavior.character import EVOLVES, RETIRED, TRAITS, CharacterConfig
from amr_rl.learning.memory import ExperienceMemory

GOOD, BAD, NONE = "attach:yellow", "attach:red", "none"
counter = {"n": 0}


def receipt(entity, observed, now, action="signal"):
    counter["n"] += 1
    return {"event_id": f"ev-{counter['n']}", "entity_id": entity, "action": action, "context": "attach:none",
            "observed": observed, "timestamp": now, "authorised": True, "images_retained": True,
            "predicted": {}, "decision": {"policy": "learned"}}


def fresh(k, n):
    """Entities never met before: their first outcomes are this session's probes."""
    return [f"s{k}-{i}" for i in range(n)]


def session(path, outcomes, *, t_end=600.0, cfg=None):
    """One session: open (a boot), live through ``outcomes`` [(entity, observed), ...], consolidate, close."""
    memory = ExperienceMemory(path, agent_id="twin")
    memory.ensure_character(cfg or CharacterConfig(), t=0.0)
    for eid in sorted({"a", "b", "c"} | {e for e, _ in outcomes}):
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
    # an empty session has no probes, so novelty bias stays where it is (no mean reversion); the
    # retired traits never move, whatever their rules would have said
    assert values["novelty_bias"] == 0.2 and values["caution"] == 0.8 and values["curiosity_type"] == 0.7
    assert writes == []


def test_raised_apart_twins_differ_on_character_in_the_documented_direction(tmp_path):
    paying = [(e, GOOD) for e in fresh(0, 6)] + [("c", GOOD)]
    barren = [(e, NONE) for e in fresh(0, 6)] + [("c", NONE)]
    rich, rw, _ = session(tmp_path / "rich.sqlite", paying)
    poor, pw, _ = session(tmp_path / "poor.sqlite", barren)
    # novelty bias follows the share of the session's probes (first attempts) that paid off
    assert rich["novelty_bias"] > 0.5 > poor["novelty_bias"]
    assert rich["novelty_bias"] - 0.5 <= 0.05 + 1e-9 and 0.5 - poor["novelty_bias"] <= 0.05 + 1e-9  # one session's cap
    assert {w["key"] for w in rw + pw} == {"novelty_bias"}  # the retired traits never move
    for w in rw + pw:
        assert w["key"] in EVOLVES and "statistic" in w
    # every write is logged with its cause, provenance and the rule
    _, _, log = session(tmp_path / "rich.sqlite", paying)
    ends = [e for e in log if e["cause"] == "session_end"]
    assert ends and all(e["provenance"] == "consolidation" for e in ends)


def test_same_world_twins_stay_identical(tmp_path):
    world = [(fresh(0, 3)[0], GOOD), ("b", BAD), (fresh(0, 3)[1], GOOD), ("c", NONE), ("b", BAD), (fresh(0, 3)[2], NONE)]
    for _ in range(3):
        x, _, _ = session(tmp_path / "x.sqlite", world)
        y, _, _ = session(tmp_path / "y.sqlite", world)
    assert x == y and x["novelty_bias"] != 0.5


def test_lock_in_recovery_novelty_bias_comes_back_within_a_stated_number_of_sessions(tmp_path):
    path = tmp_path / "lock.sqlite"
    for k in range(10):
        values, _, _ = session(path, [(e, GOOD) for e in fresh(k, 4)])  # every probe pays
    peak = values["novelty_bias"]
    assert peak > 0.7  # ten paying sessions: a keen robot (eta 0.1, cap 0.05 per session)
    sessions_back = None
    for k in range(20):
        values, _, _ = session(path, [(e, NONE) for e in fresh(100 + k, 4)])  # probes stop paying
        if values["novelty_bias"] <= 0.5 + 0.5 * (peak - 0.5):
            sessions_back = k + 1
            break
    assert sessions_back is not None and sessions_back <= 12  # half-way back within twelve sessions


def test_drift_audit_every_step_within_the_cap_and_the_bounds(tmp_path):
    path = tmp_path / "drift.sqlite"
    cfg = CharacterConfig(session_cap=0.05)
    before = None
    for k in range(16):
        world = [[(e, GOOD) for e in fresh(k, 3)], [(e, BAD) for e in fresh(k, 3)],
                 [(e, GOOD if i % 2 else BAD) for i, e in enumerate(fresh(k, 4))], []][k % 4]
        values, writes, log = session(path, world, cfg=cfg)
        if before is not None:
            for key in values:
                assert abs(values[key] - before[key]) <= cfg.session_cap + 1e-9
        for key, value in values.items():
            spec = cfg.spec(key)
            assert spec.lo <= value <= spec.hi
        before = values
    assert sum(1 for e in log if e["cause"] == "session_end") >= 12  # the empty sessions write nothing


def test_retired_traits_do_not_evolve_even_where_their_rule_would_fire(tmp_path):
    # surprise weighting's rule (toward 0.5 at a detected rule change) is documented, not applied
    reversal = [("a", GOOD)] * 6 + [("a", NONE)] * 4
    memory = ExperienceMemory(tmp_path / "rev.sqlite", agent_id="twin")
    memory.ensure_character(CharacterConfig(values={"surprise_weighting": 0.9, "caution": 0.8}), t=0.0)
    memory.upsert_entity("a", appearance={"hue": 0, "fill": 1}, now=0)
    for k, (eid, observed) in enumerate(reversal):
        memory.record_outcome(receipt(eid, observed, 1.0 + k))
    stats = memory.session_statistics()
    assert stats["change_hypotheses"] >= 1  # the statistic itself is still logged
    writes = memory.consolidate_character(600.0)
    assert not any(w["key"] in RETIRED for w in writes)
    values = {k: r["value"] for k, r in memory.character().items()}
    assert values["surprise_weighting"] == 0.9 and values["caution"] == 0.8
    memory.close()


def test_noisy_objects_leave_curiosity_type_alone_and_paying_probes_lift_novelty(tmp_path):
    import numpy as np

    rng = np.random.default_rng(1)
    noisy = [("a", GOOD)] * 6 + [("b", GOOD if rng.random() < 0.5 else BAD) for _ in range(8)]
    values, writes, _ = session(tmp_path / "noisy.sqlite", noisy)
    w = {w["key"]: w for w in writes}
    # curiosity type is retired (gate 2026-10-13): its rule is documented but no longer applied
    assert "curiosity_type" not in w and values["curiosity_type"] == 0.0
    assert w["novelty_bias"]["target"] == pytest.approx(w["novelty_bias"]["statistic"]["probes_useful_fraction"])
    assert values["approach_tempo"] == 0.5 and values["persistence"] == 0.5  # retired: no rule


def test_runtime_close_consolidates_unless_switched_off(tmp_path):
    from amr_rl.runtime.robot import RobotRuntime, RuntimeConfig

    for evolve in (True, False):
        rt = RobotRuntime.__new__(RobotRuntime)
        rt.cfg = RuntimeConfig(character_evolve=evolve)
        rt.memory = ExperienceMemory(tmp_path / f"rt-{evolve}.sqlite", agent_id="rt")
        rt.memory.ensure_character(rt.cfg.character, t=0.0)
        rt.memory.upsert_entity("a", appearance={"hue": 0, "fill": 1}, now=0)
        rt.memory.record_outcome(receipt("a", GOOD, 1.0))  # a paying probe: novelty bias has a target
        rt.now, rt.semantic, rt.decision_log = 100.0, None, []
        rt.supervisor = SimpleNamespace(stop=lambda reason: None)
        rt.chooser = SimpleNamespace(renewals=[])
        rt.adviser = Adviser(rt.cfg.adviser)
        rt.close()
        assert bool(rt.character_changes) is evolve


def test_every_trait_has_an_evolution_statement_for_the_inspector(tmp_path):
    assert set(EVOLVES) == {t.key for t in TRAITS}
    assert all("retired" in EVOLVES[k] for k in RETIRED)
    memory = ExperienceMemory(tmp_path / "i.sqlite", agent_id="i")
    memory.ensure_character(CharacterConfig(), t=0.0)
    insp = memory.character_inspector()
    assert insp["evolution"] == EVOLVES
    assert {t["key"] for t in insp["traits"] if t["retired"]} == RETIRED  # the console can show why a row is inert
    memory.upsert_entity("a", appearance={"hue": 0, "fill": 1}, now=0)
    memory.record_outcome(receipt("a", GOOD, 1.0))
    assert [w["key"] for w in memory.consolidate_character(1.0)] == ["novelty_bias"]
    assert ExperienceMemory(tmp_path / "bare.sqlite").consolidate_character(1.0) == []
