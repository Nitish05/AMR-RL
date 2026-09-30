"""Outcome learning and persistence (adapted from BB8-RL tests/test_purpose_adaptation.py).

These exercise receipts and the pure value function, not RGB, navigation or the
simulator. Negative controls: static ineffective worlds settle to idle with a
finite probe budget; replays never train twice; restart keeps knowledge.
"""

import json
import random

import pytest

from amr_rl.behavior.chooser import ChooserConfig, option_values
from amr_rl.learning.memory import ExperienceMemory, LearningConfig

CFG = ChooserConfig()
ENTITIES = ("a", "b")


def open_memory(path, **overrides):
    memory = ExperienceMemory(path, agent_id="test-agent", config=LearningConfig(**overrides))
    for eid in ENTITIES:
        memory.upsert_entity(eid, appearance={"hue": 0, "fill": 1}, now=0)
    return memory


counter = {"n": 0}


def observe(memory, entity, observed, *, event=None, now=1.0, action="signal", context="attach:none"):
    counter["n"] += 1
    return memory.record_outcome({
        "event_id": event or f"ev-{counter['n']}", "entity_id": entity, "action": action, "context": context,
        "observed": observed, "timestamp": now, "authorised": True, "images_retained": True,
        "predicted": {}, "decision": {},
    })


def choose(memory, now=2.0, need=1.0):
    best = None
    for eid in ENTITIES:
        for item in option_values(memory, eid, "attach:none", now, need=need, novelty=1.0, distance=1.0, cfg=CFG):
            if item["action"] != "signal" or item["state"]["probes"] <= 0:
                continue
            if best is None or item["value"] > best[0]:
                best = (item["value"], eid)
    return None if best is None or best[0] <= 0 else best[1]


def useful():
    return "attach:yellow"


def establish(memory, good, successes):
    other = "b" if good == "a" else "a"
    for _ in range(3):
        observe(memory, other, "none")
    for _ in range(successes):
        observe(memory, good, useful())
    return other


def test_replay_is_idempotent_and_conflict_rejected(tmp_path):
    memory = open_memory(tmp_path / "m.sqlite")
    assert observe(memory, "a", useful(), event="x") is not None
    assert observe(memory, "a", useful(), event="x") is None
    with pytest.raises(ValueError):
        observe(memory, "a", "none", event="x")
    assert memory.counts()["outcomes"] == 1


def test_unauthorised_or_unretained_receipts_never_train(tmp_path):
    memory = open_memory(tmp_path / "m.sqlite")
    base = {"event_id": "e", "entity_id": "a", "action": "signal", "context": "attach:none",
            "observed": useful(), "timestamp": 1.0}
    for extra in ({"authorised": False, "images_retained": True}, {"authorised": True}, {}):
        with pytest.raises(ValueError):
            memory.record_outcome({**base, **extra})
    assert memory.counts()["outcomes"] == 0


def test_unknown_entity_and_tokens_rejected(tmp_path):
    memory = open_memory(tmp_path / "m.sqlite")
    with pytest.raises(ValueError):
        observe(memory, "zzz", useful())
    with pytest.raises(ValueError):
        observe(memory, "a", "exploded")


def test_learning_changes_prediction_and_attitude(tmp_path):
    memory = open_memory(tmp_path / "m.sqlite")
    assert memory.attitude("a", 1.0)["attitude"] == "unknown"
    for _ in range(3):
        observe(memory, "a", useful())
        observe(memory, "b", "attach:red")
    assert memory.attitude("a", 2.0)["attitude"] == "liked"
    assert memory.attitude("b", 2.0)["attitude"] == "disliked"
    assert memory.predict("a", "signal", "attach:none", 2.0)["probabilities"]["attach:yellow"] > 0.5


def test_context_specific_expectations_with_backoff(tmp_path):
    memory = open_memory(tmp_path / "m.sqlite")
    for _ in range(4):
        observe(memory, "a", useful(), context="attach:none")
        observe(memory, "a", "none", context="attach:yellow")
    lowered = memory.predict("a", "signal", "attach:none", 2.0)["expected_valence"]
    raised = memory.predict("a", "signal", "attach:yellow", 2.0)["expected_valence"]
    assert lowered > 0.5 > raised


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("good", ["a", "b"])
@pytest.mark.parametrize("successes", [2, 3, 6, 12])
def test_early_and_late_reversals_recover(tmp_path, seed, good, successes):
    rng = random.Random(seed)
    memory = open_memory(tmp_path / "m.sqlite")
    other = establish(memory, good, successes)
    assert choose(memory) == good
    choices = []
    for index in range(10):
        choice = choose(memory, now=3.0 + index)
        assert choice is not None
        choices.append(choice)
        outcome = useful() if choice == other else ("none" if rng.random() < 0.9 else "attach:blue")
        event = f"reversal-{seed}-{index}"
        assert observe(memory, choice, outcome, event=event, now=3.0 + index) is not None
        assert observe(memory, choice, outcome, event=event, now=3.0 + index) is None
    assert other in choices[:6]
    assert choices[-3:] == [other] * 3
    assert memory.counts()["change_epoch"] >= 1


def test_static_ineffective_world_uses_finite_budget_then_idles(tmp_path):
    memory = open_memory(tmp_path / "m.sqlite")
    counts = {"a": 0, "b": 0}
    for index in range(30):
        choice = choose(memory, now=2.0 + index)
        if choice is None:
            break
        counts[choice] += 1
        observe(memory, choice, "none", now=2.0 + index)
    # Value-driven stopping may idle before the probe budget (an upper bound) is spent.
    assert 1 <= counts["a"] <= 3 and 1 <= counts["b"] <= 3
    assert choose(memory, now=100.0) is None
    assert memory.counts()["change_epoch"] == 0


def test_isolated_success_does_not_reopen_alternatives(tmp_path):
    memory = open_memory(tmp_path / "m.sqlite")
    establish(memory, "b", 1)
    tries = 0
    for index in range(10):
        choice = choose(memory, now=3.0 + index)
        if choice is None:
            break
        assert choice == "b"
        tries += 1
        observe(memory, "b", "none", now=3.0 + index)
    assert 1 <= tries <= 3
    assert choose(memory, now=10.0) is None
    assert memory.counts()["change_epoch"] == 0


def test_restart_preserves_knowledge_and_replays(tmp_path):
    path = tmp_path / "m.sqlite"
    memory = open_memory(path)
    establish(memory, "a", 3)
    before = (memory.predict("a", "signal", "attach:none", 5.0), memory.counts())
    memory.close()
    memory = ExperienceMemory(path, agent_id="test-agent")
    after = (memory.predict("a", "signal", "attach:none", 5.0), memory.counts())
    assert json.dumps(before[0], sort_keys=True) == json.dumps(after[0], sort_keys=True)
    assert after[1]["outcomes"] == before[1]["outcomes"] and after[1]["sessions"] == before[1]["sessions"] + 1
    with pytest.raises(ValueError):
        ExperienceMemory(path, agent_id="someone-else")


def test_time_decay_renews_a_single_probe_per_half_life(tmp_path):
    memory = open_memory(tmp_path / "m.sqlite", half_life=100.0)
    for index in range(3):
        observe(memory, "a", "none", now=1.0 + index)
    assert memory.option_state("a", "signal", 50.0)["probes"] == 0
    assert memory.option_state("a", "signal", 110.0)["probes"] == 1
    observe(memory, "a", "none", now=111.0)
    assert memory.option_state("a", "signal", 150.0)["probes"] == 0
    assert memory.option_state("a", "signal", 205.0)["probes"] == 0
    assert memory.option_state("a", "signal", 212.0)["probes"] == 1


def test_explicit_reset_requires_confirmation_and_changes_identity(tmp_path):
    memory = open_memory(tmp_path / "m.sqlite")
    observe(memory, "a", useful())
    with pytest.raises(ValueError):
        memory.reset("yes")
    old = memory.agent_id
    memory.reset("RESET")
    assert memory.counts()["outcomes"] == 0 and memory.agent_id != old


def test_low_need_prefers_idle(tmp_path):
    memory = open_memory(tmp_path / "m.sqlite")
    establish(memory, "a", 4)
    assert choose(memory, need=1.0) == "a"
    assert choose(memory, need=0.0) is None


def test_map_version_change_invalidates_positions_not_knowledge(tmp_path):
    memory = open_memory(tmp_path / "m.sqlite")
    memory.upsert_entity("a", appearance={"hue": 0, "fill": 1}, now=1, map_version="m1", xy=(1, 2))
    observe(memory, "a", useful())
    memory.invalidate_positions("m2")
    ent = [e for e in memory.entities() if e["entity_id"] == "a"][0]
    assert ent["x"] is None and memory.counts()["outcomes"] == 1


def test_idle_is_a_bounded_rest_so_the_robot_reconsiders():
    """Regression: an open-ended idle with a static view produced no further
    decisions for 400 s although the engineered need had regrown."""
    from amr_rl.behavior.activities import Idle

    idle = Idle(None, "nothing worth its cost")
    for k in range(int(Idle.REST_SECONDS / 0.1)):
        assert idle.step(None, 10.0 + 0.1 * k) == (0.0, 0.0) and not idle.done
    idle.step(None, 10.0 + Idle.REST_SECONDS + 0.01)
    assert idle.done and idle.result["status"] == "completed"


def test_noisy_effect_needs_a_longer_failure_run_before_a_change_is_hypothesised(tmp_path):
    """Regression (noisy world, bloom yellow with p = 0.5): three failures in a row
    were read as a consequence change and old evidence was discounted, so the robot
    backed off an option whose true value stayed positive."""
    memory = open_memory(tmp_path / "m.sqlite")
    t = 1.0
    for outcome in ["attach:yellow", "none", "attach:yellow", "none", "attach:yellow", "attach:yellow"]:
        observe(memory, "a", outcome, now=t)
        t += 1
    changes = [observe(memory, "a", "none", now=t + k)["change_hypothesis"] for k in range(3)]
    assert not any(changes)  # a 3-failure run is unremarkable at ~60 % reliability
    more = [observe(memory, "a", "none", now=t + 3 + k)["change_hypothesis"] for k in range(3)]
    assert any(more)  # a long enough run is still recognised


def test_reliable_effect_still_changes_after_three_failures(tmp_path):
    memory = open_memory(tmp_path / "m.sqlite")
    for k in range(5):
        observe(memory, "a", useful(), now=1.0 + k)
    changes = [observe(memory, "a", "none", now=10.0 + k)["change_hypothesis"] for k in range(3)]
    assert changes == [False, False, True]
