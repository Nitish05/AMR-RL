"""Baseline policies are reproducible (no simulator)."""

import numpy as np

from amr_rl.learning.memory import ExperienceMemory
from amr_rl.sim.learner_bench import MAP_VERSION, BenchRuntime


def runtime_with(ids_in_seen_order):
    memory = ExperienceMemory(":memory:")
    for k, eid in enumerate(ids_in_seen_order):
        memory.upsert_entity(eid, appearance={"hue": 60.0 * k, "fill": 0.9}, now=float(k),
                             map_version=MAP_VERSION, xy=(1.0, -0.6 + 0.6 * k), pos_sigma=0.05)
    rt = BenchRuntime(memory, policy="fixed", seed=0, room=5.0)
    rt.tracker.refresh()
    rt.pose = np.array([0.0, 0.0, 0.0])
    return rt


def test_fixed_baseline_follows_first_seen_order_not_random_ids():
    """Entity ids are random uuids in the runtime; sorting by them made the fixed
    baseline target a different fixture from run to run."""
    picks = []
    for ids in (["ent-ffffff", "ent-000000", "ent-888888"], ["ent-123456", "ent-abcdef", "ent-000001"]):
        rt = runtime_with(ids)
        choice = rt.chooser._pick(rt, rt.chooser.candidates_for(rt, 5.0))
        picks.append(ids.index(choice["entity_id"]))
    assert picks == [0, 0]  # always the first entity seen


def test_drive_clamp_is_an_evaluation_ablation_with_no_satiation_or_habituation():
    from amr_rl.behavior.chooser import Motivation, MotivationConfig

    m = Motivation(MotivationConfig(clamp=True))
    m.update(0.0)
    m.on_outcome("a", 1.0)
    m.update(300.0)
    assert m.need == 1.0 and m.novelty("a") == 1.0 and m.snapshot()["drive_clamped"]
    d = Motivation()
    d.update(0.0)
    d.on_outcome("a", 1.0)
    assert d.need < 0.6 and d.novelty("a") < 1.0 and not d.snapshot()["drive_clamped"]
