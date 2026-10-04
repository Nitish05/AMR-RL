"""Learner testbed (no physics): world rules, perception error model, short episodes."""

import numpy as np

from amr_rl.sim.learner_bench import (
    NOISE_PRESETS,
    Episode,
    EpisodeConfig,
    NoiseConfig,
    NoisyPerception,
    RobotConfig,
    SyntheticWorld,
    WorldConfig,
)


def test_world_assigns_one_reliable_and_one_unreliable_rewarding_object():
    w = SyntheticWorld(WorldConfig(), np.random.default_rng(3))
    rewarding = sorted(o.rules["signal"][1] for o in w.objects if o.rules["signal"][0] == "attach:yellow")
    assert rewarding == [0.5, 1.0]
    assert sum(o.rules["signal"][0] == "attach:red" for o in w.objects) == 1
    assert sum(o.rules["nudge"][0] == "moved" for o in w.objects) == 1
    twins = w.objects[0], w.objects[1]
    assert twins[0].hue == twins[1].hue and twins[0].width == twins[1].width


def test_inert_world_never_responds_and_reversal_moves_the_reward():
    w = SyntheticWorld(WorldConfig(world="inert"), np.random.default_rng(0))
    assert all(r[0] == "none" for o in w.objects for r in o.rules.values())
    w = SyntheticWorld(WorldConfig(switch_at=10.0), np.random.default_rng(0))
    before = {o.index for o in w.objects if o.rules["signal"][0] == "attach:yellow"}
    assert not w.maybe_switch(5.0) and w.maybe_switch(10.0)
    after = {o.index for o in w.objects if o.rules["signal"][0] == "attach:yellow"}
    assert after and not (after & before)


def test_perception_error_model():
    w = SyntheticWorld(WorldConfig(), np.random.default_rng(1))
    obj = w.objects[0]
    pose = np.array([obj.xy[0] - 1.0, obj.xy[1], 0.0])
    blind = NoisyPerception(w, NoiseConfig(miss=1.0), RobotConfig(), np.random.default_rng(0))
    assert blind.detect(pose, 0.0) == []
    clean = NoisyPerception(w, NoiseConfig(), RobotConfig(), np.random.default_rng(0))
    dets = [d for d in clean.detect(pose, 0.0) if d._true == obj.index]
    assert dets and abs(dets[0].position[0] - obj.xy[0]) < 0.15
    # A confused outcome label is consistent for the whole response, and a neighbour colour.
    obj.rules["signal"] = ("attach:yellow", 1.0)
    w.respond(obj, "signal", pose[:2], 0.0, NoiseConfig(label_confusion=1.0), True)
    tokens = {d.state_token for _ in range(5) for d in clean.detect(pose, 0.1) if d._true == obj.index}
    assert tokens <= {"attach:orange", "attach:green"} and len(tokens) == 1


def test_short_episodes_run_and_score_for_every_policy():
    for policy in ("learned", "random", "nearest", "fixed"):
        ep = Episode(EpisodeConfig(seconds=200, policy=policy, seed=0, noise=NOISE_PRESETS["clean"])).run()
        s = ep.summary()
        assert s["seconds"] >= 200 and s["entities"] >= 1
        assert s["attempts"] == s["outcomes"] + s["ambiguous"] + s["aborted"]


def test_clean_learner_finds_rewards():
    s = Episode(EpisodeConfig(seconds=600, seed=1, noise=NOISE_PRESETS["clean"])).run().summary()
    assert s["rewards"] >= 2 and s["label_accuracy"] is not None and s["label_accuracy"] > 0.9


def test_measured_flag_errors_are_one_sided_and_read_from_a_benchmark(tmp_path):
    import json

    w = SyntheticWorld(WorldConfig(), np.random.default_rng(1))
    obj = w.objects[0]
    pose = np.array([obj.xy[0] - 1.0, obj.xy[1], 0.0])

    def tokens(noise):
        p = NoisyPerception(w, noise, RobotConfig(), np.random.default_rng(0))
        return {d.state_token for _ in range(20) for d in p.detect(pose, 0.1) if d._true == obj.index}

    obj.state = "attach:none"
    assert tokens(NoiseConfig(false_flag=1.0, false_flag_color="red")) == {"attach:red"}
    assert tokens(NoiseConfig(flag_miss=1.0)) == {"attach:none"}
    obj.state = "attach:yellow"
    assert tokens(NoiseConfig(flag_miss=1.0)) == {"attach:none"}
    assert tokens(NoiseConfig(false_flag=1.0)) == {"attach:yellow"}
    rows = [{"name": "a", "range": 1.0, "truncated": False, "matched": True, "pos_err": 0.02,
             "state_truth": "attach:yellow", "state_pred": "attach:none"},
            {"name": "a", "range": 1.0, "truncated": False, "matched": True, "pos_err": 0.02,
             "state_truth": "attach:none", "state_pred": "attach:none"},
            {"name": "b", "range": 1.5, "truncated": False, "matched": False},
            {"name": "c", "range": 3.0, "truncated": False, "matched": False}]
    (tmp_path / "bench.json").write_text(json.dumps({"rows": rows}))
    n = NoiseConfig.from_measurement(tmp_path / "bench.json")
    assert abs(n.miss - 1 / 3) < 1e-9 and n.flag_miss == 1.0 and n.false_flag == 0.0
    assert abs(n.pos_sigma - 0.02 / 1.1774) < 1e-6
