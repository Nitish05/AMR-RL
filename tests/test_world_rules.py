"""World-side response rules (sim/world.choose_response; no simulator)."""

import numpy as np
import yaml

from amr_rl.robot.spec import PROJECT_ROOT
from amr_rl.sim.world import choose_response


def test_single_response_rules_draw_exactly_as_before():
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    rule = {"response": "yellow", "p": 0.5}
    old = []
    for _ in range(200):
        r = "yellow"
        if a.random() > 0.5:
            r = "none"
        old.append(r)
    assert [choose_response(rule, b) for _ in range(200)] == old
    c = np.random.default_rng(1)
    assert choose_response({}, c) == "none" and c.random() == np.random.default_rng(1).random()  # no draw spent


def test_multi_response_rules_follow_their_probabilities_with_the_remainder_none():
    rng = np.random.default_rng(0)
    rule = {"responses": [{"response": "yellow", "p": 0.5}, {"response": "red", "p": 0.25}]}
    draws = [choose_response(rule, rng) for _ in range(4000)]
    f = {k: draws.count(k) / len(draws) for k in ("yellow", "red", "none")}
    assert abs(f["yellow"] - 0.5) < 0.03 and abs(f["red"] - 0.25) < 0.03 and abs(f["none"] - 0.25) < 0.03


def test_gate_worlds_have_the_pre_registered_means():
    valence = {"yellow": 1.0, "red": -1.0, "none": 0.0}

    def moments(rule):
        opts = rule.get("responses") or [{"response": rule.get("response", "none"), "p": rule.get("p", 1.0)}]
        ps = {o["response"]: float(o["p"]) for o in opts}
        ps["none"] = ps.get("none", 0.0) + 1.0 - sum(ps.values())
        mean = sum(p * valence[k] for k, p in ps.items())
        return mean, sum(p * (valence[k] - mean) ** 2 for k, p in ps.items())

    variance = yaml.safe_load((PROJECT_ROOT / "configs/consequences/variance.yaml").read_text())
    (mb, vb), (ms, vs) = moments(variance["bloom"]["signal"]), moments(variance["stone"]["signal"])
    assert mb == ms == 0.5 and vs > vb  # equal means, stone varies more
    noisy = yaml.safe_load((PROJECT_ROOT / "configs/consequences/noisy_tv.yaml").read_text())
    assert moments(noisy["stone"]["signal"])[0] == 0.0 and moments(noisy["bloom"]["signal"]) == (1.0, 0.0)
