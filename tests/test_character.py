"""The temperament traits are engineered gains on existing behaviour parameters: 0.5 leaves
every parameter as it was, the mapping is applied only with character_active, and
re-applying it never compounds (no simulator)."""

from types import SimpleNamespace

import numpy as np
import pytest

from amr_rl.behavior.character import PRESETS, SCALED, TRAITS, config_for, effective, gain
from amr_rl.behavior.chooser import ChooserConfig, option_values
from amr_rl.learning.memory import ExperienceMemory, LearningConfig
from amr_rl.sim.learner_bench import MAP_VERSION, BenchRuntime


def test_the_defaults_are_the_behaviour_before_any_trait_existed():
    eff = effective({t.key: t.default for t in TRAITS})
    assert eff == effective({})
    assert all(v == 1.0 for k, v in eff.items() if k.endswith("_gain"))
    assert eff["chooser.value_z"] == 0.0
    assert eff["learning.initial_probes"] == LearningConfig().initial_probes
    assert eff["learning.change_failures"] == LearningConfig().change_failures
    assert eff["learning.surprise_ratio"] == LearningConfig().surprise_ratio == 1.0
    assert eff["learning.curiosity_lp"] == LearningConfig().curiosity_lp == 0.0
    assert set(SCALED.values()) <= {k.split(".")[1] for k in eff}
    # every trait's anchor is 0.5 except the mixing weight, whose unchanged value is 0
    assert {t.key: t.default for t in TRAITS if t.default != 0.5} == {"curiosity_type": 0.0}


def test_the_retired_learning_style_traits_leave_the_memory_parameters():
    # both learning-style traits are retired (gates 2026-10-13 and 2026-10-16): the stored values
    # no longer reach the memory, though the rows persist
    assert effective({"surprise_weighting": 1.0})["learning.surprise_ratio"] == 1.0
    assert effective({"surprise_weighting": 0.0})["learning.surprise_ratio"] == 1.0
    assert effective({"curiosity_type": 1.0})["learning.curiosity_lp"] == 0.0
    rt = _runtime(True, values={"surprise_weighting": 1.0, "curiosity_type": 1.0})
    rt._apply_character()
    assert rt.memory.cfg.surprise_ratio == 1.0 and rt.memory.cfg.curiosity_lp == 0.0
    assert rt.memory.character()["surprise_weighting"]["value"] == 1.0
    assert rt.memory.character()["curiosity_type"]["value"] == 1.0


def test_gains_are_bounded_and_the_speed_gain_is_capped_for_safety():
    assert gain(0.0) == 0.5 and gain(1.0) == 2.0 and gain(0.75) == pytest.approx(2 ** 0.5)
    assert gain(1.0, 0.5, 1.25) == 1.25 and gain(0.0, 0.5, 1.25) == 0.5  # the speed cap the tempo mapping used
    # approach tempo and caution are retired (gate 2, 2026-10-16): their stored values act on nothing
    assert effective({"approach_tempo": 1.0})["navigator.v_max_gain"] == 1.0
    assert effective({"approach_tempo": 0.0})["chooser.min_interval_gain"] == 1.0
    assert effective({"caution": 1.0})["chooser.value_z"] == 0.0
    assert effective({"caution": 0.0})["chooser.avoid_radius_gain"] == 1.0
    # persistence is retired (gate 2026-10-13): whatever the store holds, the probe budget stays 3
    assert effective({"persistence": 1.0})["learning.initial_probes"] == 3
    assert effective({"persistence": 0.0})["learning.initial_probes"] == 3


def test_no_preset_is_offered_after_the_second_gate_and_unknown_ones_are_refused():
    assert PRESETS == {"default": {}}  # bold and careful withdrawn (gate 2, 2026-10-16): one admitted trait
    assert config_for("default").values == {}
    for name in ("bold", "careful", "reckless"):
        with pytest.raises(ValueError):
            config_for(name)


def _memory_with_an_uncertain_option():
    memory = ExperienceMemory(":memory:")
    memory.upsert_entity("ent-a", appearance={"hue": 0.0, "fill": 0.9}, now=0.0, map_version=MAP_VERSION,
                         xy=(1.0, 0.0), pos_sigma=0.05)
    # one liked and one disliked outcome: a mean near zero with a wide posterior
    for n, observed in enumerate(("attach:green", "attach:red")):
        memory.record_outcome({"event_id": f"ev-{n}", "entity_id": "ent-a", "action": "signal",
                               "context": "attach:none", "observed": observed, "timestamp": 1.0 + n,
                               "authorised": True, "images_retained": True, "predicted": {}, "decision": {}})
    return memory


def _values(memory, cfg, now=10.0):
    rows = option_values(memory, "ent-a", "attach:none", now, need=0.8, novelty=1.0, distance=1.0, cfg=cfg)
    return {r["action"]: r for r in rows}


def test_defaults_leave_the_option_values_byte_identical():
    memory = _memory_with_an_uncertain_option()
    plain, traited = _values(memory, ChooserConfig()), _values(memory, ChooserConfig(info_gain=1.0, value_z=0.0))
    assert plain == traited


def test_caution_discounts_an_uncertain_option_and_boldness_inflates_it():
    memory = _memory_with_an_uncertain_option()
    plain = _values(memory, ChooserConfig())["signal"]
    careful = _values(memory, ChooserConfig(value_z=1.0))["signal"]
    bold = _values(memory, ChooserConfig(value_z=-1.0))["signal"]
    unc = plain["prediction"]["uncertainty"]
    assert unc > 0.1
    assert careful["value"] < plain["value"] < bold["value"]
    assert bold["value"] - careful["value"] == pytest.approx(2 * 0.8 * unc)  # need x (z range) x uncertainty
    assert careful["expected_value"] == plain["expected_value"]  # the posterior itself is untouched


def test_novelty_bias_scales_every_information_term():
    memory = _memory_with_an_uncertain_option()
    plain, keen = _values(memory, ChooserConfig()), _values(memory, ChooserConfig(info_gain=2.0))
    for action in plain:
        assert keen[action]["information_value"] == pytest.approx(2 * plain[action]["information_value"])
        assert keen[action]["expected_value"] == plain[action]["expected_value"]
    rt = BenchRuntime(memory, policy="learned", seed=0, room=5.0, chooser_cfg=ChooserConfig(info_gain=2.0))
    rt.tracker.refresh()
    rt.pose = np.array([0.0, 0.0, 0.0])
    inv = [c for c in rt.chooser.candidates_for(rt, 10.0) if c["activity"] == "investigate"]
    rt0 = BenchRuntime(memory, policy="learned", seed=0, room=5.0)
    rt0.tracker.refresh()
    rt0.pose = np.array([0.0, 0.0, 0.0])
    inv0 = [c for c in rt0.chooser.candidates_for(rt0, 10.0) if c["activity"] == "investigate"]
    if inv and inv0:
        assert inv[0]["information_value"] == pytest.approx(2 * inv0[0]["information_value"])


def _runtime(active, preset="default", values=None):
    from amr_rl.runtime.robot import RobotRuntime, RuntimeConfig

    rt = RobotRuntime.__new__(RobotRuntime)
    rt.cfg = RuntimeConfig(character_active=active, character_preset=preset)
    if preset != "default":
        rt.cfg.character = config_for(preset)
    rt.memory = ExperienceMemory(":memory:", config=rt.cfg.learning)
    rt.memory.ensure_character(rt.cfg.character, t=0.0)
    for key, value in (values or {}).items():
        rt.memory.set_trait(key, value, provenance="owner", cause="test", t=0.0)
    rt.nav = SimpleNamespace(cfg=SimpleNamespace(v_max=0.22))
    rt.character_effective, rt._character_base = None, None
    return rt


def test_the_mapping_is_inert_until_character_active():
    rt = _runtime(False, values={"novelty_bias": 0.75})
    rt._apply_character()
    assert rt.character_effective is None
    assert rt.cfg.chooser == ChooserConfig() and rt.nav.cfg.v_max == 0.22 and rt.memory.cfg.initial_probes == 3
    assert rt.memory.character()["novelty_bias"]["value"] == 0.75  # the store holds the value all the same


def test_an_active_character_sets_the_parameters_and_reapplying_never_compounds():
    rt = _runtime(True, values={"novelty_bias": 0.75, "approach_tempo": 0.75, "caution": 0.25})
    rt._apply_character()
    eff = rt.character_effective
    assert eff["values"]["novelty_bias"] == 0.75 and eff["values"]["approach_tempo"] == 0.75
    assert rt.cfg.chooser.info_gain == pytest.approx(2 ** 0.5) == rt.cfg.chooser.explore_gain
    # the retired traits' stored values act on nothing
    assert rt.nav.cfg.v_max == 0.22 and rt.cfg.chooser.value_z == 0.0 and rt.cfg.chooser.min_interval == 3.0
    assert rt.cfg.chooser.avoid_radius == 0.75 and rt.cfg.chooser.margin == pytest.approx(0.08)
    rt._apply_character()
    assert rt.cfg.chooser.info_gain == pytest.approx(2 ** 0.5)  # re-applying never compounds
    # a later owner write moves the parameters from the same base
    rt.memory.set_trait("novelty_bias", 0.5, provenance="owner", cause="test", t=1.0)
    rt._apply_character()
    assert rt.cfg.chooser.info_gain == 1.0 == rt.cfg.chooser.explore_gain


def test_a_configured_base_is_what_the_gains_scale():
    rt = _runtime(True, values={"novelty_bias": 1.0, "approach_tempo": 1.0, "persistence": 1.0})
    rt.cfg.chooser.min_interval, rt.nav.cfg.v_max, rt.cfg.chooser.info_gain = 5.0, 0.2, 1.0
    rt._apply_character()
    assert rt.cfg.chooser.info_gain == 2.0 == rt.cfg.chooser.explore_gain
    # the retired traits leave the configured base alone: speed, interval, probe budgets, margin
    assert rt.cfg.chooser.min_interval == 5.0 and rt.nav.cfg.v_max == 0.2
    assert rt.memory.cfg.initial_probes == 3 and rt.memory.cfg.change_failures == 3
    assert rt.cfg.chooser.margin == pytest.approx(0.08)


def test_engage_candidates_carry_their_uncertainty_into_the_decision_log():
    """The character gate's caution signature is the uncertainty of the chosen option, read from
    the logged candidates; the 2026-10-12 batch logged null because the record lacked it."""
    memory = _memory_with_an_uncertain_option()
    rt = BenchRuntime(memory, policy="learned", seed=0, room=5.0)
    rt.tracker.refresh()
    rt.pose = np.array([0.0, 0.0, 0.0])
    cands = rt.chooser.candidates_for(rt, 10.0)
    options = [c for c in cands if c["activity"] in ("engage", "revisit")]
    assert options, "the bench has one entity with an uncertain option"
    assert all(c["uncertainty"] is not None and c["uncertainty"] > 0.1 for c in options)
    rt.chooser.candidates, rt.chooser.chosen = cands, options[0]
    logged = rt.chooser.snapshot()["candidates"]
    assert any(c["uncertainty"] == options[0]["uncertainty"] for c in logged)


# ---------------------------------------------------------------- second pre-registration (2026-10-15)
def test_v2_caution_reads_an_untried_option_at_its_prior_and_a_tried_one_pessimistically():
    """The first careful end (z = 1.64) made every untried option's value negative and the robot
    interacted with nothing in two worlds. Pessimism now needs evidence to be pessimistic about."""
    fresh = ExperienceMemory(":memory:")
    fresh.upsert_entity("ent-new", appearance={"hue": 200.0, "fill": 0.9}, now=0.0, map_version=MAP_VERSION,
                        xy=(1.0, 0.0), pos_sigma=0.05)
    kw = dict(need=0.8, novelty=1.0, distance=1.0)
    plain = option_values(fresh, "ent-new", "attach:none", 10.0, cfg=ChooserConfig(), **kw)
    careful = option_values(fresh, "ent-new", "attach:none", 10.0, cfg=ChooserConfig(value_z=1.64), **kw)
    for a, b in zip(plain, careful):
        assert a["prediction"]["evidence"]["all_contexts"] == 0 and a["prediction"]["uncertainty"] > 0.1
        assert b["value"] == a["value"]  # nothing to be pessimistic about yet
    tried = _memory_with_an_uncertain_option()
    assert _values(tried, ChooserConfig(value_z=1.64))["signal"]["value"] < _values(tried, ChooserConfig())["signal"]["value"]


def test_v2_novelty_bias_never_discourages_exploring_space():
    """At novelty bias 0.2 (gain 0.66) one first-gate run stood still for a whole phase: it knew no
    entity and exploring a mapped room did not pay. Object information scales both ways, space only up."""
    assert effective({"novelty_bias": 0.2})["chooser.explore_gain"] == 1.0
    assert effective({"novelty_bias": 0.2})["chooser.info_gain"] == pytest.approx(gain(0.2))
    assert effective({"novelty_bias": 0.8})["chooser.explore_gain"] == pytest.approx(gain(0.8))
    rt = _runtime(True, values={"novelty_bias": 0.2})
    rt._apply_character()
    assert rt.cfg.chooser.explore_gain == 1.0 and rt.cfg.chooser.info_gain == pytest.approx(gain(0.2))
    memory = _memory_with_an_uncertain_option()
    shy = BenchRuntime(memory, policy="learned", seed=0, room=5.0, chooser_cfg=ChooserConfig(info_gain=0.5))
    base = BenchRuntime(memory, policy="learned", seed=0, room=5.0)
    for rt_ in (shy, base):
        rt_.tracker.refresh()
        rt_.pose = np.array([0.0, 0.0, 0.0])
    ex = [c for c in shy.chooser.candidates_for(shy, 10.0) if c["activity"] == "explore"]
    ex0 = [c for c in base.chooser.candidates_for(base, 10.0) if c["activity"] == "explore"]
    assert ex and ex0 and ex[0]["information_value"] == pytest.approx(ex0[0]["information_value"])


def test_retired_traits_are_inert_and_not_owner_facing():
    from amr_rl.behavior.character import ACTIVE, RETIRED, CharacterConfig

    assert RETIRED == {"persistence", "curiosity_type", "approach_tempo", "caution", "surprise_weighting"}
    assert tuple(s.key for s in ACTIVE) == ("novelty_bias",)  # the one trait that passed its gate
    assert effective({k: 1.0 for k in RETIRED}) == effective({}) == effective({k: 0.0 for k in RETIRED})
    assert not any(k in RETIRED for preset in PRESETS.values() for k in preset)
    memory = ExperienceMemory(":memory:")
    memory.ensure_character(CharacterConfig(), t=0.0)
    assert set(memory.character()) == {s.key for s in TRAITS}  # the rows persist
    assert all(s.retired for s in TRAITS if s.key in RETIRED) and all(not s.retired for s in ACTIVE)
