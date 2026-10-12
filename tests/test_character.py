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


def test_learning_style_traits_set_the_memory_parameters():
    assert effective({"surprise_weighting": 1.0})["learning.surprise_ratio"] == 2.0
    assert effective({"surprise_weighting": 0.0})["learning.surprise_ratio"] == 0.5
    assert effective({"curiosity_type": 1.0})["learning.curiosity_lp"] == 1.0
    assert effective({"curiosity_type": 0.5})["learning.curiosity_lp"] == 0.5
    rt = _runtime(True, values={"surprise_weighting": 1.0, "curiosity_type": 1.0})
    rt._apply_character()
    assert rt.memory.cfg.surprise_ratio == 2.0 and rt.memory.cfg.curiosity_lp == 1.0
    rt.memory.set_trait("curiosity_type", 0.0, provenance="owner", cause="test", t=1.0)
    rt._apply_character()
    assert rt.memory.cfg.curiosity_lp == 0.0


def test_gains_are_bounded_and_the_speed_gain_is_capped_for_safety():
    assert gain(0.0) == 0.5 and gain(1.0) == 2.0 and gain(0.75) == pytest.approx(2 ** 0.5)
    assert effective({"approach_tempo": 1.0})["navigator.v_max_gain"] == 1.25
    assert effective({"approach_tempo": 0.0})["navigator.v_max_gain"] == 0.5
    assert effective({"caution": 1.0})["chooser.value_z"] == pytest.approx(1.64)
    assert effective({"caution": 0.0})["chooser.value_z"] == pytest.approx(-1.64)
    assert effective({"persistence": 1.0})["learning.initial_probes"] == 6
    assert effective({"persistence": 0.0})["learning.initial_probes"] == 2


def test_presets_are_contrasting_and_unknown_ones_are_refused():
    bold, careful = effective(PRESETS["bold"]), effective(PRESETS["careful"])
    assert bold["navigator.v_max_gain"] > 1 > careful["navigator.v_max_gain"]
    assert bold["chooser.value_z"] < 0 < careful["chooser.value_z"]
    assert bold["chooser.info_gain"] > 1 > careful["chooser.info_gain"]
    assert config_for("bold").values == PRESETS["bold"] and config_for("default").values == {}
    with pytest.raises(ValueError):
        config_for("reckless")


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
    rt = _runtime(False, "bold")
    rt._apply_character()
    assert rt.character_effective is None
    assert rt.cfg.chooser == ChooserConfig() and rt.nav.cfg.v_max == 0.22 and rt.memory.cfg.initial_probes == 3
    assert rt.memory.character()["caution"]["value"] == 0.25  # the store holds the preset all the same


def test_an_active_character_sets_the_parameters_and_reapplying_never_compounds():
    rt = _runtime(True, "bold")
    rt._apply_character()
    eff = rt.character_effective
    assert eff["values"]["approach_tempo"] == 0.75
    assert rt.nav.cfg.v_max == pytest.approx(0.22 * 1.25) and rt.cfg.chooser.value_z == pytest.approx(-0.82)
    assert rt.cfg.chooser.min_interval == pytest.approx(3.0 / 1.25)
    assert rt.cfg.chooser.avoid_radius == pytest.approx(0.75 * 0.75)
    assert rt.cfg.chooser.margin == pytest.approx(0.08 * gain(0.6))
    assert rt.cfg.chooser.info_gain == pytest.approx(2 ** 0.5)
    rt._apply_character()
    assert rt.nav.cfg.v_max == pytest.approx(0.22 * 1.25) and rt.cfg.chooser.min_interval == pytest.approx(3.0 / 1.25)
    # a later owner write moves the parameters from the same base
    rt.memory.set_trait("approach_tempo", 0.5, provenance="owner", cause="test", t=1.0)
    rt._apply_character()
    assert rt.nav.cfg.v_max == pytest.approx(0.22) and rt.cfg.chooser.min_interval == pytest.approx(3.0)


def test_a_configured_base_is_what_the_gains_scale():
    rt = _runtime(True, values={"approach_tempo": 1.0, "persistence": 1.0})
    rt.cfg.chooser.min_interval, rt.nav.cfg.v_max = 5.0, 0.2
    rt._apply_character()
    assert rt.cfg.chooser.min_interval == pytest.approx(4.0) and rt.nav.cfg.v_max == pytest.approx(0.25)
    assert rt.memory.cfg.initial_probes == 6 and rt.memory.cfg.change_failures == 6
    assert rt.cfg.chooser.margin == pytest.approx(0.16)
