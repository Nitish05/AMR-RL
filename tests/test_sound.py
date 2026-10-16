"""The speaker is a one-way event bus on logged state transitions (no simulator)."""

import copy
import importlib
import inspect
import wave

import numpy as np
import pytest

from amr_rl.expression import sound
from amr_rl.expression.sound import (
    CUES,
    INTENSITY,
    SUBTLE,
    SoundBus,
    SoundConfig,
    render_wav,
    signature_from,
    transit_rate_hz,
)


def state(name="explore", status="tracking", interaction=None, outcome=None, since=0.0, chosen=None, **extra):
    s = {"activity": {"name": name, "target_entity": "ent-a", "since": since},
         "localization": {"status": status}, "interaction": interaction,
         "recent_outcomes": [outcome] if outcome else [], "decision": {"chosen": chosen}}
    s.update(extra)
    return s


def test_one_cue_per_visible_transition_and_none_without_one():
    bus = SoundBus(SoundConfig(min_gap_s=0.0))
    assert bus.observe(state("idle"), (0.0, 0.0), 0.0) == []  # the first snapshot sets the baseline
    assert bus.observe(state("idle"), (0.0, 0.0), 1.0) == []  # nothing changed: silence
    cues = [e["cue"] for e in bus.observe(state("explore"), (0.0, 0.0), 2.0)]
    assert cues == ["explore_start"]
    e = bus.observe(state("engage"), (0.0, 0.0), 3.0)[0]
    assert e["cue"] == "approach" and e["paired_with"] == ["activity.name", "activity.target_entity"]
    assert [x["cue"] for x in bus.observe(state("engage", interaction={"action": "signal", "status": "acting"}),
                                          (0.0, 0.0), 4.0)] == ["action"]
    good = {"event_id": "e1", "valence": 1.0}
    assert [x["cue"] for x in bus.observe(state("engage", outcome=good), (0.0, 0.0), 5.0)] == ["outcome_positive"]
    assert bus.observe(state("engage", outcome=good), (0.0, 0.0), 6.0) == []  # the same outcome again: no cue
    bad = {"event_id": "e2", "valence": -1.0}
    assert [x["cue"] for x in bus.observe(state("engage", outcome=bad), (0.0, 0.0), 7.0)] == ["outcome_aversive"]
    assert [x["cue"] for x in bus.observe(state("avoid", outcome=bad), (0.0, 0.0), 8.0)] == ["avoid"]
    assert [x["cue"] for x in bus.observe(state("avoid", status="lost", outcome=bad), (0.0, 0.0), 9.0)] == ["attention"]
    assert [x["cue"] for x in bus.observe(state("blocked", outcome=bad), (0.0, 0.0), 10.0)] == ["attention"]
    assert set(x["cue"] for x in bus.log) <= set(CUES) | {SUBTLE} | set(sound.TRANSIT)


def test_idle_long_fires_once_per_rest_at_the_threshold():
    bus = SoundBus(SoundConfig(idle_long_s=20.0))
    bus.observe(state("idle", since=0.0), (0.0, 0.0), 0.0)
    assert bus.observe(state("idle", since=0.0), (0.0, 0.0), 19.0) == []
    assert [e["cue"] for e in bus.observe(state("idle", since=0.0), (0.0, 0.0), 20.0)] == ["idle_long"]
    assert bus.observe(state("idle", since=0.0), (0.0, 0.0), 40.0) == []
    bus.observe(state("explore", since=41.0), (0.0, 0.0), 41.0)
    bus.observe(state("idle", since=50.0), (0.0, 0.0), 50.0)
    assert [e["cue"] for e in bus.observe(state("idle", since=50.0), (0.0, 0.0), 70.0)] == ["idle_long"]


def test_uncertain_is_a_subtle_companion_of_a_probe_approach():
    bus = SoundBus(SoundConfig(min_gap_s=0.0))
    bus.observe(state("explore"), (0.0, 0.0), 0.0)
    cues = [e["cue"] for e in bus.observe(state("engage", chosen={"uncertainty": 0.4}), (0.0, 0.0), 1.0)]
    assert cues == ["approach", SUBTLE]
    bus.observe(state("explore"), (0.0, 0.0), 2.0)
    cues = [e["cue"] for e in bus.observe(state("engage", chosen={"uncertainty": 0.05}), (0.0, 0.0), 3.0)]
    assert cues == ["approach"]
    assert INTENSITY[SUBTLE] < min(INTENSITY[c] for c in CUES)


def test_intensity_is_fixed_per_cue_whatever_the_surprise_or_the_character():
    bus = SoundBus(SoundConfig(min_gap_s=0.0))
    traits = [{"key": "approach_tempo", "value": 1.0}, {"key": "novelty_bias", "value": 1.0}]
    bus.observe(state("engage"), (0.0, 0.0), 0.0)
    big = bus.observe(state("engage", outcome={"event_id": "a", "valence": -1.0, "update": {"surprise": 9.0}},
                            character={"active": True, "traits": traits}), (0.0, 0.0), 1.0)[0]
    small = bus.observe(state("engage", outcome={"event_id": "b", "valence": -0.1, "update": {"surprise": 0.01}}),
                        (0.0, 0.0), 2.0)[0]
    assert big["intensity"] == small["intensity"] == INTENSITY["outcome_aversive"]
    assert big["signature"]["pitch_variability"] == 1.0 and small["signature"]["pitch_variability"] == 0.5


def test_budget_and_minimum_gap_drop_cues_and_count_them_but_outcomes_stay_synchronised():
    bus = SoundBus(SoundConfig(min_gap_s=0.8, budget_per_min=3))
    bus.observe(state("idle"), (0.0, 0.0), 0.0)
    assert [e["cue"] for e in bus.observe(state("explore"), (0.0, 0.0), 1.0)] == ["explore_start"]
    assert bus.observe(state("engage"), (0.0, 0.0), 1.2) == []  # inside the gap
    assert bus.dropped == {"approach": 1}
    assert [e["cue"] for e in bus.observe(state("engage", outcome={"event_id": "x", "valence": 1.0}),
                                          (0.0, 0.0), 1.3)] == ["outcome_positive"]  # synchronised to the flag
    assert [e["cue"] for e in bus.observe(state("avoid", outcome={"event_id": "x", "valence": 1.0}),
                                          (0.0, 0.0), 3.0)] == ["avoid"]
    assert bus.observe(state("explore", outcome={"event_id": "x", "valence": 1.0}), (0.0, 0.0), 5.0) == []
    assert bus.dropped["explore_start"] == 1  # the per-minute budget (3) is spent
    assert [e["cue"] for e in bus.observe(state("idle", since=69.0, outcome={"event_id": "x", "valence": 1.0}),
                                          (0.0, 0.0), 70.0)] == []  # idle has no transition cue
    assert [e["cue"] for e in bus.observe(state("explore", outcome={"event_id": "x", "valence": 1.0}),
                                          (0.0, 0.0), 71.0)] == ["explore_start"]  # the window moved on


def test_transit_markers_are_time_locked_to_the_command_and_the_rate_follows_speed():
    bus = SoundBus()
    assert [e["cue"] for e in bus.observe(state("explore"), (0.0, 0.0), 0.0)] == []
    on = bus.observe(state("explore"), (0.12, 0.0), 0.1)
    assert [e["cue"] for e in on] == ["transit_on"] and on[0]["t"] == 0.1 and on[0]["paired_with"] == ["command.v"]
    assert bus.transit["on"] and bus.transit["rate_hz"] == pytest.approx(transit_rate_hz(0.12))
    bus.observe(state("explore"), (0.25, 0.0), 0.2)
    assert bus.transit["rate_hz"] == 8.0 and transit_rate_hz(0.0) == 2.0 and transit_rate_hz(1.0) == 8.0
    off = bus.observe(state("explore"), (0.0, 0.0), 0.3)
    assert [e["cue"] for e in off] == ["transit_off"] and not bus.transit["on"]
    assert bus.dropped == {}  # the transit layer is outside the cue budget


def test_signature_uses_tempo_pitch_variability_and_density_only_when_the_character_is_active():
    s = state("explore", expression={"style": {"tempo": 1.2}},
              character={"active": False, "traits": [{"key": "novelty_bias", "value": 0.9}]})
    sig = signature_from(s)
    assert sig["tempo"] == 1.2 and sig["pitch_variability"] == 0.5 and sig["density"] == 0.5
    s["character"]["active"] = True
    s["character"]["traits"].append({"key": "approach_tempo", "value": 0.1})
    sig = signature_from(s)
    assert sig["pitch_variability"] == 0.9 and sig["density"] == 0.1
    assert set(sig) == {"tempo", "pitch_variability", "density", "basis"}  # no pitch, no loudness
    # a trait the state marks retired reaches no cue (approach tempo, second gate 2026-10-16)
    s["character"]["traits"][-1]["retired"] = "gate 2"
    sig = signature_from(s)
    assert sig["density"] == 0.5 and "character.traits[approach_tempo].value" not in sig["basis"]


def test_the_bus_is_read_only_and_imports_nothing_from_control_learning_or_the_simulator():
    bus = SoundBus()
    s = state("engage", outcome={"event_id": "e", "valence": 1.0})
    before = copy.deepcopy(s)
    bus.observe(state("idle"), (0.0, 0.0), 0.0)
    bus.observe(s, (0.1, 0.0), 1.0)
    assert s == before and sound.unchanged(before, s)
    src = inspect.getsource(importlib.import_module("amr_rl.expression.sound"))
    for forbidden in ("amr_rl.learning", "amr_rl.behavior", "amr_rl.control", "amr_rl.sim", "amr_rl.runtime",
                      "genesis", "openai", "anthropic", "llm"):
        assert forbidden not in src
    off = SoundBus(SoundConfig(enabled=False))
    assert off.observe(s, (0.1, 0.0), 1.0) == [] and off.snapshot()["emissions"] == 0


def test_render_wav_is_deterministic_and_the_signature_changes_length_not_loudness(tmp_path):
    base = [{"t": 0.0, "cue": "outcome_positive", "intensity": 0.6, "signature": {"tempo": 1.0, "pitch_variability": 0.5,
                                                                                  "density": 0.5}},
            {"t": 0.5, "cue": "attention", "intensity": 0.55, "signature": {"tempo": 1.0, "pitch_variability": 0.5,
                                                                            "density": 0.5}}]
    a = render_wav(base, tmp_path / "a.wav")
    b = render_wav(base)
    assert np.array_equal(a, b) and a.dtype == np.float32 and np.abs(a).max() <= 1.0
    with wave.open(str(tmp_path / "a.wav")) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getnframes() == len(a)
    brisk = [{**e, "signature": {"tempo": 1.25, "pitch_variability": 1.0, "density": 1.0}} for e in base]
    c = render_wav(brisk)
    assert len(c) == len(a) and not np.array_equal(a, c)
    assert abs(np.abs(c).max() - np.abs(a).max()) < 0.15  # the signature does not change loudness
    assert render_wav([]).shape[0] > 0
