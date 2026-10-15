"""Re-targeting holds (docs/results/r9-followup.md §12; behavior/targeting.py): an option that
never confirms is not approached again this session; an entity absent from its remembered place
waits for a sighting. Both off by default: behaviour and log strings are then unchanged. No
simulator."""

from types import SimpleNamespace

import numpy as np

from amr_rl.behavior.activities import Engage
from amr_rl.behavior.chooser import ChooserConfig
from amr_rl.behavior.targeting import ABSENT_FRAMES, TargetLedger
from amr_rl.learning.memory import ExperienceMemory
from amr_rl.sim.learner_bench import MAP_VERSION, BenchRuntime


def test_two_unconfirmed_approaches_hold_an_option_that_was_never_confirmed():
    led = TargetLedger(ChooserConfig(unconfirmed_hold=2))
    led.note_unconfirmed("crate", "signal", 10.0, xy=(0.0, 1.35))
    assert not led.held("crate", "signal")
    led.note_unconfirmed("crate", "signal", 50.0, xy=(0.0, 1.35))
    assert led.held("crate", "signal") and not led.held("crate", "nudge")
    assert led.events[-1] == {"t": 50.0, "entity_id": "crate", "action": "signal", "event": "unconfirmed",
                              "count": 2, "held": True}
    assert led.snapshot() == {"held": ["crate:signal"], "absent": []}
    # an option confirmed earlier keeps only the ordinary back-off: chance misses are not a property of the object
    led.note_confirmed("bloom", "signal")
    for t in (60.0, 70.0, 80.0):
        led.note_unconfirmed("bloom", "signal", t)
    assert not led.held("bloom", "signal")
    # a confirmation in between resets the count
    led.note_unconfirmed("stone", "nudge", 90.0)
    led.note_confirmed("stone", "nudge")
    led.note_unconfirmed("stone", "nudge", 95.0)
    assert not led.held("stone", "nudge")
    off = TargetLedger(ChooserConfig())
    for t in (1.0, 2.0, 3.0):
        off.note_unconfirmed("crate", "signal", t)
    assert not off.held("crate", "signal") and len(off.events) == 3 and "held" not in off.events[-1]


def test_an_absent_entity_waits_for_a_sighting_or_a_new_place():
    led = TargetLedger(ChooserConfig(absent_hold=True))
    led.note_unconfirmed("roller", "signal", 10.0, xy=(-1.2, -0.8), absent=True)
    assert led.events[-1]["event"] == "absent"
    assert led.absent("roller", (-1.2, -0.8), visible=False)
    assert led.absent("roller", (-1.0, -0.8), visible=False)  # still "there" within 0.3 m
    assert not led.absent("roller", (-1.2, -0.8), visible=True)  # seen again: approachable
    assert not led.absent("roller", (-0.6, -0.8), visible=False)  # remembered elsewhere now
    assert led.snapshot()["absent"] == ["roller"]
    led.note_confirmed("roller", "signal")
    assert not led.absent("roller", (-1.2, -0.8), visible=False)
    off = TargetLedger(ChooserConfig())
    off.note_unconfirmed("roller", "signal", 10.0, xy=(-1.2, -0.8), absent=True)
    assert not off.absent("roller", (-1.2, -0.8), visible=False)
    assert not off.absence_verdict(seen=False, empty_frames=ABSENT_FRAMES)
    assert led.absence_verdict(seen=False, empty_frames=ABSENT_FRAMES)
    assert not led.absence_verdict(seen=True, empty_frames=ABSENT_FRAMES)
    assert not led.absence_verdict(seen=False, empty_frames=ABSENT_FRAMES - 1)


def _bench(**chooser):
    memory = ExperienceMemory(":memory:")
    memory.upsert_entity("ent-crate", appearance={"hue": 30.0, "fill": 0.9}, now=0.0, map_version=MAP_VERSION,
                         xy=(1.0, 0.0), pos_sigma=0.05)
    memory.upsert_entity("ent-bloom", appearance={"hue": 180.0, "fill": 0.9}, now=0.0, map_version=MAP_VERSION,
                         xy=(0.0, 1.0), pos_sigma=0.05)
    rt = BenchRuntime(memory, policy="learned", seed=0, room=5.0, chooser_cfg=ChooserConfig(**chooser))
    rt.tracker.refresh()
    return rt


def _options(rt, now):
    return {(c["entity_id"], c["action"]) for c in rt.chooser.candidates_for(rt, now)
            if c["activity"] in ("engage", "revisit")}


def test_the_chooser_drops_held_options_and_absent_entities():
    rt = _bench(unconfirmed_hold=2, absent_hold=True)
    assert {("ent-crate", "signal"), ("ent-crate", "nudge"), ("ent-bloom", "signal")} <= _options(rt, 3.0)
    rt.targets.note_unconfirmed("ent-crate", "signal", 3.0, xy=(1.0, 0.0))
    rt.targets.note_unconfirmed("ent-crate", "signal", 40.0, xy=(1.0, 0.0))
    opts = _options(rt, 50.0)
    assert ("ent-crate", "signal") not in opts and ("ent-crate", "nudge") in opts and ("ent-bloom", "signal") in opts
    rt.targets.note_unconfirmed("ent-crate", "nudge", 60.0, xy=(1.0, 0.0), absent=True)
    assert not any(c["entity_id"] == "ent-crate" for c in rt.chooser.candidates_for(rt, 70.0))
    assert ("ent-bloom", "signal") in _options(rt, 70.0)


def test_with_the_switches_off_nothing_is_dropped():
    rt = _bench()
    for t in (3.0, 40.0, 60.0):
        rt.targets.note_unconfirmed("ent-crate", "signal", t, xy=(1.0, 0.0), absent=True)
    assert {("ent-crate", "signal"), ("ent-crate", "nudge")} <= _options(rt, 70.0)


def _engage_runtime(**chooser):
    rt = SimpleNamespace(pose=np.array([0.0, 0.0, 0.0]), tracker=SimpleNamespace(visible=[]),
                         last_frame_record={"raw_rgb_sha256": "0" * 64}, supervisor=SimpleNamespace(generation=1),
                         targets=TargetLedger(ChooserConfig(**chooser)), notes=[])
    rt.place_in_view = lambda xy: True
    rt.note_unreachable = lambda eid, now, missing=False: rt.notes.append((eid, round(now, 1), missing))
    rt.entity_appearance = lambda eid: None
    return rt


def _confirm_window_runs_out(rt):
    act = Engage("ent-roller", "signal", (0.6, 0.0), decision={})
    act.goto = SimpleNamespace(status="arrived")  # already at the standoff
    act.phase, act.phase_until = "confirming", 3.0
    for k in range(40):
        act.step(rt, 0.1 * k)
        if act.done:
            break
    assert act.done and act.result["status"] == "aborted"
    return act


def test_an_empty_confirmation_window_is_an_absence_only_with_the_switch_on():
    rt = _engage_runtime(absent_hold=True)
    act = _confirm_window_runs_out(rt)
    assert act.result["reason"] == "target absent from its remembered place"
    assert act.confirm_empty >= ABSENT_FRAMES and not act.confirm_seen
    assert rt.targets.absent("ent-roller", (0.6, 0.0), visible=False)
    assert rt.notes == [("ent-roller", 3.1, True)]  # the ordinary back-off still applies
    assert rt.targets.events[-1]["event"] == "absent"


def test_the_old_reason_and_no_hold_with_the_switches_off():
    rt = _engage_runtime()
    act = _confirm_window_runs_out(rt)
    assert act.result["reason"] == "target not confirmed in fresh images"
    assert not rt.targets.absent("ent-roller", (0.6, 0.0), visible=False)
    assert not rt.targets.held("ent-roller", "signal")
    assert rt.notes == [("ent-roller", 3.1, True)]
    assert rt.targets.events == [{"t": 3.1, "entity_id": "ent-roller", "action": "signal", "event": "unconfirmed",
                                  "count": 1}]
