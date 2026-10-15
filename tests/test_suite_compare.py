"""The paired suite scorer of docs/results/r9-followup.md §12 (evaluation tooling; no simulator)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("suite_compare", ROOT / "scripts" / "eval" / "suite_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(**over):
    m = {"runs": 35, "phases": 45, "failed": 0, "sim_seconds": 18000.0,
         "starts": {"relocalised": 45, "worst_err_m": 0.03, "worst_herr_deg": 2.0}, "red_le1_phases": 44,
         "contacts": [99, 0], "reloc": [130, 0], "attempts": 700, "outcomes": 480, "useful": 240, "red": 15,
         "ambiguous": 3, "rule_outcomes": {"bloom": 190, "stone": 190, "grump": 24}, "rule_outcomes_total": 404,
         "nav_failures": 10, "lost_frames": 1000, "frames": 10000, "lost_frac": 0.1,
         "nudge": {"completed": 20, "reached": 19, "share": 0.95},
         "nofix": {"approaches": 20, "seconds": 150.0, "share": 150.0 / 18000.0, "by_reason": {}}, "nofix_entities": 5,
         "targeting": {}, "targeting_by_fixture": {}, "restart": [10, 10], "reversal": (10, [50.0] * 10)}
    m.update(over)
    return m


def test_the_absolute_bounds_are_the_six_of_section_11():
    sc = _mod()
    rows = sc.absolute_bounds(_metrics())
    assert len(rows) == 6 and all(ok for _, _, ok in rows)
    assert not sc.absolute_bounds(_metrics(red_le1_phases=42))[3][2]
    assert not sc.absolute_bounds(_metrics(contacts=[99, 1]))[4][2]
    assert not sc.absolute_bounds(_metrics(reloc=[130, 1]))[5][2]
    assert not sc.absolute_bounds(_metrics(restart=[7, 10]))[1][2]
    assert not sc.absolute_bounds(_metrics(reversal=(7, [50.0] * 7 + [None] * 3)))[2][2]
    assert not sc.absolute_bounds(_metrics(failed=1))[0][2]  # a failed phase is not a relocalised start


def test_the_paired_bounds_compare_a_against_b():
    sc = _mod()
    b = _metrics(nofix={"approaches": 230, "seconds": 1592.0, "share": 1592.0 / 18526.0, "by_reason": {}})
    assert all(ok for _, _, ok in sc.paired_bounds(_metrics(), b))
    # 2 % of the time is the cap, and half of B's share
    a = _metrics(nofix={"approaches": 60, "seconds": 400.0, "share": 400.0 / 18000.0, "by_reason": {}})
    assert not sc.paired_bounds(a, b)[0][2]
    a = _metrics(nofix={"approaches": 60, "seconds": 300.0, "share": 300.0 / 18000.0, "by_reason": {}})
    b_low = _metrics(nofix={"approaches": 60, "seconds": 500.0, "share": 500.0 / 18000.0, "by_reason": {}})
    assert not sc.paired_bounds(a, b_low)[0][2]  # 1.7 % but more than half of B's 2.8 %
    assert not sc.paired_bounds(_metrics(outcomes=455), b)[1][2]  # 455 < 0.95 x 480
    assert sc.paired_bounds(_metrics(outcomes=456), b)[1][2]
    assert not sc.paired_bounds(_metrics(nav_failures=11), b)[4][2]
    assert not sc.paired_bounds(_metrics(lost_frac=0.111), b)[5][2]
    assert sc.paired_bounds(_metrics(lost_frac=0.109), b)[5][2]
    assert not sc.paired_bounds(_metrics(nudge={"completed": 20, "reached": 17, "share": 0.85}), b)[6][2]
    assert not sc.paired_bounds(_metrics(rule_outcomes_total=380), b)[3][2]


def test_runs_are_read_from_the_learning_layout_and_targeting_events_are_classed(tmp_path):
    sc = _mod()
    for name, phases in [("S-history_a-s0", [
        {"phase": {"label": "train"}, "sim_seconds": 480.0, "relocalized_at": 10.0, "frame_valid": True,
         "reloc_error_at_start": {"err_m": 0.02, "herr_rad": 0.01},
         "attitudes": {"e1": {"fixture": "bloom", "attitude": "liked", "expected_value": 0.9, "best_action": "signal"}},
         "interactions": [{"t": 30.0, "started": 20.0, "activity": "engage", "action": "signal", "status": "outcome",
                           "entity_id": "e1", "fixture": "bloom"},
                          {"t": 60.0, "started": 50.0, "activity": "revisit", "action": "signal", "status": "aborted",
                           "reason": "target not confirmed in fresh images", "entity_id": "e9", "fixture": None}],
         "outcomes": [{"t": 30.0, "entity_id": "e1", "fixture": "bloom", "action": "signal", "valence": 1.0}],
         "contacts": [], "decisions": [], "world_events": [], "trajectory": {"frames": 100, "lost_frames": 5,
                                                                             "reloc_transitions": 3, "false_reloc_transitions": 0},
         "targeting": [{"t": 60.0, "entity_id": "e9", "action": "signal", "event": "unconfirmed", "count": 2, "held": True},
                       {"t": 70.0, "entity_id": "e1", "action": "nudge", "event": "absent", "count": 1}]},
        {"phase": {"label": "test_after_restart"}, "sim_seconds": 60.0, "relocalized_at": 5.0, "frame_valid": True,
         "decisions": [{"chosen": {"activity": "revisit", "entity_id": "e1", "action": "signal"}}],
         "interactions": [], "outcomes": [], "contacts": [], "world_events": [], "trajectory": {"frames": 10, "lost_frames": 0},
         "attitudes": {"e1": {"fixture": "bloom"}}}])]:
        d = tmp_path / name / "history_a"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"experiment": "history_a", "seed": 0, "phases": phases}))
    m = sc.arm_metrics(tmp_path)
    assert m["runs"] == 1 and m["phases"] == 2 and m["restart"] == [1, 1]
    assert m["nofix"] == {"approaches": 1, "seconds": 10.0, "by_reason": {"aborted: target not confirmed in fresh images": 1},
                          "share": 10.0 / 540.0}
    assert m["outcomes"] == 1 and m["useful"] == 1 and m["rule_outcomes"]["bloom"] == 1
    assert m["targeting"] == {"unconfirmed+held": 1, "absent": 1}
    assert m["targeting_by_fixture"] == {"none": {"unconfirmed+held": 1}, "bloom": {"absent": 1}}
    assert abs(m["lost_frac"] - 5 / 110) < 1e-9
