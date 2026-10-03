"""Bounded telemetry vs complete audit recording for AMR runtime rows."""

import json

import pytest

from amr_rl.evidence.recording import RotatingJsonl, SessionRecorder, finish_recording


def amr_row(
    t,
    *,
    generation=1,
    activity="explore",
    loc="tracking",
    stopped=False,
    nav="following",
    goal=(1.0, 0.5),
    interaction=None,
    pose_x=0.0,
):
    return {
        "sim_time": t,
        "generation": generation,
        "activity": {"name": activity, "target_entity": None, "phase": "roaming"},
        "localization": {"status": loc, "pose": [pose_x, 0.0, 0.0], "position_sigma": 0.03},
        "authority": {
            "stopped": stopped,
            "autonomy_enabled": not stopped,
            "manual_active": False,
            "revoked_reason": "stop" if stopped else None,
        },
        "navigation": {"status": nav, "goal": list(goal) if goal else None},
        "interaction": interaction,
        "camera": {"frame_index": int(t * 20), "fresh": True},
    }


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_rotation_is_bounded_and_counts_evictions(tmp_path):
    log = RotatingJsonl(tmp_path / "t.jsonl", segment_bytes=1024, segments=2)
    for index in range(200):
        log.append({"type": "x", "index": index, "pad": "p" * 40})
    summary = log.summary()
    files = sorted(tmp_path.glob("t*.jsonl"))
    assert len(files) == 2
    assert sum(path.stat().st_size for path in files) <= summary["quota_bytes"]
    retained = sum(len(lines(path)) for path in files)
    assert summary["records_written"] == 200
    assert summary["records_retained"] == retained
    assert summary["records_evicted"] == 200 - retained > 0
    assert summary["history_truncated"]
    newest = lines(tmp_path / "t.jsonl")
    assert newest[-1]["index"] == 199


def test_oversized_record_is_replaced_and_counted(tmp_path):
    log = RotatingJsonl(tmp_path / "t.jsonl", segment_bytes=1024)
    log.append({"type": "blob", "data": "x" * 5000})
    (record,) = lines(tmp_path / "t.jsonl")
    assert record["type"] == "oversized_record" and record["record_type"] == "blob"
    assert log.summary()["oversized_records"] == 1


def test_telemetry_samples_logs_transitions_and_summarizes_heartbeats(tmp_path):
    recorder = SessionRecorder(tmp_path, sample_seconds=1.0)
    for step in range(100):  # 5 s at 20 Hz
        t = step * 0.05
        kwargs = {"pose_x": t}  # noisy estimate changes must not create transitions
        if 20 <= step < 40:
            kwargs["activity"] = "engage"
            kwargs["interaction"] = {
                "request_id": "r1",
                "entity_id": "ent-1",
                "action": "signal",
                "status": "observing",
            }
        if step >= 60:
            kwargs.update(stopped=True, generation=2, nav="idle", goal=None)
        recorder.row(amr_row(t, **kwargs))
        recorder.command({"action": "heartbeat"})
    recorder.command({"action": "stop"})
    summary = recorder.finish(status="ok")
    samples = lines(tmp_path / "telemetry.jsonl")
    assert [s["sim_time"] for s in samples] == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    events = lines(tmp_path / "events.jsonl")
    transitions = [e for e in events if e["type"] == "transition"]
    assert [e["row_index"] for e in transitions] == [1, 21, 41, 61]
    assert transitions[1]["activity"]["name"] == "engage"
    assert transitions[1]["interaction"]["status"] == "observing"
    assert transitions[3]["authority"]["revoked_reason"] == "stop"
    assert transitions[3]["generation"] == 2
    assert [e["type"] for e in events if e["type"] != "transition"] == ["command", "session_end"]
    assert not any(e.get("action") == "heartbeat" for e in events)
    assert summary["heartbeat_commands_summarized"] == 100
    assert summary["rows_seen"] == 100 and summary["sampled_rows"] == 5
    assert summary["transitions"] == 4
    assert json.loads((tmp_path / "recording.json").read_text()) == summary
    assert not list(tmp_path.glob("*.tmp"))


def test_interaction_status_and_goal_changes_are_transitions(tmp_path):
    recorder = SessionRecorder(tmp_path, sample_seconds=100.0)
    base = {"request_id": "r1", "entity_id": "e", "action": "push"}
    recorder.row(amr_row(0.0, interaction={**base, "status": "acting"}))
    recorder.row(amr_row(0.1, interaction={**base, "status": "observing"}))
    recorder.row(amr_row(0.2, interaction={**base, "status": "observing"}))
    recorder.row(amr_row(0.3, interaction={**base, "status": "observing"}, goal=(2.0, 0.0)))
    recorder.row(amr_row(0.4, interaction={**base, "status": "observing"}, loc="lost"))
    assert recorder.summary()["transitions"] == 4


def test_simulation_reset_restarts_sampling(tmp_path):
    recorder = SessionRecorder(tmp_path, sample_seconds=1.0)
    for t in (0.0, 0.5, 1.0, 0.0, 0.5, 1.0):
        recorder.row(amr_row(t))
    assert recorder.summary()["sampled_rows"] == 4
    assert recorder.summary()["time_resets"] == 1


def test_audit_mode_is_complete_and_unrotated(tmp_path):
    recorder = SessionRecorder(tmp_path, mode="audit", segment_bytes=1024, segments=1)
    for step in range(300):
        recorder.row(amr_row(step * 0.05, pose_x=step * 0.01))
        recorder.command({"action": "heartbeat", "step": step})
    recorder.event("note", detail="x" * 2000)
    summary = recorder.finish(status="ok")
    assert len(lines(tmp_path / "rows.jsonl")) == 300
    assert len(lines(tmp_path / "commands.jsonl")) == 300
    assert [e["type"] for e in lines(tmp_path / "events.jsonl")] == ["note", "session_end"]
    assert not list(tmp_path.glob("*.1.jsonl"))
    assert not (tmp_path / "telemetry.jsonl").exists()
    assert summary["complete_rows"] and summary["heartbeat_commands"] == 300
    assert summary["audit_records"] == {"rows.jsonl": 300, "commands.jsonl": 300, "events.jsonl": 2}


def test_invalid_configuration_rejected(tmp_path):
    with pytest.raises(ValueError):
        SessionRecorder(tmp_path, mode="everything")
    with pytest.raises(ValueError):
        SessionRecorder(tmp_path, sample_seconds=0)
    with pytest.raises(ValueError):
        RotatingJsonl(tmp_path / "x.jsonl", segment_bytes=10)


def test_finish_recording_marks_failed_on_finalization_error(tmp_path):
    recorder = SessionRecorder(tmp_path)
    manifest = {"status": "ok"}
    finish_recording(recorder, manifest)
    assert manifest["status"] == "ok" and manifest["recording"]["finished"]

    broken = SessionRecorder(tmp_path / "b")
    manifest = {"status": float("nan")}  # not JSON-encodable with allow_nan=False
    finish_recording(broken, manifest)
    assert manifest["status"] == "failed"
    assert "finalization_error" in manifest["recording"]
    assert manifest["error"].startswith("Recording finalization failed")


def test_reloc_transitions_flags_wrong_relocalisations():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))
    from common import reloc_transitions

    def row(t, status, err=None, herr=None):
        return {"t": t, "status": status, "err": err, "herr": herr}

    rows = [row(0.0, "relocalizing"), row(0.1, "tracking", 0.03, 0.01), row(0.2, "tracking", 0.03, 0.01),
            row(0.3, "lost"), row(0.4, "relocalizing"), row(0.5, "tracking", 0.82, 0.38),
            row(0.6, "lost"), row(0.7, "tracking", 0.05, 0.2)]
    out = reloc_transitions(rows)
    assert [t["t"] for t in out] == [0.1, 0.5, 0.7]
    assert [t["false"] for t in out] == [False, True, True]  # 0.82 m; then 0.2 rad > 8.6 deg
