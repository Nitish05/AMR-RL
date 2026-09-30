"""Bounded everyday telemetry, or explicitly complete independent-audit streams.

Telemetry mode is a diagnostic summary, never benchmark or audit evidence: rows
are sampled every ``sample_seconds`` of simulation time, state transitions are
logged, heartbeat commands are only counted, and each stream keeps a bounded
recent history (rotating segments) whose evictions and oversized records are
counted in ``recording.json`` after every write. Audit mode writes every row,
command (heartbeats included) and event, unsampled and unrotated.

``recording.json`` is replaced atomically (temp file + rename) but not fsynced;
a crash can lose the latest summary update, never leave a torn one.

Adapted from BB8-RL ``src/bb8_rl/session_recording.py`` (see docs/PROVENANCE.md).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA = "amr_rl.recording.v1"


def _encoded(record):
    return (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode()


class RotatingJsonl:
    """Append-only JSONL with a fixed number of size-bounded segments."""

    def __init__(self, path, *, segment_bytes=1_048_576, segments=4):
        if type(segment_bytes) is not int or segment_bytes < 1024:
            raise ValueError("Log segment size must be an integer of at least 1024 bytes")
        if type(segments) is not int or not 1 <= segments <= 16:
            raise ValueError("Log segment count must be between 1 and 16")
        self.path = Path(path)
        self.segment_bytes, self.segments = segment_bytes, segments
        self.sizes, self.counts = [0] * segments, [0] * segments
        self.written = self.evicted = self.oversized = 0

    def _path(self, index):
        return self.path if index == 0 else self.path.with_suffix(f".{index}.jsonl")

    def append(self, record):
        data = _encoded(record)
        if len(data) > self.segment_bytes:
            self.oversized += 1
            data = _encoded(
                {
                    "type": "oversized_record",
                    "original_bytes": len(data),
                    "record_type": str(record.get("type", "unknown"))[:64],
                    "detail_truncated": True,
                }
            )
        if self.sizes[0] + len(data) > self.segment_bytes:
            self.evicted += self.counts[-1]
            self._path(self.segments - 1).unlink(missing_ok=True)
            for index in range(self.segments - 1, 0, -1):
                if self._path(index - 1).exists():
                    self._path(index - 1).rename(self._path(index))
                self.sizes[index], self.counts[index] = (
                    self.sizes[index - 1],
                    self.counts[index - 1],
                )
            self.sizes[0] = self.counts[0] = 0
        with self.path.open("ab") as stream:
            stream.write(data)
        self.sizes[0] += len(data)
        self.counts[0] += 1
        self.written += 1

    def summary(self):
        return {
            "quota_bytes": self.segment_bytes * self.segments,
            "retained_bytes": sum(self.sizes),
            "records_written": self.written,
            "records_retained": sum(self.counts),
            "records_evicted": self.evicted,
            "oversized_records": self.oversized,
            "history_truncated": bool(self.evicted or self.oversized),
            "files_newest_first": [
                self._path(i).name for i in range(self.segments) if self.counts[i]
            ],
        }


def _pick(source, keys):
    return {key: source.get(key) for key in keys}


def summarize_row(row):
    """Compact AMR runtime row: identity, activity, estimate, authority, goal."""
    activity = row.get("activity") or {}
    localization = row.get("localization") or {}
    authority = row.get("authority") or {}
    navigation = row.get("navigation") or {}
    interaction = row.get("interaction")
    camera = row.get("camera") or {}
    generation = row.get("generation", authority.get("generation"))
    return {
        "sim_time": row["sim_time"],
        "generation": generation,
        "activity": _pick(activity, ("name", "target_entity", "phase")),
        "localization": _pick(localization, ("status", "pose", "position_sigma")),
        "authority": {
            "stopped": authority.get("stopped"),
            "autonomy_enabled": authority.get("autonomy_enabled", authority.get("enabled")),
            "manual_active": authority.get("manual_active"),
            "revoked_reason": authority.get("revoked_reason"),
        },
        "navigation": _pick(navigation, ("status", "goal")),
        "interaction": None
        if interaction is None
        else _pick(interaction, ("request_id", "entity_id", "action", "status")),
        "camera": _pick(camera, ("frame_index", "fresh")),
    }


def transition_key(summary):
    """Discrete state whose change is logged; noisy estimates are excluded."""
    interaction = summary["interaction"]
    return {
        "generation": summary["generation"],
        "activity": summary["activity"]["name"],
        "localization_status": summary["localization"]["status"],
        "stopped": summary["authority"]["stopped"],
        "autonomy_enabled": summary["authority"]["autonomy_enabled"],
        "revoked_reason": summary["authority"]["revoked_reason"],
        "navigation_status": summary["navigation"]["status"],
        "navigation_goal": summary["navigation"]["goal"],
        "interaction": None
        if interaction is None
        else _pick(interaction, ("request_id", "status")),
    }


class SessionRecorder:
    """Session log in ``directory``: bounded telemetry, or complete audit."""

    def __init__(
        self,
        directory,
        *,
        mode="telemetry",
        segment_bytes=1_048_576,
        segments=4,
        sample_seconds=1.0,
    ):
        if mode not in ("telemetry", "audit"):
            raise ValueError("Recording mode must be telemetry or audit")
        if not math.isfinite(sample_seconds) or sample_seconds <= 0:
            raise ValueError("Telemetry sampling interval must be finite and positive")
        self.directory, self.mode = Path(directory), mode
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sample_seconds = sample_seconds
        self.next_sample = -math.inf
        self.last_time = None
        self.last_transition = None
        self.rows_seen = self.sampled_rows = self.transitions = 0
        self.heartbeats = self.time_resets = 0
        self.audit_records = {"rows.jsonl": 0, "commands.jsonl": 0, "events.jsonl": 0}
        self.finished = False
        self.telemetry = RotatingJsonl(
            self.directory / "telemetry.jsonl", segment_bytes=segment_bytes, segments=segments
        )
        self.events = RotatingJsonl(
            self.directory / "events.jsonl", segment_bytes=segment_bytes, segments=segments
        )
        self._persist()

    def _full(self, filename, record):
        with (self.directory / filename).open("ab") as stream:
            stream.write(_encoded(record))
        self.audit_records[filename] += 1

    def command(self, record):
        if record.get("action") == "heartbeat":
            self.heartbeats += 1
        if self.mode == "audit":
            self._full("commands.jsonl", record)
        elif record.get("action") != "heartbeat":
            self.events.append({**record, "type": "command"})
            self._persist()

    def row(self, row):
        self.rows_seen += 1
        if self.mode == "audit":
            self._full("rows.jsonl", row)
            return
        summary = summarize_row(row)
        time = summary["sim_time"]
        if not math.isfinite(time):
            raise ValueError("Row sim_time must be finite")
        if self.last_time is not None and time < self.last_time:
            # Simulation reset: restart sampling from the new clock.
            self.time_resets += 1
            self.next_sample = -math.inf
        self.last_time = time
        fingerprint = _encoded(transition_key(summary))
        changed = False
        if fingerprint != self.last_transition:
            self.events.append({**summary, "type": "transition", "row_index": self.rows_seen})
            self.last_transition = fingerprint
            self.transitions += 1
            changed = True
        if time + 1e-9 >= self.next_sample:
            self.telemetry.append({**summary, "type": "sample", "row_index": self.rows_seen})
            self.next_sample = time + self.sample_seconds
            self.sampled_rows += 1
            changed = True
        if changed:
            self._persist()

    def event(self, kind, **detail):
        record = {**detail, "type": kind}
        if self.mode == "audit":
            self._full("events.jsonl", record)
        else:
            self.events.append(record)
        self._persist()

    def summary(self):
        audit = self.mode == "audit"
        result = {
            "schema": SCHEMA,
            "mode": self.mode,
            "complete_rows": audit,
            "finished": self.finished,
            "rows_seen": self.rows_seen,
            "heartbeat_commands": self.heartbeats,
            "scope": "Complete audit streams; no sampling or rotation"
            if audit
            else "Bounded diagnostic history; not audit or benchmark evidence",
        }
        if audit:
            result["audit_records"] = dict(self.audit_records)
        else:
            result.update(
                sample_seconds=self.sample_seconds,
                sampled_rows=self.sampled_rows,
                transitions=self.transitions,
                time_resets=self.time_resets,
                heartbeat_commands_summarized=self.heartbeats,
                telemetry=self.telemetry.summary(),
                events=self.events.summary(),
            )
        return result

    def _persist(self):
        path = self.directory / "recording.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.summary(), indent=2) + "\n")
        temporary.replace(path)

    def finish(self, **detail):
        self.finished = True
        try:
            self.event("session_end", **detail)
        except BaseException:
            self.finished = False
            raise
        return self.summary()


def finish_recording(recorder, manifest):
    """Keep a failed event stream from preventing the separate final manifest."""
    try:
        manifest["recording"] = recorder.finish(status=manifest["status"])
    except (OSError, TypeError, ValueError) as error:
        manifest["recording"] = recorder.summary()
        manifest["recording"]["finalization_error"] = f"{type(error).__name__}: {error}"
        manifest["status"] = "failed"
        manifest.setdefault("error", f"Recording finalization failed: {error}")
