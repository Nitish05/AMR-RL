"""Lockstep simulation harness connecting SimWorld <-> RobotRuntime.

Physics pauses while perception/decision code runs (lockstep). Results are
simulation results in simulated time; they do not establish real-time
performance on hardware.

The harness is the only place where world and runtime meet: frames go from the
world's onboard camera to the runtime; drive requests go from the runtime's
supervisor to the world's wheel backend; the runtime's screen image goes to the
world's display. Ground truth is read only by the evaluator for scoring.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..evidence.recording import SessionRecorder
from ..robot.spec import RobotSpec
from ..runtime.robot import RobotRuntime, RuntimeConfig
from . import evaluator
from .world import SimWorld, load_world_config

CONTROL_PERIOD = 0.1


class Session:
    def __init__(self, world_name, *, run_dir, memory_path, config: RuntimeConfig | None = None,
                 consequences=None, seed=0, inspection=True, map_dir=None, world_overrides=None,
                 recording_mode="telemetry", map_origin=None):
        spec = RobotSpec.load()
        wc = load_world_config(world_name)
        if world_overrides:
            wc = {**wc, **world_overrides}
        self.world = SimWorld(wc, spec=spec, seed=seed, consequences=consequences, inspection=inspection)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = RobotRuntime(spec, self.world.backend, memory_path=memory_path, run_dir=self.run_dir,
                                    config=config, map_dir=map_dir)
        self.recorder = SessionRecorder(self.run_dir / "recording", mode=recording_mode)
        for _ in range(20):  # settle on the floor before the first frame
            self.world.step()
        self.world.backend.reset_clock()
        self.world.time = 0.0
        # Scoring only: the world pose of the map frame. A fresh map starts at the robot's
        # start pose; a loaded map's frame is where that map was started (map_origin).
        self.origin = np.asarray(map_origin, float) if map_origin is not None else evaluator.true_pose(self.world)
        self.truth = []
        self.contacts = []
        self.wall_start = time.time()
        self.steps_per_control = int(round(CONTROL_PERIOD / self.world.dt))
        self.blackouts = []

    @property
    def now(self):
        return self.world.time

    def enable_autonomy(self):
        self.runtime.command({"action": "enable_autonomy", "generation": self.runtime.supervisor.generation})

    def inject_blackout(self, start, end):
        """EVALUATION FAULT INJECTION: replace onboard frames with black images
        (lens covered) between sim times ``start`` and ``end``."""
        self.blackouts.append((start, end))

    def control_step(self):
        frame = self.world.capture(self.runtime.model.version)
        if any(a <= frame.timestamp < b for a, b in self.blackouts):
            frame.rgb = np.zeros_like(frame.rgb)
        truth_at_frame = evaluator.true_pose(self.world)  # scoring only, same instant as the image
        cmd = self.runtime.last_command  # what on_frame passes to the VSLAM (logged for turn analysis)
        self.runtime.on_frame(frame)
        screen, signal = self.runtime.tick(self.world.time)
        self.world.set_screen(screen, signal_pattern=signal)
        for _ in range(self.steps_per_control):
            self.world.step()
        self._score(truth_at_frame, frame.timestamp, cmd)
        state = self.runtime.last_state
        self.recorder.row({"sim_time": state["sim_time"], "generation": state["authority"]["generation"],
                           "activity": state["activity"], "localization": state["localization"],
                           "authority": state["authority"], "navigation": state["navigation"],
                           "interaction": state["interaction"], "camera": state["camera"]})

    def _score(self, truth_pose, frame_time, cmd=None):
        truth = evaluator.to_map_frame(self.origin, truth_pose)
        est = self.runtime.pose
        err = None if est is None else evaluator.pose_error(est, truth)
        self.truth.append({"t": frame_time, "gt": truth.tolist(),
                           "est": None if est is None else [float(v) for v in est],
                           "err": None if err is None else err[0], "herr": None if err is None else err[1],
                           "sigma": self.runtime.sigma, "status": self.runtime.slam.status,
                           "activity": None if self.runtime.activity is None else self.runtime.activity.name,
                           "cmd": None if cmd is None else [float(cmd[0]), float(cmd[1])],
                           **_track_fields(self.runtime.last_track)})
        touching = evaluator.robot_contacts(self.world)
        if touching:
            self.contacts.append({"t": self.world.time, "with": touching,
                                  "activity": None if self.runtime.activity is None else
                                  f"{self.runtime.activity.name}:{self.runtime.activity.phase}"})

    def run(self, duration, *, until=None, callback=None):
        end = self.world.time + duration
        while self.world.time < end - 1e-9:
            self.control_step()
            if callback is not None:
                callback(self)
            if until is not None and until(self):
                break

    def summary(self):
        errs = [r["err"] for r in self.truth if r["err"] is not None]
        tracking = sum(r["status"] == "tracking" for r in self.truth)
        contact_events = _episodes(self.contacts)
        return {
            "sim_seconds": self.world.time, "wall_seconds": time.time() - self.wall_start,
            "timing": "lockstep simulation",
            "frames": len(self.truth), "tracking_fraction": tracking / max(1, len(self.truth)),
            "ate_rmse": float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
            "max_position_error": max(errs) if errs else None,
            "contact_episodes": contact_events,
            "world_events": [e.__dict__ for e in self.world.events],
            "map_score": evaluator.score_map(self.world, self.runtime.grid, self.origin),
            "memory": self.runtime.memory.counts(),
            "supervisor_log": self.runtime.supervisor.log[-50:],
            "activities": self.runtime.interaction_log,
            "slam_frozen_events": self.runtime.slam.frozen_events,
            "slam_reloc_log": self.runtime.slam.reloc_log,
        }

    def save_summary(self, extra=None):
        data = {**self.summary(), **(extra or {})}
        (self.run_dir / "summary.json").write_text(json.dumps(data, indent=1, default=_json_default))
        with (self.run_dir / "trajectory_scoring.jsonl").open("w") as stream:
            for row in self.truth:
                stream.write(json.dumps(row, default=_json_default) + "\n")
        self.recorder.finish(status="complete")
        return data

    def close(self):
        self.runtime.close()


def _track_fields(track):
    """Operational tracking diagnostics stored next to the scoring truth (why a frame
    was lost or relocalised); empty before the first frame."""
    if track is None:
        return {}
    out = {"reason": track.reason, "inliers": int(track.inliers), "matched": int(track.matched),
           "keyframe": bool(track.keyframe), "heading_sigma": track.heading_sigma}
    info = {k: v for k, v in (track.info or {}).items() if k in ("turn", "turn_rot", "degraded")}
    if (track.info or {}).get("turn_closure"):
        info["turn_closure"] = True
    if info:
        out["info"] = info
    return out


def _episodes(contacts, gap=0.35):
    episodes = []
    for c in contacts:
        if episodes and c["t"] - episodes[-1]["end"] <= gap and episodes[-1]["with"] == c["with"]:
            episodes[-1]["end"] = c["t"]
        else:
            episodes.append({"start": c["t"], "end": c["t"], "with": c["with"], "activity": c["activity"]})
    return episodes


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value))
