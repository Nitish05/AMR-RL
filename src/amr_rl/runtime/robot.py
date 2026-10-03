"""The robot's runtime: perception -> memory -> decisions -> supervised motion.

``RobotRuntime`` receives only (a) onboard camera frames, (b) a drive backend
implementing the command contract, and (c) operator commands. It never receives
simulator objects, ground truth, depth, segmentation or third-person images.
Its outputs are drive requests (through the supervisor) and a screen image.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..behavior.activities import Survey
from ..behavior.chooser import ActivityChooser, ChooserConfig, Motivation, MotivationConfig
from ..behavior.frontier import frontier_goals
from ..control.supervisor import Supervisor, SupervisorConfig
from ..evidence.images import ImageEvidenceArchive
from ..expression.policy import expression_from
from ..expression.screen import render as render_face
from ..learning.identity import EntityTracker
from ..learning.memory import ExperienceMemory, LearningConfig
from ..mapping.occupancy import OCCUPIED, FloorEvidenceMapper, OccupancyGrid
from ..mapping.render import render_map
from ..navigation.navigator import Navigator
from ..navigation.near_field import DepthGuard
from ..navigation.planner import Planner, PlannerConfig
from ..perception.camera_model import CameraModel
from ..perception.entities import FixtureDetector
from ..perception.near_depth import MODEL_ID as DEPTH_MODEL_ID
from ..perception.near_depth import MODEL_REVISION as DEPTH_MODEL_REVISION
from ..perception.near_depth import LazyBackend as LazyDepthBackend
from ..perception.near_depth import MonoDepthObstacles
from ..perception.near_depth import model_available as depth_model_available
from ..perception.semantic import SemanticWorker
from ..perception.vslam import PREDICTED, TRACKING, PlanarVSLAM, VSLAMConfig


@dataclass
class RuntimeConfig:
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    motivation: MotivationConfig = field(default_factory=MotivationConfig)
    chooser: ChooserConfig = field(default_factory=ChooserConfig)
    vslam: VSLAMConfig = field(default_factory=VSLAMConfig)
    policy: str = "learned"
    start_clearance_attested: float | None = 0.32  # operator attestation (m); None = no assumption
    initial_survey: bool = True
    semantic: bool = True
    audit_images: bool = False
    seed: int = 0
    # Stop for objects placed on the route (navigation/near_field.py). Needs the optional
    # depth model in the local cache (scripts/amr.sh fetch-depth-model); without it the
    # robot runs without this guard and reports so in its state.
    near_field_guard: bool = True
    # Exploration: a full 360-degree look at frontier arrivals at least this far (m)
    # from earlier panoramas (0 = only the +-50 degree sweep). Measured off: the extra
    # in-place rotations drifted the map (arena seed 1 ATE 3.6 -> 20 cm; docs/VSLAM.md).
    explore_panorama_spacing: float = 0.0


class RobotRuntime:
    def __init__(self, spec, backend, *, memory_path, run_dir, config: RuntimeConfig | None = None,
                 wall_clock=None, map_dir=None):
        self.cfg = config or RuntimeConfig()
        self.spec = spec
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model = CameraModel.from_spec(spec)
        if map_dir is not None:
            self.slam = PlanarVSLAM.load(map_dir, self.model, self.cfg.vslam)
            try:
                self.grid = OccupancyGrid.load(Path(map_dir) / "occupancy.npz", self.slam.map_version)
            except FileNotFoundError:
                self.grid = OccupancyGrid(self.slam.map_version)
        else:
            self.slam = PlanarVSLAM(self.model, self.cfg.vslam)
            self.grid = OccupancyGrid(self.slam.map_version)
        self.mapper = FloorEvidenceMapper(self.model, self.grid)
        self.planner = Planner(PlannerConfig(footprint_radius=spec.footprint_radius))
        self.nav = Navigator(self.planner)
        self.guard = None
        self.guard_status = "disabled"
        if self.cfg.near_field_guard:
            if not depth_model_available():
                self.guard_status = "unavailable: depth model not in the local cache"
            else:
                self.guard = DepthGuard(MonoDepthObstacles(self.model, backend=LazyDepthBackend()), self.grid,
                                        self.planner.cfg.footprint_radius)
                self.guard_status = f"active ({DEPTH_MODEL_ID}@{DEPTH_MODEL_REVISION[:8]})"
        self.nav.guard = self.guard
        self.supervisor = Supervisor(backend, self.cfg.supervisor, wall_clock=wall_clock or time.monotonic)
        self.detector = FixtureDetector(self.model)
        self.memory = ExperienceMemory(memory_path, config=self.cfg.learning)
        self.memory.invalidate_positions(self.slam.map_version)
        self.tracker = EntityTracker(self.memory)
        self.semantic = SemanticWorker() if self.cfg.semantic else None
        self.archive = ImageEvidenceArchive(self.run_dir, audit=self.cfg.audit_images)
        self.motivation = Motivation(self.cfg.motivation)
        self.chooser = ActivityChooser(self.cfg.chooser, policy=self.cfg.policy,
                                       rng=np.random.default_rng(self.cfg.seed))
        self.activity = None
        self.pose = None
        self.sigma = None
        self.now = 0.0
        self.trajectory = []
        self.last_frame = None
        self.last_frame_record = None
        self.last_track = None
        self.pending_commands = []
        self.last_command = (0.0, 0.0)
        self.loc_status = "initializing"
        self.last_good_pose = None
        self.recovery_plan = None
        self._interrupted = {}
        self._entity_marked = {}
        self._keepout_time = -1e9
        self._keepout_target = None
        self.acks = []
        self.triggers = []
        self.recent_outcomes = []
        self.learning_updates = []
        self.interaction_log = []
        self.decision_log = []
        self.nav_log = []
        self._unreachable = {}
        self._investigated = set()
        self._frontier_visits = {}
        self._look_spots = []  # where exploration sweeps already happened (this session)
        self._panorama_spots = []  # where full 360-degree looks happened (this session)
        self.map_progress = 100.0  # initial exploration progress prior (cells/trip)
        self._frontier_cache = (None, [])
        self._labelled = {}
        self._survey_done = not self.cfg.initial_survey
        self.screen = None
        self.expression = None
        self.last_state = {}
        self._attested = False
        self._last_visible = set()

    # ================================================================ inputs
    def command(self, command: dict):
        """Queue an operator command; processed as a batch on the next tick (Stop wins)."""
        self.pending_commands.append(dict(command))

    def on_frame(self, frame):
        self.now = frame.timestamp
        self.supervisor.observe_frame(frame.timestamp)
        self.last_frame = frame
        self.last_frame_record = self.archive.capture(frame.rgb, frame_index=frame.index, timestamp=frame.timestamp)
        result = self.slam.track(frame.rgb, frame.timestamp, commanded=self.last_command)
        self.last_track = result
        status = result.status
        self.supervisor.observe_localization(status)
        if status == TRACKING and result.pose is not None:
            if not self._attested and self.cfg.start_clearance_attested and len(self.slam.keyframes) == 1:
                # Explicit operator attestation that the start area is clear (recorded).
                self.grid.mark_free_disc(result.pose[:2], self.cfg.start_clearance_attested)
                self._attested = True
            self.pose, self.sigma = result.pose, result.position_sigma
            if not self.trajectory or np.hypot(*(np.asarray(self.trajectory[-1]) - self.pose[:2])) > 0.02:
                self.trajectory.append([float(self.pose[0]), float(self.pose[1])])
                self.trajectory = self.trajectory[-4000:]
            self.grid.mark_footprint(self.pose, self.spec.chassis["length"], self.spec.overall_width,
                                     self.spec.chassis["length"] / 2 + self.spec.chassis["axle_offset_x"])
            if result.keyframe and len(self.slam.keyframes) >= 2:
                self.mapper.update_keyframe(self.slam.keyframes)
                # Well-established triangulated points 10-45 cm high mark obstacles,
                # each landmark at most once.
                lm = self.slam.lm
                idx = np.flatnonzero(lm.alive & (lm.kind == 1) & (lm.found >= 6) & ~lm.marked)
                if len(idx):
                    lm.marked[idx] = True
                    self.mapper.add_landmark_obstacles(lm.pos[idx], low=0.10)
                self._frontier_cache = (None, [])
            if self.guard is not None and self.nav.status == "following":
                self.guard.maybe_detect(frame.timestamp, self.pose, frame.rgb)
        elif status == PREDICTED and result.pose is not None:
            # Bounded dead reckoning: usable only by bounded primitives, never for mapping.
            self.pose, self.sigma = result.pose, result.position_sigma
        else:
            self.pose, self.sigma = None, None
        self.loc_status = status
        if status == TRACKING:
            self.last_good_pose = None if result.pose is None else result.pose.copy()
        if self.supervisor.recovery is None:
            self.recovery_plan = None
        detections = self.detector.detect(frame.rgb, self.pose if status == TRACKING else None)
        if self.pose is not None and status == TRACKING:
            n_merges = len(self.tracker.merge_log)
            visible = self.tracker.update(detections, self.pose, self.sigma, self.slam.map_version, self.now,
                                          in_view=self.place_in_view)
            if len(self.tracker.merge_log) != n_merges and self.activity is not None and \
                    self.activity.target_entity is not None:
                # a duplicate identity was folded into an older one: follow it
                self.activity.target_entity = self.memory.resolve(self.activity.target_entity)
            for item in visible:
                det = item.detection
                key = item.entity_id or "tentative"
                if (det.position is not None and det.width_m and det.range_m < 2.0
                        and self.now - self._entity_marked.get(key, -1.0) >= 1.0):
                    self._entity_marked[key] = self.now
                    self._mark_entity(det)
            self._semantic_requests(frame, visible)
            self._update_keepout()
            ids = {v.entity_id for v in visible if v.entity_id}
            if ids - self._last_visible:
                self.triggers.append("new_entity_evidence")
            if any(v.ambiguous for v in visible):
                self.triggers.append("ambiguous_identity")
            self._last_visible = ids
        else:
            self.tracker.visible = []

    def _update_keepout(self):
        """Remembered entities in this map are hard obstacles for planning: a disc of
        half their measured size plus their position uncertainty (engineered margin)."""
        target = self.activity.target_entity if self.activity is not None else None
        if self.now - self._keepout_time < 1.0 and target == self._keepout_target:
            return
        self._keepout_time, self._keepout_target = self.now, target
        discs = []
        for ent in self.known_entities():
            if ent.get("x") is None or ent.get("map_version") != self.slam.map_version:
                continue
            app = ent.get("appearance") or {}
            size = max(app.get("width_m") or 0.1, app.get("height_m") or 0.1)
            if ent["entity_id"] == target:
                # the object being approached: its body only, so standoffs stay reachable
                r = float(np.clip(size / 2, 0.05, 0.15))
            else:
                r = float(np.clip(size / 2, 0.05, 0.2) + np.clip(ent.get("pos_sigma") or 0.05, 0.02, 0.1))
            discs.append((float(ent["x"]), float(ent["y"]), r))
        self.grid.set_keepout(discs)

    def _mark_entity(self, det):
        r = float(np.clip(min(det.width_m, det.height_m or det.width_m) / 2, 0.05, 0.15))
        c = np.asarray(det.position)
        angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        ring = c + r * np.stack([np.cos(angles), np.sin(angles)], 1)
        self.grid.add_hits(np.vstack([c[None], ring]), self.grid.cfg.occ_hit * 0.6)

    def _semantic_requests(self, frame, visible):
        if self.semantic is None:
            return
        for item in visible:
            if item.entity_id is None or self._labelled.get(item.entity_id, 0) >= 2:
                continue
            det = item.detection
            x0, y0, x1, y1 = det.bbox
            crop = frame.rgb[max(0, y0 - 12):y1 + 4, max(0, x0 - 6):x1 + 6]
            if crop.size and self.semantic.submit(item.entity_id, crop, frame.timestamp, self.supervisor.generation):
                self._labelled[item.entity_id] = self._labelled.get(item.entity_id, 0) + 1

    # ================================================================ helpers for activities
    def known_entities(self):
        return self.tracker._load().values()

    def entity_appearance(self, entity_id):
        ent = self.tracker._load().get(entity_id)
        return None if ent is None else ent["appearance"]

    def place_in_view(self, xy):
        if self.pose is None:
            return False
        P = np.array([[xy[0], xy[1], 0.05]])
        uv, z = self.model.project_world(P, self.pose)
        return bool(z[0] > 0.2 and z[0] < 2.5 and 5 < uv[0, 0] < self.model.width - 5
                    and 5 < uv[0, 1] < self.model.height - 5)

    def avoid_regions(self):
        regions = []
        for ent in self.known_entities():
            if ent.get("x") is None or ent.get("map_version") != self.slam.map_version:
                continue
            if self.memory.attitude(ent["entity_id"], self.now)["attitude"] == "disliked":
                regions.append(((ent["x"], ent["y"]), 0.9))
        return regions

    def note_unreachable(self, entity_id, now, missing=False):
        count = self._unreachable.get(entity_id, (0, 0.0))[0] + 1
        self._unreachable[entity_id] = (count, now + self.cfg.chooser.unreachable_backoff * min(4, count))
        self.nav_log.append({"t": now, "entity_id": entity_id, "event": "missing" if missing else "unreachable"})

    def is_unreachable(self, entity_id, now):
        item = self._unreachable.get(entity_id)
        return item is not None and now < item[1]

    def note_investigated(self, entity_id):
        self._investigated.add(entity_id)

    def was_investigated(self, entity_id):
        return entity_id in self._investigated

    def exhausted_frontiers(self):
        return {k for k, v in self._frontier_visits.items() if v >= 2}

    def wants_panorama(self):
        spacing = self.cfg.explore_panorama_spacing
        if spacing <= 0 or self.pose is None:
            return False
        here = np.asarray(self.pose[:2], float)
        return all(np.linalg.norm(here - p) >= spacing for p in self._panorama_spots)

    def note_panorama(self):
        if self.pose is not None:
            self._panorama_spots.append(np.asarray(self.pose[:2], float).copy())

    def note_frontier_visit(self, key):
        self._frontier_visits[key] = self._frontier_visits.get(key, 0) + 1
        if self.pose is not None:
            self._look_spots.append(np.asarray(self.pose[:2], float).copy())
            self._look_spots = self._look_spots[-200:]

    def note_map_progress(self, new_free_cells):
        """Learning-progress signal for exploration: map cells gained per trip (EMA)."""
        self.map_progress = 0.6 * self.map_progress + 0.4 * max(0, new_free_cells)

    def note_frontier_failure(self, key):
        self._frontier_visits[key] = self._frontier_visits.get(key, 0) + 1

    def frontiers(self):
        rev = (self.grid.revision // 25, len(self._frontier_visits))
        if self._frontier_cache[0] != rev and self.pose is not None:
            self._frontier_cache = (rev, frontier_goals(self.grid, self.planner, self.pose,
                                                        exhausted=self.exhausted_frontiers(),
                                                        keep_out=self.avoid_regions(),
                                                        visited=self._look_spots))
        return self._frontier_cache[1]

    # ================================================================ control tick
    def tick(self, now):
        self.now = now
        if self.pending_commands:
            batch, self.pending_commands = self.pending_commands, []
            for cmd, ack in zip(batch, self.supervisor.handle_batch(batch, now=now)):
                self.acks.append({"command": cmd.get("action"), **ack.to_dict(), "t": now})
                if cmd.get("action") == "goal" and ack.reason == "unknown_action":
                    self.acks[-1] = self._operator_goal(cmd, now)
                if cmd.get("action") == "reset_memory":
                    self.acks[-1] = self._reset_memory(cmd, now)
                self.acks[-1]["_cid"] = cmd.get("_cid")
            self.acks = self.acks[-50:]
        revoked = self.supervisor.check(now=now)
        if revoked:
            if self.supervisor.recovery is not None:
                self.recovery_plan = self._plan_recovery()  # needs the interrupted activity
            self._cancel_activity(revoked)
        if self.semantic is not None:
            known = {e["entity_id"] for e in self.known_entities()}
            for result in self.semantic.collect(now=now, generation=self.supervisor.generation, known_entities=known):
                self.memory.set_label(result.job.entity_id, result.label)
                self.tracker._load()[result.job.entity_id]["label"] = result.label
        self.motivation.update(now)
        v = w = 0.0
        if self.supervisor.recovery is not None:
            if self.recovery_plan is None:
                self.recovery_plan = self._plan_recovery()
            if self.activity is not None:
                self._cancel_activity("localization_lost")
            v, w = self._recovery_step(now)
            if v or w:
                ack = self.supervisor.drive(v, w, now=now, generation=self.supervisor.generation, source="recovery")
                if not ack.accepted:
                    v = w = 0.0
        elif self.supervisor.autonomy_enabled and self.pose is not None:
            v, w = self._autonomy(now)
            ack = self.supervisor.drive(v, w, now=now, generation=self.supervisor.generation)
            if not ack.accepted:
                self._cancel_activity(ack.reason)
                v = w = 0.0
        elif now < self.supervisor.manual_until:
            # Operator driving: the wheels execute the operator's command, and the VSLAM
            # must know the robot's own motion (it chains relocalisation candidates
            # through it). Still the robot's own actuation, not an external sensor.
            v, w = self.supervisor.manual_command
        self.last_command = (v, w)
        self._update_expression()
        return self.screen, bool(self.expression.signal_pattern)

    def _plan_recovery(self):
        """Choose a bounded recovery that only moves through space known to be clear:
        retrace a nudge's straight approach, or rotate in place if the last certified
        pose had full footprint clearance; otherwise wait for the operator."""
        act = self.activity
        pose = self.last_good_pose
        if act is not None and getattr(act, "action", None) == "nudge" and act.phase in ("acting", "retreating"):
            start = act.act_start_pose if act.act_start_pose is not None else pose
            if pose is not None and start is not None:
                dist = float(np.hypot(*(np.asarray(pose[:2]) - np.asarray(start[:2]))))
                return {"kind": "retrace", "remaining": min(dist + 0.02, 0.45)}
        if pose is not None and self.planner.traversable_xy(self.grid, np.asarray(pose[:2])[None])[0]:
            return {"kind": "rotate"}
        return {"kind": "wait"}

    def _recovery_step(self, now):
        plan = self.recovery_plan
        if plan["kind"] == "retrace":
            r = self.supervisor.recovery
            if r.reverse < plan["remaining"]:
                return -0.08, 0.0
            plan["kind"] = "rotate" if self.last_good_pose is not None and self.planner.traversable_xy(
                self.grid, np.asarray(self.last_good_pose[:2])[None])[0] else "wait"
            return 0.0, 0.0
        if plan["kind"] == "rotate":
            return 0.0, 0.35
        return 0.0, 0.0

    def _operator_goal(self, cmd, now):
        gen = cmd.get("generation")
        if gen != self.supervisor.generation:
            return {"command": "goal", "accepted": False, "reason": "stale_generation",
                    "generation": self.supervisor.generation, "t": now}
        if not self.supervisor.autonomy_enabled or self.pose is None:
            return {"command": "goal", "accepted": False, "reason": "autonomy_not_enabled_or_not_localized",
                    "generation": self.supervisor.generation, "t": now}
        from ..behavior.activities import Activity, Goto

        class OperatorGoal(Activity):
            name = "manual"

            def __init__(self, goal):
                super().__init__(None, "operator goal")
                self.goto = Goto(goal)
                self.preemptible = False

            def step(self, rt, t):
                self.phase = "navigating to operator goal"
                vv, ww = self.goto.step(rt, t)
                if self.goto.status in ("arrived", "rejected", "failed"):
                    self.finish(self.goto.status, reason=self.goto.reason)
                return vv, ww

        goal = np.array([float(cmd["x"]), float(cmd["y"])])
        try:
            self.planner.check_goal(self.grid, goal)
        except ValueError as error:
            self.nav.rejections += 1
            self.nav_log.append({"t": now, "event": "goal_rejected", "goal": goal.tolist(),
                                 "reason": getattr(error, "reason", str(error))})
            return {"command": "goal", "accepted": False, "reason": getattr(error, "reason", str(error)),
                    "generation": self.supervisor.generation, "t": now}
        self._cancel_activity("operator_goal")
        self.activity = OperatorGoal(goal)
        self.activity.since = now
        return {"command": "goal", "accepted": True, "reason": "goal_accepted",
                "generation": self.supervisor.generation, "t": now}

    def _reset_memory(self, cmd, now):
        try:
            self.memory.reset(cmd.get("confirm", ""))
        except ValueError as error:
            return {"command": "reset_memory", "accepted": False, "reason": str(error),
                    "generation": self.supervisor.generation, "t": now}
        self.supervisor.stop("disabled")
        self._cancel_activity("memory_reset")
        self.tracker = EntityTracker(self.memory)
        self.recent_outcomes.clear()
        self.learning_updates.clear()
        self.motivation = Motivation(self.cfg.motivation)
        return {"command": "reset_memory", "accepted": True, "reason": "memory_reset",
                "generation": self.supervisor.generation, "t": now}

    def _cancel_activity(self, reason):
        if self.activity is not None and not self.activity.done:
            self.activity.cancel(reason)
            self._log_activity_end(self.activity)
        self.activity = None
        self.nav.cancel(reason)

    def option_interrupted(self, entity_id, action):
        return self._interrupted.get((entity_id, action), 0) >= 2

    def _log_activity_end(self, act):
        if getattr(act, "_logged", False):
            return
        act._logged = True
        if act.name == "idle":
            return  # bounded rests end every few seconds; the decision log records them
        if act.name in ("engage", "revisit") and (act.result or {}).get("reason") in (
                "localization_lost", "stale_observation"):
            key = (act.target_entity, getattr(act, "action", None))
            self._interrupted[key] = self._interrupted.get(key, 0) + 1
        entry = {"t": self.now, "started": act.since, "activity": act.name, "entity_id": act.target_entity,
                 **(act.result or {})}
        if getattr(act, "action", None):
            entry["action"] = act.action
        if getattr(act, "target_xy", None) is not None:
            entry["target_xy"] = [float(v) for v in act.target_xy]
        if getattr(act, "context", None):
            entry["context"] = act.context
        if getattr(act, "decision", None):
            entry["decision"] = act.decision
        self.interaction_log.append(entry)

    def _autonomy(self, now):
        if not self._survey_done:
            if self.activity is None:
                self.activity = Survey(angle=2 * math.pi, rate=0.45, reason="initial look around")
                self.activity.since = now
            v, w = self.activity.step(self, now)
            if self.activity.done:
                self._survey_done = True
                self._log_activity_end(self.activity)
                self.activity = None
            return v, w
        trigger = self.triggers[-1] if self.triggers else None
        self.triggers.clear()
        new = self.chooser.decide(self, now, self.activity, trigger=trigger)
        if new is not None:
            self.decision_log.append({"t": now, "trigger": trigger, "chosen": self.chooser.chosen,
                                      "top": self.chooser.snapshot()["candidates"][:4]})
            self.decision_log = self.decision_log[-2000:]
            if self.activity is not None and not self.activity.done:
                self.activity.cancel(f"reconsidered ({trigger})")
                self._log_activity_end(self.activity)
            self.nav.reset_goal_state()
            self.activity = new
            self._update_keepout()  # the new target's own disc shrinks before planning
        if self.activity is None:
            return 0.0, 0.0
        v, w = self.activity.step(self, now)
        if self.activity.done:
            self._on_activity_done(self.activity, now)
            self.triggers.append("activity_finished")
            self.activity = None
        return v, w

    def _on_activity_done(self, act, now):
        if act.name in ("engage", "revisit") and act.result.get("status") == "outcome":
            receipt = act.receipt(self, now)
            frames = {b["frame_sha256"] for b in receipt["before"]} | {a["frame_sha256"] for a in receipt["after"]}
            try:
                self.archive.retain(receipt, sorted(frames))
                receipt["images_retained"] = True
                update = self.memory.record_outcome(receipt)
            except (OSError, ValueError) as error:
                act.result = {"status": "not_learned", "reason": f"evidence/memory failure: {error}"}
                self._log_activity_end(act)
                return
            valence = self.memory.cfg.valence.get(receipt["observed"], 0.0)
            self.motivation.on_outcome(act.target_entity, valence)
            self.tracker.refresh()
            record = {"event_id": receipt["event_id"], "entity_id": act.target_entity, "action": act.action,
                      "context": receipt["context"], "predicted": receipt["predicted"],
                      "observed": receipt["observed"], "valence": valence, "update": update, "timestamp": now,
                      "decision": receipt["decision"]}
            self.recent_outcomes.append(record)
            self.recent_outcomes = self.recent_outcomes[-20:]
            if update:
                self.learning_updates.append({"t": now, "entity_id": act.target_entity, **update})
                self.learning_updates = self.learning_updates[-20:]
            self.triggers.append("outcome_recorded")
        self._log_activity_end(act)

    # ================================================================ output
    def entity_summaries(self):
        visible = {v.entity_id: v for v in self.tracker.visible if v.entity_id}
        out = []
        for ent in self.known_entities():
            eid = ent["entity_id"]
            att = self.memory.attitude(eid, self.now)
            vis = visible.get(eid)
            placed = ent.get("x") is not None and ent.get("map_version") == self.slam.map_version
            out.append({
                "entity_id": eid, "label": ent.get("label") or ent["appearance"].get("color", "object"),
                "visible": vis is not None, "identity_confidence": None if vis is None else round(vis.confidence, 3),
                "ambiguous": False, "position": [ent["x"], ent["y"]] if placed else None,
                "attitude": att["attitude"], "expected_value": att["expected_value"],
                "uncertainty": att["uncertainty"], "interactions": att["interactions"],
                "best_action": att["best_action"], "movable": bool(ent.get("movable")),
                "state": self.tracker.last_state.get(eid),
            })
        for v in self.tracker.visible:
            if v.ambiguous:
                out.append({"entity_id": None, "label": f"{v.detection.color} object (ambiguous)", "visible": True,
                            "identity_confidence": 0.0, "ambiguous": True,
                            "position": None if v.detection.position is None else list(v.detection.position),
                            "attitude": "unknown", "expected_value": None, "uncertainty": None, "interactions": 0,
                            "reason": v.reason, "candidates": v.candidates})
        return out

    def snapshot(self):
        loc = self.slam.snapshot()
        loc["inliers"] = None if self.last_track is None else self.last_track.inliers
        act = self.activity.snapshot() if self.activity is not None else {
            "name": "stopped" if self.supervisor.stopped else "idle", "target_entity": None, "phase": "",
            "reason": self.supervisor.revoked_reason or "", "since": None}
        if self.supervisor.recovery is not None:
            act = {"name": "recovering", "target_entity": None, "phase": "rotating to relocalize",
                   "reason": "bounded recovery after localization loss", "since": None}
        state = {
            "schema": "amr_rl.state.v1",
            "sim_time": self.now,
            "timing": "lockstep_simulation",
            "authority": self.supervisor.snapshot(),
            "localization": loc,
            "camera": {"frame_index": None if self.last_frame is None else self.last_frame.index,
                       "timestamp": None if self.last_frame is None else self.last_frame.timestamp,
                       "age": None if self.last_frame is None else self.now - self.last_frame.timestamp,
                       "fresh": self.last_frame is not None
                       and self.now - self.last_frame.timestamp <= self.supervisor.config.max_observation_age},
            "activity": act,
            "decision": self.chooser.snapshot(),
            "entities": self.entity_summaries(),
            "interaction": self.activity.interaction() if hasattr(self.activity, "interaction") else None,
            "recent_outcomes": list(reversed(self.recent_outcomes[-8:])),
            "learning_updates": list(reversed(self.learning_updates[-8:])),
            "motivation": self.motivation.snapshot(),
            "navigation": {**self.nav.snapshot(), "near_field_guard": {
                "status": self.guard_status,
                "free_run_m": None if self.guard is None else self.guard.free_run,
                **({} if self.guard is None else self.guard.stats)}},
            "trajectory": self.trajectory[-600:],
            "memory": {"agent_id": self.memory.agent_id, "db": self.memory.path, **self.memory.counts()},
            "map": {**self.grid.counts(), "map_version": self.grid.map_version,
                    "start_clearance_attested": self.cfg.start_clearance_attested if self._attested else None},
            "semantic": None if self.semantic is None else {"backend": self.semantic.backend.name,
                                                            **self.semantic.stats},
            "acks": [{k: v for k, v in a.items() if not k.startswith("_")} for a in self.acks[-5:]],
        }
        return state

    def _update_expression(self):
        state = self.snapshot()
        self.expression = expression_from(state)
        state["expression"] = self.expression.to_dict()
        key = (self.expression.face, round(self.expression.gaze[0], 1), round(self.expression.gaze[1], 1),
               round(self.expression.openness, 2), round(self.expression.uncertainty, 2), self.expression.attitude,
               self.expression.signal_pattern, self.expression.caption)
        if key != getattr(self, "_face_key", None):
            self.screen = render_face(self.expression)
            self._face_key = key
        self.last_state = state

    def map_image(self):
        ents = [e for e in self.last_state.get("entities", []) if e.get("position")]
        img, meta = render_map(self.grid, trajectory=self.trajectory, pose=self.pose, sigma=self.sigma,
                               path=self.nav.path if self.nav.status == "following" else (),
                               goal=self.nav.goal, entities=ents, footprint_radius=self.spec.footprint_radius)
        return img, meta

    def save_map(self, directory):
        directory = Path(directory)
        meta = self.slam.save(directory)
        self.grid.save(directory / "occupancy.npz")
        return meta

    def close(self):
        self.supervisor.stop("disabled")
        if self.semantic is not None:
            self.semantic.close()
        self.memory.close()


def occupied_fraction(grid):
    return float((grid.classes() == OCCUPIED).mean())
