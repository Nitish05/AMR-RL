"""The robot's runtime: perception -> memory -> decisions -> supervised motion.

``RobotRuntime`` receives only (a) onboard camera frames, (b) a drive backend
implementing the command contract, (c) operator commands and (d) raw IMU and
wheel-encoder samples (``on_proprio``; datasheet-modelled parts). It never receives
simulator objects, ground truth, depth, segmentation or third-person images.
Its outputs are drive requests (through the supervisor) and a screen image.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..behavior.activities import Survey
from ..behavior.character import CharacterConfig, config_for, effective
from ..behavior.chooser import ActivityChooser, ChooserConfig, Motivation, MotivationConfig
from ..behavior.frontier import frontier_goals
from ..control.supervisor import Supervisor, SupervisorConfig
from ..evidence.images import ImageEvidenceArchive
from ..expression.policy import expression_from
from ..expression.screen import render as render_face
from ..learning.identity import EntityTracker
from ..learning.memory import ExperienceMemory, LearningConfig
from ..mapping.occupancy import FREE as FREE_CLASS
from ..mapping.occupancy import OCCUPIED, FloorEvidenceMapper, OccupancyGrid
from ..mapping.render import render_map
from ..navigation.navigator import Navigator
from ..navigation.near_field import DepthGuard
from ..navigation.planner import Planner, PlannerConfig
from ..odometry.fusion import OdometryConfig, WheelInertialOdometry
from ..odometry.samples import ProprioBatch
from ..odometry.wheel_model import CommandModelOdometry
from ..perception.camera_model import CameraModel
from ..perception.entities import FixtureDetector
from ..perception.near_depth import MODEL_ID as DEPTH_MODEL_ID
from ..perception.near_depth import MODEL_REVISION as DEPTH_MODEL_REVISION
from ..perception.near_depth import LazyBackend as LazyDepthBackend
from ..perception.near_depth import MonoDepthObstacles
from ..perception.near_depth import model_available as depth_model_available
from ..perception.place_recognition import MEGALOC_CODE_COMMIT, MEGALOC_REPO, make_descriptor, place_model_available
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
    # Whole-image place descriptor for loop closure ("megaloc" or None). Optional:
    # loaded only from the local cache (scripts/amr.sh fetch-place-model); without it
    # the VSLAM runs without loop closure and the state says so.
    place_descriptor: str | None = "megaloc"
    # Object detector: "open_vocab" (OmDet-Turbo boxes + engineered floor-contact and flag
    # rules, perception/open_vocab.py; docs/results/textured-worlds.md) or "fixture" (the
    # engineered saturated-colour detector). The model comes from the local cache only
    # (scripts/amr.sh fetch-detector-model); without it the colour detector is used and
    # detector_status says so. Default since 2026-10-12 (owner decision): with the IMU
    # robot in the textured arena, learning worked only with this detector (restart 6/6,
    # reversal 6/6 against 2/4 and 2/6 with the colour detector).
    detector: str = "open_vocab"
    # Round 8 (docs/results/turn-drift.md): when the VSLAM flags a frame's heading as
    # doubtful (info["degraded"]: a gated or flagged in-place turn), do not mark the
    # grid or place entities from it; in-place alignment turns may continue through
    # bounded dead reckoning instead of freezing facing the wall. Off by default.
    respect_degraded_heading: bool = False
    detector_device: str = "auto"
    # Mapping sessions (explore_only) with loop closure: when an early keyframe's
    # position relative to the robot has become uncertain in the pose graph (sigma >=
    # revisit_sigma), drive back to its pose and heading so a loop can close (at most
    # once per revisit_interval). Engineered behaviour; 0 disables.
    revisit_sigma: float = 0.0  # 0.08 evaluated in round 6, not adopted (docs/results/loop-closure.md)
    revisit_interval: float = 120.0
    revisit_early_s: float = 60.0
    # Coverage pass (mapping sessions): once frontiers run out, or when ``coverage_mode``
    # is set, visit free-space lattice points (this spacing, m) farther than
    # coverage_min_kf_dist from every keyframe and make a full survey turn there, so
    # the map has keyframes to relocalise against everywhere. 0 disables.
    # Off (round 6): the survey turns drifted the heading by up to 28 deg while still
    # "tracking" (arena s4 map ATE 4.0 -> 29.1 cm; docs/results/relocalisation.md).
    coverage_lattice: float = 0.0
    coverage_min_kf_dist: float = 0.3
    # Round 9 (docs/results/heading-r9.md). Motion source for localisation:
    # "command" = the robot's own commanded motion only (the camera-only robot);
    # "imu_encoders" = gyro + wheel encoders (ST LSM6DSOX, Pololu 4754 encoders; see
    # assets/robot/amr_spec.yaml) through odometry/fusion.py, fused in the VSLAM; when
    # no samples arrive in a frame it falls back to the command model below;
    # "command_model" = the commands through the wheel response model (WS2 soft prior).
    # Default since 2026-10-06 (validation 4 passed every required pre-registered check,
    # docs/results/heading-r9.md); "command" reproduces the camera-only robot exactly.
    odometry: str = "imu_encoders"
    odometry_cfg: OdometryConfig = field(default_factory=OdometryConfig)
    # With the IMU: stand still this long after start-up before autonomy moves the
    # wheels, so the gyro's zero-rate offset (+-1 dps typ, LSM6DSOX) is measured first
    # (standard IMU boot calibration; operator driving is not held).
    imu_boot_still_s: float = 2.0
    # The offset is re-measured at every natural stop (and learned from vision on
    # established landmarks in between). Owner decision 2026-10-05: if the robot has not
    # stopped for imu_zupt_interval_s, it stops to recalibrate, but only if necessary:
    # when the offset's uncertainty has grown above imu_zupt_min_sigma_dps. An
    # exploring robot otherwise random-walks the offset (development run: 6 deg in
    # 390 s on the gyro). Engineered. 0 disables.
    imu_zupt_interval_s: float = 120.0
    imu_zupt_min_sigma_dps: float = 0.02
    # Owner decision 2026-10-05: also recalibrate, at most every imu_zupt_temp_min_s,
    # when the IMU chip has warmed or cooled this much since the last offset measurement
    # and the offset is uncertain (warm-up after power-on shifts it by tempco x dT).
    # Development (6 worlds): 1.0 degC gave 1.28 deg / 1.94 cm against 2.11 / 4.26 with
    # the 2-minute rule alone, with 1-3 recalibration stops in 7 minutes. 0 = off.
    imu_zupt_temp_c: float = 1.0
    imu_zupt_temp_min_s: float = 20.0
    imu_zupt_max_hold_s: float = 2.5
    imu_zupt_min_frames: int = 10  # frames of stand-still measurement per recalibration stop
    # Owner decision 2026-10-12: until the offset's temperature coefficient has been
    # measured (one recalibration at a chip temperature different from boot), the
    # temperature trigger fires at this smaller change, whatever the offset's own
    # uncertainty. Validation 5: with a slow warm-up the first stop came at the 2-minute
    # rule and 72 % of the median run's worst heading error had built up by then
    # (r9-followup.md 8). 0 = off.
    imu_zupt_temp_first_c: float = 0.5
    # Owner decision 2026-10-12: a blocked drive (VSLAMConfig.odo_blocked_mode "keep_pose")
    # is handled with the pose kept: the activity is cancelled, the robot backs straight
    # off this far at this speed (at most blocked_backoff_max_s), then chooses again.
    # Stalled wheels mark the cells just ahead occupied (a contact the camera did not
    # see); spinning wheels do not (a floor, not an object). More than
    # blocked_max_per_min events in a minute stop autonomy ("blocked_repeatedly") so the
    # operator decides.
    blocked_backoff_m: float = 0.10
    blocked_backoff_speed: float = 0.06
    blocked_backoff_max_s: float = 4.0
    blocked_max_per_min: int = 3
    blocked_repeat_radius_m: float = 0.3  # a second event this close in a minute marks the spot, whatever the kind
    # The character store (behavior/character.py; owner decisions 2026-10-12): engineered
    # trait values with bounds and provenance, persisted in the experience memory. Nothing
    # reads them yet.
    character: CharacterConfig = field(default_factory=CharacterConfig)
    # The owner's preset seeds a new character (behavior/character.PRESETS); an existing
    # store keeps its values. character_active applies the trait values to the chooser, the
    # navigator and the probe budgets at boot (character.effective); off until the
    # identifiability gate has been run (docs/research/PERSONALITY_RESEARCH_2.md section 1).
    character_preset: str = "default"
    character_active: bool = False
    # This robot's gyro calibration (scripts/calibrate_imu.py); used when its unit
    # serial matches the spec's. None = datasheet prior only (+-1 % sensitivity).
    imu_calibration: str | None = "configs/calibration/imu.yaml"
    imu_unit_serial: int | None = None  # the installed IMU unit (None: the spec's unit_serial)
    # Perception-aware turning (navigation/view_check.py): avoid sweeping the camera
    # across surfaces nearer than ~0.6 m when the turn is optional; back off first
    # when it is not. Engineered behaviour, off by default.
    view_aware_turns: bool = False
    # WS4: depth-model floor mask for the VSLAM's floor lifting (VSLAMConfig.depth_floor_mask
    # selects when); one depth inference per frame is shared with the near-field guard.
    # Device for that model: "cpu" (deterministic; tests and benches) | "mps" | "auto".
    depth_device: str = "cpu"


def apply_turn_preset(cfg, name):
    """Apply a named turn-handling preset (perception/vslam.py TURN_PRESETS) and the
    runtime's degraded-heading handling that goes with it. ``None``/"none": unchanged.
    The presets are camera-only turn handling (rounds 7-8, not adopted): they set
    ``odometry="command"``, since with the IMU they would run an unevaluated hybrid.
    Apply after the other runtime options."""
    if name in (None, "none"):
        return cfg
    from ..perception.vslam import TURN_PRESETS

    for key, value in TURN_PRESETS[name].items():
        setattr(cfg.vslam, key, value)
    cfg.respect_degraded_heading = True
    cfg.odometry = "command"
    return cfg


def choose_recovery(*, loss_reason, last_command, nudge_retrace, turning_clearance, required_clearance, certified,
                    blocked=False):
    """Bounded recovery after localisation was lost (pure; unit-tested).

    * a nudge in progress: retrace its straight approach;
    * vision disagreed with a forward command (``visual_motion_inconsistent_with_
      commands``), or the wheels show the robot blocked (``blocked``: commanded forward
      travel not made without slip, judged without vision): the robot may be pressing
      on something it cannot see, so back straight off by 10 cm along the path just
      driven and wait. Never rotate: round 5 rotated in place against a box for 18 s;
    * otherwise rotate in place only if the last good pose was certified free AND
      nothing occupied (map or guard-asserted) lies within the turning circle;
    * else wait for the operator.
    """
    if nudge_retrace is not None:
        return {"kind": "retrace", "remaining": min(nudge_retrace + 0.02, 0.45)}
    v = last_command[0] if last_command else 0.0
    if (loss_reason == "visual_motion_inconsistent_with_commands" or blocked) and v > 0.02:
        return {"kind": "back_off", "remaining": 0.10}
    if certified and turning_clearance is not None and turning_clearance >= required_clearance:
        return {"kind": "rotate"}
    return {"kind": "wait"}


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
        self.nav.view_aware_turns = self.cfg.view_aware_turns
        self.nav.view.cam_forward = float(self.model.T_base_cam[0, 3])
        self.guard = None
        self.guard_status = "disabled"
        if self.cfg.near_field_guard:
            if not depth_model_available():
                self.guard_status = "unavailable: depth model not in the local cache"
            else:
                self.depth = MonoDepthObstacles(self.model, backend=LazyDepthBackend(device=self.cfg.depth_device))
                self.guard = DepthGuard(self.depth, self.grid,
                                        self.planner.cfg.footprint_radius,
                                        camera_x=float(self.model.T_base_cam[0, 3]))
                self.guard_status = f"active ({DEPTH_MODEL_ID}@{DEPTH_MODEL_REVISION[:8]})"
        self.nav.guard = self.guard
        if self.cfg.vslam.depth_floor_mask != "off":
            if self.guard is None and depth_model_available():
                self.depth = MonoDepthObstacles(self.model, backend=LazyDepthBackend(device=self.cfg.depth_device))
            self.slam.floor_mask = getattr(self, "depth", None)
        self.place_status = "disabled"
        if self.cfg.place_descriptor:
            if not place_model_available():
                self.place_status = "unavailable: place model not in the local cache (no loop closure)"
            else:
                self.slam.place = make_descriptor(self.cfg.place_descriptor)
                self.place_status = f"active ({MEGALOC_REPO}@{MEGALOC_CODE_COMMIT[:8]}, {self.slam.place.device})"
        self.slam.on_loop_closure = self._on_loop_closure
        self.loop_closures = []
        self.supervisor = Supervisor(backend, self.cfg.supervisor, wall_clock=wall_clock or time.monotonic)
        self.detector = FixtureDetector(self.model)
        self.detector_status = "fixture (engineered colour detector)"
        if self.cfg.detector == "open_vocab":
            from ..perception.open_vocab import MODELS as OV_MODELS
            from ..perception.open_vocab import load_open_vocab

            ov = load_open_vocab(self.model, device=self.cfg.detector_device)
            if ov is None:
                self.detector_status = "fixture (open-vocabulary model unavailable in the local cache)"
            else:
                self.detector = ov
                repo, rev, _ = OV_MODELS[ov.backend]
                self.detector_status = f"open_vocab ({repo}@{rev[:8]}, {ov.device})"
        self.memory = ExperienceMemory(memory_path, config=self.cfg.learning)
        self.memory.invalidate_positions(self.slam.map_version)
        if self.cfg.character_preset != "default" and not self.cfg.character.values:
            self.cfg.character = config_for(self.cfg.character_preset)
        self.memory.ensure_character(self.cfg.character, t=0.0)
        self.memory.record_boot(0.0, "restart", map_version=self.slam.map_version, map_loaded=map_dir is not None,
                                **{k: v for k, v in self.memory.counts().items() if isinstance(v, int)})
        self.character_effective, self._character_base = None, None
        self.tracker = EntityTracker(self.memory)
        self.semantic = SemanticWorker() if self.cfg.semantic else None
        self.archive = ImageEvidenceArchive(self.run_dir, audit=self.cfg.audit_images)
        self.motivation = Motivation(self.cfg.motivation)
        self._apply_character()
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
        self._revisit_last = -np.inf
        self._revisited = set()
        self.coverage_mode = False  # set by a mapping session to finish with the coverage pass
        self._look_spots = []  # where exploration sweeps already happened (this session)
        self._panorama_spots = []  # where full 360-degree looks happened (this session)
        self.map_progress = 100.0  # initial exploration progress prior (cells/trip)
        self._frontier_cache = (None, [])
        self._labelled = {}
        self._survey_done = not self.cfg.initial_survey
        self.screen = None
        self.expression = None
        self.imu_calibration_status = "none"
        self.odo = None
        if self.cfg.odometry == "imu_encoders":
            self.odo = WheelInertialOdometry(spec, self.cfg.odometry_cfg, calibration=self._load_imu_calibration())
            if not self.imu_calibration_status.startswith("loaded"):
                # Not silent: an uncalibrated unit with a -1 % sensitivity error drifted 19 deg
                # in validation 4 (scripts/calibrate_imu.py calibrates a unit).
                self.supervisor.note("imu_calibration", status=self.imu_calibration_status)
                warnings.warn(f"IMU gyro calibration not used: {self.imu_calibration_status} "
                              "(run scripts/calibrate_imu.py for this unit)", stacklevel=2)
        self.cmd_odo = CommandModelOdometry(spec) if self.cfg.odometry in ("imu_encoders", "command_model") else None
        # Sensor liveness: time of the newest sample per source; state changes are logged.
        self._sample_t = {"imu": None, "encoders": None}
        self.sensor_state = {"imu": "waiting", "encoders": "waiting"}
        self.odo_fallbacks = 0
        self._zupt_hold_until = -math.inf
        self._zupt_hold_start = None
        self._boot_t0 = None  # first control tick: the boot hold is measured from it
        self.zupt_hold_log = []  # recalibration stops: start, end, trigger, frames measured, success
        self._zupt_frames_at_hold = None
        self.zupt_holds = self.zupt_hold_failures = 0
        self.blocked_events = []  # keep-pose blocked events: t, kind, pose, activity, back-off outcome
        self._blocked_plan = None
        self._proprio = ProprioBatch([], [])
        self.last_odometry = None
        self.last_state = {}
        self._attested = False
        self._last_visible = set()

    # ================================================================ inputs
    def command(self, command: dict):
        """Queue an operator command; processed as a batch on the next tick (Stop wins)."""
        self.pending_commands.append(dict(command))

    def _load_imu_calibration(self):
        if not self.cfg.imu_calibration:
            return None
        import yaml

        from ..robot.spec import PROJECT_ROOT

        path = Path(self.cfg.imu_calibration)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path.exists():
            self.imu_calibration_status = f"missing ({path.name}): datasheet prior"
            return None
        cal = yaml.safe_load(path.read_text())
        unit = self.cfg.imu_unit_serial if self.cfg.imu_unit_serial is not None else self.spec.imu.get("unit_serial", -2)
        if int(cal.get("unit_serial", -1)) != int(unit):
            self.imu_calibration_status = "unit serial mismatch: datasheet prior"
            return None
        self.imu_calibration_status = f"loaded ({path.name}, scale {cal['gyro_scale']:.5f})"
        return cal

    def on_proprio(self, batch):
        """Raw IMU and encoder samples received since the last call (driver FIFOs)."""
        self._proprio.imu.extend(batch.imu)
        self._proprio.encoders.extend(batch.encoders)
        for source, samples in (("imu", batch.imu), ("encoders", batch.encoders)):
            if samples:
                newest = max(s.t for s in samples)
                last = self._sample_t[source]
                self._sample_t[source] = newest if last is None else max(last, newest)

    SENSOR_STALE_S = 0.3  # no samples for this long: the source counts as lost (flagged, not acted on)

    def _check_sensors(self, now):
        if self.odo is None:
            return
        for source, last in self._sample_t.items():
            state = "waiting" if last is None else ("ok" if now - last <= self.SENSOR_STALE_S else "lost")
            if state != self.sensor_state[source] and not (state == "waiting"):
                self.supervisor.note("sensor", source=source, state=state,
                                     age_s=None if last is None else round(now - last, 3))
            self.sensor_state[source] = state

    def on_frame(self, frame):
        self.now = frame.timestamp
        self._check_sensors(frame.timestamp)
        self.supervisor.observe_frame(frame.timestamp)
        self.last_frame = frame
        self.last_frame_record = self.archive.capture(frame.rgb, frame_index=frame.index, timestamp=frame.timestamp)
        odo = None
        batch, self._proprio = self._proprio, ProprioBatch([], [])
        fallback = None if self.cmd_odo is None else self.cmd_odo.step(self.last_command, frame.timestamp)
        if self.odo is not None:
            first = self.odo.t is None
            odo = self.odo.step(batch, frame.timestamp, command=self.last_command)
            if odo is None and not first and fallback is not None:
                odo = fallback  # no IMU/encoder samples this frame: the command model
                self.odo_fallbacks += 1
        elif self.cfg.odometry == "command_model":
            odo = fallback
        self.last_odometry = odo
        result = self.slam.track(frame.rgb, frame.timestamp, commanded=self.last_command, odometry=odo)
        if self.odo is not None:
            for ratio, sigma in self.slam.pop_scale_samples():
                self.odo.add_scale_sample(ratio * self.odo.gyro_scale, sigma)
            for rate, sigma in self.slam.pop_bias_feedback():
                self.odo.add_bias_feedback(rate, sigma)
            for ratio in self.slam.pop_radius_samples():
                self.odo.add_radius_sample(ratio)
        self.last_track = result
        # map evidence from this frame on is measured relative to the newest keyframe
        self.grid.anchor = self.slam.keyframes[-1].id if self.slam.keyframes else -1
        status = result.status
        self.supervisor.observe_localization(status)
        if status == TRACKING and result.pose is not None:
            if not self._attested and self.cfg.start_clearance_attested and len(self.slam.keyframes) == 1:
                # Explicit operator attestation that the start area is clear (recorded).
                self.grid.mark_free_disc(result.pose[:2], self.cfg.start_clearance_attested)
                self._attested = True
            self.pose, self.sigma = result.pose, result.position_sigma
            degraded = self.cfg.respect_degraded_heading and bool((result.info or {}).get("degraded"))
            if not self.trajectory or np.hypot(*(np.asarray(self.trajectory[-1]) - self.pose[:2])) > 0.02:
                self.trajectory.append([float(self.pose[0]), float(self.pose[1])])
                self.trajectory = self.trajectory[-4000:]
            if not degraded:
                self.grid.mark_footprint(self.pose, self.spec.chassis["length"], self.spec.overall_width,
                                         self.spec.chassis["length"] / 2 + self.spec.chassis["axle_offset_x"])
            if result.keyframe and len(self.slam.keyframes) >= 2 and not degraded:
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
            elif self.guard is not None and self.cfg.view_aware_turns and not degraded:
                self.guard.maybe_detect(frame.timestamp, self.pose, frame.rgb, record_only=True)
        elif status == PREDICTED and result.pose is not None:
            # Bounded dead reckoning: usable only by bounded primitives, never for mapping.
            self.pose, self.sigma = result.pose, result.position_sigma
        else:
            self.pose, self.sigma = None, None
        self.loc_status = status
        if (result.info or {}).get("blocked") and self.supervisor.autonomy_enabled and self.supervisor.recovery is None:
            self._on_blocked(frame.timestamp)
        if status == TRACKING:
            self.last_good_pose = None if result.pose is None else result.pose.copy()
        if self.supervisor.recovery is None:
            self.recovery_plan = None
        heading_ok = not (self.cfg.respect_degraded_heading and status == TRACKING
                          and bool((result.info or {}).get("degraded")))
        detections = self.detector.detect(frame.rgb, self.pose if status == TRACKING and heading_ok else None)
        if self.pose is not None and status == TRACKING and heading_ok:
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

    def _on_loop_closure(self, corrections, keyframe_times):
        """The VSLAM corrected its keyframes (loop closure). Move everything placed in
        the map frame with the keyframe it was measured from: occupancy evidence
        (journal replay), the trajectory, look spots, and remembered entity positions
        (by the keyframe current when last seen). Entities did not move; the map
        frame under them was corrected."""
        from bisect import bisect_right

        from ..perception.pose_graph import apply_transform

        self.grid.replay(corrections)
        ids = sorted(keyframe_times)
        times = [keyframe_times[i] for i in ids]

        def at_time(t):
            k = bisect_right(times, t) - 1
            return corrections.get(ids[max(k, 0)]) if ids else None

        latest = corrections.get(ids[-1]) if ids else None
        if latest is not None:
            if self.trajectory:
                self.trajectory = apply_transform(latest, np.asarray(self.trajectory)).tolist()[-4000:]
            self._look_spots = [apply_transform(latest, p[None])[0] for p in self._look_spots]
            self._panorama_spots = [apply_transform(latest, p[None])[0] for p in self._panorama_spots]
            if self.last_good_pose is not None:
                from ..perception.pose_graph import apply_to_pose

                self.last_good_pose = apply_to_pose(latest, self.last_good_pose)
        moved = {}
        for ent in self.memory.entities():
            if ent.get("x") is None or ent.get("map_version") != self.slam.map_version:
                continue
            T = at_time(ent.get("last_seen") or 0.0)
            if T is not None:
                moved[ent["entity_id"]] = apply_transform(T, np.array([[ent["x"], ent["y"]]]))[0]
        if moved:
            self.memory.relocate_entities(moved)
            self.tracker.refresh()
        self._frontier_cache = (None, [])
        last = self.slam.loop_log[-1] if self.slam.loop_log else {}
        self.loop_closures.append({"t": self.now, "kf": last.get("kf"), "candidate": last.get("candidate"),
                                   "correction_m": last.get("correction_m"), "entities_moved": len(moved)})

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
        if isinstance(key, tuple) and key and key[0] == "revisit":
            self._revisited.add(key[1])
            self._revisit_last = self.now
        self._frontier_visits[key] = self._frontier_visits.get(key, 0) + 1
        if self.pose is not None:
            self._look_spots.append(np.asarray(self.pose[:2], float).copy())
            self._look_spots = self._look_spots[-200:]

    def note_map_progress(self, new_free_cells):
        """Learning-progress signal for exploration: map cells gained per trip (EMA)."""
        self.map_progress = 0.6 * self.map_progress + 0.4 * max(0, new_free_cells)

    def note_frontier_failure(self, key):
        if isinstance(key, tuple) and key and key[0] == "revisit":
            self._revisited.add(key[1])
            self._revisit_last = self.now
        self._frontier_visits[key] = self._frontier_visits.get(key, 0) + 1

    def frontiers(self):
        rev = (self.grid.revision // 25, len(self._frontier_visits))
        if self._frontier_cache[0] != rev and self.pose is not None:
            self._frontier_cache = (rev, frontier_goals(self.grid, self.planner, self.pose,
                                                        exhausted=self.exhausted_frontiers(),
                                                        keep_out=self.avoid_regions(),
                                                        visited=self._look_spots))
        revisit = self._revisit_goal()
        goals = self._frontier_cache[1]
        if self.cfg.policy == "explore_only" and self.cfg.coverage_lattice > 0 and (self.coverage_mode or not goals):
            goals = self._coverage_goals() or goals
        return ([revisit] + goals) if revisit else goals

    def _coverage_goals(self, max_goals=6):
        """Certified free lattice points far from every keyframe, nearest first."""
        if self.pose is None or not self.slam.keyframes:
            return []
        step = self.cfg.coverage_lattice
        free = np.argwhere(self.grid.classes() == FREE_CLASS)
        if len(free) == 0:
            return []
        xy = self.grid.to_xy(free[:, 1], free[:, 0])
        lattice = np.unique(np.round(xy / step) * step, axis=0)
        lattice = lattice[self.planner.traversable_xy(self.grid, lattice)]
        if len(lattice) == 0:
            return []
        kf = np.array([k.pose[:2] for k in self.slam.keyframes])
        far = np.min(np.linalg.norm(lattice[:, None, :] - kf[None], axis=2), axis=1) > self.cfg.coverage_min_kf_dist
        exhausted = self.exhausted_frontiers()
        out = []
        for p in lattice[far][np.argsort(np.linalg.norm(lattice[far] - self.pose[:2], axis=1))]:
            key = ("cover", round(float(p[0]), 2), round(float(p[1]), 2))
            if key in exhausted or key in self._frontier_visits:
                continue
            heading = float(np.arctan2(p[1] - self.pose[1], p[0] - self.pose[0]))
            out.append((p.copy(), heading, 0, key))
            if len(out) >= max_goals:
                break
        return out

    def _revisit_goal(self):
        """An early keyframe pose to return to for loop closure (mapping sessions), or None."""
        cfg = self.cfg
        if (cfg.policy != "explore_only" or cfg.revisit_sigma <= 0 or self.slam.place is None or self.pose is None
                or self.now - self._revisit_last < cfg.revisit_interval or len(self.slam.keyframes) < 2):
            return None
        t0 = self.slam.keyframes[0].timestamp
        early = [k for k in self.slam.keyframes if k.timestamp - t0 <= cfg.revisit_early_s and k.gdesc is not None
                 and k.id not in self._revisited]
        if not early:
            return None
        sigma = self.slam.relative_sigma(self.slam.keyframes[-1].id)
        for k in sorted(early, key=lambda k: -sigma.get(k.id, 0.0)):
            if sigma.get(k.id, 0.0) < cfg.revisit_sigma:
                break
            if (np.hypot(*(k.pose[:2] - self.pose[:2])) >= 0.8
                    and self.planner.traversable_xy(self.grid, k.pose[:2][None])[0]):
                return (k.pose[:2].copy(), float(k.pose[2]), 0, ("revisit", k.id))
        return None

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
        if self._blocked_plan is not None and (revoked or not self.supervisor.autonomy_enabled):
            # an operator stop does not pass through _cancel_activity: close the back-off here
            self._end_backoff(now, interrupted=revoked or self.supervisor.revoked_reason or "stopped")
        if self.supervisor.recovery is not None:
            self._end_backoff(now, interrupted="localization_lost")
            if self.recovery_plan is None:
                self.recovery_plan = self._plan_recovery()
            if self.activity is not None:
                self._cancel_activity("localization_lost")
            v, w = self._recovery_step(now)
            if v or w:
                ack = self.supervisor.drive(v, w, now=now, generation=self.supervisor.generation, source="recovery")
                if not ack.accepted:
                    v = w = 0.0
        elif self.supervisor.autonomy_enabled and self.pose is not None and not self._imu_booting(now):
            v, w = self._blocked_step(now) if self._blocked_plan is not None else self._autonomy(now)
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

    def _imu_booting(self, now):
        """Hold the wheels for the IMU's zero-velocity offset measurement: at boot, and
        briefly whenever the last one is older than imu_zupt_interval_s."""
        if self.odo is None:
            return False
        if self._boot_t0 is None:
            self._boot_t0 = now
        if now - self._boot_t0 < self.cfg.imu_boot_still_s:
            if self._zupt_frames_at_hold is None:
                self._zupt_frames_at_hold = self.odo.zupt_frames
            return True
        if self.cfg.imu_zupt_interval_s <= 0:
            return False
        if self._zupt_hold_start is not None:
            # hold until the offset has actually been measured (the chassis may still
            # rock after a turn), at most imu_zupt_max_hold_s
            # ~1 s of still samples: one 0.1 s frame alone left a ~0.01 dps error
            # (gyro noise and the 8.75 mdps LSB; development run arena s0)
            done = self.odo.zupt_frames - self._zupt_frames_at_hold >= self.cfg.imu_zupt_min_frames
            if not done and now < self._zupt_hold_start + self.cfg.imu_zupt_max_hold_s:
                return True
            if not done:
                self.zupt_hold_failures += 1
            if self.zupt_hold_log:
                self.zupt_hold_log[-1].update({"end": float(now), "success": bool(done),
                                               "frames": int(self.odo.zupt_frames - self._zupt_frames_at_hold)})
            self._zupt_hold_start = None
            self._zupt_hold_until = now
            return False
        since = now - max(self.odo.last_bias_update_t, self._zupt_hold_until)
        stale = since > self.cfg.imu_zupt_interval_s
        uncertain = math.degrees(math.sqrt(self.odo.bias_var)) > self.cfg.imu_zupt_min_sigma_dps
        # Until the tempco is measured, a smaller temperature change triggers the stop,
        # whatever the offset's own uncertainty: the stop is what measures the tempco.
        first = self.cfg.imu_zupt_temp_first_c > 0 and not self.odo.tempco_known
        temp_c = self.cfg.imu_zupt_temp_c
        if first:
            temp_c = self.cfg.imu_zupt_temp_first_c if temp_c <= 0 else min(temp_c, self.cfg.imu_zupt_temp_first_c)
        warmed = (temp_c > 0 and since > self.cfg.imu_zupt_temp_min_s
                  and self.odo.temp is not None and self.odo.temp_at_update is not None
                  and abs(self.odo.temp - self.odo.temp_at_update) >= temp_c)
        if (stale and uncertain) or (warmed and (uncertain or first)):
            self._zupt_hold_start = now
            self._zupt_frames_at_hold = self.odo.zupt_frames
            self.zupt_holds += 1
            self.zupt_hold_log.append({"start": float(now),
                                       # labelled by the rule that fired it: the 1 degC rule needs an
                                       # uncertain offset AND the full change; otherwise the first-stop rule
                                       "trigger": "interval" if stale and uncertain else (
                                           "temperature" if uncertain and self.cfg.imu_zupt_temp_c > 0
                                           and abs(self.odo.temp - self.odo.temp_at_update) >= self.cfg.imu_zupt_temp_c
                                           else "temperature_first"),
                                       "tempco_known": bool(self.odo.tempco_known),
                                       "sigma_dps": math.degrees(math.sqrt(self.odo.bias_var)),
                                       "temp_change_c": None if self.odo.temp is None or self.odo.temp_at_update is None
                                       else float(self.odo.temp - self.odo.temp_at_update)})
            del self.zupt_hold_log[:-500]
            return True
        return False

    def _apply_character(self):
        """With character_active, set the parameters the trait values produce
        (character.effective) on the chooser config, the navigator and the memory's probe
        budgets. Engineered mapping; recorded in the state's character block."""
        if not self.cfg.character_active:
            self.character_effective = None
            return
        if self._character_base is None:  # the configured values, read once so re-applying never compounds
            self._character_base = {"chooser.min_interval": self.cfg.chooser.min_interval,
                                    "chooser.avoid_radius": self.cfg.chooser.avoid_radius,
                                    "chooser.margin": self.cfg.chooser.margin, "navigator.v_max": self.nav.cfg.v_max,
                                    "learning.initial_probes": self.memory.cfg.initial_probes,
                                    "learning.change_failures": self.memory.cfg.change_failures}
        values = {k: r["value"] for k, r in self.memory.character().items()}
        eff = effective(values)
        self.cfg.chooser.info_gain = eff["chooser.info_gain"]
        self.cfg.chooser.value_z = eff["chooser.value_z"]
        self.cfg.chooser.min_interval = self._character_base["chooser.min_interval"] * eff["chooser.min_interval_gain"]
        self.cfg.chooser.avoid_radius = self._character_base["chooser.avoid_radius"] * eff["chooser.avoid_radius_gain"]
        self.cfg.chooser.margin = self._character_base["chooser.margin"] * eff["chooser.margin_gain"]
        self.nav.cfg.v_max = self._character_base["navigator.v_max"] * eff["navigator.v_max_gain"]
        self.memory.cfg.initial_probes = eff["learning.initial_probes"]
        self.memory.cfg.change_failures = eff["learning.change_failures"]
        self.memory.cfg.surprise_ratio = eff["learning.surprise_ratio"]
        self.memory.cfg.curiosity_lp = eff["learning.curiosity_lp"]
        self.character_effective = {**eff, "values": values, "chooser.min_interval": self.cfg.chooser.min_interval,
                                    "chooser.avoid_radius": self.cfg.chooser.avoid_radius,
                                    "chooser.margin": self.cfg.chooser.margin, "navigator.v_max": self.nav.cfg.v_max}

    def _on_blocked(self, now):
        """The wheels were driven but vision and the encoders agree the robot did not
        move: the pose is right and something blocks the drive (owner decision
        2026-10-12: keep the pose). Record the event, cancel the activity, back straight
        off (``_blocked_step``), then choose again. Repeated events within a minute stop
        autonomy for the operator instead."""
        kind = self.slam.blocked_kind
        act = self.activity
        self.blocked_events.append({
            "t": float(now), "kind": kind, "status": self.loc_status,
            "pose": None if self.pose is None else [float(x) for x in self.pose],
            "activity": None if act is None else act.name, "action": getattr(act, "action", None)})
        del self.blocked_events[:-500]
        # The drive that was blocked: forward (+1) or reverse (-1); a turn in place has no
        # side to back away from. The VSLAM's check uses |v|, so both signs reach here.
        v_cmd = float(self.last_command[0]) if self.last_command else 0.0
        direction = 1 if v_cmd > 0.02 else -1 if v_cmd < -0.02 else 0
        self.blocked_events[-1]["drive"] = {1: "forward", -1: "reverse", 0: "turning"}[direction]
        recent = [e for e in self.blocked_events if now - e["t"] <= 60.0]
        if len(recent) > self.cfg.blocked_max_per_min:
            self._cancel_activity("blocked_repeatedly")  # also closes an open back-off
            self.supervisor.stop("blocked_repeatedly")
            self.blocked_events[-1]["stopped"] = True
            return
        # Mark the spot on the driven side when the wheels stalled (a contact), or when the
        # drive was stopped a second time within blocked_repeat_radius_m in a minute,
        # whatever the kind (a floor the robot cannot drive on; backing off and driving
        # in again repeats until the limit otherwise).
        near = [e for e in recent[:-1] if e.get("pose") is not None and self.pose is not None
                and np.hypot(e["pose"][0] - self.pose[0], e["pose"][1] - self.pose[1]) <= self.cfg.blocked_repeat_radius_m]
        if self.pose is not None and direction != 0 and (kind == "stalled" or near):
            self.blocked_events[-1]["cells_marked"] = self._mark_blocked_contact(self.pose, direction)
        if self._blocked_plan is not None:  # blocked again while backing off: keep the plan
            return
        self._cancel_activity("blocked")
        if direction == 0 or self.pose is None:
            self.blocked_events[-1]["backoff_skipped"] = "turning" if direction == 0 else "no pose"
            self.triggers.append("blocked")
            return
        # Back off only into space the map knows to be clear (as the navigator's own
        # back-off does); otherwise just choose again.
        behind = np.asarray(self.pose[:2], float) - direction * self.cfg.blocked_backoff_m * np.array(
            [math.cos(float(self.pose[2])), math.sin(float(self.pose[2]))])
        clearance = float(self.planner.obstacle_clearance_xy(self.grid, behind[None])[0])
        if clearance < self.planner.cfg.footprint_radius:
            self.blocked_events[-1]["backoff_skipped"] = f"no clearance ({clearance:.2f} m)"
            self.triggers.append("blocked")
            return
        self._blocked_plan = {"start": float(now), "dir": direction,
                              "start_pose": np.asarray(self.pose[:2], float).copy()}

    def _end_backoff(self, now, interrupted=None):
        """Close an open back-off (finished, or cut short by a revocation, an operator
        command, a loss of tracking or a repeat limit) and record its outcome."""
        plan, self._blocked_plan = self._blocked_plan, None
        if plan is None:
            return
        moved = 0.0 if plan["start_pose"] is None or self.pose is None else float(
            np.hypot(*(np.asarray(self.pose[:2], float) - plan["start_pose"])))
        if self.blocked_events:
            self.blocked_events[-1].update({
                "backed_off_m": round(moved, 3), "backoff_s": round(now - plan["start"], 2),
                "backoff_complete": bool(interrupted is None and moved >= self.cfg.blocked_backoff_m),
                **({"interrupted": interrupted} if interrupted else {})})
        if interrupted is None:
            self.triggers.append("blocked")

    def _mark_blocked_contact(self, pose, direction=1):
        """Stalled wheels: something the camera did not see is pressing on the driven
        side of the footprint. Raise the cells just beyond it, journalled like the depth
        guard's assertions, so planning routes around the spot. Returns cells changed."""
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        ahead = direction * (self.planner.cfg.footprint_radius + 0.04)
        side = np.linspace(-0.08, 0.08, 5)
        xy = np.array([[x + ahead * math.cos(th) - s * math.sin(th), y + ahead * math.sin(th) + s * math.cos(th)]
                       for s in side])
        return int(self.grid.raise_to(xy, 1.5))  # the guard's assert level: a confirmed obstacle

    def _blocked_step(self, now):
        """Back straight off along the path just driven (away from the blocked side)
        until blocked_backoff_m has been covered, measured by the kept pose, or
        blocked_backoff_max_s has passed; then hand back to the chooser."""
        plan = self._blocked_plan
        moved = 0.0 if plan["start_pose"] is None or self.pose is None else float(
            np.hypot(*(np.asarray(self.pose[:2], float) - plan["start_pose"])))
        if moved >= self.cfg.blocked_backoff_m or now - plan["start"] >= self.cfg.blocked_backoff_max_s:
            self._end_backoff(now)
            return 0.0, 0.0
        return -plan["dir"] * self.cfg.blocked_backoff_speed, 0.0

    def _plan_recovery(self):
        """Choose a bounded recovery that only moves through space known to be clear
        (see ``choose_recovery``)."""
        act = self.activity
        pose = self.last_good_pose
        retrace = None
        if act is not None and getattr(act, "action", None) == "nudge" and act.phase in ("acting", "retreating"):
            start = act.act_start_pose if act.act_start_pose is not None else pose
            if pose is not None and start is not None:
                retrace = float(np.hypot(*(np.asarray(pose[:2]) - np.asarray(start[:2]))))
        clearance = None if pose is None else float(self.planner.obstacle_clearance_xy(
            self.grid, np.asarray(pose[:2])[None])[0])
        return choose_recovery(
            loss_reason=None if self.last_track is None else self.last_track.reason,
            last_command=self.last_command, nudge_retrace=retrace,
            turning_clearance=clearance, required_clearance=self.planner.cfg.footprint_radius + self.nav.cfg.turn_margin,
            certified=pose is not None and bool(self.planner.traversable_xy(self.grid, np.asarray(pose[:2])[None])[0]),
            blocked=self.odo is not None and self.slam.cfg.odo_blocked_check
            and self.slam.blocked_by_odometry(self.now, window=0.5))

    def _recovery_step(self, now):
        plan = self.recovery_plan
        if plan["kind"] == "retrace":
            r = self.supervisor.recovery
            if r.reverse < plan["remaining"]:
                return -0.08, 0.0
            plan["kind"] = "rotate" if self.last_good_pose is not None and self.planner.traversable_xy(
                self.grid, np.asarray(self.last_good_pose[:2])[None])[0] else "wait"
            return 0.0, 0.0
        if plan["kind"] == "back_off":  # suspected contact: straight back the way it came, then wait
            if self.supervisor.recovery.reverse < plan["remaining"]:
                return -0.06, 0.0
            plan["kind"] = "wait"
            return 0.0, 0.0
        if plan["kind"] == "rotate":
            # With the gyro: a turn that does not happen (pressed against something) is
            # stopped rather than continued (round 5: 18 s rotating against a box).
            step = self.last_odometry
            if self.odo is not None and step is not None and step.rotation_source == "gyro":
                plan["cmd_rot"] = plan.get("cmd_rot", 0.0) + 0.35 * step.dt
                plan["rot"] = plan.get("rot", 0.0) + step.dth
                if plan["cmd_rot"] >= 0.35 * 1.2 and abs(plan["rot"]) < 0.25 * plan["cmd_rot"]:
                    plan["kind"] = "wait"
                    plan["reason"] = "rotation_blocked"
                    return 0.0, 0.0
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
        level = cmd.get("level", "all")  # "objects": keep identity and character; "all": a new start
        try:
            self.memory.reset(cmd.get("confirm", ""), level=level, t=now)
        except ValueError as error:
            return {"command": "reset_memory", "accepted": False, "reason": str(error),
                    "generation": self.supervisor.generation, "t": now}
        self.memory.ensure_character(self.cfg.character, t=now)
        self.supervisor.stop("disabled")
        self._cancel_activity("memory_reset")
        self.tracker = EntityTracker(self.memory)
        self.recent_outcomes.clear()
        self.learning_updates.clear()
        self.motivation = Motivation(self.cfg.motivation)
        return {"command": "reset_memory", "accepted": True, "reason": f"memory_reset:{level}",
                "generation": self.supervisor.generation, "t": now}

    def _cancel_activity(self, reason):
        if reason != "blocked":  # anything else that cancels the activity also ends an open back-off
            self._end_backoff(self.now, interrupted=reason)
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
                "localization_lost", "stale_observation", "blocked", "blocked_repeatedly"):
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

    def odometry_status(self):
        if self.odo is None:
            return {"source": self.cfg.odometry, "fusion": dict(self.slam.fusion_log)}
        o, step = self.odo, self.last_odometry
        return {"source": "imu_encoders", "imu": self.spec.imu["part"], "encoders": self.spec.encoders["part"],
                "calibration": self.imu_calibration_status,
                "gyro_bias_dps": math.degrees(o.bias), "gyro_bias_sigma_dps": math.degrees(math.sqrt(o.bias_var)),
                "bias_calibrated": o.bias_calibrated, "gyro_scale": o.gyro_scale,
                "gyro_scale_sigma": math.sqrt(o.scale_var), "scale_samples": len(o.scale_samples),
                "track_scale": o.track_scale, "wheel_radius_scale": o.radius_scale,
                "slip": None if step is None else step.slip,
                "slip_counts": dict(o.log), "fusion": dict(self.slam.fusion_log),
                "command_model_fallback_frames": self.odo_fallbacks, "zupt_holds": self.zupt_holds,
                "zupt_hold_failures": self.zupt_hold_failures, "zupt_hold_active": self._zupt_hold_start is not None,
                "blocked_active": self._blocked_plan is not None, "blocked_events": len(self.blocked_events),
                "blocked_kind": self.slam.blocked_kind,
                "temp_c": o.temp, "tempco_dps_per_c": math.degrees(o.tempco[0]), "tempco_known": bool(o.tempco_known),
                "offset_updates": o.offset_updates, "zupt_frames": o.zupt_frames,
                "sensors": dict(self.sensor_state),
                "sample_age_s": {k: None if v is None else round(self.now - v, 3) for k, v in self._sample_t.items()}}

    def snapshot(self):
        loc = self.slam.snapshot()
        loc["inliers"] = None if self.last_track is None else self.last_track.inliers
        loc["odometry"] = self.odometry_status()
        act = self.activity.snapshot() if self.activity is not None else {
            "name": "stopped" if self.supervisor.stopped else "idle", "target_entity": None, "phase": "",
            "reason": self.supervisor.revoked_reason or "", "since": None}
        if self.supervisor.recovery is not None:
            act = {"name": "recovering", "target_entity": None, "phase": "rotating to relocalize",
                   "reason": "bounded recovery after localization loss", "since": None}
        elif self._blocked_plan is not None and self.supervisor.autonomy_enabled:
            act = {"name": "blocked", "target_entity": None, "phase": "backing off",
                   "reason": f"drive blocked ({self.slam.blocked_kind}); pose kept", "since": self._blocked_plan["start"]}
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
            "character": {**self.memory.character_inspector(), "active": self.cfg.character_active,
                          "effective": self.character_effective,
                          "reads": ("the chooser, navigator and probe budgets (engineered mapping, character.effective)"
                                    if self.cfg.character_active else
                                    "no behaviour reads these values yet (character_active is off)")},
            "map": {**self.grid.counts(), "map_version": self.grid.map_version,
                    "start_clearance_attested": self.cfg.start_clearance_attested if self._attested else None,
                    "loop_closure": {"status": self.place_status, "closures": len(self.loop_closures),
                                     "last": self.loop_closures[-1] if self.loop_closures else None}},
            "detector": self.detector_status,
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
               self.expression.signal_pattern, self.expression.caption, self.expression.cue)
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
