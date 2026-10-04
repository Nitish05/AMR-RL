"""Activities: explore, investigate, engage/revisit, avoid, idle, survey.

Each activity returns body-frame (v, w) requests every control tick; the
supervisor decides whether they reach the wheels. Navigation failure ends an
activity with a *navigation* result and is never recorded as an interaction
outcome. Interaction outcomes come only from before/after onboard observations.
"""

from __future__ import annotations

import math
import uuid

import numpy as np

from ..learning.identity import appearance_distance
from ..learning.outcomes import classify
from ..perception.camera_model import wrap
from .frontier import frontier_goals

SIGNAL_STANDOFF = (0.75, 0.65, 0.9, 1.0)  # fixtures respond to signals within ~1.2 m
NUDGE_STANDOFF = (0.52, 0.47, 0.58)
INVESTIGATE_STANDOFF = (0.9, 1.1, 1.3, 1.6, 1.9)
FRONT_EXTENT = 0.14
PUSH_DEPTH = 0.03  # engineered: how far past estimated contact a nudge pushes
MAX_CREEP = 0.40


def nudge_creep(contact_range: float | None, centre_range: float | None, radius: float) -> float:
    """Forward distance a nudge creeps from where it stands.

    Prefer the range to the nearest visible floor contact (a box's front edge, a
    sphere's near-centre contact): it does not depend on guessing the object's shape.
    The older estimate (centre range minus a radius) stopped ~10 cm short of boxes
    seen corner-on, whose silhouette the detector took for round (fixed baseline,
    round 2: 0/53 nudges touched grump). For a sphere the contact-edge estimate pushes
    up to one radius deeper, which only rolls it further.
    """
    if contact_range is not None:
        creep = contact_range - FRONT_EXTENT + PUSH_DEPTH
    else:
        creep = (centre_range if centre_range is not None else NUDGE_STANDOFF[0]) - (radius + FRONT_EXTENT) + PUSH_DEPTH
    return float(np.clip(creep, 0.0, MAX_CREEP))


class Activity:
    name = "activity"

    def __init__(self, target_entity=None, reason=""):
        self.target_entity = target_entity
        self.reason = reason
        self.phase = "starting"
        self.done = False
        self.result = None  # dict describing how it ended
        self.since = None
        self.value = 0.0
        self.preemptible = True

    def finish(self, status, **detail):
        self.done = True
        self.result = {"status": status, **detail}

    def cancel(self, reason):
        if not self.done:
            self.finish("cancelled", reason=reason)

    def step(self, rt, now):  # pragma: no cover - interface
        raise NotImplementedError

    def snapshot(self):
        return {"name": self.name, "target_entity": self.target_entity, "phase": self.phase,
                "reason": self.reason, "since": self.since}


def standoff_pose(rt, target_xy, distances, prefer_from):
    """Traversable pose at one of ``distances`` from the target, facing it.

    Candidates on rings every 15 deg; preference: shortest acceptable distance
    first, then the side facing ``prefer_from``. Only certified (traversable)
    robot-centre cells qualify.
    """
    target = np.asarray(target_xy, float)
    distances = [distances] if np.isscalar(distances) else list(distances)
    base = math.atan2(prefer_from[1] - target[1], prefer_from[0] - target[0])
    for distance in distances:
        options = []
        for k in range(24):
            ang = base + math.radians(15) * k
            p = target + distance * np.array([math.cos(ang), math.sin(ang)])
            options.append((abs(wrap(ang - base)), p, math.atan2(-math.sin(ang), -math.cos(ang))))
        options.sort(key=lambda o: o[0])
        pts = np.array([o[1] for o in options])
        ok = rt.planner.traversable_xy(rt.grid, pts)
        for (_, p, heading), good in zip(options, ok):
            if good:
                return p, heading
    return None, None


class Goto:
    """Helper: navigate with the shared navigator; reports navigation-only results."""

    def __init__(self, goal, heading=None):
        self.goal, self.heading = np.asarray(goal, float), heading
        self.started = False
        self.status = "pending"
        self.reason = ""

    def step(self, rt, now):
        if not self.started:
            self.started = True
            if not rt.nav.set_goal(rt.grid, rt.pose, self.goal, heading=self.heading, now=now,
                                   avoid=rt.avoid_regions()):
                self.status, self.reason = "rejected", rt.nav.reason
                return 0.0, 0.0
            self.status = "moving"
        if getattr(rt, "loc_status", "tracking") != "tracking":
            # Ordinary navigation holds still on predicted (dead-reckoned) poses. With
            # respect_degraded_heading, an in-place alignment turn may continue on the
            # VSLAM's bounded turn prediction (rotation only), instead of freezing.
            turning = (getattr(rt, "loc_status", "") == "predicted" and rt.pose is not None
                       and getattr(getattr(rt, "cfg", None), "respect_degraded_heading", False)
                       and getattr(getattr(rt, "slam", None), "_turn", None) is not None)
            if not turning:
                return 0.0, 0.0
            v, w = rt.nav.step(rt.grid, rt.pose, rt.sigma, now)
            return (0.0, w) if abs(v) < 1e-9 else (0.0, 0.0)
        v, w = rt.nav.step(rt.grid, rt.pose, rt.sigma, now)
        if rt.nav.status == "arrived":
            self.status = "arrived"
        elif rt.nav.status in ("blocked", "rejected", "localization_uncertain", "cancelled"):
            self.status, self.reason = "failed", f"{rt.nav.status}: {rt.nav.reason}"
        return v, w


class Idle(Activity):
    """Rest in place for a bounded period, then end so the chooser reconsiders
    (need and habituation change with time even when the view does not)."""

    name = "idle"
    REST_SECONDS = 8.0

    def __init__(self, target_entity=None, reason="", rest=REST_SECONDS):
        super().__init__(target_entity, reason)
        self.rest = rest
        self.started = None

    def step(self, rt, now):
        self.phase = "resting"
        if self.started is None:
            self.started = now
        if now - self.started >= self.rest:
            self.finish("completed")
        return 0.0, 0.0


class Survey(Activity):
    """Rotate in place to observe surroundings (the camera sits ahead of the
    rotation axis, so turning also provides lateral parallax for mapping)."""

    name = "explore"

    def __init__(self, angle=2 * math.pi, rate=0.5, reason="look around"):
        super().__init__(None, reason)
        self.remaining = angle
        self.rate = rate
        self.last = None
        self.pause_until = None

    def step(self, rt, now):
        self.phase = "surveying"
        if rt.pose is None:
            return 0.0, 0.0
        if self.last is not None:
            self.remaining -= abs(wrap(rt.pose[2] - self.last))
        self.last = rt.pose[2]
        if self.remaining <= 0:
            self.finish("completed")
            return 0.0, 0.0
        # Short pauses every ~45 degrees give keyframes with stable images.
        if self.pause_until is not None and now < self.pause_until:
            return 0.0, 0.0
        if int(self.remaining / (math.pi / 4)) != int((self.remaining + 0.05) / (math.pi / 4)):
            self.pause_until = now + 0.4
        return 0.0, self.rate


class Explore(Activity):
    name = "explore"

    def __init__(self, goal, heading, gain, key, reason=""):
        super().__init__(None, reason or f"frontier with {gain} unmapped-edge cells")
        self.goto = Goto(goal, heading)
        self.key = key
        self.survey = None
        self.free_at_start = None

    def finish(self, status, **detail):
        super().finish(status, **detail)

    def step(self, rt, now):
        if self.free_at_start is None:
            self.free_at_start = rt.grid.counts()["free"]
        if self.survey is None:
            self.phase = "travelling to frontier"
            v, w = self.goto.step(rt, now)
            if self.goto.status == "arrived":
                self.survey = Survey(angle=math.radians(140), rate=0.45)
                self.survey.rate = 0.45
                self._sweep = [math.radians(50), -math.radians(100), math.radians(50)]
                coverage = isinstance(self.key, tuple) and bool(self.key) and self.key[0] == "cover"
                if coverage or getattr(rt, "wants_panorama", lambda: False)():
                    self._sweep = [2 * math.pi]  # full look: keyframes facing every way
                    rt.note_panorama()
                self._sweep_i = 0
                self._last = rt.pose[2]
                self._left = self._sweep[0]
            elif self.goto.status in ("rejected", "failed"):
                rt.note_frontier_failure(self.key)
                self.finish("navigation_failed", reason=self.goto.reason)
            return v, w
        # sweep left, right, back to centre
        self.phase = "looking around"
        if rt.pose is None:
            return 0.0, 0.0  # no pose this frame: hold (the sweep resumes when it returns)
        turned = wrap(rt.pose[2] - self._last)
        self._last = rt.pose[2]
        self._left -= turned
        target = self._sweep[self._sweep_i]
        if (target > 0 and self._left <= 0) or (target < 0 and self._left >= 0):
            self._sweep_i += 1
            if self._sweep_i >= len(self._sweep):
                rt.note_frontier_visit(self.key)
                rt.note_map_progress(rt.grid.counts()["free"] - self.free_at_start)
                self.finish("completed")
                return 0.0, 0.0
            self._left = self._sweep[self._sweep_i]
        return 0.0, 0.45 if self._sweep[self._sweep_i] > 0 else -0.45


class Investigate(Activity):
    name = "investigate"

    def __init__(self, entity_id, target_xy, reason=""):
        super().__init__(entity_id, reason or "closer look at an entity")
        self.target_xy = np.asarray(target_xy, float)
        self.goto = None
        self.dwell_until = None

    def step(self, rt, now):
        if self.goto is None:
            goal, heading = standoff_pose(rt, self.target_xy, INVESTIGATE_STANDOFF, rt.pose[:2])
            if goal is None:
                rt.note_unreachable(self.target_entity, now)
                rt.note_investigated(self.target_entity)
                self.finish("navigation_failed", reason="no certified observation pose")
                return 0.0, 0.0
            self.goto = Goto(goal, heading)
        if self.dwell_until is None:
            self.phase = "approaching"
            v, w = self.goto.step(rt, now)
            if self.goto.status == "arrived":
                self.dwell_until = now + 2.0
            elif self.goto.status in ("rejected", "failed"):
                rt.note_unreachable(self.target_entity, now)
                rt.note_investigated(self.target_entity)
                self.finish("navigation_failed", reason=self.goto.reason)
            return v, w
        self.phase = "observing"
        if now >= self.dwell_until:
            rt.note_investigated(self.target_entity)
            self.finish("completed")
        return 0.0, 0.0


class Avoid(Activity):
    name = "avoid"

    def __init__(self, entity_id, entity_xy, reason="", hold_until=None, radius=0.75):
        super().__init__(entity_id, reason or "keeping distance from a disliked entity")
        self.entity_xy = np.asarray(entity_xy, float)
        self.goto = None
        self.hold_until = hold_until  # retreats kept failing: hold still (never approach)
        self.radius = radius  # the retreat only succeeds if it ends outside this distance

    def step(self, rt, now):
        if self.hold_until is not None:
            self.phase = "holding (retreat failed)"
            if now >= self.hold_until:
                self.finish("held")
            return 0.0, 0.0
        if self.goto is None:
            away = np.asarray(rt.pose[:2]) - self.entity_xy
            ang = math.atan2(away[1], away[0])
            goal = None
            # Retreat poses 0.8-1.3 m from the entity, preferring the away direction;
            # never the far side (that path would pass the entity).
            deltas = [0.0] + [sgn * math.radians(d) for d in (30, 60, 90, 120) for sgn in (1, -1)]
            here = float(np.linalg.norm(away))
            # A retreat goal must be clearly farther than the robot is now: one within the
            # navigator's arrival tolerance "arrives" at once without moving.
            dists = [d for d in (1.1, 0.95, 1.3, 0.8) if d >= max(self.radius, here) + 0.3] or [1.3]
            for delta in deltas:
                for dist in dists:
                    p = self.entity_xy + dist * np.array([math.cos(ang + delta), math.sin(ang + delta)])
                    if rt.planner.traversable_xy(rt.grid, p[None])[0]:
                        goal = p
                        break
                if goal is not None:
                    break
            if goal is None:
                self.finish("navigation_failed", reason="no certified retreat pose")
                return 0.0, 0.0
            self.goto = Goto(goal, None)
        self.phase = "retreating"
        v, w = self.goto.step(rt, now)
        if self.goto.status == "arrived":
            if np.linalg.norm(np.asarray(rt.pose[:2]) - self.entity_xy) >= self.radius:
                self.finish("completed")
            else:  # "arrived" but still within the radius: counts towards giving up
                self.finish("navigation_failed", reason="retreat ended inside the avoid radius")
        elif self.goto.status in ("rejected", "failed"):
            self.finish("navigation_failed", reason=self.goto.reason)
        return v, w


class Engage(Activity):
    """Approach -> confirm (before window) -> act -> observe (after window) -> receipt."""

    name = "engage"

    def __init__(self, entity_id, action, target_xy, decision, *, revisit=False, reason=""):
        super().__init__(entity_id, reason)
        self.name = "revisit" if revisit else "engage"
        self.action = action
        self.target_xy = np.asarray(target_xy, float)
        self.decision = decision
        self.request_id = uuid.uuid4().hex
        self.goto = None
        self.before, self.after = [], []
        self.during_frames = []
        self.place_empty = 0
        self.phase_until = None
        self.act_start_pose = None
        self.creep = 0.0
        self.prediction = None
        self.context = None
        self.appearance = None
        self.preemptible = True
        self.status = "approaching"
        self.generation = None

    def interaction(self):
        if self.phase not in ("confirming", "acting", "retreating", "observing"):
            return None
        status = {"confirming": "confirming", "acting": "acting", "retreating": "observing",
                  "observing": "observing"}[self.phase]
        return {"request_id": self.request_id, "entity_id": self.target_entity, "action": self.action,
                "predicted": None if self.prediction is None else self.prediction["probabilities"],
                "observed": None, "status": status}

    def _target_detection(self, rt):
        best = None
        for item in rt.tracker.visible:
            det = item.detection
            if item.entity_id == self.target_entity:
                return det, item.confidence
            if self.appearance is not None and not det.partial:
                d = appearance_distance(det.descriptor(), self.appearance)
                if d < 1.2 and (best is None or d < best[1]):
                    best = (det, d)
        if best is not None and self.phase in ("observing", "retreating"):
            return best[0], 0.5
        return None, 0.0

    def step(self, rt, now):
        self.generation = rt.supervisor.generation if self.generation is None else self.generation
        if self.goto is None:
            standoff = SIGNAL_STANDOFF if self.action == "signal" else NUDGE_STANDOFF
            here = float(np.linalg.norm(self.target_xy - np.asarray(rt.pose[:2])))
            if min(standoff) - 0.08 <= here <= max(standoff) + 0.05 and abs(standoff[0] - here) <= 0.25:
                # Already at a suitable standoff (the body occupies this space): no travel.
                self.goto = Goto(rt.pose[:2])
                self.goto.status = "arrived"
                self.phase, self.phase_until = "confirming", now + 4.0
                self.preemptible = False
                return 0.0, 0.0
            goal, heading = standoff_pose(rt, self.target_xy, standoff, rt.pose[:2])
            if goal is None:
                rt.note_unreachable(self.target_entity, now)
                self.finish("navigation_failed", reason="no certified standoff pose")
                return 0.0, 0.0
            self.goto = Goto(goal, heading)
            self.phase = "approaching"
        if self.phase == "approaching":
            v, w = self.goto.step(rt, now)
            if self.goto.status == "arrived":
                self.phase, self.phase_until = "confirming", now + 3.0
                self.preemptible = False  # an interaction in progress finishes or is revoked
            elif self.goto.status in ("rejected", "failed"):
                rt.note_unreachable(self.target_entity, now)
                self.finish("navigation_failed", reason=self.goto.reason)
            return v, w
        det, conf = self._target_detection(rt)
        frame = rt.last_frame_record
        if self.phase == "confirming":
            if det is not None and conf >= 0.6 and not det.partial and det.position is not None:
                self.before.append({"token": det.state_token, "position": list(det.position),
                                    "frame_sha256": frame["raw_rgb_sha256"], "t": now,
                                    "bearing": det.bearing, "range": det.range_m,
                                    "contact_range": det.contact_range_m})
                self.appearance = rt.entity_appearance(self.target_entity)
            if len(self.before) >= 3 and self.before[-1]["t"] - self.before[0]["t"] >= 0.2:
                tokens = [b["token"] for b in self.before[-3:]]
                if len(set(tokens)) == 1:
                    self.context = tokens[0]
                    self.before = self.before[-3:]
                    self.prediction = rt.memory.predict(self.target_entity, self.action, self.context, now)
                    self.phase, self.phase_until = "acting", now + 2.2
                    self.act_start_pose = np.array(rt.pose)
                    app = rt.entity_appearance(self.target_entity) or {}
                    size = min(app.get("width_m") or 0.2, 1.1 * (app.get("height_m") or 0.2))
                    radius = float(np.clip(size / 2, 0.05, 0.15))
                    contacts = [b.get("contact_range") for b in self.before[-3:] if b.get("contact_range")]
                    self.creep = nudge_creep(float(np.median(contacts)) if contacts else None,
                                             self.before[-1]["range"], radius)
                    if self.action == "nudge":
                        self.phase_until = now + self.creep / 0.06 + 2.5
                    rt.memory.note_proposal(self.target_entity, self.action)
                    return 0.0, 0.0
            if now > self.phase_until:
                self.finish("aborted", reason="target not confirmed in fresh images")
                rt.note_unreachable(self.target_entity, now, missing=True)
            # face the target while confirming (seen bearing, else remembered position)
            if det is not None and det.bearing is not None:
                return 0.0, float(np.clip(1.2 * det.bearing, -0.4, 0.4))
            d = self.target_xy - np.asarray(rt.pose[:2])
            bearing = wrap(math.atan2(d[1], d[0]) - rt.pose[2])
            return 0.0, float(np.clip(1.2 * bearing, -0.4, 0.4)) if abs(bearing) > 0.1 else 0.0
        if self.phase == "acting":
            self.during_frames.append(frame["raw_rgb_sha256"])
            if self.action == "signal":
                if now >= self.phase_until:
                    self.phase, self.phase_until = "observing", now + 2.5
                return 0.0, 0.0
            travelled = float(np.hypot(*(np.asarray(rt.pose[:2]) - self.act_start_pose[:2])))
            if travelled >= self.creep or now >= self.phase_until:
                self.phase = "retreating"
                self.retreat_from = np.array(rt.pose)
                self.retreat_goal = travelled
                self.phase_until = now + 6.0
                return 0.0, 0.0
            w = 0.0 if det is None or det.bearing is None else float(np.clip(0.8 * det.bearing, -0.25, 0.25))
            return 0.06, w
        if self.phase == "retreating":
            self.during_frames.append(frame["raw_rgb_sha256"])
            back = float(np.hypot(*(np.asarray(rt.pose[:2]) - self.retreat_from[:2])))
            if back >= self.retreat_goal - 0.01 or now >= self.phase_until:
                self.phase, self.phase_until = "observing", now + 2.5
                return 0.0, 0.0
            return -0.10, 0.0
        if self.phase == "observing":
            if det is not None and det.position is not None and not det.partial:
                self.after.append({"token": det.state_token, "position": list(det.position),
                                   "frame_sha256": frame["raw_rgb_sha256"], "t": now})
            elif rt.place_in_view(self.before[-1]["position"]):
                self.place_empty += 1
            if now >= self.phase_until:
                observed, why = classify(self.before, self.after, place_in_view_after=self.place_empty)
                self.phase = "complete"
                if observed is None:
                    self.finish("ambiguous_outcome", reason=why)
                else:
                    self.finish("outcome", observed=observed, why=why)
            return 0.0, 0.0
        return 0.0, 0.0

    def receipt(self, rt, now):
        """Complete record of this interaction; images are retained before learning."""
        return {
            "schema": "amr_rl.interaction-receipt.v1",
            "event_id": self.request_id,
            "entity_id": self.target_entity,
            "action": self.action,
            "context": self.context,
            "observed": self.result["observed"],
            "classification_reason": self.result["why"],
            "timestamp": now,
            "generation": self.generation,
            "authorised": True,
            "identity": {"confidence_at_confirm": 0.6, "appearance": self.appearance},
            "decision": self.decision,
            "predicted": self.prediction["probabilities"],
            "predicted_expected_valence": self.prediction["expected_valence"],
            "before": self.before,
            "after": self.after[-40:],
            "during_frame_sha256": self.during_frames[-60:],
            "place_empty_frames": self.place_empty,
            "map_version": rt.slam.map_version,
            "calibration_version": rt.model.version,
            "evidence_source": "onboard_rgb",
        }


def choose_frontier(rt):
    goals = frontier_goals(rt.grid, rt.planner, rt.pose, exhausted=rt.exhausted_frontiers())
    return goals
