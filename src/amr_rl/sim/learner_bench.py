"""Learner testbed: the REAL learning layer against a fast synthetic world with
controllable perception errors.

What is real (imported unchanged from the runtime): ``ExperienceMemory`` (outcome
learning, change detection, probe budgets, persistence), ``EntityTracker``
(identity from appearance + position, duplicate merging), ``classify`` (outcome
from before/after observations), ``ActivityChooser.candidates_for`` /``_pick``
and ``Motivation``.

What is synthetic (this module, evaluation only): physics, rendering, mapping and
navigation are replaced by a point robot that travels in straight lines at a
fixed speed, a set of objects with hidden action -> consequence rules, and a
parameterised *perception error model* that turns true object states into the
detections a real camera + detector would produce: position noise and outliers,
appearance jitter and bad viewing angles (identity splits), missed detections,
per-frame state misreads, consistent mislabelled outcomes (a vision-language
model calling yellow "orange"), and hallucinated changes.

The purpose is to answer, quickly and over many seeds: does the learning layer
still learn the right things, avoid the right things, adapt and persist when its
perception is imperfect in the ways real perception will be? Ground truth here is
used only for scoring.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

import numpy as np

from ..behavior.chooser import ActivityChooser, ChooserConfig, Motivation, MotivationConfig
from ..learning import identity as identity_module
from ..learning.identity import EntityTracker, IdentityConfig
from ..learning.memory import ACTIONS, COLORS, ExperienceMemory, LearningConfig
from ..learning.outcomes import classify
from ..perception.entities import Attachment, Detection, hue_name

MAP_VERSION = "bench"
PALETTE = {"red": 0.0, "orange": 28.0, "yellow": 55.0, "green": 120.0, "cyan": 180.0, "blue": 225.0,
           "violet": 270.0, "magenta": 320.0}
BODY_COLORS = ("green", "cyan", "blue", "violet", "magenta", "orange")


# ---------------------------------------------------------------- configuration
@dataclass
class NoiseConfig:
    """Perception error model. Rates are per frame unless stated otherwise."""
    pos_sigma: float = 0.02          # m, detection position noise
    pos_outlier: float = 0.0         # probability of a 0.3-0.6 m position error
    appearance_jitter: float = 0.0   # scale of hue/size/fill jitter (1.0 = 6 deg, 2 cm, 0.03)
    bad_view: float = 0.0            # probability the object looks wider/less filled (corner-on view)
    miss: float = 0.0                # probability a visible object is not detected
    token_flip: float = 0.0          # probability the attachment state is misread in one frame
    label_confusion: float = 0.0     # per response: the new attachment is consistently reported as a
    #                                  neighbouring colour for that whole response (VLM synonym error)
    spurious_response: float = 0.0   # per interaction: a change that did not happen is reported
    name: str = "custom"


NOISE_PRESETS = {
    "clean": NoiseConfig(name="clean"),
    "low": NoiseConfig(pos_sigma=0.04, pos_outlier=0.02, appearance_jitter=0.5, bad_view=0.03, miss=0.05,
                       token_flip=0.02, label_confusion=0.05, spurious_response=0.02, name="low"),
    "medium": NoiseConfig(pos_sigma=0.06, pos_outlier=0.05, appearance_jitter=1.0, bad_view=0.08, miss=0.10,
                          token_flip=0.05, label_confusion=0.15, spurious_response=0.05, name="medium"),
    "high": NoiseConfig(pos_sigma=0.10, pos_outlier=0.10, appearance_jitter=1.5, bad_view=0.15, miss=0.20,
                        token_flip=0.10, label_confusion=0.30, spurious_response=0.10, name="high"),
}


@dataclass
class WorldConfig:
    room: float = 5.0                # square room side (m)
    n_objects: int = 8
    twins: int = 1                   # pairs of objects that look identical
    min_separation: float = 0.9
    rewarding: int = 2               # objects with a rewarding rule (one reliable, others p_unreliable)
    p_unreliable: float = 0.5
    aversive: int = 1
    movers: int = 1
    panel_seconds: float = 8.0
    world: str = "random"            # random | inert
    switch_at: float | None = None   # reversal: rewarding rules move to other objects at this time


@dataclass
class RobotConfig:
    speed: float = 0.2               # m/s
    view_range: float = 2.5
    fov_half_deg: float = 35.0
    frame_dt: float = 0.2
    signal_reach: float = 1.2        # fixtures respond to a signal within this range
    nudge_reach: float = 0.75        # a nudge touches only if the true object is this close
    idle_seconds: float = 5.0
    before_frames: int = 3           # pre-action frames kept once the target is confirmed (>= 3)


@dataclass
class TrueObject:
    index: int
    xy: np.ndarray
    hue: float
    color: str
    width: float
    height: float
    fill: float
    rules: dict                      # action -> (outcome token, probability) when no panel is up
    state: str = "attach:none"
    state_until: float = -1.0
    confused_as: str | None = None   # label error for the current response

    @property
    def appearance(self):
        return {"hue": self.hue, "color": self.color, "saturation": 0.8, "fill": self.fill,
                "width_m": self.width, "height_m": self.height}


# ---------------------------------------------------------------- world
class SyntheticWorld:
    def __init__(self, cfg: WorldConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.objects: list[TrueObject] = []
        self.switched = False
        self._place()
        self._assign_rules()

    def _place(self):
        half = self.cfg.room / 2 - 0.35
        pts = []
        while len(pts) < self.cfg.n_objects:
            p = self.rng.uniform(-half, half, 2)
            if np.linalg.norm(p) < 0.8 or any(np.linalg.norm(p - q) < self.cfg.min_separation for q in pts):
                continue
            pts.append(p)
        colors = list(self.rng.permutation(BODY_COLORS))
        for i, p in enumerate(pts):
            color = colors[i % len(colors)]
            shape = self.rng.choice(["box", "round"])
            self.objects.append(TrueObject(
                index=i, xy=np.array(p, float), hue=float((PALETTE[color] + self.rng.uniform(-5, 5)) % 360),
                color=color, width=float(self.rng.uniform(0.14, 0.3)), height=float(self.rng.uniform(0.14, 0.35)),
                fill=0.9 if shape == "box" else 0.78, rules={a: ("none", 0.0) for a in ACTIONS}))
        # Look-alike twins share an appearance (identity must use position).
        for k in range(min(self.cfg.twins, len(self.objects) // 2)):
            a, b = self.objects[2 * k], self.objects[2 * k + 1]
            b.hue, b.color, b.width, b.height, b.fill = a.hue, a.color, a.width, a.height, a.fill

    def _assign_rules(self):
        if self.cfg.world == "inert":
            return
        order = list(self.rng.permutation(len(self.objects)))
        k = 0
        for j in range(self.cfg.rewarding):
            obj = self.objects[order[k]]
            p = 1.0 if j == 0 else self.cfg.p_unreliable
            obj.rules["signal"] = ("attach:yellow", p)
            k += 1
        for _ in range(self.cfg.aversive):
            obj = self.objects[order[k]]
            obj.rules["signal"] = ("attach:red", 1.0)
            obj.rules["nudge"] = ("attach:red", 1.0)
            k += 1
        for _ in range(self.cfg.movers):
            obj = self.objects[order[k]]
            obj.rules["nudge"] = ("moved", 1.0)
            k += 1

    def maybe_switch(self, now):
        """Reversal: every rewarding rule moves to a currently inert object."""
        if self.cfg.switch_at is None or self.switched or now < self.cfg.switch_at:
            return False
        self.switched = True
        rewarding = [o for o in self.objects if o.rules["signal"][0] == "attach:yellow"]
        inert = [o for o in self.objects if all(r[0] == "none" for r in o.rules.values())]
        self.rng.shuffle(inert)
        for src, dst in zip(rewarding, inert):
            dst.rules["signal"], src.rules["signal"] = src.rules["signal"], ("none", 0.0)
        return True

    def tick(self, now):
        for o in self.objects:
            if o.state != "attach:none" and now >= o.state_until:
                o.state, o.confused_as = "attach:none", None

    def respond(self, obj: TrueObject, action, robot_xy, now, noise: NoiseConfig, reach_ok: bool):
        """Apply the hidden rule. Returns the TRUE consequence token."""
        if not reach_ok or obj.state != "attach:none":
            return "none"
        outcome, p = obj.rules[action]
        if outcome == "none" or self.rng.random() >= p:
            return "none"
        if outcome == "moved":
            d = obj.xy - np.asarray(robot_xy)
            d = d / max(np.linalg.norm(d), 1e-6)
            half = self.cfg.room / 2 - 0.3
            obj.xy = np.clip(obj.xy + 0.25 * d, -half, half)
            return "moved"
        obj.state, obj.state_until = outcome, now + self.cfg.panel_seconds
        if self.rng.random() < noise.label_confusion:
            c = outcome.split(":")[1]
            i = COLORS.index(c)
            obj.confused_as = f"attach:{COLORS[(i + self.rng.choice([-1, 1])) % len(COLORS)]}"
        return outcome

    def true_value(self, obj: TrueObject, action, valence):
        outcome, p = obj.rules[action]
        return p * valence.get(outcome, 0.0)

    def best_option(self, valence):
        """(object index, action, value) of the truly best option."""
        best = max(((o.index, a, self.true_value(o, a, valence)) for o in self.objects for a in ACTIONS),
                   key=lambda t: t[2])
        return best


# ---------------------------------------------------------------- perception
class NoisyPerception:
    def __init__(self, world: SyntheticWorld, noise: NoiseConfig, robot: RobotConfig, rng: np.random.Generator):
        self.world, self.noise, self.robot, self.rng = world, noise, robot, rng
        self.frame = 0
        self.spurious: dict | None = None  # {"object": index, "token": str, "until": t}

    def visible(self, pose, all_around=False):
        out = []
        for o in self.world.objects:
            d = o.xy - pose[:2]
            r = float(np.linalg.norm(d))
            if r > self.robot.view_range or r < 0.15:
                continue
            bearing = math.atan2(d[1], d[0]) - pose[2]
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            if not all_around and abs(bearing) > math.radians(self.robot.fov_half_deg):
                continue
            out.append((o, r, bearing))
        return out

    def _token(self, o: TrueObject, now):
        n = self.noise
        token = o.confused_as or o.state
        if self.spurious is not None and self.spurious["object"] == o.index and now <= self.spurious["until"]:
            token = self.spurious["token"]
        if self.rng.random() < n.token_flip:
            if token == "attach:none":
                token = f"attach:{COLORS[int(self.rng.integers(len(COLORS)))]}"
            else:
                token = "attach:none"
        return token

    def detect(self, pose, now, all_around=False):
        n = self.noise
        dets = []
        for o, r, bearing in self.visible(pose, all_around):
            if self.rng.random() < n.miss:
                continue
            j = n.appearance_jitter
            width = o.width + self.rng.normal(0, 0.02 * j) if j else o.width
            fill = o.fill + self.rng.normal(0, 0.03 * j) if j else o.fill
            hue = (o.hue + self.rng.normal(0, 6.0 * j)) % 360 if j else o.hue
            if self.rng.random() < n.bad_view:
                width, fill = width + 0.12, fill - 0.15
            xy = o.xy + self.rng.normal(0, n.pos_sigma, 2)
            if self.rng.random() < n.pos_outlier:
                a = self.rng.uniform(0, 2 * math.pi)
                xy = xy + self.rng.uniform(0.3, 0.6) * np.array([math.cos(a), math.sin(a)])
            token = self._token(o, now)
            atts = [] if token == "attach:none" else [
                Attachment(PALETTE.get(token.split(":")[1], 0.0), token.split(":")[1], 50, (0, 0, 5, 5))]
            det = Detection(index=len(dets), bbox=(0, 0, 20, 20), area=300, hue=float(hue), color=hue_name(hue),
                            saturation=0.8, fill=float(fill), base_uv=(10.0, 20.0), partial=False, attachments=atts,
                            position=(float(xy[0]), float(xy[1])), range_m=float(r), width_m=float(width),
                            height_m=float(o.height), bearing=float(bearing))
            det._true = o.index  # evaluation-only tag; the learning layer never reads it
            dets.append(det)
        self.frame += 1
        return dets


# ---------------------------------------------------------------- runtime shim
class _Slam:
    map_version = MAP_VERSION


class BenchRuntime:
    """The subset of RobotRuntime that ActivityChooser.candidates_for reads."""

    def __init__(self, memory: ExperienceMemory, *, policy: str, seed: int, room: float,
                 chooser_cfg: ChooserConfig | None = None, motivation_cfg: MotivationConfig | None = None):
        self.memory = memory
        self.tracker = EntityTracker(memory, IdentityConfig())
        self.motivation = Motivation(motivation_cfg)
        self.chooser = ActivityChooser(chooser_cfg, policy=policy, rng=np.random.default_rng(seed))
        self.slam = _Slam()
        self.pose = np.array([0.0, 0.0, 0.0])
        self.room = room
        self._unreachable = {}
        self._investigated = set()
        self._visited = set()
        self.map_progress = 100.0

    def known_entities(self):
        return self.tracker._load().values()

    def is_unreachable(self, entity_id, now):
        item = self._unreachable.get(entity_id)
        return item is not None and now < item[1]

    def note_unreachable(self, entity_id, now):
        count = self._unreachable.get(entity_id, (0, 0.0))[0] + 1
        self._unreachable[entity_id] = (count, now + 30.0 * min(4, count))

    def was_investigated(self, entity_id):
        return entity_id in self._investigated

    def option_interrupted(self, entity_id, action):
        return False

    def _cells(self):
        n = int(self.room)
        return [(i, j) for i in range(n) for j in range(n)]

    def cell_xy(self, c):
        return np.array([c[0] + 0.5 - self.room / 2, c[1] + 0.5 - self.room / 2])

    def frontiers(self):
        todo = [c for c in self._cells() if c not in self._visited]
        if not todo:
            return []
        here = self.pose[:2]
        c = min(todo, key=lambda c: float(np.linalg.norm(self.cell_xy(c) - here)))
        return [(self.cell_xy(c), 0.0, 20 * len(todo), c)]

    def mark_visited(self):
        for c in self._cells():
            if np.linalg.norm(self.cell_xy(c) - self.pose[:2]) <= 1.2:
                self._visited.add(c)
        self.map_progress = 150.0 if len(self._visited) < len(self._cells()) else 0.0


# ---------------------------------------------------------------- episode
@dataclass
class EpisodeConfig:
    seconds: float = 900.0
    policy: str = "learned"
    seed: int = 0
    world: WorldConfig = field(default_factory=WorldConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)


class _SeededUUID:
    def __init__(self, rng):
        self.rng = rng

    def uuid4(self):
        return uuid.UUID(int=int(self.rng.integers(0, 2 ** 63)) << 64 | int(self.rng.integers(0, 2 ** 63)), version=4)


class Episode:
    def __init__(self, cfg: EpisodeConfig, memory_path=":memory:", *, world: SyntheticWorld | None = None,
                 start_time: float = 0.0):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # reproducible entity ids (offset by the start time so a restarted session never reuses one)
        identity_module.uuid = _SeededUUID(np.random.default_rng([cfg.seed, 7919, int(start_time)]))
        self.world = world or SyntheticWorld(cfg.world, np.random.default_rng(cfg.seed + 1))
        self.memory = ExperienceMemory(memory_path, config=cfg.learning)
        self.rt = BenchRuntime(self.memory, policy=cfg.policy, seed=cfg.seed, room=cfg.world.room)
        self.perception = NoisyPerception(self.world, cfg.noise, cfg.robot, np.random.default_rng(cfg.seed + 2))
        self.now = start_time
        self.t0 = start_time
        self.events = []          # one row per interaction attempt (scoring)
        self.entity_truth = {}    # entity_id -> {true index: detection count}
        self.decisions = []
        self.switch_time = None
        self._eid = 0

    # -- primitives
    def _advance(self, dt):
        self.now += dt
        self.world.tick(self.now)
        if self.world.maybe_switch(self.now):
            self.switch_time = self.now
        self.rt.motivation.update(self.now)

    def _frame(self, all_around=False):
        dets = self.perception.detect(self.rt.pose, self.now, all_around)
        visible = self.rt.tracker.update(dets, tuple(self.rt.pose), 0.02, MAP_VERSION, self.now)
        for item in visible:
            if item.entity_id is not None:
                eid = self.memory.resolve(item.entity_id)
                counts = self.entity_truth.setdefault(eid, {})
                counts[item.detection._true] = counts.get(item.detection._true, 0) + 1
        self._advance(self.cfg.robot.frame_dt)
        return visible

    def _travel(self, goal, *, face=None):
        goal = np.asarray(goal, float)
        d = goal - self.rt.pose[:2]
        dist = float(np.linalg.norm(d))
        heading = math.atan2(d[1], d[0]) if dist > 1e-6 else self.rt.pose[2]
        self.rt.pose = np.array([*self.rt.pose[:2], heading])
        start = self.rt.pose[:2].copy()
        steps = int(dist / self.cfg.robot.speed)  # whole seconds of travel; one frame per second
        for k in range(1, steps + 1):
            self.rt.pose[:2] = start + d * (k / (steps + 1))
            self._frame()
            self._advance(1.0 - self.cfg.robot.frame_dt)
        self._advance((dist / self.cfg.robot.speed) - steps)
        self.rt.pose[:2] = goal
        if face is not None:
            f = np.asarray(face, float) - goal
            self.rt.pose[2] = math.atan2(f[1], f[0])
        self.rt.mark_visited()

    # -- activities
    def _explore(self, cand):
        goal = cand["frontier"][0]
        self._travel(goal)
        for _ in range(6):  # look around
            self.rt.pose[2] += math.pi / 3
            self._frame()

    def _investigate(self, cand):
        xy = cand["xy"]
        d = self.rt.pose[:2] - xy
        n = np.linalg.norm(d)
        goal = xy + (d / n if n > 1e-6 else np.array([1.0, 0.0])) * 0.9
        self._travel(goal, face=xy)
        for _ in range(4):
            self._frame()
        self.rt._investigated.add(cand["entity_id"])

    def _avoid(self, cand):
        if cand.get("hold_until") is not None:  # give-up hold: stand still
            self._advance(1.0)
            return
        xy = cand["xy"]
        d = self.rt.pose[:2] - xy
        n = np.linalg.norm(d)
        goal = self.rt.pose[:2] + (d / n if n > 1e-6 else np.array([1.0, 0.0])) * 0.6
        half = self.cfg.world.room / 2 - 0.3
        goal = np.clip(goal, -half, half)
        if np.linalg.norm(goal - self.rt.pose[:2]) < 0.3:  # cornered: back off toward the room centre
            c = -self.rt.pose[:2]
            goal = self.rt.pose[:2] + 0.6 * c / max(np.linalg.norm(c), 1e-6)
        self._travel(goal)
        self._frame()

    def _engage(self, cand):
        eid, action, xy = self.memory.resolve(cand["entity_id"]), cand["action"], cand["xy"]
        standoff = 0.75 if action == "signal" else 0.5
        d = self.rt.pose[:2] - xy
        n = np.linalg.norm(d)
        goal = xy + (d / n if n > 1e-6 else np.array([1.0, 0.0])) * standoff
        half = self.cfg.world.room / 2 - 0.25
        self._travel(np.clip(goal, -half, half), face=xy)
        row = {"t": self.now, "entity_id": eid, "action": action, "status": None, "true_object": None,
               "true_outcome": None, "observed": None, "true_valence": 0.0}
        # confirm the target in fresh images (as the Engage activity does)
        before, true_index, end, confirmed = [], None, self.now + 3.0, None
        want = max(3, self.cfg.robot.before_frames)
        while self.now < end:
            target = self.memory.resolve(eid)
            for item in self._frame():
                if item.entity_id is not None and self.memory.resolve(item.entity_id) == target \
                        and item.confidence >= 0.6:
                    det = item.detection
                    before.append({"token": det.state_token, "position": list(det.position),
                                   "frame_sha256": f"f{self.perception.frame}", "t": self.now})
                    true_index = det._true
            if confirmed is None and len(before) >= 3 and before[-1]["t"] - before[0]["t"] >= 0.2 and \
                    len({b["token"] for b in before[-3:]}) == 1:
                confirmed = before[-1]["token"]
            # once confirmed, keep frames showing the confirmed state (more frames -> a
            # better pre-action position estimate), up to ``before_frames``
            if confirmed is not None and sum(b["token"] == confirmed for b in before) >= want:
                break
        if confirmed is not None:
            before = [b for b in before if b["token"] == confirmed][-want:]
        else:
            row["status"] = "aborted"
            self.rt.note_unreachable(eid, self.now)
            self.events.append(row)
            return
        obj = self.world.objects[true_index]
        dist_true = float(np.linalg.norm(obj.xy - self.rt.pose[:2]))
        reach = self.cfg.robot.signal_reach if action == "signal" else self.cfg.robot.nudge_reach
        context = before[-1]["token"]
        prediction = self.memory.predict(eid, action, context, self.now)
        self.memory.note_proposal(eid, action)
        true = self.world.respond(obj, action, self.rt.pose[:2], self.now, self.cfg.noise, dist_true <= reach)
        if self.rng.random() < self.cfg.noise.spurious_response:
            fake = f"attach:{COLORS[int(self.rng.integers(len(COLORS)))]}"
            self.perception.spurious = {"object": true_index, "token": fake, "until": self.now + 6.0}
        self._advance(2.2 if action == "signal" else 4.0)
        after = []
        for _ in range(12):  # observe 2.4 s
            target = self.memory.resolve(eid)
            for item in self._frame():
                det = item.detection
                same = item.entity_id is not None and self.memory.resolve(item.entity_id) == target
                if same:
                    after.append({"token": det.state_token, "position": list(det.position),
                                  "frame_sha256": f"f{self.perception.frame}", "t": self.now})
        self.perception.spurious = None
        observed, why = classify(before, after)
        valence = self.cfg.learning.valence
        row.update({"true_object": int(true_index), "true_outcome": true, "observed": observed,
                    "true_valence": float(valence.get(true, 0.0)), "context": context,
                    "predicted_ev": prediction["expected_valence"], "why": why})
        if observed is None:
            row["status"] = "ambiguous"
            self.events.append(row)
            return
        self._eid += 1
        receipt = {"event_id": f"ev-{self.cfg.seed}-{self.t0:.0f}-{self._eid}", "entity_id": eid,
                   "action": action, "context": context, "observed": observed, "timestamp": self.now,
                   "authorised": True, "images_retained": True, "predicted": {}, "decision": {}}
        self.memory.record_outcome(receipt)
        self.rt.motivation.on_outcome(self.memory.resolve(eid), valence.get(observed, 0.0))
        self.rt.tracker.refresh()
        row["status"] = "outcome"
        self.events.append(row)

    def _idle(self, cand):
        self._advance(self.cfg.robot.idle_seconds)

    # -- loop
    def decide(self):
        cands = self.rt.chooser.candidates_for(self.rt, self.now)
        choice = self.rt.chooser._pick(self.rt, cands)
        return choice

    def run(self, seconds=None, *, initial_look=True):
        end = self.t0 + (seconds or self.cfg.seconds)
        if initial_look:
            for _ in range(6):
                self.rt.pose[2] += math.pi / 3
                self._frame()
            self.rt.mark_visited()
        while self.now < end:
            before = self.now
            choice = self.decide()
            self.decisions.append({"t": self.now, "activity": choice["activity"],
                                   "entity_id": choice.get("entity_id"), "action": choice.get("action")})
            act = choice["activity"]
            if act in ("engage", "revisit"):
                self._engage(choice)
            elif act == "explore":
                self._explore(choice)
            elif act == "investigate":
                self._investigate(choice)
            elif act == "avoid":
                self._avoid(choice)
            else:
                self._idle(choice)
            if self.now <= before:  # every decision costs time (no zero-time loops)
                self._advance(1.0)
        return self

    # -- scoring
    def entity_map(self):
        """entity -> majority true object, purity."""
        out = {}
        for eid, counts in self.entity_truth.items():
            total = sum(counts.values())
            idx, n = max(counts.items(), key=lambda kv: kv[1])
            out[eid] = (idx, n / total)
        return out

    def summary(self):
        ev = self.events
        done = [e for e in ev if e["status"] == "outcome"]
        valence = self.cfg.learning.valence
        correct = [e for e in done if e["observed"] == e["true_outcome"]]
        _, _, best_value = self.world.best_option(valence)
        emap = self.entity_map()
        ents = [e["entity_id"] for e in self.memory.entities()]
        impure = sum(1 for e in ents if e in emap and emap[e][1] < 0.8)
        objects_with_entity = len({emap[e][0] for e in ents if e in emap})
        aversive_by_object = {}
        for e in ev:
            if e.get("true_outcome") == "attach:red":
                aversive_by_object[e["true_object"]] = aversive_by_object.get(e["true_object"], 0) + 1
        # Final beliefs: the entity/action the memory values most, and what it truly is.
        top = None
        for eid in ents:
            for a in ACTIONS:
                p = self.memory.predict(eid, a, "attach:none", self.now)
                if top is None or p["expected_valence"] > top[2]:
                    top = (eid, a, p["expected_valence"])
        top_true = None if top is None or top[0] not in emap else self.world.true_value(
            self.world.objects[emap[top[0]][0]], top[1], valence)
        return {
            "seconds": self.now - self.t0,
            "attempts": len(ev),
            "outcomes": len(done),
            "ambiguous": sum(e["status"] == "ambiguous" for e in ev),
            "aborted": sum(e["status"] == "aborted" for e in ev),
            "true_valence": float(sum(e["true_valence"] for e in ev)),
            "true_valence_per_attempt": float(np.mean([e["true_valence"] for e in ev])) if ev else 0.0,
            "rewards": sum(e.get("true_outcome") in ("attach:yellow",) for e in ev),
            "aversive": sum(aversive_by_object.values()),
            "aversive_repeats": sum(max(0, n - 1) for n in aversive_by_object.values()),
            "label_accuracy": len(correct) / len(done) if done else None,
            "false_moved": sum(e["observed"] == "moved" and e["true_outcome"] != "moved" for e in done),
            "moved_recall": (sum(e["observed"] == "moved" for e in done if e["true_outcome"] == "moved")
                             / max(1, sum(e["true_outcome"] == "moved" for e in done))
                             if any(e["true_outcome"] == "moved" for e in done) else None),
            "entities": len(ents), "true_objects": len(self.world.objects),
            "objects_identified": objects_with_entity, "impure_entities": impure,
            "merges": self.memory.counts()["merges"],
            "top_belief_true_value": top_true, "best_true_value": best_value,
            "top_belief_is_best": None if top_true is None else bool(top_true >= best_value - 1e-9),
            "idle_fraction": float(np.mean([d["activity"] == "idle" for d in self.decisions])) if self.decisions else 0.0,
        }
