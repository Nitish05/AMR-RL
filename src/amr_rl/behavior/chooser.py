"""Experience-dependent activity selection with bounded reconsideration.

ENGINEERED (configured, not learned): the stimulation need and its dynamics,
habituation, action/travel costs, activity base values and switching limits.
LEARNED (from memory): outcome predictions per entity/action/context, which set
expected valence, uncertainty/information value and attitudes.

Policy: compute a value for each available activity; keep the current one unless
a reconsideration trigger fires and an alternative beats it by a margin, subject
to a minimum interval and a per-minute switch budget. Idle is always available
with value 0; nothing forces activity.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..learning.memory import ACTIONS
from .activities import Avoid, Engage, Explore, Idle, Investigate, Survey


@dataclass
class MotivationConfig:
    initial_need: float = 0.6
    need_growth: float = 0.004  # per sim second
    satiation: float = 0.35  # need reduction per unit positive valence
    habituation_half_life: float = 120.0
    idle_below: float = 0.2
    # Evaluation-only ablation: need fixed at 1, no satiation, no habituation. It
    # isolates choice quality from the engineered drive (which makes the deployed
    # policy idle once satisfied); idle stays available. Never a deployed setting.
    clamp: bool = False


@dataclass
class ChooserConfig:
    travel_cost: float = 0.05  # per metre
    action_cost: dict = field(default_factory=lambda: {"signal": 0.03, "nudge": 0.06})
    explore_weight: float = 0.15
    investigate_value: float = 0.10
    margin: float = 0.08
    # Character (behavior/character.py; set by RobotRuntime when character_active): a gain on
    # every information-value term, and decision-time pessimism, value = expected valence
    # minus value_z x its uncertainty (0 = the mean). Both 1.0 / 0.0 leave behaviour as it was.
    info_gain: float = 1.0
    value_z: float = 0.0
    min_interval: float = 3.0
    max_switches_per_minute: int = 6
    avoid_radius: float = 0.75
    unreachable_backoff: float = 30.0
    # Re-targeting holds (behavior/targeting.py; docs/results/r9-followup.md §12, pre-registered,
    # off until the suite decides): an option never confirmed in this session is not offered again
    # after unconfirmed_hold approaches in a row that did not confirm it (0 = off); an entity found
    # absent from its remembered place waits for a sighting or a new position (absent_hold).
    unconfirmed_hold: int = 0
    absent_hold: bool = False
    # Give-up rule for the forced avoid (engineered): after this many failed retreats
    # the robot holds still (it never approaches the entity) for avoid_backoff x
    # min(4, holds) seconds, then tries one retreat again.
    avoid_max_failures: int = 3
    avoid_backoff: float = 20.0
    # Frustration renewal (engineered, off; evaluated in the learning re-run): when idle
    # wins for this long although the need is unsatisfied (>= frustration_need), every
    # exhausted option tried at least this long ago gets one fresh probe. Round 6 seed 4
    # idled from the swap to the end of the run: its tried options had no probes left,
    # and no change was detected because its established effect (pushing the roller) is
    # physics and did not change. 0 = off.
    frustration_renewal_s: float = 0.0
    frustration_need: float = 0.5


class Motivation:
    """Engineered drive: 'stimulation need'. Satisfied by observed positive consequences."""

    def __init__(self, cfg: MotivationConfig | None = None):
        self.cfg = cfg or MotivationConfig()
        self.need = 1.0 if self.cfg.clamp else self.cfg.initial_need
        self.habituation: dict[str, float] = {}
        self.last = None

    def update(self, now):
        if self.cfg.clamp:
            self.last = now
            return
        if self.last is not None:
            dt = max(0.0, now - self.last)
            self.need = min(1.0, self.need + self.cfg.need_growth * dt)
            decay = 0.5 ** (dt / self.cfg.habituation_half_life)
            self.habituation = {k: v * decay for k, v in self.habituation.items() if v * decay > 1e-3}
        self.last = now

    def on_outcome(self, entity_id, valence):
        if valence > 0 and not self.cfg.clamp:
            self.need = max(0.0, self.need - self.cfg.satiation * valence)
            self.habituation[entity_id] = self.habituation.get(entity_id, 0.0) + 1.0

    def novelty(self, entity_id):
        return 1.0 / (1.0 + self.habituation.get(entity_id, 0.0))

    def snapshot(self):
        return {"stimulation_need": self.need, "engineered": True, "drive_clamped": self.cfg.clamp,
                "habituation": {k: round(v, 3) for k, v in self.habituation.items()}}


def option_values(memory, entity_id, context, now, *, need, novelty, distance, cfg):
    """Pure value computation for engaging one entity with each action.

    value = need * learned expected valence * novelty
            + curiosity bonus (learned uncertainty, engineered weight, damped by need)
            - engineered travel/action cost
    """
    preds = {a: memory.predict(entity_id, a, context, now) for a in ACTIONS}
    evs = {a: p["expected_valence"] for a, p in preds.items()}
    out = []
    for action in ACTIONS:
        pred = preds[action]
        best_other = max([0.0] + [e for a, e in evs.items() if a != action])
        info = cfg.info_gain * memory.information_value(pred, best_other) * (0.3 + 0.7 * need)
        cost = cfg.travel_cost * distance + cfg.action_cost[action]
        ev = pred["expected_valence"]
        ev_used = ev - cfg.value_z * pred["uncertainty"]  # caution: a pessimistic (or optimistic) read of the value
        out.append({"action": action, "prediction": pred, "state": memory.option_state(entity_id, action, now),
                    "expected_value": ev, "information_value": info, "cost": cost,
                    "value": need * ev_used * novelty + info - cost})
    return out


class ActivityChooser:
    def __init__(self, cfg: ChooserConfig | None = None, *, policy="learned", rng=None):
        self.cfg = cfg or ChooserConfig()
        self.policy = policy  # learned | random | nearest | fixed | explore_only
        self.rng = rng or np.random.default_rng(0)
        self.last_decision = -math.inf
        self.switch_times = deque()
        self.reconsiderations = 0
        self.candidates = []
        self.chosen = None
        self.fixed_order = None
        self._avoid_state = {}  # entity -> {"failures", "holds", "hold_until"} (this session)
        self._last_avoid = None
        self._idle_need_since = None
        self.renewals = []  # (t, options renewed) by frustration renewal

    def _note_avoid_result(self, now):
        """Read the outcome of the last Avoid once. A cancel (Stop, revocation) is not
        a failure; a completed retreat resets the entity's give-up state."""
        act = self._last_avoid
        if act is None or not act.done:
            return
        self._last_avoid = None
        status = (act.result or {}).get("status")
        st = self._avoid_state.setdefault(act.target_entity, {"failures": 0, "holds": 0, "hold_until": None})
        if status == "completed":
            self._avoid_state.pop(act.target_entity, None)
        elif status == "held":
            st["hold_until"] = None  # one retry
        elif status == "navigation_failed":
            st["failures"] += 1
            if st["failures"] >= self.cfg.avoid_max_failures:
                st["holds"] += 1
                st["hold_until"] = now + self.cfg.avoid_backoff * min(4, st["holds"])

    # ---------------------------------------------------------------- values
    def candidates_for(self, rt, now):
        need = rt.motivation.need
        pose = np.asarray(rt.pose[:2])
        out = []
        visible_ids = {v.entity_id for v in rt.tracker.visible if v.entity_id}
        self._note_avoid_result(now)
        for ent in rt.known_entities():
            eid = ent["entity_id"]
            if ent.get("x") is None or ent.get("map_version") != rt.slam.map_version:
                continue
            blocked = rt.is_unreachable(eid, now)
            xy = np.array([ent["x"], ent["y"]])
            dist = float(np.linalg.norm(xy - pose))
            att = rt.memory.attitude(eid, now)
            # Avoid comes before any skip: a disliked entity close by is always kept at a
            # distance, even one that is unreachable for engaging.
            if att["attitude"] == "disliked" and dist < self.cfg.avoid_radius:
                st = self._avoid_state.get(eid)
                hold = st["hold_until"] if st and st["hold_until"] is not None and now < st["hold_until"] else None
                basis = f"learned disliked (EV {att['expected_value']:.2f}); within {dist:.2f} m"
                if hold is not None:
                    basis += f"; retreat failed {st['failures']}x: holding still until t={hold:.0f} s"
                out.append({"activity": "avoid", "entity_id": eid, "action": None, "value": 5.0,
                            "expected_value": att["expected_value"], "information_value": 0.0, "cost": 0.0,
                            "basis": basis, "xy": xy, "hold_until": hold})
            elif eid in self._avoid_state:
                self._avoid_state.pop(eid)  # out of range: the give-up state starts afresh
            if rt.targets.absent(eid, xy, eid in visible_ids):
                continue  # looked at its remembered place and saw nothing: wait for a sighting or a new place
            if blocked and rt.was_investigated(eid):
                continue
            context = rt.tracker.last_state.get(eid, "attach:none")
            for item in ([] if blocked else option_values(rt.memory, eid, context, now, need=need,
                                                           novelty=rt.motivation.novelty(eid), distance=dist,
                                                           cfg=self.cfg)):
                action, pred, state = item["action"], item["prediction"], item["state"]
                if state["probes"] <= 0 and self.policy == "learned":
                    continue
                if rt.option_interrupted(eid, action):
                    continue  # session-scoped: this action keeps losing localization here
                if rt.targets.held(eid, action):
                    continue  # session-scoped: approaches from this action's standoff never confirm it
                value, ev, info, cost = item["value"], item["expected_value"], item["information_value"], item["cost"]
                p_top = max(pred["probabilities"].items(), key=lambda kv: kv[1])
                out.append({
                    "activity": "engage" if eid in visible_ids else "revisit",
                    "entity_id": eid, "created": ent.get("created"), "action": action, "value": value,
                    "expected_value": ev,
                    # the prediction's uncertainty travels with the record: the character gate's caution
                    # signature reads it from the decision log (the 2026-10-12 batch logged null)
                    "uncertainty": pred.get("uncertainty"),
                    "information_value": info, "cost": cost, "xy": xy, "context": context,
                    "basis": (f"{int(pred['evidence']['all_contexts'] + 0.5)} weighted outcomes; "
                              f"P({p_top[0]})={p_top[1]:.2f}; need {need:.2f}; probes {state['probes']}"),
                })
            if (att["interactions"] == 0 or blocked) and not rt.was_investigated(eid) and dist > 1.2:
                cost = self.cfg.travel_cost * dist
                iv = self.cfg.info_gain * self.cfg.investigate_value
                out.append({"activity": "investigate", "entity_id": eid, "action": None,
                            "value": iv * (0.5 + need) - cost, "expected_value": None,
                            "information_value": iv, "cost": cost, "xy": xy,
                            "basis": ("no certified interaction pose yet; look closer to map its surroundings"
                                      if blocked else "never interacted; improve identity/appearance evidence")})
        frontiers = rt.frontiers()
        if frontiers:
            goal, heading, gain, key = frontiers[0]
            dist = float(np.linalg.norm(np.asarray(goal) - pose))
            # Engineered curiosity about space, scaled by measured map-growth progress.
            progress = min(1.0, rt.map_progress / 150.0)
            info = self.cfg.info_gain * self.cfg.explore_weight * min(1.0, gain / 60.0) * progress
            cost = self.cfg.travel_cost * dist
            out.append({"activity": "explore", "entity_id": None, "action": None,
                        "value": info * (0.5 + need) - cost, "expected_value": None, "information_value": info,
                        "cost": cost, "basis": f"frontier with {gain} edge cells; recent map growth {rt.map_progress:.0f} cells/trip",
                        "frontier": frontiers[0]})
        out.append({"activity": "idle", "entity_id": None, "action": None, "value": 0.0,
                    "expected_value": 0.0, "information_value": 0.0, "cost": 0.0,
                    "basis": "baseline: do nothing" + (" (need satisfied)" if need < rt.motivation.cfg.idle_below else "")})
        return out

    def _pick(self, rt, candidates):
        forced = [c for c in candidates if c["activity"] == "avoid"]
        if forced:
            return max(forced, key=lambda c: c["value"])
        if self.policy == "operator_only":  # evaluation goal phase: only operator goals move the robot
            return candidates[-1]
        if self.policy == "explore_only":  # mapping sessions: no interactions
            explore = [c for c in candidates if c["activity"] == "explore"]
            return explore[0] if explore else candidates[-1]
        if self.policy == "learned":
            return max(candidates, key=lambda c: (c["value"], c["activity"] == "idle"))
        # Baselines choose among interaction options only (they do not use learned values).
        interactive = [c for c in candidates if c["activity"] in ("engage", "revisit")]
        if not interactive:
            others = [c for c in candidates if c["activity"] in ("explore", "investigate")]
            return max(others, key=lambda c: c["value"]) if others else candidates[-1]
        if self.policy == "random":
            return interactive[int(self.rng.integers(len(interactive)))]
        if self.policy == "nearest":
            pose = np.asarray(rt.pose[:2])
            return min(interactive, key=lambda c: (float(np.linalg.norm(c["xy"] - pose)), c["action"]))
        if self.policy == "fixed":
            # First-seen order (entity ids are random; sorting by them was not reproducible)
            ordered = sorted(interactive, key=lambda c: (c.get("created") or 0.0, c["entity_id"], c["action"]))
            return ordered[0]
        raise ValueError(f"Unknown policy {self.policy}")

    def _instantiate(self, rt, choice, now):
        kind = choice["activity"]
        if kind == "idle":
            act = Idle(None, choice["basis"])
        elif kind == "explore":
            goal, heading, gain, key = choice["frontier"]
            act = Explore(goal, heading, gain, key)
        elif kind == "investigate":
            act = Investigate(choice["entity_id"], choice["xy"], choice["basis"])
        elif kind == "avoid":
            act = Avoid(choice["entity_id"], choice["xy"], choice["basis"], hold_until=choice.get("hold_until"),
                        radius=self.cfg.avoid_radius)
            self._last_avoid = act
        else:
            decision = {k: choice[k] for k in ("activity", "value", "expected_value", "information_value", "cost",
                                               "basis")}
            decision["policy"] = self.policy
            act = Engage(choice["entity_id"], choice["action"], choice["xy"], decision,
                         revisit=kind == "revisit",
                         reason=f"{choice['action']}: expected {choice['expected_value']:+.2f} (learned), "
                                f"value {choice['value']:+.2f}")
        act.value = choice["value"]
        act.since = now
        return act

    def decide(self, rt, now, current, *, trigger=None):
        """Return a new activity or None (keep current)."""
        must = current is None or current.done
        if not must and trigger is None:
            return None
        if not must and now - self.last_decision < self.cfg.min_interval:
            return None
        while self.switch_times and now - self.switch_times[0] > 60.0:
            self.switch_times.popleft()
        candidates = self.candidates_for(rt, now)
        self.candidates = candidates
        self.last_decision = now
        if rt.pose is None:
            return None
        if not must:
            self.reconsiderations += 1
            if not current.preemptible:
                return None
            if len(self.switch_times) >= self.cfg.max_switches_per_minute:
                return None
        choice = self._pick(rt, candidates)
        if self._frustrated(rt, choice, now):
            candidates = self.candidates = self.candidates_for(rt, now)
            choice = self._pick(rt, candidates)
        if not must:
            same = (choice["activity"] == current.name or (choice["activity"] in ("engage", "revisit")
                    and current.name in ("engage", "revisit"))) and choice["entity_id"] == current.target_entity
            if same or (choice["activity"] != "avoid" and choice["value"] < current.value + self.cfg.margin):
                return None
            self.switch_times.append(now)
        self.chosen = {k: v for k, v in choice.items() if k not in ("xy", "frontier")}
        return self._instantiate(rt, choice, now)

    def _frustrated(self, rt, choice, now):
        """Frustration renewal (see ChooserConfig); True when probes were renewed."""
        cfg = self.cfg
        if cfg.frustration_renewal_s <= 0 or self.policy != "learned":
            return False
        if choice["activity"] != "idle" or rt.motivation.need < cfg.frustration_need:
            self._idle_need_since = None
            return False
        if self._idle_need_since is None:
            self._idle_need_since = now
            return False
        if now - self._idle_need_since < cfg.frustration_renewal_s:
            return False
        self._idle_need_since = None
        n = rt.memory.renew_probes(now, cfg.frustration_renewal_s)
        self.renewals.append((now, n))
        return n > 0

    def snapshot(self):
        keep = ("activity", "entity_id", "action", "value", "expected_value", "information_value", "cost", "basis",
                "uncertainty")
        return {"chosen": self.chosen, "candidates": [{k: c.get(k) for k in keep}
                                                      for c in sorted(self.candidates, key=lambda c: -c["value"])[:10]],
                "reconsiderations": self.reconsiderations, "policy": self.policy}


def initial_survey():
    return Survey(angle=2 * math.pi, rate=0.45, reason="initial look around after start")
