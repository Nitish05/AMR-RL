"""Entity identity maintenance from onboard detections.

Identity is asserted only where evidence supports it:

* appearance (hue, fill, metric size) must match a remembered entity, and
* when the entity has a remembered position in the current map version, the
  detection must be near it (a larger gate once the entity was observed to move).

Two remembered entities that both fit -> AMBIGUOUS (no learning attribution).
A remembered position that is in view but empty, plus exactly one matching
detection elsewhere -> the entity is treated as relocated (lower confidence).
New appearances become entities only after several consistent observations.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

import numpy as np

from ..perception.entities import hue_distance


@dataclass
class IdentityConfig:
    appearance_gate: float = 1.6
    position_gate: float = 0.45
    movable_gate: float = 2.5
    promote_after: int = 3
    tentative_gate: float = 0.3
    max_range: float = 3.2


@dataclass
class VisibleEntity:
    detection: object
    entity_id: str | None
    confidence: float
    ambiguous: bool
    reason: str
    candidates: list = field(default_factory=list)


def appearance_distance(det_app: dict, mem_app: dict) -> float:
    d = hue_distance(det_app["hue"], mem_app["hue"]) / 18.0
    d += abs(det_app["fill"] - mem_app["fill"]) / 0.3
    for key, scale in (("width_m", 0.07), ("height_m", 0.07)):
        a, b = det_app.get(key), mem_app.get(key)
        if a is not None and b is not None:
            d += abs(a - b) / scale
    return d


class EntityTracker:
    def __init__(self, memory, config: IdentityConfig | None = None):
        self.memory = memory
        self.cfg = config or IdentityConfig()
        self.tentative: list[dict] = []
        self.visible: list[VisibleEntity] = []
        self.last_seen: dict[str, float] = {}
        self.last_state: dict[str, str] = {}
        self._entities = None

    def _load(self):
        if self._entities is None:
            self._entities = {e["entity_id"]: e for e in self.memory.entities()}
        return self._entities

    def refresh(self):
        self._entities = None

    def update(self, detections, pose, pose_sigma, map_version, now, *, in_view=None):
        """``in_view(xy) -> bool`` tells whether a map point is currently observable
        (inside the image and range); used to detect 'remembered place is empty'."""
        entities = self._load()
        results = []
        usable = [d for d in detections if d.position is not None and d.range_m is not None
                  and d.range_m <= self.cfg.max_range and not d.partial]
        claimed = {}
        # Entities explained by a position-consistent detection in this frame cannot
        # also explain a far-away look-alike (that one may be a twin).
        consistent_now = set()
        for det in usable:
            for eid, ent in entities.items():
                if (ent.get("x") is not None and ent.get("map_version") == map_version
                        and appearance_distance(det.descriptor(), ent["appearance"]) <= self.cfg.appearance_gate
                        and math.dist(det.position, (ent["x"], ent["y"])) <= self.cfg.position_gate):
                    consistent_now.add(eid)
        for det in usable:
            app = det.descriptor()
            scored = []
            for eid, ent in entities.items():
                dist = appearance_distance(app, ent["appearance"])
                if dist > self.cfg.appearance_gate:
                    continue
                pos_known = ent.get("x") is not None and ent.get("map_version") == map_version
                gate = self.cfg.movable_gate if ent.get("movable") else self.cfg.position_gate
                gate += (pose_sigma or 0.0) + (ent.get("pos_sigma") or 0.0)
                pos_d = math.dist(det.position, (ent["x"], ent["y"])) if pos_known else None
                scored.append((eid, dist, pos_known, pos_d, gate))
            consistent = [s for s in scored if s[2] and s[3] <= s[4]]
            unplaced = [s for s in scored if not s[2]]
            displaced = [s for s in scored if s[2] and s[3] > s[4] and s[0] not in consistent_now]
            if len(consistent) == 1:
                eid, dist, _, pos_d, gate = consistent[0]
                conf = max(0.5, 1.0 - 0.25 * dist - 0.3 * pos_d / gate)
                results.append(VisibleEntity(det, eid, conf, False, "appearance+position", [s[0] for s in scored]))
            elif len(consistent) > 1:
                results.append(VisibleEntity(det, None, 0.0, True, "several remembered entities fit",
                                             [s[0] for s in consistent]))
            elif len(unplaced) == 1 and not displaced:
                eid, dist = unplaced[0][:2]
                results.append(VisibleEntity(det, eid, max(0.4, 0.75 - 0.2 * dist), False,
                                             "appearance only (no position in this map)", [eid]))
            elif len(unplaced) > 1:
                results.append(VisibleEntity(det, None, 0.0, True, "several unplaced entities fit",
                                             [s[0] for s in unplaced]))
            elif len(displaced) == 1 and not unplaced:
                eid = displaced[0][0]
                ent = entities[eid]
                empty = in_view is not None and in_view((ent["x"], ent["y"])) and not any(
                    math.dist(d.position, (ent["x"], ent["y"])) < self.cfg.position_gate for d in usable)
                if empty or ent.get("movable"):
                    results.append(VisibleEntity(det, eid, 0.65, False, "relocated (old place seen empty)"
                                                 if empty else "movable entity relocated", [eid]))
                else:
                    results.append(VisibleEntity(det, None, 0.0, True,
                                                 "matches an entity remembered elsewhere", [eid]))
            elif len(displaced) > 1:
                results.append(VisibleEntity(det, None, 0.0, True, "matches several remembered entities",
                                             [s[0] for s in displaced]))
            else:
                results.append(self._tentative(det, map_version, now))
        # One detection per entity per frame: duplicates become ambiguous.
        for item in results:
            if item.entity_id is not None:
                claimed.setdefault(item.entity_id, []).append(item)
        for items in claimed.values():
            if len(items) > 1:
                for item in items:
                    item.entity_id, item.ambiguous, item.confidence = None, True, 0.0
                    item.reason = "two detections claim one identity"
        entities = self._load()
        for item in results:
            if item.entity_id is None:
                continue
            det = item.detection
            ent = entities[item.entity_id]
            app = _blend(ent["appearance"], det.descriptor(), 0.2)
            quality = det.range_m < 2.0
            self.memory.upsert_entity(item.entity_id, appearance=app, now=now, map_version=map_version,
                                      xy=det.position if quality or ent.get("x") is None else None,
                                      pos_sigma=0.05 + 0.03 * det.range_m + (pose_sigma or 0.0))
            ent.update({"appearance": app, "map_version": map_version})
            if quality or ent.get("x") is None:
                ent.update({"x": det.position[0], "y": det.position[1],
                            "pos_sigma": 0.05 + 0.03 * det.range_m})
            self.last_seen[item.entity_id] = now
            self.last_state[item.entity_id] = det.state_token
        self.visible = results
        return results

    def _tentative(self, det, map_version, now):
        app = det.descriptor()
        for cand in self.tentative:
            if (appearance_distance(app, cand["appearance"]) < self.cfg.appearance_gate
                    and math.dist(det.position, cand["xy"]) < self.cfg.tentative_gate):
                cand["count"] += 1
                cand["xy"] = tuple(0.7 * np.array(cand["xy"]) + 0.3 * np.array(det.position))
                cand["appearance"] = _blend(cand["appearance"], app, 0.3)
                cand["last"] = now
                if cand["count"] >= self.cfg.promote_after:
                    self.tentative.remove(cand)
                    near = [e for e in self._load().values() if e.get("x") is not None and
                            e.get("map_version") == map_version and
                            appearance_distance(cand["appearance"], e["appearance"]) < 1.5 * self.cfg.appearance_gate
                            and math.dist(cand["xy"], (e["x"], e["y"])) < 0.9]
                    if near:
                        return VisibleEntity(det, None, 0.0, True, "resembles a nearby remembered entity",
                                             [e["entity_id"] for e in near])
                    eid = f"ent-{uuid.uuid4().hex[:6]}"
                    self.memory.upsert_entity(eid, appearance=cand["appearance"], now=now, map_version=map_version,
                                              xy=cand["xy"], pos_sigma=0.08, label=f"{det.color} object")
                    self.refresh()
                    return VisibleEntity(det, eid, 0.8, False, "new entity (consistent observations)", [eid])
                return VisibleEntity(det, None, 0.2, False, "tentative new entity", [])
        self.tentative = [c for c in self.tentative if now - c["last"] < 5.0][-20:]
        self.tentative.append({"appearance": app, "xy": tuple(det.position), "count": 1, "last": now})
        return VisibleEntity(det, None, 0.1, False, "tentative new entity", [])


def _blend(old: dict, new: dict, rate: float) -> dict:
    out = dict(old)
    for key in ("fill", "width_m", "height_m", "saturation"):
        if new.get(key) is not None:
            out[key] = new[key] if old.get(key) is None else (1 - rate) * old[key] + rate * new[key]
    a, b = math.radians(old["hue"]), math.radians(new["hue"])
    out["hue"] = math.degrees(math.atan2((1 - rate) * math.sin(a) + rate * math.sin(b),
                                         (1 - rate) * math.cos(a) + rate * math.cos(b))) % 360
    out["color"] = new.get("color", old.get("color"))
    return out
