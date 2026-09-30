"""Screen expression as a pure, documented function of the operator state snapshot.

``expression_from(state)`` maps an ``amr_rl.state.v1`` snapshot (see
``docs/INTERFACES.md``) to an :class:`ExpressionState`. It reads nothing else:
no clock, no randomness, no simulator ground truth, no hidden mood variable.
Every output field records in ``basis`` the snapshot fields that produced it,
and ``docs/EXPRESSION.md`` tabulates the full mapping.

The face displays the robot's *operational state* (what it is doing, how sure it
is, and what it has learned about its current target). It is not an emotion
model. ``liked``/``disliked``/``indifferent`` are only shown for an entity whose
memory entry records at least one completed interaction; before that the
attitude is forced to ``unknown``.

Rule order (first match wins for ``face``):

1. ``localization.status`` in ``lost``/``relocalizing`` -> ``confused``.
2. ``authority.stopped`` and activity is not ``manual`` -> ``stopped``.
3. ``activity.name`` table (idle, explore, investigate, engage, revisit, avoid,
   recovering, manual, stopped); missing or unrecognised -> ``neutral``.

A missing ``localization`` block (or ``position_sigma: null``) does not change
the face; it sets uncertainty to 1 and centres the gaze.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

FACES = (
    "neutral",
    "curious",
    "focused",
    "happy",
    "wary",
    "sleepy",
    "confused",
    "stopped",
    "signalling",
)
ATTITUDES = ("liked", "disliked", "indifferent", "unknown", "none")
LEARNED_ATTITUDES = ("liked", "disliked", "indifferent")

#: Eyelid opening per face (engineered display constants, not state).
OPENNESS = {
    "neutral": 0.90,
    "curious": 1.00,
    "focused": 0.72,
    "happy": 0.90,
    "wary": 0.45,
    "sleepy": 0.35,
    "confused": 0.85,
    "stopped": 0.10,
    "signalling": 0.90,
}

#: ``position_sigma`` (m) at which the localization part of uncertainty saturates.
SIGMA_FULL_SCALE = 0.15
#: Target bearing (rad) at which horizontal gaze saturates.
GAZE_FULL_SCALE = 0.65
#: Interaction statuses during which a ``signal`` action is being emitted.
SIGNAL_STATUSES = ("acting", "signalling")
CAPTION_MAX = 28

_ENGAGE_FACE = {
    "liked": "happy",
    "disliked": "wary",
    "unknown": "curious",
    "indifferent": "neutral",
    "none": "curious",
}
_ACTIVITY_FACE = {
    "idle": "sleepy",
    "explore": "curious",
    "investigate": "focused",
    "avoid": "wary",
    "recovering": "confused",
    "manual": "neutral",
    "stopped": "stopped",
}
_CAPTION_VERB = {
    "investigate": "Inspecting",
    "engage": "Engaging",
    "revisit": "Revisiting",
    "avoid": "Avoiding",
}
_CAPTION_PLAIN = {
    "idle": "Resting",
    "explore": "Exploring",
    "investigate": "Investigating",
    "engage": "Engaging",
    "revisit": "Revisiting",
    "avoid": "Avoiding",
    "recovering": "Recovering",
    "manual": "Manual drive",
    "stopped": "Stopped",
}


@dataclass(frozen=True)
class ExpressionState:
    """What the robot's screen shows, and which state fields caused it."""

    face: str = "neutral"
    #: (x, y) in [-1, 1], robot frame: +x = toward the robot's right (the viewer's
    #: left on the outward-facing screen), +y = up. ``x = -clamp(bearing / 0.65)``.
    gaze: tuple[float, float] = (0.0, 0.0)
    openness: float = OPENNESS["neutral"]
    uncertainty: float = 1.0
    attitude: str = "none"
    caption: str = ""
    signal_pattern: bool = False
    basis: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for ``state["expression"]``."""
        return {
            "face": self.face,
            "gaze": [round(self.gaze[0], 4), round(self.gaze[1], 4)],
            "openness": round(self.openness, 4),
            "uncertainty": round(self.uncertainty, 4),
            "attitude": self.attitude,
            "caption": self.caption,
            "signal_pattern": self.signal_pattern,
            "basis": {k: list(v) for k, v in self.basis.items()},
        }


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else min(x, hi)


def _num(x: Any) -> float | None:
    """Finite float or None (bools and non-numbers are rejected)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _dict(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _find_entity(state: dict, entity_id: Any) -> dict | None:
    if entity_id is None:
        return None
    for ent in state.get("entities") or ():
        if isinstance(ent, dict) and ent.get("entity_id") == entity_id:
            return ent
    return None


def learned_attitude(entity: dict | None) -> str:
    """Attitude to display for an entity entry, gated on learned outcomes.

    ``none`` without an entity; ``unknown`` unless the entry reports a learned
    attitude *and* ``interactions >= 1``.
    """
    if entity is None:
        return "none"
    att = entity.get("attitude")
    n = _num(entity.get("interactions"))
    if att in LEARNED_ATTITUDES and n is not None and n >= 1:
        return att
    return "unknown"


def _bearing(pose: Any, position: Any) -> float | None:
    """CCW-positive bearing (rad, wrapped to [-pi, pi]) of ``position`` from ``pose``."""
    if not isinstance(pose, (list, tuple)) or len(pose) < 3:
        return None
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    x, y, th = (_num(v) for v in pose[:3])
    ex, ey = (_num(v) for v in position[:2])
    if None in (x, y, th, ex, ey):
        return None
    if math.hypot(ex - x, ey - y) < 1e-6:
        return None
    b = math.atan2(ey - y, ex - x) - th
    return math.atan2(math.sin(b), math.cos(b))


def _caption(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= CAPTION_MAX:
        return text
    return text[: CAPTION_MAX - 3].rstrip() + "..."


def expression_from(state: dict | None) -> ExpressionState:
    """Map an ``amr_rl.state.v1`` snapshot to the screen expression (pure).

    Missing or ``None`` fields degrade gracefully: unknown localization gives
    uncertainty 1, a missing activity gives ``neutral``, a missing target gives
    centred gaze and attitude ``none``.
    """
    state = _dict(state)
    authority = _dict(state.get("authority"))
    loc = _dict(state.get("localization"))
    activity = _dict(state.get("activity"))
    interaction = _dict(state.get("interaction"))

    name = activity.get("name") if isinstance(activity.get("name"), str) else None
    target_id = activity.get("target_entity")
    target = _find_entity(state, target_id)
    status = loc.get("status")
    stopped = authority.get("stopped") is True
    basis: dict[str, list[str]] = {}

    # --- attitude -------------------------------------------------------
    attitude = learned_attitude(target)
    basis["attitude"] = (
        ["activity.target_entity", "entities[target].attitude", "entities[target].interactions"]
        if target is not None
        else ["activity.target_entity"]
    )

    # --- signal pattern: exactly tied to a real, perceivable action -----
    signal = interaction.get("action") == "signal" and interaction.get("status") in SIGNAL_STATUSES
    basis["signal_pattern"] = ["interaction.action", "interaction.status"]

    # --- face -----------------------------------------------------------
    lost = status in ("lost", "relocalizing")
    if lost:
        face, fb = "confused", ["localization.status"]
    elif stopped and name != "manual":
        face, fb = "stopped", ["authority.stopped", "activity.name"]
    elif name == "engage":
        if signal:
            face, fb = "signalling", ["activity.name", "interaction.action", "interaction.status"]
        else:
            face = _ENGAGE_FACE[attitude]
            fb = ["activity.name"] + basis["attitude"]
    elif name == "revisit":
        face = "happy" if attitude == "liked" else "curious"
        fb = ["activity.name"] + basis["attitude"]
    elif name in _ACTIVITY_FACE:
        face, fb = _ACTIVITY_FACE[name], ["activity.name"]
    else:
        face, fb = "neutral", ["activity.name"]
    basis["face"] = fb
    basis["openness"] = ["face"]

    # --- uncertainty ----------------------------------------------------
    sigma = _num(loc.get("position_sigma"))
    if not loc or lost or sigma is None:
        unc = 1.0
        basis["uncertainty"] = ["localization.status", "localization.position_sigma"]
    else:
        unc = _clamp(sigma / SIGMA_FULL_SCALE)
        basis["uncertainty"] = ["localization.position_sigma"]
    if target is not None:
        tu = _num(target.get("uncertainty"))
        if tu is not None:
            unc = max(unc, _clamp(tu))
            basis["uncertainty"].append("entities[target].uncertainty")
        if target.get("ambiguous") is True:
            conf = _num(target.get("identity_confidence"))
            unc = max(unc, 1.0 - _clamp(conf) if conf is not None else 1.0)
            basis["uncertainty"] += ["entities[target].ambiguous",
                                     "entities[target].identity_confidence"]

    # --- gaze -----------------------------------------------------------
    gx = 0.0
    basis["gaze"] = []
    if target is not None and not lost:
        b = _bearing(loc.get("pose"), target.get("position"))
        if b is not None:
            # CCW-positive bearing = target on the robot's left. gaze_x is
            # robot-right-positive, hence the negation. The renderer mirrors it
            # for the outward-facing screen, so the pupils move to the viewer's
            # right, i.e. physically toward the target.
            gx = -_clamp(b / GAZE_FULL_SCALE, -1.0, 1.0)
            basis["gaze"] = ["localization.pose", "activity.target_entity",
                             "entities[target].position"]

    # --- caption --------------------------------------------------------
    label = None
    if target is not None:
        label = target.get("label") or target.get("entity_id")
    elif target_id is not None:
        label = str(target_id)
    if lost:
        cap = "Lost - stopped" if stopped else (
            "Relocalizing" if status == "relocalizing" else "Lost")
        basis["caption"] = ["localization.status", "authority.stopped"]
    elif face == "stopped":
        reason = authority.get("revoked_reason")
        cap = f"Stopped ({str(reason).replace('_', ' ')})" if reason and reason != "stop" \
            else "Stopped"
        basis["caption"] = ["authority.stopped", "authority.revoked_reason"]
    elif not loc and name is None:
        cap = "No state"
        basis["caption"] = ["localization", "activity.name"]
    else:
        if face == "signalling":
            cap = f"Signalling {label}" if label else "Signalling"
        elif label and name in _CAPTION_VERB:
            cap = f"{_CAPTION_VERB[name]} {label}"
        elif name in _CAPTION_PLAIN:
            cap = _CAPTION_PLAIN[name]
        elif name:
            cap = name.replace("_", " ").capitalize()
        else:
            cap = "Waiting"
        basis["caption"] = ["activity.name", "activity.target_entity", "entities[target].label"]

    return ExpressionState(
        face=face,
        gaze=(gx, 0.0),
        openness=OPENNESS[face],
        uncertainty=float(unc),
        attitude=attitude,
        caption=_caption(cap),
        signal_pattern=bool(signal),
        basis=basis,
    )
