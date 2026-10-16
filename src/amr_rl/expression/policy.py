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
   blocked, recovering, manual, stopped); missing or unrecognised -> ``neutral``.

A missing ``localization`` block (or ``position_sigma: null``) does not change
the face; it sets uncertainty to 1 and centres the gaze.

Style (one latent expressive state, docs/research/PERSONALITY_RESEARCH_2.md section 2):
``activation`` is the chooser's engineered stimulation need as logged in the snapshot;
``tempo`` is a fixed monotone function of it, rescaled by the approach-tempo trait only
when the character is active (the same gain that scales the cruise speed, so face and
motion share one tempo), and clamped to 1 for the wary / stopped / confused faces (no
eager idiom while avoiding or stopped). The blink interval follows 1/tempo. ``cue`` is the
transparency cue shown for the first seconds after a boot, templated from logged fields
only (session count, remembered objects, the boot event, a character change since the
previous boot).
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
#: Style constants (engineered): tempo = (TEMPO_BASE + TEMPO_NEED_GAIN x activation) x trait gain,
#: so a need of 0.5 without a character gives exactly 1; the trait gain is capped like the
#: navigator's speed gain; blink interval = BLINK_BASE_S / tempo.
TEMPO_BASE = 0.75
TEMPO_NEED_GAIN = 0.5
TEMPO_TRAIT_CAP = (0.75, 1.25)
BLINK_BASE_S = 4.0
#: Faces during which the tempo is clamped to 1 (context check: no eager idiom).
CALM_FACES = ("wary", "stopped", "confused")
#: Seconds after a boot during which the transparency cue is shown.
BOOT_CUE_S = 15.0

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
    "blocked": "wary",  # backing off from something that stopped the drive (pose kept)
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
    "blocked": "Blocked - backing off",
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
    #: Style: the one latent expressive state and its fixed monotone mappings.
    activation: float = 0.5
    tempo: float = 1.0
    blink_interval_s: float = BLINK_BASE_S
    #: Transparency cue after a boot (logged fields only); empty otherwise.
    cue: str = ""
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
            "style": {"activation": round(self.activation, 4), "tempo": round(self.tempo, 4),
                      "blink_interval_s": round(self.blink_interval_s, 3)},
            "cue": self.cue,
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


def _trait_value(character: dict, key: str) -> float | None:
    """A trait's stored value, unless the trait is retired (logged, acting on nothing: no cue
    reads it; the inspector marks the row)."""
    for row in character.get("traits") or ():
        if isinstance(row, dict) and row.get("key") == key:
            return None if row.get("retired") else _num(row.get("value"))
    return None


def _style(state: dict, face: str, basis: dict) -> tuple[float, float, float]:
    """(activation, tempo, blink interval): one latent state from the logged stimulation
    need, a fixed monotone mapping, the approach-tempo trait as a gain only when the
    character is active, and the context clamp for calm faces."""
    need = _num(_dict(state.get("motivation")).get("stimulation_need"))
    activation = _clamp(need) if need is not None else 0.5
    basis["activation"] = ["motivation.stimulation_need"]
    character = _dict(state.get("character"))
    gain, gb = 1.0, ["character.active"]
    if character.get("active") is True:
        value = _trait_value(character, "approach_tempo")
        if value is not None:
            gain = _clamp(2.0 ** (2.0 * (_clamp(value) - 0.5)), *TEMPO_TRAIT_CAP)
            gb.append("character.traits[approach_tempo].value")
    tempo = (TEMPO_BASE + TEMPO_NEED_GAIN * activation) * gain
    basis["tempo"] = ["activation"] + gb
    if face in CALM_FACES:
        tempo = min(tempo, 1.0)
        basis["tempo"].append("face")
    basis["blink_interval_s"] = ["tempo"]
    return activation, tempo, BLINK_BASE_S / tempo


def _boot_cue(state: dict, basis: dict) -> str:
    """The transparency cue for the first BOOT_CUE_S seconds after the latest boot, from
    logged fields only; '' outside that window or without the fields."""
    now = _num(state.get("sim_time"))
    character = _dict(state.get("character"))
    boots = [b for b in character.get("boots") or () if isinstance(b, dict) and _num(b.get("t")) is not None]
    basis["cue"] = ["sim_time", "character.boots"]
    if now is None or not boots:
        return ""
    boots.sort(key=lambda b: _num(b["t"]))
    last = boots[-1]
    if not 0.0 <= now - _num(last["t"]) <= BOOT_CUE_S:
        return ""
    parts = [f"Session {last.get('session')}" if last.get("session") is not None else "New session"]
    objects = _num(_dict(state.get("memory")).get("entities"))
    if objects is not None:
        parts.append(f"{int(objects)} objects remembered")
        basis["cue"].append("memory.entities")
    event = last.get("event")
    if isinstance(event, str) and event != "restart":
        parts.append(event.replace("_", " ").replace(":", " "))
    previous = _num(boots[-2]["t"]) if len(boots) > 1 else None
    changes = [e for e in character.get("log") or () if isinstance(e, dict) and e.get("cause") != "created"
               and _num(e.get("t")) is not None and (previous is None or _num(e["t"]) >= previous)]
    if changes:
        parts.append("character changed")
        basis["cue"].append("character.log")
    return " · ".join(parts)


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
    elif isinstance(loc.get("odometry"), dict) and loc["odometry"].get("zupt_hold_active") is True:
        # standing still on purpose to re-measure the gyro offset (round 9): not a freeze
        cap = "Calibrating gyro"
        basis["caption"] = ["localization.odometry.zupt_hold_active"]
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

    # --- style and the boot cue -----------------------------------------
    activation, tempo, blink = _style(state, face, basis)
    cue = _boot_cue(state, basis)

    return ExpressionState(
        face=face,
        gaze=(gx, 0.0),
        openness=OPENNESS[face],
        uncertainty=float(unc),
        attitude=attitude,
        caption=_caption(cap),
        signal_pattern=bool(signal),
        activation=float(activation),
        tempo=float(tempo),
        blink_interval_s=float(blink),
        cue=cue,
        basis=basis,
    )
