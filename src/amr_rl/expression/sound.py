"""The speaker as a one-way sound event bus (SonifyIt pattern), fed by logged state
transitions in the operator snapshot and nothing else (docs/research/PERSONALITY_RESEARCH_2.md
section 5; owner decision 2026-10-12: add a speaker).

Design, all engineered:

* a small discrete cue set (eight cues), one per visible state transition, plus one
  low-salience ``uncertain`` cue when an approach is a probe or confidence is low; nothing is
  mapped continuously (interpretation of robot sounds is categorical);
* every cue pairs with a visible event and records which snapshot fields changed
  (``paired_with``); sound never carries information the face and the action do not;
* fixed intensity per cue, never scaled by surprise, by any trait or by anything else (louder
  and harsher sounds worsen perception; a loud aversive cue reads as the robot being hurt);
* a global emission budget (minimum gap and cues per minute); dropped cues are counted;
* a transit layer: on/off markers time-locked to the commanded motion of the same tick, with a
  pulse rate that follows the commanded speed (logged, not a cue);
* a per-character acoustic signature through tempo, pitch variability and density only (not
  pitch or loudness), from the expression's style tempo and, when the character is active, the
  novelty-bias and approach-tempo traits; otherwise neutral;
* in simulation the speaker is a logged virtual emission; ``render_wav`` turns a log into audio
  for evaluation with one procedural synthesiser so the cues share a timbre.

The bus reads the snapshot and the commanded motion; it writes only its own log. Control,
learning and the experience memory are byte-identical with the subscriber on or off, and
nothing else (in particular no language model) has a path to it.
"""

from __future__ import annotations

import copy
import math
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np

CUES = ("explore_start", "approach", "action", "outcome_positive", "outcome_aversive", "avoid",
        "idle_long", "attention")
SUBTLE = "uncertain"
TRANSIT = ("transit_on", "transit_off")
#: Fixed intensity per cue (0..1 amplitude); never modulated.
INTENSITY = {"explore_start": 0.5, "approach": 0.45, "action": 0.5, "outcome_positive": 0.6,
             "outcome_aversive": 0.45, "avoid": 0.5, "idle_long": 0.3, "attention": 0.55,
             SUBTLE: 0.2, "transit_on": 0.15, "transit_off": 0.15}
#: Cues that are synchronised to a visible flag and therefore skip the minimum gap (not the budget).
SYNCHRONISED = ("outcome_positive", "outcome_aversive")


@dataclass
class SoundConfig:
    enabled: bool = True
    min_gap_s: float = 0.8
    budget_per_min: int = 12
    idle_long_s: float = 20.0
    transit_min_speed: float = 0.02
    uncertain_threshold: float = 0.3  # chosen option's posterior uncertainty at or above which an approach is a probe


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _dict(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else min(x, hi)


def signature_from(state: dict | None) -> dict:
    """The acoustic signature: tempo from the expression's style (already trait-scaled only
    when the character is active), pitch variability from novelty bias and density from
    approach tempo, both neutral (0.5) unless the character is active. No pitch, no loudness."""
    state = _dict(state)
    style = _dict(_dict(state.get("expression")).get("style"))
    tempo = _num(style.get("tempo"))
    sig = {"tempo": tempo if tempo is not None else 1.0, "pitch_variability": 0.5, "density": 0.5,
           "basis": ["expression.style.tempo"]}
    character = _dict(state.get("character"))
    if character.get("active") is True:
        values = {r.get("key"): _num(r.get("value")) for r in character.get("traits") or () if isinstance(r, dict)}
        if values.get("novelty_bias") is not None:
            sig["pitch_variability"] = _clamp(values["novelty_bias"])
            sig["basis"].append("character.traits[novelty_bias].value")
        if values.get("approach_tempo") is not None:
            sig["density"] = _clamp(values["approach_tempo"])
            sig["basis"].append("character.traits[approach_tempo].value")
    sig["basis"].append("character.active")
    return sig


def transit_rate_hz(speed: float) -> float:
    """Pulse rate of the transit layer: a fixed monotone function of commanded speed."""
    return 2.0 + 6.0 * _clamp(abs(speed) / 0.25)


class SoundBus:
    """Subscriber to the operator snapshot; emits logged virtual sound events."""

    def __init__(self, config: SoundConfig | None = None):
        self.cfg = config or SoundConfig()
        self.log: list[dict] = []
        self.dropped: dict[str, int] = {}
        self._prev: dict[str, Any] = {}
        self._idle_flagged: float | None = None
        self._last_emit: float = -math.inf
        self._recent: list[float] = []
        self.transit = {"on": False, "rate_hz": 0.0, "since": None}

    # ------------------------------------------------------------ observe
    def observe(self, state: dict | None, command: tuple[float, float] | None, now: float) -> list[dict]:
        """Read one snapshot and the commanded (v, w) of the same tick; return this tick's
        emissions (also appended to ``log``). The snapshot is not modified."""
        if not self.cfg.enabled:
            return []
        state = _dict(state)
        activity = _dict(state.get("activity"))
        loc = _dict(state.get("localization"))
        interaction = _dict(state.get("interaction"))
        outcomes = state.get("recent_outcomes") or []
        latest = _dict(outcomes[0]) if outcomes else {}
        chosen = _dict(_dict(state.get("decision")).get("chosen"))
        name = activity.get("name") if isinstance(activity.get("name"), str) else None
        status = loc.get("status")
        acting = interaction.get("status") == "acting"
        prev = self._prev
        sig = signature_from(state)
        wanted: list[tuple[str, list[str]]] = []

        if prev and name != prev.get("name"):
            if name == "explore":
                wanted.append(("explore_start", ["activity.name"]))
            elif name in ("engage", "revisit"):
                wanted.append(("approach", ["activity.name", "activity.target_entity"]))
                unc = _num(chosen.get("uncertainty"))
                if unc is None:
                    unc = _num(_dict(chosen.get("prediction")).get("uncertainty"))
                if unc is not None and unc >= self.cfg.uncertain_threshold:
                    wanted.append((SUBTLE, ["decision.chosen.uncertainty"]))
            elif name == "avoid":
                wanted.append(("avoid", ["activity.name", "activity.target_entity"]))
            elif name in ("blocked", "recovering"):
                wanted.append(("attention", ["activity.name"]))
        if prev and status in ("lost", "relocalizing") and prev.get("status") not in ("lost", "relocalizing"):
            wanted.append(("attention", ["localization.status"]))
        if prev and acting and not prev.get("acting"):
            wanted.append(("action", ["interaction.action", "interaction.status"]))
        event_id = latest.get("event_id")
        if prev and event_id is not None and event_id != prev.get("event_id"):
            valence = _num(latest.get("valence"))
            if valence is not None and valence > 0:
                wanted.append(("outcome_positive", ["recent_outcomes[0].event_id", "recent_outcomes[0].valence"]))
            elif valence is not None and valence < 0:
                wanted.append(("outcome_aversive", ["recent_outcomes[0].event_id", "recent_outcomes[0].valence"]))
        since = _num(activity.get("since"))
        if name == "idle" and since is not None and now - since >= self.cfg.idle_long_s and self._idle_flagged != since:
            self._idle_flagged = since
            wanted.append(("idle_long", ["activity.name", "activity.since"]))
        elif name != "idle":
            self._idle_flagged = None

        emitted = []
        for cue, paired in wanted:
            self._recent = [t for t in self._recent if now - t < 60.0]
            if len(self._recent) >= self.cfg.budget_per_min:
                self.dropped[cue] = self.dropped.get(cue, 0) + 1
                continue
            if cue not in SYNCHRONISED and now - self._last_emit < self.cfg.min_gap_s:
                self.dropped[cue] = self.dropped.get(cue, 0) + 1
                continue
            emitted.append({"t": float(now), "cue": cue, "paired_with": paired, "intensity": INTENSITY[cue],
                            "signature": {k: v for k, v in sig.items() if k != "basis"}})
            self._last_emit = now
            self._recent.append(now)

        # transit layer: time-locked to the command of this tick
        v = _num(command[0]) if isinstance(command, (tuple, list)) and command else None
        moving = v is not None and abs(v) >= self.cfg.transit_min_speed
        if moving != self.transit["on"]:
            cue = "transit_on" if moving else "transit_off"
            emitted.append({"t": float(now), "cue": cue, "paired_with": ["command.v"], "intensity": INTENSITY[cue],
                            "signature": {k: v2 for k, v2 in sig.items() if k != "basis"}})
            self.transit = {"on": moving, "rate_hz": transit_rate_hz(v or 0.0) if moving else 0.0, "since": float(now)}
        elif moving:
            self.transit["rate_hz"] = transit_rate_hz(v or 0.0)

        self.log.extend(emitted)
        self.log = self.log[-500:]
        self._prev = {"name": name, "status": status, "acting": acting, "event_id": event_id}
        return emitted

    def snapshot(self) -> dict:
        return {"enabled": self.cfg.enabled, "emissions": len(self.log), "last": self.log[-1] if self.log else None,
                "dropped": dict(self.dropped), "transit": dict(self.transit), "cues": list(CUES),
                "reads": "operator snapshot and the commanded motion; writes only its own log"}


# ---------------------------------------------------------------- virtual speaker rendering
# One procedural synthesiser: a cue is a few sine notes with an exponential decay. The
# signature changes only the note length (tempo), the interval spread (pitch variability) and
# the note count (density); the amplitude is the cue's fixed intensity.
_CUE_NOTES = {
    "explore_start": (523.0, [0, 4]), "approach": (440.0, [0]), "action": (1046.0, [0]),
    "outcome_positive": (880.0, [0, 7]), "outcome_aversive": (110.0, [0]), "avoid": (660.0, [0, -5]),
    "idle_long": (220.0, [0]), "attention": (740.0, [0, 3, 0]), SUBTLE: (330.0, [0, -2]),
    "transit_on": (300.0, [0]), "transit_off": (300.0, [0]),
}


def _note(freq, seconds, amp, sr):
    t = np.arange(int(seconds * sr)) / sr
    env = np.exp(-4.0 * t / max(seconds, 1e-3))
    return amp * env * np.sin(2 * math.pi * freq * t)


def render_wav(emissions: list[dict], path=None, sample_rate: int = 22050) -> np.ndarray:
    """Render a log of emissions to mono float samples (and a 16-bit WAV when ``path`` is
    given). Evaluation tooling: forced-choice recognition from sound alone, annoyance over a
    session. Deterministic."""
    emissions = [e for e in emissions if e.get("cue") in _CUE_NOTES]
    if not emissions:
        out = np.zeros(sample_rate // 10, dtype=np.float32)
    else:
        t0 = min(float(e["t"]) for e in emissions)
        end = max(float(e["t"]) for e in emissions) - t0 + 1.5
        out = np.zeros(int(end * sample_rate) + 1, dtype=np.float32)
        for e in emissions:
            sig = _dict(e.get("signature"))
            tempo = max(0.5, float(sig.get("tempo", 1.0) or 1.0))
            spread = 0.5 + float(sig.get("pitch_variability", 0.5) or 0.5)  # 0.5..1.5 x the interval
            density = float(sig.get("density", 0.5) or 0.5)
            base, intervals = _CUE_NOTES[e["cue"]]
            repeats = 1 + int(round(density * 2))  # 1..3
            notes = [base * 2 ** (i * spread / 12.0) for i in intervals] * repeats
            length = 0.09 / tempo
            amp = float(INTENSITY.get(e["cue"], 0.3))
            start = int((float(e["t"]) - t0) * sample_rate)
            for k, f in enumerate(notes):
                s = start + int(k * length * 1.1 * sample_rate)
                n = _note(f, length, amp, sample_rate)
                seg = out[s:s + len(n)]
                seg += n[:len(seg)]
        out = np.clip(out, -1.0, 1.0)
    if path is not None:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes((out * 32767).astype("<i2").tobytes())
    return out


def unchanged(state_before: dict, state_after: dict) -> bool:
    """Evaluation helper: the bus must not have modified the snapshot."""
    return copy.deepcopy(state_before) == state_after


__all__ = ["CUES", "SUBTLE", "TRANSIT", "INTENSITY", "SoundConfig", "SoundBus", "signature_from",
           "transit_rate_hz", "render_wav", "unchanged"]
