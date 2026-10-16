"""Pip's character store: the engineered temperament and learning-style parameters, their
bounds and their provenance, persisted with the experience memory (owner decisions of
2026-10-12; docs/research/PERSONALITY_RESEARCH_2.md).

What is ENGINEERED: every trait's default, its bounds, the per-write caps and the rules in
``clip_write``. What is LEARNED: nothing yet. A later consolidation rule may move a value
from logged experience ("learned via an engineered update"), and an accepted, bounded,
logged LLM proposal may nudge one ("llm_advised"); the provenance of the last write is
stored with each value, and every change is logged with its cause. Traits are named by the
behaviour they set, never by an inner state; a value of 0.5 means "the behaviour the robot
had before any trait existed". ``effective`` is the whole mapping from trait values to
behaviour parameters; the runtime applies it only with ``RuntimeConfig.character_active``
(off until the identifiability gate, scripts/eval/character_gate.py, has passed), and the
inspector says which.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Who wrote a value. Only the first two may set a value anywhere in its bounds; the others
# are capped per write (origin-bound authority) and can never change bounds or anchors.
PROVENANCE = ("engineered_default", "owner", "consolidation", "llm_advised", "human_rating", "migration")
CAPPED_PROVENANCE = ("consolidation", "llm_advised", "human_rating")


@dataclass(frozen=True)
class TraitSpec:
    key: str
    sets: str        # the behaviour parameters the trait scales (see ``effective``)
    analogue: str    # the human trait it is loosely named after; informational only
    default: float = 0.5
    lo: float = 0.0
    hi: float = 1.0
    retired: str = ""  # why the trait no longer acts on behaviour; its row stays in the store for continuity


TRAITS = (
    TraitSpec("approach_tempo", "gain on the navigator's cruise speed (capped) and, inversely, on the chooser's "
              "minimum interval between decisions", "extraversion",
              retired="admission gate 2 (2026-10-16): identifiable on the driving speed (ICC 0.70) but recovery "
                      "r 0.64 against 0.7, and its slow end loses tracking more than the default"),
    TraitSpec("caution", "pessimism of an option's value: expected valence minus z x its posterior uncertainty "
              "(0.5 = the mean), and a gain on the avoid radius", "emotional stability, reversed",
              retired="admission gate 2 (2026-10-16): with the pessimism evidence-gated the careful end interacts "
                      "again, but the engage fraction does not follow the trait (ICC -0.23) and the careful end "
                      "loses tracking more"),
    TraitSpec("novelty_bias", "gain on every information-value term of the activity chooser", "openness"),
    TraitSpec("persistence", "gain on the chooser's switching margin and on the probe budgets (attempts before "
              "giving an option up, failures before a change is believed)", "conscientiousness",
              retired="admission gate 2026-10-13: the gains changed the runs but moved no measured signature"),
    TraitSpec("surprise_weighting", "evidence weight of better-than-expected over worse-than-expected outcomes in "
              "the attitude update (ratio 2^(2(x-0.5)), in [0.5, 2]), self-chosen actions only", "learning style",
              retired="admission gate 2 (2026-10-16): the attempts on the object a switch made inert do not follow "
                      "the trait (ICC -0.05, recovery r -0.46)"),
    TraitSpec("curiosity_type", "share of the chooser's uncertainty bonus that learning progress replaces (0 = the "
              "count-like bonus the robot had before any trait existed; 1 = learning progress only)",
              "learning style", default=0.0,
              retired="admission gate 2026-10-13: it moves the same signatures as novelty bias and caution "
                      "(engaging against exploring), so the recovery regression cannot tell them apart"),
)
# The anchor of every trait is 0.5 except curiosity_type, a mixing weight whose unchanged value is 0.
# A retired trait keeps its row (the store persists) but counts as its default in ``effective``,
# is left alone by consolidation and is not owner-facing (presets, cues). After the second gate
# (docs/results/character-gate-prereg.md, "Result 2", 2026-10-16) one trait is admitted: novelty
# bias; the mapping as a whole stays behind ``RuntimeConfig.character_active`` (off).
ACTIVE = tuple(s for s in TRAITS if not s.retired)
RETIRED = {s.key for s in TRAITS if s.retired}


@dataclass
class CharacterConfig:
    traits: tuple = TRAITS
    # Largest change of one trait per write by the consolidation rule (one session's drift)
    # and by an accepted LLM proposal or a rating-derived write. Engineered.
    session_cap: float = 0.05
    advised_cap: float = 0.05
    # The owner's preset: a name and the values it sets (others keep their defaults).
    preset: str = "default"
    values: dict = field(default_factory=dict)

    def spec(self, key):
        for s in self.traits:
            if s.key == key:
                return s
        raise KeyError(key)

    def initial(self, key):
        return float(self.values.get(key, self.spec(key).default))


def clip_write(current, proposed, lo, hi, provenance, cfg: CharacterConfig):
    """The value a write may set: inside [lo, hi], and for capped provenances within one
    cap of the current value. Returns (applied, clipped)."""
    if provenance not in PROVENANCE:
        raise ValueError(f"unknown provenance {provenance!r}")
    value = float(proposed)
    if provenance in CAPPED_PROVENANCE and current is not None:
        cap = cfg.session_cap if provenance == "consolidation" else cfg.advised_cap
        value = min(max(value, float(current) - cap), float(current) + cap)
    value = min(max(value, float(lo)), float(hi))
    return value, abs(value - float(proposed)) > 1e-12


# How each trait evolves at session end (ExperienceMemory.consolidate_character): the logged
# statistic, its target, theta += eta (target - theta), clipped to the per-session cap and the
# bounds; no mean reversion. Engineered rules, documented here for the inspector.
EVOLVES = {
    "caution": "retired (admission gate 2, 2026-10-16); its rule, kept for the record and not applied: toward "
               "0.5 + 0.5 x reserve, reserve = EMA(0.3) of (useful - aversive) / outcomes per session",
    "surprise_weighting": "retired (admission gate 2, 2026-10-16); its rule, kept for the record and not applied: "
                          "toward 0.5 in a session with a detected rule change, toward 0.5 + 0.1 x useful rate "
                          "otherwise",
    "curiosity_type": "toward the fraction of options engaged 3+ times this session with no learning progress",
    "novelty_bias": "toward the fraction of this session's probes (first attempts on an option) that paid off",
    "approach_tempo": "retired (admission gate 2, 2026-10-16): no rule, never changes",
    "persistence": "retired (admission gate 2026-10-13): no rule, never changes",
}
EVOLVES["curiosity_type"] = "retired (admission gate 2026-10-13): the rule is kept for the record and not applied"



# ---------------------------------------------------------------- what the traits set
# A trait value x in [0, 1] scales a parameter by GAIN(x) = 2 ** (2 (x - 0.5)): 0.5 leaves it
# as it was, 0 halves it, 1 doubles it. Engineered; the mapping is the whole of what a trait
# "is" in this robot (docs/research/PERSONALITY_RESEARCH_2.md section 1).
def gain(x, lo=0.5, hi=2.0):
    return float(min(hi, max(lo, 2.0 ** (2.0 * (float(x) - 0.5)))))


# The presets ``bold`` and ``careful`` were withdrawn by the second gate (2026-10-16): a preset
# needs more than one admitted trait to be a character the owner can read.
PRESETS = {
    "default": {},
}


def config_for(preset: str) -> CharacterConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown character preset {preset!r}; one of {sorted(PRESETS)}")
    return CharacterConfig(preset=preset, values=dict(PRESETS[preset]))


def effective(values: dict) -> dict:
    """The parameter settings a set of trait values produces. Keys name the parameter and
    the object that holds it; a missing trait counts as its default, and so does a retired
    one whatever the store holds."""
    values = {k: v for k, v in values.items() if k not in RETIRED}
    tempo = gain(values.get("approach_tempo", 0.5), 0.5, 1.25)   # speed is capped: safety, not style
    caution = float(values.get("caution", 0.5))
    novelty = gain(values.get("novelty_bias", 0.5))
    persist = gain(values.get("persistence", 0.5))
    return {
        # approach tempo: cruise speed and how soon the chooser may reconsider
        "navigator.v_max_gain": tempo,
        "chooser.min_interval_gain": 1.0 / tempo,
        # caution: decision-time pessimism (value = ev - z x uncertainty, applied by the chooser to
        # options with evidence only since 2026-10-15: an untried option is read at its prior) and
        # the avoid radius
        "chooser.value_z": 1.64 * 2.0 * (caution - 0.5),
        "chooser.avoid_radius_gain": gain(caution, 0.75, 1.5),
        # novelty bias: the information-value terms of objects (engage, investigate) both ways, and
        # exploring space only upward (2026-10-15: at 0.66 the v1 low end stood still for a whole
        # phase in a mapped room with nothing in view; a robot that knows nothing goes to look)
        "chooser.info_gain": novelty,
        "chooser.explore_gain": max(1.0, novelty),
        # persistence: how much better another activity must be before switching, attempts on an
        # option before giving up, and failures before a change is believed
        "chooser.margin_gain": persist,
        "learning.initial_probes": max(1, int(round(3 * persist))),
        "learning.change_failures": max(1, int(round(3 * persist))),
        # learning style: evidence weighting by the sign of the surprise (self-chosen actions),
        # and what the curiosity bonus rewards (uncertainty against learning progress)
        "learning.surprise_ratio": gain(values.get("surprise_weighting", 0.5)),
        "learning.curiosity_lp": float(min(1.0, max(0.0, values.get("curiosity_type", 0.0)))),
    }


# The runtime parameters the mapping scales, with the config object that holds each
# (RobotRuntime._apply_character reads the base value once and re-applies the gain).
SCALED = {"chooser.min_interval": "min_interval_gain", "chooser.avoid_radius": "avoid_radius_gain",
          "chooser.margin": "margin_gain", "navigator.v_max": "v_max_gain"}
