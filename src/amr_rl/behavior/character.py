"""Pip's character store: the engineered temperament and learning-style parameters, their
bounds and their provenance, persisted with the experience memory (owner decisions of
2026-10-12; docs/research/PERSONALITY_RESEARCH_2.md).

What is ENGINEERED: every trait's default, its bounds, the per-write caps and the rules in
``clip_write``. What is LEARNED: nothing yet. A later consolidation rule may move a value
from logged experience ("learned via an engineered update"), and an accepted, bounded,
logged LLM proposal may nudge one ("llm_advised"); the provenance of the last write is
stored with each value, and every change is logged with its cause. Traits are named by the
behaviour they will set, never by an inner state; a value of 0.5 means "the behaviour the
robot had before any trait existed". This module stores and audits; no behaviour reads the
traits yet, and the inspector says so.
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
    sets: str        # the behaviour parameters the trait will scale (documentation, not code yet)
    analogue: str    # the human trait it is loosely named after; informational only
    default: float = 0.5
    lo: float = 0.0
    hi: float = 1.0


TRAITS = (
    TraitSpec("approach_tempo", "gain on approach speed and inter-action latency relative to the learned "
              "reward rate", "extraversion"),
    TraitSpec("caution", "the quantile of an object's outcome posterior used as its value (0.5 = the mean; "
              "lower = pessimistic)", "emotional stability, reversed"),
    TraitSpec("novelty_bias", "weight on the information-value term of the activity chooser", "openness"),
    TraitSpec("persistence", "bias on the give-up threshold relative to the learned reward rate",
              "conscientiousness"),
    TraitSpec("surprise_weighting", "weighting of better-than-expected against worse-than-expected outcomes in "
              "the attitude update, self-chosen actions only", "learning style"),
    TraitSpec("curiosity_type", "mix of learning progress (1) against novelty count (0) in the information "
              "value", "learning style"),
)


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
