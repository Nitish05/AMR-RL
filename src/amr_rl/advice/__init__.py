"""Advice only when necessary: a gated, shadow-mode adviser for the character
(docs/research/PERSONALITY_RESEARCH_2.md section 3; owner decision 2026-10-12).

Stage 1 detects a logged impasse (``impasse``); stage 2 builds a robot-side payload
(``payload``), consults a backend under a budget and checks the proposal (``adviser``). In this
step the only mode is *shadow*: proposals are stored with their checks and later judged against
the next session's logged statistics by deterministic code; nothing is ever applied. The
default backend is a deterministic template (the first stage of the cascade); no language
model is wired, and the bus to the speaker has no path from here.
"""

from .adviser import Adviser, AdviserConfig, TemplateBackend
from .impasse import TRIGGERS, detect_impasses
from .payload import build_payload, grounded

__all__ = ["Adviser", "AdviserConfig", "TemplateBackend", "TRIGGERS", "detect_impasses", "build_payload", "grounded"]
