"""Classify an interaction's consequence from before/after onboard observations.

Only pixel-derived detections enter here. The classifier returns ``None``
(ambiguous -> nothing is learned) unless the evidence is consistent:

* ``moved``: the target's appearance is seen >= 0.15 m from where it was AND that
  displacement is significant against the position scatter measured in the same
  before/after frames (noisy positions alone must not look like motion), or its
  former place is in view after the action and no longer contains it;
* ``attach:<colour>``: a new attachment colour appears on the target;
* ``none``: the target is seen in place with its prior attachment state.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

MOVE_THRESHOLD = 0.15
# Engineered: a displacement must exceed this many standard errors of the
# before/after median difference (2-D, ~1 % false alarms for Gaussian scatter).
MOVE_SIGNIFICANCE = 3.0


def displacement_threshold(before_xy: np.ndarray, after_xy: np.ndarray) -> float:
    """Smallest displacement distinguishable from the measured position scatter.

    The per-axis scatter is estimated robustly from pairwise differences within
    each window (median |xi - xj| = 0.954 sigma for Gaussian scatter; deviations
    about a 3-frame median would include a zero by construction and read low);
    1.25 is the efficiency penalty of a median against a mean. Nothing here is
    learned."""
    diffs = [np.abs(w[i] - w[j]) for w in (before_xy, after_xy)
             for i in range(len(w)) for j in range(i + 1, len(w))]
    sigma = float(np.median(np.concatenate(diffs))) / 0.954 if diffs else 0.0
    se = 1.25 * sigma * math.sqrt(1.0 / len(before_xy) + 1.0 / len(after_xy))
    return max(MOVE_THRESHOLD, MOVE_SIGNIFICANCE * se)


def majority(tokens, fraction=2 / 3):
    if not tokens:
        return None
    token, count = Counter(tokens).most_common(1)[0]
    return token if count >= fraction * len(tokens) else None


def classify(before: list[dict], after: list[dict], *, place_in_view_after: int = 0,
             min_frames: int = 3) -> tuple[str | None, str]:
    """``before``/``after`` items: {"token", "position", "frame_sha256", "t"}.
    ``place_in_view_after``: after-window frames in which the former position was
    inside the camera view with no matching detection near it."""
    if len(before) < min_frames:
        return None, "insufficient_before_evidence"
    context = majority([b["token"] for b in before])
    if context is None:
        return None, "unstable_before_state"
    before_xy = np.array([b["position"] for b in before], float)
    p0 = np.median(before_xy, axis=0)
    if len(after) >= min_frames:
        positions = np.array([a["position"] for a in after], float)
        p1 = np.median(positions, axis=0)
        shift, needed = math.dist(p0, p1), displacement_threshold(before_xy, positions)
        if shift >= needed:
            return "moved", f"displaced {shift:.2f} m (needed {needed:.2f} m)"
        tokens = [a["token"] for a in after]
        # A new attachment that persists for >= min_frames consecutive frames counts,
        # even if it later disappears (transient responses are still responses).
        run, best = 0, None
        for prev, tok in zip([None] + tokens[:-1], tokens):
            run = run + 1 if tok == prev else 1
            if tok not in (context, "attach:none") and run >= min_frames:
                best = tok
        if best is not None:
            return best, "new attachment observed"
        token = majority(tokens)
        if token is None:
            return None, "unstable_after_state"
        return "none", "no visible change"
    if place_in_view_after >= min_frames:
        return "moved", "former place in view and empty"
    return None, "target not observed after action"
