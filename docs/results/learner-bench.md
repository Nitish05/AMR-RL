# Learner testbed under perception noise

**What this is:** the REAL learning layer (`ExperienceMemory`, `EntityTracker`,
`classify`, `ActivityChooser`, `Motivation`) driven by a synthetic world with no
physics: a point robot, 8 objects with hidden rules (1 reliable + 1 50 %-reliable
rewarding, 1 aversive, 1 mover, the rest inert, one pair of look-alike twins),
and a perception error model with presets `clean / low / medium / high`.
Code: `src/amr_rl/sim/learner_bench.py`, grid: `scripts/eval/learner_bench.py`
(`bash scripts/amr.sh learner-bench`). The grid of 560 episodes runs in about 15 s.

**What it is not:** evidence about Genesis, real cameras or a real home. The noise
model is engineered; its presets are guesses at what an open-vocabulary detector
and a VLM will do. What it does test is whether the learning mechanism keeps
working when its inputs are wrong in those ways. All numbers: 20 seeds per cell,
mean ± 95 % CI over seeds, true valence scored from what really happened.

## Baseline (commit `c416ccd`)

Evidence: `work/evidence/learner-bench-20261002-191819`.

| noise | true valence / attempt | rewards | aversive repeats | label accuracy | top belief truly best | impure entities |
|---|---|---|---|---|---|---|
| clean | 0.58 ± 0.07 | 16.95 ± 2.02 | 0.00 | 1.00 | 19/20 | 0.00 |
| low | 0.50 ± 0.07 | 15.45 ± 2.59 | 0.10 ± 0.13 | 0.95 | 18/20 | 0.00 |
| medium | 0.32 ± 0.07 | 12.10 ± 2.20 | 0.50 ± 0.30 | 0.86 | 15/20 | 0.05 |
| high | 0.22 ± 0.06 | 11.35 ± 2.13 | 1.40 ± 0.86 | 0.64 | 11/20 | 0.45 ± 0.27 |

Baselines (random / nearest / fixed) stay at −0.08 to +0.03 valence per attempt at
every noise level. In the inert world the robot settled through `medium` (second-half
outcomes 5–6, idle 0.75–0.80) but **not at `high`** (25.9 → 23.4 outcomes, idle 0.39).

## Diagnosis

Confusion of true vs observed outcomes (learned policy, 20 seeds, 900 s):

| world / noise | true `none` seen as `moved` | true `attach:yellow` seen as `moved` | yellow seen as orange/green |
|---|---|---|---|
| standard / medium | 7 of 372 | 3 | 39 |
| standard / high | **104 of 443** | **58** | 40 |
| inert / high | **243 of 985** | — | — |

Every false `moved` had a displacement of 0.15–0.19 m, just over the fixed
threshold. `classify` compared the median of only 3 pre-action positions with the
post-action median. With 10 cm position noise that difference exceeds 0.15 m in
about 30 % of interactions. A false `moved` is worth +0.6, so inert objects looked
rewarding (the inert world never settled), real yellow rewards were recorded as
`moved`, and the `movable` flag widened identity gates to 2.5 m (impure entities).

The two other suspects from the handoff were measured and are **not** the problem:
spurious change hypotheses in a stationary world are 0.05 / 0.55 / 0.15 per episode
(clean / medium / high), and identity impurity disappears once false moves stop.

## Fix: `moved` must beat the measured position scatter (`learning/outcomes.py`)

`moved` now needs a displacement ≥ max(0.15 m, 3 × the standard error of the
before/after median difference). The scatter is estimated from pairwise
differences within each window of the same interaction. A first version used the
median absolute deviation; it read low on 3-frame windows (the median sample has
zero deviation by construction) and was superseded
(`work/evidence/learner-bench-20261002-192241/SUPERSEDED.md`). The 3σ factor is
engineered, not learned. With exact positions (clean) nothing changes.

Evidence: `work/evidence/learner-bench-fixA-before3` (runtime-equivalent: 3
pre-action frames).

| noise | true valence / attempt | false `moved` / episode | real moves seen | top belief truly best | aversive repeats | impure entities |
|---|---|---|---|---|---|---|
| clean | 0.58 ± 0.07 | 0.00 | 1.00 | 19/20 | 0.00 | 0.00 |
| low | 0.50 ± 0.07 | 0.00 | 0.92 ± 0.10 | 18/20 | 0.10 ± 0.13 | 0.00 |
| medium | 0.32 ± 0.07 | 0.25 ± 0.24 | 0.55 ± 0.21 | 14/20 | 0.55 ± 0.30 | 0.05 |
| high | 0.18 ± 0.05 | 0.40 ± 0.26 | 0.24 ± 0.13 | 15/20 | 0.70 ± 0.43 | 0.05 ± 0.10 |

At `high`: label accuracy 0.64 → 0.78, inert world second-half outcomes
23.4 → 8.0 (idle 0.39 → 0.61), reversal tries of the old option 7.9 → 2.6.
Restart first-choice-is-best fell 13/20 → 9/20, and true valence per attempt
0.22 → 0.18. Both are within the CI, but both point the same way: **real moves are
now missed** at medium/high (recall 0.55 / 0.24), so the mover is learned less often.

## Option measured, not adopted in the runtime: 6 pre-action frames

`--before-frames 6`: after the target is confirmed, keep up to 6 frames showing the
confirmed state (within the existing 3 s confirmation window). Evidence:
`work/evidence/learner-bench-fixA-before6`.

| noise | true valence / attempt | real moves seen | top belief truly best | reversal: found new option | restart: first choice best |
|---|---|---|---|---|---|
| clean | 0.57 ± 0.08 | — | 19/20 | 20/20 | 19/20 |
| low | 0.46 ± 0.09 | 0.91 ± 0.12 | 17/20 | 18/20 | 16/20 |
| medium | 0.35 ± 0.06 | 0.86 ± 0.09 | 17/20 | 18/20 | 14/20 |
| high | 0.21 ± 0.06 | 0.33 ± 0.21 | 19/20 | 17/20 | 12/20 |

Most differences against 3 frames are within the CI. The direction is better at
medium/high, flat at clean/low. Adopting it means changing the confirmation phase
of the runtime `Engage` activity (`behavior/activities.py`), which needs a
`scripts/amr.sh test-sim` run. It has not been made.

## What remains

* **Outcome label confusion** (yellow ↔ orange/green, red → orange) is now the
  largest error at `high`. It is a perception error the learner cannot see
  through (a first red seen as orange is worth +0.2, so the robot tries again:
  aversive repeats 0.70 per episode at `high`). It belongs to the next plan step:
  VLM outcome descriptions mapped to categories, with a measured confusion matrix.
* **Real moves under 10 cm position noise** are hard to see with 3 + 12 frames.
* Hallucinated attachments on inert objects are a small residue (≈ 1 per inert
  episode at `high`).
