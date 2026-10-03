# Pre-registration: learned policy vs simple baselines (Genesis, arena)

Written and committed **before** any run of this suite (2026-10-03). Results are
reported as is, with every exclusion listed, whatever they show.

## Why

Round 3 compared total valence over a fixed 480 s. That measured two things other
than choice quality:
- **The engineered drive.** The deployed learned policy idles once its engineered
  stimulation need is met. Need rises by 0.004/s from 0.6, and each yellow lowers
  it by 0.35, so it allows roughly (0.6 + 0.004 · 480) / 0.35 ≈ 7 yellows per
  480 s. Baselines never idle.
- **Start adjacency.** On two of three usable starts the rewarding fixture was
  next to the start, so "nearest" was accidentally optimal.

The `fixed` baseline also sorted by random entity ids, so it was not reproducible
(fixed in `842602e`).

## Design

- **Policies:**
  - `clamped`: the learned policy with the drive clamped (need = 1, no satiation, no
    habituation; idle still available). This is an evaluation-only ablation.
  - `learned`: the deployed policy.
  - `random`, `nearest`, `fixed` (first-seen order).
- **Rule permutations** (`configs/consequences/`):
  - `standard`: bloom rewards signal; grump punishes.
  - `swapped`: stone rewards signal; grump punishes.
  - `mirror`: grump rewards signal; bloom punishes.
- **Starts:** the 5 arena start poses (`ARENA_STARTS`, seeds 0–4).
- **Cells:** each (permutation × start) is one cell. Each phase is a fresh memory,
  480 s, relocalised against the saved map before authority is granted.
- **Map:** the arena map that passes the round-4 closed-loop validation (WU8),
  named in the results before the suite runs.
- **Code:** the commit recorded in each run's provenance.

## Metrics

- **Primary:** *true panel valence* at 480 s, scored from the world's own fixture
  events (`report.py` `true_panel_valence`, evaluation only). Yellow = +1, red = −1,
  and a signal into a raised panel = 0.
- **Comparison:** `clamped` vs each baseline, per cell, reported as win / tie / loss.
  A tie is a difference within 1.0 (one panel). Also report the median difference
  over valid cells.
- **Secondary**, reported without being used for the claim:
  - the deployed `learned` policy's true panel valence, with the engineered cap
    stated;
  - perceived valence per attempt;
  - perceived valence over the first N attempts (N = fewest attempts of any policy in
    the cell);
  - true red panels;
  - signals into a raised panel;
  - idle fraction.

## Exclusions

A phase is excluded, and listed by name, if:
- its start relocalisation was wrong (`frame_valid == False`: > 0.2 m or > 8.6°), or
- it never relocalised (`not_relocalized`).

A cell counts only if `clamped` and the baseline being compared are both valid.

## Decision rule

The project may say "the learned policy chooses better than baseline X" only if
`clamped` wins at least 2/3 of the valid cells against X **and** the median
difference is > 0. Otherwise the result is reported as a tie or a loss.

With at most 15 cells per comparison, no significance claims are made.
