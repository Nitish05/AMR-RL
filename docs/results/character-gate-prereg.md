# Character admission gate: pre-registration

Written 2026-10-12 before any gate run. Scored by `scripts/eval/character_gate.py score`.

## Question

The four temperament traits (`approach_tempo`, `caution`, `novelty_bias`, `persistence`;
`behavior/character.py`) are engineered gains on existing behaviour parameters
(`character.effective`). Before the mapping is switched on by default
(`RuntimeConfig.character_active`), each trait must be **identifiable from behaviour an
observer can measure**: setting it must change the robot's measurable behaviour reliably
across seeds, and the set values must be recoverable from the measurements. A trait that
fails is a label without a referent and stays off (docs/research/PERSONALITY_RESEARCH_2.md
§1, the identifiability gate).

## Design

- World `arena`, map `work/evidence/i3-nocov-arena-s4/map`, experiment `history_a`; the
  480 s training phase (standard consequences) is scored. Fresh memory per run.
- Runs: default character × seeds 0–2, plus each trait at 0.2 and 0.8 with the other three
  at 0.5 × seeds 0–2: 3 + 4 × 2 × 3 = **27 runs**, all with `character_active=true` (the
  default run exercises the code path; 0.5 everywhere is the identity, tested in
  `tests/test_character.py`).
- Pinned worktree of the commit that introduced the mapping; evidence in a fresh
  `work/evidence/char-gate-<date>/`; failures retained and counted.

## Signatures (observer-measurable; no inner state)

| signature | definition |
|---|---|
| `speed_mps` | true path length / active seconds |
| `decisions_per_min`, `bout_s` | chooser decisions per minute; active seconds per decision |
| `info_frac` | frames in explore + investigate / all frames |
| `engage_frac`, `avoid_frac` | frames in engage + revisit / all; frames in avoid / all |
| `chosen_uncertainty` | mean posterior uncertainty of the chosen engage/revisit options (the chooser's own prediction, logged in the decision record) |
| `repeat_frac` | 1 − distinct (entity, action) pairs / interactions |
| `aversive_per_min` | aversive outcomes per active minute |

Primary signature per trait (fixed now): approach_tempo → `speed_mps`; caution →
`chosen_uncertainty`; novelty_bias → `info_frac`; persistence → `repeat_frac`.

## Criteria (all must hold for a trait to pass; the gate passes when all four do)

1. **Identifiability.** ICC(1) ≥ 0.4 of the trait's primary signature over its three levels
   (0.2, 0.5, 0.8; three seeds each; 9 runs).
2. **Recovery.** Leave-one-run-out ridge regression (α = 1, standardised inputs) from all
   nine signatures to the four trait values over all valid runs: Pearson r ≥ 0.7 between
   predicted and set values for the trait.
3. **No harm.** For each level: median unintended contact episodes ≤ default median + 1;
   median lost-frame fraction ≤ default median + 2 pp.

## Consequences

- All four pass: `character_active` becomes the default; presets `bold`/`careful` are
  offered to the owner; step 3c (learning-style traits) proceeds with the same gate.
- Some fail: the mapping stays switchable and off; the failing traits are either remapped
  onto a signature that does move (documented as a new pre-registration) or dropped from
  the owner-facing set. No criterion is relaxed after the fact.
- Runs that fail to relocalise or have an invalid frame are reported and excluded from
  the fit; more than 3 such runs voids the batch (re-run, not patched).

## Addendum: learning-style traits (written 2026-10-12, before any run)

The two learning-style traits are mapped onto the memory (`LearningConfig.surprise_ratio`,
`LearningConfig.curiosity_lp`; docs/LEARNING.md) and need worlds in which they can show.
Three batches of 9 runs each (defaults × seeds 0–2, plus one trait at two levels × seeds 0–2;
same map, `character_active=true`, pinned worktree), one world each:

| batch | experiment (scripts/eval/learning.py) | trait varied | levels | primary signature | expectation |
|---|---|---|---|---|---|
| reversal | `style_reversal`: standard rules, switched to `swapped` at 300 s, 660 s | `surprise_weighting` | 0.1 (r ≈ 0.57), 0.9 (r ≈ 1.74) | `post_switch_aversive` (aversive outcomes after the switch) | higher at 0.9: good news weighs more, so bloom's lost reward is believed later and the robot keeps trying it |
| noisy | `style_noisy`: `noisy_tv.yaml` (bloom learnable, stone random yellow/red, grump known), 480 s | `curiosity_type` | 0.5, 1.0 (default 0.0) | `frac_stone_late` (share of second-half interactions aimed at stone) | lower at 1.0: the random object's learning progress stays near zero, so it loses its bonus; a count-like bonus keeps visiting it |
| variance | `style_variance`: `variance.yaml` (bloom and stone equal mean +0.5, stone's variance 3×), 480 s | `caution` (report only; its admission is decided by the temperament gate) | 0.1, 0.9 | `frac_bloom_vs_stone` (`score --primary caution=frac_bloom_vs_stone`) | higher at 0.9: the pessimistic read prefers the steady object |

Criteria for `surprise_weighting` and `curiosity_type`: the same three as above, each
judged within its own batch (ICC(1) ≥ 0.4 of the primary over the three levels, 9 runs;
leave-one-run-out recovery r ≥ 0.7 over the batch's 9 runs; no harm against the batch's
defaults). The variance batch is a pre-registered prediction, not a gate: the direction is
reported, and a reversed direction is recorded as evidence against the caution mapping.
The signature set is the one `character_gate.py` computes at this commit; the per-fixture
late fractions and the two post-switch signatures were added for these batches before any
run. Nothing is changed after the runs; a failed trait stays off and is remapped under a
new pre-registration or dropped.
