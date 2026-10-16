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

## Result: the temperament gate (scored 2026-10-13)

Batch `work/evidence/char-gate-20261012/`, 27 runs from the pinned worktree of `4b5152d`
(paused overnight at the owner's request with 10 done and resumed the next morning; the six
runs interrupted mid-phase were set aside in `interrupted-20261012/` and re-run). 26 valid, 1
failed (below). Scored by `character_gate.py score` after one tooling fix: the scorer at
`4b5152d` looked for `T-*/result.json` while `learning.py` writes
`T-*/<experiment>/result.json`, so the pinned scorer reported "0 runs" (kept as
`score-pinned-0runs.log`); the fixed scorer reads the same signatures with the same rules.

**The gate fails on all four traits. `character_active` stays off, as pre-registered. No
criterion was changed after the runs.**

| trait | primary signature | ICC(1) | recovery r | harm | pass |
|---|---|---|---|---|---|
| approach_tempo | speed_mps | 0.04 | 0.40 | level 0.2: lost fraction 0.124 > default 0.088 + 0.02 | no |
| caution | chosen_uncertainty | n/a (never logged) | 0.67 | level 0.2: lost fraction 0.136 > default 0.088 + 0.02 | no |
| novelty_bias | info_frac | 0.54 | 0.38 | none | no |
| persistence | repeat_frac | −0.31 | −0.14 | none | no |

Medians by level (0.2 / 0.5 / 0.8; the full tables are in `gate.md`):

| trait | speed_mps | info_frac | engage_frac | repeat_frac | lost_frac |
|---|---|---|---|---|---|
| approach_tempo | 0.029 / 0.018 / 0.019 | 0.20 / 0.27 / 0.24 | 0.43 / 0.37 / 0.35 | 0.67 / 0.73 / 0.75 | 0.124 / 0.088 / 0.100 |
| caution | 0.037 / 0.018 / 0.023 | 0.12 / 0.27 / 0.50 | 0.55 / 0.37 / **0.00** | 0.71 / 0.73 / 0.00 | 0.136 / 0.088 / 0.043 |
| novelty_bias | 0.004 / 0.018 / 0.031 | 0.05 / 0.27 / 0.32 | 0.13 / 0.37 / 0.48 | 0.37 / 0.73 / 0.73 | 0.036 / 0.088 / 0.105 |
| persistence | 0.017 / 0.018 / 0.017 | 0.21 / 0.27 / 0.21 | 0.37 / 0.37 / 0.37 | 0.73 / 0.73 / 0.70 | 0.088 / 0.088 / 0.043 |

What each failure is:

- **approach_tempo.** The cruise-speed gain (0.66 and 1.25) did not move `speed_mps`, which is
  path length over active seconds, an average over standing at standoffs, confirming and
  idling; the robot drives at `v_max` for a small share of the time. The 0.2 level interacted
  more and lost tracking more (the harm). A remap to the commanded speed while driving is a new
  pre-registration.
- **caution.** The pre-registered primary was never measured: the chooser's candidate record
  did not carry the prediction's uncertainty, so every logged value was null (fixed with this
  result for future batches; the fix cannot be applied to these runs). The trait's effect is the
  largest in the batch and in the predicted direction on `engage_frac`: 0.44–0.56 at 0.2, 0.30–
  0.39 at the default, **0.000 in all three runs at 0.8**, where z = 1.64 makes every untried
  option's value negative and the robot only explores and investigates (info_frac 0.49–0.60).
  Recovery r 0.67 (bound 0.7). By the rule the trait fails; a remap onto `engage_frac` is a new
  pre-registration, and the 0.8 level is a behaviour an owner would read as refusal, not care.
- **novelty_bias.** Identifiable (ICC 0.54) but not recoverable (r 0.38). At 0.2 seed 1 idled the
  whole phase (60 decisions, all idle, no interaction: with the information terms at 0.66 and
  nothing liked yet, idle at value 0 outscored every option), and seed 0 failed, so n = 2.
- **persistence.** The margin and probe-budget gains changed the runs (different interaction
  sequences in every seed) but moved no measured signature.

**Failed run.** `T-novelty_bias-0.2-s0`: `struct.error: unpack requires a buffer of 4 bytes` in
pygltflib while Genesis loaded a GLB from the shared asset cache; a concurrent run was writing
the same file (the export was not atomic; fixed with this result: temporary file and rename).
One failure of 27 is within the batch's limit of 3.

**Consequence.** The mapping stays switchable and off. For the owner: remap approach tempo
(driving speed) and caution (engage fraction) under a new pre-registration, give novelty bias
more seeds, drop persistence or remap it; note that both "bold" ends (0.2 of approach tempo and
of caution) lose tracking more, so the `bold` preset is not harmless as mapped.

## Result: the learning-style gates (scored 2026-10-13)

Batches `work/evidence/char-gate-style-20261012/{reversal,noisy,variance}`, 27 runs from the
pinned worktree of `c245594` (queued behind the temperament gate; ran 13:20–15:20 PDT). 25
valid, 2 failed (below). Each batch scored on its own directory by `character_gate.py score`
(variance with `--primary caution=frac_bloom_vs_stone`, as pre-registered). The pinned scorer
again printed "0 runs" (kept as `score-pinned-0runs.*`); the working-tree scorer carries the
run-discovery fix and one reporting fix made for these batches: the primary signature's column
is shown even when a level never defines it, and the ICC row says how many levels had the
primary (before, the column vanished and the ICC was silently over fewer levels). No criterion
was changed.

**Both learning-style traits fail their gates. The variance prediction for caution could not be
evaluated. `character_active` stays off; all six traits have now failed admission.**

| batch | trait | primary signature | ICC(1) | recovery r | harm | pass |
|---|---|---|---|---|---|---|
| reversal (7 valid) | surprise_weighting | post_switch_aversive | 0.16 | −0.71 | none | no |
| noisy (9) | curiosity_type | frac_stone_late | −0.11 | 0.58 | none | no |
| variance (9, report only) | caution | frac_bloom_vs_stone | −0.44 (2 of 3 levels have the primary) | 0.95 | level 0.1: lost fraction 0.318 > default 0.152 + 0.02 | (prediction not testable) |

Medians by level (default / low / high; full tables in each `gate.md`):

| batch | primary | info_frac | engage_frac | interactions per run | lost_frac |
|---|---|---|---|---|---|
| reversal, 0.5 / 0.1 / 0.9 | 0 / 0 / 0.5 | 0.16 / 0.25 / 0.26 | 0.36 / 0.38 / 0.52 | 35 / 37 / 44 | 0.073 / 0.052 / 0.076 |
| noisy, 0.0 / 0.5 / 1.0 | 0 / 0 / 0 | 0.17 / 0.15 / **0.54** | 0.37 / 0.24 / **0.13** | 26 / 16 / 12 | 0.092 / 0.043 / 0.029 |
| variance, 0.5 / 0.1 / 0.9 | 0.33 / 0.31 / n/a | 0.29 / 0.02 / 0.50 | 0.49 / 0.65 / **0.00** | 28 / 31 / **0** | 0.152 / **0.318** / 0.043 |

What each result is:

- **surprise_weighting (reversal).** The pre-registered primary cannot move in this world. The
  switch to `swapped` makes bloom inert (valence 0) and stone rewarding; the only aversive
  object after the switch is grump, which every run had already met once before it. Aversive
  outcomes after the switch were 0 in 6 of the 7 valid runs and 1 in
  `T-surprise_weighting-0.9-s1` (a second grump visit). The expectation written in the addendum
  ("bloom's lost reward is believed later and the robot keeps trying it") describes post-switch
  attempts on bloom, not aversive outcomes: the signature was mis-specified in the
  pre-registration, and the trait fails by the rule. Post hoc and not pre-registered: post-switch
  bloom attempts were 3 / 3 / 3 at the default, 3 / 3 at 0.1 and 4 / 4 at 0.9, the predicted
  direction by one attempt. Every valid run found stone's new reward, 4–176 s after the switch
  (default 39 / 174 / 83 s, 0.1: 23 / 102 s, 0.9: 176 / 4 s), with no pattern across levels.
- **curiosity_type (noisy).** The pre-registered primary sits at the floor at every level: the
  second-half share of interactions aimed at the random object was 0 in 7 of 9 runs (0.10 at
  the default seed 0, 0.30 at 1.0 seed 0). The default robot, whose bonus is count-like, did not
  visit stone either: 5 of the batch's 164 interactions were aimed at it (bloom 88, no-fixture
  entities 50, grump 13, roller 8), so the prediction "lower at 1.0" had nothing to lower from.
  Why stone is so rarely chosen in `noisy_tv.yaml` is not determined by these runs. The
  mechanism does change behaviour, on a signature that was not the primary: the
  learning-progress bonus shifted frames from engaging (0.37 → 0.13) to exploring and
  investigating (0.17 → 0.54) and halved the interactions (26 → 12 per run). That is what
  recovery r 0.58 picks up. A remap onto `info_frac` or `engage_frac` would be a new
  pre-registration, and would make a third trait (with novelty bias and caution) whose effect is
  on the same two signatures, which the recovery regression could not then separate.
- **caution (variance, prediction only).** Untestable at 0.9: as in the temperament gate's 0.8,
  the robot interacted with nothing in any of the three runs (z = 1.64 makes every option's
  value negative; information frames 0.49–0.60), so its bloom-versus-stone share is undefined.
  At 0.1 the share was 0.31 / 0.31 / 0.41 against 0.50 / 0.08 / 0.33 at the default: no
  preference change (the steady object is not preferred more or less by the bold robot). The
  direction is neither confirmed nor reversed. The harm check fails at 0.1: the bold robot loses
  tracking in 32–33 % of frames against 9–17 % at the default, as its 0.2 level did in the
  temperament gate; it engages 62–65 % of the time and explores 2 %. Recovery r 0.95 comes from
  `engage_frac` and `info_frac` separating the levels completely, not from the primary.

**Failed runs.** `T-surprise_weighting-0.1-s0` and `T-surprise_weighting-0.9-s0`: both
`struct.error: unpack requires a buffer of 4 bytes` in pygltflib loading a half-written GLB
from the shared asset cache, the race fixed at `aa6990e`; the pinned worktree `c245594`
predates the fix, and six runs started at once on a cold cache. Two failures of 27 are within
the limit of 3; the reversal batch has n = 2 at both of its trait levels, so its ICC and
recovery rest on 7 runs (the pre-registration said 9).

**Consequence.** The mapping stays switchable and off. For the owner: re-specify surprise
weighting's signature (post-switch attempts on the object that went inert, or the time until
the memory's prediction for it falls) and re-run the reversal batch under a new
pre-registration; for curiosity type, either a world in which the default robot does visit the
random object, or a whole-phase visit count as the signature, or drop the trait; the caution
mapping's careful end refuses every interaction in a second world. Across both gates, the six
traits as mapped move three signatures (engaging, information frames, tracking loss) and leave
the others still; any remap should pick signatures that one trait moves alone.

## Pre-registration 2: four traits (written 2026-10-15, before any run)

Owner decision 2026-10-15, after both gates failed: keep four traits under corrected mappings and
signatures, retire two. Nothing in this section changes after the runs.

**Mapping and signature changes (code at this commit; `behavior/character.py`, `behavior/chooser.py`,
`scripts/eval/character_gate.py`):**

| trait | change | primary signature (ICC) | expectation |
|---|---|---|---|
| approach_tempo | mapping unchanged (`v_max` gain in [0.5, 1.25]; 1/gain on the decision interval) | `drive_speed_mps`: true path length while a drive is commanded (\|v\| > 0.02 m/s) over that time (`trajectory.driving_*`, evaluation ground truth). The first gate scored an average that included standing still. | higher at 0.8 (cap 1.25×), lower at 0.2 (0.66×) |
| caution | the pessimism `value = ev − z·uncertainty` applies only to options with evidence; an untried option is read at its prior (the first careful end refused every untried option in two worlds). Avoid-radius gain unchanged. | `engage_frac` (share of frames engaging or revisiting). Reported, no bound: `aversive_revisit_frac`, the share of interactions on an entity whose last outcome was aversive. | lower at 0.8, higher at 0.2; aversive revisits higher at 0.2 |
| novelty_bias | the information gain scales object information (engage, investigate) both ways and the explore frontier only upward (`chooser.explore_gain = max(1, gain)`): at 0.66 one first-gate run knew no entity, found exploring not worth the travel and stood still for 480 s | `info_frac` (unchanged) | higher at 0.8, lower at 0.2 |
| surprise_weighting | mapping unchanged | `post_switch_inert_attempts`: interactions aimed at the object the switch made inert (bloom in `style_reversal`) after the switch; reported: `post_switch_inert_last_s`. The first signature (aversive outcomes after the switch) could not move in that world. | higher at 0.9 (good news weighs more, so the lost reward is believed later) |
| persistence, curiosity_type | **retired** (`TraitSpec.retired`): their rows persist in the store, `effective` reads them as defaults, consolidation leaves them, presets and cues omit them | — | — |

**Design.** Two batches from one pinned worktree, `character_active=true`, the map
`i3-nocov-arena-s4`, five seeds (0–4), levels 0.2 / 0.5 / 0.8 (surprise weighting 0.1 / 0.5 / 0.9):

- T2, temperament: arena, `history_a` (480 s): 3 traits × 2 levels × 5 seeds + 5 defaults = 35 runs.
- R2, reversal: arena, `style_reversal` (660 s, switch at 300 s): surprise weighting × 2 levels × 5
  seeds + 5 defaults = 15 runs.

The re-targeting holds of r9-followup.md §12 are on in both batches if §12 adopts them and off
otherwise; the chain reads §12's verdict before the batches start and records it with the
evidence (`work/evidence/char-gate2-20261015/{T2,R2}/README.txt`).

**Criteria** (per batch, as in pre-registration 1): ICC(1) ≥ 0.4 of the primary over the three
levels; leave-one-run-out ridge recovery r ≥ 0.7 over the batch's valid runs for the batch's
traits; no harm against the batch's defaults (median unintended contacts ≤ default + 1, median
lost-frame fraction ≤ default + 2 pp). A valid run with no interaction counts (its signatures
are real behaviour) and is listed in the report; a run that crashes or does not relocalise is
excluded; more than 5 such runs of 50 void the batch.

**Consequences.** All three temperament traits pass: `character_active` becomes the default
and the presets `bold` / `careful` (tempo, caution, novelty bias) are offered. Any fails: the
mapping stays switchable and off; a failing trait is retired. Surprise weighting passes: it is
admitted alongside the temperament traits; fails: retired. The retired set is then final for
this design; further traits need a new pre-registration.

## Result 2: the second gate (scored 2026-10-16)

Batches `work/evidence/char-gate2-20261015/{T2,R2}`, 50 runs from the pinned worktree of
`5c6cca2` with the re-targeting holds of r9-followup.md §12 off (they were not adopted; the
chain recorded the verdict before any run). Paused overnight at 8 of 35 at the owner's request
and resumed; the six runs cut mid-way were set aside and re-run whole. 50 valid, 0 failed, 0
without an interaction (the explore floor of this pre-registration held: the low novelty end drove
at 0.16 m/s and interacted in every run). Scored by the pinned `character_gate.py score`.

**One trait passes: novelty bias. Approach tempo, caution and surprise weighting fail. By the
pre-registered consequence the three are retired, the presets `bold` and `careful` are
withdrawn, and `character_active` stays off. No criterion was changed after the runs.**

| batch | trait | primary signature | ICC(1) | recovery r | harm | pass |
|---|---|---|---|---|---|---|
| T2 (35) | approach_tempo | drive_speed_mps | 0.70 | 0.64 | level 0.2: lost fraction 0.127 > default 0.092 + 0.02 | no |
| T2 | caution | engage_frac | −0.23 | 0.23 | level 0.8: lost fraction 0.132 > 0.092 + 0.02 | no |
| T2 | novelty_bias | info_frac | 0.68 | 0.75 | none | **yes** |
| R2 (15) | surprise_weighting | post_switch_inert_attempts | −0.05 | −0.46 | none | no |

Medians by level (0.2 / 0.5 / 0.8; surprise weighting 0.1 / 0.5 / 0.9; full tables in each `gate.md`):

| trait | primary | engage_frac | info_frac | lost_frac |
|---|---|---|---|---|
| approach_tempo | driving speed 0.096 / 0.138 / 0.154 m/s | 0.38 / 0.37 / 0.35 | 0.24 / 0.27 / 0.24 | 0.127 / 0.092 / 0.044 |
| caution | engage 0.368 / 0.370 / 0.380 | — | 0.28 / 0.27 / 0.13 | 0.031 / 0.092 / 0.132 |
| novelty_bias | information frames 0.093 / 0.270 / 0.337 | 0.23 / 0.37 / 0.45 | — | 0.029 / 0.092 / 0.109 |
| surprise_weighting | attempts at the inert bloom 6 / 5 / 5 | 0.44 / 0.35 / 0.42 | 0.24 / 0.16 / 0.23 | 0.081 / 0.142 / 0.120 |

What each result is:

- **approach_tempo.** The driving speed is identifiable: 0.075–0.104 m/s at 0.2, 0.113–0.148 at
  the default, 0.111–0.182 at 0.8 (ICC 0.70), while the first gate's average speed stays flat
  (0.020 / 0.018 / 0.023), which confirms that measure was the first failure's cause. Recovery
  r 0.64 against the bound of 0.7. Harm: the slow end lost tracking in 12.7 % of frames (3, 26,
  13, 16 and 4 % by seed) against 9.2 % at the default, as in the first gate (12.4 %); the fast
  end lost 4.4 %. Why driving at two thirds of the cruise speed loses tracking more often is not
  determined by these runs. Retired by the rule, by 0.06 of r and 1.5 pp of lost frames.
- **caution.** With the pessimism evidence-gated the careful end interacts again (engage fraction
  0.26–0.52 at 0.8, against 0.000 in the first gate), but the engage fraction does not follow the
  trait (0.368 / 0.370 / 0.380; ICC −0.23, r 0.23). The careful end explores less (information
  frames 0.13 against 0.27) and lost tracking more (13.2 %, with 33 % and 43 % in two seeds). The
  bold end revisited the aversive grump in 3 of 5 runs (`aversive_revisit_frac` 0.067) against
  none at the default and the careful end: the predicted direction, on a secondary signature.
- **novelty_bias.** Information frames 0.06–0.22 at 0.2, 0.12–0.35 at the default, 0.32–0.39 at
  0.8 (ICC 0.68); recovery r 0.75; no harm (lost 2.9 / 9.2 / 10.9 %, the keen end 0.3 pp inside
  the bound). The keen end also engages more (0.45 against 0.37) and spends more of its time
  moving; the low end never stood still (the first gate's idle run does not recur).
- **surprise_weighting.** Attempts at the inert bloom after the switch: 3–6 at the default, 5–8 at
  0.1, 3–8 at 0.9 (ICC −0.05, r −0.46). The last such attempt came later at both non-default
  levels (medians 246 s and 317 s after the switch against 66 s), symmetric and not pre-registered,
  so not a trait effect. One run at 0.9 (seed 1) never found the stone's new reward in the 360 s
  after the switch; the other 14 did, 4–309 s after it.

**Consequence.** `character_active` stays off. `approach_tempo`, `caution` and `surprise_weighting`
are retired (`TraitSpec.retired`, with the reason): their rows persist, `effective` reads them as
defaults, consolidation leaves them, and the face and sound cues skip them (the inspector marks
the rows). The presets `bold` and `careful` are withdrawn: a preset needs more than one admitted
trait. Novelty bias is the one admitted trait: identifiable, recoverable and harmless as mapped,
switchable with `character_active`, evolving at session end toward the share of the session's
probes that paid off. The rating study (character-ratings-prereg.md) targets three retired traits
with its manipulation checks and needs a new pre-registration before it runs. The retired set is
final for this design. For the owner: approach tempo failed narrowly on a real cost (its slow end
loses tracking); re-admitting it needs a new pre-registration with that cost understood.
