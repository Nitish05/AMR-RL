# Human ratings of Pip's character: pre-registration

Written 2026-10-12, before any rater has seen a clip. Tooling: `scripts/eval/ratings.py`
(`protocol` exports the form and clip plan; `score` applies these rules). Research basis:
docs/research/PERSONALITY_RESEARCH_2.md §4. **No study has been run.** Nothing below changes
the robot: ratings never update traits.

## Study 1: trait recovery, between subjects

- **Stimuli.** Four configurations (`default`, `bold`, `careful`, and `null_control` = the
  default clip shown to a second rater group) rendered from identical seeds, camera path and
  opening 10 s, with `character_active=true`; traits at 0.1 / 0.9 for bold and careful
  (`ratings.py` `CONFIGURATIONS`). Learning-style traits are not in this study: they show only
  in statistics over many episodes (study 2, later, with time-lapse summaries).
- **Raters.** 64 per cell for d = 0.5 at 80 % power, plus 15 % for exclusions; recruited on a
  screened platform (Prolific or CloudResearch). Each rater sees one clip.
- **Disclosure at consent** (verbatim in `form.json`): the robot is simulated; its character is
  engineered parameters changed only between sessions by logged rules; no emotions, no inner
  states beyond the logged quantities. Debrief with the ledger wording. No minors, no
  vulnerable groups without an ethics review.
- **Measures.** Four bespoke manipulation-check items (hesitant–bold, reckless–careful, sticks
  to the familiar–drawn to new things, gives up quickly–keeps trying; 1–7); RoSAS Warmth,
  Competence and Discomfort as **three separate scales** (never summed; alpha reported per
  sample); eeriness (Ho & MacDorman 2010; wording to be verified against the source before use,
  flagged in the form); mind perception (Gray 2007 Agency and Experience; Weisman 2017 body /
  heart / mind); two attention checks; a clip-memory check; free text.
- **Exclusions (fixed now).** Either attention check wrong; clip-memory check wrong; under 180 s;
  duplicate IP hash; empty or off-topic free text.
- **Hypotheses and pass rules.**
  1. Each trait's manipulation item differs between its high and low configuration with
     Cohen's d ≥ 0.5 and a 95 % bootstrap CI excluding 0 (bold vs careful; caution reversed).
  2. Discomfort and eeriness in every configuration within 0.3 SD of the default's.
  3. Experience (Gray) below the scale midpoint in every configuration; the pre-registered
     profile is high mind, low heart, low body.
  4. The null-control pair shows no effect on any manipulation item (CI covers 0).
- **Outcome.** All four hold: the configurations are perceptible and not aversive, and the
  trait names may be used in the owner-facing inspector. Any fails: the names stay internal,
  the configuration that raised Discomfort is documented, and a cue redesign is planned.
  Effect sizes, CIs and alphas are reported whatever the outcome; clips and logs are published.

## Study 2: overclaiming audit, 2 × 3 (later, same tooling)

Cues on vs face-off and silent, crossed with no framing, honest-ledger framing, neutral
framing. Pass: cues raise Agency, but Experience and heart rise by < 0.3 SD; emotion-word rates
in free text reported per cell; the honest framing's cost on Warmth reported as a trade-off.
Fail triggers a cue redesign.

## Not in scope

Trust, compliance or absolute levels (video studies underestimate presence effects; only
within-study contrasts are interpreted); any update of traits from ratings (they could only
enter through the shadow adviser as a bounded, logged, provenance-tagged input, and no such
path is wired); publication of the manipulation items as an instrument (that would need the
three-stage validation with about 500 raters per stage).

## Note after the second admission gate (2026-10-16)

The study has not run. Three of its four manipulation checks (tempo, caution, persistence) and
the `bold` / `careful` configurations target traits that the admission gates retired
(docs/results/character-gate-prereg.md, "Result" and "Result 2"); the one admitted trait is
novelty bias, and the presets were withdrawn. The protocol above stands as written, but the
stimuli and checks must be re-pre-registered around the admitted trait before any clip is
recorded.
