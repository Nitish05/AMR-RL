# Pip's character: mechanisms for the owner's decisions (literature review 2, October 2026)

Desk research only: nothing here was built or tested. It follows the first review
([PERSONALITY_RESEARCH.md](PERSONALITY_RESEARCH.md)) and the owner's decisions of 2026-10-12:
- adopt all four temperament traits (approach tempo, caution, novelty bias, persistence);
- add the learning-style traits (asymmetric weighting of good and bad surprises; curiosity
  type: learning progress against novelty);
- the character persists across restarts and must be able to evolve from experience;
- it may ping an LLM for guidance, but only when necessary;
- human ratings are in scope;
- add a speaker.

**Method.** Two rounds of searches through the paper-search tools (Crossref, arXiv, Google
Scholar; Semantic Scholar was rate-limited throughout): six topic sweeps (about 160 searches,
148 papers), three citation-verification passes, one completeness critique, then four follow-up
sweeps on the gaps it found (about 115 searches, 72 papers) with their own verification pass, and
an abstract check of 16 claims that had been quoted from memory. In all, 217 citations were
re-looked-up: 213 found with matching metadata; 3 arXiv items (Bellemare 2017, Dabney 2018,
Burda 2018 RND) and OPRO (Yang 2023) could not be confirmed because the arXiv API timed out or did
not surface them; one DOI pointed at a preprint copy of a published paper (Specht 2011). Every paper cited came back in a search result;
† marks a paper whose abstract was not returned, so its finding was summarised from the
searching agent's knowledge and should be checked before quoting a number.

## Short answer

The owner's six decisions fit together into one architecture, and the literature supports each
part with one important correction and three warnings.

**The correction.** "Optimism" (asymmetric learning rates) and "caution" (risk sensitivity) are
the same mathematical operation in the standard model (Niv 2012; Mihatsch & Neuneier 2002†;
Dabney 2020†). They can both exist only if they act at different stages: the asymmetry in the
*update* of each object's attitude, caution at *decision* time as a pessimistic quantile of the
object's outcome distribution. Even then they trade off, so their separability has to be shown
by simulating Pip with known settings and recovering them from its logs (Wilson & Collins 2019).

**The warnings.**
1. In humans, learning-style parameters are not stable traits: five-week test-retest
   reliabilities were 0.02–0.71 (Schaaf 2023), and fitted RL parameters do not generalise
   across tasks (Eckstein 2021). Pip's traits will be stable only because they are engineered,
   and the ledger must say so. Any evolution needs a time constant far longer than a session.
2. Any system that learns from its own sampled experience drifts toward caution and stops
   sampling what once hurt it, the "hot stove" effect (Denrell & March 2001). An evolving
   character needs probe renewal (Pip has it), an evidence-weight floor, and a standing
   recovery test.
3. People will attribute feelings to any robot (Roomba owners did; Sung 2007†). A truthful
   mapping does not guarantee a truthful impression, so the rating studies need an overclaiming
   check (Gray 2007; Weisman 2017), not just trait recognition.

**On the LLM.** The owner's "only when necessary" has a formal home: rational metareasoning
(Russell & Wefald 1991†) and robots that try autonomous recovery first and ask only when it
fails (Tellex 2014†; Knepper 2015†), with calibrated asking (KnowNo, Ren 2023). The literature
on LLMs guiding learning agents (ELLM, Eureka, Language-to-Rewards, Reflexion, ExpeL) agrees on
one architecture: the LLM fills a bounded parameter schema or proposes hypotheses, deterministic
code applies and verifies them, the LLM never touches actuators. LLM failure modes (sycophancy,
faithfulness hallucination) are documented and have cheap checks.

## 1. One definition per trait

The six sweeps proposed overlapping definitions; the critic found three for caution and two for
novelty bias. This section fixes one each and names the identifiability test. All values are
engineered parameters that an engineered rule may move slowly (section 2); none is a measured
disposition.

| Trait (operational name) | Definition in Pip's chooser or memory | Expressed in motion and face | Identifiability test (simulate with known values, refit from logs) |
|---|---|---|---|
| **Approach tempo** | T, a multiplicative gain on the mapping from the learned environment reward rate ρ̂ to approach speed and inter-action latency (the vigor rule: latency ∝ √(effort cost / reward rate), Niv 2007†). T shifts the whole speed–latency curve without changing its slope in ρ̂; movement vigor is trait-like in humans, conserved across effectors, with no accuracy cost (Reppert 2018) | speed envelope, stand-off distance, face animation tempo | hold the give-up rule fixed, vary T: give-up counts unchanged, seconds per attempt change; a rich-against-poor world shows the shared ρ̂ slope with different intercepts |
| **Caution** | decision-time pessimism: the expected-value term uses a lower quantile (τ < 0.5) of each object's Beta outcome posterior instead of its mean (Bellemare 2017; Dabney 2018; Niv 2012) | retreat distance and avoidance decay after a red panel; hold before a nudge | Niv-style world: two objects with equal mean but different variance; preference for the steady one rises with τ; recover τ jointly with the surprise weighting below |
| **Novelty bias** | γ, the weight on the whole information-value term (directed exploration; Wilson 2014†, Gershman 2018†, Bellemare 2016) | time to first visit of a new object | horizon manipulation: preference for the less-sampled object grows with time left in the session |
| **Persistence** | κ, a multiplicative bias on the leave threshold of the marginal value theorem: keep engaging an object while its estimated rate q̂ ≥ κ·ρ̂ (Charnov 1976†; humans learn ρ̂ as a running average and use it as the leave threshold, Constantino & Daw 2015). Observable as attempts after the last reward (Lottem 2018). The hysteresis margin stays a fixed, documented perseveration term: an additive stay bias independent of outcomes and of ρ̂, which otherwise masquerades as asymmetric updating in fits (Sugawara & Katahira 2021) | retries before abandoning; revisit frequency | objects with yields 0.8 / 0.5 / 0.2 that deplete: attempts-before-leaving order by yield for every κ, and the logged ratio q̂/ρ̂ at leaving ≈ κ; richness variation (rich against poor world) separates κ (slope) from the hysteresis margin (intercept) |
| **Surprise weighting** (learning style 1) | asymmetric evidence weighting in the attitude update for better-than-expected against worse-than-expected outcomes, ratio bounded in [0.5, 2], applied only to self-chosen actions (Chambon 2020†) | none directly; shows as slower abandonment (ratio > 1) or slower re-approach (ratio < 1) after a rule change | reversal world: object A good→bad, object B bad→good; time-to-abandon and time-to-re-approach must order with the ratio; recover jointly with τ and the hysteresis margin |
| **Curiosity type** (learning style 2) | w_LP ∈ [0, 1] mixing learning progress (slow EMA minus fast EMA of outcome-prediction error; Oudeyer 2007†, Kim 2020) with the count bonus β/√n inside the information term | which objects it keeps returning to | three-object world: a learnable rule, a random-response object ("noisy TV", Burda 2018), a known object; visits to the random object decay for w_LP = 1 and persist for w_LP = 0 |

Notes on the definitions:
- **Pip's attitude update is count-based**, with evidence windows and a 900 s half-life, not a
  delta rule with a learning rate. The "asymmetric learning rate" therefore becomes asymmetric
  *weighting* of outcome evidence by the sign of its surprise. The behavioural meaning is the
  same; the ledger wording should say "evidence weighting".
- **Label the asymmetry as surprise weighting, not optimism.** Across datasets the human
  asymmetry is better described as a choice-confirmation bias, strongest for freely chosen
  actions (Palminteri & Lebreton 2022†; Chambon 2020†), it is contested in some domains (Barron
  2021 found none in a financial task), and it has a performance rationale: a mild
  confirmation bias raises harvested reward under noisy decisions, at the cost of slower
  correction after reversals (Lefebvre 2022).
- **Novelty bias and curiosity type are separable only with the right worlds**: γ sets how much
  information matters, w_LP which kind. Without a random-response object they produce the same
  behaviour and should be merged into one trait.
- **Caution's performance cost is known**: risk-averse policies collect less reward (Dabney
  2018); cautious Pips will collect fewer yellow flags. Expect it, pre-register it.
- **Persistence and tempo share one learned state.** In the canonical theories both reduce to
  the environment's long-run reward rate ρ: leave a patch when its rate falls to ρ (the marginal
  value theorem), act faster when ρ is high (vigor), and the same quantity moves both (Yoon 2018;
  Sukumar 2024). So ρ̂ is a shared learned state, never a trait; the traits are the two engineered
  deviations from it (κ on the leave threshold, T on the speed–latency mapping), which change
  different observables (give-up count against seconds per attempt). Idle time depresses ρ̂, so the
  engineered need must be logged and partialled out. An alternative give-up rule, an evidence
  threshold that the object is depleted (a drift-diffusion process; Davidson 2019; McNamara 1982†),
  is distinguishable by its sensitivity to outcome variance, so the equal-mean different-variance
  world settles which rule Pip uses.
- **Which traits may covary.** The follow-up sweep recommends a hybrid: one engineered fast–slow
  axis with fixed loadings onto all six traits, plus a persisted independent residual per trait,
  with the share of variance carried by the axis as a single tunable (start near 0.5). A pure
  "syndrome" would be a false claim about nature: pace-of-life correlations are weak and
  heterogeneous in a meta-analysis (Royauté 2018†), the personality–learning link has no
  consistent sign (Dougherty & Guillette 2018), and a one-dimensional character cannot evolve
  freely (Dochtermann & Dingemanse 2013†). Fully independent traits risk incoherent mixes (fast
  approach with extreme caution) that raters may not read as a character, since people prefer
  characters whose cues agree (Isbister & Nass 2000†; Nass & Lee 2001†). Pip's definitions of
  personality and syndrome should be the statistical ones: among-individual variance and
  among-individual correlation from multivariate mixed models (Dingemanse & Dochtermann 2013;
  Dingemanse 2010 for the intercept-plus-slope "reaction norm" form).
- **Admission gate for every trait.** A trait enters the capability ledger only if (a) its
  behavioural assay has adjusted repeatability ICC ≥ 0.4 across at least 4 sessions in at least 3
  rooms (GLMM with bootstrap CI; Nakagawa & Schielzeth 2010), and (b) it is recovered from
  simulated logs with r ≥ 0.7 and confusion with any other trait < 0.3 (Wilson & Collins 2019).
  Traits that fail (b) are merged; traits that fail (a) are reported as state, not character.
  Different optimisers can return different parameter sets with near-identical fit ("parameter
  ambiguity"; Xiong 2025, preprint), so recovery must be checked, not assumed. Fitted RL
  parameters do not generalise across tasks (Eckstein 2021), which is one more reason the ledger
  calls all six "engineered parameters", never measured dispositions.

The behavioural signatures above double as the assays animal-personality research uses for
its own trait axes (novel-object test, emergence test, open field; Réale 2007), which is the
honest vocabulary for all of this: traits named by measurable behaviour, with no inference about
inner states.

## 2. Evolution: how the character changes, and how it stays the same character

### What the sweeps agreed on

- **Two time scales.** Fast layer unchanged (outcome counts, attitudes, need). Slow layer moves
  trait values from logged lifetime statistics (TAME; ALMA; meta-gradient RL, Xu 2018; Beck 2023).
- **Bounds.** Each trait lives in a hard range; each change is small and logged with its cause.
- **Persistence.** Traits, drift variables and attitudes survive restarts; transient state
  (need, habituation, hysteresis, current goal) does not.
- **Protection against lock-in.** Probe renewal, an evidence-weight floor so unlearning is always
  possible, and a recovery test (Denrell & March 2001).
- **A protected core.** Analogous to protecting "skill neurons" while leaving the rest plastic
  (Lan 2025, preprint): traits and mature attitudes change slowly, fresh attitudes fast.

### What they disagreed on, and the resolution

| Question | Positions in the sweeps | Resolution |
|---|---|---|
| **When may traits change?** | per decision; every 100 updates; only at session end | **Only at session-end consolidation.** It matches how robot memory systems consolidate (Petit 2016), keeps within-session behaviour stationary for identifiability tests, and makes every change auditable. |
| **Mean reversion to an anchor?** | yes (OU-shaped random walk toward a seed); no | **No reversion.** The owner wants the character to evolve; reversion would undo experience whenever experience stops. Recognisability comes from hard bounds, a per-session cap and a slow time constant. The anchor is kept as a record ("how far has it drifted") for the inspector, not as a pull. |
| **Repeatability (ICC) as a control signal?** | adapt plasticity when ICC leaves [0.3, 0.8] | **No.** ICC is a between-individual variance ratio from animal populations (mean 0.37; Bell 2009†); using it as a within-robot target is a category mismatch, and estimates from 8 sessions are noise. ICC is an acceptance metric in evaluation (section 4), nothing more. |
| **Should the asymmetry track world statistics?** | yes (shadow tables pick the better ratio); no (twins converge) | **Slowly, within bounds.** Convergence of twins in the same world is the intended meaning of "evolves from experience"; individuality then comes from history and from niche-picking (Pip's own choices bias what it experiences; Stamps & Groothuis 2010). The raised-apart twins test (below) is what shows history matters. |
| **Boot-time decay of attitudes?** | apply a wall-clock recency decay at boot; never decay attitudes | **Keep the existing 900 s evidence half-life and nothing else.** Extra boot-time forgetting would read as amnesia, the documented reason families abandoned Pleo (Fernaeus 2010†). |

### Proposed update rules (for the owner to accept or change)

At the end of each session, for each trait θ_i with hard range [lo_i, hi_i] and per-session cap
c_i (e.g. 0.05 of the range):
- compute the session's logged statistic s_i (table below);
- θ_i ← clip(θ_i + η · (target_i(s_i) − θ_i), θ_i − c_i, θ_i + c_i), then clip to [lo_i, hi_i];
- η small (full-range travel in about 20 sessions), with a plasticity floor so it never reaches
  zero; optionally larger in the first sessions (a "sensitive period"; Lones 2018†, Roberts 2006†);
- log the change with its statistic, the rule, and a provenance tag.

| Trait | Session statistic that moves it | Direction | Source |
|---|---|---|---|
| caution | a "reserve": EMA of positive minus aversive outcomes, bounded; more reserve → more caution ("asset protection", Wolf 2007†; state–behaviour feedback, Sih 2015†) | up with reserve, down after long benign stretches | report 2, option C |
| surprise weighting | estimated reversal frequency | toward 1 when rules change often | report 1 F |
| curiosity type | fraction of engaged objects with zero learning progress | toward learning progress when the world has noisy objects | report 2 D |
| novelty bias | fraction of probes that paid off | up when probes pay | report 2 A |
| persistence κ, tempo T | realised net reward rate under small alternating perturbations (κ(1±ε), T(1±ε)) over blocks of engagements, a perturbation-based meta-rule; move toward the better setting | toward whatever pays in that world: fast-depleting worlds favour high κ, slow ones low | follow-up report 2, option 5; the analogue of meta-gradient RL (Xu 2018) for a count-based agent |

**Tests.**
- **Raised-apart twins:** identical settings, one reared in an aversive-dense world, one in a
  benign one, then both tested in a shared world with *new* objects. Differences on new objects
  are character; differences on known objects are knowledge (the trait-versus-knowledge confound).
- **Same-world twins:** should converge in world-driven traits while staying apart in history-
  driven ones.
- **Lock-in recovery:** ten aversive sessions, then benign ones; caution must come back by a
  stated fraction within a stated number of sessions.
- **Drift audit:** the sum of per-session changes never exceeds the cap; ablating the drift rule
  must change behaviour measurably over 20 sessions, otherwise the rule is dead weight.
- **Retention and relearning** of old attitudes after sessions with only new objects, reported
  together, because a single forgetting metric misleads (Ashley 2021; Lesort 2019).

### Persistence mechanics

- **Layered, versioned character store** with mutability classes: constitutional (traits,
  changeable only by the consolidation rule), slow (drift variables, attitudes), experience
  (raw episodes plus aggregates, with a working/long-term split as in multi-session SLAM; Labbé
  2018), transient (not persisted). Each file carries a schema version, character id, boot count,
  and a provenance tag per field: sensor-derived, consolidation, LLM-advised, human-rating.
- **Origin-bound authority** (Louck 2026, preprint): LLM-advised and rating-derived writes carry
  bounded authority and can never raise it; this is the defence against memory poisoning through
  object labels or free-text ratings.
- **Restart contract:** a restart resets transient state only; a user reset is a distinct,
  confirmed action that archives the old character rather than deleting it.
- **Transparency:** a boot cue on the face mapping to logged fields (character version, sessions
  so far, remembered objects, reset or migration happened), and an inspector listing traits,
  attitudes, drift history and provenance. Users notice and value a robot that remembers them
  (Lee 2012†, Snackbot) and read memory loss as a different agent (Reig 2020†).

### What users accept when a robot's character changes

The evidence is thin, short-horizon and mostly about adaptation toward the user rather than
autonomous drift:
- robots that adapt their style by reinforcement learning on the user's performance or social
  signals were accepted when the change was contingent on what the user did (Tapus 2008†;
  Ritschel & André 2017†; Weber 2018†);
- the one study that asked whether an internally motivated adaptive robot brings unacceptable
  unpredictability found, in a small sample, that it did not (Tanevska 2020);
- trait parameters that modulate how a robot learns were perceptible to naive raters (n = 31;
  Churamani 2022);
- no study varies the cadence of change directly. The nearest evidence: over three sessions days
  apart, perceived competence stabilised within one or two sessions, warmth moved slowly and
  perceived threat fluctuated (Paetzel 2020). Pleo owners wanted visible cumulative development
  and were frustrated when resets erased it (Fernaeus 2010†; Jacobsson 2009†).

The defensible reading: **keep competence-related behaviour stable** (localisation, navigation,
response latency to a flagged object) and **let style traits move slowly across sessions, never
within one**, each change attributable to logged events. Memory continuity matters more than the
stability of any trait. Shutdowns of robots and AI companions are grieved (Carter 2020†), and
grief was far stronger when the system had been framed as a person than as a service (Malejka
2026), which is one more argument for the honest engineered/learned labelling.

**Cross-modal consistency.** Incongruence between channels, or between an expression and its
context, lowers recognition, likeability and perceived intelligence (Tsiourti 2019; Isbister &
Nass 2000†), including for non-linguistic utterances (Rosenthal-von der Pütten & Straßmann 2018†).
Rule: one latent expressive state computed from logged quantities drives face, motion and sound
through fixed monotone mappings, with a context check (no eager idiom in avoid mode); traits only
rescale the mappings.

**Two more mechanics the sweep adds:** a visible change log ("more cautious near red panels
after 3 aversive outcomes in 2 sessions"), templated from logged fields, shown in the console and
hinted on the face at session start, with onboarding text that says the character is a set of
learned parameters that changes only between sessions; and an explicit two-level reset ("forget
objects and places, keep character" against "full new start"), each writing a marker and a
one-time plain start-of-session cue so memory loss is attributable rather than silent.

## 3. LLM guidance, only when necessary

### What "necessary" means

The sweeps converged on a two-stage gate. **Stage 1, a logged impasse** (any of):
- the existing frustration trigger has been active for two consecutive sessions;
- an object's attitude has flipped sign three or more times in ten encounters (a reversal Pip
  cannot settle);
- no learning progress on an object for N minutes while untried probes remain;
- a new object class with no outcomes and a chooser top-two margin below a calibrated threshold
  (KnowNo-style, calibrated offline on logged episodes to a target asking rate; Ren 2023);
- a consolidation rule wants to move a trait to its bound.

**Stage 2, a value-of-consultation test** (Russell & Wefald 1991†; Zilberstein 2011†): consult
only if (empirical improvement rate of past accepted proposals × expected gain) exceeds the cost
(tokens, latency, drift risk), under a budget of at most one query per session and none in
benign sessions. The improvement rate comes from a **shadow period first**: for the first M
sessions the LLM is asked but nothing is applied; its proposals are replayed in seeded
simulations to estimate their effect and its grounding error rate.

**What the budgeted-advising literature adds.** The formal home of "advice only when
necessary" is the teacher–student framework with a hard advice budget (Torrey & Taylor 2013),
where *when* the budget is spent matters more than how much: the same number of advice events
at different moments produces different learning. Its later results map directly onto Pip:
- **ask when uncertain, not on a timer** (RCMP: student-initiated advising on high epistemic
  uncertainty; Da Silva 2020), where "uncertain" for Pip is a low decayed evidence count for the
  object and action at hand plus a stalled chooser;
- **but uncertainty alone goes silent when the agent is confidently wrong**, and stale
  estimates lag the real need (Ilhan 2020), which is exactly Pip after a hidden-rule reversal, so
  the trigger must also include a stagnation or reversal-failure signal;
- **dual uncertainty**: ask only when both the agent and its cached model of past advice are
  uncertain (Sahir 2022), and **reuse stored advice** where coverage is high rather than
  re-querying (Ilhan 2021);
- **richer payloads** (values or rankings rather than single actions) are worth more under
  tight budgets (Zhu 2020), and **explanations** with each piece of advice let the student
  generalise and verify it (Guo 2022);
- the gate itself can be **learned** from myopic value-of-information and value-of-perfect-
  information features (Callaway 2018; Hay 2012);
- for cascades, **confidence-based escalation fails under distribution shift** (Jitkrittum 2023),
  again the reversal case, and a two-tier cascade captures most of the gain while deeper ones
  rarely help (Bouchard 2026, preprint; RouteLLM, Ong 2024).
- **Sycophancy is now documented at scale**: five assistants consistently matched users' stated
  views, and human preference data rewards it (Sharma 2023). The prompt must therefore never
  contain Pip's own hypothesis or preferred action; ask for ranked alternatives with a probability
  and a falsifiable prediction each, allow "don't know" without penalty, score predictions against
  outcomes (Brier), and keep a decaying trust score that suspends consultation when it falls below
  the no-information baseline.
- **Evaluation as in the advising literature**: learning curves at budgets {0, 1, 3, 10} against
  no advice; the trigger against random timing at equal budget; a cost–quality frontier over the
  escalation fraction; the metareasoning overhead counted.

### What is sent, and what may come back

- **Sent:** only robot-side records: the trait vector and anchors, outcome counts and attitudes,
  the recent activity summary, the triggers fired, and the identity record. Never simulator
  ground truth, never camera frames, never the fixtures' rule tables. The privilege-boundary test
  (`tests/test_privilege_boundary.py`) must cover the payload builder.
- **Returned, in a fixed JSON schema** (Language-to-Rewards pattern: the LLM fills parameters of
  an engineered template, Yu 2023; Eureka, Ma 2023): one of
  - a bounded delta to one trait, within the per-session cap;
  - a ranking of probes drawn from the sent list of untried options, with a one-line predicted
    outcome each (ELLM pattern, Du 2023), applied as a bonus no larger than the hysteresis margin;
  - a hypothesis with a falsifiable prediction over the next session (Reflexion, Shinn 2023;
    ExpeL, Zhao 2023), stored as text with status "hypothesis".
- **Verification before use:**
  - grounding: every object id and number in the rationale must exist in the sent log
    (faithfulness hallucination is the relevant failure; Huang 2025);
  - paraphrase consistency: two differently worded prompts must yield the same proposal
    (sycophancy and prompt sensitivity; Chen 2024);
  - separation of proposer and verifier: proposals enter a pending state and are promoted only
    after the next session's logs confirm their prediction, by deterministic code, never by the
    same LLM (Agentic Skill Discovery, Zhao 2024);
  - caching by state signature, so repeated impasses on the same object do not re-query
    (FrugalGPT, Chen 2023);
  - a cascade: deterministic templates first ("switch curiosity type after repeated zero
    progress"), a small local model second, the API model last.
- **Prompts never lead the witness.** Send neutral numbers; never "Pip seems frustrated".
- **Over-tight gates also block correct updates** (Ma 2026, preprint): track acceptance rate and
  shadow-test rejected proposals in the evaluator so the gate is tuned, not only tightened.

### Honesty

Every accepted change is recorded and described as "LLM-suggested bounded change to engineered
parameter X, accepted by rule R after test T", never "Pip decided to be braver". LLM rationales
are advisory text, stored, never surfaced as Pip's own state, never spoken (Pip does not speak;
the sweep's "Pip voices the request" variant is out of scope). Offline Eureka-style search over
the engineered defaults, with the privileged evaluator scoring candidates, is a separate
development tool and is labelled "tuned", not "learned by Pip"; whether evaluator aggregates may
be shown to an LLM at all is the owner's call.

### Evaluation

Paired A/B over at least 20 seeded sessions, LLM on against off with the same starting traits:
queries per hour (target ≤ 0.3), acceptance rate, grounding error rate (< 5 % before anything
is applied), the outcome of each accepted proposal against a matched control window, cumulative
drift, and frustration minutes. A no-op equivalence test: when the LLM returns no change,
behaviour must be bit-identical to the LLM-off run.

## 4. Human ratings

### Instruments

- **RoSAS** (Carpinella 2017†): Warmth, Competence and Discomfort, used as **three separate
  scales, not as one three-factor instrument**. The 2025 systematic validation (confirmatory factor
  analysis on open datasets, n = 1107) found the three-factor model fitted poorly, with acceptable
  fit only for word and vignette stimuli, and recommends the subscales as separate scales
  (Pithayarungsarit 2025). Its reliability must be reported per sample.
- **Eeriness** (Ho & MacDorman 2010†) or RoSAS Discomfort as the pass/fail check that no trait
  configuration or cue makes Pip aversive. Godspeed indices are intercorrelated and should not
  be reported as independent effects.
- **Mind perception** for the overclaiming check: Gray's Experience and Agency (Gray 2007), and
  Weisman's body / heart / mind (Weisman 2017). The pre-registered profile for Pip is high mind,
  low heart, low body. Robots perceived as having Experience are also rated as unnerving (Gray
  2012†), so overclaiming costs comfort as well as honesty.
- **Bespoke manipulation-check items** for the traits (hesitant–bold, careful–reckless, drawn to
  new things–sticks to the familiar, gives up quickly–keeps trying), because no observer scale
  exists for these traits (Esterwood 2021). If they are to be published as an instrument, the
  current bar is a three-stage validation with about 500 raters per stage (Yamada 2025).

### Protocols

1. **Trait recovery, between subjects.** Four configurations rendered from identical seeds,
   camera path and opening 10 s; each rater sees one clip; manipulation checks, RoSAS, eeriness,
   mind perception, free text. Pass: each trait's item differs between its high and low
   configuration with d ≥ 0.5 and a CI excluding 0; Discomfort and eeriness within 0.3 SD of
   baseline in every configuration; Experience below the midpoint. A null-control pair (the
   same clip, two rater groups) must show no effect. About 64 raters per cell for d = 0.5 at
   80 % power (Baxter 2016†), plus 15 % for exclusions; d = 0.5 is a guess, since no prior
   effect size exists for trait perception from behaviour alone.
2. **Learning-style traits need more than a clip.** Asymmetric weighting and curiosity type show
   only in statistics over many episodes. Use time-lapse summaries or trajectory overlays, and
   expect that they may not be perceptible at all; that is an open empirical question.
3. **Consistency, within subjects:** three clips of the same configuration from different rooms
   and episodes, counterbalanced; ICC(2,k) ≥ 0.6 per trait item, between-configuration
   separation above within-configuration variance (Paetzel 2020†; Perugia 2021).
4. **Behavioural distinguishability:** a simple classifier recovers the configuration from logs
   on unseen seeds (balanced accuracy ≥ 80 % over four configurations), surviving removal of any
   feature family; a human line-up task matched against the classifier's confusions (Bera 2017).
5. **Overclaiming audit, 2 × 3:** cues on against face-off and silent, crossed with no framing,
   honest ledger framing, neutral framing. Pass: cues raise Agency but Experience and heart rise
   by < 0.3 SD; emotion-word rates in free text reported per cell; the honest framing's cost on
   Warmth reported as a trade-off. Fail triggers a cue redesign.
6. **Evolution audit:** paired before/after clips around each logged trait change, including
   LLM-advised ones as a tagged subset; perceived change must have the same sign as the logged
   change in > 70 % of pairs; pairs with no change at chance.
7. **Presence bridging, later:** a small interactive condition to estimate how much absolute
   ratings shift with interactivity, and, once a physical Pip exists, a physical-versus-rendered
   comparison (Mara 2026†, HRI 2026, result not retrieved). Video studies underestimate
   presence-dependent effects (Li 2015†; Bainbridge 2011†), so only within-study contrasts are
   interpreted, never absolute levels, and never trust or compliance.

### Practice

- Pre-register hypotheses, exclusions and analyses; publish clips and logs (Leichtmann 2022).
- Recruit on Prolific or CloudResearch with filters, not raw MTurk (Peer 2021; Douglas 2023;
  disputed by Litman 2021), and screen regardless of platform: two attention checks, a clip
  memory check, minimum time, unique IP, free-text sanity. About $2 per retained respondent
  plus pay.
- Standardise clip openings: first impressions persist (Paetzel 2020†).
- Disclose at consent that the robot is simulated and its character engineered and learned; no
  deception; debrief with the ledger wording; exclude minors and vulnerable groups without a
  formal ethics review.
- Ratings never update traits directly. If the owner wants them to influence anything, they go
  through the consultation path as a bounded, logged, provenance-tagged input, and their free
  text is treated as untrusted when it reaches an LLM.

## 5. The speaker

### Design

- **A small discrete cue set** (≤ 8), one cue per logged state transition (explore start,
  approach, signal or nudge executed, positive outcome, aversive outcome, avoid, idle-long,
  relocalising or blocked), built from one procedural synthesiser so they share a timbre
  (R2-D2-style parametric synthesis; Schwenk 2014†). Interpretation of robot sounds is
  categorical, not graded (Read & Belpaeme 2016†), so continuous mappings of scalars such as
  need level will not be heard as graded; use threshold events.
- **Every sound pairs with a visible event.** Listeners read valence from the situation and
  take mainly intensity from the sound (Read & Belpaeme 2014†). Sound alone is ambiguous and
  should accompany another channel, here the face and the visible action.
- **An uncertainty cue** as an "artificial subtle expression": one short low-salience falling
  tone when the chosen action is a probe or confidence is low; nothing when confidence is high
  (Terada 2013†, one lab's line; the falling contour may read as disappointment, so test it).
- **A transit layer**: a quiet, ambient-blending pulse whose rate follows commanded wheel speed,
  with onset and offset markers time-locked to commanded motion within 100 ms, because a 100 ms
  longer sound shifted where people thought a robot stopped and would go next (Currie 2023,
  preprint). Sound improves localising an unseen robot (Cha 2018†); subtle sounds were preferred
  (Latupeirissa 2023). Display at most one or two features at once, since recognition fell when
  two were layered (Orthmann 2023).
- **Outcome cues**: a bright chime synchronised with the yellow flag, a damped thud with the red
  panel, iconic because icons are learned fastest, although they irritate more on repetition
  (Nees 2023; Amer 2018). **Do not scale their intensity with surprise or with any trait**:
  graded intensity is not perceived as graded, louder and harsher sounds worsen perception
  (Allen 2025, HRI 2025 and RA-L), and a loud aversive cue reads as the robot being hurt.
- **Need and idle cues** only at threshold crossings, with a global emission budget; names
  "idle-long" and "need-high", never "frustration" in the cue identifiers.
- **A per-character acoustic signature** derived from the traits through tempo, pitch
  variability and density, not mean pitch or loudness: in 448 human speakers mean pitch was not
  reliably linked to extraversion while pitch variability and loudness were (Ostermann 2025).
  No robot study links sound parameters to perceived traits, so this is a convention to be
  validated on Pip; musical cues raised trust on an arm but not on a humanoid (Savery 2021), so
  nothing imported from Cozmo, NAO or Pepper can be assumed to transfer.
- **Architecture**: a one-way sound event bus fed by logged state transitions (SonifyIt pattern;
  Zhang 2022†); in simulation the speaker is a logged virtual emission, optionally rendered to a
  WAV for evaluation; control, learning and the experience memory must be byte-identical with the
  subscriber on or off, and the LLM has no path to the bus.
- **Hardware**: motor noise will worsen perception on a real robot (Allen 2025; Moore 2017†;
  Frid 2022), and no surfaced paper gave speaker specifications; the speaker is a design choice,
  and its layer should blend with motor noise rather than fight it.

### Evaluation

Forced-choice recognition of state from sound alone (> 70 % for ≤ 8 cues) and with video; a
perceptual map of the cue set to confirm separation (Moore 2019†); valence and arousal ratings;
repeated-exposure annoyance over a 20–30 min session (the only repeated-exposure evidence found
is a single-session HCI study; nothing over days or weeks); localisation and motion-prediction
tasks for the transit layer; non-interference with the learning fixtures (they respond to screen
patterns and nudges, not sound, so the expected interference is nil, but it must be shown).
Subtle layers need in-room evaluation, not only video (Latupeirissa 2023).

## 6. Evidence quality and what remains unknown

- **No paper implemented learning-rate asymmetry or risk sensitivity as a deliberate robot
  trait and tested whether observers noticed.** The closest are curiosity in a shop robot, which
  raters found more humanlike and interesting (Doering 2019), and a robot dog's introversion
  recognised from speech and movement (Lee 2006†).
- **No paper formalises "the same character" as a bounded random walk**, applies ICC to robot
  logs, or measures whether people recognise the same robot across sessions from behaviour
  alone. Those protocols are proposals.
- **No empirical work on bounded, rate-limited LLM advice to an embodied agent under a privilege
  boundary.** Section 3 is assembled from ExpeL, Generative Agents, Reflexion, KnowNo and the
  metareasoning literature.
- **Several design-bearing items are preprints**: ID-RAG (identity records), Louck 2026
  (origin-bound memory authority), Lan 2025 (protected parameters), Currie 2023 (the 100 ms
  effect), Ma 2026 (anti-sycophancy trade-off), Sekoguchi 2019 (habituation model).
- **Abstract checks of claims that had been quoted from memory.** Confirmed from the abstract:
  the learning-rate asymmetry (Lefebvre 2017), volatility-adapted learning rates (Behrens 2007),
  "more to lose, more risk-averse" as an evolutionary model, not data (Wolf 2007; the label "asset
  protection" is Clark's), the pleasure signal's benefit (Lewis & Cañamero 2016), the RoSAS result
  above, and the vigor rule through a citing paper (Niv 2007). Partly confirmed: Bell 2009 is a
  repeatability meta-analysis, but the mean of 0.37 is in the body, not the abstract. Record found
  but no abstract, so the specific claim still rests on memory: Lee 2006, Kanda 2007 (the one-week
  drop in interest is from the authors' earlier two-week trial), Fernaeus 2010, Tellex 2014, Tapus
  2008, Saerbeck & Bartneck 2010 (the curvature–valence link was the weaker result), Knight &
  Simmons 2014, Pearce & Hall 1980, Keepon.
- **Not found despite searching**: Motif (intrinsic motivation from AI feedback), RouteLLM,
  LLM-based hyperparameter optimisation, Sharma et al. on sycophancy, persona-drift studies,
  Komatsu's original auditory subtle-expression paper, Weiss & Bartneck's Godspeed meta-analysis.

## 7. Decisions still open for the owner

1. **The trait definitions in section 1**, in particular caution as a decision-time quantile and
   the asymmetry relabelled as surprise weighting.
2. **No mean reversion** of traits, only bounds and caps (section 2).
3. **LLM cadence**: consolidation-time only, at most one query per session, shadow period
   first; and whether evaluator aggregates may ever be shown to an LLM in the offline tuning
   tool.
4. **Whether human ratings may ever feed back** into the character (through the consultation
   path only), or stay purely evaluative.
5. **Speaker**: no speech; discrete cues plus a quiet transit layer; the acoustic signature as
   a convention to be validated.
6. **The identifiability gate**: a trait that cannot be recovered from one session's logs is
   not presented as part of the character (Wilson & Collins 2019).

## References

Grouped by sweep. † = no abstract returned; the finding was summarised from the searching agent's knowledge.
**Learning-style traits**
- Germain Lefebvre et al. (2017). Behavioural and neural characterization of optimistic reinforcement learning. *Nature Human Behaviour 1(4)*. doi:10.1038/s41562-017-0067 †
- Stefano Palminteri, Maël Lebreton (2022). The computational roots of positivity and confirmation biases in reinforcement learning. *Trends in Cognitive Sciences 26(7)*. doi:10.1016/j.tics.2022.04.005 †
- Michiyo Sugawara, Kentaro Katahira (2021). Dissociation between asymmetric value updating and perseverance in human reinforcement learning. *Scientific Reports 11*. doi:10.1038/s41598-020-80593-7
- Germain Lefebvre et al. (2022). A Normative Account of Confirmation Bias During Reinforcement Learning. *Neural Computation 34(2)*. doi:10.1162/neco_a_01455
- Valérian Chambon et al. (2020). Information about action outcomes differentially affects learning from self-determined versus imposed choices. *Nature Human Behaviour 4(10)*. doi:10.1038/s41562-020-0919-5 †
- Neil Garrett, Tali Sharot (2017). Optimistic update bias holds firm: Three tests of robustness following Shah et al.. *Consciousness and Cognition 50*. doi:10.1016/j.concog.2016.10.013 †
- Yael Niv et al. (2012). Neural Prediction Errors Reveal a Risk-Sensitive Reinforcement-Learning Process in the Human Brain. *The Journal of Neuroscience 32(2)*. doi:10.1523/jneurosci.5498-10.2012
- Oliver Mihatsch, Ralph Neuneier (2002). Risk-Sensitive Reinforcement Learning. *Machine Learning 49(2-3)*. doi:10.1023/a:1017940631555 †
- Marc G. Bellemare et al. (2017). A Distributional Perspective on Reinforcement Learning. *arXiv (ICML 2017)*. arXiv:1707.06887
- Will Dabney et al. (2018). Implicit Quantile Networks for Distributional Reinforcement Learning. *arXiv (ICML 2018)*. arXiv:1806.06923
- Will Dabney et al. (2020). A distributional code for value in dopamine-based reinforcement learning. *Nature 577(7792)*. doi:10.1038/s41586-019-1924-6 †
- Pierre-Yves Oudeyer et al. (2007). Intrinsic Motivation Systems for Autonomous Mental Development. *IEEE Transactions on Evolutionary Computation 11(2)*. doi:10.1109/tevc.2006.890271 †
- Deepak Pathak et al. (2017). Curiosity-driven Exploration by Self-supervised Prediction. *arXiv (ICML 2017)*. arXiv:1705.05363
- Yuri Burda et al. (2018). Large-Scale Study of Curiosity-Driven Learning. *arXiv (ICLR 2019)*. arXiv:1808.04355
- Yuri Burda et al. (2018). Exploration by Random Network Distillation. *arXiv (ICLR 2019)*. arXiv:1810.12894
- Marc G. Bellemare et al. (2016). Unifying Count-Based Exploration and Intrinsic Motivation. *arXiv (NeurIPS 2016)*. arXiv:1606.01868
- Kuno Kim et al. (2020). Active World Model Learning with Progress Curiosity. *arXiv (ICML 2020)*. arXiv:2007.07853
- Robert C. Wilson et al. (2014). Humans use directed and random exploration to solve the explore-exploit dilemma. *Journal of Experimental Psychology: General 143(6)*. doi:10.1037/a0038199 †
- Samuel J. Gershman (2018). Deconstructing the human algorithms for exploration. *Cognition 173*. doi:10.1016/j.cognition.2017.12.014 †
- Rachit Dubey, Thomas L. Griffiths (2020). Reconciling novelty and complexity through a rational analysis of curiosity. *Psychological Review 127(3)*. doi:10.1037/rev0000175 †
- Takahiro Sekoguchi et al. (2019). Mathematical Model of Emotional Habituation to Novelty: Modeling with Bayesian Update and Information Theory. *arXiv (cs.IT / q-bio.NC)*. arXiv:1907.01355
- Robert C Wilson, Anne GE Collins (2019). Ten simple rules for the computational modeling of behavioral data. *eLife 8*. doi:10.7554/elife.49547
- Jessica V. Schaaf et al. (2023). Test-retest reliability of reinforcement learning parameters. *Behavior Research Methods 56(5)*. doi:10.3758/s13428-023-02203-4
- Kwan Min Lee et al. (2006). Can Robots Manifest Personality?: An Empirical Test of Personality Recognition, Social Responses, and Social Presence in Human-Robot Interaction. *Journal of Communication 56(4)*. doi:10.1111/j.1460-2466.2006.00318.x †
- Malcolm Doering et al. (2019). Curiosity Did Not Kill the Robot. *ACM Transactions on Human-Robot Interaction 8(3)*. doi:10.1145/3326462

**Evolving character**
- Zhongwen Xu et al. (2018). Meta-Gradient Reinforcement Learning. *arXiv (NeurIPS 2018)*. arXiv:1805.09801
- Zhongwen Xu et al. (2020). Meta-Gradient Reinforcement Learning with an Objective Discovered Online. *arXiv (NeurIPS 2020)*. arXiv:2007.08433 †
- Jacob Beck et al. (2023). A Tutorial on Meta-Reinforcement Learning. *arXiv; Foundations and Trends in ML (doi 10.1561/2200000080)*. arXiv:2301.08028
- Timothy E J Behrens et al. (2007). Learning the value of information in an uncertain world. *Nature Neuroscience 10(9):1214-1221*. doi:10.1038/nn1954 †
- Matthew R. Nassar et al. (2010). An Approximately Bayesian Delta-Rule Model Explains the Dynamics of Belief Updating in a Changing Environment. *The Journal of Neuroscience 30(37):12366-12378*. doi:10.1523/jneurosci.0822-10.2010
- John M. Pearce, Geoffrey Hall (1980). A model for Pavlovian learning: Variations in the effectiveness of conditioned but not of unconditioned stimuli.. *Psychological Review 87(6):532-552*. doi:10.1037/0033-295x.87.6.532 †
- Jerker Denrell, James G. March (2001). Adaptation as Information Restriction: The Hot Stove Effect. *Organization Science 12(5):523-538*. doi:10.1287/orsc.12.5.523.10092
- Max Wolf et al. (2007). Life-history trade-offs favour the evolution of animal personalities. *Nature 447(7144):581-584*. doi:10.1038/nature05835 †
- Andrew Sih et al. (2015). Animal personality and state-behaviour feedbacks: a review and guide for empiricists. *Trends in Ecology & Evolution 30(1):50-60*. doi:10.1016/j.tree.2014.11.004 †
- Alison M. Bell et al. (2009). The repeatability of behaviour: a meta-analysis. *Animal Behaviour 77(4):771-783*. doi:10.1016/j.anbehav.2008.12.022 †
- Judy Stamps, Ton G. G. Groothuis (2010). The development of animal personality: relevance, concepts and perspectives. *Biological Reviews 85(2):301-325*. doi:10.1111/j.1469-185x.2009.00103.x
- Brent W. Roberts et al. (2006). Patterns of mean-level change in personality traits across the life course: A meta-analysis of longitudinal studies.. *Psychological Bulletin 132(1):1-25*. doi:10.1037/0033-2909.132.1.1 †
- Jule Specht et al. (2011). Stability and Change of Personality Across the Life Course: The Impact of Age and Major Life Events on Mean-Level and Rank-Order Stability of the Big Five. *SSRN preprint (journal version: Journal of Personality and Social Psychology)*. doi:10.2139/ssrn.1884786
- Gaia Molinaro et al. (2024). Latent Learning Progress Drives Autonomous Goal Selection in Human Reinforcement Learning. *Advances in Neural Information Processing Systems 37:32251-32280 (PsyArXiv 10.31234/osf.io/5dtx3)*. doi:10.52202/079017-1013
- Angelo Cangelosi, Matthew Schlesinger (2015). Developmental Robotics. *The MIT Press (book)*. doi:10.7551/mitpress/9320.001.0001
- John Lones et al. (2018). A Hormone-Driven Epigenetic Mechanism for Adaptation in Autonomous Robots. *IEEE Transactions on Cognitive and Developmental Systems 10(2):445-454*. doi:10.1109/tcds.2017.2775620 †
- John Lones et al. (2016). From Sensorimotor Experiences to Cognitive Development: Investigating the Influence of Experiential Diversity on the Development of an Epigenetic Robot. *Frontiers in Robotics and AI 3*. doi:10.3389/frobt.2016.00044 †
- Timothée Lesort et al. (2019). Continual Learning for Robotics: Definition, Framework, Learning Strategies, Opportunities and Challenges. *arXiv (Information Fusion 2020)*. arXiv:1907.00182
- Jiahua Lan et al. (2025). Neuron-level Balance between Stability and Plasticity in Deep Reinforcement Learning. *arXiv*. arXiv:2504.08000
- Joon Sung Park et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *arXiv (UIST 2023)*. arXiv:2304.03442
- Andrew Zhao et al. (2023). ExpeL: LLM Agents Are Experiential Learners. *arXiv (AAAI 2024)*. arXiv:2308.10144
- Daniel Platnick et al. (2025). ID-RAG: Identity Retrieval-Augmented Generation for Long-Horizon Persona Coherence in Generative Agents. *arXiv*. arXiv:2509.25299
- Lionel P. Robert Jr. et al. (2020). A Review of Personality in Human-Robot Interactions. *Now Publishers (Foundations and Trends monograph; SSRN 10.2139/ssrn.3528496)*. doi:10.1561/9781680836639 †

**LLM guidance**
- Allen Z. Ren et al. (2023). Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners (KnowNo). *arXiv (CoRL 2023)*. arXiv:2307.01928
- Stefanie Tellex et al. (2014). Asking for Help Using Inverse Semantics. *Robotics: Science and Systems X*. doi:10.15607/rss.2014.x.024 †
- Ross A. Knepper et al. (2015). Recovering from failure by asking for help. *Autonomous Robots 39(3), 347-362*. doi:10.1007/s10514-015-9460-1 †
- Stuart Russell, Eric Wefald (1991). Principles of metareasoning. *Artificial Intelligence 49(1-3), 361-395*. doi:10.1016/0004-3702(91)90015-c †
- Stuart Russell, Eric H. Wefald (1991). Do the Right Thing: Studies in Limited Rationality. *MIT Press (monograph)*. doi:10.7551/mitpress/2474.001.0001
- Shlomo Zilberstein (2011). Metareasoning and Bounded Rationality. *In: Metareasoning (MIT Press), 27-40*. doi:10.7551/mitpress/8069.003.0005 †
- Yuqing Du et al. (2023). Guiding Pretraining in Reinforcement Learning with Large Language Models (ELLM). *arXiv (ICML 2023)*. arXiv:2302.06692
- Yecheng Jason Ma et al. (2023). Eureka: Human-Level Reward Design via Coding Large Language Models. *arXiv (ICLR 2024)*. arXiv:2310.12931
- Wenhao Yu et al. (2023). Language to Rewards for Robotic Skill Synthesis. *arXiv (CoRL 2023)*. arXiv:2306.08647
- Noah Shinn et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *arXiv (NeurIPS 2023)*. arXiv:2303.11366
- Guanzhi Wang et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. *arXiv (TMLR 2024)*. arXiv:2305.16291
- Lingjiao Chen et al. (2023). FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. *arXiv*. arXiv:2305.05176
- Kaiqu Liang et al. (2024). Introspective Planning: Aligning Robots' Uncertainty with Inherent Task Ambiguity. *arXiv (NeurIPS 2024)*. arXiv:2402.06529
- P.-Y. Oudeyer et al. (2016). Intrinsic motivation, curiosity, and learning: Theory and applications in educational technologies. *Progress in Brain Research 229, 257-284*. doi:10.1016/bs.pbr.2016.05.005 †
- Lei Huang et al. (2025). A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. *ACM Transactions on Information Systems 43(2), 1-55*. doi:10.1145/3703155
- Wei Chen et al. (2024). From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning. *arXiv (ICML 2024)*. arXiv:2409.01658
- Huanhuan Ma et al. (2026). Sycophancy Suppression Can Impair Rational Updating: Anti-Sycophancy Should Preserve the Ability to Update. *arXiv*. arXiv:2608.26511
- Xufeng Zhao et al. (2024). Agentic Skill Discovery. *arXiv (CoRL 2024 LangRob workshop)*. arXiv:2405.15019
- Jimmy Wu et al. (2023). TidyBot: Personalized Robot Assistance with Large Language Models. *arXiv / Autonomous Robots (10.1007/s10514-023-10139-z)*. arXiv:2305.05658
- Bo Deng et al. (2026). DeFA: Dependency-Guided Failure Attribution for LLM Agents. *arXiv*. arXiv:2610.01256

**Persistence and memory**
- Gregoire Pointeau, Peter Ford Dominey (2017). The Role of Autobiographical Memory in the Development of a Robot Self. *Frontiers in Neurorobotics 11*. doi:10.3389/fnbot.2017.00027 †
- Dennis Stachowicz, Geert-Jan M. Kruijff (2012). Episodic-Like Memory for Cognitive Robots. *IEEE Transactions on Autonomous Mental Development 4(1) 1-16*. doi:10.1109/tamd.2011.2159004 †
- David Vernon et al. (2015). Prospection in Cognition: The Case for Joint Episodic-Procedural Memory in Cognitive Robotics. *Frontiers in Robotics and AI 2*. doi:10.3389/frobt.2015.00019 †
- Rachel Wood et al. (2012). A review of long-term memory in natural and synthetic systems. *Adaptive Behavior 20(2) 81-103*. doi:10.1177/1059712311421219
- Mathieu Labbé, François Michaud (2022). Long-Term Online Multi-Session Graph-Based SPLAM with Memory Management. *arXiv (cs.RO); journal version Autonomous Robots, DOI 10.1007/s10514-017-9682-5*. arXiv:2301.00050
- Maxime Petit et al. (2016). Reasoning based on consolidated real world experience acquired by a humanoid robot. *Interaction Studies 17(2) 248-278*. doi:10.1075/is.17.2.04pet
- Iolanda Leite et al. (2013). Social Robots for Long-Term Interaction: A Survey. *International Journal of Social Robotics 5(2) 291-308*. doi:10.1007/s12369-013-0178-y †
- C.D. Kidd, C. Breazeal (2008). Robots at home: Understanding long-term human-robot interaction. *2008 IEEE/RSJ International Conference on Intelligent Robots and Systems, 3230-3235*. doi:10.1109/iros.2008.4651113 †
- Min Kyung Lee et al. (2012). Ripple effects of an embedded social agent. *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI 2012), 695-704*. doi:10.1145/2207676.2207776 †
- Min Kyung Lee et al. (2012). Personalization in HRI. *Proceedings of the 7th ACM/IEEE International Conference on Human-Robot Interaction (HRI 2012), 319-326*. doi:10.1145/2157689.2157804 †
- T. Kanda et al. (2007). A Two-Month Field Trial in an Elementary School for Long-Term Human–Robot Interaction. *IEEE Transactions on Robotics 23(5) 962-971*. doi:10.1109/tro.2007.904904 †
- Paul Baxter et al. (2017). Robot education peers in a situated primary school study: Personalisation promotes child learning. *PLOS ONE 12(5) e0178126*. doi:10.1371/journal.pone.0178126 †
- Catharina Vesterager Smedegaard (2019). Reframing the Role of Novelty within Social HRI: from Noise to Information. *2019 14th ACM/IEEE International Conference on Human-Robot Interaction (HRI), 411-420*. doi:10.1109/hri.2019.8673219 †
- Ja-Young Sung et al. (2007). “My Roomba Is Rambo”: Intimate Home Appliances. *UbiComp 2007, Lecture Notes in Computer Science, 145-162*. doi:10.1007/978-3-540-74853-3_9 †
- Ylva Fernaeus et al. (2010). How do you play with a robotic toy animal?. *Proceedings of the 9th International Conference on Interaction Design and Children (IDC 2010), 39-48*. doi:10.1145/1810543.1810549 †
- Michal Luria et al. (2019). Re-Embodiment and Co-Embodiment. *Proceedings of the 2019 Designing Interactive Systems Conference (DIS 2019), 633-644*. doi:10.1145/3322276.3322340 †
- Samantha Reig et al. (2020). Not Some Random Agent. *Proceedings of the 2020 ACM/IEEE International Conference on Human-Robot Interaction (HRI 2020), 289-297*. doi:10.1145/3319502.3374795 †
- Shinichi Nakagawa, Holger Schielzeth (2010). Repeatability for Gaussian and non‐Gaussian data: a practical guide for biologists. *Biological Reviews 85(4) 935-956*. doi:10.1111/j.1469-185x.2010.00141.x
- Holger Schielzeth, Shinichi Nakagawa (2020). Conditional repeatability and the variance explained by reaction norm variation in random slope models. *bioRxiv preprint (later Methods in Ecology and Evolution)*. doi:10.1101/2020.03.11.987073
- T.O. Cornwell et al. (2023). Multilevel repeatability shows selection may act on both personality and predictability, but neither is state dependent. *Animal Behaviour 195, 85-92*. doi:10.1016/j.anbehav.2022.11.004 †
- Dylan R. Ashley et al. (2021). Does the Adam Optimizer Exacerbate Catastrophic Forgetting?. *arXiv (cs.LG)*. arXiv:2102.07686
- Wei-Chieh Huang et al. (2026). A Survey of Agent Memory in the Second Half: Towards Self-Evolving and Long-Horizon Agents. *arXiv (cs.CL/cs.AI)*. arXiv:2602.06052
- Yedidel Louck (2026). Securing LLM-Agent Long-Term Memory Against Poisoning: Non-Malleable, Origin-Bound Authority with Machine-Checked Guarantees. *arXiv (cs.CR)*. arXiv:2606.24322

**Sound**
- Selma Yilmazyildiz et al. (2016). Review of Semantic-Free Utterances in Social Human-Robot Interaction. *International Journal of Human-Computer Interaction 32(1):63-85*. doi:10.1080/10447318.2015.1093856 †
- Robin Read, Tony Belpaeme (2016). People Interpret Robotic Non-linguistic Utterances Categorically. *International Journal of Social Robotics 8(1):31-50*. doi:10.1007/s12369-015-0304-0 †
- Robin Read, Tony Belpaeme (2014). Situational context directs how people affectively interpret robotic non-linguistic utterances. *Proceedings of the 2014 ACM/IEEE International Conference on Human-Robot Interaction, pp. 41-48*. doi:10.1145/2559636.2559680 †
- Robin Read, Tony Belpaeme (2014). Non-linguistic utterances should be used alongside language, rather than on their own or as a replacement. *Proceedings of the 2014 ACM/IEEE International Conference on Human-Robot Interaction (LBR), pp. 276-277*. doi:10.1145/2559636.2559836 †
- Eun-Sook Jee et al. (2010). Sound design for emotion and intention expression of socially interactive robots. *Intelligent Service Robotics 3(3):199-206*. doi:10.1007/s11370-010-0070-7 †
- Kazunori Terada et al. (2013). Expressing Certainty of a Mobile Robot with Artificial Subtle Expressions. *Transactions of the Japanese Society for Artificial Intelligence 28(3):311-319*. doi:10.1527/tjsai.28.311 †
- Markus Schwenk, Kai O. Arras (2014). R2-D2 Reloaded: A flexible sound synthesis system for sonic human-robot interaction design. *The 23rd IEEE International Symposium on Robot and Human Interactive Communication (RO-MAN), pp. 161-167*. doi:10.1109/roman.2014.6926247 †
- Adrian B. Latupeirissa et al. (2023). Probing Aesthetics Strategies for Robot Sound: Complexity and Materiality in Movement Sonification. *ACM Transactions on Human-Robot Interaction 12(4):1-22*. doi:10.1145/3585277
- Frederic Anthony Robinson et al. (2022). Designing Sound for Social Robots: Candidate Design Principles. *International Journal of Social Robotics 14(6):1507-1525*. doi:10.1007/s12369-022-00891-0
- Frederic Anthony Robinson et al. (2021). Smooth Operator (Tuning Robot Perception Through Artificial Movement Sound). *Proceedings of the 2021 ACM/IEEE International Conference on Human-Robot Interaction, pp. 53-62*. doi:10.1145/3434073.3444658 †
- Emma Frid, Roberto Bresin (2022). Perceptual Evaluation of Blended Sonification of Mechanical Robot Sounds Produced by Emotionally Expressive Gestures: Augmenting Consequential Sounds to Improve Non-verbal Robot Communication. *International Journal of Social Robotics 14(2):357-372*. doi:10.1007/s12369-021-00788-4
- Aimee Allen et al. (2025). Robots Have Been Seen and Not Heard: Effects of Consequential Sounds on Human-Perception of Robots. *IEEE Robotics and Automation Letters 10(4):3980-3987*. doi:10.1109/lra.2025.3546097 †
- Aimee Allen et al. (2025). Sound Judgment: Properties of Consequential Sounds Affecting Human-Perception of Robots. *2025 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI), pp. 94-102*. doi:10.1109/hri61500.2025.10974245 †
- Dylan Moore et al. (2017). Making Noise Intentional (A Study of Servo Sound Perception). *Proceedings of the 2017 ACM/IEEE International Conference on Human-Robot Interaction, pp. 12-21*. doi:10.1145/2909824.3020238 †
- Joel Currie et al. (2023). Consequential sound induces illusory distortions in the perception and prediction of robot motion (preprint). *OSF Preprints (Center for Open Science), preregistered studies*. doi:10.31219/osf.io/47uam
- Brian J. Zhang, Naomi T. Fitter (2023). Nonverbal Sound in Human-Robot Interaction: A Systematic Review. *ACM Transactions on Human-Robot Interaction 12(4):1-46*. doi:10.1145/3583743
- Brian J. Zhang et al. (2022). SonifyIt: Towards Transformative Sound for All Robots. *IEEE Robotics and Automation Letters 7(4):10566-10572*. doi:10.1109/lra.2022.3193228 †
- Bastian Orthmann et al. (2023). Sounding Robots: Design and Evaluation of Auditory Displays for Unintentional Human-robot Interaction. *ACM Transactions on Human-Robot Interaction 12(4):1-26*. doi:10.1145/3611655
- Elizabeth Cha et al. (2018). Effects of Robot Sound on Auditory Localization in Human-Robot Collaboration. *Proceedings of the 2018 ACM/IEEE International Conference on Human-Robot Interaction, pp. 434-442*. doi:10.1145/3171221.3171285 †
- Kim Baraka, Manuela M. Veloso (2018). Mobile Service Robot State Revealing Through Expressive Lights: Formalism, Design, and Evaluation. *International Journal of Social Robotics 10(1):65-92*. doi:10.1007/s12369-017-0431-x †
- Michael A. Nees, Eliana Liebman (2023). Auditory Icons, Earcons, Spearcons, and Speech: A Systematic Review and Meta-Analysis of Brief Audio Alerts in Human-Machine Interfaces. *Auditory Perception & Cognition 6(3-4):300-329 (abstract read from the OSF preprint record 10.31234/osf.io/g68f5)*. doi:10.1080/25742442.2023.2219201
- T. S. Amer, Todd L. Johnson (2018). Earcons Versus Auditory Icons in Communicating Computing Events: Learning and User Preference. *International Journal of Technology and Human Interaction 14(4):95-109*. doi:10.4018/ijthi.2018100106
- Richard Savery et al. (2021). Emotional musical prosody for the enhancement of trust: Audio design for robotic arm communication. *Paladyn, Journal of Behavioral Robotics 12(1):454-467*. doi:10.1515/pjbr-2021-0033
- Dylan Moore et al. (2019). Unintended Consonances (Methods to Understand Robot Motor Sound Perception). *Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems, pp. 1-12*. doi:10.1145/3290605.3300730 †
- Sabine Ostermann et al. (2025). Does pitch make for an extraverted voice? Associations of voice pitch and extraversion in group interactions. *Personality Science 6*. doi:10.1177/27000710251355896

**Human ratings and ethics**
- Colleen M. Carpinella et al. (2017). The Robotic Social Attributes Scale (RoSAS). *Proceedings of the 2017 ACM/IEEE International Conference on Human-Robot Interaction, pp. 254-262*. doi:10.1145/2909824.3020208 †
- Pawinee Pithayarungsarit et al. (2025). A Systematic Validation of the Robotic Social Attributes Scale (RoSAS). *2025 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI), pp. 1547-1552*. doi:10.1109/hri61500.2025.10973879 †
- Christoph Bartneck et al. (2009). Measurement Instruments for the Anthropomorphism, Animacy, Likeability, Perceived Intelligence, and Perceived Safety of Robots. *International Journal of Social Robotics 1(1), 71-81*. doi:10.1007/s12369-008-0001-3 †
- Chin-Chang Ho, Karl F. MacDorman (2010). Revisiting the uncanny valley theory: Developing and validating an alternative to the Godspeed indices. *Computers in Human Behavior 26(6), 1508-1518*. doi:10.1016/j.chb.2010.05.015 †
- Heather M. Gray et al. (2007). Dimensions of Mind Perception. *Science 315(5812), 619*. doi:10.1126/science.1134475
- Kurt Gray, Daniel M. Wegner (2012). Feeling robots and human zombies: Mind perception and the uncanny valley. *Cognition 125(1), 125-130*. doi:10.1016/j.cognition.2012.06.007 †
- Kara Weisman et al. (2017). Rethinking people's conceptions of mental life. *Proceedings of the National Academy of Sciences 114(43), 11374-11379*. doi:10.1073/pnas.1704347114
- Connor Esterwood, Lionel P. Robert (2021). A Systematic Review of Human and Robot Personality in Health Care Human-Robot Interaction. *Frontiers in Robotics and AI 8, 748246*. doi:10.3389/frobt.2021.748246
- Sarah Woods et al. (2006). Methodological Issues in HRI: A Comparison of Live and Video-Based Methods in Robot to Human Approach Direction Trials. *RO-MAN 2006, pp. 51-58 (companion paper: 10.1109/amc.2006.1631754)*. doi:10.1109/roman.2006.314394 †
- Jamy Li (2015). The benefit of being physically present: A survey of experimental works comparing copresent robots, telepresent robots and virtual agents. *International Journal of Human-Computer Studies 77, 23-37*. doi:10.1016/j.ijhcs.2015.01.001 †
- Wilma A. Bainbridge et al. (2011). The Benefits of Interactions with Physically Present Robots over Video-Displayed Agents. *International Journal of Social Robotics 3(1), 41-52*. doi:10.1007/s12369-010-0082-7 †
- Paul Baxter et al. (2016). From characterising three years of HRI to methodology and reporting recommendations. *2016 11th ACM/IEEE International Conference on Human-Robot Interaction (HRI), pp. 391-398*. doi:10.1109/hri.2016.7451777 †
- Tony Belpaeme (2020). Advice to New Human-Robot Interaction Researchers. *Springer Series on Bio- and Neurosystems (Human-Robot Interaction: Evaluation Methods and Their Standardization), pp. 355-369*. doi:10.1007/978-3-030-42307-0_14 †
- Benedikt Leichtmann et al. (2022). Crisis Ahead? Why Human-Robot Interaction User Studies May Have Replicability Problems and Directions for Improvement. *Frontiers in Robotics and AI 9, 838116*. doi:10.3389/frobt.2022.838116
- Benedikt Leichtmann et al. (2022). Personal Space in Human-Robot Interaction at Work: Effect of Room Size and Working Memory Load. *ACM Transactions on Human-Robot Interaction 11(4), 1-19*. doi:10.1145/3536167
- Martina Mara et al. (2026). Can VR Robots Stand in for the Real Thing? Comparing a Physical Cobot and Its Virtual Twin for User Perceptions, Experimental Effects, and Study Costs. *Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction (HRI 2026), pp. 734-743*. doi:10.1145/3757279.3785584 †
- Eyal Peer et al. (2021). Data quality of platforms and panels for online behavioral research. *Behavior Research Methods 54(4), 1643-1662 (preprint 10.2139/ssrn.3765448 also returned, with abstract)*. doi:10.3758/s13428-021-01694-3
- Benjamin D. Douglas et al. (2023). Data quality in online human-subjects research: Comparisons between MTurk, Prolific, CloudResearch, Qualtrics, and SONA. *PLOS ONE 18(3), e0279720*. doi:10.1371/journal.pone.0279720
- Leib Litman et al. (2021). Reply to MTurk, Prolific or panels? Choosing the right audience for online research. *SSRN preprint*. doi:10.2139/ssrn.3775075
- Maike Paetzel et al. (2020). The Persistence of First Impressions. *Proceedings of the 2020 ACM/IEEE International Conference on Human-Robot Interaction, pp. 73-82*. doi:10.1145/3319502.3374786 †
- Giulia Perugia et al. (2021). I Can See It in Your Eyes: Gaze as an Implicit Cue of Uncanniness and Task Performance in Repeated Interactions With Robots. *Frontiers in Robotics and AI 8, 645956*. doi:10.3389/frobt.2021.645956
- Nicolas Spatola, Olga A. Wudarczyk (2021). Ascribing emotions to robots: Explicit and implicit attribution of emotions and perceived robot anthropomorphism. *Computers in Human Behavior 124, 106934*. doi:10.1016/j.chb.2021.106934 †
- Laurel D. Riek et al. (2015). The Emerging Policy and Ethics of Human Robot Interaction. *HRI 2015 Extended Abstracts, pp. 247-248*. doi:10.1145/2701973.2714393 †
- Aniket Bera et al. (2017). Aggressive, Tense or Shy? Identifying Personality Traits from Crowd Videos. *Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI 2017), pp. 112-118*. doi:10.24963/ijcai.2017/17
- Sachie Yamada et al. (2025). Development and Validation of Moral Concern for Robot Scale (MCRS). *ACM Transactions on Human-Robot Interaction 14(4), 1-39*. doi:10.1145/3724130

**Budgeted advising and cost-aware consultation**
- Lisa Torrey, Matthew E. Taylor (2013). Teaching on a budget: agents advising agents in reinforcement learning. *AAMAS 2013, pp. 1053-1060*. doi:10.65109/kzvn5966
- Felipe Leno Da Silva et al. (2020). Uncertainty-Aware Action Advising for Deep Reinforcement Learning Agents. *AAAI 2020, vol. 34(04), pp. 5792-5799*. doi:10.1609/aaai.v34i04.6036
- Ercument Ilhan et al. (2020). Student-Initiated Action Advising via Advice Novelty. *arXiv 2010.00381; IEEE Transactions on Games (doi 10.1109/TG.2021.3113644)*. arXiv:2010.00381
- Ercument Ilhan et al. (2021). Action Advising with Advice Imitation in Deep Reinforcement Learning. *arXiv 2104.08441*. arXiv:2104.08441
- Sahir et al. (2022). Methodical Advice Collection and Reuse in Deep Reinforcement Learning. *arXiv 2204.07254*. arXiv:2204.07254
- Changxi Zhu et al. (2020). A Q-values Sharing Framework for Multiagent Reinforcement Learning under Budget Constraint. *arXiv 2011.14281; ACM TAAS (doi 10.1145/3447268)*. arXiv:2011.14281
- Felipe Leno Da Silva et al. (2020). Agents teaching agents: a survey on inter-agent transfer learning. *Autonomous Agents and Multi-Agent Systems 34(1) (doi 10.1007/s10458-019-09430-0); AAMAS 2020 extended abstract doi 10.65109/afli6845*. doi:10.1007/s10458-019-09430-0
- Yue Guo et al. (2022). Explainable Action Advising for Multi-Agent Reinforcement Learning. *arXiv 2211.07882*. arXiv:2211.07882
- Nicholas Hay et al. (2012). Selecting Computations: Theory and Applications. *UAI 2012; arXiv 1207.5879*. arXiv:1207.5879
- Frederick Callaway et al. (2018). Learning to select computations. *UAI 2018; arXiv 1711.06892*. arXiv:1711.06892
- Isaac Ong et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data. *arXiv 2406.18665*. arXiv:2406.18665
- Wittawat Jitkrittum et al. (2023). When Does Confidence-Based Cascade Deferral Suffice?. *NeurIPS 2023; arXiv 2307.02764*. arXiv:2307.02764
- Dylan Bouchard (2026). Is Escalation Worth It? On the Depth of LLM Cascades. *arXiv 2605.06350*. arXiv:2605.06350
- Chengrun Yang et al. (2023). Large Language Models as Optimizers. *ICLR 2024; arXiv 2309.03409*. arXiv:2309.03409
- Mrinank Sharma et al. (2023). Towards Understanding Sycophancy in Language Models. *ICLR 2024; arXiv 2310.13548*. arXiv:2310.13548

**Foraging and vigor: persistence and tempo**
- Eric L. Charnov (1976). Optimal foraging, the marginal value theorem. *Theoretical Population Biology 9(2):129-136*. doi:10.1016/0040-5809(76)90040-x †
- John McNamara (1982). Optimal patch use in a stochastic environment. *Theoretical Population Biology 21(2):269-288*. doi:10.1016/0040-5809(82)90018-1 †
- Benjamin Y. Hayden et al. (2011). Neuronal basis of sequential foraging decisions in a patchy environment. *Nature Neuroscience 14(7):933-939*. doi:10.1038/nn.2856
- Nils Kolling et al. (2012). Neural Mechanisms of Foraging. *Science 336(6077):95-98*. doi:10.1126/science.1216930
- Sara M. Constantino, Nathaniel D. Daw (2015). Learning the opportunity cost of time in a patch-foraging task. *Cognitive, Affective, & Behavioral Neuroscience 15(4):837-853*. doi:10.3758/s13415-015-0350-y
- Yael Niv et al. (2007). Tonic dopamine: opportunity costs and the control of response vigor. *Psychopharmacology 191(3):507-520*. doi:10.1007/s00213-006-0502-4 †
- Alexandre Zénon et al. (2016). Dopamine Manipulation Affects Response Vigor Independently of Opportunity Cost. *The Journal of Neuroscience 36(37):9516-9525*. doi:10.1523/jneurosci.4467-15.2016
- A. Ross Otto, Nathaniel D. Daw (2019). The opportunity cost of time modulates cognitive effort. *Neuropsychologia 123:92-105*. doi:10.1016/j.neuropsychologia.2018.05.006
- Reza Shadmehr et al. (2019). Movement Vigor as a Reflection of Subjective Economic Utility. *Trends in Neurosciences 42(5):323-336*. doi:10.1016/j.tins.2019.02.003 †
- Thomas R. Reppert et al. (2018). Movement vigor as a traitlike attribute of individuality. *Journal of Neurophysiology 120(2):741-757*. doi:10.1152/jn.00033.2018
- Tehrim Yoon et al. (2018). Control of movement vigor and decision making during foraging. *Proceedings of the National Academy of Sciences 115(44)*. doi:10.1073/pnas.1812979115
- Shruthi Sukumar et al. (2024). Effects of reward and effort history on decision making and movement vigor during foraging. *Journal of Neurophysiology 131(4):638-651*. doi:10.1152/jn.00092.2023
- Neil Garrett, Nathaniel D. Daw (2020). Biased belief updating and suboptimal choice in foraging decisions. *Nature Communications 11*. doi:10.1038/s41467-020-16964-5
- Gary A. Kane et al. (2019). Rats exhibit similar biases in foraging and intertemporal choice tasks. *eLife 8*. doi:10.7554/elife.48429
- Nora Harhen, Aaron Bornstein (2022). Overharvesting in human patch foraging reflects rational structure learning and adaptive planning. *PsyArXiv preprint*. doi:10.31234/osf.io/9nmvj
- Eran Lottem et al. (2018). Activation of serotonin neurons promotes active persistence in a probabilistic foraging task. *Nature Communications 9*. doi:10.1038/s41467-018-03438-y
- Jacob D. Davidson, Ahmed El Hady (2019). Foraging as an evidence accumulation process. *PLOS Computational Biology 15(7):e1007060*. doi:10.1371/journal.pcbi.1007060

**Trait covariance and measurement**
- Andrew Sih et al. (2004). Behavioral syndromes: an ecological and evolutionary overview. *Trends in Ecology & Evolution 19(7):372-378*. doi:10.1016/j.tree.2004.04.009
- Denis Réale et al. (2010). Personality and the emergence of the pace-of-life syndrome concept at the population level. *Philosophical Transactions of the Royal Society B 365(1560):4051-4063*. doi:10.1098/rstb.2010.0208
- Andrew Sih, Marco Del Giudice (2012). Linking behavioural syndromes and cognition: a behavioural ecology perspective. *Philosophical Transactions of the Royal Society B 367(1603):2762-2772*. doi:10.1098/rstb.2012.0216
- Niels J. Dingemanse et al. (2010). Behavioural reaction norms: animal personality meets individual plasticity. *Trends in Ecology & Evolution 25(2):81-89*. doi:10.1016/j.tree.2009.07.013
- Max Wolf, Franz J. Weissing (2012). Animal personalities: consequences for ecology and evolution. *Trends in Ecology & Evolution 27(8):452-461*. doi:10.1016/j.tree.2012.05.001
- Raphaël Royauté et al. (2018). Paceless life? A meta-analysis of the pace-of-life syndrome hypothesis. *Behavioral Ecology and Sociobiology 72(3)*. doi:10.1007/s00265-018-2472-z †
- Liam R. Dougherty, Lauren M. Guillette (2018). Linking personality and cognition: a meta-analysis. *Philosophical Transactions of the Royal Society B 373(1756):20170282*. doi:10.1098/rstb.2017.0282
- Niels J. Dingemanse et al. (2012). Defining behavioural syndromes and the role of 'syndrome deviation' in understanding their evolution. *Behavioral Ecology and Sociobiology 66(11):1543-1548*. doi:10.1007/s00265-012-1416-2 †
- Ned A. Dochtermann, Niels J. Dingemanse (2013). Behavioral syndromes as evolutionary constraints. *Behavioral Ecology 24(4):806-811*. doi:10.1093/beheco/art002 †
- Niels J. Dingemanse, Ned A. Dochtermann (2013). Quantifying individual variation in behaviour: mixed-effect modelling approaches. *Journal of Animal Ecology 82(1):39-54*. doi:10.1111/1365-2656.12013
- Maria K. Eckstein et al. (2021). What do reinforcement learning models measure? Interpreting model parameters in cognition and neuroscience. *Current Opinion in Behavioral Sciences 41:128-137*. doi:10.1016/j.cobeha.2021.06.004
- Hua-Dong Xiong et al. (2025). Deep Learning Improves Parameter Estimation in Reinforcement Learning Models. *bioRxiv preprint 2025.03.21.644663*. doi:10.1101/2025.03.21.644663
- Katherine Isbister, Clifford Nass (2000). Consistency of personality in interactive characters: verbal cues, non-verbal cues, and user characteristics. *International Journal of Human-Computer Studies 53(2):251-267*. doi:10.1006/ijhc.2000.0368 †
- Clifford Nass, Kwan Min Lee (2001). Does computer-synthesized speech manifest personality? Experimental tests of recognition, similarity-attraction, and consistency-attraction. *Journal of Experimental Psychology: Applied 7(3):171-181*. doi:10.1037/1076-898x.7.3.171 †

**Changing robot character, congruence, resets**
- Adriana Tapus et al. (2008). User-robot personality matching and assistive robot behavior adaptation for post-stroke rehabilitation therapy. *Intelligent Service Robotics 1(2):169-183*. doi:10.1007/s11370-008-0017-4 †
- Hannes Ritschel, Elisabeth André (2017). Real-Time Robot Personality Adaptation based on Reinforcement Learning and Social Signals. *Companion of the 2017 ACM/IEEE International Conference on Human-Robot Interaction, pp. 265-266*. doi:10.1145/3029798.3038381 †
- Klaus Weber et al. (2018). How to Shape the Humor of a Robot - Social Behavior Adaptation Based on Reinforcement Learning. *Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI), pp. 154-162 (companion abstract: AAMAS 2018, DOI 10.65109/kaog1929, abstract returned)*. doi:10.1145/3242969.3242976 †
- Ana Tanevska et al. (2020). A Socially Adaptable Framework for Human-Robot Interaction. *Frontiers in Robotics and AI 7:121*. doi:10.3389/frobt.2020.00121
- Nikhil Churamani et al. (2022). Affect-Driven Modelling of Robot Personality for Collaborative Human-Robot Interactions. *Frontiers in Robotics and AI (DOI 10.3389/frobt.2022.717193); arXiv 2010.07221*. arXiv:2010.07221
- Astrid M. Rosenthal-von der Pütten, Carolin Straßmann (2018). Interpretation of (In-)Congruent Nonverbal Behavior and Non-Linguistic Utterances. *Companion of the 2018 ACM/IEEE International Conference on Human-Robot Interaction, pp. 221-222*. doi:10.1145/3173386.3176964 †
- Christiana Tsiourti et al. (2019). Multimodal Integration of Emotional Signals from Voice, Body, and Context: Effects of (In)Congruence on Emotion Recognition and Attitudes Towards Robots. *International Journal of Social Robotics 11(4):555-573*. doi:10.1007/s12369-019-00524-z
- Mattias Jacobsson (2009). Play, belief and stories about robots: A case study of a Pleo blogging community. *RO-MAN 2009 - The 18th IEEE International Symposium on Robot and Human Interactive Communication, pp. 232-237*. doi:10.1109/roman.2009.5326213 †
- Elizabeth J. Carter et al. (2020). Death of a Robot: Social Media Reactions and Language Usage when a Robot Stops Operating. *Proceedings of the 2020 ACM/IEEE International Conference on Human-Robot Interaction, pp. 589-597*. doi:10.1145/3319502.3374794 †
- Patryk Malejka (2026). Loss Orientation and Grief Following AI-Companion Shutdown. *OMEGA - Journal of Death and Dying (online first)*. doi:10.1177/00302228261496977
- M. Asada et al. (2009). Cognitive Developmental Robotics: A Survey. *IEEE Transactions on Autonomous Mental Development 1(1):12-34 (see also Lungarella et al. 2003, Connection Science, DOI 10.1080/09540090310001655110; Cangelosi & Schlesinger 2015, MIT Press, DOI 10.7551/mitpress/9320.001.0001)*. doi:10.1109/tamd.2009.2021702 †
- Pierre-Yves Oudeyer, Linda B. Smith (2016). How Evolution May Work Through Curiosity-Driven Developmental Process. *Topics in Cognitive Science 8(2):492-502*. doi:10.1111/tops.12196

