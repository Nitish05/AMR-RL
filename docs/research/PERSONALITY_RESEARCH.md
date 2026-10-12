# Personality and character for Pip (literature review, October 2026)

Desk research only: nothing here was built or tested in this project. Searches ran on
2026-10-12 through the paper-search tools (Crossref, arXiv, Google Scholar; Semantic
Scholar was mostly rate-limited), about 130 queries in five topic sweeps. About 110
distinct papers were found; the ones that shaped an option are listed at the end.

**How far to trust each entry.**
- Metadata (authors, year, title, venue, DOI or arXiv id) was copied from search results.
  The 2026 arXiv entries were re-checked by hand.
- Where a search returned no abstract, the finding comes from the searching agent's
  knowledge of the paper. Those entries are marked †.
- The evaluation and ethics sweep did not mark these cases one by one.
- Check a finding against the paper before quoting a number.

The owner's decisions on this review (2026-10-12: all four traits, the learning-style traits, a
persistent evolving character, sparse LLM guidance, human ratings, a speaker) and the second
review that works them out are in [PERSONALITY_RESEARCH_2.md](PERSONALITY_RESEARCH_2.md).

## Short answer

Pip already has most of the structure that character architectures use. It has three of
the four TAME layers (traits, attitudes, moods, emotions); only mood is missing:
- **fixed engineered constants:** the chooser weights, need growth and satiation, travel
  and action costs, and speeds. These are not framed as traits today;
- **long-lived attitudes toward objects:** learned liked/disliked/indifferent;
- **a drive:** the engineered need;
- **a face:** a deterministic function of state.

Three things are missing:
1. One small, documented trait vector that shapes choice, motion and face together, so
   that every cue agrees.
2. Dispositions that shift slowly with the robot's own history, so two Pips with the same
   settings diverge into recognisable characters.
3. A motion and expression style: timing, anticipation, gaze and reactions.

The honesty rule is not an obstacle. The best-supported options define character by
measurable behaviour, the way animal-personality research does, with no inner-state
claims. They tie every cue to logged state, which the ethics literature asks for anyway.

**Recommended path**, if the owner wants to proceed:
1. Temperament (option 1).
2. Experience-shaped caution (option 2).
3. Motion and face style (options 4–5).

Evaluate first in simulation from behaviour logs, then with crowd-rated clips.

**Not recommended:**
- an LLM persona in the decision loop;
- matching Pip's personality to the user;
- any layer named or shown as emotion.

## What the literature says, in five points

1. **People do read personality from simple robots.** A robot dog's introversion or
   extraversion was recognised (Lee 2006). So was a robot vacuum's designed personality,
   from movement alone (Hendriks 2011). Laban-style motion features on a differential-drive
   base shifted Big Five ratings (Knight & Simmons 2014; Emir & Burns 2022). Gaze alone
   expressed extraversion (Andrist 2015). Pedestrian trajectories alone carry trait
   information (Bera 2017).
2. **Effects on acceptance are real but modest and mixed** (meta-analysis: Esterwood 2022;
   reviews: Robert 2020, Mou 2020, Santamaria 2017).
   - Extraversion is by far the most studied trait.
   - Agreeable, stable, conscientious profiles wear best over time (Paetzel-Prüsmann 2021).
   - Strong extraversion can push people away (Moujahid 2023).
   - Context changes how a trait is read (Joosse 2013†).
   - Cues that disagree hurt (Isbister & Nass 2000†).
3. **Character is best built in layers with different time scales.**
   - TAME (Moshkina 2011): fixed traits, long-lived object attitudes, slow moods, brief
     emotions.
   - ALMA (Gebhard 2005): the trait sets a baseline that slower states return to.
   - Drive-based robots get individual differences from gains and setpoints: AIBO
     (Arkin 2003), Kismet (Breazeal 2003), Cañamero's homeostatic robots (Lewis & Cañamero
     2016).
4. **Character can come from learning parameters and from history.**
   - Optimism or pessimism can come from asymmetric learning rates (Lefebvre 2017;
     Mihatsch & Neuneier 2002†).
   - Caution can come from reading the lower tail of an outcome distribution
     (Dabney 2020†).
   - Curiosity can be learning progress (Oudeyer 2007) or novelty counts (Bellemare 2016).
   - Learning from one's own sampled experience produces risk aversion by itself, the "hot
     stove" effect (Denrell & March 2001).
   - Agents with more to lose become consistently more cautious (Wolf 2007†).
   - Animal-personality research names traits by measurable behaviour only: boldness,
     exploration, activity (Réale 2007).
5. **LLM personas do not give consistent behaviour.**
   - Persona prompts shift self-reports but not behaviour (Han 2025).
   - Big Five inventories are not valid for LLMs (Zierahn 2026).
   - Personality shaping can degrade safety (Fitz 2025).
   - LLM robots hallucinate capabilities and their personas are hard to read (Ashok 2026).

## Options

Each option lists the mechanism, its evidence, its costs and risks, how to label it
honestly, and how to test it in simulation. They are ordered by recommendation.

### 1. Temperament: a small trait vector that parameterises choice, motion and face (recommended)

**Mechanism.** Three to four engineered constants, fixed for each robot instance and
versioned. They are the single source of truth for the chooser, the motion style and the
face animation, so cues can never contradict each other. Each trait is named by what it
does:

| Trait (operational name) | Big Five analogue | Parameters it sets | Main evidence |
|---|---|---|---|
| **Approach tempo** | extraversion | cruise speed and acceleration on straight segments; stand-off distance; how readily engage or signal beat idle; tempo of face animation | Tapus 2008†, Andrist 2015, Craenen 2018, Lee 2006; keep moderate (Moujahid 2023) |
| **Caution** | inverse neuroticism / emotional stability | weight on the aversive tail of outcome counts; retreat distance and avoidance decay after a red panel; hold time before a nudge | Mihatsch & Neuneier 2002†, Dabney 2020†, Paetzel-Prüsmann 2021 |
| **Novelty bias** (curiosity) | openness | weight on information value; habituation rate; willingness to try untested actions | Oudeyer 2007, Bellemare 2016, Durupinar 2011, Churamani 2022 |
| **Persistence / systematicity** | conscientiousness | chooser hysteresis margin; revisit interval; direct vs meandering exploration; smoothness of the velocity profile | Hendriks 2011, Knight & Simmons 2014† |

- **Template:** one trait vector drives every low-level parameter through a documented
  mapping (Durupinar 2011, OCEAN in a crowd simulator).
- **Personas** are weight vectors over the same utility terms (Holmgård 2019).
- Ship two contrasting presets, because character is easier to see by contrast
  (Hendriks 2011; Craenen 2018).

**Labelling.** "Engineered behaviour parameters (Big-Five-inspired)", listed in the
capability ledger. The UI shows the preset name. Pip "has" a temperament only in this
behavioural sense (Réale 2007; Sloman 2005).

**Costs and risks.**
- Traits can cost performance in contexts where they don't fit (Sih 2004).
- Different trait combinations can produce the same behaviour, e.g. high caution versus
  low curiosity. Use few traits and check they are separable (see Evaluation).
- Safety layers must stay outside trait control: collision checks, localisation-driven
  speed and yaw limits, the supervisor and probe schedules (Fitz 2025 shows trait knobs
  leaking into safety).

**Test in simulation.**
- N seeds × 2 presets in the same rooms.
- A classifier on behaviour logs identifies the preset on held-out rooms.
- The repeatability (ICC) of each behaviour metric across rooms.
- Non-inferiority on learning (restart, reversal, aversive hits) and on navigation.

### 2. Experience-shaped disposition: character that grows from Pip's own history (recommended second)

**Mechanism.** One or two slow, bounded variables updated from lifetime outcome history,
e.g. boldness ↔ caution from the long-run rate of aversive versus positive outcomes.
- They are stored with the experience memory, so they survive restarts.
- They bias the temperament parameters within documented bounds.
- Optional, from Wolf 2007†: caution can rise with accumulated positive memory ("more to
  lose").
- This is TAME's attitude idea generalised from single objects to a disposition, and it
  separates temperament (engineered) from character (learned).

**Evidence.**
- Behavioural syndromes and consistent individual differences: Sih 2004, Réale 2007.
- Caution emerging from sampled experience: Denrell & March 2001.
- Motivational systems with individual sensitivities produce recognisable traits:
  Read 2010†.
- Adaptation must be bounded, or richness turns into unpredictability (Tanevska 2020).

**Labelling.** "Learned through an engineered update rule". The value, its bounds and its
history are logged and shown in the console.

**Costs and risks.**
- **Hot-stove lock-in:** a cautious Pip stops sampling and never corrects (Denrell & March
  2001). Keep probe renewal running and bound the drift.
- **Trait versus knowledge confound:** Pip's per-object attitudes already make it depend
  on its history. Claim "character" only for differences that show on *novel* objects.

**Test in simulation.**
- "Raised apart" twins: identical presets, one reared in an aversive-dense world and one in
  a benign world, then both tested in a shared world with new objects.
- Report how far they diverge, whether the divergence recovers, and what it costs.

### 3. Learning-style traits (optional, combine with 1)

- **Optimism or pessimism.** Separate learning rates for better-than-expected and
  worse-than-expected outcomes in the attitude update (Lefebvre 2017; Mihatsch & Neuneier
  2002†).
  - Test: in reversal worlds a pessimist should re-approach later, and an optimist should
    take more aversive hits.
  - Label: "learning bias parameter". It biases value estimates by design.
- **Curiosity type.** A learning-progress "tinkerer" (Oudeyer 2007; Schmidhuber 2010†)
  versus a count-novelty "scanner" (Bellemare 2016).
  - Test: add an object with random outcomes. The learning-progress Pip should drop it and
    the novelty Pip should keep returning.
- **Arousal-like pacing.** Speed and idle bouts modulated by recent prediction-error size
  and novelty rate (Hiolle 2014). This overlaps with option 6.

### 4. Motion style within the localisation limits (recommended, alongside 1)

| Element | What it does | Evidence | Cost and limit |
|---|---|---|---|
| **Speed envelope (Laban Time/Flow)** | Per-trait acceleration and jerk limits; slow-in/slow-out S-curves; how early it decelerates on approach; translation only | Knight & Simmons 2014†/2015†; Emir & Burns 2022 (vacuum base: Flow tracked neuroticism ratings); Saerbeck & Bartneck 2010† (acceleration reads as arousal) | Yaw rate and yaw acceleration stay fixed by localisation, never by style; fast rotations caused heading drift before |
| **Anticipation beat** | Before a nudge or signal: stop, lock gaze, a 0.3–1 s hold whose length tracks real decision uncertainty, then commit | Takayama 2011†, Gielniak & Thomaz 2011†, Zhou 2017 (timing alone conveys confidence) | Seconds per interaction; the robot is stationary, so no localisation cost |
| **Gaze before move** | Pupils lead to the planner's actual next heading or waypoint 0.3–0.6 s before the base turns; glances at the target on the approach | Hart 2020†, Kawabe 2025 (eyes on the travel side raise animacy) | None; must read the real plan, never a scripted one |
| **Outcome reaction with follow-through** | A short, decaying face transient keyed to the real result (rule confirmed, unexpected, no effect); optionally a slow straight back-off of a few cm | Takayama 2011†, Ribeiro & Paiva 2012† | A back-off adds odometry and may take the target out of view; keep it slow and straight |
| **Hesitation as displayed uncertainty** | When choice entropy is high: decelerate, pause, partial approach, commit | Moon 2021 (hesitation trajectories beat plain stopping) | Efficiency; cap repetitions and log them |

- Expressive style is layered on the task motion, never added as detours (Knight & Simmons
  2015†; van Breemen 2004†).
- It must not reduce legibility, which is linked to perceived safety (Dragan 2013†;
  Lichtenthäler 2012; Kruse 2013).
- Gate all style off when localisation uncertainty is high.

### 5. Screen face: aliveness and state patterns (recommended, alongside 1)

- **Deterministic micro-expression.**
  - Blinks at task-segment boundaries.
  - Micro-saccades on new detections.
  - Lid drift with the activity.
  - A pure function of state and time: no randomness, no idle body sway (body sway would
    add heading noise).
  - Evidence: Kalegina 2018† (pupils and lids matter, and missing features read as
    unfriendly); Keepon (Kozima 2009†: attention plus a small rhythmic vocabulary carries
    a lot); Cozmo behaviour validation (Chan 2021†).
- **State border or glyph** for mapping, waiting, blocked and learning states, mapped
  formally from state to pattern (Baraka & Veloso 2018†).
- **Constraint: the screen is also a task actuator.** "Signal" shows a pattern that the
  world's rules respond to. Character animation must be kept in a separate region or time
  window, and a test must show it cannot trigger a fixture rule.

### 6. Explicit recent-outcome and arousal indices (contentious: owner decision)

**Mechanism.** A documented index such as
`M_t = Σ γ^(t−j) (w1·PE_j + w2·outcome_j)` over Pip's own prediction errors and observed
outcomes (Rutledge 2014; Eldar 2016; Blain & Rutledge 2020 weight learning-relevant
errors), plus an arousal or uncertainty index from novelty and prediction-error size
(Hiolle 2014; Breazeal 2003). The time scales are kept separate (ALMA; TAME; Kirby 2010):
1. single-outcome reaction, seconds;
2. index, minutes;
3. attitude, persistent;
4. temperament, fixed.

**Honesty.** This is the closest option to the project's "no hidden mood variable" rule.
It is allowed only if:
- it is computed from real state and logged;
- it is shown in its own face channel with a published legend;
- it is named an *index*, never "happiness";
- whether it persists across restarts is made explicit.

**Risks.**
- A loop from mood into learning amplifies itself and can oscillate (Eldar & Niv 2015).
  The index must never change stored values or attitudes; at most it shifts exploration
  within bounds (Park 2010).
- Viewers will read feelings into it, and the displayed mood of a screen-faced robot
  changed how people treated it (Kirby 2010).
- Human mood is not a recency-weighted reward average (Keren 2021, primacy weighting), so
  claim no equivalence.
- A variable that does not change behaviour is decorative and should be dropped
  (Cañamero 2005).

### 7. Appraisal record as an explanation (optional)

For each outcome, compute the appraisal variables OCC and EMA use, under computational
names (Ortony 1988; Marsella & Gratch 2009; Moerland 2018):
- desirability: the attitude value;
- expectedness: |prediction error|;
- agency: whether Pip's own command came just before the outcome, which Pip can check;
- controllability: how repeatable the outcome has been;
- novelty.

Use them to choose the short face reaction and to write a templated "why" line in the
console. Showing the robot's action selection in real time improved observers' mental
models of the robot (Wortham & Theodorou 2017).

### 8. Sound (later; needs a speaker)

Sparse, non-linguistic, rhythmic sounds, played only together with a visible event:
- rising pitch for uncertainty, falling pitch for completion (Jee 2010†);
- people hear such sounds categorically, and context overrides the sound
  (Read & Belpaeme 2014/2016†);
- continuous motor noise worsens perception, while informative sounds help
  (Allen 2024/25).

### 9. LLM persona (not recommended in the loop)

At most, offline developer-facing captions generated from logged chooser state. Each
caption is checked automatically against the logs and labelled "generated description of
engineered state". It must never act in the decision loop or see simulator truth.

Reasons:
- persona prompts do not produce consistent behaviour (Han 2025);
- human inventories are invalid for LLMs (Zierahn 2026);
- personality shaping can reduce safety (Fitz 2025);
- grounded, bounded persona templates are needed (Ashok 2026);
- LLM text tends to imply feelings, which the project rule forbids.

### Not recommended

- **Matching personality to the user.** The findings conflict: complementarity
  (Lee 2006), similarity (Tapus 2008†; Andrist 2015), half of observers (Craenen 2018),
  no effect of the user's own personality (Lim 2022). Pip also does not sense people.
- **TAME's emotion layer as emotions.** It conflicts with the project rule. Only the trait
  and attitude ideas transfer directly.

## Evaluation (simulation first)

1. **Distinguishability and stability from logs.** Possible log features:
   - speed and acceleration profiles;
   - path directness;
   - dwell and hold times;
   - stand-off distances;
   - object and action choice frequencies;
   - face-state frequencies.

   Measures:
   - a classifier identifies the preset on held-out rooms and seeds;
   - the ICC of each feature across rooms (trait-like stability);
   - the same across restarts.

   This is an engineering analogue of trait repeatability. The sweeps found no standard
   instrument for it.
2. **Trait versus knowledge.** Differences have to show up on novel objects (option 2's
   twins test).
3. **Task cost.** Pre-registered non-inferiority margins on mapping, goals, contacts and
   the learning metrics. Safety metrics may not get worse at all (Dragan 2013†:
   expressive motion has a cost).
4. **Crowd-rated rendered clips** (later; needs consent and ethics handling).
   - Design: between-subjects; character off, preset A, preset B.
   - Primary measure: RoSAS (Carpinella 2017). Its Discomfort factor doubles as a
     creepiness check. A 2025 re-validation exists (Pithayarungsarit 2025); its result was
     not retrieved.
   - Secondary: Godspeed (Bartneck 2009). Its indices are intercorrelated (Ho & MacDorman
     2010), so pre-register one primary subscale.
   - Exploratory: TIPI (Gosling 2003), which is not validated for robots.
   - Manipulation check: forced-choice recognition of the preset, against chance
     (Lee 2006; Hendriks 2011).
   - Video results are broadly comparable to live ones for approach behaviour
     (Woods 2006), but co-present robots get warmer responses (Li 2015). Label results
     "simulation, video, single exposure".
5. **Overclaiming check.** Mind-perception items for Experience versus Agency (Gray 2007),
   plus a free-response "what is Pip feeling?". Pre-register that a character condition
   must not raise Experience attribution beyond a margin. Disclosure lowers perceived
   humanlikeness but not closeness (van Straten 2020), so this has to be measured, not
   assumed.
6. **Repeated exposure.** The same raters see sessions from memory-bearing runs on day 1
   and day N. First impressions persist (Paetzel 2020), and novelty fades (Leite 2013).

## Honesty guardrails supported by the literature

1. **Every cue maps to logged state.** Every expressive cue is a deterministic function of
   logged real state, with the mapping documented. Nothing exists only to look alive
   (Boden 2017, EPSRC principles; Winfield 2021, IEEE P7001).
2. **No hidden-state deception.** Displays never conceal or misstate what Pip is doing or
   remembering (Danaher 2020).
3. **Operational names, not emotion words.** For example "low confidence", "high approach
   tempo", "learned caution" (Réale 2007; Coeckelbergh 2012; Leong & Selinger 2019).
4. **Measure what viewers believe.** A truthful mapping can still create a false
   impression (Sharkey & Sharkey 2021; Złotowski 2015), hence the overclaiming check.
5. **No children or vulnerable users** in any study without ethics review. Disclosure does
   not remove attachment (Sharkey & Sharkey 2010; van Straten 2020).
6. **Character never overrides safety or task priority.** This is enforced in control code
   and checked by the non-inferiority test.
7. **Accountability and reporting.** Trait configurations are versioned and inspectable.
   Any hand-scripted or hand-picked clip is reported as such (Riek 2012).

## Decisions for the owner

- **Traits:** which 3–4 to adopt, and the two contrasting presets.
- **Experience-shaped dispositions** (option 2): allowed, and should they persist across
  restarts?
- **Recent-outcome index** (option 6): allowed if explicit and displayed, or ruled out by
  the "no hidden mood" rule?
- **Human ratings:** in scope at all (consent, ethics, crowd platform), or log-based
  evaluation only?
- **Sound:** add a speaker to the robot specification?

## References

Grouped by topic. † = finding summarised from the searching agent's knowledge (no
abstract returned); check before quoting.

**Personality in HRI: reviews and studies**
- Robert, L. P. et al. (2020). A Review of Personality in Human–Robot Interactions. *Foundations and Trends in Information Systems* 4(2). doi:10.1561/2900000018
- Mou, Y. et al. (2020). A Systematic Review of the Personality of Robot. *IJHCI* 36(6). doi:10.1080/10447318.2019.1663008
- Santamaria, T. & Nathan-Roberts, D. (2017). Personality Measurement and Design in Human-Robot Interaction: A Systematic and Critical Review. *Proc. HFES* 61(1). doi:10.1177/1541931213601686
- Esterwood, C. et al. (2022). A Personable Robot: Meta-Analysis of Robot Personality and Human Acceptance. *IEEE RA-L* 7(3). doi:10.1109/lra.2022.3178795
- Lee, K. M., Peng, W., Jin, S.-A., Yan, C. (2006). Can Robots Manifest Personality? *J. Communication* 56(4). doi:10.1111/j.1460-2466.2006.00318.x
- Tapus, A., Ţăpuş, C., Matarić, M. (2008). User–robot personality matching and assistive robot behavior adaptation for post-stroke rehabilitation therapy. *Intelligent Service Robotics* 1(2). doi:10.1007/s11370-008-0017-4 †
- Andrist, S., Mutlu, B., Tapus, A. (2015). Look Like Me: Matching Robot Personality via Gaze to Increase Motivation. *CHI '15*. doi:10.1145/2702123.2702592
- Craenen, B. et al. (2018). Do We Really Like Robots that Match our Personality? The Case of Big-Five Traits, Godspeed Scores and Robotic Gestures. *RO-MAN 2018*, 626–631. doi:10.1109/roman.2018.8525672
- Lim, V. et al. (2022). We are all Individuals: The Role of Robot Personality and Human Traits in Trustworthy Interaction. *RO-MAN 2022*. doi:10.1109/RO-MAN53752.2022.9900772 (arXiv:2307.15568)
- Woods, S. et al. (2005). Is this robot like me? Links between human and robot personality traits. *IEEE Humanoids*. doi:10.1109/ichr.2005.1573596 †
- Isbister, K. & Nass, C. (2000). Consistency of personality in interactive characters. *IJHCS* 53(2). doi:10.1006/ijhc.2000.0368 †
- Joosse, M. et al. (2013). What you do is who you are: The role of task context in perceived social robot personality. *ICRA 2013*. doi:10.1109/icra.2013.6630863 †
- Meerbeek, B. et al. (2008). The influence of robot personality on perceived and preferred level of user control. *Interaction Studies* 9(2). doi:10.1075/is.9.2.04mee
- Hendriks, B. et al. (2011). Robot Vacuum Cleaner Personality and Behavior. *IJSR* 3(2). doi:10.1007/s12369-010-0084-5
- Moujahid, M. et al. (2023). Come Closer: The Effects of Robot Personality on Human Proxemics Behaviours. arXiv:2309.02979
- Bera, A., Randhavane, T., Manocha, D. (2017). Aggressive, Tense or Shy? Identifying Personality Traits from Crowd Videos. *IJCAI 2017*. doi:10.24963/ijcai.2017/17
- Paetzel-Prüsmann, M., Perugia, G., Castellano, G. (2021). The influence of robot personality on the development of uncanny feelings. *Computers in Human Behavior* 120. doi:10.1016/j.chb.2021.106756
- Durupinar, F. et al. (2011). How the OCEAN Personality Model Affects the Perception of Crowds. *IEEE CG&A* 31(3). doi:10.1109/mcg.2009.105
- Churamani, N. et al. (2022). Affect-Driven Modelling of Robot Personality for Collaborative Human-Robot Interactions. *Frontiers in Robotics and AI*. doi:10.3389/frobt.2022.717193 (arXiv:2010.07221)
- Bethel, C. & Murphy, R. (2008). Survey of Non-facial/Non-verbal Affective Expressions for Appearance-Constrained Robots. *IEEE TSMC-C* 38(1). doi:10.1109/tsmcc.2007.905845; companion study *Paladyn* (2010), doi:10.2478/s13230-011-0009-5

**Character architectures and motivation**
- Moshkina, L., Park, S., Arkin, R. C., Lee, J. K., Jung, H. (2011). TAME: Time-Varying Affective Response for Humanoid Robots. *IJSR* 3(3). doi:10.1007/s12369-011-0090-2; first proposed in Moshkina & Arkin, *IEEE SMC 2003*, doi:10.1109/icsmc.2003.1244505
- Park, S., Moshkina, L., Arkin, R. C. (2010). Mood as an affective component for robotic behavior with continuous adaptation via Learning Momentum. *IEEE-RAS Humanoids*. doi:10.1109/ichr.2010.5686845
- Gebhard, P. (2005). ALMA: a layered model of affect. *AAMAS 2005*. doi:10.1145/1082473.1082478
- Kirby, R., Forlizzi, J., Simmons, R. (2010). Affective social robots. *Robotics and Autonomous Systems* 58(3). doi:10.1016/j.robot.2009.09.015
- Arkin, R. C., Fujita, M., Takagi, T., Hasegawa, R. (2003). An ethological and emotional basis for human–robot interaction. *RAS* 42(3-4). doi:10.1016/s0921-8890(02)00375-5
- Breazeal, C. (2003). Emotion and sociable humanoid robots. *IJHCS* 59(1-2). doi:10.1016/s1071-5819(03)00018-1
- Lewis, M. & Cañamero, L. (2016). Hedonic quality or reward? A study of basic pleasure in homeostasis and decision making of a motivated autonomous robot. *Adaptive Behavior* 24(5). doi:10.1177/1059712316666331; Cañamero, D. (1997), *Autonomous Agents '97*, doi:10.1145/267658.267688
- Hiolle, A., Lewis, M., Cañamero, L. (2014). Arousal regulation and affective adaptation to human responsiveness by a robot that explores and learns a novel environment. *Frontiers in Neurorobotics* 8. doi:10.3389/fnbot.2014.00017
- Keramati, M. & Gutkin, B. (2014). Homeostatic reinforcement learning for integrating reward collection and physiological stability. *eLife* 3. doi:10.7554/elife.04811
- Ortony, A., Clore, G., Collins, A. (1988; 2nd ed. 2022). *The Cognitive Structure of Emotions*. Cambridge UP. doi:10.1017/cbo9780511571299
- Marsella, S. & Gratch, J. (2009). EMA: A process model of appraisal dynamics. *Cognitive Systems Research* 10(1). doi:10.1016/j.cogsys.2008.03.005
- Becker-Asano, C. & Wachsmuth, I. (2010). Affective computing with primary and secondary emotions in a virtual human. *JAAMAS* 20(1). doi:10.1007/s10458-009-9094-9
- Mascarenhas, S. et al. (2022). FAtiMA Toolkit. *ACM TiiS* 12(1). doi:10.1145/3510822
- Moerland, T., Broekens, J., Jonker, C. (2018). Emotion in reinforcement learning agents and robots: a survey. *Machine Learning* 107. doi:10.1007/s10994-017-5666-0
- Broekens, J. (2018). A temporal difference reinforcement learning theory of emotion. arXiv:1807.08941

**Mood as integrated outcomes**
- Rutledge, R. et al. (2014). A computational and neural model of momentary subjective well-being. *PNAS* 111(33). doi:10.1073/pnas.1407535111
- Keren, H. et al. (2021). *eLife* 10. doi:10.7554/elife.62051 (primacy weighting)
- Eldar, E., Rutledge, R., Dolan, R., Niv, Y. (2016). Mood as representation of momentum. *Trends in Cognitive Sciences*. https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(15)00174-6
- Eldar, E. & Niv, Y. (2015). Interaction between emotional state and learning underlies mood instability. *Nature Communications* 6. doi:10.1038/ncomms7149
- Blain, B. & Rutledge, R. (2020). Momentary subjective well-being depends on learning and not reward. *eLife* 9. doi:10.7554/elife.57977

**Character from learning parameters and experience**
- Oudeyer, P.-Y., Kaplan, F., Hafner, V. (2007). Intrinsic motivation systems for autonomous mental development. *IEEE Trans. Evolutionary Computation*. https://ieeexplore.ieee.org/abstract/document/4141061/
- Schmidhuber, J. (2010). Formal Theory of Creativity, Fun, and Intrinsic Motivation. *IEEE TAMD* 2(3). doi:10.1109/tamd.2010.2056368 †
- Bellemare, M. et al. (2016). Unifying Count-Based Exploration and Intrinsic Motivation. arXiv:1606.01868
- Lefebvre, G. et al. (2017). Behavioural and neural characterization of optimistic reinforcement learning. *Nature Human Behaviour* 1(4). doi:10.1038/s41562-017-0067
- Mihatsch, O. & Neuneier, R. (2002). Risk-Sensitive Reinforcement Learning. *Machine Learning* 49. doi:10.1023/a:1017940631555 †
- Dabney, W. et al. (2020). A distributional code for value in dopamine-based reinforcement learning. *Nature* 577. doi:10.1038/s41586-019-1924-6 †
- Denrell, J. & March, J. (2001). Adaptation as Information Restriction: The Hot Stove Effect. *Organization Science* 12(5). doi:10.1287/orsc.12.5.523.10092
- Piray, P. et al. (2019). Emotionally Aversive Cues Suppress Neural Systems Underlying Optimal Learning in Socially Anxious Individuals. *J. Neuroscience* 39(8). doi:10.1523/jneurosci.1394-18.2018
- Sih, A., Bell, A., Johnson, J. C. (2004). Behavioral syndromes: an ecological and evolutionary overview. *Trends Ecol. Evol.* 19(7). doi:10.1016/j.tree.2004.04.009
- Réale, D. et al. (2007). Integrating animal temperament within ecology and evolution. *Biological Reviews* 82(2). doi:10.1111/j.1469-185x.2007.00010.x
- Wolf, M. et al. (2007). Life-history trade-offs favour the evolution of animal personalities. *Nature* 447. doi:10.1038/nature05835 †
- Read, S. et al. (2010). A neural network model of the structure and dynamics of human personality. *Psychological Review* 117(1). doi:10.1037/a0018131 †
- Ritschel, H. & André, E. (2017). Real-Time Robot Personality Adaptation based on Reinforcement Learning and Social Signals. *HRI Companion*. doi:10.1145/3029798.3038381; Ritschel, Baur, André, *RO-MAN 2017*, doi:10.1109/roman.2017.8172330
- Ceha, J. et al. (2019). Expression of Curiosity in Social Robots. *CHI 2019*. doi:10.1145/3290605.3300636 †
- Tanevska, A. et al. (2020). A Socially Adaptable Framework for Human-Robot Interaction. *Frontiers in Robotics and AI* 7. doi:10.3389/frobt.2020.00121
- Holmgård, C. et al. (2019). Automated Playtesting With Procedural Personas Through MCTS With Evolved Heuristics. *IEEE Trans. Games* 11(4). doi:10.1109/tg.2018.2808198

**LLM personas**
- Serapio-García, G. et al. (2023). Personality Traits in Large Language Models. arXiv:2307.00184
- Han, P. et al. (2025). The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs. arXiv:2509.03730
- Zierahn, K., Cachero, C., Korhonen, A., Oliver, N. (2026). Personality Without Persons? A Psychometric Critique of Big Five Testing in Large Language Models. arXiv:2607.02325
- Fitz, S. et al. (2025). Psychometric Personality Shaping Modulates Capabilities and Safety in Language Models. arXiv:2509.16332
- Ashok, A. et al. (2026). Why did My Robot Just Change Personality? Prompting Guidelines for a Grounded Robot Persona in LLM-Based HRI. arXiv:2608.26182

**Expressive motion, faces and sound**
- Takayama, L., Dooley, D., Ju, W. (2011). Expressing thought. *HRI '11*. doi:10.1145/1957656.1957674 †
- Ribeiro, T. & Paiva, A. (2012). The illusion of robotic life. *HRI '12*. doi:10.1145/2157689.2157814 †
- Hoffman, G. & Ju, W. (2014). Designing Robots With Movement in Mind. *J. Human-Robot Interaction* 3(1). doi:10.5898/jhri.3.1.hoffman †
- Schulz, T., Torresen, J., Herstad, J. (2019). Animation Techniques in Human-Robot Interaction User Studies. *ACM THRI* 8(2). doi:10.1145/3317325
- Gielniak, M. & Thomaz, A. (2011). Generating anticipation in robot motion. *RO-MAN 2011*. doi:10.1109/roman.2011.6005255 †
- van Breemen, A. (2004). Animation engine for believable interactive user-interface robots. *IROS 2004*. doi:10.1109/iros.2004.1389845 †
- Knight, H. & Simmons, R. (2014). Expressive motion with x, y and theta: Laban Effort Features for mobile robots. *RO-MAN 2014*. doi:10.1109/roman.2014.6926264 †; (2015) Layering Laban Effort Features on Robot Task Motions, *HRI '15 Ext. Abstracts*, doi:10.1145/2701973.2702054 †
- Emir, E. & Burns, C. (2022). A Survey on Robotic Vacuum Cleaners: Evaluation of Expressive Robotic Motions based on the Framework of Laban Effort Features for Robot Personality Design. *Proc. HFES* 66(1). doi:10.1177/1071181322661040
- Saerbeck, M. & Bartneck, C. (2010). Perception of affect elicited by robot motion. *HRI 2010*. doi:10.1109/hri.2010.5453269 †
- Zhou, A., Hadfield-Menell, D., Nagabandi, A., Dragan, A. (2017). Expressive Robot Motion Timing. *HRI 2017*. doi:10.1145/2909824.3020221 (arXiv:1802.01536)
- Dragan, A., Lee, K., Srinivasa, S. (2013). Legibility and predictability of robot motion. *HRI 2013*. doi:10.1109/hri.2013.6483603 †
- Lichtenthäler, C. et al. (2012). Influence of legibility on perceived safety in a virtual human-robot path crossing task. *RO-MAN 2012*. https://ieeexplore.ieee.org/abstract/document/6343829/; Lichtenthäler & Kirsch (2016), Legibility of robot behavior: a literature review, https://hal.science/hal-01306977/
- Kruse, T., Pandey, A., Alami, R., Kirsch, A. (2013). Human-aware robot navigation: A survey. *Robotics and Autonomous Systems*. https://www.sciencedirect.com/science/article/pii/S0921889013001048
- Moon, A. et al. (2021). Design of Hesitation Gestures for Nonverbal Human-Robot Negotiation of Conflicts. *ACM THRI* 10(3). doi:10.1145/3418302
- Kalegina, A. et al. (2018). Characterizing the Design Space of Rendered Robot Faces. *HRI 2018*. doi:10.1145/3171221.3171286 †
- Kawabe, T., Akiyama, R., Yokosaka, T. (2025). The influence of eye position on the animacy impression of a cube-shaped robot in motion. *i-Perception* 16(2). doi:10.1177/20416695251323769
- Hart, J. et al. (2020). Using Human-Inspired Signals to Disambiguate Navigational Intentions. *ICSR 2020*, LNCS. doi:10.1007/978-3-030-62056-1_27 †
- Kozima, H., Michalowski, M., Nakagawa, C. (2009). Keepon. *IJSR* 1(1). doi:10.1007/s12369-008-0009-8 †
- Chan, L., Zhang, B., Fitter, N. (2021). Designing and Validating Expressive Cozmo Behaviors for Accurately Conveying Emotions. *RO-MAN 2021*. doi:10.1109/ro-man50785.2021.9515425 †
- Baraka, K. & Veloso, M. (2018). Mobile Service Robot State Revealing Through Expressive Lights. *IJSR* 10(1). doi:10.1007/s12369-017-0431-x †
- Read, R. & Belpaeme, T. (2014). Situational context directs how people affectively interpret robotic non-linguistic utterances. *HRI 2014*. doi:10.1145/2559636.2559680; (2016) People Interpret Robotic Non-linguistic Utterances Categorically, *IJSR* 8(1), doi:10.1007/s12369-015-0304-0 †
- Jee, E.-S. et al. (2010). Sound design for emotion and intention expression of socially interactive robots. *Intelligent Service Robotics* 3(3). doi:10.1007/s11370-010-0070-7 †
- Allen, A., Drummond, T., Kulić, D. (2024/25). Robots Have Been Seen and Not Heard: Effects of Consequential Sounds on Human-Perception of Robots. arXiv:2406.02938 (IEEE RA-L, doi:10.1109/LRA.2025.3546097); see also arXiv:2502.02051

**Evaluation instruments and study design**
- Bartneck, C. et al. (2009). Measurement Instruments for the Anthropomorphism, Animacy, Likeability, Perceived Intelligence, and Perceived Safety of Robots. *IJSR* 1(1). doi:10.1007/s12369-008-0001-3
- Ho, C.-C. & MacDorman, K. (2010). Revisiting the uncanny valley theory: Developing and validating an alternative to the Godspeed indices. *Computers in Human Behavior* 26(6). doi:10.1016/j.chb.2010.05.015
- Carpinella, C. et al. (2017). The Robotic Social Attributes Scale (RoSAS). *HRI '17*. doi:10.1145/2909824.3020208
- Pithayarungsarit, P. et al. (2025). A Systematic Validation of the Robotic Social Attributes Scale (RoSAS). *HRI 2025*. doi:10.1109/hri61500.2025.10973879 (result not retrieved)
- Gray, H., Gray, K., Wegner, D. (2007). Dimensions of mind perception. *Science*. https://www.science.org/doi/abs/10.1126/science.1134475
- Gosling, S., Rentfrow, P., Swann, W. (2003). A very brief measure of the Big-Five personality domains. *J. Research in Personality* 37(6). doi:10.1016/s0092-6566(03)00046-1
- Woods, S. et al. (2006). Methodological Issues in HRI: A Comparison of Live and Video-Based Methods in Robot to Human Approach Direction Trials. *RO-MAN 2006*. doi:10.1109/roman.2006.314394
- Li, J. (2015). The benefit of being physically present. *IJHCS* 77. doi:10.1016/j.ijhcs.2015.01.001
- Leite, I., Martinho, C., Paiva, A. (2013). Social Robots for Long-Term Interaction: A Survey. *IJSR* 5(2). doi:10.1007/s12369-013-0178-y
- Paetzel, M., Perugia, G., Castellano, G. (2020). The Persistence of First Impressions. *HRI 2020*. doi:10.1145/3319502.3374786
- Riek, L. (2012). Wizard of Oz Studies in HRI: A Systematic Review and New Reporting Guidelines. *J. Human-Robot Interaction*. doi:10.5898/jhri.1.1.riek

**Ethics, deception and transparency**
- Sharkey, A. & Sharkey, N. (2021). We need to talk about deception in social robotics! *Ethics and Information Technology* 23(3). doi:10.1007/s10676-020-09573-9
- Sharkey, N. & Sharkey, A. (2010). The crying shame of robot nannies. *Interaction Studies* 11(2). doi:10.1075/is.11.2.01sha
- Danaher, J. (2020). Robot Betrayal: a guide to the ethics of robotic deception. *Ethics and Information Technology*. https://link.springer.com/article/10.1007/s10676-019-09520-3
- Coeckelbergh, M. (2012). Are Emotional Robots Deceptive? *IEEE Trans. Affective Computing* 3(4). doi:10.1109/t-affc.2011.29
- Leong, B. & Selinger, E. (2019). Robot Eyes Wide Shut: Understanding Dishonest Anthropomorphism. *FAT\* '19*. doi:10.1145/3287560.3287591
- Boden, M. et al. (2017). Principles of robotics: regulating robots in the real world. *Connection Science* 29(2). doi:10.1080/09540091.2016.1271400
- Wortham, R. & Theodorou, A. (2017). Robot transparency, trust and utility. *Connection Science* 29(3). doi:10.1080/09540091.2017.1313816
- Winfield, A. et al. (2021). IEEE P7001: A Proposed Standard on Transparency. *Frontiers in Robotics and AI* 8. doi:10.3389/frobt.2021.665729
- van Straten, C., Peter, J., Kühne, R., Barco, A. (2020). Transparency about a Robot's Lack of Human Psychological Capacities. *ACM THRI* 9(2). doi:10.1145/3365668
- Arbib, M. & Fellous, J.-M. (2004). Emotions: from brain to robot. *Trends in Cognitive Sciences* 8(12). doi:10.1016/j.tics.2004.10.004
- Sloman, A., Chrisley, R., Scheutz, M. (2005). The Architectural Basis of Affective States and Processes. In *Who Needs Emotions?* (OUP). doi:10.1093/acprof:oso/9780195166194.003.0008
- Cañamero, L. (2005). Emotion understanding from the perspective of autonomous robots research. *Neural Networks* 18(4). doi:10.1016/j.neunet.2005.03.003; Cañamero (2021), *IEEE TAFFC* 12(2), doi:10.1109/taffc.2019.2908162
- Złotowski, J., Proudfoot, D., Yogeeswaran, K., Bartneck, C. (2015). Anthropomorphism: Opportunities and Challenges in Human–Robot Interaction. *IJSR* 7(3). doi:10.1007/s12369-014-0267-6
