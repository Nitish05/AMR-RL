# Research index: what the literature told the project, and what was decided

This folder holds desk research: literature reviews that shaped a design before anything was
built. Results with denominators live in `docs/results/`; several of those also open with a
literature section, listed here so that all of the project's research is in one place.

Conventions in every review: each paper cited came back from a search in that session (through
the paper-search tools or, in the earliest rounds, web search); † marks a finding summarised
from a searching agent's knowledge because no abstract was returned, to be checked before a
number is quoted; "preprint" marks unreviewed work carrying design weight.

## The reviews, in the order they happened

| When | Question | Where | What the literature said | What was decided |
|---|---|---|---|---|
| Round 4 (Oct 3) | Why did relocalisation accept wrong poses? | [results/relocalisation.md](../results/relocalisation.md) | Whole-map ORB matching is imprecise on self-similar floors; PnP collapses below ~30 % correct matches; a planar two-point RANSAC with view-change confirmation fits a ground robot | Planar two-point relocalisation with probation became the default; retrieval-first (round 5) added a small gain; range beyond 0.5 m is still the open problem |
| Round 4–6 | Can the robot see objects placed on its route? | [results/near-field-guard.md](../results/near-field-guard.md) | Monocular depth (Depth Anything V2 Small, Apache-2.0) scaled by the visible floor; larger open models were non-commercial or too coarse | Depth guard on by default when the model is installed; 0 of 5 box tests in contact |
| Round 5 (Oct 3) | How to close loops without wrong corrections? | [results/loop-closure.md](../results/loop-closure.md) §"What the literature offers" | MegaLoc/SALAD lead place-recognition benchmarks; ROVER's distortion check; GNC-Cauchy robust back-ends; two-view loop constraints suffice (2GO) | MegaLoc retrieval, planar verification against the candidate's own landmarks, SE(2) pose graph with GNC-Cauchy edges; 0 false closures in the round-9 validations once the gyro guard was added |
| Round 6 (Oct 3) | Can perception be swapped for learned models? | [results/textured-worlds.md](../results/textured-worlds.md), [results/vlm-outcome-pilot.md](../results/vlm-outcome-pilot.md) | OmDet-Turbo beat LLMDet-tiny on 320×240 frames; DINOv2 re-ID is beaten by hue in these worlds; Qwen3-VL-2B can judge interaction outcomes offline but misses 4 % of red flags | Open-vocabulary detector kept optional (now shown to make textured-world learning work: restart 6/6, reversal 6/6); VLM outcome judging not in the runtime |
| Round 7 (Oct 4) | Why does heading drift in in-place turns? | [results/turn-drift.md](../results/turn-drift.md) §"What was found first", [results/turn-drift-options.md](../results/turn-drift-options.md) | The tail is wrong depth times the lever arm, not edge aliasing; delayed landmark initialisation (ORB-SLAM, SVO, inverse depth), the command as a "virtual gyro", soft priors instead of hard gates; a differential drive cannot rotate about its camera; within 0.25 m of a wall almost no floor is visible; alternative SLAMs are GPL/non-commercial or CUDA-only | Eight camera-only candidates tested, none adopted; the physics argument led to round 9 |
| Round 9 (Oct 5–11) | Add real sensors: IMU and wheel encoders | [plans/heading-robustness-plan.md](../plans/heading-robustness-plan.md), [results/heading-r9.md](../results/heading-r9.md), [results/r9-followup.md](../results/r9-followup.md) | LSM6DSOX datasheet and Allan analogues for the error model; gyro offset as a temperature-compensated Kalman state; zero-velocity updates at stops (Kilic 2019); slip as INS-vs-wheel velocity (Kilic, Gu & Gross 2022); the MER slip check (Maimone 2007); current-based slip (Ojeda 2006); loop-closure gating as in ORB-SLAM3/VINS, Kimera-RPGO and RRR | IMU + encoder odometry became the default, validated on untouched starts; loop guard and blocked check added; two owner decisions open (blocked-wheel handling, first recalibration stop) |
| Oct 2026 | Can one VLM or VLA run all of Pip? | [VLM_VLA_SURVEY.md](VLM_VLA_SURVEY.md) | No released model does localisation, certified navigation, identity, interaction, persistent learning and expression together; navigation VLAs need a CUDA GPU and output coordinates, which breaks the semantics rule; open VLMs can take the perception layer on a Mac | Recommended path: Qwen3-VL or Moondream behind the detector contract; NoMaD in shadow mode as a waypoint proposer; LightNav-0 only with a GPU |
| Oct 12 | Personality and character: options | [PERSONALITY_RESEARCH.md](PERSONALITY_RESEARCH.md) (about 110 papers) | People read personality from simple robots' motion; effects on acceptance are modest and mixed; character architectures layer traits, attitudes, moods and emotions (TAME); traits can be learning parameters; LLM personas are not behaviourally consistent; ethics asks for every cue to map to logged state | Owner chose: all four temperament traits, learning-style traits, a persistent evolving character, sparse LLM guidance, human ratings, a speaker |
| Oct 12 | Working out those decisions | [PERSONALITY_RESEARCH_2.md](PERSONALITY_RESEARCH_2.md) (about 200 papers) | Optimism and caution are one operation unless split across update and decision time; persistence and tempo share the learned reward rate; human learning parameters are not stable traits; the hot-stove effect; budgeted advising and metareasoning for "only when necessary"; RoSAS's three-factor model fits poorly; sound is read categorically | One definition per trait with an identifiability gate; traits change only at session end, bounded, no mean reversion; LLM consultation gated, shadow period first, schema-bounded, verified; rating protocols with an overclaiming audit; a discrete, event-paired sound set |
| Oct 12 | Would a small VLA such as SmolVLA help? | [VLA_FOR_PIP.md](VLA_FOR_PIP.md) (about 115 papers) | No published VLA drives a two-wheeled robot from one camera on a CPU; SmolVLA is an arm policy that needs demonstrations on the target body and a GPU to fine-tune; headline VLA success collapses under perturbation and runtime monitors catch under half of deviations; what the VLA literature converged on is the hierarchy Pip already has (a model proposes, a classical planner verifies) and asynchronous chunked inference; foundation models help at the edges (offline outcome audit, priors for new object classes, interestingness ranking, offline cue design) | No VLA in the control loop and none for the personality work; borrow the asynchronous pattern; VLM/LLM only at the edges, read-only or bounded; if a learned micro-skill is ever tried, ACT or Diffusion Policy before SmolVLA, behind the shield; revisit when a VLA reports a usable rate on an ARM CPU |

## What the research agrees on across rounds

1. **Keep the classical, auditable core.** Every review that looked at replacing metric SLAM,
   the certified planner, the supervisor or the SQLite memory with a learned model found no
   model that could do it on this hardware, and several reasons not to (no auditable pose, no
   certification, outputs that break the semantics rule). Learned models help one layer at a
   time, behind an existing contract: pixels in, labelled regions out (detector); before/after
   images in, outcome label out (judge); image in, logged untrusted proposal out (waypoint
   proposer); logged aggregates in, bounded parameter proposal out (adviser).
2. **Physics before estimation.** The turn-drift rounds spent two rounds on camera-only
   estimators before the literature's physical argument (no floor visible near walls; the camera
   is off the turning axis) justified adding sensors. The IMU round then fixed in one step what
   eight software candidates could not.
3. **Honesty is a design constraint that the literature supports, not an obstacle.** Animal-
   personality research names traits by measurable behaviour with no inference about inner
   states; the ethics literature asks that displays never conceal real state and that machine
   nature stay transparent; the HRI evidence says incongruent cues hurt. All three point to the
   same rule the project already had: every cue maps to logged state, engineered against learned
   is labelled, emotion words are not used.
4. **Evaluation discipline.** Pre-registration with addenda, untouched seeds, paired arms,
   denominators, failures retained, and claims scoped when a criterion fails. The reviews added:
   simulate-then-recover before calling a parameter a trait, overclaiming checks in any rating
   study, and label clip-based human ratings "simulation, video, single exposure".
5. **Compute reality.** Everything runs on a Mac CPU at below real time; the eventual robot is
   small. Models that need a CUDA GPU are optional and offboard at best. Small open VLMs run
   locally; the open-vocabulary detector already does, at 0.08 s per frame on MPS.

## Open owner decisions gathered from all reviews

- ~~Blocked or spinning wheels~~: decided 2026-10-12, the pose is kept and the robot backs off
  (validation F running; adoption by its pre-registered rules).
- ~~First recalibration stop~~: decided 2026-10-12, 0.5 °C until the tempco is known (validation F).
- **Default version:** keep the re-validated round-9 build despite the failed map-error criterion
  (kept so far; validation F decides the two new defaults).
- ~~Open-vocabulary detector as default~~: decided 2026-10-12, it is the default when its model is cached.
- ~~Character~~: implemented 2026-10-12 as decided (trait definitions, no mean reversion,
  session-end-only change, shadow-only adviser with a template backend, ratings never feed back
  directly). Open: switching the mapping on, which waits for the identifiability gates
  (docs/results/character-gate-prereg.md); wiring an external model behind the gate; running the
  rating study (docs/results/character-ratings-prereg.md).
- ~~Speaker~~: implemented as a logged event bus with no speech (expression/sound.py); the
  acoustic signature remains a convention to validate with listeners.
- **Repository:** licence; removing three merged worktrees (the research documents are committed).
- **VLA:** none in the control loop (recommended); whether to spend an afternoon measuring SmolVLA's
  latency on the Mac; whether training labels for any future learned module may come from the
  evaluator's ground truth or only from teleoperation.
