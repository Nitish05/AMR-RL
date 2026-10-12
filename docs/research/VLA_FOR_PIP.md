# Would a small VLA such as SmolVLA help Pip? (literature review 3, October 2026)

Desk research only: nothing here was built or run. It answers the owner's question "what about
VLAs? would something like SmolVLA or something similar help in any way?", after the earlier survey
([VLM_VLA_SURVEY.md](VLM_VLA_SURVEY.md)) had already found that no single model can run Pip's
whole stack.

**Method.** Five topic sweeps through the paper-search tools (112 searches, 110 papers: small
and efficient VLAs; VLAs on wheeled and mobile robots; expressive behaviour from foundation
models; safety and honesty around learned policies; foundation models in the learning loop), two
citation-verification passes (104 citations re-looked-up: 95 confirmed; 7 arXiv items not
confirmable because the arXiv API returned nothing for them; 2 title mismatches corrected
below), a completeness critique, and a read of the SmolVLA paper itself, since every sweep had
only its abstract. Semantic Scholar and Google Scholar were rate-limited, so coverage rests on
arXiv and Crossref; most VLA work is arXiv-first and much of it is 2025–2026 preprints, which is
noted where a number is quoted. † marks a finding summarised from a searching agent's knowledge
because no abstract was returned.

## Short answer

**No.** A small VLA does not help Pip in any way that changes the design, and it adds nothing to
the personality work. Three indirect things do transfer, and one narrow experiment is arguable.

Why not:
- **Wrong embodiment and action space.** SmolVLA is an imitation-trained arm policy. Read from
  the paper: 450 million parameters (about 100 million in the action expert, on a SmolVLM-2
  backbone using 16 of its language layers), pretrained on fewer than 30 000 episodes from 481
  community datasets, fine-tuned and evaluated on SO-100 and SO-101 arms, 50-step action chunks,
  about 30 000 GPU-hours for the whole project. No wheeled base appears in the paper; no CPU
  latency is reported anywhere in it, only the sentence that it can be "deployed on consumer-grade
  GPUs or even CPUs" and a figure noting the policy may run on a remote GPU server. Across all five
  sweeps, no paper was found in which any VLA drives a robot like Pip (0.3 m, two wheels, one
  camera, no GPU), and VLAs, unlike VLMs, need fine-tuning for each new embodiment (Mitra 2025).
- **Compute.** Every fine-tuning recipe found needs a CUDA GPU (the smallest: a 0.5B backbone,
  8 hours on one consumer GPU, VLA-Adapter). The Mac has none. The best "onboard" results run on
  Jetson Orin GPUs (vla.cpp; EagleVLA), and compression gives 1.5–2× (SQAP-VLA) or 4–4.6× with
  ternary kernels, not the order of magnitude a Raspberry-Pi-class CPU needs.
- **Reliability.** Headline VLA scores do not survive perturbation: models above 90 % on LIBERO
  fell to 0 % when objects, layouts or instructions changed (LIBERO-PRO); paraphrased instructions
  cost 22–52 points (LIBERO-Para); runtime monitors caught 44.6 % of navigation deviations at
  11.7 % false positives (Jeong 2026); safety alignment leaves residual violations (SafeVLA;
  ShieldVLA). A learned controller cannot own safety, which is also what the project rules say.
- **Pip's hard problems are elsewhere.** Metric SLAM, certified navigation, auditable count-based
  attitudes, and cues that map to logged state are the layers the rules keep classical. The
  personality programme is about which activity to choose and how cues map to state, not how to
  turn the wheels.

What transfers:
1. **The asynchronous, chunked inference pattern** (SmolVLA; real-time chunking, Black 2025;
   A2C2, Sendai 2025; EagleVLA): slow perception produces timestamped proposals that the fast
   loop re-validates before use, so a slow module never stalls the 10 Hz tick. SmolVLA's own
   measurement: asynchronous inference finished tasks in 9.7 s against 13.75 s synchronous and
   completed 19 cycles against 9 in a fixed minute. Pip can adopt the pattern with no model.
2. **The propose-and-verify hierarchy**, which is what every working navigation VLA converged on
   (NaVILA's language-level sub-goals; Mobility VLA; O2C-Nav's one call per decision over
   projected waypoints; SmartWay; VAMOS; APPLV's planner parameters instead of actions). Pip
   already has it: the detector, the planned sparse LLM guidance and the occupancy-map planner.
3. **Foundation models at the edges, read-only or bounded** (section 4): an offline outcome
   auditor, a weak source-tagged prior for new object classes, a once-per-session interestingness
   ranking, and an LLM used offline to draft cue tables over engineered primitives.

The arguable experiment: a gated learned micro-skill for approach-and-nudge, trained from
teleoperated Genesis episodes and run behind the classical shield. If it is ever tried, the
data-efficient behaviour-cloning line is the right candidate, not a VLA: ACT learned real tasks
from 10 minutes of demonstrations (Zhao 2023), Dobb-E reached 81 % over 109 household tasks on a
wheeled Stretch from 5-minute demonstrations (Shafiullah 2023), Diffusion Policy is the
generative-policy baseline (Chi 2023). Even then it is a learned actuator path that must sit behind
the shield, be labelled learned, and carry provenance in the SQLite record.

## 1. What SmolVLA is, read from the paper

| Claim | Status |
|---|---|
| 450 M parameters, ~100 M in the action expert; SmolVLM-2 backbone, first 16 LLM layers | confirmed, body |
| pretrained on < 30 k episodes from 481 community datasets (about 10 M frames); 200 k steps, batch 256, 4 GPUs; ~30 k GPU-hours project total | confirmed, body |
| flow-matching action expert, chunks of 50 actions, 10 flow steps; fine-tuning 100 k–200 k steps | confirmed, body |
| real-world evaluation on SO-100 and SO-101 arms (pick-place, stacking, sorting); beats ACT and a ~7× larger π0 there | confirmed, body |
| "deployed on consumer-grade GPUs or even CPUs" | abstract and introduction only; **no CPU latency number anywhere in the paper**; the inference figure notes the policy may run on a remote GPU server |
| asynchronous inference: 9.7 s against 13.75 s per task; 19 against 9 cycles in 60 s | confirmed, body (real arm) |
| any wheeled or mobile base (e.g. LeKiwi) | **absent** from the paper |

The earlier survey's "SmolVLA CPU-feasible" cell came from a model card; it is now marked
unmeasured there. The only measurement that could change the Mac-side verdict is an afternoon's
work: run `smolvla_base` through the LeRobot asynchronous stack on the M-series CPU with a 320×240
frame and record the latency. It would not change the embodiment problem.

## 2. VLAs on wheeled robots: what the evidence shows

- Where a learned policy does command a mobile base, it is a large platform with substantial
  data: Mobile ALOHA (50 demonstrations per task plus co-training on a large static set), PanoVLA
  (5.5 hours of whole-body teleoperation, 2026 preprint), π0's mobile manipulators, SPOC (millions
  of simulated frames for RGB-only navigation). TidyBot++ chose a holonomic base because
  nonholonomic kinematics "create complex and time-consuming motions" for learned mobile
  manipulation; Pip is nonholonomic (a design-rationale sentence, not an experiment).
- At Pip's scale the only evidence is Duckietown: the AI Driving Olympics organisers wrote in
  2019 that learned control outperforming classical control on embodied robots "remains largely
  unproven", and the winning entry was a narrow imitation-learned lane follower (2020). Dobb-E
  (2023) weakens that claim for manipulation on a wheeled robot, but with a pretrained
  representation and tiny demonstrations, not a VLA.
- Cross-embodiment transfer: Open X-Embodiment is 22 arms; four robot types supply over 85 % of
  its real data; fine-tuning on unseen robot–gripper pairs gained 24–45 % only after heavy
  augmentation (OXE-AugE). One policy did transfer zero-shot across wheeled bodies (X-Nav;
  embodiment randomisation†), but through large-scale RL over randomised bodies, not a pretrained
  VLA.
- BARN: a learned local controller on a differential-drive Jackal competed only once wrapped in a
  finite-state machine with engineered safety checks (Mandala 2023; Xiao 2023), and "a significant
  amount of fine-tuning is still needed onsite".
- Navigation VLAs hallucinate often enough that a deviation detector and a fallback policy were
  needed (Jeong 2026, preprint).

## 3. Expression: could a foundation model generate Pip's character behaviour?

Not online, and not as the controller. What the field does is use the model offline, as a
designer over a fixed library of primitives:
- **GenEM** turns instructions into parametrised control code over existing skills (Mahadevan
  2024). **GPT-Driven Gestures** had an LLM choose motion-parameter values for named target states
  on a quadruped; 120 raters classified the states above random and comparably to human-designed
  expressions (Roy 2025). **EMOTION** generates humanoid gesture sequences by in-context learning
  (Huang 2025). The **Haru** system maps emojis in LLM output onto a fixed action library and
  documents the failure modes: repetition and hallucinated facts (Wang 2024). **Code as Policies**
  is the ancestor of all of these (Liang 2022).
- All keep a human-authored primitive library and a mapping from a named state to parameters,
  which is exactly Pip's rule.
- GPT-4 tracks human judgements of robot communicative acts (r ≈ 0.82–0.83) but rates behaviours
  more favourably than people do, and vision models failed on video stimuli (Wachowiak 2024): an
  LLM can pre-screen cue candidates, never replace human ratings.
- Grounding precedents: an expressive display driven purely by the learner's TD error made the
  learning transparent (Matarese 2020, preprint); trait parameters modulating a learner were
  perceptible (Churamani 2022); engineered expression primitives raised engagement on a
  non-anthropomorphic lamp robot (ELEGNT, Hu 2025); gaze planned from task state was more
  interpretable than reactive gaze (Mishra 2022); non-linguistic utterances can be synthesised to
  valence/arousal targets with a genetic algorithm and an acoustic model, no LLM needed (Khota
  2022, 2024); exaggerated faces beat realistic ones in one small study (Li 2020).

**Recommendation.** Use an LLM offline to propose and name candidate parametrisations of Pip's
primitives (screen pattern, approach-tempo profile, hesitation and retreat timing, sound
valence/arousal target) per logged state and trait bucket; review them; freeze them into a table.
Keep the planned sparse online LLM to choosing among pre-verified variants under schema
validation. Label the result "engineered (LLM-assisted design)". Do not put a VLA, a diffusion
policy or any free-form generator in the cue path: it would be an opaque learned actuator command
that cannot be traced to logged state.

## 4. Learning: where a foundation model honestly helps, and where it does not

Pip's learning problem is a discrete contextual bandit (two actions on a few object classes with
hidden responses) with no demonstrator and a few dozen trials per session. Count-based Beta
posteriors are the right estimator and the only auditable one. The RL-fine-tuning route for VLAs
confirms this by contrast: ConRFT needs 45–90 minutes of online fine-tuning per task with human
interventions; iRe-VLA states that online RL on VLAs is unstable and exceeds most local machines;
VLA-RL runs a 7B policy on GPU-vectorised environments; fleet-scale RL used 16 robots (Wang 2026);
and RL in one real scene turns a broad VLA into a scene-specific overfit (Choi 2026), which is
exactly what Pip's small world would do.

Where a model helps, outside the memory:
1. **Offline outcome auditor.** A SmolVLM-class model (256 M fits in under 1 GB; Marafioti 2025)
   labels logged before/after frame pairs after the session and is compared with the engineered
   response detector; disagreements go to review. Evidence that VLM judging works but needs care:
   SuccessVQA needed human annotations; video beat single images by 19 % (RoboVQA); CLIP reward
   models fail on spatial reasoning (Rocamonde 2023); pairwise preferences are more reliable than
   raw scores (RL-VLM-F). The earlier pilot with Qwen3-VL-2B missed 4 % of red flags
   ([results/vlm-outcome-pilot.md](../results/vlm-outcome-pilot.md)). The auditor writes to a
   separate table and never to the count-based memory.
2. **Weak priors for new object classes.** One offline query per new class yields one or two
   pseudo-counts in a separate, source-tagged prior column that real counts overwrite (the annealed
   guidance of VLAJS; affordance proposals as in MOKA and RoboPoint). Pip's rules are arbitrary by
   design, so a confident wrong prior slows learning for that class; cap it small.
3. **Curiosity targets.** A once-per-session LLM ranking of classes worth investigating, mixed
   with engineered learning progress (OMNI's "learnable and interesting"; ELLM), under the sparse
   guidance gate already recorded in [PERSONALITY_RESEARCH_2.md](PERSONALITY_RESEARCH_2.md).
   This evidence is from Crafter and Housekeep with text state, not robots.
4. **Not recommended:** a SOAR-style loop (a VLM proposes and judges episodes while a classical
   learner learns) needed 30 500 trajectories for a 2× gain; Pip's bandit converges in far fewer
   trials.

## 5. If a learned module is ever added: what keeps the rules

1. **Authority stays classical.** A model-predictive or control-barrier-function shield checks
   every learned proposal against the occupancy map (statistical model-predictive shielding†,
   Bastani 2021; CBF-RL, Yang 2025); the shield decision is a logged engineered state. FailBank
   documents the failure mode: persistent policy–shield disagreement stalls the robot, and a
   proposer that is overridden constantly is worse than none.
2. **Cheap monitors in the loop:** action-chunk consistency and a Mahalanobis out-of-distribution
   distance need no failure data (VLA-FAIL, 2026 preprint); a learned failure likelihood with a
   conformal threshold (SAFE); a conformal shift gate over Pip's own observables (Huang 2023);
   erratic failures caught cheaply, progression failures by a VLM queried only when time allows
   (Sentinel). Conformal guarantees assume exchangeable data; an evolving character violates that,
   so recalibration must be scheduled.
3. **Provenance before anything is wired in:** model id and hash, inputs, frame age, raw output,
   score, threshold, shield verdict, executed action, in the SQLite record and the capability
   ledger; transparency levels per stakeholder in the IEEE 7001 style†, and IEEE 7014 on emulated
   affect† (both characterised from knowledge of the standards, not from their text).
4. **Why-lines come from engineered decision variables** (Korpan 2017); VLA interpretability today
   yields steerable directions, not faithful explanations (Häon 2025).
5. **Evaluation under perturbation**, with seeds, and a SIMPLER-style measurement of the
   simulation-to-real gap before any transfer (Li 2024). Any module trained only in Genesis needs
   the domain-randomisation route to hardware (OpenAI 2018).
6. **Training labels.** The sweeps disagreed on whether a detector calibrated on evaluator ground
   truth may run operationally. That is an owner decision; whichever way, `tests/
   test_privilege_boundary.py` should enforce it.

## 6. Contradictions between the sweeps, resolved

| Question | Positions | Resolution |
|---|---|---|
| Try a SmolVLA micro-skill? | "the one defensible slot" against "never in the loop" | Not now. If ever, ACT or Diffusion Policy first, behind the shield, labelled learned; the experiment is a measurement, not a commitment |
| Who learns the interactions? | count-based memory only, against a learned proposer of signal/nudge | Counts only. Attitudes must stay reconstructible from logged interactions |
| LLM cadence | a VLM waypoint proposer every 10–30 s, against once per session | The recorded decision stands: consolidation-time, at most one query per session, shadow period first |
| Chunk latency at 10 Hz | "a second of wheel commands per chunk" against "well below 10 Hz with frame-age tags" | Neither figure is from a paper; measure before designing the supervisor's veto |
| Compression headroom | 1.5–2× against 4–4.6× ternary | The re-check trigger is stated against the ternary line: revisit when a ternary or int4 VLA reports a usable control rate on an ARM CPU without a GPU, or when LeRobot publishes a SmolVLA result on a wheeled base |
| Small-VLA size | "0.77 B" (Evo-1), "0.5 B backbone" (VLA-Adapter), "0.5–9 B" (prior survey) | SmolVLA is 450 M, read from the paper; the class spans roughly 0.4–1 B |

Two citation corrections from the verification passes: the DOI 10.3389/frobt.2022.717193 resolves
to "Affect-Driven *Learning* of Robot Behaviour for Collaborative Human-Robot Interactions"
(Churamani et al. 2022), not the arXiv title the sweeps used; and arXiv 2604.13733 carries the
title "Jump-Start Reinforcement Learning ..." in the Semantic Scholar record rather than
"Vision-Language-Action Jump-Starting ...". Seven arXiv items (Open X-Embodiment, the TNNLS VLA
survey, the AI-DO lane follower, Matarese 2020, Li 2020, BLURR, Korpan 2017) could not be
re-confirmed because the arXiv API returned nothing for them in that session; they are cited as
the searching agents recorded them.

## 7. Decisions for the owner

- **Record "no VLA in the control loop"** so the question stops recurring (done in
  [README.md](README.md) and the survey addendum).
- Whether to spend the afternoon measuring SmolVLA's latency on the Mac. It settles one cell of the
  survey and nothing else.
- Whether training or calibration labels for any future learned module may come from the
  evaluator's ground truth, or only from teleoperation.
- Whether to implement the two free wins now: the asynchronous proposal pattern for the detector
  and any VLM, and the provenance fields in the SQLite schema.

## References

Grouped by sweep. † = no abstract returned; the finding was summarised from the searching agent's knowledge.
**Small and efficient VLAs**
- Mustafa Shukor et al. (2025). SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. *arXiv preprint (cs.LG; cs.RO)*. arXiv:2506.01844
- Remi Cadene et al. (2026). LeRobot: An Open-Source Library for End-to-End Robot Learning. *arXiv preprint (cs.RO)*. arXiv:2602.22818
- Khanh D. Nguyen et al. (2026). vla.cpp: A Unified Inference Runtime for Vision-Language-Action Models. *arXiv preprint (cs.RO; cs.AI; cs.LG; eess.SY)*. arXiv:2606.08094
- Zebin Yang et al. (2026). EagleVLA: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference. *arXiv preprint (cs.RO; cs.AI)*. arXiv:2607.12659
- Junjie Wen et al. (2024). TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation. *arXiv preprint (cs.RO; cs.CV)*. arXiv:2409.12514
- Yihao Wang et al. (2025). VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model. *arXiv preprint (cs.RO)*. arXiv:2509.09372
- Moo Jin Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. *arXiv preprint (cs.RO; cs.LG)*. arXiv:2406.09246
- Moo Jin Kim et al. (2025). Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. *arXiv preprint (cs.RO; cs.AI; cs.CV; cs.LG)*. arXiv:2502.19645
- Octo Model Team et al. (2024). Octo: An Open-Source Generalist Robot Policy. *arXiv preprint (cs.RO; cs.LG)*. arXiv:2405.12213
- Kevin Black et al. (2024). pi0: A Vision-Language-Action Flow Model for General Robot Control. *arXiv preprint (cs.LG; cs.RO)*. arXiv:2410.24164
- Physical Intelligence et al. (2025). pi0.5: a Vision-Language-Action Model with Open-World Generalization. *arXiv preprint (cs.LG; cs.RO)*. arXiv:2504.16054
- Karl Pertsch et al. (2025). FAST: Efficient Action Tokenization for Vision-Language-Action Models. *arXiv preprint (cs.RO; cs.LG)*. arXiv:2501.09747
- NVIDIA et al. (2025). GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. *arXiv preprint (cs.RO; cs.AI; cs.LG)*. arXiv:2503.14734
- Anthony Brohan et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *arXiv preprint (cs.RO; cs.CL; cs.CV; cs.LG)*. arXiv:2307.15818
- Delin Qu et al. (2025). SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model. *arXiv preprint (cs.RO; cs.AI)*. arXiv:2501.15830
- Kevin Black et al. (2025). Real-Time Execution of Action Chunking Flow Policies. *arXiv preprint (cs.RO; cs.AI; cs.LG)*. arXiv:2506.07339
- Kohei Sendai et al. (2025). Leave No Observation Behind: Real-time Correction for VLA Action Chunks. *arXiv preprint (cs.RO; cs.AI; cs.CV; eess.SY)*. arXiv:2509.23224
- Hengyu Fang et al. (2025). SQAP-VLA: A Synergistic Quantization-Aware Pruning Framework for High-Performance Vision-Language-Action Models. *arXiv preprint (cs.CV; cs.AI)*. arXiv:2509.09090
- Weifan Guan et al. (2025). Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey. *arXiv preprint (cs.RO; cs.LG)*. arXiv:2510.17111
- Yueen Ma et al. (2024). A Survey on Vision-Language-Action Models for Embodied AI. *IEEE Transactions on Neural Networks and Learning Systems (DOI 10.1109/TNNLS.2025.3650584); arXiv 2405.14093*. arXiv:2405.14093
- Chancharik Mitra et al. (2025). Mechanistic Finetuning of Vision-Language-Action Models via Few-Shot Demonstrations. *arXiv preprint (cs.RO; cs.CL; cs.CV)*. arXiv:2511.22697
- An-Chieh Cheng et al. (2024). NaVILA: Legged Robot Vision-Language-Action Model for Navigation. *arXiv preprint (cs.RO; cs.CV)*. arXiv:2412.04453

**VLAs on wheeled and mobile robots**
- Open X-Embodiment Collaboration (Abby O'Neill et al.) (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. *arXiv preprint (cs.RO)*. arXiv:2310.08864
- Guanhua Ji et al. (2025). OXE-AugE: A Large-Scale Robot Augmentation of OXE for Scaling Cross-Embodiment Policy Learning. *arXiv preprint (cs.RO; cs.AI)*. arXiv:2512.13100
- Jonathan Yang et al. (2024). Pushing the Limits of Cross-Embodiment Learning for Manipulation and Navigation. *Robotics: Science and Systems XX*. doi:10.15607/rss.2024.xx.093 †
- Haitong Wang et al. (2025). X-Nav: Learning End-to-End Cross-Embodiment Navigation for Mobile Robots. *IEEE Robotics and Automation Letters 11(1):698-705 (also arXiv 2507.14731v2)*. doi:10.1109/LRA.2025.3632119
- Pranav Putta et al. (2024). Embodiment Randomization for Cross Embodiment Navigation. *2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5527-5534*. doi:10.1109/iros58592.2024.10802792 †
- Hao-Tien Lewis Chiang et al. (2024). Mobility VLA: Multimodal Instruction Navigation with Long-Context VLMs and Topological Graphs. *arXiv preprint (cs.RO; cs.AI)*. arXiv:2407.07775
- Kiana Ehsani et al. (2023). SPOC: Imitating Shortest Paths in Simulation Enables Effective Navigation and Manipulation in the Real World. *arXiv preprint (cs.RO; cs.AI; cs.CV)*. arXiv:2312.02976
- Qi Wu et al. (2024). Helpful DoggyBot: Open-World Object Fetching using Legged Robots and Vision-Language Models. *arXiv preprint (cs.RO; cs.AI; cs.CV; cs.LG)*. arXiv:2410.00231
- Zhenyu Wu et al. (2025). MoManipVLA: Transferring Vision-language-action Models for General Mobile Manipulation. *arXiv preprint (cs.RO; cs.CV)*. arXiv:2503.13446
- Xiangyu Shi et al. (2025). SmartWay: Enhanced Waypoint Prediction and Backtracking for Zero-Shot Vision-and-Language Navigation. *arXiv preprint (cs.RO; cs.CV)*. arXiv:2503.10069
- Shiqi Pan et al. (2026). One MLLM, One Call: Efficient Zero-Shot Vision-and-Language Navigation via Spatial-Aware Waypoints. *arXiv preprint (cs.CV; cs.AI)*. arXiv:2609.06476
- Jaehwan Jeong et al. (2026). Your Vision-Language-Action Model Already Has Attention Heads For Path Deviation Detection. *arXiv preprint (cs.RO; cs.CV)*. arXiv:2603.13782
- Zipeng Fu et al. (2024). Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation. *arXiv preprint (cs.RO; cs.AI; cs.CV; cs.LG; eess.SY)*. arXiv:2401.02117
- Donglin Yang et al. (2026). Learning Panorama-Aware VLA for Mobile Manipulation with Whole-Body Teleoperation. *arXiv preprint (cs.RO)*. arXiv:2608.02257
- Jimmy Wu et al. (2024). TidyBot++: An Open-Source Holonomic Mobile Manipulator for Robot Learning. *arXiv preprint (cs.RO; cs.AI; cs.CV; cs.LG)*. arXiv:2412.10447
- Julian Zilly et al. (2019). The AI Driving Olympics at NeurIPS 2018. *arXiv preprint (cs.RO)*. arXiv:1903.02503
- Mikita Sazanovich et al. (2020). Imitation Learning Approach for AI Driving Olympics Trained on Real-world and Simulation Data Simultaneously. *arXiv preprint (cs.LG; cs.CV; cs.RO; stat.ML)*. arXiv:2007.03514
- Xuesu Xiao et al. (2023). Autonomous Ground Navigation in Highly Constrained Spaces: Lessons learned from The 2nd BARN Challenge at ICRA 2023. *arXiv preprint (cs.RO)*. arXiv:2308.03205
- Hanjaya Mandala, Guilherme Christmann (2023). The BARN Challenge 2023 -- Autonomous Navigation in Highly Constrained Spaces -- Inventec Team. *arXiv preprint (cs.RO)*. arXiv:2307.14580

**Expressive behaviour from foundation models**
- Karthik Mahadevan et al. (2024). Generative Expressive Robot Behaviors using Large Language Models (GenEM). *Proceedings of the 2024 ACM/IEEE International Conference on Human-Robot Interaction (HRI), pp. 482-491*. doi:10.1145/3610977.3634999 (arXiv 2401.14673v2)
- Peide Huang et al. (2024/2025). EMOTION: Expressive Motion Sequence Generation for Humanoid Robots with In-Context Learning. *arXiv cs.RO (2410.23234v1); IEEE Robotics and Automation Letters 10(8):7699-7706 (2025)*. arXiv 2410.23234v1 / 10.1109/lra.2025.3575983
- Yuhan Hu et al. (2025). ELEGNT: Expressive and Functional Movement Design for Non-anthropomorphic Robot. *arXiv cs.RO/cs.HC*. arXiv 2501.12493v1
- Mustafa Shukor et al. (2025). SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. *arXiv cs.LG/cs.RO*. arXiv 2506.01844v1
- Zining Wang et al. (2024). Ain't Misbehavin' - Using LLMs to Generate Expressive Robot Behavior in Conversations with the Tabletop Robot Haru. *Companion of the 2024 ACM/IEEE International Conference on Human-Robot Interaction*. doi:10.1145/3610978.3640562 (arXiv 2402.11571v1)
- Liam Roy et al. (2025). GPT-Driven Gestures: Leveraging Large Language Models to Generate Expressive Robot Motion for Enhanced Human-Robot Interaction. *IEEE Robotics and Automation Letters 10:4172-4179*. doi:10.1109/LRA.2025.3547631
- Lennart Wachowiak et al. (2024). Are Large Language Models Aligned with People's Social Intuitions for Human-Robot Interactions?. *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024, pp. 2520-2527*. doi:10.1109/IROS58592.2024.10801325 (arXiv 2403.05701)
- Eshtiak Ahmed et al. (2025). A Hands-on Workshop on Designing LLM-Powered Context-Aware Behavior for Companion Robots. *International Conference on Entertainment and Media in the Ubiquitous Era (Mindtrek) 2025*. doi:10.1145/3757980.3758014
- Wenli Shi et al. (2025). Controlling Intent Expressiveness in Robot Motion with Diffusion Models. *arXiv cs.RO*. arXiv 2510.12370v1
- Nikhil Churamani et al. (2022 (arXiv 2020)). Affect-Driven Modelling of Robot Personality for Collaborative Human-Robot Interactions. *Frontiers in Robotics and AI*. doi:10.3389/frobt.2022.717193 (arXiv 2010.07221v2)
- Marco Matarese et al. (2020). Towards Transparency of TD-RL Robotic Systems with a Human Teacher. *arXiv cs.RO/cs.HC*. arXiv 2005.05926v1
- Alice Nardelli et al. (2025). Designing Empathetic Companions: Exploring Personality, Emotion, and Trust in Social Robots. *arXiv cs.HC/cs.RO*. arXiv 2504.13964v1
- Richard Savery, Gil Weinberg (2020). A Survey of Robotics and Emotion: Classifications and Models of Emotional Interaction. *arXiv cs.RO*. arXiv 2007.14838v1
- Chinmaya Mishra, Gabriel Skantze (2022). Knowing Where to Look: A Planning-based Architecture to Automate the Gaze Behavior of Social Robots. *IEEE RO-MAN 2022*. doi:10.1109/RO-MAN53752.2022.9900740 (arXiv 2210.02866v1)
- Mei Yii Lim et al. (2022 (arXiv 2023)). We are all Individuals: The Role of Robot Personality and Human Traits in Trustworthy Interaction. *IEEE RO-MAN 2022*. doi:10.1109/RO-MAN53752.2022.9900772 (arXiv 2307.15568v1)
- Carlos Herrera Perez, Emilia I. Barakova (2020). Expressivity Comes First, Movement Follows: Embodied Interaction as Intrinsically Expressive Driver of Robot Behaviour. *Modelling Human Motion (Springer), pp. 299-313*. doi:10.1007/978-3-030-46732-6_14 †
- Jamy Li et al. (2020). Usability of a Robot's Realistic Facial Expressions and Peripherals in Autistic Children's Therapy. *arXiv cs.RO/cs.HC*. arXiv 2007.12236v1
- Ahmed Khota et al. (2024). Synthesis of Non-Linguistic Utterances for Sound Design Support Using a Genetic Algorithm. *Applied Sciences 14(11):4572*. doi:10.3390/app14114572
- Ahmed Khota et al. (2022). Modelling emotional valence and arousal of non-linguistic utterances for sound design support. *9th International Conference on Kansei Engineering and Emotion Research (KEER2022), pp. 507-516*. doi:10.5821/conference-9788419184849.52
- Zebin Yang et al. (2026). EagleVLA: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference. *arXiv cs.RO/cs.AI*. arXiv 2607.12659v6
- Xiaoyu Ma et al. (2025). BLURR: A Boosted Low-Resource Inference for Vision-Language-Action Models. *arXiv cs.RO*. arXiv 2512.11769v1
- Ao Zhou et al. (2026). EcoVLA: Energy-Efficient Device-Edge Co-Inference for Vision-Language-Action Models under Real-Time Constraints. *arXiv cs.AI/cs.RO*. arXiv 2608.15502v1

**Safety, monitoring and honesty around learned policies**
- Tao Lin et al. (2025). Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment. *arXiv 2511.04555*. doi:10.48550/arXiv.2511.04555
- Florian Seligmann et al. (2026). VLA-FAIL: Efficient Task Failure Detection for Finetuned Vision-Language-Action Models. *arXiv (cs.LG; cs.CV)*. arXiv:2606.21386
- Qiao Gu et al. (2025). SAFE: Multitask Failure Detection for Vision-Language-Action Models. *arXiv (cs.RO; cs.AI)*. arXiv:2506.09937
- Christopher Agia et al. (2024). Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress. *arXiv (cs.RO; cs.AI; cs.LG)*. arXiv:2410.04640
- Mingyue Cui et al. (2026). Learning from Runtime Feedback through Failure-Bank Self-Evolution for Vision-Language-Action Models. *arXiv (cs.RO; cs.AI)*. arXiv:2609.39820
- Bo-Rong Zhang et al. (2025). SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning. *NeurIPS 2025 (arXiv 2503.03480)*. doi:10.52202/085713-5128
- Manan Tayal, A. Nambi (2026). ShieldVLA: Feasibility-Aware Safety Alignment for Vision-Language-Action Models. *arXiv 2609.13231*. arXiv:2609.13231
- Lizhi Yang et al. (2025). CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions. *arXiv (cs.RO; cs.AI; cs.LG; eess.SY)*. arXiv:2510.14959
- Osbert Bastani, Shuo Li (2021). Safe Reinforcement Learning via Statistical Model Predictive Shielding. *Robotics: Science and Systems XVII*. doi:10.15607/rss.2021.xvii.026 †
- Huang Huang et al. (2023). Conformal Policy Learning for Sensorimotor Control Under Distribution Shifts. *arXiv (cs.RO; cs.AI)*. arXiv:2311.01457
- Xueyang Zhou et al. (2025). LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization. *arXiv (cs.CV; cs.RO)*. arXiv:2510.03827
- Chanyoung Kim et al. (2026). LIBERO-Para: A Diagnostic Benchmark and Metrics for Paraphrase Robustness in VLA Models. *arXiv (cs.LG)*. arXiv:2603.28301
- Huiqiong Li et al. (2026). LIBERO-VPro: Benchmarking Closed-Loop Visual Robustness of Robotic Foundation Models. *arXiv (cs.RO; cs.CV)*. arXiv:2609.24350
- Xuanlin Li et al. (2024). Evaluating Real-World Robot Manipulation Policies in Simulation. *arXiv (cs.RO; cs.CV; cs.LG)*. arXiv:2405.05941
- Bear Häon et al. (2025). Mechanistic interpretability for steering vision-language-action models. *arXiv (cs.RO; cs.LG)*. arXiv:2509.00328
- IEEE Standards Association (2022). IEEE Standard for Transparency of Autonomous Systems (IEEE 7001-2021). *IEEE Std (standard)*. doi:10.1109/ieeestd.2022.9726144 †
- IEEE Standards Association (2024). IEEE Standard for Ethical Considerations in Emulated Empathy in Autonomous and Intelligent Systems (IEEE 7014-2024). *IEEE Std (standard)*. doi:10.1109/ieeestd.2024.10576666 †
- Raj Korpan et al. (2017). WHY: Natural Explanations from a Robot Navigator. *arXiv (cs.AI; cs.CL; cs.HC; cs.RO)*. arXiv:1709.09741
- Yuanjie Lu et al. (2026). APPLV: Adaptive Planner Parameter Learning from Vision-Language-Action Model. *arXiv (cs.RO; cs.LG)*. arXiv:2603.08862
- Mateo Guaman Castro et al. (2025). VAMOS: A Hierarchical Vision-Language-Action Model for Capability-Modulated and Steerable Navigation. *IEEE Robotics and Automation Letters, vol. 11 (arXiv 2510.20818)*. doi:10.1109/LRA.2026.3734870

**Foundation models in the learning loop**
- Andrés Marafioti et al. (2025). SmolVLM: Redefining small and efficient multimodal models. *arXiv (cs.AI; cs.CV)*. arXiv:2504.05299
- Yuhui Chen et al. (2025). ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy. *arXiv (cs.RO; cs.AI)*. arXiv:2502.05450
- Yanjiang Guo et al. (2025). Improving Vision-Language-Action Model with Online Reinforcement Learning. *arXiv (cs.RO; cs.CV; cs.LG)*. arXiv:2501.16664
- Guanxing Lu et al. (2025). VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning. *arXiv (cs.RO; cs.AI)*. arXiv:2505.18719
- Charles Xu et al. (2024). RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning. *arXiv (cs.RO; cs.AI; cs.LG)*. arXiv:2412.09858
- Angelo Moroncelli et al. (2026). Vision-Language-Action Jump-Starting for Reinforcement Learning Robotic Agents. *arXiv (cs.LG; cs.AI; cs.RO)*. arXiv:2604.13733
- Yi Wang et al. (2026). Learning While Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies. *arXiv (cs.RO)*. arXiv:2605.00416
- Andrew Choi et al. (2026). Scaling Sim-to-Real VLA Reinforcement Learning with Generative 3D Worlds. *arXiv (cs.RO; cs.AI; cs.LG)*. arXiv:2603.18532
- Yuqing Du et al. (2023). Guiding Pretraining in Reinforcement Learning with Large Language Models. *arXiv (cs.LG; cs.AI; cs.CL)*. arXiv:2302.06692
- Jenny Zhang et al. (2023). OMNI: Open-endedness via Models of human Notions of Interestingness. *arXiv (cs.AI; cs.LG)*. arXiv:2306.01711
- Juan Rocamonde et al. (2023). Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning. *arXiv (cs.LG; cs.AI)*. arXiv:2310.12921
- Yufei Wang et al. (2024). RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback. *arXiv (cs.RO; cs.AI; cs.LG)*. arXiv:2402.03681
- Yuqing Du et al. (2023). Vision-Language Models as Success Detectors. *arXiv (cs.CV; cs.AI; cs.LG)*. arXiv:2303.07280
- Pierre Sermanet et al. (2023). RoboVQA: Multimodal Long-Horizon Reasoning for Robotics. *arXiv (cs.RO)*. arXiv:2311.00899
- Zhiyuan Zhou et al. (2024). Autonomous Improvement of Instruction Following Skills via Foundation Models. *arXiv (cs.RO; cs.AI)*. arXiv:2407.20635
- Fangchen Liu et al. (2024). MOKA: Open-World Robotic Manipulation through Mark-Based Visual Prompting. *arXiv (cs.RO; cs.AI)*. arXiv:2403.03174
- Wentao Yuan et al. (2024). RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics. *arXiv (cs.RO; cs.AI; cs.CV)*. arXiv:2406.10721
- Edmond Tong et al. (2024). OVAL-Prompt: Open-Vocabulary Affordance Localization for Robot Manipulation through LLM Affordance-Grounding. *arXiv (cs.RO)*. arXiv:2404.11000
- Yecheng Jason Ma et al. (2023). Eureka: Human-Level Reward Design via Coding Large Language Models. *arXiv (cs.RO; cs.AI; cs.LG)*. arXiv:2310.12931
- Haoyu Xiong et al. (2024). Adaptive Mobile Manipulation for Articulated Objects In the Open World. *arXiv (cs.RO; cs.AI; cs.CV; cs.LG; eess.SY)*. arXiv:2401.14403
- Yogesh Girdhar, Gregory Dudek (2015). Modeling Curiosity in a Mobile Robot for Long-Term Autonomous Exploration and Monitoring. *arXiv (cs.RO; cs.CV; cs.LG); DOI 10.1007/s10514-015-9500-x*. arXiv:1509.07975

**Found by the critique pass**
- Zhao, T. Z. et al. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT). *arXiv*. arXiv:2304.13705
- Chi, C. et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *arXiv*. arXiv:2303.04137
- Ren, A. Z. et al. (2024). Diffusion Policy Policy Optimization (DPPO). *arXiv*. arXiv:2409.00588
- Liang, J. et al. (2022). Code as Policies: Language Model Programs for Embodied Control. *arXiv*. arXiv:2209.07753
- Luo, J. et al. (2024). Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning (HIL-SERL). *arXiv*. arXiv:2410.21845
- Shafiullah, N. M. M. et al. (2023). On Bringing Robots Home (Dobb-E). *arXiv*. arXiv:2311.16098
- OpenAI et al. (2018). Learning Dexterous In-Hand Manipulation. *arXiv*. arXiv:1808.00177

