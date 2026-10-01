# Can one VLM or VLA run all of Pip? (survey, October 2026)

Desk research from model cards, papers and repositories. Nothing here was run
or benchmarked in this project. Unverified items are listed at the end.

## Short answer

**No.** No released model, open or closed, does metric localisation and mapping,
certified-safe navigation, object identity, interaction, persistent learning
and expression together. Models can take over individual layers:

- **Perception and semantics:** open VLMs can do this today, on a CPU or Mac.
- **Goal or waypoint proposals:** navigation VLAs can do this, but need a CUDA GPU.
- **Kept classical:** the safety supervisor, metric SLAM and the SQLite memory.

## Candidates

| Model (release) | Trained embodiment | Inputs → outputs | Size | Weights / licence | Runs on | Fit for Pip |
|---|---|---|---|---|---|---|
| **LightNav-0** (Light Origins, arXiv Aug 2026) | Generalist; demoed on wheeled SE(2) robots | Monocular RGB video + language → pointing tokens + 10-step SE(2) waypoints | Qwen3-VL-4B backbone (HF card says 5B) | Open, Apache-2.0 per HF card | RTX 4090 in the paper | Best open navigation fit: RGB-only, waypoint output. Needs an offboard GPU |
| **Robostral Navigate** (Mistral, Jul 2026) | Wheeled, legged, aerial | Monocular RGB history + language → image-point waypoints or metric displacement | 8B VLM + 121M diffusion policy | Closed (enterprise access) | Not disclosed | Strong benchmark results, but not available |
| **InternVLA-N1 / DualVLN** (Shanghai AI Lab, Sep 2025; InternNav v0.3 Jan 2026) | Wheeled, quadruped, humanoid | RGB or RGB-D → pixel goal (2 Hz) → trajectories (30 Hz) | 7B + small policy | Code MIT; data CC BY-NC-SA | RTX 4090, ~20 GB | Good architecture match (VLM proposes, small policy executes); GPU only |
| **OmniVLA** (Berkeley/Toyota, ICRA 2026) | 10 ground platforms incl. small low robots | RGB + goal image / 2D pose / language → velocity or action chunks | 7B (+ "edge" variant) | Open, MIT | ≥ 20 GB GPU | Closest training data to a small low robot; heavy |
| **NoMaD / ViNT / GNM** (Berkeley, 2022–23) | Many ground robots | RGB context frames (+ optional goal image) → waypoints; topological graph | Tens of millions of parameters | Open | Jetson-class; CPU plausible | Cheapest navigation proposer; has an exploration mode; image goals, no language |
| NavFoM (PKU, Sep 2025) | Quadruped, drone, wheeled, car | 1–8 RGB cameras → trajectories | not stated | weights not confirmed | RTX 5090 | Reference only |
| ABot-N0 / N1 (Alibaba, 2026) | Navigation | Vision + language → trajectories; topological memory | not verified | not verified | unknown | Watch |
| NaVILA (RSS 2025) | Legged | RGB frames → language mid-level actions | 8B | Apache-2.0 code | multi-GPU | Text-action interface fits a supervisor; too heavy |
| π0 / π0.5 (openpi), GR00T N1.7, SmolVLA, OpenVLA-OFT, Magma | Arms / humanoids | RGB + state → arm or joint actions | 0.5–9B | mostly open | GPU (SmolVLA CPU-feasible) | Wrong embodiment |
| Gemini Robotics-ER 2 (preview, 2026) | Embodied reasoning | Images/video → points, boxes, trajectories, plans | closed | API only | cloud | Strong semantic layer; network-dependent |
| **Qwen3-VL** 2B/4B/8B (Oct 2025) | General VLM | Image/video → text, 2D boxes, points | 2–8B | Apache-2.0 | Mac (llama.cpp/MLX), Jetson Orin | Best open semantic backbone |
| RoboBrain 2.5 (BAAI, Jan 2026) | Embodied VLM | RGB → points, affordances, trajectories, task progress | 4B / 8B | Apache-2.0 | GPU | Good for outcome / progress judging |
| Cosmos-Reason2 (NVIDIA) | Physical-AI VLM | Video/images → reasoning, points, boxes | 2B / 8B | NVIDIA Open Model License | 2B on Orin Super Nano | Edge option on Jetson |
| **Moondream 2** (2025-04-14) | Small general VLM | Image → caption, query, detect, point | 2B | Apache-2.0 | CPU, Mac | Cheapest drop-in detector |

## Layer by layer

| Layer | Can a model do it? |
|---|---|
| Localisation and mapping | No. Navigation VLAs are reactive and produce no auditable pose or metric map. Keep ORB SLAM and the occupancy grid. |
| Safe navigation | No certification. A VLA can propose waypoints; it might add obstacle evidence for objects placed on the route, but cannot replace the certified check. |
| Object detection and identity | Detection yes (open vocabulary beats the colour detector). Identity across time still needs the entity store. |
| Interaction outcome | Good fit: a VLM can judge before/after image pairs. |
| Persistent learning from experience | No released model does this across restarts. Keep the SQLite memory. |
| Expression | Not needed. |

**Rule conflict.** Every navigation VLA outputs coordinates, which breaks the
rule that semantics never produce coordinates or motor commands. Fitting one in
means changing the rule deliberately: a logged, untrusted *proposal* channel,
projected into the map frame with the SLAM pose, accepted only if the
footprint-certified planner finds a path through known-free cells, and
reported with denominators (proposed, accepted, reached, contacts). Purely
semantic uses (labels, outcome judgements) need no rule change.

**Domain gap (unverified).** Pip's camera is 15.5 cm high, 320×240, rendered by
Genesis. Most navigation data comes from higher cameras; expect fine-tuning on
Pip's own episodes.

## Recommended path

1. **Qwen3-VL-2B/4B or Moondream 2** behind the existing detector contract
   (pixels in, labelled regions out, display-only). Replaces the colour
   detector and possibly the outcome classifier. About 1 Hz on Apple Silicon
   is enough; the 2-core container is for offline evaluation only.
2. **NoMaD or ViNT in shadow mode** as a waypoint/exploration proposer behind
   the certified planner. Executes nothing at first.
3. **Optional, GPU: LightNav-0** for language goals ("go to the blue box"),
   same proposal contract.

## First experiment

1. Log a few hundred Genesis frames from the fixture scenes (ground truth used
   only for offline scoring).
2. Run the colour detector, Moondream 2 and Qwen3-VL-2B behind the same
   detector interface.
3. Report per-fixture precision and recall, recall on non-uniformly coloured
   objects, false positives on walls and floor, and latency on the Mac and the
   2-core CPU.
4. Then compare VLM outcome labels on stored before/after pairs with the
   current classifier.
5. Then run NoMaD in shadow mode during navigation tests: how often its
   proposals would pass the certified planner, and whether it reacts to newly
   placed objects before contact.

## Not verified

Robostral Navigate's exact release date; LightNav-0's size (4B vs 5B);
OmniVLA-edge's size; ABot and NavFoM weight availability; NoMaD/ViNT/GNM
parameter counts and licence; any CPU or Apple Silicon latency figures (none
found in primary sources — measure them).

## Sources

- https://arxiv.org/html/2608.30935v1 · https://huggingface.co/LightOriginsHQ/LightNav-0
- https://arxiv.org/html/2607.20785 · https://www.marktechpost.com/2026/07/14/mistral-ai-releases-robostral-navigate-an-8b-model-enabling-robots-to-navigate-complex-environments-using-a-single-rgb-camera/
- https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf · https://github.com/InternRobotics/InternNav
- https://omnivla-nav.github.io/ · https://github.com/NHirose/OmniVLA
- https://github.com/robodhruv/visualnav-transformer · https://arxiv.org/abs/2310.07896
- https://pku-epic.github.io/NavFoM-Web/ · https://arxiv.org/abs/2509.12129
- https://arxiv.org/abs/2602.11598 · https://github.com/amap-cvlab/ABot-Navigation
- https://github.com/AnjieCheng/NaVILA · https://github.com/jzhzhang/Uni-NaVid · https://arxiv.org/abs/2407.07775
- https://github.com/Physical-Intelligence/openpi · https://huggingface.co/blog/nvidia/gr00t-n1-7 · https://huggingface.co/lerobot/smolvla_base
- https://github.com/moojink/openvla-oft · https://arxiv.org/abs/2605.02881 · https://huggingface.co/microsoft/Magma-8B
- https://ai.google.dev/gemini-api/docs/robotics-overview · https://deepmind.google/models/model-cards/gemini-robotics-on-device-2/
- https://github.com/qwenlm/qwen3-vl · https://github.com/FlagOpen/RoboBrain2.5
- https://huggingface.co/nvidia/Cosmos-Reason2-2B · https://www.jetson-ai-lab.com/tutorials/cosmos-reason2-vlm/
- https://huggingface.co/moondream/moondream-2b-2025-04-14
- https://arxiv.org/html/2606.10927v1
