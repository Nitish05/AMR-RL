# Perception swap in simulation: textured worlds, an open-vocabulary detector, measured errors (round 6, steps 5a, 5b, 5d)

**Status:**
- Textured worlds: added (`configs/worlds/arena_textured.yaml`; held out:
  `home_a_textured.yaml`).
- Open-vocabulary detector: implemented and benchmarked; **off by default**
  (`RuntimeConfig.detector = "fixture"`).
- Measured error presets: added to the learner testbed.

Step 5c, a VLM outcome describer in the runtime, was not started: its pilot missed
one criterion ([vlm-outcome-pilot.md](vlm-outcome-pilot.md)).

**Code:**
- `sim/textures.py` (`object_texture`, `flag_texture`, `poster_texture`);
- `sim/assets.py` (`textured_cylinder`, `textured_sphere`, `fixture_urdf(texture=, flag_seed=)`);
- `sim/world.py` (posters, saturated distractors, `raise_flag_for_evaluation`);
- `perception/open_vocab.py`, `perception/reid.py`;
- `scripts/dev/detector_capture.py`, `scripts/eval/detector_bench.py`, `scripts/eval/reid_bench.py`;
- `sim/learner_bench.py` (`NoiseConfig.from_measurement`).

**Evidence:**
- `work/evidence/detcap-{arena,arena_textured,home_a,home_a_textured}-20261003/`
  (300 frames each);
- `work/evidence/tex-*`, `work/evidence/floor07-*`;
- `work/evidence/learner-bench-measured-20261003/`.

## 5a: worlds the colour detector cannot handle

The textured worlds keep the arena (or home_a) layout, fixtures and response rules,
and change their appearance:
- fixtures with multi-hue saturated patterns (stripes, checks, blobs, a label band)
  on a base hue, including red and yellow patches;
- patterned flags (still mostly yellow or red, so the outcome tokens keep their
  meaning);
- four colourful wall posters;
- one saturated distractor box.

**Ground truth (scoring only).** Genesis' link-level segmentation of each fixture's
body and flags, rendered from the onboard camera at sampled poses 0.4–2.0 m from a
fixture. Every fixture's flag is set at random.

**Colour detector, objects within 2 m:**

| world | recall | precision | false positives / frame | flag-state accuracy |
|---|---|---|---|---|
| arena (plain) | 1.0 | 0.96 | 0.05 | 0.996 |
| home_a (plain) | 1.0 | 0.97 | 0.05 | 0.96 |
| arena_textured | **0.46** | **0.09** | **7.3** | 0.65 |
| home_a_textured (held out) | **0.20** | **0.04** | **5.8** | 0.61 |

The criterion was met: recall ≤ 0.5 or more than 1 false positive per frame.

**VSLAM in the textured world: worse.** Paired 420 s map builds:

| seed | arena: ATE / frames lost | arena_textured: ATE / frames lost |
|---|---|---|
| 0 | 7.8 cm / 0 | 7.2 cm / 580 |
| 2 | 4.9 cm / 237 | 8.6 cm / 501 |
| 3 | 10.1 cm / 0 | 7.1 cm / **3,429** (82 % of the run; 134 keyframes) |
| 4 | 3.4 cm / 664 | 8.9 cm / 1,025 |

More frames were lost in 4/4 pairs. The losses start the same way in both worlds,
from a rare "motion inconsistent with commands". The difference is relocalisation
afterwards: in seed 3, 1,996 attempts failed.

**Cause found.** The fixed ORB budget (900 per frame) goes to the high-contrast
posters and patterns. Features in the lower image (mostly floor) fell from a median
of 502 to 354 per frame, and floor matches are what planar relocalisation needs.
The remedy is an optional floor share of the budget (`VSLAMConfig.floor_feature_share`);
its build results are in the section below.

**This criterion is not met:** VSLAM does regress in the textured world. Runtime
experiments in it are confounded by localisation until that is fixed.

## 5b: open-vocabulary detector behind `Detection`

- **Proposals (learned).** OmDet-Turbo (Swin-T, Apache-2.0, pinned) is asked for
  "box", "cylinder", "ball" and "flag" in one pass.
- **Rules around it (engineered):**
  - an object must stand on the floor: its box's bottom edge must project onto the
    floor within 3 m;
  - a "flag" box resting on an object's top is that object's attachment;
  - the flag's colour is the majority of saturated yellow vs red pixels in the box;
  - the floor-contact offset comes from the detector's word.

**Model choice.** LLMDet-tiny was tried first. On these 320×240 frames only 2–11 % of
objects within 2 m got a box with IoU ≥ 0.5, whatever the prompt or upscale. It was
rejected. OmDet-Turbo: 100 %, median IoU 0.94.

**Calibration on arena_textured (design), then frozen.** Score threshold 0.25,
flag threshold 0.15 (lower thresholds read red patches on objects as flags), and the
3 m range limit (rejects boxes around whole walls).

The scorer counts a detection on a real object that has no ground-truth box in
that frame as "other object", not false. Examples: the crates, or the roller when
it shows under 150 pixels. All detectors are scored the same way.

| capture (objects within 2 m) | recall | precision | false positives / frame | flag state | position error, median / 90th percentile | s / frame (MPS) |
|---|---|---|---|---|---|---|
| arena_textured (design) | 0.996 | 0.98 | 0.03 | 0.962 | 2.0 / 5.4 cm | 0.08 |
| **home_a_textured (held out)** | **1.0** | **0.97** | **0.04** | **0.982** | **1.8 / 5.2 cm** | 0.08 |
| arena (plain) | 0.996 | 0.99 | 0.01 | 0.982 | 1.9 / 5.6 cm | 0.08 |
| home_a (plain) | 0.996 | 0.99 | 0.01 | 0.967 | 1.5 / 5.4 cm | 0.08 |

Criteria at ≤ 2 m: recall and precision ≥ 0.9, position ≤ 5 / 12 cm, state ≥ 0.95,
≤ 0.3 s/frame. **All met on the held-out world.**

- **Remaining error.** Mostly flags read as "none", e.g. a flag cut off by the top of
  the frame.
- **Speed.** On CPU it takes 0.9 s/frame, too slow to run every frame. The runtime
  uses MPS when available (`detector_device="auto"`).

**Re-identification.** DINOv2-S CLS embeddings of crops, scored by ROC AUC over
same-fixture vs different-fixture pairs:

| capture | DINOv2-S | hue of the crop (current cue) |
|---|---|---|
| arena_textured | 0.984 | 0.998 |
| home_a_textured (held out) | 0.984 | 0.999 |
| arena (plain) | 0.877 | 1.000 |

DINOv2 meets the 0.98 target on textured objects, but the engineered hue cue is
better in every world. This is because each textured object here keeps a dominant
base hue, which is a limit of these worlds and not a property of real objects.
Identity therefore keeps hue and size, computed from the detector's box. The
embedder stays available for worlds where colour does not identify objects.

**Not yet done for 5b.** Identity purity in a runtime run. That needs the textured
world's VSLAM problem fixed first (above).

## 5d: measured errors in the learner testbed

`NoiseConfig.from_measurement()` turns a `detector_bench.py` report into a noise preset.

- **Measured:** missed detections, position σ and outliers, flags read as none, none read as a
  flag, and colour confusion.
- **Not measured:** appearance jitter and corner-on views, which are set to the
  "low" preset.
- **Limitation:** the testbed draws frames independently, while real frames are
  correlated, so the per-frame rates are optimistic. The colour detector's ~7
  spurious objects per frame in textured worlds are not modelled either, so its
  preset understates the harm.

20 seeds per cell (`learner-bench-measured-20261003`):

| noise | true valence / attempt (learned) | reversal: found new option | restart: first choice best | inert: outcomes 1st → 2nd half |
|---|---|---|---|---|
| clean | 0.58 ± 0.08 | 19/20 | 19/20 | 18.2 → 5.7 |
| OmDet, measured (held out) | 0.60 ± 0.04 | 20/20 | 20/20 | 18.7 → 5.5 |
| colour detector in the textured world, measured | 0.13 ± 0.04 | 14/20 | 6/20 | 14.7 → 3.6 |

With the open-vocabulary detector's measured errors, the learner does as well as
with clean perception. With the colour detector in textured worlds, it does not.

The Genesis learning suite on the textured worlds (the rest of 5d) waits for the
VSLAM fix.
