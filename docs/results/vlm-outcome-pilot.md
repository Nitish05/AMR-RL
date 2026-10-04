# Can a small vision-language model tell what an interaction did? (perception swap, step 5.0)

**Status:** offline pilot, evaluation only. Nothing in the runtime changed: outcomes
still come from the engineered colour detector (`perception/entities.py`).

**Script:** `scripts/dev/vlm_outcome_pilot.py`. **Evidence:** `work/evidence/vlm-pilot-20261003/`.

## Setup

- **Data.** Retained interaction receipts from earlier Genesis learning runs: the
  robot's own before/after frames, saved by the runtime.
- **Ground truth (scoring only).** The world's own fixture events in the same phase
  (`summary.json` `world_events`), giving four classes:
  - a yellow flag raised;
  - a red flag raised;
  - no response;
  - a signal into a flag that was already up ("none, panel already up").
- **Left out.** "Moved" outcomes, which have no world event; movement stays geometry-led.
- **Available.** red 135, yellow 870, none 1,145, already up 592.
- **Samples.** Balanced, 50 per class. The design sample is seed 0. The held-out sample is seed 1, with every design receipt excluded.
- **Scoring.** The answer is read from the next-token probabilities of the option
  letters. This is deterministic, gives probabilities, and cannot produce an
  off-schema answer.
- **Hardware.** MPS on an M-series Mac, fp16.

## Design sample (seed 0, n = 200)

| question design | accuracy (95 % Wilson) | main errors |
|---|---|---|
| "what changed between these two images?" (A–F) | 0.59 | already-up flags read as "appeared" (50/50); magenta ball and cyan cylinder read as "red flag" (20) or "other colour" (12) |
| per image: "which flag stands on the object?"; outcome = after-state if different, arg-max | 0.905 (0.86–0.94) | 18/50 yellow flags missed: cut off by the top of the frame or seen edge-on |
| per image, decision on the **rise** in a flag's probability, ≥ δ | 0.955 at δ = 0.10 | δ chosen here (none → flag 1/100) and then frozen |

The per-image design puts the comparison in our code and not in the model. Only the
"state" question goes to the model.

## Held out (seed 1, n = 200, disjoint receipts, δ = 0.10 frozen)

| model | accuracy (95 % Wilson) | red → none | yellow → none | none → flag | wrong colour | s / interaction (median) |
|---|---|---|---|---|---|---|
| **Qwen3-VL-2B** | **0.935 (0.89–0.96)** | 2/50 | 10/50 | 1/100 | 0 | 0.36 |
| SmolVLM2-500M | 0.745 (0.68–0.80) | **50/50** | 1/50 | 0/100 | 0 | 0.56 |

δ was calibrated on Qwen3-VL. SmolVLM2 never raised its red-flag probability on
red-flag frames.

**Against the plan's in-runtime criteria (step 5c):**

| criterion | requirement | Qwen3-VL-2B |
|---|---|---|
| accuracy | ≥ 0.9 | **met** |
| none → yellow | ≤ 2 % | **met**: 1 % |
| red → not red | ≤ 2 % | **not met**: 4 %, 2 of 50 |
| latency | ≤ 5 s | **met**: 0.36 s |

## What this does and does not show

- **Comparison with the engineered detector.** On these flat-coloured simulated
  objects, the engineered colour detector is right on 200/200. The VLM is not
  better here, and it is not expected to be. The point of the swap is objects that
  a colour rule cannot handle, which is step 5a (textured assets).
- **Remaining error mode.** It is mostly framing: the flag is partly outside the
  single "after" frame. Next steps:
  - use the maximum over several after-frames;
  - ask about a crop that includes the space above the object.

  Either needs a fresh held-out sample. Red receipts are the scarce class: 35 remain
  unused.
- **Scope.** Two models were tested. The challenger (Gemma 4 E2B or InternVL3.5-2B)
  was not run.
