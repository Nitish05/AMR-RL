"""Offline pilot: can a small vision-language model tell what an interaction did?
(evaluation only; perception-swap stage, step 5.0)

Input: retained interaction receipts (onboard before/after frames, saved by the
runtime) from earlier Genesis runs. Ground truth (scoring only): the world's own
fixture events in the same phase (``summary.json`` ``world_events``): a yellow or
red panel raised, a signal into an already raised panel, or no response. Roller
"moved" outcomes have no world event and are left out (movement stays geometry-led).

Two question designs (``--mode``):

* ``change``: the model sees the last "before" frame and an early "after" frame and
  says what changed. First run (seed 0, n = 200): accuracy 0.59; flags that were
  already up were read as "appeared" and magenta/cyan bodies as red flags.
* ``state``: the model is asked about each image separately which flag (if any)
  stands on the object; the outcome is the after-state when it differs from the
  before-state, else "none" (the difference is computed here, not by the model).
  Decision: the flag colour whose probability rises most from before to after,
  if it rises by at least ``--delta`` (0.10, chosen on the seed-0 design sample:
  accuracy 0.955, none -> flag 1/100; the arg-max rule missed 18/50 yellow flags
  cut off by the frame edge). Scored on a disjoint sample (``--exclude``).

Answers are read from the next-token probabilities of the option letters
(deterministic, gives probabilities, cannot produce an off-schema answer) and mapped
to the memory's outcome tokens.

    PYTHONPATH=src python scripts/dev/vlm_outcome_pilot.py out.json --model qwen3vl --per-class 50
"""

import argparse
import glob
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

MODELS = {
    "qwen3vl": ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", "apache-2.0"),
    "smolvlm2": ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467", "apache-2.0"),
}
PROMPT = ("The robot faced an object (image 1, before) and then acted on it (image 2, after). "
          "Small coloured flags can pop up from the object. Compare the two images. What changed on the object?\n"
          "A: nothing changed\nB: a yellow flag appeared\nC: a red flag appeared\nD: a flag of another colour appeared\n"
          "E: the object moved away or disappeared\nF: unsure\nAnswer with one letter.")
LETTERS = "ABCDEF"
MAP = {"A": "none", "B": "attach:yellow", "C": "attach:red", "D": "attach:other", "E": "moved", "F": None}
STATE_PROMPT = ("Look at the large coloured object in the middle of the image. Is a small flat flag or panel "
                "standing up on top of it?\nA: no flag on top\nB: a yellow flag\nC: a red flag\n"
                "D: a flag of another colour\nAnswer with one letter.")
STATE_LETTERS = "ABCD"
STATE_MAP = {"A": None, "B": "yellow", "C": "red", "D": "other"}


def state_outcome(pb, pa, delta):
    """Outcome token from per-image flag-state probabilities (before, after)."""
    rise = {c: pa[c] - pb[c] for c in "BCD"}
    c = max(rise, key=rise.get)
    return "none" if rise[c] < delta else f"attach:{STATE_MAP[c]}"


def truth_for(receipt, events):
    t0 = receipt["before"][0]["t"] - 0.5 if receipt["before"] else receipt["timestamp"] - 10
    hits = [e for e in events if e["trigger"] == receipt["action"] and t0 <= e["time"] <= receipt["timestamp"] + 0.5]
    responses = {e["response"] for e in hits}
    if "yellow" in responses:
        return "attach:yellow"
    if "red" in responses:
        return "attach:red"
    if "suppressed_raised" in responses:
        return "none (panel already up)"
    return "none"


def collect(roots, per_class, seed, exclude=()):
    by = defaultdict(list)
    for f in sorted(glob.glob(f"{roots}/**/image-evidence/receipt-*.json", recursive=True)):
        if f in exclude:
            continue
        data = json.loads(Path(f).read_text())
        r = data["receipt"]
        if r.get("observed") == "moved" or not r.get("before") or not r.get("after"):
            continue
        phase = Path(f).parent.parent
        summary = phase / "summary.json"
        if not summary.exists():
            continue
        events = json.loads(summary.read_text()).get("world_events", [])
        images = data["images"]
        before = images.get(r["before"][-1]["frame_sha256"])
        after = images.get(r["after"][min(len(r["after"]) - 1, len(r["after"]) // 3)]["frame_sha256"])
        if before is None or after is None:
            continue
        if not (phase / before["png_path"]).exists() or not (phase / after["png_path"]).exists():
            continue  # older runs did not retain every frame
        by[truth_for(r, events)].append({"receipt": f, "before": str(phase / before["png_path"]),
                                         "after": str(phase / after["png_path"]), "observed_by_robot": r["observed"],
                                         "context": r["context"], "action": r["action"]})
    rng = np.random.default_rng(seed)
    sample = []
    for truth, items in sorted(by.items()):
        for i in rng.permutation(len(items))[:per_class]:
            sample.append({**items[i], "truth": truth})
    return sample, {k: len(v) for k, v in by.items()}


class Scorer:
    def __init__(self, name, device):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        repo, rev, _ = MODELS[name]
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(repo, revision=rev, local_files_only=True)
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(repo, revision=rev, local_files_only=True,
                                                                  dtype=dtype).to(device).eval()
        self.device = device
        tok = self.processor.tokenizer
        self.letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]

    def ask(self, paths, prompt, letters):
        torch = self.torch
        imgs = [Image.open(p).convert("RGB").resize((640, 480)) for p in paths]
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        inputs = self.processor(text=[text], images=imgs, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        ids = [self.letter_ids[LETTERS.index(c)] for c in letters]
        p = torch.softmax(logits[ids], dim=0).cpu().numpy()
        return {c: float(v) for c, v in zip(letters, p)}

    def change(self, before, after):
        probs = self.ask([before, after], PROMPT, LETTERS)
        return probs, MAP[max(probs, key=probs.get)]

    def state(self, before, after, delta=0.10):
        pb = self.ask([before], STATE_PROMPT, STATE_LETTERS)
        pa = self.ask([after], STATE_PROMPT, STATE_LETTERS)
        return {"before": pb, "after": pa}, state_outcome(pb, pa, delta)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--roots", default="work/evidence")
    ap.add_argument("--model", default="qwen3vl", choices=list(MODELS))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="change", choices=["change", "state"])
    ap.add_argument("--exclude", nargs="*", default=[], help="earlier pilot JSONs: leave their receipts out")
    ap.add_argument("--delta", type=float, default=0.10, help="state mode: minimum rise in a flag's probability")
    args = ap.parse_args()
    exclude = {r["receipt"] for f in args.exclude for r in json.loads(Path(f).read_text())["rows"]}
    sample, available = collect(args.roots, args.per_class, args.seed, exclude)
    print("available per truth class:", available, "| sampled:", len(sample), flush=True)
    scorer = Scorer(args.model, args.device)
    rows, conf, times = [], defaultdict(Counter), []
    for k, item in enumerate(sample):
        t0 = time.time()
        if args.mode == "state":
            probs, pred = scorer.state(item["before"], item["after"], args.delta)
        else:
            probs, pred = scorer.change(item["before"], item["after"])
        times.append(time.time() - t0)
        conf[item["truth"]][str(pred)] += 1
        rows.append({**item, "probs": probs, "pred": pred})
        if k % 20 == 0:
            print(f"  {k}/{len(sample)} {times[-1]:.2f} s", flush=True)
    correct = {"attach:yellow": "attach:yellow", "attach:red": "attach:red", "none": "none",
               "none (panel already up)": "none"}
    ok = sum(r["pred"] == correct[r["truth"]] for r in rows)
    report = {"model": MODELS[args.model], "mode": args.mode, "delta": args.delta if args.mode == "state" else None,
              "seed": args.seed, "excluded": len(exclude),
              "device": args.device, "n": len(rows), "available": available,
              "accuracy": ok / max(1, len(rows)), "accuracy_ci95": wilson(ok, len(rows)),
              "confusion": {t: dict(c) for t, c in conf.items()},
              "robot_colour_detector_accuracy": sum(
                  r["observed_by_robot"] == correct[r["truth"]] for r in rows) / max(1, len(rows)),
              "seconds_per_interaction": {"median": float(np.median(times)), "p95": float(np.percentile(times, 95))},
              "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
