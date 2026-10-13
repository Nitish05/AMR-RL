"""Human-ratings tooling: the protocol keeps RoSAS as separate scales, the scoring applies the
pre-registered exclusions and rules (no simulator, no raters)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval" / "ratings.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], check=True, capture_output=True, text=True,
                          env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}).stdout


def test_protocol_exports_separate_rosas_scales_manipulation_items_and_the_disclosure(tmp_path):
    run("protocol", "--out", str(tmp_path / "p"), "--seeds", "0", "1")
    form = json.loads((tmp_path / "p" / "form.json").read_text())
    assert set(form["rosas"]) == {"warmth", "competence", "discomfort"} and "composite" not in json.dumps(form)
    assert all(len(v["items"]) == 6 for v in form["rosas"].values())
    assert {m["trait"] for m in form["manipulation_checks"]} == {"approach_tempo", "caution", "novelty_bias", "persistence"}
    assert "simulated" in form["disclosure"] and "no emotions" in form["disclosure"]
    assert form["eeriness"]["wording_check"] and len(form["attention_checks"]) == 2
    clips = json.loads((tmp_path / "p" / "clips.json").read_text())
    assert {c["configuration"] for c in clips["clips"]} == {"default", "bold", "careful", "null_control"}
    assert all(c["opening_s"] == 10.0 and "character_active=true" in c["render"] for c in clips["clips"])
    assert (tmp_path / "p" / "responses_template.csv").read_text().startswith("rater_id,cell")


def _write_ratings(path, effect, n=20, seed=0):
    """Synthetic raters: manipulation items move with the configuration by ``effect`` scale points."""
    rng = np.random.default_rng(seed)
    sys.path.insert(0, str(SCRIPT.parent))
    import ratings as R  # noqa: E402

    rows = []
    rid = 0
    for cell in R.CONFIGURATIONS:
        for _ in range(n):
            rid += 1
            seconds = 300.0 if rid % 7 else 60.0  # every seventh rater is too fast
            items = {}
            for m in R.MANIPULATION:
                hi, lo = R.HIGH_LOW[m["trait"]]
                shift = effect if cell == hi else (-effect if cell == lo else 0.0)
                items[m["id"]] = float(np.clip(round(4 + shift + rng.normal(0, 1.0)), 1, 7))
            for k, words in R.ROSAS.items():
                base = {"warmth": 5, "competence": 6, "discomfort": 2}[k]
                for w in words:
                    items[f"rosas_{w}"] = float(np.clip(round(base + rng.normal(0, 1.0)), 1, 9))
            for k in range(len(R.EERINESS)):
                items[f"eer_{k}"] = float(np.clip(round(2 + rng.normal(0, 0.8)), 1, 7))
            for k, words in R.MIND.items():
                for w in words:
                    items[f"mind_{w}"] = float(np.clip(round((5 if k == "agency" else 2) + rng.normal(0, 0.8)), 1, 7))
            for k, words in R.WEISMAN.items():
                for w in words:
                    items[f"whm_{w}"] = float(np.clip(round((5 if k == "mind" else 2) + rng.normal(0, 0.8)), 1, 7))
            items["att_1"], items["att_2"], items["clip_memory_correct"] = 9.0, 1.0, 1.0
            if rid % 11 == 0:
                items["att_1"] = 5.0  # a failed attention check
            for item, value in items.items():
                rows.append([f"r{rid}", cell, 0, seconds, f"ip{rid}", item, value, ""])
            rows.append([f"r{rid}", cell, 0, seconds, f"ip{rid}", "free_text", "", "it looked at the box then moved"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater_id", "cell", "clip_seed", "seconds", "ip_hash", "item", "value", "free_text"])
        w.writerows(rows)


def test_score_applies_exclusions_and_passes_a_clear_manipulation(tmp_path):
    _write_ratings(tmp_path / "r.csv", effect=1.5, n=90)  # 64 per cell after the synthetic exclusions (about 22 %)
    out = run("score", str(tmp_path / "r.csv"), "--out", str(tmp_path / "s"))
    rep = json.loads((tmp_path / "s" / "ratings.json").read_text())
    assert rep["kept"] < rep["raters"] and all(any("too fast" in w or "att_1" in w for w in why)
                                                for why in rep["excluded"].values())
    assert all(v["pass"] for v in rep["manipulation"].values())
    assert rep["checks"]["experience_below_midpoint"] and rep["checks"]["comfort_within_0.3_sd"]
    assert rep["checks"]["null_control_no_effect"] is True
    assert all(v["alpha"] is not None for v in rep["scales"]["rosas_warmth"].values())
    assert rep["checks"]["sample_size_met"] and rep["pass"]
    assert "rosas_total" not in rep["scales"] and "PASS" in out


def test_score_fails_when_the_configurations_are_not_perceptible(tmp_path):
    _write_ratings(tmp_path / "flat.csv", effect=0.0, n=30, seed=3)
    run("score", str(tmp_path / "flat.csv"), "--out", str(tmp_path / "s2"))
    rep = json.loads((tmp_path / "s2" / "ratings.json").read_text())
    assert not rep["pass"] and not all(v["pass"] for v in rep["manipulation"].values())
    assert rep["checks"]["sample_size_met"] is False  # 30 per cell < 64
