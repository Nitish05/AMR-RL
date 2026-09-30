"""Shared helpers for evaluation scripts (evidence directories, provenance, JSON)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from amr_rl.evidence.provenance import capture_run_identity
from amr_rl.robot.spec import PROJECT_ROOT

EVIDENCE_ROOT = PROJECT_ROOT / "work" / "evidence"


def fresh_dir(name: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = EVIDENCE_ROOT / f"{name}-{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def provenance(run_dir: Path, configs=(), extra=None):
    # One snapshot per process: evaluations split across processes (to bound memory)
    # write provenance, provenance-2, ... into the same run directory.
    target, k = run_dir / "provenance", 1
    while target.exists():
        k += 1
        target = run_dir / f"provenance-{k}"
    return capture_run_identity(
        target,
        project_root=PROJECT_ROOT,
        config_paths=[str(p) for p in configs],
        asset_manifest=str(PROJECT_ROOT / "assets/robot/generated/manifest.json"),
        extra=extra or {},
    )


def jdefault(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(type(value))


def dump(path: Path, data):
    path.write_text(json.dumps(data, indent=1, default=jdefault))


def similarity_alignment(est, gt):
    """Umeyama similarity (2-D) of estimated to true positions: scale, rmse."""
    est, gt = np.asarray(est, float), np.asarray(gt, float)
    if len(est) < 3:
        return None
    mu_e, mu_g = est.mean(0), gt.mean(0)
    E, G = est - mu_e, gt - mu_g
    cov = G.T @ E / len(est)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    var_e = (E ** 2).sum() / len(est)
    scale = float(np.trace(np.diag(S) @ D) / var_e) if var_e > 0 else float("nan")
    aligned = scale * est @ R.T + (mu_g - scale * R @ mu_e)
    rmse = float(np.sqrt(((aligned - gt) ** 2).sum(1).mean()))
    return {"scale": scale, "rotation_deg": math.degrees(math.atan2(R[1, 0], R[0, 0])), "rmse_after_similarity": rmse}


def trajectory_metrics(truth_rows):
    rows = [r for r in truth_rows if r["est"] is not None and r["status"] == "tracking"]
    errs = [r["err"] for r in rows]
    herrs = [math.degrees(r["herr"]) for r in rows]
    gt_len = sum(math.dist(a["gt"][:2], b["gt"][:2]) for a, b in zip(truth_rows[:-1], truth_rows[1:]))
    out = {
        "frames": len(truth_rows),
        "tracking_frames": len(rows),
        "predicted_frames": sum(r["status"] == "predicted" for r in truth_rows),
        "lost_frames": sum(r["status"] in ("lost", "relocalizing") for r in truth_rows),
        "true_path_length_m": gt_len,
        "ate_rmse_m": float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
        "max_position_error_m": max(errs) if errs else None,
        "final_position_error_m": errs[-1] if errs else None,
        "max_heading_error_deg": max(herrs) if herrs else None,
        "similarity": similarity_alignment([r["est"][:2] for r in rows], [r["gt"][:2] for r in rows]),
    }
    sig = [(r["sigma"], r["err"]) for r in rows if r["sigma"]]
    if sig:
        out["sigma_coverage"] = float(np.mean([e <= s for s, e in sig]))
        out["sigma_coverage_2x"] = float(np.mean([e <= 2 * s for s, e in sig]))
    return out
