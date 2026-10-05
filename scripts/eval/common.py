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
        extra={**(extra or {}), "depth_model": _depth_model_identity()},
    )


def _depth_model_identity():
    from amr_rl.perception.near_depth import MODEL_ID, MODEL_LICENSE, MODEL_REVISION, model_available

    return {"id": MODEL_ID, "revision": MODEL_REVISION, "license": MODEL_LICENSE,
            "available_locally": bool(model_available())}


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


def operator_turn_until_tracking(session, log, *, w=0.4, timeout=40.0, reason="not tracking after recovery"):
    """Evaluation operator intervention (counted in ``log``): turn the robot slowly in
    place with manual commands until the VSLAM tracks again or ``timeout`` passes. The
    robot itself never resumes motion after its bounded recovery gave up."""
    rt = session.runtime
    log.append({"t": session.now, "intervention": "manual_relocalisation_turn", "reason": reason})
    end = session.now + timeout
    while session.now < end and rt.slam.status != "tracking":
        rt.command({"action": "manual", "v": 0.0, "w": w, "generation": rt.supervisor.generation})
        session.control_step()
    rt.command({"action": "manual", "v": 0.0, "w": 0.0, "generation": rt.supervisor.generation})
    session.control_step()
    log[-1]["end"] = session.now
    log[-1]["tracking"] = rt.slam.status == "tracking"
    return log[-1]["tracking"]


FALSE_RELOC_M = 0.2
FALSE_RELOC_RAD = 0.15  # 8.6 deg


def reloc_transitions(truth_rows, *, max_err=FALSE_RELOC_M, max_herr=FALSE_RELOC_RAD):
    """Scoring only: every lost/relocalising -> tracking transition with the true error
    of the pose the robot accepted. ``false`` marks a relocalisation that was wrong
    (> max_err m or > max_herr rad), i.e. the robot continued in a wrong frame."""
    out = []
    for prev, row in zip(truth_rows[:-1], truth_rows[1:]):
        if prev["status"] in ("lost", "relocalizing") and row["status"] == "tracking" and row["err"] is not None:
            herr = abs(row["herr"]) if row["herr"] is not None else 0.0
            out.append({"t": row["t"], "err_m": float(row["err"]), "herr_rad": float(herr),
                        "false": bool(row["err"] > max_err or herr > max_herr)})
    return out


def _truth_at(truth_rows, t):
    best = min(truth_rows, key=lambda r: abs(r["t"] - t))
    return np.asarray(best["gt"], float)


def keyframe_map_error(keyframes, truth_rows):
    """Scoring only: the FINAL keyframe poses (after any loop closure corrected the
    past) against the true pose at each keyframe's time, in the map frame."""
    if not keyframes or not truth_rows:
        return None
    errs, herrs = [], []
    for k in keyframes:
        gt = _truth_at(truth_rows, k.timestamp)
        errs.append(math.dist(k.pose[:2], gt[:2]))
        herrs.append(abs((k.pose[2] - gt[2] + math.pi) % (2 * math.pi) - math.pi))
    return {"keyframes": len(errs), "rmse_m": float(np.sqrt(np.mean(np.square(errs)))), "max_m": float(max(errs)),
            "max_heading_deg": float(np.degrees(max(herrs)))}


def score_loops(loop_log, truth_rows, *, max_err=0.1, max_herr=0.1):
    """Scoring only: was each CLOSED loop's relative pose right? Compares the loop
    constraint (candidate keyframe -> current keyframe) with the true relative pose."""
    out = []
    for e in loop_log:
        if e.get("outcome") != "closed":
            continue
        a, b = _truth_at(truth_rows, e["t_candidate"]), _truth_at(truth_rows, e["t"])
        c, s = math.cos(a[2]), math.sin(a[2])
        true_rel = np.array([c * (b[0] - a[0]) + s * (b[1] - a[1]), -s * (b[0] - a[0]) + c * (b[1] - a[1]),
                             (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi])
        pc, pl = np.asarray(e["pose_candidate"]), np.asarray(e["pose_loop"])
        c, s = math.cos(pc[2]), math.sin(pc[2])
        est_rel = np.array([c * (pl[0] - pc[0]) + s * (pl[1] - pc[1]), -s * (pl[0] - pc[0]) + c * (pl[1] - pc[1]),
                            (pl[2] - pc[2] + math.pi) % (2 * math.pi) - math.pi])
        err = float(np.hypot(*(est_rel[:2] - true_rel[:2])))
        herr = abs((est_rel[2] - true_rel[2] + math.pi) % (2 * math.pi) - math.pi)
        out.append({"t": e["t"], "kf": e["kf"], "candidate": e["candidate"], "err_m": err, "herr_rad": herr,
                    "false": bool(err > max_err or herr > max_herr), "correction_m": e.get("correction_m")})
    return out


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
    transitions = reloc_transitions(truth_rows)
    out["reloc_transitions"] = len(transitions)
    out["false_reloc_transitions"] = sum(t["false"] for t in transitions)
    sig = [(r["sigma"], r["err"]) for r in rows if r["sigma"]]
    if sig:
        out["sigma_coverage"] = float(np.mean([e <= s for s, e in sig]))
        out["sigma_coverage_2x"] = float(np.mean([e <= 2 * s for s, e in sig]))
    return out


# ---------------------------------------------------------------- round 9 options
def add_runtime_args(parser):
    """Options shared by build_map / navigation / learning (round 9)."""
    parser.add_argument("--odometry", default="command", choices=["command", "imu_encoders", "command_model"],
                        help="motion source for localisation (RuntimeConfig.odometry)")
    parser.add_argument("--runtime", nargs="*", default=[], help="RuntimeConfig overrides key=value (JSON values)")
    parser.add_argument("--odo", nargs="*", default=[], help="OdometryConfig overrides key=value (JSON values)")
    parser.add_argument("--world-set", nargs="*", default=[],
                        help="EVALUATION: world config overrides key=value (JSON), e.g. wheel_friction=0.4, "
                             "slip_patches=[...], sensor_faults={\"imu_off\":true}")
    parser.add_argument("--record-frames", action="store_true",
                        help="keep every onboard frame (audit images) and raw sensor batch for live replay")


def _kv(items):
    out = {}
    for item in items:
        key, value = item.split("=", 1)
        out[key] = json.loads(value)
    return out


def apply_runtime_args(cfg, args):
    cfg.odometry = args.odometry
    for key, value in _kv(args.runtime).items():
        if not hasattr(cfg, key):
            raise SystemExit(f"unknown RuntimeConfig field {key}")
        setattr(cfg, key, value)
    for key, value in _kv(args.odo).items():
        if not hasattr(cfg.odometry_cfg, key):
            raise SystemExit(f"unknown OdometryConfig field {key}")
        setattr(cfg.odometry_cfg, key, value)
    if args.record_frames:
        cfg.audit_images = True
    return cfg


def world_overrides(args, base=None):
    return {**(base or {}), **_kv(args.world_set)}


def start_recording(session, args):
    if args.record_frames:
        session.proprio_log = []


def save_recording(session, run_dir: Path):
    """Raw sensor batches per control step (live replay input); frames are the audit
    images written by the runtime's evidence archive."""
    if session.proprio_log is None:
        return
    imu = [[s.t, *map(float, s.gyro), *map(float, s.accel)] for b in session.proprio_log for s in b.imu]
    enc = [[s.t, s.left, s.right] for b in session.proprio_log for s in b.encoders]
    np.savez_compressed(Path(run_dir) / "proprio.npz", imu=np.asarray(imu, float).reshape(-1, 7),
                        enc=np.asarray(enc, float).reshape(-1, 3))
