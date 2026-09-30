"""Run identity capture: fresh snapshots, change detection, lock-free Git."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from amr_rl.evidence.provenance import (
    SCHEMA,
    canonical_json_sha256,
    capture_run_identity,
    verify_run_identity,
)


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch, tmp_path):
    # Never let a test discover a Git repository above tmp_path.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)


def make_project(root):
    (root / "src/amr_rl").mkdir(parents=True)
    (root / "src/amr_rl/__init__.py").write_text("VALUE = 1\n")
    (root / "src/amr_rl/__pycache__").mkdir()
    (root / "src/amr_rl/__pycache__/x.cpython-312.pyc").write_bytes(b"\0")
    (root / "scripts").mkdir()
    (root / "scripts/run.py").write_text("print('run')\n")
    (root / "configs").mkdir()
    (root / "configs/world.yaml").write_text("room: a\n")
    (root / "tests").mkdir()
    (root / "tests/test_x.py").write_text("def test_x():\n    pass\n")
    (root / "work/run-1").mkdir(parents=True)
    (root / "work/run-1/generated.json").write_text("{}")
    (root / ".venv-amr/lib").mkdir(parents=True)
    (root / ".venv-amr/lib/site.py").write_text("")
    (root / "pyproject.toml").write_text("[project]\nname = 'amr-rl'\n")
    return root


def make_assets(root):
    assets = root / "assets"
    (assets / "robot").mkdir(parents=True)
    (assets / "robot/pip.urdf").write_text("<robot name='pip'/>\n")
    digest = hashlib.sha256((assets / "robot/pip.urdf").read_bytes()).hexdigest()
    manifest = assets / "manifest.json"
    manifest.write_text(json.dumps({"files": {"robot/pip.urdf": digest}}))
    return manifest


def git(root, *args):
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def test_capture_outside_git_records_sources_configs_and_runtime(tmp_path):
    project = make_project(tmp_path / "project")
    manifest = make_assets(project)
    snapshot = tmp_path / "snapshot"
    identity = capture_run_identity(
        snapshot,
        project_root=project,
        config_paths=[project / "configs/world.yaml"],
        asset_manifest=manifest,
        extra={"seed": 7},
    )
    assert identity["schema"] == SCHEMA
    assert set(identity["source_files"]) == {
        "src/amr_rl/__init__.py",
        "scripts/run.py",
        "configs/world.yaml",
        "tests/test_x.py",
        "pyproject.toml",
    }
    record = identity["source_files"]["scripts/run.py"]
    assert record["bytes"] == len("print('run')\n")
    assert identity["git"]["amr_rl"]["available"] is False
    assert identity["git"]["genesis_studio"]["available"] is False
    assert identity["assets"]["files"]["robot/pip.urdf"]["bytes"] > 0
    config = identity["configuration"]["world.yaml"]
    assert (snapshot / config["snapshot"]).read_text() == "room: a\n"
    runtime = identity["runtime"]
    assert runtime["python"] == sys.version
    assert set(runtime["packages"]) >= {"numpy", "genesis-world", "torch", "amr-rl"}
    assert identity["extra"] == {"seed": 7}
    saved = json.loads((snapshot / "identity.json").read_text())
    assert saved == identity
    body = {k: v for k, v in saved.items() if k != "identity_sha256"}
    assert canonical_json_sha256(body) == saved["identity_sha256"]
    assert verify_run_identity(saved) == []


def test_fresh_snapshot_directory_required(tmp_path):
    project = make_project(tmp_path / "project")
    (tmp_path / "exists").mkdir()
    with pytest.raises(ValueError, match="fresh"):
        capture_run_identity(tmp_path / "exists", project_root=project)


def test_verify_detects_modified_added_and_removed_files(tmp_path):
    project = make_project(tmp_path / "project")
    identity = capture_run_identity(
        tmp_path / "snap", project_root=project, config_paths=[project / "configs/world.yaml"]
    )
    (project / "src/amr_rl/__init__.py").write_text("VALUE = 2\n")
    (project / "src/amr_rl/new.py").write_text("")
    (project / "tests/test_x.py").unlink()
    (project / "configs/world.yaml").write_text("room: b\n")
    (project / "work/run-1/generated.json").write_text("[]")  # ignored
    assert verify_run_identity(identity) == [
        "configuration/world.yaml",
        "source_files/configs/world.yaml",
        "source_files/src/amr_rl/__init__.py",
        "source_files/src/amr_rl/new.py",
        "source_files/tests/test_x.py",
    ]


def test_verify_detects_tampered_identity_record(tmp_path):
    project = make_project(tmp_path / "project")
    identity = capture_run_identity(tmp_path / "snap", project_root=project)
    identity["extra"] = {"forged": True}
    assert verify_run_identity(identity) == ["identity_sha256"]


def test_asset_manifest_mismatch_raises_before_snapshot(tmp_path):
    project = make_project(tmp_path / "project")
    manifest = make_assets(project)
    (project / "assets/robot/pip.urdf").write_text("<robot name='changed'/>\n")
    with pytest.raises(ValueError, match="robot/pip.urdf"):
        capture_run_identity(tmp_path / "snap", project_root=project, asset_manifest=manifest)
    assert not (tmp_path / "snap").exists()


def test_asset_manifest_rejects_escaping_paths(tmp_path):
    project = make_project(tmp_path / "project")
    manifest = project / "assets.json"
    manifest.write_text(json.dumps({"files": {"../outside": "0" * 64}}))
    with pytest.raises(ValueError, match="escapes"):
        capture_run_identity(tmp_path / "snap", project_root=project, asset_manifest=manifest)


def test_git_identity_is_recorded_without_writing_the_repository(tmp_path):
    project = make_project(tmp_path / "project")
    (project / ".gitignore").write_text("work/\n.venv*/\n__pycache__/\n")
    git(project, "init", "-q", "-b", "main")
    git(project, "add", "-A")
    git(project, "commit", "-q", "-m", "init")
    (project / "scripts/run.py").write_text("print('dirty')\n")
    studio = tmp_path / "studio"
    studio.mkdir()
    git(studio, "init", "-q", "-b", "dev")
    git(studio, "commit", "-q", "--allow-empty", "-m", "s")
    before = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for root in (project / ".git", studio / ".git")
        for path in root.rglob("*")
        if path.is_file()
    }
    identity = capture_run_identity(tmp_path / "snap", project_root=project, studio_root=studio)
    after = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for root in (project / ".git", studio / ".git")
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
    record = identity["git"]["amr_rl"]
    assert record["available"] and len(record["head"]) == 40
    assert record["branch"] == "main" and record["dirty"]
    assert "scripts/run.py" in record["status"]
    patch = (tmp_path / "snap" / record["diff_snapshot"]).read_text()
    assert "print('dirty')" in patch
    assert hashlib.sha256(patch.encode()).hexdigest() == record["dirty_diff_sha256"]
    studio_record = identity["git"]["genesis_studio"]
    assert studio_record["available"] and studio_record["branch"] == "dev"
    assert verify_run_identity(identity) == []
    git(studio, "commit", "-q", "--allow-empty", "-m", "s2")
    assert verify_run_identity(identity) == ["git/genesis_studio"]


def test_unborn_git_branch_is_supported(tmp_path):
    project = make_project(tmp_path / "project")
    git(project, "init", "-q", "-b", "main")
    identity = capture_run_identity(tmp_path / "snap", project_root=project)
    record = identity["git"]["amr_rl"]
    assert record["available"] and record["head"] is None and record["branch"] == "main"
    assert verify_run_identity(identity) == []


def test_provenance_does_not_import_heavy_packages(tmp_path):
    project = make_project(tmp_path / "project")
    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys\n"
        "from amr_rl.evidence.provenance import capture_run_identity\n"
        f"capture_run_identity({str(tmp_path / 'snap')!r}, project_root={str(project)!r})\n"
        "heavy = {'numpy', 'cv2', 'torch', 'genesis', 'scipy', 'trimesh', 'yaml'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    assert result.stdout.strip() == "[]"
