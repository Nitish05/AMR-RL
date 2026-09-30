"""Data-only run provenance: what was on disk and installed when a run started.

``capture_run_identity`` hashes AMR-RL's first-party sources, the configuration
files a run names, an optional asset manifest and (optionally) a Genesis Studio
checkout, and records interpreter/package versions from distribution metadata.
It never imports simulation, rendering or learning packages, never loads
weights, and only runs read-only, lock-free Git queries (``--no-optional-locks``)
so no repository or index is ever written.

Scope, stated honestly: this is a capture at one moment plus an end-of-run
re-check (``verify_run_identity``). It is not an atomic filesystem freeze, not a
signature and not publisher authentication; a file changed and restored between
capture and verification is not detected. Source files and asset bytes are
hashed, not copied; only small text configuration files and the Git working-tree
diff are snapshotted. Effective in-memory overrides must be recorded by the
caller (for example via ``extra``).

Adapted from BB8-RL ``src/bb8_rl/run_identity.py`` (see docs/PROVENANCE.md).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

SCHEMA = "amr_rl.run-identity.v1"
IDENTITY_FILENAME = "identity.json"
CONFIG_SNAPSHOT_LIMIT = 1_048_576  # bytes; larger or binary configs are hashed only

_SOURCE_DIRS = ("src", "scripts", "configs", "tests")
_SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".js",
    ".mjs",
    ".css",
    ".html",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
}
# Generated, cached or environment content is never first-party source.
_EXCLUDED_PARTS = {
    "__pycache__",
    "node_modules",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "work",
    "outputs",
    "image-evidence",
}
_PACKAGES = (
    "amr-rl",
    "genesis-world",
    "torch",
    "numpy",
    "opencv-python-headless",
    "opencv-python",
    "PyYAML",
    "scipy",
    "trimesh",
    "pydantic",
)


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value):
    """SHA-256 of sorted, compact, NaN-free JSON (the identity/receipt hash form)."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _git(root, *args):
    """Read-only Git query; returns stdout or None. Never takes optional locks."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _excluded(relative):
    return any(
        part in _EXCLUDED_PARTS or part.startswith(".venv") or part.endswith(".egg-info")
        for part in relative.parts
    )


def _safe_source(path, root, skip=None):
    relative = path.relative_to(root)
    return (
        path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(root)
        and not _excluded(relative)
        and path.suffix in _SOURCE_SUFFIXES
        and (skip is None or not path.resolve().is_relative_to(skip))
    )


def _project_sources(root, skip=None):
    files = set()
    for name in _SOURCE_DIRS:
        directory = root / name
        if directory.is_dir():
            files.update(p for p in directory.rglob("*") if _safe_source(p, root, skip))
    # Installed wheels have the package directly below the supplied root.
    if not (root / "src/amr_rl").is_dir() and (root / "amr_rl").is_dir():
        files.update(p for p in (root / "amr_rl").rglob("*") if _safe_source(p, root, skip))
    if (root / "pyproject.toml").is_file():
        files.add(root / "pyproject.toml")
    return sorted(files)


def _default_project_root():
    package = Path(__file__).resolve().parents[1]  # .../amr_rl
    candidate = package.parents[1]  # checkout root when running from src/
    return candidate if (candidate / "src/amr_rl").is_dir() else package.parent


def _record(path):
    path = Path(path).absolute()
    if not path.is_file():
        raise ValueError(f"Run identity input is not a regular file: {path}")
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": _digest(path),
        "bytes": path.stat().st_size,
    }


def _config_records(config_paths, snapshot_root=None):
    records = {}
    for index, raw in enumerate(config_paths):
        path = Path(raw).absolute()
        label = path.name if path.name not in records else f"{index:02d}-{path.name}"
        data = path.read_bytes() if path.is_file() else None
        if data is None:
            raise ValueError(f"Run identity config is not a regular file: {path}")
        record = {
            "path": str(path),
            "resolved_path": str(path.resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "snapshot": None,
        }
        if snapshot_root is not None:
            try:
                data.decode("utf-8")
                text = len(data) <= CONFIG_SNAPSHOT_LIMIT
            except UnicodeDecodeError:
                text = False
            if text:
                target = snapshot_root / "configuration" / label
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                record["snapshot"] = f"configuration/{label}"
        records[label] = record
    return records


def _asset_records(manifest_path):
    """Re-hash every manifest entry relative to the manifest; raise on mismatch."""
    manifest_path = Path(manifest_path).absolute()
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable asset manifest: {manifest_path}") from error
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        # Malformed file content, not a caller type error.
        raise ValueError("Asset manifest must contain a 'files' map of path -> sha256")  # noqa: TRY004
    root = manifest_path.resolve().parent
    records = {}
    for relative, expected in sorted(files.items()):
        candidate = Path(relative)
        target = (root / candidate).resolve()
        if candidate.is_absolute() or not target.is_relative_to(root):
            raise ValueError(f"Asset path escapes manifest directory: {relative}")
        try:
            record = _record(root / candidate)
        except ValueError as error:
            raise ValueError(f"Asset missing while recording run identity: {relative}") from error
        if record["sha256"] != expected:
            raise ValueError(
                f"Asset differs from manifest while recording run identity: {relative}"
            )
        records[relative] = record
    return {
        "manifest": _record(manifest_path),
        "root": str(root),
        "files": records,
    }


def _empty_tree(root):
    tree = _git(root, "hash-object", "-t", "tree", os.devnull)
    return tree.strip() if tree else None


def _git_identity(root, files, snapshot_root=None, name=None):
    """HEAD/branch/status/diff for the given files; read-only and lock-free."""
    if root is None:
        return {"available": False, "reason": "Root not provided"}
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None:
        return {"available": False, "root": str(root), "reason": "Not a Git checkout"}
    repository = Path(top.strip()).resolve()
    head = _git(repository, "rev-parse", "--verify", "-q", "HEAD")
    head = head.strip() if head else None
    relative = sorted(
        {
            path.resolve().relative_to(repository).as_posix()
            for path in files
            if path.resolve().is_relative_to(repository)
        }
    )
    base = head or _empty_tree(repository)  # unborn branch: diff against empty tree
    # Empty path lists must not accidentally collect unrelated files or secrets.
    if relative:
        status = _git(
            repository, "status", "--porcelain=v1", "--untracked-files=all", "--", *relative
        )
        diff = (
            _git(
                repository,
                "diff",
                base,
                "--no-ext-diff",
                "--no-textconv",
                "--binary",
                "--",
                *relative,
            )
            if base
            else None
        )
    else:
        status = diff = ""
    if status is None or diff is None:
        raise ValueError(f"Unable to capture relevant Git changes: {repository}")
    result = {
        "available": True,
        "root": str(repository),
        "head": head,
        "branch": (_git(repository, "symbolic-ref", "--short", "-q", "HEAD") or "").strip(),
        "status": status,
        "dirty": bool(status),
        "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "tracked_paths": relative,
    }
    if snapshot_root is not None:
        target = snapshot_root / "git" / f"{name}.patch"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(diff)
        result["diff_snapshot"] = target.relative_to(snapshot_root).as_posix()
    return result


def _studio_identity(root):
    """Optional external dependency: HEAD and working-tree summary only."""
    if root is None:
        return {"available": False, "reason": "Genesis Studio root not provided"}
    root = Path(root).resolve()
    if not root.is_dir():
        return {"available": False, "root": str(root), "reason": "Directory missing"}
    top = _git(root, "rev-parse", "--show-toplevel")
    head = _git(root, "rev-parse", "--verify", "-q", "HEAD") if top else None
    if top is None or head is None:
        return {"available": False, "root": str(root), "reason": "Git checkout unavailable"}
    repository = Path(top.strip()).resolve()
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=no")
    return {
        "available": True,
        "root": str(repository),
        "head": head.strip(),
        "branch": (_git(repository, "symbolic-ref", "--short", "-q", "HEAD") or "").strip(),
        "dirty": bool(status) if status is not None else None,
        "status_sha256": hashlib.sha256(status.encode()).hexdigest()
        if status is not None
        else None,
    }


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime():
    # Distribution metadata only: optional/native packages are never imported.
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: _version(name) for name in _PACKAGES},
    }


def capture_run_identity(
    snapshot_dir,
    *,
    project_root=None,
    config_paths=(),
    asset_manifest=None,
    studio_root=None,
    extra=None,
):
    """Record run inputs into ``<snapshot_dir>/identity.json`` and return the dict.

    ``snapshot_dir`` must not exist. Asset mismatches raise ``ValueError`` before
    the directory is created. ``extra`` must be JSON-serialisable.
    """
    snapshot_root = Path(snapshot_dir).absolute()
    if snapshot_root.exists():
        raise ValueError("Run identity requires a fresh snapshot directory")
    project_root = Path(project_root or _default_project_root()).resolve()
    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")
    assets = _asset_records(asset_manifest) if asset_manifest is not None else None
    extra = json.loads(json.dumps(extra if extra is not None else {}, allow_nan=False))
    snapshot_root.mkdir(parents=True)
    snapshot_root = snapshot_root.resolve()
    paths = _project_sources(project_root, skip=snapshot_root)
    sources = {path.relative_to(project_root).as_posix(): _record(path) for path in paths}
    configuration = _config_records(config_paths, snapshot_root)
    relevant = paths + [Path(record["resolved_path"]) for record in configuration.values()]
    identity = {
        "schema": SCHEMA,
        "project_root": str(project_root),
        "snapshot_dir": str(snapshot_root),
        "source_files": sources,
        "configuration": configuration,
        "assets": assets,
        "git": {
            "amr_rl": _git_identity(project_root, relevant, snapshot_root, "amr_rl"),
            "genesis_studio": _studio_identity(studio_root),
        },
        "runtime": _runtime(),
        "extra": extra,
        "scope": (
            "On-disk inputs hashed at capture; small text configs and the Git diff "
            "snapshotted; sources and assets hashed only. Not an atomic freeze, "
            "signature or publisher authentication."
        ),
    }
    identity["identity_sha256"] = canonical_json_sha256(identity)
    (snapshot_root / IDENTITY_FILENAME).write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    return identity


def _changed(previous, current):
    return current is None or any(
        current[key] != previous[key] for key in ("resolved_path", "sha256", "bytes")
    )


def _current(path):
    try:
        return _record(path)
    except (OSError, ValueError):
        return None


def verify_run_identity(identity):
    """Re-check a captured identity; return sorted change descriptions (empty = same).

    Entries are ``source_files/<rel>`` (modified, added or removed),
    ``configuration/<label>``, ``assets/<rel>`` (or ``assets/manifest``),
    ``git/amr_rl``, ``git/genesis_studio``, ``runtime/packages/<name>`` and
    ``identity_sha256`` when the record itself was altered.
    """
    changes = set()
    body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    if canonical_json_sha256(body) != identity.get("identity_sha256"):
        changes.add("identity_sha256")
    for name, previous in identity["source_files"].items():
        if _changed(previous, _current(previous["path"])):
            changes.add(f"source_files/{name}")
    project_root = Path(identity["project_root"])
    skip = Path(identity["snapshot_dir"]) if identity.get("snapshot_dir") else None
    current_paths = _project_sources(project_root, skip) if project_root.is_dir() else []
    current_names = {path.relative_to(project_root).as_posix() for path in current_paths}
    changes.update(f"source_files/{name}" for name in current_names ^ set(identity["source_files"]))
    for label, previous in identity["configuration"].items():
        if _changed(previous, _current(previous["path"])):
            changes.add(f"configuration/{label}")
    assets = identity.get("assets")
    if assets is not None:
        if _changed(assets["manifest"], _current(assets["manifest"]["path"])):
            changes.add("assets/manifest")
        for name, previous in assets["files"].items():
            if _changed(previous, _current(previous["path"])):
                changes.add(f"assets/{name}")
    git = identity["git"]
    relevant = current_paths + [
        Path(record["resolved_path"]) for record in identity["configuration"].values()
    ]
    previous = {k: v for k, v in git["amr_rl"].items() if k != "diff_snapshot"}
    try:
        current = _git_identity(project_root if previous.get("available") else None, relevant)
    except ValueError:
        current = None
    if previous.get("available") and current != previous:
        changes.add("git/amr_rl")
    studio = git["genesis_studio"]
    if studio.get("available") and _studio_identity(studio["root"]) != studio:
        changes.add("git/genesis_studio")
    changes.update(
        f"runtime/packages/{name}"
        for name, version in identity["runtime"]["packages"].items()
        if _version(name) != version
    )
    return sorted(changes)
