"""GLB export into the shared asset cache is atomic (no simulator)."""

import trimesh

from amr_rl.sim import assets


def test_export_glb_leaves_a_whole_file_and_no_temporary(tmp_path):
    """Concurrent simulations share the cache and check ``path.exists()`` before writing; a
    half-written file made one 2026-10-12 character-gate run fail in the loader."""
    path = tmp_path / "cache" / "box.glb"
    out = assets.export_glb(trimesh.creation.box(extents=(0.2, 0.2, 0.2)), path, y_up=True)
    assert out == path and path.exists() and path.read_bytes()[:4] == b"glTF"
    assert [p.name for p in path.parent.iterdir()] == ["box.glb"]  # the temporary was renamed, not left behind
    scene = trimesh.load(str(path), force="scene")
    assert len(scene.geometry) == 1
