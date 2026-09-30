"""Genesis Studio project export (validated against Studio's own model when available)."""

import os
import sys

import pytest

from amr_rl.sim.studio_export import export_project


def test_export_structure(tmp_path):
    project = export_project("arena", tmp_path / "arena.genesis.json")
    assert project["schema_version"] == 4 and project["robot"]["name"] == "amr_pip"
    names = [o["name"] for o in project["objects"]]
    assert len(names) == len(set(names)) and "ground" not in names
    assert project["sensors"][0]["parent"] == "camera_link"


def test_export_validates_with_studio_model(tmp_path):
    src = os.environ.get("GENESIS_STUDIO_SRC")
    if not src:
        pytest.skip("Set GENESIS_STUDIO_SRC=<Genesis-Studio>/apps/api/src to validate against Studio's model")
    sys.path.insert(0, src)
    from genesis_studio.models import Project

    Project.model_validate(export_project("arena", tmp_path / "arena.genesis.json"))
