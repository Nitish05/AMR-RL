"""Export an AMR-RL world as a Genesis Studio project (schema 4) for authoring/inspection.

Genesis Studio is an optional external dependency. The export approximates
textured walls/obstacles as boxes (textures are AMR-RL generated assets) and
references the generated robot URDF. AMR-RL simulations do not load Studio
projects; this is an interchange for viewing/editing scenes in Studio's native
editor. Studio is never modified by this function.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..robot.spec import PROJECT_ROOT
from .world import load_world_config


def export_project(world_name: str, out_path: Path) -> dict:
    cfg = load_world_config(world_name)
    lx, ly = cfg["room"]["size"]
    wall_h, wall_t = cfg["room"].get("wall_height", 1.0), cfg["room"].get("wall_thickness", 0.1)
    objects = []
    for name, size, xy in (
        ("wall_n", (lx + 2 * wall_t, wall_t, wall_h), (0, ly / 2 + wall_t / 2)),
        ("wall_s", (lx + 2 * wall_t, wall_t, wall_h), (0, -ly / 2 - wall_t / 2)),
        ("wall_e", (wall_t, ly, wall_h), (lx / 2 + wall_t / 2, 0)),
        ("wall_w", (wall_t, ly, wall_h), (-lx / 2 - wall_t / 2, 0)),
    ):
        objects.append({"format": "box", "name": name, "fixed": True, "size": list(size),
                        "position": [xy[0], xy[1], wall_h / 2]})
    for item in cfg.get("obstacles", []):
        size = item["size"]
        objects.append({"format": "box", "name": item["name"], "fixed": True, "size": list(size),
                        "position": [*item["pos"], size[2] / 2],
                        "euler": [0.0, 0.0, math.degrees(item.get("yaw", 0.0))]})
    for item in cfg.get("fixtures", []):
        size = item["size"]
        if item["shape"] == "sphere":
            objects.append({"format": "sphere", "name": item["name"], "radius": size[0] / 2,
                            "position": [*item["pos"], size[0] / 2]})
        elif item["shape"] == "cylinder":
            objects.append({"format": "cylinder", "name": item["name"], "fixed": True, "radius": size[0] / 2,
                            "height": size[2], "position": [*item["pos"], size[2] / 2]})
        else:
            objects.append({"format": "box", "name": item["name"], "fixed": True, "size": list(size),
                            "position": [*item["pos"], size[2] / 2]})
    light = cfg.get("lighting", {})
    lights = [{"name": f"dir{i}", "type": "directional", "vector": d["dir"], "color": d.get("color", [1, 1, 1]),
               "intensity": d["intensity"]} for i, d in enumerate(light.get("directional", []))]
    lights += [{"name": f"point{i}", "type": "point", "vector": p["pos"], "color": p.get("color", [1, 1, 1]),
                "intensity": p["intensity"]} for i, p in enumerate(light.get("points", []))]
    start = cfg.get("robot_start", [0, 0, 0])
    urdf = PROJECT_ROOT / "assets/robot/generated/amr.urdf"
    project = {
        "schema_version": 4,
        "name": f"AMR-RL {cfg['name']}",
        "robot": {"name": "amr_pip", "source": str(urdf.relative_to(PROJECT_ROOT)), "browser_url": "/assets/amr.urdf", "format": "urdf",
                  "fixed_base": False, "position": [start[0], start[1], 0.052],
                  "euler": [0.0, 0.0, math.degrees(start[2])]},
        "robots": [],
        "objects": objects,
        "environment": {"lights": lights, "ground": True,
                        "ambient_light": [light.get("ambient", 0.3)] * 3},
        "physics": {"time_step": 0.01, "substeps": 1, "max_contacts": 1024},
        "sensors": [{"name": "onboard_rgb", "type": "camera", "parent": "camera_link", "robot_name": "amr_pip",
                     "resolution": [320, 240], "fov": 60, "rate": 10}],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(project, indent=2) + "\n")
    return project
