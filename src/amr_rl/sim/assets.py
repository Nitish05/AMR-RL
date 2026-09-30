"""Sim-only mesh builders: textured boxes/quads and articulated fixture URDFs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
from PIL import Image

from ..robot.generate import _colored, _fmt, _inertial, _origin, box_inertia, rounded_box

# Uniform, saturated fixture paints (engineered, documented appearance cue).
PAINTS = {
    "cyan": (0.05, 0.78, 0.82, 1.0),
    "magenta": (0.86, 0.10, 0.62, 1.0),
    "green": (0.12, 0.70, 0.22, 1.0),
    "blue": (0.12, 0.30, 0.88, 1.0),
    "violet": (0.55, 0.20, 0.85, 1.0),
    "yellow": (0.98, 0.84, 0.05, 1.0),
    "red": (0.92, 0.08, 0.06, 1.0),
}


def _textured(vertices, faces, uv, image: np.ndarray) -> trimesh.Trimesh:
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(image), metallicFactor=0.0, roughnessFactor=0.9
    )
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


def textured_quad(width, height, image, *, z=0.0) -> trimesh.Trimesh:
    w, h = width / 2, height / 2
    vertices = np.array([[-w, -h, z], [w, -h, z], [w, h, z], [-w, h, z]], float)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return _textured(vertices, faces, uv, image)


def textured_box(size, image) -> trimesh.Trimesh:
    """Box with each face independently UV-mapped to the whole image."""
    lx, ly, lz = (s / 2 for s in size)
    faces_def = [  # (normal axis corners) counter-clockwise seen from outside
        [(lx, -ly, -lz), (lx, ly, -lz), (lx, ly, lz), (lx, -ly, lz)],
        [(-lx, ly, -lz), (-lx, -ly, -lz), (-lx, -ly, lz), (-lx, ly, lz)],
        [(lx, ly, -lz), (-lx, ly, -lz), (-lx, ly, lz), (lx, ly, lz)],
        [(-lx, -ly, -lz), (lx, -ly, -lz), (lx, -ly, lz), (-lx, -ly, lz)],
        [(-lx, -ly, lz), (lx, -ly, lz), (lx, ly, lz), (-lx, ly, lz)],
        [(-lx, ly, -lz), (lx, ly, -lz), (lx, -ly, -lz), (-lx, -ly, -lz)],
    ]
    vertices, faces, uv = [], [], []
    for quad in faces_def:
        base = len(vertices)
        vertices.extend(quad)
        uv.extend([(0, 0), (1, 0), (1, 1), (0, 1)])
        faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    return _textured(np.array(vertices, float), np.array(faces), np.array(uv, float), image)


def export_glb(mesh_or_parts, path: Path, *, y_up: bool = False) -> Path:
    """Export parts to GLB. ``y_up=True`` writes standard glTF (Y-up) axes, which
    Genesis ``morphs.Mesh`` converts back to Z-up; URDF-referenced meshes stay Z-up
    because Genesis loads URDF meshes without conversion."""
    scene = trimesh.Scene()
    parts = mesh_or_parts if isinstance(mesh_or_parts, (list, tuple)) else [mesh_or_parts]
    if y_up:
        zup_to_yup = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
        parts = [part.copy() for part in parts]
        for part in parts:
            part.apply_transform(zup_to_yup)
    for i, part in enumerate(parts):
        scene.add_geometry(part, node_name=f"part_{i}", geom_name=f"part_{i}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scene.export(file_type="glb"))
    return path


def fixture_urdf(out_dir: Path, name: str, shape: str, paint: str, size) -> Path:
    """Fixed-base fixture with two hidden indicator panels on prismatic joints.

    The panels (yellow, red) rise above the body when the world-side response
    rule fires. They are ordinary rendered geometry: the robot can only learn
    about a response by seeing a panel in its onboard RGB.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    height = float(size[2])
    if shape == "cylinder":
        radius = float(size[0]) / 2
        body = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
        collision = ("cylinder", {"radius": radius, "length": height})
        half_w = radius
    elif shape == "box":
        body = rounded_box(size, 0.02, 0.01)
        collision = ("box", {"size": list(size)})
        half_w = min(size[0], size[1]) / 2
    else:
        raise ValueError("Fixture shape must be cylinder or box")
    body = _colored(body, PAINTS[paint])
    body.apply_translation((0, 0, height / 2))
    export_glb(body, out_dir / f"{name}_body.glb")
    panel_w, panel_h = min(0.16, 1.6 * half_w), 0.12
    for color in ("yellow", "red"):
        panel = _colored(trimesh.creation.box(extents=(0.012, panel_w, panel_h)), PAINTS[color])
        export_glb(panel, out_dir / f"{name}_{color}_panel.glb")

    robot = ET.Element("robot", name=name)
    base = ET.SubElement(robot, "link", name="body")
    _inertial(base, 3.0, box_inertia(3.0, *size), (0, 0, height / 2))
    visual = ET.SubElement(base, "visual")
    _origin(visual)
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=f"{name}_body.glb")
    col = ET.SubElement(base, "collision")
    _origin(col, (0, 0, height / 2))
    kind, dims = collision
    ET.SubElement(
        ET.SubElement(col, "geometry"), kind,
        **{k: _fmt(v) if isinstance(v, list) else f"{v:.6g}" for k, v in dims.items()},
    )
    # Panels live inside the body (hidden) at q=0 and rise by `travel` when raised.
    for index, color in enumerate(("yellow", "red")):
        link = ET.SubElement(robot, "link", name=f"{color}_panel")
        _inertial(link, 0.05, box_inertia(0.05, 0.012, panel_w, panel_h))
        vis = ET.SubElement(link, "visual")
        _origin(vis)
        ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename=f"{name}_{color}_panel.glb")
        joint = ET.SubElement(robot, "joint", name=f"{color}_lift", type="prismatic")
        offset = -0.012 if index == 0 else 0.012
        _origin(joint, (offset, 0, height - panel_h / 2 - 0.01))
        ET.SubElement(joint, "parent", link="body")
        ET.SubElement(joint, "child", link=f"{color}_panel")
        ET.SubElement(joint, "axis", xyz="0 0 1")
        ET.SubElement(joint, "limit", lower="0", upper=f"{panel_h + 0.02:.4f}", effort="50", velocity="1")
    xml = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="  ")
    path = out_dir / f"{name}.urdf"
    path.write_text(xml)
    return path


def cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def yaw_matrix(yaw: float) -> np.ndarray:
    T = np.eye(4)
    T[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    return T
