"""Generate reusable robot assets (visual GLB meshes + URDF) from the YAML spec.

Visual meshes are detailed; collision geometry is deliberately simplified to
URDF primitives (box, cylinder, sphere). Mass properties are analytic for those
primitives. Run ``python -m amr_rl.robot.generate`` (or ``scripts/amr.sh
generate-robot``). Output is deterministic for a given spec and library version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np
import trimesh

from .spec import DEFAULT_SPEC, GENERATED_DIR, RobotSpec


# --------------------------------------------------------------------------
# Mesh primitives
# --------------------------------------------------------------------------
def rounded_box(size, corner_radius, edge_radius, segments=10) -> trimesh.Trimesh:
    """Convex rounded box: rounded plan-view corners and bevelled top/bottom edges."""
    lx, ly, lz = size
    corner_radius = min(corner_radius, lx / 2, ly / 2)
    edge_radius = min(edge_radius, corner_radius, lz / 2)
    points = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * (lx / 2 - corner_radius), sy * (ly / 2 - corner_radius)
            base = math.atan2(sy, sx)
            for i in range(segments + 1):
                phi = base - math.pi / 4 + (math.pi / 2) * i / segments
                for sz in (-1, 1):
                    for j in range(segments // 2 + 1):
                        psi = (math.pi / 2) * j / (segments // 2)
                        r = corner_radius - edge_radius + edge_radius * math.cos(psi)
                        z = sz * (lz / 2 - edge_radius + edge_radius * math.sin(psi))
                        points.append((cx + r * math.cos(phi), cy + r * math.sin(phi), z))
    return trimesh.convex.convex_hull(np.array(points))


def _colored(mesh: trimesh.Trimesh, rgba) -> trimesh.Trimesh:
    mesh = mesh.copy()
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[int(255 * c) for c in rgba],
        metallicFactor=0.0,
        roughnessFactor=0.7,
    )
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def _transform(mesh, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    mesh = mesh.copy()
    T = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    T[:3, 3] = xyz
    mesh.apply_transform(T)
    return mesh


def _export_glb(parts, path: Path) -> None:
    scene = trimesh.Scene()
    for index, part in enumerate(parts):
        scene.add_geometry(part, node_name=f"part_{index}", geom_name=f"part_{index}")
    path.write_bytes(scene.export(file_type="glb"))


# --------------------------------------------------------------------------
# Mass properties
# --------------------------------------------------------------------------
def box_inertia(m, x, y, z):
    return m / 12 * (y * y + z * z), m / 12 * (x * x + z * z), m / 12 * (x * x + y * y)


def cylinder_inertia_y(m, r, h):
    """Cylinder with its axis along y."""
    side = m / 12 * (3 * r * r + h * h)
    return side, m * r * r / 2, side


def sphere_inertia(m, r):
    value = 2 / 5 * m * r * r
    return value, value, value


# --------------------------------------------------------------------------
# URDF helpers
# --------------------------------------------------------------------------
def _fmt(values):
    return " ".join(f"{v:.6g}" for v in values)


def _origin(parent, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    ET.SubElement(parent, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))


def _inertial(link, mass, inertia, xyz=(0, 0, 0)):
    inertial = ET.SubElement(link, "inertial")
    _origin(inertial, xyz)
    ET.SubElement(inertial, "mass", value=f"{mass:.6g}")
    ixx, iyy, izz = inertia
    ET.SubElement(
        inertial, "inertia", ixx=f"{ixx:.6g}", ixy="0", ixz="0",
        iyy=f"{iyy:.6g}", iyz="0", izz=f"{izz:.6g}",
    )


def _visual_mesh(link, filename, name):
    visual = ET.SubElement(link, "visual", name=name)
    _origin(visual)
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "mesh", filename=filename)


def _collision(link, name, xyz, rpy, kind, **dims):
    collision = ET.SubElement(link, "collision", name=name)
    _origin(collision, xyz, rpy)
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, kind, **{k: _fmt(v) if isinstance(v, (list, tuple)) else f"{v:.6g}" for k, v in dims.items()})


def _fixed_joint(robot, name, parent, child, xyz, rpy=(0, 0, 0)):
    joint = ET.SubElement(robot, "joint", name=name, type="fixed")
    _origin(joint, xyz, rpy)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------
def generate(spec_path=DEFAULT_SPEC, out_dir=GENERATED_DIR) -> dict:
    spec = RobotSpec.load(spec_path)
    out_dir = Path(out_dir)
    meshes = out_dir / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    c, d, k, cam, s = spec.chassis, spec.drive, spec.caster, spec.camera, spec.screen
    base_z = spec.base_height

    # ---------------- visuals (each GLB is expressed in its link frame) -----
    chassis_center = spec.chassis_center
    shell = _transform(
        rounded_box((c["length"], c["width"], c["height"]), c["corner_radius"], c["edge_radius"]),
        chassis_center,
    )
    top_z = chassis_center[2] + c["height"] / 2
    bumper = _transform(
        rounded_box((0.02, c["width"] * 0.9, c["height"] * 0.45), 0.008, 0.006),
        (chassis_center[0] + c["length"] / 2 - 0.004, 0, chassis_center[2] - c["height"] * 0.12),
    )
    screen_c = spec.screen_center
    mast_h = s["mast_height"]
    mast = _transform(
        trimesh.creation.cylinder(radius=0.012, height=mast_h + 0.01, sections=24),
        (screen_c[0], 0, top_z + mast_h / 2),
    )
    base_parts = [
        _colored(shell, c["color"]),
        _colored(bumper, d["tire_color"]),
        _colored(mast, s["bezel_color"]),
    ]
    _export_glb(base_parts, meshes / "base_link.glb")

    wheel = _colored(
        trimesh.creation.cylinder(radius=d["wheel_radius"], height=d["wheel_width"], sections=48),
        d["tire_color"],
    )
    hub = _colored(
        trimesh.creation.cylinder(
            radius=d["wheel_radius"] * 0.55, height=d["wheel_width"] * 1.08, sections=32
        ),
        d["hub_color"],
    )
    # Cylinder axis is z; rotate to y (wheel axle) in the wheel link frame.
    rot = (math.pi / 2, 0, 0)
    _export_glb([_transform(wheel, rpy=rot), _transform(hub, rpy=rot)], meshes / "wheel.glb")

    _export_glb(
        [_colored(trimesh.creation.icosphere(subdivisions=3, radius=k["radius"]), k["color"])],
        meshes / "caster.glb",
    )

    housing = _colored(rounded_box((0.03, 0.05, 0.03), 0.008, 0.005), cam["housing_color"])
    lens = _colored(
        _transform(
            trimesh.creation.cylinder(radius=0.009, height=0.006, sections=32),
            (0.016, 0, 0),
            (0, math.pi / 2, 0),
        ),
        cam["lens_color"],
    )
    # camera_link: x forward, z up, pitched down about y; housing centred slightly behind the lens.
    _export_glb([_transform(housing, (-0.002, 0, 0)), lens], meshes / "camera.glb")

    bezel = _colored(
        rounded_box((s["depth"], s["width"], s["height"]), 0.012, 0.004), s["bezel_color"]
    )
    display = _colored(
        trimesh.creation.box(extents=(0.002, s["width"] * 0.9, s["height"] * 0.86)),
        s["display_color"],
    )
    _export_glb(
        [bezel, _transform(display, (s["depth"] / 2 + 0.0005, 0, 0))], meshes / "screen.glb"
    )

    # ---------------- URDF ----------------------------------------------------
    robot = ET.Element("robot", name=spec.name)
    # base_link: chassis + mast (merged), origin at axle midpoint, z = wheel radius.
    base = ET.SubElement(robot, "link", name="base_link")
    chassis_mass = c["mass"]
    _inertial(
        base,
        chassis_mass,
        box_inertia(chassis_mass, c["length"], c["width"], c["height"]),
        chassis_center,
    )
    _visual_mesh(base, "meshes/base_link.glb", "shell")
    _collision(
        base, "chassis_box", chassis_center, (0, 0, 0), "box",
        size=(c["length"] - 0.01, c["width"] - 0.004, c["height"]),
    )
    _collision(
        base, "mast", (screen_c[0], 0, top_z + mast_h / 2), (0, 0, 0), "cylinder",
        radius=0.012, length=mast_h,
    )

    for side, sign in (("left", 1), ("right", -1)):
        name = f"{side}_wheel"
        link = ET.SubElement(robot, "link", name=name)
        _inertial(link, d["wheel_mass"], cylinder_inertia_y(d["wheel_mass"], d["wheel_radius"], d["wheel_width"]))
        _visual_mesh(link, "meshes/wheel.glb", f"{side}_tire")
        _collision(
            link, f"{side}_tire", (0, 0, 0), (math.pi / 2, 0, 0), "cylinder",
            radius=d["wheel_radius"], length=d["wheel_width"],
        )
        joint = ET.SubElement(robot, "joint", name=f"{side}_wheel_joint", type="continuous")
        _origin(joint, (0, sign * d["track"] / 2, 0))
        ET.SubElement(joint, "parent", link="base_link")
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "axis", xyz="0 1 0")
        ET.SubElement(
            joint, "limit", effort=f"{d['max_wheel_torque']:.6g}",
            velocity=f"{d['max_wheel_speed']:.6g}",
        )
        ET.SubElement(joint, "dynamics", damping="0.001", friction="0.0")

    caster = ET.SubElement(robot, "link", name="caster_link")
    _inertial(caster, k["mass"], sphere_inertia(k["mass"], k["radius"]))
    _visual_mesh(caster, "meshes/caster.glb", "caster_ball")
    _collision(caster, "caster_ball", (0, 0, 0), (0, 0, 0), "sphere", radius=k["radius"])
    _fixed_joint(robot, "caster_joint", "base_link", "caster_link", spec.caster_center)

    camera_link = ET.SubElement(robot, "link", name="camera_link")
    _inertial(camera_link, 0.05, box_inertia(0.05, 0.03, 0.05, 0.03))
    _visual_mesh(camera_link, "meshes/camera.glb", "camera_housing")
    pitch = math.radians(cam["pitch_down"])
    _fixed_joint(robot, "camera_joint", "base_link", "camera_link", spec.camera_offset, (0, pitch, 0))

    screen_link = ET.SubElement(robot, "link", name="screen_link")
    _inertial(screen_link, s["mass"], box_inertia(s["mass"], s["depth"], s["width"], s["height"]))
    _visual_mesh(screen_link, "meshes/screen.glb", "screen_bezel")
    _collision(
        screen_link, "screen_box", (0, 0, 0), (0, 0, 0), "box",
        size=(s["depth"], s["width"], s["height"]),
    )
    tilt = math.radians(s["tilt_back"])
    _fixed_joint(robot, "screen_joint", "base_link", "screen_link", screen_c, (0, -tilt, 0))

    xml = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="  ")
    header = (
        "<?xml version=\"1.0\" ?>\n<!-- GENERATED by amr_rl.robot.generate from "
        "assets/robot/amr_spec.yaml. Edit the spec, not this file. -->\n"
    )
    urdf_path = out_dir / "amr.urdf"
    urdf_path.write_text(header + xml.split("\n", 1)[1])

    total_mass = chassis_mass + 2 * d["wheel_mass"] + k["mass"] + 0.05 + s["mass"]
    manifest = {
        "schema": "amr_rl.robot-assets.v1",
        "spec_sha256": hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
        "trimesh_version": trimesh.__version__,
        "dimensions_m": {
            "length": c["length"],
            "overall_width": spec.overall_width,
            "overall_height": float(screen_c[2] + base_z + s["height"] / 2 * math.cos(tilt)),
            "footprint_radius": spec.footprint_radius,
            "camera_height": cam["height_above_ground"],
        },
        "total_mass_kg": total_mass,
        "frames": ["base_link", "left_wheel", "right_wheel", "caster_link", "camera_link", "screen_link"],
        "files": {},
    }
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.relative_to(out_dir).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--out", default=str(GENERATED_DIR))
    args = parser.parse_args(argv)
    manifest = generate(args.spec, args.out)
    print(json.dumps({k: manifest[k] for k in ("dimensions_m", "total_mass_kg")}, indent=2))


if __name__ == "__main__":
    main()
