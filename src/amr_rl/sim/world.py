"""Genesis scene construction and world-side fixture mechanics (simulation only).

This module owns everything the *world* does: room geometry, lighting, fixture
response rules, the roller ball, and rendering. The robot's runtime never
receives this object; it receives onboard RGB frames, a drive backend, and a
screen framebuffer sink (see ``amr_rl.sim.harness``). Fixture response rules may
read the robot's true pose and screen content because they model how the
*environment* reacts; they are engineered and documented in each world config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..robot.spec import PROJECT_ROOT, RobotSpec
from . import assets, textures

CAMERA_ID = "onboard_rgb"
WORLD_DIR = PROJECT_ROOT / "configs" / "worlds"
CACHE_DIR = PROJECT_ROOT / "work" / "cache" / "world-assets"


def load_world_config(name_or_path) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = WORLD_DIR / f"{name_or_path}.yaml"
    config = yaml.safe_load(path.read_text())
    config["_path"] = str(path)
    return config


@dataclass
class Frame:
    """One onboard camera image. The ONLY perceptual input to the runtime."""

    index: int
    timestamp: float
    rgb: np.ndarray
    camera_id: str = CAMERA_ID
    calibration_version: str = ""


@dataclass
class FixtureState:
    name: str
    raised: str | None = None  # "yellow" | "red" | None
    lower_at: float = 0.0
    signal_since: float | None = None
    signal_consumed: bool = False
    contact_active: bool = False
    last_contact: float = -math.inf


@dataclass
class WorldEvent:
    time: float
    fixture: str
    trigger: str  # signal | nudge
    response: str  # yellow | red | none | suppressed_raised
    context: str


class SimWorld:
    def __init__(self, config: dict, *, spec: RobotSpec | None = None, seed: int = 0,
                 consequences: dict | None = None, inspection: bool = True, show_viewer=False):
        import genesis as gs

        if not gs._initialized:
            gs.init(backend=gs.cpu, logging_level="warning", seed=seed)
        self.gs = gs
        self.config = config
        self.spec = spec or RobotSpec.load()
        self.rng = np.random.default_rng(seed)
        self.dt = float(config.get("physics", {}).get("dt", 0.01))
        self.consequences = consequences if consequences is not None else config.get("consequences", {})
        self.events: list[WorldEvent] = []
        self.time = 0.0
        self.frame_index = 0
        self._screen_image = None
        self._screen_node = None
        self.signal_active = False
        self._build(inspection, show_viewer)

    # ------------------------------------------------------------------ build
    def _asset_dir(self) -> Path:
        key = assets.cache_key({k: v for k, v in self.config.items() if not k.startswith("_")})
        return CACHE_DIR / f"{self.config['name']}-{key}"

    def _build(self, inspection, show_viewer):
        gs, cfg = self.gs, self.config
        light = cfg.get("lighting", {})
        lights = [
            gs.options.vis.DirectionalLight(dir=tuple(item["dir"]), color=tuple(item.get("color", (1, 1, 1))),
                                            intensity=float(item["intensity"]))
            for item in light.get("directional", [])
        ] + [
            gs.options.vis.PointLight(pos=tuple(item["pos"]), color=tuple(item.get("color", (1, 1, 1))),
                                      intensity=float(item["intensity"]))
            for item in light.get("points", [])
        ]
        ambient = float(light.get("ambient", 0.3))
        self.scene = gs.Scene(
            show_viewer=show_viewer,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=1),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False),
            vis_options=gs.options.VisOptions(
                ambient_light=(ambient,) * 3, lights=lights, shadow=bool(light.get("shadow", True)),
                background_color=(0.05, 0.05, 0.06),
            ),
        )
        out = self._asset_dir()
        room = cfg["room"]
        lx, ly = room["size"]
        wall_h, wall_t = room.get("wall_height", 1.0), room.get("wall_thickness", 0.1)
        appearance = cfg.get("appearance", {})
        palette = appearance.get("palette", "warm")
        floor_path = out / "floor.glb"
        if not floor_path.exists():
            image = textures.floor_texture(appearance.get("floor", "wood"), appearance.get("floor_seed", 1),
                                           palette=palette)
            assets.export_glb(assets.textured_quad(lx + 2 * wall_t, ly + 2 * wall_t, image), floor_path, y_up=True)
        self.entity_names = {}
        plane = self.scene.add_entity(gs.morphs.Plane(visualization=False))
        self.plane = plane
        self.scene.add_entity(gs.morphs.Mesh(file=str(floor_path), fixed=True, collision=False,
                                             pos=(0, 0, 0.0005)))
        walls = [
            ("wall_n", (lx + 2 * wall_t, wall_t, wall_h), (0, ly / 2 + wall_t / 2)),
            ("wall_s", (lx + 2 * wall_t, wall_t, wall_h), (0, -ly / 2 - wall_t / 2)),
            ("wall_e", (wall_t, ly, wall_h), (lx / 2 + wall_t / 2, 0)),
            ("wall_w", (wall_t, ly, wall_h), (-lx / 2 - wall_t / 2, 0)),
        ]
        self.static_geometry = []
        for index, (name, size, xy) in enumerate(walls):
            path = out / f"{name}.glb"
            if not path.exists():
                image = textures.wall_texture(appearance.get("wall_seed", 2) * 10 + index, palette=palette)
                assets.export_glb(assets.textured_box(size, image), path, y_up=True)
            ent = self.scene.add_entity(gs.morphs.Mesh(file=str(path), fixed=True, pos=(*xy, wall_h / 2),
                                                       convexify=True))
            self.entity_names[ent.idx] = name
            self.static_geometry.append({"name": name, "size": size, "xy": xy, "yaw": 0.0})
        for index, item in enumerate(cfg.get("obstacles", [])):
            path = out / f"obstacle_{item['name']}.glb"
            size = tuple(item["size"])
            if not path.exists():
                image = textures.crate_texture(item.get("texture_seed", 100 + index), palette=palette)
                assets.export_glb(assets.textured_box(size, image), path, y_up=True)
            yaw = float(item.get("yaw", 0.0))
            ent = self.scene.add_entity(gs.morphs.Mesh(
                file=str(path), fixed=True, pos=(*item["pos"], size[2] / 2),
                euler=(0, 0, math.degrees(yaw)), convexify=True))
            self.entity_names[ent.idx] = item["name"]
            self.static_geometry.append({"name": item["name"], "size": size, "xy": tuple(item["pos"]), "yaw": yaw})

        self.evaluation_obstacle = None
        moved = cfg.get("nav_eval", {}).get("moved_obstacle")
        if moved:
            size = tuple(moved["size"])
            path = out / "evaluation_obstacle.glb"
            if not path.exists():
                image = textures.crate_texture(99, palette=palette)
                assets.export_glb(assets.textured_box(size, image), path, y_up=True)
            # Parked outside the room; the evaluator moves it in (environment change).
            self.evaluation_obstacle = self.scene.add_entity(gs.morphs.Mesh(
                file=str(path), fixed=False, pos=(lx + 3.0, 0.0, size[2] / 2), convexify=True),
                material=gs.materials.Rigid(rho=2000.0, friction=1.0))
            self.entity_names[self.evaluation_obstacle.idx] = "evaluation_obstacle"
            self.evaluation_obstacle_size = size

        self.fixtures = {}
        self.fixture_states = {}
        for item in cfg.get("fixtures", []):
            name = item["name"]
            if item["shape"] == "sphere":
                radius = float(item["size"][0]) / 2
                entity = self.scene.add_entity(
                    gs.morphs.Sphere(radius=radius, pos=(*item["pos"], radius + 0.001)),
                    material=gs.materials.Rigid(rho=float(item.get("density", 120.0)), friction=0.6),
                    surface=gs.surfaces.Default(color=assets.PAINTS[item["paint"]][:3]),
                )
            else:
                urdf = out / "fixtures" / f"{name}.urdf"
                if not urdf.exists():
                    assets.fixture_urdf(out / "fixtures", name, item["shape"], item["paint"], item["size"])
                entity = self.scene.add_entity(gs.morphs.URDF(
                    file=str(urdf), fixed=True, pos=(*item["pos"], 0.0),
                    euler=(0, 0, math.degrees(float(item.get("yaw", 0.0)))), merge_fixed_links=False))
            self.fixtures[name] = (item, entity)
            self.entity_names[entity.idx] = name
            self.fixture_states[name] = FixtureState(name)

        spec = self.spec
        start = cfg.get("robot_start", [0.0, 0.0, 0.0])
        self.robot = self.scene.add_entity(gs.morphs.URDF(
            file=str(PROJECT_ROOT / "assets/robot/generated/amr.urdf"),
            pos=(start[0], start[1], spec.base_height + 0.002),
            euler=(0, 0, math.degrees(start[2])),
            merge_fixed_links=True, links_to_keep=["camera_link", "screen_link", "caster_link"],
        ))
        cam = spec.camera
        self.camera = self.scene.add_camera(res=(cam["width"], cam["height"]), fov=cam["vertical_fov"],
                                            GUI=False, near=0.02, far=30.0)
        self.inspection = None
        if inspection:
            self.inspection = self.scene.add_camera(res=(480, 360), fov=55, GUI=False, debug=True,
                                                    pos=(-lx / 2 + 0.2, -ly / 2 + 0.2, 2.2),
                                                    lookat=(0.3, 0.2, 0.0))
        self.scene.build()
        # Mount the camera on the chassis link: OpenCV optical -> Genesis (OpenGL) camera axes.
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
        self.camera.attach(self.robot.get_link("base_link"), spec.camera_to_base() @ cv_to_gl)
        for _name, (item, entity) in self.fixtures.items():
            if item["shape"] != "sphere":
                dofs = [entity.get_joint(f"{c}_lift").dofs_idx_local[0] for c in ("yellow", "red")]
                entity.set_dofs_kp([400.0, 400.0], dofs)
                entity.set_dofs_kv([40.0, 40.0], dofs)
                entity.control_dofs_position([0.0, 0.0], dofs)
                item["_dofs"] = dofs
            else:
                entity.get_link(entity.links[0].name).set_friction_rolling(float(item.get("rolling_friction", 0.01)))
        from ..control.genesis_backend import GenesisWheelBackend

        self.backend = GenesisWheelBackend(self.robot, spec)

    # --------------------------------------------------------------- stepping
    def step(self):
        self.backend.apply(now=self.time)
        self.scene.step()
        self.time = round(self.time + self.dt, 9)
        self._update_fixtures()

    def capture(self, calibration_version: str) -> Frame:
        """Render the robot-mounted camera at its current physical pose."""
        self.camera.move_to_attach()
        rgb = self.camera.render(rgb=True)[0]
        self.frame_index += 1
        return Frame(self.frame_index, self.time, np.ascontiguousarray(rgb), CAMERA_ID, calibration_version)

    def render_inspection(self):
        if self.inspection is None:
            return None
        self._draw_screen()
        return np.ascontiguousarray(self.inspection.render(rgb=True)[0])

    # --------------------------------------------------------------- screen
    def set_screen(self, image: np.ndarray, *, signal_pattern: bool):
        """Robot display output. ``signal_pattern`` is what the display shows, and
        fixtures may react to it (they can 'see' the screen)."""
        self._screen_image = image
        self.signal_active = bool(signal_pattern)

    def _draw_screen(self):
        if self._screen_image is None:
            return
        import trimesh

        if self._screen_node is not None:
            self.scene.clear_debug_object(self._screen_node)
        s = self.spec.screen
        image = np.ascontiguousarray(self._screen_image[:, ::-1])  # quad UV runs toward -y
        quad = assets.textured_quad(s["width"] * 0.9, s["height"] * 0.86, image)
        quad.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (0, 1, 0)))
        quad.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0)))
        link = self.robot.get_link("screen_link")
        pos = np.asarray(link.get_pos()).reshape(3)
        quat = np.asarray(link.get_quat()).reshape(4)
        T = np.eye(4)
        T[:3, :3] = _quat_to_R(quat)
        T[:3, 3] = pos + T[:3, :3] @ np.array([s["depth"] / 2 + 0.002, 0, 0])
        self._screen_node = self.scene.draw_debug_mesh(quad, T=T)

    # --------------------------------------------------------------- fixtures
    def _robot_pose(self):
        pos = np.asarray(self.robot.get_pos()).reshape(3)
        quat = np.asarray(self.robot.get_quat()).reshape(4)
        w, x, y, z = quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return pos, yaw

    def _update_fixtures(self):
        if not self.fixtures:
            return
        self._fixture_tick = getattr(self, "_fixture_tick", 0) + 1
        if self._fixture_tick % 5:
            return  # world rules evaluated at 20 Hz
        pos, yaw = self._robot_pose()
        for name, (item, entity) in self.fixtures.items():
            if item["shape"] == "sphere":
                continue
            state = self.fixture_states[name]
            if state.raised and self.time >= state.lower_at:
                state.raised = None
                entity.control_dofs_position([0.0, 0.0], item["_dofs"])
            fxy = np.asarray(item["pos"], float)
            delta = fxy - pos[:2]
            distance = float(np.linalg.norm(delta))
            bearing = math.atan2(delta[1], delta[0]) - yaw
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            facing = distance < 1.25 and abs(bearing) < math.radians(35)
            if self.signal_active and facing:
                if state.signal_since is None:
                    state.signal_since = self.time
                if not state.signal_consumed and self.time - state.signal_since >= 1.0:
                    state.signal_consumed = True
                    self._respond(name, item, entity, "signal")
            else:
                state.signal_since, state.signal_consumed = None, False
            contact = distance < 0.6 and self._in_contact(entity)
            if contact and not state.contact_active and self.time - state.last_contact > 2.0:
                self._respond(name, item, entity, "nudge")
            if contact:
                state.last_contact = self.time
            state.contact_active = contact

    def _in_contact(self, entity) -> bool:
        try:
            contacts = entity.get_contacts(with_entity=self.robot)
        except Exception:  # noqa: BLE001 - Genesis contact query shape differs by version
            return False
        return bool(len(contacts.get("geom_a", [])))

    def _respond(self, name, item, entity, trigger):
        state = self.fixture_states[name]
        context = "raised" if state.raised else "lowered"
        rule = self.consequences.get(name, {}).get(trigger, {"response": "none", "p": 1.0})
        response = rule.get("response", "none")
        if response != "none" and self.rng.random() > float(rule.get("p", 1.0)):
            response = "none"
        if state.raised:
            response = "suppressed_raised" if response != "none" else "none"
        elif response in ("yellow", "red"):
            state.raised = response
            state.lower_at = self.time + float(rule.get("hold", 8.0))
            travel = 0.12 + 0.02
            target = [travel, 0.0] if response == "yellow" else [0.0, travel]
            entity.control_dofs_position(target, item["_dofs"])
        self.events.append(WorldEvent(self.time, name, trigger, response, context))

    def place_evaluation_obstacle(self, xy):
        """EVALUATION ONLY: the environment changes (someone puts a box down)."""
        size = self.evaluation_obstacle_size
        self.evaluation_obstacle.set_pos([float(xy[0]), float(xy[1]), size[2] / 2 + 0.002], zero_velocity=True)
        self.static_geometry.append({"name": "evaluation_obstacle", "size": size, "xy": tuple(xy), "yaw": 0.0})

    def close(self):
        pass


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
