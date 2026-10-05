"""Perception-aware turning: view check on the robot's own grid (no simulator)."""

import math

import numpy as np

from amr_rl.behavior.activities import Explore, Survey
from amr_rl.mapping.occupancy import OccupancyGrid
from amr_rl.navigation.navigator import Navigator
from amr_rl.navigation.near_field import DepthGuard
from amr_rl.navigation.view_check import (
    ViewCheckConfig,
    facing_distances,
    plan_sweep,
    plan_turn,
    sector_clear,
)

SWEEP = [math.radians(50), -math.radians(100), math.radians(50)]


def _rect_pts(x0, y0, x1, y1, step=0.025):
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    return np.array([(x, y) for x in xs for y in ys]).reshape(-1, 2)


def make_grid(free=(-2.0, -2.0, 2.0, 2.0), walls=()):
    grid = OccupancyGrid("m")
    for _ in range(4):
        grid.add_hits(_rect_pts(*free), grid.cfg.free_hit)
    for rect in walls:
        for _ in range(4):
            grid.add_hits(_rect_pts(*rect), grid.cfg.occ_hit)
    return grid


LEFT_WALL = (-2.0, 0.4, 2.0, 0.5)    # wall face 0.4 m to the left of the robot centre
FRONT_WALL = (0.5, -1.0, 0.6, 1.0)   # wall face 0.5 m ahead (0.375 m from the camera)


class FakeRuntime:
    def __init__(self, grid, pose, nav):
        self.grid, self.pose, self.nav, self.planner = grid, np.asarray(pose, float), nav, nav.planner
        self.panoramas = 0
        self.visits = []
        self.panorama_wanted = False
        self.sigma = 0.01

    def wants_panorama(self):
        return self.panorama_wanted

    def note_panorama(self):
        self.panoramas += 1

    def note_frontier_visit(self, key):
        self.visits.append(key)

    def note_map_progress(self, n):
        pass


def run_turning(act, rt, dt=0.05, max_steps=2000):
    """Integrate commanded yaw rates (v is ignored: in-place activities)."""
    start = float(rt.pose[2])
    unwrapped = start
    trace, commands = [], []
    for k in range(max_steps):
        v, w = act.step(rt, k * dt)
        commands.append((v, w))
        if act.done:
            break
        unwrapped += w * dt
        rt.pose = np.array([rt.pose[0], rt.pose[1], math.atan2(math.sin(unwrapped), math.cos(unwrapped))])
        trace.append(unwrapped - start)
    return np.array(trace), commands


def arrived_explore(rt):
    act = Explore(goal=rt.pose[:2], heading=None, gain=10, key=("frontier", 1))
    act.goto.started = True
    rt.nav.status = "arrived"
    return act


# ------------------------------------------------------------------ geometry
def test_facing_distance_on_a_synthetic_grid():
    grid = make_grid(free=(-0.6, -0.6, 0.5, 2.0), walls=[FRONT_WALL])
    pose = np.array([0.0, 0.0, 0.0])
    res = facing_distances(grid, pose, [0.0, math.pi / 2, math.pi])
    assert abs(res.distance[0] - (0.5 - 0.125)) <= grid.cfg.resolution  # measured from the camera
    assert res.source[0] == "grid" and math.isinf(res.unknown[0])
    assert math.isinf(res.distance[1])  # open floor up to max_range
    assert math.isinf(res.distance[2]) and abs(res.unknown[2] - (0.6 - 0.125)) <= grid.cfg.resolution  # unknown flagged
    # 45 deg: the camera moves sideways too; the ray meets the wall at (0.5 - 0.125 cos h) / cos h
    d45 = facing_distances(grid, pose, [math.pi / 4]).distance[0]
    assert abs(d45 - (0.5 - 0.125 * math.cos(math.pi / 4)) / math.cos(math.pi / 4)) <= 1.5 * grid.cfg.resolution


def test_extra_points_count_when_the_grid_is_still_empty():
    grid = OccupancyGrid("m")  # t = 0: nothing mapped
    pts = _rect_pts(0.4, -0.3, 0.45, 0.3, 0.02)  # a face 0.4 m ahead seen by the depth guard
    res = facing_distances(grid, np.zeros(3), [0.0, math.pi / 2], points=pts)
    assert res.source[0] == "points" and abs(res.distance[0] - 0.275) < 0.03
    assert math.isinf(res.distance[1])
    assert facing_distances(grid, np.zeros(3), [0.0]).unknown[0] < 0.05  # empty grid: all unknown, not near
    assert math.isinf(facing_distances(grid, np.zeros(3), [0.0]).distance[0])


def test_sector_fraction():
    grid = make_grid(walls=[FRONT_WALL])
    pose = np.zeros(3)
    assert sector_clear(grid, pose, 0.0, math.pi / 2) > 0.3  # the first ~47 deg face the wall
    assert sector_clear(grid, pose, math.pi / 2, math.pi) == 0.0
    open_grid = make_grid()
    assert sector_clear(open_grid, pose, 0.0, 2 * math.pi) == 0.0


# ------------------------------------------------------------------ explore look-around
def test_wall_to_one_side_shortens_the_sweep_on_that_side():
    grid = make_grid(walls=[LEFT_WALL])
    planned = plan_sweep(grid, np.zeros(3), SWEEP)
    assert 0 < planned[0] < math.radians(35) and planned[-1] == SWEEP[-1]
    rt = FakeRuntime(grid, [0.0, 0.0, 0.0], Navigator(view_aware_turns=True))
    act = arrived_explore(rt)
    trace, _ = run_turning(act, rt)
    assert act.done and act.result["view_adjusted"] == "sweep_shortened"
    assert trace.max() < math.radians(36)                          # never swung to face the wall
    assert trace.min() < -math.radians(47)                         # the open side is still swept
    assert abs(trace[-1]) < math.radians(4)                        # back to the centre heading


def test_near_box_front_left_reverses_the_sweep():
    grid = make_grid(walls=[(0.35, 0.05, 0.55, 0.4)])  # box face 0.35 m ahead, left of centre
    planned = plan_sweep(grid, np.zeros(3), SWEEP)
    assert planned[0] < 0 and len(planned) == 2  # left side dropped: look right only, then back
    rt = FakeRuntime(grid, [0.0, 0.0, 0.0], Navigator(view_aware_turns=True))
    trace, _ = run_turning(arrived_explore(rt), rt)
    assert trace.max() <= 1e-9 and trace.min() < -math.radians(30)


def test_open_floor_leaves_the_sweep_unchanged():
    grid = make_grid()
    assert plan_sweep(grid, np.zeros(3), SWEEP) == SWEEP
    on = FakeRuntime(grid, [0.0, 0.0, 0.3], Navigator(view_aware_turns=True))
    off = FakeRuntime(grid, [0.0, 0.0, 0.3], Navigator())
    a_on, a_off = arrived_explore(on), arrived_explore(off)
    t_on, c_on = run_turning(a_on, on)
    t_off, c_off = run_turning(a_off, off)
    assert c_on == c_off and a_on.result == a_off.result == {"status": "completed"}


def test_panorama_is_skipped_when_much_of_it_faces_near():
    grid = make_grid(walls=[LEFT_WALL, (-2.0, -0.5, 2.0, -0.4), FRONT_WALL])  # a dead-end alcove
    rt = FakeRuntime(grid, [0.0, 0.0, 0.0], Navigator(view_aware_turns=True))
    rt.panorama_wanted = True
    act = arrived_explore(rt)
    act.step(rt, 0.0)
    assert rt.panoramas == 0 and act._sweep != [2 * math.pi]
    assert act.view_adjusted.startswith("panorama_skipped")
    # open floor: the panorama is kept
    rt2 = FakeRuntime(make_grid(), [0.0, 0.0, 0.0], Navigator(view_aware_turns=True))
    rt2.panorama_wanted = True
    act2 = arrived_explore(rt2)
    act2.step(rt2, 0.0)
    assert rt2.panoramas == 1 and act2._sweep == [2 * math.pi]


def test_defaults_off_leave_the_sweep_and_panorama_unchanged_next_to_a_wall():
    grid = make_grid(walls=[LEFT_WALL])
    rt = FakeRuntime(grid, [0.0, 0.0, 0.0], Navigator())
    act = arrived_explore(rt)
    trace, _ = run_turning(act, rt)
    assert trace.max() > math.radians(49) and act.result == {"status": "completed"}
    rt.panorama_wanted = True
    rt.nav.status = "arrived"
    act2 = arrived_explore(rt)
    act2.step(rt, 0.0)
    assert act2._sweep == [2 * math.pi] and rt.panoramas == 1


# ------------------------------------------------------------------ initial survey (empty grid)
class _StaticGuard:
    def __init__(self, pts):
        self.pts = pts

    def view_points(self, now=None):
        return self.pts


def test_initial_survey_avoids_a_near_wall_seen_only_by_the_depth_guard():
    grid = OccupancyGrid("m")
    nav = Navigator(view_aware_turns=True)
    nav.guard = _StaticGuard(_rect_pts(-2.0, 0.4, 2.0, 0.45, 0.02))  # wall 0.4 m to the left
    rt = FakeRuntime(grid, [0.0, 0.0, 0.0], nav)
    act = Survey(angle=2 * math.pi, rate=0.45)
    trace, _ = run_turning(act, rt)
    assert act.done and act.result["view_limited"]
    assert trace.max() < math.radians(33)                    # turned back before facing the wall
    assert trace.min() > -math.radians(215)                  # stopped before it from the other side
    assert act.result["covered_deg"] > 150
    # off: the full turn, straight across the wall
    rt_off = FakeRuntime(grid, [0.0, 0.0, 0.0], Navigator())
    trace_off, _ = run_turning(Survey(angle=2 * math.pi, rate=0.45), rt_off)
    assert trace_off.max() > math.radians(355)


def test_survey_on_open_floor_is_a_full_turn():
    rt = FakeRuntime(make_grid(), [0.0, 0.0, 0.0], Navigator(view_aware_turns=True))
    act = Survey(angle=2 * math.pi, rate=0.45)
    trace, _ = run_turning(act, rt)
    assert act.result == {"status": "completed"} and trace.max() >= 2 * math.pi - 0.05


def test_depth_guard_keeps_world_points_without_touching_the_grid():
    class Det:
        def detect(self, rgb):
            return {"points": _rect_pts(0.4, -0.2, 0.45, 0.2, 0.02)}

    grid = make_grid()
    before = grid.logodds.copy()
    guard = DepthGuard(Det(), grid, 0.262)
    pose = np.array([1.0, 0.0, math.pi / 2])
    out = guard.maybe_detect(0.0, pose, np.zeros((2, 2, 3), np.uint8), record_only=True)
    assert out["free_run"] is None and guard.free_run is None
    assert np.array_equal(grid.logodds, before)
    pts = guard.view_points(0.1)
    assert len(pts) and np.allclose(pts[:, 0].mean(), 1.0, atol=0.02) and pts[:, 1].min() > 0.39
    assert len(guard.view_points(10.0)) == 0  # forgotten after view_memory


# ------------------------------------------------------------------ navigator
def _aligning(nav, pose, heading):
    nav.goal, nav.goal_heading, nav.status = np.asarray(pose[:2], float), heading, "aligning"


def test_navigator_backs_off_before_turning_away_from_a_near_wall():
    grid = make_grid(free=(-2.0, -1.0, 0.5, 1.0), walls=[FRONT_WALL])
    pose = np.array([0.0, 0.0, 0.0])
    nav = Navigator(view_aware_turns=True)
    _aligning(nav, pose, 2.0)
    v, w = nav.step(grid, pose, 0.01, 0.0)
    assert v < 0 and w == 0.0 and nav._view_turn["mode"] == "back_off"
    assert abs(nav._view_turn["value"] - 0.3) < 1e-9  # smallest candidate that clears the sweep
    mid = np.array([-0.15, 0.0, 0.0])
    assert nav.step(grid, mid, 0.01, 2.0)[0] < 0      # keeps reversing
    back = np.array([-0.3, 0.0, 0.0])
    v, w = nav.step(grid, back, 0.01, 5.0)
    assert v == 0.0 and w > 0 and abs(nav.backed_off - 0.3) < 1e-9  # then turns, budget charged
    assert nav.snapshot()["view_aware_turns"]["back_offs"] == 1


def test_navigator_back_off_respects_the_budget_and_certified_space():
    grid = make_grid(free=(-2.0, -1.0, 0.5, 1.0), walls=[FRONT_WALL])
    nav = Navigator(view_aware_turns=True)
    nav.backed_off = 0.35  # earlier back-offs used most of the budget
    _aligning(nav, np.zeros(3), 2.0)
    v, w = nav.step(grid, np.zeros(3), 0.01, 0.0)
    assert v == 0.0 and w > 0 and nav.view_stats["unavoidable"] == 1
    # unknown behind the robot: never reverse into it
    grid2 = make_grid(free=(-0.1, -1.0, 0.5, 1.0), walls=[FRONT_WALL])
    nav2 = Navigator(view_aware_turns=True)
    assert plan_turn(grid2, nav2.planner, np.zeros(3), 2.0, budget=0.4)[0] == "turn"


def test_navigator_rotate_branch_backs_off_before_turning_toward_the_path():
    grid = make_grid(free=(-2.0, -1.5, 0.5, 1.5), walls=[FRONT_WALL])
    pose = np.array([0.0, 0.0, 0.0])
    nav = Navigator(view_aware_turns=True)
    assert nav.set_goal(grid, pose, (-1.0, 1.0), now=0.0)
    v, w = nav.step(grid, pose, 0.01, 0.1)
    assert v < 0 and w == 0.0
    off = Navigator()
    assert off.set_goal(grid, pose, (-1.0, 1.0), now=0.0)
    v, w = off.step(grid, pose, 0.01, 0.1)
    assert v == 0.0 and w > 0  # default: turns in place at once


def test_navigator_turns_the_long_way_only_near_pi():
    grid = make_grid(walls=[LEFT_WALL])
    nav = Navigator(view_aware_turns=True)
    pose = np.zeros(3)
    _aligning(nav, pose, math.pi - 0.1)  # short way (left) sweeps the wall; the right is open
    v, w = nav.step(grid, pose, 0.01, 0.0)
    assert v == 0.0 and w < 0 and nav._view_turn["mode"] == "long"
    # past pi the error changes sign: from then on the turn is the ordinary one
    v, w = nav.step(grid, np.array([0.0, 0.0, -math.pi + 0.05]), 0.01, 1.0)
    assert w < 0 and nav._view_turn["mode"] == "turn"
    # a 90 deg turn toward the wall is not turned the long way
    nav2 = Navigator(view_aware_turns=True)
    _aligning(nav2, pose, 1.6)
    v, w = nav2.step(grid, pose, 0.01, 0.0)
    assert nav2._view_turn["mode"] != "long"


def test_navigator_defaults_off_turn_in_place_as_before():
    grid = make_grid(free=(-2.0, -1.0, 0.5, 1.0), walls=[FRONT_WALL])
    nav = Navigator()
    assert nav.view_aware_turns is False
    _aligning(nav, np.zeros(3), 2.0)
    assert nav.step(grid, np.zeros(3), 0.01, 0.0) == (0.0, 0.6)
    assert "view_aware_turns" not in nav.snapshot()
    assert ViewCheckConfig().near == 0.6
