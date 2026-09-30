"""Occupancy semantics and conservative planning (no simulator)."""

import math

import numpy as np
import pytest

from amr_rl.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyGrid
from amr_rl.navigation.navigator import Navigator
from amr_rl.navigation.planner import GoalRejected, Planner, PlannerConfig


def grid_with_free(rects, map_version="m"):
    grid = OccupancyGrid(map_version)
    for x0, y0, x1, y1 in rects:
        xs = np.arange(x0, x1, grid.cfg.resolution / 2)
        ys = np.arange(y0, y1, grid.cfg.resolution / 2)
        pts = np.array([(x, y) for x in xs for y in ys])
        for _ in range(4):
            grid.add_hits(pts, grid.cfg.free_hit)
    return grid


def test_unobserved_space_is_unknown_and_evidence_is_required():
    grid = OccupancyGrid("m")
    assert grid.counts()["free"] == 0 and grid.classify_xy(np.array([[0.0, 0.0]]))[0] == UNKNOWN
    grid.add_hits(np.array([[0.0, 0.0]]), grid.cfg.free_hit)
    assert grid.classify_xy(np.array([[0.0, 0.0]]))[0] == UNKNOWN  # one weak observation is not enough
    for _ in range(3):
        grid.add_hits(np.array([[0.0, 0.0]]), grid.cfg.free_hit)
    assert grid.classify_xy(np.array([[0.0, 0.0]]))[0] == FREE
    for _ in range(3):
        grid.add_hits(np.array([[0.0, 0.0]]), grid.cfg.occ_hit)
    assert grid.classify_xy(np.array([[0.0, 0.0]]))[0] == UNKNOWN  # conflicting evidence is not free
    grid.add_hits(np.array([[0.0, 0.0]]), grid.cfg.occ_hit)
    assert grid.classify_xy(np.array([[0.0, 0.0]]))[0] == OCCUPIED


def test_min_count_filters_sparse_pixels():
    grid = OccupancyGrid("m")
    assert grid.add_hits(np.array([[0.01, 0.01], [1.0, 1.0]]), -1.0, min_count=2) == 0


def test_planner_never_routes_through_unknown():
    # two free rooms joined by nothing: unreachable goal
    grid = grid_with_free([(-1.5, -1.0, 0.0, 1.0), (0.5, -1.0, 2.0, 1.0)])
    planner = Planner(PlannerConfig(footprint_radius=0.26))
    with pytest.raises(GoalRejected) as err:
        planner.plan(grid, (-0.8, 0.0), (1.3, 0.0))
    assert err.value.reason == "goal_unreachable_through_certified_space"


@pytest.mark.parametrize("goal,reason", [
    ((5.0, 5.0), "goal_in_unknown_space"),
    ((100.0, 0.0), "goal_outside_map"),
    ((-1.45, 0.0), "goal_lacks_footprint_clearance"),
])
def test_goal_rejection_reasons(goal, reason):
    grid = grid_with_free([(-1.5, -1.0, 1.5, 1.0)])
    with pytest.raises(GoalRejected) as err:
        Planner().check_goal(grid, goal)
    assert err.value.reason == reason


def test_occupied_goal_rejected():
    grid = grid_with_free([(-1.5, -1.0, 1.5, 1.0)])
    for _ in range(6):
        grid.add_hits(np.array([[0.5, 0.0]]), grid.cfg.occ_hit)
    with pytest.raises(GoalRejected) as err:
        Planner().check_goal(grid, (0.5, 0.0))
    assert err.value.reason == "goal_occupied"


def test_path_keeps_footprint_clearance_and_avoids_obstacle():
    grid = grid_with_free([(-2.0, -1.2, 2.0, 1.2)])
    for _ in range(6):
        grid.add_hits(np.array([[0.0, y] for y in np.arange(-1.2, 0.6, 0.025)]), grid.cfg.occ_hit)
    planner = Planner()
    path = planner.plan(grid, (-1.2, 0.0), (1.2, 0.0))
    dense = np.vstack([np.linspace(a, b, 20) for a, b in zip(path[:-1], path[1:])])
    assert planner.traversable_xy(grid, dense).all()
    assert planner.clearance_xy(grid, dense).min() >= 0.26


def test_navigator_stops_on_uncertainty_and_reports_arrival():
    grid = grid_with_free([(-2.0, -1.2, 2.0, 1.2)])
    nav = Navigator()
    assert nav.set_goal(grid, np.array([-1.0, 0.0, 0.0]), (0.5, 0.0))
    v, w = nav.step(grid, np.array([-1.0, 0.0, 0.0]), 0.2, 1.0)
    assert (v, w) == (0.0, 0.0) and nav.status == "localization_uncertain"
    nav.set_goal(grid, np.array([-1.0, 0.0, 0.0]), (0.5, 0.0))
    v, w = nav.step(grid, np.array([-1.0, 0.0, 0.0]), 0.01, 1.0)
    assert v > 0
    nav.step(grid, np.array([0.48, 0.0, 0.0]), 0.01, 2.0)
    assert nav.status == "arrived"


def test_navigator_rejection_counts():
    grid = grid_with_free([(-1.0, -1.0, 1.0, 1.0)])
    nav = Navigator()
    assert not nav.set_goal(grid, np.array([0.0, 0.0, 0.0]), (4.0, 0.0))
    assert nav.status == "rejected" and nav.rejections == 1


def test_grid_save_load_is_map_version_scoped(tmp_path):
    grid = grid_with_free([(-1.0, -1.0, 1.0, 1.0)], "map-a")
    grid.save(tmp_path / "g.npz")
    assert OccupancyGrid.load(tmp_path / "g.npz", "map-a").counts() == grid.counts()
    with pytest.raises(ValueError):
        OccupancyGrid.load(tmp_path / "g.npz", "map-b")


def test_ring_frontier_yields_goals_on_all_sides_not_its_centroid():
    """Regression: a frontier ringing the whole known area was one component whose
    centroid lay in free space; exhausting that single key stopped exploration."""
    from amr_rl.behavior.frontier import frontier_goals

    grid = grid_with_free([(-1.2, -1.2, 1.2, 1.2)])
    planner = Planner()
    goals = frontier_goals(grid, planner, (0.0, 0.0, 0.0))
    assert len(goals) >= 4
    keys = {g[3] for g in goals}
    assert len(keys) == len(goals)
    headings = np.array([g[1] for g in goals])
    # headings face outward in several different directions
    assert np.ptp(np.unwrap(np.sort(headings))) > np.pi / 2
    again = frontier_goals(grid, planner, (0.0, 0.0, 0.0), exhausted={goals[0][3]})
    assert goals[0][3] not in {g[3] for g in again} and len(again) >= 3


def _box_hits(grid, centre, half, times=1):
    xs = np.arange(centre[0] - half, centre[0] + half + 1e-9, 0.025)
    ys = np.arange(centre[1] - half, centre[1] + half + 1e-9, 0.025)
    pts = np.array([(x, y) for x in xs for y in ys])
    for _ in range(times):
        grid.add_hits(pts, grid.cfg.occ_hit)


def test_saturated_free_floor_needs_three_obstacle_observations():
    """Documented trade-off: one or two stray hits do not erase well-observed floor;
    a real object observed three times does (see docs/VSLAM.md)."""
    grid = grid_with_free([(-2.0, -1.2, 2.0, 1.2)])
    for _ in range(20):
        grid.add_hits(np.array([[0.5, 0.0]]), grid.cfg.free_hit)
    ix, iy = grid.to_cell([0.5, 0.0])
    for expected in (FREE, FREE):
        _box_hits(grid, (0.5, 0.0), 0.15, times=1)
        assert grid.classes()[iy[0], ix[0]] == expected
    _box_hits(grid, (0.5, 0.0), 0.15, times=1)
    assert grid.classes()[iy[0], ix[0]] != FREE


def test_navigator_stops_immediately_when_the_corridor_ahead_is_decertified():
    grid = grid_with_free([(-2.0, -0.5, 2.0, 0.5)])  # a corridor: no way around
    nav = Navigator()
    pose = np.array([-1.0, 0.0, 0.0])
    assert nav.set_goal(grid, pose, (1.2, 0.0), now=0.0)
    assert nav.step(grid, pose, 0.01, 0.1)[0] > 0
    _box_hits(grid, (-0.45, 0.0), 0.15, times=3)  # appears 0.55 m ahead, well before the replan period
    v, w = nav.step(grid, pose, 0.01, 0.2)
    assert (v, w) == (0.0, 0.0) and nav.holding  # stops at once
    for k in range(3, 40):  # replans at most once per period; no route around -> blocked
        nav.step(grid, pose, 0.01, 0.1 * k)
    assert nav.status == "blocked" and nav.replans == 1


def test_navigator_backs_off_instead_of_turning_next_to_a_known_obstacle():
    grid = grid_with_free([(-2.0, -1.2, 2.0, 1.2)])
    _box_hits(grid, (0.0, 0.0), 0.15, times=3)  # box face 0.25 m ahead of the robot centre
    nav = Navigator()
    pose = np.array([-0.4, 0.0, 0.0])
    nav.goal, nav.goal_heading, nav.status = np.array([-0.4, 0.0]), math.pi, "aligning"
    v, w = nav.step(grid, pose, 0.01, 1.0)
    assert v < 0 and w == 0.0  # reverse, do not swing the corners into the box
    far = np.array([-0.9, 0.0, 0.0])
    v, w = nav.step(grid, far, 0.01, 2.0)
    assert v == 0.0 and w != 0.0  # enough room: turn in place


def test_start_inside_clearance_band_is_not_reported_blocked():
    """Regression: the corridor check sampled the robot->carrot chord through the
    start clearance band and declared freshly planned paths blocked."""
    grid = grid_with_free([(-2.0, -0.35, 2.0, 1.2)])  # unknown below y = -0.35
    planner = Planner()
    pose = np.array([-1.0, -0.12, 0.0])  # beside the boundary, inside the band
    assert not planner.traversable_xy(grid, [pose[:2]])[0]
    nav = Navigator(planner)
    assert nav.set_goal(grid, pose, (1.0, -0.05), now=0.0)
    for k in range(1, 8):
        v, w = nav.step(grid, pose, 0.01, 0.1 * k)
    assert nav.status == "following" and v > 0


def test_frontier_goals_move_away_from_spots_already_swept():
    """Regression: every frontier tile mapped to nearly the same traversable cell
    beside the robot; repeated in-place sweeps add no translational parallax."""
    from amr_rl.behavior.frontier import frontier_goals

    grid = grid_with_free([(-0.8, -0.8, 0.8, 0.8)])
    planner = Planner()
    here = (0.0, 0.0, 0.0)
    fresh = frontier_goals(grid, planner, here)
    spots = [np.asarray(g[0]) for g in fresh]
    again = frontier_goals(grid, planner, here, visited=spots)
    moved = [min(np.linalg.norm(g[0] - s) for s in spots) for g in again]
    assert np.mean(moved) > 0.05
