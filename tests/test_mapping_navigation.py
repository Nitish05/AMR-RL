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


def test_goal_that_loses_clearance_while_driving_snaps_to_nearby_certified_cell():
    """Regression (navigation-20260930-024431): new evidence near an accepted goal
    removed its footprint clearance and the robot stopped ('blocked'). A replan may
    now move the goal to the nearest certified cell within 12 cm; farther: blocked."""
    grid = grid_with_free([(-2.0, -1.2, 2.0, 1.2)])
    nav = Navigator()
    pose = np.array([-1.0, 0.0, 0.0])
    goal = np.array([0.8, 0.0])
    assert nav.set_goal(grid, pose, goal, now=0.0)
    _box_hits(grid, (0.8, 0.33), 0.05, times=3)  # obstacle evidence 0.33 m from the goal
    assert not nav.planner.traversable_xy(grid, [goal])[0]
    v, w = nav.step(grid, pose, 0.01, 2.0)  # replan period elapsed
    assert nav.status == "following" and nav.goal_snaps == 1
    assert 0 < np.linalg.norm(nav.goal - goal) <= nav.planner.cfg.goal_snap + 1e-9
    assert nav.planner.traversable_xy(grid, [nav.goal])[0]
    # evidence right at the goal: no certified cell within 12 cm -> blocked, not moved further
    grid2 = grid_with_free([(-2.0, -1.2, 2.0, 1.2)])
    nav2 = Navigator()
    assert nav2.set_goal(grid2, pose, goal, now=0.0)
    _box_hits(grid2, (0.8, 0.0), 0.1, times=3)
    nav2.step(grid2, pose, 0.01, 2.0)
    assert nav2.status == "blocked"


def test_new_goals_are_still_checked_strictly():
    grid = grid_with_free([(-2.0, -1.2, 2.0, 1.2)])
    _box_hits(grid, (0.8, 0.33), 0.05, times=3)
    nav = Navigator()
    assert not nav.set_goal(grid, np.array([-1.0, 0.0, 0.0]), (0.8, 0.0))
    assert nav.reason == "goal_lacks_footprint_clearance"


def test_obstacle_bases_need_agreement_between_keyframe_pairs():
    """Regression: single-pair obstacle-base hits were 91 % on open floor
    (scripts/dev/obstacle_hits_probe.py); only bases seen by >= 2 pairs are applied."""
    from amr_rl.mapping.occupancy import FloorEvidenceMapper
    from amr_rl.perception.camera_model import CameraModel
    from amr_rl.robot.spec import RobotSpec

    grid = OccupancyGrid("m")
    mapper = FloorEvidenceMapper(CameraModel.from_spec(RobotSpec.load()), grid)
    real = np.array([[1.0, 0.0], [1.0, 0.05]])
    results = [{"updated": True, "base_xy": real + [0.02, 0.0]},  # same obstacle from two pairs
               {"updated": True, "base_xy": np.vstack([real, [[0.4, -0.6]]])},  # plus a one-pair stray
               {"updated": False}]
    mapper._commit_confirmed_bases(results)
    ix, iy = grid.to_cell(np.array([[1.0, 0.0], [0.4, -0.6]]))
    assert grid.logodds[iy[0], ix[0]] > 0 and grid.logodds[iy[1], ix[1]] == 0
    assert mapper.stats["bases_confirmed"] >= 2 and mapper.stats["bases_unconfirmed"] >= 1


def test_remembered_entities_are_hard_keepout_for_planning():
    """Regression (map-arena-20260930-082022, t=263 s): floor evidence had eroded a
    fixture's weak obstacle ring and the robot clipped it while turning past."""
    grid = grid_with_free([(-2.0, -1.2, 2.0, 1.2)])
    planner = Planner()
    assert planner.traversable_xy(grid, [[0.0, 0.3]])[0]
    assert grid.set_keepout([(0.0, 0.0, 0.18)]) and not grid.set_keepout([(0.0, 0.0, 0.18)])
    ix, iy = grid.to_cell(np.array([[0.0, 0.0]]))
    assert grid.classes()[iy[0], ix[0]] == OCCUPIED
    # centre within radius + footprint + margin of the disc is no longer traversable
    assert not planner.traversable_xy(grid, [[0.0, 0.3]])[0]
    path = planner.plan(grid, (-1.2, 0.0), (1.2, 0.0))
    dense = np.vstack([np.linspace(a, b, 20) for a, b in zip(path[:-1], path[1:])])
    assert np.min(np.linalg.norm(dense, axis=1)) >= 0.18 + 0.262
    assert grid.set_keepout([])  # entity forgotten/moved: space is free again
    assert planner.traversable_xy(grid, [[0.0, 0.3]])[0]


class _FakeMapper:
    """Stands in for FloorEvidenceMapper.update: returns preset floor/base points."""

    def __init__(self):
        self.base_xy = np.zeros((0, 2))
        self.free_xy = np.zeros((0, 2))
        self.calls = []

    def update(self, ref, cur, *, defer_bases=False, commit_free=True):
        self.calls.append((defer_bases, commit_free))
        return {"updated": True, "base_xy": self.base_xy, "free_xy": self.free_xy}


def _kf(t, pose):
    from amr_rl.perception.vslam import Keyframe

    return Keyframe(0, t, np.asarray(pose, float), np.zeros((0, 2)), np.zeros((0, 32), np.uint8),
                    np.zeros(0, int), np.zeros((4, 4), np.uint8))


def _frame():
    return np.zeros((4, 4), np.uint8), np.zeros((0, 2)), np.zeros((0, 32), np.uint8)


def _guarded_nav(grid):
    from amr_rl.navigation.near_field import NearFieldGuard

    mapper = _FakeMapper()
    guard = NearFieldGuard(mapper, grid)
    nav = Navigator()
    nav.guard = guard
    return nav, guard, mapper


def _floor(x0, x1):
    xs = np.arange(x0, x1, 0.025)
    return np.array([(x, y) for x in xs for y in np.arange(-0.15, 0.16, 0.025)])


def test_guard_creeps_until_the_band_ahead_is_freshly_seen_as_floor():
    grid = grid_with_free([(-2.0, -0.5, 2.0, 0.5)])
    nav, guard, mapper = _guarded_nav(grid)
    pose = np.array([-1.0, 0.0, 0.0])
    assert nav.set_goal(grid, pose, (1.2, 0.0), now=0.0)
    v, _ = nav.step(grid, pose, 0.01, 0.1)
    assert v == guard.cfg.creep_speed  # nothing verified since the map was made: creep
    mapper.free_xy = _floor(-0.8, -0.3)
    guard.maybe_probe(0.2, pose, [_kf(0.0, (-1.2, 0.0, 0.0))], _frame())
    assert guard.last_fraction > 0.9 and mapper.calls[-1] == (True, False)  # never writes the map
    v, _ = nav.step(grid, pose, 0.01, 0.3)
    assert v > guard.cfg.creep_speed  # floor just seen ahead: full speed
    v, _ = nav.step(grid, pose, 0.01, 0.2 + guard.cfg.verify_window + 0.2)
    assert v == guard.cfg.creep_speed  # the look went stale


def test_guard_holds_then_blocks_when_the_floor_ahead_cannot_be_seen():
    """A box on the route hides the floor behind its face: the band cannot be verified
    even though the old map says free, so the robot stops and the goal ends blocked."""
    grid = grid_with_free([(-2.0, -0.5, 2.0, 0.5)])
    nav, guard, mapper = _guarded_nav(grid)
    pose = np.array([-1.0, 0.0, 0.0])
    assert nav.set_goal(grid, pose, (1.2, 0.0), now=0.0)
    nav.step(grid, pose, 0.01, 0.1)  # sets the band
    mapper.free_xy = _floor(-0.8, -0.72)  # only the near edge of the band is floor
    kfs = [_kf(0.0, (-1.2, 0.0, 0.0))]
    guard.maybe_probe(0.5, pose, kfs, _frame())
    v, _ = nav.step(grid, pose, 0.01, 0.55)
    assert v == guard.cfg.creep_speed and not nav.holding  # one low probe: creep
    guard.maybe_probe(0.85, pose, kfs, _frame())
    assert nav.step(grid, pose, 0.01, 0.9) == (0.0, 0.0) and nav.holding
    nav.step(grid, pose, 0.01, 0.9 + guard.cfg.hold_timeout + 0.1)
    assert nav.status == "blocked" and nav.reason == "path_ahead_not_verified"


def test_guard_skips_probes_without_baseline_and_rejected_variant_is_off():
    from amr_rl.navigation.near_field import GuardConfig, NearFieldGuard

    grid = grid_with_free([(-2.0, -0.5, 2.0, 0.5)])
    guard = NearFieldGuard(_FakeMapper(), grid)
    guard.maybe_probe(1.0, np.array([-0.6, 0.0, 0.0]), [_kf(0.0, (-0.61, 0.0, 0.0))], _frame())
    assert guard.stats["probes"] == 0 and guard.stats["skipped"] == 1  # 1 cm: no parallax
    assert GuardConfig().assert_obstacles is False


def test_nudge_creep_uses_the_measured_contact_edge():
    from amr_rl.behavior.activities import FRONT_EXTENT, MAX_CREEP, PUSH_DEPTH, nudge_creep

    # A box seen corner-on was taken for round: its "centre" is its front edge. The old
    # estimate (centre - radius) stopped ~10 cm short; the contact edge does not.
    assert nudge_creep(0.40, 0.40, 0.11) == pytest.approx(0.40 - FRONT_EXTENT + PUSH_DEPTH)
    assert nudge_creep(None, 0.40, 0.11) == pytest.approx(0.40 - 0.11 - FRONT_EXTENT + PUSH_DEPTH)
    assert nudge_creep(0.10, 0.2, 0.1) == 0.0 and nudge_creep(2.0, 2.0, 0.1) == MAX_CREEP
