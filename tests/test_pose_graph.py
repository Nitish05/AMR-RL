"""Planar pose graph and loop-check geometry (no simulator)."""

import math

import numpy as np

from amr_rl.perception.pose_graph import (
    PoseGraph,
    apply_to_pose,
    between,
    compose,
    correction,
    trajectory_distortion,
    wrap,
)


def test_jacobians_match_finite_differences():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(0, 1, 3), rng.normal(0, 1, 3)
        Ja, Jb = PoseGraph._jacobians(a, b)
        for k in range(3):
            d = np.zeros(3)
            d[k] = 1e-6
            na = (between(a + d, b) - between(a - d, b)) / 2e-6
            nb = (between(a, b + d) - between(a, b - d)) / 2e-6
            na[2], nb[2] = wrap(na[2] * 2e-6) / 2e-6, wrap(nb[2] * 2e-6) / 2e-6
            assert np.allclose(Ja[:, k], na, atol=1e-5) and np.allclose(Jb[:, k], nb, atol=1e-5)


def square_loop(n_side=10, step=0.1, drift=0.01):
    """True square path; odometry with a heading bias; returns truth, odometry estimate."""
    truth = [np.zeros(3)]
    for _side in range(4):
        for k in range(n_side):
            last = truth[-1]
            turn = math.pi / 2 if k == n_side - 1 else 0.0
            truth.append(compose(last, np.array([step, 0.0, turn])))
    est = [truth[0].copy()]
    for a, b in zip(truth[:-1], truth[1:]):
        z = between(a, b)
        z[2] += drift
        est.append(compose(est[-1], z))
    return truth, est


def graph_from(est, sig_odo=(0.01, 0.01, 0.01)):
    g = PoseGraph()
    for i, p in enumerate(est):
        g.add_node(i, p, fixed=(i == 0))
    for i in range(len(est) - 1):
        g.add_edge(i, i + 1, between(est[i], est[i + 1]), sig_odo)
    return g


def test_true_loop_removes_accumulated_drift():
    truth, est = square_loop()
    end = len(est) - 1
    before = float(np.hypot(*(est[end][:2] - truth[end][:2])))
    g = graph_from(est)
    g.add_edge(0, end, between(truth[0], truth[end]), (0.02, 0.02, 0.01), loop=True)
    out = g.optimize()
    after = float(np.hypot(*(out[end][:2] - truth[end][:2])))
    assert before > 0.1 and after < 0.3 * before
    assert g.edges[-1].weight > 0.5  # a consistent loop keeps its weight


def test_false_loop_is_down_weighted_and_tears_the_trajectory():
    truth, est = square_loop(drift=0.0)
    g = graph_from(est)
    g.add_edge(5, 25, np.array([0.0, 0.0, 0.0]), (0.02, 0.02, 0.01), loop=True)  # aliased "same place"
    before = {i: p for i, p in enumerate(est)}
    out = g.optimize()
    assert g.edges[-1].weight < 0.05  # GNC gives the outlier ~zero weight
    # without the robust kernel the false loop distorts the trajectory a lot
    g2 = graph_from(est)
    g2.add_edge(5, 25, np.array([0.0, 0.0, 0.0]), (0.02, 0.02, 0.01), loop=True)
    forced = g2.optimize(gnc_steps=(), cauchy=1e6)
    assert trajectory_distortion(before, forced) > 0.1 > trajectory_distortion(before, out)


def test_correction_transform_maps_old_pose_to_new():
    old, new = np.array([1.0, 2.0, 0.3]), np.array([1.2, 1.9, 0.45])
    T = correction(old, new)
    assert np.allclose(apply_to_pose(T, old), new)


def test_loop_needs_a_consistent_second_keyframe():
    """Temporal consistency: one verified loop does not close; a second keyframe
    nearby that implies the same correction of the current pose does. A different
    correction, or one too many keyframes later, does not."""
    from amr_rl.perception.camera_model import CameraModel
    from amr_rl.perception.vslam import PlanarVSLAM
    from amr_rl.robot.spec import RobotSpec

    slam = PlanarVSLAM(CameraModel.from_spec(RobotSpec.load()))
    c = np.array([0.10, -0.02, 0.03])
    assert slam._confirm_loop(100, 20, c) is None
    assert slam._confirm_loop(101, 21, c + np.array([0.20, 0.0, 0.0])) is None  # a different correction
    assert slam._confirm_loop(103, 22, c + np.array([0.01, 0.01, 0.005]))["kf"] == 100
    assert slam._confirm_loop(200, 20, c) is None
    assert slam._confirm_loop(210, 20, c) is None  # 10 keyframes later: too late


def test_relative_sigma_grows_along_the_chain_and_collapses_through_a_loop():
    from amr_rl.perception.camera_model import CameraModel
    from amr_rl.perception.vslam import Keyframe, PlanarVSLAM
    from amr_rl.robot.spec import RobotSpec

    slam = PlanarVSLAM(CameraModel.from_spec(RobotSpec.load()))
    for i in range(50):
        slam.keyframes.append(Keyframe(i, float(i), np.array([0.1 * i, 0.0, 0.0]), np.zeros((0, 2)),
                                       np.zeros((0, 32), np.uint8), np.zeros(0, int)))
    sig = slam.relative_sigma(49)
    assert sig[49] == 0.0 and sig[0] > sig[40] > 0.0
    slam.loop_edges.append((2, 49, np.zeros(3)))  # keyframe 2 already tied to the current one
    tied = slam.relative_sigma(49)
    assert tied[2] <= 0.031 and tied[0] < sig[0]


def test_covisibility_keeps_reanchored_keyframes_in_place_under_a_loop_correction():
    """Shadow evaluation finding: keyframes that tracking had re-anchored to the old
    map (implicitly corrected) were dragged by a loop correction pushed through the
    odometry chain. A covisibility edge to the old keyframe holds them."""
    n = 60
    truth = [np.array([0.05 * i, 0.0, 0.0]) for i in range(n)]
    est = [t.copy() for t in truth]
    for i in range(20, 41):  # drift accumulates over keyframes 20-40...
        est[i] = truth[i] + np.array([0.0, 0.01 * (i - 19), 0.0])
    # ...then tracking re-anchors on the old map: 41+ are correct again

    def solve(with_covis):
        g = PoseGraph()
        for i, p in enumerate(est):
            g.add_node(i, p, fixed=(i == 0))
        for i in range(n - 1):
            g.add_edge(i, i + 1, between(est[i], est[i + 1]), (0.01, 0.01, 0.01))
        g.add_edge(19, 40, between(truth[19], truth[40]), (0.02, 0.02, 0.01), loop=True)  # a correct loop
        if with_covis:
            for i in range(41, n):
                g.add_edge(5, i, between(est[5], est[i]), (0.02, 0.02, 0.01))
        out = g.optimize()
        return max(float(np.hypot(*(out[i][:2] - truth[i][:2]))) for i in range(41, n))

    assert solve(False) > 0.1  # the re-anchored tail is pushed off by the correction
    assert solve(True) < 0.03
