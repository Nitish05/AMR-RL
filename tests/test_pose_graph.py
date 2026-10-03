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
