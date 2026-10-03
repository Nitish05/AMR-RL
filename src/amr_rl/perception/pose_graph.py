"""Planar (SE(2)) keyframe pose graph for loop closure.

Nodes are keyframe poses (x, y, theta) in the map frame. Edges are relative poses
measured in the first node's frame: odometry between consecutive keyframes (from the
tracker) and loop closures (from place recognition + geometric verification).
``optimize`` is Gauss-Newton on the whitened residuals; loop edges get a robust
Cauchy weight whose scale is annealed from large to small (graduated
non-convexity, as in Yang et al. 2020 / the GNC pose-graph schedules of 2024-25),
so a wrong loop that the rest of the graph disagrees with ends with ~zero weight.

``trajectory_distortion`` is the ROVER loop check (T-ASE 2026): align the optimised
trajectory to the one before the loop rigidly (planar: rotation + translation; the
metric scale is known) and take the RMS residual. A true loop bends the trajectory
gracefully (small residual); a false one tears it (large residual).

Everything here is engineered geometry; nothing is learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    """a (+) b for planar poses."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def between(a, b):
    """a^-1 (+) b: b expressed in a's frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])


def correction(old, new):
    """Rigid planar transform T with T(old) = new (applied to points/poses near old)."""
    th = wrap(new[2] - old[2])
    c, s = math.cos(th), math.sin(th)
    t = np.array([new[0] - (c * old[0] - s * old[1]), new[1] - (s * old[0] + c * old[1])])
    return np.array([t[0], t[1], th])


def apply_transform(T, xy):
    """Apply planar transform T=(tx, ty, th) to points (N,2)."""
    xy = np.atleast_2d(np.asarray(xy, float))
    c, s = math.cos(T[2]), math.sin(T[2])
    return np.stack([c * xy[:, 0] - s * xy[:, 1] + T[0], s * xy[:, 0] + c * xy[:, 1] + T[1]], 1)


def apply_to_pose(T, pose):
    p = apply_transform(T, np.asarray(pose[:2])[None])[0]
    return np.array([p[0], p[1], wrap(pose[2] + T[2])])


@dataclass
class Edge:
    i: int
    j: int
    z: np.ndarray            # measured between(pose_i, pose_j)
    sigma: np.ndarray        # (sx, sy, sth) standard deviations
    loop: bool = False
    weight: float = 1.0      # robust weight after optimisation (loop edges)


@dataclass
class PoseGraph:
    poses: dict = field(default_factory=dict)  # id -> np.array(3)
    edges: list = field(default_factory=list)
    fixed: set = field(default_factory=set)

    def add_node(self, i, pose, fixed=False):
        self.poses[i] = np.asarray(pose, float).copy()
        if fixed:
            self.fixed.add(i)

    def add_edge(self, i, j, z, sigma, loop=False):
        self.edges.append(Edge(i, j, np.asarray(z, float), np.asarray(sigma, float), loop))

    def _residual(self, e, poses):
        r = between(poses[e.i], poses[e.j]) - e.z
        r[2] = wrap(r[2])
        return r / e.sigma

    def optimize(self, iters=15, gnc_steps=(30.0, 10.0, 3.0), cauchy=3.0):
        """Gauss-Newton with GNC-annealed Cauchy weights on loop edges (scale
        ``cauchy`` x step, ending at ``cauchy`` sigma: ~the 99 % gate of a 3-DOF
        residual, so a true loop that keeps a few sigma of residual after balancing
        against odometry keeps its weight). Returns the optimised poses (dict) and
        leaves edge weights set."""
        ids = sorted(self.poses)
        free = [i for i in ids if i not in self.fixed]
        if not free:
            return dict(self.poses)
        index = {i: k for k, i in enumerate(free)}
        poses = {i: p.copy() for i, p in self.poses.items()}
        for mu in list(gnc_steps) + [None]:
            for _ in range(iters):
                rows, cols, vals, g = [], [], [], np.zeros(3 * len(free))
                for e in self.edges:
                    r = self._residual(e, poses)
                    w = 1.0
                    if e.loop:
                        c2 = (cauchy * (mu if mu is not None else 1.0)) ** 2
                        w = c2 / (c2 + float(r @ r))
                        e.weight = w
                    Ji, Jj = self._jacobians(poses[e.i], poses[e.j])
                    Ji, Jj = Ji / e.sigma[:, None], Jj / e.sigma[:, None]
                    blocks = [(e.i, Ji), (e.j, Jj)]
                    for a, Ja in blocks:
                        if a not in index:
                            continue
                        g[3 * index[a]:3 * index[a] + 3] += w * Ja.T @ r
                        for b, Jb in blocks:
                            if b not in index:
                                continue
                            H = w * Ja.T @ Jb
                            ii, jj = np.meshgrid(range(3), range(3), indexing="ij")
                            rows += list((3 * index[a] + ii).ravel())
                            cols += list((3 * index[b] + jj).ravel())
                            vals += list(H.ravel())
                n = 3 * len(free)
                H = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
                H = H + 1e-9 * coo_matrix((np.ones(n), (range(n), range(n))), shape=(n, n)).tocsr()
                dx = -spsolve(H, g)
                for i, k in index.items():
                    poses[i] = poses[i] + dx[3 * k:3 * k + 3]
                    poses[i][2] = wrap(poses[i][2])
                if np.abs(dx).max() < 1e-7:
                    break
        return poses

    @staticmethod
    def _jacobians(a, b):
        """d between(a, b) / d a and / d b."""
        c, s = math.cos(a[2]), math.sin(a[2])
        dx, dy = b[0] - a[0], b[1] - a[1]
        Ja = np.array([[-c, -s, -s * dx + c * dy],
                       [s, -c, -c * dx - s * dy],
                       [0.0, 0.0, -1.0]])
        Jb = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        return Ja, Jb


def trajectory_distortion(before: dict, after: dict) -> float:
    """ROVER-style score: RMS position residual after rigidly aligning the optimised
    keyframe positions to the positions before the loop (planar Procrustes)."""
    ids = sorted(set(before) & set(after))
    if len(ids) < 3:
        return 0.0
    P = np.array([before[i][:2] for i in ids])
    Q = np.array([after[i][:2] for i in ids])
    pc, qc = P.mean(0), Q.mean(0)
    A, B = P - pc, Q - qc
    U, _, Vt = np.linalg.svd(B.T @ A)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T
    aligned = (B @ R.T) + pc
    return float(np.sqrt(np.mean(np.sum((aligned - P) ** 2, axis=1))))
