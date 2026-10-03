"""Conservative occupancy mapping from onboard RGB keyframes.

Evidence sources (all derived from the onboard camera + its calibration + the
VSLAM pose estimate):

* **Floor evidence (free):** for two keyframes with baseline, the floor plane
  induces a homography between images. It is estimated from matched features
  and accepted only if it agrees with the pose-predicted floor homography. After
  warping, textured patches that agree photometrically (high NCC) lie on the
  floor plane; their inverse-perspective ground points get free evidence.
* **Obstacle evidence (occupied):** textured patches that disagree violate the
  floor-plane hypothesis (plane-induced parallax). In each image column, the
  lowest non-floor run above observed floor marks an obstacle base.
  Triangulated landmarks between 4 cm and 45 cm high also mark occupied cells.
* **Self evidence:** cells the robot body occupied along its estimated path.
  An operator-attested start clearance may mark the start disc free
  (recorded in the run manifest when used).

Textureless, never-observed, far (> max_range) or inconsistent regions stay
UNKNOWN. Planning treats UNKNOWN exactly like OCCUPIED.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


@dataclass
class GridConfig:
    resolution: float = 0.05
    size: float = 14.0  # metres, square, centred on the map origin
    free_hit: float = -0.45
    occ_hit: float = 0.9
    clamp: float = 4.0
    # Free-evidence saturation. Tried lower (-2, -3) so newly placed objects would
    # de-certify mapped floor faster: stray obstacle hits from live evidence then
    # eroded certified space and goal clearance (arena exploration stalled; learning
    # approaches failed with goal_lacks_footprint_clearance). Kept symmetric; a new
    # object needs 3 obstacle observations to take saturated floor out of FREE.
    free_clamp: float = 4.0
    free_threshold: float = -1.3
    occ_threshold: float = 1.0


class OccupancyGrid:
    def __init__(self, map_version: str, config: GridConfig | None = None):
        self.cfg = config or GridConfig()
        self.map_version = map_version
        n = int(round(self.cfg.size / self.cfg.resolution))
        self.n = n
        self.origin = np.array([-self.cfg.size / 2, -self.cfg.size / 2])
        self.logodds = np.zeros((n, n), np.float32)  # [iy, ix]
        self.revision = 0
        # Hard keep-out discs around remembered entities (position + size + uncertainty).
        # Derived runtime state, not saved with the map; overrides floor evidence.
        self.keepout = np.zeros((n, n), bool)
        self._keepout_key = None
        # Evidence journal for loop closure: every update with the keyframe it was
        # measured from (``anchor``, set by the runtime) and the cell centres it hit.
        # After a loop closure the grid is rebuilt from ``_base`` by replaying the
        # journal with each anchor's correction (points kept exactly, not re-quantised).
        self.anchor = -1
        self.journal = []  # (anchor, xy float32 (k, 2), delta)
        self._base = self.logodds.copy()

    # -------------------------------------------------------------- indexing
    def to_cell(self, xy):
        xy = np.atleast_2d(xy)
        ij = np.floor((xy - self.origin) / self.cfg.resolution).astype(int)
        return ij[:, 0], ij[:, 1]

    def to_xy(self, ix, iy):
        return self.origin + (np.stack([ix, iy], -1) + 0.5) * self.cfg.resolution

    def inside(self, ix, iy):
        return (ix >= 0) & (iy >= 0) & (ix < self.n) & (iy < self.n)

    # -------------------------------------------------------------- updates
    def add_hits(self, xy, delta, *, min_count=1):
        if len(xy) == 0:
            return 0
        ix, iy = self.to_cell(xy)
        ok = self.inside(ix, iy)
        flat = iy[ok] * self.n + ix[ok]
        cells, counts = np.unique(flat, return_counts=True)
        cells = cells[counts >= min_count]
        if len(cells) == 0:
            return 0
        values = self.logodds.reshape(-1)
        values[cells] = np.clip(values[cells] + delta, -self.cfg.free_clamp, self.cfg.clamp)
        self.journal.append((self.anchor, self.to_xy(cells % self.n, cells // self.n).astype(np.float32), float(delta)))
        self.revision += 1
        return len(cells)

    def replay(self, transforms):
        """Rebuild the evidence after a loop closure. ``transforms``: {anchor keyframe
        id: planar transform (tx, ty, theta)} mapping where an update was placed to
        where it belongs now; updates from other anchors are kept in place."""
        from ..perception.pose_graph import apply_transform

        values = self._base.copy().reshape(-1)
        for k, (a, xy, delta) in enumerate(self.journal):
            T = transforms.get(a)
            if T is not None and (abs(T[0]) > 1e-9 or abs(T[1]) > 1e-9 or abs(T[2]) > 1e-12):
                xy = apply_transform(T, xy).astype(np.float32)
                self.journal[k] = (a, xy, delta)
            ix, iy = self.to_cell(xy)
            ok = self.inside(ix, iy)
            cells = np.unique(iy[ok] * self.n + ix[ok])
            values[cells] = np.clip(values[cells] + delta, -self.cfg.free_clamp, self.cfg.clamp)
        self.logodds = values.reshape(self.logodds.shape)
        self.revision += 1

    def mark_free_disc(self, center, radius, strength=None):
        r = int(np.ceil(radius / self.cfg.resolution))
        cx, cy = self.to_cell(center)
        ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
        pts = self.to_xy(cx[0] + xs.ravel(), cy[0] + ys.ravel())
        keep = np.linalg.norm(pts - np.asarray(center)[None], axis=1) <= radius
        return self.add_hits(pts[keep], strength if strength is not None else self.cfg.free_hit * 3)

    def mark_footprint(self, pose, length, width, rear_offset):
        """Rectangle occupied by the robot body at an estimated pose -> free."""
        xs = np.arange(-rear_offset, length - rear_offset + 1e-9, self.cfg.resolution / 2)
        ys = np.arange(-width / 2, width / 2 + 1e-9, self.cfg.resolution / 2)
        gx, gy = np.meshgrid(xs, ys)
        c, s = np.cos(pose[2]), np.sin(pose[2])
        pts = np.stack([pose[0] + c * gx - s * gy, pose[1] + s * gx + c * gy], -1).reshape(-1, 2)
        return self.add_hits(pts, self.cfg.free_hit)

    def set_keepout(self, discs):
        """``discs``: [(x, y, radius)]. Cells within a disc are OCCUPIED for planning,
        whatever the floor evidence says (a known object is never driven through)."""
        key = tuple((round(x, 2), round(y, 2), round(r, 2)) for x, y, r in discs)
        if key == self._keepout_key:
            return False
        mask = np.zeros_like(self.keepout)
        for x, y, r in discs:
            cells = int(np.ceil(r / self.cfg.resolution)) + 1
            cx, cy = self.to_cell(np.array([x, y], float))
            ys, xs = np.mgrid[-cells:cells + 1, -cells:cells + 1]
            ix, iy = cx[0] + xs.ravel(), cy[0] + ys.ravel()
            ok = self.inside(ix, iy)
            ix, iy = ix[ok], iy[ok]
            centres = self.to_xy(ix, iy)
            near = np.linalg.norm(centres - np.array([x, y])[None], axis=1) <= r
            mask[iy[near], ix[near]] = True
        self.keepout = mask
        self._keepout_key = key
        self.revision += 1
        return True

    def classes(self):
        out = np.full(self.logodds.shape, UNKNOWN, np.uint8)
        out[self.logodds <= self.cfg.free_threshold] = FREE
        out[self.logodds >= self.cfg.occ_threshold] = OCCUPIED
        out[self.keepout] = OCCUPIED
        return out

    def classify_xy(self, xy):
        ix, iy = self.to_cell(xy)
        ok = self.inside(ix, iy)
        out = np.full(len(ix), UNKNOWN, np.uint8)
        cls = self.classes()
        out[ok] = cls[iy[ok], ix[ok]]
        return out

    def counts(self):
        c = self.classes()
        return {"free": int((c == FREE).sum()), "occupied": int((c == OCCUPIED).sum()),
                "unknown": int((c == UNKNOWN).sum())}

    def save(self, path):
        np.savez_compressed(path, logodds=self.logodds, origin=self.origin, resolution=self.cfg.resolution,
                            map_version=self.map_version)

    @classmethod
    def load(cls, path, map_version):
        data = np.load(path)
        if str(data["map_version"]) != map_version:
            raise ValueError("Occupancy grid belongs to a different map version")
        grid = cls(map_version)
        grid.logodds[:] = np.clip(data["logodds"], -grid.cfg.free_clamp, grid.cfg.clamp)
        grid._base = grid.logodds.copy()  # a loaded map is the replay base (its keyframes stay fixed)
        return grid


@dataclass
class EvidenceConfig:
    max_range: float = 2.6
    min_baseline: float = 0.035
    ncc_window: int = 9
    ncc_floor: float = 0.80
    ncc_obstacle: float = 0.35
    texture_std: float = 5.0
    min_floor_pixels_per_cell: int = 3
    homography_tolerance_px: float = 6.0
    obstacle_run: int = 4
    min_obstacle_height: float = 0.03  # shortest obstacle the free-space test must detect
    min_parallax_px: float = 3.5  # its predicted parallax must reach this to count as evidence
    pose_homography_max_gap: float = 8.0  # s between keyframes for pose-predicted homography
    min_peak: float = 0.2  # NCC advantage of floor alignment over the obstacle alignment
    # Obstacle-base evidence is committed only where >= base_confirm_pairs keyframe pairs
    # (same new keyframe, different reference keyframes) agree within one cell. A
    # single pair's base hits were 91 % on open floor (scripts/dev/obstacle_hits_probe.py).
    base_confirm_pairs: int = 2
    # Pixels above an obstacle base in its image column show the obstacle's face (or
    # what is hidden behind it): they never count as floor. Conservative for low
    # obstacles, over which real floor is visible. (false_free_probe.py)
    suppress_free_above_bases: bool = False  # A/B on one trajectory: no effect (265 vs 265 false-free)


class FloorEvidenceMapper:
    def __init__(self, model, grid: OccupancyGrid, config: EvidenceConfig | None = None):
        self.model = model
        self.grid = grid
        self.cfg = config or EvidenceConfig()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        h, w = model.height, model.width
        v, u = np.mgrid[0:h, 0:w]
        self.uv = np.stack([u.ravel(), v.ravel()], 1).astype(float)
        self.uv_h = np.column_stack([self.uv, np.ones(len(self.uv))])
        self.horizon = model.horizon_row()
        self.stats = {"updates": 0, "skipped": {}}

    def _skip(self, reason):
        self.stats["skipped"][reason] = self.stats["skipped"].get(reason, 0) + 1
        return {"updated": False, "reason": reason}

    def update(self, ref, cur, *, defer_bases=False):
        """Accumulate evidence from keyframe ``ref`` into keyframe ``cur``'s view.
        With ``defer_bases`` the obstacle-base hits are returned (``base_xy``) instead
        of applied, so the caller can require agreement between pairs."""
        if ref.gray is None or cur.gray is None:
            return self._skip("missing_images")
        c_ref = self.model.T_world_cam(ref.pose)[:3, 3]
        c_cur = self.model.T_world_cam(cur.pose)[:3, 3]
        if np.linalg.norm(c_cur - c_ref) < self.cfg.min_baseline:
            return self._skip("insufficient_baseline")
        if abs(np.arctan2(np.sin(cur.pose[2] - ref.pose[2]), np.cos(cur.pose[2] - ref.pose[2]))) > 0.8:
            return self._skip("insufficient_overlap")
        H_pred = self.model.floor_homography(cur.pose) @ np.linalg.inv(self.model.floor_homography(ref.pose))
        H = self._data_homography(ref, cur, H_pred)
        if H is None:
            # Short-horizon relative poses are accurate (drift accumulates slowly), so
            # the pose-predicted floor homography is used when image support is thin.
            if abs(cur.timestamp - ref.timestamp) > self.cfg.pose_homography_max_gap:
                return self._skip("floor_homography_unconfirmed")
            H = H_pred
            self.stats["pose_homography"] = self.stats.get("pose_homography", 0) + 1
        h, w = cur.gray.shape
        g1 = cv2.GaussianBlur(ref.gray, (0, 0), 1.0).astype(np.float32)
        g2 = cv2.GaussianBlur(cur.gray, (0, 0), 1.0).astype(np.float32)
        ground, ok = self.model.ground_points_world(self.uv, cur.pose, self.cfg.max_range)
        shift, detectable = self._parallax_shift(ground, ok, c_cur, ref.pose)
        shift = shift.reshape(h, w, 2)
        detectable = detectable.reshape(h, w)
        # Floor alignment: cur pixel -> ref pixel through the floor homography.
        Hinv = np.linalg.inv(H)
        src = self.uv_h @ Hinv.T
        base_x = (src[:, 0] / src[:, 2]).reshape(h, w).astype(np.float32)
        base_y = (src[:, 1] / src[:, 2]).reshape(h, w).astype(np.float32)
        valid = (base_x >= 3) & (base_x < w - 4) & (base_y >= 3) & (base_y < h - 4)
        k = (self.cfg.ncc_window, self.cfg.ncc_window)
        m2 = cv2.blur(g2, k)
        v2 = cv2.blur(g2 * g2, k) - m2 * m2

        def ncc_at(dx, dy):
            warped = cv2.remap(g1, base_x + dx, base_y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            m1 = cv2.blur(warped, k)
            v1 = cv2.blur(warped * warped, k) - m1 * m1
            cov = cv2.blur(warped * g2, k) - m1 * m2
            return cov / np.sqrt(np.maximum(v1 * v2, 1e-6)), v1

        ncc, v1 = ncc_at(0.0, 0.0)
        # Where would a just-detectable obstacle point have appeared? If the image
        # matches there nearly as well, the test is not discriminative -> no evidence.
        sx, sy = shift[..., 0].astype(np.float32), shift[..., 1].astype(np.float32)
        ncc_plus, _ = ncc_at(sx, sy)
        ncc_minus, _ = ncc_at(-sx, -sy)
        peak = ncc - np.maximum(ncc_plus, ncc_minus)
        textured = (v1 > self.cfg.texture_std ** 2) & (v2 > self.cfg.texture_std ** 2)
        rows = np.arange(h)[:, None] > self.horizon + 4
        usable = valid & textured & rows & detectable
        ground_xy = ground[:, :2].reshape(h, w, 2)
        ok = ok.reshape(h, w)
        floor = usable & (ncc >= self.cfg.ncc_floor) & (peak >= self.cfg.min_peak)
        nonfloor = usable & (ncc <= self.cfg.ncc_obstacle)
        bases = self._obstacle_bases(floor, nonfloor, ok)
        free_mask = floor & ok
        if self.cfg.suppress_free_above_bases and len(bases):
            above = np.zeros_like(free_mask)
            for u, base_v in bases:  # bases are sampled every 2nd column: cover u..u+1
                above[:base_v + 1, u:u + 2] = True
            free_mask &= ~above
        free_cells = self.grid.add_hits(ground_xy[free_mask], self.grid.cfg.free_hit,
                                        min_count=self.cfg.min_floor_pixels_per_cell)
        base_xy = ground_xy[bases[:, 1], bases[:, 0]] if len(bases) else np.zeros((0, 2))
        if defer_bases:
            occ_cells = 0
        else:
            occ_cells = self.grid.add_hits(base_xy, self.grid.cfg.occ_hit)
        self.stats["updates"] += 1
        return {"updated": True, "free_cells": free_cells, "occupied_cells": occ_cells, "base_xy": base_xy,
                "floor_pixels": int(free_mask.sum()), "nonfloor_pixels": int(nonfloor.sum()),
                "floor_mask": floor, "nonfloor_mask": nonfloor}

    def _parallax_shift(self, ground, ok, center, ref_pose):
        """Per pixel: the reference-image displacement between the floor point and a
        just-detectable obstacle point (minimum height, just in front of it) on the
        same current-image ray. Pixels whose displacement is below the threshold
        cannot provide evidence."""
        shift = np.zeros((len(ground), 2))
        out = np.zeros(len(ground), bool)
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            return shift, out
        P = ground[idx]
        frac = 1.0 - self.cfg.min_obstacle_height / max(center[2], 1e-3)
        Q = center + (P - center) * frac
        uv_p, zp = self.model.project_world(P, ref_pose)
        uv_q, zq = self.model.project_world(Q, ref_pose)
        delta = uv_q - uv_p
        parallax = np.linalg.norm(delta, axis=1)
        good = (zp > 0.05) & (zq > 0.05) & (parallax >= self.cfg.min_parallax_px)
        out[idx] = good
        shift[idx] = np.where(good[:, None], delta, 0.0)
        return shift, out

    def _data_homography(self, ref, cur, H_pred):
        below_r = ref.pts[:, 1] > self.horizon + 6
        below_c = cur.pts[:, 1] > self.horizon + 6
        a, b = np.flatnonzero(below_r), np.flatnonzero(below_c)
        if len(a) < 20 or len(b) < 20:
            return None
        matches = self.matcher.knnMatch(ref.desc[a], cur.desc[b], k=2)
        pairs = [(m[0].queryIdx, m[0].trainIdx) for m in matches
                 if m and m[0].distance < 50 and (len(m) < 2 or m[0].distance < 0.8 * m[1].distance)]
        if len(pairs) < 20:
            return None
        p1 = ref.pts[a[[p[0] for p in pairs]]].astype(np.float32)
        p2 = cur.pts[b[[p[1] for p in pairs]]].astype(np.float32)
        H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
        if H is None or inl is None or inl.sum() < 20:
            return None
        # Agreement with the pose-predicted floor homography on a lower-image grid.
        h, w = cur.gray.shape
        grid = np.array([[u, v, 1.0] for u in np.linspace(20, w - 20, 6)
                         for v in np.linspace(max(self.horizon + 20, 0), h - 10, 4)])
        src = (np.linalg.inv(H_pred) @ grid.T).T
        src = src[:, :2] / src[:, 2:3]
        pred = grid[:, :2]
        homog = np.column_stack([src, np.ones(len(src))]) @ H.T
        data = homog[:, :2] / homog[:, 2:3]
        if np.median(np.linalg.norm(pred - data, axis=1)) > self.cfg.homography_tolerance_px:
            return None
        return H

    def _obstacle_bases(self, floor, nonfloor, ok):
        h, w = floor.shape
        top = int(max(0, np.ceil(self.horizon + 4)))
        run = self.cfg.obstacle_run
        bases = []
        for u in range(0, w, 2):
            seen_floor = 0
            count = 0
            for v in range(h - 1, top, -1):
                if floor[v, u]:
                    seen_floor += 1
                    count = 0
                elif nonfloor[v, u]:
                    count += 1
                    if count >= run and seen_floor >= 3:
                        base_v = v + run - 1
                        if ok[base_v, u]:
                            bases.append((u, base_v))
                        break
                else:
                    count = 0
        return np.array(bases, int).reshape(-1, 2)

    def update_keyframe(self, keyframes, max_pairs=3, window=16, wide_pairs=1):
        """Pair the newest keyframe with up to ``max_pairs`` recent keyframes, plus
        ``wide_pairs`` spatially near older keyframes chosen for the largest lateral
        baseline (larger baselines make farther floor detectable)."""
        cur = keyframes[-1]
        results = []
        defer = self.cfg.base_confirm_pairs > 1
        for ref in reversed(keyframes[-1 - window:-1]):
            if len([r for r in results if r["updated"]]) >= max_pairs:
                break
            results.append(self.update(ref, cur, defer_bases=defer))
        if wide_pairs and len(keyframes) > window + 1:
            c_cur = self.model.T_world_cam(cur.pose)[:3, 3]
            heading = np.array([np.cos(cur.pose[2]), np.sin(cur.pose[2])])
            scored = []
            for ref in keyframes[:-1 - window]:
                if ref.gray is None:
                    continue
                dth = abs(np.arctan2(np.sin(cur.pose[2] - ref.pose[2]), np.cos(cur.pose[2] - ref.pose[2])))
                if dth > 0.6:
                    continue
                c_ref = self.model.T_world_cam(ref.pose)[:3, 3]
                d = c_ref[:2] - c_cur[:2]
                dist = float(np.linalg.norm(d))
                if not 0.15 <= dist <= 0.9:
                    continue
                lateral = abs(float(heading[0] * d[1] - heading[1] * d[0]))
                scored.append((lateral, ref))
            scored.sort(key=lambda item: -item[0])
            for _, ref in scored[:wide_pairs]:
                results.append(self.update(ref, cur, defer_bases=defer))
        if defer:
            self._commit_confirmed_bases(results)
        return results

    def _commit_confirmed_bases(self, results):
        """Apply obstacle-base hits supported by >= base_confirm_pairs pairs (within one
        cell, 3x3 neighbourhood); drop the rest."""
        per_pair = []
        for r in results:
            if r.get("updated") and len(r.get("base_xy", ())):
                ix, iy = self.grid.to_cell(r["base_xy"])
                ok = self.grid.inside(ix, iy)
                per_pair.append(set(zip(ix[ok].tolist(), iy[ok].tolist())))
        if len(per_pair) < self.cfg.base_confirm_pairs:
            self.stats["bases_unconfirmed"] = self.stats.get("bases_unconfirmed", 0) + sum(map(len, per_pair))
            return 0
        support = {}
        for cells in per_pair:
            near = {(x + dx, y + dy) for x, y in cells for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            for c in near:
                support[c] = support.get(c, 0) + 1
        confirmed = [c for cells in per_pair for c in cells if support.get(c, 0) >= self.cfg.base_confirm_pairs]
        confirmed = sorted(set(confirmed))
        self.stats["bases_confirmed"] = self.stats.get("bases_confirmed", 0) + len(confirmed)
        self.stats["bases_unconfirmed"] = self.stats.get("bases_unconfirmed", 0) + \
            sum(map(len, per_pair)) - len(confirmed)
        if not confirmed:
            return 0
        xy = self.grid.to_xy(np.array([c[0] for c in confirmed]), np.array([c[1] for c in confirmed]))
        return self.grid.add_hits(xy, self.grid.cfg.occ_hit)

    def add_landmark_obstacles(self, points_xyz, low=0.04, high=0.45):
        mask = (points_xyz[:, 2] > low) & (points_xyz[:, 2] < high)
        return self.grid.add_hits(points_xyz[mask, :2], self.grid.cfg.occ_hit * 0.5)
