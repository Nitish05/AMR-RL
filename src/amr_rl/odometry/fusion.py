"""Wheel-inertial odometry: IMU gyro for rotation, wheel encoders for distance.

ENGINEERED estimator (no learning). Per camera frame it turns the raw samples
received since the previous frame (``ProprioBatch``) into one planar motion
increment in the robot frame at the previous frame, with a covariance and flags:

* rotation: the gyro's z rate (body vertical), corrected for the estimated bias
  and scale, integrated at the IMU rate. On a flat floor the body z rate is the
  yaw rate (the caster rocking is zero-mean); projecting with an accelerometer
  tilt estimate was tried and removed: the accelerometer's offsets (+-20 mg typ)
  bias the tilt by 1-2 deg, which mixed cross-axis gyro terms into yaw (+0.13 %). The gyro does not see wheel slip or the camera's
  view, so it is the trusted source of rotation.
* distance: mean wheel travel from encoder counts (wheel radius from the robot
  spec times an online-checked scale). Encoders measure wheel rotation, not
  motion over the floor, so wheel slip corrupts them; the step says when slip is
  suspected and widens its translation uncertainty accordingly.
* gyro bias: estimated whenever the robot stands still (no encoder counts for
  ``zupt_min_s``; zero-velocity update), with a random-walk process between.
* slip detectors: (1) the wheels' differential rotation disagrees with the gyro
  (one wheel slipping or skidding sideways); (2) the accelerometer's forward
  specific force disagrees with the wheels' acceleration (spinning up on a
  slippery floor, hitting something while the wheels keep turning); (3) the
  wheels turn far faster than the commanded speed allows.

Noise levels come from the parts' datasheets in the robot spec (prior knowledge of
the hardware, not simulator ground truth).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .samples import ProprioBatch

DPS = math.pi / 180.0
G = 9.80665


@dataclass
class OdometryConfig:
    # Standing still: no encoder counts this long, then zupt_quiet_frames consecutive
    # quiet frames. After a turn the chassis creeps (~0.06 dps, low noise) while it
    # settles; a 0.3 s / one-frame rule took that creep as the offset (development run).
    zupt_min_s: float = 0.6
    zupt_quiet_frames: int = 3
    zupt_floor: float = 0.005 * DPS  # rad/s per frame: residual creep, not averaged away
    zupt_gyro_gate: float = 0.03     # rad/s: |gyro_z - bias| above this is not still
    # Zero-rotation update while driving straight. OFF: in simulation the chassis skids
    # (0.1-0.3 dps of yaw with equal wheel travel; up to ~2 dps just after starting off),
    # and one accepted window shifted the offset 0.08 dps (development run: 14 deg).
    zaru: bool = False
    zaru_max_frame_yaw: float = 0.003  # rad of encoder yaw per frame: "driving straight"
    zaru_window_s: float = 2.0
    zaru_min_s: float = 1.0
    zaru_floor: float = 0.0005       # rad/s*sqrt(s): unmodelled side-skid while driving straight
    zaru_gate: float = 0.01 * DPS    # rad/s on top of 3 sigma
    zaru_settle_s: float = 1.0       # straight driving this long before a window starts (starts skid)
    zaru_max_dv: float = 0.02        # m/s change per frame: steady speed only
    # rad/s/sqrt(s): offset drift between updates beyond the temperature model (2x the
    # analogue rate random walk 1.5e-4 dps/sqrt(s) plus margin for an imperfect model).
    bias_rw: float = 0.001 * DPS
    tempco_rw: float = 1e-5 * DPS    # rad/s/degC/sqrt(s): the tempco is nearly constant
    tempco_prior_factor: float = 2.0
    track_scale_init: float = 1.0    # gyro yaw / encoder yaw (scrub of a skid-steered turn)
    track_scale_gain: float = 0.05
    track_scale_bounds: tuple = (0.6, 1.4)
    radius_scale_sigma: float = 0.10  # prior: effective rolling radius vs the spec (tyre, load, floor)
    slip_floor: float = 0.03          # relative distance error without detected slip
    yaw_slip_k: float = 4.0           # differential slip: |residual| > k sigma
    yaw_slip_rel: float = 0.06
    yaw_slip_abs: float = 0.004       # rad per frame
    accel_slip: float = 2.0           # m/s^2 between accelerometer and wheels
    accel_slip_samples: int = 3
    slip_hold_s: float = 0.6
    tilt_yaw_error: float = 0.0005    # relative yaw error from caster rocking (body z vs vertical)


@dataclass
class OdometryStep:
    """Motion between two camera frames, in the robot frame at the earlier one."""

    dt: float
    dx: float
    dy: float
    dth: float
    cov: np.ndarray               # 3x3 over (dx, dy, dth)
    v: float                       # mean forward speed (m/s), from the encoders
    w: float                       # mean yaw rate (rad/s), from the gyro
    distance: float                # |wheel travel| (m)
    slip: bool = False
    stationary: bool = False
    rotation_source: str = "gyro"  # gyro | encoders
    translation_source: str = "encoders"  # encoders | command | command_model
    info: dict = field(default_factory=dict)
    trust_rotation: bool = True    # the gyro is trusted over vision on rotation; a command model is not

    def as_command(self):
        """(v, w) in the form the command-driven code paths expect."""
        return (self.v, self.w)


class WheelInertialOdometry:
    def __init__(self, spec, config: OdometryConfig | None = None, calibration: dict | None = None):
        """``calibration``: this unit's gyro calibration (scripts/calibrate_imu.py:
        ``gyro_scale`` and its ``gyro_scale_sigma``), used as the scale prior."""
        self.cfg = config or OdometryConfig()
        imu, enc, drive = spec.raw["imu"], spec.raw["encoders"], spec.raw["drive"]
        self.r = float(drive["wheel_radius"])
        self.track = float(drive["track"])
        self.cpr = float(enc["counts_per_wheel_rev"])
        g = imu["gyro"]
        bw = float(imu.get("bandwidth_hz", float(imu["output_rate_hz"]) / 2))
        self.noise_density = float(g["noise_density_dps_rthz"]) * DPS
        self.sample_noise = self.noise_density * math.sqrt(bw)
        self.lsb = float(g["lsb_mdps"]) * 1e-3 * DPS
        # Gyro z offset model: offset(T) = b0 + k (T - T0), T the chip's own temperature
        # sensor (warm-up after power-on shifts the offset by tempco x dT: +-0.010
        # dps/degC typ). State x = [b0, k] with covariance P; without temperature
        # readings it reduces to a single offset b0.
        self._x = np.array([0.0, 0.0])
        # Tempco prior: 2x the datasheet's "typ" value (typ is not guaranteed; a unit at
        # 2.5x typ drifted 0.03 dps unnoticed in development with a 1x prior).
        self._P = np.diag([(float(g["offset_sigma_dps"]) * DPS) ** 2,
                           (self.cfg.tempco_prior_factor * float(g.get("tempco_sigma_dps_per_c", 0.0)) * DPS) ** 2])
        self.temp = None   # latest chip temperature (degC)
        self.temp0 = None  # reference temperature of b0
        self.temp_at_update = None  # chip temperature at the last stand-still offset measurement
        self.zupt_frames = 0  # frames of stand-still offset measurement so far
        self.bias_calibrated = False
        self.last_zupt_t = -math.inf
        self.last_bias_update_t = -math.inf  # zero-velocity or straight-driving update
        self._zaru = None
        self._last_v = 0.0
        self._straight_s = 0.0
        # gyro scale: true rate = gyro_scale x measured. Prior from the datasheet's
        # sensitivity tolerance; refined from full in-place turns that vision closes
        # (``add_scale_sample``).
        self.gyro_scale = 1.0
        self.scale_var0 = self.scale_var = float(g["sensitivity_sigma"]) ** 2
        self.scale_samples: list[float] = []
        # Effective rolling radius / spec radius: distance travelled over the floor per
        # wheel revolution; measured from vision on straight stretches (add_radius_sample).
        self.radius_scale = 1.0
        self.radius_var = self.cfg.radius_scale_sigma ** 2
        self.radius_samples: list[float] = []
        self.calibration = None
        self.scale_prior = 1.0
        if calibration:
            self.calibration = dict(calibration)
            self.gyro_scale = float(calibration["gyro_scale"])
            self.scale_var0 = self.scale_var = float(calibration["gyro_scale_sigma"]) ** 2
            self.scale_prior = self.gyro_scale
        self._quiet_frames = 0
        self.track_scale = self.cfg.track_scale_init
        self.rest_ax = None  # forward specific force at rest (gravity x tilt + offset), from still periods
        self._last_imu_t = None
        self._last_enc = None
        self._still_since = None
        self._slip_until = -math.inf
        self._accel_bad = 0
        self._v_hist = []  # (t, encoder speed) for the wheel acceleration
        self.t = None
        self.log = {"zupt": 0, "slip_yaw": 0, "slip_accel": 0, "slip_cmd": 0}

    # -------------------------------------------------------------- samples
    def _yaw_rate(self, s, dt):
        return self.gyro_scale * (float(s.gyro[2]) - self.bias)

    # ------------------------------------------------------------ offset model
    def _h(self, temp=None):
        temp = self.temp if temp is None else temp
        d = 0.0 if (temp is None or self.temp0 is None) else temp - self.temp0
        return np.array([1.0, d])

    @property
    def bias(self):
        """Gyro z offset (rad/s) at the current chip temperature."""
        return float(self._h() @ self._x)

    @bias.setter
    def bias(self, value):
        self._x[0] += float(value) - self.bias

    @property
    def bias_var(self):
        h = self._h()
        return float(h @ self._P @ h)

    @bias_var.setter
    def bias_var(self, value):
        self._P = np.diag([float(value), self._P[1, 1]])

    @property
    def tempco(self):
        return float(self._x[1]), float(math.sqrt(max(self._P[1, 1], 0.0)))

    def _offset_update(self, meas, r_var, temp=None):
        """Kalman update with a measurement of the offset (rad/s) at ``temp``."""
        h = self._h(temp)
        S = float(h @ self._P @ h) + r_var
        K = self._P @ h / S
        self._x = self._x + K * (meas - float(h @ self._x))
        self._P = self._P - np.outer(K, h @ self._P)
        self._P = 0.5 * (self._P + self._P.T)

    def add_bias_feedback(self, rate_error: float, sigma: float):
        """Gyro rate error (rad/s, corrected units) measured by vision over a window,
        with its uncertainty (VSLAM ``_bias_window``): a Kalman update of the offset."""
        if not (math.isfinite(rate_error) and math.isfinite(sigma)) or abs(rate_error) > 0.05 or sigma <= 0:
            return
        if abs(rate_error) > 4 * math.sqrt(self.bias_var + sigma ** 2):
            self.log["bias_feedback_rejected"] = self.log.get("bias_feedback_rejected", 0) + 1
            return
        self._offset_update(self.bias + rate_error / max(self.gyro_scale, 0.5), sigma ** 2)
        self.log["bias_feedback"] = self.log.get("bias_feedback", 0) + 1

    def add_radius_sample(self, ratio: float):
        """Vision distance / encoder distance (at the current scale) over a straight,
        non-slipping stretch. Robust median of the recent samples; ratios outside
        0.7-1.3 are ignored (slip or a vision fault)."""
        value = ratio * self.radius_scale
        if not math.isfinite(value) or not 0.7 <= value <= 1.3:
            return False
        self.radius_samples = (self.radius_samples + [value])[-21:]
        x = np.asarray(self.radius_samples)
        if len(x) >= 3:
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med))) * 1.4826
            self.radius_scale = med
            self.radius_var = max((max(mad, 0.02) ** 2) / len(x), 0.005 ** 2)
        return True

    def add_scale_sample(self, ratio: float, sigma: float):
        """A measured (true / gyro) turn ratio from vision (a full turn closed on the
        same view). Robust: the scale is the median of the accepted samples, its
        variance from their spread and count; samples far from the datasheet range
        are ignored."""
        if not math.isfinite(ratio) or abs(ratio - self.scale_prior) > 0.05:
            return False
        self.scale_samples = (self.scale_samples + [float(ratio)])[-15:]
        x = np.asarray(self.scale_samples)
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med))) * 1.4826 if len(x) >= 3 else 0.0
        var = max(mad, sigma) ** 2 / len(x)
        p0, m0 = self.scale_var0, self.scale_prior
        if self.calibration is not None:
            # A unit calibration (10 closed turns, ~0.02 %) is far better than in-run
            # samples (~0.1 % each, near views included): they only monitor it, and
            # replace it when >= 5 of them disagree beyond 3 sigma (e.g. temperature).
            if len(x) < 5 or abs(med - m0) <= 3 * math.sqrt(p0 + var):
                return True
            self.log["scale_recalibrated"] = self.log.get("scale_recalibrated", 0) + 1
            self.gyro_scale, self.scale_var = med, max(var, 0.0001 ** 2)
            return True
        self.gyro_scale = m0 + p0 / (p0 + var) * (med - m0)
        self.scale_var = max(p0 * var / (p0 + var), 0.0001 ** 2)
        return True

    def step(self, batch: ProprioBatch, t: float, command=None) -> OdometryStep | None:
        """Consume the samples received since the previous frame (frame time ``t``)."""
        t0 = self.t
        self.t = t
        imu = sorted(batch.imu, key=lambda s: s.t)
        enc = sorted(batch.encoders, key=lambda s: s.t)
        if t0 is None:  # first frame: only initialise the counters
            if enc:
                self._last_enc = enc[-1]
            if imu:
                self._last_imu_t = imu[-1].t
            return None
        dt = max(t - t0, 1e-6)
        cfg = self.cfg
        # ---- rotation (gyro), cumulative yaw at each IMU sample
        yaw_t, yaw_c, yaw = [t0], [0.0], 0.0
        gz_int = imu_time = 0.0  # raw gyro z integral (offset included) for the straight-driving update
        for s in imu:
            h = s.t - self._last_imu_t if self._last_imu_t is not None else 0.0
            self._last_imu_t = s.t
            rate = self._yaw_rate(s, h)
            yaw += rate * h
            gz_int += float(s.gyro[2]) * h
            imu_time += h
            if getattr(s, "temp", None) is not None:
                self.temp = 0.95 * self.temp + 0.05 * float(s.temp) if self.temp is not None else float(s.temp)
                if self.temp0 is None:
                    self.temp0 = self.temp
            yaw_t.append(s.t)
            yaw_c.append(yaw)
        have_imu = len(imu) > 0
        # ---- distance (encoders), integrated along the gyro heading
        x = y = 0.0
        dl_tot = dr_tot = 0.0
        speeds = []
        last = self._last_enc
        for s in enc:
            if last is not None:
                dl = (s.left - last.left) * 2 * math.pi / self.cpr
                dr = (s.right - last.right) * 2 * math.pi / self.cpr
                ds = self.radius_scale * self.r * (dl + dr) / 2
                th = float(np.interp(s.t, yaw_t, yaw_c)) if have_imu else \
                    self.r * (dr_tot + dl_tot * 0 + dr - dl) / self.track  # fallback: encoder heading
                x += ds * math.cos(th)
                y += ds * math.sin(th)
                dl_tot += dl
                dr_tot += dr
                h = s.t - last.t
                if h > 0:
                    speeds.append((s.t, ds / h))
            last = s
        self._last_enc = last
        have_enc = len(enc) > 0
        dist_signed = self.radius_scale * self.r * (dl_tot + dr_tot) / 2
        enc_yaw = self.r * (dr_tot - dl_tot) / self.track
        if not have_imu and not have_enc:
            return None
        if not have_imu:
            yaw = self.track_scale * enc_yaw
        # ---- standing still: zero-velocity update of the gyro bias. Still = no
        # encoder counts for zupt_min_s AND this frame's gyro quiet on all axes (the
        # chassis keeps rocking for a moment after the wheels stop). Each frame's own
        # samples are one independent measurement of the offset.
        moved = abs(dl_tot) > 0 or abs(dr_tot) > 0
        stationary = False
        # Without encoder samples (dropout) stillness is judged from the robot's own
        # zero command plus a quiet gyro (validation 3: with encoders off the offset was
        # never measured and the heading drifted 64 deg in 420 s).
        idle_cmd = command is not None and abs(float(command[0])) < 1e-9 and abs(float(command[1])) < 1e-9
        if (have_enc and not moved) or (not have_enc and have_imu and idle_cmd):
            if self._still_since is None:
                self._still_since = t0
            g_arr = np.array([s_.gyro for s_ in imu]) if imu else np.zeros((0, 3))
            quiet_lim = 4 * self.sample_noise + self.lsb
            quiet = len(g_arr) >= 3 and bool(np.all(g_arr.std(0) < quiet_lim))  # (x/y have own offsets)
            self._quiet_frames = self._quiet_frames + 1 if quiet else 0
            if quiet and t - self._still_since >= cfg.zupt_min_s and self._quiet_frames >= cfg.zupt_quiet_frames:
                mean = float(g_arr[:, 2].mean())
                if abs(mean - self.bias) < max(cfg.zupt_gyro_gate, 4 * math.sqrt(self.bias_var)):
                    stationary = True
                    r_var = (self.sample_noise ** 2 + self.lsb ** 2 / 12) / len(g_arr) + cfg.zupt_floor ** 2
                    self._offset_update(mean, r_var)
                    self.bias_calibrated = True
                    self.last_zupt_t = self.last_bias_update_t = t
                    self.temp_at_update = self.temp
                    self.zupt_frames += 1
                    self.log["zupt"] += 1
                    ax = float(np.mean([s_.accel[0] for s_ in imu]))
                    self.rest_ax = ax if self.rest_ax is None else 0.9 * self.rest_ax + 0.1 * ax
        else:
            self._still_since = None
            self._quiet_frames = 0
        self._P[0, 0] += cfg.bias_rw ** 2 * dt
        self._P[1, 1] += cfg.tempco_rw ** 2 * dt
        # ---- slip detection
        slip_reasons = []
        if have_imu and have_enc and abs(enc_yaw) > 1e-4:
            if abs(yaw) > 0.02 and not self._slip_active(t):
                ratio = yaw / enc_yaw
                lo, hi = cfg.track_scale_bounds
                if lo <= ratio <= hi:
                    self.track_scale += cfg.track_scale_gain * (ratio - self.track_scale)
            res = yaw - self.track_scale * enc_yaw
            sig = cfg.yaw_slip_rel * abs(yaw) + cfg.yaw_slip_abs
            if abs(res) > cfg.yaw_slip_k * sig / 2:
                slip_reasons.append("yaw")
                self.log["slip_yaw"] += 1
        if have_imu and len(speeds) >= 2:
            if self._accel_check(imu, speeds):
                slip_reasons.append("accel")
                self.log["slip_accel"] += 1
        if command is not None and have_enc:
            v_cmd = abs(float(command[0]))
            # the wheels turned much faster than any commanded speed: spinning
            if abs(dist_signed) / dt > 1.6 * v_cmd + 0.08 and v_cmd > 0.0:
                slip_reasons.append("command")
                self.log["slip_cmd"] += 1
        if slip_reasons:
            self._slip_until = t + cfg.slip_hold_s
        slip = self._slip_active(t)
        # ---- straight driving: zero-rotation update of the gyro offset (gyrodometry).
        # Wheels turning (nearly) equally and no slip: the body's yaw is the encoders'
        # (small) differential yaw, so the raw gyro integral minus it, over >= 1 s of
        # such frames, measures the offset. Encoder quantisation (1.4 mrad per count)
        # limits one window to ~0.03 dps; repeated windows average it down. No stop needed.
        straight = (cfg.zaru and have_imu and have_enc and moved and not slip and imu_time > 0
                    and abs(enc_yaw) <= cfg.zaru_max_frame_yaw
                    and abs(dist_signed / dt - self._last_v) <= cfg.zaru_max_dv)
        self._last_v = dist_signed / dt if have_enc else 0.0
        self._straight_s = self._straight_s + dt if straight else 0.0
        if straight and self._straight_s < cfg.zaru_settle_s:
            self._zaru = None
        elif straight:
            w = self._zaru or [0.0, 0.0, 0.0]
            w[0] += gz_int
            w[1] += self.track_scale * enc_yaw / max(self.gyro_scale, 0.5)
            w[2] += imu_time
            self._zaru = w
            if w[2] >= cfg.zaru_window_s:
                self._zaru_update(*w)
                self._zaru = None
        else:
            if self._zaru is not None and self._zaru[2] >= cfg.zaru_min_s:
                self._zaru_update(*self._zaru)
            self._zaru = None
        # ---- covariance (robot frame at the previous frame)
        dist = abs(dist_signed)
        if have_imu:
            var_th = (self.noise_density ** 2) * dt + self.bias_var * dt ** 2 \
                + ((math.sqrt(self.scale_var) + cfg.tilt_yaw_error) * abs(yaw)) ** 2 + (self.lsb * dt) ** 2
            rot_src = "gyro"
        else:
            var_th = (0.15 * abs(yaw) + 0.01) ** 2
            rot_src = "encoders"
        if have_enc:
            rel = math.hypot(math.sqrt(self.radius_var), cfg.slip_floor)
            var_s = (rel * dist) ** 2 + (self.r * 2 * math.pi / self.cpr) ** 2
            if slip:
                var_s += (dist + 0.03) ** 2
            trans_src = "encoders"
        else:
            v_cmd = 0.0 if command is None else float(command[0])
            x, y = v_cmd * dt * math.cos(yaw / 2), v_cmd * dt * math.sin(yaw / 2)
            dist = abs(v_cmd) * dt
            var_s = (0.5 * dist + 0.01) ** 2
            trans_src = "command"
        c, s_ = math.cos(yaw / 2), math.sin(yaw / 2)
        var_lat = (dist ** 2) * var_th + (0.002 + 0.01 * abs(yaw) * 0.125) ** 2 + (0.0 if not slip else (0.3 * dist) ** 2)
        R = np.array([[c, -s_], [s_, c]])
        cov = np.zeros((3, 3))
        cov[:2, :2] = R @ np.diag([var_s, var_lat]) @ R.T
        cov[2, 2] = var_th
        v = (dist_signed if have_enc else (0.0 if command is None else float(command[0])) * dt) / dt
        return OdometryStep(dt, float(x), float(y), float(yaw), cov, float(v), float(yaw / dt), dist,
                            slip=slip, stationary=stationary, rotation_source=rot_src,
                            translation_source=trans_src,
                            info={"bias": self.bias, "bias_sigma": math.sqrt(self.bias_var), "temp": self.temp,
                                  "tempco": self.tempco[0],
                                  "track_scale": self.track_scale, "gyro_scale": self.gyro_scale,
                                  "radius_scale": self.radius_scale,
                                  "scale_sigma": math.sqrt(self.scale_var), "enc_yaw": enc_yaw,
                                  "slip_reasons": slip_reasons})

    def _zaru_update(self, gz_int, enc_yaw_raw, T):
        meas = (gz_int - enc_yaw_raw) / T
        q = self.r * 2 * math.pi / self.cpr / self.track  # one count of differential yaw
        r_var = (2 * q / T) ** 2 + (self.cfg.zaru_floor / math.sqrt(T)) ** 2
        # Equal wheel travel does not guarantee no rotation: in simulation the chassis
        # skids on caster and wheels (~0.2 dps while "straight", more when starting
        # off). Only windows consistent with how far the offset can have drifted are used.
        if abs(meas - self.bias) > 3 * math.sqrt(self.bias_var + r_var) + self.cfg.zaru_gate:
            self.log["zaru_rejected"] = self.log.get("zaru_rejected", 0) + 1
            return
        self._offset_update(meas, r_var)
        self.last_bias_update_t = self.t
        self.log["zaru"] = self.log.get("zaru", 0) + 1

    def _slip_active(self, t):
        return t < self._slip_until

    def _accel_check(self, imu, speeds):
        """Forward specific force (gravity removed with the tilt estimate) vs the
        wheels' acceleration, both low-passed over ~30 ms."""
        st = np.array([s[0] for s in speeds])
        sv = np.array([s[1] for s in speeds])
        self._v_hist = (self._v_hist + list(zip(st, sv)))[-40:]
        ht = np.array([h[0] for h in self._v_hist])
        hv = np.array([h[1] for h in self._v_hist])
        if len(ht) < 5:
            return False
        bad = 0
        for s in imu:
            sel = np.abs(ht - s.t) <= 0.025
            if sel.sum() < 3:
                continue
            a_wheel = np.polyfit(ht[sel] - s.t, hv[sel], 1)[0]
            a_imu = float(s.accel[0]) - (self.rest_ax or 0.0)
            if abs(a_imu - a_wheel) > self.cfg.accel_slip:
                self._accel_bad += 1
                if self._accel_bad >= self.cfg.accel_slip_samples:
                    bad += 1
            else:
                self._accel_bad = 0
        return bad > 0
