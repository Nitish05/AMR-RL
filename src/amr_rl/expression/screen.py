"""Deterministic face framebuffer for the robot's screen.

``render(expr, size=(width, height))`` draws an :class:`ExpressionState` into an
``uint8`` RGB array of shape ``(height, width, 3)``. It uses only the fields of
``expr`` (no clock, no randomness), so identical expressions give identical
pixels. Drawing happens at ``_SS``x resolution and is area-downsampled for
smooth edges at the native 192x132.

Visual element -> ExpressionState field (see ``docs/EXPRESSION.md``):

* eye shapes / brows / glyphs -> ``face``
* eyelid height               -> ``openness``
* pupil and eye offset        -> ``gaze`` (robot frame: +x = robot's right, which is
  the *viewer's left* on the outward-facing screen, so +gaze_x moves pupils to
  image -x; +gaze_y = up = image -y)
* bottom bar ('?' + fill)     -> ``uncertainty``
* top-right pip               -> ``attitude``
* concentric ring pattern     -> ``signal_pattern`` (drawn iff True)
"""

from __future__ import annotations

import cv2
import numpy as np

from amr_rl.expression.policy import ExpressionState

_SS = 4  # supersampling factor

# RGB colours (engineered display constants)
BG = (10, 14, 22)
EYE = (120, 225, 255)
EYE_DIM = (70, 130, 150)
PUPIL = (8, 20, 34)
GLINT = (235, 250, 255)
BROW = (255, 176, 64)
STOP_RED = (235, 40, 40)
TRACK = (38, 46, 60)
TEXT = (150, 165, 185)
RING_A = (255, 214, 0)
RING_B = (255, 40, 170)
ATTITUDE_COLOURS = {
    "liked": (60, 210, 90),
    "disliked": (230, 50, 50),
    "indifferent": (140, 145, 155),
}


def _lerp(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


class _Canvas:
    """Supersampled drawing helpers; all public coords are in output pixels."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.img = np.zeros((h * _SS, w * _SS, 3), np.uint8)
        self.img[:] = BG

    @staticmethod
    def p(x: float, y: float) -> tuple[int, int]:
        return round(x * _SS), round(y * _SS)

    @staticmethod
    def s(v: float) -> int:
        return max(1, round(v * _SS))

    def mask(self) -> np.ndarray:
        return np.zeros(self.img.shape[:2], np.uint8)

    def fill(self, mask: np.ndarray, colour) -> None:
        self.img[mask > 0] = colour

    def rounded_rect(self, m, cx, cy, w, h, r, val=255):
        r = min(r, w / 2, h / 2)
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        cv2.rectangle(m, self.p(x0 + r, y0), self.p(x1 - r, y1), val, -1)
        cv2.rectangle(m, self.p(x0, y0 + r), self.p(x1, y1 - r), val, -1)
        for ex, ey in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(m, self.p(ex, ey), self.s(r), val, -1, cv2.LINE_AA)

    def poly(self, m, pts, val=255):
        arr = np.array([self.p(x, y) for x, y in pts], np.int32)
        cv2.fillPoly(m, [arr], val, cv2.LINE_AA)

    def line(self, a, b, colour, t):
        cv2.line(self.img, self.p(*a), self.p(*b), colour, self.s(t), cv2.LINE_AA)

    def arc(self, c, axes, a0, a1, colour, t):
        cv2.ellipse(self.img, self.p(*c), (self.s(axes[0]), self.s(axes[1])), 0, a0, a1,
                    colour, self.s(t), cv2.LINE_AA)

    def circle(self, c, r, colour, t=-1):
        cv2.circle(self.img, self.p(*c), self.s(r), colour, -1 if t < 0 else self.s(t),
                   cv2.LINE_AA)

    def text(self, s, org, scale, colour, thick):
        cv2.putText(self.img, s, self.p(*org), cv2.FONT_HERSHEY_SIMPLEX, scale * _SS, colour,
                    self.s(thick), cv2.LINE_AA)

    def out(self) -> np.ndarray:
        return cv2.resize(self.img, (self.w, self.h), interpolation=cv2.INTER_AREA)


def _superellipse(cx, cy, a, b, n=3.2, steps=72):
    t = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    x = cx + a * np.sign(ct) * np.abs(ct) ** (2 / n)
    y = cy + b * np.sign(st) * np.abs(st) ** (2 / n)
    return list(zip(x, y))


def _eye(c: _Canvas, cx, cy, ew, eh, openness, gaze, lid_tilt=0.0, pupil_scale=1.0,
         colour=EYE, outline=False):
    """Squircle eye with an upper lid; ``lid_tilt`` slants the lid (px, + = viewer-right up)."""
    eye = c.mask()
    c.poly(eye, _superellipse(cx, cy, ew / 2, eh / 2))
    closed = (1.0 - float(np.clip(openness, 0.0, 1.0))) * eh
    top = cy - eh / 2 + closed
    if closed > 0.5:
        # lid edge: slight downward sag in the middle, optional slant
        xs = np.linspace(cx - ew / 2 - 2, cx + ew / 2 + 2, 24)
        u = (xs - cx) / (ew / 2 + 2)
        ys = top + min(closed, eh * 0.12) * 0.35 * (1 - u * u) - lid_tilt * u
        big = 2 * max(ew, eh)
        pts = [(cx - big, cy - big), (cx + big, cy - big), (cx + big, ys[-1])]
        pts += list(zip(xs[::-1], ys[::-1])) + [(cx - big, ys[0])]
        lid = c.mask()
        c.poly(lid, pts)
        eye[lid > 0] = 0
    if outline:
        ring = c.mask()
        cv2.dilate(eye, np.ones((3 * _SS, 3 * _SS), np.uint8), dst=ring)
        c.fill(ring, BG)
    c.fill(eye, colour)
    pr = min(ew, eh) * 0.23 * pupil_scale
    visible_h = max(eh - closed, 1.0)
    py = top + visible_h * 0.55 if closed > 0.5 else cy
    px = cx + gaze[0] * (ew / 2 - pr - 3)
    py = min(py + gaze[1] * (eh / 2 - pr - 3), cy + eh / 2 - pr - 2)
    pup = c.mask()
    cv2.circle(pup, c.p(px, py), c.s(pr), 255, -1, cv2.LINE_AA)
    c.fill(np.minimum(pup, eye), PUPIL)
    glint = c.mask()
    cv2.circle(glint, c.p(px + pr * 0.38, py - pr * 0.38), c.s(max(pr * 0.3, 1.2)), 255, -1,
               cv2.LINE_AA)
    c.fill(np.minimum(glint, eye), GLINT)


def _rings(c: _Canvas, cx, cy, rmax):
    """Static high-contrast concentric rings (the perceivable signal pattern)."""
    band = rmax / 7.0
    for i in range(7, 0, -1):
        c.circle((cx, cy), band * i, RING_A if i % 2 else RING_B)
    c.circle((cx, cy), band * 0.6, (255, 255, 255))


def _uncertainty_bar(c: _Canvas, u: float):
    y, bh = c.h - 9, 5
    x0, x1 = 17, c.w - 22
    c.text("?", (5, y + 5.5), 0.36, TEXT, 1.1)
    m = c.mask()
    c.rounded_rect(m, (x0 + x1) / 2, y + bh / 2, x1 - x0, bh, bh / 2)
    c.fill(m, TRACK)
    u = float(np.clip(u, 0.0, 1.0))
    if u > 0.005:
        wfill = max((x1 - x0) * u, bh)
        m = c.mask()
        c.rounded_rect(m, x0 + wfill / 2, y + bh / 2, wfill, bh, bh / 2)
        c.fill(m, _lerp((70, 200, 170), (255, 170, 40), u))


def _attitude_pip(c: _Canvas, att: str):
    centre, r = (c.w - 10, 10), 4.5
    if att in ATTITUDE_COLOURS:
        c.circle(centre, r, ATTITUDE_COLOURS[att])
    elif att == "unknown":
        c.circle(centre, r, (235, 235, 235), 1.2)


def render(expr: ExpressionState, size: tuple[int, int] = (192, 132)) -> np.ndarray:
    """Draw ``expr``; ``size`` is (width, height); returns uint8 RGB (height, width, 3)."""
    w, h = int(size[0]), int(size[1])
    c = _Canvas(w, h)
    face = expr.face
    # ExpressionState.gaze is in the robot's frame: +x = robot's right, +y = up.
    # The screen faces outward and is viewed unmirrored, so the robot's right is
    # the viewer's left (image -x) and up is image -y. Convert to image offsets.
    gx = -float(np.clip(expr.gaze[0], -1, 1))
    gy = -float(np.clip(expr.gaze[1], -1, 1))
    open_ = float(np.clip(expr.openness, 0, 1))
    k = min(w / 192.0, h / 132.0)
    cy = h * 0.44
    sep = 44 * k
    ew, eh = 48 * k, 56 * k
    shift = gx * 7 * k  # the whole face drifts a little toward the gaze (image px)
    lx, rx = w / 2 - sep + shift, w / 2 + sep + shift

    if expr.signal_pattern:
        _rings(c, w / 2 + shift, cy, min(h * 0.46, w * 0.4))

    if face == "happy":
        for ex in (lx, rx):
            c.arc((ex, cy + 8 * k), (ew * 0.46, eh * 0.36), 190, 350, EYE, 7 * k)
    elif face == "stopped":
        for ex in (lx, rx):
            c.line((ex - ew * 0.45, cy), (ex + ew * 0.45, cy), EYE_DIM, 6 * k)
        s = 7 * k
        m = c.mask()
        c.rounded_rect(m, w / 2, cy + 24 * k, 2 * s, 2 * s, 1.5 * k)
        c.fill(m, STOP_RED)
    elif face == "confused":
        _eye(c, lx, cy + 2 * k, ew, eh, 1.0, (gx, gy), lid_tilt=0)
        _eye(c, rx, cy + 6 * k, ew * 0.8, eh * 0.8, open_ * 0.7, (gx, gy),
             lid_tilt=-6 * k, pupil_scale=0.9)
        c.line((rx - ew * 0.4, cy - eh * 0.52), (rx + ew * 0.35, cy - eh * 0.36), BROW, 3.5 * k)
        c.text("?", (w / 2 - 6 * k, cy - eh * 0.25), 0.9 * k, BROW, 2.6 * k)
    elif face == "wary":
        tilt = 9 * k
        _eye(c, lx, cy + 4 * k, ew, eh * 0.9, open_ + 0.12, (gx, gy), lid_tilt=tilt)
        _eye(c, rx, cy + 4 * k, ew, eh * 0.9, open_ + 0.12, (gx, gy), lid_tilt=-tilt)
        top = cy + 4 * k - eh * 0.45 + (1 - open_ - 0.12) * eh * 0.9
        # slanted brows, inner ends low (lx is viewer-left eye; its inner end is +x)
        c.line((lx - ew * 0.55, top - 15 * k), (lx + ew * 0.5, top - 5 * k), BROW, 4 * k)
        c.line((rx + ew * 0.55, top - 15 * k), (rx - ew * 0.5, top - 5 * k), BROW, 4 * k)
    elif face == "sleepy":
        for ex in (lx, rx):
            _eye(c, ex, cy + 6 * k, ew, eh * 0.85, open_, (gx, 0.3), colour=EYE_DIM)
        c.text("z", (w - 38 * k, cy - 22 * k), 0.45 * k, EYE_DIM, 1.4 * k)
        c.text("z", (w - 30 * k, cy - 30 * k), 0.32 * k, EYE_DIM, 1.2 * k)
    elif face == "focused":
        for ex in (lx, rx):
            _eye(c, ex, cy, ew * 0.82, eh * 0.78, open_ + 0.2, (gx, gy), pupil_scale=1.25)
        for ex in (lx, rx):
            c.line((ex - ew * 0.38, cy - eh * 0.5), (ex + ew * 0.38, cy - eh * 0.5), BROW,
                   3 * k)
    elif face == "curious":
        _eye(c, lx, cy, ew, eh, open_, (gx, gy), pupil_scale=1.1)
        _eye(c, rx, cy - 2 * k, ew * 1.08, eh * 1.08, open_, (gx, gy), pupil_scale=1.1)
        c.arc((rx, cy - eh * 0.62), (ew * 0.42, 8 * k), 200, 340, BROW, 3 * k)
    elif face == "signalling":
        for ex in (lx, rx):
            _eye(c, ex, cy, ew * 0.72, eh * 0.72, open_, (gx, gy), outline=True)
    else:  # neutral and anything unrecognised
        for ex in (lx, rx):
            _eye(c, ex, cy, ew, eh, open_, (gx, gy))

    _uncertainty_bar(c, expr.uncertainty)
    _attitude_pip(c, expr.attitude)
    return c.out()


def render_png(expr: ExpressionState, size: tuple[int, int] = (192, 132)) -> bytes:
    """PNG bytes of :func:`render` (for ``GET /api/screen.png``)."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(render(expr, size), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
