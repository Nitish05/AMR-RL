"""Deterministic procedural textures for the simulated rooms (sim-only).

Floors and walls are deliberately desaturated (HSV saturation below ~0.35) so the
fixture detector's saturated-colour cue is an engineered, documented property of
the environment, not something the detector learned. Textures are rich in corners
because monocular feature tracking needs texture; a textureless room is a
documented failure mode, not something these assets hide.
"""

from __future__ import annotations

import cv2
import numpy as np


def _noise(rng, shape, scales=(8, 32, 128), weights=(0.5, 0.3, 0.2)):
    h, w = shape
    total = np.zeros(shape, np.float32)
    for scale, weight in zip(scales, weights):
        small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
        total += weight * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    total -= total.min()
    return total / max(1e-6, total.max())


def _desaturate(rgb: np.ndarray, max_s=0.33) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.minimum(hsv[..., 1], int(255 * max_s))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


PALETTES = {
    "warm": [(170, 140, 110), (120, 95, 75), (200, 180, 150), (90, 75, 65)],
    "cool": [(140, 150, 160), (95, 105, 120), (185, 190, 195), (70, 78, 90)],
    "neutral": [(150, 150, 145), (110, 108, 104), (195, 192, 186), (80, 80, 78)],
}


def floor_texture(kind: str, seed: int, size: int = 4096, palette: str = "warm") -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = np.array(PALETTES[palette], np.float32)
    if kind == "wood":
        img = np.zeros((size, size, 3), np.float32)
        plank_w = size // 16
        for i in range(16):
            base = colors[rng.integers(0, 2)] * rng.uniform(0.85, 1.15)
            grain = _noise(rng, (size, plank_w), scales=(4, 16, 64))
            streak = np.sin(np.linspace(0, rng.uniform(20, 60), size))[:, None] * 0.08
            tone = (0.75 + 0.5 * grain + streak)[..., None]
            img[:, i * plank_w:(i + 1) * plank_w] = base * tone
            # staggered plank joints
            for y in rng.integers(0, size, 2):
                img[y:y + 3, i * plank_w:(i + 1) * plank_w] *= 0.55
            img[:, i * plank_w:i * plank_w + 3] *= 0.55
        for _ in range(size * size // 4000):  # knots/specks: good corners
            x, y = rng.integers(0, size, 2)
            r = int(rng.integers(2, 7))
            cv2.circle(img, (int(x), int(y)), r, tuple(float(c) for c in colors[3] * 0.8), -1)
    elif kind == "speckle":
        base = colors[2]
        img = np.ones((size, size, 3), np.float32) * base
        img *= (0.8 + 0.4 * _noise(rng, (size, size)))[..., None]
        for _ in range(size * size // 1500):
            x, y = rng.integers(0, size, 2)
            color = colors[rng.integers(0, 4)] * rng.uniform(0.6, 1.1)
            cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 6)), tuple(map(float, color)), -1)
    elif kind == "tiles":
        img = np.ones((size, size, 3), np.float32) * colors[2]
        tile = size // 12
        for i in range(12):
            for j in range(12):
                shade = rng.uniform(0.75, 1.1) * (colors[0] if (i + j) % 2 else colors[2])
                img[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile] = shade
        img *= (0.85 + 0.3 * _noise(rng, (size, size), scales=(3, 12, 48)))[..., None]
        for k in range(13):
            img[k * tile:k * tile + 4, :] *= 0.5
            img[:, k * tile:k * tile + 4] *= 0.5
        for _ in range(size * size // 5000):
            x, y = rng.integers(0, size, 2)
            cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 4)), tuple(map(float, colors[3])), -1)
    else:
        raise ValueError(f"Unknown floor texture {kind}")
    # fine grain so close-range views keep trackable corners
    img *= (0.9 + 0.2 * rng.random((size, size), dtype=np.float32))[..., None]
    return _desaturate(np.clip(img, 0, 255).astype(np.uint8))


def wall_texture(seed: int, width: int = 2048, height: int = 512, palette: str = "warm",
                 posters: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = np.array(PALETTES[palette], np.float32)
    img = np.ones((height, width, 3), np.float32) * colors[2]
    img *= (0.85 + 0.3 * _noise(rng, (height, width), scales=(4, 24, 96)))[..., None]
    # skirting board: strong horizontal edge near the floor
    img[int(height * 0.9):] = colors[3] * 0.9
    for _ in range(posters):
        pw, ph = int(rng.integers(width // 14, width // 6)), int(rng.integers(height // 5, height // 2))
        x, y = int(rng.integers(0, width - pw)), int(rng.integers(height // 12, int(height * 0.8) - ph))
        img[y:y + ph, x:x + pw] = colors[rng.integers(0, 4)] * rng.uniform(0.6, 1.2)
        for _ in range(int(rng.integers(10, 30))):
            cx, cy = x + int(rng.integers(0, pw)), y + int(rng.integers(0, ph))
            color = tuple(float(c) for c in colors[rng.integers(0, 4)] * rng.uniform(0.3, 1.3))
            if rng.random() < 0.5:
                cv2.circle(img, (cx, cy), int(rng.integers(3, 18)), color, -1)
            else:
                cv2.rectangle(img, (cx, cy), (cx + int(rng.integers(4, 30)), cy + int(rng.integers(3, 20))), color, -1)
        cv2.rectangle(img, (x, y), (x + pw, y + ph), tuple(map(float, colors[3] * 0.5)), 3)
    for _ in range(40):  # text-like strokes
        x, y = int(rng.integers(0, width - 60)), int(rng.integers(20, int(height * 0.85)))
        for k in range(int(rng.integers(4, 12))):
            cv2.line(img, (x + 6 * k, y), (x + 6 * k + 3, y - int(rng.integers(4, 10))),
                     tuple(map(float, colors[3] * 0.6)), 2)
    return _desaturate(np.clip(img, 0, 255).astype(np.uint8))


def crate_texture(seed: int, size: int = 512, palette: str = "warm") -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = np.array(PALETTES[palette], np.float32)
    img = np.ones((size, size, 3), np.float32) * colors[1] * rng.uniform(0.9, 1.2)
    img *= (0.75 + 0.5 * _noise(rng, (size, size), scales=(3, 12, 48)))[..., None]
    for k in range(0, size, size // 6):
        img[k:k + 5] *= 0.6
    cv2.rectangle(img, (6, 6), (size - 7, size - 7), tuple(map(float, colors[3] * 0.5)), 10)
    cv2.line(img, (10, 10), (size - 10, size - 10), tuple(map(float, colors[3] * 0.6)), 8)
    for _ in range(120):
        x, y = rng.integers(0, size, 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 6)), tuple(map(float, colors[3])), -1)
    return _desaturate(np.clip(img, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- textured objects
# Perception-swap worlds (docs/results/textured-worlds.md): objects, flags and
# posters that are NOT uniformly painted, so a saturated-colour rule no longer
# identifies them. These textures are deliberately saturated and multi-hue.

OBJECT_HUES = {  # base hue (deg) of a textured object, so objects stay distinguishable
    "cyan": 185, "magenta": 320, "green": 125, "blue": 225, "violet": 275, "orange": 28, "teal": 165,
}


def _hsv_color(h_deg, s, v):
    hsv = np.uint8([[[int(h_deg / 2) % 180, int(255 * s), int(255 * v)]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def object_texture(seed: int, base: str = "cyan", style: str = "stripes", size: int = 512) -> np.ndarray:
    """Saturated multi-hue pattern on a base hue: stripes, checks, blobs or a label band.
    Roughly 40-60 % of the area is the base hue, the rest other hues, white and dark."""
    rng = np.random.default_rng(seed)
    h0 = OBJECT_HUES.get(base, 185)
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = _hsv_color(h0, 0.8, 0.85)
    others = [(h0 + d) % 360 for d in rng.choice([60, 100, 140, 180, 220, 260, 300], size=3, replace=False)]
    palette = [_hsv_color(h, rng.uniform(0.6, 0.95), rng.uniform(0.6, 0.95)) for h in others]
    palette += [(235, 235, 230), (35, 35, 40)]
    if style == "stripes":
        width = int(rng.integers(size // 16, size // 7))
        for k, y in enumerate(range(0, size, 2 * width)):
            cv2.rectangle(img, (0, y), (size, y + width), palette[k % len(palette)], -1)
    elif style == "checker":
        cell = int(rng.integers(size // 10, size // 5))
        for i in range(0, size, cell):
            for j in range(0, size, cell):
                if ((i // cell) + (j // cell)) % 2:
                    cv2.rectangle(img, (j, i), (j + cell, i + cell), palette[int(rng.integers(len(palette)))], -1)
    elif style == "blobs":
        for _ in range(int(rng.integers(14, 26))):
            c = tuple(int(v) for v in rng.integers(0, size, 2))
            cv2.circle(img, c, int(rng.integers(size // 20, size // 7)), palette[int(rng.integers(len(palette)))], -1)
    elif style == "label":
        y0 = int(size * rng.uniform(0.3, 0.45))
        cv2.rectangle(img, (0, y0), (size, y0 + size // 4), palette[-2], -1)
        for k in range(int(rng.integers(5, 10))):
            x = int(rng.integers(0, size - 60))
            cv2.putText(img, "ABCDEFGHKMRSTXZ"[int(rng.integers(15))], (x, y0 + size // 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, palette[k % 3], 6)
        for x in range(0, size, size // 8):
            cv2.line(img, (x, 0), (x + size // 10, y0), palette[-1], 5)
    else:
        raise ValueError(f"unknown object texture style {style!r}")
    noise = 0.85 + 0.3 * _noise(rng, (size, size), scales=(4, 16, 64))
    return np.clip(img.astype(np.float32) * noise[..., None], 0, 255).astype(np.uint8)


def flag_texture(color: str, seed: int, size: int = 256) -> np.ndarray:
    """A yellow or red flag with a pattern (stripes or dots in white/dark); the
    named hue stays dominant (>= ~60 % of the area) so the outcome tokens keep meaning."""
    rng = np.random.default_rng(seed)
    base = _hsv_color(52 if color == "yellow" else 2, 0.95, 0.95 if color == "yellow" else 0.85)
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = base
    mark = (240, 240, 235) if rng.random() < 0.5 else (30, 30, 35)
    if rng.random() < 0.5:
        w = size // 10
        for x in range(-size, size, 4 * w):
            pts = np.array([[x, size], [x + w, size], [x + w + size, 0], [x + size, 0]], np.int32)
            cv2.fillPoly(img, [pts], mark)
    else:
        for y in range(size // 8, size, size // 4):
            for x in range(size // 8, size, size // 4):
                cv2.circle(img, (x, y), size // 14, mark, -1)
    return img


def poster_texture(seed: int, width: int = 512, height: int = 384) -> np.ndarray:
    """A colourful wall poster: saturated blocks, discs and bands of many hues
    (including yellow and red), i.e. distractors for a saturated-colour detector."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = _hsv_color(float(rng.uniform(0, 360)), 0.25, 0.9)
    for _ in range(int(rng.integers(6, 12))):
        color = _hsv_color(float(rng.uniform(0, 360)), rng.uniform(0.7, 1.0), rng.uniform(0.6, 1.0))
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(30, 180)), y + int(rng.integers(20, 120))), color, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(15, 80)), color, -1)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (30, 30, 30), 8)
    return img
