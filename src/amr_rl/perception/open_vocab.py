"""Open-vocabulary object detector behind the ``Detection`` interface (perception swap,
step 5b; docs/results/textured-worlds.md). Onboard RGB only.

A pretrained open-vocabulary detector proposes boxes for a few object words. What
the detector does NOT provide is engineered, stated plainly:

* objects must stand on the floor: the box's bottom edge is the floor contact and
  must project onto the floor through the calibrated camera (wall posters and
  things above the horizon are rejected);
* the visible state ("attachment"): with ``flag_mode="detect"`` the detector is also
  asked for "flag" in the same pass, and a flag box resting on an object's top
  (geometric rule) is that object's attachment; its colour is the majority of
  saturated yellow vs red pixels inside the flag box (flags stay yellow/red by
  design, so the outcome tokens keep their meaning). ``flag_mode="band"`` reads the
  colour share in a band directly above the object's box instead;
* the floor-contact offset to the object's centre comes from the detector's word
  ("ball": none, else half the width) instead of the colour detector's mask fill.

Models are pinned (``MODELS``), loaded from the local cache only, and optional:
without them ``load_open_vocab`` returns None and the runtime keeps the colour
detector.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .entities import Attachment, Detection, circular_mean_deg, hue_name

MODELS = {
    "llmdet": ("iSEE-Laboratory/llmdet_tiny", "d05199165a19320a9236396e20fca0a5e065189c", "apache-2.0"),
    "omdet": ("omlab/omdet-turbo-swin-tiny-hf", "7fe93cecfb770c4d76cf71163956221249cab566", "apache-2.0"),
}
DEFAULT_LABELS = ("box", "cylinder", "ball")


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-9)


class OpenVocabDetector:
    # Defaults frozen on the arena_textured design capture (docs/results/textured-worlds.md)
    # before the held-out home_a_textured capture was scored. LLMDet-tiny was tried
    # first and rejected (2-11 % of objects boxed at IoU >= 0.5 on 320x240 frames).
    def __init__(self, model, *, backend="omdet", device="cpu", labels=DEFAULT_LABELS, threshold=0.25,
                 upscale=1.0, nms_iou=0.5, flag_band=0.7, flag_min_share=0.12, max_range=3.0, min_box_px=6,
                 flag_mode="detect", flag_label="flag", flag_threshold=0.15, flag_color_share=0.25):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        repo, rev, _ = MODELS[backend]
        self.torch = torch
        self.model = model
        self.backend = backend
        self.device = device
        self.labels = list(labels)
        self.threshold = threshold
        self.upscale = upscale
        self.nms_iou = nms_iou
        self.flag_band = flag_band
        self.flag_min_share = flag_min_share
        self.max_range = max_range
        self.min_box_px = min_box_px
        self.flag_mode = flag_mode
        self.flag_label = flag_label
        self.flag_threshold = threshold if flag_threshold is None else flag_threshold
        self.flag_color_share = flag_color_share
        self.processor = AutoProcessor.from_pretrained(repo, revision=rev, local_files_only=True)
        self.net = AutoModelForZeroShotObjectDetection.from_pretrained(
            repo, revision=rev, local_files_only=True).to(device).eval()

    # ------------------------------------------------------------ proposals
    def proposals(self, rgb):
        """[(box [x0, y0, x1, y1] in pixels, score, label)] after class-agnostic NMS."""
        from PIL import Image

        H, W = rgb.shape[:2]
        im = Image.fromarray(rgb)
        if self.upscale != 1.0:
            im = im.resize((int(W * self.upscale), int(H * self.upscale)), Image.BICUBIC)
        labels = self.labels + ([self.flag_label] if self.flag_mode == "detect" else [])
        if self.backend == "omdet":
            inputs = self.processor(images=im, text=labels, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(images=im, text=[labels], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.net(**inputs)
        size = [(H, W)]
        floor = min(self.threshold, self.flag_threshold)
        if self.backend == "omdet":
            res = self.processor.post_process_grounded_object_detection(
                out, text_labels=[labels], target_sizes=size, threshold=floor, nms_threshold=self.nms_iou)[0]
            labels = res.get("text_labels", res.get("classes"))
        else:
            res = self.processor.post_process_grounded_object_detection(
                out, threshold=floor, target_sizes=size, text_labels=[labels])[0]
            labels = res.get("text_labels", res.get("labels"))
        boxes = res["boxes"].float().cpu().numpy()
        scores = res["scores"].float().cpu().numpy()
        order = np.argsort(-scores)
        kept = []
        for i in order:
            b = np.clip(boxes[i], [0, 0, 0, 0], [W, H, W, H])
            label = str(labels[i])
            if scores[i] < (self.flag_threshold if label == self.flag_label else self.threshold):
                continue
            if b[2] - b[0] < self.min_box_px or b[3] - b[1] < self.min_box_px:
                continue
            same = [k for k in kept if (k[2] == self.flag_label) == (label == self.flag_label)]
            if any(box_iou(b, k[0]) > self.nms_iou for k in same):  # NMS within objects / within flags
                continue
            kept.append((b, float(scores[i]), label))
        return kept

    # ------------------------------------------------------------ detection
    def detect(self, rgb: np.ndarray, pose=None) -> list[Detection]:
        H, W = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        detections = []
        props = self.proposals(rgb)
        flags = [p for p in props if p[2] == self.flag_label]
        for box, score, label in props:
            if label == self.flag_label:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            det = Detection(index=len(detections), bbox=(x0, y0, x1, y1), area=(x1 - x0) * (y1 - y0),
                            hue=self._hue(hsv, x0, y0, x1, y1), color="", saturation=0.0,
                            fill=0.0 if label == "ball" else 1.0, base_uv=((x0 + x1) / 2, float(y1 - 1)),
                            partial=bool(x0 <= 1 or y0 <= 1 or x1 >= W - 1 or y1 >= H - 1))
            det.color = hue_name(det.hue)
            det.label, det.score = label, score
            if not self._localize(det, pose):
                continue  # no floor contact: a poster, a wall feature, or the floor edge cut off
            flag = self._flag_box(hsv, flags, x0, y0, x1, y1) if self.flag_mode == "detect" \
                else self._flag(hsv, x0, y0, x1, y1)
            if flag is not None:
                det.attachments.append(flag)
            det.index = len(detections)
            detections.append(det)
        return detections

    @staticmethod
    def _hue(hsv, x0, y0, x1, y1):
        crop = hsv[y0:y1, x0:x1].reshape(-1, 3)
        sat = crop[crop[:, 1] > 80]
        return circular_mean_deg(sat[:, 0] * 2.0) if len(sat) else 0.0

    def _flag_box(self, hsv, flags, x0, y0, x1, y1):
        """The best-scoring flag box resting on this object's top, coloured by its pixels."""
        h = y1 - y0
        best = None
        for (fx0, fy0, fx1, fy1), score, _ in flags:
            overlap = min(fx1, x1) - max(fx0, x0)
            if overlap < 0.5 * (fx1 - fx0) or not (y0 - 0.35 * h <= fy1 <= y0 + 0.6 * h) or fy0 > y0 + 0.2 * h:
                continue
            if best is None or score > best[1]:
                best = ((fx0, fy0, fx1, fy1), score)
        if best is None:
            return None
        fx0, fy0, fx1, fy1 = (int(round(v)) for v in best[0])
        crop = hsv[fy0:fy1, fx0:fx1].reshape(-1, 3)
        if len(crop) == 0:
            return None
        hue = crop[:, 0].astype(np.float32) * 2.0
        strong = (crop[:, 1] >= 130) & (crop[:, 2] >= 60)
        shares = {"yellow": (strong & (hue >= 40) & (hue < 70)).mean(),
                  "red": (strong & ((hue < 15) | (hue >= 345))).mean()}
        color = max(shares, key=shares.get)
        if shares[color] < self.flag_color_share:
            return None
        return Attachment(hue=55.0 if color == "yellow" else 0.0, color=color,
                          area=int(shares[color] * len(crop)), bbox=(fx0, fy0, fx1, fy1))

    def _flag(self, hsv, x0, y0, x1, y1):
        """Saturated yellow/red share in the band directly above the object's box."""
        h = max(2, int(self.flag_band * (y1 - y0)))
        top = max(0, y0 - h)
        if y0 - top < 2:
            return None
        band = hsv[top:y0 + max(1, (y1 - y0) // 10), x0:x1].reshape(-1, 3)
        if len(band) == 0:
            return None
        hue = band[:, 0].astype(np.float32) * 2.0
        strong = (band[:, 1] >= 130) & (band[:, 2] >= 60)
        yellow = strong & (hue >= 40) & (hue < 70)
        red = strong & ((hue < 15) | (hue >= 345))
        shares = {"yellow": yellow.mean(), "red": red.mean()}
        color = max(shares, key=shares.get)
        if shares[color] < self.flag_min_share:
            return None
        return Attachment(hue=55.0 if color == "yellow" else 0.0, color=color,
                          area=int(shares[color] * len(band)), bbox=(x0, top, x1, y0))

    def _localize(self, det: Detection, pose) -> bool:
        """Floor contact -> range/bearing/size, and map position when a pose is given."""
        if det.partial and det.bbox[3] >= self.model.height - 1:
            return False
        base_pose = (0.0, 0.0, 0.0)
        P, ok = self.model.ground_points_world(np.array([det.base_uv]), base_pose, max_range=self.max_range)
        if not ok[0]:
            return False
        cam = self.model.T_world_cam(base_pose)[:3, 3]
        contact = P[0]
        slant = float(np.linalg.norm(contact - cam))
        x0, y0, x1, y1 = det.bbox
        det.width_m = (x1 - x0) * slant / self.model.K[0, 0]
        det.height_m = (y1 - y0) * slant / self.model.K[1, 1]
        det.contact_range_m = float(np.linalg.norm(contact[:2]))
        ray = contact[:2] - cam[:2]
        ray /= max(np.linalg.norm(ray), 1e-6)
        offset = 0.0 if det.fill < 0.5 else det.width_m / 2  # from the detector's word, see module doc
        centre = contact[:2] + ray * offset
        det.range_m = float(np.linalg.norm(centre))
        if det.range_m > self.max_range:
            return False  # e.g. a box around a whole wall: its "centre" lies outside any room
        det.bearing = math.atan2(centre[1], centre[0])
        if pose is not None:
            c, s = math.cos(pose[2]), math.sin(pose[2])
            det.position = (pose[0] + c * centre[0] - s * centre[1], pose[1] + s * centre[0] + c * centre[1])
        return True


def load_open_vocab(model, device="auto", **kwargs):
    """The detector, or None when torch/transformers or the pinned weights are missing.
    ``device="auto"``: Apple MPS when available (0.08 s/frame), else CPU (~0.9 s/frame)."""
    try:
        if device == "auto":
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cpu"
        return OpenVocabDetector(model, device=device, **kwargs)
    except (ImportError, OSError) as exc:  # not installed / not cached: the runtime keeps the colour detector
        print(f"open-vocabulary detector unavailable: {exc}")
        return None
