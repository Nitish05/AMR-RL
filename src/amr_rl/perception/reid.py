"""Appearance embeddings for re-identification (perception swap, step 5b; learned).

The crop of a detected object is embedded with DINOv2-S (self-supervised ViT,
Apache-2.0, pinned revision, local cache only); identity compares embeddings by
cosine similarity instead of the colour detector's hue/fill cue. Optional: without
torch/transformers or the weights, ``load_embedder`` returns None.
"""

from __future__ import annotations

import numpy as np

MODEL_ID = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MODEL_LICENSE = "apache-2.0"


class CropEmbedder:
    def __init__(self, device="cpu", pad=0.1, size=224):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.pad = pad
        self.size = size
        self.net = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).to(device).eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def _crop(self, rgb, box):
        import cv2

        H, W = rgb.shape[:2]
        x0, y0, x1, y1 = (float(v) for v in box)
        px, py = self.pad * (x1 - x0), self.pad * (y1 - y0)
        x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
        x1, y1 = int(min(W, x1 + px)), int(min(H, y1 + py))
        crop = rgb[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
        crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        return ((crop - self.mean) / self.std).transpose(2, 0, 1)

    def embed(self, rgb, box) -> np.ndarray:
        """Unit-norm CLS embedding (384-d) of the padded crop."""
        return self.embed_many(rgb, [box])[0]

    def embed_many(self, rgb, boxes) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, 384), np.float32)
        x = self.torch.from_numpy(np.stack([self._crop(rgb, b) for b in boxes])).to(self.device)
        with self.torch.inference_mode():
            out = self.net(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy()
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def load_embedder(**kwargs):
    try:
        return CropEmbedder(**kwargs)
    except (ImportError, OSError) as exc:
        print(f"re-identification embedder unavailable: {exc}")
        return None
