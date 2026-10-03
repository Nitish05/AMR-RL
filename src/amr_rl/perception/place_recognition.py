"""Whole-image place descriptors for keyframe retrieval (loop closure, relocalisation).

Camera-only contract: an onboard RGB image in, an L2-normalised vector out; places
are compared by cosine similarity. No simulator state.

Two descriptors (``make_descriptor``):

* ``megaloc``: MegaLoc (Berton & Masone, 2025; MIT licence), a DINOv2 ViT-B/14
  backbone with optimal-transport (SALAD-style) aggregation, 8448-d. Optional
  dependency (torch, torchvision); code and weights are pinned and loaded only from
  the local caches outside the repository (``scripts/amr.sh fetch-place-model``
  downloads them). LEARNED by its authors on public datasets, not by this robot.
* ``orb_bow``: a bag of binary words. The vocabulary is clustered (k-majority,
  Hamming) from the ORB descriptors of the robot's own saved map, words are
  weighted by inverse frequency over the map's landmarks. No weights at all.
  ENGINEERED: vocabulary size, weighting.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

MEGALOC_REPO = "gmberton/MegaLoc"
MEGALOC_CODE_COMMIT = "5fe0dd697c4a70ba3e23607f6716ab3c606b16db"
MEGALOC_WEIGHTS_REPO = "gberton/MegaLoc"
MEGALOC_WEIGHTS_REVISION = "a0f34722c4297ff787e022433799250180860af7"
MEGALOC_LICENSE = "MIT"

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


class MegaLocDescriptor:
    name = "megaloc"
    licence = MEGALOC_LICENSE

    def __init__(self, *, allow_download=False, device=None):
        import torch

        self.torch = torch
        hub = Path(torch.hub.get_dir()) / f"gmberton_MegaLoc_{MEGALOC_CODE_COMMIT}"
        if not hub.exists():
            if not allow_download:
                raise FileNotFoundError("MegaLoc code not in the local cache: run scripts/amr.sh fetch-place-model")
            torch.hub.list(f"{MEGALOC_REPO}:{MEGALOC_CODE_COMMIT}", trust_repo=True)
        sys.path.insert(0, str(hub))
        try:
            from megaloc_model import MegaLoc  # pinned third-party code (MIT), outside the repo
        finally:
            sys.path.remove(str(hub))
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        weights = hf_hub_download(MEGALOC_WEIGHTS_REPO, "model.safetensors", revision=MEGALOC_WEIGHTS_REVISION,
                                  local_files_only=not allow_download)
        model = MegaLoc()
        model.load_state_dict(load_file(weights))
        if device is None:
            device = os.environ.get("AMR_PLACE_DEVICE", "cpu")  # cpu: deterministic; mps/cuda ~4x faster
        self.device = device
        self.model = model.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def describe(self, rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        # 320x240 onboard frames -> multiples of 14 keeping the aspect ratio (322 x 238)
        img = cv2.resize(rgb, (322, 238), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = ((x - self.mean) / self.std).to(self.device)
        with torch.inference_mode():
            d = self.model(x)[0].float().cpu().numpy()
        return d / max(float(np.linalg.norm(d)), 1e-12)


class OrbBowDescriptor:
    """Bag of binary words from the robot's own map (no external weights)."""

    name = "orb_bow"

    def __init__(self, map_descriptors: np.ndarray, *, words=512, iters=8, seed=0, n_features=900):
        rng = np.random.default_rng(seed)
        data = map_descriptors[rng.choice(len(map_descriptors), min(len(map_descriptors), 30000), replace=False)]
        self.vocab = data[rng.choice(len(data), words, replace=False)].copy()
        bits = np.unpackbits(data, axis=1)
        for _ in range(iters):  # k-majority: Hamming assignment, bitwise majority update
            assign = self._assign(data)
            for k in range(words):
                members = bits[assign == k]
                if len(members):
                    self.vocab[k] = np.packbits((members.mean(0) >= 0.5).astype(np.uint8))
        counts = np.bincount(self._assign(map_descriptors), minlength=words)
        self.idf = np.log((len(map_descriptors) + 1) / (counts + 1)).astype(np.float32)
        self.orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=6)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def _assign(self, desc):
        out = np.empty(len(desc), int)
        for s in range(0, len(desc), 4096):
            block = desc[s:s + 4096]
            ham = _POPCOUNT[np.bitwise_xor(block[:, None, :], self.vocab[None])].sum(2)
            out[s:s + 4096] = ham.argmin(1)
        return out

    def describe_features(self, desc: np.ndarray) -> np.ndarray:
        v = np.zeros(len(self.vocab), np.float32)
        if desc is not None and len(desc):
            np.add.at(v, self._assign(desc), 1.0)
        v *= self.idf
        return v / max(float(np.linalg.norm(v)), 1e-12)

    def describe(self, rgb: np.ndarray) -> np.ndarray:
        gray = self.clahe.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        _, desc = self.orb.detectAndCompute(gray, None)
        return self.describe_features(desc)


def place_model_available() -> bool:
    """Is the pinned MegaLoc code + weights in the local caches (no network)?"""
    try:
        import torch
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    hub = Path(torch.hub.get_dir()) / f"gmberton_MegaLoc_{MEGALOC_CODE_COMMIT}"
    weights = try_to_load_from_cache(MEGALOC_WEIGHTS_REPO, "model.safetensors", revision=MEGALOC_WEIGHTS_REVISION)
    return hub.exists() and isinstance(weights, str)


def make_descriptor(name, *, map_dir=None, allow_download=False, **kw):
    if name == "megaloc":
        return MegaLocDescriptor(allow_download=allow_download, **kw)
    if name == "orb_bow":
        if map_dir is None:
            raise ValueError("orb_bow needs a saved map to build its vocabulary")
        return OrbBowDescriptor(np.load(Path(map_dir) / "landmarks.npz")["desc"], **kw)
    raise ValueError(f"unknown place descriptor {name}")
