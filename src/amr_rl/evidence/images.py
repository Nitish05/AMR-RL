"""Bounded live RGB buffer; immutable PNGs retained only for admitted receipts.

Everyday operation keeps recent onboard frames in memory (``buffer_frames``) and
writes nothing. When a caller admits a receipt (for example an interaction
outcome) that cites frames by raw-RGB SHA-256, ``retain`` publishes each cited
PNG and a ``receipt-<sha>.json`` exclusively (temp file + fsync + hard link +
directory fsync), never replacing an existing file. Exact replay of an identical
receipt verifies what is on disk (decoding PNGs when the frame has left the
buffer); conflicting content raises ``ValueError`` and is preserved untouched.

Scope: integrity against accidental overwrite, partial writes and mismatched
replays within one filesystem. Hashes are not signatures; anyone with write
access can replace the whole directory consistently. Audit mode additionally
writes every captured frame as it arrives.

Adapted from BB8-RL ``src/bb8_rl/visual_evidence.py`` (see docs/PROVENANCE.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

RECEIPT_SCHEMA = "amr_rl.image-receipt.v1"
DIRECTORY_NAME = "image-evidence"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_IMAGE_KEYS = {
    "png_path",
    "png_sha256",
    "raw_rgb_sha256",
    "shape",
    "frame_index",
    "timestamp",
    "retained",
}
_ARCHIVE_KEYS = {"schema", "receipt_sha256", "receipt", "images"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def receipt_sha256(receipt):
    """SHA-256 of the canonical JSON of a receipt (its file-name identity)."""
    return hashlib.sha256(_canonical(receipt).encode()).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_immutable(path, payload):
    """Publish complete bytes exclusively; preserve and reject conflicting files."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing image evidence differs: {path.name}")
        _sync_directory(path.parent)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".image-evidence-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            if handle.write(payload) != len(payload):
                raise OSError("Incomplete image evidence write")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A hard link publishes complete bytes without replacing any existing
            # artifact, including one concurrently created after the first check.
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"Existing image evidence differs: {path.name}") from None
        # Return only after both bytes and the published name are durable. A failed
        # directory flush withholds the receipt from the caller.
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _decoded_rgb_sha256(pixels):
    rgb = cv2.imdecode(np.frombuffer(pixels, np.uint8), cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    return hashlib.sha256(
        np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).tobytes()
    ).hexdigest()


def _verify_archive(payload, directory, *, receipt=None, hashes=None, buffered=None):
    """Validate receipt JSON bytes and every PNG they cite; return the parsed dict."""
    try:
        archive = json.loads(payload)
        if (
            not isinstance(archive, dict)
            or set(archive) != _ARCHIVE_KEYS
            or archive["schema"] != RECEIPT_SCHEMA
            or not isinstance(archive["images"], dict)
            or payload != _json_bytes(archive)
            or receipt_sha256(archive["receipt"]) != archive["receipt_sha256"]
        ):
            raise ValueError("Existing image receipt differs or is malformed")
        if receipt is not None and _canonical(archive["receipt"]) != _canonical(receipt):
            raise ValueError("Existing image receipt differs")
        if hashes is not None and set(archive["images"]) != hashes:
            raise ValueError("Existing image receipt differs in cited frames")
        if not archive["images"]:
            raise ValueError("Existing image receipt cites no frames")
        for raw_hash, record in archive["images"].items():
            if (
                not isinstance(record, dict)
                or set(record) != _IMAGE_KEYS
                or not _HEX64.fullmatch(raw_hash)
                or record["png_path"] != f"{DIRECTORY_NAME}/{raw_hash}.png"
                or record["raw_rgb_sha256"] != raw_hash
                or record["retained"] is not True
            ):
                raise ValueError("Existing image record differs")
            pixels = (directory / f"{raw_hash}.png").read_bytes()
            if hashlib.sha256(pixels).hexdigest() != record["png_sha256"]:
                raise ValueError("Existing image differs")
            if buffered is not None and raw_hash in buffered:
                if pixels != buffered[raw_hash][0]:
                    raise ValueError("Existing image differs")
            elif _decoded_rgb_sha256(pixels) != raw_hash:
                raise ValueError("Existing image pixels differ")
    except (TypeError, KeyError, UnicodeDecodeError, json.JSONDecodeError, cv2.error) as error:
        raise ValueError("Invalid existing image receipt") from error
    except FileNotFoundError as error:
        raise ValueError(f"Receipt cites a missing image: {error.filename}") from error
    return archive


def load_receipt(path):
    """Load ``receipt-<sha>.json`` and verify its name, bytes and every cited PNG."""
    path = Path(path)
    archive = _verify_archive(path.read_bytes(), path.parent)
    if path.name != f"receipt-{archive['receipt_sha256']}.json":
        raise ValueError("Image receipt file name does not match its content hash")
    return archive


class ImageEvidenceArchive:
    """Recent-frame buffer plus exclusive publication of receipt-cited frames."""

    def __init__(self, run_dir, *, audit=False, buffer_frames=256):
        if type(buffer_frames) is not int or buffer_frames < 1:
            raise ValueError("buffer_frames must be a positive integer")
        self.directory = Path(run_dir) / DIRECTORY_NAME
        self.directory.mkdir(parents=True, exist_ok=True)
        self.audit = audit
        self.buffer_frames = buffer_frames
        self.frames = OrderedDict()
        self.retained = set()

    def capture(self, rgb, *, frame_index, timestamp):
        """Buffer one HxWx3 uint8 RGB frame; returns its (copied) image record."""
        rgb = np.ascontiguousarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
            raise ValueError("Image evidence requires a non-empty HxWx3 uint8 RGB frame")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, (int, np.integer))
            or not math.isfinite(timestamp)
        ):
            raise ValueError("frame_index must be an int and timestamp finite")
        frame_index = int(frame_index)
        raw_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise OSError("Could not encode image evidence")
        payload = encoded.tobytes()
        record = {
            "png_path": f"{DIRECTORY_NAME}/{raw_hash}.png",
            "png_sha256": hashlib.sha256(payload).hexdigest(),
            "raw_rgb_sha256": raw_hash,
            "shape": list(rgb.shape),
            "frame_index": frame_index,
            "timestamp": float(timestamp),
            "retained": False,
        }
        self.frames[raw_hash] = (payload, record)
        self.frames.move_to_end(raw_hash)
        while len(self.frames) > self.buffer_frames:
            self.frames.popitem(last=False)
        if self.audit:
            return self._save(raw_hash)
        return dict(record)

    def _save(self, raw_hash):
        payload, record = self.frames[raw_hash]
        _write_immutable(self.directory / f"{raw_hash}.png", payload)
        record["retained"] = True
        return dict(record)

    def retain(self, receipt, frame_hashes):
        """Publish cited frames and ``receipt-<sha>.json``; return the stored dict.

        Identical replays verify existing files (even after buffer eviction or a
        restart). Conflicting existing content, frames absent from the buffer on
        first retention, malformed hashes and empty citations raise ValueError.
        """
        if not isinstance(receipt, dict):
            raise TypeError("Receipt must be a JSON object (dict)")
        hashes = set(frame_hashes)
        if not hashes:
            raise ValueError("Receipt must cite at least one frame")
        if not all(isinstance(h, str) and _HEX64.fullmatch(h) for h in hashes):
            raise ValueError("Frame hashes must be lowercase hex SHA-256 digests")
        digest = receipt_sha256(receipt)
        path = self.directory / f"receipt-{digest}.json"
        if path.exists():
            archive = _verify_archive(
                path.read_bytes(),
                self.directory,
                receipt=receipt,
                hashes=hashes,
                buffered=self.frames,
            )
            _sync_directory(self.directory)
        else:
            missing = hashes - set(self.frames)
            if missing:
                raise ValueError(
                    f"{len(missing)} receipt frame(s) missing from bounded evidence buffer"
                )
            images = {key: self._save(key) for key in sorted(hashes)}
            archive = {
                "schema": RECEIPT_SCHEMA,
                "receipt_sha256": digest,
                "receipt": receipt,
                "images": images,
            }
            payload = _json_bytes(archive)
            _write_immutable(path, payload)
            archive = json.loads(payload)
        self.retained.add(digest)
        return archive
