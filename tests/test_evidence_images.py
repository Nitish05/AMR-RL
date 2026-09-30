"""Durable image receipts without unbounded idle-session frame logging.

Ported from BB8-RL tests/test_visual_evidence.py to the AMR receipt API.
"""

import hashlib
import json
import os
import stat

import cv2
import numpy as np
import pytest

from amr_rl.evidence import images
from amr_rl.evidence.images import ImageEvidenceArchive, load_receipt, receipt_sha256


def frame(value):
    return np.full((8, 8, 3), value, dtype=np.uint8)


def receipt(tag="a", **extra):
    return {"event_id": tag * 8, "action": "signal", **extra}


def receipt_files(archive):
    return list(archive.directory.glob("receipt-*.json"))


def test_live_buffer_does_not_write_idle_pixels_and_retains_receipt(tmp_path):
    archive = ImageEvidenceArchive(tmp_path)
    records = [archive.capture(frame(i), frame_index=i, timestamp=i * 0.05) for i in range(3)]
    assert not list(archive.directory.iterdir())
    hashes = [r["raw_rgb_sha256"] for r in records]
    stored = archive.retain(receipt(), hashes)
    assert archive.retain(receipt(), hashes) == stored
    assert len(list(archive.directory.iterdir())) == 4
    path = archive.directory / f"receipt-{receipt_sha256(receipt())}.json"
    saved = json.loads(path.read_text())
    assert saved == stored == load_receipt(path)
    assert saved["receipt"] == receipt()
    for record in saved["images"].values():
        png = tmp_path / record["png_path"]
        assert hashlib.sha256(png.read_bytes()).hexdigest() == record["png_sha256"]
        rgb = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == record["raw_rgb_sha256"]
    # Captured records returned to callers are copies, not the buffered state.
    assert not records[0]["retained"]


def test_buffer_bounded_and_missing_frames_rejected(tmp_path):
    archive = ImageEvidenceArchive(tmp_path, buffer_frames=16)
    first = archive.capture(frame(0), frame_index=0, timestamp=0)
    for index in range(1, 40):
        archive.capture(frame(index), frame_index=index, timestamp=index * 0.05)
    assert len(archive.frames) == 16
    with pytest.raises(ValueError, match="missing"):
        archive.retain(receipt(), [first["raw_rgb_sha256"]])
    assert not list(archive.directory.iterdir())


@pytest.mark.parametrize("hashes", [[], ["../../etc/passwd"], ["A" * 64]])
def test_empty_or_malformed_frame_citations_rejected(tmp_path, hashes):
    archive = ImageEvidenceArchive(tmp_path)
    archive.capture(frame(1), frame_index=0, timestamp=0)
    with pytest.raises(ValueError):
        archive.retain(receipt(), hashes)
    assert not list(archive.directory.iterdir())


def test_audit_retains_every_distinct_image(tmp_path):
    archive = ImageEvidenceArchive(tmp_path, audit=True, buffer_frames=8)
    for index in range(20):
        record = archive.capture(frame(index), frame_index=index, timestamp=index * 0.05)
        assert record["retained"]
        assert (tmp_path / record["png_path"]).is_file()
    assert len(list(archive.directory.glob("*.png"))) == 20


@pytest.mark.parametrize("artifact", ["png", "receipt"])
@pytest.mark.parametrize("new_receipt", [False, True])
def test_partial_atomic_write_leaves_no_published_fragment_and_retry_is_valid(
    tmp_path, monkeypatch, artifact, new_receipt
):
    archive = ImageEvidenceArchive(tmp_path)
    record = archive.capture(frame(15), frame_index=1, timestamp=0.05)
    evidence = receipt()
    original = images.tempfile.NamedTemporaryFile
    failed = []

    class InterruptedFile:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, payload):
            selected = (
                payload.startswith(b"\x89PNG") if artifact == "png" else payload.startswith(b"{")
            )
            if selected and not failed:
                failed.append(True)
                self.handle.write(payload[:8])
                self.handle.flush()
                raise OSError("partial device write")
            return self.handle.write(payload)

    monkeypatch.setattr(
        images.tempfile,
        "NamedTemporaryFile",
        lambda *args, **kwargs: InterruptedFile(original(*args, **kwargs)),
    )
    with pytest.raises(OSError, match="partial device write"):
        archive.retain(evidence, [record["raw_rgb_sha256"]])
    assert not receipt_files(archive)
    assert not list(archive.directory.glob(".image-evidence-*.tmp"))
    if artifact == "png":
        assert not list(archive.directory.glob("*.png"))
    if new_receipt:
        evidence = receipt("b")
    saved = archive.retain(evidence, [record["raw_rgb_sha256"]])
    assert saved["receipt"] == evidence
    image = saved["images"][record["raw_rgb_sha256"]]
    assert (
        hashlib.sha256((tmp_path / image["png_path"]).read_bytes()).hexdigest()
        == image["png_sha256"]
    )
    assert not list(archive.directory.glob(".image-evidence-*.tmp"))


@pytest.mark.parametrize("new_receipt", [False, True])
def test_corrupt_existing_png_is_preserved_and_rejected_even_after_retention(tmp_path, new_receipt):
    archive = ImageEvidenceArchive(tmp_path)
    record = archive.capture(frame(20), frame_index=0, timestamp=0)
    archive.retain(receipt(), [record["raw_rgb_sha256"]])
    target = tmp_path / record["png_path"]
    target.write_bytes(b"partial old PNG")
    evidence = receipt("b") if new_receipt else receipt()
    with pytest.raises(ValueError, match="differ"):
        archive.retain(evidence, [record["raw_rgb_sha256"]])
    assert target.read_bytes() == b"partial old PNG"
    assert len(receipt_files(archive)) == 1
    with pytest.raises(ValueError):
        load_receipt(receipt_files(archive)[0])


@pytest.mark.parametrize("corruption", ["partial", "conflicting", "equal_value_wrong_type"])
def test_corrupt_existing_receipt_is_preserved_and_rejected(tmp_path, corruption):
    archive = ImageEvidenceArchive(tmp_path)
    record = archive.capture(frame(20), frame_index=0, timestamp=0)
    evidence = receipt(generation=1)
    archive.retain(evidence, [record["raw_rgb_sha256"]])
    path = receipt_files(archive)[0]
    if corruption == "partial":
        corrupted = b'{"receipt":'
    else:
        payload = json.loads(path.read_text())
        if corruption == "equal_value_wrong_type":
            payload["receipt"]["generation"] = True
        else:
            payload["receipt"]["unexpected"] = True
        corrupted = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(corrupted)
    with pytest.raises(ValueError):
        archive.retain(evidence, [record["raw_rgb_sha256"]])
    with pytest.raises(ValueError):
        load_receipt(path)
    assert path.read_bytes() == corrupted


def test_replay_with_different_frames_for_same_receipt_is_rejected(tmp_path):
    archive = ImageEvidenceArchive(tmp_path)
    first = archive.capture(frame(1), frame_index=0, timestamp=0)
    second = archive.capture(frame(2), frame_index=1, timestamp=0.05)
    archive.retain(receipt(), [first["raw_rgb_sha256"]])
    with pytest.raises(ValueError, match="differs"):
        archive.retain(receipt(), [first["raw_rgb_sha256"], second["raw_rgb_sha256"]])
    assert len(list(archive.directory.glob("*.png"))) == 1


def test_exact_receipt_replay_survives_eviction_and_archive_restart(tmp_path):
    archive = ImageEvidenceArchive(tmp_path, buffer_frames=4)
    record = archive.capture(frame(0), frame_index=0, timestamp=0)
    archive.retain(receipt(), [record["raw_rgb_sha256"]])
    original = {path.name: path.read_bytes() for path in archive.directory.iterdir()}
    for index in range(1, 10):
        archive.capture(frame(index), frame_index=index, timestamp=index * 0.05)
    assert record["raw_rgb_sha256"] not in archive.frames
    archive.retain(receipt(), [record["raw_rgb_sha256"]])
    restarted = ImageEvidenceArchive(tmp_path)
    restarted.retain(receipt(), [record["raw_rgb_sha256"]])
    assert {path.name: path.read_bytes() for path in archive.directory.iterdir()} == original


def test_replay_after_eviction_decodes_png_and_rejects_substituted_valid_image(tmp_path):
    archive = ImageEvidenceArchive(tmp_path, buffer_frames=2)
    record = archive.capture(frame(3), frame_index=0, timestamp=0)
    archive.retain(receipt(), [record["raw_rgb_sha256"]])
    for index in range(1, 5):
        archive.capture(frame(10 + index), frame_index=index, timestamp=index)
    # A different, valid PNG with a consistent file hash in the receipt still fails
    # because decoded pixels no longer match the cited raw RGB hash.
    ok, other = cv2.imencode(".png", frame(99))
    assert ok
    png = tmp_path / record["png_path"]
    png.write_bytes(other.tobytes())
    path = receipt_files(archive)[0]
    payload = json.loads(path.read_text())
    payload["images"][record["raw_rgb_sha256"]]["png_sha256"] = hashlib.sha256(
        other.tobytes()
    ).hexdigest()
    path.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    with pytest.raises(ValueError, match="pixels differ"):
        archive.retain(receipt(), [record["raw_rgb_sha256"]])


def test_atomic_publication_never_replaces_a_concurrent_conflicting_artifact(tmp_path, monkeypatch):
    archive = ImageEvidenceArchive(tmp_path)
    record = archive.capture(frame(5), frame_index=0, timestamp=0)
    original = images.os.link

    def conflict(source, destination):
        destination.write_bytes(b"concurrent original")
        return original(source, destination)

    monkeypatch.setattr(images.os, "link", conflict)
    with pytest.raises(ValueError, match="differs"):
        archive.retain(receipt(), [record["raw_rgb_sha256"]])
    assert (tmp_path / record["png_path"]).read_bytes() == b"concurrent original"
    assert not receipt_files(archive)
    assert not list(archive.directory.glob(".image-evidence-*.tmp"))


@pytest.mark.parametrize("failed_publication", [1, 2])
def test_directory_sync_failure_withholds_retention_and_retries_complete_artifacts(
    tmp_path, monkeypatch, failed_publication
):
    archive = ImageEvidenceArchive(tmp_path)
    record = archive.capture(frame(7), frame_index=0, timestamp=0)
    evidence = receipt()
    original = os.fsync
    directory_calls = []

    def interrupted(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_calls.append(True)
            if len(directory_calls) == failed_publication:
                raise OSError("directory sync failed")
        return original(descriptor)

    monkeypatch.setattr(images.os, "fsync", interrupted)
    with pytest.raises(OSError, match="directory sync failed"):
        archive.retain(evidence, [record["raw_rgb_sha256"]])
    assert not archive.retained
    assert not list(archive.directory.glob(".image-evidence-*.tmp"))
    published = {path.name: path.read_bytes() for path in archive.directory.iterdir()}
    archive.retain(evidence, [record["raw_rgb_sha256"]])
    assert len(directory_calls) > failed_publication
    assert archive.retained == {receipt_sha256(evidence)}
    assert all((archive.directory / name).read_bytes() == data for name, data in published.items())


def test_load_receipt_rejects_renamed_file(tmp_path):
    archive = ImageEvidenceArchive(tmp_path)
    record = archive.capture(frame(4), frame_index=0, timestamp=0)
    archive.retain(receipt(), [record["raw_rgb_sha256"]])
    path = receipt_files(archive)[0]
    renamed = path.with_name(f"receipt-{'0' * 64}.json")
    renamed.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="name"):
        load_receipt(renamed)
