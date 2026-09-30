"""Run evidence: provenance capture, image receipts and session recording.

Submodules are imported lazily so that ``amr_rl.evidence.provenance`` and
``amr_rl.evidence.recording`` never pull in OpenCV/NumPy; only ``images`` does.
"""

from importlib import import_module

_EXPORTS = {
    "capture_run_identity": "provenance",
    "verify_run_identity": "provenance",
    "ImageEvidenceArchive": "images",
    "load_receipt": "images",
    "RotatingJsonl": "recording",
    "SessionRecorder": "recording",
    "finish_recording": "recording",
}

__all__ = [
    "ImageEvidenceArchive",
    "RotatingJsonl",
    "SessionRecorder",
    "capture_run_identity",
    "finish_recording",
    "load_receipt",
    "verify_run_identity",
]


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
