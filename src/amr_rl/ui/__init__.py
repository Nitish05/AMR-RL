"""Operator console static assets (vanilla HTML/CSS/JS, no build step).

The server (owned by the integrator) serves ``index.html``, ``app.js`` and
``style.css`` from :data:`STATIC_DIR` side by side (the page references them by
relative URL) and implements the ``/api/*`` endpoints in ``docs/INTERFACES.md``.
:func:`read_static` is a traversal-safe helper for doing that.
"""

from __future__ import annotations

from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"

CONTENT_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}


def read_static(name: str) -> tuple[bytes, str] | None:
    """Return ``(body, content_type)`` for a whitelisted asset, else ``None``."""
    name = name.lstrip("/") or "index.html"
    if name not in CONTENT_TYPES:
        return None
    return (STATIC_DIR / name).read_bytes(), CONTENT_TYPES[name]


__all__ = ["CONTENT_TYPES", "STATIC_DIR", "read_static"]
