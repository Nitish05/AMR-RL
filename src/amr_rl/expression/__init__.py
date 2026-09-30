"""Robot screen expression: a documented, pure function of the operator state.

``policy`` has no third-party dependencies; ``screen`` (OpenCV/NumPy) is
imported lazily so importing ``expression_from`` stays lightweight.
"""

from importlib import import_module

from amr_rl.expression.policy import ExpressionState, expression_from

__all__ = ["ExpressionState", "expression_from", "render", "render_png"]


def __getattr__(name):
    if name in ("render", "render_png"):
        return getattr(import_module(f"{__name__}.screen"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
