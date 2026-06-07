"""Deprecated: import from `flightframe.renderer` instead."""

import flightframe.renderer as _renderer

from flightframe.renderer import *  # noqa: F403

# Private helpers used by legacy Streamlit UI code.
_draw_overlay_rgba = _renderer._draw_overlay_rgba
