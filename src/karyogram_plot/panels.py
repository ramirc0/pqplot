"""Child panels of a host Axes, placed at draw time."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

Cell = Callable[[float, float], tuple[float, float, float, float]]
"""Maps the host's along and across size in points to a panel rectangle.

The rectangle is ``(along_start, along_end, across_start, across_end)`` in
points from the host's along and across starts.
"""
Orientation = Literal["horizontal", "vertical"]


def add_panel(host: Axes, cell: Cell, orientation: Orientation, **kwargs) -> Axes:
    """Add a child Axes that `cell` places inside the host at each draw.

    The along axis runs down a vertical host and right across a horizontal
    one. The across axis runs right across a vertical host and down a
    horizontal one. The panel is a child of the host, so it is not in
    ``figure.axes``, and layout engines count it in the host's tight bbox.

    Parameters
    ----------
    host : matplotlib.axes.Axes
        Axes that holds the panel.
    cell : callable
        Rectangle of the panel for a given host size, as in `Cell`.
    orientation : {"horizontal", "vertical"}
        Direction of the along axis.
    **kwargs
        Passed to `matplotlib.axes.Axes.inset_axes`, such as ``sharex``.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, host = plt.subplots()
    >>> panel = add_panel(host, lambda along, across: (0, along / 2, 0, across), "vertical")
    >>> panel in host.child_axes
    True
    """
    panel = host.inset_axes([0, 0, 1, 1], **kwargs)
    panel.set_axes_locator(_CellLocator(host, cell, orientation))
    return panel


class _CellLocator:
    """Turns a cell in points into the panel's figure position."""

    def __init__(self, host: Axes, cell: Cell, orientation: Orientation):
        self.host = host
        self.cell = cell
        self.orientation = orientation

    def __call__(self, ax: Axes, renderer) -> Bbox:
        box = self.host.bbox
        pixels_per_point = self.host.figure.dpi / 72
        width, height = box.width / pixels_per_point, box.height / pixels_per_point
        if self.orientation == "vertical":
            a0, a1, c0, c1 = self.cell(height, width)
            x0, x1, y0, y1 = c0, c1, height - a1, height - a0
        else:
            a0, a1, c0, c1 = self.cell(width, height)
            x0, x1, y0, y1 = a0, a1, height - c1, height - c0
        extents = [
            box.x0 + x0 * pixels_per_point,
            box.y0 + y0 * pixels_per_point,
            box.x0 + x1 * pixels_per_point,
            box.y0 + y1 * pixels_per_point,
        ]
        return Bbox.from_extents(*extents).transformed(self.host.figure.transSubfigure.inverted())
