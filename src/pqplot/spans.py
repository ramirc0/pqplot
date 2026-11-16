"""Boxes over spans along a chromosome, with a minimum width in points."""

from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib.collections import PolyCollection


class Spans(PolyCollection):
    """Boxes over spans along one axis, at least `min_points` wide on screen.

    Each span covers ``[start, end]`` along the axis and ``across`` on the
    other axis, in the collection's transform. At each draw, spans narrower
    than `min_points` widen about their middle, so they stay visible at any
    scale.

    Parameters
    ----------
    starts, ends : array-like of float
        Spans along the axis, in data coordinates.
    across : tuple of float
        Low and high edge on the other axis.
    orientation : {"horizontal", "vertical"}
        Direction of the along axis. Horizontal puts it on x.
    min_points : float
        Smallest drawn width, in points. Zero keeps every span as given.
    **kwargs
        Passed to `matplotlib.collections.PolyCollection`, such as
        ``transform`` or ``facecolors``.

    Attributes
    ----------
    starts, ends : numpy.ndarray
        Spans as given, before widening.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> spans = Spans([0, 50], [10, 60], (0, 1), orientation="horizontal", min_points=2)
    >>> len(ax.add_collection(spans).get_paths())
    2
    """

    def __init__(
        self,
        starts: object,
        ends: object,
        across: tuple[float, float],
        *,
        orientation: Literal["horizontal", "vertical"],
        min_points: float,
        **kwargs,
    ):
        self.starts = np.asarray(starts, dtype=float)
        self.ends = np.asarray(ends, dtype=float)
        self.across = across
        self.orientation = orientation
        self.min_points = min_points
        super().__init__(self._boxes(self.starts, self.ends), **kwargs)

    def draw(self, renderer) -> None:
        """Widen narrow spans for the current scale, then draw.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> _ = ax.add_collection(Spans([0], [1], (0, 1), orientation="vertical", min_points=2))
        >>> fig.draw_without_rendering()
        """
        if self.min_points > 0:
            zero, one = self._boxes(np.zeros(1), np.ones(1))[0, :2]
            p0, p1 = self.get_transform().transform([zero, one])
            min_half = renderer.points_to_pixels(self.min_points) / np.hypot(*(p1 - p0)) / 2
            mid = (self.starts + self.ends) / 2
            half = np.maximum((self.ends - self.starts) / 2, min_half)
            self.set_verts(self._boxes(mid - half, mid + half))
        super().draw(renderer)

    def _boxes(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Return one quad per span, starting with its two low-edge corners."""
        along = np.stack([starts, ends, ends, starts], axis=1)
        low, high = self.across
        across = np.broadcast_to([low, low, high, high], along.shape)
        pair = (along, across) if self.orientation == "horizontal" else (across, along)
        return np.stack(pair, axis=-1)
