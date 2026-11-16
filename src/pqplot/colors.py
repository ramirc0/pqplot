"""Colors for categories and values, shared by every data layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba, to_rgba_array
from matplotlib.patches import Patch


def value_scale(values: np.ndarray, cmap: object, norm: Normalize | None) -> ScalarMappable:
    """Return a color scale fit to the values unless `norm` sets its limits.

    Examples
    --------
    >>> value_scale(np.array([2.0, 8.0]), "magma", None).norm.vmax
    8.0
    """
    scale = ScalarMappable(norm, cmap)
    scale.set_array(values)
    scale.autoscale_None()
    return scale


def swatch(color: np.ndarray) -> Artist:
    """Return a legend handle for a filled category."""
    return Patch(facecolor=color, edgecolor="none")


def category_colors(
    values: np.ndarray | None,
    rows: int,
    shown: np.ndarray,
    palette: Mapping | None,
    legend: dict[str, Artist],
    handle: Callable[[np.ndarray], Artist] = swatch,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the category index of each of `rows` rows and the category colors.

    Only the `shown` rows count. Their categories join `legend` with handles
    made by `handle`. A category already in `legend` keeps its color unless
    `palette` names one. New categories take the next colors of the theme's
    cycle. Without categories, every row gets the first color of the cycle
    and the legend stays as it is.

    Raises
    ------
    KeyError
        If `palette` lacks a category.

    Examples
    --------
    >>> legend = {}
    >>> codes, swatches = category_colors(np.array(["b", "a", "b"]), 3, np.arange(3), None, legend)
    >>> codes.tolist(), list(legend)
    ([0, 1, 0], ['b', 'a'])
    """
    if values is None:
        return np.zeros(rows, dtype=int), to_rgba_array(["C0"])
    labels, shown_codes = _categories(values[shown], palette)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    swatches = []
    for label in labels.tolist():
        if palette is not None:
            color = to_rgba(palette[label])
        elif str(label) in legend:
            color = _color_of(legend[str(label)])
        else:
            color = to_rgba(cycle[len(legend) % len(cycle)])
        legend[str(label)] = handle(np.array(color))
        swatches.append(color)
    codes = np.zeros(rows, dtype=int)
    codes[shown] = shown_codes
    return codes, to_rgba_array(swatches) if swatches else np.empty((0, 4))


def _color_of(handle: Artist) -> tuple[float, float, float, float]:
    """Return the color a legend handle shows."""
    return to_rgba(handle.get_facecolor() if isinstance(handle, Patch) else handle.get_color())


def _categories(values: np.ndarray, palette: Mapping | None) -> tuple[np.ndarray, np.ndarray]:
    """Return the categories in legend order and each row's index into them."""
    uniques, first, codes = np.unique(values, return_index=True, return_inverse=True)
    if palette is None:
        order = np.argsort(first)
    else:
        missing = [u for u in uniques.tolist() if u not in palette]
        if missing:
            raise KeyError(f"palette lacks categories {missing}")
        keys = list(palette)
        order = np.argsort([keys.index(u) for u in uniques.tolist()])
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return uniques[order], rank[codes]
