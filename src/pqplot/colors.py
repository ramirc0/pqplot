"""Colors for categories and values, shared by every data layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba_array
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


def _swatch(color: np.ndarray) -> Artist:
    """Return a legend handle for a filled category."""
    return Patch(facecolor=color, edgecolor="none")


def category_colors(
    values: np.ndarray | None,
    rows: int,
    palette: Mapping | None,
    legend: dict[str, Artist],
    handle: Callable[[np.ndarray], Artist] = _swatch,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's category index and the category colors.

    Categories join `legend` with handles made by `handle`. Without
    categories, every row gets the first color of the theme's cycle and the
    legend stays as it is.

    Raises
    ------
    KeyError
        If `palette` lacks a category.

    Examples
    --------
    >>> legend = {}
    >>> codes, swatches = category_colors(np.array(["b", "a", "b"]), 3, None, legend)
    >>> codes.tolist(), list(legend)
    ([0, 1, 0], ['b', 'a'])
    """
    if values is None:
        return np.zeros(rows, dtype=int), to_rgba_array(["C0"])
    labels, codes = _categories(values, palette)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    swatches = to_rgba_array(
        [cycle[i % len(cycle)] for i in range(len(labels))]
        if palette is None
        else [palette[x] for x in labels.tolist()]
    )
    for label, swatch in zip(labels, swatches, strict=True):
        legend[str(label)] = handle(swatch)
    return codes, swatches


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
