"""Label placement along a chromosome, free of any drawing library."""

from __future__ import annotations

import numpy as np


def capacity(gap: float, low: float, high: float) -> int:
    """Return how many labels fit between two bounds at a given spacing.

    Examples
    --------
    >>> capacity(4, 0, 10)
    3
    """
    return int(np.floor((high - low) / gap + 1e-9)) + 1


def spread(targets: np.ndarray, gap: float, low: float, high: float) -> np.ndarray:
    """Place labels at least `gap` apart, as close to their targets as possible.

    The result minimizes the squared distance to the targets, keeps their
    order and stays inside the bounds. It uses pool-adjacent-violators
    isotonic regression, so it runs in linear time and is deterministic.

    Parameters
    ----------
    targets : numpy.ndarray
        Positions the labels point at, sorted.
    gap : float
        Smallest distance between neighboring labels.
    low, high : float
        Bounds of the placed labels.

    Returns
    -------
    numpy.ndarray
        Label positions, in the order of `targets`.

    Raises
    ------
    ValueError
        If `targets` is not sorted or more than `capacity` labels are given.

    Examples
    --------
    >>> spread(np.array([50.0, 50.0, 50.0]), 4, 0, 100).tolist()
    [46.0, 50.0, 54.0]
    """
    if np.any(np.diff(targets) < 0):
        raise ValueError("targets must be sorted")
    if len(targets) > capacity(gap, low, high):
        raise ValueError(f"{len(targets)} labels do not fit {gap} apart in [{low}, {high}]")
    if len(targets) == 0:
        return targets.astype(float)
    steps = gap * np.arange(len(targets))
    return np.clip(_isotonic(targets - steps), low, high - steps[-1]) + steps


def _isotonic(values: np.ndarray) -> np.ndarray:
    """Return the non-decreasing least-squares fit to `values`."""
    means, weights = [], []
    for value in values:
        means.append(float(value))
        weights.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            m2, w2 = means.pop(), weights.pop()
            m1, w1 = means.pop(), weights.pop()
            means.append((m1 * w1 + m2 * w2) / (w1 + w2))
            weights.append(w1 + w2)
    return np.repeat(means, weights)
