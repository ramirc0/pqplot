"""Chromosome outline geometry in bp, free of any drawing library.

Positions run along the chromosome in bp. Positions across it are in body
half-widths, so the body edges sit at -1 and 1.
"""

from __future__ import annotations

import numpy as np

from pqplot.genome import Chromosome

PINCH = 0.42
"""Half-width at the centromere midpoint."""
STALK = 0.55
"""Half-width along a stalk."""
KNOB = 1.6
"""Length of a telocentric knob from position 0 to its waist, in caps."""
WAIST = 0.9
"""Shortest half-length of the centromere waist, in caps."""
ARC = 25


def vertices(
    chromosome: Chromosome, cap: float, *, stalks: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return the closed outline of a chromosome body.

    The outline has round telomere caps, a waist at the centromere and
    narrow stalks. A telocentric chromosome gets a round knob at position
    0, with its waist `KNOB` caps in. An unknown centromere gets no waist.

    Parameters
    ----------
    chromosome : Chromosome
        Chromosome to outline.
    cap : float
        Cap radius in bp. Pass the body half-width converted to bp, so caps
        come out round. It shrinks to fit a short chromosome.
    stalks : bool, default True
        Whether ``stalk`` bands narrow the body.

    Raises
    ------
    ValueError
        If the cap is not positive.

    Returns
    -------
    along, across : numpy.ndarray
        Vertex positions along the chromosome in bp and across it in
        half-widths. The ring runs from the left tip over the top edge and
        back along the bottom edge. It does not repeat its first vertex.

    Examples
    --------
    >>> along, across = vertices(Chromosome("chr1", 1000, centromere=(480, 520)), cap=10)
    >>> float(along.min()), float(along.max()), float(across.max())
    (0.0, 1000.0, 1.0)
    """
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    cap = min(cap, chromosome.length / 2)
    lo, hi = cap, chromosome.length - cap
    mid, waist = None, WAIST * cap
    if chromosome.telocentric:
        mid = KNOB * cap
        waist = mid - lo
    elif chromosome.centromere is not None:
        mid = sum(chromosome.centromere) / 2
        waist = max(waist, np.ptp(chromosome.centromere) / 2)
    spans = [(b.start, b.end) for b in chromosome.bands if stalks and b.stain.name == "stalk"]

    def width(s: float, right: bool) -> float:
        w = 1.0 if mid is None else 1 - (1 - PINCH) * max(0.0, 1 - abs(s - mid) / waist)
        for s0, s1 in spans:
            if (s0 <= s < s1) if right else (s0 < s <= s1):
                w = min(w, STALK)
        return w

    breaks = [lo, hi, *(b for span in spans for b in span)]
    if mid is not None:
        breaks += [mid - waist, mid, mid + waist]
    edge = []
    for s in sorted({float(np.clip(b, lo, hi)) for b in breaks}):
        left, right = width(s, right=False), width(s, right=True)
        edge += [(s, left)] if left == right else [(s, left), (s, right)]
    s, w = np.array(edge).T
    tip = np.linspace(np.pi, np.pi / 2, ARC // 2 + 1)[:-1]
    turn = np.linspace(np.pi / 2, -np.pi / 2, ARC)
    tail = np.linspace(-np.pi / 2, -np.pi, ARC // 2 + 1)[1:-1]
    along = np.concatenate(
        [lo + cap * np.cos(tip), s, hi + cap * np.cos(turn), s[::-1], lo + cap * np.cos(tail)]
    )
    across = np.concatenate([np.sin(tip), w, np.sin(turn), -w[::-1], np.sin(tail)])
    return along, across
