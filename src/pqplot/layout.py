"""Panel layout of a karyogram, free of any drawing library.

A karyogram lays chromosomes out in groups. The along axis runs down a
vertical karyogram and right across a horizontal one. Groups follow each
other along it, and the chromosomes of a group sit side by side across it.
Each chromosome column holds its slots: tracks before the body, the body,
and tracks after it. Across sizes are in body widths. Along sizes are in bp.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pqplot.genome import Chromosome

TRACK_GAP = 0.2
"""Space between slots of one chromosome, in body widths."""
COPY_GAP = 0.25
"""Space between copies of one chromosome, in body widths."""
COLUMN_GAP = 1.0
"""Space between chromosomes of one group, in body widths."""
GROUP_GAP = 24.0
"""Space between groups, in points. It holds the chromosome names."""
MAX_BODY_POINTS = 14.0
"""Widest body, in points. Columns center in a host wider than that allows."""

Align = Literal["start", "centromere", "end"]


def wrap[T](items: Sequence[T], groups: int) -> list[list[T]]:
    """Split items into groups of even size, in order.

    Parameters
    ----------
    items : sequence
        Items to split.
    groups : int
        Number of groups. Fewer come back when there are fewer items.

    Returns
    -------
    list of list

    Raises
    ------
    ValueError
        If `groups` is less than 1.

    Examples
    --------
    >>> wrap([1, 2, 3, 4, 5], 2)
    [[1, 2, 3], [4, 5]]
    """
    if groups < 1:
        raise ValueError(f"groups must be at least 1, got {groups}")
    size = -(-len(items) // groups)
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def bounds(chromosomes: Sequence[Chromosome], align: Align) -> list[tuple[float, float]]:
    """Return the along-axis limits that align chromosomes in one group.

    Every chromosome gets limits of the same length, so the group shares one
    bp scale.

    Parameters
    ----------
    chromosomes : sequence of Chromosome
        Chromosomes of one group.
    align : {"start", "centromere", "end"}
        Line up the starts, the centromere midpoints, or the ends. A
        telocentric or unknown centromere counts as position 0.

    Returns
    -------
    list of tuple of float
        ``(low, high)`` in bp for each chromosome.

    Raises
    ------
    ValueError
        If `align` is not one of the choices.

    Examples
    --------
    >>> row = [Chromosome("1", 1000, centromere=(400, 500)), Chromosome("2", 600)]
    >>> bounds(row, "end")
    [(0, 1000), (-400, 600)]
    """
    longest = max(c.length for c in chromosomes)
    if align == "start":
        return [(0, longest) for _ in chromosomes]
    if align == "end":
        return [(c.length - longest, c.length) for c in chromosomes]
    if align != "centromere":
        raise ValueError(f"align must be start, centromere or end, got {align!r}")
    mids = [sum(c.centromere) / 2 if c.centromere else 0.0 for c in chromosomes]
    p = max(mids)
    q = max(c.length - mid for c, mid in zip(chromosomes, mids, strict=True))
    return [(mid - p, mid + q) for mid in mids]


@dataclass(frozen=True)
class Slot:
    """A strip across a chromosome column that holds its body or one track.

    Parameters
    ----------
    track : str or None
        Track name, or None for the body.
    start : float
        Distance from the column start, in body widths.
    size : float
        Width in body widths.

    Examples
    --------
    >>> Slot(None, 0.0, 1.0).track is None
    True
    """

    track: str | None
    start: float
    size: float


def slots(
    before: Sequence[tuple[str, float]], copies: int, after: Sequence[tuple[str, float]]
) -> list[Slot]:
    """Stack the tracks and the body across one chromosome column.

    Parameters
    ----------
    before, after : sequence of (str, float)
        Track names and sizes in body widths, before and after the body.
    copies : int
        Number of copies drawn side by side in the body slot.

    Returns
    -------
    list of Slot
        Slots in order across the column.

    Examples
    --------
    >>> [s.track for s in slots([("genes", 1.0)], copies=1, after=[])]
    ['genes', None]
    """
    stack = [*before, (None, copies + (copies - 1) * COPY_GAP), *after]
    result, start = [], 0.0
    for track, size in stack:
        result.append(Slot(track, start, size))
        start += size + TRACK_GAP
    return result


@dataclass(frozen=True)
class Grid:
    """Rectangles of every panel, sized to fit a host.

    Groups share one bp scale and fill the along extent between gaps of
    `GROUP_GAP` points. Columns share one body width and fill the across
    extent, up to bodies `MAX_BODY_POINTS` wide. Beyond that they center.

    Parameters
    ----------
    spans : tuple of float
        Along length of each group in bp.
    columns : int
        Chromosomes in the largest group.
    slots : list of Slot
        Slots of every column.

    Examples
    --------
    >>> grid = Grid((1000.0,), 2, slots([], copies=1, after=[]))
    >>> grid.cell(0, 1, grid.slots[0], along=100, across=30)
    (0.0, 100.0, 20.0, 30.0)
    """

    spans: tuple[float, ...]
    columns: int
    slots: list[Slot]

    @property
    def width(self) -> float:
        """Width of one column in body widths.

        Examples
        --------
        >>> Grid((1.0,), 1, slots([("a", 1.0)], copies=1, after=[])).width
        2.2
        """
        last = self.slots[-1]
        return last.start + last.size

    def cell(
        self, group: int, column: int, slot: Slot, along: float, across: float
    ) -> tuple[float, float, float, float]:
        """Return the rectangle of one panel inside a host.

        Parameters
        ----------
        group, column : int
            Position of the chromosome.
        slot : Slot
            Slot of the panel.
        along, across : float
            Host size in points.

        Returns
        -------
        tuple of float
            ``(along_start, along_end, across_start, across_end)`` in points
            from the host's along and across starts.

        Examples
        --------
        >>> grid = Grid((300.0, 100.0), 1, slots([], copies=1, after=[]))
        >>> grid.cell(1, 0, grid.slots[0], along=424, across=10)
        (324.0, 424.0, 0.0, 10.0)
        """
        gaps = (len(self.spans) - 1) * GROUP_GAP
        points_per_bp = max(along - gaps, 0.0) / sum(self.spans)
        a0 = sum(self.spans[:group]) * points_per_bp + group * GROUP_GAP
        bodies = self.columns * self.width + (self.columns - 1) * COLUMN_GAP
        unit = min(across / bodies, MAX_BODY_POINTS)
        c0 = (across - bodies * unit) / 2 + (column * (self.width + COLUMN_GAP) + slot.start) * unit
        return a0, a0 + self.spans[group] * points_per_bp, c0, c0 + slot.size * unit

    def extent(self, points_per_bp: float, points_per_body: float) -> tuple[float, float]:
        """Return the along and across size of the grid at a given scale.

        Parameters
        ----------
        points_per_bp : float
            Along scale.
        points_per_body : float
            Body width in points.

        Returns
        -------
        tuple of float
            Along and across size in points.

        Examples
        --------
        >>> Grid((1000.0,), 2, slots([], copies=1, after=[])).extent(0.1, 10)
        (100.0, 30.0)
        """
        along = sum(self.spans) * points_per_bp + (len(self.spans) - 1) * GROUP_GAP
        across = (self.columns * self.width + (self.columns - 1) * COLUMN_GAP) * points_per_body
        return along, across
