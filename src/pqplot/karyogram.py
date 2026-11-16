"""Whole-genome karyograms built from child panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.transforms import Bbox
from matplotlib.typing import ColorType

from pqplot import layout
from pqplot.colors import category_colors, value_scale
from pqplot.columns import column, rows_by_chrom
from pqplot.genome import Chromosome, Genome, chrom_key
from pqplot.ideograms import Ideogram
from pqplot.panels import Orientation, add_panel
from pqplot.tracks import TrackPanels, style_track

BODY_POINTS = 7.0
"""Body width in a karyogram's own figure, in points."""
LONGEST_POINTS = {"vertical": 180.0, "horizontal": 432.0}
"""Length of the longest group in a karyogram's own figure, in points."""
NAME_POINTS = {"vertical": 16.0, "horizontal": 40.0}
"""Room for chromosome names in a karyogram's own figure, in points at a 10 point font."""
PAD_POINTS = 6.0
"""Room for the layout engine's padding in a karyogram's own figure, in points."""
COLORBAR_POINTS = (7.0, 144.0)
"""Width and longest length of a colorbar, in points."""
COLORBAR_GAP = 8.0
"""Space between the karyogram and its colorbar, in points."""


@dataclass(frozen=True)
class Track:
    """A strip of data beside every chromosome of a karyogram.

    Parameters
    ----------
    name : str
        Name used to find the track's panels.
    side : {"before", "after"}, default "before"
        Side of the body, in order across the chromosome column. Before is
        left of a vertical karyogram and above a horizontal one.
    size : float, default 2
        Width across the chromosome, in body widths.

    Raises
    ------
    ValueError
        If the name is empty, the side is unknown or the size is not
        positive.

    Examples
    --------
    >>> Track("cnv", side="after", size=1.5).size
    1.5
    """

    name: str
    side: Literal["before", "after"] = "before"
    size: float = 2.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"track name must be a non-empty string, got {self.name!r}")
        if not (np.isfinite(self.size) and self.size > 0):
            raise ValueError(
                f"track size must be a positive number of body widths, got {self.size}"
            )
        if self.side not in ("before", "after"):
            raise ValueError(f"side must be before or after, got {self.side!r}")


class Karyogram:
    """Chromosomes of a genome side by side, each in its own panel.

    The host Axes is hidden. Each chromosome gets a body panel and one panel
    per track. Panels are plain Axes with positions in bp, so matplotlib
    calls draw on them directly. They are placed at draw time and follow
    resizes and layout engines. Every panel shares one bp scale.

    Parameters
    ----------
    genome : Genome
        Chromosomes to draw.
    ax : matplotlib.axes.Axes, optional
        Host Axes to fill. By default, a new figure sized to the content.
    orientation : {"vertical", "horizontal"}, default "vertical"
        Direction of the chromosomes. A horizontal karyogram is the vertical
        one transposed.
    groups : int or sequence of sequence of str, default 1
        Number of groups to wrap the chromosomes into, or the chromosome
        names of each group. Groups stack down a vertical karyogram and sit
        side by side in a horizontal one.
    align : {"start", "centromere", "end"}, default "start"
        What lines up within a group: the starts, the centromeres or the
        ends. Starts are at the top of a vertical karyogram and on the left
        of a horizontal one.
    chroms : sequence of str, optional
        Chromosomes to draw, in order. By default, all of `genome`. Not
        allowed with explicit `groups`.
    copies : int, default 1
        Copies of each chromosome drawn side by side, such as 2 for
        homologs.
    bands : bool, default True
        Whether to draw bands.
    tracks : sequence of Track, optional
        Data tracks beside every chromosome, in order across it.

    Raises
    ------
    ValueError
        If an argument is out of range, track names repeat, a chromosome is
        drawn twice, or there is nothing to draw.
    KeyError
        If a chromosome is not in `genome`.

    Attributes
    ----------
    ax : matplotlib.axes.Axes
        Host Axes.

    Examples
    --------
    >>> from pqplot import Genome
    >>> kary = Karyogram(Genome.bundled("hg38"), groups=2, align="centromere")
    >>> len(kary.panels())
    24
    """

    def __init__(
        self,
        genome: Genome,
        ax: Axes | None = None,
        *,
        orientation: Orientation = "vertical",
        groups: int | Sequence[Sequence[str]] = 1,
        align: layout.Align = "start",
        chroms: Sequence[str] | None = None,
        copies: int = 1,
        bands: bool = True,
        tracks: Sequence[Track] = (),
    ):
        if orientation not in ("horizontal", "vertical"):
            raise ValueError(f"orientation must be horizontal or vertical, got {orientation!r}")
        if not isinstance(copies, int) or copies < 1:
            raise ValueError(f"copies must be a whole number from 1 up, got {copies}")
        if isinstance(chroms, str) or (
            not isinstance(groups, int) and any(isinstance(g, str) for g in groups)
        ):
            raise ValueError("chroms and each group must be sequences of names, not one string")
        names = [t.name for t in tracks]
        if len(set(names)) < len(names):
            raise ValueError(f"track names must be unique, got {names}")
        if isinstance(groups, int):
            chosen = [genome[c] for c in chroms] if chroms is not None else list(genome)
            grouped = layout.wrap(chosen, groups) if chosen else []
        elif chroms is not None:
            raise ValueError("pass chroms or explicit groups, not both")
        else:
            grouped = [[genome[c] for c in group] for group in groups]
        _check_drawable(grouped)
        limits = [layout.bounds(group, align) for group in grouped]
        grid = layout.Grid(
            tuple(bounds[0][1] - bounds[0][0] for bounds in limits),
            max(len(group) for group in grouped),
            layout.slots(
                [(t.name, t.size) for t in tracks if t.side == "before"],
                copies,
                [(t.name, t.size) for t in tracks if t.side == "after"],
            ),
        )
        self._sized_to_content = ax is None
        if ax is None:
            ax = _own_axes(grid, orientation)
        ax.set_axis_off()
        self.ax = ax
        self._panels: dict[tuple[str, str | None], Axes] = {}
        self._ideograms: dict[str, list[Ideogram]] = {}
        self._legend: dict[str, Artist] = {}
        self._colorbar: tuple[Colorbar, float] | None = None
        self._orientation = orientation
        self._tracks = {t.name: t for t in tracks}
        body = next(s for s in grid.slots if s.track is None)
        tracks_first: dict[str, Axes] = {}
        along, value = ("sharey", "sharex") if orientation == "vertical" else ("sharex", "sharey")
        for g, group in enumerate(grouped):
            for j, (chromosome, (lo, hi)) in enumerate(zip(group, limits[g], strict=True)):
                key = chrom_key(chromosome.name)
                panel = add_panel(ax, partial(grid.cell, g, j, body), orientation)
                self._ideograms[key] = _draw_body(
                    panel, chromosome, orientation, copies, bands, lo, hi
                )
                self._panels[key, None] = panel
                for slot in grid.slots:
                    if slot.track is None:
                        continue
                    first = tracks_first.get(slot.track)
                    track = add_panel(
                        ax,
                        partial(grid.cell, g, j, slot),
                        orientation,
                        **{along: panel, value: first},
                    )
                    edge = 0 if orientation == "vertical" else len(grouped) - 1
                    style_track(track, orientation, labels=(g, j) == (edge, 0))
                    tracks_first.setdefault(slot.track, track)
                    self._panels[key, slot.track] = track

    @property
    def figure(self) -> Figure:
        """Figure that holds the karyogram.

        Examples
        --------
        >>> from pqplot import Genome
        >>> Karyogram(Genome.bundled("mm39")).figure.get_layout_engine() is not None
        True
        """
        return self.ax.figure

    def panel(self, chrom: str, track: str | None = None) -> Axes:
        """Return the panel of one chromosome.

        Parameters
        ----------
        chrom : str
            Chromosome name. Names match across styles.
        track : str, optional
            Track name. By default, the body panel.

        Returns
        -------
        matplotlib.axes.Axes
            Plain Axes with positions in bp along the chromosome.

        Raises
        ------
        KeyError
            If the chromosome or track is not drawn.

        Examples
        --------
        >>> from pqplot import Genome
        >>> hg38 = Genome.bundled("hg38")
        >>> kary = Karyogram(hg38, orientation="horizontal")
        >>> kary.panel("chr17").get_xlim() == (0, hg38["chr1"].length)
        True
        """
        try:
            return self._panels[chrom_key(chrom), track]
        except KeyError:
            raise KeyError(f"no panel for chromosome {chrom!r} and track {track!r}") from None

    def panels(self, track: str | None = None) -> dict[str, Axes]:
        """Return the panels of one track for every chromosome, in order.

        Parameters
        ----------
        track : str, optional
            Track name. By default, the body panels.

        Returns
        -------
        dict of str to matplotlib.axes.Axes
            Panels keyed by chromosome name.

        Raises
        ------
        KeyError
            If the karyogram has no such track.

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), chroms=["1", "2"], tracks=[Track("cnv")])
        >>> list(kary.panels("cnv"))
        ['chr1', 'chr2']
        """
        if track is not None and track not in self._tracks:
            raise KeyError(f"no track {track!r}")
        return {
            ideograms[0].chromosome.name: self._panels[key, track]
            for key, ideograms in self._ideograms.items()
            if (key, track) in self._panels
        }

    def track(self, name: str) -> TrackPanels:
        """Return the panels of one track, to draw data into.

        Parameters
        ----------
        name : str
            Track name.

        Returns
        -------
        TrackPanels

        Raises
        ------
        KeyError
            If the karyogram has no such track.

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("cnv")])
        >>> points = kary.track("cnv").scatter({"chrom": ["chr1"], "pos": [5e6], "value": [0.3]})
        >>> list(points)
        ['chr1']
        """
        if name not in self._tracks:
            raise KeyError(f"no track {name!r}")
        side = self._tracks[name].side
        return TrackPanels(self.panels(name), self._orientation, side, self._legend)

    def ideograms(self, chrom: str) -> list[Ideogram]:
        """Return the drawn copies of one chromosome.

        Parameters
        ----------
        chrom : str
            Chromosome name. Names match across styles.

        Returns
        -------
        list of Ideogram
            One per copy, in order across the body panel.

        Examples
        --------
        >>> from pqplot import Genome
        >>> len(Karyogram(Genome.bundled("hg38"), copies=2).ideograms("chrX"))
        2
        """
        return self._ideograms[chrom_key(chrom)]

    def paint(
        self,
        data: object = None,
        *,
        chrom: object = "chrom",
        start: object = None,
        end: object = None,
        color: object = None,
        value: object = None,
        palette: Mapping[object, ColorType] | None = None,
        cmap: str | Colormap | None = None,
        norm: Normalize | None = None,
        copy: object = None,
    ) -> ScalarMappable | None:
        """Fill chromosome bodies by category or by value.

        Touching spans of one category merge into one. Small spans keep a
        minimum width in points and draw over larger ones. Rows on
        chromosomes that are not drawn are skipped.

        Each column argument is a column name in `data` or the values
        themselves.

        Parameters
        ----------
        data : DataFrame or dict, optional
            Table with one row per span, such as a polars or pandas
            DataFrame.
        chrom : str or array-like, default "chrom"
            Chromosome of each span. Names match across styles.
        start, end : str or array-like, optional
            Span in bp. Without them, each row fills its whole chromosome.
        color : str or array-like, optional
            Category of each span. Categories get colors from `palette` and
            appear in `legend`. Without `color` or `value`, every span gets
            the first color of the theme's cycle.
        value : str or array-like, optional
            Number for each span, colored through `cmap` and `norm`.
        palette : mapping, optional
            Color of each category, in legend order. By default, the
            theme's color cycle in order of first appearance.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap for `value`. By default, the theme's.
        norm : matplotlib.colors.Normalize, optional
            Scaling of `value`. By default, its minimum to its maximum.
        copy : str or array-like, optional
            Copy of each span, such as the haplotype. Sorted levels map to
            copies left to right. Without it, spans fill every copy.

        Returns
        -------
        matplotlib.cm.ScalarMappable or None
            The mapping for `value`, for a colorbar. None for categories.

        Raises
        ------
        ValueError
            If both `color` and `value` are given, only one of `start` and
            `end` is given, or `copy` has more levels than drawn copies.
        KeyError
            If `palette` lacks a category.

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), groups=2)
        >>> segments = {"chrom": ["chr1", "chr7"], "start": [0, 5e6], "end": [4e7, 9e6],
        ...             "state": ["gain", "loss"]}
        >>> kary.paint(segments, start="start", end="end", color="state")
        >>> bins = {"chrom": ["chr2"] * 2, "start": [0, 1e6], "end": [1e6, 2e6], "n": [3, 9]}
        >>> heat = kary.paint(bins, start="start", end="end", value="n", cmap="magma")
        >>> heat.norm.vmax
        9.0
        """
        if color is not None and value is not None:
            raise ValueError("pass color or value, not both")
        if (start is None) != (end is None):
            raise ValueError("pass start and end together, or neither")
        chroms = column(data, chrom)
        copy_index = None
        if copy is not None:
            levels, copy_index = np.unique(column(data, copy), return_inverse=True)
            copies = len(next(iter(self._ideograms.values())))
            if len(levels) > copies:
                raise ValueError(f"copy has {len(levels)} levels for {copies} copies")
        mappable = None
        if value is not None:
            values = column(data, value).astype(float)
            mappable = value_scale(values, cmap, norm)
            # Each row is its own category, so no spans merge.
            codes, swatches = np.arange(len(values)), mappable.to_rgba(values)
        else:
            categories = None if color is None else column(data, color)
            codes, swatches = category_colors(categories, len(chroms), palette, self._legend)
        starts = None if start is None else column(data, start).astype(float)
        ends = None if end is None else column(data, end).astype(float)
        for key, rows in rows_by_chrom(chroms).items():
            for k, ideo in enumerate(self._ideograms.get(key, [])):
                mine = rows if copy_index is None else rows[copy_index[rows] == k]
                if len(mine) == 0:
                    continue
                if starts is None:
                    s0, s1 = np.zeros(len(mine)), np.full(len(mine), ideo.chromosome.length)
                else:
                    s0, s1 = starts[mine], ends[mine]
                s0, s1, c = _merge(s0, s1, codes[mine])
                ideo.paint(s0, s1, swatches[c])
        return mappable

    def colorbar(self, mappable: ScalarMappable, label: str | None = None, **kwargs) -> Colorbar:
        """Draw a colorbar right of the karyogram, top-aligned.

        The bar is `COLORBAR_POINTS` in size, whatever the size of the
        karyogram. When the karyogram made its own figure, the figure grows
        to hold the bar, so the chromosomes keep their size. A second call
        replaces the bar. A legend also sits at the top right, so move one of
        them when drawing both.

        Parameters
        ----------
        mappable : matplotlib.cm.ScalarMappable
            Color scale, such as the return value of `paint` or
            `TrackPanels.heatmap`.
        label : str, optional
            Label along the bar.
        **kwargs
            Passed to `matplotlib.figure.Figure.colorbar`.

        Returns
        -------
        matplotlib.colorbar.Colorbar

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), chroms=["1"])
        >>> bins = {"chrom": ["1"], "start": [0], "end": [1e6], "n": [7.0]}
        >>> heat = kary.paint(bins, start="start", end="end", value="n")
        >>> kary.colorbar(heat, label="cCREs per Mb").ax.get_ylabel()
        'cCREs per Mb'
        """
        old_room = 0.0
        if self._colorbar is not None:
            old, old_room = self._colorbar
            old.remove()
        cax = self.ax.inset_axes([1, 0, 0, 1])
        cax.set_axes_locator(partial(_beside, self.ax))
        bar = self.figure.colorbar(mappable, cax=cax, label=label, **kwargs)
        room = 0.0
        if self._sized_to_content:
            room = cax.get_tightbbox().width / self.figure.dpi + COLORBAR_GAP / 72
            self.figure.set_figwidth(self.figure.get_figwidth() + room - old_room)
            self._fit_height(COLORBAR_POINTS[1] / 72)
        self._colorbar = (bar, room)
        return bar

    def legend(self, title: str | None = None, **kwargs) -> Legend:
        """Draw a legend of the categories painted or marked so far.

        Parameters
        ----------
        title : str, optional
            Legend title.
        **kwargs
            Passed to `matplotlib.axes.Axes.legend`. By default, the
            legend sits outside the host Axes, at its top right, without a
            frame.

        When the karyogram made its own figure, the figure grows to hold
        the legend, so the chromosomes keep their size.

        Returns
        -------
        matplotlib.legend.Legend

        Raises
        ------
        ValueError
            If nothing with categories was drawn yet.

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), chroms=["1"])
        >>> kary.paint({"chrom": ["1"], "split": ["test"]}, color="split")
        >>> [t.get_text() for t in kary.legend(title="Split").get_texts()]
        ['test']
        """
        if not self._legend:
            raise ValueError("no categories to show; draw a layer with color= first")
        old = self.ax.get_legend()
        room = _inches(old) if old else -PAD_POINTS / 72
        options = {"loc": "upper left", "bbox_to_anchor": (1, 1), "frameon": False}
        legend = self.ax.legend(
            list(self._legend.values()), list(self._legend), title=title, **options | kwargs
        )
        if self._sized_to_content:
            self.figure.set_figwidth(self.figure.get_figwidth() + _inches(legend) - room)
            self._fit_height(legend.get_window_extent().height / self.figure.dpi)
        return legend

    def _fit_height(self, inches: float) -> None:
        """Grow the figure to at least the given height plus padding."""
        needed = inches + 2 * PAD_POINTS / 72
        if self.figure.get_figheight() < needed:
            self.figure.set_figheight(needed)


def _check_drawable(grouped: list[list[Chromosome]]) -> None:
    """Raise unless every group has chromosomes and none repeats."""
    if not grouped:
        raise ValueError("no chromosomes to draw")
    if not all(grouped):
        raise ValueError("groups must not hold an empty group")
    seen = set()
    for chromosome in (c for group in grouped for c in group):
        if chromosome.name in seen:
            raise ValueError(f"chromosome {chromosome.name!r} is drawn more than once")
        seen.add(chromosome.name)


def _beside(host: Axes, cax: Axes, renderer) -> Bbox:
    """Place a colorbar right of the host, top-aligned, with a fixed size in points."""
    box = host.bbox
    pixels_per_point = host.figure.dpi / 72
    width, longest = (size * pixels_per_point for size in COLORBAR_POINTS)
    height = min(box.height, longest)
    x0 = box.x1 + COLORBAR_GAP * pixels_per_point
    return Bbox.from_bounds(x0, box.y1 - height, width, height).transformed(
        host.figure.transSubfigure.inverted()
    )


def _inches(artist: Artist) -> float:
    """Return the width of an artist in inches."""
    return artist.get_window_extent().width / artist.figure.dpi


def _merge(
    starts: np.ndarray, ends: np.ndarray, codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge spans that touch or overlap the previous span of the same code, by start."""
    order = np.argsort(starts, kind="stable")
    starts, ends, codes = starts[order], ends[order], codes[order]
    head = np.ones(len(starts), dtype=bool)
    head[1:] = (codes[1:] != codes[:-1]) | (starts[1:] > ends[:-1])
    heads = np.flatnonzero(head)
    return starts[heads], np.maximum.reduceat(ends, heads), codes[heads]


def _draw_body(
    panel: Axes,
    chromosome: Chromosome,
    orientation: Orientation,
    copies: int,
    bands: bool,
    lo: float,
    hi: float,
) -> list[Ideogram]:
    """Draw the copies of one chromosome and its name into a body panel."""
    ideograms = [
        Ideogram(
            chromosome,
            panel,
            orientation=orientation,
            center=k * (1 + layout.COPY_GAP) + 0.5,
            bands=bands,
        )
        for k in range(copies)
    ]
    across = (0, copies + (copies - 1) * layout.COPY_GAP)
    panel.set_axis_off()
    if orientation == "vertical":
        panel.set(xlim=across, ylim=(hi, lo))
        panel.annotate(
            chromosome.label,
            (0.5, 0),
            xycoords="axes fraction",
            xytext=(0, -3),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize="small",
        )
    else:
        panel.set(xlim=(lo, hi), ylim=across[::-1])
        panel.annotate(
            chromosome.name,
            (0, 0.5),
            xycoords="axes fraction",
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize="small",
        )
    return ideograms


def _own_axes(grid: layout.Grid, orientation: Orientation) -> Axes:
    """Make a figure sized so the grid gets its default scale."""
    along, across = grid.extent(LONGEST_POINTS[orientation] / max(grid.spans), BODY_POINTS)
    along += NAME_POINTS[orientation] * plt.rcParams["font.size"] / 10
    width, height = (across, along) if orientation == "vertical" else (along, across)
    size = ((width + PAD_POINTS) / 72, (height + PAD_POINTS) / 72)
    _, ax = plt.subplots(figsize=size, layout="constrained")
    return ax
