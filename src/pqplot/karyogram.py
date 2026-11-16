"""Whole-genome karyograms built from child panels."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, Normalize, to_rgba_array
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, StepPatch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox, IdentityTransform, ScaledTranslation, Transform
from matplotlib.typing import ColorType

from pqplot import layout
from pqplot.columns import column, rows_by_chrom
from pqplot.genome import Chromosome, Genome, chrom_key
from pqplot.ideograms import MIN_PAINT_POINTS, Ideogram
from pqplot.labels import capacity, spread
from pqplot.panels import Orientation, add_panel
from pqplot.spans import Spans

BODY_POINTS = 7.0
"""Body width in a karyogram's own figure, in points."""
LONGEST_POINTS = {"vertical": 180.0, "horizontal": 432.0}
"""Length of the longest group in a karyogram's own figure, in points."""
NAME_POINTS = {"vertical": 16.0, "horizontal": 40.0}
"""Room for chromosome names in a karyogram's own figure, in points."""
PAD_POINTS = 6.0
"""Room for the layout engine's padding in a karyogram's own figure, in points."""
COLORBAR_POINTS = (7.0, 144.0)
"""Width and longest length of a colorbar, in points."""
COLORBAR_GAP = 8.0
"""Space between the karyogram and its colorbar, in points."""
LEADER_POINTS = (5.0, 7.0, 10.0)
"""Start, elbow and end of a label leader, in points from the track edge."""


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

    Examples
    --------
    >>> Track("cnv", side="after", size=1.5).size
    1.5
    """

    name: str
    side: Literal["before", "after"] = "before"
    size: float = 2.0

    def __post_init__(self):
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
        If an argument is out of range or track names repeat.
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
        if copies < 1:
            raise ValueError(f"copies must be at least 1, got {copies}")
        names = [t.name for t in tracks]
        if len(set(names)) < len(names):
            raise ValueError(f"track names must be unique, got {names}")
        if isinstance(groups, int):
            chosen = [genome[c] for c in chroms] if chroms is not None else list(genome)
            grouped = layout.wrap(chosen, groups)
        elif chroms is not None:
            raise ValueError("pass chroms or explicit groups, not both")
        else:
            grouped = [[genome[c] for c in group] for group in groups]
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
                    _style_track(track, orientation, labels=(g, j) == (edge, 0))
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

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), chroms=["1", "2"], tracks=[Track("cnv")])
        >>> list(kary.panels("cnv"))
        ['chr1', 'chr2']
        """
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
            mappable = _value_scale(values, cmap, norm)
            # Each row is its own category, so no spans merge.
            codes, swatches = np.arange(len(values)), mappable.to_rgba(values)
        else:
            categories = None if color is None else column(data, color)
            codes, swatches = _category_colors(categories, len(chroms), palette, self._legend)
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
        karyogram. When the karyogram made its own figure, the figure widens
        to hold the bar, so the chromosomes keep their size. A legend also
        sits at the top right, so move one of them when drawing both.

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
        cax = self.ax.inset_axes([1, 0, 0, 1])
        cax.set_axes_locator(partial(_beside, self.ax))
        bar = self.figure.colorbar(mappable, cax=cax, label=label, **kwargs)
        if self._sized_to_content:
            room = cax.get_tightbbox().width / self.figure.dpi + COLORBAR_GAP / 72
            self.figure.set_figwidth(self.figure.get_figwidth() + room)
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

        When the karyogram made its own figure, the figure widens to hold
        the legend, so the chromosomes keep their size.

        Returns
        -------
        matplotlib.legend.Legend

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), chroms=["1"])
        >>> kary.paint({"chrom": ["1"], "split": ["test"]}, color="split")
        >>> [t.get_text() for t in kary.legend(title="Split").get_texts()]
        ['test']
        """
        old = self.ax.get_legend()
        room = _inches(old) if old else -PAD_POINTS / 72
        options = {"loc": "upper left", "bbox_to_anchor": (1, 1), "frameon": False}
        legend = self.ax.legend(
            list(self._legend.values()), list(self._legend), title=title, **options | kwargs
        )
        if self._sized_to_content:
            self.figure.set_figwidth(self.figure.get_figwidth() + _inches(legend) - room)
        return legend


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


class TrackPanels:
    """Draws data into one track, across every chromosome of a karyogram.

    Get one from `Karyogram.track`. Each method takes a table with one row
    per point or bin, splits the rows by chromosome and draws each group
    into its chromosome's panel. Rows on chromosomes without a panel are
    skipped. Positions are in bp. Panels of one track share their value
    axis, so one call scales them all.

    Each column argument is a column name in `data` or the values
    themselves. Extra keyword arguments go to the matplotlib call. Arrays
    and Series among them with one value per row, such as ``c`` or ``s``,
    are split by chromosome too.

    Parameters
    ----------
    panels : dict of str to matplotlib.axes.Axes
        Track panels keyed by chromosome name, in drawing order.
    orientation : {"horizontal", "vertical"}
        Direction of the chromosomes.
    side : {"before", "after"}
        Side of the body the track sits on. Marks and labels start from
        the edge that faces the body.
    legend : dict of str to matplotlib.artist.Artist
        Legend entries of the karyogram, which `intervals` and `mark` add
        to.

    Examples
    --------
    >>> from pqplot import Genome
    >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("cnv")])
    >>> type(kary.track("cnv")).__name__
    'TrackPanels'
    """

    def __init__(
        self,
        panels: dict[str, Axes],
        orientation: Orientation,
        side: Literal["before", "after"],
        legend: dict[str, Artist],
    ):
        self._panels = {chrom_key(name): (name, panel) for name, panel in panels.items()}
        self._orientation = orientation
        self._side = side
        self._legend = legend

    def mark(
        self,
        data: object = None,
        *,
        chrom: object = "chrom",
        pos: object = "pos",
        color: object = None,
        palette: Mapping[object, ColorType] | None = None,
        size: float = 16.0,
        **kwargs,
    ) -> dict[str, PathCollection]:
        """Draw a triangle at each position, on the track edge that faces the body.

        Each triangle points at the body. The track's value axis is hidden.

        Parameters
        ----------
        data : DataFrame or dict, optional
            Table with one row per mark.
        chrom : str or array-like, default "chrom"
            Chromosome of each mark.
        pos : str or array-like, default "pos"
            Position in bp.
        color : str or array-like, optional
            Category of each mark. Categories get colors from `palette` and
            appear in `Karyogram.legend`. By default, every mark gets the
            first color of the theme's cycle.
        palette : mapping, optional
            Color of each category, in legend order. By default, the
            theme's color cycle in order of first appearance.
        size : float, default 16
            Marker area in points squared, as in `matplotlib.axes.Axes.scatter`.
        **kwargs
            Passed to `matplotlib.axes.Axes.scatter`.

        Returns
        -------
        dict of str to matplotlib.collections.PathCollection
            Marks keyed by chromosome name.

        Raises
        ------
        KeyError
            If `palette` lacks a category.

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("hits", "after", 5)])
        >>> gwas = {"chrom": ["1", "2"], "pos": [55e6, 21e6], "trait": ["LDL", "LDL"]}
        >>> list(kary.track("hits").mark(gwas, color="trait"))
        ['chr1', 'chr2']
        """
        along = column(data, pos).astype(float)
        categories = None if color is None else column(data, color)
        marker = {(1, 0): "<", (-1, 0): ">", (0, -1): "^", (0, 1): "v"}[self._away()]

        def handle(swatch: np.ndarray) -> Artist:
            return Line2D(
                [],
                [],
                marker=marker,
                markersize=np.sqrt(size),
                linestyle="none",
                color=swatch,
                markeredgewidth=0,
            )

        codes, swatches = _category_colors(categories, len(along), palette, self._legend, handle)
        inset = np.array(self._away()) * np.sqrt(size) / 2 / 72
        kwargs = {"linewidths": 0, "clip_on": False, "zorder": 3, **kwargs}
        self._hide_values()
        marks = {}
        for name, panel, rows in self._groups(data, chrom):
            transform = self._across_axes(panel) + ScaledTranslation(
                *inset, panel.figure.dpi_scale_trans
            )
            marks[name] = panel.scatter(
                *self._xy(along[rows], np.full(len(rows), self._inner())),
                s=size,
                marker=marker,
                c=swatches[codes[rows]],
                transform=transform,
                **kwargs,
            )
        return marks

    def label(
        self,
        data: object = None,
        *,
        chrom: object = "chrom",
        pos: object = "pos",
        text: object = "text",
        fontsize: float | str = "x-small",
        **kwargs,
    ) -> dict[str, LabelLane]:
        """Write a label for each position, joined to it by a leader line.

        Labels start from the track edge that faces the body. Crowded
        labels spread apart at draw time and keep their order. Labels that
        still do not fit are hidden from the last row up, with a warning,
        so put the important rows first. Labels do not count in the
        layout, so size the track to fit the longest one. The track's value
        axis is hidden.

        Parameters
        ----------
        data : DataFrame or dict, optional
            Table with one row per label.
        chrom : str or array-like, default "chrom"
            Chromosome of each label.
        pos : str or array-like, default "pos"
            Position in bp the label points at.
        text : str or array-like, default "text"
            Label text.
        fontsize : float or str, default "x-small"
            Font size of the labels.
        **kwargs
            Passed to `matplotlib.text.Text`.

        Returns
        -------
        dict of str to LabelLane
            Labels keyed by chromosome name.

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("genes", "after", 5)])
        >>> genes = {"chrom": ["1", "2"], "start": [55e6, 21e6], "symbol": ["PCSK9", "APOB"]}
        >>> lanes = kary.track("genes").label(genes, pos="start", text="symbol")
        >>> [t.get_text() for t in lanes["chr1"].texts]
        ['PCSK9']
        """
        along = column(data, pos).astype(float)
        texts = column(data, text).astype(str)
        self._hide_values()
        return {
            name: LabelLane(
                panel,
                along[rows],
                texts[rows],
                orientation=self._orientation,
                away=self._away(),
                fontsize=fontsize,
                **kwargs,
            )
            for name, panel, rows in self._groups(data, chrom)
        }

    def scatter(
        self,
        data: object = None,
        *,
        chrom: object = "chrom",
        pos: object = "pos",
        value: object = "value",
        **kwargs,
    ) -> dict[str, PathCollection]:
        """Draw one marker per row at its position and value.

        A numeric ``c`` gets one color scale across every chromosome, unless
        ``norm``, ``vmin`` or ``vmax`` sets it. Markers have no edge unless
        ``linewidths`` sets one, because themes that ring markers in the
        background color would hide small, dense points.

        Parameters
        ----------
        data : DataFrame or dict, optional
            Table with one row per point.
        chrom : str or array-like, default "chrom"
            Chromosome of each point.
        pos : str or array-like, default "pos"
            Position in bp.
        value : str or array-like, default "value"
            Value on the track's value axis.
        **kwargs
            Passed to `matplotlib.axes.Axes.scatter`.

        Returns
        -------
        dict of str to matplotlib.collections.PathCollection
            Markers keyed by chromosome name.

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("cnv")])
        >>> import numpy as np
        >>> bins = {"chrom": ["1", "2"], "mid": [5e5, 7e5], "log2": np.array([0.4, -1.2])}
        >>> points = kary.track("cnv").scatter(bins, pos="mid", value="log2", c=bins["log2"])
        >>> list(points)
        ['chr1', 'chr2']
        """
        along, values = column(data, pos), column(data, value)
        kwargs = {"linewidths": 0, **kwargs}
        colors = np.asarray(kwargs.get("c", ()))
        if (
            colors.dtype.kind in "fiu"
            and colors.size
            and not {"norm", "vmin", "vmax"} & set(kwargs)
        ):
            kwargs |= {"vmin": np.nanmin(colors), "vmax": np.nanmax(colors)}
        return {
            name: panel.scatter(
                *self._xy(along[rows], values[rows]), **_split_styles(kwargs, rows, len(along))
            )
            for name, panel, rows in self._groups(data, chrom)
        }

    def line(
        self,
        data: object = None,
        *,
        chrom: object = "chrom",
        pos: object = "pos",
        value: object = "value",
        **kwargs,
    ) -> dict[str, Line2D]:
        """Draw a line through the rows of each chromosome, in position order.

        Parameters
        ----------
        data : DataFrame or dict, optional
            Table with one row per point.
        chrom : str or array-like, default "chrom"
            Chromosome of each point.
        pos : str or array-like, default "pos"
            Position in bp.
        value : str or array-like, default "value"
            Value on the track's value axis.
        **kwargs
            Passed to `matplotlib.axes.Axes.plot`.

        Returns
        -------
        dict of str to matplotlib.lines.Line2D
            Lines keyed by chromosome name.

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("gc")])
        >>> gc = {"chrom": ["1", "1"], "pos": [2e6, 1e6], "value": [0.41, 0.38]}
        >>> kary.track("gc").line(gc)["chr1"].get_ydata().tolist()
        [1000000.0, 2000000.0]
        """
        along, values = column(data, pos), column(data, value)
        lines = {}
        for name, panel, rows in self._groups(data, chrom):
            rows = rows[np.argsort(along[rows], kind="stable")]
            (lines[name],) = panel.plot(*self._xy(along[rows], values[rows]), **kwargs)
        return lines

    def area(
        self,
        data: object = None,
        *,
        chrom: object = "chrom",
        start: object = "start",
        end: object = "end",
        value: object = "value",
        baseline: float = 0.0,
        **kwargs,
    ) -> dict[str, StepPatch]:
        """Fill from a baseline to each bin's value, as steps.

        Bins of one chromosome must not overlap. Gaps between them stay
        empty.

        Parameters
        ----------
        data : DataFrame or dict, optional
            Table with one row per bin.
        chrom : str or array-like, default "chrom"
            Chromosome of each bin.
        start, end : str or array-like, default "start" and "end"
            Span of each bin in bp.
        value : str or array-like, default "value"
            Height of each bin on the track's value axis.
        baseline : float, default 0
            Value the fill starts from.
        **kwargs
            Passed to `matplotlib.axes.Axes.stairs`.

        Returns
        -------
        dict of str to matplotlib.patches.StepPatch
            Fills keyed by chromosome name.

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("genes")])
        >>> density = {"chrom": ["1", "1"], "start": [0, 1e6], "end": [1e6, 2e6], "value": [3, 8]}
        >>> kary.track("genes").area(density)["chr1"].get_data().edges.tolist()
        [0.0, 1000000.0, 2000000.0]
        """
        starts, ends = column(data, start).astype(float), column(data, end).astype(float)
        values = column(data, value).astype(float)
        across = "horizontal" if self._orientation == "vertical" else "vertical"
        areas = {}
        for name, panel, rows in self._groups(data, chrom):
            rows = rows[np.argsort(starts[rows], kind="stable")]
            edges, heights = _steps(starts[rows], ends[rows], values[rows])
            areas[name] = panel.stairs(
                heights, edges, baseline=baseline, fill=True, orientation=across, **kwargs
            )
        return areas

    def heatmap(
        self,
        data: object = None,
        *,
        chrom: object = "chrom",
        start: object = "start",
        end: object = "end",
        value: object = "value",
        cmap: str | Colormap | None = None,
        norm: Normalize | None = None,
        **kwargs,
    ) -> ScalarMappable:
        """Color each bin across the full track width by its value.

        Parameters
        ----------
        data : DataFrame or dict, optional
            Table with one row per bin.
        chrom : str or array-like, default "chrom"
            Chromosome of each bin.
        start, end : str or array-like, default "start" and "end"
            Span of each bin in bp.
        value : str or array-like, default "value"
            Number colored through `cmap` and `norm`.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap. By default, the theme's.
        norm : matplotlib.colors.Normalize, optional
            Scaling of `value`. By default, its minimum to its maximum.
        **kwargs
            Passed to `pqplot.spans.Spans`.

        Returns
        -------
        matplotlib.cm.ScalarMappable
            The color scale, for a colorbar.

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("expr")])
        >>> bins = {"chrom": ["1", "1"], "start": [0, 1e6], "end": [1e6, 2e6], "value": [3, 8]}
        >>> kary.track("expr").heatmap(bins, cmap="magma").norm.vmax
        8.0
        """
        starts, ends = column(data, start).astype(float), column(data, end).astype(float)
        values = column(data, value).astype(float)
        scale = _value_scale(values, cmap, norm)
        for _, panel, rows in self._groups(data, chrom):
            colors = scale.to_rgba(values[rows])
            self._add_spans(panel, starts[rows], ends[rows], colors, 0.0, **kwargs)
        return scale

    def intervals(
        self,
        data: object = None,
        *,
        chrom: object = "chrom",
        start: object = "start",
        end: object = "end",
        color: object = None,
        palette: Mapping[object, ColorType] | None = None,
        **kwargs,
    ) -> dict[str, Spans]:
        """Draw a box across the track width over each interval.

        Intervals narrower than `MIN_PAINT_POINTS` widen at draw time, so
        small features such as peaks stay visible.

        Parameters
        ----------
        data : DataFrame or dict, optional
            Table with one row per interval.
        chrom : str or array-like, default "chrom"
            Chromosome of each interval.
        start, end : str or array-like, default "start" and "end"
            Span of each interval in bp.
        color : str or array-like, optional
            Category of each interval. Categories get colors from `palette`
            and appear in `Karyogram.legend`. By default, every interval
            gets the first color of the theme's cycle.
        palette : mapping, optional
            Color of each category, in legend order. By default, the
            theme's color cycle in order of first appearance.
        **kwargs
            Passed to `pqplot.spans.Spans`.

        Returns
        -------
        dict of str to pqplot.spans.Spans
            Boxes keyed by chromosome name.

        Raises
        ------
        KeyError
            If `palette` lacks a category.

        Examples
        --------
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("peaks")])
        >>> peaks = {"chrom": ["1", "2"], "start": [1e6, 3e6], "end": [1.0002e6, 3.0001e6]}
        >>> list(kary.track("peaks").intervals(peaks))
        ['chr1', 'chr2']
        """
        starts, ends = column(data, start).astype(float), column(data, end).astype(float)
        categories = None if color is None else column(data, color)
        codes, swatches = _category_colors(categories, len(starts), palette, self._legend)
        return {
            name: self._add_spans(
                panel, starts[rows], ends[rows], swatches[codes[rows]], MIN_PAINT_POINTS, **kwargs
            )
            for name, panel, rows in self._groups(data, chrom)
        }

    def _groups(self, data: object, chrom: object) -> Iterator[tuple[str, Axes, np.ndarray]]:
        """Yield each drawn chromosome with its panel and rows, in drawing order."""
        rows = rows_by_chrom(column(data, chrom))
        for key, (name, panel) in self._panels.items():
            if key in rows:
                yield name, panel, rows[key]

    def _xy(self, along: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (values, along) if self._orientation == "vertical" else (along, values)

    def _hide_values(self) -> None:
        """Hide the value axis, which means nothing for marks and labels."""
        for _, panel in self._panels.values():
            (panel.xaxis if self._orientation == "vertical" else panel.yaxis).set_visible(False)

    def _away(self) -> tuple[int, int]:
        """Return the display direction that points away from the body."""
        sign = 1 if self._side == "after" else -1
        return (sign, 0) if self._orientation == "vertical" else (0, -sign)

    def _inner(self) -> float:
        """Return the across position of the edge facing the body, as an axes fraction."""
        return 0.0 if sum(self._away()) > 0 else 1.0

    def _across_axes(self, panel: Axes) -> Transform:
        """Return the transform with along in data and across in axes fractions."""
        return (
            panel.get_yaxis_transform()
            if self._orientation == "vertical"
            else panel.get_xaxis_transform()
        )

    def _add_spans(
        self,
        panel: Axes,
        starts: np.ndarray,
        ends: np.ndarray,
        colors: np.ndarray,
        min_points: float,
        **kwargs,
    ) -> Spans:
        """Add spans that fill the track width, with a thin edge to hide seams."""
        transform = self._across_axes(panel)
        kwargs = {"facecolors": colors, "edgecolors": colors, "linewidths": 0.25, **kwargs}
        spans = Spans(
            starts,
            ends,
            (0.0, 1.0),
            orientation=self._orientation,
            min_points=min_points,
            transform=transform,
            **kwargs,
        )
        panel.add_collection(spans, autolim=False)
        return spans


def _split_styles(styles: dict, rows: np.ndarray, length: int) -> dict:
    """Pick out the rows of style arrays that hold one value per row."""
    return {
        key: np.asarray(style)[rows]
        if hasattr(style, "__array__") and np.ndim(style) >= 1 and len(style) == length
        else style
        for key, style in styles.items()
    }


def _steps(
    starts: np.ndarray, ends: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return step edges and heights for sorted bins, with NaN steps over gaps."""
    gap = np.append(False, starts[1:] > ends[:-1])
    slot = np.arange(len(starts)) + np.cumsum(gap)
    heights = np.full(len(starts) + gap.sum(), np.nan)
    heights[slot] = values
    edges = np.empty(len(heights) + 1)
    edges[0] = starts[0]
    edges[slot + 1] = ends
    edges[slot[gap]] = starts[gap]
    return edges, heights


def _value_scale(values: np.ndarray, cmap: object, norm: Normalize | None) -> ScalarMappable:
    """Return a color scale fit to the values unless `norm` sets its limits."""
    scale = ScalarMappable(norm, cmap)
    scale.set_array(values)
    scale.autoscale_None()
    return scale


def _swatch(color: np.ndarray) -> Artist:
    """Return a legend handle for a filled category."""
    return Patch(facecolor=color, edgecolor="none")


def _category_colors(
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
    along += NAME_POINTS[orientation]
    width, height = (across, along) if orientation == "vertical" else (along, across)
    size = ((width + PAD_POINTS) / 72, (height + PAD_POINTS) / 72)
    _, ax = plt.subplots(figsize=size, layout="constrained")
    return ax


class LabelLane(Artist):
    """Labels in a track panel, spread along the chromosome at each draw.

    Each label sits beyond the panel edge that faces the body, joined to
    its position by a leader line. At draw time, labels closer than one
    line of text spread apart with `pqplot.labels.spread`, keeping their
    order. Labels that do not fit are hidden from the last one up, with a
    warning.

    Parameters
    ----------
    panel : matplotlib.axes.Axes
        Track panel to draw in.
    positions : numpy.ndarray
        Positions in bp the labels point at.
    texts : numpy.ndarray of str
        Label texts, most important first.
    orientation : {"horizontal", "vertical"}
        Direction of the chromosome.
    away : tuple of int
        Display direction that points away from the body.
    fontsize : float or str
        Font size of the labels.
    **kwargs
        Passed to `matplotlib.text.Text`.

    Attributes
    ----------
    texts : list of matplotlib.text.Text
        Labels, in the order given.
    leaders : matplotlib.collections.LineCollection
        Leader lines, one per label.

    Examples
    --------
    >>> import numpy as np
    >>> from pqplot import Genome
    >>> kary = Karyogram(Genome.bundled("hg38"), chroms=["1"], tracks=[Track("genes", "after", 5)])
    >>> panel = kary.panel("1", "genes")
    >>> lane = LabelLane(panel, np.array([5e7]), np.array(["TP73"]),
    ...                  orientation="vertical", away=(1, 0), fontsize="x-small")
    >>> lane.texts[0].get_text()
    'TP73'
    """

    def __init__(
        self,
        panel: Axes,
        positions: np.ndarray,
        texts: np.ndarray,
        *,
        orientation: Orientation,
        away: tuple[int, int],
        fontsize: float | str,
        **kwargs,
    ):
        super().__init__()
        self.positions = positions
        self.orientation = orientation
        self.away = np.array(away)
        rotation = 0 if orientation == "vertical" else 90
        # Align the end of each label nearest the body to its anchor.
        ha = "left" if away[0] > 0 or away[1] > 0 else "right"
        self.texts = [
            panel.text(
                0,
                0,
                text,
                transform=IdentityTransform(),
                fontsize=fontsize,
                rotation=rotation,
                rotation_mode="anchor",
                ha=ha,
                va="center",
                clip_on=False,
                in_layout=False,
                zorder=3,
                **kwargs,
            )
            for text in texts
        ]
        self.leaders = LineCollection(
            [],
            colors=plt.rcParams["axes.edgecolor"],
            linewidths=0.4,
            transform=IdentityTransform(),
            clip_on=False,
            zorder=2.95,
        )
        self.leaders.set_in_layout(False)
        panel.add_collection(self.leaders, autolim=False)
        self.set_zorder(2.9)
        self.set_in_layout(False)
        panel.add_artist(self)

    def draw(self, renderer) -> None:
        """Place the labels and leaders for the current panel size.

        Examples
        --------
        >>> import numpy as np
        >>> from pqplot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), chroms=["1"], tracks=[Track("g", "after", 5)])
        >>> lane = LabelLane(kary.panel("1", "g"), np.array([5e7]), np.array(["TP73"]),
        ...                  orientation="vertical", away=(1, 0), fontsize="x-small")
        >>> kary.figure.draw_without_rendering()
        >>> lane.texts[0].get_position() != (0, 0)
        True
        """
        if not self.texts:
            return
        along = 1 if self.orientation == "vertical" else 0
        box = self.axes.bbox
        points = renderer.points_to_pixels
        line = points(self.texts[0].get_fontsize()) * 1.2
        low, high = sorted((box.p0[along], box.p1[along]))
        low, high = low + line / 2, high - line / 2
        fits = min(len(self.texts), capacity(line, low, high))
        if fits < len(self.texts):
            warnings.warn(
                f"{len(self.texts) - fits} labels do not fit; hiding the last ones. "
                "Use a smaller font or fewer labels.",
                stacklevel=2,
            )
        xy = np.zeros((len(self.positions), 2))
        xy[:, along] = self.positions
        targets = self.axes.transData.transform(xy)[:, along]
        shown = np.arange(fits)
        order = shown[np.argsort(targets[shown], kind="stable")]
        placed = np.empty(fits)
        placed[np.argsort(order)] = spread(targets[order], line, low, high)
        inner = box.p0 if self.away.sum() > 0 else box.p1
        edge = inner[1 - along]
        start, elbow, end = (edge + self.away[1 - along] * points(d) for d in LEADER_POINTS)
        segments = []
        for i, text in enumerate(self.texts):
            text.set_visible(i < fits)
            if i >= fits:
                continue
            path = np.empty((3, 2))
            path[:, along] = [targets[i], targets[i], placed[i]]
            path[:, 1 - along] = [start, elbow, end]
            segments.append(path)
            anchor = path[-1].copy()
            anchor[1 - along] += self.away[1 - along] * points(1)
            text.set_position(anchor)
        self.leaders.set_segments(segments)


class _OuterTicks(MaxNLocator):
    """Keeps only the lowest and highest round ticks in view, for narrow tracks."""

    def __init__(self):
        super().__init__(nbins=2, steps=[1, 2, 2.5, 5, 10])

    def tick_values(self, vmin: float, vmax: float) -> np.ndarray:
        ticks = super().tick_values(vmin, vmax)
        low, high = sorted((vmin, vmax))
        slack = 1e-9 * (high - low)
        inside = ticks[(ticks >= low - slack) & (ticks <= high + slack)]
        return inside[[0, -1]] if len(inside) > 2 else inside


def _style_track(panel: Axes, orientation: Orientation, labels: bool) -> None:
    """Show only the value axis of a track panel, and only when `labels` is set.

    The value axis goes on the side away from the chromosome names: the top
    in a vertical karyogram and the right in a horizontal one.
    """
    if orientation == "vertical":
        along, value = panel.yaxis, panel.xaxis
        value.tick_top()
    else:
        along, value = panel.xaxis, panel.yaxis
        value.tick_right()
    along.set_visible(False)
    value.set_major_locator(_OuterTicks())
    value.set_visible(labels)
    value.set_tick_params(labelsize="x-small", length=2)
    panel.patch.set_visible(False)
    for spine in panel.spines.values():
        spine.set_visible(False)
