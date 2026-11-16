"""Whole-genome karyograms built from child panels."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection
from matplotlib.colors import Colormap, Normalize, to_rgba_array
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, StepPatch
from matplotlib.typing import ColorType

from karyogram_plot import layout
from karyogram_plot.columns import column, rows_by_chrom
from karyogram_plot.genome import Chromosome, Genome, chrom_key
from karyogram_plot.ideograms import MIN_PAINT_POINTS, Ideogram
from karyogram_plot.panels import Orientation, add_panel
from karyogram_plot.spans import Spans

BODY_POINTS = 7.0
"""Body width in a karyogram's own figure, in points."""
LONGEST_POINTS = {"vertical": 180.0, "horizontal": 432.0}
"""Length of the longest group in a karyogram's own figure, in points."""
NAME_POINTS = {"vertical": 16.0, "horizontal": 40.0}
"""Room for chromosome names in a karyogram's own figure, in points."""
PAD_POINTS = 6.0
"""Room for the layout engine's padding in a karyogram's own figure, in points."""


@dataclass(frozen=True)
class Track:
    """A strip of data beside every chromosome of a karyogram.

    Parameters
    ----------
    name : str
        Name used to find the track's panels.
    side : {"above", "below"}, default "above"
        Side of the body. In a vertical karyogram, above is left and below
        is right.
    size : float, default 2
        Width across the chromosome, in body widths.

    Examples
    --------
    >>> Track("cnv", side="below", size=1.5).size
    1.5
    """

    name: str
    side: Literal["above", "below"] = "above"
    size: float = 2.0

    def __post_init__(self):
        if self.side not in ("above", "below"):
            raise ValueError(f"side must be above or below, got {self.side!r}")


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
    rows : int or sequence of sequence of str, default 1
        Number of groups to wrap the chromosomes into, or the chromosome
        names of each group. Groups stack down a vertical karyogram and sit
        side by side in a horizontal one.
    align : {"top", "centromere", "bottom"}, default "top"
        What lines up within a group: the starts, the centromeres or the
        ends. In a horizontal karyogram, top is left.
    chroms : sequence of str, optional
        Chromosomes to draw, in order. By default, all of `genome`. Not
        allowed with explicit `rows`.
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
    >>> from karyogram_plot import Genome
    >>> kary = Karyogram(Genome.bundled("hg38"), rows=2, align="centromere")
    >>> len(kary.panels())
    24
    """

    def __init__(
        self,
        genome: Genome,
        ax: Axes | None = None,
        *,
        orientation: Orientation = "vertical",
        rows: int | Sequence[Sequence[str]] = 1,
        align: layout.Align = "top",
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
        if isinstance(rows, int):
            chosen = [genome[c] for c in chroms] if chroms is not None else list(genome)
            groups = layout.wrap(chosen, rows)
        elif chroms is not None:
            raise ValueError("pass chroms or explicit rows, not both")
        else:
            groups = [[genome[c] for c in row] for row in rows]
        limits = [layout.bounds(group, align) for group in groups]
        grid = layout.Grid(
            tuple(bounds[0][1] - bounds[0][0] for bounds in limits),
            max(len(group) for group in groups),
            layout.slots(
                [(t.name, t.size) for t in tracks if t.side == "above"],
                copies,
                [(t.name, t.size) for t in tracks if t.side == "below"],
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
        body = next(s for s in grid.slots if s.track is None)
        tracks_first: dict[str, Axes] = {}
        along, value = ("sharey", "sharex") if orientation == "vertical" else ("sharex", "sharey")
        for g, group in enumerate(groups):
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
                    _style_track(track, orientation, labels=j == 0)
                    tracks_first.setdefault(slot.track, track)
                    self._panels[key, slot.track] = track

    @property
    def figure(self) -> Figure:
        """Figure that holds the karyogram.

        Examples
        --------
        >>> from karyogram_plot import Genome
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
        >>> from karyogram_plot import Genome
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
        >>> from karyogram_plot import Genome
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
        >>> from karyogram_plot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("cnv")])
        >>> points = kary.track("cnv").scatter({"chrom": ["chr1"], "pos": [5e6], "value": [0.3]})
        >>> list(points)
        ['chr1']
        """
        panels = self.panels(name)
        if not panels:
            raise KeyError(f"no track {name!r}")
        return TrackPanels(panels, self._orientation, self._legend)

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
        >>> from karyogram_plot import Genome
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
        >>> from karyogram_plot import Genome
        >>> kary = Karyogram(Genome.bundled("hg38"), rows=2)
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
        >>> from karyogram_plot import Genome
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
    legend : dict of str to matplotlib.artist.Artist
        Legend entries of the karyogram, which `intervals` adds to.

    Examples
    --------
    >>> from karyogram_plot import Genome
    >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("cnv")])
    >>> type(kary.track("cnv")).__name__
    'TrackPanels'
    """

    def __init__(
        self, panels: dict[str, Axes], orientation: Orientation, legend: dict[str, Artist]
    ):
        self._panels = {chrom_key(name): (name, panel) for name, panel in panels.items()}
        self._orientation = orientation
        self._legend = legend

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
        >>> from karyogram_plot import Genome
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
        >>> from karyogram_plot import Genome
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
        >>> from karyogram_plot import Genome
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
            Passed to `karyogram_plot.spans.Spans`.

        Returns
        -------
        matplotlib.cm.ScalarMappable
            The color scale, for a colorbar.

        Examples
        --------
        >>> from karyogram_plot import Genome
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
            Passed to `karyogram_plot.spans.Spans`.

        Returns
        -------
        dict of str to karyogram_plot.spans.Spans
            Boxes keyed by chromosome name.

        Raises
        ------
        KeyError
            If `palette` lacks a category.

        Examples
        --------
        >>> from karyogram_plot import Genome
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
        transform = (
            panel.get_yaxis_transform()
            if self._orientation == "vertical"
            else panel.get_xaxis_transform()
        )
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


def _category_colors(
    values: np.ndarray | None, rows: int, palette: Mapping | None, legend: dict[str, Artist]
) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's category index and the category colors.

    Categories join `legend`. Without categories, every row gets the first
    color of the theme's cycle and the legend stays as it is.
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
        legend[str(label)] = Patch(facecolor=swatch, edgecolor="none")
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


def _style_track(panel: Axes, orientation: Orientation, labels: bool) -> None:
    """Show only the value axis of a track panel, and only when `labels` is set."""
    along, value = (
        (panel.yaxis, panel.xaxis) if orientation == "vertical" else (panel.xaxis, panel.yaxis)
    )
    along.set_visible(False)
    value.set_visible(labels)
    value.set_tick_params(labelsize="x-small", length=2)
    panel.patch.set_visible(False)
    for spine in panel.spines.values():
        spine.set_visible(False)
