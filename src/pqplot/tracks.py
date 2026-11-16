"""Data layers drawn into the track panels of a karyogram."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import Colormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import StepPatch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import IdentityTransform, ScaledTranslation, Transform
from matplotlib.typing import ColorType

from pqplot.colors import category_colors, value_scale
from pqplot.columns import column, rows_by_chrom
from pqplot.genome import chrom_key
from pqplot.ideograms import MIN_PAINT_POINTS
from pqplot.labels import capacity, spread
from pqplot.panels import Orientation
from pqplot.spans import Spans

LEADER_POINTS = (5.0, 7.0, 10.0)
"""Start, elbow and end of a label leader, in points from the track edge."""


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
    >>> from pqplot import Genome, Karyogram, Track
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
        >>> from pqplot import Genome, Karyogram, Track
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

        codes, swatches = category_colors(categories, len(along), palette, self._legend, handle)
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
        >>> from pqplot import Genome, Karyogram, Track
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
        >>> from pqplot import Genome, Karyogram, Track
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
        >>> from pqplot import Genome, Karyogram, Track
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
        >>> from pqplot import Genome, Karyogram, Track
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
        >>> from pqplot import Genome, Karyogram, Track
        >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("expr")])
        >>> bins = {"chrom": ["1", "1"], "start": [0, 1e6], "end": [1e6, 2e6], "value": [3, 8]}
        >>> kary.track("expr").heatmap(bins, cmap="magma").norm.vmax
        8.0
        """
        starts, ends = column(data, start).astype(float), column(data, end).astype(float)
        values = column(data, value).astype(float)
        scale = value_scale(values, cmap, norm)
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
        >>> from pqplot import Genome, Karyogram, Track
        >>> kary = Karyogram(Genome.bundled("hg38"), tracks=[Track("peaks")])
        >>> peaks = {"chrom": ["1", "2"], "start": [1e6, 3e6], "end": [1.0002e6, 3.0001e6]}
        >>> list(kary.track("peaks").intervals(peaks))
        ['chr1', 'chr2']
        """
        starts, ends = column(data, start).astype(float), column(data, end).astype(float)
        categories = None if color is None else column(data, color)
        codes, swatches = category_colors(categories, len(starts), palette, self._legend)
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
    >>> from pqplot import Genome, Karyogram, Track
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
        >>> from pqplot import Genome, Karyogram, Track
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


def style_track(panel: Axes, orientation: Orientation, labels: bool) -> None:
    """Show only the value axis of a track panel, and only when `labels` is set.

    The value axis goes on the side away from the chromosome names: the top
    in a vertical karyogram and the right in a horizontal one. It ticks only
    its outermost round values, so narrow tracks stay legible.

    Parameters
    ----------
    panel : matplotlib.axes.Axes
        Track panel.
    orientation : {"horizontal", "vertical"}
        Direction of the chromosomes.
    labels : bool
        Whether this panel shows the value axis.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, panel = plt.subplots()
    >>> style_track(panel, "vertical", labels=True)
    >>> panel.xaxis.get_ticks_position()
    'top'
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
