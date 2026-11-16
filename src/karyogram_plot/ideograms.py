"""Ideogram artist, single-chromosome plots and genome-wide axis helpers."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba, to_rgba_array
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.transforms import IdentityTransform
from matplotlib.typing import ColorType

from karyogram_plot import outline
from karyogram_plot.genome import Chromosome, LinearGenome, Region, Stain, chrom_key
from karyogram_plot.spans import Spans

STAIN_COLORS = {"acen": "#b5524a", "gvar": "#c8d1dc", "stalk": "#7f8fa6"}
"""Colors of stains that are not G-bands. G-bands blend the theme's colors."""
BODY = 0.12
"""Share of the foreground color in an unbanded body."""
SHADE = 0.06
"""Share of the foreground color in shaded chromosome spans."""
BOX = 1.35
"""Half-height of a locus box, in body half-widths."""
MIN_BOX_POINTS = 3.0
"""Smallest width of a locus box, in points."""
STRIP_HALF = 0.3
"""Body half-width in a genome strip, as a share of half the Axes height."""
MIN_PAINT_POINTS = 1.5
"""Smallest width of a painted span, in points."""
OVERSHOOT = 1.2


class Ideogram(PathPatch):
    """One chromosome drawn in an Axes, with positions in bp.

    The artist is the chromosome outline. Its round caps and centromere
    waist are sized at draw time, so they stay round after a resize. The
    bands are a separate collection clipped to the outline. Colors come from
    the active theme when the ideogram is made.

    Parameters
    ----------
    chromosome : Chromosome
        Chromosome to draw.
    ax : matplotlib.axes.Axes
        Axes to draw into.
    orientation : {"horizontal", "vertical"}, default "horizontal"
        Direction of the chromosome. Horizontal puts bp on the x axis.
    offset : float, default 0
        Axis position of the chromosome's position 0.
    center : float, default 0
        Axis position of the body's midline, across the chromosome.
    half : float, default 0.5
        Body half-width in axis units, across the chromosome.
    bands : bool, default True
        Whether to draw bands and stalks. Without them the body is one
        plain span.
    **kwargs
        Passed to `matplotlib.patches.PathPatch` to style the outline.

    Attributes
    ----------
    chromosome : Chromosome
        The chromosome drawn.
    bands : karyogram_plot.spans.Spans
        Band fills, one polygon per band.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from karyogram_plot import Genome
    >>> fig, ax = plt.subplots()
    >>> ideo = Ideogram(Genome.bundled("hg38")["chr17"], ax)
    >>> len(ideo.bands.get_paths())
    24
    """

    def __init__(
        self,
        chromosome: Chromosome,
        ax: Axes,
        *,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
        offset: float = 0.0,
        center: float = 0.0,
        half: float = 0.5,
        bands: bool = True,
        **kwargs,
    ):
        if orientation not in ("horizontal", "vertical"):
            raise ValueError(f"orientation must be horizontal or vertical, got {orientation!r}")
        kwargs = {
            "facecolor": "none",
            "edgecolor": plt.rcParams["axes.edgecolor"],
            "linewidth": 0.5,
            "zorder": 1.5,
            "gid": f"{chromosome.name}-outline",
            **kwargs,
        }
        super().__init__(Path(np.zeros((1, 2))), clip_on=False, **kwargs)
        self.chromosome = chromosome
        self.orientation = orientation
        self.offset = offset
        self.center = center
        self.half = half
        self._stalks = bands
        ax.add_patch(self)
        ax.autoscale_view()
        starts, ends, colors = _spans(chromosome, bands)
        # A thin edge in each band's own color hides seams in PDF viewers.
        self.bands = Spans(
            starts + offset,
            ends + offset,
            self._overshoot(),
            orientation=orientation,
            min_points=0,
            facecolors=colors,
            edgecolors=colors,
            linewidths=0.25,
            zorder=self.get_zorder() - 0.1,
            gid=f"{chromosome.name}-bands",
        )
        ax.add_collection(self.bands, autolim=False)
        self.bands.set_clip_path(self)
        self.bands.set_clip_box(None)
        if chromosome.telocentric:
            acen = STAIN_COLORS["acen"]
            knob = _Knob(
                self,
                facecolor=acen,
                edgecolor=acen,
                linewidth=0.25,
                zorder=self.get_zorder() - 0.05,
                gid=f"{chromosome.name}-knob",
            )
            ax.add_artist(knob)
            knob.set_clip_path(self)
            knob.set_clip_box(None)

    def xy(self, along: object, across: object) -> np.ndarray:
        """Map chromosome coordinates to data coordinates.

        Parameters
        ----------
        along : array-like of float
            Positions along the chromosome in bp.
        across : array-like of float
            Positions across it in body half-widths. The body edges are at
            -1 and 1.

        Returns
        -------
        numpy.ndarray
            Data coordinates with a last axis of length 2.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from karyogram_plot import Chromosome
        >>> fig, ax = plt.subplots()
        >>> Ideogram(Chromosome("chr1", 100), ax, offset=10, half=0.4).xy([5], [1]).tolist()
        [[15.0, 0.4]]
        """
        along, across = np.broadcast_arrays(
            self.offset + np.asarray(along, dtype=float),
            self.center + self.half * np.asarray(across, dtype=float),
        )
        pair = (along, across) if self.orientation == "horizontal" else (across, along)
        return np.stack(pair, axis=-1)

    def paint(self, starts: object, ends: object, colors: object, **kwargs) -> Spans:
        """Fill spans of the body with color, over the bands.

        Spans narrower than `MIN_PAINT_POINTS` widen about their middle at
        draw time, so small spans stay visible. Narrower spans draw over
        wider ones. The fills are clipped to the outline and sit under a
        telocentric knob.

        Parameters
        ----------
        starts, ends : array-like of float
            Spans along the chromosome in bp.
        colors : color or sequence of color
            One color for every span, or one per span.
        **kwargs
            Passed to `matplotlib.collections.PolyCollection`.

        Returns
        -------
        karyogram_plot.spans.Spans
            The fills, widest first.

        Raises
        ------
        ValueError
            If the number of colors matches neither 1 nor the spans.

        Examples
        --------
        >>> from karyogram_plot import Genome
        >>> ideo = ideogram(Genome.bundled("hg38")["chr9"])
        >>> fills = ideo.paint([0, 1e8], [4e7, 1.2e8], ["tab:red", "tab:blue"])
        >>> len(fills.get_facecolors())
        2
        """
        starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
        colors = to_rgba_array(colors)
        if len(colors) not in (1, len(starts)):
            raise ValueError(f"got {len(colors)} colors for {len(starts)} spans")
        order = np.argsort(starts - ends, kind="stable")
        if len(colors) > 1:
            colors = colors[order]
        kwargs = {
            "facecolors": colors,
            "edgecolors": colors,
            "linewidths": 0.25,
            "zorder": self.get_zorder() - 0.08,
            "gid": f"{self.chromosome.name}-paint",
            **kwargs,
        }
        fills = Spans(
            self.offset + starts[order],
            self.offset + ends[order],
            self._overshoot(),
            orientation=self.orientation,
            min_points=MIN_PAINT_POINTS,
            **kwargs,
        )
        self.axes.add_collection(fills, autolim=False)
        fills.set_clip_path(self)
        fills.set_clip_box(None)
        return fills

    def get_path(self) -> Path:
        """Return the outline, sized for the current Axes scale.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from karyogram_plot import Chromosome
        >>> fig, ax = plt.subplots()
        >>> Ideogram(Chromosome("chr1", 100), ax).get_path().vertices.shape[1]
        2
        """
        along, across = outline.vertices(self.chromosome, self._cap(), stalks=self._stalks)
        xy = self.xy(along, across)
        return Path(np.vstack([xy, xy[:1]]), closed=True)

    def zoom(
        self, region: Region, into: Axes, *, color: ColorType = "C3"
    ) -> tuple[PathPatch, PathPatch]:
        """Box a locus on the ideogram and connect it to another Axes.

        The box keeps a width of at least `MIN_BOX_POINTS`, so small loci
        stay visible. Two lines join the box to the near corners of the
        target Axes. Both are placed at draw time and follow layout changes.

        Parameters
        ----------
        region : Region
            Locus on this chromosome.
        into : matplotlib.axes.Axes
            Axes that shows the locus, usually with x limits set to it.
        color : color, default "C3"
            Color of the box and the lines.

        Returns
        -------
        box, lines : matplotlib.patches.PathPatch
            The locus box and the connecting lines.

        Raises
        ------
        ValueError
            If the region lies on another chromosome.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from karyogram_plot import Genome, Region
        >>> locus = Region.parse("chr17:43,044,295-43,125,483")
        >>> fig, (head, body) = plt.subplots(2, 1, height_ratios=[1, 5])
        >>> ideo = ideogram(Genome.bundled("hg38")[locus.chrom], ax=head)
        >>> _ = body.set_xlim(locus.start, locus.end)
        >>> box, lines = ideo.zoom(locus, into=body)
        """
        if chrom_key(region.chrom) != chrom_key(self.chromosome.name):
            raise ValueError(f"region on {region.chrom} is not on {self.chromosome.name}")
        box = _LocusBox(
            self,
            region,
            edgecolor=color,
            facecolor="none",
            linewidth=0.75,
            zorder=self.get_zorder() + 0.1,
            clip_on=False,
        )
        self.axes.add_artist(box)
        lines = _ZoomLines(box, into, edgecolor=color, facecolor="none", linewidth=0.5)
        lines.set_in_layout(False)
        self.figure.add_artist(lines)
        return box, lines

    def _cap(self) -> float:
        """Return the cap radius in bp at the current Axes scale."""
        pixels_per_bp, half_pixels = self._scale()
        cap = half_pixels / pixels_per_bp if pixels_per_bp > 0 and half_pixels > 0 else None
        return cap or 0.02 * self.chromosome.length

    def _overshoot(self) -> tuple[float, float]:
        """Return the across edges of fills that overshoot the body, for clipping."""
        return self.center - OVERSHOOT * self.half, self.center + OVERSHOOT * self.half

    def _scale(self) -> tuple[float, float]:
        """Return pixels per bp along the body and the body half-width in pixels."""
        length = self.chromosome.length
        p0, p1, p2 = self.axes.transData.transform(self.xy([0, length, 0], [0, 0, 1]))
        return float(np.hypot(*(p1 - p0)) / length), float(np.hypot(*(p2 - p0)))


class _LocusBox(PathPatch):
    """Box around a locus, at least `MIN_BOX_POINTS` wide."""

    def __init__(self, ideogram: Ideogram, region: Region, **kwargs):
        super().__init__(Path(np.zeros((1, 2))), **kwargs)
        self.ideogram = ideogram
        self.region = region

    def extent(self) -> tuple[float, float]:
        pixels_per_bp, _ = self.ideogram._scale()
        min_bp = MIN_BOX_POINTS * self.figure.dpi / 72 / pixels_per_bp
        mid = (self.region.start + self.region.end) / 2
        half = max((self.region.end - self.region.start) / 2, min_bp / 2)
        return mid - half, mid + half

    def get_path(self) -> Path:
        return _span(self.ideogram, *self.extent(), BOX)


class _Knob(PathPatch):
    """Centromere fill of a telocentric knob, from position 0 to its waist."""

    def __init__(self, ideogram: Ideogram, **kwargs):
        super().__init__(Path(np.zeros((1, 2))), **kwargs)
        self.ideogram = ideogram

    def get_path(self) -> Path:
        return _span(self.ideogram, 0, outline.KNOB * self.ideogram._cap(), OVERSHOOT)


def _span(ideogram: Ideogram, start: float, end: float, half: float) -> Path:
    """Return a closed box over a chromosome span, `half` body half-widths tall."""
    return Path(
        ideogram.xy([start, end, end, start, start], [-half, -half, half, half, -half]),
        closed=True,
    )


class _ZoomLines(PathPatch):
    """Two lines from a locus box to the near corners of an Axes, in pixels."""

    def __init__(self, box: _LocusBox, target: Axes, **kwargs):
        super().__init__(Path(np.zeros((1, 2))), transform=IdentityTransform(), **kwargs)
        self.box = box
        self.target = target

    def get_path(self) -> Path:
        ideogram = self.box.ideogram
        s0, s1 = self.box.extent()
        sides = [
            ideogram.axes.transData.transform(ideogram.xy([s0, s1], [t, t])) for t in (-BOX, BOX)
        ]
        bbox = self.target.get_window_extent()
        center = (bbox.p0 + bbox.p1) / 2
        side = min(sides, key=lambda pts: np.hypot(*(pts.mean(axis=0) - center)))
        along = 0 if ideogram.orientation == "horizontal" else 1
        side = side[np.argsort(side[:, along])]
        ends = np.empty((2, 2))
        ends[:, along] = sorted([bbox.p0[along], bbox.p1[along]])
        edge = bbox.p1 if side[0, 1 - along] > center[1 - along] else bbox.p0
        ends[:, 1 - along] = edge[1 - along]
        return Path(
            [side[0], ends[0], side[1], ends[1]],
            [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO],
        )


class _SparseLabels(Artist):
    """Shows only the labels that fit from left to right, decided at draw time."""

    def __init__(self, ax: Axes, texts: list[Text], gap_points: float = 2.0):
        super().__init__()
        self.texts = sorted(texts, key=lambda t: t.get_position()[0])
        self.gap_points = gap_points
        self.set_zorder(min(t.get_zorder() for t in texts) - 0.1)
        self.set_in_layout(False)
        ax.add_artist(self)

    def draw(self, renderer) -> None:
        for text in self.texts:
            text.set_visible(True)
        boxes = [text.get_window_extent(renderer) for text in self.texts]
        gap = renderer.points_to_pixels(self.gap_points)
        for text, keep in zip(
            self.texts, _fitting([(b.x0, b.x1) for b in boxes], gap), strict=True
        ):
            text.set_visible(keep)


class _SparseTicks(Artist):
    """Blanks the x tick labels that would overlap a neighbor, decided at draw time."""

    def __init__(self, ax: Axes, names: list[str], gap_points: float = 2.0):
        super().__init__()
        self.names = names
        self.gap_points = gap_points
        self.set_zorder(ax.xaxis.get_zorder() - 0.1)
        self.set_in_layout(False)
        ax.add_artist(self)

    def draw(self, renderer) -> None:
        axis = self.axes.xaxis
        font = axis.get_ticklabels()[0].get_fontproperties()
        locs = axis.get_ticklocs()
        x = self.axes.transData.transform(np.column_stack([locs, np.zeros_like(locs)]))[:, 0]
        widths = [
            renderer.get_text_width_height_descent(n, font, ismath=False)[0] for n in self.names
        ]
        spans = [(xi - w / 2, xi + w / 2) for xi, w in zip(x, widths, strict=True)]
        keep = _fitting(spans, renderer.points_to_pixels(self.gap_points))
        axis.set_ticklabels([n if k else "" for n, k in zip(self.names, keep, strict=True)])


def ideogram(
    chromosome: Chromosome, ax: Axes | None = None, *, bands: bool = True, labels: bool = True
) -> Ideogram:
    """Draw one chromosome across an Axes, with x in bp.

    The name goes to the left of the chromosome. Band names go below it,
    and those that would overlap a neighbor are hidden at draw time.

    Parameters
    ----------
    chromosome : Chromosome
        Chromosome to draw.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. By default, a new figure sized for one
        chromosome.
    bands : bool, default True
        Whether to draw bands.
    labels : bool, default True
        Whether to name the bands.

    Returns
    -------
    Ideogram

    Examples
    --------
    >>> from karyogram_plot import Genome
    >>> ideo = ideogram(Genome.bundled("hg38")["chr17"])
    >>> ideo.axes.get_xlim() == (0, ideo.chromosome.length)
    True
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 0.75), layout="constrained")
    ideo = Ideogram(chromosome, ax, bands=bands)
    ax.set(xlim=(0, chromosome.length), ylim=(-1, 1))
    ax.set_axis_off()
    ax.annotate(
        chromosome.name,
        xy=(0, 0.5),
        xycoords=ideo,
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        va="center",
    )
    if labels and chromosome.bands:
        _SparseLabels(
            ax, [_below(ax, (b.start + b.end) / 2, b.name, ideo.half) for b in chromosome.bands]
        )
    return ideo


def genome_strip(
    linear: LinearGenome, ax: Axes | None = None, *, bands: bool = True, labels: bool = True
) -> list[Ideogram]:
    """Draw every chromosome of a concatenated genome in one row.

    Parameters
    ----------
    linear : LinearGenome
        Chromosomes and their offsets on the axis.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. By default, a new figure sized for one strip.
    bands : bool, default True
        Whether to draw bands.
    labels : bool, default True
        Whether to name the chromosomes below the strip. Names that would
        overlap a neighbor are hidden at draw time.

    Returns
    -------
    list of Ideogram
        One per chromosome, in axis order.

    Examples
    --------
    >>> from karyogram_plot import Genome
    >>> strip = genome_strip(Genome.bundled("mm39").concatenated(gap=5_000_000))
    >>> len(strip), strip[1].offset
    (21, 200154279.0)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 0.6), layout="constrained")
    ideograms = [
        Ideogram(c, ax, offset=linear.offset(c.name), half=STRIP_HALF, bands=bands)
        for c in linear.genome
    ]
    ax.set(xlim=(0, linear.length), ylim=(-1, 1))
    ax.set_axis_off()
    if labels:
        _SparseLabels(
            ax,
            [
                _below(ax, i.offset + i.chromosome.length / 2, i.chromosome.label, STRIP_HALF)
                for i in ideograms
            ],
        )
    return ideograms


def chromosome_axis(
    linear: LinearGenome, ax: Axes, *, shade: bool | ColorType = True
) -> PolyCollection | None:
    """Label a concatenated x axis by chromosome, for genome-wide plots.

    Ticks go to the chromosome midpoints, named without the ``chr`` prefix.
    Names that would overlap a neighbor are hidden at draw time. Every other
    chromosome gets a shaded background. The x limits are set to
    the whole genome.

    Parameters
    ----------
    linear : LinearGenome
        Chromosomes and their offsets on the axis.
    ax : matplotlib.axes.Axes
        Axes whose x axis holds concatenated positions.
    shade : bool or color, default True
        Shading of the second, fourth and later chromosomes. True blends a
        little of the theme's foreground into the background. False draws
        none.

    Returns
    -------
    matplotlib.collections.PolyCollection or None
        The shaded spans, or None without shading.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from karyogram_plot import Genome
    >>> linear = Genome.bundled("hg38").concatenated(gap=5_000_000)
    >>> fig, ax = plt.subplots()
    >>> _ = ax.scatter(linear.position(["chr1", "chr2"], [1e6, 5e7]), [3.0, 8.0])
    >>> shade = chromosome_axis(linear, ax)
    >>> [label.get_text() for label in ax.get_xticklabels()][:3]
    ['1', '2', '3']
    """
    chromosomes = list(linear.genome)
    starts = np.array([linear.offset(c.name) for c in chromosomes])
    ends = starts + [c.length for c in chromosomes]
    ax.set_xticks((starts + ends) / 2, [c.label for c in chromosomes])
    ax.tick_params(axis="x", length=0)
    _SparseTicks(ax, [c.label for c in chromosomes])
    ax.set_xlim(0, linear.length)
    if shade is False:
        return None
    x0, x1 = starts[1::2], ends[1::2]
    quads = np.stack(
        [np.stack([x0, x1, x1, x0], axis=1), np.tile([0, 0, 1, 1], (len(x0), 1))], axis=-1
    )
    spans = PolyCollection(
        quads,
        facecolors=_blend(SHADE) if shade is True else shade,
        linewidths=0,
        zorder=0,
        transform=ax.get_xaxis_transform(),
        gid="chromosome-shade",
    )
    ax.add_collection(spans, autolim=False)
    return spans


def _fitting(spans: list[tuple[float, float]], gap: float) -> list[bool]:
    """Keep spans from left to right that clear the last kept span by `gap`."""
    keep, right = [], -np.inf
    for x0, x1 in spans:
        keep.append(bool(x0 >= right + gap))
        if keep[-1]:
            right = x1
    return keep


def _below(ax: Axes, x: float, text: str, half: float) -> Text:
    return ax.text(x, -half - 0.1, text, ha="center", va="top", fontsize="x-small", clip_on=False)


def _blend(share: float) -> tuple[float, float, float, float]:
    """Mix the theme's foreground into its background."""
    paper = np.array(to_rgba(plt.rcParams["axes.facecolor"]))
    ink = np.array(to_rgba(plt.rcParams["axes.edgecolor"]))
    return tuple((1 - share) * paper + share * ink)


def _stain_color(stain: Stain) -> ColorType:
    density = stain.density
    if density is not None:
        return _blend(density)
    return STAIN_COLORS.get(stain.name, _blend(0))


def _spans(chromosome: Chromosome, bands: bool) -> tuple[np.ndarray, np.ndarray, list]:
    """Return starts, ends and colors of the body fills."""
    if bands and chromosome.bands:
        spans = [(b.start, b.end, _stain_color(b.stain)) for b in chromosome.bands]
    else:
        spans = [(0, chromosome.length, _blend(BODY))]
    starts, ends, colors = zip(*spans, strict=True)
    return np.array(starts, dtype=float), np.array(ends, dtype=float), list(colors)
