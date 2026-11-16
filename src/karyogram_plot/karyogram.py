"""Whole-genome karyograms built from child panels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from karyogram_plot import layout
from karyogram_plot.genome import Chromosome, Genome, chrom_key
from karyogram_plot.ideograms import Ideogram
from karyogram_plot.panels import Orientation, add_panel

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
        if ax is None:
            ax = _own_axes(grid, orientation)
        ax.set_axis_off()
        self.ax = ax
        self._panels: dict[tuple[str, str | None], Axes] = {}
        self._ideograms: dict[str, list[Ideogram]] = {}
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
