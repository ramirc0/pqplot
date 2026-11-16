import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytest
from matplotlib.colors import to_rgba

import karyogram_plot as kp

HG38 = kp.Genome.bundled("hg38")
SEGMENTS = {
    "chrom": ["chr1", "chr1", "chr1", "1", "chr7", "chrM"],
    "start": [0.0, 10e6, 20e6, 30e6, 5e6, 0.0],
    "end": [10e6, 20e6, 30e6, 60e6, 9e6, 100.0],
    "state": ["gain", "gain", "loss", "gain", "loss", "gain"],
}


def fills(kary, chrom, copy=0):
    """Painted collections on one copy of a chromosome, in paint order."""
    ideo = kary.ideograms(chrom)[copy]
    return [c for c in ideo.axes.collections if getattr(c, "ideogram", None) is ideo]


def spans(collection):
    """(start, end, color) of each painted span, in start order."""
    rows = zip(
        collection.starts, collection.ends, map(tuple, collection.get_facecolors()), strict=True
    )
    return sorted((float(s), float(e), c) for s, e, c in rows)


@pytest.mark.parametrize("make", [dict, pd.DataFrame, pl.DataFrame])
def test_paint_merges_touching_spans_of_one_category(make):
    kary = kp.Karyogram(HG38, chroms=["1", "7"])
    kary.paint(make(SEGMENTS), start="start", end="end", color="state")
    gain, loss = to_rgba("C0"), to_rgba("C1")
    (chr1,) = fills(kary, "1")
    assert spans(chr1) == [(0, 20e6, gain), (20e6, 30e6, loss), (30e6, 60e6, gain)]
    (chr7,) = fills(kary, "7")
    assert spans(chr7) == [(5e6, 9e6, loss)]


def test_paint_skips_chromosomes_that_are_not_drawn():
    kary = kp.Karyogram(HG38, chroms=["7"])
    kary.paint(SEGMENTS, start="start", end="end", color="state")
    assert len(fills(kary, "7")) == 1


def test_paint_takes_a_palette_and_lists_it_in_the_legend():
    kary = kp.Karyogram(HG38, chroms=["1", "7"])
    palette = {"loss": "tab:blue", "gain": "tab:red"}
    kary.paint(SEGMENTS, start="start", end="end", color="state", palette=palette)
    assert spans(fills(kary, "7")[0])[0][2] == to_rgba("tab:blue")
    legend = kary.legend(title="CNV")
    assert [t.get_text() for t in legend.get_texts()] == ["loss", "gain"]
    assert legend.get_title().get_text() == "CNV"
    kary.figure.draw_without_rendering()


def test_paint_palette_must_cover_every_category():
    kary = kp.Karyogram(HG38, chroms=["1"])
    with pytest.raises(KeyError, match="loss"):
        kary.paint(SEGMENTS, start="start", end="end", color="state", palette={"gain": "red"})


def test_paint_values_through_a_colormap():
    kary = kp.Karyogram(HG38)
    bins = {"chrom": ["chr1"] * 3, "start": [0, 1e6, 2e6], "end": [1e6, 2e6, 3e6]}
    heat = kary.paint(bins | {"n": [0.0, 5.0, 10.0]}, start="start", end="end", value="n")
    assert (heat.norm.vmin, heat.norm.vmax) == (0, 10)
    (chr1,) = fills(kary, "1")
    np.testing.assert_allclose([c for *_, c in spans(chr1)], heat.to_rgba([0.0, 5.0, 10.0]))
    kary.figure.colorbar(heat, ax=kary.ax)
    kary.figure.draw_without_rendering()


def test_paint_without_spans_fills_whole_chromosomes():
    kary = kp.Karyogram(HG38, chroms=["1", "2"])
    kary.paint({"chrom": ["1", "2"], "split": ["train", "test"]}, color="split")
    assert spans(fills(kary, "2")[0]) == [(0, HG38["2"].length, to_rgba("C1"))]


def test_paint_without_categories_uses_one_color():
    kary = kp.Karyogram(HG38, chroms=["1"])
    kary.paint({"chrom": ["1"], "start": [0], "end": [5e6]}, start="start", end="end")
    assert spans(fills(kary, "1")[0]) == [(0, 5e6, to_rgba("C0"))]
    assert kary.legend().get_texts() == []


def test_paint_copies_by_column():
    kary = kp.Karyogram(HG38, chroms=["1"], copies=2)
    tracts = {
        "chrom": ["1", "1", "1"],
        "start": [0, 0, 5e7],
        "end": [5e7, 1e8, 1e8],
        "ancestry": ["AFR", "EUR", "EUR"],
        "hap": [1, 2, 1],
    }
    kary.paint(tracts, start="start", end="end", color="ancestry", copy="hap")
    afr, eur = to_rgba("C0"), to_rgba("C1")
    assert spans(fills(kary, "1", copy=0)[0]) == [(0, 5e7, afr), (5e7, 1e8, eur)]
    assert spans(fills(kary, "1", copy=1)[0]) == [(0, 1e8, eur)]


def test_paint_without_a_copy_column_paints_every_copy():
    kary = kp.Karyogram(HG38, chroms=["1"], copies=2)
    kary.paint({"chrom": ["1"], "state": ["gain"]}, color="state")
    assert len(fills(kary, "1", copy=0)) == len(fills(kary, "1", copy=1)) == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"color": "state", "value": "start"}, "color or value"),
        ({"start": "start"}, "start and end"),
        ({"copy": "state"}, "copies"),
    ],
)
def test_paint_rejects_bad_arguments(kwargs, match):
    kary = kp.Karyogram(HG38, chroms=["1"])
    with pytest.raises(ValueError, match=match):
        kary.paint(SEGMENTS, **kwargs)


def test_paint_in_a_user_axes_with_a_legend_keeps_layout_clean():
    fig, (a, b) = plt.subplots(1, 2, layout="constrained", figsize=(8, 4))
    kary = kp.Karyogram(HG38, ax=a, rows=2)
    kary.paint(SEGMENTS, start="start", end="end", color="state")
    legend = kary.legend()
    fig.draw_without_rendering()
    assert legend.get_window_extent().x1 <= b.get_window_extent().x0


def body_points(kary):
    return kary.panel("1").get_window_extent().width * 72 / kary.figure.dpi


def test_legend_widens_the_karyogram_own_figure_and_keeps_body_width():
    kary = kp.Karyogram(HG38, chroms=["1", "2"])
    kary.figure.draw_without_rendering()
    before = body_points(kary), kary.figure.get_figwidth()
    kary.paint({"chrom": ["1"], "state": ["homozygous deletion"]}, color="state")
    kary.legend(title="CNV")
    kary.figure.draw_without_rendering()
    assert body_points(kary) == pytest.approx(before[0], rel=0.05)
    assert kary.figure.get_figwidth() > before[1]


def test_legend_leaves_a_user_figure_size_alone():
    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
    kary = kp.Karyogram(HG38, ax=ax)
    kary.paint({"chrom": ["1"], "state": ["gain"]}, color="state")
    kary.legend()
    assert tuple(fig.get_size_inches()) == (6, 3)


def test_a_second_legend_replaces_the_first_without_growing_again():
    kary = kp.Karyogram(HG38, chroms=["1", "2"])
    kary.paint({"chrom": ["1"], "state": ["gain"]}, color="state")
    kary.legend()
    width = kary.figure.get_figwidth()
    kary.legend()
    assert kary.figure.get_figwidth() == pytest.approx(width)
