import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytest
from matplotlib.colors import to_rgba

import pqplot as pqp

HG38 = pqp.Genome.bundled("hg38")
SEGMENTS = {
    "chrom": ["chr1", "chr1", "chr1", "1", "chr7", "chrM"],
    "start": [0.0, 10e6, 20e6, 30e6, 5e6, 0.0],
    "end": [10e6, 20e6, 30e6, 60e6, 9e6, 100.0],
    "state": ["gain", "gain", "loss", "gain", "loss", "gain"],
}


def fills(kary, chrom, copy=0):
    """Painted collections on one copy of a chromosome, in paint order."""
    ideo = kary.ideograms(chrom)[copy]
    painted = [c for c in ideo.axes.collections if c.get_gid() == f"{ideo.chromosome.name}-paint"]
    return [c for c in painted if np.mean(c.across) == ideo.center]


def spans(collection):
    """(start, end, color) of each painted span, in start order."""
    rows = zip(
        collection.starts, collection.ends, map(tuple, collection.get_facecolors()), strict=True
    )
    return sorted((float(s), float(e), c) for s, e, c in rows)


@pytest.mark.parametrize("make", [dict, pd.DataFrame, pl.DataFrame])
def test_paint_merges_touching_spans_of_one_category(make):
    kary = pqp.Karyogram(HG38, chroms=["1", "7"])
    kary.paint(make(SEGMENTS), start="start", end="end", color="state")
    gain, loss = to_rgba("C0"), to_rgba("C1")
    (chr1,) = fills(kary, "1")
    assert spans(chr1) == [(0, 20e6, gain), (20e6, 30e6, loss), (30e6, 60e6, gain)]
    (chr7,) = fills(kary, "7")
    assert spans(chr7) == [(5e6, 9e6, loss)]


def test_paint_skips_chromosomes_that_are_not_drawn():
    kary = pqp.Karyogram(HG38, chroms=["7"])
    kary.paint(SEGMENTS, start="start", end="end", color="state")
    assert len(fills(kary, "7")) == 1


def test_paint_takes_a_palette_and_lists_it_in_the_legend():
    kary = pqp.Karyogram(HG38, chroms=["1", "7"])
    palette = {"loss": "tab:blue", "gain": "tab:red"}
    kary.paint(SEGMENTS, start="start", end="end", color="state", palette=palette)
    assert spans(fills(kary, "7")[0])[0][2] == to_rgba("tab:blue")
    legend = kary.legend(title="CNV")
    assert [t.get_text() for t in legend.get_texts()] == ["loss", "gain"]
    assert legend.get_title().get_text() == "CNV"
    kary.figure.draw_without_rendering()


def test_paint_palette_must_cover_every_category():
    kary = pqp.Karyogram(HG38, chroms=["1"])
    with pytest.raises(KeyError, match="loss"):
        kary.paint(SEGMENTS, start="start", end="end", color="state", palette={"gain": "red"})


def test_paint_values_through_a_colormap():
    kary = pqp.Karyogram(HG38)
    bins = {"chrom": ["chr1"] * 3, "start": [0, 1e6, 2e6], "end": [1e6, 2e6, 3e6]}
    heat = kary.paint(bins | {"n": [0.0, 5.0, 10.0]}, start="start", end="end", value="n")
    assert (heat.norm.vmin, heat.norm.vmax) == (0, 10)
    (chr1,) = fills(kary, "1")
    np.testing.assert_allclose([c for *_, c in spans(chr1)], heat.to_rgba([0.0, 5.0, 10.0]))
    kary.figure.colorbar(heat, ax=kary.ax)
    kary.figure.draw_without_rendering()


def test_paint_without_spans_fills_whole_chromosomes():
    kary = pqp.Karyogram(HG38, chroms=["1", "2"])
    kary.paint({"chrom": ["1", "2"], "split": ["train", "test"]}, color="split")
    assert spans(fills(kary, "2")[0]) == [(0, HG38["2"].length, to_rgba("C1"))]


def test_paint_without_categories_uses_one_color():
    kary = pqp.Karyogram(HG38, chroms=["1"])
    kary.paint({"chrom": ["1"], "start": [0], "end": [5e6]}, start="start", end="end")
    assert spans(fills(kary, "1")[0]) == [(0, 5e6, to_rgba("C0"))]
    with pytest.raises(ValueError, match="categories"):
        kary.legend()


def test_paint_copies_by_column():
    kary = pqp.Karyogram(HG38, chroms=["1"], copies=2)
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
    kary = pqp.Karyogram(HG38, chroms=["1"], copies=2)
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
    kary = pqp.Karyogram(HG38, chroms=["1"])
    with pytest.raises(ValueError, match=match):
        kary.paint(SEGMENTS, **kwargs)


def test_paint_in_a_user_axes_with_a_legend_keeps_layout_clean():
    fig, (a, b) = plt.subplots(1, 2, layout="constrained", figsize=(8, 4))
    kary = pqp.Karyogram(HG38, ax=a, groups=2)
    kary.paint(SEGMENTS, start="start", end="end", color="state")
    legend = kary.legend()
    fig.draw_without_rendering()
    assert legend.get_window_extent().x1 <= b.get_window_extent().x0


def body_points(kary):
    return kary.panel("1").get_window_extent().width * 72 / kary.figure.dpi


def test_legend_widens_the_karyogram_own_figure_and_keeps_body_width():
    kary = pqp.Karyogram(HG38, chroms=["1", "2"])
    kary.figure.draw_without_rendering()
    before = body_points(kary), kary.figure.get_figwidth()
    kary.paint({"chrom": ["1"], "state": ["homozygous deletion"]}, color="state")
    kary.legend(title="CNV")
    kary.figure.draw_without_rendering()
    assert body_points(kary) == pytest.approx(before[0], rel=0.05)
    assert kary.figure.get_figwidth() > before[1]


def test_legend_leaves_a_user_figure_size_alone():
    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
    kary = pqp.Karyogram(HG38, ax=ax)
    kary.paint({"chrom": ["1"], "state": ["gain"]}, color="state")
    kary.legend()
    assert tuple(fig.get_size_inches()) == (6, 3)


def test_a_second_legend_replaces_the_first_without_growing_again():
    kary = pqp.Karyogram(HG38, chroms=["1", "2"])
    kary.paint({"chrom": ["1"], "state": ["gain"]}, color="state")
    kary.legend()
    width = kary.figure.get_figwidth()
    kary.legend()
    assert kary.figure.get_figwidth() == pytest.approx(width)


def bins(chroms):
    return {
        "chrom": chroms,
        "start": [0.0] * len(chroms),
        "end": [5e6] * len(chroms),
        "n": np.arange(len(chroms), dtype=float),
    }


def test_colorbar_widens_the_karyogram_own_figure_and_keeps_body_width():
    kary = pqp.Karyogram(HG38, chroms=["1", "2"])
    kary.figure.draw_without_rendering()
    before = body_points(kary), kary.figure.get_figwidth()
    heat = kary.paint(bins(["1", "2"]), start="start", end="end", value="n")
    bar = kary.colorbar(heat, label="cCREs per Mb")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kary.figure.draw_without_rendering()
    assert body_points(kary) == pytest.approx(before[0], rel=0.05)
    assert kary.figure.get_figwidth() > before[1]
    assert bar.ax.yaxis.get_label().get_text() == "cCREs per Mb"
    host, cax = kary.ax.get_window_extent(), bar.ax.get_window_extent()
    assert cax.x0 > host.x1
    assert cax.width * 72 / kary.figure.dpi == pytest.approx(pqp.karyogram.COLORBAR_POINTS[0])


def test_colorbar_in_a_user_figure_keeps_its_size_and_stays_inside():
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    kary = pqp.Karyogram(HG38, ax=a)
    bar = kary.colorbar(kary.paint(bins(["1"]), start="start", end="end", value="n"))
    fig.draw_without_rendering()
    assert tuple(fig.get_size_inches()) == (8, 4)
    assert bar.ax.get_tightbbox().x1 <= b.get_window_extent().x0
    assert bar.ax.get_window_extent().height * 72 / fig.dpi == pytest.approx(
        pqp.karyogram.COLORBAR_POINTS[1]
    )


def test_paint_skips_spans_with_missing_coordinates_and_keeps_the_rest():
    kary = pqp.Karyogram(HG38, chroms=["1"])
    rows = pl.DataFrame(
        {"chrom": ["1"] * 3, "start": [0.0, 5e6, 20e6], "end": [10e6, None, 30e6], "s": ["a"] * 3}
    )
    kary.paint(rows, start="start", end="end", color="s")
    assert [s[:2] for s in spans(fills(kary, "1")[0])] == [(0, 10e6), (20e6, 30e6)]


def test_paint_skips_rows_with_a_missing_category_or_copy():
    kary = pqp.Karyogram(HG38, chroms=["1"], copies=2)
    rows = {"chrom": ["1"] * 3, "s": ["a", None, "b"], "hap": [1, 2, None]}
    kary.paint(rows, color="s", copy="hap")
    assert [t.get_text() for t in kary.legend().get_texts()] == ["a"]


def test_one_category_keeps_its_color_across_paint_calls():
    kary = pqp.Karyogram(HG38, chroms=["1", "2"])
    kary.paint({"chrom": ["1"], "s": ["a"]}, color="s")
    kary.paint(
        {"chrom": ["2", "2"], "start": [0, 5e7], "end": [1e7, 6e7], "s": ["b", "a"]},
        start="start",
        end="end",
        color="s",
    )
    colors = {s[:2]: s[2] for s in spans(fills(kary, "2")[0])}
    assert colors[(5e7, 6e7)] == to_rgba("C0")
    assert colors[(0, 1e7)] == to_rgba("C1")


def test_legend_lists_only_categories_that_were_drawn():
    kary = pqp.Karyogram(HG38, chroms=["1"])
    kary.paint({"chrom": ["1", "chrM"], "s": ["shown", "ghost"]}, color="s")
    assert [t.get_text() for t in kary.legend().get_texts()] == ["shown"]


def test_value_scale_spans_only_the_drawn_rows():
    kary = pqp.Karyogram(HG38, chroms=["1"])
    bins = {"chrom": ["1", "2"], "start": [0, 0], "end": [1e6, 1e6], "n": [1.0, 1000.0]}
    assert kary.paint(bins, start="start", end="end", value="n").norm.vmax == 1


@pytest.mark.parametrize("column", ["start", "s"])
def test_paint_rejects_columns_of_different_lengths(column):
    rows = {"chrom": ["1", "1"], "start": [0, 5e6], "end": [1e6, 6e6], "s": ["a", "b"]}
    rows[column] = rows[column][:1]
    with pytest.raises(ValueError, match="length"):
        pqp.Karyogram(HG38, chroms=["1"]).paint(rows, start="start", end="end", color="s")


def test_paint_names_a_missing_column():
    with pytest.raises(KeyError, match="'state'"):
        pqp.Karyogram(HG38, chroms=["1"]).paint({"chrom": ["1"]}, color="state")
