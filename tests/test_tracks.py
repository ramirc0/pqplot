import warnings

import numpy as np
import polars as pl
import pytest
from matplotlib.colors import to_rgba

import pqplot as pqp

HG38 = pqp.Genome.bundled("hg38")
TRACKS = [pqp.Track("genes"), pqp.Track("cnv", side="after")]
POINTS = {
    "chrom": ["chr2", "chr1", "chr1", "chrM"],
    "pos": [5e6, 3e6, 1e6, 10.0],
    "log2": [0.5, -1.0, 2.0, 9.0],
}
BINS = {
    "chrom": ["chr1"] * 3 + ["chr2"],
    "start": [0.0, 1e6, 3e6, 0.0],
    "end": [1e6, 2e6, 4e6, 1e6],
    "n": [4.0, 2.0, 8.0, 6.0],
}


def karyogram(orientation="vertical"):
    return pqp.Karyogram(HG38, chroms=["1", "2"], orientation=orientation, tracks=TRACKS)


def test_scatter_puts_values_across_a_vertical_track():
    kary = karyogram()
    points = kary.track("cnv").scatter(POINTS, pos="pos", value="log2")
    assert list(points) == ["chr1", "chr2"]
    assert points["chr1"].axes is kary.panel("1", "cnv")
    assert points["chr1"].get_offsets().tolist() == [[-1.0, 3e6], [2.0, 1e6]]


def test_scatter_puts_values_up_a_horizontal_track():
    kary = karyogram("horizontal")
    points = kary.track("cnv").scatter(pl.DataFrame(POINTS), pos="pos", value="log2")
    assert points["chr2"].get_offsets().tolist() == [[5e6, 0.5]]


def test_scatter_splits_per_row_styles_and_shares_one_color_scale():
    kary = karyogram()
    points = kary.track("cnv").scatter(
        POINTS, pos="pos", value="log2", c=np.array(POINTS["log2"]), s=4
    )
    assert points["chr1"].get_array().tolist() == [-1.0, 2.0]
    assert points["chr1"].norm.vmin == points["chr2"].norm.vmin == -1.0
    assert points["chr1"].norm.vmax == points["chr2"].norm.vmax == 2.0
    assert points["chr2"].get_sizes().tolist() == [4]


def test_track_panels_share_one_value_axis():
    kary = karyogram()
    kary.track("cnv").scatter(POINTS, pos="pos", value="log2")
    assert kary.panel("1", "cnv").get_xlim() == kary.panel("2", "cnv").get_xlim()


def test_line_runs_in_position_order():
    kary = karyogram("horizontal")
    lines = kary.track("cnv").line(POINTS, pos="pos", value="log2")
    assert lines["chr1"].get_xydata().tolist() == [[1e6, 2.0], [3e6, -1.0]]


def test_area_steps_over_bins_with_gaps_between_them():
    kary = karyogram()
    areas = kary.track("genes").area(BINS, start="start", end="end", value="n")
    steps = areas["chr1"].get_data()
    assert steps.edges.tolist() == [0, 1e6, 2e6, 3e6, 4e6]
    np.testing.assert_array_equal(steps.values, [4, 2, np.nan, 8])
    assert areas["chr1"].orientation == "horizontal"


def test_heatmap_fills_the_track_width_through_one_colormap():
    kary = karyogram()
    heat = kary.track("genes").heatmap(BINS, start="start", end="end", value="n", cmap="magma")
    assert (heat.norm.vmin, heat.norm.vmax) == (2, 8)
    panel = kary.panel("2", "genes")
    (cells,) = panel.collections
    np.testing.assert_allclose(cells.get_facecolors(), heat.to_rgba([6.0]))
    kary.figure.draw_without_rendering()
    box = panel.get_window_extent()
    drawn = cells.get_transform().transform(cells.get_paths()[0].vertices)
    assert (drawn[:, 0].min(), drawn[:, 0].max()) == pytest.approx((box.x0, box.x1))


def test_intervals_color_by_category_with_a_legend_and_keep_a_minimum_width():
    kary = karyogram("horizontal")
    peaks = BINS | {"end": [1e6, 2e6, 3e6 + 10, 1e6], "kind": ["a", "b", "a", "a"]}
    boxes = kary.track("genes").intervals(peaks, start="start", end="end", color="kind")
    np.testing.assert_allclose(
        boxes["chr1"].get_facecolors(), [to_rgba(c) for c in ["C0", "C1", "C0"]]
    )
    assert boxes["chr1"].min_points > 0
    assert [t.get_text() for t in kary.legend().get_texts()] == ["a", "b"]


def test_unknown_track_raises():
    with pytest.raises(KeyError, match="nope"):
        karyogram().track("nope")


def test_tracks_draw_without_warnings():
    kary = karyogram("horizontal")
    kary.track("cnv").scatter(POINTS, pos="pos", value="log2", s=2, rasterized=True)
    kary.track("genes").area(BINS, start="start", end="end", value="n")
    kary.figure.draw_without_rendering()


def test_scatter_markers_have_no_halo_unless_asked():
    kary = karyogram()
    plain = kary.track("cnv").scatter(POINTS, pos="pos", value="log2", s=0.5)
    assert plain["chr1"].get_linewidths().tolist() == [0]
    haloed = kary.track("cnv").scatter(POINTS, pos="pos", value="log2", linewidths=1)
    assert haloed["chr1"].get_linewidths().tolist() == [1]


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_value_axis_labels_clear_the_chromosome_names(orientation):
    kary = pqp.Karyogram(HG38, orientation=orientation, groups=2, tracks=TRACKS)
    kary.track("cnv").scatter(POINTS, pos="pos", value="log2")
    kary.figure.draw_without_rendering()
    renderer = kary.figure.canvas.get_renderer()
    names = [t.get_window_extent(renderer) for p in kary.panels().values() for t in p.texts]
    ticks = [
        label.get_window_extent(renderer)
        for track in ("genes", "cnv")
        for panel in kary.panels(track).values()
        for axis in (panel.xaxis, panel.yaxis)
        if axis.get_visible()
        for label in axis.get_ticklabels(which="both")
        if label.get_visible() and label.get_text()
    ]
    assert ticks
    assert not [(t, n) for t in ticks for n in names if t.overlaps(n)]


def visible_ticks(axis):
    low, high = sorted(axis.get_view_interval())
    return [t for t in axis.get_majorticklocs() if low <= t <= high]


def test_value_axis_ticks_only_the_outermost_round_values():
    kary = karyogram()
    kary.track("cnv").scatter(POINTS, pos="pos", value="log2")
    panel = kary.panel("1", "cnv")
    panel.set_xlim(-2, 2)
    assert visible_ticks(panel.xaxis) == [-2, 2]
    kary.track("genes").area(BINS, start="start", end="end", value="n")
    genes = kary.panel("1", "genes")
    genes.set_xlim(0, 52)
    assert visible_ticks(genes.xaxis) == [0, 50]


def test_value_axis_tick_labels_do_not_overlap_in_a_narrow_track():
    kary = pqp.Karyogram(HG38, tracks=[pqp.Track("cnv", size=1.5)])
    kary.track("cnv").scatter(POINTS, pos="pos", value="log2")
    kary.panel("1", "cnv").set_xlim(-2, 2)
    kary.figure.draw_without_rendering()
    renderer = kary.figure.canvas.get_renderer()
    labels = [
        t.get_window_extent(renderer)
        for t in kary.panel("1", "cnv").xaxis.get_ticklabels()
        if t.get_visible() and t.get_text()
    ]
    assert len(labels) == 2
    assert not labels[0].overlaps(labels[1])


def test_labels_with_missing_positions_leave_the_others_in_place():
    kary = pqp.Karyogram(HG38, chroms=["1"], tracks=[pqp.Track("g", "after", 5)])
    genes = {"chrom": ["1"] * 3, "pos": [10e6, np.nan, 90e6], "text": ["a", "b", "c"]}
    lanes = kary.track("g").label(genes)
    kary.figure.draw_without_rendering()
    assert [t.get_text() for t in lanes["chr1"].texts] == ["a", "c"]
    assert all(np.isfinite(t.get_position()).all() for t in lanes["chr1"].texts)


@pytest.mark.parametrize(
    "bins",
    [
        {"start": [0, 1e6], "end": [2e6, 3e6]},
        {"start": [0, 1e6], "end": [5e6, 2e6]},
        {"start": [2e6], "end": [1e6]},
    ],
)
def test_area_rejects_overlapping_or_reversed_bins(bins):
    rows = {"chrom": ["1"] * len(bins["start"]), "value": [1.0] * len(bins["start"]), **bins}
    with pytest.raises(ValueError, match="chr1"):
        karyogram().track("genes").area(rows)


def test_area_and_line_skip_rows_with_missing_coordinates():
    kary = karyogram()
    rows = {
        "chrom": ["1"] * 3,
        "start": [0, np.nan, 2e6],
        "end": [1e6, 2e6, 3e6],
        "pos": [1, None, 3],
        "value": [1.0, 2.0, None],
    }
    assert kary.track("genes").area(rows)["chr1"].get_data().edges.tolist() == [0, 1e6, 2e6, 3e6]
    assert kary.track("cnv").line(rows)["chr1"].get_ydata().tolist() == [1, 3]


@pytest.mark.parametrize("limit", ["vmin", "vmax"])
def test_scatter_shares_one_color_scale_when_one_limit_is_given(limit):
    kary = pqp.Karyogram(HG38, chroms=["1", "2"], tracks=[pqp.Track("cnv")])
    points = {"chrom": ["1", "2"], "pos": [1e6, 1e6], "value": [1.0, 100.0]}
    drawn = kary.track("cnv").scatter(
        points, c=np.array(points["value"]), **{limit: 0.5 if limit == "vmin" else 200}
    )
    assert {c.norm.vmin for c in drawn.values()} | {c.norm.vmax for c in drawn.values()} <= {
        0.5,
        1.0,
        100.0,
        200.0,
    }
    assert len({(c.norm.vmin, c.norm.vmax) for c in drawn.values()}) == 1


def test_scatter_splits_plain_lists_of_colors_and_sizes():
    kary = karyogram()
    drawn = kary.track("cnv").scatter(POINTS, value="log2", c=[1.0, 2.0, 3.0, 4.0], s=[1, 2, 3, 4])
    assert drawn["chr1"].get_sizes().tolist() == [2, 3]


def test_scatter_with_rgb_colors_does_not_warn():
    kary = karyogram()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kary.track("cnv").scatter(POINTS, value="log2", c=np.tile([1.0, 0, 0], (4, 1)))
        kary.figure.draw_without_rendering()


def test_heatmap_scale_spans_only_the_drawn_rows():
    kary = karyogram()
    rows = {**BINS, "chrom": BINS["chrom"][:3] + ["chr9"], "n": [1.0, 2.0, 3.0, 1000.0]}
    assert kary.track("genes").heatmap(rows, value="n").norm.vmax == 3


def test_marks_skip_missing_categories():
    kary = karyogram()
    hits = {"chrom": ["1", "1"], "pos": [1e6, 2e6], "trait": ["LDL", None]}
    assert len(kary.track("cnv").mark(hits, color="trait")["chr1"].get_offsets()) == 1
