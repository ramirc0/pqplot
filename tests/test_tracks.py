import numpy as np
import polars as pl
import pytest
from matplotlib.colors import to_rgba

import karyogram_plot as kp

HG38 = kp.Genome.bundled("hg38")
TRACKS = [kp.Track("genes"), kp.Track("cnv", side="below")]
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
    return kp.Karyogram(HG38, chroms=["1", "2"], orientation=orientation, tracks=TRACKS)


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
    assert points["chr1"].norm.vmax == 9.0
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
