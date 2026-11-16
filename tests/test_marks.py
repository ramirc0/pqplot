import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle

import pqplot as pqp

HG38 = pqp.Genome.bundled("hg38")
HITS = {
    "chrom": ["chr1", "chr1", "chr2", "chrM"],
    "pos": [55e6, 230e6, 21e6, 5.0],
    "trait": ["LDL", "BMI", "LDL", "BMI"],
    "gene": ["PCSK9", "AGT", "APOB", "ND1"],
}


def karyogram(orientation="vertical", side="after", size=5):
    return pqp.Karyogram(
        HG38, chroms=["1", "2"], orientation=orientation, tracks=[pqp.Track("hits", side, size)]
    )


def drawn(kary):
    kary.figure.draw_without_rendering()
    return kary.figure.canvas.get_renderer()


def along_pixel(panel, pos, orientation):
    xy = (0, pos) if orientation == "vertical" else (pos, 0)
    return panel.transData.transform(xy)[1 if orientation == "vertical" else 0]


@pytest.mark.parametrize(
    ("orientation", "side", "marker"),
    [
        ("vertical", "after", "<"),
        ("vertical", "before", ">"),
        ("horizontal", "after", "^"),
        ("horizontal", "before", "v"),
    ],
)
def test_marks_point_at_the_body_from_the_inner_edge(orientation, side, marker):
    kary = karyogram(orientation, side)
    marks = kary.track("hits").mark(HITS, pos="pos")
    assert list(marks) == ["chr1", "chr2"]
    drawn(kary)
    panel = kary.panel("1", "hits")
    points = marks["chr1"]
    style = MarkerStyle(marker)
    expected_path = style.get_path().transformed(style.get_transform())
    np.testing.assert_allclose(points.get_paths()[0].vertices, expected_path.vertices)
    centers = points.get_offset_transform().transform(points.get_offsets())
    box = panel.get_window_extent()
    half = np.sqrt(points.get_sizes()[0]) / 2 * kary.figure.dpi / 72
    axis = 1 if orientation == "vertical" else 0
    inner = {
        ("vertical", "after"): box.x0 + half,
        ("vertical", "before"): box.x1 - half,
        ("horizontal", "after"): box.y1 - half,
        ("horizontal", "before"): box.y0 + half,
    }[orientation, side]
    np.testing.assert_allclose(centers[:, 1 - axis], inner)
    expected = [along_pixel(panel, p, orientation) for p in (55e6, 230e6)]
    np.testing.assert_allclose(centers[:, axis], expected)


def test_marks_color_by_category_with_marker_legend_entries():
    kary = karyogram()
    marks = kary.track("hits").mark(HITS, pos="pos", color="trait")
    np.testing.assert_allclose(marks["chr1"].get_facecolors(), [to_rgba("C0"), to_rgba("C1")])
    legend = kary.legend()
    assert [t.get_text() for t in legend.get_texts()] == ["LDL", "BMI"]
    assert legend.legend_handles[0].get_marker() == "<"


def visible(lane):
    return [t for t in lane.texts if t.get_visible()]


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_crowded_labels_spread_in_order_without_overlap(orientation):
    kary = karyogram(orientation)
    genes = {"chrom": ["1"] * 6, "pos": [50e6 + i * 1e5 for i in range(6)], "gene": list("ABCDEF")}
    lanes = kary.track("hits").label(genes, pos="pos", text="gene")
    renderer = drawn(kary)
    boxes = [t.get_window_extent(renderer) for t in visible(lanes["chr1"])]
    assert len(boxes) == 6
    axis = 1 if orientation == "vertical" else 0
    mids = [(b.p0[axis] + b.p1[axis]) / 2 for b in boxes]
    assert mids == sorted(mids, reverse=orientation == "vertical")
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    panel = kary.panel("1", "hits").get_window_extent()
    for box in boxes:
        assert panel.x0 - 1 <= box.x0 and box.x1 <= panel.x1 + 1
        assert panel.y0 - 1 <= box.y0 and box.y1 <= panel.y1 + 1


def test_leaders_run_from_each_target_to_its_label():
    kary = karyogram()
    lanes = kary.track("hits").label(HITS, pos="pos", text="gene")
    drawn(kary)
    panel = kary.panel("1", "hits")
    leaders = lanes["chr1"].leaders.get_segments()
    starts = sorted(segment[0][1] for segment in leaders)
    expected = sorted(along_pixel(panel, p, "vertical") for p in (55e6, 230e6))
    np.testing.assert_allclose(starts, expected)
    for segment, text in zip(leaders, lanes["chr1"].texts, strict=True):
        assert segment[-1][1] == pytest.approx(text.get_position()[1])
        assert segment[0][0] < segment[-1][0] <= text.get_position()[0]


def test_labels_that_do_not_fit_drop_in_input_order_with_a_warning():
    kary = karyogram()
    count = 400
    genes = {"chrom": ["2"] * count, "pos": np.linspace(0, 2e8, count), "gene": ["G"] * count}
    lanes = kary.track("hits").label(genes, pos="pos", text="gene")
    with pytest.warns(UserWarning, match="labels do not fit"):
        drawn(kary)
    shown = visible(lanes["chr2"])
    assert 0 < len(shown) < count
    assert shown == lanes["chr2"].texts[: len(shown)]


def test_mark_legend_handles_match_the_mark_size():
    kary = karyogram()
    kary.track("hits").mark(HITS, pos="pos", color="trait", size=25)
    handle = kary.legend().legend_handles[0]
    assert handle.get_markersize() == pytest.approx(5)


@pytest.mark.parametrize(
    "draw", [lambda track: track.mark(HITS), lambda track: track.label(HITS, text="gene")]
)
def test_marks_and_labels_hide_the_track_value_axis(draw):
    kary = karyogram()
    draw(kary.track("hits"))
    assert not any(p.xaxis.get_visible() for p in kary.panels("hits").values())
