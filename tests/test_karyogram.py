import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pqplot as pqp

HG38 = pqp.Genome.bundled("hg38")
MM39 = pqp.Genome.bundled("mm39")


def drawn(kary):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kary.figure.draw_without_rendering()
    return kary


def pixel(panel, along, orientation="vertical"):
    """Display coordinate of a bp position along a panel."""
    xy = (0, along) if orientation == "vertical" else (along, 0)
    return panel.transData.transform(xy)[1 if orientation == "vertical" else 0]


def test_every_chromosome_gets_a_body_panel_in_genome_order():
    kary = pqp.Karyogram(HG38)
    assert list(kary.panels()) == [c.name for c in HG38]
    assert kary.panel("1") is kary.panel("chr1")


def test_panels_are_children_of_the_host_and_inside_it():
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    kary = drawn(pqp.Karyogram(HG38, ax=ax))
    assert kary.ax is ax
    assert kary.figure is fig
    for panel in kary.panels().values():
        assert panel in ax.child_axes
        box = panel.get_window_extent()
        assert ax.bbox.x0 - 1e-6 <= box.x0 < box.x1 <= ax.bbox.x1 + 1e-6
        assert ax.bbox.y0 - 1e-6 <= box.y0 < box.y1 <= ax.bbox.y1 + 1e-6


def test_own_figure_uses_constrained_layout():
    kary = pqp.Karyogram(HG38)
    assert kary.figure.get_layout_engine() is not None
    drawn(kary)


def test_vertical_columns_run_left_to_right_and_rows_top_to_bottom():
    kary = drawn(pqp.Karyogram(HG38, rows=2))
    boxes = {name: p.get_window_extent() for name, p in kary.panels().items()}
    assert boxes["chr1"].x0 < boxes["chr2"].x0
    assert boxes["chr13"].x0 == pytest.approx(boxes["chr1"].x0)
    assert boxes["chr13"].y1 < boxes["chr1"].y0


def test_rows_take_explicit_groups():
    rows = [["1", "2", "3"], ["X", "Y"]]
    kary = pqp.Karyogram(HG38, rows=rows)
    assert list(kary.panels()) == ["chr1", "chr2", "chr3", "chrX", "chrY"]


def test_rows_and_chroms_are_exclusive():
    with pytest.raises(ValueError, match="chroms"):
        pqp.Karyogram(HG38, rows=[["1"]], chroms=["1"])


def test_all_rows_share_one_bp_scale():
    kary = drawn(pqp.Karyogram(HG38, rows=2))
    scales = [abs(pixel(p, 1e8) - pixel(p, 0)) / 1e8 for p in kary.panels().values()]
    np.testing.assert_allclose(scales, scales[0])


def test_top_alignment_puts_p_terminals_level_and_p_on_top():
    kary = drawn(pqp.Karyogram(HG38, chroms=["1", "21"]))
    a, b = kary.panel("1"), kary.panel("21")
    assert pixel(a, 0) == pytest.approx(pixel(b, 0))
    assert pixel(a, 0) > pixel(a, 1e8)


@pytest.mark.parametrize(
    ("align", "position"),
    [("centromere", lambda c: sum(c.centromere) / 2), ("bottom", lambda c: c.length)],
)
def test_alignment_lines_up_the_chosen_position(align, position):
    kary = drawn(pqp.Karyogram(HG38, chroms=["1", "13", "21"], align=align))
    levels = [pixel(kary.panel(c), position(HG38[c])) for c in ("1", "13", "21")]
    np.testing.assert_allclose(levels, levels[0])


def test_names_sit_between_rows():
    kary = drawn(pqp.Karyogram(HG38, rows=2))
    renderer = kary.figure.canvas.get_renderer()
    name = next(t for t in kary.panel("1").texts if t.get_text() == "1")
    box = name.get_window_extent(renderer)
    assert box.y0 > kary.panel("13").get_window_extent().y1
    assert box.y1 < kary.panel("1").get_window_extent().y0


def test_horizontal_rows_stack_down_with_positions_left_to_right():
    kary = drawn(pqp.Karyogram(HG38, orientation="horizontal", chroms=["1", "2"]))
    a, b = kary.panel("1"), kary.panel("2")
    assert a.get_window_extent().y0 > b.get_window_extent().y1
    assert pixel(a, 0, "horizontal") == pytest.approx(pixel(b, 0, "horizontal"))
    assert pixel(a, 1e8, "horizontal") > pixel(a, 0, "horizontal")


def test_horizontal_names_sit_left_of_the_body():
    kary = drawn(pqp.Karyogram(HG38, orientation="horizontal", chroms=["1", "2"]))
    renderer = kary.figure.canvas.get_renderer()
    name = next(t for t in kary.panel("2").texts if t.get_text() == "chr2")
    assert name.get_window_extent(renderer).x1 < kary.panel("2").get_window_extent().x0


TRACKS = [pqp.Track("genes", size=1.0), pqp.Track("cnv", side="below", size=1.5)]


def test_horizontal_tracks_sit_above_and_below_the_body():
    kary = drawn(pqp.Karyogram(HG38, orientation="horizontal", chroms=["1"], tracks=TRACKS))
    genes, body, cnv = (kary.panel("1", t).get_window_extent() for t in ("genes", None, "cnv"))
    assert genes.y0 > body.y1
    assert body.y0 > cnv.y1
    assert cnv.height == pytest.approx(1.5 * genes.height)


def test_vertical_tracks_sit_left_and_right_of_the_body():
    kary = drawn(pqp.Karyogram(HG38, chroms=["1", "2"], tracks=TRACKS))
    genes, body, cnv = (kary.panel("1", t).get_window_extent() for t in ("genes", None, "cnv"))
    assert genes.x1 < body.x0
    assert body.x1 < cnv.x0
    assert cnv.x1 < kary.panel("2", "genes").get_window_extent().x0


def test_tracks_share_the_bp_axis_with_their_body():
    kary = drawn(pqp.Karyogram(HG38, orientation="horizontal", chroms=["1"], tracks=TRACKS))
    body, cnv = kary.panel("1"), kary.panel("1", "cnv")
    assert cnv.get_xlim() == body.get_xlim()
    assert pixel(cnv, 5e7, "horizontal") == pytest.approx(pixel(body, 5e7, "horizontal"))


def test_one_track_shares_its_value_axis_across_chromosomes():
    kary = pqp.Karyogram(HG38, orientation="horizontal", chroms=["1", "2"], tracks=TRACKS)
    kary.panel("1", "cnv").set_ylim(-2, 2)
    assert kary.panel("2", "cnv").get_ylim() == (-2, 2)
    assert kary.panel("2", "genes").get_ylim() != (-2, 2)


def test_panels_by_track_name():
    kary = pqp.Karyogram(HG38, chroms=["1", "2"], tracks=TRACKS)
    assert list(kary.panels("cnv")) == ["chr1", "chr2"]
    with pytest.raises(KeyError, match="nope"):
        kary.panel("1", "nope")


def test_copies_sit_side_by_side_in_one_body_panel():
    kary = drawn(pqp.Karyogram(HG38, chroms=["1"], copies=2))
    first, second = (i.get_window_extent() for i in kary.ideograms("1"))
    assert first.x1 < second.x0
    assert first.width == pytest.approx(second.width)
    body = kary.panel("1").get_window_extent()
    assert body.x0 <= first.x0 + 1 and second.x1 <= body.x1 + 1


def test_bodies_follow_the_bands_flag():
    kary = pqp.Karyogram(HG38, chroms=["1"], bands=False)
    assert len(kary.ideograms("1")[0].bands.get_paths()) == 1


def test_mouse_karyogram_draws_without_warnings():
    drawn(pqp.Karyogram(MM39, rows=2))


def test_body_width_stays_fixed_in_points_in_its_own_figure():
    kary = drawn(pqp.Karyogram(HG38, chroms=["1", "2"]))
    width = kary.panel("1").get_window_extent().width * 72 / kary.figure.dpi
    assert width == pytest.approx(pqp.karyogram.BODY_POINTS, rel=0.25)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"orientation": "diagonal"}, "orientation"),
        ({"align": "middle"}, "align"),
        ({"copies": 0}, "copies"),
        ({"tracks": [pqp.Track("a"), pqp.Track("a")]}, "unique"),
    ],
)
def test_bad_arguments_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pqp.Karyogram(HG38, **kwargs)


def test_track_rejects_an_unknown_side():
    with pytest.raises(ValueError, match="side"):
        pqp.Track("a", side="left")


@pytest.mark.parametrize(
    ("orientation", "labeled"), [("vertical", "chr1"), ("horizontal", "chr13")]
)
def test_track_value_axis_shows_once_on_the_edge_away_from_the_names(orientation, labeled):
    kary = pqp.Karyogram(HG38, orientation=orientation, rows=2, tracks=TRACKS)
    panels = kary.panels("cnv")
    value = "xaxis" if orientation == "vertical" else "yaxis"
    along = "yaxis" if orientation == "vertical" else "xaxis"
    assert [name for name, p in panels.items() if getattr(p, value).get_visible()] == [labeled]
    assert not any(getattr(p, along).get_visible() for p in panels.values())
    side = getattr(panels[labeled], value).get_ticks_position()
    assert side == ("top" if orientation == "vertical" else "right")
