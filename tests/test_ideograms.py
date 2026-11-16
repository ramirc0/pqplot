import matplotlib.pyplot as plt
import numpy as np
import poikilos as pk
import pytest
from matplotlib.colors import to_rgba

import karyogram_plot as kp
from karyogram_plot import ideograms, outline

HG38 = kp.Genome.bundled("hg38")
BRCA1 = kp.Region.parse("chr17:43,044,295-43,125,483")


def display(artist):
    path = artist.get_path()
    return artist.get_transform().transform(path.vertices)


def assert_round_tip(points, tip, radius, along):
    """Vertices within one radius of the tip lie on a circle of that radius."""
    across = 1 - along
    near = np.abs(points[:, along] - tip[along]) <= radius * (1 + 1e-9)
    center = np.array(tip, dtype=float)
    center[along] += radius if points[:, along].min() >= tip[along] - 1e-6 else -radius
    np.testing.assert_allclose(np.hypot(*(points[near] - center).T), radius, rtol=1e-6)
    assert np.ptp(points[near][:, across]) == pytest.approx(2 * radius, rel=1e-6)


def ends(values):
    return values.min(), values.max()


def body_half_pixels(ideo, along):
    p0, p1 = ideo.axes.transData.transform(ideo.xy([0, 0], [0, 1]))
    return abs(p1[1 - along] - p0[1 - along])


def test_caps_are_round_in_pixels_and_follow_resizes():
    ideo = kp.ideogram(HG38["chr1"])
    fig = ideo.figure
    for size in [(6, 0.8), (12, 2)]:
        fig.set_size_inches(size)
        fig.draw_without_rendering()
        points = display(ideo)
        r = body_half_pixels(ideo, along=0)
        left = points[np.argmin(points[:, 0])]
        right = points[np.argmax(points[:, 0])]
        assert_round_tip(points, left, r, along=0)
        assert_round_tip(points, right, r, along=0)


def test_vertical_caps_are_round_with_inverted_axis():
    fig, ax = plt.subplots(figsize=(1, 5))
    ideo = kp.Ideogram(HG38["chr1"], ax, orientation="vertical")
    ax.set(xlim=(-1, 1), ylim=(HG38["chr1"].length, 0))
    fig.draw_without_rendering()
    points = display(ideo)
    r = body_half_pixels(ideo, along=1)
    assert_round_tip(points, points[np.argmin(points[:, 1])], r, along=1)
    assert_round_tip(points, points[np.argmax(points[:, 1])], r, along=1)


def test_bands_sit_at_their_positions_in_bp():
    fig, ax = plt.subplots()
    chrom = HG38["chr17"]
    ideo = kp.Ideogram(chrom, ax, offset=1000)
    fig.draw_without_rendering()
    starts = [path.vertices[:, 0].min() for path in ideo.bands.get_paths()]
    assert starts == [1000 + band.start for band in chrom.bands]


def test_bands_are_clipped_to_the_outline_not_the_axes():
    fig, ax = plt.subplots()
    ideo = kp.Ideogram(HG38["chr1"], ax)
    assert ideo.bands.get_clip_path() is not None
    assert ideo.bands.get_clip_box() is None
    assert not ideo.get_clip_on()


def test_band_colors_follow_the_theme():
    with pk.context("plain-dark"):
        fig, ax = plt.subplots()
        ideo = kp.Ideogram(HG38["chr1"], ax)
        paper = to_rgba(plt.rcParams["axes.facecolor"])
        ink = to_rgba(plt.rcParams["axes.edgecolor"])
    colors = ideo.bands.get_facecolors()
    stains = [band.stain.name for band in HG38["chr1"].bands]
    np.testing.assert_allclose(colors[stains.index("gneg")], paper)
    np.testing.assert_allclose(colors[stains.index("gpos100")], ink)
    assert to_rgba(ideo.get_edgecolor()) == ink


def test_unbanded_body_is_one_span():
    fig, ax = plt.subplots()
    ideo = kp.Ideogram(HG38["chr1"], ax, bands=False)
    assert len(ideo.bands.get_paths()) == 1


def knob(ideo):
    return next(p for p in ideo.axes.patches if p.get_gid() == f"{ideo.chromosome.name}-knob")


def test_telocentric_knob_is_painted_as_centromere_up_to_the_pinch():
    ideo = kp.ideogram(kp.Genome.bundled("mm39")["chr1"])
    ideo.figure.draw_without_rendering()
    pixels_per_bp, half_pixels = ideo._scale()
    x = knob(ideo).get_path().vertices[:, 0]
    assert x.min() == 0
    assert x.max() == pytest.approx(outline.KNOB * half_pixels / pixels_per_bp)
    np.testing.assert_allclose(
        to_rgba(knob(ideo).get_facecolor()), to_rgba(ideograms.STAIN_COLORS["acen"])
    )
    assert knob(ideo).get_clip_path() is not None


def test_metacentric_chromosome_has_no_knob():
    ideo = kp.ideogram(HG38["chr1"])
    assert not any(p.get_gid() == "chr1-knob" for p in ideo.axes.patches)


def test_genome_strip_knobs_stay_inside_their_chromosomes():
    strip = kp.genome_strip(kp.Genome.bundled("mm39").concatenated(gap=5_000_000))
    strip[0].figure.draw_without_rendering()
    for ideo in strip:
        x = ideo.get_path().vertices[:, 0]
        assert x.min() == pytest.approx(ideo.offset)
        assert x.max() == pytest.approx(ideo.offset + ideo.chromosome.length)


def test_name_sits_left_of_a_telocentric_knob():
    ideo = kp.ideogram(kp.Genome.bundled("mm39")["chr1"])
    fig = ideo.figure
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    name = next(t for t in ideo.axes.texts if t.get_text() == "chr1")
    assert name.get_window_extent(renderer).x1 <= ideo.get_window_extent(renderer).x0


def test_band_labels_do_not_overlap():
    ideo = kp.ideogram(HG38["chr1"])
    fig = ideo.figure
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    names = {band.name for band in HG38["chr1"].bands}
    shown = [t for t in ideo.axes.texts if t.get_text() in names and t.get_visible()]
    boxes = sorted((t.get_window_extent(renderer) for t in shown), key=lambda b: b.x0)
    assert len(boxes) > 3
    assert all(a.x1 <= b.x0 for a, b in zip(boxes, boxes[1:], strict=False))


def test_band_labels_can_be_left_out():
    ideo = kp.ideogram(HG38["chr1"], labels=False)
    assert [t.get_text() for t in ideo.axes.texts] == ["chr1"]


def m1(region, head_on_top=True):
    fig, axes = plt.subplots(2, 1, height_ratios=[1, 5] if head_on_top else [5, 1])
    head, body = axes if head_on_top else axes[::-1]
    ideo = kp.ideogram(HG38[region.chrom], ax=head)
    body.plot([region.start, region.end], [0, 1])
    body.set_xlim(region.start, region.end)
    box, lines = ideo.zoom(region, into=body)
    fig.draw_without_rendering()
    return ideo, body, box, lines


def test_zoom_lines_run_from_the_box_to_the_target_corners():
    region = kp.Region("chr17", 20_000_000, 40_000_000)
    ideo, body, box, lines = m1(region)
    target = body.get_window_extent()
    corners = ideo.axes.transData.transform(ideo.xy([region.start, region.end], [-1.35, -1.35]))
    np.testing.assert_allclose(
        display(lines), [corners[0], (target.x0, target.y1), corners[1], (target.x1, target.y1)]
    )
    np.testing.assert_allclose(display(box)[:, 0].min(), corners[0, 0])


def test_zoom_into_axes_above_uses_its_bottom_edge():
    region = kp.Region("chr17", 20_000_000, 40_000_000)
    ideo, body, _, lines = m1(region, head_on_top=False)
    target = body.get_window_extent()
    end = display(lines)[[1, 3]]
    np.testing.assert_allclose(end, [(target.x0, target.y0), (target.x1, target.y0)])
    assert display(lines)[0, 1] < target.y0


def test_small_locus_box_keeps_a_minimum_width_in_points():
    ideo, _, box, _ = m1(BRCA1)
    width = np.ptp(display(box)[:, 0])
    assert width == pytest.approx(ideograms.MIN_BOX_POINTS * ideo.figure.dpi / 72)
    middle = ideo.axes.transData.transform((ideo.offset + (BRCA1.start + BRCA1.end) / 2, 0))
    assert np.mean(ends(display(box)[:, 0])) == pytest.approx(middle[0])


def test_zoom_rejects_a_region_on_another_chromosome():
    fig, (head, body) = plt.subplots(2, 1)
    ideo = kp.ideogram(HG38["chr1"], ax=head)
    with pytest.raises(ValueError, match="chr17"):
        ideo.zoom(BRCA1, into=body)


def test_genome_strip_draws_each_chromosome_at_its_offset():
    linear = HG38.concatenated(gap=5_000_000)
    ideograms = kp.genome_strip(linear)
    assert [i.chromosome.name for i in ideograms] == [c.name for c in HG38]
    assert ideograms[1].offset == linear.offset("chr2")
    assert ideograms[0].axes.get_xlim() == (0, linear.length)
    ideograms[0].figure.draw_without_rendering()


def test_chromosome_axis_labels_midpoints_and_shades_alternate_chromosomes():
    linear = HG38.concatenated(gap=5_000_000)
    fig, ax = plt.subplots()
    ax.scatter(linear.position(["chr1", "chrY"], [1, 2]), [3, 4])
    shade = kp.chromosome_axis(linear, ax)
    fig.draw_without_rendering()
    chr2 = HG38["chr2"]
    assert ax.get_xticks()[1] == pytest.approx(linear.offset("chr2") + chr2.length / 2)
    assert [t.get_text() for t in ax.get_xticklabels()][:3] == ["1", "2", "3"]
    assert len(shade.get_paths()) == len(HG38) // 2
    assert shade.get_paths()[0].vertices[:, 0].min() == linear.offset("chr2")
    assert ax.get_xlim() == (0, linear.length)


def test_chromosome_axis_shading_can_be_turned_off():
    fig, ax = plt.subplots()
    assert kp.chromosome_axis(HG38.concatenated(), ax, shade=False) is None
    assert not ax.collections


def test_ideogram_makes_its_own_figure_or_uses_the_given_axes():
    ideo = kp.ideogram(HG38["chrX"])
    assert ideo.figure.get_layout_engine() is not None
    fig, ax = plt.subplots()
    assert kp.ideogram(HG38["chrX"], ax=ax).axes is ax


def test_chromosome_axis_hides_tick_labels_that_would_overlap():
    linear = HG38.concatenated(gap=5_000_000)
    fig, ax = plt.subplots(figsize=(7, 2.5))
    kp.chromosome_axis(linear, ax)
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    shown = [t for t in ax.get_xticklabels() if t.get_text()]
    boxes = [t.get_window_extent(renderer) for t in shown]
    assert 12 < len(shown) < len(HG38)
    assert all(a.x1 <= b.x0 for a, b in zip(boxes, boxes[1:], strict=False))


def painted_extents(spans):
    """Along-axis extent of each painted span in data coordinates."""
    return [ends(path.vertices[:, 0]) for path in spans.get_paths()]


def test_paint_fills_spans_in_bp_clipped_to_the_outline():
    fig, ax = plt.subplots()
    ideo = kp.Ideogram(HG38["chr1"], ax, offset=1000)
    spans = ideo.paint([0, 50e6], [40e6, 120e6], ["C1", "C2"])
    fig.draw_without_rendering()
    assert sorted(painted_extents(spans)) == [(1000, 1000 + 40e6), (1000 + 50e6, 1000 + 120e6)]
    assert spans.get_clip_path() is not None
    assert spans.get_clip_box() is None
    assert ideo.bands.get_zorder() < spans.get_zorder() < ideo.get_zorder()


def test_paint_sits_below_a_telocentric_knob():
    fig, ax = plt.subplots()
    ideo = kp.Ideogram(kp.Genome.bundled("mm39")["chr1"], ax)
    spans = ideo.paint([0], [1e8], "C1")
    assert spans.get_zorder() < knob(ideo).get_zorder()


def test_small_painted_spans_keep_a_minimum_width_and_draw_on_top():
    ideo = kp.ideogram(HG38["chr1"])
    spans = ideo.paint([0, 100e6], [200e6, 100e6 + 1000], ["C1", "C2"])
    fig = ideo.figure
    for size in [(6.5, 0.75), (13, 0.75)]:
        fig.set_size_inches(size)
        fig.draw_without_rendering()
        small = spans.get_paths()[-1]
        pixels = ax_pixels(ideo, ends(small.vertices[:, 0]))
        assert np.diff(pixels)[0] == pytest.approx(ideograms.MIN_PAINT_POINTS * fig.dpi / 72)
        assert np.mean(ends(small.vertices[:, 0])) == pytest.approx(100e6 + 500)
    np.testing.assert_allclose(spans.get_facecolors()[-1], to_rgba("C2"))


def ax_pixels(ideo, xs):
    return ideo.axes.transData.transform([(x, 0) for x in xs])[:, 0]


def test_paint_takes_one_color_or_one_per_span():
    fig, ax = plt.subplots()
    ideo = kp.Ideogram(HG38["chr1"], ax)
    assert len(ideo.paint([0, 5e7], [1e7, 6e7], "C1").get_facecolors()) == 1
    with pytest.raises(ValueError, match="colors"):
        ideo.paint([0, 5e7], [1e7, 6e7], ["C1", "C2", "C3"])
