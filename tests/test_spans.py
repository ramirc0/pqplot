import matplotlib.pyplot as plt
import numpy as np
import pytest

from karyogram_plot.spans import Spans


def along_pixels(spans, orientation):
    axis = 0 if orientation == "horizontal" else 1
    transform = spans.get_transform()
    return [np.ptp(transform.transform(p.vertices)[:, axis]) for p in spans.get_paths()]


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_spans_keep_their_bp_extent_when_wide(orientation):
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 1000), ylim=(0, 1000))
    spans = Spans([100, 500], [300, 900], (0.2, 0.8), orientation=orientation, min_points=2)
    ax.add_collection(spans)
    fig.draw_without_rendering()
    axis = 0 if orientation == "horizontal" else 1
    extents = [(p.vertices[:, axis].min(), p.vertices[:, axis].max()) for p in spans.get_paths()]
    assert extents == [(100, 300), (500, 900)]
    across = spans.get_paths()[0].vertices[:, 1 - axis]
    assert (across.min(), across.max()) == (0.2, 0.8)


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_narrow_spans_widen_to_min_points_about_their_middle(orientation):
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 1e8), ylim=(1e8, 0))
    spans = Spans([5e7], [5e7 + 10], (0, 1), orientation=orientation, min_points=2)
    ax.add_collection(spans)
    fig.draw_without_rendering()
    assert along_pixels(spans, orientation)[0] == pytest.approx(2 * fig.dpi / 72)
    axis = 0 if orientation == "horizontal" else 1
    along = spans.get_paths()[0].vertices[:, axis]
    assert (along.min() + along.max()) / 2 == pytest.approx(5e7 + 5)


def test_spans_in_a_blended_transform_span_the_axes_across():
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 1e8), ylim=(-5, 5))
    spans = Spans(
        [0],
        [10],
        (0, 1),
        orientation="horizontal",
        min_points=3,
        transform=ax.get_xaxis_transform(),
    )
    ax.add_collection(spans, autolim=False)
    fig.draw_without_rendering()
    assert along_pixels(spans, "horizontal")[0] == pytest.approx(3 * fig.dpi / 72)
    assert np.ptp(spans.get_transform().transform(spans.get_paths()[0].vertices)[:, 1]) == (
        pytest.approx(ax.bbox.height)
    )
