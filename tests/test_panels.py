import matplotlib.pyplot as plt
import numpy as np
import pytest

from karyogram_plot.panels import add_panel


def top_half(along, across):
    return 0.0, along / 2, 0.0, across


@pytest.mark.parametrize(
    ("orientation", "expected"),
    [
        ("vertical", lambda b: (b.x0, b.y0 + b.height / 2, b.x1, b.y1)),
        ("horizontal", lambda b: (b.x0, b.y0, b.x0 + b.width / 2, b.y1)),
    ],
)
def test_panel_takes_its_cell_along_the_orientation(orientation, expected):
    fig, host = plt.subplots(layout="constrained")
    panel = add_panel(host, top_half, orientation)
    fig.draw_without_rendering()
    np.testing.assert_allclose(panel.get_window_extent().extents, expected(host.bbox))


def test_panel_follows_a_resize():
    fig, host = plt.subplots(layout="constrained")
    panel = add_panel(host, top_half, "vertical")
    fig.set_size_inches(9, 3)
    fig.draw_without_rendering()
    assert panel.get_window_extent().width == pytest.approx(host.bbox.width)


def test_cell_sizes_are_in_points():
    sizes = []

    def cell(along, across):
        sizes.append((along, across))
        return 0.0, along, 0.0, across

    fig, host = plt.subplots(figsize=(4, 3), dpi=200)
    add_panel(host, cell, "horizontal")
    fig.draw_without_rendering()
    assert sizes[-1] == pytest.approx((host.bbox.width * 72 / 200, host.bbox.height * 72 / 200))


def test_panel_is_a_child_of_the_host():
    fig, host = plt.subplots()
    panel = add_panel(host, top_half, "vertical")
    assert panel in host.child_axes
    assert panel not in fig.axes
