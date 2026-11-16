import numpy as np
import pytest

from karyogram_plot import outline
from karyogram_plot.genome import Band, Chromosome, Stain

CAP = 10.0


def upper_edge(along, across):
    """Vertices of the top side, from the left cap to the right cap."""
    top = across > -1e-9
    return along[top], across[top]


def body(along, across, cap=CAP):
    """Vertices between the caps."""
    inside = (along >= along.min() + cap) & (along <= along.max() - cap)
    return along[inside], across[inside]


def test_metacentric_outline_spans_the_chromosome():
    along, across = outline.vertices(Chromosome("1", 1000, centromere=(480, 520)), CAP)
    assert along.min() == pytest.approx(0)
    assert along.max() == pytest.approx(1000)
    assert np.abs(across).max() == pytest.approx(1)


def test_outline_is_symmetric_across_the_axis():
    along, across = outline.vertices(Chromosome("1", 1000, centromere=(480, 520)), CAP)
    top = sorted(zip(*upper_edge(along, across), strict=True))
    low = across < 1e-9
    bottom = sorted(zip(along[low], -across[low], strict=True))
    np.testing.assert_allclose(top, bottom, atol=1e-9)


def test_centromere_pinches_to_the_waist_at_its_midpoint():
    along, across = outline.vertices(Chromosome("1", 1000, centromere=(480, 520)), CAP)
    s, t = upper_edge(along, across)
    assert t[np.argmin(np.abs(s - 500))] == pytest.approx(outline.PINCH)
    assert body(along, across)[1].max() == pytest.approx(1)
    assert np.abs(body(along, across)[1]).min() == pytest.approx(outline.PINCH)


def test_waist_spans_the_centromere():
    along, across = outline.vertices(Chromosome("1", 1000, centromere=(400, 600)), CAP)
    s, t = upper_edge(along, across)
    assert np.interp(400, s, t) == pytest.approx(1)
    assert np.interp(450, s, t) < 1


def test_short_centromere_still_gets_a_waist_near_a_cap_long():
    along, across = outline.vertices(Chromosome("1", 1000, centromere=(499, 501)), CAP)
    s, t = upper_edge(along, across)
    assert np.interp(500 - 0.5 * CAP, s, t) < 1


def test_unknown_centromere_draws_no_waist():
    along, across = outline.vertices(Chromosome("1", 1000), CAP)
    _, t = body(along, across)
    np.testing.assert_allclose(np.abs(t), 1)


def test_telocentric_knob_sits_inside_the_chromosome():
    along, across = outline.vertices(Chromosome("1", 1000, centromere=(0, 0)), CAP)
    assert along.min() == pytest.approx(0)
    assert along.max() == pytest.approx(1000)
    s, t = upper_edge(along, across)
    assert np.interp(outline.KNOB * CAP, s, t) == pytest.approx(outline.PINCH)


def stalked():
    bands = (
        Band(0, 200, "p13", Stain("gvar")),
        Band(200, 300, "p12", Stain("stalk")),
        Band(300, 400, "p11", Stain("acen")),
        Band(400, 1000, "q11", Stain("gneg")),
    )
    return Chromosome("13", 1000, bands, centromere=(300, 400))


def test_stalks_narrow_the_body():
    along, across = outline.vertices(stalked(), CAP)
    s, t = upper_edge(along, across)
    assert np.interp(250, s, t) == pytest.approx(outline.STALK)
    assert np.interp(150, s, t) == pytest.approx(1)


def test_stalks_can_be_left_out():
    along, across = outline.vertices(stalked(), CAP, stalks=False)
    s, t = upper_edge(along, across)
    assert np.interp(250, s, t) == pytest.approx(1)


def test_stalk_that_ends_where_the_waist_starts_steps_once():
    along, across = outline.vertices(stalked(), CAP)
    s, t = upper_edge(along, across)
    assert np.all(np.diff(s) >= 0)
    np.testing.assert_allclose(t[np.isclose(s, 300)], [outline.STALK, 1])


def test_cap_shrinks_to_fit_a_short_chromosome():
    along, _ = outline.vertices(Chromosome("1", 10, centromere=(4, 6)), CAP)
    assert along.min() == pytest.approx(0)
    assert along.max() == pytest.approx(10)
