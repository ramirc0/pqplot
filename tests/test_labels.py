import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pqplot.labels import capacity, spread


def test_spaced_targets_stay_put():
    np.testing.assert_allclose(spread(np.array([0.0, 10.0, 30.0]), 5, 0, 100), [0, 10, 30])


def test_crowded_targets_spread_evenly_about_their_mean():
    np.testing.assert_allclose(spread(np.array([50.0, 50.0, 50.0]), 4, 0, 100), [46, 50, 54])


def test_spread_stays_inside_the_bounds():
    np.testing.assert_allclose(spread(np.array([0.0, 1.0]), 4, 0, 100), [0, 4])
    np.testing.assert_allclose(spread(np.array([99.0, 100.0]), 4, 0, 100), [96, 100])


def test_spread_of_nothing():
    assert spread(np.array([]), 4, 0, 100).size == 0


def test_capacity_counts_labels_that_fit():
    assert capacity(4, 0, 100) == 26
    assert capacity(4, 0, 0) == 1


@given(
    st.lists(st.floats(0, 1000), max_size=40),
    st.floats(0.5, 20),
)
def test_spread_keeps_order_and_gaps_within_bounds(targets, gap):
    targets = np.sort(np.array(targets))
    if len(targets) > capacity(gap, 0, 1000):
        return
    placed = spread(targets, gap, 0, 1000)
    assert placed.shape == targets.shape
    assert np.all(np.diff(placed) >= gap - 1e-6)
    assert np.all(placed >= -1e-6)
    assert np.all(placed <= 1000 + 1e-6)


def test_spread_needs_sorted_targets_that_fit():
    with pytest.raises(ValueError, match="sorted"):
        spread(np.array([5.0, 1.0]), 1, 0, 10)
    with pytest.raises(ValueError, match="fit"):
        spread(np.zeros(5), 5, 0, 10)
