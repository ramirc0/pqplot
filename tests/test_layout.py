import pytest
from hypothesis import given
from hypothesis import strategies as st

from pqplot import layout
from pqplot.genome import Chromosome

ROW = [
    Chromosome("1", 1000, centromere=(400, 500)),
    Chromosome("2", 800, centromere=(100, 200)),
    Chromosome("3", 600, centromere=(0, 0)),
]


def test_wrap_splits_items_into_even_groups_in_order():
    assert layout.wrap(list(range(5)), 2) == [[0, 1, 2], [3, 4]]
    assert layout.wrap(list(range(24)), 2) == [list(range(12)), list(range(12, 24))]
    assert layout.wrap([1, 2], 5) == [[1], [2]]


def test_wrap_rejects_fewer_than_one_group():
    with pytest.raises(ValueError, match="groups"):
        layout.wrap([1], 0)


def test_start_alignment_starts_every_chromosome_at_zero():
    assert layout.bounds(ROW, "start") == [(0, 1000), (0, 1000), (0, 1000)]


def test_end_alignment_ends_every_chromosome_together():
    assert layout.bounds(ROW, "end") == [(0, 1000), (-200, 800), (-400, 600)]


def test_centromere_alignment_lines_up_centromere_midpoints():
    bounds = layout.bounds(ROW, "centromere")
    mids = [450, 150, 0]
    assert {mid - lo for mid, (lo, _) in zip(mids, bounds, strict=True)} == {450}
    assert {hi - lo for lo, hi in bounds} == {450 + 650}


def test_unknown_centromere_aligns_like_position_zero():
    row = [Chromosome("1", 1000, centromere=(400, 500)), Chromosome("2", 100)]
    assert layout.bounds(row, "centromere")[1] == (-450, 550)


def test_bounds_reject_an_unknown_alignment():
    with pytest.raises(ValueError, match="align"):
        layout.bounds(ROW, "middle")


def test_slots_put_tracks_before_then_body_then_after():
    slots = layout.slots([("genes", 1.0)], copies=2, after=[("cnv", 1.5)])
    assert [s.track for s in slots] == ["genes", None, "cnv"]
    body = slots[1]
    assert body.start == pytest.approx(1 + layout.TRACK_GAP)
    assert body.size == pytest.approx(2 + layout.COPY_GAP)
    assert slots[2].start == pytest.approx(body.start + body.size + layout.TRACK_GAP)


def test_body_alone_is_one_slot_at_zero():
    assert layout.slots([], copies=1, after=[]) == [layout.Slot(None, 0.0, 1.0)]


def grid(spans=(1000.0, 500.0), columns=3, after=()):
    return layout.Grid(spans, columns, layout.slots([], copies=1, after=list(after)))


def test_groups_share_one_bp_scale_and_fill_the_along_extent():
    g = grid()
    body = g.slots[0]
    a0, a1, _, _ = g.cell(0, 0, body, along=400, across=100)
    b0, b1, _, _ = g.cell(1, 0, body, along=400, across=100)
    assert a0 == 0
    assert b1 == pytest.approx(400)
    assert b0 - a1 == pytest.approx(layout.GROUP_GAP)
    assert (a1 - a0) / 1000 == pytest.approx((b1 - b0) / 500)


def test_columns_fill_the_across_extent():
    g = grid()
    body = g.slots[0]
    first = g.cell(0, 0, body, along=400, across=100)
    last = g.cell(0, 2, body, along=400, across=100)
    assert first[2] == 0
    assert last[3] == pytest.approx(100)
    unit = first[3] - first[2]
    assert g.cell(0, 1, body, along=400, across=100)[2] - first[3] == pytest.approx(
        layout.COLUMN_GAP * unit
    )


def test_extent_measures_the_grid_at_a_given_scale():
    g = grid()
    along, across = g.extent(points_per_bp=0.1, points_per_body=10)
    assert along == pytest.approx(150 + layout.GROUP_GAP)
    assert across == pytest.approx(10 * (3 + 2 * layout.COLUMN_GAP))


@given(
    spans=st.lists(st.floats(1, 1e9), min_size=1, max_size=4),
    columns=st.integers(1, 12),
    tracks=st.lists(st.floats(0.1, 5), max_size=3),
    along=st.floats(200, 2000),
    across=st.floats(50, 2000),
)
def test_cells_stay_inside_the_host_and_do_not_overlap(spans, columns, tracks, along, across):
    g = grid(tuple(spans), columns, after=[(str(i), s) for i, s in enumerate(tracks)])
    cells = [
        g.cell(i, j, slot, along, across)
        for i in range(len(spans))
        for j in range(columns)
        for slot in g.slots
    ]
    for a0, a1, c0, c1 in cells:
        assert -1e-6 <= a0 < a1 <= along + 1e-6
        assert -1e-6 <= c0 < c1 <= across + 1e-6
    for n, (a0, a1, c0, c1) in enumerate(cells):
        for b0, b1, d0, d1 in cells[n + 1 :]:
            assert a1 <= b0 + 1e-6 or b1 <= a0 + 1e-6 or c1 <= d0 + 1e-6 or d1 <= c0 + 1e-6
