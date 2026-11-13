"""Tests for src/connectivity.py (segment turn tables)."""

import shapefile
from connectivity import (
    CONFIRMED,
    PENDING,
    Segment,
    build_connectivity,
    load_segments,
)


def turn_pairs(table, status=None):
    return {
        (t.from_segment, t.to_segment)
        for t in table.turns
        if status is None or t.status == status
    }


def test_same_coordinates_different_levels_do_not_connect():
    # Elevated segment 1 directly above ground segment 2: identical
    # endpoints, but different levels must never share an intersection.
    segs = [
        Segment(1, (0, 0), (10, 0), level=1),
        Segment(2, (10, 0), (20, 0), level=1),
        Segment(3, (10, 0), (10, 10), level=0),
    ]
    table = build_connectivity(segs)
    # Two-way segments: the junction works in both directions.
    assert turn_pairs(table, CONFIRMED) == {(1, 2), (2, 1)}
    # The only junction is the level-1 one between segments 1 and 2.
    assert [(inter.level, inter.members) for inter in table.intersections] == [
        (1, ((1, "end"), (2, "start")))
    ]


def test_tolerance_merges_endpoints_with_common_position():
    segs = [
        Segment(1, (0, 0), (10, 0), end_tol=0.6),
        Segment(2, (10.5, 0.4), (20, 0), start_tol=0.6),
    ]
    table = build_connectivity(segs)
    assert turn_pairs(table, CONFIRMED) == {(1, 2), (2, 1)}
    (inter,) = table.intersections
    # The common position must lie within both tolerance squares.
    assert 9.4 <= inter.point[0] <= 10.6
    assert -0.2 <= inter.point[1] <= 0.6


def test_no_common_position_means_no_junction():
    segs = [
        Segment(1, (0, 0), (10, 0), end_tol=0.4),
        Segment(2, (11, 0), (20, 0), start_tol=0.4),
    ]
    table = build_connectivity(segs)
    assert table.turns == ()
    assert table.intersections == ()


def test_near_neighbour_chain_is_not_merged_into_big_junction():
    # A--B are compatible and B--C are compatible, but A and C cannot
    # share a position.  Chaining near neighbours would merge all three
    # into one big junction; the partition must not.
    segs = [
        Segment("A", (-10, 0), (0.0, 0), end_tol=1.0),
        Segment("B", (1.5, 0), (5, 0), start_tol=1.0),
        Segment("C", (0, 9), (3.0, 0), end_tol=1.0),
    ]
    table = build_connectivity(segs)
    assert table.num_partitions == 2
    for inter in table.intersections:
        assert len(inter.members) == 2  # never all three together
    # Both minimal partitions put B with either A or C, so every turn
    # is still to be confirmed.
    assert turn_pairs(table, CONFIRMED) == set()
    assert turn_pairs(table, PENDING) == {
        ("A", "B"),
        ("B", "A"),
        ("B", "C"),
        ("C", "B"),
    }


def test_oneway_direction_is_respected():
    # One-way 1 ends at the junction, one-way 2 starts there:
    # only 1 -> 2 may be taken.
    segs = [
        Segment(1, (0, 0), (10, 0), oneway=True),
        Segment(2, (10, 0), (20, 0), oneway=True),
    ]
    table = build_connectivity(segs)
    assert turn_pairs(table, CONFIRMED) == {(1, 2)}


def test_reversed_oneway_via_loader(tmp_path):
    path = str(tmp_path / "roads")
    with shapefile.Writer(path, shapeType=shapefile.POLYLINE) as w:
        w.field("LEVEL", "N")
        w.field("ONEWAY", "C", size=10)
        w.field("TOL", "F", decimal=3)
        # Record 0: one-way to->from, so it is entered at (10, 0).
        w.line([[[0, 0], [10, 0]]])
        w.record(0, "TF", 0.0)
        # Record 1: two-way continuation.
        w.line([[[10, 0], [20, 0]]])
        w.record(0, "", 0.0)
    segs = load_segments(
        path + ".shp",
        level_field="LEVEL",
        oneway_field="ONEWAY",
        tolerance_field="TOL",
    )
    assert segs[0].oneway and segs[0].start == (10, 0)
    assert not segs[1].oneway
    table = build_connectivity(segs)
    # Segment 0 is one-way ending at (0, 0): at the (10, 0) junction it
    # can only be entered, never exited.
    assert turn_pairs(table, CONFIRMED) == {(1, 0)}


def test_ambiguous_partition_reports_pending_turns():
    # Junction with four endpoints where endpoint "e" of segment 1 sits
    # between two mutually incompatible endpoints: two minimal
    # partitions exist, and only turns common to both are confirmed.
    segs = [
        Segment(1, (0, 0), (10, 0), end_tol=1.0),
        Segment(2, (11, 0), (20, 0), start_tol=1.0),
        Segment(3, (5, 5), (12, 0), end_tol=1.0),
        Segment(4, (5, -5), (13, 0), end_tol=1.0),
        # A separate, unambiguous junction.
        Segment(5, (100, 0), (110, 0)),
        Segment(6, (110, 0), (120, 0)),
    ]
    # Junction endpoints: 1e=(10,0), 2s=(11,0), 3e=(12,0), 4e=(13,0),
    # tolerance sum 2.0.  Every pair is compatible except 1e vs 4e
    # (distance 3), so the minimal partitions use two intersections;
    # the unattached endpoints 2s/3e may sit on either side, giving
    # four equally minimal partitions.  No junction turn there is
    # allowed by all four, so all of them are still to be confirmed.
    table = build_connectivity(segs)
    assert table.num_partitions == 4
    assert turn_pairs(table, CONFIRMED) == {(5, 6), (6, 5)}
    pending = turn_pairs(table, PENDING)
    assert (1, 2) in pending and (4, 2) in pending and (2, 3) in pending
    # 1 and 4 never share an intersection in any minimal partition.
    assert (1, 4) not in turn_pairs(table)


def test_multi_part_polyline_becomes_separate_segments(tmp_path):
    path = str(tmp_path / "multi")
    with shapefile.Writer(path, shapeType=shapefile.POLYLINE) as w:
        w.field("LEVEL", "N")
        w.line([[[0, 0], [5, 0]], [[5, 0], [9, 0]]])
        w.record(0)
    segs = load_segments(path + ".shp", level_field="LEVEL")
    assert [s.seg_id for s in segs] == [(0, 0), (0, 1)]
    table = build_connectivity(segs)
    assert turn_pairs(table, CONFIRMED) == {((0, 0), (0, 1)), ((0, 1), (0, 0))}
