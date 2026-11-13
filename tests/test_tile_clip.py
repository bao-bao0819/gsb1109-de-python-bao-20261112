"""Tests for tile_clip: clipping Z/M polylines to tiles and merging them back."""

import math

import pytest

from tile_clip import (
    ClippedPiece,
    MergedLine,
    Tile,
    clip_to_tiles,
    merge_pieces,
    parts_from_shape,
)

# 2x2 tiling of [0, 10] x [0, 10].
TILES = [
    Tile("SW", 0, 0, 5, 5),
    Tile("SE", 5, 0, 10, 5),
    Tile("NW", 0, 5, 5, 10),
    Tile("NE", 5, 5, 10, 10),
]


def test_cut_points_are_interpolated():
    # A straight road with linear Z and M (mileage 0 -> 80 over 8 units).
    features = [("road-1", [[(1, 1, 0, 0), (9, 1, 8, 80)]])]
    pieces = clip_to_tiles(features, TILES)

    assert len(pieces) == 2
    by_tile = {p.tile_id: p for p in pieces}
    sw, se = by_tile["SW"], by_tile["SE"]

    # Cut point at x=5 gets interpolated Z and M, not endpoint values.
    assert sw.points == [(1, 1, 0, 0), (5, 1, 4, 40)]
    assert se.points == [(5, 1, 4, 40), (9, 1, 8, 80)]

    # Each piece knows its source feature, part and start/end mileage.
    assert sw.feature_id == se.feature_id == "road-1"
    assert sw.part_index == se.part_index == 0
    assert (sw.start_m, sw.end_m) == (0, 40)
    assert (se.start_m, se.end_m) == (40, 80)

    # Arc-length positions along the source part.
    assert sw.start_s == 0
    assert sw.end_s == pytest.approx(4.0)
    assert se.start_s == pytest.approx(4.0)
    assert se.end_s == pytest.approx(8.0)


def test_road_on_shared_edge_belongs_to_one_side():
    # The road runs exactly along the shared edge x=5.
    features = [("edge-road", [[(5, 1, 0, 0), (5, 9, 0, 80)]])]
    pieces = clip_to_tiles(features, TILES)

    # Crosses y=5, so two pieces, both owned by the east side only.
    assert [p.tile_id for p in pieces] == ["SE", "NE"]

    merged = merge_pieces(pieces)
    assert len(merged) == 1
    line = merged[0]
    # The cut point (5, 5) appears exactly once after merging.
    assert line.points == [(5, 1, 0, 0), (5, 5, 0, 40), (5, 9, 0, 80)]
    assert (line.start_m, line.end_m) == (0, 80)


def test_missing_m_is_not_treated_as_zero():
    # M missing (None) at the first vertex: interpolation must stay missing.
    features = [("no-m", [[(1, 1, 0, None), (9, 1, 0, 80)]])]
    pieces = clip_to_tiles(features, TILES)
    by_tile = {p.tile_id: p for p in pieces}

    cut_m = by_tile["SW"].points[-1][3]
    assert cut_m is None
    assert cut_m != 0.0
    assert by_tile["SE"].points[0][3] is None

    merged = merge_pieces(pieces)
    assert len(merged) == 1
    assert merged[0].start_m is None
    assert merged[0].end_m == 80

    # Entirely unmeasured line: all M values stay None, never 0.
    features = [("all-missing", [[(1, 1, 0, None), (9, 1, 0, None)]])]
    pieces = clip_to_tiles(features, TILES)
    assert all(m is None for p in pieces for *_, m in p.points)


def test_nan_m_is_also_missing():
    features = [("nan-m", [[(1, 1, 0, float("nan")), (9, 1, 0, 80)]])]
    pieces = clip_to_tiles(features, TILES)
    assert all(m is None for p in pieces for *_, m in p.points[:1])
    merged = merge_pieces(pieces)
    assert merged[0].start_m is None


def test_merge_restores_source_order_and_dedupes_cut_points():
    # A zig-zag crossing several tiles; pieces are fed in scrambled order.
    features = [("zig", [[(1, 1, 0, 0), (9, 1, 8, 80), (9, 9, 16, 160)]])]
    pieces = clip_to_tiles(features, TILES)
    assert {p.tile_id for p in pieces} == {"SW", "SE", "NE"}

    merged = merge_pieces(list(reversed(pieces)))
    assert len(merged) == 1
    line = merged[0]
    xs = [p[0] for p in line.points]
    # Source order restored: x grows monotonically along the merged line.
    assert xs == sorted(xs)
    # Cut points at x=5 and y=5 each appear exactly once.
    assert sum(1 for p in line.points if p[0] == 5) == 1
    assert sum(1 for p in line.points if p[1] == 5) == 1
    # Mileage runs continuously from 0 to 160.
    assert line.start_m == 0
    assert line.end_m == 160
    ms = [p[3] for p in line.points]
    assert ms == sorted(ms)


def test_disconnected_parts_stay_disconnected():
    features = [
        (
            "two-parts",
            [
                [(1, 1, 0, 0), (9, 1, 0, 80)],
                [(1, 9, 0, 100), (9, 9, 0, 180)],
            ],
        )
    ]
    pieces = clip_to_tiles(features, TILES)
    merged = merge_pieces(pieces)

    assert len(merged) == 2
    assert {line.part_index for line in merged} == {0, 1}
    for line in merged:
        assert line.feature_id == "two-parts"
    # Each part keeps its own mileage range.
    by_part = {line.part_index: line for line in merged}
    assert (by_part[0].start_m, by_part[0].end_m) == (0, 80)
    assert (by_part[1].start_m, by_part[1].end_m) == (100, 180)


def test_gap_between_tiles_stays_broken():
    # Tiles do not cover x in (5, 7): the line leaves the tiled area.
    tiles = [Tile("L", 0, 0, 5, 5), Tile("R", 7, 0, 10, 5)]
    features = [("gapped", [[(1, 1, 0, 0), (9, 1, 0, 80)]])]
    pieces = clip_to_tiles(features, tiles)
    assert len(pieces) == 2

    merged = merge_pieces(pieces)
    # Not joined across the gap, even though feature and part match.
    assert len(merged) == 2
    merged.sort(key=lambda line: line.start_s)
    assert merged[0].end_s == pytest.approx(4.0)
    assert merged[1].start_s == pytest.approx(6.0)
    # Mileage of the exposed ends is interpolated from the source segment.
    assert merged[0].end_m == pytest.approx(40.0)
    assert merged[1].start_m == pytest.approx(60.0)


def test_outer_max_boundary_is_kept():
    # Points on the outermost max edges of the tiling are not dropped.
    features = [("rim", [[(9, 9, 0, 0), (10, 10, 1, 10)]])]
    pieces = clip_to_tiles(features, TILES)
    assert len(pieces) == 1
    assert pieces[0].tile_id == "NE"
    assert pieces[0].points[-1] == (10, 10, 1, 10)


def test_parts_from_shape_roundtrip():
    shapefile = pytest.importorskip("shapefile")
    shape = shapefile.Shape(
        shapeType=shapefile.POLYLINEZ,
        points=[[1, 1, 0, 0], [9, 1, 8, 80], [9, 9, 16, 160]],
        parts=[0],
    )
    parts = parts_from_shape(shape)
    assert parts == [[(1, 1, 0, 0), (9, 1, 8, 80), (9, 9, 16, 160)]]

    pieces = clip_to_tiles([("from-shape", parts)], TILES)
    merged = merge_pieces(pieces)
    assert len(merged) == 1
    assert (merged[0].start_m, merged[0].end_m) == (0, 160)


def test_merge_pieces_accepts_hand_built_pieces():
    # Merge does not depend on clip output ordering, only on source positions.
    pieces = [
        ClippedPiece("f", 0, "B", [(5, 0, 1, 50), (8, 0, 4, 80)], 5.0, 8.0),
        ClippedPiece("f", 0, "A", [(0, 0, 0, 0), (5, 0, 1, 50)], 0.0, 5.0),
    ]
    merged = merge_pieces(pieces)
    assert len(merged) == 1
    line = merged[0]
    assert isinstance(line, MergedLine)
    assert line.points == [(0, 0, 0, 0), (5, 0, 1, 50), (8, 0, 4, 80)]
    assert line.tile_ids == ("A", "B")
    assert (line.start_m, line.end_m) == (0, 80)
