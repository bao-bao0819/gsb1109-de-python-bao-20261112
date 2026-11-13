"""
Tests for the minimum-vertex line simplification helpers
(simplify_lines, Shape.simplify and friends).
"""

from __future__ import annotations

import io
import math
import random

import pytest

import shapefile
from shapefile import (
    DegeneratePartError,
    SimplifiedShape,
    ToleranceError,
    VertexMapping,
    flow_direction_traceable,
    simplify_lines,
)

# --------------------------------------------------------------------------
# Basic retention and tolerance behaviour
# --------------------------------------------------------------------------


def test_endpoints_are_always_retained():
    points = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)]
    result = simplify_lines([points], 100.0)
    assert result.kept == [[0, 4]]
    assert result.lines == [[(0, 0), (4, 0)]]


def test_zero_tolerance_retains_every_vertex():
    points = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)]
    result = simplify_lines([points], 0.0)
    assert result.kept == [list(range(len(points)))]
    assert result.lines == [points]


def test_tight_tolerance_forces_intermediate_vertices():
    points = [(0, 0), (1, 5), (2, 0)]
    # The middle point is 5 units off the chord; a tight tolerance keeps it.
    assert simplify_lines([points], 1.0).kept == [[0, 1, 2]]
    assert simplify_lines([points], 10.0).kept == [[0, 2]]


def test_per_point_tolerances_govern_each_vertex():
    # Only the middle vertex has a restrictive tolerance.
    points = [(0, 0), (1, 5), (2, 0)]
    kept = simplify_lines([points], [[10.0, 1.0, 10.0]]).kept
    assert kept == [[0, 1, 2]]
    # Loosening that vertex's own tolerance lets the chord replace it.
    assert simplify_lines([points], [[10.0, 10.0, 10.0]]).kept == [[0, 2]]


def test_flat_tolerance_sequence_spans_all_parts():
    lines = [[(0, 0), (1, 1), (2, 0)], [(5, 5), (6, 6), (7, 5)]]
    flat = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    result = simplify_lines(lines, flat)
    assert result.kept == [[0, 2], [0, 2]]


def test_wrong_length_tolerance_is_rejected():
    with pytest.raises(ToleranceError):
        simplify_lines([[(0, 0), (1, 1)]], [1.0, 2.0, 3.0])
    with pytest.raises(ToleranceError):
        simplify_lines([[(0, 0), (1, 1)]], [[1.0]])


@pytest.mark.parametrize("bad", [-0.1, float("nan")])
def test_invalid_tolerance_values_are_rejected(bad):
    with pytest.raises(ToleranceError):
        simplify_lines([[(0, 0), (1, 1)]], bad)
    with pytest.raises(ToleranceError):
        simplify_lines([[(0, 0), (1, 1)]], [0.0, bad])


@pytest.mark.parametrize("bad", [None, True, "1.0"])
def test_non_numeric_tolerance_input_is_rejected(bad):
    with pytest.raises(ToleranceError):
        simplify_lines([[(0, 0), (1, 1)]], bad)


def test_non_sequence_forced_input_is_rejected():
    with pytest.raises(shapefile.ShapefileException):
        simplify_lines([[(0, 0), (1, 1)]], 1.0, forced=True)


def test_three_d_and_measured_points_keep_their_full_tuples():
    points = [(0, 0, 10.0), (1, 1, 11.0), (2, 0, 12.0)]
    result = simplify_lines([points], 100.0)
    assert result.lines == [[(0, 0, 10.0), (2, 0, 12.0)]]

    measured = [(0, 0, 0.5), (1, 5, None), (2, 0, 0.9)]
    result_m = simplify_lines([measured], 1.0)
    assert result_m.kept == [[0, 1, 2]]
    assert result_m.lines == [measured]


# --------------------------------------------------------------------------
# Forced vertices, confluence points and part boundaries
# --------------------------------------------------------------------------


def test_forced_markers_are_retained():
    points = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)]
    result = simplify_lines(
        [points], 100.0, forced=[[False, False, True, False, False]]
    )
    assert result.kept == [[0, 2, 4]]


def test_flat_forced_markers_span_all_parts():
    lines = [[(0, 0), (1, 1), (2, 0)], [(5, 5), (6, 6), (7, 5)]]
    # Keep the middle vertex of each part via a flat marker sequence.
    result = simplify_lines(
        lines, 100.0, forced=[False, True, False, False, True, False]
    )
    assert result.kept == [[0, 1, 2], [0, 1, 2]]


def test_confluence_points_are_retained_across_parts():
    lines = [
        [(0, 0), (5, 5), (9, 9)],
        [(1, 1), (5, 5), (2, 2)],
    ]
    result = simplify_lines(lines, 100.0, confluence=[(0, 1), (1, 1)])
    assert result.kept == [[0, 1, 2], [0, 1, 2]]


def test_each_part_boundary_is_retained():
    lines = [
        [(0, 0), (1, 1), (2, 2)],
        [(9, 9), (8, 8), (7, 7)],
        [(4, 4), (5, 5), (6, 6)],
    ]
    result = simplify_lines(lines, 100.0)
    assert len(result.lines) == 3
    for kept, part in zip(result.kept, lines):
        assert kept[0] == 0
        assert kept[-1] == len(part) - 1


def test_confluence_index_validation():
    with pytest.raises(shapefile.ShapefileException):
        simplify_lines([[(0, 0), (1, 1)]], 1.0, confluence=[(5, 0)])
    with pytest.raises(shapefile.ShapefileException):
        simplify_lines([[(0, 0), (1, 1)]], 1.0, confluence=[(0, 9)])


# --------------------------------------------------------------------------
# Mapping output: coverage, ordering, distances and projections
# --------------------------------------------------------------------------


def _assert_mapping_consistency(points, tolerances, result):
    (kept,) = result.kept
    (mappings,) = result.mappings
    assert len(mappings) == len(points)
    n = len(points)

    # The first edge also covers any leading duplicate of the retained start.
    edge_low = {0: 0}
    edge_high = {}
    for edge in range(len(kept) - 1):
        edge_high[edge] = kept[edge + 1]
        if edge > 0:
            edge_low[edge] = kept[edge]

    edge_starts = {edge: kept[edge] for edge in range(len(kept) - 1)}
    previous_t_by_edge: dict[int, float] = {}
    for index, mapping in enumerate(mappings):
        assert isinstance(mapping, VertexMapping)
        start_index = edge_starts[mapping.new_edge]
        assert mapping.new_position == mapping.new_edge
        end_index = kept[mapping.new_edge + 1]
        assert edge_low[mapping.new_edge] <= index <= edge_high[mapping.new_edge]
        assert index < n

        start = points[start_index]
        end = points[end_index]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        assert 0.0 <= mapping.t <= 1.0
        projected = (start[0] + mapping.t * dx, start[1] + mapping.t * dy)
        distance = math.hypot(
            points[index][0] - projected[0], points[index][1] - projected[1]
        )
        assert distance == pytest.approx(mapping.distance, abs=1e-9)
        # Every point must stay within its own permitted offset.
        assert mapping.distance <= tolerances[index] + 1e-9
        # Projection parameters never decrease along one replacement edge.
        earlier = previous_t_by_edge.get(mapping.new_edge, 0.0)
        assert mapping.t + 1e-12 >= earlier
        previous_t_by_edge[mapping.new_edge] = mapping.t


def test_mapping_covers_every_vertex_in_order():
    points = [(0, 0), (1, 0.5), (2, -0.5), (3, 0.5), (4, 0)]
    tolerances = [0.6] * len(points)
    result = simplify_lines([points], tolerances)
    _assert_mapping_consistency(points, tolerances, result)


def test_retained_vertices_map_to_edge_endpoints():
    points = [(0, 0), (1, 5), (2, 0), (3, 5), (4, 0)]
    result = simplify_lines([points], 1.0)
    (kept,) = result.kept
    (mappings,) = result.mappings
    for vertex in kept[:-1]:
        assert mappings[vertex].t == 0.0
        assert mappings[vertex].distance == 0.0
    assert mappings[kept[-1]].t == 1.0


# --------------------------------------------------------------------------
# Fold-backs, repeated coordinates and flow direction
# --------------------------------------------------------------------------


def test_foldback_keeps_an_anchor_and_stays_traceable():
    # Out-and-back reach: a single edge would collapse to zero length / backtrack.
    points = [(0, 0), (1, 0), (2, 0), (1, 0), (0, 0)]
    result = simplify_lines([points], 100.0, check_flow=True)
    (kept,) = result.kept
    (mappings,) = result.mappings
    assert len(kept) >= 3
    assert flow_direction_traceable(points, kept, mappings)


def test_repeated_identical_endpoints_collapse_but_remain_mapped():
    points = [(0, 0), (0, 0), (1, 1), (2, 0), (2, 0)]
    result = simplify_lines([points], 100.0)
    (kept,) = result.kept
    (mappings,) = result.mappings
    # Duplicate boundary vertices are not kept, but every index is mapped.
    assert kept == [1, 3]
    assert len(mappings) == len(points)
    assert all(mapping.new_edge == 0 for mapping in mappings)
    assert mappings[0].t == 0.0 and mappings[1].t == 0.0
    assert mappings[3].t == 1.0 and mappings[4].t == 1.0
    assert flow_direction_traceable(points, kept, mappings)


def test_interior_duplicate_run_does_not_create_zero_length_edge():
    points = [(0, 0), (1, 0), (1, 0), (2, 0)]
    result = simplify_lines([points], 100.0)
    (kept,) = result.kept
    assert result.lines == [[(0, 0), (2, 0)]]
    assert flow_direction_traceable(points, kept, result.mappings[0])


def test_degenerate_part_without_two_distinct_points_raises():
    with pytest.raises(DegeneratePartError):
        simplify_lines([[(1, 1), (1, 1)]], 1.0)
    with pytest.raises(DegeneratePartError):
        simplify_lines([[(1, 1)]], 1.0)


def test_flow_direction_traceable_rejects_zero_length_edge():
    points = [(0, 0), (0, 0), (1, 0)]
    # Hand-built "kept" with a coincident endpoint pair is not traceable.
    kept = [0, 1, 2]
    mappings = [
        VertexMapping(0, 0, 0.0, 0.0),
        VertexMapping(0, 0, 0.0, 1.0),
        VertexMapping(1, 1, 0.0, 1.0),
    ]
    assert not flow_direction_traceable(points, kept, mappings)


# --------------------------------------------------------------------------
# Optimality of the dynamic program, checked against brute force
# --------------------------------------------------------------------------


def _edge_is_feasible(points, tolerances, i, j, forced):
    start = points[i]
    end = points[j]
    if start == end:  # zero-length replacement edge
        return False
    previous_t = 0.0
    length_sq = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
    for k in range(i + 1, j):
        if forced[k]:  # forced vertices cannot be spanned over
            return False
        t = (
            (points[k][0] - start[0]) * (end[0] - start[0])
            + (points[k][1] - start[1]) * (end[1] - start[1])
        ) / length_sq
        t = min(1.0, max(0.0, t))
        if t + 1e-12 < previous_t:
            return False
        projected = (
            start[0] + t * (end[0] - start[0]),
            start[1] + t * (end[1] - start[1]),
        )
        distance = math.hypot(points[k][0] - projected[0], points[k][1] - projected[1])
        if distance > tolerances[k] + 1e-12:
            return False
        previous_t = t
    return True


def _brute_force_minimum(points, tolerances, forced):
    n = len(points)
    required = {0, n - 1, *(i for i, flag in enumerate(forced) if flag)}
    best = n + 1
    for mask in range(1 << n):
        chosen = [i for i in range(n) if mask & (1 << i)]
        if not required.issubset(chosen):
            continue
        if all(
            _edge_is_feasible(points, tolerances, a, b, forced)
            for a, b in zip(chosen, chosen[1:])
        ):
            best = min(best, len(chosen))
    return best


def test_dp_matches_brute_force_minimum():
    rng = random.Random(20261113)
    for _ in range(300):
        n = rng.randint(3, 10)
        # Keep distinct, non-duplicated endpoints so the brute-force model and
        # the implementation force the same boundary vertices.
        points = [(rng.randint(0, 4), rng.randint(0, 4)) for _ in range(n - 2)]
        points = [(0, 0), *points, (8, 8)]
        points[1] = (1, 1) if points[1] == (0, 0) else points[1]
        points[-2] = (7, 7) if points[-2] == (8, 8) else points[-2]
        tolerances = [rng.choice([0.0, 0.5, 1.0, 3.0, 100.0]) for _ in range(n)]
        forced = [rng.random() < 0.25 for _ in range(n)]
        forced[0] = forced[-1] = False
        expected = _brute_force_minimum(points, tolerances, forced)
        if expected > n:
            # Coincident consecutive required vertices make any traceable
            # replacement edge impossible; both models must agree it fails.
            with pytest.raises(DegeneratePartError):
                simplify_lines([points], [tolerances], [forced])
            continue
        result = simplify_lines([points], [tolerances], [forced])
        assert len(result.kept[0]) == expected
        # Independently re-verify the returned solution is itself feasible.
        _assert_mapping_consistency(points, tolerances, result)


# --------------------------------------------------------------------------
# Shape.simplify integration
# --------------------------------------------------------------------------


def test_shape_simplify_returns_same_shape_type():
    line = shapefile.Polyline(
        [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)],
        [(10, 10), (11, 11), (12, 10)],
    )
    result = line.simplify(100.0)
    assert isinstance(result, SimplifiedShape)
    assert result.shape.shapeType == shapefile.POLYLINE
    assert len(result.shape.points) == sum(len(kept) for kept in result.kept)
    assert list(result.shape.parts) == [0, 2]


def test_shape_simplify_rejects_point_and_multipoint_shapes():
    with pytest.raises(shapefile.ShapefileException):
        shapefile.Point(1, 1).simplify(1.0)
    with pytest.raises(shapefile.ShapefileException):
        shapefile.MultiPoint((1, 1), (2, 2)).simplify(1.0)


def test_shape_simplify_preserves_z_and_m_in_order():
    line = shapefile.PolylineZ(
        [(0, 0, 1.0), (1, 0, 2.0), (2, 0, 3.0), (3, 0, 4.0)],
        z=[1.0, 2.0, 3.0, 4.0],
        m=[0.1, 0.2, 0.3, 0.4],
    )
    result = line.simplify(100.0)
    assert result.kept == [[0, 3]]
    assert list(result.shape.z) == [1.0, 4.0]
    assert list(result.shape.m) == [0.1, 0.4]
    assert result.shape.zbox == shapefile.ZBox(1.0, 4.0)
    assert result.shape.mbox == shapefile.MBox(0.1, 0.4)


def test_shape_simplify_confluence_by_coordinate():
    line = shapefile.Polyline([(0, 0), (5, 5), (9, 9)], [(1, 1), (5, 5), (2, 2)])
    result = line.simplify(100.0, confluence=[(5, 5)])
    assert result.kept == [[0, 1, 2], [0, 1, 2]]


def test_polygon_ring_closure_vertex_is_retained():
    polygon = shapefile.Polygon([(0, 0), (1, 1), (2, 0), (2, 2), (0, 2), (0, 0)])
    result = polygon.simplify(100.0)
    points = result.shape.points
    assert points[0] == points[-1]


def test_simplified_shape_roundtrips_through_bytes():
    original = shapefile.Polyline(
        [(0, 0), (1, 0), (2, 0), (1, 0), (0, 0)],
        [(10, 10), (11, 11), (12, 10), (13, 11), (14, 10)],
    )
    simplified = original.simplify(100.0, check_flow=True).shape

    shp_stream = io.BytesIO()
    shx_stream = io.BytesIO()
    dbf_stream = io.BytesIO()
    with shapefile.Writer(
        shapeType=shapefile.POLYLINE,
        shp=shp_stream,
        shx=shx_stream,
        dbf=dbf_stream,
    ) as writer:
        writer.field("id", "N")
        writer.shape(simplified)
        writer.record(1)

    shp_stream.seek(0)
    with shapefile.Reader(shp=shp_stream) as reader:
        read_back = reader.shape(0)
        assert read_back.points == simplified.points
        assert list(read_back.parts) == list(simplified.parts)


def test_empty_input_raises():
    with pytest.raises(DegeneratePartError):
        simplify_lines([], 1.0)
