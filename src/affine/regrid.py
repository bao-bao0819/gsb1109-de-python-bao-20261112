"""Area-weighted redistribution of raster totals between two grids.

An old pixel may straddle several new pixels (diagonally it can touch
four). This module redistributes every old pixel's total over the new
grid in proportion to the true area of intersection with each new pixel,
using exact polygon clipping rather than a point or center sampling.

Three books are kept for every input window so that nothing is lost:

* the share landing on *valid* new pixels inside the target window,
* the share landing on *masked* new pixels inside the target window, and
* the share landing entirely outside the target window.

The three sum back to the original total. Pixels that only touch along an
edge or at a corner have zero intersection area and so contribute nothing
to one another.

Only :class:`~affine.Affine` from this package and the Python standard
library are required. Plain nested sequences are accepted, so Numpy arrays
work as inputs without Numpy being a hard dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

from affine import Affine

__all__ = ["RegridError", "RegridResult", "regrid_window"]

# Intersection areas at or below this fraction of an old pixel's area are
# treated as edge/corner-only contact, suppressing floating-point noise.
_AREA_EPSILON = 1e-12

Point = tuple[float, float]
Polygon = list[Point]
InsideTest = Callable[[Point], bool]
Intersector = Callable[[Point, Point], Point]


class RegridError(ValueError):
    """Raised when regridding inputs are inconsistent or degenerate."""


@dataclass(frozen=True)
class RegridResult:
    """Outcome of redistributing one input window.

    Attributes
    ----------
    data : list of list of float
        Target grid with the same shape as ``valid_mask``. Valid pixels
        carry their redistributed total; masked pixels are zero.
    masked : float
        Total share that fell on masked new pixels.
    outside : float
        Total share that fell outside the target window.
    """

    data: list[list[float]]
    masked: float
    outside: float


def regrid_window(
    source: Sequence[Sequence[float]],
    src_transform: Affine,
    dst_transform: Affine,
    valid_mask: Sequence[Sequence[bool]],
) -> RegridResult:
    """Redistribute old pixel totals onto a new grid by intersection area.

    Each old pixel's total is split over the new pixels in proportion to
    the area of the polygon shared between them. Shares landing on masked
    new pixels or beyond the target window are accounted for separately
    instead of being silently dropped.

    Parameters
    ----------
    source : sequence of sequences of float
        Old pixel totals, shape ``(src_rows, src_cols)``.
    src_transform : Affine
        Transform mapping old pixel ``(col, row)`` corners to world
        coordinates.
    dst_transform : Affine
        Transform mapping new pixel ``(col, row)`` corners to world
        coordinates. It may have a negative determinant (a flipped grid);
        such a grid never yields a negative output.
    valid_mask : sequence of sequences of bool
        ``True`` marks usable new pixels, shape ``(dst_rows, dst_cols)``.

    Returns
    -------
    RegridResult
        The redistributed grid plus the masked and outside ledgers.

    Raises
    ------
    RegridError
        If the arrays have mismatched or empty shapes, the mask has the
        wrong shape, or either transform is degenerate and cannot be
        inverted.
    """
    src = _as_float_matrix(source, "source")
    mask = _as_bool_matrix(valid_mask, "valid_mask")
    src_rows, src_cols = _shape(src)
    dst_rows, dst_cols = _shape(mask)
    if not src_rows or not src_cols:
        raise RegridError("source must contain at least one pixel")
    if not dst_rows or not dst_cols:
        raise RegridError("valid_mask must contain at least one pixel")
    for row in src:
        if len(row) != src_cols:
            raise RegridError("source rows must all have the same length")
    for row in mask:
        if len(row) != dst_cols:
            raise RegridError("valid_mask rows must all have the same length")
    if src_transform.is_degenerate:
        raise RegridError("src_transform is degenerate and cannot be inverted")
    if dst_transform.is_degenerate:
        raise RegridError("dst_transform is degenerate and cannot be inverted")

    # Map world coordinates back to fractional new-pixel (col, row).
    inverse_dst = ~dst_transform

    target = [[0.0 for _ in range(dst_cols)] for _ in range(dst_rows)]
    masked_total = 0.0
    outside_total = 0.0

    unit_square = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

    for src_row in range(src_rows):
        for src_col in range(src_cols):
            total = src[src_row][src_col]
            if total == 0.0:
                continue

            # World-space corners of this old pixel, then its shape in
            # fractional new-pixel coordinates.
            world_corners = [
                src_transform @ (src_col + u, src_row + v) for u, v in unit_square
            ]
            old_polygon = [inverse_dst @ point for point in world_corners]
            old_area = abs(_polygon_area(old_polygon))
            if old_area <= 0.0:
                # Degenerate (zero-area) old pixel: nothing to distribute.
                continue

            inside, outside_area = _clip_to_window(old_polygon, dst_rows, dst_cols)
            if outside_area > 0.0:
                outside_total += total * outside_area / old_area
            if not inside:
                continue

            col_min, col_max, row_min, row_max = _candidate_bounds(
                inside, dst_rows, dst_cols
            )
            area_tolerance = _AREA_EPSILON * max(old_area, 1.0)
            for new_row in range(row_min, row_max):
                for new_col in range(col_min, col_max):
                    cell = (
                        (new_col, new_row),
                        (new_col + 1.0, new_row),
                        (new_col + 1.0, new_row + 1.0),
                        (new_col, new_row + 1.0),
                    )
                    overlap = abs(_polygon_area(_clip_polygon(inside, cell)))
                    if overlap <= area_tolerance:
                        # Edge- or corner-only contact has zero area; the
                        # absolute value also keeps flipped (negative
                        # determinant) grids non-negative.
                        continue
                    share = total * overlap / old_area
                    if mask[new_row][new_col]:
                        target[new_row][new_col] += share
                    else:
                        masked_total += share

    return RegridResult(data=target, masked=masked_total, outside=outside_total)


def _as_float_matrix(values: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    """Copy a numeric grid into a plain matrix of floats."""
    matrix: list[list[float]] = []
    for row in values:
        matrix.append([float(item) for item in row])
    if not matrix:
        raise RegridError(f"{name} must contain at least one row")
    return matrix


def _as_bool_matrix(values: Sequence[Sequence[bool]], name: str) -> list[list[bool]]:
    """Copy a truth-value grid into a plain matrix of booleans."""
    matrix: list[list[bool]] = []
    for row in values:
        matrix.append([bool(item) for item in row])
    if not matrix:
        raise RegridError(f"{name} must contain at least one row")
    return matrix


def _shape(matrix: Sequence[Sequence[object]]) -> tuple[int, int]:
    """Return the row and column counts of a non-empty matrix."""
    return len(matrix), len(matrix[0])


def _candidate_bounds(
    polygon: Polygon, dst_rows: int, dst_cols: int
) -> tuple[int, int, int, int]:
    """Integer new-pixel index range that could overlap the polygon."""
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    col_min = max(0, math.floor(min(xs)))
    col_max = min(dst_cols, math.ceil(max(xs)))
    row_min = max(0, math.floor(min(ys)))
    row_max = min(dst_rows, math.ceil(max(ys)))
    return col_min, col_max, row_min, row_max


def _clip_to_window(
    polygon: Polygon, dst_rows: int, dst_cols: int
) -> tuple[Polygon, float]:
    """Clip a polygon to the target window.

    Returns the clipped polygon (possibly empty) and the area removed.
    Each boundary is applied separately so a segment crossing the whole
    window is never dropped.
    """
    original_area = abs(_polygon_area(polygon))
    clipped = polygon
    for axis, bound, keep_greater in (
        ("x", 0.0, True),
        ("x", float(dst_cols), False),
        ("y", 0.0, True),
        ("y", float(dst_rows), False),
    ):
        clipped = _clip_axis(clipped, axis, bound, keep_greater)
        if not clipped:
            return [], original_area
    remaining_area = abs(_polygon_area(clipped))
    # Clamp against floating-point drift when the polygon is fully inside.
    return clipped, max(0.0, original_area - remaining_area)


def _clip_axis(
    polygon: Polygon, axis: str, bound: float, keep_greater: bool
) -> Polygon:
    """Clip against one axis-aligned boundary with Sutherland-Hodgman."""

    def inside(point: Point) -> bool:
        value = point[0] if axis == "x" else point[1]
        return value >= bound if keep_greater else value <= bound

    def intersection(start: Point, end: Point) -> Point:
        start_value = start[0] if axis == "x" else start[1]
        end_value = end[0] if axis == "x" else end[1]
        ratio = (bound - start_value) / (end_value - start_value)
        return (
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        )

    return _run_sutherland_hodgman(polygon, inside, intersection)


def _run_sutherland_hodgman(
    polygon: Polygon, inside: InsideTest, intersection: Intersector
) -> Polygon:
    """Clip a polygon against a single convex clipping half-plane."""
    output: Polygon = []
    if not polygon:
        return output
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def _clip_polygon(subject: Polygon, clip: Polygon) -> Polygon:
    """Clip a convex subject polygon by a convex clip polygon."""
    output = subject
    for index in range(len(clip)):
        edge_start = clip[index]
        edge_end = clip[(index + 1) % len(clip)]
        output = _clip_oriented_edge(output, edge_start, edge_end)
        if not output:
            break
    return output


def _clip_oriented_edge(
    polygon: Polygon, edge_start: Point, edge_end: Point
) -> Polygon:
    """Sutherland-Hodgman against one oriented clip edge.

    The interior is the half-plane to the left of the directed edge.
    """
    edge_x = edge_end[0] - edge_start[0]
    edge_y = edge_end[1] - edge_start[1]

    def is_inside(point: Point) -> bool:
        return (
            edge_x * (point[1] - edge_start[1]) - edge_y * (point[0] - edge_start[0])
        ) >= 0.0

    def intersection(start: Point, end: Point) -> Point:
        start_side = edge_x * (start[1] - edge_start[1]) - edge_y * (
            start[0] - edge_start[0]
        )
        end_side = edge_x * (end[1] - edge_start[1]) - edge_y * (end[0] - edge_start[0])
        ratio = start_side / (start_side - end_side)
        return (
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        )

    return _run_sutherland_hodgman(polygon, is_inside, intersection)


def _polygon_area(polygon: Polygon) -> float:
    """Signed shoelace area; zero for fewer than three vertices."""
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for index, (x, y) in enumerate(polygon):
        next_x, next_y = polygon[(index + 1) % len(polygon)]
        total += x * next_y - next_x * y
    return total / 2.0
