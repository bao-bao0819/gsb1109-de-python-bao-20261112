"""Selection of overview levels and tiles for windowed raster reads.

Given a target window in world coordinates and a maximum acceptable
pixel size, choose the cheapest overview level and the set of tiles
that must be read to cover the window.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import TYPE_CHECKING

from attrs import define, field, validators

if TYPE_CHECKING:
    from affine import Affine

__all__ = ["OverviewLevel", "OverviewSelection", "select_overview"]

# Relative tolerance for resolution, coverage, and intersection checks.
_RTOL = 1e-9


def _max_stretch(transform: Affine) -> float:
    """Largest singular value of the linear part of a transform.

    This is the maximum world-coordinate length of a unit pixel step,
    taken over all directions.
    """
    a, b, d, e = transform.a, transform.b, transform.d, transform.e
    trace = a * a + b * b + d * d + e * e
    det2 = (a * e - b * d) ** 2
    delta = max(0.0, trace * trace / 4.0 - det2)
    return math.sqrt(trace / 2.0 + math.sqrt(delta))


def _polygon_area(points: list[tuple[float, float]]) -> float:
    """Absolute area of a polygon given by its vertices."""
    total = 0.0
    for i, (x0, y0) in enumerate(points):
        x1, y1 = points[(i + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def _clip_half_plane(
    points: list[tuple[float, float]], inside, intersect
) -> list[tuple[float, float]]:
    """Clip a polygon to a half plane (Sutherland-Hodgman step)."""
    if not points:
        return []
    clipped = []
    prev = points[-1]
    prev_inside = inside(prev)
    for cur in points:
        cur_inside = inside(cur)
        if cur_inside:
            if not prev_inside:
                clipped.append(intersect(prev, cur))
            clipped.append(cur)
        elif prev_inside:
            clipped.append(intersect(prev, cur))
        prev, prev_inside = cur, cur_inside
    return clipped


def _clip_to_rect(
    points: list[tuple[float, float]],
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> list[tuple[float, float]]:
    """Clip a polygon to an axis-aligned rectangle."""

    def clip_vertical(points, x, keep_ge):
        def intersect(p, q):
            t = (x - p[0]) / (q[0] - p[0])
            return (x, p[1] + t * (q[1] - p[1]))

        if keep_ge:
            return _clip_half_plane(points, lambda p: p[0] >= x, intersect)
        return _clip_half_plane(points, lambda p: p[0] <= x, intersect)

    def clip_horizontal(points, y, keep_ge):
        def intersect(p, q):
            t = (y - p[1]) / (q[1] - p[1])
            return (p[0] + t * (q[0] - p[0]), y)

        if keep_ge:
            return _clip_half_plane(points, lambda p: p[1] >= y, intersect)
        return _clip_half_plane(points, lambda p: p[1] <= y, intersect)

    points = clip_vertical(points, xmin, True)
    points = clip_vertical(points, xmax, False)
    points = clip_horizontal(points, ymin, True)
    return clip_horizontal(points, ymax, False)


def _normalize_tile_sizes(sizes) -> dict[tuple[int, int], int]:
    """Normalize tile sizes to a mapping of (tile_col, tile_row) to bytes."""
    if isinstance(sizes, Mapping):
        return dict(sizes)
    return {
        (col, row): int(size)
        for row, row_sizes in enumerate(sizes)
        for col, size in enumerate(row_sizes)
    }


@define(frozen=True)
class OverviewLevel:
    """One candidate overview level of a tiled raster.

    Parameters
    ----------
    transform : Affine
        Maps pixel (col, row) coordinates to world coordinates.
    width, height : int
        Size of the level in pixels. Levels may be clipped, so this
        extent can be smaller than the full image extent.
    tile_width, tile_height : int
        Tile size in pixels.
    tile_sizes : mapping or sequence of sequences
        Compressed size in bytes of each tile. Either a mapping of
        ``(tile_col, tile_row)`` to bytes, or row-major nested
        sequences ``tile_sizes[row][col]``. Tiles missing from a
        mapping are treated as zero bytes.
    """

    transform: Affine
    width: int = field(validator=validators.gt(0))
    height: int = field(validator=validators.gt(0))
    tile_width: int = field(validator=validators.gt(0))
    tile_height: int = field(validator=validators.gt(0))
    tile_sizes: dict[tuple[int, int], int] = field(converter=_normalize_tile_sizes)


@define(frozen=True)
class OverviewSelection:
    """Outcome of :func:`select_overview`.

    Attributes
    ----------
    level_index : int or None
        Index of the chosen level, or None when no level qualifies.
    tiles : tuple of (int, int)
        ``(tile_col, tile_row)`` pairs that must be read.
    total_bytes : int
        Sum of the compressed sizes of the selected tiles.
    resolution_ok : bool
        True when at least one level meets the resolution limit.
    coverage_ok : bool
        True when at least one level fully covers the window.
    """

    level_index: int | None
    tiles: tuple[tuple[int, int], ...]
    total_bytes: int
    resolution_ok: bool
    coverage_ok: bool

    @property
    def ok(self) -> bool:
        """True when a qualifying level and tile set were found."""
        return self.level_index is not None

    @property
    def unmet(self) -> tuple[str, ...]:
        """Names of the conditions no candidate level could satisfy."""
        return tuple(
            name
            for name, met in (
                ("resolution", self.resolution_ok),
                ("coverage", self.coverage_ok),
            )
            if not met
        )


def _tiles_for_window(
    level: OverviewLevel, pixel_points: list[tuple[float, float]]
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Tiles of a level with positive-area intersection with the window.

    Tiles that only touch the window along an edge or at a point have
    zero intersection area and are not read.
    """
    tiles = []
    total_bytes = 0
    ncols = -(-level.width // level.tile_width)
    nrows = -(-level.height // level.tile_height)
    for tile_row in range(nrows):
        ymin = tile_row * level.tile_height
        ymax = min(ymin + level.tile_height, level.height)
        for tile_col in range(ncols):
            xmin = tile_col * level.tile_width
            xmax = min(xmin + level.tile_width, level.width)
            clipped = _clip_to_rect(pixel_points, xmin, ymin, xmax, ymax)
            area_tol = _RTOL * max(1.0, (xmax - xmin) * (ymax - ymin))
            if len(clipped) >= 3 and _polygon_area(clipped) > area_tol:
                tiles.append((tile_col, tile_row))
                total_bytes += level.tile_sizes.get((tile_col, tile_row), 0)
    return tuple(tiles), total_bytes


def select_overview(
    window: Sequence[tuple[float, float]],
    max_resolution: float,
    levels: Sequence[OverviewLevel],
) -> OverviewSelection:
    """Choose the cheapest overview level and tiles covering a window.

    A level qualifies when the maximum stretch of its pixels in any
    direction does not exceed ``max_resolution`` and its (possibly
    clipped) extent fully covers ``window``. Among qualifying levels
    the one whose required tiles sum to the fewest compressed bytes
    wins; the highest-numbered overview is not assumed to be cheapest.
    Tiles that only touch the window along an edge or at a point are
    not read.

    Parameters
    ----------
    window : sequence of (float, float)
        Vertices of the target window polygon in world coordinates.
    max_resolution : float
        Maximum acceptable pixel size in world units, measured as the
        largest stretch of a pixel in any direction.
    levels : sequence of OverviewLevel
        Candidate levels, including the full-resolution level.

    Returns
    -------
    OverviewSelection
        The chosen level and tiles. When no level qualifies,
        ``level_index`` is None and ``resolution_ok``/``coverage_ok``
        report which condition could not be met.
    """
    points = [(float(x), float(y)) for x, y in window]
    if len(points) < 3:
        raise ValueError("window polygon requires at least 3 vertices")
    if not max_resolution > 0.0:
        raise ValueError("max_resolution must be positive")
    candidates = list(levels)
    if not candidates:
        raise ValueError("at least one overview level is required")

    resolution_ok = False
    coverage_ok = False
    best = None
    for index, level in enumerate(candidates):
        stretch = _max_stretch(level.transform)
        meets_resolution = stretch <= max_resolution * (1.0 + _RTOL)
        resolution_ok = resolution_ok or meets_resolution

        inverse = ~level.transform
        pixel_points = [inverse @ point for point in points]
        tol = _RTOL * max(level.width, level.height, 1)
        covers = all(
            -tol <= col <= level.width + tol and -tol <= row <= level.height + tol
            for col, row in pixel_points
        )
        coverage_ok = coverage_ok or covers
        if not (meets_resolution and covers):
            continue

        tiles, total_bytes = _tiles_for_window(level, pixel_points)
        ranking = (total_bytes, len(tiles), index)
        if best is None or ranking < best[0]:
            best = (ranking, index, tiles, total_bytes)

    if best is None:
        return OverviewSelection(None, (), 0, resolution_ok, coverage_ok)
    _, index, tiles, total_bytes = best
    return OverviewSelection(index, tiles, total_bytes, resolution_ok, coverage_ok)
