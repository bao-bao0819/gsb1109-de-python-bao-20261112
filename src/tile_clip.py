"""Clip Z/M polylines to rectangular tiles and stitch the pieces back together.

Road polylines carry Z (elevation) and M (mileage / measure) values.  This
module cuts such lines against a set of rectangular tiles, interpolating the
Z and M values of the cut points linearly along the original segments, and
can later merge the per-tile pieces back into full lines.

Conventions:

* Points are ``(x, y, z, m)`` tuples.  ``z`` and ``m`` may be ``None``
  (or NaN) when unmeasured.  A missing M is *never* treated as zero: if
  either endpoint of a segment lacks an M value, interpolated cut points
  on that segment get ``None`` as well.
* Tiles are half-open: a point on a shared edge belongs to the tile on
  the right (larger x) or top (larger y) side only, so a road running
  exactly along a common tile edge is emitted by a single tile.  The
  outer maximum edges of the whole tiling are included, so nothing on
  the boundary of the tiled area is lost.
* Every clipped piece records its source feature, source part index and
  the start/end positions along the source part, both as arc length
  (``start_s``/``end_s``) and as mileage (``start_m``/``end_m``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Optional, Sequence, Tuple

# A point is (x, y, z, m); z and m may be None when unmeasured.
Point = Tuple[float, float, Optional[float], Optional[float]]

__all__ = [
    "Point",
    "Tile",
    "ClippedPiece",
    "MergedLine",
    "clip_to_tiles",
    "merge_pieces",
    "parts_from_shape",
]


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _as_xyzm(point: Sequence) -> Point:
    x = float(point[0])
    y = float(point[1])
    z = point[2] if len(point) > 2 else None
    m = point[3] if len(point) > 3 else None
    return (x, y, None if _is_missing(z) else z, None if _is_missing(m) else m)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_optional(a, b, t: float):
    # Exact endpoints keep their own value, even if the other end is missing.
    if t == 0.0:
        return None if _is_missing(a) else a
    if t == 1.0:
        return None if _is_missing(b) else b
    # Missing measures/elevations propagate; they are never treated as zero.
    if _is_missing(a) or _is_missing(b):
        return None
    return _lerp(a, b, t)


def _interp_point(p0: Point, p1: Point, t: float) -> Point:
    return (
        _lerp(p0[0], p1[0], t),
        _lerp(p0[1], p1[1], t),
        _lerp_optional(p0[2], p1[2], t),
        _lerp_optional(p0[3], p1[3], t),
    )


@dataclass(frozen=True)
class Tile:
    """A rectangular tile: [xmin, xmax) x [ymin, ymax)."""

    tile_id: Hashable
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class ClippedPiece:
    """The part of one source part that falls inside one tile."""

    feature_id: Hashable
    part_index: int
    tile_id: Hashable
    points: list  # list[Point], first/last may be interpolated cut points
    start_s: float  # arc length along the source part at points[0]
    end_s: float  # arc length along the source part at points[-1]

    @property
    def start_m(self):
        """Mileage (M) at the start of the piece, None if unmeasured."""
        return self.points[0][3]

    @property
    def end_m(self):
        """Mileage (M) at the end of the piece, None if unmeasured."""
        return self.points[-1][3]


@dataclass
class MergedLine:
    """A line stitched back together from per-tile pieces."""

    feature_id: Hashable
    part_index: int
    points: list  # list[Point]; shared cut points appear exactly once
    start_s: float
    end_s: float
    tile_ids: tuple = field(default_factory=tuple)

    @property
    def start_m(self):
        return self.points[0][3]

    @property
    def end_m(self):
        return self.points[-1][3]


def _clip_segment_to_box(p0: Point, p1: Point, tile: Tile):
    """Liang-Barsky clip of segment p0->p1 against the tile's closed box.

    Returns the parameter interval (t0, t1) of the portion inside the box,
    or None if the segment misses the box entirely.
    """
    x0, y0 = p0[0], p0[1]
    dx = p1[0] - x0
    dy = p1[1] - y0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0 - tile.xmin),
        (dx, tile.xmax - x0),
        (-dy, y0 - tile.ymin),
        (dy, tile.ymax - y0),
    ):
        if p == 0:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r
    return t0, t1


def _owns(tile: Tile, x: float, y: float, global_xmax: float, global_ymax: float) -> bool:
    """Half-open ownership: shared edges belong to the right/top tile only.

    The outermost max edges of the whole tiling are included so boundary
    points of the tiled area are not dropped.
    """
    in_x = tile.xmin <= x < tile.xmax or (x == tile.xmax == global_xmax)
    in_y = tile.ymin <= y < tile.ymax or (y == tile.ymax == global_ymax)
    return in_x and in_y


def clip_to_tiles(
    features: Iterable,
    tiles: Iterable[Tile],
) -> list:
    """Clip every part of every feature against every tile.

    ``features`` is an iterable of ``(feature_id, parts)`` where ``parts``
    is a list of parts, each part a sequence of ``(x, y[, z[, m]])`` points.

    Returns a list of :class:`ClippedPiece`, one per non-empty intersection
    of a source segment chain with a tile.  Cut points are interpolated
    along the original segments, including their Z and M values.
    """
    tiles = list(tiles)
    if not tiles:
        return []
    global_xmax = max(t.xmax for t in tiles)
    global_ymax = max(t.ymax for t in tiles)

    pieces = []
    for feature_id, parts in features:
        for part_index, raw_part in enumerate(parts):
            part = [_as_xyzm(p) for p in raw_part]
            # Cumulative arc length along the source part.
            s = [0.0]
            for i in range(1, len(part)):
                s.append(
                    s[-1]
                    + math.hypot(
                        part[i][0] - part[i - 1][0],
                        part[i][1] - part[i - 1][1],
                    )
                )
            for i in range(len(part) - 1):
                p0, p1 = part[i], part[i + 1]
                seg_len = s[i + 1] - s[i]
                if seg_len == 0.0:
                    continue
                for tile in tiles:
                    span = _clip_segment_to_box(p0, p1, tile)
                    if span is None:
                        continue
                    t0, t1 = span
                    if t1 - t0 <= 1e-12:
                        continue
                    a = _interp_point(p0, p1, t0)
                    b = _interp_point(p0, p1, t1)
                    mid = _interp_point(p0, p1, (t0 + t1) / 2.0)
                    if not _owns(tile, mid[0], mid[1], global_xmax, global_ymax):
                        continue
                    pieces.append(
                        ClippedPiece(
                            feature_id=feature_id,
                            part_index=part_index,
                            tile_id=tile.tile_id,
                            points=[a, b],
                            start_s=s[i] + t0 * seg_len,
                            end_s=s[i] + t1 * seg_len,
                        )
                    )
    return pieces


def _close(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def _same_point(p: Point, q: Point, tol: float) -> bool:
    return _close(p[0], q[0], tol) and _close(p[1], q[1], tol)


def merge_pieces(pieces: Iterable[ClippedPiece], tol: float = 1e-9) -> list:
    """Stitch per-tile pieces back into lines, in source order.

    Pieces are grouped by ``(feature_id, part_index)`` and sorted by their
    position along the source part.  Consecutive pieces that meet at a
    shared cut point are joined and the duplicated cut point is kept only
    once.  Pieces separated by a gap (the line left the tiled area) or
    belonging to different parts are never joined, so disconnected parts
    stay disconnected.
    """
    groups = {}
    for piece in pieces:
        groups.setdefault((piece.feature_id, piece.part_index), []).append(piece)

    lines = []
    for (feature_id, part_index), group in groups.items():
        group.sort(key=lambda piece: (piece.start_s, piece.end_s))
        cur_points = None
        cur_start_s = 0.0
        cur_end_s = 0.0
        cur_tiles: list = []

        def flush():
            lines.append(
                MergedLine(
                    feature_id=feature_id,
                    part_index=part_index,
                    points=cur_points,
                    start_s=cur_start_s,
                    end_s=cur_end_s,
                    tile_ids=tuple(cur_tiles),
                )
            )

        for piece in group:
            if (
                cur_points is not None
                and _close(piece.start_s, cur_end_s, tol)
                and _same_point(cur_points[-1], piece.points[0], tol)
            ):
                # Shared cut point: keep it once.
                cur_points.extend(piece.points[1:])
                cur_end_s = piece.end_s
                cur_tiles.append(piece.tile_id)
            else:
                if cur_points is not None:
                    flush()
                cur_points = list(piece.points)
                cur_start_s = piece.start_s
                cur_end_s = piece.end_s
                cur_tiles = [piece.tile_id]
        if cur_points is not None:
            flush()
    return lines


def parts_from_shape(shape) -> list:
    """Convert a pyshp :class:`shapefile.Shape` into tile-clip parts.

    Returns a list of parts of ``(x, y, z, m)`` tuples, using ``None`` for
    Z/M values the shape does not carry (pyshp already decodes missing M
    values as ``None``).
    """
    points = shape.points
    zs = getattr(shape, "z", None) or []
    ms = getattr(shape, "m", None) or []
    indexes = list(shape.parts) + [len(points)]
    parts = []
    for i in range(len(indexes) - 1):
        part = []
        for j in range(indexes[i], indexes[i + 1]):
            z = zs[j] if j < len(zs) else None
            m = ms[j] if j < len(ms) else None
            part.append((points[j][0], points[j][1], z, m))
        parts.append(part)
    return parts
