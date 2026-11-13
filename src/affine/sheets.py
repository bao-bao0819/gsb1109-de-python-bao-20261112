"""Exact map-sheet coverage queries for batch imports.

Observation points are distributed to map sheets (partitions) that may
carry their own rotations or negative-direction axes. To decide which
sheets truly cover a point, this module inverts each sheet's affine
transform using exact rational arithmetic instead of floats.

All coefficients and world coordinates are given as integers, integer
strings, or finite decimal strings (``Fraction`` and finite ``Decimal``
instances are also accepted). Floats are rejected on purpose: with large
world coordinates and very small pixel sizes, converting to float first
can shift an exact business boundary by a whole column or row.

Pixel coordinates follow the half-open convention: the left and top
edges of a sheet are inclusive, the right and bottom edges exclusive.
A point exactly on a shared edge therefore belongs to exactly one
sheet, which prevents a record from being written to two partitions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
import math
import re
from typing import Union

from affine import TransformNotInvertibleError

__all__ = [
    "Coverage",
    "CoverageResult",
    "MapSheet",
    "SheetHit",
    "locate_point",
]

NumberLike = Union[int, str, Fraction, Decimal]

_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _exact(value: NumberLike) -> Fraction:
    """Parse an integer, finite decimal, Fraction, or Decimal exactly.

    Floats are rejected so that callers cannot lose precision before
    the exact boundary arithmetic happens.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid coordinate or coefficient")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"not a finite decimal: {value!r}")
        return Fraction(value)
    if isinstance(value, str):
        text = value.strip()
        if not _DECIMAL_RE.match(text):
            raise ValueError(f"not an integer or finite decimal: {value!r}")
        return Fraction(text)
    raise TypeError(
        f"expected int, str, Fraction, or Decimal, got {type(value).__name__}; "
        "pass integer strings or finite decimals to avoid float rounding"
    )


def _exact_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class MapSheet:
    """A map sheet partition with an exact world-from-pixel transform.

    The coefficients ``a`` through ``f`` map pixel coordinates to world
    coordinates, exactly like ``Affine``::

        x = a * col + b * row + c
        y = d * col + e * row + f

    Rotations, shears, and negative-direction axes (for example
    ``e < 0`` for a north-up sheet) are all supported. ``width`` and
    ``height`` are the sheet's extent in pixels; pixel coordinates
    ``0 <= col < width`` and ``0 <= row < height`` are covered.
    """

    sheet_id: str
    a: Fraction
    b: Fraction
    c: Fraction
    d: Fraction
    e: Fraction
    f: Fraction
    width: int
    height: int

    def __init__(
        self,
        sheet_id: str,
        a: NumberLike,
        b: NumberLike,
        c: NumberLike,
        d: NumberLike,
        e: NumberLike,
        f: NumberLike,
        width: int,
        height: int,
    ) -> None:
        object.__setattr__(self, "sheet_id", str(sheet_id))
        object.__setattr__(self, "a", _exact(a))
        object.__setattr__(self, "b", _exact(b))
        object.__setattr__(self, "c", _exact(c))
        object.__setattr__(self, "d", _exact(d))
        object.__setattr__(self, "e", _exact(e))
        object.__setattr__(self, "f", _exact(f))
        object.__setattr__(self, "width", _exact_dimension(width, "width"))
        object.__setattr__(self, "height", _exact_dimension(height, "height"))

    def pixel_coords(self, x: NumberLike, y: NumberLike) -> tuple[Fraction, Fraction]:
        """Return the exact fractional pixel coordinates of a world point.

        Parameters
        ----------
        x, y : int, str, Fraction, or Decimal
            World coordinates as integers or finite decimals.

        Returns
        -------
        tuple of two Fraction
            Exact fractional (col, row) pixel coordinates.

        Raises
        ------
        TransformNotInvertibleError
            If the sheet's transform is degenerate.
        """
        px = _exact(x)
        py = _exact(y)
        det = self.a * self.e - self.b * self.d
        if det == 0:
            raise TransformNotInvertibleError(
                f"sheet {self.sheet_id!r} has a degenerate transform"
            )
        dx = px - self.c
        dy = py - self.f
        col = (self.e * dx - self.b * dy) / det
        row = (self.a * dy - self.d * dx) / det
        return col, row


@dataclass(frozen=True)
class SheetHit:
    """One sheet that truly covers the queried point.

    Attributes
    ----------
    sheet_id : str
        Identifier of the covering sheet.
    col, row : int
        Pixel indices of the point within the sheet.
    col_exact, row_exact : Fraction
        Exact fractional pixel coordinates, kept for auditing boundary
        decisions.
    """

    sheet_id: str
    col: int
    row: int
    col_exact: Fraction
    row_exact: Fraction


class Coverage(Enum):
    """How the queried sheets cover a point."""

    NONE = "none"
    """No sheet covers the point."""

    INTERIOR = "interior"
    """Exactly one sheet covers the point, away from any sheet edge."""

    EDGE = "edge"
    """The point lies exactly on a sheet edge (shared or outer).

    The half-open convention assigns it to exactly one sheet: the one
    for which the edge is the left or top edge.
    """

    OVERLAP = "overlap"
    """Multiple sheets legitimately cover the point."""


@dataclass(frozen=True)
class CoverageResult:
    """Result of locating a point against a set of map sheets."""

    coverage: Coverage
    hits: tuple[SheetHit, ...]

    @property
    def sheet_ids(self) -> tuple[str, ...]:
        """Identifiers of all sheets that truly cover the point."""
        return tuple(hit.sheet_id for hit in self.hits)


def locate_point(
    sheets: Iterable[MapSheet], x: NumberLike, y: NumberLike
) -> CoverageResult:
    """Return every sheet that truly covers the world point (x, y).

    Parameters
    ----------
    sheets : iterable of MapSheet
        The candidate partitions.
    x, y : int, str, Fraction, or Decimal
        World coordinates as integers or finite decimals.

    Returns
    -------
    CoverageResult
        ``coverage`` is ``Coverage.NONE`` when no sheet covers the
        point, ``Coverage.OVERLAP`` when several sheets legitimately
        cover it, ``Coverage.EDGE`` when exactly one sheet covers it
        and the point lies exactly on a sheet edge, and
        ``Coverage.INTERIOR`` otherwise. ``hits`` lists each covering
        sheet with its pixel indices, in input order.
    """
    px = _exact(x)
    py = _exact(y)
    hits = []
    on_edge = False
    for sheet in sheets:
        col, row = sheet.pixel_coords(px, py)
        covers = 0 <= col < sheet.width and 0 <= row < sheet.height
        if covers:
            hits.append(
                SheetHit(
                    sheet_id=sheet.sheet_id,
                    col=math.floor(col),
                    row=math.floor(row),
                    col_exact=col,
                    row_exact=row,
                )
            )
            if col == 0 or row == 0:
                on_edge = True
        else:
            within_closed = 0 <= col <= sheet.width and 0 <= row <= sheet.height
            on_boundary = (
                col == 0 or col == sheet.width or row == 0 or row == sheet.height
            )
            if within_closed and on_boundary:
                on_edge = True
    if not hits:
        coverage = Coverage.NONE
    elif len(hits) > 1:
        coverage = Coverage.OVERLAP
    elif on_edge:
        coverage = Coverage.EDGE
    else:
        coverage = Coverage.INTERIOR
    return CoverageResult(coverage=coverage, hits=tuple(hits))
