"""Affine transformation matrices.

The Affine package is derived from Casey Duncan's Planar package. See the
copyright statement below.
"""

#############################################################################
# Copyright (c) 2010 by Casey Duncan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name(s) of the copyright holders nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#############################################################################

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from functools import cached_property
import math
from typing import overload
import warnings

from attrs import astuple, define, field

__all__ = [
    "Affine",
    "GridAlignment",
    "Window",
    "align_grid",
]
__author__ = "Sean Gillies"
__version__ = "3.0.1"

EPSILON: float = 1e-5
EPSILON2: float = 1e-10

# Default tolerance for aligning two pixel grids, expressed as a fraction
# of the smaller pixel size.  It is deliberately much tighter than a
# pixel: a grid is either shared exactly (up to floating-point noise) or
# resampling would be required.
DEFAULT_GRID_TOLERANCE: float = 1e-6


class AffineError(Exception):
    pass


class TransformNotInvertibleError(AffineError):
    """The transform could not be inverted."""


class UndefinedRotationError(AffineError):
    """The rotation angle could not be computed for this transform."""


class GridAlignmentError(AffineError):
    """Two raster grids cannot be mapped onto one another by whole pixels."""


#: The signed permutation matrices, as ``(a, b, d, e)`` coefficients, that
#: preserve a square pixel grid.  They express axis order swaps and flips in
#: addition to the identity.
_GRID_PERMUTATIONS = (
    (1, 0, 0, 1),
    (1, 0, 0, -1),
    (-1, 0, 0, 1),
    (-1, 0, 0, -1),
    (0, 1, 1, 0),
    (0, 1, -1, 0),
    (0, -1, 1, 0),
    (0, -1, -1, 0),
)


@define(frozen=True)
class Window:
    """A half-open window of pixels in a raster.

    Coordinates are rows and columns, matching the ``(row, col)`` convention
    of raster shapes.  ``row_stop`` and ``col_stop`` are exclusive, so the
    number of pixels is ``height * width``.  A window with a zero height or
    width contains no pixels; this is how edge-only and corner-only contact
    is represented.
    """

    row_start: int = field(converter=int)
    row_stop: int = field(converter=int)
    col_start: int = field(converter=int)
    col_stop: int = field(converter=int)

    def __attrs_post_init__(self) -> None:
        if self.row_start < 0 or self.col_start < 0:
            raise ValueError("window starts must be non-negative")
        if self.row_stop < self.row_start or self.col_stop < self.col_start:
            raise ValueError("window stops must not precede their starts")

    @property
    def height(self) -> int:
        """Number of rows in the window."""
        return self.row_stop - self.row_start

    @property
    def width(self) -> int:
        """Number of columns in the window."""
        return self.col_stop - self.col_start

    @property
    def count(self) -> int:
        """Number of pixels in the window."""
        return self.height * self.width

    @property
    def shape(self) -> tuple[int, int]:
        """The ``(rows, cols)`` shape of the window."""
        return (self.height, self.width)

    @property
    def is_empty(self) -> bool:
        """True if the window contains no pixels."""
        return self.count == 0


@define(frozen=True)
class GridAlignment:
    """A whole-pixel mapping from one raster grid to another.

    The mapping is expressed in pixel corner coordinates::

        col_b = a * col_a + b * row_a + col_offset
        row_b = d * col_a + e * row_a + row_offset

    ``a``, ``b``, ``d``, and ``e`` form a signed permutation matrix, so the
    mapping may translate by whole pixels, swap the row and column axes, and
    flip either axis.  It may not rotate by an angle that is not a multiple of
    90 degrees, scale, or shear.

    Attributes
    ----------
    column_mapping, row_mapping : tuple of three ints
        ``(coefficient for col_a, coefficient for row_a, offset)`` for the
        destination column and row respectively.
    transform : Affine
        The same mapping as an affine transform from source pixel corner
        coordinates to destination pixel corner coordinates.
    worst_case_deviation : float
        The largest grid-point discrepancy, in world units, over the full
        source and destination extents.  It is evaluated at the corners of
        both coverages, where the extrema of this affine residual must occur.
    worst_case_deviation_pixels : float
        The same discrepancy expressed in source pixels.
    tolerance : float
        The acceptance tolerance in pixels that was used.
    overlap_a, overlap_b : Window
        The overlapping pixel window in the source and destination raster.
        Either is empty when the rasters have no area overlap, including when
        they meet only along an edge or at a corner.
    """

    column_mapping: tuple[int, int, int]
    row_mapping: tuple[int, int, int]
    transform: Affine
    worst_case_deviation: float
    worst_case_deviation_pixels: float
    tolerance: float
    overlap_a: Window
    overlap_b: Window

    @property
    def permutation(self) -> tuple[int, int, int, int]:
        """The signed permutation matrix as ``(a, b, d, e)``."""
        return (
            self.column_mapping[0],
            self.column_mapping[1],
            self.row_mapping[0],
            self.row_mapping[1],
        )

    @property
    def offset(self) -> tuple[int, int]:
        """The integer ``(col, row)`` translation in destination pixels."""
        return (self.column_mapping[2], self.row_mapping[2])

    def map_to_b(self, col: int, row: int) -> tuple[int, int]:
        """Map a source pixel corner ``(col, row)`` to destination pixels."""
        return (
            self.column_mapping[0] * col
            + self.column_mapping[1] * row
            + self.column_mapping[2],
            self.row_mapping[0] * col + self.row_mapping[1] * row + self.row_mapping[2],
        )

    def map_to_a(self, col: int, row: int) -> tuple[int, int]:
        """Map a destination pixel corner ``(col, row)`` back to source pixels."""
        inverse = ~self.transform
        mapped = inverse @ (float(col), float(row))
        return (round(mapped[0]), round(mapped[1]))

    @property
    def overlap(self) -> Window:
        """The overlap window in the source raster (alias of ``overlap_a``)."""
        return self.overlap_a

    @property
    def overlap_pixels(self) -> int:
        """Number of pixels in the overlap; zero for edge or corner contact."""
        return self.overlap_a.count


def cos_sin_deg(deg: float) -> tuple[float, float]:
    """Return the cosine and sin for the given angle in degrees.

    With special-case handling of multiples of 90 for perfect right
    angles.
    """
    deg = deg % 360.0
    if deg == 90.0:
        return 0.0, 1.0
    if deg == 180.0:
        return -1.0, 0
    if deg == 270.0:
        return 0, -1.0
    rad = math.radians(deg)
    return math.cos(rad), math.sin(rad)


@define(frozen=True)
class Affine:
    """Two dimensional affine transform for 2D linear mapping.

    Parallel lines are preserved by these transforms. Affine transforms
    can perform any combination of translations, scales/flips, shears,
    and rotations. Class methods are provided to conveniently compose
    transforms from these operations.

    Parameters
    ----------
    a, b, c, d, e, f : float
        Coefficients of the 3 x 3 augmented affine transformation matrix.

    g, h, i : float, optional
        Coefficients of the 3 x 3 augmented affine transformation matrix.

    Attributes
    ----------
    a, b, c, d, e, f, g, h, i : float
        Coefficients of the 3 x 3 augmented affine transformation matrix.

        .. code-block:: none

            | x' |   | a  b  c | | x |
            | y' | = | d  e  f | | y |
            | 1  |   | g  h  i | | 1 |

        `g`, `h`, and `i` are always 0, 0, and 1.

    Notes
    -----
    Multiplication of a transform and an (x, y) vector *always* returns
    the column vector that is the matrix multiplication product of the
    transform and (x, y) as a column vector, no matter which is on the
    left or right side. This is obviously not the case for matrices and
    vectors in general, but provides a convenience for users of this
    class.

    The Affine package is derived from Casey Duncan's Planar package.
    See the copyright statement.
    """

    a: float = field(converter=float)
    b: float = field(converter=float)
    c: float = field(converter=float)
    d: float = field(converter=float)
    e: float = field(converter=float)
    f: float = field(converter=float)

    # The class has 3 attributes that don't have to be specified: g, h,
    # and i. If they are, the given value has to be the same as the
    # default value. This allows a new instances to be created from the
    # tuple form of another, like Affine(*Affine.identity()).

    g: float = field(default=0.0, converter=float)

    @g.validator
    def _check_g(self, attribute, value):
        if value != 0.0:
            raise ValueError("g must be equal to 0.0")

    h: float = field(default=0.0, converter=float)

    @h.validator
    def _check_h(self, attribute, value):
        if value != 0.0:
            raise ValueError("h must be equal to 0.0")

    i: float = field(default=1.0, converter=float)

    @i.validator
    def _check_i(self, attribute, value):
        if value != 1.0:
            raise ValueError("i must be equal to 1.0")

    @classmethod
    def from_gdal(
        cls, c: float, a: float, b: float, f: float, d: float, e: float
    ) -> Affine:
        """Use same coefficient order as GDAL's GetGeoTransform().

        Parameters
        ----------
        c, a, b, f, d, e : float
            Parameters ordered by GDAL's GeoTransform.

        Returns
        -------
        Affine
        """
        return cls(a, b, c, d, e, f)

    @classmethod
    def identity(cls) -> Affine:
        """Return the identity transform.

        Returns
        -------
        Affine
        """
        return identity

    @classmethod
    def translation(cls, xoff: float, yoff: float) -> Affine:
        """Create a translation transform from an offset vector.

        Parameters
        ----------
        xoff, yoff : float
            Translation offsets in x and y directions.

        Returns
        -------
        Affine
        """
        return cls(1.0, 0.0, xoff, 0.0, 1.0, yoff)

    @classmethod
    def scale(cls, *scaling: float) -> Affine:
        """Create a scaling transform from a scalar or vector.

        Parameters
        ----------
        *scaling : float or sequence of two floats
            One or two scaling factors. A scalar value will scale in
            both dimensions equally. A vector scaling value scales the
            dimensions independently.

        Returns
        -------
        Affine
        """
        if len(scaling) == 1:
            sx = scaling[0]
            sy = sx
        else:
            sx, sy = scaling
        return cls(sx, 0.0, 0.0, 0.0, sy, 0.0)

    @classmethod
    def shear(cls, x_angle: float = 0.0, y_angle: float = 0.0) -> Affine:
        """Create a shear transform along one or both axes.

        Parameters
        ----------
        x_angle, y_angle : float
            Shear angles in degrees parallel to the x- and y-axis.

        Returns
        -------
        Affine
        """
        mx = math.tan(math.radians(x_angle))
        my = math.tan(math.radians(y_angle))
        return cls(1.0, mx, 0.0, my, 1.0, 0.0)

    @classmethod
    def rotation(cls, angle: float, pivot: Sequence[float] | None = None) -> Affine:
        """Create a rotation transform at the specified angle.

        Parameters
        ----------
        angle : float
            Rotation angle in degrees, counter-clockwise about the pivot
            point.
        pivot : sequence of float (px, py), optional
            Pivot point coordinates to rotate around. If None (default),
            the pivot point is the coordinate system origin (0.0, 0.0).

        Returns
        -------
        Affine
        """
        ca, sa = cos_sin_deg(angle)
        if pivot is None:
            return cls(ca, -sa, 0.0, sa, ca, 0.0)
        px, py = pivot
        # fmt: off
        return cls(
            ca, -sa, px - px * ca + py * sa,
            sa, ca, py - px * sa - py * ca,
        )
        # fmt: on

    @classmethod
    def permutation(cls, *scaling: float) -> Affine:
        """Create the permutation transform.

        For 2x2 matrices, there is only one permutation matrix that is
        not the identity.

        Parameters
        ----------
        *scaling : any
            Ignored.

        Returns
        -------
        Affine
        """
        return cls(0.0, 1.0, 0.0, 1.0, 0.0, 0.0)

    def __str__(self) -> str:
        """Concise string representation."""
        return (
            f"|{self.a: .2f},{self.b: .2f},{self.c: .2f}|\n"
            f"|{self.d: .2f},{self.e: .2f},{self.f: .2f}|\n"
            f"|{self.g: .2f},{self.h: .2f},{self.i: .2f}|"
        )

    def __repr__(self) -> str:
        """Precise string representation."""
        return (
            f"Affine({self.a!r}, {self.b!r}, {self.c!r},\n"
            f"       {self.d!r}, {self.e!r}, {self.f!r})"
        )

    def to_gdal(self) -> tuple[float, float, float, float, float, float]:
        """Return same coefficient order expected by GDAL's SetGeoTransform().

        Returns
        -------
        tuple
            Ordered: c, a, b, f, d, e.
        """
        return (self.c, self.a, self.b, self.f, self.d, self.e)

    def to_shapely(self) -> tuple[float, float, float, float, float, float]:
        """Return affine transformation parameters for shapely's affinity module.

        Returns
        -------
        tuple
            Ordered: a, b, d, e, c, f.
        """
        return (self.a, self.b, self.d, self.e, self.c, self.f)

    @property
    def xoff(self) -> float:
        """Alias for 'c'."""
        return self.c

    @property
    def yoff(self) -> float:
        """Alias for 'f'."""
        return self.f

    @cached_property
    def determinant(self) -> float:
        """Evaluate the determinant of the transform matrix.

        This value is equal to the area scaling factor when the
        transform is applied to a shape.

        Returns
        -------
        float
        """
        return self.a * self.e - self.b * self.d

    @property
    def _scaling(self) -> tuple[float, float]:
        """The absolute scaling factors of the transformation.

        This tuple represents the absolute value of the scaling factors
        of the transformation, sorted from bigger to smaller.
        """
        a, b, d, e = self.a, self.b, self.d, self.e

        # The singular values are the square root of the eigenvalues
        # of the matrix times its transpose, M M*
        # Computing trace and determinant of M M*
        trace = a**2 + b**2 + d**2 + e**2
        det2 = (a * e - b * d) ** 2

        delta = trace**2 / 4.0 - det2
        if delta < EPSILON2:
            delta = 0.0

        sqrt_delta = math.sqrt(delta)
        l1 = math.sqrt(trace / 2.0 + sqrt_delta)
        l2 = math.sqrt(trace / 2.0 - sqrt_delta)
        return l1, l2

    @property
    def eccentricity(self) -> float:
        """The eccentricity of the affine transformation.

        This value represents the eccentricity of an ellipse under
        this affine transformation.

        Raises
        ------
        NotImplementedError
            For improper transformations.
        """
        l1, l2 = self._scaling
        return math.sqrt(l1**2 - l2**2) / l1

    @property
    def rotation_angle(self) -> float:
        """The rotation angle in degrees of the affine transformation.

        This is the rotation angle in degrees of the affine
        transformation, assuming it is in the form M = R S, where R is
        a rotation and S is a scaling.

        Raises
        ------
        UndefinedRotationError
            For improper and degenerate transformations.
        """
        if self.is_proper or self.is_degenerate:
            l1, _ = self._scaling
            y, x = self.d / l1, self.a / l1
            return math.degrees(math.atan2(y, x))
        raise UndefinedRotationError

    @property
    def is_identity(self) -> bool:
        """True if this transform equals the identity matrix, within rounding limits."""
        return self is identity or self.almost_equals(identity, EPSILON)

    @property
    def is_rectilinear(self) -> bool:
        """True if the transform is rectilinear.

        i.e., whether a shape would remain axis-aligned, within rounding
        limits, after applying the transform.
        """
        return (abs(self.a) < EPSILON and abs(self.e) < EPSILON) or (
            abs(self.d) < EPSILON and abs(self.b) < EPSILON
        )

    @property
    def is_conformal(self) -> bool:
        """True if the transform is conformal.

        i.e., if angles between points are preserved after applying the
        transform, within rounding limits.  This implies that the
        transform has no effective shear.
        """
        return abs(self.a * self.b + self.d * self.e) < EPSILON

    @property
    def is_orthonormal(self) -> bool:
        """True if the transform is orthonormal.

        Which means that the transform represents a rigid motion, which
        has no effective scaling or shear. Mathematically, this means
        that the axis vectors of the transform matrix are perpendicular
        and unit-length.  Applying an orthonormal transform to a shape
        always results in a congruent shape.
        """
        a, b, d, e = self.a, self.b, self.d, self.e
        return (
            self.is_conformal
            and abs(1.0 - (a * a + d * d)) < EPSILON
            and abs(1.0 - (b * b + e * e)) < EPSILON
        )

    @cached_property
    def is_degenerate(self) -> bool:
        """Return True if this transform is degenerate.

        A degenerate transform will collapse a shape to an effective area
        of zero, and cannot be inverted.

        Returns
        -------
        bool
        """
        return self.determinant == 0.0

    @cached_property
    def is_proper(self) -> bool:
        """Return True if this transform is proper.

        A proper transform (with a positive determinant) does not include
        reflection.

        Returns
        -------
        bool
        """
        return self.determinant > 0.0

    @property
    def column_vectors(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """The values of the transform as three 2D column vectors.

        Returns
        -------
        tuple of three tuple pairs
            Ordered (a, d), (b, e), (c, f).
        """
        return (self.a, self.d), (self.b, self.e), (self.c, self.f)

    def almost_equals(self, other: Affine, precision: float | None = None) -> bool:
        """Compare transforms for approximate equality.

        Parameters
        ----------
        other : Affine
            Transform being compared.
        precision : float, default EPSILON
            Precision to use to evaluate equality.

        Returns
        -------
        bool
            True if absolute difference between each element
            of each respective transform matrix < ``precision``.
        """
        precision = precision or EPSILON
        return all(abs(sv - ov) < precision for sv, ov in zip(self, other))

    @cached_property
    def _astuple(self) -> tuple[float]:
        return astuple(self)

    def __getitem__(self, index):
        return self._astuple[index]

    def __iter__(self):
        return iter(self._astuple)

    def __len__(self):
        return 9

    def __gt__(self, other) -> bool:
        return NotImplemented

    __ge__ = __lt__ = __le__ = __gt__

    # Override from base class. We do not support entrywise
    # addition, subtraction or scalar multiplication because
    # the result is not an affine transform

    def __add__(self, other):
        raise TypeError("Operation not supported")

    __iadd__ = __add__

    @overload
    def __matmul__(self, other: Affine) -> Affine: ...
    @overload
    def __matmul__(self, other: tuple[float, float]) -> tuple[float, float]: ...
    @overload
    def __matmul__(
        self, other: tuple[float, float, float]
    ) -> tuple[float, float, float]: ...
    # For other float sequences, we don't know the returned tuple length here
    @overload
    def __matmul__(self, other: Sequence[float]) -> tuple[float, ...]: ...
    def __matmul__(self, other):
        """Matrix multiplication.

        Apply the transform using matrix multiplication, creating
        a resulting object of the same type.  A transform may be applied
        to another transform, a vector, vector array, or shape.

        Parameters
        ----------
        other : Affine or iterable of (vx, vy, [vw])

        Returns
        -------
        Affine or a tuple of two or three items
        """
        sa, sb, sc, sd, se, sf = self[:6]
        if isinstance(other, Affine):
            oa, ob, oc, od, oe, of = other[:6]
            return self.__class__(
                sa * oa + sb * od,
                sa * ob + sb * oe,
                sa * oc + sb * of + sc,
                sd * oa + se * od,
                sd * ob + se * oe,
                sd * oc + se * of + sf,
            )
        # vector of 2 or 3 items
        try:
            num_items = len(other)
        except (TypeError, ValueError):
            return NotImplemented
        if num_items == 2:
            vx, vy = other
        elif num_items == 3:
            vx, vy, vw = other
            vw_eq_one = vw == 1.0
            try:
                is_eq_one = bool(vw_eq_one)
                msg = "third value must be 1.0"
            except ValueError:
                is_eq_one = (vw_eq_one).all()
                msg = "third values must all be 1.0"
            if not is_eq_one:
                raise ValueError(msg)
        else:
            raise TypeError("expected vector of 2 or 3 items")
        px = vx * sa + vy * sb + sc
        py = vx * sd + vy * se + sf
        if num_items == 2:
            return (px, py)
        return (px, py, vw)

    def __rmatmul__(self, other):
        return NotImplemented

    def __imatmul__(self, other):  # type: ignore
        if not isinstance(other, Affine):
            raise TypeError("Operation not supported")
        return NotImplemented

    @overload
    def __mul__(self, other: Affine) -> Affine: ...
    @overload
    def __mul__(self, other: tuple[float, float]) -> tuple[float, float]: ...
    def __mul__(self, other):
        """Multiplication.

        Apply the transform using matrix multiplication, creating
        a resulting object of the same type.  A transform may be applied
        to another transform, a vector, vector array, or shape.

        Parameters
        ----------
        other : Affine or iterable of (vx, vy)

        Returns
        -------
        Affine or a tuple of two items

        .. deprecated:: 3.1.0
            Use `@` matmul instead of `*` mul operator for matrix multiplication.
        """
        # TODO: consider elevating to DeprecationWarning for 3.2
        warnings.warn(
            "Use `@` matmul instead of `*` mul operator for matrix multiplication",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        if isinstance(other, Affine):
            return self.__matmul__(other)
        try:
            _, _ = other
            return self.__matmul__(other)
        except (ValueError, TypeError):
            return NotImplemented

    def __rmul__(self, other):
        return NotImplemented

    def __imul__(self, other):  # type: ignore
        if isinstance(other, tuple):
            warnings.warn(
                "in-place multiplication with tuple is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )
        return NotImplemented

    def itransform(self, seq: MutableSequence[Sequence[float]]) -> None:
        """Transform a sequence of points or vectors in-place.

        Parameters
        ----------
        seq : mutable sequence

        Returns
        -------
        None
            The input sequence is mutated in-place.
        """
        if self is not identity and self != identity:
            sa, sb, sc, sd, se, sf = self[:6]
            for i, (x, y) in enumerate(seq):
                seq[i] = (x * sa + y * sb + sc, x * sd + y * se + sf)

    def __invert__(self):
        """Return the inverse transform.

        Raises
        ------
        TransformNotInvertible
            If the transform is degenerate.
        """
        if self.is_degenerate:
            raise TransformNotInvertibleError("Cannot invert degenerate transform")
        idet = 1.0 / self.determinant
        sa, sb, sc, sd, se, sf = self[:6]
        ra = se * idet
        rb = -sb * idet
        rd = -sd * idet
        re = sa * idet
        # fmt: off
        return self.__class__(
            ra, rb, -sc * ra - sf * rb,
            rd, re, -sc * rd - sf * re,
        )
        # fmt: on

    def __getnewargs__(self):
        """Pickle protocol support.

        Notes
        -----
        Normal unpickling creates a situation where __new__ receives all
        9 elements rather than the 6 that are required for the
        constructor.  This method ensures that only the 6 are provided.
        """
        return self[:6]


identity = Affine(1, 0, 0, 0, 1, 0)
"""The identity transform"""

# Miscellaneous utilities


def _checked_shape(shape: Sequence[int], name: str) -> tuple[int, int]:
    """Return ``(rows, cols)`` from a raster shape."""
    try:
        rows, cols = shape
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (rows, cols) pair") from None
    if isinstance(rows, bool) or isinstance(cols, bool):
        raise TypeError(f"{name} dimensions must be integers")
    rows, cols = int(rows), int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return rows, cols


def _pixel_scale(transform: Affine) -> float:
    """Return the shortest pixel edge length of a raster transform."""
    return min(
        math.hypot(transform.a, transform.d),
        math.hypot(transform.b, transform.e),
    )


def _grid_deviation(
    source: Affine,
    destination: Affine,
    inverse_destination: Affine,
    mapping: Affine,
    p: tuple[float, float],
) -> float:
    """Physical distance between one grid node under two raster transforms.

    The comparison is carried out in destination pixel space to avoid
    cancelling large world coordinates, then converted to world units with
    the destination transform's linear part.
    """
    q = mapping @ p
    source_pixel = inverse_destination @ (source @ p)
    offset_col = source_pixel[0] - q[0]
    offset_row = source_pixel[1] - q[1]
    return math.hypot(
        destination.a * offset_col + destination.b * offset_row,
        destination.d * offset_col + destination.e * offset_row,
    )


def _corner_residuals(
    source: Affine,
    destination: Affine,
    mapping: Affine,
    source_shape: tuple[int, int],
    destination_shape: tuple[int, int],
) -> float:
    """Maximum grid-node residual over both full raster extents.

    The residual is the physical distance between the same grid node
    expressed through both rasters.  Every residual component is affine in
    pixel coordinates, so its worst magnitude over a rectangular extent is
    attained at a corner.  Both extents are checked: a mapping that looks
    fine at the source origin can drift arbitrarily far at a far corner.
    """
    source_rows, source_cols = source_shape
    destination_rows, destination_cols = destination_shape
    inverse_destination = ~destination
    inverse_mapping = ~mapping

    worst = 0.0
    for col, row in (
        (0.0, 0.0),
        (float(source_cols), 0.0),
        (0.0, float(source_rows)),
        (float(source_cols), float(source_rows)),
    ):
        worst = max(
            worst,
            _grid_deviation(
                source, destination, inverse_destination, mapping, (col, row)
            ),
        )

    for col, row in (
        (0.0, 0.0),
        (float(destination_cols), 0.0),
        (0.0, float(destination_rows)),
        (float(destination_cols), float(destination_rows)),
    ):
        p = inverse_mapping @ (col, row)
        worst = max(
            worst,
            _grid_deviation(source, destination, inverse_destination, mapping, p),
        )
    return worst


def _rect_q_bounds(
    mapping: Affine, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Bounds of a raster rectangle under an integer pixel mapping."""
    rows, cols = shape
    qs = [
        mapping @ (float(col), float(row))
        for col, row in ((0, 0), (cols, 0), (0, rows), (cols, rows))
    ]
    return (
        round(min(q[0] for q in qs)),
        round(max(q[0] for q in qs)),
        round(min(q[1] for q in qs)),
        round(max(q[1] for q in qs)),
    )


def _overlap_windows(
    mapping: Affine,
    source_shape: tuple[int, int],
    destination_shape: tuple[int, int],
) -> tuple[Window, Window]:
    """Half-open overlap windows in source and destination pixel coordinates.

    An overlap is an area intersection of pixel cells.  When the rasters
    only share an edge or a corner the intersection has zero width or zero
    height, and the returned windows contain no pixels.
    """
    source_q = _rect_q_bounds(mapping, source_shape)
    dest_rows, dest_cols = destination_shape
    col_start = max(source_q[0], 0)
    col_stop = min(source_q[1], dest_cols)
    row_start = max(source_q[2], 0)
    row_stop = min(source_q[3], dest_rows)
    if col_stop <= col_start or row_stop <= row_start:
        empty = Window(0, 0, 0, 0)
        return empty, empty

    destination_window = Window(row_start, row_stop, col_start, col_stop)

    inverse = ~mapping
    source_ps = [
        inverse @ (float(col), float(row))
        for col, row in (
            (col_start, row_start),
            (col_stop, row_start),
            (col_start, row_stop),
            (col_stop, row_stop),
        )
    ]
    src_col_start = round(min(p[0] for p in source_ps))
    src_col_stop = round(max(p[0] for p in source_ps))
    src_row_start = round(min(p[1] for p in source_ps))
    src_row_stop = round(max(p[1] for p in source_ps))
    src_rows, src_cols = source_shape
    src_col_start = max(src_col_start, 0)
    src_col_stop = min(src_col_stop, src_cols)
    src_row_start = max(src_row_start, 0)
    src_row_stop = min(src_row_stop, src_rows)
    return (
        Window(src_row_start, src_row_stop, src_col_start, src_col_stop),
        destination_window,
    )


def align_grid(
    transform_a: Affine,
    shape_a: Sequence[int],
    transform_b: Affine,
    shape_b: Sequence[int],
    *,
    tolerance: float = DEFAULT_GRID_TOLERANCE,
) -> GridAlignment:
    """Align two raster grids by whole pixels, swapping and flipping axes.

    The search is restricted to mappings of the form ``q = P @ p + t``
    where ``P`` is a signed permutation matrix and ``t`` is an integer
    translation in destination pixels.  Such a mapping changes axis order,
    flips an axis, or translates by whole pixels, but never rotates by a
    non-right angle, scales, or shears, so no resampling would be needed.

    Candidate mappings are not accepted on coefficient similarity alone.
    For every candidate, the Euclidean residual between the same grid node
    represented through both rasters is evaluated at the corners of both
    full extents.  A tiny pixel-size mismatch accumulates with distance and
    is rejected when it makes any covered grid point deviate further than
    the tolerance.

    Parameters
    ----------
    transform_a, transform_b : Affine
        Affine transforms from pixel corner coordinates to world
        coordinates.
    shape_a, shape_b : sequence of two ints
        Raster shapes as ``(rows, cols)``.
    tolerance : float, optional
        Largest allowed grid-node discrepancy as a fraction of the shorter
        pixel edge.  Defaults to one millionth of a pixel, which admits
        floating-point round-off but not any genuine resampling.

    Returns
    -------
    GridAlignment
        The integer mapping, its worst-case deviation, and the half-open
        overlap windows.  The windows are empty when the rasters only touch
        along an edge or at a corner.

    Raises
    ------
    GridAlignmentError
        If no signed-permutation and whole-pixel translation aligns both
        full extents within ``tolerance``.
    ValueError
        If a shape is not a positive ``(rows, cols)`` pair.
    TransformNotInvertibleError
        If either transform is degenerate.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    source_rows, source_cols = _checked_shape(shape_a, "shape_a")
    destination_rows, destination_cols = _checked_shape(shape_b, "shape_b")
    source_shape = (source_rows, source_cols)
    destination_shape = (destination_rows, destination_cols)

    if transform_a.is_degenerate:
        raise TransformNotInvertibleError("source transform is degenerate")
    if transform_b.is_degenerate:
        raise TransformNotInvertibleError("destination transform is degenerate")

    pixel_scale = min(_pixel_scale(transform_a), _pixel_scale(transform_b))
    if pixel_scale == 0.0:
        raise TransformNotInvertibleError("pixel edge has zero length")
    allowed_deviation = tolerance * pixel_scale

    # The source origin maps to this pixel coordinate in the destination
    # grid.  Candidate whole-pixel offsets are seeded from several reference
    # points of the source extent under every candidate permutation; taking
    # both floor and ceiling means floating-point round-off near an integer
    # cannot hide the true offset.
    offset_candidates: set[tuple[int, int]] = set()
    reference_points = (
        (0, 0),
        (source_cols / 2.0, source_rows / 2.0),
        (float(source_cols), 0),
        (0, float(source_rows)),
        (float(source_cols), float(source_rows)),
    )
    for pa, pb, pd, pe in _GRID_PERMUTATIONS:
        basis = Affine(float(pa), float(pb), 0.0, float(pd), float(pe), 0.0)
        for col, row in reference_points:
            world = transform_a @ (col, row)
            pixel = ~transform_b @ world
            seed = basis @ (col, row)
            for candidate_col in (math.floor(pixel[0]), math.ceil(pixel[0])):
                for candidate_row in (math.floor(pixel[1]), math.ceil(pixel[1])):
                    offset_candidates.add(
                        (
                            round(candidate_col - seed[0]),
                            round(candidate_row - seed[1]),
                        )
                    )

    best: GridAlignment | None = None
    best_worst = math.inf
    for pa, pb, pd, pe in _GRID_PERMUTATIONS:
        for offset_col, offset_row in offset_candidates:
            mapping = Affine(
                float(pa),
                float(pb),
                float(offset_col),
                float(pd),
                float(pe),
                float(offset_row),
            )
            worst = _corner_residuals(
                transform_a,
                transform_b,
                mapping,
                source_shape,
                destination_shape,
            )
            if worst < best_worst:
                overlap_a, overlap_b = _overlap_windows(
                    mapping, source_shape, destination_shape
                )
                best_worst = worst
                best = GridAlignment(
                    column_mapping=(pa, pb, offset_col),
                    row_mapping=(pd, pe, offset_row),
                    transform=mapping,
                    worst_case_deviation=worst,
                    worst_case_deviation_pixels=worst / pixel_scale,
                    tolerance=tolerance,
                    overlap_a=overlap_a,
                    overlap_b=overlap_b,
                )

    if best is None or best_worst > allowed_deviation:
        raise GridAlignmentError(
            "no whole-pixel grid alignment within tolerance "
            f"{tolerance} px (best worst-case deviation "
            f"{(best_worst / pixel_scale) if best else math.inf:.6g} px)"
        )
    return best


def loadsw(s: str) -> Affine:
    """Return Affine from the contents of a world file string.

    This method also translates the coefficients from center- to
    corner-based coordinates.

    Parameters
    ----------
    s : str
        String with 6 floats ordered in a world file.

    Returns
    -------
    Affine
    """
    if not hasattr(s, "split"):
        raise TypeError("Cannot split input string")
    coeffs = s.split()
    if len(coeffs) != 6:
        raise ValueError(f"Expected 6 coefficients, found {len(coeffs)}")
    a, d, b, e, c, f = (float(x) for x in coeffs)
    center = Affine(a, b, c, d, e, f)
    return center @ Affine.translation(-0.5, -0.5)


def dumpsw(obj: Affine) -> str:
    """Return string for a world file.

    This method also translates the coefficients from corner- to
    center-based coordinates.

    Returns
    -------
    str
    """
    center = obj @ Affine.translation(0.5, 0.5)
    return "\n".join(repr(getattr(center, x)) for x in list("adbecf")) + "\n"


def set_epsilon(epsilon: float) -> None:
    """Set the global absolute error value and rounding limit.

    This value is accessible via the affine.EPSILON global variable.

    Parameters
    ----------
    epsilon : float
        The global absolute error value and rounding limit for
        approximate floating point comparison operations.

    Returns
    -------
    None

    Notes
    -----
    The default value of ``0.00001`` is suitable for values that are in
    the "countable range". You may need a larger epsilon when using
    large absolute values, and a smaller value for very small values
    close to zero. Otherwise approximate comparison operations will not
    behave as expected.
    """
    global EPSILON, EPSILON2
    EPSILON = float(epsilon)
    EPSILON2 = EPSILON**2


set_epsilon(1e-5)
