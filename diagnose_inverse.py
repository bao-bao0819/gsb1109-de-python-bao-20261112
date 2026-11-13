"""Exact-rational diagnosis of non-invertible / ill-conditioned image transforms.

Each "image" is described by six finite-decimal affine coefficients

    | x |   | a  b  c | | col |
    | y | = | d  e  f | | row |
    | 1 |   | 0  0  1 | |  1  |

mapping integer pixel (col, row) centers to world coordinates.  Coefficients
and query points are all finite decimals, so they are parsed with
``fractions.Fraction`` and every comparison stays exact: no rounding can turn
a singular matrix into an invertible one or merge distinct pixel centers.

For every image the diagnosis reports, per query:

* ``none``  -- no pixel center in the finite grid maps to the world point,
* ``one``   -- exactly one pixel center maps to it,
* ``many``  -- several centers coincide there (two distinct witnesses given).

Singularity (``det == 0``) is kept separate from answerability: a rank-1
transform collapses the image to a line and a rank-0 transform to a single
point, yet points on that line/point still have answers (one or many).  The
invertible but cancellation-prone case is solved exactly as well, exposing
where ordinary float inversion returns the wrong pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

Number = str | int | float
Coeffs = tuple[Number, Number, Number, Number, Number, Number]


@dataclass(frozen=True)
class Diagnosis:
    """Outcome of locating pixel centers for one world-coordinate query."""

    query: tuple[Fraction, Fraction]
    rank: int
    exact_determinant: Fraction
    float_determinant: float
    cancellation: bool
    status: str  # "none", "one", or "many"
    witnesses: tuple[tuple[int, int], ...]

    @property
    def has_answer(self) -> bool:
        return self.status != "none"


def _fraction(value: Number) -> Fraction:
    """Parse a finite decimal (or number) into an exact Fraction."""
    return Fraction(value) if isinstance(value, int) else Fraction(str(value))


def _linear_rank(a: Fraction, b: Fraction, d: Fraction, e: Fraction) -> int:
    if a == 0 and b == 0 and d == 0 and e == 0:
        return 0
    if a * e - b * d == 0:
        return 1
    return 2


def locate_centers(
    coeffs: Coeffs, ncols: int, nrows: int, x: Number, y: Number
) -> Diagnosis:
    """Locate every in-grid pixel center mapping exactly to ``(x, y)``.

    Parameters
    ----------
    coeffs:
        The six affine coefficients ``(a, b, c, d, e, f)`` as finite decimals.
    ncols, nrows:
        Finite extent of the image: ``col`` in ``range(ncols)`` and
        ``row`` in ``range(nrows)``.
    x, y:
        World-coordinate query, again finite decimals.

    Returns
    -------
    Diagnosis
        Exact rank, determinants, cancellation flag, and all witness pixels.
    """
    a, b, c, d, e, f = (_fraction(v) for v in coeffs)
    target_x, target_y = _fraction(x), _fraction(y)

    determinant = a * e - b * d
    float_determinant = (
        float(coeffs[0]) * float(coeffs[4])
        - float(coeffs[1]) * float(coeffs[3])
    )
    rank = _linear_rank(a, b, d, e)
    cancellation = (
        determinant != 0
        and float_determinant != 0.0
        and abs(float_determinant)
        < 1e-12
        * max(
            abs(float(coeffs[0])) * abs(float(coeffs[4])),
            abs(float(coeffs[1])) * abs(float(coeffs[3])),
            1.0,
        )
    )

    residual_x, residual_y = target_x - c, target_y - f
    witnesses: list[tuple[int, int]] = []

    if rank == 2:
        # Unique preimage; keep it only if it is an in-grid integer center.
        col = (e * residual_x - b * residual_y) / determinant
        row = (-d * residual_x + a * residual_y) / determinant
        if (
            col.denominator == 1
            and row.denominator == 1
            and 0 <= col < ncols
            and 0 <= row < nrows
        ):
            witnesses.append((col.numerator, row.numerator))
    else:
        # Singular: the finite grid is small, so enumerate it exactly rather
        # than pretending the inverse exists.
        for row in range(nrows):
            for col in range(ncols):
                if (
                    a * col + b * row == residual_x
                    and d * col + e * row == residual_y
                ):
                    witnesses.append((col, row))

    status = {0: "none", 1: "one"}.get(len(witnesses), "many")
    return Diagnosis(
        query=(target_x, target_y),
        rank=rank,
        exact_determinant=determinant,
        float_determinant=float_determinant,
        cancellation=cancellation,
        status=status,
        witnesses=tuple(witnesses),
    )


def _forward(coeffs: Coeffs, col: int, row: int) -> tuple[Fraction, Fraction]:
    a, b, c, d, e, f = (_fraction(v) for v in coeffs)
    return a * col + b * row + c, d * col + e * row + f


def _report(name: str, coeffs: Coeffs, ncols: int, nrows: int,
            queries: Iterable[tuple[Number, Number]]) -> list[Diagnosis]:
    print(f"\n### {name}  grid={ncols}x{nrows}  coeffs={coeffs}")
    results: list[Diagnosis] = []
    for x, y in queries:
        diag = locate_centers(coeffs, ncols, nrows, x, y)
        results.append(diag)
        label = {
            "none": "NO pixel center",
            "one": "one center",
            "many": "MANY centers coincide",
        }[diag.status]
        print(f"  query=({x}, {y})  rank={diag.rank}  -> {label}")
        print(
            f"      exact det={diag.exact_determinant}  "
            f"float det={diag.float_determinant!r}  "
            f"cancellation={diag.cancellation}"
        )
        for col, row in diag.witnesses:
            mapped = _forward(coeffs, col, row)
            assert mapped == diag.query
            print(f"      witness (col={col}, row={row}) -> world{mapped}")
        if diag.status == "many":
            assert len({p for p in diag.witnesses}) >= 2
    return results


def main() -> None:
    # Image A: rank 1. The linear part (0.1, 0.2 / 0.2, 0.4) collapses the
    # plane onto the line through (10, 20) in direction (1, 2).
    line_coeffs: Coeffs = ("0.1", "0.2", "10.0", "0.2", "0.4", "20.0")
    line_results = _report(
        "Image A: coverage segment degenerates to a line",
        line_coeffs,
        4,
        4,
        [
            ("10.2", "20.4"),  # on the line, two centers share it
            ("10.9", "21.8"),  # on the line, only the corner (3, 3)
            ("10.2", "20.5"),  # off the line, no center
        ],
    )
    many, one, none = line_results
    assert many.status == "many" and many.witnesses == ((2, 0), (0, 1))
    assert one.status == "one" and one.witnesses == ((3, 3),)
    assert none.status == "none"

    # Image B: rank 0. Every pixel center maps to the single point (5, 7).
    point_coeffs: Coeffs = ("0", "0", "5.0", "0", "0", "7.0")
    point_results = _report(
        "Image B: all pixel centers fall on one point",
        point_coeffs,
        4,
        4,
        [("5.0", "7.0"), ("5.0", "8.0")],
    )
    assert point_results[0].status == "many"
    assert len(point_results[0].witnesses) == 16
    assert point_results[1].status == "none"

    # Image C: full rank, exact determinant 1, but the two products differ by
    # one part in 1e16 so float evaluation yields det = 2.0 (cancellation).
    ill_coeffs: Coeffs = (
        "100000000",
        "99999999.99999999",
        "0",
        "100000000",
        "100000000",
        "0",
    )
    ill_results = _report(
        "Image C: nonzero determinant with severe float cancellation",
        ill_coeffs,
        4,
        4,
        [
            ("199999999.99999999", "200000000"),  # exact preimage (1, 1)
            ("999999999.99999995", "1000000000"),  # exact preimage (5, 5), out of grid
        ],
    )
    inside, outside = ill_results
    assert inside.cancellation and inside.status == "one"
    assert inside.witnesses == ((1, 1),)
    assert outside.status == "none"

    # Naive float inversion of the same query picks the wrong pixel: it
    # returns (2, 0), whereas the exact answer is the in-grid center (1, 1).
    from affine import Affine

    float_transform = Affine(*(float(v) for v in ill_coeffs))
    float_query = (float("199999999.99999999"), 200000000.0)
    naive_pixel = tuple(round(v) for v in (~float_transform @ float_query))
    print(
        "\nFloat-inversion check on Image C: naive inverse gives "
        f"{naive_pixel}, exact diagnosis gives {inside.witnesses[0]}."
    )
    assert naive_pixel != inside.witnesses[0]


if __name__ == "__main__":
    main()
