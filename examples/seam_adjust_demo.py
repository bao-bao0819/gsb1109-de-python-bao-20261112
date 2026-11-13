"""End-to-end seam adjustment example.

Three north-up flight-line sheets share two seams.  The last sheet
carries a one-pixel integer drift; the second seam was only measured
as an along-track projection.  The solver finds the minimum-change fix,
applies it to the real world coordinates and reports the observations
sitting exactly at their tolerance.
"""

from __future__ import annotations

from affine import Affine
from seam_adjust import Sheet, Tie, solve


def main() -> None:
    """Build the demo problem, solve it and print the adjustment report."""
    pixel_size = 5.0

    def transform(x0: float, y0: float = 0.0) -> Affine:
        return Affine(pixel_size, 0.0, x0, 0.0, -pixel_size, y0)

    sheets = [
        Sheet("L1", transform(0.0), dx_range=(-2, 2), dy_range=(-2, 2)),
        Sheet("L2", transform(495.0), dx_range=(-2, 2), dy_range=(-2, 2)),
        Sheet("L3", transform(1002.5), dx_range=(-2, 2), dy_range=(-2, 2)),
    ]

    ties = [
        Tie(
            name="seam-L1-L2",
            sheet_a="L1",
            sheet_b="L2",
            pixel_a=(99.0, 20.0),
            pixel_b=(0.0, 20.0),
            tolerance=0.5,
        ),
        Tie(
            name="seam-L2-L3",
            sheet_a="L2",
            sheet_b="L3",
            pixel_a=(100.0, 10.0),
            pixel_b=(0.0, 10.0),
            tolerance=2.5,
            along_track=(1.0, 0.0),
        ),
    ]

    result = solve(sheets, ties)
    print(result.report())
    print()
    print("corrected world coordinates of the measured pixel centres:")
    for name, (point_a, point_b) in result.corrected_points.items():
        print(
            f"  {name}: A=({point_a[0]:.3f}, {point_a[1]:.3f})"
            f"  B=({point_b[0]:.3f}, {point_b[1]:.3f})"
        )
    print()
    print("observations exactly at tolerance:", result.boundary_ties or "none")


if __name__ == "__main__":
    main()
