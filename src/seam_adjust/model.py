"""Input model for integer pixel-shift seam adjustment."""

from __future__ import annotations

from dataclasses import dataclass

from affine import Affine


@dataclass(frozen=True)
class Sheet:
    """One map sheet whose linear part is calibrated.

    Parameters
    ----------
    name
        Unique sheet identifier.
    transform
        Affine mapping pixel (col, row) centres to world coordinates.
    dx_range, dy_range
        Inclusive integer pixel-shift bounds ``(lo, hi)``.  The zero
        shift must lie inside both ranges.
    """

    name: str
    transform: Affine
    dx_range: tuple[int, int]
    dy_range: tuple[int, int]

    def __post_init__(self) -> None:
        dx_lo, dx_hi = self.dx_range
        dy_lo, dy_hi = self.dy_range
        if dx_lo > dx_hi or dy_lo > dy_hi:
            raise ValueError(f"sheet {self.name!r}: empty shift range")
        if not (dx_lo <= 0 <= dx_hi and dy_lo <= 0 <= dy_hi):
            raise ValueError(
                f"sheet {self.name!r}: shift ranges must contain the zero shift"
            )

    def domain(self) -> list[tuple[int, int]]:
        """All admissible integer pixel shifts, zero first."""
        values = [
            (dx, dy)
            for dx in range(self.dx_range[0], self.dx_range[1] + 1)
            for dy in range(self.dy_range[0], self.dy_range[1] + 1)
        ]
        values.sort(key=lambda v: (v != (0, 0), abs(v[0]) + abs(v[1]), v))
        return values


@dataclass(frozen=True)
class Tie:
    """One seam observation of the same feature on two sheets.

    Parameters
    ----------
    name
        Unique observation identifier.
    sheet_a, sheet_b
        The two sheets joined by this observation.
    pixel_a, pixel_b
        Measured pixel centres ``(col, row)`` of the feature on each
        sheet.
    tolerance
        Admissible residual magnitude in world units.
    along_track
        Unit vector in world coordinates pointing along the flight
        track.  When ``None`` the observation is a full 2-D match;
        when given, only the residual projection on this direction is
        constrained (the across-track component was not measured).
    radial
        Full 2-D matches only: when true the tolerance bounds the
        Euclidean residual norm; when false each component is bounded
        independently by the same tolerance.
    """

    name: str
    sheet_a: str
    sheet_b: str
    pixel_a: tuple[float, float]
    pixel_b: tuple[float, float]
    tolerance: float
    along_track: tuple[float, float] | None = None
    radial: bool = False

    def __post_init__(self) -> None:
        if self.sheet_a == self.sheet_b:
            raise ValueError(f"tie {self.name!r}: a tie must join two distinct sheets")
        if self.tolerance < 0.0:
            raise ValueError(f"tie {self.name!r}: tolerance must be non-negative")
        if self.along_track is not None:
            ux, uy = self.along_track
            norm = (ux * ux + uy * uy) ** 0.5
            if norm == 0.0:
                raise ValueError(
                    f"tie {self.name!r}: along-track vector must be non-zero"
                )
            object.__setattr__(self, "along_track", (ux / norm, uy / norm))

    @property
    def projected(self) -> bool:
        """True when only the along-track projection was measured."""
        return self.along_track is not None


@dataclass(frozen=True)
class TieResidual:
    """Evaluated residual of one observation under a chosen adjustment."""

    tie: str
    sheet_a: str
    sheet_b: str
    residual: tuple[float, float]
    magnitude: float
    projected: float | None
    tolerance: float
    at_boundary: bool
