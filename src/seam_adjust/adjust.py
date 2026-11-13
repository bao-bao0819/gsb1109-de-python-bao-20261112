"""Public API for integer pixel-shift seam adjustment."""

from __future__ import annotations

from dataclasses import dataclass
import math

from affine import Affine

from .engine import InfeasibleAdjustmentError, ShiftSolver, _apply_linear
from .model import Sheet, Tie, TieResidual

__all__ = [
    "AdjustmentResult",
    "InfeasibleAdjustmentError",
    "Sheet",
    "Tie",
    "TieResidual",
    "solve",
]

_BOUNDARY_EPS = 1e-10


@dataclass(frozen=True)
class AdjustmentResult:
    """Outcome of a seam adjustment run.

    Parameters
    ----------
    shifts
        Chosen integer pixel shift ``(dx, dy)`` per sheet name.
    adjusted_transforms
        Corrected pixel-to-world transform per sheet.
    world_shifts
        World-coordinate displacement vector per sheet.
    changed_sheets
        Names of the sheets moved away from the zero shift.
    moved_count
        Number of changed sheets (the primary objective).
    total_world_displacement
        Sum of world displacement norms (the tie-breaking objective).
    uncertain
        Sheets whose shift is not uniquely determined, mapped to every
        shift they take among globally optimal solutions.
    residuals
        Evaluated residual of every seam observation.
    corrected_points
        Corrected world coordinates of each measured pixel centre, as
        ``(point_on_sheet_a, point_on_sheet_b)`` per observation.
    """

    shifts: dict[str, tuple[int, int]]
    adjusted_transforms: dict[str, Affine]
    world_shifts: dict[str, tuple[float, float]]
    changed_sheets: list[str]
    moved_count: int
    total_world_displacement: float
    uncertain: dict[str, set[tuple[int, int]]]
    residuals: list[TieResidual]
    corrected_points: dict[str, tuple[tuple[float, float], tuple[float, float]]]

    @property
    def boundary_ties(self) -> list[str]:
        """Names of observations whose residual sits exactly at tolerance."""
        return [residual.tie for residual in self.residuals if residual.at_boundary]

    def report(self) -> str:
        """Render a human-readable summary of the result."""
        changed = ", ".join(self.changed_sheets) or "none"
        lines = [
            f"changed sheets: {self.moved_count} -> {changed}",
            f"total world displacement: {self.total_world_displacement:.6g}",
            "pixel shifts:",
        ]
        for name, shift in self.shifts.items():
            flag = "  (uncertain)" if name in self.uncertain else ""
            lines.append(f"  {name}: dx={shift[0]:+d} dy={shift[1]:+d}{flag}")
        if self.uncertain:
            lines.append("undetermined sheets:")
            for name, values in self.uncertain.items():
                rendered = ", ".join(f"({dx:+d},{dy:+d})" for dx, dy in sorted(values))
                lines.append(f"  {name}: {rendered}")
        lines.append("seam residuals:")
        for residual in self.residuals:
            rx, ry = residual.residual
            tag = " BOUNDARY" if residual.at_boundary else ""
            if residual.projected is not None:
                lines.append(
                    f"  {residual.tie}: along-track={residual.projected:.6g}"
                    f" / tol={residual.tolerance:.6g}{tag}"
                )
            else:
                lines.append(
                    f"  {residual.tie}: residual=({rx:.6g}, {ry:.6g})"
                    f" |mag|={residual.magnitude:.6g}"
                    f" / tol={residual.tolerance:.6g}{tag}"
                )
        return "\n".join(lines)


def _boundary(value: float, tolerance: float) -> bool:
    scale = max(1.0, abs(tolerance), abs(value))
    return abs(abs(value) - tolerance) <= _BOUNDARY_EPS * scale


def _evaluate(
    solver: ShiftSolver,
    assignment: dict[int, tuple[int, int]],
) -> tuple[
    list[TieResidual],
    dict[str, tuple[tuple[float, float], tuple[float, float]]],
]:
    residuals: list[TieResidual] = []
    corrected: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for geometry in solver.geometries:
        tie = geometry.tie
        ia = solver.index[tie.sheet_a]
        ib = solver.index[tie.sheet_b]
        rx, ry = geometry.residual(assignment[ia], assignment[ib])
        magnitude = math.hypot(rx, ry)
        if geometry.direction is not None:
            projection = rx * geometry.direction[0] + ry * geometry.direction[1]
            at_boundary = _boundary(projection, tie.tolerance)
        else:
            projection = None
            if tie.radial:
                at_boundary = _boundary(magnitude, tie.tolerance)
            else:
                at_boundary = _boundary(rx, tie.tolerance) or _boundary(
                    ry, tie.tolerance
                )
        residuals.append(
            TieResidual(
                tie=tie.name,
                sheet_a=tie.sheet_a,
                sheet_b=tie.sheet_b,
                residual=(rx, ry),
                magnitude=magnitude,
                projected=projection,
                tolerance=tie.tolerance,
                at_boundary=at_boundary,
            )
        )
        point_a = solver.sheets[ia].transform @ tie.pixel_a
        point_b = solver.sheets[ib].transform @ tie.pixel_b
        delta_a = _apply_linear(solver.linear[ia], assignment[ia])
        delta_b = _apply_linear(solver.linear[ib], assignment[ib])
        corrected[tie.name] = (
            (point_a[0] + delta_a[0], point_a[1] + delta_a[1]),
            (point_b[0] + delta_b[0], point_b[1] + delta_b[1]),
        )
    return residuals, corrected


def solve(sheets: list[Sheet], ties: list[Tie]) -> AdjustmentResult:
    """Find the optimal integer pixel-shift adjustment.

    Parameters
    ----------
    sheets
        Sheets with calibrated linear transforms and finite shift ranges.
    ties
        Seam observations, full 2-D matches or along-track projections.

    Returns
    -------
    AdjustmentResult
        Optimal shifts, corrected transforms, uncertainty map, residual
        evaluation and boundary observations.

    Raises
    ------
    InfeasibleAdjustmentError
        When no shift combination inside the ranges satisfies all ties.
    ValueError
        When the input references unknown or duplicated identifiers.
    """
    tie_names = [tie.name for tie in ties]
    if len(set(tie_names)) != len(tie_names):
        raise ValueError("tie names must be unique")
    solver = ShiftSolver(sheets, ties)
    assignment, moved_count, total_cost = solver.solve_optimal()

    shifts = {sheet.name: assignment[i] for i, sheet in enumerate(sheets)}
    adjusted_transforms: dict[str, Affine] = {}
    world_shifts: dict[str, tuple[float, float]] = {}
    for i, sheet in enumerate(sheets):
        dx, dy = assignment[i]
        corrected_transform = sheet.transform @ Affine.translation(dx, dy)
        adjusted_transforms[sheet.name] = corrected_transform
        world_shifts[sheet.name] = (
            corrected_transform.c - sheet.transform.c,
            corrected_transform.f - sheet.transform.f,
        )

    uncertain: dict[str, set[tuple[int, int]]] = {}
    for i, sheet in enumerate(sheets):
        options = solver.possible_values(i, moved_count, total_cost)
        if len(options) > 1:
            uncertain[sheet.name] = options

    residuals, corrected_points = _evaluate(solver, assignment)
    changed = [sheet.name for i, sheet in enumerate(sheets) if assignment[i] != (0, 0)]
    return AdjustmentResult(
        shifts=shifts,
        adjusted_transforms=adjusted_transforms,
        world_shifts=world_shifts,
        changed_sheets=changed,
        moved_count=moved_count,
        total_world_displacement=total_cost,
        uncertain=uncertain,
        residuals=residuals,
        corrected_points=corrected_points,
    )
