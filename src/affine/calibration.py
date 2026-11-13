"""Robust affine calibration from ground control points.

Control point records pair pixel coordinates (column, row) with field
world coordinates (east, north). When importing an old map, at most one
record is allowed to be wrong, and either world component of a record may
be missing. This module enumerates every two dimensional affine
calibration that can explain the trustworthy records, accepting residuals
component by component against explicit tolerances, and decides whether a
pixel may be admitted into the database.

A calibration maps pixel coordinates to world coordinates.

.. code-block:: none

    | east  |   | a  b  c | | col |
    | north | = | d  e  f | | row |
    | 1     |   | 0  0  1 | |  1  |

The east and north subsystems share the same design rows
``(col, row, 1)`` but have independent coefficients, so a record whose
east coordinate is missing still constrains the north subsystem and vice
versa.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

from . import Affine, AffineError

__all__ = [
    "AdmissionResult",
    "AdmissionViolation",
    "Bounds",
    "Calibration",
    "CalibrationConflictError",
    "CandidateFailure",
    "ControlPoint",
    "PointResidual",
    "ResidualFailure",
    "admit_pixel",
    "evaluate_pixel",
    "find_calibrations",
]

_SINGULAR_EPS = 1e-12
_COEFF_EPS = 1e-9


class CalibrationConflictError(AffineError):
    """No calibration can explain the records under the residual bounds.

    Parameters
    ----------
    failures : tuple of CandidateFailure
        Residual failures observed for every discarded-record hypothesis
        that was tried.
    conflicting_records : tuple, optional
        Identifiers of the records judged incompatible with the strongest
        component-wise consensus of the other records. When the geometry
        is too degenerate to establish a consensus, this falls back to
        every record appearing in ``failures``.

    Attributes
    ----------
    failures : tuple of CandidateFailure
        Structured diagnostics identifying the records that conflict with
        each hypothesis.
    conflicting_records : tuple
        Identifiers of the records most likely responsible for the
        conflict.
    """

    def __init__(
        self,
        failures: Sequence[CandidateFailure],
        conflicting_records: Sequence[object] | None = None,
    ) -> None:
        self.failures: tuple[CandidateFailure, ...] = tuple(failures)
        if conflicting_records is None:
            conflicting_records = []
            for candidate in self.failures:
                for failure in candidate.failures:
                    if failure.point_id not in conflicting_records:
                        conflicting_records.append(failure.point_id)
        self.conflicting_records: tuple[object, ...] = tuple(conflicting_records)
        if self.conflicting_records:
            ids = ", ".join(repr(point_id) for point_id in self.conflicting_records)
            detail = f"conflicting records: {ids}"
        else:
            detail = "the trusted records are insufficient or geometrically degenerate"
        super().__init__(
            "no affine calibration satisfies the residual bounds; " + detail
        )


@dataclass(frozen=True)
class ControlPoint:
    """A single (column, row) to (east, north) control record.

    Parameters
    ----------
    point_id : object
        Identifier of the record, used in diagnostics.
    col, row : float
        Pixel column and row.
    east : float or None
        Field east coordinate, or None when the component is missing.
    north : float or None
        Field north coordinate, or None when the component is missing.
    """

    point_id: object
    col: float
    row: float
    east: float | None
    north: float | None


@dataclass(frozen=True)
class PointResidual:
    """Signed calibration residuals for one trusted control point.

    Parameters
    ----------
    point_id : object
        Identifier of the control record.
    east : float or None
        Predicted minus observed east coordinate, or None when missing.
    north : float or None
        Predicted minus observed north coordinate, or None when missing.
    """

    point_id: object
    east: float | None
    north: float | None


@dataclass(frozen=True)
class Calibration:
    """A feasible affine calibration and its acceptance evidence.

    Parameters
    ----------
    transform : Affine
        Affine transform mapping pixel coordinates to world coordinates.
    discarded : tuple
        Identifiers of the records treated as wrong. Empty when no record
        is discarded.
    residuals : tuple of PointResidual
        Signed residuals of every trusted record, component by component.
    """

    transform: Affine
    discarded: tuple[object, ...]
    residuals: tuple[PointResidual, ...]


@dataclass(frozen=True)
class ResidualFailure:
    """A residual that violates its component tolerance.

    Parameters
    ----------
    point_id : object
        Identifier of the offending control record.
    component : str
        ``"east"`` or ``"north"``.
    residual : float
        Signed residual, predicted minus observed coordinate.
    tolerance : float
        Residual bound that was exceeded.
    """

    point_id: object
    component: str
    residual: float
    tolerance: float


@dataclass(frozen=True)
class CandidateFailure:
    """Failures observed when testing one discarded-record hypothesis.

    Parameters
    ----------
    discarded : tuple
        Identifiers of the records discarded by the hypothesis.
    failures : tuple of ResidualFailure
        Residual violations among the trusted records.
    underdetermined : bool
        True when the trusted records do not constrain both coordinate
        subsystems enough to solve for all six coefficients.
    """

    discarded: tuple[object, ...]
    failures: tuple[ResidualFailure, ...]
    underdetermined: bool


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned acceptance box in world coordinates.

    Parameters
    ----------
    min_east, min_north : float
        Lower inclusive bounds.
    max_east, max_north : float
        Upper inclusive bounds.
    """

    min_east: float
    min_north: float
    max_east: float
    max_north: float

    def __post_init__(self) -> None:
        if self.min_east > self.max_east or self.min_north > self.max_north:
            raise ValueError("bounds minima must not exceed maxima")


@dataclass(frozen=True)
class AdmissionViolation:
    """A calibration that maps the candidate pixel outside the box.

    Parameters
    ----------
    calibration : Calibration
        Feasible calibration producing the out-of-box coordinates.
    pixel : tuple of float
        Candidate pixel (column, row).
    world : tuple of float
        World coordinates (east, north) computed for the pixel.
    failed_components : tuple of str
        Components whose acceptance bound was violated.
    """

    calibration: Calibration
    pixel: tuple[float, float]
    world: tuple[float, float]
    failed_components: tuple[str, ...]


@dataclass(frozen=True)
class AdmissionResult:
    """Outcome of admitting a pixel under every feasible calibration.

    Parameters
    ----------
    admitted : bool
        True only when every feasible calibration maps the pixel inside
        the acceptance box.
    pixel : tuple of float
        Candidate pixel (column, row).
    world_bounds : Bounds
        Tightest box enclosing the pixel's world coordinates under every
        feasible calibration.
    violations : tuple of AdmissionViolation
        Calibrations and pixel positions that leave the acceptance box.
    """

    admitted: bool
    pixel: tuple[float, float]
    world_bounds: Bounds
    violations: tuple[AdmissionViolation, ...]


class _SingularSystemError(Exception):
    """Raised internally when a normal equation system has no unique solve."""


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)  # type: ignore[arg-type]


def _normalize_points(
    points: Iterable[ControlPoint | Sequence[float]],
) -> tuple[ControlPoint, ...]:
    normalized: list[ControlPoint] = []
    for index, point in enumerate(points):
        if isinstance(point, ControlPoint):
            normalized.append(point)
            continue
        record = tuple(point)
        if len(record) == 4:
            col, row, east, north = record
            point_id: object = index
        elif len(record) == 5:
            point_id, col, row, east, north = record
        else:
            raise TypeError(
                "control points must be ControlPoint instances, 4-tuples "
                "(col, row, east, north) or 5-tuples "
                "(id, col, row, east, north); use None for missing values"
            )
        normalized.append(
            ControlPoint(
                point_id,
                float(col),
                float(row),
                _optional_float(east),
                _optional_float(north),
            )
        )
    if not normalized:
        raise ValueError("at least one control point is required")
    return tuple(normalized)


def _solve_3x3(
    matrix: Sequence[Sequence[float]], rhs: Sequence[float]
) -> tuple[float, float, float]:
    """Solve a 3x3 linear system with partial pivoting.

    Raises
    ------
    _SingularSystemError
        When the system does not have a unique solution.
    """
    system = [[*matrix[row], rhs[row]] for row in range(3)]
    scale = max(abs(value) for row in system for value in row[:3])
    for pivot_col in range(3):
        pivot_row = max(
            range(pivot_col, 3),
            key=lambda row: abs(system[row][pivot_col]),
        )
        pivot = system[pivot_row][pivot_col]
        if abs(pivot) <= _SINGULAR_EPS * max(1.0, scale):
            raise _SingularSystemError
        system[pivot_col], system[pivot_row] = system[pivot_row], system[pivot_col]
        pivot_row = pivot_col
        for row in range(pivot_col + 1, 3):
            factor = system[row][pivot_col] / system[pivot_row][pivot_col]
            if factor == 0.0:
                continue
            for col in range(pivot_col, 4):
                system[row][col] -= factor * system[pivot_row][col]
    solution = [0.0, 0.0, 0.0]
    for row in range(2, -1, -1):
        remainder = system[row][3] - sum(
            system[row][col] * solution[col] for col in range(row + 1, 3)
        )
        solution[row] = remainder / system[row][row]
    return solution[0], solution[1], solution[2]


def _normal_equations(
    points: Sequence[ControlPoint],
    weights: Sequence[float],
    component: str,
) -> tuple[tuple[list[float], ...], list[float], int]:
    """Build weighted normal equations for one world coordinate component."""
    matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
    rhs = [0.0, 0.0, 0.0]
    observation_count = 0
    for point, weight in zip(points, weights):
        if weight == 0.0:
            continue
        observed = getattr(point, component)
        if observed is None:
            continue
        observation_count += 1
        design = (point.col, point.row, 1.0)
        for i in range(3):
            rhs[i] += weight * design[i] * observed
            for j in range(3):
                matrix[i][j] += weight * design[i] * design[j]
    return tuple(matrix), rhs, observation_count


def _same_calibration(first: Affine, second: Affine) -> bool:
    return all(
        abs(left - right) <= _COEFF_EPS * max(1.0, abs(left), abs(right))
        for left, right in zip(first[:6], second[:6])
    )


def _component_support(
    points: Sequence[ControlPoint],
    component: str,
    tolerance: float,
) -> tuple[frozenset[int], frozenset[int]]:
    """Return the best consensus and its unsupported records for a component.

    Exact affine coefficients are interpolated from every non-collinear
    triple of records observing the component. The triple explaining the
    most records within ``tolerance`` defines the consensus.
    """
    observed = [
        index
        for index, point in enumerate(points)
        if getattr(point, component) is not None
    ]
    best_support: frozenset[int] = frozenset()
    for triple in combinations(observed, 3):
        matrix = [[points[i].col, points[i].row, 1.0] for i in triple]
        rhs = [getattr(points[i], component) for i in triple]
        try:
            coeffs = _solve_3x3(matrix, rhs)
        except _SingularSystemError:
            continue
        support = set(triple)
        for index in observed:
            if index in support:
                continue
            point = points[index]
            predicted = coeffs[0] * point.col + coeffs[1] * point.row + coeffs[2]
            if abs(predicted - getattr(point, component)) <= tolerance:
                support.add(index)
        if len(support) > len(best_support):
            best_support = frozenset(support)
    observed_set = frozenset(observed)
    return best_support, observed_set - best_support


def _conflicting_records(
    points: Sequence[ControlPoint],
    east_tolerance: float,
    north_tolerance: float,
) -> tuple[object, ...]:
    """Identify records incompatible with the strongest component consensus."""
    conflict_indices: set[int] = set()
    for component, tolerance in (
        ("east", east_tolerance),
        ("north", north_tolerance),
    ):
        support, unsupported = _component_support(points, component, tolerance)
        if not support:
            continue
        conflict_indices.update(unsupported)
    return tuple(points[index].point_id for index in sorted(conflict_indices))


def _evaluate_hypothesis(
    points: Sequence[ControlPoint],
    discarded: tuple[int, ...],
    east_tolerance: float,
    north_tolerance: float,
) -> Calibration | CandidateFailure:
    weights = [0.0 if index in discarded else 1.0 for index in range(len(points))]
    discarded_ids = tuple(points[index].point_id for index in discarded)

    try:
        east_matrix, east_rhs, east_count = _normal_equations(points, weights, "east")
        north_matrix, north_rhs, north_count = _normal_equations(
            points, weights, "north"
        )
        if east_count == 0 or north_count == 0:
            raise _SingularSystemError
        east_coeffs = _solve_3x3(east_matrix, east_rhs)
        north_coeffs = _solve_3x3(north_matrix, north_rhs)
    except _SingularSystemError:
        return CandidateFailure(discarded_ids, (), True)

    transform = Affine(*east_coeffs, *north_coeffs)
    failures: list[ResidualFailure] = []
    residuals: list[PointResidual] = []
    for point, weight in zip(points, weights):
        if weight == 0.0:
            continue
        predicted_east, predicted_north = transform @ (point.col, point.row)
        east_residual: float | None = None
        north_residual: float | None = None
        if point.east is not None:
            east_residual = predicted_east - point.east
            if abs(east_residual) > east_tolerance:
                failures.append(
                    ResidualFailure(
                        point.point_id, "east", east_residual, east_tolerance
                    )
                )
        if point.north is not None:
            north_residual = predicted_north - point.north
            if abs(north_residual) > north_tolerance:
                failures.append(
                    ResidualFailure(
                        point.point_id, "north", north_residual, north_tolerance
                    )
                )
        if east_residual is not None or north_residual is not None:
            residuals.append(
                PointResidual(point.point_id, east_residual, north_residual)
            )
    if failures:
        return CandidateFailure(discarded_ids, tuple(failures), False)
    return Calibration(transform, discarded_ids, tuple(residuals))


def find_calibrations(
    points: Iterable[ControlPoint | Sequence[float]],
    east_tolerance: float,
    north_tolerance: float,
    *,
    max_outliers: int = 1,
) -> tuple[Calibration, ...]:
    """Find every affine calibration explaining the control records.

    Each hypothesis trusts all records except a discarded subset of at
    most ``max_outliers`` records. The six affine coefficients are fitted
    independently for the east and north subsystems by weighted least
    squares on the normal equations, so records missing one component
    still constrain the other one. A hypothesis is feasible only when
    every trusted, observed component residual stays within its
    tolerance. Calibrations equal up to floating point precision are
    reported once.

    Parameters
    ----------
    points : iterable
        ``ControlPoint`` instances, 4-tuples
        ``(col, row, east, north)`` or 5-tuples
        ``(id, col, row, east, north)``. Use None for a missing
        component. Tuple records without an id use their position as id.
    east_tolerance : float
        Maximum accepted absolute east residual. Must be non-negative.
    north_tolerance : float
        Maximum accepted absolute north residual. Must be non-negative.
    max_outliers : int, default 1
        Maximum number of records that may be treated as wrong.

    Returns
    -------
    tuple of Calibration
        All distinct feasible calibrations, ordered from the hypothesis
        discarding no record to hypotheses discarding records.

    Raises
    ------
    ValueError
        For invalid tolerances, outlier counts or empty input.
    CalibrationConflictError
        When no hypothesis yields a feasible calibration. The exception
        carries the residual failures for every hypothesis tried.
    """
    if east_tolerance < 0.0 or north_tolerance < 0.0:
        raise ValueError("residual tolerances must be non-negative")
    if max_outliers < 0:
        raise ValueError("max_outliers must be non-negative")
    normalized = _normalize_points(points)
    if max_outliers >= len(normalized):
        raise ValueError("max_outliers must leave at least one trusted record")

    hypotheses: list[tuple[int, ...]] = [()]
    for count in range(1, max_outliers + 1):
        hypotheses.extend(combinations(range(len(normalized)), count))

    feasible: list[Calibration] = []
    candidate_failures: list[CandidateFailure] = []
    for hypothesis in hypotheses:
        outcome = _evaluate_hypothesis(
            normalized, hypothesis, east_tolerance, north_tolerance
        )
        if isinstance(outcome, Calibration):
            if not any(
                _same_calibration(outcome.transform, existing.transform)
                for existing in feasible
            ):
                feasible.append(outcome)
        else:
            candidate_failures.append(outcome)

    if not feasible:
        conflicts = _conflicting_records(normalized, east_tolerance, north_tolerance)
        raise CalibrationConflictError(tuple(candidate_failures), conflicts)
    return tuple(feasible)


def admit_pixel(
    calibrations: Sequence[Calibration],
    col: float,
    row: float,
    acceptance_bounds: Bounds,
) -> AdmissionResult:
    """Check a pixel's world coordinates against every feasible calibration.

    The pixel is admitted only when its east and north coordinates fall
    inside the acceptance box under all feasible calibrations. When at
    least one calibration maps the pixel outside the box, the violating
    calibration and pixel position are returned instead.

    Parameters
    ----------
    calibrations : sequence of Calibration
        Feasible calibrations returned by :func:`find_calibrations`.
    col, row : float
        Candidate pixel coordinates.
    acceptance_bounds : Bounds
        Inclusive acceptance box in world coordinates.

    Returns
    -------
    AdmissionResult
        Admission decision with the observed world bounds and any
        violations.

    Raises
    ------
    ValueError
        When no calibration is supplied.
    """
    if not calibrations:
        raise ValueError("at least one calibration is required")
    pixel = (float(col), float(row))
    violations: list[AdmissionViolation] = []
    easts: list[float] = []
    norths: list[float] = []
    for calibration in calibrations:
        east, north = calibration.transform @ pixel
        easts.append(east)
        norths.append(north)
        failed: list[str] = []
        if not acceptance_bounds.min_east <= east <= acceptance_bounds.max_east:
            failed.append("east")
        if not acceptance_bounds.min_north <= north <= acceptance_bounds.max_north:
            failed.append("north")
        if failed:
            violations.append(
                AdmissionViolation(calibration, pixel, (east, north), tuple(failed))
            )
    observed = Bounds(min(easts), min(norths), max(easts), max(norths))
    return AdmissionResult(not violations, pixel, observed, tuple(violations))


def evaluate_pixel(
    points: Iterable[ControlPoint | Sequence[float]],
    pixel: Sequence[float],
    acceptance_bounds: Bounds,
    *,
    east_tolerance: float,
    north_tolerance: float,
    max_outliers: int = 1,
) -> AdmissionResult:
    """Calibrate from control points and admit a pixel in one step.

    Parameters
    ----------
    points : iterable
        Control records, see :func:`find_calibrations`.
    pixel : sequence of float
        Candidate pixel ``(col, row)``.
    acceptance_bounds : Bounds
        Inclusive acceptance box in world coordinates.
    east_tolerance : float
        Maximum accepted absolute east residual.
    north_tolerance : float
        Maximum accepted absolute north residual.
    max_outliers : int, default 1
        Maximum number of records that may be treated as wrong.

    Returns
    -------
    AdmissionResult
        Admission decision across every feasible calibration.

    Raises
    ------
    CalibrationConflictError
        When no hypothesis yields a feasible calibration.
    """
    col, row = pixel
    calibrations = find_calibrations(
        points,
        east_tolerance,
        north_tolerance,
        max_outliers=max_outliers,
    )
    return admit_pixel(calibrations, col, row, acceptance_bounds)
