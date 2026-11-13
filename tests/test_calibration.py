"""Tests for robust affine calibration from control points."""

import pytest

from affine import (
    AdmissionResult,
    Affine,
    AffineError,
    Bounds,
    Calibration,
    CalibrationConflictError,
    CandidateFailure,
    ControlPoint,
    ResidualFailure,
    admit_pixel,
    evaluate_pixel,
    find_calibrations,
)

TRUE_TRANSFORM = Affine(2.0, 0.3, 100.0, -0.2, -1.5, 500.0)
PIXEL_LOCATIONS = [
    (0.0, 0.0),
    (10.0, 4.0),
    (25.0, 13.0),
    (7.0, 31.0),
    (40.0, 20.0),
    (18.0, 9.0),
]


def make_point(index, col, row, *, east=None, north=None):
    true_east, true_north = TRUE_TRANSFORM @ (col, row)
    return ControlPoint(
        f"cp-{index}",
        col,
        row,
        true_east if east is None else east,
        true_north if north is None else north,
    )


def make_points():
    return [
        make_point(index, col, row) for index, (col, row) in enumerate(PIXEL_LOCATIONS)
    ]


def test_exact_points_recover_single_transform():
    calibrations = find_calibrations(make_points(), 1e-6, 1e-6)
    assert len(calibrations) == 1
    calibration = calibrations[0]
    assert calibration.discarded == ()
    assert calibration.transform.almost_equals(TRUE_TRANSFORM)
    assert len(calibration.residuals) == len(PIXEL_LOCATIONS)
    assert all(abs(r.east) < 1e-9 for r in calibration.residuals)
    assert all(abs(r.north) < 1e-9 for r in calibration.residuals)


def test_subset_fits_are_deduplicated():
    points = make_points()
    calibrations = find_calibrations(points, 1e-6, 1e-6)
    transforms = [c.transform for c in calibrations]
    assert len(transforms) == len(
        {tuple(round(v, 10) for v in t[:6]) for t in transforms}
    )


def test_one_wrong_east_record_is_discarded():
    points = make_points()
    bad = points[3]
    points[3] = ControlPoint(bad.point_id, bad.col, bad.row, bad.east + 50.0, bad.north)
    calibrations = find_calibrations(points, 1e-6, 1e-6)
    assert len(calibrations) == 1
    calibration = calibrations[0]
    assert calibration.discarded == ("cp-3",)
    assert calibration.transform.almost_equals(TRUE_TRANSFORM)
    assert all(r.point_id != "cp-3" for r in calibration.residuals)


def test_one_wrong_north_record_is_discarded():
    points = make_points()
    bad = points[2]
    points[2] = ControlPoint(bad.point_id, bad.col, bad.row, bad.east, bad.north - 42.0)
    calibrations = find_calibrations(points, 1e-6, 1e-6)
    assert len(calibrations) == 1
    assert calibrations[0].discarded == ("cp-2",)
    assert calibrations[0].transform.almost_equals(TRUE_TRANSFORM)


def test_missing_east_still_constrains_north():
    points = make_points()
    partial = points[4]
    points[4] = ControlPoint(
        partial.point_id, partial.col, partial.row, None, partial.north
    )
    calibrations = find_calibrations(points, 1e-6, 1e-6)
    assert len(calibrations) == 1
    calibration = calibrations[0]
    residual = next(r for r in calibration.residuals if r.point_id == "cp-4")
    assert residual.east is None
    assert abs(residual.north) < 1e-9
    assert calibration.transform.almost_equals(TRUE_TRANSFORM)


def test_missing_north_still_constrains_east():
    points = make_points()
    partial = points[0]
    points[0] = ControlPoint(
        partial.point_id, partial.col, partial.row, partial.east, None
    )
    calibrations = find_calibrations(points, 1e-6, 1e-6)
    assert len(calibrations) == 1
    calibration = calibrations[0]
    residual = next(r for r in calibration.residuals if r.point_id == "cp-0")
    assert residual.north is None
    assert abs(residual.east) < 1e-9
    assert calibration.transform.almost_equals(TRUE_TRANSFORM)


def test_fully_missing_record_is_ignored():
    points = make_points()
    points.append(ControlPoint("cp-missing", 99.0, 99.0, None, None))
    calibrations = find_calibrations(points, 1e-6, 1e-6)
    assert len(calibrations) == 1
    assert calibrations[0].transform.almost_equals(TRUE_TRANSFORM)
    assert all(r.point_id != "cp-missing" for r in calibrations[0].residuals)


def test_tuple_input_forms():
    points = make_points()
    tuple_points = [(p.point_id, p.col, p.row, p.east, p.north) for p in points]
    calibrations = find_calibrations(tuple_points, 1e-6, 1e-6)
    assert len(calibrations) == 1
    assert calibrations[0].transform.almost_equals(TRUE_TRANSFORM)

    short_points = [(p.col, p.row, p.east, p.north) for p in points]
    calibrations = find_calibrations(short_points, 1e-6, 1e-6)
    assert calibrations[0].residuals[0].point_id == 0


def test_residual_at_tolerance_boundary_accepted():
    points = make_points()[:4]
    bad = points[1]
    points[1] = ControlPoint(bad.point_id, bad.col, bad.row, bad.east + 0.5, bad.north)
    generous = find_calibrations(points, 10.0, 1e-6, max_outliers=0)[0]
    max_east_residual = max(abs(r.east) for r in generous.residuals)
    accepted = find_calibrations(points, max_east_residual, 1e-6, max_outliers=0)
    assert accepted[0].discarded == ()
    with pytest.raises(CalibrationConflictError):
        find_calibrations(points, max_east_residual - 1e-9, 1e-6, max_outliers=0)


def test_multiple_feasible_interpretations():
    points = make_points()
    shifted = points[3]
    points[3] = ControlPoint(
        shifted.point_id,
        shifted.col,
        shifted.row,
        shifted.east + 0.4,
        shifted.north,
    )
    calibrations = find_calibrations(points, 0.5, 0.5)
    discard_choices = {c.discarded for c in calibrations}
    assert () in discard_choices
    assert ("cp-3",) in discard_choices
    assert len(calibrations) >= 2
    transforms = [c.transform for c in calibrations]
    assert any(
        not t.almost_equals(transforms[0], precision=1e-10) for t in transforms[1:]
    )


def test_admission_confirmed_when_box_covers_all_interpretations():
    points = make_points()
    shifted = points[3]
    points[3] = ControlPoint(
        shifted.point_id,
        shifted.col,
        shifted.row,
        shifted.east + 0.4,
        shifted.north,
    )
    calibrations = find_calibrations(points, 0.5, 0.5)
    pixel = (12.0, 8.0)
    result = admit_pixel(calibrations, *pixel, Bounds(-1e9, -1e9, 1e9, 1e9))
    assert isinstance(result, AdmissionResult)
    assert result.admitted
    assert result.violations == ()
    easts = [c.transform @ pixel for c in calibrations]
    assert result.world_bounds.min_east == min(e for e, _ in easts)
    assert result.world_bounds.max_east == max(e for e, _ in easts)


def test_admission_rejected_reports_violating_calibration():
    points = make_points()
    shifted = points[3]
    points[3] = ControlPoint(
        shifted.point_id,
        shifted.col,
        shifted.row,
        shifted.east + 0.4,
        shifted.north,
    )
    calibrations = find_calibrations(points, 0.5, 0.5)
    pixel = (12.0, 8.0)
    reference = calibrations[0].transform @ pixel
    tight_box = Bounds(
        reference[0] - 1e-6,
        reference[1] - 1e-6,
        reference[0] + 1e-6,
        reference[1] + 1e-6,
    )
    result = admit_pixel(calibrations, *pixel, tight_box)
    assert not result.admitted
    assert result.violations
    violation = result.violations[0]
    expected_world = violation.calibration.transform @ pixel
    assert violation.world == expected_world
    assert violation.pixel == pixel
    assert set(violation.failed_components) & {"east", "north"}


def test_evaluate_pixel_end_to_end():
    points = make_points()
    pixel = (5.0, 5.0)
    east, north = TRUE_TRANSFORM @ pixel
    box = Bounds(east - 1.0, north - 1.0, east + 1.0, north + 1.0)
    result = evaluate_pixel(
        points, pixel, box, east_tolerance=1e-6, north_tolerance=1e-6
    )
    assert result.admitted
    assert result.world_bounds.min_east == pytest.approx(east, abs=1e-9)


def test_conflict_when_two_records_wrong():
    points = make_points()
    first = points[1]
    points[1] = ControlPoint(
        first.point_id, first.col, first.row, first.east + 50.0, first.north
    )
    second = points[4]
    points[4] = ControlPoint(
        second.point_id, second.col, second.row, second.east, second.north - 60.0
    )
    with pytest.raises(CalibrationConflictError) as exc_info:
        find_calibrations(points, 1e-6, 1e-6)
    error = exc_info.value
    assert isinstance(error, AffineError)
    assert error.failures
    assert all(isinstance(candidate, CandidateFailure) for candidate in error.failures)
    residual_failures = [
        failure for candidate in error.failures for failure in candidate.failures
    ]
    assert residual_failures
    assert all(isinstance(failure, ResidualFailure) for failure in residual_failures)
    assert error.conflicting_records == ("cp-1", "cp-4")
    message = str(error)
    assert "cp-1" in message
    assert "cp-4" in message


def test_underdetermined_hypothesis_reports_conflict():
    points = make_points()[:2]
    with pytest.raises(CalibrationConflictError) as exc_info:
        find_calibrations(points, 1e-6, 1e-6)
    error = exc_info.value
    assert all(candidate.underdetermined for candidate in error.failures)
    assert error.conflicting_records == ()
    assert "degenerate" in str(error)


def test_max_outliers_zero_forbids_discards():
    points = make_points()
    bad = points[3]
    points[3] = ControlPoint(bad.point_id, bad.col, bad.row, bad.east + 50.0, bad.north)
    with pytest.raises(CalibrationConflictError):
        find_calibrations(points, 1e-6, 1e-6, max_outliers=0)


def test_invalid_arguments():
    points = make_points()
    with pytest.raises(ValueError, match="tolerances must be non-negative"):
        find_calibrations(points, -1.0, 1.0)
    with pytest.raises(ValueError, match="tolerances must be non-negative"):
        find_calibrations(points, 1.0, -1.0)
    with pytest.raises(ValueError, match="max_outliers must be non-negative"):
        find_calibrations(points, 1.0, 1.0, max_outliers=-1)
    with pytest.raises(
        ValueError, match="max_outliers must leave at least one trusted record"
    ):
        find_calibrations(points, 1.0, 1.0, max_outliers=len(points))
    with pytest.raises(ValueError, match="at least one control point"):
        find_calibrations([], 1.0, 1.0)
    with pytest.raises(TypeError, match="control points must be"):
        find_calibrations([(1.0, 2.0, 3.0)], 1.0, 1.0)
    with pytest.raises(ValueError, match="minima must not exceed maxima"):
        Bounds(0.0, 0.0, -1.0, 1.0)
    with pytest.raises(ValueError, match="at least one calibration"):
        admit_pixel([], 0.0, 0.0, Bounds(0.0, 0.0, 1.0, 1.0))


def test_calibration_is_frozen_dataclass():
    from dataclasses import FrozenInstanceError

    calibration = find_calibrations(make_points(), 1e-6, 1e-6)[0]
    assert isinstance(calibration, Calibration)
    with pytest.raises(FrozenInstanceError):
        calibration.discarded = ("x",)


def test_conflict_failures_identify_components():
    points = make_points()
    bad = points[2]
    points[2] = ControlPoint(bad.point_id, bad.col, bad.row, bad.east, bad.north - 42.0)
    with pytest.raises(CalibrationConflictError) as exc_info:
        find_calibrations(points, 1e-6, 1e-6, max_outliers=0)
    no_discard_failure = next(
        candidate for candidate in exc_info.value.failures if candidate.discarded == ()
    )
    north_failures = [
        failure
        for failure in no_discard_failure.failures
        if failure.component == "north"
    ]
    target = next(failure for failure in north_failures if failure.point_id == "cp-2")
    assert target.tolerance == 1e-6
    assert abs(target.residual) > 1.0
    assert abs(target.residual) == max(
        abs(failure.residual) for failure in north_failures
    )


def test_conflict_consensus_pinpoints_single_bad_record():
    points = make_points()
    bad = points[3]
    points[3] = ControlPoint(bad.point_id, bad.col, bad.row, bad.east + 50.0, bad.north)
    with pytest.raises(CalibrationConflictError) as exc_info:
        find_calibrations(points, 1e-6, 1e-6, max_outliers=0)
    assert exc_info.value.conflicting_records == ("cp-3",)
