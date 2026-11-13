"""Tests for the integer pixel-shift seam adjustment solver."""

from __future__ import annotations

import math

import pytest

from affine import Affine
from seam_adjust import (
    InfeasibleAdjustmentError,
    Sheet,
    Tie,
    solve,
)


def north_up(pixel_size, x0=0.0, y0=0.0):
    return Affine(pixel_size, 0.0, x0, 0.0, -pixel_size, y0)


def test_zero_shift_is_preferred_when_ties_already_pass():
    sheets = [
        Sheet("S1", north_up(10.0), (-2, 2), (-2, 2)),
        Sheet("S2", north_up(10.0, x0=100.0), (-2, 2), (-2, 2)),
    ]
    ties = [
        Tie(
            "t1",
            "S1",
            "S2",
            (10.0, 0.0),
            (0.0, 0.0),
            tolerance=0.5,
        )
    ]
    result = solve(sheets, ties)
    assert result.moved_count == 0
    assert result.shifts == {"S1": (0, 0), "S2": (0, 0)}
    assert result.changed_sheets == []
    assert result.uncertain == {}
    assert result.total_world_displacement == pytest.approx(0.0)


def test_cheapest_world_displacement_wins_tie_break():
    sheets = [
        Sheet("S1", north_up(10.0), (-1, 1), (-1, 1)),
        Sheet("S2", north_up(20.0, x0=15.0), (-1, 1), (-1, 1)),
    ]
    ties = [Tie("t1", "S1", "S2", (0.0, 0.0), (0.0, 0.0), tolerance=5.0)]
    result = solve(sheets, ties)
    assert result.moved_count == 1
    assert result.shifts["S1"] == (1, 0)
    assert result.shifts["S2"] == (0, 0)
    assert result.total_world_displacement == pytest.approx(10.0)
    residual = result.residuals[0]
    assert residual.residual[0] == pytest.approx(-5.0)
    assert residual.at_boundary is True
    assert result.boundary_ties == ["t1"]


def test_equal_cost_alternatives_are_flagged_uncertain():
    sheets = [
        Sheet("S1", north_up(10.0), (-2, 2), (-2, 2)),
        Sheet("S2", north_up(10.0, x0=10.0), (-2, 2), (-2, 2)),
    ]
    ties = [Tie("t1", "S1", "S2", (0.0, 0.0), (0.0, 0.0), tolerance=0.0)]
    result = solve(sheets, ties)
    assert result.moved_count == 1
    assert result.total_world_displacement == pytest.approx(10.0)
    assert result.uncertain == {
        "S1": {(0, 0), (1, 0)},
        "S2": {(0, 0), (-1, 0)},
    }
    chosen = result.shifts
    assert chosen in (
        {"S1": (1, 0), "S2": (0, 0)},
        {"S1": (0, 0), "S2": (-1, 0)},
    )
    assert abs(result.residuals[0].residual[0]) <= 1e-9


def test_along_track_projection_ignores_across_track_gap():
    sheets = [
        Sheet("S1", north_up(10.0), (-2, 2), (-2, 2)),
        Sheet("S2", north_up(10.0, x0=12.0, y0=30.0), (-2, 2), (-2, 2)),
    ]
    ties = [
        Tie(
            "t1",
            "S1",
            "S2",
            (0.0, 0.0),
            (0.0, 0.0),
            tolerance=2.0,
            along_track=(2.0, 0.0),
        )
    ]
    result = solve(sheets, ties)
    assert result.moved_count == 1
    residual = result.residuals[0]
    assert abs(residual.projected) == pytest.approx(2.0)
    assert residual.residual[1] == pytest.approx(-30.0)
    assert residual.at_boundary is True
    assert result.boundary_ties == ["t1"]
    assert set(result.uncertain) == {"S1", "S2"}


def test_projected_tie_inside_tolerance_is_not_boundary():
    sheets = [
        Sheet("S1", north_up(10.0), (-2, 2), (-2, 2)),
        Sheet("S2", north_up(10.0, x0=10.0, y0=30.0), (-2, 2), (-2, 2)),
    ]
    ties = [
        Tie(
            "t1",
            "S1",
            "S2",
            (0.0, 0.0),
            (0.0, 0.0),
            tolerance=2.0,
            along_track=(1.0, 0.0),
        )
    ]
    result = solve(sheets, ties)
    assert result.residuals[0].projected == pytest.approx(0.0)
    assert result.residuals[0].at_boundary is False


def test_radial_and_box_tolerance_differ():
    sheets = [
        Sheet("S1", north_up(10.0), (-1, 1), (0, 0)),
        Sheet("S2", north_up(10.0, x0=12.0, y0=-12.0), (-1, 1), (0, 0)),
    ]
    common = dict(
        name="t1",
        sheet_a="S1",
        sheet_b="S2",
        pixel_a=(0.0, 0.0),
        pixel_b=(0.0, 0.0),
    )
    boxed = solve(sheets, [Tie(**common, tolerance=12.0)])
    assert boxed.moved_count == 0
    assert boxed.residuals[0].residual == pytest.approx((-12.0, 12.0))
    assert boxed.residuals[0].at_boundary is True

    radial_tol = math.hypot(2.0, 12.0)
    radial = solve(sheets, [Tie(**common, tolerance=radial_tol, radial=True)])
    assert radial.moved_count == 1
    assert radial.residuals[0].magnitude == pytest.approx(radial_tol)
    assert radial.residuals[0].at_boundary is True

    with pytest.raises(InfeasibleAdjustmentError):
        solve(sheets, [Tie(**common, tolerance=12.0, radial=True)])


def test_infeasible_when_gap_exceeds_ranges():
    sheets = [
        Sheet("S1", north_up(10.0), (-1, 1), (-1, 1)),
        Sheet("S2", north_up(10.0, x0=100.0), (-1, 1), (-1, 1)),
    ]
    ties = [Tie("t1", "S1", "S2", (0.0, 0.0), (0.0, 0.0), tolerance=0.0)]
    with pytest.raises(InfeasibleAdjustmentError):
        solve(sheets, ties)


def test_rotated_raster_uses_linear_part_for_world_shift():
    rotated = Affine(0.0, -10.0, 0.0, 10.0, 0.0, 0.0)
    sheets = [
        Sheet("S1", rotated, (-1, 1), (-1, 1)),
        Sheet("S2", north_up(10.0, y0=-10.0), (-1, 1), (0, 0)),
    ]
    ties = [Tie("t1", "S1", "S2", (0.0, 0.0), (0.0, 0.0), tolerance=0.0)]
    result = solve(sheets, ties)
    assert result.shifts["S1"] == (-1, 0)
    assert result.shifts["S2"] == (0, 0)
    assert result.world_shifts["S1"] == pytest.approx((0.0, -10.0))
    point_a, point_b = result.corrected_points["t1"]
    assert point_a == pytest.approx((0.0, -10.0))
    assert point_b == pytest.approx((0.0, -10.0))


def test_adjusted_transform_matches_corrected_points():
    sheets = [
        Sheet("S1", north_up(10.0), (-2, 2), (-2, 2)),
        Sheet("S2", north_up(10.0, x0=22.0), (-2, 2), (-2, 2)),
    ]
    ties = [Tie("t1", "S1", "S2", (3.0, 1.0), (2.0, 1.0), tolerance=3.0)]
    result = solve(sheets, ties)
    for sheet in sheets:
        dx, dy = result.shifts[sheet.name]
        expected = sheet.transform @ Affine.translation(dx, dy)
        assert result.adjusted_transforms[sheet.name].almost_equals(expected)
    point_a, point_b = result.corrected_points["t1"]
    assert point_a == pytest.approx(result.adjusted_transforms["S1"] @ ties[0].pixel_a)
    assert point_b == pytest.approx(result.adjusted_transforms["S2"] @ ties[0].pixel_b)


def test_chain_prefers_moving_the_single_bad_sheet():
    sheets = [
        Sheet("S1", north_up(10.0), (-2, 2), (-2, 2)),
        Sheet("S2", north_up(10.0, x0=112.0), (-2, 2), (-2, 2)),
        Sheet("S3", north_up(10.0, x0=200.0), (-2, 2), (-2, 2)),
    ]
    ties = [
        Tie("t1", "S1", "S2", (10.0, 0.0), (0.0, 0.0), tolerance=5.0),
        Tie("t2", "S2", "S3", (10.0, 0.0), (0.0, 0.0), tolerance=5.0),
    ]
    result = solve(sheets, ties)
    assert result.moved_count == 1
    assert result.changed_sheets == ["S2"]
    assert result.shifts["S2"] == (-1, 0)
    assert result.uncertain == {}
    for residual in result.residuals:
        assert abs(residual.residual[0]) == pytest.approx(2.0)
        assert residual.at_boundary is False


def test_validation_errors():
    good = Sheet("S1", north_up(10.0), (-1, 1), (-1, 1))
    with pytest.raises(ValueError, match="zero shift"):
        Sheet("bad", north_up(10.0), (1, 2), (-1, 1))
    with pytest.raises(ValueError, match="distinct sheets"):
        Tie("t", "S1", "S1", (0.0, 0.0), (0.0, 0.0), tolerance=1.0)
    with pytest.raises(ValueError, match="non-zero"):
        Tie(
            "t",
            "S1",
            "S2",
            (0.0, 0.0),
            (0.0, 0.0),
            tolerance=1.0,
            along_track=(0.0, 0.0),
        )
    with pytest.raises(ValueError, match="unique"):
        solve([good, good], [])
    with pytest.raises(ValueError, match="unknown sheet"):
        solve(
            [good],
            [Tie("t", "S1", "ghost", (0.0, 0.0), (0.0, 0.0), tolerance=1.0)],
        )
    with pytest.raises(ValueError, match="tie names must be unique"):
        solve(
            [good, Sheet("S2", north_up(10.0), (-1, 1), (-1, 1))],
            [
                Tie("t", "S1", "S2", (0.0, 0.0), (0.0, 0.0), tolerance=1.0),
                Tie("t", "S1", "S2", (1.0, 0.0), (1.0, 0.0), tolerance=1.0),
            ],
        )


def test_report_contains_key_sections():
    sheets = [
        Sheet("S1", north_up(10.0), (-1, 1), (-1, 1)),
        Sheet("S2", north_up(10.0, x0=10.0), (-1, 1), (-1, 1)),
    ]
    ties = [Tie("t1", "S1", "S2", (0.0, 0.0), (0.0, 0.0), tolerance=0.0)]
    text = solve(sheets, ties).report()
    assert "changed sheets: 1" in text
    assert "undetermined sheets:" in text
    assert "seam residuals:" in text
    assert "t1" in text
