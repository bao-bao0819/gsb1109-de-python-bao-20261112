"""Randomised cross-checks against exhaustive enumeration."""

from __future__ import annotations

import itertools
import math
import random

import pytest

from affine import Affine
from seam_adjust import InfeasibleAdjustmentError, Sheet, Tie, solve

_LINEAR_PARTS = [
    ((5.0, 0.0), (0.0, -5.0)),
    ((10.0, 0.0), (0.0, -10.0)),
    ((0.0, -10.0), (10.0, 0.0)),
    ((7.0, 0.3), (-0.3, 7.0)),
]
_TOLERANCES = [0.0, 2.5, 5.0, 7.5, 10.0]


def _world(transform, pixel):
    return transform @ pixel


def _linear(transform):
    return (transform.a, transform.b), (transform.d, transform.e)


def _apply(linear, shift):
    (a, b), (d, e) = linear
    dx, dy = shift
    return a * dx + b * dy, d * dx + e * dy


def _is_feasible(sheets, ties, assignment):
    for tie in ties:
        sheet_a = next(s for s in sheets if s.name == tie.sheet_a)
        sheet_b = next(s for s in sheets if s.name == tie.sheet_b)
        base_a = _world(sheet_a.transform, tie.pixel_a)
        base_b = _world(sheet_b.transform, tie.pixel_b)
        da = _apply(_linear(sheet_a.transform), assignment[tie.sheet_a])
        db = _apply(_linear(sheet_b.transform), assignment[tie.sheet_b])
        rx = (base_a[0] + da[0]) - (base_b[0] + db[0])
        ry = (base_a[1] + da[1]) - (base_b[1] + db[1])
        if tie.along_track is not None:
            projection = rx * tie.along_track[0] + ry * tie.along_track[1]
            if abs(projection) > tie.tolerance + 1e-9:
                return False
        elif tie.radial:
            if math.hypot(rx, ry) > tie.tolerance + 1e-9:
                return False
        elif abs(rx) > tie.tolerance + 1e-9 or abs(ry) > tie.tolerance + 1e-9:
            return False
    return True


def _expected_boundary(sheets, ties, assignment):
    names = []
    for tie in ties:
        sheet_a = next(s for s in sheets if s.name == tie.sheet_a)
        sheet_b = next(s for s in sheets if s.name == tie.sheet_b)
        base_a = _world(sheet_a.transform, tie.pixel_a)
        base_b = _world(sheet_b.transform, tie.pixel_b)
        da = _apply(_linear(sheet_a.transform), assignment[tie.sheet_a])
        db = _apply(_linear(sheet_b.transform), assignment[tie.sheet_b])
        rx = (base_a[0] + da[0]) - (base_b[0] + db[0])
        ry = (base_a[1] + da[1]) - (base_b[1] + db[1])
        if tie.along_track is not None:
            value = abs(rx * tie.along_track[0] + ry * tie.along_track[1])
        elif tie.radial:
            value = math.hypot(rx, ry)
        else:
            if abs(abs(rx) - tie.tolerance) <= 1e-9 * max(1.0, tie.tolerance):
                names.append(tie.name)
                continue
            value = abs(ry)
        if abs(value - tie.tolerance) <= 1e-9 * max(1.0, tie.tolerance, value):
            names.append(tie.name)
    return names


def _random_problem(rng):
    count = rng.randint(2, 4)
    sheets = []
    for i in range(count):
        (a, b), (d, e) = rng.choice(_LINEAR_PARTS)
        transform = Affine(a, b, rng.uniform(-50, 50), d, e, rng.uniform(-50, 50))
        sheets.append(Sheet(f"S{i}", transform, (-1, 1), (-1, 1)))
    ties = []
    for k in range(rng.randint(2, 6)):
        ia, ib = rng.sample(range(count), 2)
        projected = rng.random() < 0.5
        angle = rng.uniform(0.0, 2.0 * math.pi)
        ties.append(
            Tie(
                name=f"t{k}",
                sheet_a=f"S{ia}",
                sheet_b=f"S{ib}",
                pixel_a=(rng.uniform(0, 20), rng.uniform(0, 20)),
                pixel_b=(rng.uniform(0, 20), rng.uniform(0, 20)),
                tolerance=rng.choice(_TOLERANCES),
                along_track=(math.cos(angle), math.sin(angle)) if projected else None,
                radial=False if projected else rng.random() < 0.4,
            )
        )
    return sheets, ties


@pytest.mark.parametrize("seed", range(40))
def test_solver_matches_brute_force(seed):
    rng = random.Random(20261113 + seed)
    sheets, ties = _random_problem(rng)
    domain = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)]
    names = [sheet.name for sheet in sheets]
    feasible = []
    for combo in itertools.product(domain, repeat=len(names)):
        assignment = dict(zip(names, combo))
        if _is_feasible(sheets, ties, assignment):
            moved = sum(shift != (0, 0) for shift in assignment.values())
            cost = sum(
                math.hypot(*_apply(_linear(sheet.transform), assignment[sheet.name]))
                for sheet in sheets
            )
            feasible.append((moved, cost, assignment))

    if not feasible:
        with pytest.raises(InfeasibleAdjustmentError):
            solve(sheets, ties)
        return

    min_moved = min(item[0] for item in feasible)
    best = [item for item in feasible if item[0] == min_moved]
    min_cost = min(item[1] for item in best)
    best = [item for item in best if abs(item[1] - min_cost) <= 1e-9]

    result = solve(sheets, ties)
    assert result.moved_count == min_moved
    assert result.total_world_displacement == pytest.approx(min_cost)
    assert (result.moved_count, result.total_world_displacement) == (
        min_moved,
        pytest.approx(min_cost),
    )
    chosen = result.shifts
    assert any(chosen == item[2] for item in best)

    expected_options = {name: {item[2][name] for item in best} for name in names}
    expected_uncertain = {
        name: values for name, values in expected_options.items() if len(values) > 1
    }
    assert result.uncertain == expected_uncertain
    for name in names:
        assert result.shifts[name] in expected_options[name]

    assert sorted(result.boundary_ties) == sorted(
        _expected_boundary(sheets, ties, chosen)
    )
