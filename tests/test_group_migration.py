"""Tests for target-grid group migration segments."""

import math
import time

import pytest

from affine import (
    Affine,
    GroupMigration,
    TransformNotInvertibleError,
    reproject_group_migrations,
)


def brute_force_migrations(
    old_transform,
    new_transform,
    target_transform,
    target_width,
    target_height,
    source_window,
):
    """Per-pixel reference implementation of the migration contract."""
    inv_target = ~target_transform
    row_start, col_start, row_end, col_end = source_window

    def group(transform, row, col):
        x, y = transform @ (col + 0.5, row + 0.5)
        u, v = inv_target @ (x, y)
        target_col = math.floor(u)
        target_row = math.floor(v)
        if 0 <= target_col < target_width and 0 <= target_row < target_height:
            return target_col, target_row
        return None

    expected = []
    for row in range(row_start, row_end):
        run = None
        for col in range(col_start, col_end):
            old_group = group(old_transform, row, col)
            new_group = group(new_transform, row, col)
            if old_group == new_group:
                if run is not None:
                    expected.append(run)
                    run = None
                continue
            if (
                run is not None
                and run.col_end == col
                and run.old_group == old_group
                and run.new_group == new_group
            ):
                run = GroupMigration(row, run.col_start, col + 1, old_group, new_group)
            else:
                if run is not None:
                    expected.append(run)
                run = GroupMigration(row, col, col + 1, old_group, new_group)
        if run is not None:
            expected.append(run)
    return expected


def assert_matches_brute_force(
    old_transform,
    new_transform,
    target_transform,
    target_width,
    target_height,
    source_window,
):
    result = reproject_group_migrations(
        old_transform,
        new_transform,
        target_transform,
        target_width,
        target_height,
        source_window,
    )
    expected = brute_force_migrations(
        old_transform,
        new_transform,
        target_transform,
        target_width,
        target_height,
        source_window,
    )
    assert result == expected

    row_start, col_start, row_end, col_end = source_window
    changed = set()
    for migration in result:
        assert col_start <= migration.col_start < migration.col_end <= col_end
        assert row_start <= migration.row < row_end
        for col in range(migration.col_start, migration.col_end):
            changed.add((migration.row, col))
    assert len(changed) == sum(
        migration.col_end - migration.col_start for migration in result
    )

    inv_target = ~target_transform

    def group(transform, row, col):
        x, y = transform @ (col + 0.5, row + 0.5)
        u, v = inv_target @ (x, y)
        target_col = math.floor(u)
        target_row = math.floor(v)
        if 0 <= target_col < target_width and 0 <= target_row < target_height:
            return target_col, target_row
        return None

    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            differs = group(old_transform, row, col) != group(new_transform, row, col)
            assert ((row, col) in changed) == differs


def test_north_up_half_pixel_shift():
    old_transform = Affine.identity()
    new_transform = Affine.translation(0.5, 0.0)
    target_transform = Affine.identity()
    assert_matches_brute_force(
        old_transform, new_transform, target_transform, 8, 8, (0, 0, 4, 10)
    )


def test_outside_target_is_its_own_group():
    old_transform = Affine.identity()
    new_transform = Affine.translation(3.25, -2.5)
    target_transform = Affine.identity()
    result = reproject_group_migrations(
        old_transform,
        new_transform,
        target_transform,
        4,
        4,
        (0, 0, 6, 6),
    )
    assert result
    outside_entries = [
        migration
        for migration in result
        if migration.old_group is None or migration.new_group is None
    ]
    assert outside_entries
    for migration in outside_entries:
        if migration.old_group is None:
            assert migration.new_group is None or (
                0 <= migration.new_group[0] < 4 and 0 <= migration.new_group[1] < 4
            )
    assert_matches_brute_force(
        old_transform,
        new_transform,
        target_transform,
        4,
        4,
        (0, 0, 6, 6),
    )


def test_rotated_target_grid():
    target_transform = Affine.rotation(15.0) @ Affine.scale(0.7, 0.9)
    old_transform = Affine.scale(0.4, 0.6)
    new_transform = Affine.translation(0.3, -0.2) @ old_transform
    assert_matches_brute_force(
        old_transform, new_transform, target_transform, 12, 12, (0, 0, 9, 13)
    )


def test_shear_and_row_dependent_groups():
    old_transform = Affine(0.3, 0.1, 1.0, 0.05, 0.4, -2.0)
    new_transform = Affine(0.31, 0.12, 1.1, 0.04, 0.38, -1.9)
    assert_matches_brute_force(
        old_transform, new_transform, Affine.identity(), 10, 10, (0, 0, 12, 14)
    )


def test_negative_pixel_size_target():
    target_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 8.0)
    old_transform = Affine.identity()
    new_transform = Affine.translation(0.25, 0.75)
    assert_matches_brute_force(
        old_transform, new_transform, target_transform, 8, 8, (0, 0, 8, 8)
    )


def test_no_changes_returns_empty():
    transform = Affine(0.3, 0.1, 1.0, 0.05, 0.4, -2.0)
    result = reproject_group_migrations(
        transform, transform, Affine.identity(), 10, 10, (0, 0, 12, 14)
    )
    assert result == []


def test_shift_that_never_crosses_a_boundary():
    old_transform = Affine.identity()
    new_transform = Affine.translation(0.1, 0.1)
    result = reproject_group_migrations(
        old_transform,
        new_transform,
        Affine.identity(),
        100,
        100,
        (0, 0, 5, 20),
    )
    assert result == []


def test_boundary_landing_exactly_on_integer():
    old_transform = Affine.identity()
    new_transform = Affine.translation(0.5, 0.5)
    assert_matches_brute_force(
        old_transform,
        new_transform,
        Affine.identity(),
        6,
        6,
        (0, 0, 6, 6),
    )


def test_wide_window_with_few_boundaries():
    slope = 1e-9
    old_transform = Affine(slope, 0.0, 0.0, 0.0, slope, 0.0)
    new_transform = Affine(slope, 0.0, 0.95, 0.0, slope, 0.0)
    window = (0, 0, 2, 100_000_000)

    start = time.perf_counter()
    result = reproject_group_migrations(
        old_transform,
        new_transform,
        Affine.identity(),
        20,
        20,
        window,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    assert len(result) == 2
    for migration in result:
        assert migration.old_group == (0, 0)
        assert migration.new_group == (1, 0)
        assert migration.col_end - migration.col_start > 40_000_000

    crossing_window = (0, 49_999_995, 1, 50_000_005)
    assert_matches_brute_force(
        old_transform,
        new_transform,
        Affine.identity(),
        20,
        20,
        crossing_window,
    )


def test_segments_are_maximal_and_constant():
    old_transform = Affine(0.35, 0.07, 0.4, 0.02, 0.41, 0.2)
    new_transform = Affine(0.37, 0.09, 0.9, 0.06, 0.33, -0.4)
    target_transform = Affine.scale(0.5, 0.8)
    source_window = (0, 0, 7, 25)
    result = reproject_group_migrations(
        old_transform,
        new_transform,
        target_transform,
        16,
        16,
        source_window,
    )
    by_row = {}
    for migration in result:
        by_row.setdefault(migration.row, []).append(migration)
    for migrations in by_row.values():
        for earlier, later in zip(migrations, migrations[1:]):
            assert earlier.col_end <= later.col_start
            if earlier.col_end == later.col_start:
                assert (
                    earlier.old_group != later.old_group
                    or earlier.new_group != later.new_group
                )


def test_invalid_window_raises():
    with pytest.raises(ValueError, match="source_window"):
        reproject_group_migrations(
            Affine.identity(),
            Affine.identity(),
            Affine.identity(),
            4,
            4,
            (2, 0, 1, 4),
        )


def test_invalid_target_dimensions_raise():
    with pytest.raises(ValueError, match="target grid"):
        reproject_group_migrations(
            Affine.identity(),
            Affine.identity(),
            Affine.identity(),
            0,
            4,
            (0, 0, 2, 2),
        )


def test_degenerate_target_transform_raises():
    with pytest.raises(TransformNotInvertibleError):
        reproject_group_migrations(
            Affine.identity(),
            Affine.identity(),
            Affine(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            4,
            4,
            (0, 0, 2, 2),
        )


def test_empty_window_is_allowed():
    result = reproject_group_migrations(
        Affine.identity(),
        Affine.translation(0.5, 0.0),
        Affine.identity(),
        4,
        4,
        (3, 0, 3, 10),
    )
    assert result == []
