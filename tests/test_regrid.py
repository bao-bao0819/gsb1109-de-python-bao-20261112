"""Tests for area-weighted window regridding."""

import math

import pytest

from affine import Affine
from affine.regrid import RegridError, RegridResult, regrid_window

try:
    import numpy as np
except ImportError:
    np = None


def total_of(result):
    return sum(sum(row) for row in result.data) + result.masked + result.outside


def test_returns_result_and_preserves_shape():
    src = [[10.0]]
    mask = [[True, True], [True, True]]
    result = regrid_window(src, Affine(2, 0, 0, 0, 2, 0), Affine.identity(), mask)
    assert isinstance(result, RegridResult)
    assert len(result.data) == 2
    assert all(len(row) == 2 for row in result.data)


def test_one_old_pixel_split_evenly_over_four_new_pixels():
    # One 2x2 old pixel centered on the corner of four unit new pixels.
    result = regrid_window(
        [[40.0]],
        Affine(2, 0, 0, 0, 2, 0),
        Affine.identity(),
        [[True, True], [True, True]],
    )
    assert result.data == [[10.0, 10.0], [10.0, 10.0]]
    assert result.masked == 0.0
    assert result.outside == 0.0
    assert total_of(result) == pytest.approx(40.0)


def test_half_offset_pixel_diagonally_touches_four_cells():
    # Old unit pixel centered on the corner (1, 1) of the new grid.
    result = regrid_window(
        [[4.0]],
        Affine(1, 0, 0.5, 0, 1, 0.5),
        Affine.identity(),
        [[True, True], [True, True]],
    )
    for row in result.data:
        for value in row:
            assert value == pytest.approx(1.0)
    assert total_of(result) == pytest.approx(4.0)


def test_identical_grids_move_totals_one_to_one():
    source = [[1.0, 2.0], [3.0, 4.0]]
    mask = [[True, True], [True, True]]
    result = regrid_window(source, Affine.identity(), Affine.identity(), mask)
    assert result.data == source
    assert result.masked == 0.0
    assert result.outside == 0.0


def test_masked_pixels_keep_their_share_in_a_separate_ledger():
    mask = [[True, False], [True, True]]
    result = regrid_window(
        [[40.0]],
        Affine(2, 0, 0, 0, 2, 0),
        Affine.identity(),
        mask,
    )
    assert result.data[0][1] == 0.0
    assert result.masked == pytest.approx(10.0)
    assert total_of(result) == pytest.approx(40.0)


def test_share_outside_target_window_is_booked_separately():
    # A 2x2 source window mapped onto a single-unit target window:
    # three quarters of the total lies outside.
    source = [[1.0, 1.0], [1.0, 1.0]]
    result = regrid_window(
        source,
        Affine.identity(),
        Affine.identity(),
        [[True]],
    )
    assert result.data == [[1.0]]
    assert result.masked == 0.0
    assert result.outside == pytest.approx(3.0)
    assert total_of(result) == pytest.approx(4.0)


def test_edge_only_contact_contributes_nothing():
    # Old pixel occupies world x in [1, 2]; the target cell covers [0, 1],
    # so they share only the boundary line x = 1 (zero area).
    result = regrid_window(
        [[100.0]],
        Affine(1, 0, 1.0, 0, 1, 0.0),
        Affine.identity(),
        [[True]],
    )
    assert result.data == [[0.0]]
    assert result.masked == 0.0
    assert result.outside == pytest.approx(100.0)
    assert total_of(result) == pytest.approx(100.0)


def test_corner_only_contact_contributes_nothing():
    result = regrid_window(
        [[8.0]],
        Affine(1, 0, 1.0, 0, 1, 1.0),
        Affine.identity(),
        [[True]],
    )
    assert result.data == [[0.0]]
    assert result.outside == pytest.approx(8.0)
    assert total_of(result) == pytest.approx(8.0)


def test_negative_determinant_dst_grid_has_non_negative_output():
    # Flip the x axis: new column 0 covers old column 1 and vice versa.
    source = [[1.0, 2.0], [3.0, 4.0]]
    flipped = Affine(-1, 0, 2.0, 0, 1, 0.0)
    assert flipped.determinant < 0.0
    result = regrid_window(
        source,
        Affine.identity(),
        flipped,
        [[True, True], [True, True]],
    )
    for row in result.data:
        assert all(value >= 0.0 for value in row)
    assert result.data[0] == [2.0, 1.0]
    assert result.data[1] == [4.0, 3.0]
    assert total_of(result) == pytest.approx(10.0)


def test_negative_determinant_source_grid():
    flipped_src = Affine(-1, 0, 2.0, 0, 1, 0.0)
    source = [[5.0, 7.0], [9.0, 11.0]]
    result = regrid_window(
        source,
        flipped_src,
        Affine.identity(),
        [[True, True], [True, True]],
    )
    for row in result.data:
        assert all(value >= 0.0 for value in row)
    assert total_of(result) == pytest.approx(32.0)


def test_rotated_grid_conserves_total_and_stays_non_negative():
    rotation = Affine.rotation(30.0) @ Affine.scale(0.6)
    source = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]
    src_transform = Affine.scale(0.6) @ Affine.translation(-2.0, -2.0)
    mask = [[True] * 8 for _ in range(8)]
    result = regrid_window(source, src_transform, rotation, mask)
    grand_total = sum(sum(row) for row in source)
    for row in result.data:
        assert all(value >= 0.0 for value in row)
    assert total_of(result) == pytest.approx(grand_total)


def test_zero_total_pixels_are_ignored():
    result = regrid_window(
        [[0.0]],
        Affine.identity(),
        Affine.identity(),
        [[True]],
    )
    assert result.data == [[0.0]]
    assert result.masked == 0.0
    assert result.outside == 0.0


def test_degenerate_transform_is_rejected():
    degenerate = Affine(1, 0, 0, 0, 0, 0)
    with pytest.raises(RegridError):
        regrid_window([[1.0]], degenerate, Affine.identity(), [[True]])
    with pytest.raises(RegridError):
        regrid_window([[1.0]], Affine.identity(), degenerate, [[True]])


def test_ragged_source_is_rejected():
    with pytest.raises(RegridError):
        regrid_window(
            [[1.0, 2.0], [3.0]],
            Affine.identity(),
            Affine.identity(),
            [[True, True], [True, True]],
        )


def test_empty_window_is_rejected():
    with pytest.raises(RegridError):
        regrid_window([], Affine.identity(), Affine.identity(), [[True]])
    with pytest.raises(RegridError):
        regrid_window([[1.0]], Affine.identity(), Affine.identity(), [])


@pytest.mark.skipif(np is None, reason="requires numpy")
def test_accepts_numpy_arrays():
    source = np.full((2, 2), 4.0)
    mask = np.ones((3, 3), dtype=bool)
    result = regrid_window(source, Affine.scale(1.5), Affine.identity(), mask)
    assert total_of(result) == pytest.approx(16.0)


@pytest.mark.skipif(np is None, reason="requires numpy")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_transforms_always_balance_the_three_ledgers(seed):
    rng = np.random.default_rng(seed)
    angle = math.radians(rng.uniform(-180.0, 180.0))
    scale = rng.uniform(0.4, 1.3)
    rotation = Affine(
        math.cos(angle) * scale,
        -math.sin(angle) * scale,
        rng.uniform(-1.0, 1.0),
        math.sin(angle) * scale,
        math.cos(angle) * scale,
        rng.uniform(-1.0, 1.0),
    )
    source = rng.uniform(0.0, 10.0, size=(5, 5)).tolist()
    mask = rng.uniform(size=(7, 7)) > 0.3
    result = regrid_window(source, Affine.scale(0.8), rotation, mask)
    grand_total = sum(sum(row) for row in source)
    for row in result.data:
        assert all(value >= 0.0 for value in row)
    assert result.masked >= 0.0
    assert result.outside >= 0.0
    assert total_of(result) == pytest.approx(grand_total, rel=1e-9)
