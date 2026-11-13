"""Grid alignment tests."""

import pytest

from affine import (
    Affine,
    GridAlignment,
    GridAlignmentError,
    TransformNotInvertibleError,
    Window,
    align_grid,
)


def test_identical_grids():
    transform = Affine(10, 0, 1000, 0, -10, 2000)
    alignment = align_grid(transform, (20, 30), transform, (40, 50))
    assert isinstance(alignment, GridAlignment)
    assert alignment.column_mapping == (1, 0, 0)
    assert alignment.row_mapping == (0, 1, 0)
    assert alignment.offset == (0, 0)
    assert alignment.worst_case_deviation_pixels < 1e-9
    assert alignment.overlap_a == Window(0, 20, 0, 30)
    assert alignment.overlap_b == Window(0, 20, 0, 30)
    assert alignment.overlap_pixels == 600


def test_whole_pixel_translation():
    a = Affine(10, 0, 1000, 0, -10, 2000)
    b = Affine(10, 0, 1030, 0, -10, 1980)
    alignment = align_grid(a, (10, 10), b, (10, 10))
    assert alignment.column_mapping == (1, 0, -3)
    assert alignment.row_mapping == (0, 1, -2)
    assert alignment.overlap_a == Window(2, 10, 3, 10)
    assert alignment.overlap_b == Window(0, 8, 0, 7)
    assert alignment.overlap_a.shape == (8, 7)
    assert alignment.overlap_pixels == 56


def test_axis_swap():
    a = Affine(10, 0, 0, 0, -10, 0)
    b = Affine(0, 10, 0, -10, 0, 0)
    alignment = align_grid(a, (10, 20), b, (20, 10))
    assert alignment.permutation == (0, 1, 1, 0)
    assert alignment.column_mapping == (0, 1, 0)
    assert alignment.row_mapping == (1, 0, 0)
    # Source (col, row) maps to destination (row, col).
    assert alignment.transform @ (0.0, 0.0) == (0.0, 0.0)
    assert alignment.transform @ (1.0, 0.0) == (0.0, 1.0)
    assert alignment.transform @ (0.0, 1.0) == (1.0, 0.0)
    assert alignment.overlap_a == Window(0, 10, 0, 20)
    assert alignment.overlap_b == Window(0, 20, 0, 10)
    assert alignment.overlap_pixels == 200


def test_flip_both_axes():
    a = Affine(5, 0, 0, 0, 5, 0)
    b = Affine(-5, 0, 100, 0, -5, 100)
    alignment = align_grid(a, (20, 20), b, (20, 20))
    assert alignment.permutation == (-1, 0, 0, -1)
    assert alignment.offset == (20, 20)
    assert alignment.overlap_a == Window(0, 20, 0, 20)
    assert alignment.overlap_b == Window(0, 20, 0, 20)
    assert alignment.overlap_pixels == 400


def test_rotation_90_degrees():
    a = Affine(10, 0, 0, 0, -10, 0)
    b = Affine.rotation(90) @ Affine(10, 0, 0, 0, -10, 0)
    alignment = align_grid(a, (8, 5), b, (5, 8))
    assert alignment.permutation == (0, -1, 1, 0)
    assert alignment.worst_case_deviation_pixels < 1e-9


def test_non_right_rotation_is_rejected():
    a = Affine(10, 0, 0, 0, -10, 0)
    b = Affine.rotation(0.5, pivot=(50, 50)) @ a
    with pytest.raises(GridAlignmentError):
        align_grid(a, (10, 10), b, (10, 10))


def test_edge_contact_has_no_overlap_pixels():
    a = Affine(10, 0, 0, 0, -10, 100)
    b = Affine(10, 0, 100, 0, -10, 100)
    alignment = align_grid(a, (10, 10), b, (10, 10))
    assert alignment.column_mapping == (1, 0, -10)
    assert alignment.row_mapping == (0, 1, 0)
    assert alignment.overlap_a.is_empty
    assert alignment.overlap_b.is_empty
    assert alignment.overlap_pixels == 0


def test_corner_contact_has_no_overlap_pixels():
    a = Affine(10, 0, 0, 0, -10, 100)
    b = Affine(10, 0, 100, 0, -10, 0)
    alignment = align_grid(a, (10, 10), b, (10, 10))
    assert alignment.column_mapping == (1, 0, -10)
    assert alignment.row_mapping == (0, 1, -10)
    assert alignment.overlap_pixels == 0


def test_corner_contact_with_axis_swap():
    a = Affine(10, 0, 0, 0, -10, 100)
    b = Affine(0, 10, 100, 10, 0, 0)
    alignment = align_grid(a, (10, 10), b, (10, 10))
    assert alignment.overlap_pixels == 0


def test_contained_raster():
    a = Affine(10, 0, 0, 0, -10, 1000)
    b = Affine(10, 0, 0, 0, -10, 1000)
    alignment = align_grid(a, (50, 50), b, (10, 10))
    assert alignment.overlap_a == Window(0, 10, 0, 10)
    assert alignment.overlap_b == Window(0, 10, 0, 10)


def test_tiny_pixel_size_mismatch_is_rejected():
    # The size coefficients differ by only 1e-7, far below EPSILON, but
    # across 1000 pixels that accumulates to a visible grid deviation.
    a = Affine(10, 0, 0, 0, -10, 0)
    b = Affine(10 + 1e-7, 0, 0, 0, -10, 0)
    assert a.almost_equals(b)
    with pytest.raises(GridAlignmentError):
        align_grid(a, (1000, 1000), b, (1000, 1000))


def test_small_extent_tolerates_round_off_only():
    a = Affine(10, 0, 0, 0, -10, 0)
    b = Affine(10 + 1e-12, 0, 0, 0, -10, 0)
    alignment = align_grid(a, (1, 1), b, (1, 1), tolerance=1e-3)
    assert alignment.worst_case_deviation_pixels < 1e-6


def test_worst_case_corner_not_origin():
    a = Affine(10, 0, 0, 0, -10, 0)
    # A tiny shear vanishes at the origin and grows to metres at the far
    # corner; comparing coefficients at the origin would see nothing.
    b = a @ Affine(1, 1e-3, 0, 0, 1, 0)
    with pytest.raises(GridAlignmentError):
        align_grid(a, (1000, 1000), b, (1000, 1000), tolerance=0.1)


def test_map_to_a_and_b():
    a = Affine(10, 0, 1000, 0, -10, 2000)
    b = Affine(10, 0, 1030, 0, -10, 1980)
    alignment = align_grid(a, (10, 10), b, (10, 10))
    for col in range(10):
        for row in range(10):
            mapped = alignment.map_to_b(col, row)
            assert alignment.map_to_a(*mapped) == (col, row)
    assert alignment.map_to_b(3, 2) == (0, 0)


def test_invalid_shapes():
    a = Affine.identity()
    with pytest.raises(ValueError, match="must be positive"):
        align_grid(a, (0, 10), a, (10, 10))
    with pytest.raises(ValueError, match="rows, cols"):
        align_grid(a, (10,), a, (10, 10))


def test_degenerate_transform():
    with pytest.raises(TransformNotInvertibleError):
        align_grid(Affine.scale(0), (10, 10), Affine.identity(), (10, 10))
