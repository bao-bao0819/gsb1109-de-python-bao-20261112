"""Tests of affine.overview."""

import math

import pytest

from affine import Affine, OverviewLevel, select_overview


def make_level(transform, width, height, tile_width, tile_height, size=100, sizes=None):
    if sizes is None:
        ncols = math.ceil(width / tile_width)
        nrows = math.ceil(height / tile_height)
        sizes = {(c, r): size for r in range(nrows) for c in range(ncols)}
    return OverviewLevel(transform, width, height, tile_width, tile_height, sizes)


def square(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_window_within_single_tile():
    level = make_level(Affine.scale(1.0), 512, 512, 256, 256)
    sel = select_overview(square(10, 10, 100, 100), 2.0, [level])
    assert sel.ok
    assert sel.level_index == 0
    assert sel.tiles == ((0, 0),)
    assert sel.total_bytes == 100
    assert sel.resolution_ok
    assert sel.coverage_ok
    assert sel.unmet == ()


def test_window_spanning_multiple_tiles():
    level = make_level(Affine.scale(1.0), 512, 512, 256, 256)
    sel = select_overview(square(10, 10, 300, 300), 2.0, [level])
    assert sel.ok
    assert sel.tiles == ((0, 0), (1, 0), (0, 1), (1, 1))
    assert sel.total_bytes == 400


def test_edge_aligned_window_reads_no_neighbor():
    level = make_level(Affine.scale(1.0), 512, 512, 256, 256)
    sel = select_overview(square(0, 0, 256, 256), 2.0, [level])
    assert sel.tiles == ((0, 0),)
    assert sel.total_bytes == 100


def test_slanted_window_edge_touch_excluded():
    level = make_level(Affine.scale(1.0), 512, 512, 256, 256)
    window = [(256, 0), (256, 256), (0, 128)]
    sel = select_overview(window, 2.0, [level])
    assert sel.tiles == ((0, 0),)


def test_slanted_narrow_window_point_touch_excluded():
    level = make_level(Affine.scale(1.0), 512, 512, 256, 256)
    window = [(4, 4), (256, 256), (254, 258), (2, 6)]
    sel = select_overview(window, 2.0, [level])
    assert sel.tiles == ((0, 0), (0, 1))


def test_north_up_transform():
    transform = Affine(1, 0, 0, 0, -1, 512)
    level = make_level(transform, 512, 512, 256, 256)
    sel = select_overview(square(10, 412, 100, 502), 2.0, [level])
    assert sel.tiles == ((0, 0),)


def test_partial_tile_at_level_edge():
    level = make_level(Affine.scale(1.0), 300, 512, 256, 256)
    sel = select_overview(square(260, 10, 300, 100), 2.0, [level])
    assert sel.tiles == ((1, 0),)


def test_resolution_limit_excludes_coarse_level():
    fine = make_level(Affine.scale(1.0), 512, 512, 256, 256, size=100)
    coarse = make_level(Affine.scale(4.0), 128, 128, 128, 128, size=10)
    sel = select_overview(square(0, 0, 100, 100), 2.0, [fine, coarse])
    assert sel.level_index == 0
    assert sel.total_bytes == 100


def test_max_stretch_in_any_direction():
    rotated = make_level(
        Affine.rotation(45) @ Affine.scale(3.0, 1.0), 512, 512, 512, 512, size=1
    )
    uniform = make_level(Affine.scale(2.0), 256, 256, 256, 256, size=500)
    window = [(100, 100), (300, 300), (250, 350), (50, 150)]
    sel = select_overview(window, 2.5, [rotated, uniform])
    assert sel.level_index == 1
    assert sel.total_bytes == 500


def test_highest_overview_not_assumed_cheapest():
    level0 = make_level(Affine.scale(1.0), 512, 512, 256, 256, size=100)
    level1 = make_level(Affine.scale(2.0), 256, 256, 256, 256, size=80)
    level2 = make_level(Affine.scale(4.0), 128, 128, 128, 128, size=5000)
    sel = select_overview(square(0, 0, 100, 100), 10.0, [level0, level1, level2])
    assert sel.level_index == 1
    assert sel.total_bytes == 80


def test_clipped_level_not_covering_window():
    full = make_level(Affine.scale(1.0), 512, 512, 256, 256, size=100)
    clipped = make_level(Affine.scale(4.0), 64, 64, 64, 64, size=1)
    sel = select_overview(square(0, 0, 512, 512), 10.0, [full, clipped])
    assert sel.level_index == 0
    assert sel.total_bytes == 400
    assert sel.coverage_ok


def test_no_level_meets_resolution():
    coarse = make_level(Affine.scale(8.0), 64, 64, 64, 64)
    sel = select_overview(square(0, 0, 100, 100), 2.0, [coarse])
    assert not sel.ok
    assert sel.level_index is None
    assert sel.tiles == ()
    assert sel.total_bytes == 0
    assert not sel.resolution_ok
    assert sel.coverage_ok
    assert sel.unmet == ("resolution",)


def test_no_level_covers_window():
    level = make_level(Affine.scale(1.0), 512, 512, 256, 256)
    sel = select_overview(square(1000, 1000, 1100, 1100), 2.0, [level])
    assert not sel.ok
    assert sel.resolution_ok
    assert not sel.coverage_ok
    assert sel.unmet == ("coverage",)


def test_no_level_meets_either_condition():
    coarse = make_level(Affine.scale(8.0), 64, 64, 64, 64)
    sel = select_overview(square(1000, 1000, 1100, 1100), 2.0, [coarse])
    assert not sel.ok
    assert sel.unmet == ("resolution", "coverage")


def test_nested_sequence_tile_sizes():
    level = OverviewLevel(Affine.scale(1.0), 512, 512, 256, 256, [[10, 20], [30, 40]])
    sel = select_overview(square(0, 0, 512, 512), 2.0, [level])
    assert sel.tiles == ((0, 0), (1, 0), (0, 1), (1, 1))
    assert sel.total_bytes == 100


def test_missing_tile_counts_zero_bytes():
    level = make_level(Affine.scale(1.0), 512, 512, 256, 256, sizes={(1, 1): 50})
    sel = select_overview(square(300, 300, 400, 400), 2.0, [level])
    assert sel.tiles == ((1, 1),)
    assert sel.total_bytes == 50


def test_equal_cost_prefers_fewer_tiles_then_lower_index():
    one_tile = make_level(Affine.scale(2.0), 256, 256, 256, 256, size=100)
    four_tiles = make_level(Affine.scale(2.0), 256, 256, 128, 128, size=25)
    window = square(1, 1, 511, 511)
    sel = select_overview(window, 4.0, [one_tile, four_tiles])
    assert sel.level_index == 0
    assert sel.tiles == ((0, 0),)
    assert sel.total_bytes == 100
    same_cost = make_level(Affine.scale(2.0), 256, 256, 256, 256, size=100)
    sel = select_overview(window, 4.0, [one_tile, same_cost])
    assert sel.level_index == 0


def test_invalid_window_raises():
    level = make_level(Affine.scale(1.0), 512, 512, 256, 256)
    with pytest.raises(ValueError, match="at least 3 vertices"):
        select_overview([(0, 0), (1, 1)], 1.0, [level])


def test_invalid_resolution_raises():
    level = make_level(Affine.scale(1.0), 512, 512, 256, 256)
    with pytest.raises(ValueError, match="max_resolution"):
        select_overview(square(0, 0, 10, 10), 0.0, [level])


def test_empty_levels_raises():
    with pytest.raises(ValueError, match="at least one"):
        select_overview(square(0, 0, 10, 10), 1.0, [])


def test_invalid_level_dimensions_raise():
    with pytest.raises(ValueError, match="width"):
        OverviewLevel(Affine.scale(1.0), 0, 512, 256, 256, {})
    with pytest.raises(ValueError, match="tile_width"):
        OverviewLevel(Affine.scale(1.0), 512, 512, 0, 256, {})
