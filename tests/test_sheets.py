"""Tests of exact map-sheet coverage queries."""

from fractions import Fraction

import pytest

from affine import (
    Coverage,
    MapSheet,
    TransformNotInvertibleError,
    locate_point,
)


def north_up(sheet_id, c, f, width=100, height=100, pixel="10"):
    """A north-up sheet: x grows with col, y shrinks with row."""
    return MapSheet(
        sheet_id,
        a=pixel,
        b="0",
        c=c,
        d="0",
        e=f"-{pixel}",
        f=f,
        width=width,
        height=height,
    )


class TestInterior:
    def test_single_sheet_interior(self):
        sheet = north_up("A", c="500000", f="4000000")
        result = locate_point([sheet], "500505", "3999955")
        assert result.coverage is Coverage.INTERIOR
        assert result.sheet_ids == ("A",)
        (hit,) = result.hits
        assert (hit.col, hit.row) == (50, 4)
        assert hit.col_exact == Fraction(101, 2)
        assert hit.row_exact == Fraction(9, 2)

    def test_origin_pixel_is_covered(self):
        sheet = north_up("A", c="500000", f="4000000")
        result = locate_point([sheet], "500000", "4000000")
        assert result.coverage is Coverage.EDGE
        assert result.hits[0].col == 0
        assert result.hits[0].row == 0

    def test_last_pixel_is_covered(self):
        sheet = north_up("A", c="500000", f="4000000")
        result = locate_point([sheet], "500999.999", "3999000.001")
        assert result.coverage is Coverage.INTERIOR
        assert (result.hits[0].col, result.hits[0].row) == (99, 99)


class TestSharedEdges:
    def test_vertical_shared_edge_has_exactly_one_owner(self):
        west = north_up("W", c="500000", f="4000000")
        east = north_up("E", c="501000", f="4000000")
        result = locate_point([west, east], "501000", "3999500")
        assert result.coverage is Coverage.EDGE
        assert result.sheet_ids == ("E",)
        assert (result.hits[0].col, result.hits[0].row) == (0, 50)

    def test_horizontal_shared_edge_with_negative_row_axis(self):
        north = north_up("N", c="500000", f="4000000")
        south = north_up("S", c="500000", f="3999000")
        result = locate_point([north, south], "500500", "3999000")
        assert result.coverage is Coverage.EDGE
        assert result.sheet_ids == ("S",)
        assert (result.hits[0].col, result.hits[0].row) == (50, 0)

    def test_four_sheet_corner_has_exactly_one_owner(self):
        nw = north_up("NW", c="500000", f="4001000")
        ne = north_up("NE", c="501000", f="4001000")
        sw = north_up("SW", c="500000", f="4000000")
        se = north_up("SE", c="501000", f="4000000")
        result = locate_point([nw, ne, sw, se], "501000", "4000000")
        assert result.coverage is Coverage.EDGE
        assert result.sheet_ids == ("SE",)
        assert (result.hits[0].col, result.hits[0].row) == (0, 0)

    def test_exclusive_outer_edge_is_not_covered(self):
        sheet = north_up("A", c="500000", f="4000000")
        result = locate_point([sheet], "501000", "3999500")
        assert result.coverage is Coverage.NONE
        assert result.hits == ()


class TestOverlap:
    def test_legitimate_overlap_returns_both_sheets(self):
        a = north_up("A", c="500000", f="4000000")
        b = north_up("B", c="500500", f="4000500")
        result = locate_point([a, b], "500750", "3999750")
        assert result.coverage is Coverage.OVERLAP
        assert result.sheet_ids == ("A", "B")
        assert (result.hits[0].col, result.hits[0].row) == (75, 25)
        assert (result.hits[1].col, result.hits[1].row) == (25, 75)


class TestRotation:
    def test_rotated_sheet(self):
        # 90 degree rotation with 10 m pixels:
        # x = -10 * row + 1000, y = 10 * col + 2000
        sheet = MapSheet(
            "R",
            a="0",
            b="-10",
            c="1000",
            d="10",
            e="0",
            f="2000",
            width=50,
            height=50,
        )
        result = locate_point([sheet], "990", "2050")
        assert result.coverage is Coverage.INTERIOR
        assert (result.hits[0].col, result.hits[0].row) == (5, 1)

    def test_rotated_sheet_shared_edge(self):
        left = MapSheet(
            "L",
            a="0",
            b="-10",
            c="1000",
            d="10",
            e="0",
            f="2000",
            width=50,
            height=50,
        )
        right = MapSheet(
            "R",
            a="0",
            b="-10",
            c="1000",
            d="10",
            e="0",
            f="2500",
            width=50,
            height=50,
        )
        # y = 2500 is the exclusive col == width edge of L and the
        # inclusive col == 0 edge of R.
        result = locate_point([left, right], "900", "2500")
        assert result.coverage is Coverage.EDGE
        assert result.sheet_ids == ("R",)


class TestExactBoundaries:
    def test_tiny_pixels_with_large_coordinates(self):
        sheet = north_up(
            "BIG",
            c="1000000",
            f="8000000",
            width=2000000,
            height=2000000,
            pixel="0.000001",
        )
        # Exactly 1234567 pixels of 1e-6 past the origin. As a float,
        # 1000001.234567 cannot be represented exactly and the column
        # can shift to 1234566.
        result = locate_point([sheet], "1000001.234567", "7999999.999999")
        assert result.coverage is Coverage.INTERIOR
        hit = result.hits[0]
        assert hit.col == 1234567
        assert hit.col_exact == 1234567
        assert hit.row == 1

    def test_sheet_boundary_at_an_exact_tiny_pixel_edge(self):
        west = north_up(
            "W",
            c="1000000",
            f="8000000",
            width=1234567,
            height=10,
            pixel="0.000001",
        )
        east = north_up(
            "E",
            c="1000001.234567",
            f="8000000",
            width=10,
            height=10,
            pixel="0.000001",
        )
        result = locate_point([west, east], "1000001.234567", "7999999.999999")
        assert result.coverage is Coverage.EDGE
        assert result.sheet_ids == ("E",)
        assert result.hits[0].col == 0

    def test_integer_string_inputs(self):
        sheet = north_up("A", c="500000", f="4000000")
        result = locate_point([sheet], "500010", "3999990")
        assert (result.hits[0].col, result.hits[0].row) == (1, 1)


class TestInputValidation:
    def test_float_world_coordinates_are_rejected(self):
        sheet = north_up("A", c="500000", f="4000000")
        with pytest.raises(TypeError):
            locate_point([sheet], 500000.5, "3999999")

    def test_float_coefficients_are_rejected(self):
        with pytest.raises(TypeError):
            MapSheet(
                "A",
                a=10.0,
                b="0",
                c="0",
                d="0",
                e="-10",
                f="0",
                width=10,
                height=10,
            )

    def test_non_decimal_strings_are_rejected(self):
        sheet = north_up("A", c="500000", f="4000000")
        with pytest.raises(ValueError, match="not an integer or finite decimal"):
            locate_point([sheet], "1/3", "3999999")

    def test_degenerate_sheet_raises(self):
        sheet = MapSheet(
            "DEG",
            a="1",
            b="1",
            c="0",
            d="1",
            e="1",
            f="0",
            width=10,
            height=10,
        )
        with pytest.raises(TransformNotInvertibleError):
            locate_point([sheet], "1", "1")

    def test_non_positive_dimensions_are_rejected(self):
        with pytest.raises(ValueError, match="width must be positive"):
            MapSheet(
                "A",
                a="10",
                b="0",
                c="0",
                d="0",
                e="-10",
                f="0",
                width=0,
                height=10,
            )


class TestNoCoverage:
    def test_point_outside_all_sheets(self):
        sheet = north_up("A", c="500000", f="4000000")
        result = locate_point([sheet], "600000", "3999500")
        assert result.coverage is Coverage.NONE
        assert result.hits == ()
        assert result.sheet_ids == ()
