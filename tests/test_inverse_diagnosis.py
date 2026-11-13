"""Tests for exact-rational inverse diagnosis of singular/ill-conditioned maps."""

import pytest

from diagnose_inverse import locate_centers

LINE = ("0.1", "0.2", "10.0", "0.2", "0.4", "20.0")
POINT = ("0", "0", "5.0", "0", "0", "7.0")
ILL = (
    "100000000",
    "99999999.99999999",
    "0",
    "100000000",
    "100000000",
    "0",
)


def test_rank1_line_has_coincident_centers_with_two_witnesses():
    diag = locate_centers(LINE, 4, 4, "10.2", "20.4")
    assert diag.rank == 1
    assert diag.exact_determinant == 0
    assert diag.status == "many"
    # Two distinct (col, row) centers landing on the same world point.
    assert diag.witnesses == ((2, 0), (0, 1))
    assert len(set(diag.witnesses)) == 2


def test_rank1_line_can_have_a_single_center():
    diag = locate_centers(LINE, 4, 4, "10.9", "21.8")
    assert diag.rank == 1
    assert diag.status == "one"
    assert diag.witnesses == ((3, 3),)


def test_rank1_line_query_off_the_line_has_no_center():
    diag = locate_centers(LINE, 4, 4, "10.2", "20.5")
    assert diag.rank == 1
    assert diag.status == "none"
    assert diag.witnesses == ()
    # Singular does not mean "no answer": other queries on the line do.
    assert locate_centers(LINE, 4, 4, "10.9", "21.8").has_answer


def test_rank0_collapses_every_center_to_one_point():
    hit = locate_centers(POINT, 4, 4, "5.0", "7.0")
    assert hit.rank == 0
    assert hit.exact_determinant == 0
    assert hit.status == "many"
    assert len(hit.witnesses) == 16
    miss = locate_centers(POINT, 4, 4, "5.0", "8.0")
    assert miss.status == "none"


def test_near_singular_invertible_solved_exactly_despite_cancellation():
    inside = locate_centers(ILL, 4, 4, "199999999.99999999", "200000000")
    assert inside.rank == 2
    assert inside.exact_determinant == 1
    # Float arithmetic loses the exact determinant to cancellation.
    assert inside.float_determinant == pytest.approx(2.0)
    assert inside.cancellation
    assert inside.status == "one"
    assert inside.witnesses == ((1, 1),)


def test_near_singular_preimage_outside_grid_has_no_center():
    outside = locate_centers(ILL, 4, 4, "999999999.99999995", "1000000000")
    assert outside.status == "none"
    assert outside.witnesses == ()


def test_float_inversion_picks_wrong_pixel_on_ill_conditioned_map():
    from affine import Affine

    transform = Affine(*(float(v) for v in ILL))
    # The exact decimal query is parsed directly as a Fraction; converting it
    # through float first would round it away from its true value.
    exact = locate_centers(
        ILL, 4, 4, "199999999.99999999", "200000000"
    ).witnesses[0]
    naive = tuple(
        round(v)
        for v in (~transform @ (float("199999999.99999999"), 200000000.0))
    )
    assert naive != exact
    assert naive == (2, 0)
    assert exact == (1, 1)
