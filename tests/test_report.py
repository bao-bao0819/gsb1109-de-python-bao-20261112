import random

import pytest

from uncertainties import UFloat
from uncertainties import report
from uncertainties.report import (
    HALF_EVEN,
    HALF_UP,
    Interval,
    check_report,
    parse_report_row,
    rounding_interval,
)
from decimal import Decimal


parse_cases = [
    # text, nominal, nominal unit, sigma, sigma unit, signif digits, shorthand
    ("10.00(50)", "10.00", "0.01", "0.50", "0.01", 2, True),
    ("12.34(142)", "12.34", "0.01", "1.42", "0.01", 3, True),
    ("1.2(3.4)", "1.2", "0.1", "3.4", "0.1", 2, True),
    ("680(41)e-3", "0.680", "0.001", "0.041", "0.001", 2, True),
    ("23.2", "23.2", "0.1", "0.1", "0.1", 1, True),
    ("12.58+/-0.23", "12.58", "0.01", "0.23", "0.01", 2, False),
    ("-1.23e2+/-4", "-123", "1", "4", "1", 1, False),
    # Trailing / leading decimal points still pin the last digit:
    ("31.", "31", "1", "1", "1", 1, True),
    (".5", "0.5", "0.1", "0.1", "0.1", 1, True),
    ("-3(0.)", "-3", "1", "0", "1", 0, True),
    (".1(2.)", "0.1", "0.1", "2", "1", 1, True),
]


@pytest.mark.parametrize(
    "text, nominal, nominal_unit, sigma, sigma_unit, digits, shorthand",
    parse_cases,
)
def test_parse_report_row(
    text, nominal, nominal_unit, sigma, sigma_unit, digits, shorthand
):
    row = parse_report_row(text)
    assert row.nominal == Decimal(nominal)
    assert row.nominal_unit == Decimal(nominal_unit)
    assert row.sigma == Decimal(sigma)
    assert row.sigma_unit == Decimal(sigma_unit)
    assert row.uncert_signif_digits == digits
    assert row.shorthand is shorthand


def test_parse_report_row_accepts_parsed_object():
    row = parse_report_row("10.00(50)")
    assert check_report(row, parse_report_row("3.00(20)"), "7.0(5)").consistent


def test_parse_report_row_rejects_garbage():
    with pytest.raises(ValueError):
        parse_report_row("not a measurement")
    with pytest.raises(ValueError):
        parse_report_row(12.34)


def test_rounding_interval_half_even_ties():
    # An even grid index wins both adjacent ties; an odd grid index
    # loses both of them (the ties go to its even neighbors):
    even = rounding_interval(Decimal("1.0"), Decimal("0.1"), HALF_EVEN)
    assert even == Interval(
        Decimal("0.95"), Decimal("1.05"), True, True
    )
    odd = rounding_interval(Decimal("1.1"), Decimal("0.1"), HALF_EVEN)
    assert odd == Interval(
        Decimal("1.05"), Decimal("1.15"), False, False
    )
    # The tie itself follows half-even:
    assert not odd.contains(Decimal("1.05"))
    assert even.contains(Decimal("1.05"))


def test_rounding_interval_half_up_ties():
    interval = rounding_interval(Decimal("1.0"), Decimal("0.1"), HALF_UP)
    assert interval.lower_inclusive is True
    assert interval.upper_inclusive is False
    assert interval.contains(Decimal("0.95"))
    assert not interval.contains(Decimal("1.05"))


def test_rounding_interval_sigma_clipped_at_zero():
    interval = rounding_interval(
        Decimal("0.0"), Decimal("0.1"), HALF_EVEN, non_negative=True
    )
    assert interval.lower == 0
    assert interval.lower_inclusive is True
    assert interval.contains(0)
    # Without the physical constraint, negative values remain feasible:
    nominal = rounding_interval(Decimal("0.0"), Decimal("0.1"), HALF_EVEN)
    assert nominal.lower == Decimal("-0.05")


def test_rounding_interval_unknown_rule():
    with pytest.raises(ValueError):
        rounding_interval(Decimal("1"), Decimal("1"), "half-sideways")


def test_consistent_report_round_trips_textually():
    originals = ["10.00(50)", "3.00(20)", "7.0(5)"]
    result = check_report(*originals)
    assert result.consistent
    assert result.boundary_only is False
    assert result.formatted == originals
    assert -1 <= result.rho <= 1
    # Every witness value lies inside the reconstructed intervals:
    for row, quantity in zip(
        (parse_report_row(text) for text in originals),
        (result.witness["A"], result.witness["B"], result.witness["A-B"]),
    ):
        nominal_interval, sigma_interval = row.rounding_intervals()
        assert nominal_interval.contains(
            Decimal(str(quantity.nominal_value))
        )
        assert sigma_interval.contains(Decimal(str(quantity.std_dev)))


def test_inconsistent_nominal_values():
    result = check_report("10.00(50)", "3.00(20)", "8.0(5)")
    assert not result.consistent
    assert not result.boundary_only
    assert "A - B" in result.reason


def test_inconsistent_uncertainties_need_illegal_correlation():
    result = check_report("10.00(50)", "3.00(20)", "7.0(9)")
    assert not result.consistent
    assert not result.boundary_only
    assert "correlation coefficient" in result.reason


def test_nominal_values_only_meet_on_excluded_boundary():
    # Half-even: the A - B difference box reaches 1.05, which the
    # printed 1.1 row excludes (1.1 has an odd last digit):
    result = check_report("1.0(1)", "0.0(1)", "1.1(2)")
    assert not result.consistent
    assert result.boundary_only
    assert result.witness == {}


def test_uncertainties_only_meet_on_excluded_boundary():
    # Reachable sigma upper bound is 0.50 (excluded); the diff row unit
    # is 1, so its interval starts at the included tie 0.5:
    result = check_report("10.0(3)", "3.0(1)", "7(1)")
    assert not result.consistent
    assert result.boundary_only


@pytest.mark.parametrize(
    "rows",
    [
        # Perfectly anti-correlated limit, sigma_C = |sigma_A - sigma_B|:
        ["10.0(5)", "3.0(2)", "7.0(3)"],
        # Fully correlated limit, sigma_C = sigma_A + sigma_B:
        ["10.0(5)", "3.0(2)", "7.0(7)"],
    ],
)
def test_correlation_limits_are_feasible(rows):
    result = check_report(*rows)
    assert result.consistent, result.reason
    assert -1.0 - 1e-12 <= result.rho <= 1.0 + 1e-12


def test_zero_uncertainty_row():
    result = check_report("10.0(0)", "3.0(2)", "7.0(2)")
    assert result.consistent
    assert result.formatted == ["10.0(0)", "3.0(2)", "7.0(2)"]

    result_pm = check_report("10.0+/-0", "3.0+/-0.2", "7.0+/-0.2")
    assert result_pm.consistent
    assert result_pm.formatted == ["10.0+/-0", "3.0+/-0.2", "7.0+/-0.2"]


def test_signed_zero_nominal_is_preserved():
    # A printed "-0.0" must round-trip with its sign, and a printed
    # "0.0" must not acquire a negative sign:
    negative = check_report("0.0(1)", "0.0(1)", "-0.0(1)")
    assert negative.consistent, negative.reason
    assert negative.formatted == ["0.0(1)", "0.0(1)", "-0.0(1)"]

    positive = check_report("0.0(1)", "0.0(1)", "0.0(1)")
    assert positive.consistent, positive.reason
    assert positive.formatted == ["0.0(1)", "0.0(1)", "0.0(1)"]


def test_negative_nominals_and_exponents():
    # The original factored-exponent layout is preserved on reformat:
    result = check_report(
        "680(41)e-3", "100(5)e-3", "580(42)e-3"
    )
    assert result.consistent
    assert result.formatted == [
        "680(41)e-3",
        "100(5)e-3",
        "580(42)e-3",
    ]

    result = check_report("-1.23(4)", "2.00(5)", "-3.23(7)")
    assert result.consistent
    assert result.formatted == ["-1.23(4)", "2.00(5)", "-3.23(7)"]

    # Factored exponent over an explicit +/- body also round-trips:
    result = check_report(
        "(1.50+/-0.10)e+02", "(0.50+/-0.10)e+02", "(1.00+/-0.14)e+02"
    )
    assert result.consistent, result.reason
    assert result.formatted == [
        "(1.50+/-0.10)e+02",
        "(0.50+/-0.10)e+02",
        "(1.00+/-0.14)e+02",
    ]


def test_half_up_changes_boundary_membership():
    even = rounding_interval(Decimal("1.1"), Decimal("0.1"), HALF_EVEN)
    up = rounding_interval(Decimal("1.1"), Decimal("0.1"), HALF_UP)
    # Half-even excludes the lower tie (odd digit); half-up includes it:
    assert not even.lower_inclusive
    assert up.lower_inclusive


def test_generated_reports_are_always_consistent():
    rng = random.Random(20261113)
    for _ in range(200):
        value_a = rng.uniform(-50, 50)
        value_b = rng.uniform(-50, 50)
        sigma_a = rng.uniform(0.05, 5)
        sigma_b = rng.uniform(0.05, 5)
        rho = rng.uniform(-0.95, 0.95)

        variable_a, variable_b = report._build_correlated_pair(
            value_a, value_b, sigma_a, sigma_b, rho
        )
        variable_diff = variable_a - variable_b

        rows = []
        for digits, quantity in (
            (1, variable_a),
            (2, variable_b),
            (rng.choice((1, 2)), variable_diff),
        ):
            rows.append(format(quantity, ".%duS" % digits))

        result = check_report(*rows)
        assert result.consistent, rows
        assert result.formatted == rows
        assert -1.0 - 1e-9 <= result.rho <= 1.0 + 1e-9

        # The reconstructed pair must itself satisfy A - B numerically:
        reconstructed_diff = (
            result.witness["A"] - result.witness["B"]
        )
        assert reconstructed_diff.nominal_value == pytest.approx(
            result.witness["A-B"].nominal_value
        )


def test_witness_pair_has_requested_sigmas():
    result = check_report("10.00(50)", "3.00(20)", "7.0(5)")
    assert result.consistent
    row_a = parse_report_row("10.00(50)")
    row_b = parse_report_row("3.00(20)")
    _, sigma_interval_a = row_a.rounding_intervals()
    _, sigma_interval_b = row_b.rounding_intervals()
    assert sigma_interval_a.contains(
        Decimal(str(result.witness["A"].std_dev))
    )
    assert sigma_interval_b.contains(
        Decimal(str(result.witness["B"].std_dev))
    )
    # The constructed witnesses are regular numbers with uncertainty:
    assert isinstance(result.witness["A"], UFloat)
    assert isinstance(result.witness["B"], UFloat)
