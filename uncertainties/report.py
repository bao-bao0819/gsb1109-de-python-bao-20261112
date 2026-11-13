# coding=utf-8

"""
Consistency checks for rounded measurement reports.

An old report may only contain three rounded rows, A, B and A-B, each
giving a nominal value and a parenthesized uncertainty with its own
display precision::

    A = 10.00(50)
    B = 3.00(20)
    A - B = 7.0(6)

The underlying (unrounded) measurements can only take values in the
rounding intervals that correspond to the printed last digits.  The
three rows are compatible if some pair of measurements exists whose
values, standard deviations and error correlation reproduce all three
rows.

The correlation coefficient between the errors of A and B must satisfy
-1 <= rho <= 1.  For C = A - B, error propagation gives

    sigma_C**2 = sigma_A**2 + sigma_B**2 - 2*rho*sigma_A*sigma_B,

so the legal correlation coefficient is

    rho = (sigma_A**2 + sigma_B**2 - sigma_C**2)
          / (2*sigma_A*sigma_B),

equivalently, some (sigma_A, sigma_B, sigma_C) must satisfy

    |sigma_A - sigma_B| <= sigma_C <= sigma_A + sigma_B.

Half-unit rounding boundaries are handled exactly: with the default
half-even rounding rule (the rule used by Python's ``round()``), a tie
at a rounding boundary is given to the neighbor with an even last
digit, so the feasible intervals are half-open.  A report that would
only hold on a boundary that rounds *away* from the printed digit is
reported as inconsistent, with the actual rounding direction recorded.
"""

from __future__ import division

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP, localcontext
import math
import re
from warnings import catch_warnings, simplefilter

from .core import ufloat


HALF_EVEN = "half-even"
HALF_UP = "half-up"

_ROUNDING_MODES = {
    HALF_EVEN: ROUND_HALF_EVEN,
    HALF_UP: ROUND_HALF_UP,
}

# A decimal number without sign: "12", "12.", "12.34", ".34".  A
# trailing decimal point is meaningful: it pins the last digit to the
# units place (e.g. "31." has the same unit as "31").
_DECIMAL_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"

# A parenthesized uncertainty, possibly with its own decimal point:
_PAREN_UNCERT_RE = re.compile(
    r"""
    ^\s*
    (?P<sign>[+-])?
    (?P<main>"""
    + _DECIMAL_NUMBER
    + r""")
    (?:\((?P<uncert>"""
    + _DECIMAL_NUMBER
    + r""")\))?
    (?:\s*[eE]\s*(?P<exp>[+-]?\d+))?
    \s*$
    """,
    re.VERBOSE,
)

# A value followed by an explicit +/- (or ±) uncertainty:
_PM_RE = re.compile(
    r"""
    ^\s*
    (?P<sign>[+-])?
    (?P<main>"""
    + _DECIMAL_NUMBER
    + r""")
    (?:\s*[eE]\s*(?P<exp>[+-]?\d+))?
    \s*(?:\+/-|±|\+-|-\+)
    \s*(?P<uncert>"""
    + _DECIMAL_NUMBER
    + r""")
    (?:\s*[eE]\s*(?P<uncert_exp>[+-]?\d+))?
    \s*$
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Interval:
    """
    Closed or half-open interval of feasible values on a decimal grid.

    Endpoints are exact :class:`~decimal.Decimal` values so that
    half-unit rounding boundaries can be represented without floating
    point ambiguity.
    """

    lower: Decimal
    upper: Decimal
    lower_inclusive: bool
    upper_inclusive: bool

    def contains(self, value):
        if self.lower_inclusive:
            above_lower = value >= self.lower
        else:
            above_lower = value > self.lower
        if self.upper_inclusive:
            below_upper = value <= self.upper
        else:
            below_upper = value < self.upper
        return above_lower and below_upper

    @property
    def midpoint(self):
        return (self.lower + self.upper) / 2

    def __str__(self):
        left = "[" if self.lower_inclusive else "("
        right = "]" if self.upper_inclusive else ")"
        return "%s%s, %s%s" % (left, self.lower, self.upper, right)


@dataclass(frozen=True)
class ReportRow:
    """
    One parsed report row.

    ``nominal_unit`` and ``sigma_unit`` are the value of the last printed
    digit of the nominal value and of the uncertainty, respectively,
    after any exponent is applied.
    """

    text: str
    nominal: Decimal
    nominal_unit: Decimal
    sigma: Decimal
    sigma_unit: Decimal
    uncert_signif_digits: int
    shorthand: bool
    negative: bool
    exponent: int
    sign: str
    main_text: str
    uncert_text: object
    main_places: int
    uncert_places: int
    uncert_has_point: bool
    factored: bool
    exp_suffix: str

    def rounding_intervals(self, rounding_rule=HALF_EVEN):
        """
        Return the feasible ``(nominal interval, sigma interval)``.
        """
        return (
            rounding_interval(
                self.nominal, self.nominal_unit, rounding_rule
            ),
            rounding_interval(
                self.sigma,
                self.sigma_unit,
                rounding_rule,
                non_negative=True,
            ),
        )


def _unit_from_fraction(fraction):
    """Return the last-digit unit for a decimal fraction text (or 1)."""
    if fraction is None:
        return Decimal(1)
    return Decimal(1).scaleb(-len(fraction))


# A factored exponent over a parenthesized whole row, like
# "(1.5+/-0.1)e+02" or "(1.23(4))e-3":
_GLOBAL_EXP_RE = re.compile(
    r"""
    ^\s*
    \((?P<body>.*)\)
    (?:\s*[eE]\s*(?P<exp>[+-]?\d+))?
    \s*$
    """,
    re.VERBOSE,
)


def _fraction_places(number_text):
    """Number of fractional digits in a decimal text (0 if none)."""
    _, _, fraction = number_text.partition(".")
    return len(fraction)


def _exp_suffix(exponent_text):
    """Keep the original exponent spelling (e.g. "e-3", not "e-03")."""
    return "" if exponent_text is None else "e" + exponent_text


def parse_report_row(text):
    """
    Parse one report row like ``"12.34(142)"`` or ``"680(41)e-3"``.

    Rows using the explicit ``+/-`` or ``±`` notation and rows with a
    factored exponent like ``"(1.5+/-0.1)e+02"`` are also accepted.  The
    decimal text of the row is kept so that the last-digit units can be
    recovered exactly.

    Raises :class:`ValueError` if the row cannot be parsed.
    """
    if not isinstance(text, str):
        raise ValueError("Report rows must be strings, got %r" % (text,))

    text = text.strip()
    global_exp = 0
    global_match = _GLOBAL_EXP_RE.match(text)
    if global_match and (
        "+/-" in global_match.group("body")
        or "±" in global_match.group("body")
        or "+-" in global_match.group("body")
    ):
        # Factored exponent over a "+/-" body, e.g. "(1.5+/-0.1)e+02":
        body = global_match.group("body")
        global_exp = int(global_match.group("exp") or 0)
        suffix = _exp_suffix(global_match.group("exp"))
        parsed = parse_report_row(body)
        factor = Decimal(1).scaleb(global_exp)
        return ReportRow(
            text=text,
            nominal=parsed.nominal * factor,
            nominal_unit=parsed.nominal_unit * factor,
            sigma=parsed.sigma * factor,
            sigma_unit=parsed.sigma_unit * factor,
            uncert_signif_digits=parsed.uncert_signif_digits,
            shorthand=False,
            negative=parsed.negative,
            exponent=global_exp,
            sign=parsed.sign,
            main_text=parsed.main_text,
            uncert_text=parsed.uncert_text,
            main_places=parsed.main_places,
            uncert_places=parsed.uncert_places,
            uncert_has_point=parsed.uncert_has_point,
            factored=True,
            exp_suffix=suffix,
        )

    match = _PAREN_UNCERT_RE.match(text)
    shorthand = True

    if match:
        groups = match.groupdict()
        sign = groups["sign"] or ""
        main_text = groups["main"]
        uncert_text = groups["uncert"]
        exponent = int(groups["exp"] or 0)

        main_places = _fraction_places(main_text)
        main_unit = _unit_from_fraction(
            main_text.partition(".")[2] or None
        )
        nominal = Decimal(sign + main_text)

        if uncert_text is None:
            # An implicit uncertainty of 1 on the last printed digit:
            sigma = main_unit
            sigma_unit = main_unit
            uncert_signif_digits = 1
            uncert_places = main_places
        elif "." in uncert_text:
            # The uncertainty is given in full, with its own last digit:
            uncert_places = _fraction_places(uncert_text)
            sigma_unit = _unit_from_fraction(
                uncert_text.partition(".")[2] or None
            )
            sigma = Decimal(uncert_text)
            uncert_signif_digits = len(
                uncert_text.replace(".", "").lstrip("0")
            )
        else:
            # The integer uncertainty is aligned on the nominal last
            # digit: "12.34(142)" means an uncertainty of 1.42.
            sigma_unit = main_unit
            sigma = Decimal(uncert_text) * main_unit
            uncert_signif_digits = len(uncert_text.lstrip("0"))
            uncert_places = main_places
    else:
        match = _PM_RE.match(text)
        if not match:
            raise ValueError(
                "Cannot parse report row %r: expected a form like '12.34(5)'"
                % text
            )
        shorthand = False
        groups = match.groupdict()
        sign = groups["sign"] or ""
        main_text = groups["main"]
        exponent = int(groups["exp"] or 0)
        main_places = _fraction_places(main_text)
        main_unit = _unit_from_fraction(
            main_text.partition(".")[2] or None
        )
        nominal = Decimal(sign + main_text)

        uncert_text = groups["uncert"]
        uncert_places = _fraction_places(uncert_text)
        sigma_unit = _unit_from_fraction(
            uncert_text.partition(".")[2] or None
        )
        sigma = Decimal(uncert_text)
        uncert_signif_digits = len(
            uncert_text.replace(".", "").lstrip("0")
        )
        if groups["uncert_exp"] is not None:
            uncert_factor = Decimal(1).scaleb(int(groups["uncert_exp"]))
            sigma *= uncert_factor
            sigma_unit *= uncert_factor

    factor = Decimal(1).scaleb(exponent)
    nominal *= factor
    main_unit *= factor
    if shorthand:
        # The factored exponent applies to the whole shorthand row,
        # including its parenthesized uncertainty:
        sigma *= factor
        sigma_unit *= factor

    return ReportRow(
        text=text,
        nominal=nominal,
        nominal_unit=main_unit,
        sigma=sigma,
        sigma_unit=sigma_unit,
        uncert_signif_digits=uncert_signif_digits,
        shorthand=shorthand,
        negative=(sign == "-" and nominal == 0),
        exponent=exponent,
        sign=sign,
        main_text=main_text,
        uncert_text=uncert_text,
        main_places=main_places,
        uncert_places=uncert_places,
        uncert_has_point=(uncert_text is not None and "." in uncert_text),
        factored=False,
        exp_suffix=_exp_suffix(groups["exp"]),
    )


def rounding_interval(value, unit, rounding_rule=HALF_EVEN, non_negative=False):
    """
    Return the interval of numbers that round to ``value``.

    ``value`` must lie on the decimal grid whose last digit is ``unit``
    (so ``value / unit`` is an integer).  Ties at the half-unit
    boundaries are assigned according to ``rounding_rule``:

    - ``HALF_EVEN`` (default): ties go to the neighbor with an even last
      digit, like Python's ``round()``;
    - ``HALF_UP``: ties go to the neighbor with the larger magnitude.

    Whether each boundary actually belongs to the interval is decided by
    rounding the boundary itself with the requested rule, so the result
    is exact for positive and negative grid values alike.

    When ``non_negative`` is True, the interval is clipped at zero, as
    required for a standard deviation.
    """
    if rounding_rule not in _ROUNDING_MODES:
        raise ValueError(
            "Unknown rounding rule %r (expected %r or %r)"
            % (rounding_rule, HALF_EVEN, HALF_UP)
        )

    rounding = _ROUNDING_MODES[rounding_rule]
    half = unit / 2
    lower = value - half
    upper = value + half

    with localcontext() as context:
        context.prec = 60
        lower_inclusive = lower.quantize(unit, rounding=rounding) == value
        upper_inclusive = upper.quantize(unit, rounding=rounding) == value

    if non_negative and lower < 0 <= value:
        # A standard deviation cannot be negative: a zero-uncertainty
        # row has its physical interval clipped at 0.
        lower = Decimal(0)
        lower_inclusive = True

    return Interval(lower, upper, lower_inclusive, upper_inclusive)


def _intersection(left, right):
    """
    Return the intersection of two intervals, or ``None``.

    A touching point is included only when both intervals include it;
    otherwise the rows would rely on a boundary that rounds away from a
    printed digit.
    """
    if left.lower >= right.lower:
        lower, lower_inclusive = left.lower, left.lower_inclusive
        if left.lower == right.lower:
            lower_inclusive = left.lower_inclusive and right.lower_inclusive
    else:
        lower, lower_inclusive = right.lower, right.lower_inclusive

    if left.upper <= right.upper:
        upper, upper_inclusive = left.upper, left.upper_inclusive
        if left.upper == right.upper:
            upper_inclusive = left.upper_inclusive and right.upper_inclusive
    else:
        upper, upper_inclusive = right.upper, right.upper_inclusive

    if lower > upper:
        return None
    if lower == upper and not (lower_inclusive and upper_inclusive):
        return None
    return Interval(lower, upper, lower_inclusive, upper_inclusive)


def _difference_interval(interval_a, interval_b):
    """Return interval_a - interval_b (Minkowski difference)."""
    return Interval(
        interval_a.lower - interval_b.upper,
        interval_a.upper - interval_b.lower,
        interval_a.lower_inclusive and interval_b.upper_inclusive,
        interval_a.upper_inclusive and interval_b.lower_inclusive,
    )


def _is_negative_zero_row(row):
    """Whether the row prints a signed negative zero (e.g. "-0.0")."""
    return row.nominal == 0 and row.negative


def _signed_zero_interval(interval, row):
    """
    Restrict a zero-grid interval to the sign printed on the row.

    A row like "-0.0(3)" can only come from a non-positive underlying
    value, whereas "0.0(3)" only from a non-negative one.  Non-zero
    rows keep their full rounding interval.
    """
    if row.nominal != 0:
        return interval
    if _is_negative_zero_row(row):
        return Interval(
            interval.lower, Decimal(0), interval.lower_inclusive, True
        )
    return Interval(
        Decimal(0), interval.upper, True, interval.upper_inclusive
    )


def _signed_float(value, row):
    """Convert a Decimal witness to float, preserving a printed -0 sign."""
    result = float(value)
    if result == 0.0 and _is_negative_zero_row(row):
        return math.copysign(0.0, -1.0)
    return result


def _witness_for_difference(interval_a, interval_b, target):
    """
    Return ``(a, b)`` with a in interval_a, b in interval_b and a-b=target.

    Writing b = a - target turns the condition on b into an interval on
    a, so a witness exists exactly when interval_a intersects
    interval_b shifted by target.
    """
    shifted_b = Interval(
        interval_b.lower + target,
        interval_b.upper + target,
        interval_b.lower_inclusive,
        interval_b.upper_inclusive,
    )
    feasible_a = _intersection(interval_a, shifted_b)
    if feasible_a is None:
        return None
    value_a = (
        feasible_a.midpoint
        if feasible_a.lower != feasible_a.upper
        else feasible_a.lower
    )
    value_b = value_a - target
    if not interval_b.contains(value_b):
        return None
    return value_a, value_b


def _witness_for_sigmas(interval_a, interval_b, target_c):
    """
    Return ``(sigma_a, sigma_b)`` with |sigma_a-sigma_b| <= target_c
    <= sigma_a + sigma_b, inside the two sigma intervals.

    A pair exists when some sigma_a can be paired with a sigma_b in
    interval_b.  Existence of such a sigma_b bounds sigma_a by:

        sigma_a >= target_c - interval_b.upper   (sum constraint)
        sigma_a <= interval_b.upper + target_c   (difference constraint)
        sigma_a >= interval_b.lower - target_c   (difference constraint)

    Once sigma_a is chosen, sigma_b must additionally lie in
    [sigma_a - target_c, sigma_a + target_c] and above
    target_c - sigma_a.
    """
    huge = Decimal("1e100")

    sum_constraint = Interval(
        target_c - interval_b.upper,
        huge,
        interval_b.upper_inclusive,
        True,
    )
    diff_upper = Interval(
        -huge,
        interval_b.upper + target_c,
        True,
        interval_b.upper_inclusive,
    )
    diff_lower = Interval(
        interval_b.lower - target_c,
        huge,
        interval_b.lower_inclusive,
        True,
    )

    feasible_a = interval_a
    for constraint in (sum_constraint, diff_upper, diff_lower):
        feasible_a = _intersection(feasible_a, constraint)
        if feasible_a is None:
            return None

    sigma_a = (
        feasible_a.midpoint
        if feasible_a.lower != feasible_a.upper
        else feasible_a.lower
    )

    sigma_b_lower = max(
        interval_b.lower, sigma_a - target_c, target_c - sigma_a
    )
    sigma_b_upper = min(interval_b.upper, sigma_a + target_c)
    if sigma_b_lower > sigma_b_upper:
        return None

    lower_inclusive = (
        interval_b.lower_inclusive
        or sigma_b_lower != interval_b.lower
    )
    upper_inclusive = (
        interval_b.upper_inclusive
        or sigma_b_upper != interval_b.upper
    )
    feasible_b = Interval(
        sigma_b_lower, sigma_b_upper, lower_inclusive, upper_inclusive
    )
    if not interval_b.contains(feasible_b.midpoint):
        return None
    sigma_b = feasible_b.midpoint

    if (
        interval_a.contains(sigma_a)
        and interval_b.contains(sigma_b)
        and abs(sigma_a - sigma_b) <= target_c <= sigma_a + sigma_b
    ):
        return sigma_a, sigma_b
    return None


def _correlation(sigma_a, sigma_b, sigma_c):
    """
    Return the correlation coefficient required by the three sigmas.

    ``None`` is returned when the correlation is undefined (one of the
    standard deviations is zero).
    """
    if sigma_a == 0 or sigma_b == 0:
        return None
    rho = (sigma_a**2 + sigma_b**2 - sigma_c**2) / (2 * sigma_a * sigma_b)
    return float(rho)


def _requantize(value, unit, rounding_rule):
    """Round a float witness back onto its decimal reporting grid."""
    rounding = _ROUNDING_MODES[rounding_rule]
    with localcontext() as context:
        context.prec = 60
        return Decimal(str(value)).quantize(unit, rounding=rounding)


def _fixed(decimal_value, places, trailing_point=False):
    """
    Format a Decimal with exactly ``places`` fractional digits.

    A value with no fractional part keeps a trailing decimal point when
    ``trailing_point`` is True (e.g. "31." rather than "31").
    """
    quantized = decimal_value.quantize(Decimal(1).scaleb(-places))
    text = format(quantized, "f")
    if places == 0:
        if trailing_point and "." not in text:
            text += "."
        elif not trailing_point and "." in text:
            text = text.rstrip(".")
    return text


def _has_trailing_point(number_text):
    return number_text is not None and number_text.endswith(".")


def _match_leading_point(text, original):
    """Drop a leading zero when the original text starts with a dot."""
    if original is not None and original.lstrip("+-").startswith("."):
        return text.replace("0.", ".", 1)
    return text


def _render_row(row, nominal, sigma):
    """
    Rebuild the printed text for a row from grid-rounded values.

    The original layout (shorthand vs ``+/-``, implicit uncertainty,
    trailing decimal point, factored exponent, number of fractional
    digits) is reproduced exactly, so a witness that rounds onto the
    row's grids yields the original text.
    """
    factor = Decimal(1).scaleb(row.exponent)
    nominal_mantissa = (nominal / factor).quantize(
        Decimal(1).scaleb(-row.main_places)
    )
    nominal_text = _match_leading_point(
        _fixed(
            nominal_mantissa,
            row.main_places,
            trailing_point=_has_trailing_point(row.main_text),
        ),
        row.main_text,
    )

    if row.shorthand:
        main_unit_mantissa = row.nominal_unit / factor
        sigma_mantissa = (sigma / factor).quantize(
            Decimal(1).scaleb(-row.uncert_places)
        )
        if row.uncert_text is None:
            # No parenthesized uncertainty was printed: none is added,
            # so the row keeps its implicit one-last-digit form.
            return nominal_text + row.exp_suffix
        if row.uncert_has_point:
            # Fully written uncertainty with its own decimal point
            # (e.g. "1.2(3.4)"), possibly with a trailing point:
            uncert_text = _match_leading_point(
                _fixed(
                    sigma_mantissa,
                    row.uncert_places,
                    trailing_point=_has_trailing_point(row.uncert_text),
                ),
                row.uncert_text,
            )
        else:
            # Integer parenthesized uncertainty aligned on the nominal
            # last digit (e.g. "10.00(50)", "10.0(0)"):
            count = (
                sigma_mantissa / main_unit_mantissa
            ).quantize(Decimal(1))
            uncert_text = format(count, "f")
        return "%s(%s)%s" % (
            nominal_text,
            uncert_text,
            row.exp_suffix,
        )

    sigma_mantissa = (sigma / factor).quantize(
        Decimal(1).scaleb(-row.uncert_places)
    )
    uncert_text = _match_leading_point(
        _fixed(
            sigma_mantissa,
            row.uncert_places,
            trailing_point=_has_trailing_point(row.uncert_text),
        ),
        row.uncert_text,
    )
    body = "%s+/-%s" % (nominal_text, uncert_text)
    if row.factored:
        return "(%s)%s" % (body, row.exp_suffix)
    return body + row.exp_suffix


def _build_correlated_pair(value_a, value_b, sigma_a, sigma_b, rho):
    """
    Build two correlated UFloats with the given nominals and sigmas.

    Uses a pure-Python Cholesky construction so that NumPy is not
    required: A = a0 + sigma_A*X and
    B = b0 + sigma_B*(rho*X + sqrt(1-rho**2)*Y), with X, Y independent.
    """
    with catch_warnings():
        simplefilter("ignore")
        if rho is None:
            # One sigma is zero: the correlation is undefined and the
            # zero-sigma quantity is exact.
            return ufloat(value_a, sigma_a, "A"), ufloat(value_b, sigma_b, "B")

        independent_x = ufloat(0.0, 1.0, "X")
        independent_y = ufloat(0.0, 1.0, "Y")
        variable_a = value_a + sigma_a * independent_x
        variable_b = (
            value_b
            + sigma_b * rho * independent_x
            + sigma_b * math.sqrt(max(0.0, 1.0 - rho * rho)) * independent_y
        )
        return variable_a, variable_b


@dataclass
class ReportConsistency:
    """
    Result of :func:`check_report`.

    ``consistent`` is True exactly when one pair of underlying
    measurements can reproduce all three rounded rows.  When the rows can
    only meet on a half-unit rounding boundary that rounds away from the
    printed digits, ``consistent`` is False and ``boundary_only`` is
    True.  When consistent, ``witness`` gives concrete A and B values
    and ``formatted`` gives strings that round back to every original
    row.
    """

    consistent: bool
    reason: str
    boundary_only: bool
    rho: object
    intervals: dict
    witness: dict
    formatted: list


def check_report(row_a, row_b, row_diff, rounding_rule=HALF_EVEN):
    """
    Check whether three rounded rows can come from one pair of measurements.

    ``row_a``, ``row_b`` and ``row_diff`` are the printed rows for A, B
    and A - B, as strings (e.g. ``"10.00(50)"``), or parsed
    :class:`ReportRow` objects.  The feasible rounding intervals are
    reconstructed from the last printed digit of each value, using
    ``rounding_rule`` for the half-unit boundaries, and two conditions
    are checked together:

    1. the nominal values must satisfy A - B;
    2. the standard deviations must admit a correlation coefficient in
       [-1, 1].

    A :class:`ReportConsistency` is returned.  When the rows are
    compatible, it contains a witness pair of measurements and
    reformatted strings that agree with every original row.
    """
    parsed_a = row_a if isinstance(row_a, ReportRow) else parse_report_row(row_a)
    parsed_b = row_b if isinstance(row_b, ReportRow) else parse_report_row(row_b)
    parsed_diff = (
        row_diff if isinstance(row_diff, ReportRow) else parse_report_row(row_diff)
    )
    rows = [parsed_a, parsed_b, parsed_diff]

    (nom_a, sig_a), (nom_b, sig_b), (nom_d, sig_d) = (
        row.rounding_intervals(rounding_rule) for row in rows
    )

    result_kwargs = {
        "consistent": False,
        "reason": "",
        "boundary_only": False,
        "rho": None,
        "intervals": {
            "A": {"nominal": nom_a, "sigma": sig_a},
            "B": {"nominal": nom_b, "sigma": sig_b},
            "A-B": {"nominal": nom_d, "sigma": sig_d},
        },
        "witness": {},
        "formatted": [],
    }

    # 1. Nominal-value feasibility: the difference box of the A and B
    # nominal intervals must intersect the printed A - B interval.  A
    # printed negative zero (e.g. "-0.0") restricts the witness to the
    # non-positive part of its zero-grid interval.
    restricted_nom_a = _signed_zero_interval(nom_a, parsed_a)
    restricted_nom_b = _signed_zero_interval(nom_b, parsed_b)
    restricted_nom_d = _signed_zero_interval(nom_d, parsed_diff)

    diff_box = _difference_interval(restricted_nom_a, restricted_nom_b)
    nominal_overlap = _intersection(diff_box, restricted_nom_d)
    if nominal_overlap is None:
        closed_overlap = _closed_box_overlap(diff_box, restricted_nom_d)
        if closed_overlap:
            result_kwargs.update(
                reason=(
                    "The nominal values only meet on a half-unit rounding"
                    " boundary that rounds away from the printed last digit"
                    " (tie not awarded to this row)."
                ),
                boundary_only=True,
            )
        else:
            result_kwargs.update(
                reason=(
                    "The nominal rounding intervals cannot satisfy A - B:"
                    " feasible A - B is %s but the report prints %s."
                    % (diff_box, nom_d)
                )
            )
        return ReportConsistency(**result_kwargs)

    target_diff = (
        nominal_overlap.midpoint
        if nominal_overlap.lower != nominal_overlap.upper
        else nominal_overlap.lower
    )
    if _is_negative_zero_row(parsed_diff):
        target_diff = min(target_diff, Decimal(0))
    nominal_witness = _witness_for_difference(
        restricted_nom_a, restricted_nom_b, target_diff
    )
    if nominal_witness is None:
        result_kwargs.update(
            reason=(
                "The nominal values only meet on a half-unit rounding"
                " boundary that rounds away from a printed last digit."
            ),
            boundary_only=True,
        )
        return ReportConsistency(**result_kwargs)

    value_a, value_b = nominal_witness

    # 2. Standard-deviation feasibility: over the sigma boxes, the
    # reachable sigma_(A-B) values fill
    # [min|sigma_A-sigma_B|, sigma_A+sigma_B].
    reachable = _reachable_sigma_interval(sig_a, sig_b)
    sigma_overlap = _intersection(reachable, sig_d)
    if sigma_overlap is None:
        if _closed_box_overlap(reachable, sig_d):
            result_kwargs.update(
                reason=(
                    "The uncertainties only meet on a half-unit rounding"
                    " boundary that rounds away from a printed uncertainty"
                    " digit (tie not awarded to this row)."
                ),
                boundary_only=True,
            )
        else:
            result_kwargs.update(
                reason=(
                    "No correlation coefficient in [-1, 1] can produce the"
                    " printed uncertainties: the reachable sigma_(A-B)"
                    " interval is %s but the report prints %s."
                    % (reachable, sig_d)
                )
            )
        return ReportConsistency(**result_kwargs)

    target_sigma_c = (
        sigma_overlap.midpoint
        if sigma_overlap.lower != sigma_overlap.upper
        else sigma_overlap.lower
    )
    sigma_witness = _witness_for_sigmas(sig_a, sig_b, target_sigma_c)
    if sigma_witness is None:
        result_kwargs.update(
            reason=(
                "The uncertainties only meet on a half-unit rounding"
                " boundary that rounds away from a printed uncertainty digit."
            ),
            boundary_only=True,
        )
        return ReportConsistency(**result_kwargs)

    sigma_a_value, sigma_b_value = sigma_witness
    sigma_d_value = target_sigma_c
    rho = _correlation(sigma_a_value, sigma_b_value, sigma_d_value)
    if rho is not None and not (-1.0 - 1e-12 <= rho <= 1.0 + 1e-12):
        result_kwargs.update(
            reason=(
                "The required correlation coefficient %.6f is outside"
                " the legal [-1, 1] range." % rho
            )
        )
        return ReportConsistency(**result_kwargs)
    if rho is not None:
        rho = max(-1.0, min(1.0, rho))

    with catch_warnings():
        simplefilter("ignore")
        witness_a, witness_b = _build_correlated_pair(
            _signed_float(value_a, parsed_a),
            _signed_float(value_b, parsed_b),
            float(sigma_a_value),
            float(sigma_b_value),
            rho,
        )
        witness_diff = witness_a - witness_b

        formatted = []
        quantities = (witness_a, witness_b, witness_diff)
        for row, quantity in zip(rows, quantities):
            # Grid-rounded Decimal values are rendered with the row's
            # own layout instead of the library formatter, so that an
            # automatically chosen exponent cannot change the text:
            rounded_nominal = _requantize(
                quantity.nominal_value, row.nominal_unit, rounding_rule
            )
            rounded_sigma = _requantize(
                quantity.std_dev, row.sigma_unit, rounding_rule
            )
            if rounded_nominal != row.nominal or rounded_sigma != row.sigma:
                result_kwargs.update(
                    reason=(
                        "The witness %s does not reformat to the original row"
                        " %r." % (quantity, row.text)
                    )
                )
                return ReportConsistency(**result_kwargs)

            formatted.append(
                _render_row(row, rounded_nominal, rounded_sigma)
            )

    result_kwargs.update(
        consistent=True,
        reason="",
        rho=rho,
        witness={
            "A": witness_a,
            "B": witness_b,
            "A-B": witness_diff,
        },
        formatted=formatted,
    )
    return ReportConsistency(**result_kwargs)


def _closed_box_overlap(left, right):
    """Whether the closures of two intervals touch or overlap."""
    return not (
        left.upper < right.lower or right.upper < left.lower
    )


def _reachable_sigma_interval(interval_a, interval_b):
    """
    Return the sigma_C values reachable from the two sigma boxes.

    With C = A - B and rho in [-1, 1]:

        |sigma_A - sigma_B| <= sigma_C <= sigma_A + sigma_B.
    """
    if interval_a.upper < interval_b.lower:
        lower = interval_b.lower - interval_a.upper
        lower_inclusive = (
            interval_b.lower_inclusive and interval_a.upper_inclusive
        )
    elif interval_b.upper < interval_a.lower:
        lower = interval_a.lower - interval_b.upper
        lower_inclusive = (
            interval_a.lower_inclusive and interval_b.upper_inclusive
        )
    else:
        lower = Decimal(0)
        lower_inclusive = True

    upper = interval_a.upper + interval_b.upper
    upper_inclusive = interval_a.upper_inclusive and interval_b.upper_inclusive
    return Interval(lower, upper, lower_inclusive, upper_inclusive)
