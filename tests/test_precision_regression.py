"""Regression tests for sub-microsecond precision loss.

Receiver timestamps may carry more than six fractional digits. Arrow stores
everything on a :class:`datetime.datetime`, whose finest granularity is the
microsecond (``10**-6`` seconds), so the ``S`` token rounds the *rational*
fractional second described by the source string to six digits.

Every assertion in this module is anchored on the exact rational value of the
source string, computed with :class:`fractions.Fraction` instead of floating
point. The tests distinguish three situations that look identical once the
strings have become datetimes:

1. true simultaneity -- two strings denote the same rational instant (often via
   different UTC offsets), so merging them is correct;
2. representational precision collisions -- two distinct rational instants round
   to the same microsecond and become indistinguishable datetimes;
3. reordered rows -- the source row order inside a collision bucket, or the
   wall-clock order across offsets, can no longer be derived from the datetime.

Rounding boundary pairs are built at the second, day and UTC-offset day-change
edges, including the half-to-even tie at the seventh fractional digit.
"""

import re
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from typing import List, Tuple

import pytest

import arrow
from arrow.formatter import DateTimeFormatter
from arrow.parser import DateTimeParser

ISO_PARTS_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})[T ]"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r"(?:[.,](?P<fraction>\d+))?"
    r"(?P<tz>Z|(?P<sign>[+-])(?P<offset_hour>\d{2}):?(?P<offset_minute>\d{2}))?$"
)

_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
MICROSECONDS_PER_SECOND = 1_000_000


def exact_fractional_second(fraction: str) -> Fraction:
    """Return the exact rational value of a fractional-second digit string."""
    return Fraction(int(fraction), 10 ** len(fraction))


def exact_microsecond(fraction: str) -> int:
    """Round the rational fractional second to whole microseconds.

    ``Fraction.__round__`` implements round-half-to-even on the exact value,
    which is the mathematically correct mapping of the source string.
    """
    return round(exact_fractional_second(fraction) * MICROSECONDS_PER_SECOND)


def exact_rational_instant(text: str) -> Fraction:
    """Return the exact UTC instant of an ISO string in rational seconds.

    The offset is applied to the wall components, so strings written with
    different UTC offsets can be compared as pure rational numbers without any
    floating-point involvement.
    """
    match = ISO_PARTS_RE.match(text)
    assert match is not None, f"not an ISO-like timestamp: {text!r}"
    groups = match.groupdict()

    day_ordinal = date(
        int(groups["year"]), int(groups["month"]), int(groups["day"])
    ).toordinal()
    whole_seconds = (
        (day_ordinal - _EPOCH_ORDINAL) * 86400
        + int(groups["hour"]) * 3600
        + int(groups["minute"]) * 60
        + int(groups["second"])
    )
    instant = Fraction(whole_seconds)
    if groups["fraction"] is not None:
        instant += exact_fractional_second(groups["fraction"])

    if groups["tz"] is not None and groups["tz"] != "Z":
        offset_seconds = (
            int(groups["offset_hour"]) * 3600
            + int(groups["offset_minute"]) * 60
        )
        if groups["sign"] == "+":
            instant -= offset_seconds
        else:
            instant += offset_seconds

    return instant


class TestRationalOracle:
    """The Fraction-based oracle must order strings exactly as written."""

    @pytest.mark.parametrize(
        "fraction, expected",
        [
            ("1", Fraction(1, 10)),
            ("123456", Fraction(123456, 1_000_000)),
            ("1234565", Fraction(1234565, 10_000_000)),
            ("12345651", Fraction(12345651, 100_000_000)),
        ],
    )
    def test_fractional_second_is_exact(self, fraction, expected):
        assert exact_fractional_second(fraction) == expected

    def test_float_cannot_hold_the_seventh_digit_exactly(self):
        rational = exact_fractional_second("1234565")
        # A reduced fraction only fits in binary floating point when its
        # denominator is a power of two; here the factor of five survives.
        assert rational.denominator % 5 == 0
        assert rational != Fraction(float(rational))

    @pytest.mark.parametrize(
        "fraction, expected_microsecond",
        [
            ("9876535", 987654),  # tie, rounds up to the even digit
            ("9876545", 987654),  # tie, rounds down to the even digit
            ("0000005", 0),  # tie next to zero, zero is even
            ("0000015", 2),  # tie, rounds up to the even digit
            ("1234564999", 123456),  # strictly below the midpoint
            ("1234565001", 123457),  # strictly above the midpoint
            ("9999995", 1_000_000),  # exact tie rounds up and overflows
        ],
    )
    def test_exact_half_to_even_rounding(self, fraction, expected_microsecond):
        assert exact_microsecond(fraction) == expected_microsecond

    def test_distinct_strings_keep_strict_rational_ordering(self):
        earlier = "2013-01-01T12:30:45.1234555"
        later = "2013-01-01T12:30:45.1234564"
        assert exact_rational_instant(earlier) < exact_rational_instant(later)

    def test_rational_oracle_sees_through_timezone_offsets(self):
        utc_string = "2012-12-31T19:00:00.0000000Z"
        offset_string = "2013-01-01T00:00:00.0000000+05:00"
        assert exact_rational_instant(utc_string) == exact_rational_instant(
            offset_string
        )


@pytest.mark.usefixtures("dt_parser")
class TestParserRoundingAgainstOracle:
    """Parsing agrees with the rational oracle whenever only seven digits decide."""

    @pytest.mark.parametrize(
        "fraction, expected_microsecond",
        [
            ("9876535", 987654),
            ("9876545", 987654),
            ("0000005", 0),
            ("0000015", 2),
            ("9876535001", 987654),
            ("9876544999", 987654),
            ("1234574999", 123457),
        ],
    )
    def test_parse_iso_matches_exact_rational(
        self, fraction, expected_microsecond
    ):
        text = f"2013-01-01T12:30:45.{fraction}"
        parsed = self.parser.parse_iso(text)
        assert parsed.microsecond == expected_microsecond
        assert parsed.microsecond == exact_microsecond(fraction)

    def test_explicit_format_and_iso_agree(self):
        text = "2013-01-01 12:30:45.9876535"
        fmt = "YYYY-MM-DD HH:mm:ss.S"
        expected = datetime(2013, 1, 1, 12, 30, 45, 987654)
        assert self.parser.parse(text, fmt) == expected
        assert self.parser.parse_iso(text) == expected

    def test_eighth_digit_tail_is_ignored_by_parser_rule(self):
        # The parser only inspects the seventh digit, whereas the exact oracle
        # rounds the full rational value. A trailing "1" pushes the true value
        # above the midpoint, but parsing still rounds down. This is the precise
        # point where the microsecond mapping stops representing the source.
        tie_down = "2013-01-01T12:30:45.12345650"
        above_tie = "2013-01-01T12:30:45.12345651"

        assert exact_microsecond("12345650") == 123456
        assert exact_microsecond("12345651") == 123457

        parsed_tie = self.parser.parse_iso(tie_down)
        parsed_above = self.parser.parse_iso(above_tie)
        assert parsed_tie.microsecond == 123456
        assert parsed_above.microsecond == 123456
        assert parsed_tie == parsed_above

    def test_rounded_mapping_is_monotonic_never_inverts(self):
        # Every pair that is strictly ordered by the rational oracle is only
        # collapsed to equality when parsed; the rounded mapping never swaps a
        # strict "before" into a strict "after".
        fractions = [
            "1234555",
            "1234564",
            "12345650",
            "12345651",
            "1234574",
            "1234575",
        ]
        texts = [f"2013-01-01T12:30:45.{fraction}" for fraction in fractions]
        parsed = [self.parser.parse_iso(text) for text in texts]

        for left, right in zip(parsed, parsed[1:]):
            assert left <= right
        strict_pairs = sum(
            left < right for left, right in zip(parsed, parsed[1:])
        )
        assert strict_pairs < len(fractions) - 1


@pytest.mark.usefixtures("dt_parser")
class TestCarryBoundaryPairs:
    """Rounding pairs sitting exactly on the second and day rollover edges."""

    def test_second_boundary_pair(self):
        carried = self.parser.parse_iso("2013-01-01T12:30:45.9999995")
        not_carried = self.parser.parse_iso("2013-01-01T12:30:45.9999994999")

        assert carried == datetime(2013, 1, 1, 12, 30, 46, 0)
        assert not_carried == datetime(2013, 1, 1, 12, 30, 45, 999999)
        assert exact_rational_instant(
            "2013-01-01T12:30:45.9999994999"
        ) < exact_rational_instant("2013-01-01T12:30:45.9999995")

    def test_minute_boundary_pair(self):
        assert self.parser.parse_iso(
            "2013-01-01T12:30:59.9999996"
        ) == datetime(2013, 1, 1, 12, 31, 0, 0)
        assert self.parser.parse_iso(
            "2013-01-01T12:30:59.9999994"
        ) == datetime(2013, 1, 1, 12, 30, 59, 999999)

    def test_day_boundary_pair(self):
        last_moment = self.parser.parse_iso("2013-01-01T23:59:59.9999999")
        first_moment = self.parser.parse_iso("2013-01-02T00:00:00.0000000")
        assert last_moment == datetime(2013, 1, 2, 0, 0, 0, 0)
        assert first_moment == datetime(2013, 1, 2, 0, 0, 0, 0)
        assert last_moment == first_moment

    def test_month_and_year_boundaries(self):
        assert self.parser.parse_iso(
            "2013-01-31T23:59:59.9999999"
        ) == datetime(2013, 2, 1, 0, 0, 0)
        assert self.parser.parse_iso(
            "2013-12-31T23:59:59.9999999"
        ) == datetime(2014, 1, 1, 0, 0, 0)


@pytest.mark.usefixtures("dt_parser")
class TestTimezoneDayBoundary:
    """Carries and collisions where a UTC offset changes the calendar day."""

    def test_carry_pushes_local_date_past_midnight_behind_utc(self):
        # Local time is five hours behind UTC; the carry flips the local date.
        parsed = self.parser.parse_iso("2013-01-01T23:59:59.9999999-05:00")
        assert parsed == datetime(
            2013, 1, 2, 0, 0, 0, tzinfo=timezone(timedelta(hours=-5))
        )
        assert parsed.astimezone(timezone.utc) == datetime(
            2013, 1, 2, 5, 0, 0, tzinfo=timezone.utc
        )

    def test_carry_pushes_local_date_ahead_of_utc(self):
        # Local time is five hours ahead: local midnight on Jan 2 is still
        # Jan 1 in UTC, so the wall date and UTC date disagree.
        parsed = self.parser.parse_iso("2013-01-01T23:59:59.9999995+05:00")
        assert parsed == datetime(
            2013, 1, 2, 0, 0, 0, tzinfo=timezone(timedelta(hours=5))
        )
        assert parsed.astimezone(timezone.utc) == datetime(
            2013, 1, 1, 19, 0, 0, tzinfo=timezone.utc
        )

    def test_true_simultaneous_instants_across_day_boundary(self):
        utc_string = "2012-12-31T19:00:00.0000000Z"
        offset_string = "2013-01-01T00:00:00.0000000+05:00"

        utc_dt = self.parser.parse_iso(utc_string)
        offset_dt = self.parser.parse_iso(offset_string)

        # Same rational instant: equality is genuine, not a precision collision.
        assert exact_rational_instant(utc_string) == exact_rational_instant(
            offset_string
        )
        assert utc_dt == offset_dt
        # Yet the wall dates are different and cannot be made identical.
        assert utc_dt.date() != offset_dt.date()

    def test_wall_order_and_utc_instant_order_conflict(self):
        ahead = "2013-01-01T23:00:00.1234565+05:00"
        utc_later_wall = "2013-01-01T20:00:00.1234564Z"

        ahead_dt = self.parser.parse_iso(ahead)
        utc_dt = self.parser.parse_iso(utc_later_wall)

        # Reading the wall components says +05:00 happens later; the rational
        # UTC instant says it happens two hours earlier.
        assert ahead_dt.replace(tzinfo=None) > utc_dt.replace(tzinfo=None)
        assert exact_rational_instant(ahead) < exact_rational_instant(
            utc_later_wall
        )
        assert ahead_dt < utc_dt


@pytest.mark.usefixtures("dt_parser")
class TestPrecisionCollisions:
    """Distinct rational instants that become one datetime."""

    def test_same_second_collision_pair(self):
        earlier = "2013-01-01T12:30:45.1234555"
        later = "2013-01-01T12:30:45.1234564"

        earlier_dt = self.parser.parse_iso(earlier)
        later_dt = self.parser.parse_iso(later)

        assert exact_rational_instant(earlier) < exact_rational_instant(later)
        assert earlier_dt == later_dt == datetime(2013, 1, 1, 12, 30, 45, 123456)
        assert len({earlier_dt, later_dt}) == 1

    def test_collision_through_second_carry(self):
        before = "2013-01-01T12:30:45.9999999"
        after = "2013-01-01T12:30:46.0000001"

        before_dt = self.parser.parse_iso(before)
        after_dt = self.parser.parse_iso(after)

        assert exact_rational_instant(before) < exact_rational_instant(after)
        assert before_dt == after_dt == datetime(2013, 1, 1, 12, 30, 46, 0)

    def test_collision_through_tie_tail_divergence(self):
        tie = "2013-01-01T12:30:45.12345650"
        above = "2013-01-01T12:30:45.12345651"

        # The exact oracle keeps them apart; parsing does not.
        assert exact_microsecond("12345650") != exact_microsecond("12345651")
        assert self.parser.parse_iso(tie) == self.parser.parse_iso(above)

    def test_true_equality_is_distinguishable_from_collision(self):
        # Precision collision: different rational instants.
        collision_a = exact_rational_instant("2013-01-01T12:30:45.1234555")
        collision_b = exact_rational_instant("2013-01-01T12:30:45.1234564")
        assert collision_a != collision_b

        # True simultaneity: identical rational instants via offsets.
        simultaneous_a = exact_rational_instant("2013-01-01T12:00:00.0000000Z")
        simultaneous_b = exact_rational_instant(
            "2013-01-01T17:00:00.0000000+05:00"
        )
        assert simultaneous_a == simultaneous_b
        assert self.parser.parse_iso(
            "2013-01-01T12:00:00.0000000Z"
        ) == self.parser.parse_iso("2013-01-01T17:00:00.0000000+05:00")


class TestFormatReparseRoundTrip:
    """Formatting to six digits and re-parsing closes the loop on the loss."""

    def setup_method(self):
        self.parser = DateTimeParser()
        self.formatter = DateTimeFormatter()
        self.fmt = "YYYY-MM-DDTHH:mm:ss.SSSSSS"

    def _round_trip(self, text: str) -> Tuple[datetime, str, datetime]:
        parsed = self.parser.parse_iso(text)
        formatted = self.formatter.format(parsed, self.fmt)
        reparsed = self.parser.parse_iso(formatted)
        return parsed, formatted, reparsed

    def test_round_trip_is_stable_for_the_datetime(self):
        for text in [
            "2013-01-01T12:30:45.9876545",
            "2013-01-01T12:30:45.9999995",
            "2013-01-31T23:59:59.9999999",
        ]:
            parsed, formatted, reparsed = self._round_trip(text)
            assert reparsed == parsed
            assert len(formatted.split(".", 1)[1]) == 6

    def test_carried_value_formats_on_the_new_second(self):
        parsed, formatted, reparsed = self._round_trip(
            "2013-01-01T12:30:45.9999995"
        )
        assert formatted == "2013-01-01T12:30:46.000000"
        assert reparsed == datetime(2013, 1, 1, 12, 30, 46, 0)

    def test_colliding_sources_produce_one_canonical_string(self):
        sources = [
            "2013-01-01T12:30:45.1234555",
            "2013-01-01T12:30:45.1234564",
            "2013-01-01T12:30:45.12345651",
        ]
        formatted = [self._round_trip(text)[1] for text in sources]
        assert formatted == ["2013-01-01T12:30:45.123456"] * 3
        assert len(set(formatted)) == 1

    def test_round_trip_keeps_timezone_offset(self):
        parsed = self.parser.parse_iso("2013-01-01T23:59:59.9999995+05:00")
        formatted = self.formatter.format(
            parsed, "YYYY-MM-DDTHH:mm:ss.SSSSSSZZ"
        )
        assert formatted == "2013-01-02T00:00:00.000000+05:00"
        assert self.parser.parse_iso(formatted) == parsed

    def test_public_arrow_api_shares_the_same_loss(self):
        arrow_a = arrow.get("2013-01-01T12:30:45.1234555")
        arrow_b = arrow.get("2013-01-01T12:30:45.1234564")
        assert arrow_a == arrow_b
        assert arrow_a.format(self.fmt) == arrow_b.format(self.fmt)


class TestPartitionAndOrdering:
    """Bucketing and sorting once the source strings have become datetimes."""

    def setup_method(self):
        self.parser = DateTimeParser()

    def _parse_all(self, texts: List[str]) -> List[datetime]:
        return [self.parser.parse_iso(text) for text in texts]

    def test_partition_by_datetime_merges_collision_bucket(self):
        sources = [
            "2013-01-01T12:30:45.1234555",
            "2013-01-01T12:30:45.1234564",
            "2013-01-01T12:30:45.1234574",
        ]
        buckets = {}
        for text in sources:
            buckets.setdefault(self.parser.parse_iso(text), []).append(text)

        # Three rational instants collapse into two microsecond buckets.
        assert len(buckets) == 2
        merged_bucket = buckets[datetime(2013, 1, 1, 12, 30, 45, 123456)]
        assert merged_bucket == sources[:2]

    def test_stable_sort_keeps_source_order_only_via_row_position(self):
        sources = [
            "2013-01-01T12:30:45.1234564",
            "2013-01-01T12:30:45.1234555",
        ]
        parsed = self._parse_all(sources)

        # Python's sort is stable: equal keys retain the incoming row order.
        assert sorted(parsed) == parsed
        # Feeding the rows in reverse yields the reverse bucket order while the
        # datetimes themselves stay pairwise equal, so nothing on the datetime
        # says which ordering was originally received.
        reversed_parsed = self._parse_all(list(reversed(sources)))
        assert sorted(reversed_parsed) == reversed_parsed
        assert parsed[0] == reversed_parsed[0]

    def test_dedup_removes_a_row_without_any_trace(self):
        sources = [
            "2013-01-01T12:30:45.1234555",
            "2013-01-01T12:30:45.1234564",
            "2013-01-01T12:30:45.1234574",
        ]
        unique = set(self._parse_all(sources))
        assert len(unique) == 2
        assert exact_rational_instant(sources[0]) != exact_rational_instant(
            sources[1]
        )

    def test_offset_wall_order_needs_tzinfo_to_correct(self):
        texts = [
            "2013-01-01T23:00:00.1234565+05:00",
            "2013-01-01T20:00:00.1234564Z",
        ]
        parsed = self._parse_all(texts)

        # Aware datetimes sort on the UTC instant, recovering true order.
        assert sorted(parsed) == parsed
        # If only naive wall datetimes survive, sorting flips the true order.
        naive = [dt.replace(tzinfo=None) for dt in parsed]
        assert sorted(naive) == list(reversed(naive))

    def test_rational_sequence_loses_strictness_without_inversion(self):
        fractions = [
            "1234555",
            "1234564",
            "12345650",
            "12345651",
            "1234574",
            "9999999",
        ]
        texts = [f"2013-01-01T12:30:45.{fraction}" for fraction in fractions]
        rationals = [exact_rational_instant(text) for text in texts]
        parsed = self._parse_all(texts)

        assert rationals == sorted(rationals)
        assert all(left <= right for left, right in zip(parsed, parsed[1:]))
        assert any(left == right for left, right in zip(parsed, parsed[1:]))


class TestIrrecoverableOrderFromDatetime:
    """Documenting which source orderings can never be reconstructed."""

    def setup_method(self):
        self.parser = DateTimeParser()

    def test_sub_microsecond_order_is_unrecoverable(self):
        earlier = "2013-01-01T12:30:45.1234555"
        later = "2013-01-01T12:30:45.1234564"
        sole_datetime = self.parser.parse_iso(earlier)

        assert self.parser.parse_iso(later) == sole_datetime
        # Handed only this datetime, both sources map to it; no field records
        # which string arrived first or that two events existed at all.
        assert sole_datetime.microsecond == 123456
        assert exact_rational_instant(earlier) < exact_rational_instant(later)

    def test_carried_source_leaves_no_trace_of_original_second(self):
        carried = self.parser.parse_iso("2013-01-01T12:30:45.9999995")
        exact_second = self.parser.parse_iso("2013-01-01T12:30:46.0000000")
        assert carried == exact_second
        assert carried.second == 46 and carried.microsecond == 0

    def test_naive_datetime_destroys_cross_offset_order(self):
        ahead = self.parser.parse_iso(
            "2013-01-01T23:00:00.1234565+05:00"
        ).replace(tzinfo=None)
        behind = self.parser.parse_iso(
            "2013-01-01T20:00:00.1234564Z"
        ).replace(tzinfo=None)

        # As naive wall values the true earlier UTC instant looks later; with
        # the offset gone there is no longer any way to correct that reading.
        assert ahead > behind
        assert (
            exact_rational_instant("2013-01-01T23:00:00.1234565+05:00")
            < exact_rational_instant("2013-01-01T20:00:00.1234564Z")
        )

    def test_lost_distinctions_inventory(self):
        # Each entry lists sources that become one single datetime; the size of
        # the source group is information the datetime cannot give back.
        groups = [
            [
                "2013-01-01T12:30:45.1234555",
                "2013-01-01T12:30:45.1234564",
            ],
            [
                "2013-01-01T12:30:45.9999999",
                "2013-01-01T12:30:46.0000001",
            ],
            [
                "2013-01-01T12:30:45.12345650",
                "2013-01-01T12:30:45.12345651",
            ],
        ]
        for group in groups:
            parsed = {self.parser.parse_iso(text) for text in group}
            assert len(parsed) == 1
            assert (
                len({exact_rational_instant(text) for text in group})
                == len(group)
            )
