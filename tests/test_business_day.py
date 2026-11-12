try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, List

import pytest
import pytz

import arrow
from arrow.business_day import (
    FOLD_FIRST,
    FOLD_SECOND,
    GAP_EARLIEST,
    GAP_LATEST,
    BusinessDayPartitioner,
    BusinessDayRule,
)

NY = "America/New_York"
UTC = timezone.utc


def utc(y: int, mo: int, d: int, h: int = 0, mi: int = 0, s: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, s, tzinfo=UTC)


class TestBusinessDayRule:
    def test_default_boundary_is_local_0230(self):
        rule = BusinessDayRule(tz=NY)
        # 2024-06-01 02:30 in New York is UTC-04:00 -> 06:30 UTC.
        assert rule.boundary_at(date(2024, 6, 1)) == arrow.Arrow(
            2024, 6, 1, 6, 30, tzinfo=UTC
        )

    def test_accepts_tzinfo_object(self):
        rule = BusinessDayRule(tz=ZoneInfo(NY))
        assert rule.boundary_at(date(2024, 6, 1)) == arrow.Arrow(
            2024, 6, 1, 6, 30, tzinfo=UTC
        )

    def test_custom_boundary_time(self):
        rule = BusinessDayRule(tz=NY, boundary_time=time(0, 0))
        assert rule.boundary_at(date(2024, 6, 1)) == arrow.Arrow(
            2024, 6, 1, 4, 0, tzinfo=UTC
        )

    def test_invalid_gap_and_fold_choices(self):
        with pytest.raises(ValueError):
            BusinessDayRule(tz=NY, gap="middle")
        with pytest.raises(ValueError):
            BusinessDayRule(tz=NY, fold="middle")

    def test_invalid_boundary_type(self):
        with pytest.raises(TypeError):
            BusinessDayRule(tz=NY, boundary_time="02:30")  # type: ignore[arg-type]

    def test_empty_effective_interval(self):
        change = utc(2024, 6, 15)
        with pytest.raises(ValueError):
            BusinessDayRule(tz=NY, start=change, end=change)

    def test_applies_at_respects_interval(self):
        change = utc(2024, 6, 15)
        rule = BusinessDayRule(tz=NY, end=change)
        assert rule.applies_at(change - timedelta(seconds=1))
        assert not rule.applies_at(change)


class TestGapAndFoldBoundaries:
    # America/New_York springs forward 2024-03-10 02:00 -> 03:00, so 02:30 is
    # a non-existent wall time.
    def test_spring_forward_gap_earliest(self):
        rule = BusinessDayRule(tz=NY, gap=GAP_EARLIEST)
        assert rule.boundary_at(date(2024, 3, 10)) == arrow.Arrow(
            2024, 3, 10, 6, 30, tzinfo=UTC
        )

    def test_spring_forward_gap_latest(self):
        rule = BusinessDayRule(tz=NY, gap=GAP_LATEST)
        assert rule.boundary_at(date(2024, 3, 10)) == arrow.Arrow(
            2024, 3, 10, 7, 30, tzinfo=UTC
        )

    # America/New_York falls back 2024-11-03 02:00 -> 01:00, so 01:30 occurs
    # twice. A 01:30 boundary exercises the fold choice directly.
    def test_fall_back_fold_first(self):
        rule = BusinessDayRule(
            tz=NY, boundary_time=time(1, 30), fold=FOLD_FIRST
        )
        assert rule.boundary_at(date(2024, 11, 3)) == arrow.Arrow(
            2024, 11, 3, 5, 30, tzinfo=UTC
        )

    def test_fall_back_fold_second(self):
        rule = BusinessDayRule(
            tz=NY, boundary_time=time(1, 30), fold=FOLD_SECOND
        )
        assert rule.boundary_at(date(2024, 11, 3)) == arrow.Arrow(
            2024, 11, 3, 6, 30, tzinfo=UTC
        )


class TestPytzParity:
    # pytz zones must resolve identically to PEP 495 (zoneinfo) zones even
    # though naive datetimes cannot be attached with a plain .replace().
    @pytest.mark.parametrize(
        "local_date, expected",
        [
            (date(2024, 6, 1), arrow.Arrow(2024, 6, 1, 6, 30, tzinfo=UTC)),
            (date(2024, 3, 10), arrow.Arrow(2024, 3, 10, 6, 30, tzinfo=UTC)),
        ],
    )
    def test_pytz_matches_zoneinfo_boundaries(
        self, local_date: date, expected: arrow.Arrow
    ) -> None:
        pytz_rule = BusinessDayRule(tz=pytz.timezone(NY))
        zoneinfo_rule = BusinessDayRule(tz=ZoneInfo(NY))
        assert pytz_rule.boundary_at(local_date) == expected
        assert pytz_rule.boundary_at(local_date) == zoneinfo_rule.boundary_at(
            local_date
        )

    def test_pytz_gap_and_fold_choices_match_zoneinfo(self):
        for gap in (GAP_EARLIEST, GAP_LATEST):
            pytz_rule = BusinessDayRule(tz=pytz.timezone(NY), gap=gap)
            zone_rule = BusinessDayRule(tz=ZoneInfo(NY), gap=gap)
            assert pytz_rule.boundary_at(date(2024, 3, 10)) == zone_rule.boundary_at(
                date(2024, 3, 10)
            )
        for fold in (FOLD_FIRST, FOLD_SECOND):
            pytz_rule = BusinessDayRule(
                tz=pytz.timezone(NY), boundary_time=time(1, 30), fold=fold
            )
            zone_rule = BusinessDayRule(
                tz=ZoneInfo(NY), boundary_time=time(1, 30), fold=fold
            )
            assert pytz_rule.boundary_at(date(2024, 11, 3)) == zone_rule.boundary_at(
                date(2024, 11, 3)
            )


class TestPartition:
    def test_events_sorted_in_real_time_order(self):
        rule = BusinessDayRule(tz=NY)
        partitioner = BusinessDayPartitioner(rule)
        events = [
            arrow.get(utc(2024, 6, 2, 10)),
            arrow.get(utc(2024, 6, 1, 10)),
            arrow.get(utc(2024, 6, 1, 5)),  # just before 06:30 boundary
        ]
        assignments = partitioner.partition(events)
        instants = [assignment.instant.datetime for assignment in assignments]
        assert instants == sorted(instants)

    def test_boundary_inclusive_start_and_exclusive_end(self):
        partitioner = BusinessDayPartitioner(BusinessDayRule(tz=NY))
        boundary = utc(2024, 6, 1, 6, 30)
        day = partitioner.business_day_of(boundary)
        assert day.contains(boundary)
        assert not day.contains(day.end.datetime)
        previous = partitioner.business_day_of(boundary - timedelta(microseconds=1))
        assert previous.date == date(2024, 5, 31)
        assert partitioner.business_day_of(boundary).date == date(2024, 6, 1)

    def test_each_event_assigned_to_exactly_one_day_across_gap(self):
        partitioner = BusinessDayPartitioner(
            BusinessDayRule(tz=NY, gap=GAP_EARLIEST)
        )
        # Minute-by-minute across the whole spring-forward weekend; every minute
        # must be contained by the single day it is assigned to.
        current = utc(2024, 3, 9)
        end = utc(2024, 3, 12)
        seen_days: List[date] = []
        while current < end:
            day = partitioner.business_day_of(current)
            assert day.contains(current)
            if not seen_days or seen_days[-1] != day.date:
                seen_days.append(day.date)
            current += timedelta(minutes=1)
        # Four contiguous local dates even though one 02:30 boundary did not
        # exist; the span starts during the evening of the first local date.
        assert seen_days == [
            date(2024, 3, 8),
            date(2024, 3, 9),
            date(2024, 3, 10),
            date(2024, 3, 11),
        ]

    def test_gap_choice_moves_events_on_transition_day(self):
        events = [
            arrow.get(utc(2024, 3, 10, 7, 0)),  # between 06:30 and 07:30 UTC
        ]
        earliest = BusinessDayPartitioner(
            BusinessDayRule(tz=NY, gap=GAP_EARLIEST)
        ).partition(events)[0]
        latest = BusinessDayPartitioner(
            BusinessDayRule(tz=NY, gap=GAP_LATEST)
        ).partition(events)[0]
        assert earliest.business_day.date == date(2024, 3, 10)
        # With the later boundary, 07:00 UTC still belongs to the previous day.
        assert latest.business_day.date == date(2024, 3, 9)

    def test_fold_keeps_every_instant_in_one_day(self):
        partitioner = BusinessDayPartitioner(
            BusinessDayRule(
                tz=NY, boundary_time=time(1, 30), fold=FOLD_SECOND
            )
        )
        current = utc(2024, 11, 2)
        end = utc(2024, 11, 5)
        seen_days: List[date] = []
        while current < end:
            day = partitioner.business_day_of(current)
            assert day.contains(current)
            if not seen_days or seen_days[-1] != day.date:
                seen_days.append(day.date)
            current += timedelta(minutes=1)
        # The span begins during the evening of the first local date.
        assert seen_days == [
            date(2024, 11, 1),
            date(2024, 11, 2),
            date(2024, 11, 3),
            date(2024, 11, 4),
        ]

    def test_naive_datetime_treated_as_utc(self):
        partitioner = BusinessDayPartitioner(BusinessDayRule(tz=NY))
        day = partitioner.business_day_of(datetime(2024, 6, 1, 6, 30))
        assert day.date == date(2024, 6, 1)

    def test_custom_event_key(self):
        partitioner = BusinessDayPartitioner(BusinessDayRule(tz=NY))
        events = [
            {"id": "a", "ts": utc(2024, 6, 1, 10)},
            {"id": "b", "ts": utc(2024, 6, 1, 5)},
        ]
        assignments = partitioner.partition(events, key=lambda event: event["ts"])
        assert [assignment.event["id"] for assignment in assignments] == ["b", "a"]
        assert all(
            assignment.business_day.date == date(2024, 5, 31)
            or assignment.business_day.date == date(2024, 6, 1)
            for assignment in assignments
        )

    def test_missing_key_for_plain_objects_raises(self):
        partitioner = BusinessDayPartitioner(BusinessDayRule(tz=NY))
        with pytest.raises(TypeError):
            partitioner.partition([object()])

    def test_arrow_entry_points(self):
        rules = BusinessDayRule(tz=NY)
        partitioner = arrow.Arrow.business_day_partitioner(rules)
        assert isinstance(partitioner, BusinessDayPartitioner)

        instant = arrow.get(utc(2024, 6, 1, 7))
        day = instant.business_day(rules)
        assert day.date == date(2024, 6, 1)

        assignments = arrow.Arrow.partition_business_days(
            [instant], rules
        )
        assert assignments[0].business_day.date == date(2024, 6, 1)


class TestEffectiveRules:
    def test_rule_change_uses_each_interval(self):
        change = utc(2024, 6, 15, 4)
        rules = [
            BusinessDayRule(tz=NY, boundary_time=time(2, 30), end=change),
            BusinessDayRule(tz=NY, boundary_time=time(0, 0), start=change),
        ]
        partitioner = BusinessDayPartitioner(rules)
        # Just before the change the old 02:30 boundary (06:30 UTC) still wins,
        # so the early hours keep the previous calendar date.
        assert (
            partitioner.business_day_of(utc(2024, 6, 15, 3, 59)).date
            == date(2024, 6, 14)
        )
        # From the change onward midnight local (04:00 UTC) starts the day.
        assert partitioner.business_day_of(utc(2024, 6, 15, 4, 0, 1)).date == date(
            2024, 6, 15
        )

    def test_overlapping_intervals_rejected(self):
        change = utc(2024, 6, 15)
        with pytest.raises(ValueError):
            BusinessDayPartitioner(
                [
                    BusinessDayRule(tz=NY, end=change + timedelta(hours=1)),
                    BusinessDayRule(tz=NY, start=change),
                ]
            )

    def test_different_timezones_rejected(self):
        change = utc(2024, 6, 15)
        with pytest.raises(ValueError):
            BusinessDayPartitioner(
                [
                    BusinessDayRule(tz=NY, end=change),
                    BusinessDayRule(tz="UTC", start=change),
                ]
            )

    def test_uncovered_instant_raises(self):
        partitioner = BusinessDayPartitioner(
            BusinessDayRule(tz=NY, start=utc(2024, 1, 1))
        )
        with pytest.raises(ValueError):
            partitioner.business_day_of(utc(2020, 1, 1))

    def test_empty_rule_set_rejected(self):
        with pytest.raises(ValueError):
            BusinessDayPartitioner([])


class TestReplay:
    def _rulesets(self) -> Any:
        change = utc(2024, 6, 15)
        old = BusinessDayPartitioner(
            [
                BusinessDayRule(tz=NY, boundary_time=time(2, 30), end=change),
                BusinessDayRule(tz=NY, boundary_time=time(2, 30), start=change),
            ]
        )
        new = [
            BusinessDayRule(tz=NY, boundary_time=time(0, 0), end=change),
            BusinessDayRule(tz=NY, boundary_time=time(0, 0), start=change),
        ]
        return old, new, change

    def test_replay_reports_moved_days_and_boundaries(self):
        old, new, _ = self._rulesets()
        events = [
            arrow.get(utc(2024, 6, 14, 5)),   # 01:00 local, flips 5/31-style day
            arrow.get(utc(2024, 7, 1, 12)),
        ]
        results = old.replay(new, events)
        moved = results[0]
        assert len(results) == 1
        assert moved.old_day.date != moved.new_day.date
        assert moved.old_boundary == (moved.old_day.start, moved.old_day.end)
        assert moved.new_boundary == (moved.new_day.start, moved.new_day.end)
        # New midnight boundary starts at 04:00 UTC.
        assert moved.new_day.start == arrow.Arrow(2024, 6, 14, 4, 0, tzinfo=UTC)

    def test_replay_window_limits_affected_history(self):
        old, new, change = self._rulesets()
        # 05:00 UTC flips from the 13th to the 14th once midnight becomes the
        # boundary; the July event keeps the same business day under both rules.
        inside = arrow.get(utc(2024, 6, 14, 5))
        outside = arrow.get(utc(2024, 7, 1))
        window = (change - timedelta(days=2), change + timedelta(days=2))
        results = old.replay(new, [outside, inside], window=window)
        assert [result.event for result in results] == [inside]
        assert results[0].old_day.date == date(2024, 6, 13)
        assert results[0].new_day.date == date(2024, 6, 14)

    def test_replay_accepts_partitioner(self):
        old, new_rules, _ = self._rulesets()
        target = BusinessDayPartitioner(new_rules)
        events = [arrow.get(utc(2024, 6, 14, 5))]
        results = old.replay(target, events)
        assert len(results) == 1

    def test_replay_results_in_time_order(self):
        old, new, _ = self._rulesets()
        events = [
            arrow.get(utc(2024, 6, 16, 5)),
            arrow.get(utc(2024, 6, 14, 5)),
            arrow.get(utc(2024, 6, 15, 5)),
        ]
        results = old.replay(new, events)
        instants = [result.instant.datetime for result in results]
        assert instants == sorted(instants)

    def test_unchanged_events_are_not_reported(self):
        rule = BusinessDayRule(tz=NY)
        partitioner = BusinessDayPartitioner(rule)
        events = [arrow.get(utc(2024, 6, 1, 12))]
        assert partitioner.replay(rule, events) == []


class TestExhaustiveCoverage:
    @pytest.mark.parametrize("boundary_time", [time(0, 0), time(1, 30), time(2, 30)])
    @pytest.mark.parametrize("gap", [GAP_EARLIEST, GAP_LATEST])
    @pytest.mark.parametrize("fold", [FOLD_FIRST, FOLD_SECOND])
    def test_every_minute_has_exactly_one_business_day(
        self, boundary_time: time, gap: str, fold: str
    ) -> None:
        partitioner = BusinessDayPartitioner(
            BusinessDayRule(
                tz=NY, boundary_time=boundary_time, gap=gap, fold=fold
            )
        )
        for span_start, span_end in (
            (utc(2024, 3, 9), utc(2024, 3, 12)),   # spring forward
            (utc(2024, 11, 2), utc(2024, 11, 5)),  # fall back
        ):
            current = span_start
            boundaries: List[datetime] = []
            while current < span_end:
                day = partitioner.business_day_of(current)
                assert day.contains(current)
                if not boundaries or boundaries[-1] != day.start.datetime:
                    boundaries.append(day.start.datetime)
                current += timedelta(minutes=1)
            assert boundaries == sorted(set(boundaries))
