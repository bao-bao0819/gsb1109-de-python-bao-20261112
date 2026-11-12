from datetime import datetime, timedelta

import pytest
from dateutil import tz

import arrow
from arrow.sampling import (
    daily_ranges,
    replay_range,
    stratified_plan,
    stratified_sample,
)

NYC = tz.gettz("America/New_York")
UTC = tz.UTC


class TestStrata:
    def test_normal_day_has_24_full_hour_strata(self):
        start = arrow.get(2026, 3, 1, tzinfo=UTC)
        end = arrow.get(2026, 3, 2, tzinfo=UTC)

        plan = stratified_plan(start, end, budget=0)

        assert len(plan.strata) == 24
        assert all(s.duration.total_seconds() == 3600 for s in plan.strata)
        assert all(s.fold == 0 for s in plan.strata)
        assert len({s.key for s in plan.strata}) == 24

    def test_fallback_day_has_two_independent_0100_strata(self):
        start = arrow.get(2026, 11, 1, tzinfo=NYC)
        end = arrow.get(2026, 11, 2, tzinfo=NYC)

        plan = stratified_plan(start, end, budget=0)

        assert len(plan.strata) == 25
        one_am = [s for s in plan.strata if s.local_start.hour == 1]
        assert len(one_am) == 2
        assert [s.fold for s in one_am] == [0, 1]
        assert one_am[0].key != one_am[1].key
        # each occurrence of the repeated hour lasts a real hour
        assert all(s.duration.total_seconds() == 3600 for s in one_am)
        assert one_am[0].utc_end == one_am[1].utc_start
        # the day really lasts 25 hours
        total = sum((s.duration for s in plan.strata), timedelta()).total_seconds()
        assert total == 25 * 3600

    def test_spring_forward_day_skips_the_missing_hour(self):
        start = arrow.get(2026, 3, 8, tzinfo=NYC)
        end = arrow.get(2026, 3, 9, tzinfo=NYC)

        plan = stratified_plan(start, end, budget=0)

        assert len(plan.strata) == 23
        assert not any(s.local_start.hour == 2 for s in plan.strata)
        total = sum((s.duration for s in plan.strata), timedelta()).total_seconds()
        assert total == 23 * 3600

    def test_strata_are_clipped_to_the_range(self):
        start = arrow.get(2026, 3, 1, 6, 30, tzinfo=UTC)
        end = arrow.get(2026, 3, 1, 8, 15, tzinfo=UTC)

        plan = stratified_plan(start, end, budget=0)

        durations = [s.duration.total_seconds() for s in plan.strata]
        assert durations == [1800, 3600, 900]


class TestAllocation:
    def test_quotas_are_proportional_to_true_durations(self):
        start = arrow.get(2026, 3, 1, 0, 0, tzinfo=UTC)
        end = arrow.get(2026, 3, 1, 3, 0, tzinfo=UTC)

        plan = stratified_plan(start, end, budget=6)
        # make the middle hour half as long by clipping the range
        clipped = stratified_plan(
            arrow.get(2026, 3, 1, 0, 0, tzinfo=UTC),
            arrow.get(2026, 3, 1, 2, 30, tzinfo=UTC),
            budget=5,
        )

        assert sum(plan.allocation.values()) == 6
        assert list(plan.allocation.values()) == [2, 2, 2]
        # durations 3600, 3600, 1800 -> quotas 2, 2, 1
        assert list(clipped.allocation.values()) == [2, 2, 1]

    def test_ties_are_broken_by_utc_order(self):
        start = arrow.get(2026, 3, 1, tzinfo=UTC)
        end = arrow.get(2026, 3, 2, tzinfo=UTC)

        plan = stratified_plan(start, end, budget=1)

        quotas = plan.allocation
        assert sum(quotas.values()) == 1
        assert quotas[plan.strata[0].key] == 1

    def test_minimums_are_honored_before_proportional_share(self):
        start = arrow.get(2026, 3, 1, tzinfo=UTC)
        end = arrow.get(2026, 3, 1, 4, tzinfo=UTC)

        plan = stratified_plan(start, end, budget=10, min_per_stratum=2)

        assert all(quota >= 2 for quota in plan.allocation.values())
        assert sum(plan.allocation.values()) == 10

    def test_maximums_cap_allocation_and_report_shortfalls(self):
        start = arrow.get(2026, 3, 1, tzinfo=UTC)
        end = arrow.get(2026, 3, 1, 2, tzinfo=UTC)

        plan = stratified_plan(start, end, budget=10, max_per_stratum=3)

        assert list(plan.allocation.values()) == [3, 3]
        assert sum(s.missing for s in plan.shortfalls) == 4
        assert {s.stratum_key for s in plan.shortfalls} == set(plan.allocation)

    def test_no_shortfalls_when_budget_is_feasible(self):
        start = arrow.get(2026, 3, 1, tzinfo=UTC)
        end = arrow.get(2026, 3, 2, tzinfo=UTC)

        plan = stratified_plan(start, end, budget=100, max_per_stratum=5)

        assert plan.shortfalls == ()

    def test_infeasible_minimum_raises(self):
        start = arrow.get(2026, 3, 1, tzinfo=UTC)
        end = arrow.get(2026, 3, 2, tzinfo=UTC)

        with pytest.raises(ValueError):
            stratified_plan(start, end, budget=10, min_per_stratum=1)

    def test_invalid_arguments_raise(self):
        start = arrow.get(2026, 3, 1, tzinfo=UTC)
        end = arrow.get(2026, 3, 2, tzinfo=UTC)

        with pytest.raises(ValueError):
            stratified_plan(end, start, budget=1)
        with pytest.raises(ValueError):
            stratified_plan(start, end, budget=-1)
        with pytest.raises(ValueError):
            stratified_plan(start, end, budget=1, min_per_stratum=-1)
        with pytest.raises(ValueError):
            stratified_plan(start, end, budget=1, min_per_stratum=2, max_per_stratum=1)
        with pytest.raises(ValueError):
            stratified_plan(datetime(2026, 3, 1), end, budget=1)


class TestSampling:
    def test_samples_are_uniform_instants_inside_their_stratum(self):
        start = arrow.get(2026, 3, 1, tzinfo=UTC)
        end = arrow.get(2026, 3, 2, tzinfo=UTC)

        plan = stratified_plan(start, end, budget=240, seed=7)
        samples = plan.samples()
        strata = {s.key: s for s in plan.strata}

        assert len(samples) == 240
        for sample in samples:
            stratum = strata[sample.stratum_key]
            assert stratum.utc_start <= sample.instant.to("utc") < stratum.utc_end
        assert len({s.instant for s in samples}) == 240

    def test_sampling_is_deterministic_per_seed(self):
        start = arrow.get(2026, 3, 1, tzinfo=NYC)
        end = arrow.get(2026, 3, 3, tzinfo=NYC)

        first = [s.sample_id for s in stratified_sample(start, end, 50, seed=42)]
        again = [s.sample_id for s in stratified_sample(start, end, 50, seed=42)]
        other = stratified_sample(start, end, 50, seed=43)

        assert first == again
        assert [s.instant for s in stratified_sample(start, end, 50, seed=42)] == [
            s.instant for s in stratified_sample(start, end, 50, seed=42)
        ]
        assert {s.instant for s in other} != {
            s.instant for s in stratified_sample(start, end, 50, seed=42)
        }

    def test_samples_fall_back_hours_are_sampled_independently(self):
        start = arrow.get(2026, 11, 1, tzinfo=NYC)
        end = arrow.get(2026, 11, 2, tzinfo=NYC)

        plan = stratified_plan(start, end, budget=250, min_per_stratum=1, seed=3)
        samples = plan.samples()
        one_am = [s for s in plan.strata if s.local_start.hour == 1]

        for stratum in one_am:
            drawn = [s for s in samples if s.stratum_key == stratum.key]
            assert len(drawn) == plan.allocation[stratum.key]
            assert len(drawn) >= 1


class TestReplay:
    def test_daily_replay_reproduces_global_sample_ids(self):
        start = arrow.get(2026, 10, 31, 12, 30, tzinfo=NYC)
        end = arrow.get(2026, 11, 2, 6, 15, tzinfo=NYC)

        plan = stratified_plan(start, end, budget=300, min_per_stratum=1, seed=11)
        global_ids = [s.sample_id for s in plan.samples()]

        # replay as per-day tasks against the same global allocation table
        daily_ids = []
        for day_start, day_end in plan.daily_ranges():
            daily_ids.extend(s.sample_id for s in plan.samples_between(day_start, day_end))
        assert daily_ids == global_ids

        # replay from the serialized table only, rebuilding strata per day
        table = plan.allocation
        rebuilt_ids = []
        for day_start, day_end in daily_ranges(start, end):
            rebuilt_ids.extend(
                s.sample_id for s in replay_range(day_start, day_end, table, seed=11)
            )
        assert rebuilt_ids == global_ids

    def test_daily_ranges_cover_the_range_without_gaps(self):
        start = arrow.get(2026, 3, 7, 12, 30, tzinfo=NYC)
        end = arrow.get(2026, 3, 9, 6, 15, tzinfo=NYC)

        ranges = daily_ranges(start, end)

        assert ranges[0][0] == start
        assert ranges[-1][1] == end
        assert len(ranges) == 3
        for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
            assert prev_end == next_start

    def test_replay_with_missing_table_entry_raises(self):
        start = arrow.get(2026, 3, 1, tzinfo=UTC)
        end = arrow.get(2026, 3, 2, tzinfo=UTC)

        with pytest.raises(KeyError):
            replay_range(start, end, {}, seed=0)
