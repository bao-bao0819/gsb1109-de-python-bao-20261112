try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from datetime import datetime, timedelta, timezone

import pytest

from arrow import Arrow
from arrow.device_uptime import (
    ACTION_REVOKE,
    DailyUptime,
    DeviceEvent,
    STATE_CONFLICT,
    STATE_OFFLINE,
    STATE_ONLINE,
    build_state_timeline,
    normalize_events,
    summarize,
    total_interval_seconds,
)

SHANGHAI = "Asia/Shanghai"
NEW_YORK = "America/New_York"


def _ts(year, month, day, hour=0, minute=0, second=0, tz=SHANGHAI):
    return Arrow.fromdatetime(
        datetime(year, month, day, hour, minute, second)
    ).replace(tzinfo=ZoneInfo(tz))


def _rows_by_day(rows):
    return {row.day: row for row in rows}


class TestNormalization:
    def test_exact_resend_is_dropped(self):
        event = DeviceEvent("dev", 1, _ts(2026, 3, 1, 6), state=STATE_ONLINE)
        result = normalize_events([event, event])
        assert len(result) == 1

    def test_resend_with_different_payload_is_rejected(self):
        first = DeviceEvent("dev", 1, _ts(2026, 3, 1, 6), state=STATE_ONLINE)
        resent = DeviceEvent("dev", 1, _ts(2026, 3, 1, 7), state=STATE_OFFLINE)
        with pytest.raises(ValueError, match="Conflicting resend"):
            normalize_events([first, resent])

    def test_revoke_removes_the_named_event(self):
        online = DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE)
        false_stop = DeviceEvent("dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE)
        retraction = DeviceEvent(
            "dev", 2, _ts(2026, 3, 1, 2), action=ACTION_REVOKE
        )
        real_stop = DeviceEvent("dev", 3, _ts(2026, 3, 1, 3), state=STATE_OFFLINE)
        result = normalize_events([online, false_stop, retraction, real_stop])
        assert [e.sequence for e in result] == [1, 3]

    def test_reject_revoke_carrying_state(self):
        event = DeviceEvent(
            "dev", 1, _ts(2026, 3, 1), action=ACTION_REVOKE, state=STATE_ONLINE
        )
        with pytest.raises(ValueError, match="revoke events must not carry a state"):
            normalize_events([event])

    def test_correction_supersedes_target_event(self):
        online = DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE)
        early_stop = DeviceEvent("dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE)
        corrected_stop = DeviceEvent(
            "dev",
            3,
            _ts(2026, 3, 1, 2),
            state=STATE_OFFLINE,
            corrected_sequence=2,
        )
        result = normalize_events([online, early_stop, corrected_stop])
        assert [e.sequence for e in result] == [1, 3]

    def test_sequence_isolated_between_devices(self):
        a = DeviceEvent("a", 1, _ts(2026, 3, 1), state=STATE_ONLINE)
        b = DeviceEvent("b", 1, _ts(2026, 3, 1), state=STATE_OFFLINE)
        result = normalize_events([a, b])
        assert {e.device_id for e in result} == {"a", "b"}

    def test_subsecond_payload_difference_is_still_a_conflict(self):
        first = DeviceEvent("dev", 1, 1000.0, state=STATE_ONLINE)
        resent = DeviceEvent("dev", 1, 1000.5, state=STATE_ONLINE)
        with pytest.raises(ValueError, match="Conflicting resend"):
            normalize_events([first, resent])

    def test_exact_float_resend_is_dropped(self):
        event = DeviceEvent("dev", 1, 1000.5, state=STATE_ONLINE)
        assert len(normalize_events([event, event])) == 1

    def test_invalid_event_time_type_is_rejected(self):
        with pytest.raises(TypeError, match="at must be an Arrow"):
            normalize_events(
                [DeviceEvent("dev", 1, "2026-03-01", state=STATE_ONLINE)]
            )

    def test_revoke_arriving_before_its_event_still_voids_it(self):
        online = DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE)
        false_stop = DeviceEvent("dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE)
        retraction = DeviceEvent(
            "dev", 2, _ts(2026, 3, 1, 2), action=ACTION_REVOKE
        )
        result = normalize_events([retraction, false_stop, online])
        assert [e.sequence for e in result] == [1]

    def test_correction_arriving_before_its_target_still_supersedes(self):
        online = DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE)
        early_stop = DeviceEvent("dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE)
        corrected_stop = DeviceEvent(
            "dev",
            3,
            _ts(2026, 3, 1, 2),
            state=STATE_OFFLINE,
            corrected_sequence=2,
        )
        result = normalize_events([corrected_stop, early_stop, online])
        assert [e.sequence for e in result] == [1, 3]

    def test_revoked_correction_restores_its_target(self):
        online = DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE)
        early_stop = DeviceEvent("dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE)
        corrected_stop = DeviceEvent(
            "dev",
            3,
            _ts(2026, 3, 1, 2),
            state=STATE_OFFLINE,
            corrected_sequence=2,
        )
        correction_revoke = DeviceEvent(
            "dev", 3, _ts(2026, 3, 1, 3), action=ACTION_REVOKE
        )
        result = normalize_events(
            [online, early_stop, corrected_stop, correction_revoke]
        )
        assert [e.sequence for e in result] == [1, 2]

    def test_shuffled_stream_matches_ordered_stream(self):
        import random

        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE),
            DeviceEvent("dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE),
            DeviceEvent(
                "dev", 2, _ts(2026, 3, 1, 2), action=ACTION_REVOKE
            ),
            DeviceEvent("dev", 3, _ts(2026, 3, 1, 3), state=STATE_OFFLINE),
            DeviceEvent("dev", 3, _ts(2026, 3, 1, 3), state=STATE_OFFLINE),
        ]
        shuffled = events[:]
        random.Random(7).shuffle(shuffled)
        assert normalize_events(shuffled) == normalize_events(events)


class TestValidation:
    def test_invalid_fields_are_rejected(self):
        cases = [
            (dict(device_id=""), "device_id must be a non-empty string"),
            (dict(device_id=1), "device_id must be a non-empty string"),
            (dict(sequence="1"), "sequence must be a non-negative integer"),
            (dict(sequence=True), "sequence must be a non-negative integer"),
            (dict(sequence=-1), "sequence must be a non-negative integer"),
            (dict(action="pause"), "action must be one of"),
            (dict(source=""), "source must be a non-empty string"),
            (dict(state=None), "report state must be one of"),
            (dict(state="standby"), "report state must be one of"),
            (dict(corrected_sequence=-1), "corrected_sequence must be"),
        ]
        for overrides, message in cases:
            kwargs = dict(
                device_id="dev",
                sequence=1,
                at=_ts(2026, 3, 1),
                state=STATE_ONLINE,
            )
            kwargs.update(overrides)
            with pytest.raises(ValueError, match=message):
                normalize_events([DeviceEvent(**kwargs)])

    def test_correction_on_revoke_is_rejected(self):
        event = DeviceEvent(
            "dev",
            1,
            _ts(2026, 3, 1),
            action=ACTION_REVOKE,
            corrected_sequence=0,
        )
        with pytest.raises(ValueError, match="only report events"):
            normalize_events([event])

    def test_non_event_input_is_rejected(self):
        with pytest.raises(TypeError, match="expected DeviceEvent"):
            normalize_events([("dev", 1)])

    def test_datetime_event_times_support_sorting_and_dedup(self):
        first = DeviceEvent(
            "dev",
            1,
            datetime(2026, 3, 1, 8, 0, tzinfo=ZoneInfo(SHANGHAI)),
            state=STATE_ONLINE,
        )
        duplicate = DeviceEvent(
            "dev",
            1,
            datetime(2026, 3, 1, 8, 0, tzinfo=ZoneInfo(SHANGHAI)),
            state=STATE_ONLINE,
        )
        assert normalize_events([duplicate, first]) == [first]

    def test_timezone_object_and_invalid_timezone_expression(self):
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1), state=STATE_ONLINE),
            DeviceEvent("dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE),
        ]
        rows = summarize(
            events,
            ZoneInfo(SHANGHAI),
            _ts(2026, 3, 1),
            _ts(2026, 3, 2),
        )
        assert rows[0].online_seconds == 3600
        with pytest.raises(ValueError, match="not recognized as a timezone"):
            summarize(events, "Mars/Olympus", _ts(2026, 3, 1), _ts(2026, 3, 2))

    def test_unsupported_window_time_type_is_rejected(self):
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1), state=STATE_ONLINE),
            DeviceEvent("dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE),
        ]
        with pytest.raises(TypeError, match="Unsupported event time"):
            summarize(events, SHANGHAI, "2026-03-01", _ts(2026, 3, 2))

    def test_to_arrow_rejects_unsupported_types(self):
        from arrow.device_uptime import _to_arrow

        with pytest.raises(TypeError, match="Unsupported event time"):
            _to_arrow(True, ZoneInfo(SHANGHAI))
        with pytest.raises(TypeError, match="Unsupported event time"):
            _to_arrow("2026-03-01", ZoneInfo(SHANGHAI))

    def test_window_ending_before_start_returns_no_rows(self):
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1), state=STATE_ONLINE),
            DeviceEvent("dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE),
        ]
        assert summarize(
            events, SHANGHAI, _ts(2026, 3, 2), _ts(2026, 3, 1)
        ) == []

    def test_utc_measurement_skips_segments_before_window(self):
        start, end = _ts(2026, 3, 2), _ts(2026, 3, 3)
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE),
            DeviceEvent("dev", 2, _ts(2026, 3, 1, 6), state=STATE_OFFLINE),
            DeviceEvent("dev", 3, _ts(2026, 3, 2, 6), state=STATE_ONLINE),
            DeviceEvent("dev", 4, _ts(2026, 3, 2, 12), state=STATE_OFFLINE),
        ]
        totals = total_interval_seconds(events, SHANGHAI, start, end)["dev"]
        assert totals["online_seconds"] == 6 * 3600
        assert totals["conflict_seconds"] == 0


class TestTimeline:
    def test_revoked_stop_keeps_device_online(self):
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE),
            DeviceEvent("dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE),
            DeviceEvent(
                "dev", 2, _ts(2026, 3, 1, 2), action=ACTION_REVOKE
            ),
            DeviceEvent("dev", 3, _ts(2026, 3, 1, 3), state=STATE_OFFLINE),
        ]
        timeline = build_state_timeline(events)["dev"]
        assert len(timeline) == 2
        assert timeline[0].state == STATE_ONLINE
        assert timeline[0].end == _ts(2026, 3, 1, 3)

    def test_opposite_source_states_become_conflict_segment(self):
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE, source="gw"),
            DeviceEvent(
                "dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE, source="sensor"
            ),
            DeviceEvent("dev", 3, _ts(2026, 3, 1, 2), state=STATE_OFFLINE, source="gw"),
        ]
        timeline = build_state_timeline(events)["dev"]
        states = [(seg.state, seg.conflict) for seg in timeline]
        assert states == [
            (STATE_ONLINE, False),
            (STATE_CONFLICT, True),
            (STATE_OFFLINE, False),
        ]

    def test_same_state_from_two_sources_counts_once(self):
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE, source="gw"),
            DeviceEvent(
                "dev", 2, _ts(2026, 3, 1, 0), state=STATE_ONLINE, source="sensor"
            ),
            DeviceEvent("dev", 3, _ts(2026, 3, 1, 1), state=STATE_OFFLINE, source="gw"),
            DeviceEvent(
                "dev", 4, _ts(2026, 3, 1, 1), state=STATE_OFFLINE, source="sensor"
            ),
        ]
        timeline = build_state_timeline(events)["dev"]
        assert len(timeline) == 2
        assert timeline[0].state == STATE_ONLINE
        assert not timeline[0].conflict
        assert (
            timeline[0].end.int_timestamp - timeline[0].start.int_timestamp == 3600
        )


class TestSummarize:
    def _window(self):
        return _ts(2026, 3, 1), _ts(2026, 3, 4)

    def test_split_across_local_natural_days(self):
        start, end = self._window()
        events = [
            DeviceEvent(
                "dev",
                1,
                datetime(2026, 3, 1, 22, 0, tzinfo=timezone.utc),
                state=STATE_ONLINE,
            ),
            DeviceEvent(
                "dev",
                2,
                datetime(2026, 3, 2, 2, 0, tzinfo=timezone.utc),
                state=STATE_OFFLINE,
            ),
            DeviceEvent(
                "dev",
                3,
                datetime(2026, 3, 2, 20, 0, tzinfo=timezone.utc),
                state=STATE_ONLINE,
            ),
            DeviceEvent(
                "dev",
                4,
                datetime(2026, 3, 2, 22, 0, tzinfo=timezone.utc),
                state=STATE_OFFLINE,
            ),
        ]
        rows = summarize(events, SHANGHAI, start, end)
        by_day = _rows_by_day(rows)
        assert by_day[datetime(2026, 3, 1).date()].online_seconds == 0
        assert by_day[datetime(2026, 3, 2).date()].online_seconds == 4 * 3600
        assert by_day[datetime(2026, 3, 3).date()].online_seconds == 2 * 3600
        assert all(row.conflict_seconds == 0 for row in rows)

    def test_daily_sum_matches_direct_utc_interval_measurement(self):
        start, end = self._window()
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1, 6), state=STATE_ONLINE),
            DeviceEvent("dev", 2, _ts(2026, 3, 1, 10), state=STATE_OFFLINE),
            DeviceEvent("dev", 3, _ts(2026, 3, 2, 4), state=STATE_ONLINE),
            DeviceEvent("dev", 4, _ts(2026, 3, 2, 6), state=STATE_OFFLINE),
        ]
        rows = summarize(events, SHANGHAI, start, end)
        totals = total_interval_seconds(events, SHANGHAI, start, end)
        online = sum(row.online_seconds for row in rows)
        conflict = sum(row.conflict_seconds for row in rows)
        assert online == totals["dev"]["online_seconds"]
        assert conflict == totals["dev"]["conflict_seconds"]
        assert online == 6 * 3600

    def test_conflict_seconds_are_a_separate_column(self):
        start, end = _ts(2026, 3, 1), _ts(2026, 3, 1, 3)
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE, source="gw"),
            DeviceEvent(
                "dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE, source="sensor"
            ),
            DeviceEvent(
                "dev", 3, _ts(2026, 3, 1, 2), state=STATE_ONLINE, source="sensor"
            ),
        ]
        rows = summarize(events, SHANGHAI, start, end)
        row = rows[0]
        assert row.online_seconds == 2 * 3600
        assert row.conflict_seconds == 3600
        totals = total_interval_seconds(events, SHANGHAI, start, end)["dev"]
        assert totals == {
            "online_seconds": 2 * 3600,
            "conflict_seconds": 3600,
        }

    def test_naive_event_times_are_localized_to_ward_tz(self):
        start, end = _ts(2026, 3, 1), _ts(2026, 3, 2)
        naive_events = [
            DeviceEvent(
                "dev",
                1,
                datetime(2026, 3, 1, 8, 0),
                state=STATE_ONLINE,
            ),
            DeviceEvent(
                "dev",
                2,
                datetime(2026, 3, 1, 10, 0),
                state=STATE_OFFLINE,
            ),
        ]
        aware_events = [
            DeviceEvent(
                "dev",
                1,
                datetime(2026, 3, 1, 8, 0, tzinfo=ZoneInfo(SHANGHAI)),
                state=STATE_ONLINE,
            ),
            DeviceEvent(
                "dev",
                2,
                datetime(2026, 3, 1, 10, 0, tzinfo=ZoneInfo(SHANGHAI)),
                state=STATE_OFFLINE,
            ),
        ]
        assert summarize(naive_events, SHANGHAI, start, end) == summarize(
            aware_events, SHANGHAI, start, end
        )

    def test_utc_window_boundaries_split_on_ward_natural_days(self):
        start = datetime(2026, 3, 8, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            DeviceEvent(
                "dev",
                1,
                datetime(2026, 3, 8, 0, 0, tzinfo=timezone.utc),
                state=STATE_ONLINE,
            ),
            DeviceEvent(
                "dev",
                2,
                datetime(2026, 3, 9, 16, 0, tzinfo=timezone.utc),
                state=STATE_OFFLINE,
            ),
        ]
        rows = summarize(events, SHANGHAI, start, end)
        by_day = _rows_by_day(rows)
        assert [row.day for row in rows] == [
            datetime(2026, 3, 8).date(),
            datetime(2026, 3, 9).date(),
            datetime(2026, 3, 10).date(),
        ]
        assert by_day[datetime(2026, 3, 8).date()].online_seconds == 16 * 3600
        assert by_day[datetime(2026, 3, 9).date()].online_seconds == 86400
        assert by_day[datetime(2026, 3, 10).date()].online_seconds == 0
        totals = total_interval_seconds(events, SHANGHAI, start, end)["dev"]
        assert totals["online_seconds"] == 40 * 3600
        assert sum(row.online_seconds for row in rows) == totals["online_seconds"]

    def test_numeric_timestamps_and_fixed_offset_tz(self):
        start = datetime(1970, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=8)))
        end = datetime(1970, 1, 2, 0, 0, tzinfo=timezone(timedelta(hours=8)))
        events = [
            DeviceEvent("dev", 1, 0, state=STATE_ONLINE),
            DeviceEvent("dev", 2, 1800, state=STATE_OFFLINE),
        ]
        rows = summarize(events, "+08:00", start, end)
        epoch_day = datetime(1970, 1, 1).date()
        assert rows == [DailyUptime("dev", epoch_day, 1800, 0)]

    def test_dst_spring_forward_day_has_23_hours(self):
        start = _ts(2026, 3, 7, 0, tz=NEW_YORK)
        end = _ts(2026, 3, 10, 0, tz=NEW_YORK)
        events = [
            DeviceEvent(
                "dev",
                1,
                datetime(2026, 3, 7, 12, tzinfo=ZoneInfo(NEW_YORK)),
                state=STATE_ONLINE,
            ),
            DeviceEvent(
                "dev",
                2,
                datetime(2026, 3, 9, 12, tzinfo=ZoneInfo(NEW_YORK)),
                state=STATE_OFFLINE,
            ),
        ]
        rows = summarize(events, NEW_YORK, start, end)
        by_day = _rows_by_day(rows)
        assert by_day[datetime(2026, 3, 7).date()].online_seconds == 12 * 3600
        assert by_day[datetime(2026, 3, 8).date()].online_seconds == 23 * 3600
        assert by_day[datetime(2026, 3, 9).date()].online_seconds == 12 * 3600

        totals = total_interval_seconds(events, NEW_YORK, start, end)["dev"]
        assert totals["online_seconds"] == 47 * 3600
        assert sum(row.online_seconds for row in rows) == 47 * 3600

    def test_dst_fall_back_day_has_25_hours(self):
        start = _ts(2026, 10, 31, 0, tz=NEW_YORK)
        end = _ts(2026, 11, 3, 0, tz=NEW_YORK)
        events = [
            DeviceEvent(
                "dev",
                1,
                datetime(2026, 10, 31, 12, tzinfo=ZoneInfo(NEW_YORK)),
                state=STATE_ONLINE,
            ),
            DeviceEvent(
                "dev",
                2,
                datetime(2026, 11, 2, 12, tzinfo=ZoneInfo(NEW_YORK)),
                state=STATE_OFFLINE,
            ),
        ]
        rows = summarize(events, NEW_YORK, start, end)
        by_day = _rows_by_day(rows)
        assert by_day[datetime(2026, 10, 31).date()].online_seconds == 12 * 3600
        assert by_day[datetime(2026, 11, 1).date()].online_seconds == 25 * 3600
        assert by_day[datetime(2026, 11, 2).date()].online_seconds == 12 * 3600

        totals = total_interval_seconds(events, NEW_YORK, start, end)["dev"]
        assert totals["online_seconds"] == 49 * 3600
        assert sum(row.online_seconds for row in rows) == 49 * 3600

    def test_correction_can_change_the_asserted_state(self):
        start, end = _ts(2026, 3, 1), _ts(2026, 3, 2)
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE),
            DeviceEvent("dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE),
            DeviceEvent(
                "dev",
                3,
                _ts(2026, 3, 1, 1),
                state=STATE_ONLINE,
                corrected_sequence=2,
            ),
            DeviceEvent("dev", 4, _ts(2026, 3, 1, 3), state=STATE_OFFLINE),
        ]
        rows = summarize(events, SHANGHAI, start, end)
        assert rows[0].online_seconds == 3 * 3600
        assert rows[0].conflict_seconds == 0

    def test_multiple_devices_are_independent_and_reconcile(self):
        start, end = _ts(2026, 3, 1), _ts(2026, 3, 3)
        events = [
            DeviceEvent("a", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE),
            DeviceEvent("a", 2, _ts(2026, 3, 1, 12), state=STATE_OFFLINE),
            DeviceEvent("b", 1, _ts(2026, 3, 1, 6), state=STATE_ONLINE, source="gw"),
            DeviceEvent(
                "b", 2, _ts(2026, 3, 1, 18), state=STATE_OFFLINE, source="sensor"
            ),
            DeviceEvent(
                "b", 3, _ts(2026, 3, 2, 0), state=STATE_OFFLINE, source="gw"
            ),
        ]
        rows = summarize(events, SHANGHAI, start, end)
        totals = total_interval_seconds(events, SHANGHAI, start, end)
        for device_id in ("a", "b"):
            device_rows = [row for row in rows if row.device_id == device_id]
            assert (
                sum(row.online_seconds for row in device_rows)
                == totals[device_id]["online_seconds"]
            )
            assert (
                sum(row.conflict_seconds for row in device_rows)
                == totals[device_id]["conflict_seconds"]
            )
        assert totals["a"]["online_seconds"] == 12 * 3600
        assert totals["b"] == {
            "online_seconds": 12 * 3600,
            "conflict_seconds": 6 * 3600,
        }

    def test_window_clips_segments_on_both_sides(self):
        start, end = _ts(2026, 3, 1, 1), _ts(2026, 3, 1, 3)
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE),
            DeviceEvent("dev", 2, _ts(2026, 3, 1, 4), state=STATE_OFFLINE),
        ]
        rows = summarize(events, SHANGHAI, start, end)
        assert rows == [
            DailyUptime("dev", datetime(2026, 3, 1).date(), 2 * 3600, 0)
        ]
        totals = total_interval_seconds(events, SHANGHAI, start, end)["dev"]
        assert totals["online_seconds"] == 2 * 3600

    def test_default_window_requires_a_closing_event(self):
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1), state=STATE_ONLINE),
        ]
        with pytest.raises(ValueError, match="explicit end horizon"):
            summarize(events, SHANGHAI)

    def test_open_ended_segment_is_counted_with_explicit_horizon(self):
        start, end = _ts(2026, 3, 1), _ts(2026, 3, 2)
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE),
        ]
        rows = summarize(events, SHANGHAI, start, end)
        assert rows[0].online_seconds == 86400
        totals = total_interval_seconds(events, SHANGHAI, start, end)["dev"]
        assert totals["online_seconds"] == 86400

    def test_every_local_day_in_window_gets_a_row(self):
        start, end = _ts(2026, 3, 1), _ts(2026, 3, 4)
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 2, 0), state=STATE_ONLINE),
            DeviceEvent("dev", 2, _ts(2026, 3, 2, 1), state=STATE_OFFLINE),
        ]
        rows = summarize(events, SHANGHAI, start, end)
        assert [row.day for row in rows] == [
            datetime(2026, 3, 1).date(),
            datetime(2026, 3, 2).date(),
            datetime(2026, 3, 3).date(),
        ]

    def test_empty_input_returns_empty_result(self):
        assert summarize([], SHANGHAI, _ts(2026, 3, 1), _ts(2026, 3, 2)) == []
        assert total_interval_seconds([]) == {}

    def test_partial_window_arguments_are_rejected(self):
        events = [
            DeviceEvent("dev", 1, _ts(2026, 3, 1), state=STATE_ONLINE),
            DeviceEvent("dev", 2, _ts(2026, 3, 1, 1), state=STATE_OFFLINE),
        ]
        with pytest.raises(ValueError, match="provided together"):
            summarize(events, SHANGHAI, start=_ts(2026, 3, 1))
        with pytest.raises(ValueError, match="provided together"):
            total_interval_seconds(events, SHANGHAI, end=_ts(2026, 3, 2))

    def test_default_window_reconciles_with_open_segment_device(self):
        events = [
            DeviceEvent("a", 1, _ts(2026, 3, 1, 0), state=STATE_ONLINE),
            DeviceEvent("a", 2, _ts(2026, 3, 1, 4), state=STATE_OFFLINE),
            DeviceEvent("b", 1, _ts(2026, 3, 1, 2), state=STATE_ONLINE),
        ]
        rows = summarize(events, SHANGHAI)
        totals = total_interval_seconds(events, SHANGHAI)
        for device_id in ("a", "b"):
            device_rows = [row for row in rows if row.device_id == device_id]
            assert (
                sum(row.online_seconds for row in device_rows)
                == totals[device_id]["online_seconds"]
            )
        assert totals["b"]["online_seconds"] == 2 * 3600

    def test_random_streams_always_reconcile_with_utc_measurement(self):
        import random

        rng = random.Random(20260308)
        origin = _ts(2026, 3, 1).int_timestamp
        horizon = _ts(2026, 3, 8).int_timestamp
        sources = ("gw", "sensor", "nurse")
        events = []
        for device_index in range(4):
            device_id = f"dev-{device_index}"
            sequence = 0
            for source in sources:
                cursor = origin + rng.randrange(0, 3600)
                while cursor < horizon:
                    sequence += 1
                    events.append(
                        DeviceEvent(
                            device_id,
                            sequence,
                            cursor,
                            state=rng.choice((STATE_ONLINE, STATE_OFFLINE)),
                            source=source,
                        )
                    )
                    cursor += rng.randrange(900, 4 * 3600)
        start = _ts(2026, 3, 1)
        end = _ts(2026, 3, 8)
        rows = summarize(events, SHANGHAI, start, end)
        totals = total_interval_seconds(events, SHANGHAI, start, end)
        for device_id in totals:
            device_rows = [row for row in rows if row.device_id == device_id]
            assert sum(row.online_seconds for row in device_rows) == totals[
                device_id
            ]["online_seconds"]
            assert sum(row.conflict_seconds for row in device_rows) == totals[
                device_id
            ]["conflict_seconds"]
            for row in device_rows:
                assert row.online_seconds + row.conflict_seconds <= 86400
