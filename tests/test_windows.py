from datetime import datetime, timedelta, timezone

import pytest

from arrow import arrow
from arrow.windows import (
    AssignmentSegment,
    ConflictSegment,
    MaintenanceWindow,
    WindowAssignment,
    assign_windows,
)

UTC = timezone.utc


def at(
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
) -> arrow.Arrow:
    return arrow.Arrow(2024, 1, 1, hour, minute, second, microsecond, tzinfo=UTC)


class TestMaintenanceWindow:
    def test_normalizes_naive_to_utc(self):
        window = MaintenanceWindow("k", datetime(2024, 1, 1), datetime(2024, 1, 2))

        assert window.start == arrow.Arrow(2024, 1, 1, tzinfo=UTC)
        assert window.end == arrow.Arrow(2024, 1, 2, tzinfo=UTC)
        assert window.start.tzinfo == UTC

    def test_normalizes_aware_offsets_to_same_utc_instant(self):
        plus_five = timezone(timedelta(hours=5))
        local = MaintenanceWindow(
            "k",
            datetime(2024, 1, 1, 5, tzinfo=plus_five),
            datetime(2024, 1, 1, 9, tzinfo=plus_five),
        )
        utc = MaintenanceWindow("k", at(0), at(4))

        assert local.start == utc.start
        assert local.end == utc.end

    def test_rejects_empty_or_inverted_window(self):
        with pytest.raises(ValueError):
            MaintenanceWindow("bad", at(2), at(2))
        with pytest.raises(ValueError):
            MaintenanceWindow("bad", at(3), at(2))

    def test_contains_is_half_open(self):
        window = MaintenanceWindow("k", at(1), at(2))

        assert window.contains(at(1))
        assert window.contains(at(1, 59))
        assert not window.contains(at(2))
        assert not window.contains(at(0, 59))

    def test_from_span_builds_half_open_frame(self):
        window = MaintenanceWindow.from_span(
            "k", arrow.Arrow(2024, 1, 1, 10, tzinfo=UTC), "day"
        )

        assert window.start == arrow.Arrow(2024, 1, 1, tzinfo=UTC)
        assert window.end == arrow.Arrow(2024, 1, 2, tzinfo=UTC)
        assert not window.contains(arrow.Arrow(2024, 1, 2, tzinfo=UTC))

    def test_from_span_month_and_week(self):
        month = MaintenanceWindow.from_span(
            "m", arrow.Arrow(2024, 2, 15, tzinfo=UTC), "month"
        )
        assert month.start == arrow.Arrow(2024, 2, 1, tzinfo=UTC)
        assert month.end == arrow.Arrow(2024, 3, 1, tzinfo=UTC)

        week = MaintenanceWindow.from_span(
            "w", arrow.Arrow(2024, 1, 3, tzinfo=UTC), "week", week_start=1
        )
        assert week.start == arrow.Arrow(2024, 1, 1, tzinfo=UTC)
        assert week.end == arrow.Arrow(2024, 1, 8, tzinfo=UTC)

    def test_from_span_ignores_display_bounds(self):
        window = MaintenanceWindow.from_span(
            "k", arrow.Arrow(2024, 1, 1, 10, tzinfo=UTC), "hour", bounds="[)"
        )
        assert window.start == at(10)
        assert window.end == at(11)

    def test_from_span_range_builds_abutting_half_open_windows(self):
        windows = MaintenanceWindow.from_span_range(
            "svc", "hour", datetime(2024, 1, 1, 12), datetime(2024, 1, 1, 15), tz="UTC"
        )

        assert len(windows) == 4
        assert all(window.source == "svc" for window in windows)
        assert [window.start for window in windows] == [
            at(12),
            at(13),
            at(14),
            at(15),
        ]
        assert [window.end for window in windows] == [
            at(13),
            at(14),
            at(15),
            at(16),
        ]
        assert all(
            windows[index].end == windows[index + 1].start
            for index in range(len(windows) - 1)
        )


class TestAssignWindows:
    def test_single_window_is_one_segment(self):
        assignment = assign_windows([MaintenanceWindow("a", at(0), at(3))])

        assert assignment.segments == [
            AssignmentSegment(at(0), at(3), assignment.windows[0])
        ]
        assert assignment.conflicts == []

    def test_higher_priority_wins_overlap(self):
        low = MaintenanceWindow("a", at(0), at(3), priority=1)
        high = MaintenanceWindow("b", at(2), at(4), priority=2)

        assignment = assign_windows([low, high])

        assert [(s.start, s.end, s.key) for s in assignment.segments] == [
            (at(0), at(2), "a"),
            (at(2), at(4), "b"),
        ]

    def test_three_layered_priorities(self):
        x = MaintenanceWindow("x", at(0), at(6), priority=1)
        y = MaintenanceWindow("y", at(1), at(5), priority=2)
        z = MaintenanceWindow("z", at(2), at(4), priority=3)

        assignment = assign_windows([x, y, z])

        assert [(s.start, s.end, s.key) for s in assignment.segments] == [
            (at(0), at(1), "x"),
            (at(1), at(2), "y"),
            (at(2), at(4), "z"),
            (at(4), at(5), "y"),
            (at(5), at(6), "x"),
        ]

    def test_same_priority_different_keys_is_conflict(self):
        c1 = MaintenanceWindow("c1", at(1), at(2), priority=5)
        c2 = MaintenanceWindow("c2", at(1), at(2), priority=5)

        assignment = assign_windows([c1, c2])

        assert assignment.segments == []
        assert len(assignment.conflicts) == 1
        conflict = assignment.conflicts[0]
        assert conflict.start == at(1)
        assert conflict.end == at(2)
        assert set(conflict.keys) == {"c1", "c2"}
        assert conflict.priority == 5
        assert assignment.winner_at(at(1, 30)) is None
        assert assignment.conflict_at(at(1, 30)) is conflict

    def test_same_priority_same_key_is_not_conflict(self):
        first = MaintenanceWindow(
            "k", at(0), at(2), priority=1, source="first"
        )
        second = MaintenanceWindow(
            "k", at(1), at(3), priority=1, source="second"
        )

        assignment = assign_windows([first, second])

        assert assignment.conflicts == []
        assert [(s.start, s.end, s.source) for s in assignment.segments] == [
            (at(0), at(2), "first"),
            (at(2), at(3), "second"),
        ]

    def test_lower_priority_does_not_create_conflict(self):
        high = MaintenanceWindow("a", at(0), at(4), priority=9)
        low = MaintenanceWindow("b", at(1), at(3), priority=1)

        assignment = assign_windows([high, low])

        assert assignment.conflicts == []
        assert [(s.start, s.end, s.key) for s in assignment.segments] == [
            (at(0), at(4), "a")
        ]

    def test_compacts_when_winner_stays_constant(self):
        outer = MaintenanceWindow("outer", at(0), at(5), priority=2)
        blip = MaintenanceWindow("blip", at(2), at(3), priority=1)

        assignment = assign_windows([outer, blip])

        # The lower-priority blip never wins, so outer covers one compacted span.
        assert [(s.start, s.end, s.key) for s in assignment.segments] == [
            (at(0), at(5), "outer")
        ]

    def test_compacts_adjacent_same_key_and_source_windows(self):
        first = MaintenanceWindow("svc", at(0), at(2), source="series")
        second = MaintenanceWindow("svc", at(2), at(4), source="series")

        assignment = assign_windows([first, second])

        assert [(s.start, s.end, s.key) for s in assignment.segments] == [
            (at(0), at(4), "svc")
        ]

    def test_conflicts_with_different_participants_are_separate(self):
        c1 = MaintenanceWindow("c1", at(0), at(4), priority=5)
        c2 = MaintenanceWindow("c2", at(0), at(4), priority=5)
        c3 = MaintenanceWindow("c3", at(2), at(3), priority=5)

        assignment = assign_windows([c1, c2, c3])

        assert [
            (c.start, c.end, tuple(sorted(c.keys))) for c in assignment.conflicts
        ] == [
            (at(0), at(2), ("c1", "c2")),
            (at(2), at(3), ("c1", "c2", "c3")),
            (at(3), at(4), ("c1", "c2")),
        ]

    def test_boundaries_are_real_sub_minute_instants(self):
        early = MaintenanceWindow("p", at(0, 0, 0, 0), at(0, 0, 0, 500), priority=1)
        late = MaintenanceWindow(
            "q", at(0, 0, 0, 250), at(0, 0, 0, 750), priority=2
        )

        assignment = assign_windows([early, late])

        assert [(s.start, s.end, s.key) for s in assignment.segments] == [
            (at(0, 0, 0, 0), at(0, 0, 0, 250), "p"),
            (at(0, 0, 0, 250), at(0, 0, 0, 750), "q"),
        ]
        assert assignment.winner_at(at(0, 0, 0, 249)).key == "p"
        assert assignment.winner_at(at(0, 0, 0, 250)).key == "q"

    def test_segments_and_conflicts_partition_coverage(self):
        a = MaintenanceWindow("a", at(0), at(4), priority=1)
        b = MaintenanceWindow("b", at(2), at(6), priority=2)
        tie = MaintenanceWindow("c", at(5), at(7), priority=2)

        assignment = assign_windows([a, b, tie])

        assert assignment.uncovered() == []
        assert assignment.is_covered(at(0))
        assert not assignment.is_covered(at(8))
        # End instants are excluded.
        assert not assignment.is_covered(at(7))


class TestOffsetContinuity:
    def test_adjacent_local_windows_abut_at_real_utc_instant(self):
        nyc = "America/New_York"
        # 23:00 local on the DST transition evening and 00:00 local the next day.
        first = MaintenanceWindow.from_span(
            "w1", arrow.Arrow(2024, 3, 9, 23, tzinfo=nyc), "hour"
        )
        second = MaintenanceWindow.from_span(
            "w2", arrow.Arrow(2024, 3, 10, 0, tzinfo=nyc), "hour"
        )

        assignment = assign_windows([first, second])
        segments = sorted(assignment.segments, key=lambda segment: segment.start)

        assert segments[0].end == segments[1].start
        assert assignment.uncovered() == []
        # UTC boundaries reflect the spring-forward offset change.
        assert segments[0].start == arrow.Arrow(2024, 3, 10, 4, tzinfo=UTC)
        assert segments[1].start == arrow.Arrow(2024, 3, 10, 5, tzinfo=UTC)


class TestReassign:
    def setup_method(self):
        self.a = MaintenanceWindow("a", at(0), at(4), priority=1)
        self.b = MaintenanceWindow("b", at(2), at(6), priority=2)
        self.assignment = assign_windows([self.a, self.b])

    def test_no_change_is_empty_diff(self):
        diff = self.assignment.reassign([self.a, self.b])

        assert diff.is_empty
        assert diff.revoked == []
        assert diff.granted == []

    def test_shrinking_window_revokes_only_changed_instants(self):
        shrunk_b = MaintenanceWindow("b", at(3), at(6), priority=2)

        diff = self.assignment.reassign([self.a, shrunk_b])

        assert [(r.start, r.end, r.key) for r in diff.revoked] == [
            (at(2), at(3), "b")
        ]
        assert [(g.start, g.end, g.key) for g in diff.granted] == [
            (at(2), at(3), "a")
        ]
        # Ownership of the instants b still covers is untouched.
        assert diff.assignment.winner_at(at(4)).key == "b"

    def test_adding_window_grants_only_newly_won_instants(self):
        new_window = MaintenanceWindow("g", at(5), at(7), priority=9)

        diff = self.assignment.reassign([self.a, self.b, new_window])

        assert [(r.start, r.end, r.key) for r in diff.revoked] == [
            (at(5), at(6), "b")
        ]
        assert [
            (segment.start, segment.end, segment.key)
            for segment in diff.granted
        ] == [(at(5), at(7), "g")]

    def test_removing_window(self):
        diff = self.assignment.reassign([self.a])

        assert [(r.start, r.end, r.key) for r in diff.revoked] == [
            (at(2), at(6), "b")
        ]
        assert [(g.start, g.end, g.key) for g in diff.granted] == [
            (at(2), at(4), "a")
        ]

    def test_priority_change_emerges_conflict(self):
        c = MaintenanceWindow("c", at(2), at(3), priority=2)

        diff = self.assignment.reassign([self.a, self.b, c])

        assert len(diff.granted) == 1
        conflict = diff.granted[0]
        assert isinstance(conflict, ConflictSegment)
        assert (conflict.start, conflict.end) == (at(2), at(3))
        assert set(conflict.keys) == {"b", "c"}

    def test_change_far_away_leaves_other_attribution_intact(self):
        moved_a = MaintenanceWindow("a", at(10), at(11), priority=1)

        diff = self.assignment.reassign([moved_a, self.b])

        assert [(r.start, r.end, r.key) for r in diff.revoked] == [
            (at(0), at(2), "a")
        ]
        assert [(g.start, g.end, g.key) for g in diff.granted] == [
            (at(10), at(11), "a")
        ]
        assert diff.assignment.winner_at(at(4)).key == "b"

    def test_resolving_conflict_by_priority_bump(self):
        tied = [
            MaintenanceWindow("a", at(0), at(4), priority=2),
            MaintenanceWindow("b", at(0), at(4), priority=2),
        ]
        conflicted = assign_windows(tied)
        promoted = MaintenanceWindow("a", at(0), at(4), priority=3)

        diff = conflicted.reassign([promoted, tied[1]])

        assert len(diff.revoked) == 1
        assert isinstance(diff.revoked[0], ConflictSegment)
        assert (diff.revoked[0].start, diff.revoked[0].end) == (at(0), at(4))
        assert [(g.start, g.end, g.key) for g in diff.granted] == [
            (at(0), at(4), "a")
        ]

    def test_clearing_all_windows_revokes_everything(self):
        diff = self.assignment.reassign([])

        assert diff.assignment.segments == []
        assert diff.assignment.conflicts == []
        assert diff.granted == []
        assert len(diff.revoked) == 2

    def test_reordering_windows_is_no_change(self):
        diff = self.assignment.reassign([self.b, self.a])

        assert diff.is_empty

    def test_diff_returns_rebuilt_assignment(self):
        shrunk_b = MaintenanceWindow("b", at(3), at(6), priority=2)
        diff = self.assignment.reassign([self.a, shrunk_b])

        assert isinstance(diff.assignment, WindowAssignment)
        assert [(s.start, s.end, s.key) for s in diff.assignment.segments] == [
            (at(0), at(3), "a"),
            (at(3), at(6), "b"),
        ]
