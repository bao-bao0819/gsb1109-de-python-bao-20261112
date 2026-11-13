import pytest

import portion as P


class TestScaleTableBasics:
    def test_empty_table(self):
        table = P.ScaleTable()
        assert table.sources() == ()
        assert table.coverage().empty
        assert table.effective() == []
        assert table.effective_at(0) is None
        assert table.evidence() == []
        assert table.revision == 0

    def test_single_source_contribution(self):
        table = P.ScaleTable()
        result = table.contribute("s1", P.closed(0, 10), 1.5)

        assert table.sources() == ("s1",)
        assert table.coverage() == P.closed(0, 10)
        assert table.coverage("s1") == P.closed(0, 10)

        segment, effective = table.effective()[0]
        assert segment == P.closed(0, 10)
        assert effective.value == 1.5
        assert effective.sources == ("s1",)
        assert effective.defined and not effective.conflicting
        assert effective.values == (1.5,)

        assert [change.kind for change in result] == [P.NEW_COVERAGE]
        assert table.revision == 1

    def test_contribution_keeps_each_source(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 10), 1.0)
        table.contribute("s2", P.closed(0, 10), 1.0)

        evidence = table.evidence()
        assert [entry[0] for entry in evidence] == ["s1", "s2"]
        assert all(entry[2] == 1.0 for entry in evidence)

        effective = table.effective_at(5)
        assert effective.value == 1.0
        assert effective.sources == ("s1", "s2")

    def test_unknown_source_raises(self):
        table = P.ScaleTable()
        with pytest.raises(KeyError):
            table.coverage("missing")
        with pytest.raises(KeyError):
            table.retract("missing")

    def test_partial_retraction_is_idempotent(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 4), 1.0)

        # An unknown source on a segment retraction is a no-op, not an error.
        missing = table.retract("ghost", P.closed(0, 2))
        assert len(missing) == 0

        # Withdrawing the whole contribution in two overlapping halves.
        first = table.retract("s1", P.closed(0, 2))
        second = table.retract("s1", P.closed(2, 4))
        assert "s1" not in table.sources()
        assert table.effective() == []

        # Further retractions on the emptied source stay empty.
        assert len(table.retract("s1", P.closed(0, 4))) == 0
        assert len(first) == 1 and len(second) == 1

    def test_singleton_contribution_and_query(self):
        table = P.ScaleTable()
        table.contribute("s1", 3, 9.0)
        assert table.coverage() == P.singleton(3)
        assert table.effective_at(3).value == 9.0
        assert table.effective_at(2) is None

    def test_overwrite_only_affects_own_evidence(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 4), 1.0)
        table.contribute("s2", P.closed(0, 4), 1.0)
        table.contribute("s1", P.closed(0, 4), 2.0)

        conflict = table.effective_at(1)
        assert conflict.conflicting
        assert conflict.sources == ("s1", "s2")
        assert conflict.values == (2.0, 1.0)
        assert table.evidence_at(1) == [("s1", 2.0), ("s2", 1.0)]

    def test_sources_filtered_by_interval(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 5), 1.0)
        table.contribute("s2", P.closed(5, 10), 1.0)
        assert table.sources(P.closed(0, 2)) == ("s1",)
        assert table.sources(P.closed(6, 7)) == ("s2",)
        assert table.sources(P.singleton(5)) == ("s1", "s2")


class TestEndpointConflict:
    def _build(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 10), 1.0)
        table.contribute("s2", P.closed(0, 10), 1.0)
        return table

    def test_third_source_disagrees_only_at_endpoint(self):
        table = self._build()
        result = table.contribute("s3", P.closed(8, 10), 2.0)

        view = table.effective()
        assert len(view) == 2

        agreed_segment, agreed = view[0]
        assert agreed_segment == P.closedopen(0, 8)
        assert agreed.value == 1.0
        assert agreed.sources == ("s1", "s2")

        conflict_segment, conflict = view[1]
        assert conflict_segment == P.closed(8, 10)
        assert conflict.conflicting
        assert conflict.value is None
        assert conflict.sources == ("s1", "s2", "s3")
        assert conflict.values == (1.0, 2.0)

        # The endpoint itself carries the conflict.
        assert table.effective_at(8).conflicting
        assert not table.effective_at(7).conflicting

        assert [change.kind for change in result] == [P.CONFLICT_EMERGED]
        assert result.conflict_emerged[0].interval == P.closed(8, 10)

    def test_conflict_visible_in_restricted_query(self):
        table = self._build()
        table.contribute("s3", P.closed(8, 10), 2.0)

        queried = table.effective(P.closed(9, 12))
        assert len(queried) == 1
        segment, effective = queried[0]
        assert segment == P.closed(9, 10)
        assert effective.conflicting


class TestSameValueDistinctEvidence:
    def test_adjacent_segments_remain_separable(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 10), 7.0)
        table.contribute("s2", P.closed(5, 10), 7.0)

        view = table.effective()
        assert len(view) == 2

        first_segment, first = view[0]
        second_segment, second = view[1]
        assert first_segment == P.closedopen(0, 5)
        assert second_segment == P.closed(5, 10)
        assert first.value == second.value == 7.0
        assert first.sources == ("s1",)
        assert second.sources == ("s1", "s2")

    def test_segments_can_be_queried_independently(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 10), 7.0)
        table.contribute("s2", P.closed(5, 10), 7.0)

        left = table.effective(P.closed(1, 2))
        right = table.effective(P.closed(6, 7))
        assert len(left) == 1 and left[0][1].sources == ("s1",)
        assert len(right) == 1 and right[0][1].sources == ("s1", "s2")
        assert table.effective_at(4).sources == ("s1",)
        assert table.effective_at(6).sources == ("s1", "s2")


class TestReconciliationKinds:
    def test_value_changed(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 4), 1.0)
        result = table.contribute("s1", P.closed(0, 4), 9.0)

        assert [change.kind for change in result] == [P.VALUE_CHANGED]
        change = result.value_changed[0]
        assert change.before.value == 1.0
        assert change.after.value == 9.0
        assert change.interval == P.closed(0, 4)
        assert table.effective_at(2).value == 9.0

    def test_evidence_lost_value_unchanged(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 4), 1.0)
        table.contribute("s2", P.closed(0, 4), 1.0)
        result = table.retract("s2")

        assert [change.kind for change in result] == [P.EVIDENCE_LOST]
        change = result.evidence_lost[0]
        assert change.before.value == change.after.value == 1.0
        assert change.before.sources == ("s1", "s2")
        assert change.after.sources == ("s1",)
        assert table.effective_at(2).value == 1.0

    def test_evidence_gained_value_unchanged(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 4), 1.0)
        result = table.contribute("s2", P.closed(2, 6), 1.0)

        gained = result.evidence_gained
        assert len(gained) == 1
        assert gained[0].interval == P.closed(2, 4)
        assert gained[0].before.sources == ("s1",)
        assert gained[0].after.sources == ("s1", "s2")
        assert result.new_coverage[0].interval == P.openclosed(4, 6)

    def test_conflict_resolved_to_unique_value(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 4), 1.0)
        table.contribute("s2", P.closed(0, 4), 2.0)
        result = table.retract("s2")

        assert [change.kind for change in result] == [P.CONFLICT_RESOLVED]
        change = result.conflict_resolved[0]
        assert change.before.conflicting
        assert not change.after.conflicting
        assert change.after.value == 1.0
        assert table.effective_at(2).value == 1.0

    def test_coverage_lost(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 4), 1.0)
        result = table.retract("s1")

        assert [change.kind for change in result] == [P.COVERAGE_LOST]
        assert result.coverage_lost[0].after is None
        assert table.effective_at(2) is None
        assert table.sources() == ()

    def test_retracting_uncovered_segment_is_noop(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 4), 1.0)
        revision = table.revision
        result = table.retract("s1", P.closed(10, 12))
        assert len(result) == 0
        assert table.revision == revision
        assert table.coverage() == P.closed(0, 4)


class TestRetractionEquivalence:
    def _build(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 10), 1.0)
        table.contribute("s2", P.closed(0, 10), 1.0)
        table.contribute("s3", P.closed(8, 10), 2.0)
        return table

    def test_half_then_rest_equals_whole_retraction(self):
        split = self._build()
        first = split.retract("s3", P.closed(8, 9))
        second = split.retract("s3", P.closed(9, 10))

        whole = self._build().retract("s3")

        # Both paths converge on the same effective calibration.
        assert split.effective() == self._expected_final_view()
        assert [change.kind for change in first] == [P.CONFLICT_RESOLVED]
        assert first.conflict_resolved[0].interval == P.closed(8, 9)

        # The second step also recovers from conflict on the remaining piece.
        assert [change.kind for change in second] == [P.CONFLICT_RESOLVED]
        assert second.conflict_resolved[0].interval == P.openclosed(9, 10)

        # The whole retraction explains the same segment as one conflict recovery.
        assert [change.kind for change in whole] == [P.CONFLICT_RESOLVED]
        assert whole.conflict_resolved[0].interval == P.closed(8, 10)

    def _expected_final_view(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 10), 1.0)
        table.contribute("s2", P.closed(0, 10), 1.0)
        return table.effective()

    def test_both_paths_share_final_effective_scale(self):
        split = self._build()
        split.retract("s3", P.closed(8, 9))
        split.retract("s3", P.closed(9, 10))

        whole_table = self._build()
        whole_table.retract("s3")

        assert split.effective() == whole_table.effective()
        assert split.evidence() == whole_table.evidence()
        assert split.sources() == whole_table.sources() == ("s1", "s2")

    def test_partial_retraction_preserves_other_sources(self):
        table = self._build()
        table.retract("s3", P.singleton(9))

        point_9 = table.effective_at(9)
        assert not point_9.conflicting
        assert point_9.sources == ("s1", "s2")

        point_8 = table.effective_at(8)
        assert point_8.conflicting
        assert "s3" in point_8.sources

    def test_retract_source_alias(self):
        table = self._build()
        result = table.retract_source("s3")
        assert "s3" not in table.sources()
        assert result.conflict_resolved


class TestCopyAndReconciliation:
    def test_copy_is_independent(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 4), 1.0)
        clone = table.copy()
        clone.retract("s1")

        assert "s1" in table.sources()
        assert "s1" not in clone.sources()
        assert table.effective_at(2).value == 1.0
        assert clone.effective_at(2) is None

    def test_reconciliation_domains_and_boolean(self):
        table = P.ScaleTable()
        result = table.contribute("s1", P.closed(0, 4), 1.0)
        assert result.domains() == P.closed(0, 4)
        assert bool(result) and len(result) == 1

        noop = table.retract("s1", P.closed(10, 12))
        assert not noop and noop.domains().empty

    def test_reconciliation_is_iterable(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 10), 1.0)
        result = table.contribute("s2", P.closed(5, 10), 2.0)

        kinds = {change.kind for change in result}
        assert kinds == {P.CONFLICT_EMERGED}
        assert result.conflict_emerged[0].interval == P.closed(5, 10)


class TestGapsAndSegments:
    def test_gap_between_same_value_segments_is_preserved(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 2), 1.0)
        table.contribute("s1", P.closed(5, 7), 1.0)

        view = table.effective()
        assert len(view) == 2
        assert view[0][0] == P.closed(0, 2)
        assert view[1][0] == P.closed(5, 7)
        assert table.effective_at(3) is None
        assert table.coverage() == P.closed(0, 2) | P.closed(5, 7)

    def test_non_adjacent_changes_are_not_merged(self):
        table = P.ScaleTable()
        table.contribute("s1", P.closed(0, 10), 1.0)
        table.contribute("s2", P.closed(0, 2), 2.0)
        table.contribute("s2", P.closed(8, 10), 2.0)

        result = table.retract("s2")
        intervals = [change.interval for change in result.conflict_resolved]
        assert len(intervals) == 2
        assert intervals == [P.closed(0, 2), P.closed(8, 10)]
        assert result.domains() == P.closed(0, 2) | P.closed(8, 10)

    def test_partition_keeps_inner_boundary_under_wider_interval(self):
        table = P.ScaleTable()
        table.contribute("a", P.closed(0, 4), 1.0)
        table.contribute("b", P.closed(2, 6), 1.0)
        table.contribute("c", P.closed(0, 6), 2.0)

        conflict = table.effective()
        domains = [segment for segment, _ in conflict]
        assert P.closedopen(0, 2) in domains
        assert P.closed(2, 4) in domains
        assert P.openclosed(4, 6) in domains
        for _, effective in conflict:
            assert effective.conflicting
