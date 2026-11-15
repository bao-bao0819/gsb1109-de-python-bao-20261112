import pytest

import portion as P
from portion.publish import (
    REASON_BUCKET_BELOW_K,
    REASON_SEGMENT_BELOW_K,
)


def sources_of(plan):
    return [tuple(sorted(rng.sources)) for rng in plan.ranges]


class TestPlanPublication:
    def test_simple_distribution(self):
        buckets = [[1, 2], [3, 4], [1, 5]]
        plan = P.plan_publication(buckets, k=2)

        assert plan.suppressed == ()
        assert len(plan.ranges) == 3
        assert [(r.start, r.stop) for r in plan.ranges] == [
            (0, 1),
            (1, 2),
            (2, 3),
        ]
        assert sources_of(plan) == [(1, 2), (3, 4), (1, 5)]
        assert [r.bucket_count for r in plan.ranges] == [1, 1, 1]
        assert [r.span for r in plan.ranges] == [1, 1, 1]

    def test_same_source_counts_once(self):
        # Every bucket alone has only one distinct source, but adjacent
        # buckets together aggregate enough distinct sources.
        buckets = [{1}, {1, 2}, {2, 3}]
        plan = P.plan_publication(buckets, k=2)

        assert plan.suppressed == ()
        assert [(r.start, r.stop) for r in plan.ranges] == [(0, 2), (2, 3)]
        assert sources_of(plan) == [(1, 2), (2, 3)]

    def test_maximum_number_of_ranges(self):
        # Both [0, 3) and [0, 1)+[1, 3) are valid, so the split wins.
        buckets = [{1, 2}, {3}, {3, 4}]
        plan = P.plan_publication(buckets, k=2)
        assert len(plan.ranges) == 2
        assert [(r.start, r.stop) for r in plan.ranges] == [(0, 1), (1, 3)]

    def test_tie_break_on_max_span(self):
        # The deficient bucket 1 must merge either left or right. Both
        # options give three ranges; the one avoiding the span-10 bucket has
        # the smaller largest span and must be chosen.
        buckets = [
            P.Bucket({1, 2}, span=1),
            P.Bucket({3}, span=1),
            P.Bucket({4, 5}, span=10),
            P.Bucket({6, 7}, span=1),
        ]
        plan = P.plan_publication(buckets, k=2)
        assert len(plan.ranges) == 3
        assert [(r.start, r.stop) for r in plan.ranges] == [
            (0, 2),
            (2, 3),
            (3, 4),
        ]
        assert [r.span for r in plan.ranges] == [2, 10, 1]
        assert max(r.span for r in plan.ranges) == 10

    def test_tie_break_on_lexicographic_cuts(self):
        # Two partitions give two ranges with the same largest span; the
        # lexicographically earliest cut point wins.
        buckets = [
            P.Bucket({1, 2}, span=2),
            P.Bucket({3}, span=2),
            P.Bucket({4, 5}, span=2),
        ]
        plan = P.plan_publication(buckets, k=2)
        assert [(r.start, r.stop) for r in plan.ranges] == [(0, 1), (1, 3)]

    def test_legal_boundaries_are_respected(self):
        buckets = [{1, 2}, {3, 4}, {5, 6}]
        plan = P.plan_publication(buckets, k=2, legal_boundaries=[1])
        assert [(r.start, r.stop) for r in plan.ranges] == [
            (0, 1),
            (1, 2),
            (2, 3),
        ]

        # The boundary forbids merging buckets 0 and 1 even when required.
        buckets = [{1}, {1, 2}, {3, 4}]
        plan = P.plan_publication(buckets, k=2, legal_boundaries=[1])
        assert [(r.start, r.stop) for r in plan.ranges] == [(1, 2), (2, 3)]
        assert len(plan.suppressed) == 1
        suppressed = plan.suppressed[0]
        assert (suppressed.start, suppressed.stop) == (0, 1)
        assert suppressed.sources == frozenset({1})
        codes = [reason.code for reason in suppressed.reasons]
        assert REASON_SEGMENT_BELOW_K in codes
        assert REASON_BUCKET_BELOW_K in codes
        assert all(
            reason.bucket == 0
            for reason in suppressed.reasons
            if reason.code == REASON_BUCKET_BELOW_K
        )

    def test_suppressed_segment_is_not_completed_by_neighbours(self):
        # Bucket 1 cannot be helped by buckets 0 or 2 once the legal
        # boundaries isolate it.
        buckets = [{1, 2}, {9}, {3, 4}]
        plan = P.plan_publication(buckets, k=2, legal_boundaries=[1, 2])
        assert len(plan.ranges) == 2
        assert [(r.start, r.stop) for r in plan.ranges] == [(0, 1), (2, 3)]
        assert len(plan.suppressed) == 1
        assert (plan.suppressed[0].start, plan.suppressed[0].stop) == (1, 2)

    def test_entire_distribution_suppressed(self):
        plan = P.plan_publication([{1}, {1}], k=2)
        assert plan.ranges == ()
        assert len(plan.suppressed) == 1
        suppressed = plan.suppressed[0]
        assert (suppressed.start, suppressed.stop) == (0, 2)
        assert suppressed.sources == frozenset({1})
        assert suppressed.bucket_count == 2
        assert [reason.code for reason in suppressed.reasons] == [
            REASON_SEGMENT_BELOW_K,
            REASON_BUCKET_BELOW_K,
            REASON_BUCKET_BELOW_K,
        ]

    def test_empty_distribution(self):
        plan = P.plan_publication([], k=1)
        assert plan.ranges == ()
        assert plan.suppressed == ()

    def test_span_accumulation(self):
        buckets = [
            P.Bucket({1}, span=2),
            P.Bucket({1, 2}, span=3),
            P.Bucket({3}, span=4),
            P.Bucket({3, 4}, span=5),
        ]
        plan = P.plan_publication(buckets, k=2)
        assert [(r.start, r.stop, r.span) for r in plan.ranges] == [
            (0, 2, 5),
            (2, 4, 9),
        ]

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            P.plan_publication([{1}], k=0)
        with pytest.raises(ValueError):
            P.plan_publication([{1}], k=1, legal_boundaries=[0])
        with pytest.raises(ValueError):
            P.plan_publication([{1}], k=1, legal_boundaries=[1])
        with pytest.raises(ValueError):
            P.plan_publication([P.Bucket({1}, span=-1)], k=1)

    def test_determinism_and_idempotence(self):
        buckets = [{1}, {2}, {1, 2}, {3}, {3, 4}]
        first = P.plan_publication(buckets, k=2)
        second = P.plan_publication(buckets, k=2)
        assert first == second
