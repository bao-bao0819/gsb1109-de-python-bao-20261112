import pytest

import portion as P


def make_migrator(**kwargs):
    old_buckets = [P.closedopen(0, 10), P.closedopen(10, 20)]
    new_buckets = [
        P.closedopen(0, 5),
        P.closedopen(5, 15),
        P.closedopen(15, 20),
    ]
    old_counts = {
        P.closedopen(0, 10): 10,
        P.closedopen(10, 20): 10,
    }
    old_counts.update(kwargs.pop("old_counts", {}))
    return P.HistogramMigrator(
        old_buckets, new_buckets, old_counts, kwargs.pop("exact_counts", None)
    )


class TestSingleBucketMigration:
    def test_exact_when_layouts_coincide(self):
        old = [P.closedopen(0, 10), P.closedopen(10, 20)]
        new = [P.closedopen(0, 10), P.closedopen(10, 20)]
        counts = {old[0]: 4, old[1]: 6}
        migrator = P.HistogramMigrator(old, new, counts)

        first = migrator.migrate(new[0])
        second = migrator.migrate(new[1])
        assert first.status is P.MigrationStatus.EXACT
        assert (first.lower, first.upper) == (4, 4)
        assert second.status is P.MigrationStatus.EXACT
        assert second.as_range() == (6, 6)

    def test_bounded_when_boundary_moves(self):
        migrator = make_migrator()
        n1, n2, n3 = migrator.new_buckets

        assert migrator.migrate(n1).status is P.MigrationStatus.BOUNDED
        assert migrator.migrate(n1).as_range() == (0, 10)
        assert migrator.migrate(n2).as_range() == (0, 20)
        assert migrator.migrate(n3).as_range() == (0, 10)

    def test_exact_threshold_is_peeled_off(self):
        migrator = make_migrator(exact_counts={10: 3})
        n1, n2, n3 = migrator.new_buckets

        # Threshold 10 lives in the second old bucket and lands in n2.
        assert migrator.migrate(n1).as_range() == (0, 10)
        assert migrator.migrate(n2).as_range() == (3, 20)
        assert migrator.migrate(n3).as_range() == (0, 7)

    def test_exact_bucket_from_threshold_only(self):
        old = [P.closedopen(0, 10)]
        new = [P.closedopen(0, 10)]
        migrator = P.HistogramMigrator(
            old, new, {old[0]: 3}, exact_counts={5: 3}
        )
        verdict = migrator.migrate(new[0])
        assert verdict.status is P.MigrationStatus.EXACT
        assert verdict.as_range() == (3, 3)

    def test_migrate_all_keeps_order(self):
        migrator = make_migrator()
        verdicts = migrator.migrate_all()
        assert [verdict.bucket for verdict in verdicts] == list(
            migrator.new_buckets
        )


class TestGroupRange:
    def test_group_bounds_are_tighter_than_sum_of_maxima(self):
        migrator = make_migrator()
        n1, n2, n3 = migrator.new_buckets

        group = migrator.group_range([n1, n2])
        assert group.status is P.MigrationStatus.BOUNDED
        # The shared old pool is counted once, not twice, and is forced in.
        assert group.as_range() == (10, 20)
        assert group.upper < migrator.migrate(n1).upper + migrator.migrate(n2).upper

    def test_group_lower_bound_forces_internal_pools(self):
        migrator = make_migrator()
        n1, n2, n3 = migrator.new_buckets

        # n1 and n2 together cover every destination of the first old pool,
        # so its 10 samples are forced into the group.
        assert migrator.group_range([n1, n2]).lower == 10
        # All buckets together recover the whole batch exactly.
        whole = migrator.group_range([n1, n2, n3])
        assert whole.status is P.MigrationStatus.EXACT
        assert whole.as_range() == (20, 20)

    def test_disjoint_pools_group_is_sum(self):
        migrator = make_migrator()
        n1, n2, n3 = migrator.new_buckets

        # n1 and n3 draw from distinct old pools.
        group = migrator.group_range([n1, n3])
        assert group.as_range() == (0, 20)

    def test_single_bucket_group_matches_migration(self):
        migrator = make_migrator()
        n1 = migrator.new_buckets[0]
        group = migrator.group_range(n1)
        verdict = migrator.migrate(n1)
        assert group.as_range() == verdict.as_range()

    def test_duplicate_buckets_are_counted_once(self):
        migrator = make_migrator()
        n1, n2 = migrator.new_buckets[:2]
        once = migrator.group_range([n1, n2])
        twice = migrator.group_range([n1, n2, n1])
        assert twice.buckets == (n1, n2)
        assert twice.as_range() == once.as_range()


class TestEdgeCases:
    def test_zero_counts(self):
        bucket = P.closedopen(0, 10)
        migrator = P.HistogramMigrator([bucket], [bucket], {bucket: 0})
        verdict = migrator.migrate(bucket)
        assert verdict.status is P.MigrationStatus.EXACT
        assert verdict.as_range() == (0, 0)

    def test_singleton_new_bucket_matches_threshold(self):
        old = [P.closedopen(0, 10)]
        rest = P.closedopen(0, 5) | P.open(5, 10)
        new = [P.singleton(5), rest]
        migrator = P.HistogramMigrator(
            old, new, {old[0]: 6}, exact_counts={5: 2}
        )
        point, leftover = new
        assert migrator.migrate(point).as_range() == (2, 2)
        assert migrator.migrate(leftover).as_range() == (4, 4)

    def test_unbounded_buckets(self):
        old = [P.closedopen(-P.inf, 0), P.closedopen(0, P.inf)]
        new = [P.closedopen(-P.inf, 5), P.closedopen(5, P.inf)]
        counts = {old[0]: 3, old[1]: 7}
        migrator = P.HistogramMigrator(old, new, counts)
        n1, n2 = new
        assert migrator.migrate(n1).as_range() == (3, 10)
        assert migrator.group_range([n1, n2]).as_range() == (10, 10)


class TestFeasibleCombination:
    def test_feasible_allocation(self):
        migrator = make_migrator()
        n1, n2, n3 = migrator.new_buckets

        assert migrator.is_feasible({n1: 0, n2: 20, n3: 0})
        assert migrator.is_feasible({n1: 10, n2: 0, n3: 10})
        assert migrator.is_feasible({n1: 4, n2: 11, n3: 5})

    def test_individual_maxima_not_jointly_achievable(self):
        migrator = make_migrator()
        n1, n2, n3 = migrator.new_buckets

        # Each maximum is reachable on its own.
        assert migrator.is_feasible({n1: 10, n2: 0, n3: 10})
        assert migrator.is_feasible({n1: 0, n2: 20, n3: 0})
        assert migrator.migrate(n1).upper == 10
        assert migrator.migrate(n2).upper == 20

        # Their sum (30) exceeds the batch and the tight group bound (20).
        assert not migrator.is_feasible({n1: 10, n2: 20, n3: 0})
        verdict = migrator.feasible_combination({n1: 10, n2: 20, n3: 0})
        assert verdict.status is P.MigrationStatus.IMPOSSIBLE
        assert migrator.group_range([n1, n2]).upper == 20

    def test_routing_impossibility_with_matching_total(self):
        p1 = P.closedopen(0, 5)
        p2 = P.closedopen(5, 10)
        p3 = P.closedopen(10, 20)
        old = [p1, p2, p3]
        new = [P.closedopen(0, 10), P.closedopen(10, 20)]
        counts = {p1: 5, p2: 5, p3: 10}
        migrator = P.HistogramMigrator(old, new, counts)

        n1, n2 = new
        # The second new bucket aligns with p3, so its count is exact.
        assert migrator.migrate(n2).as_range() == (10, 10)
        verdict = migrator.feasible_combination({n1: 5, n2: 15})
        assert verdict.status is P.MigrationStatus.IMPOSSIBLE
        assert not migrator.is_feasible({n1: 5, n2: 15})

    def test_exact_counts_constrain_feasibility(self):
        migrator = make_migrator(exact_counts={10: 3})
        n1, n2, n3 = migrator.new_buckets

        # n2 must hold at least the 3 exact samples.
        verdict = migrator.feasible_combination({n1: 10, n2: 2, n3: 5})
        assert verdict.status is P.MigrationStatus.IMPOSSIBLE
        assert migrator.is_feasible({n1: 7, n2: 8, n3: 5})

    def test_requires_complete_vector(self):
        migrator = make_migrator()
        n1 = migrator.new_buckets[0]
        with pytest.raises(ValueError):
            migrator.feasible_combination({n1: 5})


class TestValidation:
    def test_missing_old_count(self):
        with pytest.raises(ValueError):
            P.HistogramMigrator(
                [P.closedopen(0, 10)], [P.closedopen(0, 10)], {}
            )

    def test_negative_counts(self):
        bucket = P.closedopen(0, 10)
        with pytest.raises(ValueError):
            P.HistogramMigrator([bucket], [bucket], {bucket: -1})
        with pytest.raises(ValueError):
            P.HistogramMigrator(
                [bucket], [bucket], {bucket: 1}, exact_counts={5: -1}
            )

    def test_overlapping_buckets_rejected(self):
        with pytest.raises(ValueError):
            P.HistogramMigrator(
                [P.closedopen(0, 10), P.closedopen(5, 15)],
                [P.closedopen(0, 10)],
                {P.closedopen(0, 10): 1, P.closedopen(5, 15): 1},
            )

    def test_threshold_outside_old_layout(self):
        bucket = P.closedopen(0, 10)
        with pytest.raises(ValueError):
            P.HistogramMigrator(
                [bucket], [bucket], {bucket: 2}, exact_counts={42: 1}
            )

    def test_exact_counts_exceed_bucket_total(self):
        bucket = P.closedopen(0, 10)
        with pytest.raises(ValueError):
            P.HistogramMigrator(
                [bucket], [bucket], {bucket: 1}, exact_counts={5: 2}
            )

    def test_unknown_bucket_key(self):
        migrator = make_migrator()
        with pytest.raises(KeyError):
            migrator.migrate(P.closedopen(100, 200))
        with pytest.raises(KeyError):
            migrator.group_range(P.closedopen(100, 200))
