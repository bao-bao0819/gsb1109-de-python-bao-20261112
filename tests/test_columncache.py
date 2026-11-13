import unittest

from cachetools.columncache import (
    CachedGroup,
    ColumnCache,
    ColumnSource,
    ConflictingColumnsError,
    plan_column_read,
)


class Group:
    """Small helper building groups with compact positional data."""

    def __init__(
        self, group_id, revision, columns, checksum=0, unpack_cost=1.0, payload=None
    ):
        if isinstance(columns, dict):
            checksums = columns
        else:
            checksums = {column: checksum for column in columns}
        self.group = CachedGroup(
            group_id,
            revision,
            checksums,
            unpack_cost=unpack_cost,
            payload=payload,
        )


def group(*args, **kwargs):
    return Group(*args, **kwargs).group


class PlanColumnReadTest(unittest.TestCase):
    def test_full_hit_single_group(self):
        groups = [group("g1", "r1", ("a", "b"))]
        plan = plan_column_read(groups, ("a", "b"), "r1", remote_column_cost=1.0)
        self.assertEqual([g.group_id for g in plan.groups], ["g1"])
        self.assertEqual(plan.remote_columns, ())
        self.assertTrue(plan.hit)
        self.assertEqual(plan.unpack_cost, 1.0)
        self.assertEqual(plan.remote_cost, 0.0)
        self.assertEqual(plan.total_cost, 1.0)

    def test_combines_groups_covering_request(self):
        groups = [
            group("g1", "r1", ("a",), unpack_cost=1.0),
            group("g2", "r1", ("b",), unpack_cost=1.0),
        ]
        plan = plan_column_read(groups, ("a", "b"), "r1")
        self.assertEqual([g.group_id for g in plan.groups], ["g1", "g2"])
        self.assertEqual(plan.remote_columns, ())
        self.assertEqual(plan.total_cost, 2.0)

    def test_supplemental_read_for_missing_columns(self):
        groups = [group("g1", "r1", ("a",), unpack_cost=1.0)]
        plan = plan_column_read(groups, ("a", "b", "c"), "r1", remote_column_cost=1.0)
        self.assertEqual([g.group_id for g in plan.groups], ["g1"])
        self.assertEqual(plan.remote_columns, ("b", "c"))
        self.assertFalse(plan.hit)
        self.assertEqual(plan.total_cost, 3.0)

    def test_skips_expensive_group_when_supplement_is_cheaper(self):
        groups = [group("g1", "r1", ("a",), unpack_cost=10.0)]
        plan = plan_column_read(groups, ("a", "b"), "r1", remote_column_cost=1.0)
        # Unpacking costs 10 + 1 for b; skipping costs 2 remote reads.
        self.assertEqual(plan.groups, ())
        self.assertEqual(plan.remote_columns, ("a", "b"))
        self.assertEqual(plan.total_cost, 2.0)

    def test_empty_subset_is_considered(self):
        groups = [group("g1", "r1", ("a",), unpack_cost=5.0)]
        plan = plan_column_read(groups, ("a",), "r1", remote_column_cost=1.0)
        self.assertEqual(plan.groups, ())
        self.assertEqual(plan.remote_columns, ("a",))

    def test_full_remote_when_cheaper(self):
        groups = [
            group("g1", "r1", ("a",), unpack_cost=3.0),
            group("g2", "r1", ("b",), unpack_cost=3.0),
        ]
        plan = plan_column_read(
            groups,
            ("a", "b"),
            "r1",
            remote_column_cost=2.0,
            full_remote_cost=3.0,
        )
        self.assertTrue(plan.full_remote)
        self.assertEqual(plan.groups, ())
        self.assertEqual(plan.remote_columns, ("a", "b"))
        self.assertEqual(plan.total_cost, 3.0)

    def test_full_remote_callable(self):
        groups = [group("g1", "r1", ("a",), unpack_cost=8.0)]
        plan = plan_column_read(
            groups,
            ("a", "b"),
            "r1",
            remote_column_cost=4.0,
            full_remote_cost=lambda requested: 5.0,
        )
        self.assertTrue(plan.full_remote)
        self.assertEqual(plan.remote_cost, 5.0)

    def test_cache_combination_beats_full_remote(self):
        groups = [
            group("g1", "r1", ("a",), unpack_cost=1.0),
            group("g2", "r1", ("b",), unpack_cost=1.0),
        ]
        plan = plan_column_read(
            groups,
            ("a", "b"),
            "r1",
            remote_column_cost=10.0,
            full_remote_cost=5.0,
        )
        self.assertFalse(plan.full_remote)
        self.assertEqual([g.group_id for g in plan.groups], ["g1", "g2"])
        self.assertEqual(plan.total_cost, 2.0)

    def test_ignores_other_revisions(self):
        groups = [
            group("g1", "r0", ("a", "b")),
            group("g2", "r1", ("a",)),
        ]
        plan = plan_column_read(groups, ("a", "b"), "r1", remote_column_cost=1.0)
        self.assertEqual([g.group_id for g in plan.groups], ["g2"])
        self.assertEqual(plan.remote_columns, ("b",))

    def test_conflicting_checksums_raise(self):
        groups = [
            group("g1", "r1", {"a": 1}),
            group("g2", "r1", {"a": 2}),
        ]
        with self.assertRaises(ConflictingColumnsError) as ctx:
            plan_column_read(groups, ("a",), "r1")
        self.assertEqual(ctx.exception.column, "a")
        self.assertEqual(ctx.exception.checksums, (1, 2))
        self.assertEqual(ctx.exception.group_ids, ("g1", "g2"))

    def test_conflict_only_raises_for_requested_columns(self):
        groups = [
            group("g1", "r1", {"a": 1, "x": 1}),
            group("g2", "r1", {"a": 1, "x": 2}),
        ]
        plan = plan_column_read(groups, ("a",), "r1")
        self.assertEqual([g.group_id for g in plan.groups], ["g1"])

    def test_conflict_across_other_revision_is_ignored(self):
        groups = [
            group("g1", "r0", {"a": 1}),
            group("g2", "r1", {"a": 2}),
        ]
        plan = plan_column_read(groups, ("a",), "r1", remote_column_cost=1.0)
        self.assertEqual([g.group_id for g in plan.groups], ["g2"])

    def test_equal_checksums_on_overlap_combine(self):
        groups = [
            group("g1", "r1", {"a": 7, "b": 1}),
            group("g2", "r1", {"a": 7, "c": 2}),
        ]
        plan = plan_column_read(groups, ("a", "b", "c"), "r1", remote_column_cost=10.0)
        self.assertEqual([g.group_id for g in plan.groups], ["g1", "g2"])
        self.assertEqual(plan.remote_columns, ())

    def test_overlapping_group_charged_once(self):
        cheap = group("g1", "r1", {"a": 1, "b": 1}, unpack_cost=1.0)
        extra = group("g2", "r1", {"c": 1}, unpack_cost=1.0)
        plan = plan_column_read(
            [cheap, extra],
            ("a", "b", "c"),
            "r1",
            remote_column_cost=10.0,
        )
        self.assertEqual([g.group_id for g in plan.groups], ["g1", "g2"])
        self.assertEqual(plan.unpack_cost, 2.0)

    def test_duplicate_request_columns_preserved_in_remote(self):
        groups = [group("g1", "r1", ("a",))]
        plan = plan_column_read(
            groups,
            ("a", "b", "b", "a"),
            "r1",
            remote_column_cost=1.0,
        )
        self.assertEqual(plan.remote_columns, ("b",))

    def test_remote_column_cost_callable(self):
        groups = [group("g1", "r1", ("a",), unpack_cost=1.0)]

        def cost(missing):
            return 7.0 * len(missing)

        plan = plan_column_read(groups, ("a", "b"), "r1", remote_column_cost=cost)
        self.assertEqual([g.group_id for g in plan.groups], ["g1"])
        self.assertEqual(plan.remote_columns, ("b",))
        self.assertEqual(plan.remote_cost, 7.0)
        self.assertEqual(plan.total_cost, 8.0)

    def test_prefers_fewer_groups_on_tied_cost(self):
        groups = [
            group("g1", "r1", {"a": 1}, unpack_cost=1.0),
            group("g2", "r1", {"b": 1}, unpack_cost=1.0),
            group("g3", "r1", {"a": 1, "b": 1}, unpack_cost=2.0),
        ]
        plan = plan_column_read(groups, ("a", "b"), "r1", remote_column_cost=2.0)
        # g1 + g2 and g3 both cost 2; the one-group plan wins.
        self.assertEqual([g.group_id for g in plan.groups], ["g3"])

    def test_no_groups(self):
        plan = plan_column_read((), ("a", "b"), "r1", remote_column_cost=1.0)
        self.assertEqual(plan.groups, ())
        self.assertEqual(plan.remote_columns, ("a", "b"))
        self.assertEqual(plan.total_cost, 2.0)
        self.assertFalse(plan.full_remote)

    def test_group_without_requested_columns_ignored(self):
        groups = [group("g1", "r1", ("z",))]
        plan = plan_column_read(groups, ("a",), "r1", remote_column_cost=1.0)
        self.assertEqual(plan.groups, ())
        self.assertEqual(plan.remote_columns, ("a",))

    def test_plan_revision(self):
        plan = plan_column_read((), ("a",), "r7", remote_column_cost=1.0)
        self.assertEqual(plan.revision, "r7")


class RecordingSource(ColumnSource):
    def __init__(
        self, values, checksums=None, supplemental=1.0, full_remote=1.0, unpack_map=None
    ):
        self.values = values
        self._checksums = checksums
        self.supplemental = supplemental
        self.full = full_remote
        self.unpack_map = unpack_map or {}
        self.reads = []
        self.unpacked = []

    def read(self, partition, columns, revision):
        self.reads.append((partition, tuple(columns), revision))
        return {column: self.values[column] for column in columns}

    def unpack(self, grp):
        self.unpacked.append(grp.group_id)
        if grp.group_id in self.unpack_map:
            return self.unpack_map[grp.group_id]
        return super().unpack(grp)

    def supplemental_cost(self, partition, columns):
        return self.supplemental * len(columns)

    def full_remote_cost(self, partition, columns, revision):
        return self.full

    def checksums(self, partition, columns, revision, values):
        if self._checksums is None:
            return None
        return {column: self._checksums[column] for column in columns}


class ColumnCacheTest(unittest.TestCase):
    def _cache(self, source, **kwargs):
        cache = ColumnCache(source, **kwargs)
        return cache

    def test_full_hit_returns_values_without_source_read(self):
        source = RecordingSource({"a": 1, "b": 2})
        cache = self._cache(source)
        cache.add(
            "p",
            "r1",
            {"a": 10, "b": 20},
            payload={"a": 1, "b": 2},
        )
        values, plan = cache.read_with_plan("p", ("a", "b"), "r1")
        self.assertEqual(values, [1, 2])
        self.assertTrue(plan.hit)
        self.assertEqual(source.reads, [])

    def test_supplemental_read_fills_missing_columns(self):
        source = RecordingSource(
            {"a": 1, "b": 2, "c": 3},
            supplemental=1.0,
            full_remote=10.0,
        )
        cache = self._cache(source)
        cache.add("p", "r1", {"a": 10}, payload={"a": 1})
        values, plan = cache.read_with_plan("p", ("a", "b", "c"), "r1")
        self.assertEqual(values, [1, 2, 3])
        self.assertEqual(plan.remote_columns, ("b", "c"))
        self.assertEqual(source.reads, [("p", ("b", "c"), "r1")])

    def test_full_remote_when_cheapest(self):
        source = RecordingSource({"a": 1, "b": 2}, supplemental=1.0, full_remote=1.0)
        cache = self._cache(source)
        cache.add("p", "r1", {"a": 10}, unpack_cost=5.0, payload={"a": 1})
        values, plan = cache.read_with_plan("p", ("a", "b"), "r1")
        self.assertEqual(values, [1, 2])
        self.assertTrue(plan.full_remote)
        self.assertEqual(source.reads, [("p", ("a", "b"), "r1")])

    def test_conflict_falls_back_to_full_remote(self):
        source = RecordingSource({"a": 1, "b": 2}, supplemental=1.0, full_remote=2.0)
        cache = self._cache(source)
        cache.add("p", "r1", {"a": 1, "b": 1}, payload={"a": 1, "b": 2})
        cache.add("p", "r1", {"a": 2}, payload={"a": 99})
        values, plan = cache.read_with_plan("p", ("a", "b"), "r1")
        self.assertEqual(values, [1, 2])
        self.assertTrue(plan.full_remote)
        self.assertEqual(source.reads, [("p", ("a", "b"), "r1")])
        self.assertEqual(source.unpacked, [])

    def test_result_order_and_duplicates_match_request(self):
        source = RecordingSource({"a": 1, "b": 2, "c": 3})
        cache = self._cache(source)
        cache.add("p", "r1", {"b": 20}, payload={"b": 2})
        values = cache.read("p", ("c", "b", "a", "b", "c"), "r1")
        self.assertEqual(values, [3, 2, 1, 2, 3])

    def test_other_revision_triggers_remote(self):
        source = RecordingSource({"a": 1})
        cache = self._cache(source)
        cache.add("p", "r0", {"a": 10}, payload={"a": 0})
        values, plan = cache.read_with_plan("p", ("a",), "r1")
        self.assertEqual(values, [1])
        self.assertEqual(source.reads, [("p", ("a",), "r1")])
        self.assertEqual(plan.remote_columns, ("a",))

    def test_partitions_are_independent(self):
        source = RecordingSource({"a": 1})
        cache = self._cache(source)
        cache.add("p1", "r1", {"a": 10}, payload={"a": 1})
        cache.read("p2", ("a",), "r1")
        self.assertEqual(source.reads, [("p2", ("a",), "r1")])
        self.assertEqual([g.group_id for g in cache.groups("p1")], [0])
        self.assertEqual(cache.groups("p2"), ())

    def test_supplemental_result_is_cached_when_checksums_known(self):
        source = RecordingSource(
            {"a": 1, "b": 2},
            checksums={"a": 10, "b": 20},
            supplemental=1.0,
            full_remote=10.0,
        )
        cache = self._cache(source)
        cache.add("p", "r1", {"a": 10}, payload={"a": 1})
        cache.read("p", ("a", "b"), "r1")
        groups = cache.groups("p")
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[1].checksums, {"b": 20})
        # Second read of b is now fully served from the cache.
        source.reads.clear()
        values = cache.read("p", ("b",), "r1")
        self.assertEqual(values, [2])
        self.assertEqual(source.reads, [])

    def test_remote_result_not_cached_without_checksums(self):
        source = RecordingSource({"a": 1})
        cache = self._cache(source)
        cache.read("p", ("a",), "r1")
        self.assertEqual(cache.groups("p"), ())

    def test_stored_group_payloads_are_unpacked(self):
        source = RecordingSource({"a": 1})
        cache = self._cache(source)
        cache.add("p", "r1", {"a": 10}, payload={"a": 42})
        self.assertEqual(cache.read("p", ("a",), "r1"), [42])
        self.assertEqual(source.unpacked, [0])

    def test_missing_source_column_raises_keyerror(self):
        class PartialSource(ColumnSource):
            def read(self, partition, columns, revision):
                return {}

        cache = ColumnCache(PartialSource())
        with self.assertRaises(KeyError):
            cache.read("p", ("a",), "r1")

    def test_default_unpack_cost(self):
        source = RecordingSource({"a": 1})
        cache = ColumnCache(source, default_unpack_cost=4.0)
        group = cache.add("p", "r1", {"a": 10}, payload={"a": 1})
        self.assertEqual(group.unpack_cost, 4.0)

    def test_explicit_group_id(self):
        source = RecordingSource({"a": 1})
        cache = ColumnCache(source)
        grp = cache.add("p", "r1", {"a": 10}, payload={"a": 1}, group_id="x")
        self.assertEqual(grp.group_id, "x")


class RepresentationAndDefaultsTest(unittest.TestCase):
    def test_group_repr(self):
        grp = group("g1", "r1", ("a", "b"), unpack_cost=2.5)
        text = repr(grp)
        self.assertIn("g1", text)
        self.assertIn("r1", text)
        self.assertIn("2.5", text)

    def test_group_equality(self):
        first = group("g1", "r1", {"a": 1}, payload={"a": 1})
        second = group("g1", "r1", {"a": 1}, payload={"a": 1})
        self.assertEqual(first, second)
        self.assertNotEqual(first, group("g2", "r1", {"a": 1}))
        self.assertEqual(first.__eq__("not a group"), NotImplemented)

    def test_plan_repr_and_equality(self):
        groups = [group("g1", "r1", ("a",))]
        plan = plan_column_read(groups, ("a", "b"), "r1")
        text = repr(plan)
        self.assertIn("g1", text)
        self.assertIn("b", text)
        same = plan_column_read([group("g1", "r1", ("a",))], ("a", "b"), "r1")
        self.assertEqual(plan, same)
        self.assertEqual(plan.__eq__("not a plan"), NotImplemented)

    def test_base_source_defaults(self):
        source = ColumnSource()
        with self.assertRaises(NotImplementedError):
            source.read("p", ("a",), "r1")
        with self.assertRaises(ValueError):
            source.unpack(group("g1", "r1", ("a",)))
        self.assertEqual(source.supplemental_cost("p", ("a", "b")), 2.0)
        self.assertEqual(source.full_remote_cost("p", ("a", "b"), "r1"), 2.0)
        self.assertIsNone(source.checksums("p", ("a",), "r1", {"a": 1}))


if __name__ == "__main__":
    unittest.main()
