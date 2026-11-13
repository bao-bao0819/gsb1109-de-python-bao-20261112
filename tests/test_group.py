import unittest

from cachetools import GroupCache, GroupMiss, GroupSnapshot

from . import CacheTestMixin


class GroupCacheTest(unittest.TestCase, CacheTestMixin):
    Cache = GroupCache

    def test_publish_snapshot(self):
        cache = GroupCache[int, int](maxsize=10)
        cache.publish("report", {"a": 1, "b": 2, "c": 3}, {1: 10, 2: 20, 3: 30})

        with cache.snapshot("report") as snap:
            self.assertIsInstance(snap, GroupSnapshot)
            self.assertEqual("report", snap.group)
            self.assertEqual({"a": 1, "b": 2, "c": 3}, snap.manifest)
            self.assertFalse(snap.closed)
            self.assertEqual(3, len(snap))
            self.assertEqual({"a", "b", "c"}, set(snap))
            self.assertEqual(10, snap["a"])
            self.assertEqual(20, snap["b"])
            self.assertEqual(30, snap["c"])
            self.assertEqual(60, sum(snap.values()))
            self.assertEqual({"a": 10, "b": 20, "c": 30}, dict(snap))
            self.assertEqual({"a": 10, "b": 20, "c": 30}, snap)
            with self.assertRaises(KeyError):
                snap["x"]
        self.assertTrue(snap.closed)

    def test_publish_reuse_unchanged(self):
        cache = GroupCache[int, int](maxsize=10)
        cache.publish("report", {"a": 1, "b": 2, "c": 3}, {1: 10, 2: 20, 3: 30})

        # recompute shard b only; a and c keep their old tokens
        cache.publish("report", {"a": 1, "b": 4, "c": 3}, {4: 25})

        self.assertEqual({"a": 1, "b": 4, "c": 3}, cache.manifest("report"))
        with cache.snapshot("report") as snap:
            self.assertEqual({"a": 10, "b": 25, "c": 30}, dict(snap))

    def test_publish_reclaims_superseded(self):
        cache = GroupCache[int, int](maxsize=3)
        cache.publish("report", {"a": 1, "b": 2, "c": 3}, {1: 10, 2: 20, 3: 30})

        # the cache is full; the superseded value of shard a is
        # reclaimed to make room for the new version
        cache.publish("report", {"a": 4, "b": 2, "c": 3}, {4: 40})

        self.assertEqual(3, len(cache))
        self.assertEqual({"a": 4, "b": 2, "c": 3}, cache.manifest("report"))
        with cache.snapshot("report") as snap:
            self.assertEqual({"a": 40, "b": 20, "c": 30}, dict(snap))

    def test_publish_shared_token(self):
        cache = GroupCache[int, int](maxsize=10)
        # all three shards share a single token and cached value
        cache.publish("report", {"a": 1, "b": 1, "c": 1}, {1: 10})
        cache.publish("report", {"a": 1, "b": 2, "c": 1}, {2: 25})

        with cache.snapshot("report") as snap:
            self.assertEqual({"a": 10, "b": 25, "c": 10}, dict(snap))

    def test_publish_validation(self):
        cache = GroupCache[int, int](maxsize=10)

        with self.assertRaises(TypeError):
            cache.publish("report", [("a", 1)], {1: 10})
        with self.assertRaises(TypeError):
            cache.publish("report", {"a": 1}, [(1, 10)])
        with self.assertRaises(ValueError):
            cache.publish("report", {"a": 1, "b": 2}, {1: 10})

        # nothing must be cached after failed publishes
        self.assertEqual(0, len(cache))
        with self.assertRaises(KeyError):
            cache.manifest("report")

    def test_publish_does_not_fit(self):
        cache = GroupCache[int, int](maxsize=2)
        with self.assertRaises(ValueError):
            cache.publish("report", {"a": 1, "b": 2, "c": 3}, {1: 10, 2: 20, 3: 30})

        # the manifest must not be published
        with self.assertRaises(KeyError):
            cache.manifest("report")
        with self.assertRaises(KeyError):
            cache.snapshot("report")

    def test_manifest(self):
        cache = GroupCache[int, int](maxsize=10)
        with self.assertRaises(KeyError):
            cache.manifest("report")

        cache.publish("report", {"a": 1}, {1: 10})
        manifest = cache.manifest("report")
        self.assertEqual({"a": 1}, manifest)

        # the returned manifest is a copy
        manifest["b"] = 2
        self.assertEqual({"a": 1}, cache.manifest("report"))

    def test_snapshot_unknown_group(self):
        cache = GroupCache[int, int](maxsize=10)
        with self.assertRaises(KeyError):
            cache.snapshot("report")

    def test_snapshot_survives_publish(self):
        cache = GroupCache[int, int](maxsize=10)
        cache.publish("report", {"a": 1, "b": 2}, {1: 10, 2: 20})

        old = cache.snapshot("report")
        cache.publish("report", {"a": 3, "b": 2}, {3: 99})

        # the old reader still sees its own published set of values
        self.assertEqual({"a": 10, "b": 20}, dict(old))
        old.close()

        with cache.snapshot("report") as new:
            self.assertEqual({"a": 99, "b": 20}, dict(new))

    def test_snapshot_missing_shards(self):
        cache = GroupCache[int, int](maxsize=4)
        cache.publish("report", {"a": 1, "b": 2, "c": 3}, {1: 10, 2: 20, 3: 30})

        # evict one shard value by publishing another group
        cache.publish("other", {"x": 9, "y": 8}, {9: 90, 8: 80})

        with self.assertRaises(GroupMiss) as cm:
            cache.snapshot("report")
        self.assertIsInstance(cm.exception, KeyError)
        self.assertEqual("report", cm.exception.group)
        self.assertEqual(1, len(cm.exception.missing))
        self.assertLess(cm.exception.missing, {"a", "b", "c", "x"})
        self.assertIn("missing", str(cm.exception))

        # the failed snapshot must not leak pins
        while cache:
            cache.popitem()
        self.assertEqual(0, len(cache))

    def test_snapshot_pins_values(self):
        cache = GroupCache[int, int](maxsize=3)
        cache.publish("report", {"a": 1, "b": 2, "c": 3}, {1: 10, 2: 20, 3: 30})

        snap = cache.snapshot("report")
        # the cache is full and all values are pinned by the snapshot
        with self.assertRaises(KeyError):
            cache.publish("report", {"a": 4, "b": 2, "c": 3}, {4: 40})
        with self.assertRaises(KeyError):
            cache.popitem()

        # the open snapshot is unaffected, and no new manifest was published
        self.assertEqual({"a": 10, "b": 20, "c": 30}, dict(snap))
        self.assertEqual({"a": 1, "b": 2, "c": 3}, cache.manifest("report"))

        snap.close()
        # after close, publishing the new version succeeds
        cache.publish("report", {"a": 4, "b": 2, "c": 3}, {4: 40})
        self.assertEqual({"a": 4, "b": 2, "c": 3}, cache.manifest("report"))
        with cache.snapshot("report") as new:
            self.assertEqual({"a": 40, "b": 20, "c": 30}, dict(new))

    def test_snapshot_concurrent_readers(self):
        cache = GroupCache[int, int](maxsize=3)
        cache.publish("report", {"a": 1, "b": 2, "c": 3}, {1: 10, 2: 20, 3: 30})

        first = cache.snapshot("report")
        second = cache.snapshot("report")
        first.close()

        # values are still pinned by the second snapshot
        with self.assertRaises(KeyError):
            cache.popitem()
        self.assertEqual({"a": 10, "b": 20, "c": 30}, dict(second))

        second.close()
        cache.popitem()  # must not raise once all snapshots are closed
        self.assertEqual(2, len(cache))

    def test_snapshot_close_idempotent(self):
        cache = GroupCache[int, int](maxsize=10)
        cache.publish("report", {"a": 1}, {1: 10})

        snap = cache.snapshot("report")
        snap.close()
        snap.close()
        self.assertTrue(snap.closed)
        # values remain readable after close
        self.assertEqual(10, snap["a"])

    def test_snapshot_repr(self):
        cache = GroupCache[int, int](maxsize=10)
        cache.publish("report", {"a": 1}, {1: 10})

        with cache.snapshot("report") as snap:
            self.assertEqual(
                "GroupSnapshot(group='report', manifest={'a': 1})", repr(snap)
            )

    def test_clear_manifests(self):
        cache = GroupCache[int, int](maxsize=10)
        cache.publish("report", {"a": 1}, {1: 10})
        cache.clear()

        with self.assertRaises(KeyError):
            cache.manifest("report")
        with self.assertRaises(KeyError):
            cache.snapshot("report")

        # verify groups can be republished after clear
        cache.publish("report", {"a": 2}, {2: 20})
        with cache.snapshot("report") as snap:
            self.assertEqual({"a": 20}, dict(snap))

    def test_pickle_group(self):
        import pickle

        source = GroupCache[int, int](maxsize=10)
        source.publish("report", {"a": 1, "b": 2}, {1: 10, 2: 20})

        cache = pickle.loads(pickle.dumps(source))
        self.assertEqual({"a": 1, "b": 2}, cache.manifest("report"))
        with cache.snapshot("report") as snap:
            self.assertEqual({"a": 10, "b": 20}, dict(snap))
