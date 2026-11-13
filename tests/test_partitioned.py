import unittest

from cachetools import PartitionedCache

from . import CacheTestMixin


class PartitionedCacheTest(unittest.TestCase, CacheTestMixin):
    Cache = PartitionedCache

    def test_partitioned_defaults(self):
        cache = PartitionedCache(maxsize=10)

        self.assertEqual({}, cache.reserved)
        self.assertEqual({}, cache.usage)
        self.assertEqual((), cache.last_evicted)
        self.assertIsNone(cache.getdataset("key"))

        cache[1] = 1
        self.assertIsNone(cache.owner(1))
        self.assertEqual({None: 1}, cache.usage)

    def test_partitioned_reserved_negative(self):
        with self.assertRaises(ValueError):
            PartitionedCache(maxsize=10, reserved={"a": -1})

    def test_partitioned_reserved_property_copy(self):
        cache = PartitionedCache(maxsize=10, reserved={"a": 2})

        self.assertEqual({"a": 2}, cache.reserved)
        cache.reserved["a"] = 0  # mutating the copy must not stick
        self.assertEqual({"a": 2}, cache.reserved)

    def test_partitioned_owner(self):
        cache = PartitionedCache[str, int](
            maxsize=4, reserved={"a": 2}, getdataset=lambda key: key[0]
        )

        cache["a1"] = 1
        cache["b1"] = 2
        self.assertEqual("a", cache.owner("a1"))
        self.assertEqual("b", cache.owner("b1"))
        with self.assertRaises(KeyError):
            cache.owner("c1")

    def test_partitioned_getdataset_subclass(self):
        class Cache(PartitionedCache[str, int]):
            @staticmethod
            def getdataset(key):
                return key[0]

        cache = Cache(maxsize=4, reserved={"a": 2})
        cache["a1"] = 1
        self.assertEqual("a", cache.owner("a1"))

    def test_partitioned_usage(self):
        cache = PartitionedCache[str, int](
            maxsize=10,
            reserved={"a": 4, "b": 4},
            getdataset=lambda key: key[0],
            getsizeof=lambda value: value,
        )

        cache["a1"] = 2
        cache["b1"] = 3
        self.assertEqual({"a": 2, "b": 3}, cache.usage)

        # replacing an item adjusts its dataset's usage by the difference
        cache["a1"] = 3
        self.assertEqual({"a": 3, "b": 3}, cache.usage)

        del cache["b1"]
        self.assertEqual({"a": 3, "b": 0}, cache.usage)

    def test_partitioned_borrow_idle_reserved(self):
        cache = PartitionedCache[str, int](
            maxsize=10,
            reserved={"a": 5, "b": 5},
            getdataset=lambda key: key[0],
            getsizeof=lambda value: value,
        )

        # dataset b may borrow a's idle reserved size
        cache["b1"] = 4
        cache["b2"] = 4
        self.assertEqual(8, cache.currsize)
        self.assertEqual({"a": 0, "b": 8}, cache.usage)

        # when a claims its share, b's items are evicted first
        cache["a1"] = 4
        self.assertEqual((("b1", "b", 4),), cache.last_evicted)
        self.assertIn("b2", cache)
        self.assertIn("a1", cache)
        self.assertEqual({"a": 4, "b": 4}, cache.usage)

    def test_partitioned_evict_over_reserved_first(self):
        cache = PartitionedCache[str, int](
            maxsize=6,
            reserved={"a": 2, "b": 2},
            getdataset=lambda key: key[0],
            getsizeof=lambda value: value,
        )

        cache["a1"] = 2
        cache["b1"] = 2
        cache["b2"] = 2  # b exceeds its reserved size
        self.assertEqual(6, cache.currsize)

        cache["a2"] = 2
        self.assertEqual((("b1", "b", 2),), cache.last_evicted)
        self.assertNotIn("b1", cache)
        self.assertIn("a1", cache)
        self.assertIn("b2", cache)
        self.assertIn("a2", cache)

    def test_partitioned_evict_lru_among_over_reserved(self):
        cache = PartitionedCache[str, int](
            maxsize=6,
            reserved={"a": 1, "b": 1},
            getdataset=lambda key: key[0],
            getsizeof=lambda value: value,
        )

        cache["a1"] = 2
        cache["b1"] = 2
        cache["b2"] = 2
        cache["a1"]  # touch a1, making b1 the least recently used item
        cache["c1"] = 2

        # all datasets exceed their reserved size, so the least recently
        # used item overall is evicted
        self.assertEqual((("b1", "b", 2),), cache.last_evicted)
        self.assertNotIn("b1", cache)
        self.assertIn("a1", cache)
        self.assertIn("b2", cache)

    def test_partitioned_evict_spans_datasets(self):
        cache = PartitionedCache[str, int](
            maxsize=6,
            reserved={"a": 1, "b": 1},
            getdataset=lambda key: key[0],
            getsizeof=lambda value: value,
        )

        cache["a1"] = 2
        cache["b1"] = 2
        cache["c1"] = 2
        cache["d1"] = 3

        # items are indivisible, so more bytes than needed are freed
        self.assertEqual((("a1", "a", 2), ("b1", "b", 2)), cache.last_evicted)
        self.assertIn("c1", cache)
        self.assertIn("d1", cache)
        self.assertEqual(5, cache.currsize)

    def test_partitioned_evict_fallback_lru(self):
        cache = PartitionedCache[str, int](
            maxsize=4,
            reserved={"a": 3, "b": 3},
            getdataset=lambda key: key[0],
            getsizeof=lambda value: value,
        )

        cache["a1"] = 2
        cache["b1"] = 2
        cache["a1"]  # touch a1, making b1 the least recently used item
        cache["c1"] = 1

        # no dataset exceeds its reserved size, so the least recently
        # used item overall is evicted
        self.assertEqual((("b1", "b", 2),), cache.last_evicted)
        self.assertNotIn("b1", cache)
        self.assertIn("a1", cache)
        self.assertIn("c1", cache)

    def test_partitioned_evict_unreserved_first(self):
        cache = PartitionedCache[str, int](
            maxsize=4, reserved={"a": 4}, getdataset=lambda key: key[0]
        )

        cache["b1"] = 1
        cache["a1"] = 1
        cache["a2"] = 1
        cache["a3"] = 1
        cache["b1"]  # touch b1, making a1 the least recently used item
        cache["a4"] = 1

        # items of a dataset without reserved size are evicted first
        self.assertEqual((("b1", "b", 1),), cache.last_evicted)
        self.assertNotIn("b1", cache)
        self.assertIn("a1", cache)

    def test_partitioned_value_too_large_keeps_cache(self):
        cache = PartitionedCache[str, int](
            maxsize=4,
            reserved={"a": 2},
            getdataset=lambda key: key[0],
            getsizeof=lambda value: value,
        )

        cache["a1"] = 2
        cache["b1"] = 2
        with self.assertRaises(ValueError):
            cache["c1"] = 5

        # a rejected insertion must not modify the cache
        self.assertEqual({"a1": 2, "b1": 2}, dict(cache))
        self.assertEqual(4, cache.currsize)
        self.assertEqual({"a": 2, "b": 2}, cache.usage)
        self.assertEqual((), cache.last_evicted)

    def test_partitioned_replace_evicts_self(self):
        cache = PartitionedCache[int, int](maxsize=10, getsizeof=lambda value: value)

        cache[1] = 5
        cache[2] = 5
        cache[1] = 9

        self.assertEqual({1: 9}, dict(cache))
        self.assertEqual(9, cache.currsize)
        self.assertEqual({None: 9}, cache.usage)
        self.assertEqual(((1, None, 5), (2, None, 5)), cache.last_evicted)

    def test_partitioned_last_evicted_reset(self):
        cache = PartitionedCache[int, int](maxsize=2, getsizeof=lambda value: value)

        cache[1] = 2
        cache[2] = 2
        self.assertEqual(((1, None, 2),), cache.last_evicted)

        cache[2] = 1  # replacing with a smaller item evicts nothing
        self.assertEqual((), cache.last_evicted)

    def test_partitioned_popitem(self):
        cache = PartitionedCache[str, int](
            maxsize=6,
            reserved={"a": 2, "b": 2},
            getdataset=lambda key: key[0],
            getsizeof=lambda value: value,
        )

        cache["a1"] = 2
        cache["b1"] = 2
        cache["b2"] = 2

        # b exceeds its reserved size, so its items go first
        self.assertEqual(("b1", 2), cache.popitem())
        self.assertEqual(("a1", 2), cache.popitem())
        self.assertEqual(("b2", 2), cache.popitem())
        with self.assertRaises(KeyError):
            cache.popitem()

    def test_partitioned_clear(self):
        cache = PartitionedCache[str, int](
            maxsize=4, reserved={"a": 2}, getdataset=lambda key: key[0]
        )

        cache["a1"] = 1
        cache["b1"] = 1
        cache.clear()

        self.assertEqual(0, len(cache))
        self.assertEqual({"a": 0}, cache.usage)
        self.assertEqual((), cache.last_evicted)

        # verify cache can be reused after clear
        cache["a1"] = 1
        self.assertEqual({"a": 1}, cache.usage)
        self.assertEqual("a", cache.owner("a1"))

    def test_partitioned_unreserved_dataset_usage(self):
        cache = PartitionedCache[str, int](
            maxsize=4, reserved={"a": 4}, getdataset=lambda key: key[0]
        )

        cache["b1"] = 1
        self.assertEqual({"a": 0, "b": 1}, cache.usage)

    def test_partitioned_pickle(self):
        import pickle

        source = PartitionedCache(maxsize=4, reserved={"a": 2})
        source[1] = 1

        cache = pickle.loads(pickle.dumps(source))
        self.assertEqual(source, cache)
        self.assertEqual({"a": 2}, cache.reserved)
        self.assertEqual({"a": 0, None: 1}, cache.usage)

        cache[2] = 2
        self.assertEqual({1: 1, 2: 2}, dict(cache))
