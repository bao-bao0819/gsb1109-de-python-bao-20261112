import unittest
from typing import Any

from cachetools import LRUCache, TTLCache
from cachetools.source import Generation, Hit, SourceCache, SourceCacheRecord


class Timer:
    def __init__(self):
        self.time = 0.0

    def __call__(self):
        return self.time


def ttl_cache_factory(timer):
    class ManualTTLCache(TTLCache):
        def __init__(self, maxsize):
            super().__init__(maxsize, 10, timer=timer)

    return ManualTTLCache


class SizedLRUCache(LRUCache):
    def __init__(self, maxsize):
        def record_size(record):
            return len(record.value) if isinstance(record.value, str) else 1

        super().__init__(maxsize, record_size)


class SourceCacheTest(unittest.TestCase):
    def test_seal_returns_generation(self):
        cache = SourceCache()
        generation = cache.seal("2026-11-13")
        self.assertIsInstance(generation, Generation)
        self.assertEqual(("2026-11-13", generation.revision), generation)
        self.assertEqual(cache.sealed("2026-11-13"), generation)
        self.assertIsNone(cache.sealed("2026-11-12"))

    def test_seal_twice_starts_new_generation(self):
        cache = SourceCache()
        first = cache.seal("day-1")
        second = cache.seal("day-1")
        self.assertNotEqual(first, second)
        self.assertEqual(cache.sealed("day-1"), second)

    def test_miss_is_distinct_from_empty(self):
        cache = SourceCache()
        self.assertIsNone(cache.get_hit("device-1"))
        self.assertIsNone(cache.get_value("device-1"))
        self.assertEqual("default", cache.get_value("device-1", "default"))
        self.assertFalse(cache.is_empty("device-1"))
        self.assertEqual(cache.generations("device-1"), ())
        self.assertNotIn("device-1", cache)

    def test_empty_without_sealed_generation_rejected(self):
        cache = SourceCache()
        with self.assertRaises(ValueError):
            cache.store_empty("device-1")

    def test_empty_after_seal_is_cached_hit(self):
        cache = SourceCache()
        generation = cache.seal("2026-11-13")
        cache.store_empty("device-1", generation)

        self.assertEqual(len(cache), 1)
        self.assertIn("device-1", cache)
        self.assertTrue(cache.is_empty("device-1"))
        self.assertEqual(cache.generations("device-1"), (generation,))

        hit = cache.get_hit("device-1")
        self.assertEqual(hit, Hit("device-1", None, True, (generation,)))
        self.assertTrue(hit.empty)
        self.assertIsNone(hit.value)
        self.assertEqual(hit.generations, (generation,))
        self.assertIsNone(cache.get_value("device-1"))

    def test_store_empty_keyword_generations(self):
        cache = SourceCache()
        generation = cache.seal("day-1")
        cache.store_empty("device-1", generations=(generation,))
        self.assertTrue(cache.is_empty("device-1"))

    def test_positive_result_hit(self):
        cache = SourceCache()
        generation = cache.seal("day-1")
        cache.store_value("device-1", "payload", generation)

        hit = cache.get_hit("device-1")
        self.assertEqual(hit, Hit("device-1", "payload", False, (generation,)))
        self.assertFalse(hit.empty)
        self.assertEqual(hit.value, "payload")
        self.assertEqual(cache.get_value("device-1"), "payload")
        self.assertFalse(cache.is_empty("device-1"))

    def test_positive_result_without_generations(self):
        cache = SourceCache()
        cache.store_value("device-1", "payload")
        hit = cache.get_hit("device-1")
        self.assertFalse(hit.empty)
        self.assertEqual(hit.generations, ())

    def test_withdraw_invalidates_dependent_empty(self):
        cache = SourceCache()
        generation = cache.seal("2026-11-13")
        cache.store_empty("device-1", generation)
        cache.store_empty("device-2", generation)

        removed = cache.withdraw("2026-11-13")
        self.assertEqual(set(removed), {"device-1", "device-2"})
        self.assertIsNone(cache.get_hit("device-1"))
        self.assertIsNone(cache.get_hit("device-2"))
        self.assertIsNone(cache.sealed("2026-11-13"))
        self.assertEqual(len(cache), 0)

    def test_withdraw_keeps_other_sources_empty(self):
        cache = SourceCache()
        day1 = cache.seal("2026-11-12")
        day2 = cache.seal("2026-11-13")
        cache.store_empty("device-1", day1)
        cache.store_empty("device-2", day2)

        removed = cache.withdraw("2026-11-12")
        self.assertEqual(removed, ("device-1",))
        self.assertIsNone(cache.get_hit("device-1"))
        hit = cache.get_hit("device-2")
        self.assertTrue(hit.empty)
        self.assertEqual(hit.generations, (day2,))
        self.assertEqual(cache.sealed("2026-11-13"), day2)

    def test_withdraw_keeps_positive_results(self):
        cache = SourceCache()
        generation = cache.seal("day-1")
        cache.store_value("device-1", "payload", generation)
        cache.store_empty("device-2", generation)

        self.assertEqual(cache.withdraw("day-1"), ("device-2",))
        hit = cache.get_hit("device-1")
        self.assertFalse(hit.empty)
        self.assertEqual(hit.value, "payload")
        self.assertEqual(hit.generations, (generation,))

    def test_empty_depending_on_multiple_sources(self):
        cache = SourceCache()
        day1 = cache.seal("day-1")
        day2 = cache.seal("day-2")
        cache.store_empty("device-1", generations=(day1, day2))

        self.assertEqual(cache.withdraw("day-1"), ("device-1",))
        self.assertIsNone(cache.get_hit("device-1"))

    def test_withdraw_either_of_multiple_sources_invalidates(self):
        cache = SourceCache()
        day1 = cache.seal("day-1")
        day2 = cache.seal("day-2")
        cache.store_empty("device-1", generations=(day1, day2))
        cache.store_empty("device-2", day2)

        self.assertEqual(set(cache.withdraw("day-2")), {"device-1", "device-2"})
        self.assertIsNone(cache.get_hit("device-1"))
        self.assertIsNone(cache.get_hit("device-2"))
        self.assertEqual(cache.sealed("day-1"), day1)

    def test_withdraw_unknown_source(self):
        cache = SourceCache()
        self.assertEqual(cache.withdraw("never-sealed"), ())

    def test_withdraw_twice(self):
        cache = SourceCache()
        generation = cache.seal("day-1")
        cache.store_empty("device-1", generation)
        self.assertEqual(cache.withdraw("day-1"), ("device-1",))
        self.assertEqual(cache.withdraw("day-1"), ())

    def test_stale_generation_rejected(self):
        cache = SourceCache()
        first = cache.seal("day-1")
        cache.seal("day-1")
        with self.assertRaises(ValueError):
            cache.store_empty("device-1", first)

    def test_foreign_generation_rejected(self):
        cache = SourceCache()
        cache.seal("day-1")
        foreign = Generation("day-1", 999)
        with self.assertRaises(ValueError):
            cache.store_empty("device-1", foreign)

    def test_non_generation_rejected(self):
        cache = SourceCache()
        cache.seal("day-1")
        with self.assertRaises(TypeError):
            cache.store_empty("device-1", ("day-1", 1))

    def test_duplicate_markers_are_deduplicated(self):
        cache = SourceCache()
        generation = cache.seal("day-1")
        cache.store_empty("device-1", generations=(generation, generation))
        self.assertEqual(cache.generations("device-1"), (generation,))
        self.assertEqual(cache.withdraw("day-1"), ("device-1",))

    def test_overwrite_empty_with_positive(self):
        cache = SourceCache()
        generation = cache.seal("day-1")
        cache.store_empty("device-1", generation)
        cache.store_value("device-1", "payload")
        self.assertEqual(cache.withdraw("day-1"), ())
        self.assertEqual(cache.get_value("device-1"), "payload")

    def test_overwrite_multi_source_empty_cleans_all_dependencies(self):
        cache = SourceCache()
        day1 = cache.seal("day-1")
        day2 = cache.seal("day-2")
        cache.store_empty("device-1", generations=(day1, day2))
        cache.store_empty("device-2", day2)
        cache.store_value("device-1", "payload")
        self.assertEqual(cache.withdraw("day-1"), ())
        self.assertEqual(cache.withdraw("day-2"), ("device-2",))
        self.assertEqual(cache.get_value("device-1"), "payload")

    def test_overwrite_positive_with_empty(self):
        cache = SourceCache()
        generation = cache.seal("day-1")
        cache.store_value("device-1", "payload")
        cache.store_empty("device-1", generation)
        self.assertEqual(cache.withdraw("day-1"), ("device-1",))
        self.assertIsNone(cache.get_hit("device-1"))

    def test_clear_keeps_completion_markers(self):
        cache = SourceCache()
        generation = cache.seal("day-1")
        cache.store_empty("device-1", generation)
        cache.store_value("device-2", "payload")
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.sealed("day-1"), generation)
        cache.store_empty("device-3", generation)
        self.assertTrue(cache.is_empty("device-3"))

    def test_mapping_interface(self):
        cache = SourceCache()
        generation = cache.seal("day-1")
        cache.store_empty("device-1", generation)
        cache.store_value("device-2", "payload")
        self.assertEqual(set(cache), {"device-1", "device-2"})
        self.assertEqual(set(cache.keys()), {"device-1", "device-2"})
        self.assertEqual(cache.maxsize, float("inf"))

    def test_eviction_cleans_up_dependency_tracking(self):
        cache = SourceCache(maxsize=2, cache_factory=LRUCache)
        day1 = cache.seal("day-1")
        day2 = cache.seal("day-2")
        cache.store_empty("device-1", day1)
        cache.store_empty("device-2", day2)
        cache.store_value("device-3", "payload")  # evicts device-1
        self.assertNotIn("device-1", cache)
        self.assertEqual(cache.withdraw("day-1"), ())
        self.assertEqual(cache.withdraw("day-2"), ("device-2",))

    def test_evicted_key_can_be_restored_as_positive(self):
        cache = SourceCache(maxsize=1, cache_factory=LRUCache)
        generation = cache.seal("day-1")
        cache.store_empty("device-1", generation)
        cache.store_value("device-2", "payload")  # evicts device-1
        cache.store_value("device-1", "payload")
        self.assertEqual(cache.withdraw("day-1"), ())
        self.assertEqual(cache.get_value("device-1"), "payload")

    def test_oversized_store_keeps_existing_empty(self):
        cache = SourceCache(maxsize=5, cache_factory=SizedLRUCache)
        generation = cache.seal("day-1")
        cache.store_empty("device-1", generation)
        with self.assertRaises(ValueError):
            cache.store_value("device-1", "way too large a payload")
        self.assertTrue(cache.is_empty("device-1"))
        self.assertEqual(cache.withdraw("day-1"), ("device-1",))

    def test_expired_empty_is_not_reported_on_withdraw(self):
        timer = Timer()
        cache = SourceCache(maxsize=10, cache_factory=ttl_cache_factory(timer))
        generation = cache.seal("day-1")
        cache.store_empty("device-1", generation)
        timer.time = 11
        self.assertEqual(cache.withdraw("day-1"), ())
        self.assertIsNone(cache.get_hit("device-1"))

    def test_expiry_cleans_up_dependency_tracking(self):
        timer = Timer()
        cache = SourceCache(maxsize=10, cache_factory=ttl_cache_factory(timer))
        day1 = cache.seal("day-1")
        day2 = cache.seal("day-2")
        cache.store_empty("device-1", day1)
        timer.time = 11
        cache.store_empty("device-2", day2)  # triggers expiry of device-1
        self.assertNotIn("device-1", cache)
        self.assertEqual(cache.withdraw("day-1"), ())
        self.assertEqual(cache.withdraw("day-2"), ("device-2",))

    def test_record_namedtuple(self):
        generation = Generation("day-1", 1)
        record = SourceCacheRecord(None, True, (generation,))
        self.assertEqual(record.value, None)
        self.assertTrue(record.empty)
        self.assertEqual(record.generations, (generation,))
        with self.assertRaises(AttributeError):
            record.empty = False

    def test_positive_and_empty_coexist_show_generation(self):
        cache = SourceCache()
        first = cache.seal("day-1")
        cache.store_value("device-1", "payload")
        cache.store_empty("device-2", first)
        second = cache.seal("day-1")
        cache.store_empty("device-3", second)

        positive = cache.get_hit("device-1")
        old_empty = cache.get_hit("device-2")
        new_empty = cache.get_hit("device-3")
        self.assertFalse(positive.empty)
        self.assertEqual(positive.generations, ())
        self.assertTrue(old_empty.empty)
        self.assertEqual(old_empty.generations, (first,))
        self.assertTrue(new_empty.empty)
        self.assertEqual(new_empty.generations, (second,))
        self.assertEqual(set(cache.withdraw("day-1")), {"device-2", "device-3"})

    def test_stubs(self):
        # type stubs must infer key, value and source types
        cache: SourceCache[str, dict[str, Any]] = SourceCache()
        generation: Generation[str] = cache.seal("day-1")
        sealed: Generation[str] | None = cache.sealed("day-1")
        self.assertEqual(sealed, generation)

        cache.store_value("device-1", {"v": 1}, generation)
        cache.store_empty("device-2", generation)

        hit: Hit[str, dict[str, Any]] | None = cache.get_hit("device-1")
        assert hit is not None
        value: dict[str, Any] | None = cache.get_value("device-1")
        self.assertEqual(value, {"v": 1})
        self.assertIsNone(cache.get_value("missing"))

        sized: SourceCache[str, str] = SourceCache(maxsize=4, cache_factory=LRUCache)
        self.assertEqual(sized.maxsize, 4)

        record: SourceCacheRecord[str] = SourceCacheRecord(
            "payload", False, (generation,)
        )
        self.assertEqual(record.value, "payload")


if __name__ == "__main__":
    unittest.main()
