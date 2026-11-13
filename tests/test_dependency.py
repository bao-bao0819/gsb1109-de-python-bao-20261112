import pickle
import threading
import time
import unittest

import cachetools

from . import CacheTestMixin


class DependencyCacheTest(unittest.TestCase, CacheTestMixin):
    Cache = cachetools.DependencyCache


class DependencyTest(unittest.TestCase):
    def setUp(self):
        self.cache = cachetools.DependencyCache(maxsize=100)

    def test_compute(self):
        calls = []

        def func():
            calls.append(None)
            return 42

        self.assertEqual(self.cache.compute("key", func), 42)
        self.assertEqual(self.cache["key"], 42)
        # cached value is returned without calling func again
        self.assertEqual(self.cache.compute("key", func), 42)
        self.assertEqual(len(calls), 1)

    def test_compute_error(self):
        def func():
            raise ValueError("test error")

        with self.assertRaises(ValueError):
            self.cache.compute("key", func)
        self.assertNotIn("key", self.cache)
        self.assertEqual(self.cache.dependencies("key"), frozenset())
        # pending computation is cleaned up, so this does not deadlock
        self.assertEqual(self.cache.compute("key", lambda: 42), 42)

    def test_dependencies(self):
        self.cache["b"] = 2
        self.cache["c"] = 3

        value = self.cache.compute("a", lambda: self.cache["b"] + self.cache["c"])

        self.assertEqual(value, 5)
        self.assertEqual(self.cache.dependencies("a"), {"b", "c"})
        self.assertEqual(self.cache.dependencies("b"), frozenset())
        self.assertEqual(self.cache.dependencies("unknown"), frozenset())

    def test_dependencies_nested_compute(self):
        def outer():
            return self.cache.compute("inner", lambda: 1) + 1

        self.assertEqual(self.cache.compute("outer", outer), 2)
        self.assertEqual(self.cache.dependencies("outer"), {"inner"})
        self.assertEqual(self.cache.dependencies("inner"), frozenset())
        # a cache hit of a nested compute() still counts as a dependency
        self.cache.revise("outer")
        self.assertEqual(self.cache.compute("outer", outer), 2)
        self.assertEqual(self.cache.dependencies("outer"), {"inner"})

    def test_dependencies_dynamic(self):
        # different branches read different sources; a successful
        # recomputation replaces the recorded dependencies
        self.cache["peak_source"] = 1
        self.cache["offpeak_source"] = 2
        peak = True

        def seasonal():
            source = "peak_source" if peak else "offpeak_source"
            return self.cache[source] * 10

        self.assertEqual(self.cache.compute("sales", seasonal), 10)
        self.assertEqual(self.cache.dependencies("sales"), {"peak_source"})

        self.cache.revise("peak_source")
        self.assertNotIn("sales", self.cache)

        peak = False
        self.assertEqual(self.cache.compute("sales", seasonal), 20)
        self.assertEqual(self.cache.dependencies("sales"), {"offpeak_source"})

        # the outdated source no longer invalidates the item...
        self.assertEqual(self.cache.revise("peak_source"), {"peak_source"})
        self.assertEqual(self.cache["sales"], 20)
        # ...but the source actually read does
        self.assertEqual(
            self.cache.revise("offpeak_source"), {"offpeak_source", "sales"}
        )
        self.assertNotIn("sales", self.cache)

    def test_dependencies_failed_recompute(self):
        # a failed recomputation must not erase the dependencies
        # recorded by the last successful computation
        self.cache["source"] = 1
        self.assertEqual(
            self.cache.compute("feature", lambda: self.cache["source"] * 2), 2
        )
        self.assertEqual(self.cache.dependencies("feature"), {"source"})

        self.cache.revise("source")
        self.assertNotIn("feature", self.cache)
        # dependency evidence survives invalidation of the value
        self.assertEqual(self.cache.dependencies("feature"), {"source"})

        def failing():
            raise ValueError("test error")

        with self.assertRaises(ValueError):
            self.cache.compute("feature", failing)
        self.assertNotIn("feature", self.cache)
        self.assertEqual(self.cache.dependencies("feature"), {"source"})

    def test_dependencies_value_too_large(self):
        cache = cachetools.DependencyCache(maxsize=2, getsizeof=len)
        cache["source"] = "x"
        self.assertEqual(cache.compute("feature", lambda: cache["source"]), "x")
        self.assertEqual(cache.dependencies("feature"), {"source"})

        cache.revise("source")
        # value too large to cache: returned, but not stored, and the
        # previously recorded dependencies are left unchanged
        self.assertEqual(cache.compute("feature", lambda: "xyz"), "xyz")
        self.assertNotIn("feature", cache)
        self.assertEqual(cache.dependencies("feature"), {"source"})

    def test_revise(self):
        self.cache["source"] = 1
        self.cache["unrelated"] = 2
        self.cache.compute("feature", lambda: self.cache["source"] * 2)

        invalidated = self.cache.revise("source")

        self.assertEqual(invalidated, {"source", "feature"})
        self.assertNotIn("source", self.cache)
        self.assertNotIn("feature", self.cache)
        # unrelated items are not affected
        self.assertEqual(self.cache["unrelated"], 2)

    def test_revise_transitive(self):
        self.cache["source"] = 1
        self.cache.compute("feature", lambda: self.cache["source"] * 2)
        self.cache.compute("top", lambda: self.cache["feature"] * 3)

        invalidated = self.cache.revise("source")

        self.assertEqual(invalidated, {"source", "feature", "top"})
        self.assertEqual(len(self.cache), 0)

    def test_revise_transitive_after_eviction(self):
        # dependency records are kept when values are evicted, so
        # invalidation remains transitive through evicted items
        self.cache["source"] = 1
        self.cache.compute("feature", lambda: self.cache["source"] * 2)
        self.cache.compute("top", lambda: self.cache["feature"] * 3)
        del self.cache["feature"]

        invalidated = self.cache.revise("source")

        self.assertEqual(invalidated, {"source", "feature", "top"})
        self.assertNotIn("top", self.cache)

    def test_revise_multiple_keys(self):
        self.cache["a"] = 1
        self.cache["b"] = 2
        self.cache.compute("ab", lambda: self.cache["a"] + self.cache["b"])

        invalidated = self.cache.revise("a", "b")

        self.assertEqual(invalidated, {"a", "b", "ab"})
        self.assertEqual(len(self.cache), 0)

    def test_revise_unknown_key(self):
        self.assertEqual(self.cache.revise("unknown"), {"unknown"})

    def test_clear_dependencies(self):
        self.cache["source"] = 1
        self.cache.compute("feature", lambda: self.cache["source"] * 2)

        self.cache.clear()

        self.assertEqual(self.cache.dependencies("feature"), frozenset())
        self.assertEqual(self.cache.revise("source"), {"source"})

    def test_pickle_dependencies(self):
        self.cache["source"] = 1
        self.cache.compute("feature", lambda: self.cache["source"] * 2)

        cache = pickle.loads(pickle.dumps(self.cache))

        self.assertEqual(cache.dependencies("feature"), {"source"})
        self.assertEqual(cache.revise("source"), {"source", "feature"})
        self.assertNotIn("feature", cache)

    def test_cycle_same_key(self):
        def func():
            return self.cache.compute("a", lambda: 1)

        with self.assertRaises(cachetools.CyclicDependencyError) as cm:
            self.cache.compute("a", func)
        self.assertEqual(cm.exception.keys, ("a", "a"))
        # pending computation is cleaned up, so this does not deadlock
        self.assertEqual(self.cache.compute("a", lambda: 1), 1)

    def test_cycle_nested(self):
        def func_a():
            return self.cache.compute("b", func_b)

        def func_b():
            return self.cache.compute("a", lambda: 1)

        with self.assertRaises(cachetools.CyclicDependencyError) as cm:
            self.cache.compute("a", func_a)
        self.assertEqual(cm.exception.keys, ("a", "b", "a"))
        self.assertEqual(self.cache.compute("a", lambda: 1), 1)

    def test_cycle_error_attributes(self):
        error = cachetools.CyclicDependencyError(["a", "b"])
        self.assertEqual(error.keys, ("a", "b"))
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("'a'", str(error))
        self.assertIn("'b'", str(error))


class DependencyThreadingTest(unittest.TestCase):
    TIMEOUT = 10

    def test_concurrent_compute(self):
        cache = cachetools.DependencyCache(maxsize=2)
        calls = []
        started = threading.Event()
        release = threading.Event()

        def func():
            calls.append(None)
            started.set()
            release.wait(self.TIMEOUT)
            return object()

        results = [None] * 2

        def run(index):
            results[index] = cache.compute("key", func)

        first = threading.Thread(target=run, args=[0])
        first.start()
        self.assertTrue(started.wait(self.TIMEOUT))
        second = threading.Thread(target=run, args=[1])
        second.start()
        # second caller must block on the pending key
        deadline = time.monotonic() + self.TIMEOUT
        while time.monotonic() < deadline:
            with cache._DependencyCache__condition:
                if cache._DependencyCache__waiting:
                    break
            time.sleep(0.01)
        with cache._DependencyCache__condition:
            self.assertTrue(cache._DependencyCache__waiting)
        release.set()
        for thread in (first, second):
            thread.join(timeout=self.TIMEOUT)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(results[0])
        self.assertIs(results[0], results[1])

    def test_concurrent_compute_error(self):
        # when the pending computation fails, a waiting thread
        # computes the value itself
        cache = cachetools.DependencyCache(maxsize=2)
        calls = []
        started = threading.Event()
        release = threading.Event()

        def func():
            calls.append(None)
            if len(calls) == 1:
                started.set()
                release.wait(self.TIMEOUT)
                raise ValueError("test error")
            return 42

        results = [None] * 2
        errors = []

        def run(index):
            try:
                results[index] = cache.compute("key", func)
            except ValueError:
                errors.append(index)

        first = threading.Thread(target=run, args=[0])
        first.start()
        self.assertTrue(started.wait(self.TIMEOUT))
        second = threading.Thread(target=run, args=[1])
        second.start()
        deadline = time.monotonic() + self.TIMEOUT
        while time.monotonic() < deadline:
            with cache._DependencyCache__condition:
                if cache._DependencyCache__waiting:
                    break
            time.sleep(0.01)
        release.set()
        for thread in (first, second):
            thread.join(timeout=self.TIMEOUT)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [0])
        self.assertEqual(results[1], 42)
        self.assertEqual(cache["key"], 42)

    def test_cyclic_dependency(self):
        # feature A (re)computed by one thread now reads feature B,
        # while feature B computed by another thread reads feature A:
        # the wait must end with a CyclicDependencyError naming the
        # participating features and partitions, and unrelated
        # partitions must continue to produce values
        cache = cachetools.DependencyCache(maxsize=100)
        key_a = ("feature_a", "store_1")
        key_b = ("feature_b", "store_1")
        key_c = ("feature_c", "store_2")
        a_ready = threading.Event()
        b_ready = threading.Event()
        results = {}
        errors = []

        def func_a():
            a_ready.set()
            b_ready.wait(self.TIMEOUT)
            return cache.compute(key_b, func_b)

        def func_b():
            b_ready.set()
            a_ready.wait(self.TIMEOUT)
            return cache.compute(key_a, func_a)

        def run(name, key, func):
            try:
                results[name] = cache.compute(key, func)
            except cachetools.CyclicDependencyError as e:
                errors.append(e)

        threads = [
            threading.Thread(target=run, args=("a", key_a, func_a)),
            threading.Thread(target=run, args=("b", key_b, func_b)),
            threading.Thread(target=run, args=("c", key_c, lambda: 42)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=self.TIMEOUT)
            self.assertFalse(thread.is_alive())

        # both cyclic computations ended with an error instead of
        # deadlocking, and the cycle names both features
        self.assertEqual(len(errors), 2)
        participants = set()
        for error in errors:
            participants.update(error.keys)
        self.assertEqual(participants, {key_a, key_b})

        # the unrelated store in the same cache kept producing
        self.assertEqual(results, {"c": 42})
        self.assertEqual(cache[key_c], 42)

        # pending computations are cleaned up, so the cache stays usable
        self.assertEqual(cache.compute(key_a, lambda: 1), 1)
        self.assertEqual(cache.compute(key_b, lambda: 2), 2)
