import unittest
from threading import Barrier, Condition, Event, Lock, Thread

from cachetools import LRUCache, cached, cachedmethod


class WaitCondition(Condition):
    def __init__(self, lock=None):
        super().__init__(lock)
        self.waiting = Event()

    def wait(self, timeout=None):
        self.waiting.set()
        return super().wait(timeout)


class ThreadingTest(unittest.TestCase):
    NTHREADS = 10

    TIMEOUT = 10

    def test_cached_lock(self):
        for info in (False, True):
            with self.subTest(info=info):
                barrier = Barrier(self.NTHREADS, timeout=self.TIMEOUT)
                cache = LRUCache(1)

                @cached(cache, lock=Lock(), info=info)
                def func(barrier=barrier):
                    result = object()
                    barrier.wait()
                    return result

                self.run_lock_test(func, cache, info)

    def test_cachedmethod_lock(self):
        for info in (False, True):
            with self.subTest(info=info):
                barrier = Barrier(self.NTHREADS, timeout=self.TIMEOUT)
                cache = LRUCache(1)
                lock = Lock()

                class Cached:
                    @cachedmethod(lambda _: cache, lock=lambda _: lock, info=info)
                    def method(self, barrier=barrier):
                        result = object()
                        barrier.wait()
                        return result

                self.run_lock_test(Cached().method, cache, info)

    def test_cached_cond(self):
        for info in (False, True):
            with self.subTest(info=info):
                cache = LRUCache(1)
                cond = WaitCondition()
                started = Event()
                release = Event()
                calls = []

                @cached(cache, condition=cond, info=info)
                def func(calls=calls, started=started, release=release):
                    calls.append(None)
                    started.set()
                    release.wait(self.TIMEOUT)
                    return object()

                self.run_cond_wait_test(func, cond, started, release, calls)

    def test_cachedmethod_cond(self):
        for info in (False, True):
            with self.subTest(info=info):
                cache = LRUCache(1)
                cond = WaitCondition()
                started = Event()
                release = Event()
                calls = []

                class Cached:
                    @cachedmethod(lambda _: cache, condition=lambda _: cond, info=info)
                    def method(
                        self,
                        calls=calls,
                        started=started,
                        release=release,
                        timeout=self.TIMEOUT,
                    ):
                        calls.append(None)
                        started.set()
                        release.wait(timeout)
                        return object()

                func = Cached().method
                self.run_cond_wait_test(func, cond, started, release, calls)

    def test_cached_lock_cond(self):
        for info in (False, True):
            with self.subTest(info=info):
                cache = LRUCache(1)
                lock = Lock()
                cond = WaitCondition(lock)
                started = Event()
                release = Event()
                calls = []

                @cached(cache, lock=lock, condition=cond, info=info)
                def func(calls=calls, started=started, release=release):
                    calls.append(None)
                    started.set()
                    release.wait(self.TIMEOUT)
                    return object()

                self.run_cond_wait_test(func, cond, started, release, calls)

    def test_cachedmethod_lock_cond(self):
        for info in (False, True):
            with self.subTest(info=info):
                cache = LRUCache(1)
                lock = Lock()
                cond = WaitCondition(lock)
                started = Event()
                release = Event()
                calls = []

                class Cached:
                    @cachedmethod(
                        lambda _: cache,
                        lock=lambda _: lock,
                        condition=lambda _: cond,
                        info=info,
                    )
                    def method(
                        self,
                        calls=calls,
                        started=started,
                        release=release,
                        timeout=self.TIMEOUT,
                    ):
                        calls.append(None)
                        started.set()
                        release.wait(timeout)
                        return object()

                func = Cached().method
                self.run_cond_wait_test(func, cond, started, release, calls)

    def test_cached_unrelated_lock_cond(self):
        # a condition variable not created with the lock fails on
        # the first cache miss, even without concurrent calls
        for info in (False, True):
            with self.subTest(info=info):

                @cached(LRUCache(1), lock=Lock(), condition=Condition(), info=info)
                def func():
                    return object()

                with self.assertRaises(RuntimeError):
                    func()

    def test_cachedmethod_unrelated_lock_cond(self):
        for info in (False, True):
            with self.subTest(info=info):
                cache = LRUCache(1)
                lock = Lock()
                cond = Condition()

                class Cached:
                    @cachedmethod(
                        lambda _: cache,
                        lock=lambda _: lock,
                        condition=lambda _: cond,
                        info=info,
                    )
                    def method(self):
                        return object()

                with self.assertRaises(RuntimeError):
                    Cached().method()

    def run_cond_wait_test(self, func, cond, started, release, calls):
        results = [None] * 2

        def run(index):
            results[index] = func()

        first = Thread(target=run, args=[0])
        first.start()
        self.assertTrue(started.wait(self.TIMEOUT))
        second = Thread(target=run, args=[1])
        second.start()
        # second caller must block on the pending key
        self.assertTrue(cond.waiting.wait(self.TIMEOUT))
        release.set()
        for thread in (first, second):
            thread.join(timeout=self.TIMEOUT)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(results[0])
        self.assertIs(results[0], results[1])

    def run_lock_test(self, func, cache, info):
        self.run_threading_test(func, cache)

        if info:
            cache_info = func.cache_info()
            self.assertEqual(cache_info.hits + cache_info.misses, self.NTHREADS)
            self.assertGreaterEqual(cache_info.misses, 1)

    def run_threading_test(self, func, cache):
        results = [None] * self.NTHREADS

        def run(index):
            results[index] = func()

        threads = [Thread(target=run, args=[i]) for i in range(self.NTHREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=self.TIMEOUT)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(cache), 1)

        # multiple threads may compute the value, but the first one
        # should always win and all concurrent invocations should
        # return the same result
        cached_result = next(iter(cache.values()))
        for result in results:
            self.assertIs(result, cached_result)


class InvalidationRaceTest(unittest.TestCase):
    # a key invalidated while a computation for it is still in
    # flight: invalidation only affects values already stored in the
    # cache, it cannot retract the pending computation, whose stale
    # result is still written back

    TIMEOUT = 10

    def run_invalidation_race(self):
        # caller A computes a partition summary from the old input
        # while caller B waits on the same key; meanwhile the job
        # publishes new input and invalidates the key
        cache = LRUCache(4)
        cond = WaitCondition()
        published = ["v1"]
        reads = []
        computing = Event()
        release = Event()

        @cached(cache, condition=cond)
        def summary(
            partition,
            published=published,
            reads=reads,
            computing=computing,
            release=release,
            timeout=self.TIMEOUT,
        ):
            version = published[0]
            reads.append(version)
            if not release.is_set():
                computing.set()
                release.wait(timeout)
            return (partition, version)

        results = [None, None]

        def run(index):
            results[index] = summary("part-0")

        first = Thread(target=run, args=[0])
        first.start()
        self.assertTrue(computing.wait(self.TIMEOUT))
        second = Thread(target=run, args=[1])
        second.start()
        # second caller must block on the pending key
        self.assertTrue(cond.waiting.wait(self.TIMEOUT))

        published[0] = "v2"
        cache.pop(summary.cache_key("part-0"), None)

        release.set()
        for thread in (first, second):
            thread.join(timeout=self.TIMEOUT)
            self.assertFalse(thread.is_alive())
        return summary, cache, results, reads

    def test_waiter_reads_stale_answer(self):
        _, _, results, reads = self.run_invalidation_race()
        # the original caller computed from the old input, and the
        # caller waiting on its pending key received that stale
        # answer; no recomputation took place
        self.assertEqual(reads, ["v1"])
        self.assertEqual(results[0], ("part-0", "v1"))
        self.assertEqual(results[1], ("part-0", "v1"))

    def test_stale_write_repollutes_next_batch(self):
        summary, cache, _, reads = self.run_invalidation_race()
        # the stale result was written back after the invalidation,
        # so a caller arriving now is served the stale value from
        # cache without recomputing
        self.assertEqual(summary("part-0"), ("part-0", "v1"))
        self.assertEqual(reads, ["v1"])
        self.assertEqual(cache[summary.cache_key("part-0")], ("part-0", "v1"))

    def test_invalidation_effective_once_quiescent(self):
        summary, cache, _, reads = self.run_invalidation_race()
        # all the interface can promise: with no computation in
        # flight, invalidating the key takes effect and the next
        # caller recomputes from the new input
        cache.pop(summary.cache_key("part-0"), None)
        self.assertEqual(summary("part-0"), ("part-0", "v2"))
        self.assertEqual(reads, ["v1", "v2"])
        self.assertEqual(cache[summary.cache_key("part-0")], ("part-0", "v2"))
