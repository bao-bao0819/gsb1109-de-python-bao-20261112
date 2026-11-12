"""Standalone LRU and LFU simulations used for the comparison.

The eviction rules mirror :class:`cachetools.LRUCache` and
:class:`cachetools.LFUCache` on the unit-size fast path:

* LRU evicts the least recently used partition.
* LFU evicts the partition with the smallest use count; partitions
  tied on count are broken by least recent use.

A simulation works on a plain request sequence starting from an empty
cache, so it can be replayed identically for both policies and, when
needed, against the real ``cachetools`` classes.
"""

from collections import OrderedDict


class _BasePolicy:
    name = "base"

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data = OrderedDict()

    def access(self, key):
        """Request ``key``; return True on a miss (origin fetch)."""
        if key in self._data:
            self._on_hit(key)
            return False
        self._on_miss(key)
        return True

    def _on_hit(self, key):
        raise NotImplementedError

    def _on_miss(self, key):
        raise NotImplementedError

    def contents(self):
        return tuple(self._data.keys())

    def snapshot(self):
        raise NotImplementedError

    def restore(self, state):
        raise NotImplementedError


class LRU(_BasePolicy):
    """Least Recently Used, matching ``cachetools.LRUCache``."""

    name = "LRU"

    def _on_hit(self, key):
        self._data.move_to_end(key)

    def _on_miss(self, key):
        if len(self._data) >= self.capacity:
            self._data.popitem(last=False)
        self._data[key] = None

    def snapshot(self):
        return tuple(self._data.keys())

    def restore(self, state):
        self._data = OrderedDict((key, None) for key in state)


class LFU(_BasePolicy):
    """Least Frequently Used with LRU tie-break, matching ``cachetools``."""

    name = "LFU"

    def __init__(self, capacity):
        super().__init__(capacity)
        self._count = {}

    def _on_hit(self, key):
        self._count[key] += 1
        # Keep a true recency order so count ties resolve by LRU, exactly
        # like the per-count OrderedDict in cachetools.LFUCache.
        self._data.move_to_end(key)

    def _on_miss(self, key):
        if len(self._data) >= self.capacity:
            # Iteration is LRU order; the first smallest-count key is thus
            # the least recently used one among the minimum-count keys.
            victim = min(self._data, key=self._count.__getitem__)
            del self._data[victim]
            del self._count[victim]
        self._data[key] = None
        self._count[key] = 1

    def contents(self):
        return tuple(self._data.keys())

    def snapshot(self):
        # OrderedDict order is the global recency order, which together
        # with the use counts fully determines LFU/LRU tie-break state.
        return tuple((key, self._count[key]) for key in self._data)

    def restore(self, state):
        self._data = OrderedDict()
        self._count = {}
        for key, count in state:
            self._data[key] = None
            self._count[key] = count


def replay(policy, sequence):
    """Replay a sequence, returning per-request ``(key, miss)`` steps."""
    steps = []
    for key in sequence:
        steps.append((key, policy.access(key)))
    return steps


def evaluate_sequence(sequence, capacity):
    """Return ``(lru_misses, lfu_misses)`` for one concrete sequence."""
    lru = LRU(capacity)
    lfu = LFU(capacity)
    lru_misses = sum(lru.access(key) for key in sequence)
    lfu_misses = sum(lfu.access(key) for key in sequence)
    return lru_misses, lfu_misses
