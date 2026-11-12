"""Source-aware caches with completion markers for empty results.

A cache lookup usually stores positive results, i.e. values that were
found.  An *empty* result ("this key has no data") is different: it is
only trustworthy once the data source, e.g. a collection manifest for
a given day, has been fully processed.  This module ties empty results
to the completion markers of the sources that were searched.

A source is marked as complete with :meth:`SourceCache.seal`, which
returns an immutable :class:`Generation` marker.  Empty results may
only be stored together with the markers of the sealed sources they
depend on.  If a source is later reopened via
:meth:`SourceCache.withdraw`, every empty result depending on it is
invalidated, while positive results and empty results derived from
other sources stay valid.  Every :class:`Hit` records the exact
generation(s) it is based on, so callers can tell which manifest
generation a read was served from.

"""

__all__ = (
    "Generation",
    "Hit",
    "SourceCache",
    "SourceCacheRecord",
)

import collections
import itertools
import math

from . import Cache


class Generation(collections.namedtuple("Generation", "source revision")):
    """Immutable completion marker for one generation of a source."""

    __slots__ = ()


class SourceCacheRecord(
    collections.namedtuple("SourceCacheRecord", "value empty generations")
):
    """One cached entry: its `value`, whether it is a confirmed empty
    result, and the :class:`Generation` markers it is based on.

    """

    __slots__ = ()


class Hit(collections.namedtuple("Hit", "key value empty generations")):
    """The result of a successful :meth:`SourceCache.get_hit` lookup.

    `empty` is :const:`True` for a confirmed empty result, in which
    case `value` is :const:`None` and `generations` lists the sealed
    source generations the emptiness was confirmed against.  For a
    positive result `empty` is :const:`False`; any `generations`
    attached to it describe where the value came from but never cause
    invalidation.

    """

    __slots__ = ()


def _make_cache(factory, maxsize, forget):
    class _TrackingCache(factory):
        def popitem(self):
            key, value = super().popitem()
            forget(key, value)
            return key, value

    if hasattr(factory, "expire"):

        def expire(self, time=None):
            expired = factory.expire(self, time)
            for key, value in expired:
                forget(key, value)
            return expired

        _TrackingCache.expire = expire

    return _TrackingCache(maxsize)


class SourceCache:
    """Cache storing positive results and source-dependent empty results.

    Positive results can be stored at any time with
    :meth:`store_value`.  Empty results can only be stored with
    :meth:`store_empty` together with the :class:`Generation` markers
    of one or more sealed sources, and are invalidated automatically
    when any of those sources is withdrawn.

    The underlying key/value cache is created from `cache_factory`,
    which must be a :class:`~cachetools.Cache` subclass, and bounded by
    `maxsize`.  Evicted or expired records have their dependency
    tracking cleaned up automatically.

    """

    __marker = object()

    def __init__(self, maxsize=math.inf, *, cache_factory=Cache):
        self.__cache = _make_cache(cache_factory, maxsize, self.__forget)
        self.__generations = {}
        self.__dependents = {}
        self.__revisions = itertools.count(1)

    def seal(self, source):
        """Mark the current generation of `source` as complete.

        Returns the :class:`Generation` marker identifying this
        generation.  Sealing the same source again starts a new
        generation without touching already cached records.

        """
        revision = next(self.__revisions)
        self.__generations[source] = revision
        return Generation(source, revision)

    def withdraw(self, source):
        """Reopen `source` and invalidate its dependent empty results.

        Every confirmed empty result depending on the source (in any
        generation) is removed from the cache and its key is returned
        as a tuple, in unspecified order.  Positive results and empty
        results depending only on other sources are kept.  Withdrawing
        a source that was never sealed returns an empty tuple.

        """
        if self.__generations.pop(source, None) is None:
            return ()
        keys = self.__dependents.pop(source, None)
        if keys is None:
            return ()
        removed = []
        for key in tuple(keys):
            record = self.__cache.get(key)
            if record is None:
                # evicted or expired underneath us; its tracking was
                # cleaned up by popitem()/expire() already
                continue
            del self.__cache[key]
            self.__forget(key, record, skipped=source)
            removed.append(key)
        return tuple(removed)

    def store_value(self, key, value, generation=None, *, generations=()):
        """Store a positive result `value` for `key`.

        Source markers may be attached to record where the value came
        from; they are validated but positive results are never
        invalidated by :meth:`withdraw`.

        """
        markers = self.__resolve(generation, generations)
        self.__put(key, SourceCacheRecord(value, False, markers))

    def store_empty(self, key, generation=None, *, generations=()):
        """Store a confirmed empty result for `key`.

        At least one current :class:`Generation` marker of a sealed
        source must be given; the empty result is invalidated once any
        of those sources is withdrawn.  Storing an empty result without
        markers raises :exc:`ValueError`, and stale or unknown markers
        raise :exc:`ValueError` as well.

        """
        markers = self.__resolve(generation, generations)
        if not markers:
            raise ValueError(
                "empty result requires at least one sealed source generation"
            )
        self.__put(key, SourceCacheRecord(None, True, markers))

    def get_hit(self, key):
        """Return the :class:`Hit` for `key`, or :const:`None` on a miss."""
        record = self.__cache.get(key)
        if record is None:
            return None
        return Hit(key, record.value, record.empty, record.generations)

    def get_value(self, key, default=None):
        """Return the cached value for `key`, or `default` on a miss.

        Confirmed empty results and misses both return :const:`None`
        unless another `default` is given; use :meth:`get_hit` or
        :meth:`is_empty` to tell them apart.

        """
        hit = self.get_hit(key)
        if hit is None:
            return default
        return hit.value

    def is_empty(self, key):
        """Return whether `key` holds a confirmed empty result."""
        record = self.__cache.get(key)
        return record is not None and record.empty

    def sealed(self, source):
        """Return the current :class:`Generation` of a sealed source,
        or :const:`None` if the source is not sealed.

        """
        revision = self.__generations.get(source)
        if revision is None:
            return None
        return Generation(source, revision)

    def generations(self, key):
        """Return the :class:`Generation` markers recorded for `key`,
        or an empty tuple on a miss.

        """
        record = self.__cache.get(key)
        if record is None:
            return ()
        return record.generations

    def clear(self):
        """Remove all cached records.

        Completion markers are kept, since they describe the state of
        the sources rather than cache contents; new empty results can
        be stored against sealed sources without sealing them again.

        """
        self.__cache.clear()
        self.__dependents.clear()

    def keys(self):
        """Return a view of the cached keys."""
        return self.__cache.keys()

    @property
    def maxsize(self):
        """The maximum size of the underlying cache."""
        return self.__cache.maxsize

    def __contains__(self, key):
        return key in self.__cache

    def __iter__(self):
        return iter(self.__cache)

    def __len__(self):
        return len(self.__cache)

    def __resolve(self, generation, generations):
        if generation is None:
            markers = tuple(generations)
        else:
            markers = (generation, *generations)
        resolved = []
        seen = set()
        for marker in markers:
            if not isinstance(marker, Generation):
                raise TypeError("generation must be a Generation instance")
            revision = self.__generations.get(marker.source)
            if revision != marker.revision:
                raise ValueError("stale or unknown source generation: %r" % (marker,))
            if marker.source not in seen:
                seen.add(marker.source)
                resolved.append(marker)
        return tuple(resolved)

    def __put(self, key, record):
        old = self.__cache.get(key, self.__marker)
        self.__cache[key] = record
        if old is not self.__marker:
            self.__forget(key, old)
        self.__remember(key, record)

    def __remember(self, key, record):
        if not record.empty:
            return
        for marker in record.generations:
            self.__dependents.setdefault(marker.source, set()).add(key)

    def __forget(self, key, record, skipped=None):
        if not record.empty:
            return
        for marker in record.generations:
            if marker.source == skipped:
                continue
            keys = self.__dependents[marker.source]
            keys.discard(key)
            if not keys:
                del self.__dependents[marker.source]
