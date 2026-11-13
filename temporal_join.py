"""Caching time-dependent dimension joins with :mod:`cachetools`.

A March order may only be keyed in May, and the province and tax
dimension tables are joined as of the *order* time, so the same store
key can yield different results in March and in May.  This module
caches such joins: the caller loads every dimension record with its
effective interval ``[valid_from, valid_to)`` and a revision code, and
a cached result is reused only when *both* tables resolve to the same
interpretation -- the same pair of revision codes -- at the order
time.

A retroactive correction is loaded as a new record with a new
revision code covering the corrected interval.  Orders before or
after that interval still resolve to the old revisions and keep
hitting the existing cache entries, orders inside the interval miss
and are recomputed, and other stores sharing the corrected tax record
pick up the new result without any explicit invalidation.

Revision codes must uniquely identify a record version across a
table; reusing a code for a changed record would allow stale joins.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Hashable

from cachetools import CacheInfo, LRUCache, cached

__all__ = ["DimRecord", "TemporalDimTable", "TemporalJoinCache"]


@dataclass(frozen=True)
class DimRecord:
    """One version of a dimension record.

    `key` is the natural key (store id, region code, ...), the record
    is effective for ``valid_from <= at < valid_to`` (use `date.max`
    for open-ended records), `revision` is the caller-supplied
    revision code, and `value` is the payload.
    """

    key: Hashable
    valid_from: date
    valid_to: date
    revision: Hashable
    value: Any


class TemporalDimTable:
    """A dimension table versioned by effective interval."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._versions: dict[Hashable, list[DimRecord]] = {}

    def add(self, record: DimRecord) -> None:
        """Load `record`; overlapping intervals are allowed.

        When intervals overlap -- e.g. a retroactive correction nested
        inside an older record -- the version with the latest
        `valid_from` wins.
        """
        versions = self._versions.setdefault(record.key, [])
        versions.append(record)
        versions.sort(key=lambda version: version.valid_from)

    def resolve(self, key: Hashable, at: date) -> DimRecord | None:
        """Return the record effective at `at`, or `None`."""
        best: DimRecord | None = None
        for record in self._versions.get(key, ()):
            if record.valid_from <= at < record.valid_to:
                if best is None or record.valid_from > best.valid_from:
                    best = record
        return best


def _join_key(province: DimRecord, tax: DimRecord) -> tuple[Hashable, Hashable]:
    # reuse is allowed only when both tables make the same
    # interpretation, i.e. resolve to the same revision codes
    return (province.revision, tax.revision)


class TemporalJoinCache:
    """Caches joins of a province and a tax dimension table.

    Province records map a store key to ``{"region": ...}`` and tax
    records map a region code to ``{"rate": ...}``; both are resolved
    at the order time, so a backfilled order is enriched with the
    interpretation of its order date, not of its entry date.
    """

    def __init__(self, maxsize: int = 1024) -> None:
        self.province = TemporalDimTable("province")
        self.tax = TemporalDimTable("tax")
        self.cache: LRUCache[tuple[Hashable, Hashable], dict[str, Any]] = LRUCache(
            maxsize
        )

        @cached(self.cache, key=_join_key, info=True)
        def join(province: DimRecord, tax: DimRecord) -> dict[str, Any]:
            return {
                "region": province.value["region"],
                "tax_rate": tax.value["rate"],
                "province_revision": province.revision,
                "tax_revision": tax.revision,
            }

        self._join = join

    def enrich(self, store: Hashable, ordered_at: date) -> dict[str, Any]:
        """Return the cached join for `store` as of `ordered_at`."""
        province = self.province.resolve(store, ordered_at)
        if province is None:
            raise KeyError(f"no province record for {store!r} at {ordered_at}")
        region = province.value["region"]
        tax = self.tax.resolve(region, ordered_at)
        if tax is None:
            raise KeyError(f"no tax record for {region!r} at {ordered_at}")
        return self._join(province, tax)

    def cache_info(self) -> CacheInfo:
        """Return hit/miss statistics of the join cache."""
        return self._join.cache_info()
