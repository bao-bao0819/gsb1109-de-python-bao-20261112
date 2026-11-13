"""
Migration of latency histograms between two bucket layouts.

An *old* histogram only stores, for every old bucket, how many samples fell in
it. Some old sources additionally record how many samples are *exactly equal*
to a threshold: such a count cannot be told apart from the rest of its bucket
in an ordinary histogram, so it is handled as its own singleton cell.

A *new* layout moves a few bucket boundaries. The exact per-bucket counts of the
new layout cannot always be recovered from the old, coarser information: a new
bucket that straddles an old boundary only admits bounds on its true count.

Three situations are distinguished:

* ``EXACT`` -- the count can be rebuilt exactly from the old data (a whole old
  bucket that falls into a single new bucket, or an "exactly equal" count).
* ``BOUNDED`` -- only a lower and an upper bound on the true count are known.
* ``IMPOSSIBLE`` -- a requested combination of per-bucket counts cannot be
  produced by one and the same batch of samples.

The model behind the bounds: after peeling off the exact singleton cells, every
old bucket keeps a *pool* of indistinguishable samples. A pool that overlaps
several new buckets may distribute its samples freely among them. Pools are
independent, which makes the tight group bounds straightforward and lets the
feasibility of an explicit count vector be decided with a bipartite flow.
"""

from dataclasses import dataclass
from enum import Enum

from .const import Bound
from .interval import Interval

__all__ = [
    "MigrationStatus",
    "BucketMigration",
    "GroupRange",
    "HistogramMigrator",
]


class MigrationStatus(Enum):
    """How accurately a count can be recovered from the old histogram."""

    EXACT = "exact"
    BOUNDED = "bounded"
    IMPOSSIBLE = "impossible"


@dataclass(frozen=True)
class BucketMigration:
    """
    Migration verdict for a single new bucket.

    ``lower`` and ``upper`` enclose every count compatible with the old data.
    They coincide when ``status`` is EXACT. An IMPOSSIBLE verdict never applies
    to a lone bucket (it needs an explicit requested count); use
    :meth:`HistogramMigrator.feasible_combination` for that.
    """

    bucket: Interval
    status: MigrationStatus
    lower: int
    upper: int
    reason: str = ""

    @property
    def exact(self):
        """True when the bucket count is known exactly."""
        return self.status is MigrationStatus.EXACT

    def as_range(self):
        """Return the ``(lower, upper)`` bounds."""
        return self.lower, self.upper


@dataclass(frozen=True)
class GroupRange:
    """
    Feasible range of the *sum* of a group of new buckets.

    The bounds are tight for the group as a whole, not the sum of the
    per-bucket bounds: an old pool shared by two buckets of the group is
    counted only once in the upper bound, because the same samples cannot feed
    both buckets.
    """

    buckets: tuple
    status: MigrationStatus
    lower: object = None
    upper: object = None
    reason: str = ""

    def as_range(self):
        """Return the ``(lower, upper)`` bounds, or None when impossible."""
        if self.status is MigrationStatus.IMPOSSIBLE:
            return None
        return self.lower, self.upper


@dataclass(frozen=True)
class _Pool:
    """
    The free samples of one old bucket, after exact singletons are removed.

    ``destinations`` are the new buckets intersecting the bucket's residual
    region. When there is a single destination, every sample is forced there.
    """

    source: Interval
    size: int
    destinations: tuple


def _singleton(value):
    return Interval.from_atomic(Bound.CLOSED, value, value, Bound.CLOSED)


class HistogramMigrator:
    """
    Migrate an old latency histogram onto a new bucket layout.

    :param old_buckets: disjoint intervals partitioning the sampled values.
    :param new_buckets: disjoint intervals covering every old bucket.
    :param old_counts: mapping from each old bucket to its number of samples.
    :param exact_counts: optional mapping from a threshold value to the number
        of samples exactly equal to it.
    """

    def __init__(self, old_buckets, new_buckets, old_counts, exact_counts=None):
        self.old_buckets = tuple(old_buckets)
        self.new_buckets = tuple(new_buckets)
        self.old_counts = dict(old_counts)
        self.exact_counts = dict(exact_counts or {})

        self._check_disjoint("old_buckets", self.old_buckets)
        self._check_disjoint("new_buckets", self.new_buckets)
        self._check_counts()

        new_cover = self._union(self.new_buckets)
        old_cover = self._union(self.old_buckets)
        for threshold in self.exact_counts:
            if _singleton(threshold) not in old_cover:
                raise ValueError(
                    f"Exact threshold {threshold} belongs to no old bucket."
                )

        self._exact_by_bucket = {bucket: 0 for bucket in self.new_buckets}
        self._pools = self._build_pools(new_cover)
        self._migrations = {
            bucket: self._migrate_bucket(bucket) for bucket in self.new_buckets
        }

    @staticmethod
    def _check_disjoint(name, buckets):
        for bucket in buckets:
            if not isinstance(bucket, Interval):
                raise TypeError(f"{name} must contain Interval instances.")
        for index, bucket in enumerate(buckets):
            for other in buckets[index + 1 :]:
                if bucket.overlaps(other):
                    raise ValueError(
                        f"{name} buckets overlap: {bucket} and {other}."
                    )

    def _check_counts(self):
        for bucket in self.old_buckets:
            if bucket not in self.old_counts:
                raise ValueError(f"Missing old count for bucket {bucket}.")
            if self.old_counts[bucket] < 0:
                raise ValueError(f"Old count for bucket {bucket} is negative.")
        for threshold, count in self.exact_counts.items():
            if count < 0:
                raise ValueError(
                    f"Exact count for threshold {threshold} is negative."
                )

    @staticmethod
    def _union(buckets):
        result = Interval()
        for bucket in buckets:
            result = result | bucket
        return result

    def _build_pools(self, new_cover):
        pools = []

        for old_bucket in self.old_buckets:
            residual = old_bucket
            exact_total = 0

            for threshold, count in self.exact_counts.items():
                point = _singleton(threshold)
                if point.overlaps(old_bucket):
                    if not point.overlaps(new_cover):
                        raise ValueError(
                            f"Threshold {threshold} is not covered by any new bucket."
                        )
                    target = self._bucket_containing(point)
                    self._exact_by_bucket[target] += count
                    residual = residual - point
                    exact_total += count

            pool_size = self.old_counts[old_bucket] - exact_total
            if pool_size < 0:
                raise ValueError(
                    f"Exact counts in {old_bucket} exceed its total count."
                )
            if not (residual in new_cover):
                raise ValueError(
                    f"Old bucket {old_bucket} is not fully covered by new buckets."
                )

            destinations = tuple(
                bucket
                for bucket in self.new_buckets
                if not (residual & bucket).empty
            )
            if not destinations and pool_size:
                raise ValueError(
                    f"Old bucket {old_bucket} has {pool_size} samples but no "
                    "new bucket to migrate them to."
                )

            pools.append(
                _Pool(
                    source=old_bucket,
                    size=pool_size,
                    destinations=destinations,
                )
            )

        return pools

    def _bucket_containing(self, point):
        for bucket in self.new_buckets:
            if point in bucket:
                return bucket
        raise ValueError(f"No new bucket contains {point}.")

    def _pools_touching(self, bucket):
        return [pool for pool in self._pools if bucket in pool.destinations]

    def _migrate_bucket(self, bucket):
        fixed = self._exact_by_bucket[bucket]
        forced = 0
        optional = 0

        for pool in self._pools_touching(bucket):
            if len(pool.destinations) == 1:
                forced += pool.size
            else:
                optional += pool.size

        lower = fixed + forced
        upper = fixed + forced + optional

        if lower == upper:
            return BucketMigration(
                bucket=bucket,
                status=MigrationStatus.EXACT,
                lower=lower,
                upper=upper,
                reason="all contributing samples are unambiguously assigned",
            )
        return BucketMigration(
            bucket=bucket,
            status=MigrationStatus.BOUNDED,
            lower=lower,
            upper=upper,
            reason="shares free samples across an old bucket boundary",
        )

    def migrate(self, bucket):
        """Return the :class:`BucketMigration` of a single new bucket."""
        try:
            return self._migrations[bucket]
        except KeyError:
            raise KeyError(f"Unknown new bucket {bucket}.") from None

    def migrate_all(self):
        """Return every :class:`BucketMigration`, in new-bucket order."""
        return [self._migrations[bucket] for bucket in self.new_buckets]

    def group_range(self, buckets):
        """
        Tight feasible range for the sum of a group of new buckets.

        A pool contributes its full size to the lower bound only when all its
        destination buckets are inside the group; it contributes to the upper
        bound as soon as one destination is inside. Every pool is counted at
        most once regardless of how many group buckets it touches.

        :param buckets: a new bucket or an iterable of new buckets.
        :return: a :class:`GroupRange`.
        """
        group = self._as_group(buckets)
        group_set = set(group)

        fixed = sum(self._exact_by_bucket[bucket] for bucket in group)
        lower = fixed
        upper = fixed

        for pool in self._pools:
            inside = [bucket for bucket in pool.destinations if bucket in group_set]
            if not inside:
                continue
            upper += pool.size
            if len(inside) == len(pool.destinations):
                lower += pool.size

        status = MigrationStatus.EXACT if lower == upper else MigrationStatus.BOUNDED
        reason = (
            "group count is fully determined"
            if status is MigrationStatus.EXACT
            else "tight bounds from a jointly feasible allocation"
        )
        return GroupRange(
            buckets=group,
            status=status,
            lower=lower,
            upper=upper,
            reason=reason,
        )

    def feasible_combination(self, targets):
        """
        Decide whether requested per-bucket counts share one feasible batch.

        :param targets: mapping from new buckets to the counts they should hold.
        :return: a :class:`GroupRange` with status EXACT (the combination is
            reachable) or IMPOSSIBLE (no allocation of the old samples matches).
        """
        items = list(targets.items())
        group = tuple(bucket for bucket, _ in items)

        for bucket in group:
            if bucket not in self._migrations:
                raise KeyError(f"Unknown new bucket {bucket}.")
        if set(group) != set(self.new_buckets):
            raise ValueError(
                "feasible_combination expects a count for every new bucket."
            )

        needs = {}
        for bucket, wanted in items:
            if wanted < 0:
                return self._impossible(group, f"negative count for {bucket}.")
            free_need = wanted - self._exact_by_bucket[bucket]
            if free_need < 0:
                return self._impossible(
                    group,
                    f"exact samples already exceed the requested count of {bucket}.",
                )
            needs[bucket] = free_need

        total_need = sum(needs.values())
        total_free = sum(pool.size for pool in self._pools)
        if total_need != total_free:
            return self._impossible(
                group,
                f"requested free samples {total_need} differ from available "
                f"{total_free}.",
            )

        if self._max_flow(needs) != total_free:
            return self._impossible(
                group,
                "old buckets cannot route enough samples to the requested buckets.",
            )

        return GroupRange(
            buckets=group,
            status=MigrationStatus.EXACT,
            lower=sum(targets.values()),
            upper=sum(targets.values()),
            reason="a jointly feasible allocation of the old samples exists",
        )

    def is_feasible(self, targets):
        """Boolean shortcut for :meth:`feasible_combination`."""
        return self.feasible_combination(targets).status is MigrationStatus.EXACT

    def _max_flow(self, needs):
        # Bipartite flow: source -> old pools -> new buckets -> sink.
        sources = [pool for pool in self._pools if pool.size]
        bucket_index = {bucket: index for index, bucket in enumerate(self.new_buckets)}
        source_base = 1
        bucket_base = source_base + len(sources)
        sink = bucket_base + len(self.new_buckets)

        graph = [[] for _ in range(sink + 1)]

        def add_edge(origin, target, capacity):
            graph[origin].append([target, capacity, len(graph[target])])
            graph[target].append([origin, 0, len(graph[origin]) - 1])

        for pool_index, pool in enumerate(sources):
            pool_node = source_base + pool_index
            add_edge(0, pool_node, pool.size)
            for bucket in pool.destinations:
                add_edge(pool_node, bucket_base + bucket_index[bucket], pool.size)

        for bucket, need in needs.items():
            add_edge(bucket_base + bucket_index[bucket], sink, need)

        flow = 0
        while True:
            level = [-1] * len(graph)
            level[0] = 0
            queue = [0]
            for node in queue:
                for target, capacity, reverse in graph[node]:
                    if capacity > 0 and level[target] < 0:
                        level[target] = level[node] + 1
                        queue.append(target)
            if level[sink] < 0:
                return flow

            def augment(node, pushed):
                if node == sink:
                    return pushed
                while edge_pointer[node] < len(graph[node]):
                    edge = graph[node][edge_pointer[node]]
                    target, capacity, reverse = edge
                    if capacity > 0 and level[target] == level[node] + 1:
                        sent = augment(target, min(pushed, capacity))
                        if sent:
                            edge[1] -= sent
                            graph[target][reverse][1] += sent
                            return sent
                    edge_pointer[node] += 1
                return 0

            edge_pointer = [0] * len(graph)
            while True:
                sent = augment(0, float("inf"))
                if not sent:
                    break
                flow += sent

    @staticmethod
    def _impossible(group, reason):
        return GroupRange(
            buckets=group,
            status=MigrationStatus.IMPOSSIBLE,
            reason=reason,
        )

    def _as_group(self, buckets):
        if isinstance(buckets, Interval):
            group = (buckets,)
        else:
            group = tuple(buckets)
        if not group:
            raise ValueError("At least one bucket is required.")
        unknown = [bucket for bucket in group if bucket not in self._migrations]
        if unknown:
            raise KeyError(f"Unknown new buckets: {unknown}.")
        # Drop duplicates while preserving the first occurrence.
        return tuple(dict.fromkeys(group))
