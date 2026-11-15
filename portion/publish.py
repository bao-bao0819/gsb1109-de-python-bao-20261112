"""
Publication planning for meter-reading distributions.

A distribution is made of ordered raw buckets. Each bucket holds the set of
source identifiers it collects readings from, together with the scale span it
covers. A distribution can only be published as the totals of adjacent
buckets, and every published range must aggregate at least ``k`` distinct
sources. The same source appearing in several buckets only counts once for
the range it belongs to.

Some boundaries are legal boundaries that no published range is allowed to
cross. The buckets are therefore split into independent legal segments, and
each segment is solved on its own: a segment that cannot satisfy the
distinct-source threshold is suppressed as a whole, with the reasons listed,
instead of borrowing data from the neighbouring segments.

Within every segment, the number of published ranges is maximised. Ties are
broken by minimising the largest scale span found among the merged ranges,
then by choosing the lexicographically earliest sequence of cut points.
"""

from collections import namedtuple

Bucket = namedtuple("Bucket", ["sources", "span"])
"""
A raw bucket, with the iterable of source identifiers it contains and the
scale span it covers (defaults to 1 when a bucket is provided without one).
"""

PublishedRange = namedtuple(
    "PublishedRange", ["start", "stop", "sources", "span", "bucket_count"]
)
"""
A range of adjacent buckets that can be published, from bucket index
``start`` (inclusive) to ``stop`` (exclusive), together with its distinct
sources, its total scale span and the number of buckets it aggregates.
"""

SuppressedRange = namedtuple(
    "SuppressedRange",
    ["start", "stop", "sources", "span", "bucket_count", "reasons"],
)
"""
A legal segment that cannot be published. It is reported on its own, with
the reasons why no valid partition exists, instead of being completed with
data coming from neighbouring segments.
"""

Reason = namedtuple("Reason", ["code", "bucket"])
"""
A suppression reason. ``bucket`` is the index of the offending bucket, or
``None`` for a reason that applies to the whole segment.
"""

Plan = namedtuple("Plan", ["ranges", "suppressed"])
"""
The result of :func:`plan_publication`, with the published ranges and the
suppressed segments, both reported in bucket order.
"""

REASON_SEGMENT_BELOW_K = "segment_distinct_sources_below_k"
REASON_BUCKET_BELOW_K = "bucket_distinct_sources_below_k"


_SegmentPlan = namedtuple("_SegmentPlan", ["count", "max_span", "cuts"])


def _normalize_buckets(buckets):
    normalized = []
    for index, bucket in enumerate(buckets):
        if isinstance(bucket, Bucket):
            span = bucket.span
            sources = frozenset(bucket.sources)
        else:
            span = 1
            sources = frozenset(bucket)
        if span < 0:
            raise ValueError(
                "bucket {} has a negative scale span: {}".format(index, span)
            )
        normalized.append(Bucket(sources, span))
    return normalized


def _collect(buckets, start, stop):
    sources = set()
    span = 0
    for index in range(start, stop):
        sources |= buckets[index].sources
        span += buckets[index].span
    return frozenset(sources), span


def _published(buckets, start, stop):
    sources, span = _collect(buckets, start, stop)
    return PublishedRange(start, stop, sources, span, stop - start)


def _suppressed(buckets, start, stop, k):
    sources, span = _collect(buckets, start, stop)
    reasons = [Reason(REASON_SEGMENT_BELOW_K, None)]
    for index in range(start, stop):
        if len(buckets[index].sources) < k:
            reasons.append(Reason(REASON_BUCKET_BELOW_K, index))
    return SuppressedRange(
        start, stop, sources, span, stop - start, tuple(reasons)
    )


def _plan_segment(buckets, start, stop, k):
    """
    Return the best :class:`_SegmentPlan` for the legal segment
    ``buckets[start:stop]``, or ``None`` when no valid partition exists.

    The selection is lexicographic: maximise the number of ranges, then
    minimise the largest span, then take the earliest cut points. It is
    solved in three exact phases because keeping a single best state per
    prefix would prune cut sequences that only become optimal once later
    ranges absorb their larger span:

    1. a scalar DP finds the minimum possible largest span for every number
       of ranges, so the maximal range count and its span are known;
    2. a boolean feasibility table is restricted to ranges whose span does
       not exceed that optimal largest span;
    3. the earliest feasible cut points are reconstructed greedily, which
       yields the lexicographically earliest cut sequence.
    """
    length = stop - start

    unions = [[None] * (length + 1) for _ in range(length + 1)]
    spans = [[0] * (length + 1) for _ in range(length + 1)]
    for left in range(length):
        sources = set()
        total_span = 0
        for right in range(left + 1, length + 1):
            bucket = buckets[start + right - 1]
            sources |= bucket.sources
            total_span += bucket.span
            unions[left][right] = frozenset(sources)
            spans[left][right] = total_span

    infinity = float("inf")

    # min_span[count][right] is the smallest achievable largest span when
    # covering the first "right" buckets with exactly "count" ranges.
    min_span = [[infinity] * (length + 1) for _ in range(length + 1)]
    min_span[0][0] = 0
    for count in range(1, length + 1):
        for right in range(count, length + 1):
            best = infinity
            for left in range(count - 1, right):
                previous = min_span[count - 1][left]
                if previous == infinity:
                    continue
                if len(unions[left][right]) < k:
                    continue
                best = min(best, max(previous, spans[left][right]))
            min_span[count][right] = best

    max_count = 0
    optimal_span = infinity
    for count in range(length, 0, -1):
        if min_span[count][length] < infinity:
            max_count = count
            optimal_span = min_span[count][length]
            break
    if max_count == 0:
        return None

    # feasible[remaining][i] tells whether buckets[i:length] can be covered
    # by exactly "remaining" valid ranges whose span stays under the cap.
    feasible = [[False] * (length + 1) for _ in range(max_count + 1)]
    feasible[0][length] = True
    for remaining in range(1, max_count + 1):
        for index in range(length - 1, -1, -1):
            for cut in range(index + 1, length + 1):
                if (
                    len(unions[index][cut]) >= k
                    and spans[index][cut] <= optimal_span
                    and feasible[remaining - 1][cut]
                ):
                    feasible[remaining][index] = True
                    break

    cuts = []
    index = 0
    for remaining in range(max_count, 0, -1):
        for cut in range(index + 1, length + 1):
            if (
                len(unions[index][cut]) >= k
                and spans[index][cut] <= optimal_span
                and feasible[remaining - 1][cut]
            ):
                if cut < length:
                    cuts.append(start + cut)
                index = cut
                break
    return _SegmentPlan(max_count, optimal_span, tuple(cuts))


def plan_publication(buckets, k, legal_boundaries=()):
    """
    Plan how a meter-reading distribution can be published.

    :param buckets: an ordered sequence of buckets. Each bucket is either an
        iterable of source identifiers or a :class:`Bucket` whose ``span``
        gives the scale span it covers.
    :param k: the minimum number of distinct sources every published range
        must aggregate. The same source only counts once per range.
    :param legal_boundaries: bucket indices where a published range is
        required to start or end, i.e. boundaries that no range may cross.
    :return: a :class:`Plan` with the publishable ranges and the suppressed
        legal segments, both ordered by bucket index.
    """
    if k < 1:
        raise ValueError("k must be at least 1, got {}".format(k))

    buckets = _normalize_buckets(buckets)
    size = len(buckets)

    boundaries = set(legal_boundaries)
    for boundary in boundaries:
        if not isinstance(boundary, int) or not (1 <= boundary < size):
            raise ValueError(
                "legal boundary must be a bucket index in [1, {}), got {}".format(
                    size, boundary
                )
            )

    cuts = sorted(boundaries | {0, size})
    ranges = []
    suppressed = []
    for start, stop in zip(cuts, cuts[1:]):
        segment = _plan_segment(buckets, start, stop, k)
        if segment is None:
            suppressed.append(_suppressed(buckets, start, stop, k))
            continue
        bounds = (start,) + segment.cuts + (stop,)
        for left, right in zip(bounds, bounds[1:]):
            ranges.append(_published(buckets, left, right))
    return Plan(tuple(ranges), tuple(suppressed))
