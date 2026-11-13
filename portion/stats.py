"""
Order statistics of interval-valued observations.

This module provides tools to reason about the k-th smallest reading of a
sample whose exact readings are unknown. Each observation is described by
an interval (possibly a union of intervals, e.g. "below the detection
limit" or "within one of two ranges") and a positive integer frequency.
The unknown reading of an observation is repeated as a whole by its
frequency: a frequency cannot be split across several values.
"""

from .const import Bound, inf
from .interval import Interval

__all__ = ["KthSmallestCheck", "check_kth_smallest", "kth_smallest_values"]


class KthSmallestCheck:
    """
    Result of :func:`check_kth_smallest` for a candidate value.

    Attributes:
     - `value`: the candidate value;
     - `k`: the considered rank;
     - `reachable`: True if `value` can be the k-th smallest reading;
     - `readings`: if reachable, a tuple with one original reading per
       observation. Repeating each reading by the frequency of its
       observation leads to `value` being the k-th smallest reading;
     - `reason`: if not reachable, an explanation of why every feasible
       combination of readings skips the k-th rank at `value`.
    """

    def __init__(self, *, value, k, reachable, readings=None, reason=None):
        self.value = value
        self.k = k
        self.reachable = reachable
        self.readings = readings
        self.reason = reason

    def __bool__(self):
        return self.reachable

    def __repr__(self):
        if self.reachable:
            return (
                f"KthSmallestCheck(value={self.value!r}, k={self.k}, "
                f"reachable=True, readings={self.readings!r})"
            )
        return (
            f"KthSmallestCheck(value={self.value!r}, k={self.k}, "
            f"reachable=False, reason={self.reason!r})"
        )


def kth_smallest_values(observations, k):
    """
    Return the set of all the possible values of the k-th smallest reading.

    Each observation is a `(interval, frequency)` pair where `interval`
    describes the values the unknown reading can take (open bounds, infinite
    bounds and unions of intervals are supported) and `frequency` is a
    positive integer. The unknown reading of an observation is repeated as
    a whole by its frequency: it cannot be split across several values.

    The k-th smallest reading is 1-based: `k=1` refers to the smallest
    reading, and `k` cannot exceed the sum of the frequencies.

    The returned interval preserves open bounds and internal holes, and is
    an instance of the class of the first observation.

    :param observations: an iterable of `(interval, frequency)` pairs.
    :param k: rank of the considered reading (1-based).
    :return: an interval containing exactly the feasible values.
    """
    observations = _validate(observations, k)
    klass = type(observations[0][0])

    # The feasibility of a value only changes at the bounds of the
    # observations, so it suffices to test each bound and each open
    # interval between two consecutive bounds.
    points = sorted(
        {
            point
            for interval, _ in observations
            for piece in interval
            for point in (piece.lower, piece.upper)
            if point != -inf and point != inf
        }
    )

    values = klass()
    lower = -inf
    for upper in points + [inf]:
        if _feasible(_classify_between(observations, lower, upper), k):
            values = values | klass.from_atomic(Bound.OPEN, lower, upper, Bound.OPEN)
        if upper != inf and _feasible(_classify_at(observations, upper), k):
            values = values | klass.from_atomic(
                Bound.CLOSED, upper, upper, Bound.CLOSED
            )
        lower = upper
    return values


def check_kth_smallest(observations, k, value):
    """
    Check whether `value` can be the k-th smallest reading.

    Observations and `k` follow the same conventions as in
    :func:`kth_smallest_values`.

    If `value` is reachable, the returned result holds one original reading
    per observation in `readings` (each reading belonging to its observation
    interval) such that repeating these readings by their frequencies makes
    `value` the k-th smallest reading. Otherwise, `reason` explains why
    every feasible combination of readings skips the k-th rank at `value`.

    :param observations: an iterable of `(interval, frequency)` pairs.
    :param k: rank of the considered reading (1-based).
    :param value: the candidate value for the k-th smallest reading.
    :return: a :class:`KthSmallestCheck` instance.
    """
    observations = _validate(observations, k)
    n = sum(frequency for _, frequency in observations)
    classification = _classify_at(observations, value)
    lo, hi = _rank_window(classification, k)

    frequencies = [frequency for _, frequency in classification.straddlers]
    hi = min(hi, sum(frequencies))
    sums, parents = _subset_sums(frequencies, max(hi, 0))
    target = next((s for s in range(max(lo, 0), hi + 1) if s in sums), None)

    if target is None:
        reason = _unreachable_reason(n, k, value, classification)
        return KthSmallestCheck(value=value, k=k, reachable=False, reason=reason)

    # Recover which straddling observations are assigned below value.
    below = set()
    s = target
    while s > 0:
        s, position = parents[s]
        below.add(classification.straddlers[position][0])

    klass = type(observations[0][0])
    below_domain = klass.from_atomic(Bound.OPEN, -inf, value, Bound.OPEN)
    above_domain = klass.from_atomic(Bound.OPEN, value, inf, Bound.OPEN)

    readings = []
    for index, (interval, _) in enumerate(observations):
        if value in interval:
            readings.append(value)
        elif _forced_below(interval, value) or index in below:
            readings.append(_pick(interval & below_domain))
        else:
            readings.append(_pick(interval & above_domain))

    return KthSmallestCheck(
        value=value, k=k, reachable=True, readings=tuple(readings)
    )


def _validate(observations, k):
    """
    Turn `observations` into a list of `(interval, frequency)` pairs, and
    check that intervals are non-empty, frequencies are positive integers,
    and `k` is a valid rank.
    """
    pairs = []
    for item in observations:
        if isinstance(item, Interval):
            raise TypeError("observations must be (interval, frequency) pairs")
        try:
            interval, frequency = item
        except (TypeError, ValueError):
            raise TypeError(
                "observations must be (interval, frequency) pairs"
            ) from None
        if not isinstance(interval, Interval):
            raise TypeError(f"expected an Interval, got {interval!r}")
        if interval.empty:
            raise ValueError("observation intervals must not be empty")
        if (
            not isinstance(frequency, int)
            or isinstance(frequency, bool)
            or frequency < 1
        ):
            raise ValueError("frequencies must be positive integers")
        pairs.append((interval, frequency))

    if not pairs:
        raise ValueError("at least one observation is required")

    n = sum(frequency for _, frequency in pairs)
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n:
        raise ValueError(f"k must be an integer between 1 and {n}")

    return pairs


def _forced_below(interval, value):
    """
    True if all the values of `interval` are strictly smaller than `value`.
    """
    return interval.upper < value or (
        interval.upper == value and interval.right is Bound.OPEN
    )


def _forced_above(interval, value):
    """
    True if all the values of `interval` are strictly greater than `value`.
    """
    return interval.lower > value or (
        interval.lower == value and interval.left is Bound.OPEN
    )


def _classify_at(observations, value):
    """
    Classify the observations with respect to `value`: observations that
    necessarily lie below (resp. above) `value`, observations that can be
    equal to `value`, and the remaining ("straddling") observations, returned
    as (index, frequency) pairs.
    """
    low = equal = high = 0
    straddlers = []
    for index, (interval, frequency) in enumerate(observations):
        if value in interval:
            equal += frequency
        elif _forced_below(interval, value):
            low += frequency
        elif _forced_above(interval, value):
            high += frequency
        else:
            straddlers.append((index, frequency))
    return _Classification(low, equal, high, straddlers)


def _classify_between(observations, lower, upper):
    """
    Same as `_classify_at`, but for any value strictly between `lower` and
    `upper`, where `lower` and `upper` are two consecutive bounds of the
    observations (so the classification does not depend on the exact value).
    """
    low = equal = high = 0
    straddlers = []
    for index, (interval, frequency) in enumerate(observations):
        if any(piece.lower <= lower and piece.upper >= upper for piece in interval):
            equal += frequency
        elif interval.upper <= lower:
            low += frequency
        elif interval.lower >= upper:
            high += frequency
        else:
            straddlers.append((index, frequency))
    return _Classification(low, equal, high, straddlers)


class _Classification:
    __slots__ = ("low", "equal", "high", "straddlers")

    def __init__(self, low, equal, high, straddlers):
        self.low = low
        self.equal = equal
        self.high = high
        self.straddlers = straddlers


def _rank_window(classification, k):
    """
    Return the `[lo, hi]` window for the number of straddling copies that
    must be assigned below the candidate value for the k-th smallest reading
    to reach this value. An empty window means the value is not reachable.
    """
    lo = k - classification.low - classification.equal
    hi = k - 1 - classification.low
    return lo, hi


def _feasible(classification, k):
    """
    True if the k-th smallest reading can take a value with the given
    classification of observations.
    """
    lo, hi = _rank_window(classification, k)
    hi = min(hi, sum(frequency for _, frequency in classification.straddlers))
    lo = max(lo, 0)
    if lo > hi:
        return False
    sums, _ = _subset_sums(
        [frequency for _, frequency in classification.straddlers], hi
    )
    return any(lo <= s for s in sums)


def _subset_sums(frequencies, cap):
    """
    Return the set of achievable sums of subsets of `frequencies` (capped at
    `cap`), together with a dict mapping each sum to the (previous sum,
    position) pair that produced it, so a subset can be recovered.
    """
    sums = {0}
    parents = {}
    for position, frequency in enumerate(frequencies):
        for s in list(sums):
            ns = s + frequency
            if ns <= cap and ns not in sums:
                sums.add(ns)
                parents[ns] = (s, position)
    return sums, parents


def _pick(interval):
    """
    Pick a value within a non-empty interval.
    """
    piece = next(iter(interval))
    if piece.left is Bound.CLOSED:
        return piece.lower
    if piece.right is Bound.CLOSED:
        return piece.upper
    if piece.lower != -inf and piece.upper != inf:
        return (piece.lower + piece.upper) / 2
    if piece.lower != -inf:
        return piece.lower + 1
    if piece.upper != inf:
        return piece.upper - 1
    return 0


def _unreachable_reason(n, k, value, classification):
    """
    Explain why the k-th smallest reading cannot be equal to `value`.
    """
    if classification.equal == 0:
        return (
            f"{value!r} is not a possible reading of any observation, so "
            f"every feasible combination of readings skips rank {k} at {value!r}."
        )
    if classification.low > k - 1:
        return (
            f"at least {classification.low} reading(s) are necessarily smaller "
            f"than {value!r}, so the {k}-th smallest reading is necessarily "
            f"smaller than {value!r}."
        )
    if classification.high > n - k:
        return (
            f"at least {classification.high} reading(s) are necessarily greater "
            f"than {value!r}, so the {k}-th smallest reading is necessarily "
            f"greater than {value!r}."
        )
    frequencies = [frequency for _, frequency in classification.straddlers]
    sums, _ = _subset_sums(frequencies, sum(frequencies))
    achievable = sorted(classification.low + s for s in sums)
    return (
        f"the frequencies {sorted(frequencies)} of the observations straddling "
        f"{value!r} cannot be split between values below and above {value!r}: "
        f"the number of readings below {value!r} can only be {achievable}, "
        f"while rank {k} requires it to be between {k - classification.equal} "
        f"and {k - 1}, so every feasible combination of readings skips rank {k}."
    )
