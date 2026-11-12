"""Exhaustive reference implementation for small samples.

This enumerates every completion of a log and evaluates both policies
directly.  It is only practical when the number of completions is small
(see :meth:`AccessLog.completion_count`) and serves as an independent
check of :func:`analysis.reach.reachable_bounds`.
"""

from collections import namedtuple

from .model import AccessLog
from .policies import evaluate_sequence

BruteResult = namedtuple(
    "BruteResult",
    "min_diff max_diff min_sequence max_sequence min_counts max_counts completions",
)


def brute_force_bounds(log, capacity):
    """Enumerate all completions and return the exact diff range."""
    if not isinstance(log, AccessLog):
        raise TypeError("log must be an AccessLog")
    if capacity < 1:
        raise ValueError("capacity must be positive")

    best_min = None
    best_max = None
    min_sequence = None
    max_sequence = None
    min_counts = None
    max_counts = None
    completions = 0

    for sequence in log.completions():
        completions += 1
        lru_misses, lfu_misses = evaluate_sequence(sequence, capacity)
        diff = lru_misses - lfu_misses
        if best_min is None or diff < best_min:
            best_min = diff
            min_sequence = sequence
            min_counts = (lru_misses, lfu_misses)
        if best_max is None or diff > best_max:
            best_max = diff
            max_sequence = sequence
            max_counts = (lru_misses, lfu_misses)

    return BruteResult(
        best_min,
        best_max,
        min_sequence,
        max_sequence,
        min_counts,
        max_counts,
        completions,
    )
