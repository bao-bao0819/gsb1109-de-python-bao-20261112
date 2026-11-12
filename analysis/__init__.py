"""Reachability analysis of LRU versus LFU origin fetches.

A sampled partition access log only fixes part of the request order;
each missing segment (gap) records its length and the set of partitions
that may appear in it.  This package enumerates the possible
completions of such a log and computes the exact attainable range of
the difference in origin fetch (cache miss) counts between LRU and LFU,
with both bounds witnessed on one and the same completed sequence.
"""

from .model import AccessLog, Gap, iter_completions, log_from_pair, make_log
from .policies import LFU, LRU, evaluate_sequence, replay
from .reach import ReachBounds, reachable_bounds
from .bruteforce import brute_force_bounds

__all__ = [
    "AccessLog",
    "Gap",
    "LFU",
    "LRU",
    "ReachBounds",
    "brute_force_bounds",
    "evaluate_sequence",
    "iter_completions",
    "log_from_pair",
    "make_log",
    "reachable_bounds",
    "replay",
]
