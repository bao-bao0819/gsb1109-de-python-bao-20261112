"""Exact attainable range of the LRU/LFU fetch-count difference.

The difference is defined as ``d = LRU_misses - LFU_misses`` over a
single completed sequence, so a positive value means LFU saves origin
fetches compared with LRU.

Both extremes are produced on one and the same kind of object -- a
concrete completion of the sampled log -- by keeping, for every joint
cache state (an LRU snapshot paired with an LFU snapshot), the smallest
and largest ``d`` reachable at that state together with the request
sequence that witnesses it.  Missing segments are expanded by an inner
dynamic program over their length and allowed partition set.
"""

from collections import namedtuple

from .model import AccessLog, Gap
from .policies import LFU, LRU, evaluate_sequence

# cell = [min_d, min_witness, max_d, max_witness]
ReachBounds = namedtuple(
    "ReachBounds",
    "min_diff max_diff min_sequence max_sequence min_counts max_counts max_frontier",
)


def _blank_policies(capacity):
    return LRU(capacity), LFU(capacity)


def _access(snapshot_pair, key, capacity):
    """Advance one joint state by a single request."""
    lru_snapshot, lfu_snapshot = snapshot_pair
    lru, lfu = _blank_policies(capacity)
    lru.restore(lru_snapshot)
    lfu.restore(lfu_snapshot)
    lru_miss = lru.access(key)
    lfu_miss = lfu.access(key)
    return (lru.snapshot(), lfu.snapshot()), lru_miss - lfu_miss


def _merge(frontier, state, diff, witness):
    cell = frontier.get(state)
    if cell is None:
        frontier[state] = [diff, witness, diff, witness]
        return
    if diff < cell[0]:
        cell[0] = diff
        cell[1] = witness
    if diff > cell[2]:
        cell[2] = diff
        cell[3] = witness


def _expand_gap(frontier, gap, capacity):
    """Replace the frontier after filling one missing segment."""
    next_frontier = {}
    allowed = sorted(gap.allowed)
    for state, cell in frontier.items():
        # Seed the inner program with both extremes known at this state.
        inner = {state: [cell[0], cell[1], cell[2], cell[3]]}
        for _ in range(gap.length):
            step = {}
            for inner_state, inner_cell in inner.items():
                for index in (0, 2):
                    diff = inner_cell[index]
                    witness = inner_cell[index + 1]
                    for key in allowed:
                        new_state, delta = _access(inner_state, key, capacity)
                        _merge(
                            step,
                            new_state,
                            diff + delta,
                            witness + (key,),
                        )
            inner = step
        for state, inner_cell in inner.items():
            _merge(next_frontier, state, inner_cell[0], inner_cell[1])
            _merge(next_frontier, state, inner_cell[2], inner_cell[3])
    return next_frontier


def reachable_bounds(log, capacity):
    """Compute the attainable range of ``LRU_misses - LFU_misses``.

    Returns a :class:`ReachBounds` whose ``min_sequence`` and
    ``max_sequence`` are concrete completions of ``log`` witnessing the
    two bounds, each validated by replaying both policies from empty.
    """
    if not isinstance(log, AccessLog):
        raise TypeError("log must be an AccessLog")
    if capacity < 1:
        raise ValueError("capacity must be positive")

    empty = ((), ())
    frontier = {empty: [0, (), 0, ()]}
    max_frontier = 1

    for event in log.events:
        if isinstance(event, Gap):
            if event.length == 0:
                continue
            frontier = _expand_gap(frontier, event, capacity)
        else:
            next_frontier = {}
            for state, cell in frontier.items():
                for index in (0, 2):
                    diff = cell[index]
                    witness = cell[index + 1]
                    new_state, delta = _access(state, event, capacity)
                    _merge(
                        next_frontier,
                        new_state,
                        diff + delta,
                        witness + (event,),
                    )
            frontier = next_frontier
        max_frontier = max(max_frontier, len(frontier))

    min_diff = min(cell[0] for cell in frontier.values())
    max_diff = max(cell[2] for cell in frontier.values())
    min_sequence = next(cell[1] for cell in frontier.values() if cell[0] == min_diff)
    max_sequence = next(cell[3] for cell in frontier.values() if cell[2] == max_diff)

    min_counts = evaluate_sequence(min_sequence, capacity)
    max_counts = evaluate_sequence(max_sequence, capacity)
    return ReachBounds(
        min_diff,
        max_diff,
        min_sequence,
        max_sequence,
        min_counts,
        max_counts,
        max_frontier,
    )


def verdict(bounds):
    """Human-readable conclusion from a :class:`ReachBounds`."""
    if bounds.min_diff == bounds.max_diff:
        value = bounds.min_diff
        if value == 0:
            return "tie on every completion"
        return "LFU always saves %d fetch(es)" % value
    if bounds.min_diff >= 0:
        return "LFU never worse; saves up to %d fetch(es)" % bounds.max_diff
    if bounds.max_diff <= 0:
        return "LFU never better; costs up to %d extra fetch(es)" % (-bounds.min_diff)
    return "reversal possible: diff ranges from %d to %d" % (
        bounds.min_diff,
        bounds.max_diff,
    )
