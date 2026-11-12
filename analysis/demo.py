"""Runnable comparison on small samples, including a reversal case.

Run with::

    python -m analysis.demo

It prints the attainable range of ``LRU_misses - LFU_misses`` for a
sampled log whose single missing segment can make either policy win,
and shows the two concrete completions that prove the opposite
conclusions, replayed request by request.
"""

from .model import AccessLog, Gap
from .policies import LFU, LRU
from .reach import reachable_bounds, verdict


def trace(sequence, capacity):
    """Replay one sequence on both policies, returning per-step rows."""
    lru = LRU(capacity)
    lfu = LFU(capacity)
    rows = []
    for key in sequence:
        lru_miss = lru.access(key)
        lfu_miss = lfu.access(key)
        rows.append(
            {
                "key": key,
                "lru_miss": lru_miss,
                "lfu_miss": lfu_miss,
                "lru_contents": lru.contents(),
                "lfu_contents": lfu.contents(),
            }
        )
    return rows


def _format_row(row):
    def mark(miss):
        return "MISS" if miss else "hit "

    return "  req {key!r:>4} | LRU {lru} {lru_c} | LFU {lfu} {lfu_c}".format(
        key=row["key"],
        lru=mark(row["lru_miss"]),
        lru_c=row["lru_contents"],
        lfu=mark(row["lfu_miss"]),
        lfu_c=row["lfu_contents"],
    )


def reversal_log():
    """Sampled log with one gap that can make either policy look better."""
    # Observed prefix leaves both caches holding {2, 3}; the missing
    # request may be partition 1 or partition 2.
    return AccessLog([1, 1, 2, 3, Gap(1, (1, 2))])


def describe_reversal(capacity=2):
    log = reversal_log()
    bounds = reachable_bounds(log, capacity)
    lines = []
    lines.append("sampled log: %s" % (log.events,))
    lines.append("cache capacity: %d" % capacity)
    lines.append("completions consistent with the sample: %d" % log.completion_count())
    lines.append(
        "attainable LRU_misses - LFU_misses: [%d, %d]"
        % (bounds.min_diff, bounds.max_diff)
    )
    lines.append("conclusion: %s" % verdict(bounds))
    lines.append("")

    lines.append(
        "Replay A (gap -> %r), LFU saves a fetch (diff = %d):"
        % (bounds.max_sequence[-1], bounds.max_diff)
    )
    lines.append(
        "  full sequence: %s  LRU=%d LFU=%d"
        % (
            bounds.max_sequence,
            bounds.max_counts[0],
            bounds.max_counts[1],
        )
    )
    lines.extend(_format_row(row) for row in trace(bounds.max_sequence, capacity))
    lines.append("")

    lines.append(
        "Replay B (gap -> %r), LFU costs an extra fetch (diff = %d):"
        % (bounds.min_sequence[-1], bounds.min_diff)
    )
    lines.append(
        "  full sequence: %s  LRU=%d LFU=%d"
        % (
            bounds.min_sequence,
            bounds.min_counts[0],
            bounds.min_counts[1],
        )
    )
    lines.extend(_format_row(row) for row in trace(bounds.min_sequence, capacity))
    return "\n".join(lines)


def main():
    print(describe_reversal())


if __name__ == "__main__":
    main()
