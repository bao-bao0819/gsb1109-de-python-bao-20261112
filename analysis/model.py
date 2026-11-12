"""Sampled access log with missing segments (gaps).

A log keeps the observed request order and, wherever the sample is
missing, a :class:`Gap` recording how many requests were lost and the
set of partitions those requests are allowed to reference.  A
*completion* is any concrete request sequence obtained by filling each
gap with ``length`` keys drawn (with repetition) from its allowed set.
"""

from collections import namedtuple
from itertools import product

# A fixed observed request is a bare partition key; a Gap marks a run of
# unobserved requests.  Both appear as elements of ``AccessLog.events``.
Gap = namedtuple("Gap", "length allowed")


class AccessLog:
    """Observed requests interleaved with missing segments."""

    def __init__(self, events=()):
        self.events = []
        for event in events:
            if isinstance(event, Gap):
                if event.length < 0:
                    raise ValueError("gap length must be non-negative")
                if not event.allowed and event.length:
                    raise ValueError("non-empty gap needs an allowed set")
                self.events.append(Gap(event.length, frozenset(event.allowed)))
            else:
                self.events.append(event)

    def add_request(self, key):
        self.events.append(key)
        return self

    def add_gap(self, length, allowed):
        self.events.append(Gap(length, frozenset(allowed)))
        return self

    def gap_count(self):
        return sum(isinstance(event, Gap) for event in self.events)

    def completion_count(self):
        """Number of distinct completions (product of gap choices)."""
        total = 1
        for event in self.events:
            if isinstance(event, Gap):
                total *= len(event.allowed) ** event.length
        return total

    def completions(self):
        """Yield every concrete request sequence that completes the log."""
        choices_per_event = []
        for event in self.events:
            if isinstance(event, Gap):
                fills = [
                    tuple(fill)
                    for fill in product(sorted(event.allowed), repeat=event.length)
                ]
                choices_per_event.append(fills)
            else:
                choices_per_event.append([(event,)])
        for parts in product(*choices_per_event):
            sequence = []
            for part in parts:
                sequence.extend(part)
            yield tuple(sequence)

    def __repr__(self):
        return "AccessLog(%r)" % (self.events,)


def iter_completions(log):
    """Convenience wrapper around :meth:`AccessLog.completions`."""
    return log.completions()


def make_log(fixed_requests, gaps):
    """Build a log from fixed requests and ``(position, length, allowed)``.

    ``gaps`` maps the index *before which* a missing segment sits in the
    list of fixed requests, mirroring how a sampler reports a hole.
    """
    events = []
    gaps = dict(gaps)
    for position, key in enumerate(fixed_requests):
        if position in gaps:
            length, allowed = gaps[position]
            events.append(Gap(length, allowed))
        events.append(key)
    if len(fixed_requests) in gaps:
        length, allowed = gaps[len(fixed_requests)]
        events.append(Gap(length, allowed))
    return AccessLog(events)


def log_from_pair(observed, gap_specs):
    """Build a log from an observed sequence and trailing/inline gaps.

    Each spec is ``(after_index, length, allowed)``; ``after_index`` is
    the index of the observed request after which the gap appears, with
    ``-1`` placing it before the first request.
    """
    events = []
    by_after = {}
    for after_index, length, allowed in gap_specs:
        by_after.setdefault(after_index, []).append(Gap(length, allowed))
    for index, key in enumerate(observed):
        events.extend(by_after.get(index - 1, ()))
        events.append(key)
    events.extend(by_after.get(len(observed) - 1, ()))
    return AccessLog(events)
