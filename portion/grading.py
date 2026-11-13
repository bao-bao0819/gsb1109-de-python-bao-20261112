"""
Compile interval-based grading rules into a queryable, order-independent map.

A grading rule associates an interval with a grade for a given source. Rule
intervals may use arbitrary open/closed endpoints and may be reduced to a
singleton to override a broader rule on a single value.

Specificity is defined by strict containment only: a rule is more specific
than another one when its interval is strictly contained in the other's.
Rules whose intervals overlap without either one containing the other have no
precedence over each other and produce a conflict.

The compilation is purely functional: the resulting :class:`GradingMap` only
depends on the set of rules, not on the order they were provided. Hence a
sample located on a boundary can never change grade because the input order
changed.
"""

from dataclasses import dataclass, field
from typing import Any

from .const import Bound, inf
from .dict import IntervalDict
from .interval import Interval

UNIQUE = "unique"
CONFLICT = "conflict"
GAP = "gap"

# A boundary key is a (value, kind) pair where kind 0 denotes the point itself
# (a closed boundary) and kind 1 denotes the open region right after the value
# (an open boundary). Ordering is lexicographic: P(v) < O(v) < P(w) for v < w.
_POINT = 0
_OPEN = 1


class GradingConflict(LookupError):
    """Raised when querying a grade on a segment where rules conflict."""


@dataclass(frozen=True)
class GradeRule:
    """
    A single grading rule.

    :param interval: the interval covered by the rule.
    :param grade: the grade attributed to samples in the interval.
    :param source: identifier of the source that issued the rule (optional).
    """

    interval: Interval
    grade: Any
    source: Any = None

    def __post_init__(self):
        if not isinstance(self.interval, Interval):
            raise TypeError(
                f"interval must be an Interval instance, got {self.interval!r}"
            )
        if self.interval.empty:
            raise ValueError(f"Cannot build a rule on an empty interval: {self!r}")

    def __repr__(self):
        return f"GradeRule(source={self.source!r}, interval={self.interval!r}, grade={self.grade!r})"


@dataclass(frozen=True)
class Segment:
    """
    A maximal region on which the set of applicable rules is constant.

    :param interval: the region covered by the segment.
    :param status: one of UNIQUE, CONFLICT or GAP.
    :param grade: the unique grade, or None when the segment is a conflict or
        a gap.
    :param basis: the rules that support the grade (all the active rules whose
        grade matches the unique one). Empty for conflicts and gaps.
    :param reasons: all the rules that are active on the segment. Empty for
        gaps.
    """

    interval: Interval
    status: str
    grade: Any = None
    basis: tuple = field(default_factory=tuple)
    reasons: tuple = field(default_factory=tuple)

    @property
    def is_unique(self):
        return self.status == UNIQUE

    @property
    def is_conflict(self):
        return self.status == CONFLICT

    @property
    def is_gap(self):
        return self.status == GAP


def _as_rule(item, index):
    if isinstance(item, GradeRule):
        return item
    if isinstance(item, tuple) and len(item) in (2, 3):
        if len(item) == 2:
            interval, grade = item
            source = None
        else:
            interval, grade, source = item
        return GradeRule(interval=interval, grade=grade, source=source)
    raise TypeError(f"Unsupported rule at position {index}: {item!r}")


def _strictly_contains(outer, inner):
    return inner in outer and inner != outer


def _decide(active):
    """
    Return (grade, basis) for a constant set of active rules, or None when the
    active rules conflict. An empty set is handled by the caller as a gap.

    Grades are only compared for equality, so they do not need to be hashable.
    """
    grade = active[0].grade
    if all(rule.grade == grade for rule in active):
        # Every active rule agrees: same-level sources keep a shared basis.
        return grade, tuple(active)

    # Grades differ: only strict containment can grant precedence. The minimal
    # intervals (the ones no other distinct interval strictly contains) are the
    # most specific rules and they must all agree to resolve the disagreement.
    distinct_intervals = {rule.interval for rule in active}
    minimal = [
        interval
        for interval in distinct_intervals
        if not any(_strictly_contains(interval, other) for other in distinct_intervals)
    ]

    covering_rules = [rule for rule in active if rule.interval in minimal]
    specific_grade = covering_rules[0].grade
    if any(rule.grade != specific_grade for rule in covering_rules[1:]):
        # Incomparable or co-covering most specific rules disagree.
        return None

    basis = tuple(rule for rule in active if rule.grade == specific_grade)
    return specific_grade, basis


def _segment_interval(klass, start_key, next_key):
    """Rebuild the geometric interval between two consecutive boundary keys."""
    if start_key is None:
        lower, left = -inf, Bound.OPEN
    else:
        value, kind = start_key
        lower = value
        left = Bound.CLOSED if kind == _POINT else Bound.OPEN

    if next_key is None:
        upper, right = inf, Bound.OPEN
    else:
        value, kind = next_key
        upper = value
        right = Bound.OPEN if kind == _POINT else Bound.CLOSED

    return klass.from_atomic(left, lower, upper, right)


class _OrderedRule:
    """Internal carrier pairing a rule with a display-stable order index."""

    __slots__ = ("rule", "order")

    def __init__(self, rule, order):
        self.rule = rule
        self.order = order


def _build_segments(ordered_rules, klass):
    # Boundary key -> list of (activate?, ordered rule). Activation and
    # deactivation at the very same key are applied together, modelling the
    # exact point.
    events = {}
    initially_active = set()

    for carrier in ordered_rules:
        for atomic in carrier.rule.interval:
            if atomic.lower == -inf:
                initially_active.add(carrier)
            else:
                start_kind = _POINT if atomic.left == Bound.CLOSED else _OPEN
                events.setdefault((atomic.lower, start_kind), []).append(
                    (True, carrier)
                )

            if atomic.upper != inf:
                end_kind = _OPEN if atomic.right == Bound.CLOSED else _POINT
                events.setdefault((atomic.upper, end_kind), []).append((False, carrier))

    ordered_keys = sorted(events)
    active = set(initially_active)
    segments = []
    run_start = None

    def emit(start_key, next_key, active_carriers):
        interval = _segment_interval(klass, start_key, next_key)
        if interval.empty:
            return
        carriers = tuple(sorted(active_carriers, key=lambda carrier: carrier.order))
        active_rules = tuple(carrier.rule for carrier in carriers)
        if not active_rules:
            segments.append(Segment(interval, GAP))
            return
        decision = _decide(active_rules)
        if decision is None:
            segments.append(Segment(interval, CONFLICT, None, tuple(), active_rules))
        else:
            grade, basis = decision
            segments.append(Segment(interval, UNIQUE, grade, basis, active_rules))

    for key in ordered_keys:
        next_active = set(active)
        for activated, carrier in events[key]:
            if activated:
                next_active.add(carrier)
            else:
                next_active.discard(carrier)

        if next_active != active:
            emit(run_start, key, active)
            run_start = key
            active = next_active

    # Trailing run, possibly an uncovered gap extending to +inf.
    emit(run_start, None, active)

    if not segments:
        # No boundary at all: the whole line is a gap.
        segments.append(
            Segment(klass.from_atomic(Bound.OPEN, -inf, inf, Bound.OPEN), GAP)
        )

    return segments


class GradingMap:
    """
    The result of compiling grading rules: an ordered, exhaustive partition of
    the considered domain into uniquely graded, conflicting or uncovered
    segments.

    Instances are not meant to be created directly, use :func:`compile_grading`.
    """

    __slots__ = ("_klass", "_segments", "_upper_keys", "_rules")

    def __init__(self, segments, rules, klass):
        self._klass = klass
        self._segments = tuple(segments)
        self._rules = tuple(rules)
        self._upper_keys = [
            (segment.interval.upper, segment.interval.right)
            for segment in self._segments
        ]

    @property
    def rules(self):
        """All the rules that were compiled, deduplicated and ordered."""
        return self._rules

    def segments(self):
        """Return the exhaustive, ordered list of segments."""
        return self._segments

    def classify(self, value):
        """
        Return the :class:`Segment` that contains the given sample value.

        :raises KeyError: if the value lies outside the compiled domain.
        """
        index = self._index_of(value)
        if index is None:
            raise KeyError(value)
        return self._segments[index]

    def grade_for(self, value):
        """
        Return the unique grade of the segment containing the given value.

        :raises GradingConflict: if the value falls in a conflict segment.
        :raises KeyError: if the value is in a gap or outside the domain.
        """
        segment = self.classify(value)
        if segment.is_conflict:
            raise GradingConflict(
                f"Conflicting rules at {value!r}: {segment.reasons!r}"
            )
        if segment.is_gap:
            raise KeyError(value)
        return segment.grade

    def covered(self):
        """Return the interval union of all the non-gap segments."""
        return self._klass(
            *(segment.interval for segment in self._segments if not segment.is_gap)
        )

    def gaps(self):
        """Return the interval union of the uncovered segments."""
        return self._klass(
            *(segment.interval for segment in self._segments if segment.is_gap)
        )

    def conflicts(self):
        """Return the interval union of the conflict segments."""
        return self._klass(
            *(segment.interval for segment in self._segments if segment.is_conflict)
        )

    def unique_intervals(self):
        """Return an :class:`IntervalDict` mapping uniquely graded regions to grades."""
        dict_klass = type(
            "GradingIntervalDict", (IntervalDict,), {"_klass": self._klass}
        )
        mapping = dict_klass()
        for segment in self._segments:
            if segment.is_unique:
                mapping[segment.interval] = segment.grade
        return mapping

    def _index_of(self, value):
        # Segments are disjoint and ordered; locate a candidate by upper bound.
        boundaries = self._upper_keys
        low, high = 0, len(boundaries) - 1
        while low <= high:
            middle = (low + high) // 2
            upper, right = boundaries[middle]
            if value > upper or (value == upper and right == Bound.OPEN):
                low = middle + 1
            else:
                high = middle - 1
        candidate = low
        if (
            candidate < len(self._segments)
            and value in self._segments[candidate].interval
        ):
            return candidate
        return None

    def __contains__(self, value):
        index = self._index_of(value)
        return index is not None and not self._segments[index].is_gap

    def __iter__(self):
        return iter(self._segments)

    def __len__(self):
        return len(self._segments)

    def __eq__(self, other):
        if isinstance(other, GradingMap):
            return self._segments == other._segments
        return NotImplemented

    def __repr__(self):
        body = ", ".join(
            f"{segment.interval!r}: {segment.status}"
            + (f"({segment.grade!r})" if segment.is_unique else "")
            for segment in self._segments
        )
        return f"GradingMap({{{body}}})"


def compile_grading(rules, *, domain=None, klass=Interval):
    """
    Compile grading rules into an order-independent :class:`GradingMap`.

    :param rules: an iterable of :class:`GradeRule` instances or of
        ``(interval, grade)`` / ``(interval, grade, source)`` tuples.
    :param domain: an optional Interval restricting the reported segments.
        Regions outside the domain are ignored rather than reported as gaps.
    :param klass: the Interval class to use (defaults to :class:`Interval`).
    :return: a compiled :class:`GradingMap`.
    """
    unique_rules = []
    seen = []
    for position, item in enumerate(rules):
        rule = _as_rule(item, position)
        if not any(
            existing.source == rule.source
            and existing.interval == rule.interval
            and existing.grade == rule.grade
            for existing in seen
        ):
            seen.append(rule)
            unique_rules.append(rule)

    # A canonical, content-based order keeps basis/reasons display identical
    # whatever the input order is. Every decision itself is purely set-based
    # and relies on interval geometry and grade equality only.
    carriers = tuple(
        _OrderedRule(rule, order)
        for order, rule in enumerate(
            sorted(
                unique_rules,
                key=lambda rule: (
                    str(rule.source),
                    repr(rule.interval),
                    repr(rule.grade),
                ),
            )
        )
    )
    ordered_rules = tuple(carrier.rule for carrier in carriers)

    raw_segments = _build_segments(carriers, klass)

    if domain is None:
        segments = raw_segments
    else:
        segments = []
        for segment in raw_segments:
            clipped = segment.interval & domain
            for atomic in clipped:
                segments.append(
                    Segment(
                        atomic,
                        segment.status,
                        segment.grade,
                        segment.basis,
                        segment.reasons,
                    )
                )

    return GradingMap(segments, ordered_rules, klass)
