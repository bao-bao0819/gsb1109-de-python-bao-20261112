"""
Discrete gear-level storage for continuous intervals.

This module provides a runnable semantic correspondence between the legacy
continuous reading of an interval (any real value within the bounds is
legal) and a discrete reading where only the published gears ("attainable"
sensor levels) can be stored.

Gears may be negative, may contain gaps, and consecutive gears do not need
to be evenly spaced, so a fixed-step discrete domain (see
`AbstractDiscreteInterval`) cannot represent them. Bounds that are not
gears themselves are still handled by magnitude comparison.

This feature is experimental and backward incompatible changes may occur
even in minor or patch updates of portion.
"""

from collections import namedtuple

from .func import singleton
from .interval import Interval

__all__ = ["GearTable", "Mismatch"]


class Mismatch(
    namedtuple("Mismatch", ["operation", "gear", "in_continuous", "in_discrete"])
):
    """
    Report of a discrepancy between a continuous result and a discrete one.

    `operation` is the name of the checked operation, `gear` is the smallest
    attainable gear on which both sides disagree, and `in_continuous` /
    `in_discrete` are the membership verdicts of that gear on each side.
    """

    __slots__ = ()

    def __str__(self):
        return (
            f"{self.operation}: smallest disagreeing gear {self.gear!r} is "
            f"{'in' if self.in_continuous else 'not in'} the continuous result, "
            f"{'in' if self.in_discrete else 'not in'} the discrete result"
        )


class GearTable:
    """
    Published set of attainable gears (the discrete universe).

    A gear table maps continuous intervals to sets of gears and provides
    union, intersection, difference and complement (relative to the whole
    table) on these sets, as well as a comparison of the two semantics.
    """

    def __init__(self, gears):
        """
        Create a gear table from the published gears.

        :param gears: an iterable of distinct, comparable values.
        """
        values = sorted(set(gears))
        if not values:
            raise ValueError("Gear table must contain at least one gear")
        self._gears = tuple(values)
        self._all = frozenset(values)

    @property
    def gears(self):
        """
        Published gears, sorted in increasing order.
        """
        return self._gears

    def __len__(self):
        return len(self._gears)

    def __iter__(self):
        return iter(self._gears)

    def __contains__(self, value):
        return value in self._all

    def __eq__(self, other):
        if isinstance(other, GearTable):
            return self._gears == other._gears
        return NotImplemented

    def __repr__(self):
        return f"GearTable({', '.join(repr(g) for g in self._gears)})"

    def _checked(self, gears):
        gears = frozenset(gears)
        unknown = gears - self._all
        if unknown:
            raise ValueError(f"Values are not published gears: {sorted(unknown)!r}")
        return gears

    def read(self, interval):
        """
        Return the discrete reading of a continuous interval, i.e. the set of
        gears it contains. Bounds that are not gears are still handled by
        magnitude comparison, and open bounds exclude boundary gears.

        :param interval: an Interval instance.
        :return: a frozenset of gears.
        """
        if not isinstance(interval, Interval):
            raise TypeError(f"Unsupported type {type(interval)} for {interval}")
        return frozenset(g for g in self._gears if g in interval)

    def to_interval(self, gears):
        """
        Return the continuous view of a set of gears, as a union of
        singletons.

        :param gears: an iterable of published gears.
        :return: an Interval instance.
        """
        return Interval(*(singleton(g) for g in sorted(self._checked(gears))))

    def union(self, a, b):
        """
        Discrete union of two gear sets.
        """
        return self._checked(a) | self._checked(b)

    def intersection(self, a, b):
        """
        Discrete intersection of two gear sets.
        """
        return self._checked(a) & self._checked(b)

    def difference(self, a, b):
        """
        Discrete difference of two gear sets.
        """
        return self._checked(a) - self._checked(b)

    def complement(self, a):
        """
        Discrete complement of a gear set, relative to the whole table.
        """
        return self._all - self._checked(a)

    def compare(self, continuous, discrete, operation="compare"):
        """
        Compare a continuous result with a discrete (gear set) result.

        Only published gears are examined: isolated points of the continuous
        domain that are not attainable gears are never reported as
        differences.

        :param continuous: an Interval instance.
        :param discrete: an iterable of published gears.
        :param operation: name of the operation being checked.
        :return: a Mismatch for the smallest disagreeing gear, or None if
            both semantics agree on every gear.
        """
        if not isinstance(continuous, Interval):
            raise TypeError(f"Unsupported type {type(continuous)} for {continuous}")
        discrete = self._checked(discrete)
        for gear in self._gears:
            in_continuous = gear in continuous
            in_discrete = gear in discrete
            if in_continuous != in_discrete:
                return Mismatch(operation, gear, in_continuous, in_discrete)
        return None

    def check(self, a, b=None):
        """
        Cross-check the continuous semantics of given interval(s) against the
        discrete gear-set reading for union, intersection, difference (if `b`
        is provided) and complement (relative to the whole table).

        :param a: an Interval instance.
        :param b: an optional Interval instance.
        :return: a dict mapping each operation name to a Mismatch instance,
            or to None if both semantics agree for that operation.
        """
        if not isinstance(a, Interval):
            raise TypeError(f"Unsupported type {type(a)} for {a}")
        d_a = self.read(a)

        results = {}
        if b is not None:
            if not isinstance(b, Interval):
                raise TypeError(f"Unsupported type {type(b)} for {b}")
            d_b = self.read(b)
            results["union"] = self.compare(a | b, self.union(d_a, d_b), "union")
            results["intersection"] = self.compare(
                a & b, self.intersection(d_a, d_b), "intersection"
            )
            results["difference"] = self.compare(
                a - b, self.difference(d_a, d_b), "difference"
            )
        results["complement"] = self.compare(~a, self.complement(d_a), "complement")
        return results
