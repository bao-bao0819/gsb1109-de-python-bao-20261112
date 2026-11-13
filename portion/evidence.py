"""
Source-attributed calibration tables.

A :class:`ScaleTable` keeps every calibration contribution attached to its
originating *source* (e.g. a sensor). When several sources report the same
corrected value on the same scale segment, their contributions are preserved as
distinct pieces of evidence instead of being merged into one anonymous value.
This makes it possible to:

* keep the source contribution inside the calibration lookup table;
* retract a single segment (or the whole contribution) of a single source;
* explain how the *effective* calibration changes when evidence is added or
  withdrawn.

Reconciliation outputs distinguish three situations required by the domain:

* ``VALUE_CHANGED``      -- "最终值变了": the unique effective value changes;
* ``EVIDENCE_LOST``      -- "值没变但证据少了": the value stays the same but one
  of its supporting sources has been withdrawn;
* ``CONFLICT_RESOLVED``  -- "从冲突恢复成唯一值": disagreeing evidence used to
  make the value undefined, and the remaining evidence again agrees on one
  unique value.

Two adjacent segments carrying the same continuous value but backed by
different evidence stay distinguishable: they produce distinct
:class:`EffectiveValue` entries and can be queried independently.
"""

from collections import namedtuple

from .const import Bound
from .dict import IntervalDict
from .interval import Interval


# Reconciliation kinds.
NEW_COVERAGE = "new_coverage"
COVERAGE_LOST = "coverage_lost"
VALUE_CHANGED = "value_changed"
EVIDENCE_GAINED = "evidence_gained"
EVIDENCE_LOST = "evidence_lost"
CONFLICT_EMERGED = "conflict_emerged"
CONFLICT_RESOLVED = "conflict_resolved"


class EffectiveValue(namedtuple("EffectiveValue", ["value", "sources", "conflicting", "values"])):
    """
    The effective calibration on a scale segment.

    :param value: the unique corrected value when ``conflicting`` is False,
        otherwise None (no value is defined while evidence disagrees).
    :param sources: ordered tuple of the sources that actively support this
        segment (all of them when conflicting).
    :param conflicting: True when the active evidence disagrees.
    :param values: ordered tuple of the distinct values reported by the active
        evidence; it reduces to ``(value,)`` when there is no conflict.
    """

    __slots__ = ()

    @property
    def defined(self):
        """True when a unique effective value is defined."""
        return not self.conflicting


class Change(namedtuple("Change", ["interval", "kind", "before", "after"])):
    """
    A single reconciliation change.

    :param interval: the scale segment affected by the change.
    :param kind: one of the module-level kind constants.
    :param before: the :class:`EffectiveValue` before the operation, or None
        when the segment was not covered.
    :param after: the :class:`EffectiveValue` after the operation, or None
        when the segment is no longer covered.
    """

    __slots__ = ()


class Reconciliation(namedtuple("Reconciliation", ["changes"])):
    """
    The outcome of adding or retracting evidence.

    It is an iterable of :class:`Change` instances, grouped into maximal
    segments sharing the same kind and the same before/after effective values.
    Convenience selectors are provided for the three domain-level situations
    (value changed, evidence lost, conflict resolved) and their symmetric
    counterparts.
    """

    __slots__ = ()

    def _of(self, kind):
        return tuple(change for change in self.changes if change.kind == kind)

    @property
    def value_changed(self):
        """Segments whose unique effective value changed."""
        return self._of(VALUE_CHANGED)

    @property
    def evidence_lost(self):
        """Segments that kept their value but lost at least one source."""
        return self._of(EVIDENCE_LOST)

    @property
    def evidence_gained(self):
        """Segments that kept their value while gaining a source."""
        return self._of(EVIDENCE_GAINED)

    @property
    def conflict_resolved(self):
        """Segments moving from disagreeing evidence to a unique value."""
        return self._of(CONFLICT_RESOLVED)

    @property
    def conflict_emerged(self):
        """Segments moving from a unique value to disagreeing evidence."""
        return self._of(CONFLICT_EMERGED)

    @property
    def new_coverage(self):
        """Previously uncovered segments now covered by a unique value."""
        return self._of(NEW_COVERAGE)

    @property
    def coverage_lost(self):
        """Segments no longer covered by any evidence."""
        return self._of(COVERAGE_LOST)

    def domains(self):
        """Return the union of every changed segment as an Interval."""
        return Interval(*(change.interval for change in self.changes))

    def __iter__(self):
        return iter(self.changes)

    def __len__(self):
        return len(self.changes)

    def __bool__(self):
        return bool(self.changes)


def _as_interval(interval):
    if isinstance(interval, Interval):
        return interval
    return Interval.from_atomic(Bound.CLOSED, interval, interval, Bound.CLOSED)


def _partition(atomic_intervals):
    """
    Split a collection of atomic intervals into a disjoint partition whose
    boundaries align with every input interval.
    """
    parts = []
    for piece in atomic_intervals:
        overlap_total = Interval()
        updated = []
        for part in parts:
            shared = part & piece
            if shared.empty:
                updated.append(part)
            else:
                overlap_total = overlap_total | shared
                updated.extend(iter(part - piece))
                updated.extend(iter(shared))
        updated.extend(iter(piece - overlap_total))
        parts = updated
    parts.sort(key=lambda part: (part.lower, part.upper))
    return parts


class ScaleTable:
    """
    A calibration lookup table that retains source contributions.

    Each source owns an :class:`~portion.IntervalDict` mapping the scale
    segments it calibrated to the corrected values it reported. A source can
    therefore contribute several disjoint segments, and the same physical value
    reported by two sources remains two independently retractable pieces of
    evidence.

    The *effective* calibration on a segment is derived from all active
    evidence: agreeing evidence defines a unique value together with the list
    of supporting sources; disagreeing evidence leaves the value undefined and
    exposes the conflict instead of silently letting one source win.
    """

    __slots__ = ("_sources", "_revision")

    def __init__(self):
        # Insertion-ordered mapping: source -> IntervalDict(value coverage).
        self._sources = {}
        self._revision = 0

    @property
    def revision(self):
        """Monotonic counter incremented on every effective modification."""
        return self._revision

    def copy(self):
        """Return an independent deep-enough copy of this table."""
        clone = ScaleTable()
        for source, data in self._sources.items():
            clone._sources[source] = data.copy()
        clone._revision = self._revision
        return clone

    def sources(self, interval=None):
        """
        Return the ordered tuple of sources with active evidence.

        :param interval: optional scale segment; when provided, only sources
            covering (part of) it are returned.
        """
        if interval is None:
            return tuple(self._sources)
        interval = _as_interval(interval)
        return tuple(
            source
            for source, data in self._sources.items()
            if data[interval].domain()
        )

    def coverage(self, source=None):
        """
        Return the scale interval covered by evidence.

        :param source: optional source; when provided, only that source's
            coverage is returned (raises KeyError if unknown).
        """
        if source is None:
            domain = Interval()
            for data in self._sources.values():
                domain = domain | data.domain()
            return domain
        return self._source_data(source).domain()

    def _source_data(self, source):
        try:
            return self._sources[source]
        except KeyError:
            raise KeyError(f"Unknown source {source!r}") from None

    def contribute(self, source, interval, value):
        """
        Add (or overwrite) one source's calibration on a scale segment.

        Overwriting only affects this source's own evidence on the segment;
        evidence from other sources is untouched.

        :param source: identifier of the contributing source.
        :param interval: the calibrated scale segment (a single value is
            interpreted as a singleton).
        :param value: the corrected value reported by the source.
        :return: a :class:`Reconciliation` describing the effective changes.
        """
        interval = _as_interval(interval)
        before = self._view()

        if not interval.empty:
            data = self._sources.setdefault(source, IntervalDict())
            data[data.find(value) | interval] = value

        after = self._view()
        reconciliation = self._diff(before, after)
        if reconciliation:
            self._revision += 1
        return reconciliation

    def retract(self, source, interval=None):
        """
        Withdraw one source's evidence on a scale segment.

        :param source: identifier of the source whose evidence is withdrawn.
        :param interval: the segment to withdraw; a single value is interpreted
            as a singleton, and None withdraws the whole contribution of the
            source. Withdrawing a segment is idempotent: when the source has
            no evidence left (or never had any), the result is an empty
            reconciliation instead of an error, so a contribution can be
            withdrawn piece by piece in any number of steps.
        :return: a :class:`Reconciliation` describing the effective changes.
        :raises KeyError: only when the whole contribution of an unknown
            source is requested (``interval`` is None).
        """
        before = self._view()

        if interval is None:
            if source not in self._sources:
                raise KeyError(f"Unknown source {source!r}")
            del self._sources[source]
        else:
            interval = _as_interval(interval)
            data = self._sources.get(source)
            if data is not None and not interval.empty and data[interval].domain():
                del data[interval]
                if not data.domain():
                    del self._sources[source]

        after = self._view()
        reconciliation = self._diff(before, after)
        if reconciliation:
            self._revision += 1
        return reconciliation

    def retract_source(self, source):
        """Withdraw the whole contribution of a source."""
        return self.retract(source)

    def effective(self, interval=None):
        """
        Return the effective calibration as a list of ``(segment, value)``
        pairs where ``value`` is an :class:`EffectiveValue`.

        Adjacent segments with the same corrected value but distinct supporting
        sources are kept separate. Uncovered and conflicting segments are
        included, so conflicts are visible in the output.

        :param interval: optional segment restricting the query.
        """
        view = self._view()
        if interval is None:
            return list(view)

        interval = _as_interval(interval)
        return [
            (segment & interval, effective_value)
            for segment, effective_value in view
            if segment.overlaps(interval)
        ]

    def effective_at(self, point):
        """
        Return the :class:`EffectiveValue` at a single scale point, or None if
        no source covers it.
        """
        evidence = []
        for source, data in self._sources.items():
            try:
                evidence.append((source, data[point]))
            except KeyError:
                continue

        if not evidence:
            return None
        return self._summarize(evidence)

    def evidence(self, interval=None):
        """
        Return the raw, per-source evidence as a list of
        ``(source, segment, value)`` triples.

        Two sources reporting the same value on overlapping segments appear as
        two distinct triples and can be queried and retracted independently.

        :param interval: optional segment restricting the query.
        """
        result = []
        for source, data in self._sources.items():
            if interval is None:
                items = data.items()
            else:
                items = data[_as_interval(interval)].items()
            for segment, value in items:
                result.append((source, segment, value))
        return result

    def evidence_at(self, point):
        """
        Return the ``(source, value)`` evidence triples supporting a single
        scale point, in source insertion order.
        """
        result = []
        for source, data in self._sources.items():
            try:
                result.append((source, data[point]))
            except KeyError:
                continue
        return result

    # ----- internals -----------------------------------------------------

    def _view(self):
        """Build the maximal-segment effective view of the current state."""
        atomics = [
            atomic
            for data in self._sources.values()
            for atomic in data.domain()
        ]
        partitioned = _partition(atomics)

        segments = []
        for part in partitioned:
            evidence = []
            for source, data in self._sources.items():
                subset = data[part]
                if subset.domain():
                    evidence.append((source, next(iter(subset.values()))))
            if evidence:
                segments.append((part, self._summarize(evidence)))

        merged = []
        run_parts = []
        run_value = ...
        for part, effective_value in segments:
            adjacent = bool(run_parts) and (run_parts[-1] | part).atomic
            if (
                run_value is not ...
                and effective_value == run_value
                and adjacent
            ):
                run_parts.append(part)
            else:
                if run_parts:
                    merged.append((Interval(*run_parts), run_value))
                run_parts = [part]
                run_value = effective_value
        if run_parts:
            merged.append((Interval(*run_parts), run_value))
        return merged

    @staticmethod
    def _summarize(evidence):
        sources = tuple(source for source, _ in evidence)
        distinct = []
        for _, value in evidence:
            if value not in distinct:
                distinct.append(value)

        if len(distinct) == 1:
            unique = distinct[0]
            return EffectiveValue(unique, sources, False, (unique,))
        return EffectiveValue(None, sources, True, tuple(distinct))

    @staticmethod
    def _lookup(view, part):
        for segment, effective_value in view:
            if part in segment:
                return effective_value
        return None

    def _diff(self, before, after):
        atomics = [
            atomic
            for view in (before, after)
            for segment, _ in view
            for atomic in segment
        ]
        grouped = []
        run_parts = []
        run_signature = ...
        for part in _partition(atomics):
            old = self._lookup(before, part)
            new = self._lookup(after, part)
            if old == new:
                # An unchanged segment breaks any run of identical changes.
                if run_parts:
                    grouped.append(self._flush(run_parts, run_signature))
                    run_parts = []
                    run_signature = ...
                continue
            kind = self._kind(old, new)
            signature = (kind, old, new)
            if run_signature is not ... and signature == run_signature:
                run_parts.append(part)
            else:
                if run_parts:
                    grouped.append(self._flush(run_parts, run_signature))
                run_parts = [part]
                run_signature = signature
        if run_parts:
            grouped.append(self._flush(run_parts, run_signature))
        return Reconciliation(tuple(grouped))

    @staticmethod
    def _flush(run_parts, signature):
        return Change(Interval(*run_parts), signature[0], signature[1], signature[2])

    @staticmethod
    def _kind(before, after):
        if before is None:
            return CONFLICT_EMERGED if after.conflicting else NEW_COVERAGE
        if after is None:
            return COVERAGE_LOST

        if before.conflicting and not after.conflicting:
            return CONFLICT_RESOLVED
        if not before.conflicting and after.conflicting:
            return CONFLICT_EMERGED

        if not before.conflicting and before.value != after.value:
            return VALUE_CHANGED

        old_sources = set(before.sources)
        new_sources = set(after.sources)
        if new_sources < old_sources:
            return EVIDENCE_LOST
        if old_sources < new_sources:
            return EVIDENCE_GAINED
        if before.sources != after.sources:
            return EVIDENCE_LOST
        return VALUE_CHANGED if before.values != after.values else EVIDENCE_LOST
