"""
Diagnose the coordinate axis order (``(latitude, longitude)`` versus
``(longitude, latitude)``) of data partitions in which both imported columns
happen to fall within +/-90 degrees, so that swapping the columns still
produces a syntactically valid :class:`geopy.point.Point`.

The input is a batch of records carrying station identifiers.  Every
partition uses exactly one axis order; some stations have trusted
coordinates; the same station may show up in several partitions, in which
case the reconstructed points must agree within a given distance tolerance.

Three kinds of constraints decide whether an axis-order hypothesis --
``normal`` (the stored pair is latitude/longitude) or ``swapped`` (the
stored pair is longitude/latitude) -- is viable for a partition:

1. Range constraints.  A hypothesis is only viable if reconstructing every
   record of the partition yields a valid point, i.e. the value used as
   latitude lies within the [-90, 90] range.
2. Trusted-coordinate constraints.  For records whose station has a trusted
   coordinate, the reconstructed point must lie within ``tolerance_km`` of
   it.
3. Cross-partition station constraints.  Points reconstructed for the same
   station from different records must lie within ``tolerance_km`` of each
   other (this also covers repeated stations inside one partition).

Each constraint is encoded as a bitmask over the ``2 ** k`` possible
assignments of an axis order to the ``k`` partitions.  A set of constraints
is jointly satisfiable exactly when the bitwise conjunction of its masks is
non-zero, which makes enumerating the viable explanations and extracting a
minimal contradictory constraint group straightforward.  Memory grows as
``2 ** k`` bits per constraint, so the exhaustive search is suited to
batches with at most around twenty partitions.
"""

import math
from dataclasses import dataclass, field
from itertools import combinations

from geopy.distance import distance
from geopy.point import Point

NORMAL = "normal"
SWAPPED = "swapped"
_ORDERS = (NORMAL, SWAPPED)


@dataclass(frozen=True)
class StationRecord:
    """A single imported ``(first, second)`` pair for a station."""

    partition: str
    station: str
    first: float
    second: float
    index: int = 0


@dataclass(frozen=True)
class TrustedPoint:
    """A trusted reference coordinate for a station."""

    station: str
    point: Point


@dataclass(frozen=True)
class Constraint:
    """A constraint that an axis-order assignment must satisfy.

    ``kind`` is one of ``"range"``, ``"trusted"`` or ``"cross_partition"``.
    ``records`` holds the one or two :class:`StationRecord` instances
    involved and ``detail`` is a human-readable description.
    """

    kind: str
    records: tuple
    detail: str


@dataclass(frozen=True)
class Explanation:
    """One complete viable assignment of axis orders to partitions."""

    orders: dict
    points: dict


@dataclass(frozen=True)
class AxisOrderDiagnosis:
    """Result of :func:`diagnose_axis_order`.

    * ``partitions``: all partitions seen in the input, in first-seen order.
    * ``explanations``: every viable complete assignment, each with the
      reconstructed :class:`geopy.point.Point` for every record, keyed by
      the record position in the input.
    * ``forced``: partitions whose axis order is identical in every viable
      explanation, mapped to that order.
    * ``ambiguous``: partitions whose order still varies between viable
      explanations.
    * ``conflict``: an inclusion-minimal group of constraints that cannot
      hold simultaneously, or ``None`` when at least one explanation exists.
    """

    partitions: tuple
    explanations: tuple
    forced: dict = field(default_factory=dict)
    ambiguous: tuple = ()
    conflict: object = None

    @property
    def is_consistent(self):
        return self.conflict is None

    @property
    def is_unique(self):
        return self.conflict is None and len(self.explanations) == 1


def _as_point(value):
    if isinstance(value, Point):
        return Point(value)
    return Point(value[0], value[1])


def _coerce_records(records):
    coerced = []
    for index, record in enumerate(records):
        if isinstance(record, StationRecord):
            coerced.append(StationRecord(
                record.partition, record.station,
                float(record.first), float(record.second), index,
            ))
        else:
            partition, station, first, second = record
            coerced.append(StationRecord(
                partition, station, float(first), float(second), index,
            ))
    return coerced


def _coerce_trusted(trusted):
    if trusted is None:
        return {}
    if isinstance(trusted, dict):
        items = trusted.items()
    else:
        items = ((item.station, item.point) for item in trusted)
    return {station: _as_point(point) for station, point in items}


def reconstruct(record, order):
    """Reconstruct a :class:`Point` for a record under an axis order.

    Returns ``None`` when the order would put a value outside [-90, 90] in
    the latitude slot, i.e. when no legal :class:`Point` can be built.
    """

    if order == NORMAL:
        latitude, longitude = record.first, record.second
    elif order == SWAPPED:
        latitude, longitude = record.second, record.first
    else:
        raise ValueError("Unknown axis order: %r" % (order,))
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        return None
    if abs(latitude) > 90:
        return None
    return Point(latitude, longitude)


def _within_tolerance(first, second, tolerance_km):
    return distance(first, second).km <= tolerance_km


def _order_bit(assignment_bits, partition_index):
    return (assignment_bits >> partition_index) & 1


def _build_constraints(records, trusted, partition_index, tolerance_km):
    """Return ``(Constraint, holds)`` pairs; ``holds(bits)`` tests an
    assignment encoded as a bit-per-partition integer."""

    constraints = []

    for record in records:
        def range_holds(bits, record=record,
                        partition_index=partition_index):
            order = _ORDERS[_order_bit(bits, partition_index[record.partition])]
            return reconstruct(record, order) is not None

        constraints.append((Constraint(
            "range", (record,),
            "partition %r record #%d (station %r, stored pair %r): the "
            "value used as latitude must stay within [-90, 90]"
            % (record.partition, record.index, record.station,
               (record.first, record.second)),
        ), range_holds))

    for record in records:
        if record.station not in trusted:
            continue
        reference = trusted[record.station]

        def trusted_holds(bits, record=record, reference=reference,
                          tolerance_km=tolerance_km,
                          partition_index=partition_index):
            order = _ORDERS[_order_bit(bits, partition_index[record.partition])]
            point = reconstruct(record, order)
            return point is not None and _within_tolerance(
                point, reference, tolerance_km)

        constraints.append((Constraint(
            "trusted", (record,),
            "partition %r record #%d (station %r) must reconstruct within "
            "%.6g km of trusted point %r"
            % (record.partition, record.index, record.station,
               tolerance_km, tuple(reference)),
        ), trusted_holds))

    by_station = {}
    for record in records:
        by_station.setdefault(record.station, []).append(record)
    for station, station_records in by_station.items():
        for left, right in combinations(station_records, 2):
            def cross_holds(bits, left=left, right=right,
                            tolerance_km=tolerance_km,
                            partition_index=partition_index):
                left_order = _ORDERS[
                    _order_bit(bits, partition_index[left.partition])]
                right_order = _ORDERS[
                    _order_bit(bits, partition_index[right.partition])]
                left_point = reconstruct(left, left_order)
                right_point = reconstruct(right, right_order)
                return (
                    left_point is not None
                    and right_point is not None
                    and _within_tolerance(
                        left_point, right_point, tolerance_km)
                )

            constraints.append((Constraint(
                "cross_partition", (left, right),
                "station %r: partition %r record #%d and partition %r "
                "record #%d must reconstruct within %.6g km of each other"
                % (station, left.partition, left.index,
                   right.partition, right.index, tolerance_km),
            ), cross_holds))

    return constraints


def _constraint_masks(constraints, assignment_count):
    masks = []
    for _constraint, holds in constraints:
        mask = 0
        for bits in range(assignment_count):
            if holds(bits):
                mask |= 1 << bits
        masks.append(mask)
    return masks


def _minimal_conflict(constraints, masks):
    """Return an inclusion-minimal unsatisfiable subset of constraints."""

    core = list(range(len(constraints)))

    def satisfiable(indices):
        combined = None
        for index in indices:
            combined = masks[index] if combined is None else combined & masks[index]
            if combined == 0:
                return False
        return True

    for index in list(core):
        candidate = [other for other in core if other != index]
        if not satisfiable(candidate):
            core = candidate
    return tuple(constraints[index][0] for index in core)


def _bits_to_orders(bits, partitions):
    return {
        partition: _ORDERS[_order_bit(bits, position)]
        for position, partition in enumerate(partitions)
    }


def diagnose_axis_order(records, trusted=None, tolerance_km=1.0):
    """Diagnose which partitions store swapped longitude/latitude pairs.

    :param records: iterable of :class:`StationRecord` objects or
        ``(partition, station, first, second)`` tuples with the two imported
        columns in storage order.
    :param trusted: mapping of station identifiers to trusted
        :class:`geopy.point.Point` values or ``(latitude, longitude)``
        tuples, or an iterable of :class:`TrustedPoint`.
    :param float tolerance_km: maximum allowed geodesic distance between
        coordinates that are supposed to describe the same station.
    :rtype: AxisOrderDiagnosis
    """

    if tolerance_km < 0:
        raise ValueError("tolerance_km must be non-negative")

    records = _coerce_records(records)
    trusted = _coerce_trusted(trusted)
    if not records:
        raise ValueError("at least one record is required")

    partitions = []
    for record in records:
        if record.partition not in partitions:
            partitions.append(record.partition)
    partition_index = {
        partition: position for position, partition in enumerate(partitions)
    }
    assignment_count = 1 << len(partitions)

    constraints = _build_constraints(
        records, trusted, partition_index, tolerance_km)
    masks = _constraint_masks(constraints, assignment_count)

    viable_mask = (1 << assignment_count) - 1
    for mask in masks:
        viable_mask &= mask
        if viable_mask == 0:
            break

    if viable_mask == 0:
        conflict = _minimal_conflict(constraints, masks)
        return AxisOrderDiagnosis(
            partitions=tuple(partitions),
            explanations=(),
            forced={},
            ambiguous=tuple(partitions),
            conflict=conflict,
        )

    explanations = []
    for bits in range(assignment_count):
        if not (viable_mask & (1 << bits)):
            continue
        orders = _bits_to_orders(bits, partitions)
        points = {
            record.index: reconstruct(record, orders[record.partition])
            for record in records
        }
        explanations.append(Explanation(orders=orders, points=points))

    forced = {}
    ambiguous = []
    for position, partition in enumerate(partitions):
        orders_seen = {
            _order_bit(bits, position)
            for bits in range(assignment_count)
            if viable_mask & (1 << bits)
        }
        if len(orders_seen) == 1:
            forced[partition] = _ORDERS[orders_seen.pop()]
        else:
            ambiguous.append(partition)

    return AxisOrderDiagnosis(
        partitions=tuple(partitions),
        explanations=tuple(explanations),
        forced=forced,
        ambiguous=tuple(ambiguous),
        conflict=None,
    )


def format_report(diagnosis):
    """Render an :class:`AxisOrderDiagnosis` as a human-readable report."""

    lines = []
    if diagnosis.conflict is not None:
        lines.append("CONTRADICTION: no axis-order assignment satisfies all "
                     "constraints. Minimal conflicting constraint group:")
        for constraint in diagnosis.conflict:
            lines.append("  - [%s] %s" % (constraint.kind, constraint.detail))
        return "\n".join(lines)

    if diagnosis.forced:
        lines.append("Partitions with a forced axis order:")
        for partition in diagnosis.partitions:
            if partition in diagnosis.forced:
                lines.append("  - %r: %s"
                             % (partition, diagnosis.forced[partition]))
    else:
        lines.append("No partition has a forced axis order.")

    if diagnosis.ambiguous:
        lines.append("Ambiguous partitions: %r" % (diagnosis.ambiguous,))

    lines.append("Viable explanations: %d" % len(diagnosis.explanations))
    for number, explanation in enumerate(diagnosis.explanations, start=1):
        lines.append("  Explanation %d:" % number)
        for partition in diagnosis.partitions:
            lines.append("    %r -> %s"
                         % (partition, explanation.orders[partition]))
        for index in sorted(explanation.points):
            lines.append("    record #%d -> %r"
                         % (index, tuple(explanation.points[index])))
    return "\n".join(lines)
