"""Column-wise inference of ambiguous date formats.

A value such as ``03/04/05`` is readable under several legitimate formats
(``MM/DD/YY``, ``DD/MM/YY``, ``YY/MM/DD``).  This module infers, per column,
which of a finite set of candidate formats are usable, given:

* the observed values in each column,
* verified dates for individual cells (:class:`Anchor` calibration records),
* cross-column relations such as "shipment cannot precede the order"
  (:class:`ColumnRelation`).

When several format assignments remain plausible, a row is released only if
every surviving interpretation of that row falls into the same target
partition (by default the ``(year, month)`` bucket).  Rows that are not
released report two concrete, feasible interpretations.

Adding a calibration record narrows the surviving candidates, and results are
independent of the order in which candidate formats were listed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import datetime as dt_datetime
from itertools import product
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .api import get
from .parser import ParserError

_INVALID = object()

PartitionFn = Callable[[date], Hashable]


def _month_partition(value: date) -> Tuple[int, int]:
    return (value.year, value.month)


def _as_date(value: "date | dt_datetime") -> date:
    if isinstance(value, dt_datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"expected a date or datetime, got {type(value)!r}")


@dataclass(frozen=True)
class Anchor:
    """A verified date for one cell, used as a calibration record."""

    column: str
    row: int
    value: date


@dataclass(frozen=True)
class ColumnRelation:
    """Row-wise ordering constraint: ``earlier`` must not follow ``later``."""

    earlier: str
    later: str


@dataclass(frozen=True)
class Interpretation:
    """One feasible parse of a cell under a surviving format assignment."""

    formats: Tuple[Tuple[str, str], ...]
    value: date


@dataclass(frozen=True)
class RowDecision:
    """Outcome for one row of a target column.

    ``resolved`` rows are released: every surviving interpretation falls in
    ``partition``.  Unresolved rows carry exactly two feasible
    ``interpretations`` (fewer only when that many exist).
    """

    row: int
    resolved: bool
    partition: Optional[Hashable]
    interpretations: Tuple[Interpretation, ...]


@dataclass(frozen=True)
class InferenceResult:
    """Surviving formats, global assignments, and per-row decisions."""

    formats: Mapping[str, Tuple[str, ...]]
    assignments: Tuple[Mapping[str, str], ...]
    decisions: Mapping[str, Tuple[RowDecision, ...]]

    @property
    def determined(self) -> bool:
        """True when every column has narrowed to exactly one format."""
        return all(len(formats) == 1 for formats in self.formats.values())


class FormatInferer:
    """Infers usable date formats per column from values, anchors, relations."""

    def __init__(
        self,
        columns: Mapping[str, Sequence[Optional[str]]],
        formats: Iterable[str],
        anchors: Iterable[Anchor] = (),
        relations: Iterable[ColumnRelation] = (),
        partition: Optional[PartitionFn] = None,
        targets: Optional[Iterable[str]] = None,
    ) -> None:
        if not columns:
            raise ValueError("at least one column is required")
        deduped = sorted(set(formats))
        if not deduped:
            raise ValueError("at least one candidate format is required")

        self._columns: Dict[str, Tuple[Optional[str], ...]] = {
            name: tuple(values) for name, values in columns.items()
        }
        self._formats: Tuple[str, ...] = tuple(deduped)
        self._relations: Tuple[ColumnRelation, ...] = tuple(relations)
        self._partition: PartitionFn = partition or _month_partition
        self._targets: Tuple[str, ...] = (
            tuple(self._columns) if targets is None else tuple(targets)
        )
        self._anchors: List[Anchor] = []
        self._parse_cache: Dict[Tuple[str, str], Tuple[object, ...]] = {}

        for anchor in anchors:
            self.add_anchor(anchor.column, anchor.row, anchor.value)
        for relation in self._relations:
            for name in (relation.earlier, relation.later):
                if name not in self._columns:
                    raise ValueError(f"relation references unknown column {name!r}")
        for name in self._targets:
            if name not in self._columns:
                raise ValueError(f"target references unknown column {name!r}")

    @property
    def formats(self) -> Tuple[str, ...]:
        """The candidate formats, deduplicated and canonically ordered."""
        return self._formats

    @property
    def anchors(self) -> Tuple[Anchor, ...]:
        return tuple(self._anchors)

    def add_anchor(self, column: str, row: int, value: "date | dt_datetime") -> None:
        """Add a calibration record: ``column[row]`` is known to be ``value``."""
        if column not in self._columns:
            raise ValueError(f"anchor references unknown column {column!r}")
        if not 0 <= row < len(self._columns[column]):
            raise ValueError(f"anchor row {row} out of range for column {column!r}")
        self._anchors.append(Anchor(column=column, row=row, value=_as_date(value)))

    def calibrate(
        self, column: str, row: int, value: "date | dt_datetime"
    ) -> InferenceResult:
        """Add a calibration record and return the narrowed inference."""
        self.add_anchor(column, row, value)
        return self.infer()

    def infer(self) -> InferenceResult:
        """Compute surviving formats, assignments, and row decisions."""
        viable = {
            name: tuple(fmt for fmt in self._formats if self._is_viable(name, fmt))
            for name in self._columns
        }
        names = sorted(self._columns)
        assignments = tuple(
            dict(zip(names, combo))
            for combo in product(*(viable[name] for name in names))
            if self._satisfies_relations(names, combo)
        )
        surviving = {
            name: tuple(
                fmt for fmt in self._formats if any(a[name] == fmt for a in assignments)
            )
            for name in self._columns
        }
        decisions = {
            name: self._decide(name, assignments) for name in self._targets
        }
        return InferenceResult(
            formats=surviving, assignments=assignments, decisions=decisions
        )

    def _parsed_column(self, column: str, fmt: str) -> Tuple[object, ...]:
        key = (column, fmt)
        if key not in self._parse_cache:
            parsed: List[object] = []
            for raw in self._columns[column]:
                if raw is None or raw == "":
                    parsed.append(None)
                    continue
                try:
                    parsed.append(get(raw, fmt).date())
                except (ParserError, ValueError, OverflowError):
                    parsed.append(_INVALID)
            self._parse_cache[key] = tuple(parsed)
        return self._parse_cache[key]

    def _is_viable(self, column: str, fmt: str) -> bool:
        parsed = self._parsed_column(column, fmt)
        if any(value is _INVALID for value in parsed):
            return False
        return all(
            anchor.column != column or parsed[anchor.row] == anchor.value
            for anchor in self._anchors
        )

    def _satisfies_relations(
        self, names: Sequence[str], combo: Tuple[str, ...]
    ) -> bool:
        assignment = dict(zip(names, combo))
        for relation in self._relations:
            earlier = self._parsed_column(
                relation.earlier, assignment[relation.earlier]
            )
            later = self._parsed_column(relation.later, assignment[relation.later])
            for first, second in zip(earlier, later):
                if first is not None and second is not None and first > second:
                    return False
        return True

    def _decide(
        self, column: str, assignments: Tuple[Mapping[str, str], ...]
    ) -> Tuple[RowDecision, ...]:
        decisions = []
        for row, raw in enumerate(self._columns[column]):
            if raw is None or raw == "":
                decisions.append(
                    RowDecision(
                        row=row, resolved=True, partition=None, interpretations=()
                    )
                )
                continue
            interpretations: List[Interpretation] = []
            seen: Set[date] = set()
            for assignment in assignments:
                value = self._parsed_column(column, assignment[column])[row]
                if value in seen:
                    continue
                seen.add(value)
                key = tuple(sorted(assignment.items()))
                interpretations.append(Interpretation(formats=key, value=value))
            partitions = {self._partition(item.value) for item in interpretations}
            if len(partitions) <= 1:
                partition = partitions.pop() if partitions else None
                decisions.append(
                    RowDecision(
                        row=row,
                        resolved=True,
                        partition=partition,
                        interpretations=tuple(interpretations),
                    )
                )
            else:
                decisions.append(
                    RowDecision(
                        row=row,
                        resolved=False,
                        partition=None,
                        interpretations=tuple(interpretations[:2]),
                    )
                )
        return tuple(decisions)
