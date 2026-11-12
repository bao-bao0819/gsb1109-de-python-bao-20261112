"""Equipment-inspection work-order schedule analysis.

Work orders are ingested from the ``VTODO`` entries of an iCalendar stream.
The standard ``UID``/``SUMMARY`` properties identify each order while the
scheduling data lives in source extension fields:

* ``X-PLANNED-HOURS``  - planned (estimated) work hours
* ``X-ACTUAL-HOURS``   - actual work hours
* ``X-PREDECESSORS``   - UIDs of work orders that must finish first
* ``X-BATCH-START``    - calendar-level batch kick-off instant

Given the batch kick-off instant and unlimited parallel workstations, every
order starts the moment all of its predecessors have finished.  The earliest
acceptance instant is therefore the end of the longest dependency path (the
critical chain) measured from the batch start.

Orders whose predecessors cannot be resolved inside the calendar (or which
depend, transitively, on such an order) do not form a dependency-closed set.
They are excluded from every published result; only closed terminal results are
reported.

Replacing an order's actual duration by its planned duration simulates removing
its delay.  ``minimum_corrections`` finds the smallest set of delayed orders
that must be corrected together to pull acceptance forward by a requested
amount - parallel critical chains may each require a correction before either
advances the batch.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import vobject

PLANNED_PROP = "x-planned-hours"
ACTUAL_PROP = "x-actual-hours"
PREDECESSORS_PROP = "x-predecessors"
BATCH_START_PROP = "x-batch-start"

_EPS = 1e-9
_UTC = timezone.utc
_SPLIT_PREDS = re.compile(r"[,;\s]+")
_LINE_WITH_VALUE = re.compile(r"^(?P<name>[^:;]+)(?P<params>(?:;[^:]*)?):(?P<value>.*)$")


def _normalize_predecessor_separators(raw: str) -> str:
    """Protect predecessor UID lists from vobject's comma text decoding.

    Unknown text properties go through ``TextBehavior``, which treats a comma as
    a multi-value separator and discards every value after the first. Rewriting
    commas to semicolons on ``X-PREDECESSORS`` lines (after unfolding) keeps the
    full predecessor list regardless of the separator the feed uses.
    """

    unfolded: list[str] = []
    for line in raw.splitlines():
        if line[:1] in (" ", "\t") and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)

    normalized: list[str] = []
    for line in unfolded:
        match = _LINE_WITH_VALUE.match(line)
        if match and match.group("name").split(".")[-1].strip().upper() == PREDECESSORS_PROP.upper():
            value = match.group("value").replace(",", ";")
            line = line[: match.start("value")] + value
        normalized.append(line)
    return "\n".join(normalized)


class ScheduleDataError(ValueError):
    """Raised when the calendar cannot describe a schedulable batch."""


@dataclass(frozen=True)
class WorkOrder:
    """A single inspection work order parsed from a VTODO."""

    uid: str
    summary: str
    planned_hours: float
    actual_hours: float
    predecessor_uids: tuple[str, ...]

    @property
    def delay_hours(self) -> float:
        """Extra hours actually spent beyond the plan (never negative)."""

        return max(0.0, self.actual_hours - self.planned_hours)

    @property
    def is_delayed(self) -> bool:
        return self.actual_hours > self.planned_hours + _EPS


@dataclass
class BatchSchedule:
    """A parsed batch restricted to its dependency-closed work orders."""

    batch_start: datetime
    work_orders: dict[str, WorkOrder]
    predecessors: dict[str, list[str]]
    closed_uids: set[str]
    open_uids: set[str]

    def work_order(self, uid: str) -> WorkOrder:
        return self.work_orders[uid]

    @property
    def delayed_closed_uids(self) -> list[str]:
        """Closed, delayed orders, the only candidates for a correction."""

        return sorted(uid for uid in self.closed_uids if self.work_orders[uid].is_delayed)


@dataclass
class ScenarioResult:
    """The recomputed schedule for one choice of work-order durations."""

    label: str
    corrected_uids: frozenset[str]
    durations: dict[str, float]
    early_start: dict[str, float]
    early_finish: dict[str, float]
    critical_chains: list[list[str]]
    acceptance_offset_hours: float
    batch_start: datetime

    @property
    def acceptance_at(self) -> datetime:
        return self.batch_start + timedelta(hours=self.acceptance_offset_hours)

    @property
    def critical_chain(self) -> list[str] | None:
        """The unique critical chain, or ``None`` when chains run in parallel."""

        return self.critical_chains[0] if len(self.critical_chains) == 1 else None


@dataclass
class OptimizationResult:
    """Outcome of searching for the minimum corrective set."""

    required_advance_hours: float
    achieved: bool
    baseline: ScenarioResult
    minimal_corrections: tuple[str, ...] | None
    corrected: ScenarioResult | None
    best_achievable: ScenarioResult
    best_advance_hours: float


def _parse_datetime(raw: object) -> datetime:
    if isinstance(raw, datetime):
        return raw
    text = str(raw).strip()
    if not text:
        raise ScheduleDataError("batch start instant is empty")
    if text.endswith(("Z", "z")):
        try:
            return datetime.strptime(text[:-1], "%Y%m%dT%H%M%S").replace(tzinfo=_UTC)
        except ValueError:
            text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ScheduleDataError(f"unparseable batch start instant: {raw!r}") from exc


def _hours(line_value: object, uid: str, prop: str) -> float:
    try:
        hours = float(line_value)
    except (TypeError, ValueError) as exc:
        raise ScheduleDataError(f"work order {uid}: {prop} is not numeric: {line_value!r}") from exc
    if hours < 0:
        raise ScheduleDataError(f"work order {uid}: {prop} must be non-negative")
    return hours


def _line(component, name: str):
    lines = component.contents.get(name)
    return lines[0] if lines else None


def _split_predecessors(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    seen: set[str] = set()
    ordered: list[str] = []
    for token in _SPLIT_PREDS.split(str(raw).strip()):
        if token and token not in seen:
            seen.add(token)
            ordered.append(token)
    return tuple(ordered)


def parse_calendar(source: str | Path | io.IOBase) -> BatchSchedule:
    """Parse an iCalendar source into a :class:`BatchSchedule`.

    ``source`` may be raw iCalendar text, a filesystem path, or an open text
    stream.
    """

    if isinstance(source, Path):
        raw_text = source.read_text(encoding="utf-8")
    elif isinstance(source, str):
        raw_text = source
    else:
        raw_text = source.read()
    calendar = vobject.readOne(io.StringIO(_normalize_predecessor_separators(raw_text)))

    start_line = _line(calendar, BATCH_START_PROP)
    if start_line is None:
        raise ScheduleDataError(f"calendar is missing {BATCH_START_PROP.upper()}")
    batch_start = _parse_datetime(start_line.value)

    work_orders: dict[str, WorkOrder] = {}
    declared_predecessors: dict[str, tuple[str, ...]] = {}
    for todo in calendar.contents.get("vtodo", []):
        uid = todo.getChildValue("uid")
        if not uid:
            raise ScheduleDataError("every VTODO must carry a UID")
        if uid in work_orders:
            raise ScheduleDataError(f"duplicate VTODO UID: {uid}")
        planned_line = _line(todo, PLANNED_PROP)
        actual_line = _line(todo, ACTUAL_PROP)
        if planned_line is None or actual_line is None:
            raise ScheduleDataError(
                f"work order {uid} requires both {PLANNED_PROP.upper()} and {ACTUAL_PROP.upper()}"
            )
        preds = _split_predecessors(todo.getChildValue(PREDECESSORS_PROP))
        work_orders[uid] = WorkOrder(
            uid=uid,
            summary=todo.getChildValue("summary", "") or "",
            planned_hours=_hours(planned_line.value, uid, PLANNED_PROP),
            actual_hours=_hours(actual_line.value, uid, ACTUAL_PROP),
            predecessor_uids=preds,
        )
        declared_predecessors[uid] = preds

    if not work_orders:
        raise ScheduleDataError("calendar contains no VTODO work orders")

    predecessors: dict[str, list[str]] = {
        uid: [pred for pred in preds if pred in work_orders]
        for uid, preds in declared_predecessors.items()
    }

    _raise_if_cyclic(work_orders, predecessors)

    closed_uids, open_uids = _dependency_closure(work_orders, declared_predecessors)
    return BatchSchedule(
        batch_start=batch_start,
        work_orders=work_orders,
        predecessors=predecessors,
        closed_uids=closed_uids,
        open_uids=open_uids,
    )


def _raise_if_cyclic(
    work_orders: dict[str, WorkOrder], predecessors: dict[str, list[str]]
) -> None:
    indegree = {uid: 0 for uid in work_orders}
    dependents: dict[str, list[str]] = {uid: [] for uid in work_orders}
    for uid, preds in predecessors.items():
        indegree[uid] = len(preds)
        for pred in preds:
            dependents[pred].append(uid)
    ready = [uid for uid, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        uid = ready.pop()
        visited += 1
        for dependent in dependents[uid]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    if visited != len(work_orders):
        cyclic = sorted(uid for uid, degree in indegree.items() if degree > 0)
        raise ScheduleDataError(f"dependency cycle detected among work orders: {cyclic}")


def _dependency_closure(
    work_orders: dict[str, WorkOrder], declared_predecessors: dict[str, tuple[str, ...]]
) -> tuple[set[str], set[str]]:
    """Return closed orders and the orders left without full dependency closure.

    ``declared_predecessors`` keeps every named predecessor, including UIDs that
    do not exist in the feed.  An order is closed when every named predecessor
    exists and is itself closed; roots (no predecessors) seed the closed set, so
    an order naming a missing predecessor - and anything chained behind it -
    remains open because the unresolved UID can never enter the closed set.
    """

    closed: set[str] = set()
    pending = set(work_orders)
    progress = True
    while progress:
        progress = False
        for uid in list(pending):
            if all(pred in closed for pred in declared_predecessors[uid]):
                closed.add(uid)
                pending.remove(uid)
                progress = True
    return closed, pending


def _topological_order(batch: BatchSchedule) -> list[str]:
    indegree = {uid: len(batch.predecessors[uid]) for uid in batch.closed_uids}
    dependents: dict[str, list[str]] = {uid: [] for uid in batch.closed_uids}
    for uid in batch.closed_uids:
        for pred in batch.predecessors[uid]:
            dependents[pred].append(uid)
    ready = sorted(uid for uid, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        uid = ready.pop(0)
        order.append(uid)
        for dependent in sorted(dependents[uid]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort()
    if len(order) != len(batch.closed_uids):
        raise ScheduleDataError("closed work orders contain a dependency cycle")
    return order


def simulate(
    batch: BatchSchedule, corrected_uids: Iterable[str] = (), label: str | None = None
) -> ScenarioResult:
    """Run the unlimited-parallel forward pass for one duration scenario.

    Orders in ``corrected_uids`` use their planned duration; every other closed
    order keeps its actual duration.  Open (dependency-unclosed) orders never
    participate.
    """

    corrected = frozenset(corrected_uids)
    unknown = corrected - batch.closed_uids
    if unknown:
        raise ScheduleDataError(
            f"cannot correct unknown or dependency-open work orders: {sorted(unknown)}"
        )

    order = _topological_order(batch)
    durations: dict[str, float] = {}
    early_start: dict[str, float] = {}
    early_finish: dict[str, float] = {}
    for uid in order:
        order_obj = batch.work_order(uid)
        durations[uid] = order_obj.planned_hours if uid in corrected else order_obj.actual_hours
        start = max((early_finish[pred] for pred in batch.predecessors[uid]), default=0.0)
        early_start[uid] = start
        early_finish[uid] = start + durations[uid]

    finish = max(early_finish.values(), default=0.0)
    chains = _critical_chains(batch, early_start, early_finish, finish)
    if label is None:
        label = "corrected: " + ", ".join(sorted(corrected)) if corrected else "baseline (actual hours)"
    return ScenarioResult(
        label=label,
        corrected_uids=corrected,
        durations=durations,
        early_start=early_start,
        early_finish=early_finish,
        critical_chains=chains,
        acceptance_offset_hours=finish,
        batch_start=batch.batch_start,
    )


def _critical_chains(
    batch: BatchSchedule,
    early_start: dict[str, float],
    early_finish: dict[str, float],
    finish: float,
) -> list[list[str]]:
    terminals = sorted(
        uid for uid in batch.closed_uids if abs(early_finish[uid] - finish) <= _EPS
    )
    chains: set[tuple[str, ...]] = set()

    def backtrack(uid: str, suffix: list[str]) -> None:
        critical_preds = [
            pred
            for pred in batch.predecessors[uid]
            if abs(early_finish[pred] - early_start[uid]) <= _EPS
        ]
        if not critical_preds:
            chains.add(tuple([uid, *suffix]))
            return
        for pred in sorted(critical_preds):
            backtrack(pred, [uid, *suffix])

    for terminal in terminals:
        backtrack(terminal, [])
    return [list(chain) for chain in sorted(chains)]


def advance_hours(batch: BatchSchedule, corrected_uids: Iterable[str] = ()) -> float:
    """How much earlier acceptance becomes if the given orders are corrected."""

    baseline = simulate(batch)
    scenario = simulate(batch, corrected_uids)
    return baseline.acceptance_offset_hours - scenario.acceptance_offset_hours


def minimum_corrections(batch: BatchSchedule, required_advance_hours: float) -> OptimizationResult:
    """Find the fewest delayed orders whose correction achieves the advance.

    Parallel critical chains are handled directly: a correction on one chain is
    credited only once the competing chain no longer dominates, so the search
    returns the joint set when neither chain alone can move the terminal.

    The search enumerates candidate subsets in increasing cardinality, so it is
    exact but exponential in the number of delayed closed orders; it is intended
    for bounded inspection batches.
    """

    if required_advance_hours < 0:
        raise ScheduleDataError("required advance must be non-negative")

    baseline = simulate(batch)
    target_offset = baseline.acceptance_offset_hours - required_advance_hours
    candidates = batch.delayed_closed_uids
    best_achievable = simulate(batch, candidates, label="all delayed orders corrected")
    best_advance = baseline.acceptance_offset_hours - best_achievable.acceptance_offset_hours

    if required_advance_hours <= _EPS:
        return OptimizationResult(
            required_advance_hours=required_advance_hours,
            achieved=True,
            baseline=baseline,
            minimal_corrections=(),
            corrected=baseline,
            best_achievable=best_achievable,
            best_advance_hours=best_advance,
        )

    if best_achievable.acceptance_offset_hours > target_offset + _EPS:
        return OptimizationResult(
            required_advance_hours=required_advance_hours,
            achieved=False,
            baseline=baseline,
            minimal_corrections=None,
            corrected=None,
            best_achievable=best_achievable,
            best_advance_hours=best_advance,
        )

    for size in range(1, len(candidates) + 1):
        for combo in combinations(candidates, size):
            scenario = simulate(batch, combo)
            if scenario.acceptance_offset_hours <= target_offset + _EPS:
                return OptimizationResult(
                    required_advance_hours=required_advance_hours,
                    achieved=True,
                    baseline=baseline,
                    minimal_corrections=combo,
                    corrected=scenario,
                    best_achievable=best_achievable,
                    best_advance_hours=best_advance,
                )
    raise ScheduleDataError("internal error: feasible correction set was not reached")


def _format_hours(value: float) -> str:
    return f"{value:.1f}" if value % 1 else f"{value:.0f}"


def _print_scenario(result: ScenarioResult, batch: BatchSchedule) -> None:
    print(f"  scenario: {result.label}")
    print(f"  earliest acceptance: {result.acceptance_at.isoformat()} "
          f"({_format_hours(result.acceptance_offset_hours)} h after batch start)")
    if len(result.critical_chains) == 1:
        print(f"  critical chain: {' -> '.join(result.critical_chains[0])}")
    else:
        print("  critical chains (parallel):")
        for chain in result.critical_chains:
            print(f"    - {' -> '.join(chain)}")
    print("  work orders:")
    header = f"    {'UID':<8}{'PLAN':>6}{'ACTUAL':>8}{'DELAY':>7}  {'START':>6} {'FINISH':>7}  PREDECESSORS"
    print(header)
    for uid in sorted(result.early_start):
        wo = batch.work_order(uid)
        preds = ",".join(wo.predecessor_uids) or "-"
        print(
            f"    {uid:<8}{_format_hours(wo.planned_hours):>6}"
            f"{_format_hours(wo.actual_hours):>8}{_format_hours(wo.delay_hours):>7}  "
            f"{_format_hours(result.early_start[uid]):>6} "
            f"{_format_hours(result.early_finish[uid]):>7}  {preds}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze inspection work-order delays from a VTODO calendar."
    )
    parser.add_argument("calendar", type=Path, help="path to the iCalendar (.ics) file")
    parser.add_argument(
        "--advance",
        type=float,
        default=None,
        help="required acceptance advance in hours; reports the minimum corrections",
    )
    args = parser.parse_args(argv)

    try:
        batch = parse_calendar(args.calendar)
    except (ScheduleDataError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"batch start: {batch.batch_start.isoformat()}")
    print(f"work orders: {len(batch.work_orders)} total, {len(batch.closed_uids)} "
          f"dependency-closed, {len(batch.open_uids)} open")
    if batch.open_uids:
        print(f"open (excluded, missing predecessor closure): {', '.join(sorted(batch.open_uids))}")
    print()
    baseline = simulate(batch)
    _print_scenario(baseline, batch)

    if args.advance is not None:
        print()
        outcome = minimum_corrections(batch, args.advance)
        print(f"requested advance: {_format_hours(outcome.required_advance_hours)} h")
        print(f"maximum achievable advance: {_format_hours(outcome.best_advance_hours)} h")
        if not outcome.achieved:
            print("result: not achievable even after correcting every delayed closed order")
            return 1
        print(f"minimum corrections ({len(outcome.minimal_corrections)}): "
              f"{', '.join(outcome.minimal_corrections)}")
        _print_scenario(outcome.corrected, batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
