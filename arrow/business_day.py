"""Partition UTC events into store business days.

A store's business day starts at a fixed local wall-clock time (by default
02:30). Business days are therefore intervals on the UTC timeline whose
boundaries are obtained by locating that wall time in the store timezone for
each calendar date.

Two DST edge cases are handled explicitly:

- a spring-forward gap can make the boundary wall time non-existent; the rule
  selects either the earlier or the later real instant of the gap.
- a fall-back fold can make the boundary wall time occur twice; the rule
  selects either the first (``fold=0``) or the second (``fold=1``) instant.

Rules may carry an effective interval, allowing a store to change its policy
over time. Events are always walked in real (UTC) time order and every event is
assigned to exactly one business day, even on days where the boundary does not
exist or occurs twice.
"""

from dataclasses import dataclass, field
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from datetime import timedelta, timezone
from datetime import tzinfo as dt_tzinfo
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

from dateutil import tz as dateutil_tz

from arrow.arrow import TZ_EXPR, Arrow

try:
    import pytz
except ImportError:  # pragma: no cover
    pytz = None  # type: ignore[assignment]

#: Spring-forward boundary: keep the earlier real instant of the gap.
GAP_EARLIEST = "earliest"
#: Spring-forward boundary: keep the later real instant of the gap.
GAP_LATEST = "latest"

#: Fall-back boundary: keep the first occurrence of the repeated wall time.
FOLD_FIRST = "first"
#: Fall-back boundary: keep the second occurrence of the repeated wall time.
FOLD_SECOND = "second"

_GAP_CHOICES = (GAP_EARLIEST, GAP_LATEST)
_FOLD_CHOICES = (FOLD_FIRST, FOLD_SECOND)

_DT_EXPR = Union[Arrow, dt_datetime]
_KEY_FUNC = Callable[[Any], _DT_EXPR]


def _forever_past() -> dt_datetime:
    return dt_datetime.min.replace(tzinfo=timezone.utc)


def _forever_future() -> dt_datetime:
    return dt_datetime.max.replace(tzinfo=timezone.utc)


def _as_utc_datetime(value: _DT_EXPR) -> dt_datetime:
    """Coerce an Arrow/datetime expression into a tz-aware UTC datetime."""
    current = value.datetime if isinstance(value, Arrow) else value
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _coerce_bound(value: Optional[_DT_EXPR], default: dt_datetime) -> dt_datetime:
    return default if value is None else _as_utc_datetime(value)


def _coerce_tzinfo(tz_expr: TZ_EXPR) -> dt_tzinfo:
    if isinstance(tz_expr, dt_tzinfo):
        return tz_expr
    return Arrow._get_tzinfo(tz_expr)


def _timezone_key(tzinfo: dt_tzinfo) -> str:
    """Stable identity of a timezone so rules can be checked for the same zone."""
    zone_name = getattr(tzinfo, "key", None) or getattr(tzinfo, "zone", None)
    return str(zone_name if zone_name is not None else tzinfo)


def _is_pytz(tzinfo: dt_tzinfo) -> bool:
    return pytz is not None and isinstance(tzinfo, pytz.tzinfo.BaseTzInfo)


def _pytz_boundary_utc(local_date: dt_date, rule: "BusinessDayRule") -> dt_datetime:
    """Locate a boundary through pytz, which localizes naive times explicitly."""
    assert pytz is not None
    tzinfo: Any = rule.tzinfo
    naive = dt_datetime.combine(local_date, rule.boundary_time)
    try:
        localized = tzinfo.localize(naive, is_dst=None)
    except pytz.exceptions.AmbiguousTimeError:
        # is_dst=True keeps the DST (first) occurrence; False the second.
        localized = tzinfo.localize(naive, is_dst=rule.fold == FOLD_FIRST)
    except pytz.exceptions.NonExistentTimeError:
        # is_dst=True applies the DST offset (earlier instant), False the later.
        earlier = tzinfo.localize(naive, is_dst=True).astimezone(timezone.utc)
        later = tzinfo.localize(naive, is_dst=False).astimezone(timezone.utc)
        return earlier if rule.gap == GAP_EARLIEST else later
    return localized.astimezone(timezone.utc)


def _local_boundary_utc(local_date: dt_date, rule: "BusinessDayRule") -> dt_datetime:
    """Return the UTC instant of one rule's boundary on a given local date."""
    if _is_pytz(rule.tzinfo):
        return _pytz_boundary_utc(local_date, rule)

    naive = dt_datetime.combine(local_date, rule.boundary_time)
    tzinfo = rule.tzinfo

    candidate = naive.replace(tzinfo=tzinfo, fold=0)
    if not dateutil_tz.datetime_exists(candidate):
        # Non-existent wall time (spring-forward gap). PEP 495 exposes the two
        # real instants bracketing the gap through fold=0 and fold=1.
        later = naive.replace(tzinfo=tzinfo, fold=0).astimezone(timezone.utc)
        earlier = naive.replace(tzinfo=tzinfo, fold=1).astimezone(timezone.utc)
        return earlier if rule.gap == GAP_EARLIEST else later

    if dateutil_tz.datetime_ambiguous(candidate):
        # Repeated wall time (fall-back fold): fold selects the occurrence.
        fold = 1 if rule.fold == FOLD_SECOND else 0
        return naive.replace(tzinfo=tzinfo, fold=fold).astimezone(timezone.utc)

    return candidate.astimezone(timezone.utc)


@dataclass(frozen=True)
class BusinessDayRule:
    """A store rule mapping UTC instants onto local business days.

    :param tz: the store timezone, as a tzinfo object or a timezone expression
        (e.g. ``"America/New_York"``).
    :param start: (optional) UTC instant from which the rule is effective
        (inclusive). Defaults to the beginning of time.
    :param end: (optional) UTC instant until which the rule is effective
        (exclusive). Defaults to forever.
    :param boundary_time: (optional) local wall time at which a business day
        starts. Defaults to ``time(2, 30)``.
    :param gap: (optional) boundary choice when the wall time does not exist
        during a spring-forward gap: ``"earliest"`` or ``"latest"``. Defaults
        to ``"earliest"``.
    :param fold: (optional) boundary choice when the wall time occurs twice
        during a fall-back fold: ``"first"`` or ``"second"``. Defaults to
        ``"first"``.
    """

    tz: TZ_EXPR
    start: Optional[_DT_EXPR] = None
    end: Optional[_DT_EXPR] = None
    boundary_time: dt_time = dt_time(2, 30)
    gap: str = GAP_EARLIEST
    fold: str = FOLD_FIRST
    tzinfo: dt_tzinfo = field(init=False, repr=False, compare=False)
    effective_start: dt_datetime = field(init=False, repr=False, compare=False)
    effective_end: dt_datetime = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_time, dt_time):
            raise TypeError("boundary_time must be a datetime.time instance.")
        if self.gap not in _GAP_CHOICES:
            raise ValueError(f"gap must be one of {_GAP_CHOICES!r}, got {self.gap!r}.")
        if self.fold not in _FOLD_CHOICES:
            raise ValueError(
                f"fold must be one of {_FOLD_CHOICES!r}, got {self.fold!r}."
            )

        tzinfo = _coerce_tzinfo(self.tz)
        effective_start = _coerce_bound(self.start, _forever_past())
        effective_end = _coerce_bound(self.end, _forever_future())
        if effective_start >= effective_end:
            raise ValueError(
                "Rule effective interval is empty: start must be before end."
            )

        object.__setattr__(self, "tzinfo", tzinfo)
        object.__setattr__(self, "effective_start", effective_start)
        object.__setattr__(self, "effective_end", effective_end)

    def applies_at(self, instant: dt_datetime) -> bool:
        """Return whether the rule is in effect at the given UTC instant."""
        return self.effective_start <= instant < self.effective_end

    def boundary_at(self, local_date: dt_date) -> Arrow:
        """Return the UTC ``Arrow`` boundary starting ``local_date``'s business day."""
        return Arrow.fromdatetime(_local_boundary_utc(local_date, self))


@dataclass(frozen=True)
class BusinessDay:
    """One business day interval on the UTC timeline."""

    date: dt_date
    start: Arrow
    end: Arrow
    rule: BusinessDayRule

    def contains(self, instant: _DT_EXPR) -> bool:
        """Return whether ``instant`` falls in this business day (start inclusive)."""
        current = _as_utc_datetime(instant)
        return self.start.datetime <= current < self.end.datetime


@dataclass(frozen=True)
class BusinessDayAssignment:
    """The business day assigned to one event."""

    event: Any
    instant: Arrow
    business_day: BusinessDay


@dataclass(frozen=True)
class ReassignedEvent:
    """An event whose business day changed after a rule change."""

    event: Any
    instant: Arrow
    old_day: BusinessDay
    new_day: BusinessDay

    @property
    def old_boundary(self) -> Tuple[Arrow, Arrow]:
        """The original ``(start, end)`` UTC boundaries of the event."""
        return self.old_day.start, self.old_day.end

    @property
    def new_boundary(self) -> Tuple[Arrow, Arrow]:
        """The new ``(start, end)`` UTC boundaries of the event."""
        return self.new_day.start, self.new_day.end


class BusinessDayPartitioner:
    """Partition UTC events into business days under one or more store rules.

    :param rules: one :class:`BusinessDayRule` or an ordered sequence of rules
        whose effective intervals must not overlap and must cover every
        instant that is partitioned.
    """

    def __init__(
        self, rules: Union[BusinessDayRule, Sequence[BusinessDayRule]]
    ) -> None:
        if isinstance(rules, BusinessDayRule):
            rule_list: List[BusinessDayRule] = [rules]
        else:
            rule_list = list(rules)

        if not rule_list:
            raise ValueError("At least one business day rule is required.")

        ordered = sorted(rule_list, key=lambda rule: rule.effective_start)
        for previous, current in zip(ordered, ordered[1:]):
            if current.effective_start < previous.effective_end:
                raise ValueError(
                    "Business day rule effective intervals must not overlap."
                )

        timezone_keys = {_timezone_key(rule.tzinfo) for rule in ordered}
        if len(timezone_keys) != 1:
            raise ValueError(
                "All rules of one store must use the same timezone; rule changes "
                "may only alter the boundary time and DST edge choices."
            )

        self.rules: Tuple[BusinessDayRule, ...] = tuple(ordered)

    def _rule_at(self, instant: dt_datetime) -> BusinessDayRule:
        for rule in self.rules:
            if rule.applies_at(instant):
                return rule
        raise ValueError(
            f"No business day rule is in effect at {instant.isoformat()}. "
            "Add a rule covering this instant."
        )

    def _rule_for_date(self, local_date: dt_date) -> BusinessDayRule:
        """Rule governing a local date, defaulting to the nearest rule outside coverage.

        Local noon is used as the probe because it never falls inside a DST gap.
        """
        tzinfo = self.rules[0].tzinfo
        noon_naive = dt_datetime.combine(local_date, dt_time(12))
        if _is_pytz(tzinfo):
            noon_utc = tzinfo.localize(noon_naive).astimezone(timezone.utc)
        else:
            noon_utc = noon_naive.replace(tzinfo=tzinfo).astimezone(timezone.utc)
        if noon_utc < self.rules[0].effective_start:
            return self.rules[0]
        if noon_utc >= self.rules[-1].effective_end:
            return self.rules[-1]
        return self._rule_at(noon_utc)

    def _boundary_for_date(
        self, local_date: dt_date
    ) -> Tuple[dt_datetime, BusinessDayRule]:
        """UTC boundary starting the business day of ``local_date`` and its rule.

        The rule is the one in effect at the boundary instant itself. The choice
        is resolved to a fixed point so that, when a rule change falls near a
        boundary, the two candidate wall times cannot oscillate: each rule's
        boundary is self-consistent on its own side of the change.
        """
        rule = self._rule_for_date(local_date)
        for _ in range(len(self.rules) + 1):
            boundary = _local_boundary_utc(local_date, rule)
            try:
                governing = self._rule_at(boundary)
            except ValueError:
                governing = self._rule_for_date(local_date)
            if governing is rule:
                return boundary, rule
            rule = governing
        raise RuntimeError(
            "Could not resolve a stable business day boundary; rule effective "
            "intervals are not ordered consistently around the boundary."
        )

    def _business_day(self, instant: dt_datetime) -> BusinessDay:
        """Locate the one business day containing an instant, in UTC order.

        Boundary candidates are walked one local date at a time. At every step
        the rule governing the candidate boundary is consulted, so a boundary
        that does not exist or occurs twice never leaves an instant unassigned:
        the adjacent real boundaries still delimit exactly one day containing
        it. Consecutive local boundaries are strictly increasing in UTC, so the
        walk always terminates.
        """
        entry_rule = self._rule_at(instant)
        local_date = instant.astimezone(entry_rule.tzinfo).date()
        start, start_rule = self._boundary_for_date(local_date)

        # Walk backwards to the last boundary at or before the instant.
        while start > instant:
            previous_date = (
                start.astimezone(start_rule.tzinfo).date() - timedelta(days=1)
            )
            start, start_rule = self._boundary_for_date(previous_date)

        # Walk forwards to the first boundary strictly after the instant.
        end, end_rule = start, start_rule
        while end <= instant:
            next_date = end.astimezone(end_rule.tzinfo).date() + timedelta(days=1)
            end, end_rule = self._boundary_for_date(next_date)

        day_date = start.astimezone(start_rule.tzinfo).date()
        return BusinessDay(
            day_date,
            Arrow.fromdatetime(start),
            Arrow.fromdatetime(end),
            start_rule,
        )

    def partition(
        self,
        events: Iterable[Any],
        key: Optional[_KEY_FUNC] = None,
    ) -> List[BusinessDayAssignment]:
        """Assign every event to exactly one business day.

        :param events: events carrying a UTC instant. Raw datetimes/Arrows may
            be passed directly, or arbitrary objects with a time accessor.
        :param key: (optional) callable returning the UTC instant (Arrow or
            tz-aware datetime) of an event. Required for non-datetime events.
        :return: assignments in real (UTC) time order.
        """
        accessor = _identity_key if key is None else key

        normalized = [(_as_utc_datetime(accessor(event)), event) for event in events]
        normalized.sort(key=lambda item: item[0])

        assignments: List[BusinessDayAssignment] = []
        for instant, event in normalized:
            day = self._business_day(instant)
            assignments.append(
                BusinessDayAssignment(event, Arrow.fromdatetime(instant), day)
            )
        return assignments

    def business_day_of(self, instant: _DT_EXPR) -> BusinessDay:
        """Return the business day containing a single UTC instant."""
        return self._business_day(_as_utc_datetime(instant))

    def replay(
        self,
        new_rules: Union[BusinessDayRule, Sequence[BusinessDayRule]],
        events: Iterable[Any],
        key: Optional[_KEY_FUNC] = None,
        window: Optional[Tuple[_DT_EXPR, _DT_EXPR]] = None,
    ) -> List[ReassignedEvent]:
        """Replay events under new rules and report the business days that moved.

        Only events in ``window`` (UTC, start inclusive, end exclusive) are
        replayed; when omitted, every event is checked. Events assigned to the
        same business day (by local date) are skipped, so only affected history
        is returned; each reported event still carries its old and new UTC
        boundaries.

        :param new_rules: the replacement rule set.
        :param events: the historical events, as for :meth:`partition`.
        :param key: (optional) instant accessor for non-datetime events.
        :param window: (optional) UTC ``(start, end)`` interval to replay.
        :return: reassignments in real (UTC) time order, each carrying the old
            and new business days and their original/new UTC boundaries.
        """
        accessor = _identity_key if key is None else key
        target = (
            new_rules
            if isinstance(new_rules, BusinessDayPartitioner)
            else BusinessDayPartitioner(new_rules)
        )

        window_start = (
            _as_utc_datetime(window[0]) if window is not None else _forever_past()
        )
        window_end = (
            _as_utc_datetime(window[1]) if window is not None else _forever_future()
        )

        normalized = []
        for event in events:
            instant = _as_utc_datetime(accessor(event))
            if window_start <= instant < window_end:
                normalized.append((instant, event))
        normalized.sort(key=lambda item: item[0])

        results: List[ReassignedEvent] = []
        for instant, event in normalized:
            old_day = self._business_day(instant)
            new_day = target._business_day(instant)
            if old_day.date != new_day.date:
                results.append(
                    ReassignedEvent(
                        event,
                        Arrow.fromdatetime(instant),
                        old_day,
                        new_day,
                    )
                )
        return results


def _identity_key(event: Any) -> _DT_EXPR:
    if isinstance(event, (Arrow, dt_datetime)):
        return event
    raise TypeError(
        "Events are not datetimes; provide a key() returning the event's UTC instant."
    )


__all__ = [
    "BusinessDay",
    "BusinessDayAssignment",
    "BusinessDayPartitioner",
    "BusinessDayRule",
    "ReassignedEvent",
    "GAP_EARLIEST",
    "GAP_LATEST",
    "FOLD_FIRST",
    "FOLD_SECOND",
]
