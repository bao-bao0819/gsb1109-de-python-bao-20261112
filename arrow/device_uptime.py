"""Device online-time accounting from start/stop telemetry events.

The ward (病区) reports equipment start/stop events from several sources.  This
module turns that noisy event stream into per-device, per-natural-day exposure
seconds in the ward's local timezone, using :class:`arrow.Arrow` to draw the
day boundaries so DST transitions and fixed-offset wards are both handled
correctly.

Telemetry rules supported:

* Every event carries a per-device monotonic ``sequence`` (号).  An exact
  resend of the same sequence is a no-op; a resend with a different payload is
  rejected as a conflicting duplicate.
* A ``revoke`` action voids the device's previously accepted event with that
  sequence, e.g. retracting a false stop so the device stays online.
* A report may carry ``corrected_sequence`` to supersede an earlier event.
* Each source independently asserts a state at an instant.  When two sources
  assert opposite states for the same instant the interval is *conflicting*
  and its seconds are reported in a separate column, never double counted.
* Each device only ever contributes exposure once for a single instant.

The daily figures produced by :func:`summarize` reconcile exactly with a
direct UTC-interval measurement from :func:`total_interval_seconds`: summing
the adjacent local days gives the same totals as measuring the merged UTC
spans.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, tzinfo as dt_tzinfo, timezone
from typing import Any, Dict, List, Optional, Sequence as Seq, Set, Tuple, Union

import arrow
from arrow import parser as arrow_parser

STATE_ONLINE = "online"
STATE_OFFLINE = "offline"
STATE_CONFLICT = "conflict"
STATES = (STATE_ONLINE, STATE_OFFLINE)

ACTION_REPORT = "report"
ACTION_REVOKE = "revoke"
ACTIONS = (ACTION_REPORT, ACTION_REVOKE)

_EVENT_TIME = Union[arrow.Arrow, datetime, int, float]


@dataclass(frozen=True)
class DeviceEvent:
    """A single start/stop telemetry record.

    :param device_id: the equipment the record belongs to.
    :param sequence: per-device monotonic event number (号).
    :param at: event time as an :class:`~arrow.arrow.Arrow`, an aware/naive
        ``datetime`` (naive values are localized to the ward timezone), or a
        numeric UTC timestamp.
    :param action: ``"report"`` asserts ``state``; ``"revoke"`` voids the
        previously accepted event with the same ``sequence``.
    :param state: ``"online"`` or ``"offline"`` for reports, ``None`` for
        revocations.
    :param source: identifier of the reporting feed; each source's state is
        tracked independently.
    :param corrected_sequence: when set, the report supersedes the device's
        earlier event with that sequence number.
    """

    device_id: str
    sequence: int
    at: _EVENT_TIME
    action: str = ACTION_REPORT
    state: Optional[str] = None
    source: str = "default"
    corrected_sequence: Optional[int] = None


@dataclass(frozen=True)
class DailyUptime:
    """Whole-second exposure of one device on one local natural day."""

    device_id: str
    day: date
    online_seconds: int
    conflict_seconds: int


@dataclass(frozen=True)
class _Segment:
    """A maximal interval over which the merged device status is constant."""

    start: arrow.Arrow
    end: Optional[arrow.Arrow]
    state: str
    conflict: bool


def _resolve_tz(tz_expr: Any) -> dt_tzinfo:
    if tz_expr is None:
        return timezone.utc
    if isinstance(tz_expr, dt_tzinfo):
        return tz_expr
    try:
        return arrow_parser.TzinfoParser.parse(tz_expr)
    except arrow_parser.ParserError:
        raise ValueError(f"{tz_expr!r} not recognized as a timezone.")


def _to_arrow(value: _EVENT_TIME, ward_tz: dt_tzinfo) -> arrow.Arrow:
    if isinstance(value, arrow.Arrow):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return arrow.Arrow.fromdatetime(value).replace(tzinfo=ward_tz)
        return arrow.Arrow.fromdatetime(value)
    if isinstance(value, bool):
        raise TypeError(f"Unsupported event time {value!r}.")
    if isinstance(value, (int, float)):
        return arrow.Arrow.utcfromtimestamp(value)
    raise TypeError(
        f"Unsupported event time {value!r}; use an Arrow, datetime or numeric timestamp."
    )


def _instant(value: _EVENT_TIME) -> float:
    if isinstance(value, arrow.Arrow):
        return value.timestamp()
    if isinstance(value, datetime):
        return value.timestamp()
    return float(value)


def _sort_key(event: DeviceEvent) -> Tuple[str, float, int, str]:
    return event.device_id, _instant(event.at), event.sequence, event.source


def _validate(event: DeviceEvent) -> None:
    if not isinstance(event.device_id, str) or not event.device_id:
        raise ValueError("device_id must be a non-empty string.")
    if isinstance(event.at, bool) or not isinstance(
        event.at, (arrow.Arrow, datetime, int, float)
    ):
        raise TypeError(
            f"at must be an Arrow, datetime or numeric timestamp, got {event.at!r}."
        )
    if not isinstance(event.sequence, int) or isinstance(event.sequence, bool):
        raise ValueError("sequence must be a non-negative integer.")
    if event.sequence < 0:
        raise ValueError("sequence must be a non-negative integer.")
    if event.action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}, got {event.action!r}.")
    if not isinstance(event.source, str) or not event.source:
        raise ValueError("source must be a non-empty string.")
    if event.action == ACTION_REPORT:
        if event.state not in STATES:
            raise ValueError(
                f"report state must be one of {STATES}, got {event.state!r}."
            )
    elif event.state is not None:
        raise ValueError("revoke events must not carry a state.")
    if event.corrected_sequence is not None:
        if event.action != ACTION_REPORT:
            raise ValueError("only report events may carry corrected_sequence.")
        if (
            not isinstance(event.corrected_sequence, int)
            or isinstance(event.corrected_sequence, bool)
            or event.corrected_sequence < 0
        ):
            raise ValueError("corrected_sequence must be a non-negative integer.")


def normalize_events(events: Seq[DeviceEvent]) -> List[DeviceEvent]:
    """Validate, de-duplicate and order the raw event stream.

    Processing is order-independent: revocations and corrections may arrive
    before the event they target.  Resends of an already accepted
    ``(device, sequence)`` pair are dropped when identical and raise
    :class:`ValueError` when their payload differs.  Corrections replace the
    targeted event and revocations remove it; when a correction is itself
    revoked, the superseded target stays in the stream.  The returned events
    are ordered per device by event time, sequence and source.
    """

    reports: Dict[Tuple[str, int], DeviceEvent] = {}
    revokes: Set[Tuple[str, int]] = set()
    for raw in events:
        if not isinstance(raw, DeviceEvent):
            raise TypeError(f"expected DeviceEvent, got {type(raw).__name__}.")
        _validate(raw)

        key = (raw.device_id, raw.sequence)
        if raw.action == ACTION_REVOKE:
            revokes.add(key)
            reports.pop(key, None)
            continue

        if key in revokes:
            continue
        if key in reports:
            earlier = reports[key]
            same_payload = (
                _instant(earlier.at) == _instant(raw.at)
                and earlier.state == raw.state
                and earlier.source == raw.source
                and earlier.corrected_sequence == raw.corrected_sequence
            )
            if not same_payload:
                raise ValueError(
                    f"Conflicting resend of sequence {raw.sequence} for device "
                    f"{raw.device_id!r}; use corrected_sequence to supersede it."
                )
            continue
        reports[key] = raw

    for key in revokes:
        reports.pop(key, None)

    for event in list(reports.values()):
        if event.corrected_sequence is not None:
            reports.pop((event.device_id, event.corrected_sequence), None)

    return sorted(reports.values(), key=_sort_key)


def build_state_timeline(events: Seq[DeviceEvent]) -> Dict[str, List[_Segment]]:
    """Merge the normalized events into constant-status segments per device.

    Reports from every source are folded together at each event instant:
    identical states produce a clean segment, opposite states produce a
    conflict segment.  A trailing segment with no later event is left open
    (``end is None``).
    """

    ordered = normalize_events(events)
    by_device: Dict[str, List[DeviceEvent]] = {}
    for event in ordered:
        by_device.setdefault(event.device_id, []).append(event)

    timelines: Dict[str, List[_Segment]] = {}
    for device_id, device_events in by_device.items():
        source_states: Dict[str, str] = {}
        segments: List[_Segment] = []
        open_start: Optional[arrow.Arrow] = None
        open_state: Optional[str] = None
        open_conflict = False

        for event in device_events:
            at = arrow.Arrow.utcfromtimestamp(_instant(event.at))
            if open_start is not None and at > open_start:
                segments.append(
                    _Segment(open_start, at, open_state, open_conflict)  # type: ignore[arg-type]
                )

            source_states[event.source] = event.state  # type: ignore[assignment]
            distinct = set(source_states.values())
            open_conflict = len(distinct) > 1
            open_state = STATE_CONFLICT if open_conflict else next(iter(distinct))
            open_start = at

        if open_start is not None:
            segments.append(
                _Segment(open_start, None, open_state, open_conflict)  # type: ignore[arg-type]
            )

        if segments:
            timelines[device_id] = segments

    return timelines


def _seconds(lo: arrow.Arrow, hi: arrow.Arrow) -> int:
    return hi.int_timestamp - lo.int_timestamp


def _local_days(
    window_start: arrow.Arrow, window_end: arrow.Arrow
) -> List[Tuple[arrow.Arrow, arrow.Arrow, date]]:
    days: List[Tuple[arrow.Arrow, arrow.Arrow, date]] = []
    cursor = window_start.floor("day")
    while cursor < window_end:
        day_start, day_end = cursor.span("day", bounds="[]")
        days.append((day_start, day_end, cursor.date()))
        cursor = day_end
    return days


def _prepare(
    events: Seq[DeviceEvent], ward_tz: Any
) -> Tuple[Dict[str, List[_Segment]], dt_tzinfo]:
    tzinfo = _resolve_tz(ward_tz)
    prepared = [
        DeviceEvent(
            device_id=e.device_id,
            sequence=e.sequence,
            at=_to_arrow(e.at, tzinfo),
            action=e.action,
            state=e.state,
            source=e.source,
            corrected_sequence=e.corrected_sequence,
        )
        for e in events
    ]
    return build_state_timeline(prepared), tzinfo


def _default_window(
    timelines: Dict[str, List[_Segment]],
) -> Tuple[arrow.Arrow, arrow.Arrow]:
    starts = [seg.start for segs in timelines.values() for seg in segs]
    ends = [
        seg.end
        for segs in timelines.values()
        for seg in segs
        if seg.end is not None
    ]
    if not starts or not ends:
        raise ValueError(
            "an explicit end horizon is required when the timeline never closes."
        )
    return min(starts), max(ends)


def summarize(
    events: Seq[DeviceEvent],
    ward_tz: Any,
    start: Optional[_EVENT_TIME] = None,
    end: Optional[_EVENT_TIME] = None,
) -> List[DailyUptime]:
    """Return per-device online/conflict seconds for each local natural day.

    :param events: raw telemetry records (order does not matter).
    :param ward_tz: timezone expression for the ward's local natural days.
    :param start: optional report window start (inclusive); naive values use
        the ward timezone.
    :param end: optional report window end (exclusive); open-ended segments
        are only counted when this horizon is supplied.

    One :class:`DailyUptime` row is emitted for every device and every local
    day intersecting the window, including zero-exposure days.  Conflict
    seconds and online seconds are mutually exclusive columns.
    """

    timelines, tzinfo = _prepare(events, ward_tz)
    if not timelines:
        return []

    if start is not None and end is not None:
        window_start = _to_arrow(start, tzinfo).to(tzinfo)
        window_end = _to_arrow(end, tzinfo).to(tzinfo)
    elif start is None and end is None:
        window_start, window_end = (
            boundary.to(tzinfo) for boundary in _default_window(timelines)
        )
    else:
        raise ValueError("start and end must be provided together.")
    if window_end <= window_start:
        return []

    days = _local_days(window_start, window_end)
    rows: List[DailyUptime] = []
    for device_id in sorted(timelines):
        online_by_day: Dict[date, int] = {}
        conflict_by_day: Dict[date, int] = {}
        for segment in timelines[device_id]:
            for day_start, day_end, day in days:
                lo = segment.start if segment.start > day_start else day_start
                if lo < window_start:
                    lo = window_start
                hi = window_end if segment.end is None else segment.end
                if hi > day_end:
                    hi = day_end
                if hi > window_end:
                    hi = window_end
                if hi <= lo:
                    continue
                if segment.conflict:
                    conflict_by_day[day] = conflict_by_day.get(day, 0) + _seconds(
                        lo, hi
                    )
                elif segment.state == STATE_ONLINE:
                    online_by_day[day] = online_by_day.get(day, 0) + _seconds(lo, hi)
        for _, _, day in days:
            rows.append(
                DailyUptime(
                    device_id,
                    day,
                    online_by_day.get(day, 0),
                    conflict_by_day.get(day, 0),
                )
            )

    return rows


def total_interval_seconds(
    events: Seq[DeviceEvent],
    ward_tz: Any = None,
    start: Optional[_EVENT_TIME] = None,
    end: Optional[_EVENT_TIME] = None,
) -> Dict[str, Dict[str, int]]:
    """Measure exposure directly over merged UTC intervals.

    Independent cross-check for :func:`summarize`: the segments are clipped to
    the reporting window and summed in absolute time without any day
    splitting, so the totals must equal the sum of adjacent local days.  With
    no explicit window both functions use the same default window (first event
    to the last finite segment end); an explicit ``end`` horizon is required
    to include open-ended segments.
    """

    timelines, tzinfo = _prepare(events, ward_tz)
    if not timelines:
        return {}

    if start is not None and end is not None:
        window_start = _to_arrow(start, tzinfo)
        window_end = _to_arrow(end, tzinfo)
    elif start is None and end is None:
        window_start, window_end = _default_window(timelines)
    else:
        raise ValueError("start and end must be provided together.")

    totals: Dict[str, Dict[str, int]] = {}
    for device_id, segments in timelines.items():
        online = 0
        conflict = 0
        for segment in segments:
            if segment.end is not None and segment.end <= window_start:
                continue
            lo = segment.start if segment.start > window_start else window_start
            hi = window_end if segment.end is None else segment.end
            if hi > window_end:
                hi = window_end
            if hi <= lo:
                continue
            if segment.conflict:
                conflict += _seconds(lo, hi)
            elif segment.state == STATE_ONLINE:
                online += _seconds(lo, hi)
        totals[device_id] = {
            "online_seconds": online,
            "conflict_seconds": conflict,
        }

    return totals
