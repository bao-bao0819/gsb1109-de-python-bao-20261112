"""Assign UTC instants to possibly overlapping maintenance windows.

Windows are produced from :class:`Arrow <arrow.arrow.Arrow>` spans/ranges and
treated as **half-open** intervals ``[start, end)`` of *real* UTC instants.
The helpers in this module never sample time minute-by-minute: every boundary
is an actual instant, so changing a UTC offset only affects whether two
adjacent segments describe the same instant and never reshapes a segment.

The typical entry points are:

* :class:`MaintenanceWindow` -- one window, carrying a business key and a
  priority (higher wins).
* :class:`WindowAssignment` -- the compacted result, where every covered
  instant belongs to exactly one winning window, except for instants where
  two windows share the highest priority but have different business keys,
  which are reported separately as conflicts.
* :func:`assign_windows` -- build an assignment from a collection of windows.
* :meth:`WindowAssignment.reassign` -- re-run the assignment after locally
  changing a window, revoking only the assignments that actually changed.

"""

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime as dt_datetime
from datetime import timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from arrow.arrow import Arrow, _T_FRAMES

__all__ = [
    "MaintenanceWindow",
    "AssignmentSegment",
    "ConflictSegment",
    "WindowAssignment",
    "AssignmentDiff",
    "assign_windows",
]

_TZ_UTC = timezone.utc


def _utc_arrow(value: Union[Arrow, dt_datetime]) -> Arrow:
    """Return ``value`` as an :class:`Arrow` pinned to UTC.

    Naive inputs are assumed to already be UTC, mirroring ``Arrow``'s own
    default.  Aware inputs are converted to the same UTC instant.

    """
    if isinstance(value, Arrow):
        arrow_value = value
    else:
        arrow_value = Arrow.fromdatetime(value)
    if arrow_value.tzinfo is not None:
        arrow_value = arrow_value.to(_TZ_UTC)
    return arrow_value


@dataclass(frozen=True, init=False)
class MaintenanceWindow:
    """A half-open maintenance window ``[start, end)`` on the UTC timeline.

    :param key: business key identifying the owning system/team.  Segments are
        attributed to a window through this key.
    :param start: inclusive start instant.  Naive values are treated as UTC.
    :param end: exclusive end instant.  Naive values are treated as UTC.
    :param priority: ranking used when windows overlap; the window with the
        higher value wins.  Ties between different ``key`` values are conflicts.
    :param source: optional free-form pointer back to the span/range (or any
        caller object) this window was derived from.

    Both boundaries are normalized to UTC :class:`Arrow` instances, so the
    :attr:`start` and :attr:`end` fields always carry an ``Arrow`` even when the
    constructor receives a plain ``datetime``.

    """

    key: Any
    start: Arrow
    end: Arrow
    priority: int
    source: Any

    def __init__(
        self,
        key: Any,
        start: Union[Arrow, dt_datetime],
        end: Union[Arrow, dt_datetime],
        priority: int = 0,
        source: Any = None,
    ) -> None:
        start_arrow = _utc_arrow(start)
        end_arrow = _utc_arrow(end)
        if end_arrow <= start_arrow:
            raise ValueError(
                f"window {key!r} must have a start before its end "
                f"(got [{start_arrow}, {end_arrow}))."
            )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "start", start_arrow)
        object.__setattr__(self, "end", end_arrow)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "source", source)

    @classmethod
    def from_span(
        cls,
        key: Any,
        anchor: Union[Arrow, dt_datetime],
        frame: _T_FRAMES,
        priority: int = 0,
        source: Any = None,
        count: int = 1,
        **span_kwargs: Any,
    ) -> "MaintenanceWindow":
        """Build a window from an ``Arrow.span`` frame around ``anchor``.

        Windows are always half-open, so the frame is requested with inclusive
        bounds on both sides; the returned ceiling is the exact frame boundary
        and becomes the exclusive end of the window.

        """
        if not isinstance(anchor, Arrow):
            anchor = Arrow.fromdatetime(anchor)
        # Asking ``span`` for inclusive bounds on both sides yields the exact
        # frame boundary, whose ceiling is already the exclusive end.
        span_kwargs.pop("bounds", None)
        floor, ceil = anchor.span(frame, count=count, bounds="[]", **span_kwargs)
        return cls(key, floor, ceil, priority=priority, source=source)

    @classmethod
    def from_span_range(
        cls,
        key: Any,
        frame: _T_FRAMES,
        start: Union[Arrow, dt_datetime],
        end: Union[Arrow, dt_datetime],
        priority: int = 0,
        source: Any = None,
        **span_range_kwargs: Any,
    ) -> List["MaintenanceWindow"]:
        """Build one window per frame from an ``Arrow.span_range`` series.

        Every returned window shares ``key`` and ``source`` (defaulting to
        ``key``), so adjacent frames that stay owned by the same series are
        compacted into a single segment by :func:`assign_windows`.

        """
        series_source = key if source is None else source
        start_arrow = _utc_arrow(start)
        end_arrow = _utc_arrow(end)
        span_range_kwargs.pop("bounds", None)
        windows: List[MaintenanceWindow] = []
        for floor, ceil in Arrow.span_range(
            frame,
            start_arrow.datetime,
            end_arrow.datetime,
            bounds="[]",
            **span_range_kwargs,
        ):
            windows.append(
                cls(key, floor, ceil, priority=priority, source=series_source)
            )
        return windows

    @property
    def span(self) -> Tuple[Arrow, Arrow]:
        """The normalized ``(start, end)`` UTC half-open boundaries."""
        return self.start, self.end

    def contains(self, instant: Union[Arrow, dt_datetime]) -> bool:
        """Return whether ``instant`` falls inside ``[start, end)``."""
        instant = _utc_arrow(instant)
        return self.start <= instant < self.end


@dataclass(frozen=True)
class AssignmentSegment:
    """A compacted half-open region won by a single window.

    ``[start, end)`` is maximal: the winning window is constant across it and
    differs on either side.

    """

    start: Arrow
    end: Arrow
    window: MaintenanceWindow

    @property
    def key(self) -> Any:
        """Business key of the winning window."""
        return self.window.key

    @property
    def source(self) -> Any:
        """Source pointer of the winning window."""
        return self.window.source

    def contains(self, instant: Union[Arrow, dt_datetime]) -> bool:
        """Return whether ``instant`` falls inside this segment."""
        instant = _utc_arrow(instant)
        return self.start <= instant < self.end


@dataclass(frozen=True)
class ConflictSegment:
    """A half-open region where equal-priority windows with different keys tie.

    No unique winner exists across ``[start, end)``: every window in
    ``windows`` shares the highest priority and has a distinct business key.

    """

    start: Arrow
    end: Arrow
    windows: Tuple[MaintenanceWindow, ...]

    @property
    def priority(self) -> int:
        """Priority shared by every tied window."""
        return self.windows[0].priority

    @property
    def keys(self) -> Tuple[Any, ...]:
        """Distinct business keys involved in the conflict."""
        return tuple(window.key for window in self.windows)

    def contains(self, instant: Union[Arrow, dt_datetime]) -> bool:
        """Return whether ``instant`` falls inside this conflict."""
        instant = _utc_arrow(instant)
        return self.start <= instant < self.end


def _sweep(windows: List[MaintenanceWindow]) -> List[Dict[str, Any]]:
    """Partition the timeline into constant-coverage elementary regions.

    A sweep over all window boundaries yields, for each elementary half-open
    region, the active window set.  Adjacent regions are kept separate even when
    the active set is unchanged; :func:`_build_segments` compacts them once the
    winning window is known.

    """
    boundaries = sorted(
        {bound for window in windows for bound in (window.start, window.end)}
    )
    if not boundaries:
        return []

    regions: List[Dict[str, Any]] = []
    active: List[MaintenanceWindow] = []
    starts: Dict[Arrow, List[MaintenanceWindow]] = {}
    ends: Dict[Arrow, List[MaintenanceWindow]] = {}
    for window in windows:
        starts.setdefault(window.start, []).append(window)
        ends.setdefault(window.end, []).append(window)

    previous = boundaries[0]
    for boundary in boundaries:
        if boundary > previous and active:
            regions.append(
                {
                    "start": previous,
                    "end": boundary,
                    "active": tuple(active),
                }
            )
        for window in ends.get(boundary, ()):
            active.remove(window)
        active.extend(starts.get(boundary, ()))
        previous = boundary
    return regions


def _resolve(
    active: Tuple[MaintenanceWindow, ...],
) -> Tuple[int, List[MaintenanceWindow]]:
    """Return ``(best_priority, contenders)`` for an active window set."""
    best_priority = max(window.priority for window in active)
    contenders = [window for window in active if window.priority == best_priority]
    return best_priority, contenders


def _same_winner(left: MaintenanceWindow, right: MaintenanceWindow) -> bool:
    """Return whether two windows carry the same attribution.

    Two windows are treated as one continuous owner when they share both a
    business key and a source, e.g. consecutive frames from the same
    ``span_range`` series split only at an internal boundary.

    """
    return bool(left.key == right.key and left.source == right.source)


def _build_segments(
    windows: Iterable[MaintenanceWindow],
) -> Tuple[List[AssignmentSegment], List[ConflictSegment]]:
    """Resolve windows into compacted winner and conflict segments."""
    window_list = list(windows)
    regions = _sweep(window_list)

    winners: List[AssignmentSegment] = []
    conflicts: List[ConflictSegment] = []

    for region in regions:
        active = region["active"]
        _, contenders = _resolve(active)
        distinct_keys = {window.key for window in contenders}
        if len(distinct_keys) == 1:
            winner = contenders[0]
            if (
                winners
                and winners[-1].end == region["start"]
                and _same_winner(winners[-1].window, winner)
            ):
                # Extend the previous segment; the active set changed underneath
                # but the owning window (business key + source) did not.
                winners[-1] = AssignmentSegment(
                    winners[-1].start, region["end"], winner
                )
            else:
                winners.append(
                    AssignmentSegment(region["start"], region["end"], winner)
                )
        else:
            tied = tuple(sorted(contenders, key=lambda window: repr(window.key)))
            if (
                conflicts
                and conflicts[-1].windows == tied
                and conflicts[-1].end == region["start"]
            ):
                conflicts[-1] = ConflictSegment(
                    conflicts[-1].start, region["end"], tied
                )
            else:
                conflicts.append(ConflictSegment(region["start"], region["end"], tied))

    winners.sort(key=lambda segment: segment.start)
    conflicts.sort(key=lambda segment: segment.start)
    return winners, conflicts


@dataclass
class WindowAssignment:
    """Result of assigning instants to maintenance windows.

    :param windows: the windows the assignment was built from.
    :param segments: compacted, non-overlapping winner segments.
    :param conflicts: compacted regions with an unresolved same-priority tie.

    ``segments`` and ``conflicts`` are mutually exclusive: together they cover
    every instant that at least one window covers, each exactly once.

    """

    windows: Tuple[MaintenanceWindow, ...]
    segments: List[AssignmentSegment] = field(default_factory=list)
    conflicts: List[ConflictSegment] = field(default_factory=list)

    def winner_at(
        self, instant: Union[Arrow, dt_datetime]
    ) -> Optional[MaintenanceWindow]:
        """Return the unique window owning ``instant``, or ``None``.

        Conflicting and uncovered instants both return ``None``; use
        :meth:`conflict_at` to distinguish the two cases.

        """
        instant = _utc_arrow(instant)
        for segment in self.segments:
            if segment.contains(instant):
                return segment.window
        return None

    def conflict_at(
        self, instant: Union[Arrow, dt_datetime]
    ) -> Optional[ConflictSegment]:
        """Return the conflict covering ``instant``, if any."""
        instant = _utc_arrow(instant)
        for conflict in self.conflicts:
            if conflict.contains(instant):
                return conflict
        return None

    def is_covered(self, instant: Union[Arrow, dt_datetime]) -> bool:
        """Return whether any window (winner or conflict) covers ``instant``."""
        instant = _utc_arrow(instant)
        return (
            self.winner_at(instant) is not None
            or self.conflict_at(instant) is not None
        )

    def uncovered(self) -> List[Tuple[Arrow, Arrow]]:
        """Return half-open gaps inside the overall covered envelope.

        Useful for verifying that compaction leaves no instants unaccounted for
        inside the union of all windows.

        """
        covered = sorted(
            [
                (segment.start, segment.end)
                for segment in self.segments
            ]
            + [(conflict.start, conflict.end) for conflict in self.conflicts],
            key=lambda pair: pair[0],
        )
        gaps: List[Tuple[Arrow, Arrow]] = []
        cursor: Optional[Arrow] = None
        for start, end in covered:
            if cursor is not None and start > cursor:
                gaps.append((cursor, start))
            if cursor is None or end > cursor:
                cursor = end
        return gaps

    def reassign(
        self, windows: Iterable[MaintenanceWindow]
    ) -> "AssignmentDiff":
        """Re-run the assignment after locally changing some windows.

        Only the instants whose winner/conflict status actually changes are
        reported in the returned :class:`AssignmentDiff`; assignments outside
        the influence of the changed windows are left untouched.

        :param windows: the full, updated window collection.  Windows are
            matched by value: any window added, removed, or whose bounds,
            priority, or key differs from :attr:`windows` counts as a change,
            and the diff is clipped to the envelope of those changes.

        """
        new_windows = list(windows)
        changed = _changed_windows(list(self.windows), new_windows)
        affected_envelope = _envelope(changed)

        new_assignment = assign_windows(new_windows)

        revoked: List[Union[AssignmentSegment, ConflictSegment]] = []
        granted: List[Union[AssignmentSegment, ConflictSegment]] = []

        if affected_envelope is not None:
            env_start, env_end = affected_envelope
            old_affected = _clip_regions(
                self._regions_overlapping(env_start, env_end), env_start, env_end
            )
            new_affected = _clip_regions(
                new_assignment._regions_overlapping(env_start, env_end),
                env_start,
                env_end,
            )
            revoked = _subtract_regions(old_affected, new_affected)
            granted = _subtract_regions(new_affected, old_affected)

        return AssignmentDiff(
            assignment=new_assignment,
            revoked=revoked,
            granted=granted,
            changed_windows=tuple(changed),
        )

    def _regions_overlapping(
        self, start: Arrow, end: Arrow
    ) -> List[Union[AssignmentSegment, ConflictSegment]]:
        """Return winner/conflict regions overlapping ``[start, end)``.

        The result is merged back into chronological order so callers can rely
        on sorted, non-overlapping input (e.g. for binary search).

        """
        regions: List[Union[AssignmentSegment, ConflictSegment]] = []
        for segment in self.segments:
            if segment.start < end and segment.end > start:
                regions.append(segment)
        for conflict in self.conflicts:
            if conflict.start < end and conflict.end > start:
                regions.append(conflict)
        regions.sort(key=lambda region: region.start)
        return regions


@dataclass(frozen=True)
class AssignmentDiff:
    """The outcome of an incremental :meth:`WindowAssignment.reassign`.

    :param assignment: the fully rebuilt assignment.
    :param revoked: regions that lost their previous attribution.
    :param granted: regions that gained a new attribution.
    :param changed_windows: the windows added or removed relative to the
        previous assignment (a modified window appears as its old and new
        instances).

    ``revoked`` and ``granted`` are compacted half-open regions, clipped to the
    envelope of the changed windows and restricted to instants whose
    attribution genuinely differs.

    """

    assignment: WindowAssignment
    revoked: List[Union[AssignmentSegment, ConflictSegment]]
    granted: List[Union[AssignmentSegment, ConflictSegment]]
    changed_windows: Tuple[MaintenanceWindow, ...]

    @property
    def is_empty(self) -> bool:
        """Return whether the change left every attribution untouched."""
        return not self.revoked and not self.granted


def _envelope(
    windows: Iterable[MaintenanceWindow],
) -> Optional[Tuple[Arrow, Arrow]]:
    """Return the ``[min start, max end)`` envelope of ``windows``."""
    window_list = list(windows)
    if not window_list:
        return None
    return (
        min(window.start for window in window_list),
        max(window.end for window in window_list),
    )


def _changed_windows(
    old: List[MaintenanceWindow],
    new: List[MaintenanceWindow],
) -> List[MaintenanceWindow]:
    """Return windows added or removed between two collections.

    A modified window shows up as its old instance (removed) and its new
    instance (added).  Frozen-dataclass equality makes the value comparison
    sensitive to bounds, priority, key, and source.

    """
    old_remaining = list(old)
    added: List[MaintenanceWindow] = []
    for window in new:
        try:
            old_remaining.remove(window)
        except ValueError:
            added.append(window)
    # Anything left in old_remaining was removed.
    return added + old_remaining


def _clip_regions(
    regions: List[Union[AssignmentSegment, ConflictSegment]],
    start: Arrow,
    end: Arrow,
) -> List[Union[AssignmentSegment, ConflictSegment]]:
    """Truncate half-open regions to the envelope ``[start, end)``."""
    clipped: List[Union[AssignmentSegment, ConflictSegment]] = []
    for region in regions:
        clipped_start = max(region.start, start)
        clipped_end = min(region.end, end)
        if clipped_start < clipped_end:
            clipped.append(_with_bounds(region, clipped_start, clipped_end))
    return clipped


def _region_identity(
    region: Union[AssignmentSegment, ConflictSegment]
) -> Tuple[Any, ...]:
    """Canonical attribution of a region, for equality comparison.

    Attribution is ownership by business key, not the concrete window object:
    shrinking a window leaves its ownership of the instants it still covers
    untouched.  Conflicts are identified by the full set of tied business keys.

    """
    if isinstance(region, AssignmentSegment):
        return ("winner", region.key)
    return ("conflict",) + tuple(sorted(region.keys, key=repr))


def _subtract_regions(
    source: List[Union[AssignmentSegment, ConflictSegment]],
    other: List[Union[AssignmentSegment, ConflictSegment]],
) -> List[Union[AssignmentSegment, ConflictSegment]]:
    """Return parts of ``source`` whose attribution differs from ``other``.

    Both inputs are half-open, non-overlapping, and time-ordered.  The result is
    compacted: adjacent fragments with the same attribution are rejoined.

    """
    boundaries = sorted(
        {
            bound
            for region in source + other
            for bound in (region.start, region.end)
        }
    )
    if not boundaries:
        return []

    source_starts = [region.start for region in source]
    other_starts = [region.start for region in other]

    def find(
        regions: List[Union[AssignmentSegment, ConflictSegment]],
        starts: List[Arrow],
        instant: Arrow,
    ) -> Optional[Union[AssignmentSegment, ConflictSegment]]:
        index = bisect_right(starts, instant) - 1
        if index >= 0 and regions[index].start <= instant < regions[index].end:
            return regions[index]
        return None

    fragments: List[
        Tuple[Arrow, Arrow, Union[AssignmentSegment, ConflictSegment]]
    ] = []
    previous = boundaries[0]
    for boundary in boundaries[1:]:
        source_region = find(source, source_starts, previous)
        other_region = find(other, other_starts, previous)
        if source_region is not None and (
            other_region is None
            or _region_identity(source_region) != _region_identity(other_region)
        ):
            fragments.append((previous, boundary, source_region))
        previous = boundary

    return _compact_fragments(fragments)


def _compact_fragments(
    fragments: List[
        Tuple[Arrow, Arrow, Union[AssignmentSegment, ConflictSegment]]
    ]
) -> List[Union[AssignmentSegment, ConflictSegment]]:
    """Join adjacent fragments sharing an attribution and touching instants."""
    compacted: List[Union[AssignmentSegment, ConflictSegment]] = []
    for start, end, region in fragments:
        if compacted and compacted[-1].end == start and _same_attribution(
            compacted[-1], region
        ):
            previous = compacted.pop()
            compacted.append(_with_end(previous, end))
        else:
            compacted.append(_with_end(region, end) if region.end != end else region)
    return compacted


def _same_attribution(
    left: Union[AssignmentSegment, ConflictSegment],
    right: Union[AssignmentSegment, ConflictSegment],
) -> bool:
    """Return whether two regions describe the same attribution."""
    return _region_identity(left) == _region_identity(right)


def _with_end(
    region: Union[AssignmentSegment, ConflictSegment],
    end: Arrow,
) -> Union[AssignmentSegment, ConflictSegment]:
    """Return a copy of ``region`` ending at ``end``."""
    return _with_bounds(region, region.start, end)


def _with_bounds(
    region: Union[AssignmentSegment, ConflictSegment],
    start: Arrow,
    end: Arrow,
) -> Union[AssignmentSegment, ConflictSegment]:
    """Return a copy of ``region`` with the given half-open bounds."""
    if isinstance(region, AssignmentSegment):
        return AssignmentSegment(start, end, region.window)
    return ConflictSegment(start, end, region.windows)


def assign_windows(
    windows: Iterable[MaintenanceWindow],
) -> WindowAssignment:
    """Assign every covered UTC instant to a unique winning window.

    :param windows: maintenance windows, produced from Arrow spans/ranges or
        built directly from instants.  They may overlap arbitrarily.
    :returns: a :class:`WindowAssignment` holding compacted half-open winner
        segments and any same-priority conflict regions.

    Higher ``priority`` wins at overlaps.  When windows with different business
    keys tie for the highest priority, the overlapping instants are reported as
    a :class:`ConflictSegment` instead of being arbitrarily assigned.

    Usage::

        >>> import arrow
        >>> from arrow.windows import MaintenanceWindow, assign_windows
        >>> a = MaintenanceWindow("a", arrow.get(2024, 1, 1, tzinfo="UTC"),
        ...                       arrow.get(2024, 1, 1, 3, tzinfo="UTC"), priority=1)
        >>> b = MaintenanceWindow("b", arrow.get(2024, 1, 1, 2, tzinfo="UTC"),
        ...                       arrow.get(2024, 1, 1, 4, tzinfo="UTC"), priority=2)
        >>> assignment = assign_windows([a, b])
        >>> for segment in assignment.segments:
        ...     print(segment.start, segment.end, segment.key)
        ...
        2024-01-01T00:00:00+00:00 2024-01-01T02:00:00+00:00 a
        2024-01-01T02:00:00+00:00 2024-01-01T04:00:00+00:00 b

    """
    window_tuple = tuple(windows)
    segments, conflicts = _build_segments(window_tuple)
    return WindowAssignment(
        windows=window_tuple,
        segments=segments,
        conflicts=conflicts,
    )
