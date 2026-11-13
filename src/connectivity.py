"""Segment connectivity (turn) tables for split road-network polylines.

Road-network sources often split one logical road into many polyline
parts.  Two endpoints may share the exact same coordinates yet belong to
different levels (e.g. an elevated road above a ground-level road), so
geometry alone cannot decide which segments meet at an intersection.

This module builds a connectivity (turn) table from:

- the two endpoints of every polyline (part),
- a per-record level attribute from the DBF (endpoints only meet on the
  same level),
- a per-record one-way attribute from the DBF (a one-way segment can
  only be entered at its start and left at its end),
- a per-endpoint coordinate tolerance supplied by the source.

Grouping rules
--------------
Each endpoint may be relocated anywhere inside its tolerance square
``[x - tol, x + tol] x [y - tol, y + tol]``.  A set of endpoints may
share an intersection only if a single position satisfies *all* of them
(the tolerance squares have a common intersection) and all endpoints
are on the same level.  Consequently endpoints are never chained into a
large junction through near-neighbour transitivity: if no common
position exists for the whole chain, it is split into several smaller
intersections instead of being merged.

The endpoints of one level are partitioned into the fewest possible
intersections.  When several partitions tie with the same minimal
number of intersections, turns allowed by *every* minimal partition are
reported as ``"confirmed"`` and turns allowed by only some of them are
reported as ``"pending"`` (still to be confirmed against the source).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass

__all__ = [
    "Segment",
    "Turn",
    "Intersection",
    "ConnectivityTable",
    "build_connectivity",
    "load_segments",
]

_START = "start"
_END = "end"

# An endpoint is identified by its segment id and which end it is.
EndpointKey = tuple[Hashable, str]

CONFIRMED = "confirmed"
PENDING = "pending"


@dataclass(frozen=True)
class Segment:
    """One directed-or-not polyline part of the road network.

    ``start``/``end`` are ``(x, y)`` endpoint coordinates.  ``oneway``
    segments may only be entered at ``start`` and left at ``end``;
    two-way segments may be entered and left at both ends.
    ``start_tol``/``end_tol`` are the per-endpoint coordinate deviations
    allowed by the source (the endpoint may be relocated anywhere inside
    the square of half-side ``tol`` around its recorded position).
    """

    seg_id: Hashable
    start: tuple[float, float]
    end: tuple[float, float]
    level: Hashable = 0
    oneway: bool = False
    start_tol: float = 0.0
    end_tol: float = 0.0


@dataclass(frozen=True)
class Turn:
    """A possible continuation from one segment onto another.

    ``status`` is ``"confirmed"`` when every minimal intersection
    partition allows the turn, or ``"pending"`` when only some of them
    do and the source still has to confirm it.
    """

    from_segment: Hashable
    to_segment: Hashable
    level: Hashable
    status: str


@dataclass(frozen=True)
class Intersection:
    """A junction cluster from the reference (first minimal) partition.

    ``point`` is a feasible common position: every member endpoint can
    be relocated to it within its own tolerance.
    """

    level: Hashable
    members: tuple[EndpointKey, ...]
    point: tuple[float, float]


@dataclass(frozen=True)
class ConnectivityTable:
    """Result of :func:`build_connectivity`.

    ``turns`` holds confirmed and pending turns.  ``intersections`` is
    the reference partition's junctions (clusters of two or more
    endpoints).  ``num_partitions`` counts the minimal intersection
    partitions that were found (per level, multiplied together), and
    ``exhaustive`` is ``False`` when enumeration hit the partition cap,
    meaning some "confirmed" turns might actually still be ambiguous.
    """

    turns: tuple[Turn, ...]
    intersections: tuple[Intersection, ...]
    num_partitions: int
    exhaustive: bool

    def confirmed_turns(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.status == CONFIRMED]

    def pending_turns(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.status == PENDING]

    def successors(
        self, seg_id: Hashable, confirmed_only: bool = False
    ) -> list[Hashable]:
        """Segment ids reachable from the exit end of ``seg_id``."""
        return [
            turn.to_segment
            for turn in self.turns
            if turn.from_segment == seg_id
            and not (confirmed_only and turn.status != CONFIRMED)
        ]


@dataclass
class _Endpoint:
    key: EndpointKey
    seg_id: Hashable
    x: float
    y: float
    tol: float
    entry: bool
    exit: bool


def _compatible(a: _Endpoint, b: _Endpoint) -> bool:
    """True when the two tolerance squares intersect."""
    return abs(a.x - b.x) <= a.tol + b.tol and abs(a.y - b.y) <= a.tol + b.tol


def _components(compatible: list[list[bool]]) -> list[list[int]]:
    """Connected components of the pairwise-compatibility graph."""
    n = len(compatible)
    seen = [False] * n
    comps = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for u in range(n):
                if compatible[v][u] and not seen[u]:
                    seen[u] = True
                    stack.append(u)
        comps.append(sorted(comp))
    return comps


def _min_colorings(conflict: list[frozenset], cap: int) -> tuple[list[tuple], bool]:
    """All minimum colourings of the conflict graph (canonical form).

    Returns ``(colourings, exhaustive)``.  A colouring is a tuple of
    frozensets of vertex indices (the colour classes).  For box
    tolerances a colour class is exactly a set of endpoints whose
    tolerance squares share a common position, so each class is a valid
    intersection.
    """
    n = len(conflict)
    if n == 0:
        return [()], True
    order = sorted(range(n), key=lambda v: (-len(conflict[v]), v))
    for k in range(1, n + 1):
        sols: list[tuple] = []
        colors = [-1] * n

        def backtrack(pos: int, ncolors: int) -> None:
            if len(sols) >= cap:
                return
            if pos == n:
                classes = [[] for _ in range(max(colors) + 1)]
                for v, c in enumerate(colors):
                    classes[c].append(v)
                sols.append(tuple(frozenset(c) for c in classes))
                return
            v = order[pos]
            forbidden = {colors[u] for u in conflict[v] if colors[u] >= 0}
            # Canonical form: colours are introduced in order, so only
            # colours 0..ncolors may be assigned here.
            for c in range(min(ncolors + 1, k)):
                if c not in forbidden:
                    colors[v] = c
                    backtrack(pos + 1, max(ncolors, c + 1))
                    colors[v] = -1

        backtrack(0, 0)
        if sols:
            return sols, len(sols) < cap
    return [()], False  # unreachable: k = n always colours the graph


def _partition_level(endpoints: list[_Endpoint], cap: int) -> tuple[list[tuple], bool]:
    """All minimal partitions of one level's endpoints into junctions."""
    n = len(endpoints)
    compatible = [
        [i == j or _compatible(endpoints[i], endpoints[j]) for j in range(n)]
        for i in range(n)
    ]
    partitions: list[tuple] = [()]
    exhaustive = True
    for comp in _components(compatible):
        local = {g: i for i, g in enumerate(comp)}
        conflict = [
            frozenset(local[u] for u in comp if u != v and not compatible[v][u])
            for v in comp
        ]
        colorings, comp_exhaustive = _min_colorings(conflict, cap)
        exhaustive = exhaustive and comp_exhaustive
        merged: list[tuple] = []
        for prefix in partitions:
            for coloring in colorings:
                merged.append(
                    prefix
                    + tuple(frozenset(comp[local] for local in cls) for cls in coloring)
                )
                if len(merged) >= cap:
                    break
            if len(merged) >= cap:
                break
        if len(merged) >= cap:
            exhaustive = False
        partitions = merged
    return partitions, exhaustive


def _turns_of_partition(partition: tuple, endpoints: list[_Endpoint]) -> set[tuple]:
    """``(from_segment, to_segment)`` pairs connected by this partition."""
    cluster_of = {}
    for ci, cluster in enumerate(partition):
        for ei in cluster:
            cluster_of[ei] = ci
    exits: dict[int, set] = {}
    entries: dict[int, set] = {}
    for ei, ep in enumerate(endpoints):
        ci = cluster_of[ei]
        if ep.exit:
            exits.setdefault(ci, set()).add(ep.seg_id)
        if ep.entry:
            entries.setdefault(ci, set()).add(ep.seg_id)
    turns = set()
    for ci, from_ids in exits.items():
        for to_id in entries.get(ci, ()):
            for from_id in from_ids:
                if from_id != to_id:  # no U-turn back onto the same segment
                    turns.add((from_id, to_id))
    return turns


def _common_point(cluster: tuple, endpoints: list[_Endpoint]) -> tuple:
    """A position reachable by every endpoint of the cluster."""
    lo_x = max(endpoints[ei].x - endpoints[ei].tol for ei in cluster)
    hi_x = min(endpoints[ei].x + endpoints[ei].tol for ei in cluster)
    lo_y = max(endpoints[ei].y - endpoints[ei].tol for ei in cluster)
    hi_y = min(endpoints[ei].y + endpoints[ei].tol for ei in cluster)
    return ((lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0)


def build_connectivity(
    segments: Iterable[Segment], partition_cap: int = 10000
) -> ConnectivityTable:
    """Build the segment connectivity (turn) table.

    ``partition_cap`` bounds how many minimal intersection partitions
    are enumerated per level; if it is hit, the result's ``exhaustive``
    flag is ``False`` and some "confirmed" turns may in fact be
    ambiguous.
    """
    by_level: dict[Hashable, list[_Endpoint]] = {}
    for seg in segments:
        sx, sy = seg.start
        ex, ey = seg.end
        eps = by_level.setdefault(seg.level, [])
        eps.append(
            _Endpoint(
                key=(seg.seg_id, _START),
                seg_id=seg.seg_id,
                x=sx,
                y=sy,
                tol=seg.start_tol,
                entry=True,
                exit=not seg.oneway,
            )
        )
        eps.append(
            _Endpoint(
                key=(seg.seg_id, _END),
                seg_id=seg.seg_id,
                x=ex,
                y=ey,
                tol=seg.end_tol,
                entry=not seg.oneway,
                exit=True,
            )
        )

    all_turns: list[Turn] = []
    intersections: list[Intersection] = []
    num_partitions = 1
    exhaustive = True
    for level in sorted(by_level, key=repr):
        endpoints = by_level[level]
        partitions, level_exhaustive = _partition_level(endpoints, partition_cap)
        exhaustive = exhaustive and level_exhaustive
        num_partitions *= len(partitions)

        confirmed: set | None = None
        possible: set = set()
        for partition in partitions:
            turns = _turns_of_partition(partition, endpoints)
            possible |= turns
            confirmed = turns if confirmed is None else confirmed & turns
        for from_id, to_id in sorted(possible, key=repr):
            status = CONFIRMED if (from_id, to_id) in confirmed else PENDING
            all_turns.append(Turn(from_id, to_id, level, status))

        # Reference junctions from the first minimal partition.
        for cluster in partitions[0]:
            if len(cluster) < 2:
                continue
            members = tuple(sorted((endpoints[ei].key for ei in cluster), key=repr))
            intersections.append(
                Intersection(level, members, _common_point(cluster, endpoints))
            )

    return ConnectivityTable(
        turns=tuple(all_turns),
        intersections=tuple(intersections),
        num_partitions=num_partitions,
        exhaustive=exhaustive,
    )


_ONEWAY_FORWARD = {"1", "FT", "F", "T", "Y", "YES", "ONEWAY", "FORWARD"}
_ONEWAY_REVERSE = {"-1", "TF", "R", "REVERSE", "BACKWARD"}


def _parse_oneway(value) -> tuple[bool, bool]:
    """Return ``(oneway, reversed)`` for a DBF one-way attribute value."""
    if value is None:
        return False, False
    if isinstance(value, str):
        text = value.strip().upper()
        if text in _ONEWAY_FORWARD:
            return True, False
        if text in _ONEWAY_REVERSE:
            return True, True
        return False, False
    return bool(value), False


def load_segments(
    source,
    level_field: str,
    oneway_field: str | None = None,
    tolerance_field: str | None = None,
    default_tolerance: float = 0.0,
    id_field: str | None = None,
) -> list[Segment]:
    """Load segments from a polyline shapefile (path or ``shapefile.Reader``).

    Every part of every polyline record becomes one :class:`Segment`.
    ``level_field`` names the DBF level attribute, ``oneway_field`` the
    one-way attribute (values like ``"FT"``/``"TF"``/``"1"``/``"-1"`` or
    booleans; anything else means two-way), and ``tolerance_field`` a
    per-record endpoint tolerance (falling back to
    ``default_tolerance``).  ``id_field`` names a DBF field to use as
    segment id; record indices are used otherwise.
    """
    import shapefile

    reader = (
        source if isinstance(source, shapefile.Reader) else shapefile.Reader(source)
    )
    segments: list[Segment] = []
    for rec_index, sr in enumerate(reader.iterShapeRecords()):
        record = sr.record
        base_id = record[id_field] if id_field else rec_index
        level = record[level_field]
        oneway = reversed_ = False
        if oneway_field:
            oneway, reversed_ = _parse_oneway(record[oneway_field])
        tol = default_tolerance
        if tolerance_field:
            value = record[tolerance_field]
            if value is not None:
                tol = float(value)

        shape = sr.shape
        points = shape.points
        starts = list(shape.parts) or [0]
        starts.append(len(points))
        for part_index in range(len(starts) - 1):
            part = points[starts[part_index] : starts[part_index + 1]]
            if len(part) < 2:
                continue
            seg_id: Hashable = base_id if len(starts) == 2 else (base_id, part_index)
            first, last = tuple(part[0][:2]), tuple(part[-1][:2])
            if reversed_:
                first, last = last, first
            segments.append(
                Segment(
                    seg_id=seg_id,
                    start=first,
                    end=last,
                    level=level,
                    oneway=oneway,
                    start_tol=tol,
                    end_tol=tol,
                )
            )
    return segments
