"""
Temporal constraint solving over integer-valued events.

Each event has a discrete set of admissible integer times, represented by a
portion.Interval whose atomic intervals are closed integer ranges. Gaps inside
such an interval set model times excluded by power outages or other
unavailability periods. Those gaps are structural: a feasible schedule only
ever picks integer times that truly belong to the admissible set, and the
gaps are never silently filled with continuous time.

Three kinds of binary records are supported on top of the per-event domains:

 - precedence ("a" happens no later than "b");
 - minimum gap ("b" happens at least ``d`` time units after "a");
 - maximum wait ("b" happens at most ``d`` time units after "a").

All of them reduce to difference constraints ``t[b] - t[a] >= w`` on integer
variables. The module decides, for a pair of events, whether their order is
necessary, reversible, forced to be simultaneous, or whether the whole set of
records is inconsistent. It exhibits concrete feasible schedules supporting
each conclusion and, when no schedule exists, reports the conflicting records.
"""

from collections import namedtuple

from .const import Bound, inf
from .interval import Interval
from .func import closed, closedopen, openclosed

__all__ = [
    "TemporalConstraint",
    "TemporalModel",
    "OrderResult",
    "InconsistentRecords",
    "make_domain",
    "before",
    "min_gap",
    "max_wait",
]


# A single binary record. ``kind`` is one of "before", "min_gap" or
# "max_wait" and ``value`` carries the integer distance (unused for "before").
TemporalConstraint = namedtuple("TemporalConstraint", ["kind", "a", "b", "value"])


def before(a, b):
    """Event ``a`` happens no later than event ``b`` (``t[b] - t[a] >= 0``)."""
    return TemporalConstraint("before", a, b, 0)


def min_gap(a, b, distance):
    """Event ``b`` happens at least ``distance`` time units after ``a``."""
    distance = _check_distance(distance)
    return TemporalConstraint("min_gap", a, b, distance)


def max_wait(a, b, distance):
    """Event ``b`` happens at most ``distance`` time units after ``a``."""
    distance = _check_distance(distance)
    return TemporalConstraint("max_wait", a, b, distance)


def _check_distance(distance):
    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
        raise ValueError("distance must be a non-negative integer")
    return distance


def make_domain(*ranges):
    """
    Build an admissible-time domain from closed integer ranges.

    Each argument is either an integer (a singleton time) or a pair
    ``(lower, upper)`` of finite integers. Disconnected ranges are kept
    disconnected, so the times in between remain excluded (e.g. power cuts).

    >>> make_domain(1, (3, 4), (8, 10))
    [1] | [3,4] | [8,10]
    """
    atomics = []
    for rng in ranges:
        if isinstance(rng, int) and not isinstance(rng, bool):
            atomics.append(closed(rng, rng))
        else:
            lower, upper = rng
            atomics.append(closed(lower, upper))
    return Interval(*atomics)


class InconsistentRecords(ValueError):
    """Raised when the temporal records admit no feasible schedule."""

    def __init__(self, conflicts):
        self.conflicts = conflicts
        super().__init__(
            "inconsistent temporal records: "
            + "; ".join(_format_conflict(c) for c in conflicts)
        )


# Verdict for the relative order of two events:
#  - "necessary": one event necessarily happens (strictly) before the other;
#  - "reversible": both strict orders admit a feasible schedule;
#  - "simultaneous": the two events are forced to share the same time;
#  - "infeasible": the whole set of records has no solution.
OrderResult = namedtuple(
    "OrderResult", ["verdict", "earlier", "schedule", "reverse_schedule", "conflicts"]
)


class TemporalModel:
    """
    A set of integer-time events with admissible-time domains and temporal
    records relating them.

    :param domains: mapping of event to a portion.Interval of closed integer
        ranges (see :func:`make_domain`).
    :param constraints: iterable of :class:`TemporalConstraint` instances.
    """

    def __init__(self, domains, constraints=()):
        self._domains = {}
        for event, domain in domains.items():
            self._domains[event] = self._normalize_domain(event, domain)
        self._constraints = []
        for constraint in constraints:
            self.add(constraint)
        self._closure = None
        self._cycle = None
        self._reduced = None
        self._conflicts = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_domain(event, domain):
        if not isinstance(domain, Interval):
            raise TypeError(f"domain of event {event!r} must be a portion.Interval")
        atomics = []
        for atomic in domain:
            lower, upper = atomic.lower, atomic.upper
            if lower in (-inf, inf) or upper in (-inf, inf):
                raise ValueError(
                    f"domain of event {event!r} must use finite integer bounds"
                )
            if atomic.left is Bound.OPEN:
                lower += 1
            if atomic.right is Bound.OPEN:
                upper -= 1
            if lower <= upper:
                atomics.append(closed(lower, upper))
        return Interval(*atomics)

    def add(self, constraint):
        """Register a :class:`TemporalConstraint` and return ``self``."""
        if not isinstance(constraint, TemporalConstraint):
            raise TypeError(
                "constraint must be built with before(), min_gap() or max_wait()"
            )
        if constraint.a not in self._domains:
            raise KeyError(f"unknown event {constraint.a!r}")
        if constraint.b not in self._domains:
            raise KeyError(f"unknown event {constraint.b!r}")
        self._constraints.append(constraint)
        self._closure = None
        self._cycle = None
        self._reduced = None
        self._conflicts = None
        return self

    def add_before(self, a, b):
        return self.add(before(a, b))

    def add_min_gap(self, a, b, distance):
        return self.add(min_gap(a, b, distance))

    def add_max_wait(self, a, b, distance):
        return self.add(max_wait(a, b, distance))

    @property
    def events(self):
        return tuple(self._domains)

    @property
    def constraints(self):
        return tuple(self._constraints)

    def domain(self, event):
        """Original admissible-time domain of ``event`` (gaps preserved)."""
        return self._domains[event]

    # ------------------------------------------------------------------ #
    # Difference-constraint closure
    # ------------------------------------------------------------------ #
    def _build_closure(self):
        events = self.events
        n = len(events)
        index = {event: i for i, event in enumerate(events)}
        neg = [[-inf] * n for _ in range(n)]
        pred = [[None] * n for _ in range(n)]
        for i in range(n):
            neg[i][i] = 0
        for constraint in self._constraints:
            i, j = index[constraint.a], index[constraint.b]
            if constraint.kind == "max_wait":
                # t[b] - t[a] <= d  <=>  t[a] - t[b] >= -d
                weight, head, tail = -constraint.value, j, i
            else:
                weight = constraint.value
                head, tail = i, j
            if weight > neg[head][tail]:
                neg[head][tail] = weight
                pred[head][tail] = head
        # Floyd-Warshall on the lower-bound differences, tracking predecessors
        # so that a positive diagonal (infeasible cycle) can be explained.
        for k in range(n):
            for i in range(n):
                if neg[i][k] == -inf:
                    continue
                for j in range(n):
                    if neg[k][j] == -inf:
                        continue
                    candidate = neg[i][k] + neg[k][j]
                    if candidate > neg[i][j]:
                        neg[i][j] = candidate
                        pred[i][j] = pred[k][j]

        cycle = None
        for i in range(n):
            if neg[i][i] > 0:
                cycle = self._cycle_conflict(i, pred)
                break
        self._closure = (index, neg, pred)
        self._cycle = cycle

    def _ensure_closure(self):
        if self._closure is None:
            self._build_closure()
        return self._closure

    def _edge_constraints(self, source, target):
        """Original records that imply a lower-bound edge source -> target."""
        found = []
        for constraint in self._constraints:
            if constraint.kind == "max_wait":
                head, tail = constraint.b, constraint.a
            else:
                head, tail = constraint.a, constraint.b
            if head == source and tail == target:
                found.append(constraint)
        return found

    def _cycle_conflict(self, i, pred):
        events = self.events
        path = [i]
        current = pred[i][i]
        guard = 0
        while current != i and current is not None and guard <= len(events):
            path.append(current)
            current = pred[i][current]
            guard += 1
        path.append(i)
        path.reverse()
        involved = []
        for source, target in zip(path, path[1:]):
            involved.extend(self._edge_constraints(source, target))
        if not involved:
            involved = list(self._constraints)
        return {
            "kind": "cycle",
            "events": [events[node] for node in path],
            "constraints": involved,
        }

    # ------------------------------------------------------------------ #
    # Domain propagation (gap-aware)
    # ------------------------------------------------------------------ #
    def _propagate(self, domains):
        """
        Tighten the given domains with the closed difference constraints.

        Raises InconsistentRecords when an event loses every admissible time,
        distinguishing numeric contradictions from times trapped in excluded
        gaps. Returns the reduced domains otherwise.
        """
        for event, domain in domains.items():
            if domain.empty:
                raise InconsistentRecords(
                    [{"kind": "empty_domain", "event": event}]
                )

        index, neg, _ = self._ensure_closure()
        events = self.events
        n = len(events)
        reduced = dict(domains)

        while True:
            changed = False
            for j in range(n):
                event = events[j]
                low, low_source = -inf, None
                high, high_source = inf, None
                for i in range(n):
                    if neg[i][j] != -inf:
                        candidate = reduced[events[i]].lower + neg[i][j]
                        if candidate > low:
                            low, low_source = candidate, i
                    # t[j] - t[i] <= w  <=>  t[i] - t[j] >= -w
                    if neg[j][i] != -inf:
                        candidate = reduced[events[i]].upper - neg[j][i]
                        if candidate < high:
                            high, high_source = candidate, i

                if low > high:
                    conflict = self._numeric_conflict(
                        event, low, high, low_source, high_source
                    )
                    raise InconsistentRecords([conflict])

                window = reduced[event] & closed(low, high)
                if window.empty:
                    raise InconsistentRecords(
                        [
                            self._gap_conflict(
                                event, low, high, low_source, high_source
                            )
                        ]
                    )
                if window != reduced[event]:
                    reduced[event] = window
                    changed = True
            if not changed:
                return reduced

    def _numeric_conflict(self, event, low, high, low_source, high_source):
        constraint = self._binding_constraint(low_source, high_source, event)
        return {
            "kind": "domain_emptied",
            "event": event,
            "constraint": constraint,
            "window": closed(low, high),
        }

    def _gap_conflict(self, event, low, high, low_source, high_source):
        constraint = self._binding_constraint(low_source, high_source, event)
        return {
            "kind": "excluded_gap",
            "event": event,
            "constraint": constraint,
            "window": closed(low, high),
            "admissible": self._domains[event],
        }

    def _binding_constraint(self, low_source, high_source, event):
        events = self.events
        for source in (low_source, high_source):
            if source is None:
                continue
            edge = self._edge_constraints(events[source], event)
            if edge:
                return edge[0]
            # An upper bound t[j] <= t[i] + w corresponds to edge j -> i.
            edge = self._edge_constraints(event, events[source])
            if edge:
                return edge[0]
        return None

    def solve(self):
        """
        Propagate every record and return the reduced admissible domains.

        Raises :class:`InconsistentRecords` (with an explanation of the
        conflicting records) when no schedule exists. Reduced domains keep
        their disconnected structure: excluded gaps are never filled.
        """
        if self._conflicts is not None:
            raise InconsistentRecords(self._conflicts)
        if self._reduced is None:
            self._ensure_closure()
            if self._cycle is not None:
                self._conflicts = [self._cycle]
                raise InconsistentRecords(self._conflicts)
            try:
                self._reduced = self._propagate(dict(self._domains))
            except InconsistentRecords as error:
                self._conflicts = error.conflicts
                raise
        return dict(self._reduced)

    def is_consistent(self):
        """Return True when at least one feasible schedule exists."""
        try:
            self.solve()
        except InconsistentRecords:
            return False
        return True

    def conflicts(self):
        """Return the conflict explanations, or an empty list if consistent."""
        try:
            self.solve()
        except InconsistentRecords as error:
            return list(error.conflicts)
        return []

    def reduced_domain(self, event):
        return self.solve()[event]

    # ------------------------------------------------------------------ #
    # Schedule construction
    # ------------------------------------------------------------------ #
    def feasible_schedule(self, prefer="earliest"):
        """
        Return a feasible schedule mapping every event to one integer time.

        ``prefer`` is either "earliest" or "latest" and selects the value
        ordering used during search. Every returned time belongs to the
        original admissible domain, so excluded gaps are never used.
        """
        if prefer not in ("earliest", "latest"):
            raise ValueError('prefer must be "earliest" or "latest"')
        reduced = self.solve()
        schedule = self._greedy(reduced, prefer)
        if schedule is not None and self._is_valid(schedule):
            return schedule
        schedule = self._search(dict(self._domains), prefer)
        if schedule is None:  # pragma: no cover - solve() already ruled this out
            raise InconsistentRecords(self.conflicts() or [{"kind": "empty_domain"}])
        return schedule

    def _greedy(self, reduced, prefer):
        """Fast path: assign extremal values, one precedence SCC at a time."""
        index, neg, _ = self._ensure_closure()
        schedule = {}
        current = dict(reduced)
        for group in self._precedence_sccs():
            low, high = -inf, inf
            for earlier, value in schedule.items():
                i = index[earlier]
                for event in group:
                    j = index[event]
                    if neg[i][j] != -inf:
                        low = max(low, value + neg[i][j])
                    if neg[j][i] != -inf:
                        high = min(high, value - neg[j][i])

            common = None
            for event in group:
                common = current[event] if common is None else common & current[event]
            if prefer == "earliest":
                chosen = _first_at_or_after(common, low)
                if chosen is None or chosen > high:
                    return None
            else:
                chosen = _last_at_or_before(common, high)
                if chosen is None or chosen < low:
                    return None
            for event in group:
                schedule[event] = chosen
                current[event] = closed(chosen, chosen)
        return schedule

    def _precedence_sccs(self):
        """
        Strongly connected components of the non-negativity precedence graph,
        returned in topological order. Events inside one component are linked
        by closed precedence cycles and therefore share a common time.
        """
        index, neg, _ = self._ensure_closure()
        events = self.events
        n = len(events)
        adjacency = [
            [j for j in range(n) if i != j and neg[i][j] >= 0] for i in range(n)
        ]
        reverse = [[] for _ in range(n)]
        for i, outs in enumerate(adjacency):
            for j in outs:
                reverse[j].append(i)

        seen = [False] * n
        finish_order = []
        for start in range(n):
            if seen[start]:
                continue
            seen[start] = True
            stack = [(start, 0)]
            while stack:
                node, k = stack[-1]
                if k < len(adjacency[node]):
                    nxt = adjacency[node][k]
                    stack[-1] = (node, k + 1)
                    if not seen[nxt]:
                        seen[nxt] = True
                        stack.append((nxt, 0))
                else:
                    finish_order.append(node)
                    stack.pop()

        component = [-1] * n
        groups = []
        for start in reversed(finish_order):
            if component[start] != -1:
                continue
            gid = len(groups)
            members = []
            component[start] = gid
            stack = [start]
            while stack:
                node = stack.pop()
                members.append(node)
                for nxt in reverse[node]:
                    if component[nxt] == -1:
                        component[nxt] = gid
                        stack.append(nxt)
            groups.append(members)
        return [[events[i] for i in members] for members in groups]

    def _search(self, domains, prefer):
        """Backtracking search with propagation, MRV and extremal values."""
        try:
            reduced = self._propagate(domains)
        except InconsistentRecords:
            return None

        pending = [
            event
            for event in self.events
            if not _is_singleton(reduced[event])
        ]
        if not pending:
            return {event: reduced[event].lower for event in self.events}

        event = min(
            pending,
            key=lambda e: (len(reduced[e]), reduced[e].upper - reduced[e].lower),
        )
        atomics = list(reduced[event])
        if prefer == "latest":
            atomics.reverse()
        for atomic in atomics:
            for value in self._candidate_values(atomic, prefer):
                child = dict(domains)
                child[event] = closed(value, value)
                result = self._search(child, prefer)
                if result is not None:
                    return result
        return None

    @staticmethod
    def _candidate_values(atomic, prefer):
        """Extremal values of an atomic range first; fall back to the rest."""
        lower, upper = atomic.lower, atomic.upper
        if lower == upper:
            return [lower]
        if prefer == "earliest":
            return [lower, upper] + list(range(lower + 1, upper))
        return [upper, lower] + list(range(upper - 1, lower, -1))

    def _is_valid(self, schedule):
        for event, domain in self._domains.items():
            if schedule[event] not in domain:
                return False
        for constraint in self._constraints:
            delta = schedule[constraint.b] - schedule[constraint.a]
            if constraint.kind == "before":
                if delta < 0:
                    return False
            elif constraint.kind == "max_wait":
                if delta > constraint.value:
                    return False
            else:
                if delta < constraint.value:
                    return False
        return True

    # ------------------------------------------------------------------ #
    # Relative order of two events
    # ------------------------------------------------------------------ #
    def order_of(self, a, b):
        """
        Classify the relative order of events ``a`` and ``b``.

        See :class:`OrderResult` for the possible verdicts. Both supporting
        schedules are provided whenever they exist; when the records are
        inconsistent, the conflicting records are returned instead.
        """
        if a not in self._domains or b not in self._domains:
            raise KeyError("unknown event in order_of")
        try:
            self.solve()
        except InconsistentRecords as error:
            return OrderResult("infeasible", None, None, None, list(error.conflicts))

        le_ab = self._satisfiable(before(a, b))       # t[a] <= t[b]
        le_ba = self._satisfiable(before(b, a))       # t[b] <= t[a]
        lt_ab = self._satisfiable(min_gap(a, b, 1))   # t[a] <  t[b]
        lt_ba = self._satisfiable(min_gap(b, a, 1))   # t[b] <  t[a]

        if lt_ab and lt_ba:
            verdict, earlier = "reversible", None
            schedule = self._extended(self._clone(min_gap(a, b, 1)), "earliest")
            reverse = self._extended(self._clone(min_gap(b, a, 1)), "latest")
        elif lt_ab:
            verdict, earlier = "necessary", a
            schedule = self._extended(self._clone(min_gap(a, b, 1)), "earliest")
            reverse = None
        elif lt_ba:
            verdict, earlier = "necessary", b
            schedule = self._extended(self._clone(min_gap(b, a, 1)), "latest")
            reverse = None
        elif le_ab and not le_ba:
            verdict, earlier = "necessary", a
            schedule = self._extended(self._clone(before(a, b)), "earliest")
            reverse = None
        elif le_ba and not le_ab:
            verdict, earlier = "necessary", b
            schedule = self._extended(self._clone(before(b, a)), "latest")
            reverse = None
        else:
            verdict, earlier = "simultaneous", None
            schedule = self._extended(self._clone(before(a, b)), "earliest")
            reverse = None
        return OrderResult(verdict, earlier, schedule, reverse, [])

    def _satisfiable(self, constraint):
        clone = self._clone(constraint)
        return clone.is_consistent()

    def _clone(self, extra):
        model = TemporalModel(self._domains)
        for constraint in self._constraints:
            model.add(constraint)
        model.add(extra)
        return model

    @staticmethod
    def _extended(model, prefer):
        return model.feasible_schedule(prefer)

    # ------------------------------------------------------------------ #
    # Connectivity
    # ------------------------------------------------------------------ #
    def components(self):
        """
        Return the connected components of events linked by temporal records.

        Events in different components have no record relating them, so their
        times are chosen independently; disconnected possible-time windows are
        never treated as one continuous timeline.
        """
        neighbours = {event: set() for event in self.events}
        for constraint in self._constraints:
            neighbours[constraint.a].add(constraint.b)
            neighbours[constraint.b].add(constraint.a)
        seen = set()
        groups = []
        for event in self.events:
            if event in seen:
                continue
            stack = [event]
            group = set()
            while stack:
                current = stack.pop()
                if current in group:
                    continue
                group.add(current)
                stack.extend(neighbours[current] - group)
            seen |= group
            groups.append(frozenset(group))
        return groups


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _is_singleton(domain):
    return domain.atomic and domain.lower == domain.upper


def _first_at_or_after(domain, value):
    if domain.empty or domain.upper < value:
        return None
    capped = domain & closedopen(value, inf)
    if capped.empty:
        return None
    return capped.lower


def _last_at_or_before(domain, value):
    if domain.empty or domain.lower > value:
        return None
    capped = domain & openclosed(-inf, value)
    if capped.empty:
        return None
    return capped.upper


def _format_conflict(conflict):
    kind = conflict["kind"]
    if kind == "empty_domain":
        return f"event {conflict['event']!r} has no admissible time"
    if kind == "cycle":
        chain = " < ".join(repr(event) for event in conflict["events"])
        records = ", ".join(repr(c) for c in conflict["constraints"])
        return f"strict ordering cycle ({chain}) created by {records}"
    event = conflict["event"]
    constraint = conflict.get("constraint")
    window = conflict.get("window")
    if kind == "excluded_gap":
        admissible = conflict["admissible"]
        return (
            f"event {event!r} is forced into {window}, whose integer times all "
            f"fall in excluded gaps of its admissible set {admissible}"
            + (f" (binding record {constraint!r})" if constraint else "")
        )
    return (
        f"no admissible time left for event {event!r} within {window}"
        + (f" due to {constraint!r}" if constraint else "")
    )
