"""Bounds on production during downtime slots from overlapping counter windows.

Each counter reports an exact piece count for its own time window, with no
per-item timestamps inside the window.  Windows overlap, every time slot has
a production capacity, and a set of slots is flagged as downtime.  Because a
single overlapping window is enough to make any duration-proportional split
infeasible, the only sound way to bound the downtime output is to solve the
allocation problem exactly.

Model (transportation problem, solved as min-cost flow):

* each window ``w`` is a supply node holding ``count_w`` pieces,
* each slot ``t`` is a demand node accepting at most ``capacity_t`` pieces
  (slots outside the downtime set constrain the solution as well),
* edge ``w -> t`` exists when slot ``t`` lies inside window ``w`` and costs
  1 per piece when ``t`` is a downtime slot, 0 otherwise.

The minimum-cost feasible flow gives the fewest pieces that must have been
produced during downtime; rerunning with the cost on non-downtime slots
gives the most.  All counts are integers, and the flow decomposition yields
an explicit assignment satisfying every window.

JSON CLI (reads stdin, writes stdout)::

    {
      "capacities": [3, 3, 3],
      "windows": [{"start": 0, "end": 1, "count": 4},
                  {"start": 1, "end": 2, "count": 4}],
      "downtime": [1]
    }

Slot indices are 0-based and window ends are inclusive.
"""

from __future__ import annotations

import heapq
import json
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

__all__ = [
    "Window",
    "Allocation",
    "DowntimeBounds",
    "InfeasibleError",
    "downtime_bounds",
]


class InfeasibleError(ValueError):
    """Raised when no assignment can satisfy all windows and capacities."""


@dataclass(frozen=True)
class Window:
    """One counter report: ``count`` pieces produced in slots start..end."""

    start: int  # first slot of the window, inclusive
    end: int  # last slot of the window, inclusive
    count: int  # exact pieces uploaded by the counter for this window


@dataclass
class Allocation:
    """A concrete assignment of every reported piece to a slot."""

    slot_totals: List[int]  # pieces produced in each slot (all slots)
    per_window: List[Dict[int, int]]  # per window: slot -> pieces from that window

    def downtime_pieces(self, downtime: Set[int]) -> int:
        return sum(self.slot_totals[t] for t in downtime)


@dataclass
class DowntimeBounds:
    """Achievable minimum and maximum downtime output, with witnesses."""

    minimum: int
    maximum: int
    min_allocation: Allocation
    max_allocation: Allocation


class _MinCostFlow:
    """Successive shortest paths with Johnson potentials (integral flows)."""

    __slots__ = ("graph",)

    def __init__(self, nodes: int) -> None:
        self.graph: List[List[list]] = [[] for _ in range(nodes)]

    def add_edge(self, frm: int, to: int, cap: int, cost: int) -> int:
        fwd = [to, cap, cost, len(self.graph[to])]
        rev = [frm, 0, -cost, len(self.graph[frm])]
        self.graph[frm].append(fwd)
        self.graph[to].append(rev)
        return len(self.graph[frm]) - 1

    def edge(self, node: int, index: int) -> list:
        return self.graph[node][index]

    def min_cost_flow(self, source: int, sink: int, amount: int) -> Tuple[int, int]:
        """Push up to ``amount`` flow; return (flow, cost). Costs must be >= 0."""
        n = len(self.graph)
        potential = [0] * n
        flow = 0
        cost = 0
        while flow < amount:
            dist = [None] * n
            dist[source] = 0
            prev: List[Tuple[int, int] | None] = [None] * n
            heap = [(0, source)]
            while heap:
                d, v = heapq.heappop(heap)
                if dist[v] is not None and d > dist[v]:
                    continue
                for i, e in enumerate(self.graph[v]):
                    if e[1] <= 0:
                        continue
                    nd = d + e[2] + potential[v] - potential[e[0]]
                    if dist[e[0]] is None or nd < dist[e[0]]:
                        dist[e[0]] = nd
                        prev[e[0]] = (v, i)
                        heapq.heappush(heap, (nd, e[0]))
            if dist[sink] is None:
                break
            for v in range(n):
                if dist[v] is not None:
                    potential[v] += dist[v]
            add = amount - flow
            v = sink
            while v != source:
                u, i = prev[v]
                add = min(add, self.graph[u][i][1])
                v = u
            v = sink
            while v != source:
                u, i = prev[v]
                e = self.graph[u][i]
                e[1] -= add
                self.graph[v][e[3]][1] += add
                v = u
            flow += add
            cost += add * potential[sink]
        return flow, cost


def _validate(capacities: Sequence[int], windows: Sequence[Window], downtime: Set[int]) -> None:
    n = len(capacities)
    if n == 0:
        raise ValueError("at least one time slot is required")
    for t, cap in enumerate(capacities):
        if cap < 0:
            raise ValueError(f"negative capacity {cap} in slot {t}")
    for t in downtime:
        if not 0 <= t < n:
            raise ValueError(f"downtime slot {t} outside 0..{n - 1}")
    for w in windows:
        if w.count < 0:
            raise ValueError(f"negative count in window {w}")
        if not 0 <= w.start <= w.end < n:
            raise ValueError(f"window {w} spans slots outside 0..{n - 1}")


def _optimal_allocation(
    capacities: Sequence[int],
    windows: Sequence[Window],
    costly: Set[int],
) -> Tuple[Allocation, int]:
    """Minimise pieces landing in ``costly`` slots; return allocation and cost."""
    n = len(capacities)
    w_count = len(windows)
    source, sink = 0, 1 + w_count + n
    mcf = _MinCostFlow(sink + 1)
    for i, w in enumerate(windows):
        mcf.add_edge(source, 1 + i, w.count, 0)
    slot_edges: List[List[Tuple[int, int]]] = []
    for i, w in enumerate(windows):
        edges = []
        for t in range(w.start, w.end + 1):
            idx = mcf.add_edge(1 + i, 1 + w_count + t, w.count, 1 if t in costly else 0)
            edges.append((t, idx))
        slot_edges.append(edges)
    for t in range(n):
        mcf.add_edge(1 + w_count + t, sink, capacities[t], 0)

    total = sum(w.count for w in windows)
    flow, cost = mcf.min_cost_flow(source, sink, total)
    if flow < total:
        raise InfeasibleError(
            f"windows report {total} pieces but capacities/windows allow at most {flow}"
        )

    per_window: List[Dict[int, int]] = []
    slot_totals = [0] * n
    for i, w in enumerate(windows):
        alloc: Dict[int, int] = {}
        for t, idx in slot_edges[i]:
            used = w.count - mcf.edge(1 + i, idx)[1]
            if used:
                alloc[t] = used
                slot_totals[t] += used
        per_window.append(alloc)
    return Allocation(slot_totals, per_window), cost


def downtime_bounds(
    capacities: Sequence[int],
    windows: Iterable[Window | Tuple[int, int, int]],
    downtime: Iterable[int],
) -> DowntimeBounds:
    """Minimum/maximum pieces that may fall in ``downtime`` slots.

    ``capacities[t]`` bounds total production in slot ``t`` (slots outside
    the downtime set constrain the answer as well).  Each window is a
    ``Window`` or ``(start, end, count)`` triple with inclusive integer slot
    ends.  Returns bounds together with assignments satisfying every window.
    Raises :class:`InfeasibleError` when the reports are mutually impossible.
    """
    windows = [w if isinstance(w, Window) else Window(*w) for w in windows]
    downtime_set = set(downtime)
    _validate(capacities, windows, downtime_set)

    min_alloc, minimum = _optimal_allocation(capacities, windows, downtime_set)
    outside = set(range(len(capacities))) - downtime_set
    max_alloc, min_outside = _optimal_allocation(capacities, windows, outside)
    total = sum(w.count for w in windows)
    maximum = total - min_outside
    return DowntimeBounds(minimum, maximum, min_alloc, max_alloc)


def _allocation_to_json(allocation: Allocation, windows: Sequence[Window]) -> dict:
    return {
        "slot_totals": allocation.slot_totals,
        "windows": [
            {
                "start": w.start,
                "end": w.end,
                "count": w.count,
                "allocation": {str(t): p for t, p in sorted(alloc.items())},
            }
            for w, alloc in zip(windows, allocation.per_window)
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    payload = json.load(sys.stdin if argv is None else open(argv[0]))
    capacities = payload["capacities"]
    windows = [Window(w["start"], w["end"], w["count"]) for w in payload["windows"]]
    downtime = set(payload["downtime"])
    try:
        bounds = downtime_bounds(capacities, windows, downtime)
    except InfeasibleError as exc:
        json.dump({"error": str(exc)}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1
    json.dump(
        {
            "minimum": bounds.minimum,
            "maximum": bounds.maximum,
            "min_allocation": _allocation_to_json(bounds.min_allocation, windows),
            "max_allocation": _allocation_to_json(bounds.max_allocation, windows),
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
