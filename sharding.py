"""Optimal re-cut of an integer primary-key space among fixed backends.

The key space is already partitioned into shards. Each shard covers a
closed integer interval ``[start, end]``, holds a known number of records
and currently lives on a backend. Gaps between shards are holes: they hold
no records and therefore never cost anything to move.

The new layout must:

* assign every backend (in a fixed order) exactly one non-empty, contiguous
  interval of the key space, with new boundaries chosen from a set of
  permitted cut points;
* never put more records on a backend than its capacity; record counts are
  what matters, so a single-key hotspot cannot be discounted by the length
  of its interval;
* minimise the number of records that actually change backend. Records in
  existing holes are not part of any shard and are never moved.

Cut-point semantics
-------------------
A cut at position ``x`` is a boundary *between* integer keys ``x`` and
``x + 1``. A cut can only be used where it does not split an existing
shard: it must coincide with a shard end or fall inside a hole (in which
case it is canonicalised to the end of the shard immediately before the
hole). Cuts landing inside a shard cannot split it and are reported as
unusable.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "Shard",
    "ShardMove",
    "Migration",
    "BlockingSegment",
    "RebalancePlan",
    "rebalance",
    "format_plan",
]


@dataclass(frozen=True)
class Shard:
    """An existing shard: closed interval ``[start, end]`` on one backend."""

    start: int
    end: int
    records: int
    backend: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("shard bounds must be ints")
        if self.start > self.end:
            raise ValueError(f"empty shard {self.start!r} > {self.end!r}")
        if self.records < 0:
            raise ValueError("record counts must be non-negative")


@dataclass(frozen=True)
class ShardMove:
    """Where one existing shard ends up after rebalancing."""

    start: int
    end: int
    records: int
    source: str | None
    target: str

    @property
    def moved(self) -> bool:
        return self.source != self.target


@dataclass(frozen=True)
class Migration:
    """All records flowing from one source backend to one target backend."""

    source: str | None
    target: str
    records: int
    shards: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BlockingSegment:
    """An indivisible segment that no backend can absorb."""

    start: int
    end: int
    records: int
    capacity_limit: int


@dataclass
class RebalancePlan:
    feasible: bool
    machines: tuple[str, ...]
    boundaries: tuple[int, ...]
    moves: tuple[ShardMove, ...]
    migrations: tuple[Migration, ...]
    loads: dict[str, int]
    moved_records: int
    unusable_cuts: tuple[int, ...]
    blocking: tuple[BlockingSegment, ...]
    reason: str | None

    def assert_ok(self) -> "RebalancePlan":
        if not self.feasible:
            raise RuntimeError(self.reason or "no feasible rebalancing")
        return self


def _validate(shards: list[Shard], capacities: Mapping[str, int]) -> None:
    if not shards:
        raise ValueError("at least one shard is required")
    if not capacities:
        raise ValueError("at least one backend is required")
    for name, cap in capacities.items():
        if not isinstance(cap, int) or cap < 0:
            raise ValueError(f"capacity of {name!r} must be a non-negative int")
    prev: Shard | None = None
    for shard in shards:
        if prev is not None and shard.start <= prev.end:
            raise ValueError(
                f"shards overlap or are unordered around [{shard.start},{shard.end}]"
            )
        prev = shard


def _canonical_boundaries(
    shards: list[Shard], cuts: frozenset[int]
) -> tuple[set[int], list[int]]:
    """Map raw cut positions to shard-end boundaries.

    Returns the set of usable internal boundary positions (shard ends) and
    the list of raw cuts that cannot be used because they split a shard.
    """
    starts = [shard.start for shard in shards]
    first_start = shards[0].start
    last_end = shards[-1].end
    usable: set[int] = set()
    unusable: list[int] = []
    for cut in sorted(cuts):
        if cut <= first_start - 1 or cut > last_end:
            unusable.append(cut)
            continue
        idx = bisect_right(starts, cut) - 1
        shard = shards[idx]
        if cut <= shard.end:
            if cut == shard.end:
                usable.add(cut)
            else:
                unusable.append(cut)
        else:
            # Cut lies in the hole following this shard: same logical
            # boundary as the shard's own end.
            usable.add(shard.end)
    usable.discard(last_end)
    return usable, unusable


def rebalance(
    shards: list[Shard],
    capacities: Mapping[str, int],
    cuts: list[int] | frozenset[int] | None = None,
) -> RebalancePlan:
    """Find the feasible re-cut that moves the fewest records.

    ``capacities`` is an ordered mapping: its insertion order fixes the
    backend order, and each backend receives exactly one contiguous segment
    in that order. ``cuts`` are the permitted boundary positions (see the
    module docstring for their semantics).

    Returns a :class:`RebalancePlan`. When no feasible partition exists the
    plan is marked infeasible and explains which indivisible segment (for
    example a single-key hotspot) no backend can hold.
    """
    shards = list(shards)
    _validate(shards, capacities)
    machines = tuple(capacities)
    caps = [capacities[name] for name in machines]
    raw_cuts = frozenset(cuts) if cuts is not None else frozenset()

    internal, unusable = _canonical_boundaries(shards, raw_cuts)
    boundary_ends = internal | {shards[-1].end}

    # Group shards into indivisible cells separated by usable boundaries.
    cells: list[list[int]] = []
    current: list[int] = []
    for idx, shard in enumerate(shards):
        current.append(idx)
        if shard.end in boundary_ends:
            cells.append(current)
            current = []
    if current:  # pragma: no cover - last end is always a boundary
        cells.append(current)

    cell_count = len(cells)
    machine_count = len(machines)

    cell_span = [(idxs[0], idxs[-1] + 1) for idxs in cells]
    cell_start_key = [shards[si].start for si, _ in cell_span]
    cell_end_key = [shards[sj - 1].end for _, sj in cell_span]

    weight_prefix = [0]
    for idxs in cells:
        weight_prefix.append(weight_prefix[-1] + sum(shards[i].records for i in idxs))

    # Prefix movement cost per target machine: cost of assigning shard
    # prefix [0, j) to that machine.
    cost_prefix: list[list[int]] = []
    for machine in machines:
        prefix = [0]
        for shard in shards:
            prefix.append(
                prefix[-1] + (shard.records if shard.backend != machine else 0)
            )
        cost_prefix.append(prefix)

    max_capacity = max(caps)
    blocking = tuple(
        BlockingSegment(
            cell_start_key[i],
            cell_end_key[i],
            weight_prefix[i + 1] - weight_prefix[i],
            max_capacity,
        )
        for i in range(cell_count)
        if weight_prefix[i + 1] - weight_prefix[i] > max_capacity
    )

    reason: str | None = None
    if machine_count > cell_count:
        reason = (
            f"cannot give {machine_count} backends one non-empty segment each: "
            f"only {cell_count} indivisible segment(s) are available"
        )
        return RebalancePlan(
            False, machines, (), (), (), {}, 0, tuple(unusable), blocking, reason
        )

    # DP over (machines used, cells covered): minimise moved records.
    infinity = float("inf")
    dp = [[infinity] * (cell_count + 1) for _ in range(machine_count + 1)]
    choice = [[-1] * (cell_count + 1) for _ in range(machine_count + 1)]
    dp[0][0] = 0

    for k in range(1, machine_count + 1):
        cap = caps[k - 1]
        prefix = cost_prefix[k - 1]
        for j in range(1, cell_count + 1):
            best = infinity
            best_i = -1
            for i in range(k - 1, j):
                if dp[k - 1][i] == infinity:
                    continue
                weight = weight_prefix[j] - weight_prefix[i]
                if weight > cap:
                    continue
                shard_lo, _ = cell_span[i]
                _, shard_hi = cell_span[j - 1]
                cost = dp[k - 1][i] + prefix[shard_hi] - prefix[shard_lo]
                if cost < best or (cost == best and (best_i == -1 or i < best_i)):
                    best = cost
                    best_i = i
            dp[k][j] = best
            choice[k][j] = best_i

    if dp[machine_count][cell_count] == infinity:
        if blocking:
            detail = "; ".join(
                f"[{seg.start},{seg.end}] holds {seg.records} records but the "
                f"largest backend capacity is {seg.capacity_limit}"
                for seg in blocking
            )
            reason = (
                "no cut can separate the oversized hotspot(s): "
                f"{detail}, and no backend can hold them"
            )
        else:
            stuck = next(
                (
                    k
                    for k in range(1, machine_count + 1)
                    if all(dp[k][j] == infinity for j in range(cell_count + 1))
                ),
                None,
            )
            if stuck is not None:
                name = machines[stuck - 1]
                reason = (
                    f"backend order cannot be satisfied: {name!r} (capacity "
                    f"{caps[stuck - 1]}) cannot take any non-empty contiguous "
                    "segment given the segments already assigned to the backends "
                    "before it"
                )
            else:
                reason = (
                    "no partition of the indivisible segments over the ordered "
                    f"backends {list(machines)} satisfies all capacities "
                    f"{caps}"
                )
        return RebalancePlan(
            False, machines, (), (), (), {}, 0, tuple(unusable), blocking, reason
        )

    # Backtrack the optimal partition into cell ranges per machine.
    ranges: list[tuple[int, int]] = [None] * machine_count  # type: ignore[list-item]
    j = cell_count
    for k in range(machine_count, 0, -1):
        i = choice[k][j]
        ranges[k - 1] = (i, j)
        j = i

    boundaries = tuple(cell_end_key[ranges[k][1] - 1] for k in range(machine_count - 1))
    moves: list[ShardMove] = []
    migrations: dict[tuple[str | None, str], Migration] = {}
    loads: dict[str, int] = {}
    moved_records = 0

    for k, (lo, hi) in enumerate(ranges):
        target = machines[k]
        loads[target] = weight_prefix[hi] - weight_prefix[lo]
        shard_lo = cell_span[lo][0]
        shard_hi = cell_span[hi - 1][1]
        for idx in range(shard_lo, shard_hi):
            shard = shards[idx]
            move = ShardMove(
                shard.start, shard.end, shard.records, shard.backend, target
            )
            moves.append(move)
            if move.moved:
                moved_records += move.records
                key = (shard.backend, target)
                existing = migrations.get(key)
                if existing is None:
                    migrations[key] = Migration(
                        shard.backend,
                        target,
                        shard.records,
                        ((shard.start, shard.end),),
                    )
                else:
                    migrations[key] = Migration(
                        existing.source,
                        existing.target,
                        existing.records + shard.records,
                        existing.shards + ((shard.start, shard.end),),
                    )

    return RebalancePlan(
        feasible=True,
        machines=machines,
        boundaries=boundaries,
        moves=tuple(moves),
        migrations=tuple(migrations.values()),
        loads=loads,
        moved_records=moved_records,
        unusable_cuts=tuple(unusable),
        blocking=(),
        reason=None,
    )


def format_plan(plan: RebalancePlan) -> str:
    """Render a plan as a human-readable report."""
    if not plan.feasible:
        lines = [f"INFEASIBLE: {plan.reason}"]
        for seg in plan.blocking:
            lines.append(
                f"  oversized segment [{seg.start},{seg.end}]: "
                f"{seg.records} records > capacity {seg.capacity_limit}"
            )
        if plan.unusable_cuts:
            lines.append(f"  unusable cuts: {list(plan.unusable_cuts)}")
        return "\n".join(lines)

    lines = [f"boundaries (cut after key): {list(plan.boundaries)}"]
    for name in plan.machines:
        lines.append(f"  {name}: load {plan.loads[name]} / capacity")
    lines.append(f"moved records: {plan.moved_records}")
    for move in plan.moves:
        flag = "stay" if not move.moved else "MOVE"
        lines.append(
            f"  [{move.start},{move.end}] {move.records} records: "
            f"{move.source} -> {move.target} ({flag})"
        )
    for migration in plan.migrations:
        lines.append(
            f"  migration {migration.source} -> {migration.target}: "
            f"{migration.records} records across {list(migration.shards)}"
        )
    if plan.unusable_cuts:
        lines.append(f"  unusable cuts: {list(plan.unusable_cuts)}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    demo_shards = [
        Shard(0, 99, 1000, "A"),
        Shard(200, 299, 500, "A"),  # hole [100,199]: no records to move
        Shard(300, 300, 9000, "B"),  # single-key hotspot
        Shard(301, 400, 800, "B"),
        Shard(401, 500, 700, "C"),
    ]
    demo_cuts = [99, 150, 299, 300, 400]

    print("== feasible scenario ==")
    ok = rebalance(demo_shards, {"A": 2000, "B": 9500, "C": 1500}, demo_cuts)
    print(format_plan(ok))

    print("\n== hotspot cannot fit ==")
    bad = rebalance(demo_shards, {"A": 2000, "B": 2000, "C": 1500}, demo_cuts)
    print(format_plan(bad))
