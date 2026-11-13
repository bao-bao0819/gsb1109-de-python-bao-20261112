"""Tests for the shard rebalancer, including brute-force optimality checks."""

from itertools import combinations

import pytest

from sharding import (
    BlockingSegment,
    Shard,
    format_plan,
    rebalance,
)


def brute_force(shards, capacities, cuts):
    """Reference implementation: enumerate every legal boundary combination."""
    from sharding import _canonical_boundaries

    machines = tuple(capacities)
    internal, _ = _canonical_boundaries(shards, frozenset(cuts))
    internal_indices = sorted(
        {i for i, shard in enumerate(shards) if shard.end in internal}
    )
    m = len(machines)
    best = None

    if m > len(internal_indices) + 1:
        return None

    for chosen in combinations(internal_indices, m - 1):
        bounds = (-1, *chosen, len(shards) - 1)
        feasible = True
        cost = 0
        for k in range(m):
            lo, hi = bounds[k] + 1, bounds[k + 1] + 1
            weight = sum(s.records for s in shards[lo:hi])
            if weight > capacities[machines[k]]:
                feasible = False
                break
            cost += sum(s.records for s in shards[lo:hi] if s.backend != machines[k])
        if feasible and (best is None or cost < best):
            best = cost
    return best


def test_basic_two_machines_prefers_keeping_records_in_place():
    shards = [
        Shard(0, 9, 100, "A"),
        Shard(10, 19, 100, "A"),
        Shard(20, 29, 100, "B"),
        Shard(30, 39, 100, "B"),
    ]
    plan = rebalance(shards, {"A": 250, "B": 250}, cuts=[19])
    assert plan.feasible
    assert plan.boundaries == (19,)
    assert plan.moved_records == 0
    assert all(not move.moved for move in plan.moves)
    assert plan.loads == {"A": 200, "B": 200}


def test_capacity_forces_moves_and_picks_cheapest_boundary():
    shards = [
        Shard(0, 9, 100, "A"),
        Shard(10, 19, 100, "A"),
        Shard(20, 29, 60, "B"),
        Shard(30, 39, 40, "B"),
    ]
    # A can hold at most 200, so boundary after index 1 keeps everything put.
    plan = rebalance(shards, {"A": 200, "B": 200}, cuts=[9, 19, 29])
    assert plan.feasible and plan.boundaries == (19,) and plan.moved_records == 0

    # A capped at 160: boundary after index 1 infeasible (200 > 160),
    # next option after index 0 moves 100; boundary after index 2 moves 160.
    tight = rebalance(shards, {"A": 160, "B": 200}, cuts=[9, 19, 29])
    assert tight.feasible
    assert tight.boundaries == (9,)
    assert tight.moved_records == 100
    move = next(m for m in tight.moves if m.start == 10)
    assert move.moved and move.source == "A" and move.target == "B"
    assert tight.loads == {"A": 100, "B": 200}


def test_hole_cut_canonicalises_and_holes_cost_nothing():
    shards = [
        Shard(0, 99, 1000, "A"),
        Shard(200, 299, 500, "A"),  # hole [100,199]
        Shard(300, 399, 700, "B"),
    ]
    # Cut at 150 sits inside the hole and equals a boundary after shard 0.
    plan = rebalance(shards, {"A": 1000, "B": 1200}, cuts=[150])
    assert plan.feasible
    assert plan.boundaries == (99,)
    assert plan.moved_records == 500
    assert plan.unusable_cuts == ()


def test_cut_inside_shard_is_unusable():
    shards = [Shard(0, 100, 10, "A"), Shard(101, 200, 10, "B")]
    plan = rebalance(shards, {"A": 20}, cuts=[50])
    assert plan.feasible
    assert plan.boundaries == ()
    assert plan.unusable_cuts == (50,)
    assert plan.loads == {"A": 20}
    assert plan.moved_records == 10


def test_no_usable_boundary_for_two_machines_is_infeasible():
    shards = [Shard(0, 100, 10, "A"), Shard(101, 200, 10, "B")]
    plan = rebalance(shards, {"A": 20, "B": 20}, cuts=[50])
    assert not plan.feasible
    assert plan.unusable_cuts == (50,)
    assert "non-empty segment" in plan.reason


def test_single_key_hotspot_blocks_when_no_machine_fits():
    shards = [
        Shard(0, 99, 500, "A"),
        Shard(100, 100, 9000, "A"),  # one key, cannot be split by any cut
        Shard(101, 200, 500, "B"),
    ]
    plan = rebalance(shards, {"A": 2000, "B": 2000}, cuts=[99, 100, 200])
    assert not plan.feasible
    assert len(plan.blocking) == 1
    seg = plan.blocking[0]
    assert isinstance(seg, BlockingSegment)
    assert (seg.start, seg.end, seg.records) == (100, 100, 9000)
    assert seg.capacity_limit == 2000
    assert "hotspot" in plan.reason


def test_hotspot_fits_on_large_middle_machine():
    shards = [
        Shard(0, 99, 500, "A"),
        Shard(100, 100, 9000, "A"),
        Shard(101, 200, 500, "B"),
    ]
    plan = rebalance(
        shards,
        {"A": 500, "B": 9500, "C": 500},
        cuts=[99, 100],
    )
    assert plan.feasible
    assert plan.boundaries == (99, 100)
    assert plan.loads == {"A": 500, "B": 9000, "C": 500}
    # The hotspot (9000) and the last shard (500) both change backend.
    assert plan.moved_records == 9500


def test_migration_aggregation_groups_shards_per_source_target_pair():
    shards = [
        Shard(0, 9, 10, "A"),
        Shard(10, 19, 20, "A"),
        Shard(20, 29, 30, "B"),
    ]
    plan = rebalance(shards, {"X": 60}, cuts=[])
    assert plan.feasible
    assert len(plan.migrations) == 2
    by_pair = {(mg.source, mg.target): mg for mg in plan.migrations}
    mg = by_pair[("A", "X")]
    assert mg.records == 30
    assert mg.shards == ((0, 9), (10, 19))
    assert by_pair[("B", "X")].records == 30
    assert sum(mg.records for mg in plan.migrations) == plan.moved_records


def test_more_machines_than_segments_is_infeasible():
    shards = [Shard(0, 10, 5, "A")]
    plan = rebalance(shards, {"A": 5, "B": 5}, cuts=[])
    assert not plan.feasible
    assert "non-empty segment" in plan.reason


def test_capacity_order_constraint_diagnosed():
    # Small machine first can never receive the only heavy shard.
    shards = [Shard(0, 10, 100, "A"), Shard(11, 20, 1, "B")]
    plan = rebalance(shards, {"small": 1, "big": 200}, cuts=[10])
    assert not plan.feasible
    assert plan.blocking == ()
    assert "small" in plan.reason


def test_generic_infeasibility_when_every_machine_row_is_locally_reachable():
    # weights 5,5,100,2,5 with caps 10/100/3: every backend can take some
    # prefix segment, yet no full ordered partition exists (tail cannot be
    # split into a segment of at most 3 around the 100-weight shard).
    shards = [Shard(i, i, weight, "X") for i, weight in enumerate([5, 5, 100, 2, 5])]
    plan = rebalance(shards, {"M0": 10, "M1": 100, "M2": 3}, cuts=[0, 1, 2, 3])
    assert not plan.feasible
    assert plan.blocking == ()
    assert "no partition" in plan.reason


def test_zero_record_shards_and_holes_never_cost_moves():
    shards = [
        Shard(0, 9, 0, "A"),
        Shard(100, 109, 0, None),
        Shard(200, 209, 5, "B"),
    ]
    plan = rebalance(shards, {"A": 5, "B": 5}, cuts=[9, 109])
    assert plan.feasible
    assert plan.moved_records == 0
    assert plan.loads == {"A": 0, "B": 5}


def test_format_plan_covers_feasible_and_infeasible():
    shards = [Shard(0, 9, 10, "A"), Shard(10, 19, 10, "B")]
    text = format_plan(rebalance(shards, {"A": 20}, cuts=[]))
    assert "boundaries" in text and "moved records: 10" in text

    bad = rebalance([Shard(0, 0, 99, "A")], {"A": 1}, cuts=[])
    bad_text = format_plan(bad)
    assert "INFEASIBLE" in bad_text and "99" in bad_text


def test_validation_errors():
    with pytest.raises(ValueError):
        rebalance([], {"A": 1})
    with pytest.raises(ValueError):
        rebalance([Shard(0, 1, 1)], {})
    with pytest.raises(ValueError):
        rebalance([Shard(0, 1, 1)], {"A": -1})
    with pytest.raises(ValueError):
        rebalance([Shard(0, 5, 1, "A"), Shard(3, 8, 1, "B")], {"A": 2, "B": 2})
    with pytest.raises(ValueError):
        Shard(5, 4, 1)


@pytest.mark.parametrize("seed", range(60))
def test_dp_matches_brute_force_on_random_instances(seed):
    import random

    rng = random.Random(seed)
    shard_count = rng.randint(1, 6)
    machine_count = rng.randint(1, 4)
    shards = []
    cursor = rng.randint(0, 5)
    backends = [f"M{i}" for i in range(machine_count)]
    for _ in range(shard_count):
        length = rng.randint(1, 4)
        records = rng.choice([0, 1, 5, 10, 50, 100])
        shards.append(Shard(cursor, cursor + length - 1, records, rng.choice(backends)))
        cursor += length + rng.randint(0, 3)  # occasional holes

    capacities = {name: rng.randint(0, 150) for name in backends}
    valid_cuts = [s.end for s in shards[:-1]]
    hole_cuts = [c for s in shards for c in (s.end + 1, s.end + 2)]
    inside_cuts = [rng.randint(s.start, s.end - 1) for s in shards if s.end > s.start]
    cuts = rng.sample(
        sorted(set(valid_cuts + hole_cuts + inside_cuts)),
        k=rng.randint(0, len(set(valid_cuts + hole_cuts + inside_cuts))),
    )

    plan = rebalance(shards, capacities, cuts)
    expected = brute_force(shards, capacities, cuts)

    if expected is None:
        assert not plan.feasible
        return

    assert plan.feasible, plan.reason
    assert plan.moved_records == expected

    # Independently re-derive every property of the returned partition.
    assert len(plan.boundaries) == machine_count - 1
    covered = 0
    total_load = 0
    for k, machine in enumerate(plan.machines):
        mine = [m for m in plan.moves if m.target == machine]
        assert mine, "every backend receives a non-empty contiguous segment"
        starts = [
            shards.index(next(s for s in shards if s.start == m.start)) for m in mine
        ]
        assert starts == list(range(min(starts), max(starts) + 1)), "contiguous"
        load = sum(m.records for m in mine)
        assert load <= capacities[machine]
        assert plan.loads[machine] == load
        total_load += load
        covered += len(mine)
    assert covered == len(shards)
    assert total_load == sum(s.records for s in shards)
    assert sum(m.records for m in plan.migrations) == plan.moved_records


def test_demo_scenario_from_module():
    shards = [
        Shard(0, 99, 1000, "A"),
        Shard(200, 299, 500, "A"),
        Shard(300, 300, 9000, "B"),
        Shard(301, 400, 800, "B"),
        Shard(401, 500, 700, "C"),
    ]
    ok = rebalance(shards, {"A": 2000, "B": 9500, "C": 1500}, [99, 150, 299, 300, 400])
    assert ok.feasible
    assert ok.boundaries == (299, 300)
    # The hotspot sits alone on B; [301,400] joins C while [401,500] stays.
    assert ok.loads == {"A": 1500, "B": 9000, "C": 1500}
    assert ok.moved_records == 800
    pair = {(mg.source, mg.target): mg.records for mg in ok.migrations}
    assert pair == {("B", "C"): 800}

    bad = rebalance(shards, {"A": 2000, "B": 2000, "C": 1500}, [99, 150, 299, 300, 400])
    assert not bad.feasible
    assert bad.blocking[0].records == 9000
    assert (bad.blocking[0].start, bad.blocking[0].end) == (300, 300)
