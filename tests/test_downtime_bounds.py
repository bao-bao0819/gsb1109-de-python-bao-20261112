import itertools
import random

import pytest

from downtime_bounds import (
    Allocation,
    DowntimeBounds,
    InfeasibleError,
    Window,
    downtime_bounds,
)


def check_allocation(capacities, windows, allocation: Allocation):
    """Every window satisfied, every slot within capacity."""
    assert all(0 <= x <= cap for x, cap in zip(allocation.slot_totals, capacities))
    for w, alloc in zip(windows, allocation.per_window):
        assert sum(alloc.values()) == w.count
        assert all(w.start <= t <= w.end and p > 0 for t, p in alloc.items())
    recomputed = [0] * len(capacities)
    for alloc in allocation.per_window:
        for t, p in alloc.items():
            recomputed[t] += p
    assert recomputed == allocation.slot_totals


def check_bounds(capacities, windows, downtime, bounds: DowntimeBounds):
    check_allocation(capacities, windows, bounds.min_allocation)
    check_allocation(capacities, windows, bounds.max_allocation)
    assert bounds.min_allocation.downtime_pieces(downtime) == bounds.minimum
    assert bounds.max_allocation.downtime_pieces(downtime) == bounds.maximum
    assert 0 <= bounds.minimum <= bounds.maximum <= sum(w.count for w in windows)


def brute_force(capacities, windows, downtime):
    """Exhaustive per-window split; returns (min, max) downtime pieces."""
    n = len(capacities)

    def splits(w):
        slots = range(w.start, w.end + 1)
        yield from (
            zip(slots, parts)
            for parts in itertools.product(range(w.count + 1), repeat=len(list(slots)))
            if sum(parts) == w.count
        )

    best = [None, None]
    totals = [0] * n

    def go(i):
        if i == len(windows):
            if all(totals[t] <= capacities[t] for t in range(n)):
                down = sum(totals[t] for t in downtime)
                best[0] = down if best[0] is None else min(best[0], down)
                best[1] = down if best[1] is None else max(best[1], down)
            return
        for assignment in splits(windows[i]):
            assignment = list(assignment)
            if any(totals[t] + p > capacities[t] for t, p in assignment):
                continue
            for t, p in assignment:
                totals[t] += p
            go(i + 1)
            for t, p in assignment:
                totals[t] -= p

    go(0)
    return best


def test_proportional_split_is_infeasible_but_flow_solves():
    # Two overlapping windows of 4 pieces each over 3 slots of capacity 3.
    # A duration-proportional split puts 2+2=4 pieces in slot 1 (> capacity),
    # yet exact bounds exist: 2 <= downtime pieces <= 3.
    capacities = [3, 3, 3]
    windows = [Window(0, 1, 4), Window(1, 2, 4)]
    bounds = downtime_bounds(capacities, windows, downtime={1})
    assert (bounds.minimum, bounds.maximum) == (2, 3)
    check_bounds(capacities, windows, {1}, bounds)


def test_non_overlapping_window_is_exact():
    bounds = downtime_bounds([5, 5], [Window(0, 0, 3)], downtime={0})
    assert (bounds.minimum, bounds.maximum) == (3, 3)
    assert bounds.min_allocation.slot_totals == [3, 0]


def test_outside_slots_constrain_downtime():
    # Window covers slots 0..2 with 6 pieces; only slot 1 is downtime.
    # Outside slots hold at most 2+2, so at least 2 pieces fell in downtime.
    capacities = [2, 10, 2]
    windows = [Window(0, 2, 6)]
    bounds = downtime_bounds(capacities, windows, downtime={1})
    assert (bounds.minimum, bounds.maximum) == (2, 6)
    check_bounds(capacities, windows, {1}, bounds)


def test_infeasible_reports_raise():
    with pytest.raises(InfeasibleError):
        downtime_bounds([1, 1], [Window(0, 1, 3)], downtime=set())


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        downtime_bounds([1], [Window(0, 2, 1)], downtime=set())
    with pytest.raises(ValueError):
        downtime_bounds([1], [Window(0, 0, 1)], downtime={5})
    with pytest.raises(ValueError):
        downtime_bounds([], [], downtime=set())


def test_tuple_windows_and_iterable_downtime():
    bounds = downtime_bounds([4, 4], [(0, 1, 5)], downtime=[0])
    assert (bounds.minimum, bounds.maximum) == (1, 4)


def test_against_brute_force_on_random_instances():
    rng = random.Random(20261112)
    for _ in range(60):
        n = rng.randint(1, 4)
        capacities = [rng.randint(0, 4) for _ in range(n)]
        windows = []
        for _ in range(rng.randint(1, 3)):
            a = rng.randrange(n)
            b = rng.randrange(a, n)
            windows.append(Window(a, b, rng.randint(0, 5)))
        downtime = {t for t in range(n) if rng.random() < 0.5}
        lo, hi = brute_force(capacities, windows, downtime)
        if lo is None:
            with pytest.raises(InfeasibleError):
                downtime_bounds(capacities, windows, downtime)
        else:
            bounds = downtime_bounds(capacities, windows, downtime)
            assert (bounds.minimum, bounds.maximum) == (lo, hi)
            check_bounds(capacities, windows, downtime, bounds)
