import itertools
import random

import pytest

import portion as P


# --------------------------------------------------------------------------- #
# Domain construction
# --------------------------------------------------------------------------- #
class TestMakeDomain:
    def test_singletons_and_ranges(self):
        domain = P.make_domain(2, (4, 6), 9)
        assert domain == P.closed(2, 2) | P.closed(4, 6) | P.closed(9, 9)
        assert list(P.iterate(domain, 1)) == [2, 4, 5, 6, 9]

    def test_disconnected_ranges_keep_gaps(self):
        domain = P.make_domain((1, 2), (8, 10))
        assert len(domain) == 2
        assert 3 not in domain and 7 not in domain
        assert list(P.iterate(domain, 1)) == [1, 2, 8, 9, 10]

    def test_open_bounds_normalized_to_integer_values(self):
        model = P.TemporalModel({"a": P.openclosed(0, 4) - P.closed(2, 2)})
        # Open lower bound 0 excludes 0; the removed singleton keeps a gap at 2.
        assert list(P.iterate(model.domain("a"), 1)) == [1, 3, 4]

    def test_infinite_bounds_rejected(self):
        with pytest.raises(ValueError):
            P.TemporalModel({"a": P.closed(0, P.inf)})

    def test_wrong_domain_type_rejected(self):
        with pytest.raises(TypeError):
            P.TemporalModel({"a": (1, 5)})


class TestConstraintFactories:
    def test_distance_validation(self):
        with pytest.raises(ValueError):
            P.min_gap("a", "b", -1)
        with pytest.raises(ValueError):
            P.max_wait("a", "b", 1.5)
        with pytest.raises(ValueError):
            P.min_gap("a", "b", True)

    def test_factories(self):
        assert P.before("a", "b") == P.TemporalConstraint("before", "a", "b", 0)
        assert P.min_gap("a", "b", 3) == P.TemporalConstraint("min_gap", "a", "b", 3)
        assert P.max_wait("a", "b", 2) == P.TemporalConstraint("max_wait", "a", "b", 2)

    def test_unknown_event_rejected(self):
        model = P.TemporalModel({"a": P.make_domain((1, 5))})
        with pytest.raises(KeyError):
            model.add(P.before("a", "b"))


# --------------------------------------------------------------------------- #
# Propagation and feasible schedules
# --------------------------------------------------------------------------- #
class TestPropagation:
    def test_reduced_domain_preserves_gaps(self):
        model = P.TemporalModel(
            {
                "a": P.make_domain((1, 2), (6, 8)),
                "b": P.make_domain((2, 3), (7, 9)),
            },
            [P.min_gap("a", "b", 2)],
        )
        assert model.reduced_domain("a") == P.make_domain((1, 2), (6, 7))
        assert model.reduced_domain("b") == P.make_domain(3, (7, 9))
        assert not model.reduced_domain("a").atomic

    def test_reduced_domain_never_extends_original(self):
        model = P.TemporalModel(
            {"a": P.make_domain(1, 5, 9), "b": P.make_domain((2, 8))},
            [P.before("b", "a")],
        )
        for event in model.events:
            assert model.reduced_domain(event) in model.domain(event)

    def test_schedule_jumps_over_gap_instead_of_filling_it(self):
        # b is fixed at 4 and must follow a by at least 2: a <= 2, so the
        # second admissible window [6, 7] of a must be dropped entirely.
        model = P.TemporalModel(
            {"a": P.make_domain((1, 2), (6, 7)), "b": P.make_domain(4)},
            [P.min_gap("a", "b", 2)],
        )
        schedule = model.feasible_schedule()
        assert schedule == {"a": 1, "b": 4}
        assert schedule["a"] in model.domain("a")

    def test_earliest_and_latest_schedules(self):
        model = P.TemporalModel(
            {
                "a": P.make_domain((1, 3), (8, 10)),
                "b": P.make_domain((2, 4), (9, 11)),
            },
            [P.min_gap("a", "b", 1), P.max_wait("a", "b", 3)],
        )
        earliest = model.feasible_schedule("earliest")
        latest = model.feasible_schedule("latest")
        assert earliest == {"a": 1, "b": 2}
        assert latest == {"a": 10, "b": 11}
        assert _is_feasible(model, earliest)
        assert _is_feasible(model, latest)

    def test_invalid_prefer_rejected(self):
        model = P.TemporalModel({"a": P.make_domain((1, 3))})
        with pytest.raises(ValueError):
            model.feasible_schedule("middle")

    def test_add_resets_cached_state(self):
        model = P.TemporalModel(
            {"a": P.make_domain((1, 5)), "b": P.make_domain((1, 5))}
        )
        assert model.is_consistent()
        model.add(P.min_gap("a", "b", 10))
        assert not model.is_consistent()


# --------------------------------------------------------------------------- #
# Order verdicts
# --------------------------------------------------------------------------- #
class TestOrderOf:
    def test_reversible_with_supporting_schedules(self):
        model = P.TemporalModel(
            {"a": P.make_domain((1, 5)), "b": P.make_domain((1, 5))}
        )
        result = model.order_of("a", "b")
        assert result.verdict == "reversible"
        assert result.earlier is None
        assert result.schedule["a"] < result.schedule["b"]
        assert result.reverse_schedule["b"] < result.reverse_schedule["a"]
        assert _is_feasible(model, result.schedule)
        assert _is_feasible(model, result.reverse_schedule)

    def test_necessary_order(self):
        model = P.TemporalModel(
            {"a": P.make_domain((1, 3)), "b": P.make_domain((5, 8))}
        )
        result = model.order_of("a", "b")
        assert result.verdict == "necessary"
        assert result.earlier == "a"
        assert result.schedule["a"] < result.schedule["b"]
        assert result.reverse_schedule is None

    def test_necessary_order_induced_by_min_gap(self):
        model = P.TemporalModel(
            {"a": P.make_domain((1, 6)), "b": P.make_domain((1, 6))},
            [P.min_gap("a", "b", 2)],
        )
        result = model.order_of("a", "b")
        assert result.verdict == "necessary"
        assert result.earlier == "a"

    def test_simultaneous(self):
        model = P.TemporalModel(
            {"a": P.make_domain((1, 5)), "b": P.make_domain((1, 5))},
            [P.before("a", "b"), P.before("b", "a")],
        )
        result = model.order_of("a", "b")
        assert result.verdict == "simultaneous"
        assert result.schedule["a"] == result.schedule["b"]
        assert result.reverse_schedule is None

    def test_simultaneous_through_zero_max_wait(self):
        model = P.TemporalModel(
            {"a": P.make_domain(1, 2), "b": P.make_domain(1, 2)},
            [P.max_wait("a", "b", 0), P.before("a", "b")],
        )
        result = model.order_of("a", "b")
        assert result.verdict == "simultaneous"
        assert result.schedule == {"a": 1, "b": 1}

    def test_infeasible_returns_conflicts_instead_of_schedule(self):
        model = P.TemporalModel(
            {event: P.make_domain((1, 10)) for event in "abc"},
            [
                P.min_gap("a", "b", 1),
                P.min_gap("b", "c", 1),
                P.min_gap("c", "a", 1),
            ],
        )
        result = model.order_of("a", "b")
        assert result.verdict == "infeasible"
        assert result.schedule is None
        assert result.conflicts[0]["kind"] == "cycle"
        assert result.conflicts[0]["events"] == ["a", "b", "c", "a"]

    def test_unknown_event(self):
        model = P.TemporalModel({"a": P.make_domain((1, 3))})
        with pytest.raises(KeyError):
            model.order_of("a", "b")


# --------------------------------------------------------------------------- #
# Inconsistency explanations
# --------------------------------------------------------------------------- #
class TestConflicts:
    def test_strict_ordering_cycle(self):
        model = P.TemporalModel(
            {"a": P.make_domain((1, 4)), "b": P.make_domain((1, 4))},
            [P.min_gap("a", "b", 1), P.min_gap("b", "a", 1)],
        )
        conflicts = model.conflicts()
        assert conflicts[0]["kind"] == "cycle"
        assert set(conflicts[0]["constraints"]) == {
            P.min_gap("a", "b", 1),
            P.min_gap("b", "a", 1),
        }
        with pytest.raises(P.InconsistentRecords):
            model.feasible_schedule()

    def test_max_wait_cycle_through_negative_edge(self):
        # t[b] - t[a] <= 0 together with t[a] < t[b] is contradictory.
        model = P.TemporalModel(
            {"a": P.make_domain((1, 4)), "b": P.make_domain((1, 4))},
            [P.max_wait("a", "b", 0), P.min_gap("a", "b", 1)],
        )
        assert not model.is_consistent()
        assert model.conflicts()[0]["kind"] == "cycle"

    def test_numeric_domain_contradiction(self):
        model = P.TemporalModel(
            {"a": P.make_domain((1, 2)), "b": P.make_domain((1, 2))},
            [P.min_gap("a", "b", 10)],
        )
        conflict = model.conflicts()[0]
        assert conflict["kind"] == "domain_emptied"
        assert conflict["event"] == "b" or conflict["event"] == "a"
        assert conflict["constraint"] == P.min_gap("a", "b", 10)

    def test_excluded_gap_is_not_filled(self):
        # a may only happen at 1 or 10; b is fixed at 5 and waits at most 2
        # units after a. Filling the outage gap would wrongly allow a = 3..5.
        model = P.TemporalModel(
            {"a": P.make_domain(1, 10), "b": P.make_domain(5)},
            [P.max_wait("a", "b", 2), P.before("a", "b")],
        )
        conflict = model.conflicts()[0]
        assert conflict["kind"] == "excluded_gap"
        assert conflict["event"] == "a"
        assert conflict["admissible"] == P.make_domain(1, 10)
        assert 3 in conflict["window"] and 5 in conflict["window"]

    def test_empty_domain(self):
        model = P.TemporalModel({"a": P.make_domain((1, 0))})
        assert model.conflicts()[0]["kind"] == "empty_domain"


# --------------------------------------------------------------------------- #
# Connectivity
# --------------------------------------------------------------------------- #
class TestComponents:
    def test_independent_groups_are_separate(self):
        model = P.TemporalModel(
            {
                "a": P.make_domain((1, 2)),
                "b": P.make_domain((8, 9)),
                "c": P.make_domain((1, 2)),
            },
            [P.min_gap("a", "b", 1)],
        )
        components = model.components()
        assert frozenset(map(frozenset, components)) == frozenset(
            [frozenset(("a", "b")), frozenset(("c",))]
        )

    def test_isolated_events_do_not_share_time_windows(self):
        # c has no record tying it to a and b; its outage gap stays its own.
        model = P.TemporalModel(
            {
                "a": P.make_domain(1, 5),
                "b": P.make_domain((2, 6)),
                "c": P.make_domain(1, 9),
            },
            [P.before("a", "b")],
        )
        schedule = model.feasible_schedule()
        assert schedule["c"] == 1
        assert all(value in model.domain(event) for event, value in schedule.items())


# --------------------------------------------------------------------------- #
# Brute-force cross-validation on random small instances
# --------------------------------------------------------------------------- #
def _values(model, event):
    return list(P.iterate(model.domain(event), 1))


def _constraint_holds(constraint, assignment):
    delta = assignment[constraint.b] - assignment[constraint.a]
    if constraint.kind == "before":
        return delta >= 0
    if constraint.kind == "min_gap":
        return delta >= constraint.value
    return delta <= constraint.value


def _brute_force(model):
    events = model.events
    options = [_values(model, event) for event in events]
    solutions = []
    for combo in itertools.product(*options):
        assignment = dict(zip(events, combo))
        if all(_constraint_holds(c, assignment) for c in model.constraints):
            solutions.append(assignment)
    return solutions


def _brute_force_verdict(model, a, b, solutions):
    if not solutions:
        return "infeasible"
    a_before = any(s[a] < s[b] for s in solutions)
    b_before = any(s[b] < s[a] for s in solutions)
    if a_before and b_before:
        return "reversible"
    if a_before:
        return "necessary"
    if b_before:
        return "necessary"
    return "simultaneous"


def _is_feasible(model, assignment):
    if any(value not in model.domain(event) for event, value in assignment.items()):
        return False
    return all(_constraint_holds(c, assignment) for c in model.constraints)


def _random_model(rng):
    names = ["a", "b", "c"]
    domains = {}
    for name in names:
        chosen = sorted(rng.sample(range(0, 6), rng.randint(1, 4)))
        domains[name] = P.make_domain(*chosen)

    model = P.TemporalModel(domains)
    for _ in range(rng.randint(0, 4)):
        a, b = rng.sample(names, 2)
        kind = rng.choice(["before", "min_gap", "max_wait"])
        if kind == "before":
            model.add(P.before(a, b))
        elif kind == "min_gap":
            model.add(P.min_gap(a, b, rng.randint(0, 3)))
        else:
            model.add(P.max_wait(a, b, rng.randint(0, 3)))
    return model


@pytest.mark.parametrize("seed", range(200))
def test_random_models_match_brute_force(seed):
    rng = random.Random(seed)
    model = _random_model(rng)
    solutions = _brute_force(model)

    assert model.is_consistent() == bool(solutions)
    if not solutions:
        assert model.conflicts()
        with pytest.raises(P.InconsistentRecords):
            model.feasible_schedule()
        return

    for prefer in ("earliest", "latest"):
        schedule = model.feasible_schedule(prefer)
        assert _is_feasible(model, schedule)
        # No time may come from an excluded gap.
        for event, value in schedule.items():
            assert value in _values(model, event)

    for a, b in itertools.combinations(model.events, 2):
        result = model.order_of(a, b)
        expected = _brute_force_verdict(model, a, b, solutions)
        if expected == "necessary":
            assert result.verdict == "necessary"
            expected_earlier = a if any(s[a] < s[b] for s in solutions) else b
            assert result.earlier == expected_earlier
        else:
            assert result.verdict == expected
        if result.schedule is not None:
            assert _is_feasible(model, result.schedule)
        if result.reverse_schedule is not None:
            assert _is_feasible(model, result.reverse_schedule)
            assert result.reverse_schedule[b] < result.reverse_schedule[a]
