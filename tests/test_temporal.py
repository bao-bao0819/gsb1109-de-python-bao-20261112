from datetime import date, datetime, timedelta

import pytest

from anytree import Node
from anytree.temporal import (
    REASON_AMBIGUOUS_PARENT,
    REASON_CYCLE,
    TemporalAmbiguityError,
    TemporalDimension,
    TemporalEdge,
    TemporalLoopError,
    TemporalPublishError,
    TemporalTreeError,
)

from .helper import eq_

D = date


def build_category_tree():
    """Tea moves from food to beverages in June, with a late-arriving record."""
    dim = TemporalDimension()
    root = Node("root")
    food = Node("food", parent=root)
    beverages = Node("beverages", parent=root)
    tea = Node("tea", parent=food)
    dim.add_edge(food, root, D(2026, 1, 1))
    dim.add_edge(beverages, root, D(2026, 1, 1))
    # current open-ended relation first, historical interval arrives late
    dim.add_edge(tea, beverages, D(2026, 6, 1))
    dim.add_edge(tea, food, D(2026, 1, 1), D(2026, 6, 1))
    return dim, root, food, beverages, tea


def test_late_migration_keeps_historical_business_time():
    dim, _root, food, beverages, tea = build_category_tree()

    # April order still rolls up through food, regardless of insertion order
    eq_([node.name for node in dim.path_at(tea, D(2026, 4, 15))], ["root", "food", "tea"])
    eq_([node.name for node in dim.ancestors_at(tea, D(2026, 4, 15))], ["root", "food"])
    eq_(dim.parent_at(tea, D(2026, 5, 31)), food)

    # boundary: half-open intervals, the move applies on June 1st
    eq_(dim.parent_at(tea, D(2026, 5, 31)), food)
    eq_(dim.parent_at(tea, D(2026, 6, 1)), beverages)
    eq_([node.name for node in dim.path_at(tea, D(2026, 6, 2))], ["root", "beverages", "tea"])

    result = dim.validate()
    assert result.valid
    eq_(result.invalid_dates, ())
    eq_(result.nodes, ())


def test_half_open_interval_boundaries():
    dim = TemporalDimension()
    a = Node("a")
    b = Node("b")
    dim.add_edge(a, b, D(2026, 1, 1), D(2026, 1, 10))
    eq_(dim.parent_at(a, D(2026, 1, 1)), b)
    eq_(dim.parent_at(a, D(2026, 1, 9)), b)
    eq_(dim.parent_at(a, D(2026, 1, 10)), None)
    eq_(dim.ancestors_at(a, D(2026, 1, 10)), ())
    assert dim.is_valid_at(D(2026, 1, 10))


def test_open_ended_interval():
    dim = TemporalDimension()
    a = Node("a")
    b = Node("b")
    dim.add_edge(a, b, D(2026, 1, 1))
    eq_(dim.parent_at(a, D(2035, 12, 31)), b)
    assert dim.validate().valid


def test_datetime_business_time():
    dim = TemporalDimension()
    a = Node("a")
    b = Node("b")
    start = datetime(2026, 1, 1, 0, 0, 0)
    dim.add_edge(a, b, start, start + timedelta(hours=12))
    eq_(dim.parent_at(a, start + timedelta(hours=11)), b)
    eq_(dim.parent_at(a, start + timedelta(hours=12)), None)


def test_explicit_detachment_interval():
    dim = TemporalDimension()
    a = Node("a")
    b = Node("b")
    dim.add_edge(a, b, D(2026, 1, 1), D(2026, 2, 1))
    dim.add_edge(a, None, D(2026, 2, 1), D(2026, 3, 1))
    dim.add_edge(a, b, D(2026, 3, 1))
    eq_(dim.parent_at(a, D(2026, 1, 15)), b)
    eq_(dim.parent_at(a, D(2026, 2, 15)), None)
    # a is detached and b has no parent of its own, so both are roots that month
    eq_(set(dim.roots_at(D(2026, 2, 15))), {a, b})
    eq_(dim.parent_at(a, D(2026, 3, 2)), b)
    assert dim.validate().valid


def test_full_ancestor_chain_follows_each_version():
    dim = TemporalDimension()
    root = Node("root")
    old = Node("old")
    new = Node("new")
    leaf = Node("leaf")
    dim.add_edge(old, root, D(2026, 1, 1))
    dim.add_edge(new, root, D(2026, 1, 1))
    dim.add_edge(leaf, old, D(2026, 1, 1), D(2026, 6, 1))
    dim.add_edge(leaf, new, D(2026, 6, 1))

    eq_([node.name for node in dim.ancestors_at(leaf, D(2026, 5, 31))], ["root", "old"])
    eq_([node.name for node in dim.path_at(leaf, D(2026, 5, 31))], ["root", "old", "leaf"])
    eq_([node.name for node in dim.ancestors_at(leaf, D(2026, 6, 1))], ["root", "new"])
    eq_([node.name for node in dim.path_at(leaf, D(2026, 7, 1))], ["root", "new", "leaf"])
    eq_(dim.ancestors_at(root, D(2026, 1, 1)), ())
    eq_(dim.path_at(root, D(2026, 1, 1)), (root,))


def test_unregistered_node_rejected():
    dim = TemporalDimension()
    stray = Node("stray")
    with pytest.raises(TemporalTreeError):
        dim.parent_at(stray, D(2026, 1, 1))
    with pytest.raises(TemporalTreeError):
        dim.ancestors_at(stray, D(2026, 1, 1))


def test_non_node_object_rejected():
    dim = TemporalDimension()
    with pytest.raises(TemporalTreeError):
        dim.add_node("not-a-node")
    with pytest.raises(TemporalTreeError):
        dim.add_edge(Node("child"), "not-a-node", D(2026, 1, 1))
    with pytest.raises(TemporalTreeError):
        dim.add_edge("not-a-node", Node("parent"), D(2026, 1, 1))


def test_invalid_interval_rejected():
    dim = TemporalDimension()
    a = Node("a")
    b = Node("b")
    with pytest.raises(ValueError):
        dim.add_edge(a, b, D(2026, 2, 1), D(2026, 2, 1))
    with pytest.raises(ValueError):
        dim.add_edge(a, b, D(2026, 3, 1), D(2026, 2, 1))


def test_edges_and_nodes_registration():
    dim = TemporalDimension()
    a = Node("a")
    b = Node("b")
    c = Node("c")
    dim.add_node(a)
    dim.add_edge(b, a, D(2026, 1, 1), D(2026, 2, 1))
    dim.add_edge(b, c, D(2026, 2, 1))
    eq_(dim.nodes, (a, b, c))
    edges = dim.edges_of(b)
    eq_(len(edges), 2)
    eq_(edges[0], TemporalEdge(b, a, D(2026, 1, 1), D(2026, 2, 1)))
    eq_(edges[1], TemporalEdge(b, c, D(2026, 2, 1), None))
    eq_(dim.edges, edges)
    eq_(dim.edges_of(a), ())


def test_overlapping_parents_are_ambiguous():
    dim = TemporalDimension()
    a = Node("a")
    p1 = Node("p1")
    p2 = Node("p2")
    dim.add_edge(a, p1, D(2026, 1, 1), D(2026, 3, 1))
    dim.add_edge(a, p2, D(2026, 2, 1), D(2026, 4, 1))

    result = dim.validate()
    assert not result.valid
    eq_(result.invalid_dates, ((D(2026, 2, 1), D(2026, 3, 1)),))
    eq_(len(result.failures), 1)
    failure = result.failures[0]
    eq_(failure.reason, REASON_AMBIGUOUS_PARENT)
    eq_(failure.valid_from, D(2026, 2, 1))
    eq_(failure.valid_to, D(2026, 3, 1))
    eq_(set(failure.nodes), {a, p1, p2})
    eq_(set(result.nodes), {a, p1, p2})

    # unaffected dates still resolve
    eq_(dim.parent_at(a, D(2026, 1, 15)), p1)
    eq_(dim.parent_at(a, D(2026, 3, 15)), p2)

    with pytest.raises(TemporalAmbiguityError) as exc_info:
        dim.parent_at(a, D(2026, 2, 15))
    eq_(set(exc_info.value.nodes), {a, p1, p2})
    with pytest.raises(TemporalAmbiguityError):
        dim.ancestors_at(a, D(2026, 2, 15))


def test_cycle_only_on_overlapping_dates():
    dim = TemporalDimension()
    a = Node("a")
    b = Node("b")
    c = Node("c")
    # c -> b and b -> a are fine for the whole window
    dim.add_edge(c, b, D(2026, 1, 1), D(2026, 3, 1))
    dim.add_edge(b, a, D(2026, 1, 1))
    # a -> c only closes the loop in January
    dim.add_edge(a, c, D(2026, 1, 1), D(2026, 2, 1))

    result = dim.validate()
    assert not result.valid
    eq_(result.invalid_dates, ((D(2026, 1, 1), D(2026, 2, 1)),))
    eq_(len(result.failures), 1)
    failure = result.failures[0]
    eq_(failure.reason, REASON_CYCLE)
    eq_(set(failure.nodes), {a, b, c})
    eq_(set(result.nodes), {a, b, c})

    # February onwards remains a clean tree and queryable by version
    assert dim.is_valid_at(D(2026, 2, 15))
    eq_([node.name for node in dim.path_at(c, D(2026, 2, 15))], ["a", "b", "c"])
    eq_(dim.roots_at(D(2026, 2, 15)), (a,))
    assert not dim.is_valid_at(D(2026, 1, 10))

    with pytest.raises(TemporalLoopError) as exc_info:
        dim.ancestors_at(c, D(2026, 1, 10))
    eq_(set(exc_info.value.nodes), {a, b, c})
    with pytest.raises(TemporalLoopError):
        dim.path_at(b, D(2026, 1, 20))


def test_self_loop_detected():
    dim = TemporalDimension()
    a = Node("a")
    dim.add_edge(a, a, D(2026, 1, 1), D(2026, 2, 1))
    result = dim.validate()
    assert not result.valid
    eq_(result.invalid_dates, ((D(2026, 1, 1), D(2026, 2, 1)),))
    eq_(result.failures[0].reason, REASON_CYCLE)
    eq_(result.failures[0].nodes, (a,))
    with pytest.raises(TemporalLoopError):
        dim.ancestors_at(a, D(2026, 1, 1))


def test_multiple_invalid_segments_merged_into_date_list():
    dim = TemporalDimension()
    a = Node("a")
    b = Node("b")
    c = Node("c")
    # cycle in January and March, clean in between
    dim.add_edge(b, a, D(2026, 1, 1))
    dim.add_edge(a, b, D(2026, 1, 1), D(2026, 2, 1))
    dim.add_edge(a, c, D(2026, 3, 1), D(2026, 4, 1))
    dim.add_edge(c, b, D(2026, 3, 1), D(2026, 4, 1))

    result = dim.validate()
    assert not result.valid
    eq_(result.invalid_dates, ((D(2026, 1, 1), D(2026, 2, 1)), (D(2026, 3, 1), D(2026, 4, 1))))
    # February still publishes
    eq_([node.name for node in dim.path_at(b, D(2026, 2, 15))], ["a", "b"])


def test_adjacent_failure_ranges_are_merged():
    dim = TemporalDimension()
    a = Node("a")
    b = Node("b")
    c = Node("c")
    # January: a <-> b cycle; February: a has two parents; the failing ranges touch
    dim.add_edge(b, a, D(2026, 1, 1), D(2026, 2, 1))
    dim.add_edge(a, b, D(2026, 1, 1))
    dim.add_edge(a, c, D(2026, 2, 1), D(2026, 3, 1))
    result = dim.validate()
    assert not result.valid
    eq_(result.invalid_dates, ((D(2026, 1, 1), D(2026, 3, 1)),))
    eq_([failure.reason for failure in result.failures], [REASON_CYCLE, REASON_AMBIGUOUS_PARENT])
    eq_(set(result.nodes), {a, b, c})
    # March is unaffected and still queryable
    eq_(dim.parent_at(a, D(2026, 3, 1)), b)


def test_open_ended_failure_range():
    dim = TemporalDimension()
    a = Node("a")
    b = Node("b")
    dim.add_edge(a, b, D(2026, 1, 1))
    dim.add_edge(b, a, D(2026, 1, 1))
    result = dim.validate()
    assert not result.valid
    eq_(result.invalid_dates, ((D(2026, 1, 1), None),))
    eq_(result.failures[0].reason, REASON_CYCLE)
    with pytest.raises(TemporalLoopError):
        dim.ancestors_at(a, D(2030, 1, 1))


def test_ambiguity_and_cycle_in_same_segment():
    dim = TemporalDimension()
    a = Node("a")
    b = Node("b")
    c = Node("c")
    p1 = Node("p1")
    p2 = Node("p2")
    # cycle a <-> b
    dim.add_edge(a, b, D(2026, 1, 1), D(2026, 2, 1))
    dim.add_edge(b, a, D(2026, 1, 1), D(2026, 2, 1))
    # ambiguous parent for c in the very same segment
    dim.add_edge(c, p1, D(2026, 1, 1), D(2026, 2, 1))
    dim.add_edge(c, p2, D(2026, 1, 1), D(2026, 2, 1))

    result = dim.validate()
    assert not result.valid
    # the two failures cover the same range and are merged into one date entry
    eq_(result.invalid_dates, ((D(2026, 1, 1), D(2026, 2, 1)),))
    eq_(sorted(failure.reason for failure in result.failures), [REASON_AMBIGUOUS_PARENT, REASON_CYCLE])
    eq_(set(result.nodes), {a, b, c, p1, p2})
    # after the segment ends there are no relations left, so later dates publish
    assert dim.is_valid_at(D(2026, 3, 1))


def test_publish_raises_with_result_and_passes_when_clean():
    dim = TemporalDimension()
    a = Node("a")
    b = Node("b")
    dim.add_edge(a, b, D(2026, 1, 1), D(2026, 2, 1))
    dim.add_edge(b, a, D(2026, 1, 1), D(2026, 2, 1))
    with pytest.raises(TemporalPublishError) as exc_info:
        dim.publish()
    eq_(exc_info.value.result.invalid_dates, ((D(2026, 1, 1), D(2026, 2, 1)),))

    clean = TemporalDimension()
    root = Node("root")
    child = Node("child")
    clean.add_edge(child, root, D(2026, 1, 1))
    result = clean.publish()
    assert result.valid


def test_empty_dimension_is_valid():
    dim = TemporalDimension()
    result = dim.validate()
    assert result.valid
    assert bool(result)
    eq_(result.invalid_dates, ())
    eq_(dim.nodes, ())
    eq_(dim.edges, ())
    dim.publish()


def test_validation_result_repr():
    dim = TemporalDimension()
    assert repr(dim.validate()) == "TemporalValidationResult(valid, invalid_dates=())"


def test_forest_with_multiple_roots():
    dim = TemporalDimension()
    r1 = Node("r1")
    r2 = Node("r2")
    child = Node("child")
    dim.add_node(r1)
    dim.add_node(r2)
    dim.add_edge(child, r1, D(2026, 1, 1))
    eq_(sorted(node.name for node in dim.roots_at(D(2026, 1, 1))), ["r1", "r2"])
    assert dim.validate().valid
