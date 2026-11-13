"""Tests for partially disclosed hierarchy bounds."""

from itertools import product
from types import SimpleNamespace

import pytest

from anytree import (
    AnyNode,
    DisclosureBounds,
    DisclosureConflictError,
    Node,
    disclosure_bounds,
    disclosure_group_bounds,
)

from .helper import eq_


def _sample_tree():
    root = Node("root")
    x = Node("x", parent=root)
    a = Node("a", parent=x)
    b = Node("b", parent=x)
    y = Node("y", parent=root)
    c = Node("c", parent=y)
    d = Node("d", parent=y)
    e = Node("e", parent=root)
    return SimpleNamespace(root=root, x=x, y=y, a=a, b=b, c=c, d=d, e=e)


def test_leaf_bounds_basic():
    t = _sample_tree()
    bounds = disclosure_bounds(t.root, {t.root: 10, t.x: 4, t.y: 3})
    eq_(bounds[t.a], DisclosureBounds(0, 4))
    eq_(bounds[t.b], DisclosureBounds(0, 4))
    eq_(bounds[t.c], DisclosureBounds(0, 3))
    eq_(bounds[t.d], DisclosureBounds(0, 3))
    eq_(bounds[t.e], DisclosureBounds(3, 3))


def test_single_undisclosed_leaf_is_pinned():
    t = _sample_tree()
    bounds = disclosure_bounds(t.root, {t.root: 10, t.x: 4})
    eq_(bounds[t.c], DisclosureBounds(0, 6))
    eq_(bounds[t.d], DisclosureBounds(0, 6))
    eq_(bounds[t.e], DisclosureBounds(0, 6))
    eq_(bounds[t.a], DisclosureBounds(0, 4))

    pinned_tree = Node("r")
    px = Node("x", parent=pinned_tree)
    pa = Node("a", parent=px)
    pb = Node("b", parent=pinned_tree)
    pinned_bounds = disclosure_bounds(pinned_tree, {pinned_tree: 10, px: 4})
    eq_(pinned_bounds[pa], DisclosureBounds(4, 4))
    eq_(pinned_bounds[pb], DisclosureBounds(6, 6))


def test_nested_subtotals_share_the_remaining_slack():
    t = _sample_tree()
    bounds = disclosure_bounds(t.root, {t.root: 20, t.x: 6, t.a: 2, t.c: 1})
    eq_(bounds[t.a], DisclosureBounds(2, 2))
    eq_(bounds[t.b], DisclosureBounds(4, 4))
    eq_(bounds[t.c], DisclosureBounds(1, 1))
    eq_(bounds[t.d], DisclosureBounds(0, 13))
    eq_(bounds[t.e], DisclosureBounds(0, 13))


def test_reported_leaf_is_fixed():
    t = _sample_tree()
    bounds = disclosure_bounds(t.root, {t.root: 10, t.x: 4, t.e: 5})
    eq_(bounds[t.e], DisclosureBounds(5, 5))
    eq_(bounds[t.c], DisclosureBounds(0, 1))
    eq_(bounds[t.d], DisclosureBounds(0, 1))
    eq_(bounds[t.a], DisclosureBounds(0, 4))


def test_missing_ancestor_subtotal_means_unknown_upper_bound():
    t = _sample_tree()
    bounds = disclosure_bounds(t.root, {t.x: 4})
    eq_(bounds[t.a], DisclosureBounds(0, 4))
    eq_(bounds[t.b], DisclosureBounds(0, 4))
    eq_(bounds[t.c], DisclosureBounds(0, None))
    eq_(bounds[t.d], DisclosureBounds(0, None))
    eq_(bounds[t.e], DisclosureBounds(0, None))
    assert not bounds[t.e].bounded
    assert bounds[t.a].bounded


def test_conflicting_nested_subtotals():
    t = _sample_tree()
    with pytest.raises(DisclosureConflictError) as excinfo:
        disclosure_bounds(t.root, {t.root: 3, t.x: 4})
    eq_(set(excinfo.value.nodes), {t.root, t.x})


def test_conflicting_deeply_nested_subtotals():
    t = _sample_tree()
    with pytest.raises(DisclosureConflictError) as excinfo:
        disclosure_bounds(t.root, {t.root: 10, t.x: 9, t.a: 5, t.b: 6})
    eq_(set(excinfo.value.nodes), {t.x, t.a, t.b})


def test_conflict_when_all_leaves_disclosed_but_slack_remains():
    t = _sample_tree()
    with pytest.raises(DisclosureConflictError) as excinfo:
        disclosure_bounds(t.root, {t.x: 5, t.a: 2, t.b: 2})
    eq_(set(excinfo.value.nodes), {t.x, t.a, t.b})


def test_feasible_when_disclosed_leaves_exactly_fill_subtotal():
    t = _sample_tree()
    bounds = disclosure_bounds(t.root, {t.x: 5, t.a: 2, t.b: 3})
    eq_(bounds[t.a], DisclosureBounds(2, 2))
    eq_(bounds[t.b], DisclosureBounds(3, 3))


def test_groups_mapping_for_leaf_bounds():
    t = _sample_tree()
    groups = {"first": (t.a, t.c, t.e), "second": (t.b, t.d)}
    result = disclosure_bounds(t.root, {t.root: 10, t.x: 4, t.y: 3}, groups)
    eq_(set(result), {"first", "second"})
    eq_(result["first"][t.a], DisclosureBounds(0, 4))
    eq_(result["first"][t.e], DisclosureBounds(3, 3))
    eq_(result["second"][t.d], DisclosureBounds(0, 3))


def test_flat_group_iterable():
    t = _sample_tree()
    result = disclosure_bounds(t.root, {t.root: 10, t.x: 4}, [t.a, t.e])
    eq_(set(result), {t.a, t.e})
    eq_(result[t.e], DisclosureBounds(0, 6))
    eq_(result[t.a], DisclosureBounds(0, 4))


def test_group_share_bounds():
    t = _sample_tree()
    groups = {"g1": (t.a, t.c), "g2": (t.b,), "g3": (t.e,)}
    result = disclosure_group_bounds(t.root, {t.root: 20, t.x: 6, t.y: 4}, groups)
    eq_(result["g1"], DisclosureBounds(0, 10))
    eq_(result["g2"], DisclosureBounds(0, 6))
    eq_(result["g3"], DisclosureBounds(10, 10))


def test_group_share_lower_bound_from_companion_leaves():
    t = _sample_tree()
    result = disclosure_group_bounds(t.root, {t.root: 10, t.x: 4}, {"g": (t.a,)})
    eq_(result["g"], DisclosureBounds(0, 4))
    result = disclosure_group_bounds(t.root, {t.root: 10, t.x: 4}, {"g": (t.a, t.b)})
    eq_(result["g"], DisclosureBounds(4, 4))
    result = disclosure_group_bounds(t.root, {t.root: 10, t.x: 4}, {"g": (t.a, t.e)})
    eq_(result["g"], DisclosureBounds(0, 10))
    result = disclosure_group_bounds(t.root, {t.root: 10, t.x: 4}, {"g": (t.c, t.d, t.e)})
    eq_(result["g"], DisclosureBounds(6, 6))


def test_group_share_unknown_upper_bound():
    t = _sample_tree()
    result = disclosure_group_bounds(t.root, {t.x: 4}, {"g": (t.a, t.e)})
    eq_(result["g"], DisclosureBounds(0, None))
    flat = disclosure_group_bounds(t.root, {t.x: 4}, [t.a, t.b])
    eq_(flat, DisclosureBounds(4, 4))


def test_group_with_only_reported_leaves_has_zero_share():
    t = _sample_tree()
    result = disclosure_group_bounds(
        t.root,
        {t.root: 10, t.x: 4, t.e: 6},
        {"known": (t.e,), "mixed": (t.e, t.a)},
    )
    eq_(result["known"], DisclosureBounds(0, 0))
    eq_(result["mixed"], DisclosureBounds(0, 4))


def test_leaf_attr_keying():
    root = AnyNode(id="root")
    left = AnyNode(id="left", parent=root)
    p = AnyNode(id="p", parent=left)
    q = AnyNode(id="q", parent=left)
    r = AnyNode(id="r", parent=root)
    bounds = disclosure_bounds(root, {"root": 9, "left": 5}, {"g": ("p", "r")}, leaf_attr="id")
    eq_(bounds["g"]["p"], DisclosureBounds(0, 5))
    eq_(bounds["g"]["r"], DisclosureBounds(4, 4))
    group = disclosure_group_bounds(root, {"root": 9, "left": 5}, {"g": ("p", "r")}, leaf_attr="id")
    eq_(group["g"], DisclosureBounds(4, 9))


def test_duplicate_leaf_attr_rejected():
    root = AnyNode(code="r")
    AnyNode(code="dup", parent=root)
    AnyNode(code="dup", parent=root)
    with pytest.raises(ValueError):
        disclosure_bounds(root, {"r": 2}, leaf_attr="code")


def test_invalid_subtotal_values():
    t = _sample_tree()
    with pytest.raises(ValueError):
        disclosure_bounds(t.root, {t.x: -1})
    with pytest.raises(TypeError):
        disclosure_bounds(t.root, {t.x: 1.5})
    with pytest.raises(TypeError):
        disclosure_bounds(t.root, [(t.x, 1)])
    with pytest.raises(ValueError):
        disclosure_bounds(t.root, {Node("foreign"): 1})


def test_invalid_groups():
    t = _sample_tree()
    with pytest.raises(ValueError):
        disclosure_bounds(t.root, {t.root: 10}, [t.x])
    with pytest.raises(TypeError):
        disclosure_bounds(t.root, {t.root: 10}, "not-a-group")
    with pytest.raises(ValueError):
        disclosure_group_bounds(t.root, {t.root: 10}, {"g": [Node("foreign")]})


def _brute_force_bounds(tree, subtotals, cap=8):
    leaves = tree.leaves
    feasible = []
    for values in product(range(cap + 1), repeat=len(leaves)):
        assignment = dict(zip(leaves, values))
        if all(sum(assignment[leaf] for leaf in node.leaves) == total for node, total in subtotals.items()):
            feasible.append(assignment)
    assert feasible, "test setup needs at least one feasible assignment"
    leaf_bounds = {}
    for leaf in leaves:
        vals = [assignment[leaf] for assignment in feasible]
        leaf_bounds[leaf] = (min(vals), max(vals))
    return feasible, leaf_bounds


def test_bounds_match_brute_force_leaf_extrema():
    t = _sample_tree()
    subtotals = {t.root: 12, t.x: 5, t.y: 4}
    _, exact = _brute_force_bounds(t.root, subtotals)
    bounds = disclosure_bounds(t.root, subtotals)
    for leaf, (lower, upper) in exact.items():
        eq_(bounds[leaf], DisclosureBounds(lower, upper))


def test_group_bounds_match_brute_force_group_extrema():
    t = _sample_tree()
    subtotals = {t.root: 14, t.x: 6}
    groups = {"g1": (t.a, t.c), "g2": (t.b, t.d, t.e)}
    feasible, _ = _brute_force_bounds(t.root, subtotals)
    exact = {}
    for name, members in groups.items():
        totals = [sum(assignment[leaf] for leaf in members) for assignment in feasible]
        exact[name] = (min(totals), max(totals))
    result = disclosure_group_bounds(t.root, subtotals, groups)
    for name, (lower, upper) in exact.items():
        eq_(result[name], DisclosureBounds(lower, upper))


def test_bounds_are_integrally_attainable():
    t = _sample_tree()
    subtotals = {t.root: 11, t.x: 7}
    feasible, _ = _brute_force_bounds(t.root, subtotals, cap=11)
    leaf_bounds = disclosure_bounds(t.root, subtotals)
    for leaf in (t.a, t.b, t.c, t.d, t.e):
        lower, upper = leaf_bounds[leaf]
        assert any(assignment[leaf] == lower for assignment in feasible)
        assert any(assignment[leaf] == upper for assignment in feasible)
    group_bounds = disclosure_group_bounds(t.root, subtotals, {"g": (t.a, t.c, t.e)})
    lower, upper = group_bounds["g"]
    totals = [assignment[t.a] + assignment[t.c] + assignment[t.e] for assignment in feasible]
    assert min(totals) == lower
    assert max(totals) == upper
