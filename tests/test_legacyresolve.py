import pytest

import anytree as at
from anytree import Node
from anytree.legacyresolve import (
    ConflictGroup,
    JointGroup,
    LegacyRecord,
    LegacyResolveError,
    LegacyResolver,
    SameNode,
    SameParent,
)


def _tree():
    # corp(1)
    # ├── north(2)
    # │   ├── sales(4)
    # │   │   └── ops(8)
    # │   │       └── sales(10)
    # │   └── ops(6)
    # └── south(3)
    #     ├── sales(5)
    #     └── ops(7)
    corp = Node("corp", id=1)
    north = Node("north", parent=corp, id=2)
    south = Node("south", parent=corp, id=3)
    sales_north = Node("sales", parent=north, id=4)
    sales_south = Node("sales", parent=south, id=5)
    ops_north = Node("ops", parent=north, id=6)
    ops_south = Node("ops", parent=south, id=7)
    ops_deep = Node("ops", parent=sales_north, id=8)
    sales_deep = Node("sales", parent=ops_deep, id=10)
    return corp, {
        "corp": corp,
        "north": north,
        "south": south,
        "sales_north": sales_north,
        "sales_south": sales_south,
        "ops_north": ops_north,
        "ops_south": ops_south,
        "ops_deep": ops_deep,
        "sales_deep": sales_deep,
    }


def test_candidates_duplicate_names():
    """Department names may occur more than once."""
    corp, nodes = _tree()
    resolver = LegacyResolver(corp)
    assert resolver.candidates(("corp", "sales")) == [
        nodes["sales_north"],
        nodes["sales_deep"],
        nodes["sales_south"],
    ]
    assert resolver.candidates(("sales",)) == [
        nodes["sales_north"],
        nodes["sales_deep"],
        nodes["sales_south"],
    ]


def test_candidates_gap_in_the_middle():
    """Missing segments may be in the middle of the path."""
    corp, nodes = _tree()
    resolver = LegacyResolver(corp)
    # 'ops' between 'north' and 'sales' is missing in the fragments
    assert resolver.candidates(("corp", "north", "sales")) == [
        nodes["sales_north"],
        nodes["sales_deep"],
    ]
    # order matters: 'ops' before 'sales' does not match 'sales_north'
    assert resolver.candidates(("ops", "sales")) == [nodes["sales_deep"]]
    # repeated fragments require repeated occurrences on the path
    assert resolver.candidates(("sales", "sales")) == [nodes["sales_deep"]]


def test_candidates_empty_fragments():
    corp, _ = _tree()
    resolver = LegacyResolver(corp)
    with pytest.raises(LegacyResolveError):
        resolver.candidates(())


def test_resolve_unique():
    corp, nodes = _tree()
    resolver = LegacyResolver(corp)
    record = LegacyRecord("E1", ("corp", "south", "ops"))
    resolution = resolver.resolve([record])
    assert resolution.complete
    assert resolution.resolved == {"E1": nodes["ops_south"]}
    assert resolution.resolved_ids == {"E1": 7}
    assert not resolution.ambiguous
    assert not resolution.conflicts
    assert not resolution.unmatched


def test_resolve_unmatched():
    corp, _ = _tree()
    resolver = LegacyResolver(corp)
    record = LegacyRecord("E1", ("corp", "unknown"))
    resolution = resolver.resolve([record])
    assert not resolution.complete
    assert resolution.unmatched == {"E1": record}
    assert not resolution.resolved


def test_resolve_ambiguous_single_record():
    """Not enough evidence: all feasible mappings of one record are reported."""
    corp, nodes = _tree()
    resolver = LegacyResolver(corp)
    resolution = resolver.resolve([LegacyRecord("E1", ("corp", "ops"))])
    assert not resolution.complete
    assert len(resolution.ambiguous) == 1
    group = resolution.ambiguous[0]
    assert group.keys == ("E1",)
    assert group.placements == [
        {"E1": nodes["ops_deep"]},
        {"E1": nodes["ops_north"]},
        {"E1": nodes["ops_south"]},
    ]
    assert group.open == ("E1",)
    assert group.fixed == {}


def test_resolve_jointly_unique():
    """Two records with two options each, same-parent constraint rules out one combination."""
    corp = Node("corp", id=1)
    north = Node("north", parent=corp, id=2)
    south = Node("south", parent=corp, id=3)
    sales_north = Node("sales", parent=north, id=4)
    sales_south = Node("sales", parent=south, id=5)
    ops_south = Node("ops", parent=south, id=7)
    ops_nested = Node("ops", parent=sales_north, id=8)

    resolver = LegacyResolver(corp)
    alice = LegacyRecord("E1", ("corp", "sales"))
    bob = LegacyRecord("E2", ("corp", "ops"))
    # individually ambiguous
    assert resolver.candidates(alice.fragments) == [sales_north, sales_south]
    assert resolver.candidates(bob.fragments) == [ops_nested, ops_south]

    resolution = resolver.resolve([alice, bob], constraints=[SameParent("E1", "E2")])
    # only (sales_south, ops_south) share a parent
    assert resolution.complete
    assert resolution.resolved == {"E1": sales_south, "E2": ops_south}
    assert resolution.resolved_ids == {"E1": 5, "E2": 7}


def test_resolve_jointly_ambiguous():
    """Same-parent evidence is not sufficient: two feasible joint mappings remain."""
    corp = Node("corp", id=1)
    north = Node("north", parent=corp, id=2)
    south = Node("south", parent=corp, id=3)
    sales_north = Node("sales", parent=north, id=4)
    sales_south = Node("sales", parent=south, id=5)
    ops_north = Node("ops", parent=north, id=6)
    ops_south = Node("ops", parent=south, id=7)

    resolver = LegacyResolver(corp)
    alice = LegacyRecord("E1", ("sales",))
    bob = LegacyRecord("E2", ("ops",))
    resolution = resolver.resolve([alice, bob], constraints=[SameParent("E1", "E2")])
    assert not resolution.complete
    assert len(resolution.ambiguous) == 1
    group = resolution.ambiguous[0]
    assert group.keys == ("E1", "E2")
    assert group.placements == [
        {"E1": sales_north, "E2": ops_north},
        {"E1": sales_south, "E2": ops_south},
    ]
    assert not group.truncated
    assert group.fixed == {}
    assert group.open == ("E1", "E2")
    assert not resolution.resolved


def test_resolve_same_node():
    """Records known to have been at the very same node."""
    corp, nodes = _tree()
    resolver = LegacyResolver(corp)
    alice = LegacyRecord("E1", ("corp", "sales"))
    bob = LegacyRecord("E2", ("corp", "south", "sales"))
    resolution = resolver.resolve([alice, bob], constraints=[SameNode("E1", "E2")])
    assert resolution.complete
    assert resolution.resolved == {"E1": nodes["sales_south"], "E2": nodes["sales_south"]}


def test_resolve_conflict():
    """Contradictory constraints: no joint mapping exists."""
    corp, nodes = _tree()
    resolver = LegacyResolver(corp)
    alice = LegacyRecord("E1", ("corp", "sales"))
    bob = LegacyRecord("E2", ("corp", "ops"))
    resolution = resolver.resolve([alice, bob], constraints=[SameNode("E1", "E2")])
    assert not resolution.complete
    assert not resolution.resolved
    assert len(resolution.conflicts) == 1
    conflict = resolution.conflicts[0]
    assert conflict.keys == ("E1", "E2")
    # the mutually contradictory placement options are reported
    assert conflict.candidates == {
        "E1": (nodes["sales_north"], nodes["sales_deep"], nodes["sales_south"]),
        "E2": (nodes["ops_deep"], nodes["ops_north"], nodes["ops_south"]),
    }


def test_resolve_conflict_unmatched_member():
    """A record without candidates makes its whole component a conflict."""
    corp, _ = _tree()
    resolver = LegacyResolver(corp)
    alice = LegacyRecord("E1", ("corp", "sales"))
    bob = LegacyRecord("E2", ("corp", "unknown"))
    resolution = resolver.resolve([alice, bob], constraints=[SameParent("E1", "E2")])
    assert len(resolution.conflicts) == 1
    assert resolution.conflicts[0].candidates["E2"] == ()
    assert not resolution.unmatched


def test_resolve_constraint_chain():
    """Constraints chain records into one component."""
    corp = Node("corp", id=1)
    north = Node("north", parent=corp, id=2)
    south = Node("south", parent=corp, id=3)
    sales_north = Node("sales", parent=north, id=4)
    Node("sales", parent=south, id=5)
    ops_north = Node("ops", parent=north, id=6)
    Node("ops", parent=south, id=7)

    resolver = LegacyResolver(corp)
    alice = LegacyRecord("E1", ("sales",))
    bob = LegacyRecord("E2", ("ops",))
    carol = LegacyRecord("E3", ("corp", "north", "sales"))
    resolution = resolver.resolve(
        [alice, bob, carol],
        constraints=[SameParent("E1", "E2"), SameNode("E1", "E3")],
    )
    # carol anchors the whole chain to the north branch
    assert resolution.complete
    assert resolution.resolved == {"E1": sales_north, "E2": ops_north, "E3": sales_north}


def test_resolve_independent_records():
    """Unconstrained records are resolved independently."""
    corp, nodes = _tree()
    resolver = LegacyResolver(corp)
    alice = LegacyRecord("E1", ("corp", "south", "sales"))
    bob = LegacyRecord("E2", ("corp", "south", "ops"))
    resolution = resolver.resolve([alice, bob])
    assert resolution.complete
    assert resolution.resolved == {"E1": nodes["sales_south"], "E2": nodes["ops_south"]}


def test_resolve_max_solutions():
    corp, _ = _tree()
    resolver = LegacyResolver(corp)
    resolution = resolver.resolve([LegacyRecord("E1", ("corp", "sales"))], max_solutions=2)
    group = resolution.ambiguous[0]
    assert group.truncated
    assert len(group.placements) == 2
    # truncated groups are not promoted to resolved
    assert not resolution.resolved


def test_resolve_invalid_input():
    corp, _ = _tree()
    resolver = LegacyResolver(corp)
    with pytest.raises(LegacyResolveError, match="not a LegacyRecord"):
        resolver.resolve([("E1", ("corp",))])
    with pytest.raises(LegacyResolveError, match="duplicate legacy key"):
        resolver.resolve([LegacyRecord("E1", ("corp",)), LegacyRecord("E1", ("corp",))])
    with pytest.raises(LegacyResolveError, match="unknown legacy key"):
        resolver.resolve([LegacyRecord("E1", ("corp",))], constraints=[SameParent("E1", "E2")])
    with pytest.raises(LegacyResolveError, match="not a RecordConstraint"):
        resolver.resolve([LegacyRecord("E1", ("corp",))], constraints=[("E1", "E2")])


def test_record_requires_fragments():
    with pytest.raises(LegacyResolveError, match="no fragments"):
        LegacyRecord("E1", ())


def test_constraint_requires_two_keys():
    with pytest.raises(LegacyResolveError, match="at least two keys"):
        SameParent("E1")
    with pytest.raises(LegacyResolveError, match="at least two keys"):
        SameNode("E1")


def test_constraint_accepts_key_list():
    assert SameParent(["E1", "E2"]).keys == ("E1", "E2")
    assert SameNode(("E1", "E2", "E3")).keys == ("E1", "E2", "E3")


def test_constraint_consistent():
    _, nodes = _tree()
    same_parent = SameParent("E1", "E2")
    assert same_parent.consistent({})
    assert same_parent.consistent({"E1": nodes["sales_north"]})
    assert same_parent.consistent({"E1": nodes["sales_north"], "E2": nodes["ops_north"]})
    assert not same_parent.consistent({"E1": nodes["sales_north"], "E2": nodes["ops_south"]})
    same_node = SameNode("E1", "E2")
    assert same_node.consistent({"E1": nodes["sales_north"], "E2": nodes["sales_north"]})
    assert not same_node.consistent({"E1": nodes["sales_north"], "E2": nodes["sales_south"]})


def test_joint_group_fixed():
    node_a = Node("a")
    node_b = Node("b", parent=node_a)
    node_c = Node("c", parent=node_a)
    group = JointGroup(
        ("E1", "E2"),
        [{"E1": node_b, "E2": node_c}, {"E1": node_c, "E2": node_c}],
    )
    assert group.fixed == {"E2": node_c}
    assert group.open == ("E1",)
    assert repr(group) == "JointGroup(keys=('E1', 'E2'), placements=2)"


def test_conflict_group_repr():
    node_a = Node("a")
    group = ConflictGroup(("E1",), {"E1": [node_a]})
    assert repr(group) == "ConflictGroup(keys=('E1',))"


def test_resolution_node_ids():
    corp, _ = _tree()
    resolver = LegacyResolver(corp)
    resolution = resolver.resolve([LegacyRecord("E1", ("corp", "south", "ops"))])
    assert resolution.node_ids(resolution.resolved) == {"E1": 7}
    assert repr(resolution) == "Resolution(resolved=1, ambiguous=0, conflicts=0, unmatched=0)"


def test_pathattr():
    """Match against a custom node attribute."""
    corp = Node("corp", key="c")
    node = Node("node", parent=corp, key="n")
    resolver = LegacyResolver(corp, pathattr="key")
    assert resolver.candidates(("c", "n")) == [node]
    assert resolver.candidates(("corp", "node")) == []


def test_package_exports():
    assert at.LegacyResolver is LegacyResolver
    assert at.LegacyRecord is LegacyRecord
    assert at.LegacyResolveError is LegacyResolveError
    assert at.SameParent is SameParent
    assert at.SameNode is SameNode
    assert at.Resolution is not None
    assert at.JointGroup is JointGroup
    assert at.ConflictGroup is ConflictGroup
    assert at.RecordConstraint is not None
