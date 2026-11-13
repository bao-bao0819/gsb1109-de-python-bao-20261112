"""Tests for organizational dimension table publishing."""

import io
import json

from anytree import AnyNode, Node, OrderConflictError, OrgStructureError, PreOrderIter, publish_org_tree

from .helper import assert_raises, eq_

RELATIONS = {
    "root": None,
    "a": "root",
    "b": "root",
    "c": "root",
    "x": "a",
    "y": "a",
    "z": "b",
}

OLD = {
    "root": None,
    "c": "root",
    "b": "root",
    "a": "root",
    "z": "b",
    "y": "a",
    "x": "a",
}


def child_ids(node):
    return tuple(child.id for child in node.children)


def test_publish_builds_complete_ordered_tree():
    """The final relations become one complete tree with stable identifiers."""
    publication = publish_org_tree(RELATIONS)
    root = publication.root
    eq_(root.id, "root")
    eq_(child_ids(root), ("a", "b", "c"))
    eq_(child_ids(root.children[0]), ("x", "y"))
    eq_(child_ids(root.children[1]), ("z",))
    eq_(len(publication), len(RELATIONS))
    eq_(set(publication.nodes), set(RELATIONS))
    eq_(tuple(node.id for node in PreOrderIter(root)), ("root", "a", "x", "y", "b", "z", "c"))


def test_explicit_ordering_swap_two_departments():
    """An explicit constraint swaps two sibling departments."""
    publication = publish_org_tree(RELATIONS, [("b", "a")], old=OLD)
    eq_(child_ids(publication.root), ("c", "b", "a"))
    eq_(publication.verify(), True)


def test_minimal_inversions_of_old_sibling_order():
    """Old sibling relationships are kept unless an explicit constraint overrides them."""
    publication = publish_org_tree(RELATIONS, [("b", "c")], old=OLD)
    # old root order was c, b, a and b must precede c; the unique minimum
    # inversion arrangement is b, c, a.
    eq_(child_ids(publication.root), ("b", "c", "a"))
    # untouched sibling groups keep the old order
    node_a = publication.nodes["a"]
    eq_(child_ids(node_a), ("y", "x"))
    eq_(publication.verify(), True)


def test_ties_resolved_by_stable_identifier():
    """Departments without an old position fall back to stable id ordering."""
    relations = dict(RELATIONS)
    relations["newbie"] = "root"
    publication = publish_org_tree(relations, old=OLD)
    eq_(child_ids(publication.root), ("c", "b", "a", "newbie"))
    eq_(publication.verify(), True)


def test_no_old_organization_uses_stable_identifiers():
    """Without an old organization every sibling group is id ordered."""
    publication = publish_org_tree(RELATIONS)
    eq_(child_ids(publication.root), ("a", "b", "c"))
    eq_(child_ids(publication.nodes["a"]), ("x", "y"))


def test_subtree_detach_and_reattach():
    """A subtree moved out and its old parent re-attached stays consistent."""
    relations = dict(RELATIONS)
    relations["x"] = "b"  # x leaves subtree a and joins b
    publication = publish_org_tree(relations, [("x", "z")], old=OLD)
    eq_(child_ids(publication.nodes["a"]), ("y",))
    eq_(child_ids(publication.nodes["b"]), ("x", "z"))
    eq_(publication.verify(), True)


def test_transitive_constraints():
    """Chained constraints are honored together."""
    publication = publish_org_tree(RELATIONS, [("c", "b"), ("b", "a")], old=OLD)
    eq_(child_ids(publication.root), ("c", "b", "a"))


def test_duplicate_constraint_entries_are_allowed():
    """Repeating a constraint is harmless and reported identically."""
    publication = publish_org_tree(RELATIONS, [("b", "a"), ("b", "a")], old=OLD)
    eq_(child_ids(publication.root), ("c", "b", "a"))


def test_conflicting_constraints_reported_with_entries():
    """Mutually contradictory constraints surface every conflicting entry."""
    try:
        publish_org_tree(RELATIONS, [("a", "b"), ("b", "a")])
    except OrderConflictError as error:
        eq_(error.conflicts, (("a", "b"), ("b", "a")))
        assert any("cyclic" in detail for detail in error.details)
    else:  # pragma: no cover
        raise AssertionError("OrderConflictError not raised")


def test_constraint_between_non_siblings_is_conflict():
    """Ordering departments with different parents cannot be expressed."""
    with assert_raises(
        OrderConflictError,
        "Conflicting ordering constraints: ['x' -> 'z'] (constraints between non-siblings: 'x'->'z')",
    ):
        publish_org_tree(RELATIONS, [("x", "z")])


def test_self_constraint_is_conflict():
    """A department cannot be ordered before itself."""
    try:
        publish_org_tree(RELATIONS, [("a", "a")])
    except OrderConflictError as error:
        eq_(error.conflicts, (("a", "a"),))
    else:  # pragma: no cover
        raise AssertionError("OrderConflictError not raised")


def test_unknown_department_in_relations():
    """Unknown parent identifiers reject the batch."""
    with assert_raises(OrgStructureError, "Relations reference unknown parent departments: ghost."):
        publish_org_tree({"root": None, "a": "ghost"})


def test_unknown_department_in_constraint():
    """Constraints on unknown departments reject the batch."""
    with assert_raises(OrgStructureError, "Ordering constraints reference unknown departments: 'ghost'."):
        publish_org_tree(RELATIONS, [("a", "ghost")])


def test_multiple_roots_rejected():
    """Only one complete tree can be published."""
    with assert_raises(OrgStructureError, "An organization must have exactly one root, found 2: 'a', 'b'."):
        publish_org_tree({"a": None, "b": None})


def test_cycle_rejected():
    """Cyclic parent relations reject the batch."""
    with assert_raises(OrgStructureError, "Relations contain a cycle involving departments: 'a', 'b'."):
        publish_org_tree({"root": None, "a": "b", "b": "a"})


def test_empty_relations_rejected():
    """An empty batch is not a complete tree."""
    with assert_raises(OrgStructureError, "Cannot publish an empty organization: no departments were supplied."):
        publish_org_tree({})


def test_relations_as_pairs_iterable():
    """Relations may be supplied as (id, parent) pairs."""
    pairs = ((identifier, parent) for identifier, parent in RELATIONS.items())
    publication = publish_org_tree(pairs)
    eq_(tuple(node.id for node in PreOrderIter(publication.root)), ("root", "a", "x", "y", "b", "z", "c"))


def test_duplicate_relation_rejected():
    """Duplicate final relations for one id are ambiguous."""
    with assert_raises(OrgStructureError, "Duplicate final relation for department 'a'."):
        publish_org_tree([("root", None), ("a", "root"), ("a", "root")])


def test_two_submissions_are_identical():
    """Both internal submission orders produce the same organization."""
    publication = publish_org_tree(RELATIONS, [("c", "a"), ("b", "a")], old=OLD)

    def signature(root):
        return tuple((node.id, child_ids(node)) for node in PreOrderIter(root))

    eq_(signature(publication.root), signature(publication.secondary_root))
    eq_(publication.verify(), True)
    # the two arrangements are independent node instances
    assert publication.root is not publication.secondary_root


def test_query_and_export_see_same_structure():
    """Queryers, dict export and JSON export observe one identical ordering."""
    publication = publish_org_tree(RELATIONS, [("b", "a")], old=OLD)
    data = publication.to_dict()
    root_ids = [child["id"] for child in data["children"]]
    eq_(root_ids, ["c", "b", "a"])
    json_data = json.loads(publication.to_json())
    eq_([child["id"] for child in json_data["children"]], root_ids)
    buffer = io.StringIO()
    publication.write(buffer)
    eq_([child["id"] for child in json.loads(buffer.getvalue())["children"]], root_ids)
    eq_(child_ids(publication.root), ("c", "b", "a"))


def test_old_organization_accepts_tree_node():
    """The previous organization may be passed as an existing tree."""
    old_root = AnyNode(id="root")
    old_b = AnyNode(id="b", parent=old_root)
    old_a = AnyNode(id="a", parent=old_root)
    AnyNode(id="z", parent=old_b)
    AnyNode(id="y", parent=old_a)
    AnyNode(id="x", parent=old_a)
    relations = {
        identifier: parent
        for identifier, parent in RELATIONS.items()
        if identifier in {"root", "a", "b", "x", "y", "z"}
    }
    publication = publish_org_tree(relations, old=old_root)
    eq_(child_ids(publication.root), ("b", "a"))
    eq_(child_ids(publication.nodes["a"]), ("y", "x"))


def test_custom_node_class_and_id_attribute():
    """Stable identifiers can live in a custom attribute of a custom node class."""
    relations = {10: None, 11: 10, 12: 10}
    publication = publish_org_tree(relations, [(12, 11)], nodecls=Node, id_attr="name")
    eq_(tuple(node.name for node in publication.root.children), (12, 11))
    eq_(isinstance(publication.root, Node), True)
