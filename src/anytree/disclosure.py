"""Bounds for undisclosed values in a partially reported hierarchy.

External institutions often deliver only some branches of a hierarchy and,
in addition, report subtotals for whole subtrees.  This module derives the
tightest reachable lower and upper bounds for the values of undisclosed
leaves, and of requested groups of such leaves, under these assumptions:

* every leaf value is a non-negative integer;
* a reported subtotal equals the sum of *all* leaf values in its subtree,
  including leaves and nested subtotals that were not delivered;
* a missing entry is unknown (it is **not** zero).

The constraints decompose into independent blocks: each reported node owns
the leaves whose nearest reported ancestor it is.  Within a block the nested
reported subtotals are fixed, leaving one non-negative integer "slack"
shared by the block's undisclosed leaves.  Different blocks do not exchange
values, so all bounds follow from these per-block slacks alone.

Example:
    >>> from anytree import Node
    >>> root = Node("root")
    >>> x = Node("x", parent=root)
    >>> a = Node("a", parent=x)
    >>> b = Node("b", parent=x)
    >>> c = Node("c", parent=root)
    >>> bounds = disclosure_bounds(root, {root: 10, x: 4})
    >>> bounds[a]
    DisclosureBounds(lower=0, upper=4)
    >>> bounds[c]
    DisclosureBounds(lower=6, upper=6)

A leaf without any reported ancestor has no finite upper bound:

    >>> disclosure_bounds(root, {x: 4})[c]
    DisclosureBounds(lower=0, upper=None)

Subtotals that cannot hold simultaneously raise a
:any:`DisclosureConflictError`:

    >>> disclosure_bounds(root, {root: 3, x: 4})  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    anytree.disclosure.DisclosureConflictError: Inconsistent subtotals reported on: ...
"""

from collections import defaultdict, namedtuple
from collections.abc import Mapping

from .iterators import PreOrderIter
from .node.exceptions import TreeError

__all__ = [
    "DisclosureBounds",
    "DisclosureConflictError",
    "disclosure_bounds",
    "disclosure_group_bounds",
]


class DisclosureConflictError(TreeError):
    """Reported subtotals cannot hold at the same time.

    The :attr:`nodes` attribute holds a set of reported nodes whose subtotals
    contradict each other: the subtotals of the nested reported nodes sum to
    more than the subtotal of their nearest reported ancestor.
    """

    def __init__(self, nodes, message=None):
        self.nodes = tuple(nodes)
        if message is None:
            labels = ", ".join(_node_label(node) for node in self.nodes)
            message = f"Inconsistent subtotals reported on: {labels}"
        super().__init__(message)


class DisclosureBounds(namedtuple("DisclosureBounds", ("lower", "upper"))):
    """Reachable interval ``[lower, upper]`` of an undisclosed value.

    ``upper`` is ``None`` when no finite upper bound can be derived.
    """

    __slots__ = ()

    @property
    def bounded(self):
        """Return ``True`` when a finite upper bound is known."""
        return self.upper is not None


def disclosure_bounds(tree, subtotals, groups=None, *, leaf_attr=None):
    """Compute reachable bounds of individual undisclosed leaf values.

    Args:
        tree: root node of the hierarchy.
        subtotals: mapping of reported node -> non-negative integer subtotal.
            When `leaf_attr` is set, nodes may instead be keyed by the value
            of that attribute.
        groups: requested grouping of leaves. Either a mapping of group name
            to iterable of leaves, or a single iterable of leaves. When
            ``None``, bounds are returned for every leaf of `tree`.
        leaf_attr: when set, `subtotals` and `groups` are keyed by the value
            of this node attribute instead of by the node objects themselves.

    Returns:
        A mapping of leaf to :any:`DisclosureBounds` when `groups` is ``None``
        or a flat iterable, or a mapping of group name to such a mapping when
        `groups` is a mapping. Keys are node objects, or attribute values when
        `leaf_attr` is set. A finite upper bound is ``None`` when no reported
        ancestor constrains the leaf.

    Raises:
        DisclosureConflictError: when the reported subtotals are mutually
            inconsistent; ``error.nodes`` names a conflicting subtotal set.
        TypeError: when an argument has an unsupported type.
        ValueError: when a subtotal is negative, a key does not belong to
            `tree`, or a group member is not a leaf.
    """
    model = _build_model(tree, subtotals, leaf_attr)
    group_map, flat_output = _resolve_groups(groups, model.leaves, model.node_set, model.attr_index, leaf_attr)
    results = {name: _leaf_bounds(members, model, leaf_attr) for name, members in group_map.items()}
    return results[None] if flat_output else results


def disclosure_group_bounds(tree, subtotals, groups, *, leaf_attr=None):
    """Compute reachable bounds of the total undisclosed share per group.

    For each group, all leaf members that do not carry their own reported
    subtotal are summed; the function returns the tightest interval of that
    sum.  A group whose members are all disclosed individually has share
    ``(0, 0)``; a group containing a leaf without any reported ancestor has no
    finite upper bound and yields ``upper=None``.

    Args:
        tree: root node of the hierarchy.
        subtotals: mapping of reported node -> non-negative integer subtotal.
        groups: mapping of group name to iterable of leaves, or a single
            iterable of leaves.
        leaf_attr: when set, `subtotals` and `groups` are keyed by the value
            of this node attribute instead of by the node objects themselves.

    Returns:
        A mapping of group name to :any:`DisclosureBounds` when `groups` is a
        mapping, or a single :any:`DisclosureBounds` when `groups` is a flat
        iterable.

    Raises:
        DisclosureConflictError: when the reported subtotals are mutually
            inconsistent; ``error.nodes`` names a conflicting subtotal set.
        TypeError: when an argument has an unsupported type.
        ValueError: when a subtotal is negative, a key does not belong to
            `tree`, or a group member is not a leaf.
    """
    model = _build_model(tree, subtotals, leaf_attr)
    group_map, flat_output = _resolve_groups(groups, model.leaves, model.node_set, model.attr_index, leaf_attr)
    results = {name: _group_bounds(members, model) for name, members in group_map.items()}
    return results[None] if flat_output else results


class _DisclosureModel:
    """Preprocessed subtotal constraints of one tree."""

    def __init__(self, leaves, node_set, attr_index, reported, owner_of, slack, block_sizes):
        self.leaves = leaves
        self.node_set = node_set
        self.attr_index = attr_index
        self.reported = reported
        self.owner_of = owner_of
        self.slack = slack
        self.block_sizes = block_sizes


def _build_model(tree, subtotals, leaf_attr):
    all_nodes = list(PreOrderIter(tree))
    leaves = tuple(node for node in all_nodes if node.is_leaf)
    node_set = set(all_nodes)
    attr_index = _build_attr_index(all_nodes, leaf_attr)
    reported = _resolve_subtotals(subtotals, node_set, attr_index, leaf_attr)

    reported_set = set(reported)
    nested_of = defaultdict(list)
    owner_of = {}
    for node in all_nodes:
        owner = _nearest_reported_ancestor(node, reported_set)
        if node in reported_set:
            if owner is not None:
                nested_of[owner].append(node)
        elif node.is_leaf:
            owner_of[node] = owner

    slack = {}
    for node in all_nodes:
        if node not in reported:
            continue
        nested = nested_of.get(node, ())
        remainder = reported[node] - sum(reported[descendant] for descendant in nested)
        if remainder < 0:
            raise DisclosureConflictError([node, *nested])
        slack[node] = remainder

    block_sizes = defaultdict(int)
    for leaf, owner in owner_of.items():
        if owner is not None:
            block_sizes[owner] += 1
    for node in all_nodes:
        if node in reported and not node.is_leaf and block_sizes[node] == 0 and slack[node] > 0:
            raise DisclosureConflictError([node, *nested_of.get(node, ())])
    return _DisclosureModel(leaves, node_set, attr_index, reported, owner_of, slack, block_sizes)


def _build_attr_index(all_nodes, leaf_attr):
    if leaf_attr is None:
        return None
    index = {}
    for node in all_nodes:
        key = getattr(node, leaf_attr)
        if key in index:
            raise ValueError(f"leaf_attr {leaf_attr!r} is not unique: {key!r} identifies more than one node")
        index[key] = node
    return index


def _resolve_subtotals(subtotals, node_set, attr_index, leaf_attr):
    if not isinstance(subtotals, Mapping):
        raise TypeError("subtotals must be a mapping of node to non-negative integer")
    reported = {}
    for key, value in subtotals.items():
        node = _resolve_node(key, node_set, attr_index, leaf_attr)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Subtotal of {key!r} must be an integer, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"Subtotal of {key!r} must be non-negative, got {value}")
        reported[node] = value
    return reported


def _resolve_node(key, node_set, attr_index, leaf_attr):
    node = key if leaf_attr is None else attr_index.get(key)
    if node is None or node not in node_set:
        raise ValueError(f"Node {key!r} is not part of the tree")
    return node


def _resolve_groups(groups, leaves, node_set, attr_index, leaf_attr):
    def resolve_leaf(key):
        node = _resolve_node(key, node_set, attr_index, leaf_attr)
        if not node.is_leaf:
            raise ValueError(f"Group member {key!r} is not a leaf of the tree")
        return node

    if groups is None:
        return {None: list(leaves)}, True
    if isinstance(groups, Mapping):
        normalized = {
            name: list(dict.fromkeys(resolve_leaf(key) for key in members)) for name, members in groups.items()
        }
        return normalized, False
    if isinstance(groups, str):
        raise TypeError("groups must be a mapping or an iterable of leaf nodes")
    try:
        members = list(dict.fromkeys(resolve_leaf(key) for key in groups))
    except TypeError as exc:
        raise TypeError("groups must be a mapping or an iterable of leaf nodes") from exc
    return {None: members}, True


def _nearest_reported_ancestor(node, reported_set):
    owner = None
    for ancestor in node.ancestors:
        if ancestor in reported_set:
            owner = ancestor
    return owner


def _output_key(leaf, leaf_attr):
    return leaf if leaf_attr is None else getattr(leaf, leaf_attr)


def _leaf_bounds(members, model, leaf_attr):
    per_leaf = {}
    for leaf in members:
        key = _output_key(leaf, leaf_attr)
        if leaf in model.reported:
            value = model.reported[leaf]
            per_leaf[key] = DisclosureBounds(value, value)
            continue
        owner = model.owner_of[leaf]
        if owner is None:
            per_leaf[key] = DisclosureBounds(0, None)
        elif model.block_sizes[owner] == 1:
            per_leaf[key] = DisclosureBounds(model.slack[owner], model.slack[owner])
        else:
            per_leaf[key] = DisclosureBounds(0, model.slack[owner])
    return per_leaf


def _group_bounds(members, model):
    covered = defaultdict(int)
    unbounded = 0
    for leaf in members:
        if leaf in model.reported:
            continue
        owner = model.owner_of[leaf]
        if owner is None:
            unbounded += 1
        else:
            covered[owner] += 1

    lower = 0
    upper = 0
    for owner, count in covered.items():
        slack = model.slack[owner]
        upper += slack
        if count == model.block_sizes[owner]:
            lower += slack
    if unbounded:
        return DisclosureBounds(lower, None)
    return DisclosureBounds(lower, upper)


def _node_label(node):
    name = getattr(node, "name", None)
    return repr(name) if name is not None else repr(node)
