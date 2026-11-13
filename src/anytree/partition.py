"""
Materialized-partition pruning for hierarchical aggregation tables.

The tree models a hierarchy of materialized aggregation partitions. Every node
stores its own on-disk size and the query loss incurred when its whole subtree is
reverted to on-the-fly computation. A subtree can only be reverted as a whole and
a subtree containing a pinned (always-resident) descendant cannot be pruned at
all.

Given a mandatory amount of bytes to release, :func:`plan_partition` selects a set
of pairwise non-containing pruning roots with minimum total query loss. The
optimization is a tree knapsack solved by post-order dynamic programming.
"""

from __future__ import annotations

from dataclasses import dataclass

from anytree.iterators import PostOrderIter


@dataclass(frozen=True)
class PartitionResult:
    """
    Result of a partition pruning plan.

    Attributes:
        feasible: True if a plan releasing at least the requested bytes exists.
        roots: Pruning roots, i.e. nodes whose whole subtrees are reverted.
            The nodes are pairwise non-containing.
        released_bytes: Total bytes released by `roots`.
        query_loss: Total query loss incurred by `roots`.
        max_releasable_bytes: Maximum bytes releasable without violating the
            pinning constraint, independent of the requested amount.
    """

    feasible: bool
    roots: tuple
    released_bytes: int
    query_loss: float
    max_releasable_bytes: int


def _number(node, attr_name):
    value = getattr(node, attr_name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"node {node!r} attribute {attr_name!r} must be numeric, got {value!r}")
    if value < 0:
        raise ValueError(f"node {node!r} attribute {attr_name!r} must be non-negative, got {value!r}")
    return value


def plan_partition(root, required_bytes, size_attr="bytes", loss_attr="loss", pinned_attr="pinned") -> PartitionResult:
    """
    Select pairwise non-containing pruning roots releasing at least `required_bytes`.

    Each node carries a storage size (`size_attr`), a query loss for reverting its
    whole subtree (`loss_attr`) and an optional pinned flag (`pinned_attr`). A node
    whose subtree contains a pinned node is never eligible as a pruning root, so
    pinning effectively blocks every ancestor subtree while leaving the subtrees
    of non-pinned siblings available.

    The dynamic programming table of each node maps a released-byte capacity to
    the minimum query loss. Capacities are clamped to `required_bytes`, which keeps
    the tables pseudo-polynomial in the requested amount while still handling
    plans that overshoot the target.

    Args:
        root: Root node of the partition hierarchy.
        required_bytes: Minimum number of bytes that must be released.
        size_attr: Name of the node attribute holding the stored byte count.
            The name "size" is reserved by NodeMixin (subtree node count) and
            must not be used as a data attribute.
        loss_attr: Name of the node attribute holding the query loss.
        pinned_attr: Name of the node attribute holding the pinned flag.

    Returns:
        PartitionResult: The optimal feasible plan, or the maximum releasable
            plan when the request cannot be satisfied.

    Raises:
        ValueError: If `required_bytes` is negative or a node holds negative data.
        TypeError: If `required_bytes` or node data is not numeric.

    Example:
        >>> from anytree import Node
        >>> root = Node("root", bytes=0, loss=0)
        >>> big = Node("big", parent=root, bytes=100, loss=1000)
        >>> small_a = Node("small_a", parent=root, bytes=60, loss=10)
        >>> small_b = Node("small_b", parent=root, bytes=60, loss=10)
        >>> result = plan_partition(root, 100)
        >>> result.feasible
        True
        >>> tuple(node.name for node in result.roots)
        ('small_a', 'small_b')
        >>> result.released_bytes, result.query_loss
        (120, 20)
    """
    if not isinstance(required_bytes, int) or isinstance(required_bytes, bool):
        raise TypeError(f"required_bytes must be an integer, got {required_bytes!r}")
    if required_bytes < 0:
        raise ValueError(f"required_bytes must be non-negative, got {required_bytes!r}")

    sub_bytes, sub_loss, pinned_below = _aggregate(root, size_attr, loss_attr, pinned_attr)

    # dp[node] maps clamped capacity -> (minimum loss, choice, split)
    # choice 0: keep the node; split holds the capacity handed to each child.
    # choice 1: prune the whole subtree; split is None.
    dp = {}
    for node in PostOrderIter(root):
        current = {0: (0, 0, ())}
        for child in node.children:
            child_table = dp[child]
            merged = {}
            for cap_one, (loss_one, _choice_one, split_one) in current.items():
                for cap_two, (loss_two, _choice_two, _split_two) in child_table.items():
                    cap = min(cap_one + cap_two, required_bytes)
                    loss = loss_one + loss_two
                    old = merged.get(cap)
                    if old is None or loss < old[0]:
                        merged[cap] = (loss, 0, (*split_one, cap_two))
            current = merged
        if not pinned_below[node]:
            cap = min(sub_bytes[node], required_bytes)
            loss = sub_loss[node]
            old = current.get(cap)
            if old is None or loss < old[0]:
                current[cap] = (loss, 1, None)
        dp[node] = current

    max_releasable, max_choice = _max_releasable(root, sub_bytes, pinned_below)

    if required_bytes not in dp[root]:
        roots = tuple(_reconstruct_max(root, max_choice))
        released = sum(sub_bytes[node] for node in roots)
        loss = sum(sub_loss[node] for node in roots)
        return PartitionResult(False, roots, released, loss, max_releasable)

    optimal_loss = dp[root][required_bytes][0]
    roots = tuple(_reconstruct(root, required_bytes, dp))
    released = sum(sub_bytes[node] for node in roots)
    loss = sum(sub_loss[node] for node in roots)
    # Re-summing the selected subtree losses yields the DP optimum.
    assert abs(loss - optimal_loss) <= 1e-9 * max(1.0, abs(optimal_loss))
    return PartitionResult(True, roots, released, loss, max_releasable)


def _aggregate(root, size_attr, loss_attr, pinned_attr):
    sub_bytes = {}
    sub_loss = {}
    pinned_below = {}
    for node in PostOrderIter(root):
        own_bytes = _number(node, size_attr)
        own_loss = _number(node, loss_attr)
        pinned = bool(getattr(node, pinned_attr, False))
        pinned_below[node] = pinned or any(pinned_below[child] for child in node.children)
        sub_bytes[node] = own_bytes + sum(sub_bytes[child] for child in node.children)
        sub_loss[node] = own_loss + sum(sub_loss[child] for child in node.children)
    return sub_bytes, sub_loss, pinned_below


def _max_releasable(root, sub_bytes, pinned_below):
    """Scalar DP: maximum bytes each subtree can release, with prune choices."""
    max_bytes = {}
    choice = {}
    for node in PostOrderIter(root):
        keep_bytes = sum(max_bytes[child] for child in node.children)
        if not pinned_below[node] and sub_bytes[node] > keep_bytes:
            max_bytes[node] = sub_bytes[node]
            choice[node] = 1
        else:
            max_bytes[node] = keep_bytes
            choice[node] = 0
    return max_bytes[root], choice


def _reconstruct_max(root, choice):
    stack = [root]
    while stack:
        node = stack.pop()
        if choice[node] == 1:
            yield node
        else:
            stack.extend(node.children)


def _reconstruct(node, cap, dp):
    _loss, pruned, split = dp[node][cap]
    if pruned:
        yield node
        return
    for child, child_cap in zip(node.children, split):
        yield from _reconstruct(child, child_cap, dp)
