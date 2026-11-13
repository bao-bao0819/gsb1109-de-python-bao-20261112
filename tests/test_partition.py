"""Tests for materialized-partition pruning."""

import random

from anytree import Node, plan_partition

from .helper import assert_raises, eq_


def _names(result):
    return tuple(node.name for node in result.roots)


def _assert_antichain(result):
    roots = result.roots
    for index, node in enumerate(roots):
        for other in roots[index + 1 :]:
            assert node not in other.path
            assert other not in node.path


def _build_greedy_counterexample():
    #            root
    #          /  |  \
    #         R   A   B        R: subtree 100 bytes, loss 1000
    #       /   \             A, B: 60 bytes each, loss 10 each
    #      R1   R2            R1, R2: 30 bytes each, loss 300 each
    root = Node("root", bytes=0, loss=0)
    r_node = Node("R", parent=root, bytes=40, loss=400)
    Node("R1", parent=r_node, bytes=30, loss=300)
    Node("R2", parent=r_node, bytes=30, loss=300)
    Node("A", parent=root, bytes=60, loss=10)
    Node("B", parent=root, bytes=60, loss=10)
    return root, r_node


def test_greedy_largest_root_is_worse():
    """Picking the largest root first meets the target but at higher loss."""
    root, r_node = _build_greedy_counterexample()
    result = plan_partition(root, 100)

    eq_(result.feasible, True)
    eq_(_names(result), ("A", "B"))
    eq_(result.released_bytes, 120)
    eq_(result.query_loss, 20)
    _assert_antichain(result)

    # Greedy "largest subtree first": R is the biggest single pruning root
    # (100 bytes). It satisfies the target immediately but costs 1000 loss,
    # 50x the optimal plan built from two smaller sibling roots.
    r_subtree_bytes = r_node.bytes + sum(child.bytes for child in r_node.children)
    r_subtree_loss = r_node.loss + sum(child.loss for child in r_node.children)
    eq_(r_subtree_bytes, 100)
    eq_(r_subtree_loss, 1000)
    assert r_subtree_loss > result.query_loss


def test_optimal_prefers_two_small_subtrees():
    """Two cheap subtrees can beat one large expensive root."""
    root = Node("root", bytes=0, loss=0)
    Node("big", parent=root, bytes=100, loss=1000)
    Node("small_a", parent=root, bytes=60, loss=10)
    Node("small_b", parent=root, bytes=60, loss=10)

    result = plan_partition(root, 100)
    eq_(result.feasible, True)
    eq_(_names(result), ("small_a", "small_b"))
    eq_(result.released_bytes, 120)
    eq_(result.query_loss, 20)
    _assert_antichain(result)


def test_whole_subtree_is_pruned_atomically():
    # A descendant alone is too small; only pruning the whole subtree works.
    root = Node("root", bytes=0, loss=0)
    r_node = Node("R", parent=root, bytes=40, loss=5)
    Node("R1", parent=r_node, bytes=60, loss=5)

    result = plan_partition(root, 70)
    eq_(result.feasible, True)
    eq_(_names(result), ("R",))
    eq_(result.released_bytes, 100)
    eq_(result.query_loss, 10)
    _assert_antichain(result)


def test_pinned_node_blocks_ancestor_subtrees():
    root = Node("root", bytes=0, loss=0)
    blocked = Node("blocked", parent=root, bytes=100, loss=5)
    Node("leaf", parent=blocked, bytes=100, loss=5, pinned=True)
    Node("free", parent=root, bytes=40, loss=50)

    result = plan_partition(root, 40)
    eq_(result.feasible, True)
    eq_(_names(result), ("free",))
    eq_(result.released_bytes, 40)
    eq_(result.query_loss, 50)

    # The pinned branch cannot be touched at all.
    infeasible = plan_partition(root, 41)
    eq_(infeasible.feasible, False)
    eq_(_names(infeasible), ("free",))
    eq_(infeasible.released_bytes, 40)
    eq_(infeasible.max_releasable_bytes, 40)
    _assert_antichain(infeasible)


def test_pinned_root_cannot_be_pruned_but_descendants_can():
    # The pinned root itself is never a pruning root; its non-pinned child
    # subtree does not contain a pinned node and stays eligible.
    root = Node("root", bytes=10, loss=1, pinned=True)
    child = Node("child", parent=root, bytes=20, loss=2)
    result = plan_partition(root, 1)
    eq_(result.feasible, True)
    eq_(_names(result), ("child",))
    eq_(result.released_bytes, 20)
    eq_(result.max_releasable_bytes, 20)
    assert root not in result.roots
    assert child in result.roots

    # A pinned leaf blocks its whole subtree chain; nothing else is available.
    leaf = Node("leaf", parent=child, bytes=5, loss=1, pinned=True)
    blocked = plan_partition(root, 1)
    eq_(blocked.feasible, False)
    eq_(blocked.roots, ())
    eq_(blocked.max_releasable_bytes, 0)
    assert leaf.pinned


def test_zero_requirement_keeps_everything():
    root, _r_node = _build_greedy_counterexample()
    result = plan_partition(root, 0)
    eq_(result.feasible, True)
    eq_(result.roots, ())
    eq_(result.released_bytes, 0)
    eq_(result.query_loss, 0)
    eq_(result.max_releasable_bytes, 220)


def test_sibling_subtrees_remain_eligible_under_pinned_uncle():
    root = Node("root", bytes=0, loss=0)
    pinned_branch = Node("pinned_branch", parent=root, bytes=100, loss=1)
    Node("pinned_leaf", parent=pinned_branch, bytes=50, loss=1, pinned=True)
    free_branch = Node("free_branch", parent=root, bytes=30, loss=7)
    Node("free_leaf", parent=free_branch, bytes=20, loss=3)

    result = plan_partition(root, 50)
    eq_(result.feasible, True)
    eq_(_names(result), ("free_branch",))
    eq_(result.released_bytes, 50)
    eq_(result.query_loss, 10)
    _assert_antichain(result)


def test_custom_attribute_names():
    # "size" is a built-in Node property, so deliberately use other names.
    root = Node("root", stored=0, penalty=0)
    Node("a", parent=root, stored=80, penalty=3)
    Node("b", parent=root, stored=80, penalty=4, resident=True)
    result = plan_partition(root, 80, size_attr="stored", loss_attr="penalty", pinned_attr="resident")
    eq_(result.feasible, True)
    eq_(_names(result), ("a",))


def test_invalid_arguments():
    root = Node("root", bytes=10, loss=1)
    with assert_raises(TypeError, "required_bytes must be an integer, got 1.5"):
        plan_partition(root, 1.5)
    with assert_raises(ValueError, "required_bytes must be non-negative, got -1"):
        plan_partition(root, -1)

    bad = Node("bad", bytes=-1, loss=1)
    with assert_raises(
        ValueError,
        "node Node('/bad', bytes=-1, loss=1) attribute 'bytes' must be non-negative, got -1",
    ):
        plan_partition(bad, 1)

    weird = Node("weird", bytes="x", loss=1)
    with assert_raises(
        TypeError,
        "node Node('/weird', bytes='x', loss=1) attribute 'bytes' must be numeric, got 'x'",
    ):
        plan_partition(weird, 1)


def _brute_force_optimal(root, required_bytes, pinned_attr):
    nodes = [*list(root.descendants), root]
    eligible = [
        node
        for node in nodes
        if not any(getattr(n, pinned_attr, False) for n in [node, *list(node.descendants)])
    ]

    def is_antichain(candidates):
        for index, node in enumerate(candidates):
            for other in candidates[index + 1 :]:
                if node in other.path or other in node.path:
                    return False
        return True

    def subtree_sum(node, attr):
        return sum(getattr(n, attr) for n in [node, *node.descendants])

    best = None
    for mask in range(1 << len(eligible)):
        candidates = tuple(eligible[index] for index in range(len(eligible)) if mask & (1 << index))
        if not is_antichain(candidates):
            continue
        released = sum(subtree_sum(n, "bytes") for n in candidates)
        if released < required_bytes:
            continue
        loss = sum(subtree_sum(n, "loss") for n in candidates)
        if best is None or loss < best[0] or (loss == best[0] and released < best[1]):
            best = (loss, released, candidates)
    return best


def test_matches_brute_force_on_random_trees():
    rng = random.Random(20261113)
    for seed in range(40):
        rng.seed(seed)
        root = Node("n0", bytes=rng.randint(0, 5), loss=rng.randint(0, 9))
        frontier = [root]
        for index in range(1, 9):
            parent = rng.choice(frontier)
            node = Node(
                f"n{index}",
                parent=parent,
                bytes=rng.randint(0, 8),
                loss=rng.randint(0, 9),
                pinned=rng.random() < 0.15,
            )
            frontier.append(node)

        total_bytes = sum(n.bytes for n in root.descendants) + root.bytes
        for required in (0, 1, 5, 10, total_bytes, total_bytes + 1):
            result = plan_partition(root, required)
            _assert_antichain(result)
            best = _brute_force_optimal(root, required, "pinned")
            if best is None:
                eq_(result.feasible, False)
                fallback = plan_partition(root, total_bytes + 1)
                eq_(result.max_releasable_bytes, fallback.max_releasable_bytes)
                eq_(result.released_bytes, fallback.max_releasable_bytes)
            else:
                eq_(result.feasible, True)
                eq_(result.query_loss, best[0])
                assert result.released_bytes >= required
