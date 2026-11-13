from anytree import AssetInventory, Node, SymlinkNode
from anytree.inventory import LoopOccurrence, SubtreeInventory

from .helper import eq_


def test_plain_tree_counts_each_node_once():
    root = Node("root")
    child = Node("child", parent=root)
    Node("grandchild", parent=child)

    report = AssetInventory().analyze(root)

    eq_(report.entry_count, 3)
    eq_(report.asset_count, 3)
    eq_(report.asset_ids, ("child", "grandchild", "root"))
    eq_(report.shared_assets, ())
    eq_(report.loops, ())
    eq_(len(report.subtrees), 1)
    eq_(report.subtrees[0].entry_count, 3)
    eq_(report.subtrees[0].asset_count, 3)


def test_single_node_and_iterable_entries_are_equivalent():
    root = Node("root")
    Node("child", parent=root)

    single = AssetInventory().analyze(root)
    iterable = AssetInventory().analyze([root])

    eq_(single.entry_count, iterable.entry_count)
    eq_(single.asset_count, iterable.asset_count)


def test_link_is_second_entry_of_same_asset():
    themes = Node("themes")
    asset_a = Node("assetA", parent=themes)
    asset_b = Node("assetB", parent=asset_a)
    other = Node("other")
    link_a = SymlinkNode(asset_a, parent=other)

    report = AssetInventory().analyze([themes, other])

    eq_(report.entry_count, 5)
    eq_(report.asset_count, 4)
    eq_(report.asset_ids, ("assetA", "assetB", "other", "themes"))
    eq_(report.shared_assets, (("assetA", 2),))
    eq_(report.entry_paths["assetA"], (("themes", "assetA"), ("other", "assetA")))
    eq_(link_a.name, "assetA")

    subtree_themes, subtree_other = report.subtrees
    eq_(isinstance(subtree_themes, SubtreeInventory), True)
    eq_((subtree_themes.entry_count, subtree_themes.asset_count), (3, 3))
    eq_((subtree_other.entry_count, subtree_other.asset_count), (2, 2))
    eq_(subtree_other.asset_ids, ("assetA", "other"))


def test_link_does_not_share_descendants_of_target():
    entry_one = Node("e1")
    shared = Node("shared", parent=entry_one)
    Node("descendant", parent=shared)
    entry_two = Node("e2")
    SymlinkNode(shared, parent=entry_two)

    report = AssetInventory().analyze([entry_one, entry_two])

    eq_(report.entry_count, 5)
    eq_(report.asset_count, 4)
    eq_(report.shared_assets, (("shared", 2),))
    eq_(report.entry_paths["shared"], (("e1", "shared"), ("e2", "shared")))


def test_subtree_discrepancies_explain_dedup_gap():
    entry_one = Node("e1")
    shared = Node("shared", parent=entry_one)
    entry_two = Node("e2")
    SymlinkNode(shared, parent=entry_two)
    entry_three = Node("e3")
    SymlinkNode(shared, parent=entry_three)

    report = AssetInventory().analyze([entry_one, entry_two, entry_three])

    subtree_sum = sum(subtree.asset_count for subtree in report.subtrees)
    eq_(subtree_sum, 6)
    eq_(report.asset_count, 4)
    eq_(subtree_sum - report.asset_count, 2)
    eq_(report.subtree_discrepancies(), (("shared", ("e1", "e2", "e3")),))

    explained = sum(len(owners) - 1 for _, owners in report.subtree_discrepancies())
    eq_(explained, subtree_sum - report.asset_count)

    entry_sum = sum(subtree.entry_count for subtree in report.subtrees)
    eq_(entry_sum, report.entry_count)
    duplicated = sum(count - 1 for _, count in report.shared_assets)
    eq_(report.entry_count - report.asset_count, duplicated)


def test_link_back_to_ancestor_records_loop_and_stops():
    root = Node("root")
    directory = Node("dir", parent=root)
    loop_link = SymlinkNode(root, parent=directory)

    report = AssetInventory().analyze(root)

    eq_(report.entry_count, 3)
    eq_(report.asset_count, 2)
    eq_(len(report.loops), 1)
    loop = report.loops[0]
    eq_(isinstance(loop, LoopOccurrence), True)
    eq_(loop.target, root)
    eq_(loop.path_names, ("root", "dir", "root"))
    eq_(loop.path[-1], loop_link)
    eq_(report.entry_paths["root"], (("root",), ("root", "dir", "root")))
    eq_(report.shared_assets, (("root", 2),))
    eq_(report.subtrees[0].loops, (loop,))


def test_link_to_grandparent_is_a_loop():
    root = Node("root")
    middle = Node("middle", parent=root)
    leaf = Node("leaf", parent=middle)
    SymlinkNode(root, parent=leaf)

    report = AssetInventory().analyze(root)

    eq_(report.entry_count, 4)
    eq_(report.asset_count, 3)
    eq_(len(report.loops), 1)
    eq_(report.loops[0].target, root)
    eq_(report.loops[0].path_names, ("root", "middle", "leaf", "root"))


def test_acyclic_path_still_counts_when_another_path_loops():
    root = Node("root")
    directory = Node("dir", parent=root)
    SymlinkNode(root, parent=directory)

    side = Node("side")
    SymlinkNode(root, parent=side)

    report = AssetInventory().analyze([root, side])

    eq_(len(report.loops), 1)
    eq_(report.loops[0].path_names, ("root", "dir", "root"))

    root_subtree, side_subtree = report.subtrees
    eq_((root_subtree.entry_count, len(root_subtree.loops)), (3, 1))
    eq_((side_subtree.entry_count, len(side_subtree.loops)), (2, 0))

    eq_(report.entry_count, 5)
    eq_(report.asset_count, 3)
    eq_(
        report.entry_paths["root"],
        (("root",), ("root", "dir", "root"), ("side", "root")),
    )
    eq_(report.subtree_discrepancies(), (("root", ("root", "side")),))


def test_repeated_acyclic_links_all_count_as_entries():
    root = Node("root")
    target = Node("target", parent=root)
    SymlinkNode(target, parent=root)
    SymlinkNode(target, parent=root)

    report = AssetInventory().analyze(root)

    eq_(report.entry_count, 4)
    eq_(report.asset_count, 2)
    eq_(report.shared_assets, (("target", 3),))
    eq_(
        report.entry_paths["target"],
        (
            ("root", "target"),
            ("root", "target"),
            ("root", "target"),
        ),
    )
    eq_(report.loops, ())


def test_chained_links_resolve_to_real_asset():
    root = Node("root")
    real = Node("real", parent=root)
    middle_link = SymlinkNode(real)
    top_link = SymlinkNode(middle_link, parent=root)

    report = AssetInventory().analyze(root)

    eq_(report.entry_count, 3)
    eq_(report.asset_count, 2)
    eq_(report.shared_assets, (("real", 2),))
    eq_(report.entry_paths["real"], (("root", "real"), ("root", "real")))
    eq_(top_link.target, middle_link)


def test_custom_identifier_attribute():
    root = Node("root", asset_id="ID-0")
    first = Node("first", parent=root, asset_id="ID-1")
    second = Node("second", parent=root, asset_id="ID-1")
    other = Node("other", asset_id="ID-9")
    SymlinkNode(first, parent=other)

    report = AssetInventory(idattr="asset_id").analyze([root, other])

    eq_(report.entry_count, 5)
    eq_(report.asset_count, 3)
    eq_(report.asset_ids, ("ID-0", "ID-1", "ID-9"))
    eq_(report.shared_assets, (("ID-1", 3),))
    eq_(
        report.entry_paths["ID-1"],
        (("root", "first"), ("root", "second"), ("other", "first")),
    )


def test_empty_entry_list():
    report = AssetInventory().analyze([])

    eq_(report.entry_count, 0)
    eq_(report.asset_count, 0)
    eq_(report.asset_ids, ())
    eq_(report.subtrees, ())
    eq_(report.loops, ())
    eq_(repr(report), "InventoryReport(entry_count=0, asset_count=0, loops=0)")
