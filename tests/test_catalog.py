"""Tests for the asset catalog subscription diff."""

from anytree import AssetEntry, CatalogDiff, Node, PathChange, Subscription, SymlinkNode, catalogdiff, find_by_attr

from .helper import eq_


def _old_catalog():
    root = Node("root", id="root")
    assets = Node("assets", parent=root, id="assets")
    photos = Node("photos", parent=assets, id="photos")
    Node("cat", parent=photos, id="cat")
    Node("dog", parent=photos, id="dog")
    music = Node("music", parent=assets, id="music")
    Node("rock", parent=music, id="rock")
    Node("inbox", parent=root, id="inbox")
    return root


def _final_catalog():
    """Final tree: `photos` moved to `inbox`, `cat` renamed to `kitty`."""
    root = Node("root", id="root")
    assets = Node("assets", parent=root, id="assets")
    music = Node("music", parent=assets, id="music")
    Node("rock", parent=music, id="rock")
    inbox = Node("inbox", parent=root, id="inbox")
    photos = Node("photos", parent=inbox, id="photos")
    Node("kitty", parent=photos, id="cat")
    Node("dog", parent=photos, id="dog")
    return root


def test_subscription():
    """Subscription matches by hierarchy and level."""
    sub = Subscription("/root/assets", maxlevel=2)
    eq_(sub.path, ("root", "assets"))
    eq_(sub, Subscription(("root", "assets"), maxlevel=2))
    eq_(repr(sub), "Subscription('/root/assets', maxlevel=2)")
    assert sub.matches(("root", "assets"))
    assert sub.matches(("root", "assets", "photos"))
    assert not sub.matches(("root", "assets", "photos", "cat"))
    assert not sub.matches(("root", "inbox"))
    assert not sub.matches(("root",))
    assert Subscription().matches(("anything", "at", "all"))
    eq_(repr(Subscription()), "Subscription('/')")


def test_unchanged():
    """Identical catalogs produce an empty, minimal change list."""
    diff = catalogdiff(_old_catalog(), _old_catalog())
    eq_(diff, CatalogDiff((), (), ()))
    assert diff.empty


def test_added_removed():
    """Assets without stable id counterpart are added and removed, not renamed."""
    old = Node("root", id="root")
    Node("gone", parent=old, id="gone")
    new = Node("root", id="root")
    Node("fresh", parent=new, id="fresh")
    diff = catalogdiff(old, new)
    eq_(diff.added, (AssetEntry("fresh", "/root/fresh"),))
    eq_(diff.removed, (AssetEntry("gone", "/root/gone"),))
    eq_(diff.moved, ())
    assert not diff.empty


def test_subtree_move_within_subscription():
    """Children of a moved subtree report display path changes, even if unchanged themselves."""
    diff = catalogdiff(_old_catalog(), _final_catalog())
    eq_(diff.added, ())
    eq_(diff.removed, ())
    eq_(
        diff.moved,
        (
            PathChange("photos", "/root/assets/photos", "/root/inbox/photos"),
            PathChange("cat", "/root/assets/photos/cat", "/root/inbox/photos/kitty"),
            PathChange("dog", "/root/assets/photos/dog", "/root/inbox/photos/dog"),
        ),
    )


def test_subtree_move_exits_subscription():
    """Children of a subtree moved out of the subscribed area leave the subscription."""
    diff = catalogdiff(_old_catalog(), _final_catalog(), Subscription("/root/assets"))
    eq_(diff.added, ())
    eq_(diff.moved, ())
    eq_(
        diff.removed,
        (
            AssetEntry("photos", "/root/assets/photos"),
            AssetEntry("cat", "/root/assets/photos/cat"),
            AssetEntry("dog", "/root/assets/photos/dog"),
        ),
    )


def test_subtree_move_enters_subscription():
    """Children of a subtree moved into the subscribed area are added."""
    diff = catalogdiff(_old_catalog(), _final_catalog(), Subscription("/root/inbox"))
    eq_(diff.removed, ())
    eq_(diff.moved, ())
    eq_(
        diff.added,
        (
            AssetEntry("photos", "/root/inbox/photos"),
            AssetEntry("cat", "/root/inbox/photos/kitty"),
            AssetEntry("dog", "/root/inbox/photos/dog"),
        ),
    )


def test_move_then_rename_same_as_final_tree():
    """Moving first and renaming afterwards notifies like publishing the final tree directly."""
    old = _old_catalog()
    stepwise = _old_catalog()
    photos = find_by_attr(stepwise, "photos")
    photos.parent = find_by_attr(stepwise, "inbox")
    cat = find_by_attr(stepwise, "cat")
    cat.name = "kitty"
    eq_(catalogdiff(old, stepwise), catalogdiff(old, _final_catalog()))


def test_maxlevel():
    """Subscription levels limit the matched hierarchy."""
    sub = Subscription("/root/assets", maxlevel=1)
    eq_(catalogdiff(_old_catalog(), _final_catalog(), sub), CatalogDiff((), (), ()))
    sub = Subscription("/root/assets", maxlevel=2)
    diff = catalogdiff(_old_catalog(), _final_catalog(), sub)
    eq_(diff.added, ())
    eq_(diff.moved, ())
    eq_(diff.removed, (AssetEntry("photos", "/root/assets/photos"),))


def test_nodes_without_id_are_skipped():
    """Nodes without a stable id are no assets, but contribute to display paths."""
    old = Node("root", id="root")
    folder = Node("folder", parent=old)
    Node("leaf", parent=folder, id="leaf")
    new = Node("root", id="root")
    folder2 = Node("folder2", parent=new)
    Node("leaf", parent=folder2, id="leaf")
    diff = catalogdiff(old, new)
    eq_(diff.added, ())
    eq_(diff.removed, ())
    eq_(diff.moved, (PathChange("leaf", "/root/folder/leaf", "/root/folder2/leaf"),))


def test_idattr():
    """The stable id attribute is configurable."""
    old = Node("root", aid="root")
    Node("gone", parent=old, aid="gone")
    new = Node("root", aid="root")
    Node("fresh", parent=new, aid="fresh")
    diff = catalogdiff(old, new, idattr="aid")
    eq_(diff.added, (AssetEntry("fresh", "/root/fresh"),))
    eq_(diff.removed, (AssetEntry("gone", "/root/gone"),))


def _linked_catalog():
    root = Node("root", id="root")
    a = Node("a", parent=root, id="a")
    x = Node("x", parent=a, id="x")
    b = Node("b", parent=root, id="b")
    SymlinkNode(x, parent=b)
    return root, x


def test_second_entry_dropped():
    """An asset visible through two entries remains visible when one entry is dropped."""
    old, _x = _linked_catalog()
    new = Node("root", id="root")
    a = Node("a", parent=new, id="a")
    Node("x", parent=a, id="x")
    Node("b", parent=new, id="b")
    eq_(catalogdiff(old, new), CatalogDiff((), (), ()))


def test_main_entry_dropped():
    """An asset losing its main entry stays subscribed via its second entry."""
    old, _x = _linked_catalog()
    new = Node("root", id="root")
    Node("a", parent=new, id="a")
    b = Node("b", parent=new, id="b")
    x2 = Node("x", id="x")
    SymlinkNode(x2, parent=b)
    diff = catalogdiff(old, new)
    eq_(diff.added, ())
    eq_(diff.removed, ())
    eq_(diff.moved, (PathChange("x", "/root/a/x", "/root/b/x"),))


def test_all_entries_dropped():
    """An asset is removed only after its last visible entry is dropped."""
    old, _x = _linked_catalog()
    new = Node("root", id="root")
    Node("a", parent=new, id="a")
    Node("b", parent=new, id="b")
    diff = catalogdiff(old, new)
    eq_(diff.added, ())
    eq_(diff.moved, ())
    eq_(diff.removed, (AssetEntry("x", "/root/a/x"),))
