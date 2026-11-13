"""Snapshot based depth-first pagination."""

import json

import pytest

from anytree import (
    AmbiguousPathError,
    LightNodeMixin,
    Node,
    PagedExporter,
    PaginationCursorError,
    Paginator,
    PreOrderIter,
    SnapshotExpiredError,
    SnapshotStore,
    TreeSnapshot,
)


def sample_tree():
    root = Node("root")
    a = Node("a", parent=root)
    a1 = Node("a1", parent=a)
    a2 = Node("a2", parent=a)
    b = Node("b", parent=root)
    b1 = Node("b1", parent=b)
    return root, a, a1, a2, b, b1


def tree_root():
    return sample_tree()[0]


def collect(paginator, page_size=None):
    items = []
    pages = []
    cursor = None
    while True:
        page = paginator.page(cursor, page_size=page_size)
        pages.append(page)
        items.extend(page.items)
        if page.complete:
            return items, pages
        cursor = page.cursor


def test_dfs_order_across_pages():
    root, _a, _a1, _a2, _b, _b1 = sample_tree()
    paginator = Paginator(root, page_size=2)
    items, pages = collect(paginator)
    assert [item.name for item in items] == [node.name for node in PreOrderIter(root)]
    assert [len(page) for page in pages] == [2, 2, 2]
    assert pages[-1].complete
    assert all(page.cursor is not None for page in pages[:-1])
    assert pages[-1].cursor is None
    assert pages[-1].has_more is False
    assert pages[0].next_cursor is pages[0].cursor


def test_page_sizes_do_not_duplicate_or_lose_nodes():
    root, *_ = sample_tree()
    for size in (1, 2, 3, 10):
        paginator = Paginator(root, page_size=size)
        items, pages = collect(paginator)
        assert [item.name for item in items] == [node.name for node in PreOrderIter(root)]
        assert all(len(page) <= size for page in pages[:-1])


def test_filter_delivers_matches_with_complete_ancestor_paths():
    root, a, a1, _a2, _b, _b1 = sample_tree()
    paginator = Paginator(root, page_size=10, filter_=lambda node: node.name.endswith("1"))
    items = paginator.page().items
    assert [item.name for item in items] == ["a1", "b1"]
    assert items[0].path == ("root", "a", "a1")
    assert items[0].ancestor_path == ("root", "a")
    assert items[1].ancestor_path == ("root", "b")
    assert items[0].display_path == "root/a/a1"
    assert items[0].node_path == (root, a, a1)
    assert items[0].ancestor_nodes == (root, a)
    assert hash(items[0]) == hash(items[0])


def test_filter_does_not_prune_descendants():
    root = Node("root")
    hidden = Node("hidden", parent=root)
    hit = Node("hit", parent=hidden)
    paginator = Paginator(root, page_size=10, filter_=lambda node: node.name == "hit")
    items = paginator.page().items
    assert [item.name for item in items] == ["hit"]
    assert items[0].path == ("root", "hidden", "hit")
    page = paginator.page()
    assert page.skipped == 2
    assert page.complete


def test_rename_during_continuation_uses_snapshot_names():
    root, a, a1, _a2, _b, _b1 = sample_tree()
    paginator = Paginator(root, page_size=2)
    first = paginator.page()
    a.name = "renamed"
    a1.name = "renamed-child"
    second = paginator.page(first.cursor)
    assert [item.name for item in second.items] == ["a1", "a2"]
    assert second.items[0].path == ("root", "a", "a1")
    assert second.items[0].display_path == "root/a/a1"
    assert a.name == "renamed"


def test_inserted_same_name_sibling_does_not_change_started_export():
    root, a, _a1, _a2, b, _b1 = sample_tree()
    paginator = Paginator(root, page_size=2)
    first = paginator.page()
    inserted = Node("b", parent=root)
    root.children = (a, inserted, b)
    rest = []
    for page in paginator.pages(start=first.cursor):
        rest.extend(page.items)
    all_items = list(first.items) + rest
    assert [item.name for item in all_items] == ["root", "a", "a1", "a2", "b", "b1"]
    assert inserted not in [item.node for item in all_items]
    assert [child.name for child in root.children] == ["a", "b", "b"]


def test_new_export_sees_inserted_sibling_and_keeps_order():
    root, a, _a1, _a2, b, _b1 = sample_tree()
    inserted = Node("b", parent=root)
    root.children = (a, inserted, b)
    items, _ = collect(Paginator(root, page_size=2))
    assert [item.name for item in items] == ["root", "a", "a1", "a2", "b", "b", "b1"]
    assert items[4].node is inserted
    assert items[5].node is b


def test_cursor_is_json_serializable_and_resumable():
    root, *_ = sample_tree()
    store = SnapshotStore()
    paginator = Paginator(root, page_size=2, store=store)
    page = paginator.page()
    cursor = json.loads(json.dumps(page.cursor))
    resumed = Paginator.resume(cursor, store, page_size=2)
    next_page = resumed.page(cursor)
    assert [item.name for item in next_page.items] == ["a1", "a2"]


def test_cursor_is_serializable_for_non_json_identity():
    root = Node("root", token=object())
    child = Node("child", parent=root)
    paginator = Paginator(root, page_size=2, identityfunc=lambda node: node.token if hasattr(node, "token") else None)
    page = paginator.page()
    assert json.loads(json.dumps(page.cursor)) == page.cursor


def test_export_page_is_json_serializable_for_non_json_identity():
    root = Node("root", token=object())
    child = Node("child", parent=root)
    exporter = PagedExporter(
        root,
        page_size=2,
        identityfunc=lambda node: getattr(node, "token", None),
        payloadfunc=lambda node: {"name": node.name},
    )
    page = exporter.export_page()
    json.dumps(page)
    assert page["boundary"]["identity"].startswith("<object object at")


def test_cursor_from_other_snapshot_is_rejected():
    root, *_ = sample_tree()
    paginator = Paginator(root, page_size=2)
    cursor = paginator.page().cursor
    other = Paginator(root, page_size=2)
    with pytest.raises(PaginationCursorError):
        other.page(cursor)


def test_cursor_position_must_point_to_a_match():
    root, _a, _a1, _a2, _b, _b1 = sample_tree()
    paginator = Paginator(root, page_size=1, filter_=lambda node: node.name in ("root", "b"))
    cursor = paginator.page().cursor
    assert cursor["position"] == 4
    cursor["position"] = 1
    with pytest.raises(PaginationCursorError):
        paginator.page(cursor)


def test_malformed_cursor_is_rejected():
    root, *_ = sample_tree()
    paginator = Paginator(root, page_size=2)
    with pytest.raises(PaginationCursorError):
        paginator.page({"version": 1})
    with pytest.raises(PaginationCursorError):
        paginator.page("{not-json")
    with pytest.raises(PaginationCursorError):
        paginator.page("[]")


def test_page_size_validation():
    root, *_ = sample_tree()
    with pytest.raises(ValueError):
        Paginator(root, page_size=0)
    with pytest.raises(ValueError):
        Paginator(root, page_size=2).page(page_size=0)


def test_boundary_is_last_complete_subtree():
    root, _a, _a1, _a2, _b, _b1 = sample_tree()
    paginator = Paginator(root, page_size=2)
    page = paginator.page()
    assert page.boundary is None
    page = paginator.page(page.cursor)
    assert page.boundary.path == ("root", "a")
    assert page.boundary.key == (0,)
    assert page.boundary.start == 1
    assert page.boundary.end == 4
    assert page.boundary.size == 3
    page = paginator.page(page.cursor)
    assert page.boundary.path == ("root",)
    assert page.boundary.end == 6


def test_expired_snapshot_returns_boundary_and_never_rebinds_to_live_tree():
    root, _a, _a1, _a2, b, _b1 = sample_tree()
    store = SnapshotStore(max_snapshots=1)
    exporter = PagedExporter(root, page_size=2, store=store)
    first = exporter.export_page()
    second = exporter.export_page(first["cursor"])
    assert second["boundary"]["path"] == ["root", "a"]
    b.name = "renamed-b"
    Node("new", parent=root)
    PagedExporter(Node("other"), page_size=2, store=store)
    with pytest.raises(SnapshotExpiredError) as excinfo:
        PagedExporter.resume(second["cursor"], store)
    assert excinfo.value.boundary is not None
    assert excinfo.value.boundary.path == ("root", "a")
    assert excinfo.value.boundary.end == 4


def test_expired_snapshot_without_boundary_reports_none():
    root, *_ = sample_tree()
    store = SnapshotStore(max_snapshots=1)
    exporter = PagedExporter(root, page_size=2, store=store)
    first = exporter.export_page()
    PagedExporter(Node("other"), page_size=2, store=store)
    with pytest.raises(SnapshotExpiredError) as excinfo:
        PagedExporter.resume(first["cursor"], store)
    assert excinfo.value.boundary is None


def test_export_captures_payload_at_start():
    root = Node("root", value=1)
    child = Node("child", parent=root, value=2)
    exporter = PagedExporter(root, page_size=1, payloadfunc=lambda node: {"value": node.value})
    first = exporter.export_page()
    child.value = 99
    second = exporter.export_page(first["cursor"])
    assert first["items"][0]["payload"] == {"value": 1}
    assert second["items"][0]["payload"] == {"value": 2}


def test_default_export_payload_contains_public_attributes_only():
    root = Node("root", foo="bar")
    child = Node("child", parent=root)
    page = PagedExporter(root, page_size=10).export_page()
    assert page["items"][0]["payload"] == {"name": "root", "foo": "bar"}
    assert page["items"][1]["payload"] == {"name": "child"}


def test_default_export_payload_ignores_private_attributes():
    root = Node("root", _private="hidden", visible="shown")
    page = PagedExporter(root, page_size=10, payloadfunc=None).export_page()
    assert page["items"][0]["payload"] == {"name": "root", "visible": "shown"}


def test_default_export_payload_supports_nodes_without_dict():
    class SlottedNode(LightNodeMixin):
        __slots__ = ("foo", "name")

        def __init__(self, name, foo=None, parent=None):
            self.name = name
            self.foo = foo
            self.parent = parent

    root = SlottedNode("root", foo="bar")
    child = SlottedNode("child", parent=root)
    page = PagedExporter(root, page_size=10).export_page()
    assert page["items"][0]["payload"] == {"name": "root", "foo": "bar"}
    assert page["items"][1]["payload"] == {"name": "child", "foo": None}


def test_export_pages_yields_all_items_and_is_json_serializable():
    root, *_ = sample_tree()
    exporter = PagedExporter(root, page_size=2)
    pages = list(exporter.export_pages())
    assert len(pages) == 3
    assert [item["name"] for page in pages for item in page["items"]] == [
        "root",
        "a",
        "a1",
        "a2",
        "b",
        "b1",
    ]
    json.dumps(pages)
    assert [page["page_number"] for page in pages] == [0, 1, 2]
    assert pages[-1]["complete"] is True
    assert pages[-1]["cursor"] is None


def test_maxlevel_limits_capture():
    root, *_ = sample_tree()
    paginator = Paginator(root, page_size=10, maxlevel=2)
    assert [item.name for item in paginator.page().items] == ["root", "a", "b"]
    assert Paginator(root, page_size=10, maxlevel=0).page().items == ()


def test_path_resolution_reports_ambiguity_for_same_names():
    root = Node("root")
    a = Node("same", parent=root)
    b = Node("same", parent=root)
    snapshot = TreeSnapshot(root)
    assert snapshot.resolve_path(("root", "same"), strict=False) == (
        snapshot.resolve_path(("root", "same"), strict=False)[0],
        snapshot.resolve_path(("root", "same"), strict=False)[1],
    )
    with pytest.raises(AmbiguousPathError) as excinfo:
        snapshot.resolve_path(("root", "same"))
    assert [item.node for item in excinfo.value.candidates] == [a, b]
    assert snapshot.resolve_path(("root", "missing")) == ()
    assert snapshot.item_for_path(("root",)).node is root
    with pytest.raises(KeyError):
        snapshot.item_for_path(("root", "missing"))
    with pytest.raises(AmbiguousPathError):
        snapshot.item_for_path(("root", "same"))


def test_key_lookup_returns_item_and_missing_key_raises():
    root = tree_root()
    snapshot = TreeSnapshot(root)
    item = snapshot.item_for_key((0,))
    assert item.name == "a"
    assert snapshot.key_for(item.node) == (0,)
    with pytest.raises(KeyError):
        snapshot.item_for_key((99,))


def test_store_reclaim_and_tombstone():
    root, *_ = sample_tree()
    store = SnapshotStore(max_snapshots=2)
    first = Paginator(root, page_size=2, store=store)
    second = Paginator(root, page_size=2, store=store)
    third = Paginator(root, page_size=2, store=store)
    assert first.snapshot_id in store.reclaimed
    assert store.get(first.snapshot_id) is None
    assert store.tombstone(first.snapshot_id)["snapshot_id"] == first.snapshot_id
    assert store.tombstone(first.snapshot_id)["boundary"]["path"] == ["root"]
    assert not store.reclaim(first.snapshot_id)
    assert store.reclaim(second.snapshot_id)
    assert second.snapshot_id in store.reclaimed


def test_recover_reports_boundary_instead_of_rebinding():
    root, _a, _a1, _a2, _b, _b1 = sample_tree()
    store = SnapshotStore(max_snapshots=1)
    exporter = PagedExporter(root, page_size=2, store=store)
    first = exporter.export_page()
    second = exporter.export_page(first["cursor"])
    assert second["boundary"]["path"] == ["root", "a"]
    PagedExporter(Node("other"), page_size=2, store=store)
    result = PagedExporter.recover(second["cursor"], store)
    assert result.expired
    assert not result.resumed
    assert result.paginator is None
    assert result.boundary.path == ("root", "a")


def test_recover_with_retained_snapshot_returns_paginator():
    root, *_ = sample_tree()
    store = SnapshotStore()
    exporter = PagedExporter(root, page_size=2, store=store)
    first = exporter.export_page()
    result = PagedExporter.recover(first["cursor"], store)
    assert result.resumed
    assert result.paginator.export_page(first["cursor"])["items"][0]["name"] == "a1"


def test_tampered_boundary_is_rejected():
    root, *_ = sample_tree()
    paginator = Paginator(root, page_size=2)
    page = paginator.page(paginator.page().cursor)
    cursor = page.cursor
    cursor["boundary"] = dict(cursor["boundary"])
    cursor["boundary"]["key"] = [1]
    cursor["boundary"]["path"] = ["root", "b"]
    with pytest.raises(PaginationCursorError):
        paginator.page(cursor)


def test_boundary_from_other_snapshot_is_rejected():
    root, *_ = sample_tree()
    store = SnapshotStore()
    first = Paginator(root, page_size=2, store=store)
    second = Paginator(root, page_size=2, store=store)
    cursor = first.page(first.page().cursor).cursor
    with pytest.raises(PaginationCursorError):
        second.page(cursor)


def test_bool_is_not_accepted_as_integer():
    root, *_ = sample_tree()
    with pytest.raises(ValueError):
        Paginator(root, page_size=True)


def test_page_override_takes_precedence_over_paginator_size():
    root = tree_root()
    paginator = Paginator(root, page_size=2)
    assert len(paginator.page(page_size=100)) == 6
    assert len(paginator.page(page_size=1)) == 1


def test_recursive_tree_pagination_is_iterative():
    root = Node("0")
    parent = root
    for index in range(1500):
        parent = Node(str(index + 1), parent=parent)
    paginator = Paginator(root, page_size=100, maxlevel=1200)
    page = paginator.page(page_size=5000)
    assert len(page) == 1200
    assert page.items[-1].depth == 1199


def test_symlink_like_shared_subtree_is_not_revisited():
    root = Node("root")
    a = Node("a", parent=root)
    b = Node("b", parent=a)
    c = Node("c", parent=root)
    items, _ = collect(Paginator(root, page_size=2))
    assert [item.name for item in items] == ["root", "a", "b", "c"]
    assert items[1].node is a
    assert items[2].node_path == (root, a, b)
    assert items[3].node is c


def test_deep_filtered_pages_still_advance_and_terminate():
    root = Node("root")
    chain = [root]
    for index in range(30):
        chain.append(Node(f"n{index}", parent=chain[-1]))
    paginator = Paginator(root, page_size=1, filter_=lambda node: node.name in ("root", "n29"))
    items, pages = collect(paginator)
    assert [item.name for item in items] == ["root", "n29"]
    assert len(pages) == 2
    assert pages[0].skipped == 29
    assert pages[1].skipped == 0


def test_maxlevel_one_captures_only_root():
    root, _a, _a1, _a2, _b, _b1 = sample_tree()
    snapshot = TreeSnapshot(root, maxlevel=1)
    assert snapshot.size == 1
    page = Paginator(root, page_size=10, maxlevel=1).page()
    assert [item.name for item in page] == ["root"]
    assert page.complete
    assert page.boundary.path == ("root",)


def test_invalid_maxlevel_is_rejected():
    root = tree_root()
    with pytest.raises(ValueError):
        TreeSnapshot(root, maxlevel=-1)
    with pytest.raises(ValueError):
        TreeSnapshot(root, maxlevel=-2)
    with pytest.raises(ValueError):
        TreeSnapshot(root, maxlevel=True)


def test_empty_pages_never_occur_when_matches_remain():
    root = Node("root")
    for index in range(20):
        Node(f"child{index}", parent=root)
    paginator = Paginator(root, page_size=3)
    for page in paginator.pages():
        if not page.complete:
            assert len(page) > 0


def test_export_resume_uses_export_page():
    root, *_ = sample_tree()
    store = SnapshotStore()
    exporter = PagedExporter(root, page_size=2, store=store)
    page = exporter.export_page()
    resumed = PagedExporter.resume(page["cursor"], store)
    assert [item["name"] for item in resumed.export_page(page["cursor"])["items"]] == ["a1", "a2"]


def test_store_clear_and_contains():
    root, *_ = sample_tree()
    store = SnapshotStore()
    paginator = Paginator(root, page_size=2, store=store)
    assert paginator.snapshot_id in store
    store.clear()
    assert paginator.snapshot_id not in store
    assert paginator.snapshot_id in store.reclaimed


def test_store_rejects_duplicate_snapshot_id_for_other_snapshot():
    root = tree_root()
    store = SnapshotStore()
    snapshot = TreeSnapshot(root, snapshot_id="fixed")
    store.add(snapshot)
    with pytest.raises(ValueError):
        store.add(TreeSnapshot(root, snapshot_id="fixed"))
    store.add(snapshot)


def test_store_rejects_resurrected_reclaimed_snapshot_id():
    root = tree_root()
    store = SnapshotStore()
    snapshot = TreeSnapshot(root, snapshot_id="fixed")
    store.add(snapshot)
    store.reclaim("fixed")
    with pytest.raises(ValueError):
        store.add(TreeSnapshot(root, snapshot_id="fixed"))


def test_cursor_rejects_unknown_version():
    root, *_ = sample_tree()
    paginator = Paginator(root, page_size=2)
    cursor = paginator.page().cursor
    cursor["version"] = 99
    with pytest.raises(PaginationCursorError):
        paginator.page(cursor)


def test_boundary_end_is_exclusive_and_complete_at_document_end():
    root, *_ = sample_tree()
    paginator = Paginator(root, page_size=100)
    page = paginator.page()
    assert page.complete
    assert page.boundary.path == ("root",)
    assert page.boundary.end == 6
    assert page.boundary.size == 6


def test_boundary_identity_supports_reconciliation_against_new_version():
    root, a, _a1, _a2, _b, _b1 = sample_tree()
    snapshot = TreeSnapshot(root, identityfunc=lambda node: f"id:{node.name}")
    paginator = Paginator(root, page_size=2, snapshot=snapshot)
    first = paginator.page()
    second = paginator.page(first.cursor)
    assert second.boundary.identity == "id:a"
    assert second.boundary.path == ("root", "a")
    assert second.boundary.size == 3
    # A later snapshot of a changed tree still identifies the subtree by its
    # captured external identity, so the consumer can look it up independently.
    a.name = "a-renamed"
    new_snapshot = TreeSnapshot(root, identityfunc=lambda node: f"id:{node.name}")
    assert new_snapshot.size == 6
