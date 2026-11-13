Pagination
==========

Large trees are often delivered to a consumer page by page.  The
:any:`anytree.pagination` module provides depth-first pagination with a stable
snapshot per run.

Snapshot semantics
------------------

Starting a pagination captures an immutable snapshot of the tree:

* the set of member nodes,
* sibling order,
* display names used for ancestor paths,
* filter verdicts,
* optional payloads.

Pages requested later are served from this snapshot.  Changing a display name
or inserting a sibling while a pagination is running does not affect the
running pagination.  A fresh pagination sees the live tree instead.

>>> from anytree import Node, Paginator
>>> root = Node("root")
>>> a = Node("a", parent=root)
>>> a1 = Node("a1", parent=a)
>>> b = Node("b", parent=root)
>>> paginator = Paginator(root, page_size=1)
>>> page = paginator.page()
>>> [item.display_path for item in page]
['root']
>>> a.name = "renamed"
>>> page = paginator.page(page.cursor)
>>> [(item.name, item.display_path) for item in page]
[('a', 'root/a')]

Filtered pages
--------------

A filter only selects delivered nodes.  Non-matching nodes are still
traversed, so a delivered node always carries its complete ancestor path:

>>> paginator = Paginator(root, page_size=10, filter_=lambda node: node.name == "a1")
>>> [(item.name, item.ancestor_path) for item in paginator.page()]
[('a1', ('root', 'renamed'))]

Each page item offers the display path and the node objects along that path:

>>> item = paginator.page().items[0]
>>> item.path
('root', 'renamed', 'a1')
>>> item.ancestor_nodes
(Node('/root'), Node('/root/renamed'))

Since the snapshot permits indistinguishable display names (for example
inserted same-name siblings), a display path may match several nodes.  Each
item therefore also carries a structural ``key``.  Use it to distinguish them:

>>> root = Node("root")
>>> first = Node("same", parent=root)
>>> second = Node("same", parent=root)
>>> [item.key for item in Paginator(root, page_size=10).page()]
[(), (0,), (1,)]

Cursors
-------

Every page carries a JSON-serializable cursor.  ``None`` marks the final page.
A consumer which only needs nodes of interest can discard non-matching nodes
and later resume with the last cursor it processed:

>>> page = Paginator(root, page_size=2).page()
>>> import json
>>> json.loads(json.dumps(page.cursor))["page_number"]
1

Page boundaries are also reported explicitly.  A boundary describes the last
complete subtree in depth-first order, including its captured path, key,
depth-first ``start``/``end`` positions, ``size`` and ``matched_count``.
Only subtrees whose whole traversal lies before the cursor position qualify.

Resuming requires the original snapshot.  Use :any:`SnapshotStore` to retain
snapshots for a bounded time and to resume with :any:`Paginator.resume`.

Snapshot reclamation
--------------------

When a snapshot has been reclaimed, the original membership, sibling order
and display names are no longer known.  The current live tree must not be
used to continue the old pagination.

Instead, resuming raises :any:`SnapshotExpiredError`.  Its ``boundary`` is the
last complete subtree which the cursor's snapshot had fully traversed.  The
boundary contains the captured display path, structural key, depth-first
start/end positions, subtree size and optional external identity.  A consumer
reconciles its data up to that boundary and starts a new pagination.

:any:`Paginator.recover` and :any:`PagedExporter.recover` return the same
boundary as a :any:`ResumeResult` without raising.

Exporting
---------

:any:`PagedExporter` adds payload capture and produces plain dictionaries:
>>> from anytree import PagedExporter
>>> exporter = PagedExporter(root, page_size=2)
>>> page = exporter.export_page()
>>> page["items"][0]["payload"]
{'name': 'root'}
