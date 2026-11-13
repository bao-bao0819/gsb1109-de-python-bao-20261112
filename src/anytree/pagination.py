"""Snapshot based, depth-first pagination of a tree.

The pagination API delivers a tree as an ordered sequence of pages.
Every pagination starts with an immutable snapshot, which captures membership,
sibling order and display names.  Later pages are served from that snapshot,
even if the live tree is modified in between.

Only nodes matching an optional filter are delivered, but every delivered node
carries its complete ancestor path as captured by the snapshot.

>>> from anytree import Node
>>> root = Node("root")
>>> a = Node("a", parent=root)
>>> a1 = Node("a1", parent=a)
>>> a2 = Node("a2", parent=a)
>>> b = Node("b", parent=root)

>>> paginator = Paginator(root, page_size=2)
>>> page = paginator.page()
>>> [item.name for item in page]
['root', 'a']
>>> page.complete
False
>>> page = paginator.page(page.cursor)
>>> [item.name for item in page]
['a1', 'a2']
>>> page.boundary.path
('root', 'a')
>>> page.cursor is not None
True
>>> len(paginator.page(page.cursor))
1

Cursors are JSON serializable and can be handed to another process.  The
consumer resumes against the original snapshot through a :any:`SnapshotStore`.
When the store has reclaimed the snapshot, :any:`SnapshotExpiredError` reports
the last complete subtree boundary instead of silently continuing on the live
tree.
"""

from __future__ import annotations

import json
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AmbiguousPathError",
    "Page",
    "PageItem",
    "PagedExporter",
    "PaginationCursorError",
    "PaginationError",
    "Paginator",
    "ResumeResult",
    "SnapshotExpiredError",
    "SnapshotStore",
    "SubtreeBoundary",
    "TreeSnapshot",
]

_CURSOR_VERSION = 1
_INTERNAL_ATTRS = {
    "_NodeMixin__children",
    "_NodeMixin__parent",
    "_LightNodeMixin__children",
    "_LightNodeMixin__parent",
    "__children",
    "__parent",
}


class PaginationError(RuntimeError):
    """Base class for pagination errors."""


class PaginationCursorError(PaginationError, ValueError):
    """Raised when a pagination cursor is malformed or belongs to another export."""


class AmbiguousPathError(PaginationError, ValueError):
    """Raised when a display path matches more than one snapshot node."""

    def __init__(self, path, candidates):
        super().__init__(f"Path {path!r} matches {len(candidates)} snapshot nodes.")
        self.path = tuple(path)
        self.candidates = tuple(candidates)


class SnapshotExpiredError(PaginationError):
    """Raised when a resumed pagination refers to a reclaimed snapshot.

    The old snapshot cannot be replaced by the current live tree, because
    membership, sibling order and display names may have changed.  Instead the
    error provides the last complete subtree boundary that the consumer can
    reconcile and restart from.

    Attributes:
        cursor (dict): The cursor which was used for the failed resume.
        boundary (Optional[SubtreeBoundary]): Last complete subtree boundary
            known by the cursor, ``None`` if no complete subtree had been
            reached.
    """

    def __init__(self, cursor, boundary):
        if boundary is None:
            message = "The pagination snapshot is no longer available and no complete subtree boundary was reached."
        else:
            message = (
                "The pagination snapshot is no longer available. "
                f"Reconcile up to the last complete subtree boundary at {boundary.display_path!r}."
            )
        super().__init__(message)
        self.cursor = dict(cursor)
        self.boundary = boundary


@dataclass(frozen=True)
class SubtreeBoundary:
    """A complete subtree which has been fully traversed by a pagination.

    Attributes:
        snapshot_id (str): Snapshot which produced the boundary.
        key (tuple): Structural key of the subtree root inside the snapshot.
        path (tuple): Display path from the pagination root, as captured by the
            snapshot.
        start (int): Inclusive depth-first position of the subtree root.
        end (int): Exclusive depth-first position of the subtree.
        size (int): Number of captured nodes inside the subtree.
        matched_count (int): Number of matching nodes delivered up to `end`.
        identity: Optional external identity captured by `identityfunc`.
    """

    snapshot_id: str
    key: tuple
    path: tuple
    start: int
    end: int
    size: int
    matched_count: int
    identity: Any = None

    @property
    def name(self):
        """Display name of the boundary node in the snapshot."""
        return self.path[-1]

    @property
    def display_path(self):
        """Slash joined display path of the boundary node."""
        return "/".join(str(name) for name in self.path)

    @property
    def depth(self):
        """Depth of the boundary node relative to the pagination root."""
        return len(self.key)

    def to_dict(self):
        """Return a JSON-serializable dictionary representation."""
        return {
            "snapshot_id": self.snapshot_id,
            "key": list(self.key),
            "path": list(self.path),
            "start": self.start,
            "end": self.end,
            "size": self.size,
            "matched_count": self.matched_count,
            "identity": self.identity,
        }

    @classmethod
    def from_dict(cls, data):
        """Build a boundary from its dictionary representation."""
        _validate_mapping(data, "boundary")
        try:
            key = tuple(data["key"])
            path = tuple(data["path"])
            start = data["start"]
            end = data["end"]
            size = data["size"]
        except KeyError as exc:
            raise PaginationCursorError(f"Boundary is missing {exc.args[0]!r}.") from exc
        _validate_ints(key, "boundary key")
        if len(path) != len(key) + 1:
            raise PaginationCursorError("Boundary path length must match key depth.")
        for fieldname, value in (("start", start), ("end", end), ("size", size)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PaginationCursorError(f"Boundary {fieldname} must be a non-negative integer.")
        if start < 0 or end < start or size != end - start:
            raise PaginationCursorError("Boundary positions are inconsistent.")
        matched_count = data.get("matched_count", 0)
        if not isinstance(matched_count, int) or isinstance(matched_count, bool) or matched_count < 0:
            raise PaginationCursorError("Boundary matched_count must be a non-negative integer.")
        snapshot_id = data.get("snapshot_id")
        if not isinstance(snapshot_id, str):
            raise PaginationCursorError("Boundary snapshot_id must be a string.")
        return cls(
            snapshot_id=snapshot_id,
            key=key,
            path=path,
            start=start,
            end=end,
            size=size,
            matched_count=matched_count,
            identity=data.get("identity"),
        )


@dataclass(frozen=True, eq=False)
class PageItem:
    """A single matching node delivered on a page.

    Attributes:
        node: The tree node.
        key (tuple): Structural key of the node in the snapshot.
        path (tuple): Complete display path from the pagination root down to
            `node`, including the root and `node` itself.
        node_path (tuple): The nodes along `path`, captured by the snapshot.
        depth (int): Depth relative to the pagination root (root is ``0``).
        payload: Payload captured at snapshot time, or ``None`` for plain
            paginations.
    """

    node: Any
    key: tuple
    path: tuple
    node_path: tuple
    depth: int
    payload: Any = field(default=None)

    def __eq__(self, other):
        if not isinstance(other, PageItem):
            return NotImplemented
        return self.node is other.node and self.key == other.key and self.path == other.path

    def __hash__(self):
        return hash((id(self.node), self.key, self.path))

    @property
    def name(self):
        """Display name of the node as captured by the snapshot."""
        return self.path[-1]

    @property
    def ancestor_path(self):
        """Display path of all ancestors of the node, excluding the node itself."""
        return self.path[:-1]

    @property
    def ancestors(self):
        """Alias of :any:`PageItem.ancestor_path`."""
        return self.ancestor_path

    @property
    def ancestor_nodes(self):
        """Ancestor nodes of the item, excluding the node itself."""
        return self.node_path[:-1]

    @property
    def display_path(self):
        """Slash joined display path of the node."""
        separator = getattr(self.node, "separator", "/")
        return separator.join(str(name) for name in self.path)

    def to_dict(self):
        """Return a JSON-serializable dictionary representation."""
        return {
            "key": list(self.key),
            "path": list(self.path),
            "ancestor_path": list(self.ancestor_path),
            "name": self.name,
            "depth": self.depth,
        }


@dataclass(frozen=True)
class Page:
    """One page of a pagination.

    Attributes:
        snapshot_id (str): Identifier of the snapshot which serves the page.
        page_number (int): Zero based page number.
        items (tuple): Matching :any:`PageItem` objects in depth-first order.
        cursor (Optional[dict]): Cursor for the next page, ``None`` on the last
            page.
        boundary (Optional[SubtreeBoundary]): Last complete subtree boundary
            known after this page.
        complete (bool): Whether this is the last page.
        skipped (int): Number of non-matching nodes traversed for this page.
    """

    snapshot_id: str
    page_number: int
    items: tuple
    cursor: Any
    boundary: Any
    complete: bool
    skipped: int = 0

    @property
    def matched_count(self):
        """Number of matching items on this page."""
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    @property
    def nodes(self):
        """Matching tree nodes in depth-first order."""
        return tuple(item.node for item in self.items)

    @property
    def next_cursor(self):
        """Alias of :any:`Page.cursor`."""
        return self.cursor

    @property
    def has_more(self):
        """Whether another page is available."""
        return not self.complete


class _Entry:
    __slots__ = ("depth", "identity", "key", "matched", "node", "node_path", "path", "payload", "start")

    def __init__(self, node, key, path, node_path, depth, matched, payload, identity=None):
        self.node = node
        self.key = key
        self.path = path
        self.node_path = node_path
        self.depth = depth
        self.matched = matched
        self.payload = payload
        self.identity = identity
        self.start = None


class TreeSnapshot:
    r"""Immutable depth-first snapshot of a tree.

    The snapshot captures all nodes, their sibling order and their display
    names.  Filtering does not prune the tree: non-matching nodes are still
    traversed and remain part of ancestor paths.

    Args:
        root: Root node of the tree to capture.

    Keyword Args:
        filter\\_: Function called with every node.  Only nodes it accepts are
            delivered on pages.  All nodes contribute to ancestor paths.
        pathfunc: Function called with every node to produce its display name.
            Defaults to the node name.  The result is captured on creation.
        payloadfunc: Function called with every node to capture a payload.
        identityfunc: Function called with every node to capture an external
            identity, which is carried by subtree boundaries so a consumer can
            reconcile them against another tree version.
        maxlevel (int): Maximum number of levels to capture.  ``1`` captures
            only the root, ``None`` captures everything.
        snapshot_id (str): Explicit identifier, by default a random UUID.
    """

    def __init__(
        self, root, filter_=None, pathfunc=None, payloadfunc=None, identityfunc=None, maxlevel=None, snapshot_id=None
    ):
        if maxlevel is not None and (
            not isinstance(maxlevel, int) or isinstance(maxlevel, bool) or maxlevel < 0
        ):
            raise ValueError("maxlevel must be a non-negative integer or None.")
        self.root = root
        self.filter_ = filter_
        self.pathfunc = pathfunc or _default_path_element
        self.payloadfunc = payloadfunc
        self.identityfunc = identityfunc
        self.maxlevel = maxlevel
        self.snapshot_id = snapshot_id or uuid.uuid4().hex
        self._entries = []
        self._ends = []
        self._matched_prefix = []
        self._boundaries = []
        self._build()

    def _build(self):
        filter_ = self.filter_ or _default_filter
        maxlevel = self.maxlevel
        entries = []
        stack = [(self.root, (), (), (), 0)]
        while stack:
            node, key, parentpath, parentnodes, depth = stack.pop()
            if maxlevel is not None and depth >= maxlevel:
                continue
            path = (*parentpath, self.pathfunc(node))
            node_path = (*parentnodes, node)
            matched = bool(filter_(node))
            payload = self.payloadfunc(node) if self.payloadfunc else None
            identity = self.identityfunc(node) if self.identityfunc else None
            entry = _Entry(node, key, path, node_path, depth, matched, payload, identity)
            entry.start = len(entries)
            entries.append(entry)
            children = tuple(node.children)
            for index in reversed(range(len(children))):
                stack.append((children[index], (*key, index), path, node_path, depth + 1))
        self._entries = entries
        self._ends = self._subtree_ends(entries)
        self._matched_prefix = self._prefix_counts(entries)
        self._boundaries = self._boundary_indexes(entries, self._ends)

    @staticmethod
    def _subtree_ends(entries):
        ends = [len(entries)] * len(entries)
        openentries = []
        for index, entry in enumerate(entries):
            while openentries and entries[openentries[-1]].depth >= entry.depth:
                ends[openentries.pop()] = index
            openentries.append(index)
        return ends

    @staticmethod
    def _prefix_counts(entries):
        counts = [0]
        for entry in entries:
            counts.append(counts[-1] + (1 if entry.matched else 0))
        return counts

    @staticmethod
    def _boundary_indexes(entries, ends):
        shallowest = {}
        for index, entry in enumerate(entries):
            end = ends[index]
            current = shallowest.get(end)
            if current is None or entry.depth < entries[current].depth:
                shallowest[end] = index
        boundaries = [None] * (len(entries) + 1)
        latest = None
        for position in range(len(entries) + 1):
            latest = shallowest.get(position, latest)
            boundaries[position] = latest
        return boundaries

    @property
    def size(self):
        """Number of captured nodes, including non-matching nodes."""
        return len(self._entries)

    @property
    def matched_count(self):
        """Number of nodes matching the snapshot filter."""
        return self._matched_prefix[-1]

    def boundary_at(self, position):
        """Return the last complete subtree boundary before `position`.

        Only subtrees whose depth-first traversal is entirely contained in
        ``[0, position)`` qualify.  A subtree is therefore an exact
        reconciliation anchor: everything up to its ``end`` is known to be
        complete.  No boundary is reported while a subtree is still being
        traversed from the beginning, which prevents a consumer from
        reconciling a partially delivered subtree.
        """
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise ValueError("position must be a non-negative integer.")
        if position > len(self._entries):
            raise ValueError("position is outside the snapshot.")
        index = self._boundaries[position]
        if index is None:
            return None
        entry = self._entries[index]
        return SubtreeBoundary(
            snapshot_id=self.snapshot_id,
            key=entry.key,
            path=entry.path,
            start=entry.start,
            end=self._ends[index],
            size=self._ends[index] - entry.start,
            matched_count=self._matched_prefix[self._ends[index]],
            identity=entry.identity,
        )

    def key_for(self, node):
        """Return the structural key of `node` in the snapshot, or ``None``."""
        for entry in self._entries:
            if entry.node is node:
                return entry.key
        return None

    def item_for_key(self, key):
        """Return the snapshot item with structural `key`.

        Raises:
            KeyError: if `key` is not part of this snapshot.
        """
        for entry in self._entries:
            if entry.key == tuple(key):
                return self._item(entry)
        raise KeyError(tuple(key))

    @staticmethod
    def _item(entry):
        return PageItem(
            node=entry.node,
            key=entry.key,
            path=entry.path,
            node_path=entry.node_path,
            depth=entry.depth,
            payload=entry.payload,
        )

    def items_for_path(self, path):
        """Return all snapshot items whose captured display path equals `path`."""
        path = tuple(path)
        return tuple(self._item(entry) for entry in self._entries if entry.path == path)

    def item_for_path(self, path):
        """Return exactly one item for `path`.

        Raises:
            KeyError: if no snapshot item matches `path`.
            AmbiguousPathError: if several items match `path`.
        """
        items = self.items_for_path(path)
        if not items:
            raise KeyError(tuple(path))
        if len(items) > 1:
            raise AmbiguousPathError(path, items)
        return items[0]

    def resolve_path(self, path, strict=True):
        """Resolve a captured display path to snapshot items.

        Since the snapshot permits indistinguishable names (for example
        inserted same-name siblings), a path may match multiple nodes.  With
        `strict` an :any:`AmbiguousPathError` is raised instead of silently
        choosing one of them.

        Args:
            path: Sequence of display names from the pagination root.
            strict (bool): Raise on ambiguity.

        Returns:
            tuple: All matching :any:`PageItem` objects, empty if none.

        Raises:
            AmbiguousPathError: if `strict` and more than one entry matches.
        """
        items = self.items_for_path(path)
        if strict and len(items) > 1:
            raise AmbiguousPathError(path, items)
        return items


class SnapshotStore:
    """A bounded store of reusable pagination snapshots.

    Consumers resume a pagination by snapshot id.  When a snapshot has been
    reclaimed, resuming raises :any:`SnapshotExpiredError`; the store never
    substitutes a newer snapshot.

    Keyword Args:
        max_snapshots (int): Maximum number of retained snapshots.  The least
            recently used snapshot is reclaimed first.  ``None`` retains all
            snapshots.
    """

    def __init__(self, max_snapshots=None):
        if max_snapshots is not None and (
            not isinstance(max_snapshots, int) or isinstance(max_snapshots, bool) or max_snapshots < 1
        ):
            raise ValueError("max_snapshots must be a positive integer or None.")
        self.max_snapshots = max_snapshots
        self._snapshots = OrderedDict()
        self._reclaimed = []
        self._tombstones = OrderedDict()

    def add(self, snapshot):
        """Register `snapshot` and return it.

        Raises:
            ValueError: if the identifier is already in use or has already
                been reclaimed.  Reusing a reclaimed identifier would
                silently rebind old cursors to a newer tree, which is
                explicitly unsupported.
        """
        existing = self._snapshots.get(snapshot.snapshot_id)
        if existing is not None and existing is not snapshot:
            raise ValueError(f"Snapshot id {snapshot.snapshot_id!r} is already in use.")
        if snapshot.snapshot_id in self._tombstones:
            raise ValueError(f"Snapshot id {snapshot.snapshot_id!r} has already been reclaimed.")
        self._snapshots[snapshot.snapshot_id] = snapshot
        self._snapshots.move_to_end(snapshot.snapshot_id)
        while self.max_snapshots is not None and len(self._snapshots) > self.max_snapshots:
            expired_id, expired = self._snapshots.popitem(last=False)
            self._reclaim_snapshot(expired_id, expired)
        return snapshot

    def get(self, snapshot_id):
        """Return the snapshot with `snapshot_id`, or ``None`` if reclaimed."""
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is not None:
            self._snapshots.move_to_end(snapshot_id)
        return snapshot

    def reclaim(self, snapshot_id):
        """Drop `snapshot_id` and return whether it was present."""
        snapshot = self._snapshots.pop(snapshot_id, None)
        if snapshot is None:
            return False
        self._reclaim_snapshot(snapshot_id, snapshot)
        return True

    def clear(self):
        """Drop all retained snapshots."""
        for snapshot_id, snapshot in self._snapshots.items():
            self._reclaim_snapshot(snapshot_id, snapshot)
        self._snapshots.clear()

    def __contains__(self, snapshot_id):
        return snapshot_id in self._snapshots

    def __len__(self):
        return len(self._snapshots)

    @property
    def reclaimed(self):
        """Identifiers of snapshots that have been reclaimed by this store."""
        return tuple(self._reclaimed)

    def tombstone(self, snapshot_id):
        """Boundary data retained for a reclaimed snapshot.

        Returns:
            dict: ``{"snapshot_id": ..., "boundary": ...}`` or ``None`` if the
            identifier is unknown.  The boundary covers the whole snapshot
            and can be used to reconcile a consumer which has no cursor.
        """
        return self._tombstones.get(snapshot_id)

    def _reclaim_snapshot(self, snapshot_id, snapshot):
        self._reclaimed.append(snapshot_id)
        self._tombstones[snapshot_id] = {
            "snapshot_id": snapshot_id,
            "boundary": _boundary_to_dict(snapshot.boundary_at(snapshot.size)),
        }


class Paginator:
    r"""Depth-first paginator over an immutable :any:`TreeSnapshot`.

    Args:
        root: Root node of the tree.

    Keyword Args:
        page_size (int): Maximum number of matching nodes per page.
        filter\\_: Function called with every node.  Only accepted nodes are
            delivered, all nodes contribute to ancestor paths.
        pathfunc: Function called with every node to produce its display name.
        maxlevel (int): Maximum number of captured levels.
        store (SnapshotStore): Store for resumable snapshots.  A private store
            is created if omitted.
        snapshot (TreeSnapshot): Pre-built snapshot, mainly for advanced use.
        identityfunc: Function called with every node to capture an external
            identity for reconciliation boundaries.
    """

    def __init__(
        self,
        root,
        page_size=10,
        filter_=None,
        pathfunc=None,
        maxlevel=None,
        store=None,
        snapshot=None,
        identityfunc=None,
    ):
        if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size < 1:
            raise ValueError("page_size must be a positive integer.")
        self.store = store if store is not None else SnapshotStore()
        if snapshot is None:
            snapshot = TreeSnapshot(
                root, filter_=filter_, pathfunc=pathfunc, maxlevel=maxlevel, identityfunc=identityfunc
            )
        self.root = snapshot.root
        self.snapshot = self.store.add(snapshot)
        self.page_size = page_size

    @property
    def snapshot_id(self):
        """Identifier of the snapshot serving this paginator."""
        return self.snapshot.snapshot_id

    def page(self, cursor=None, page_size=None):
        """Return the page selected by `cursor`, or the first page if omitted."""
        size = self._validate_page_size(page_size)
        parsed = self._validate_cursor(cursor) if cursor is not None else None
        position = parsed["position"] if parsed is not None else 0
        page_number = parsed["page_number"] if parsed is not None else 0
        entries = self.snapshot._entries
        items = []
        skipped = 0
        index = position
        while index < len(entries) and len(items) < size:
            entry = entries[index]
            if entry.matched:
                items.append(self.snapshot._item(entry))
            else:
                skipped += 1
            index += 1
        while index < len(entries) and not entries[index].matched:
            skipped += 1
            index += 1
        complete = index >= len(entries)
        next_cursor = None
        if not complete:
            next_cursor = {
                "version": _CURSOR_VERSION,
                "snapshot_id": self.snapshot.snapshot_id,
                "position": index,
                "page_number": page_number + 1,
                "page_size": size,
                "boundary": _boundary_to_dict(self.snapshot.boundary_at(index)),
            }
        return Page(
            snapshot_id=self.snapshot.snapshot_id,
            page_number=page_number,
            items=tuple(items),
            cursor=next_cursor,
            boundary=self.snapshot.boundary_at(index),
            complete=complete,
            skipped=skipped,
        )

    def pages(self, start=None, page_size=None):
        """Yield pages in depth-first order, starting at cursor `start`."""
        cursor = start
        while True:
            page = self.page(cursor, page_size=page_size)
            yield page
            if page.complete:
                return
            cursor = page.cursor

    @classmethod
    def resume(cls, cursor, store, page_size=None):
        """Resume a pagination from `cursor` against `store`.

        Raises:
            SnapshotExpiredError: if `store` has reclaimed the snapshot.  The
                exception carries the last complete subtree boundary.  The
                current live tree is intentionally not used as a substitute.
            PaginationCursorError: if `cursor` is malformed or inconsistent.
        """
        parsed = _parse_cursor(cursor)
        snapshot = store.get(parsed["snapshot_id"])
        if snapshot is None:
            raise SnapshotExpiredError(parsed, _boundary_from_dict(parsed.get("boundary")))
        paginator = cls.__new__(cls)
        paginator.store = store
        paginator.root = snapshot.root
        paginator.snapshot = snapshot
        paginator.page_size = paginator._validate_page_size(page_size, default=parsed["page_size"])
        paginator._validate_cursor(parsed, validate_page_size=False)
        return paginator

    @classmethod
    def recover(cls, cursor, store, page_size=None):
        """Try to resume `cursor` and report the outcome without raising on expiry.

        Returns:
            ResumeResult: ``resumed`` is ``True`` with a usable paginator, or
            ``False`` with the last complete subtree boundary carried by the
            cursor.  The live tree is never consulted as a replacement.

        Raises:
            PaginationCursorError: if the cursor itself is malformed or a
                boundary claims more than the cursor has consumed.
        """
        parsed = _parse_cursor(cursor)
        snapshot = store.get(parsed["snapshot_id"])
        boundary = _boundary_from_dict(parsed.get("boundary"))
        if snapshot is None:
            return ResumeResult(cursor=parsed, boundary=boundary, paginator=None)
        return ResumeResult(
            cursor=parsed,
            boundary=boundary,
            paginator=cls.resume(parsed, store, page_size=page_size),
        )

    def _validate_page_size(self, page_size, default=None):
        if page_size is not None:
            size = page_size
        elif default is not None:
            size = default
        else:
            size = self.page_size
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("page_size must be a positive integer.")
        return size

    def _validate_cursor(self, cursor, validate_page_size=True):
        parsed = _parse_cursor(cursor)
        if parsed["snapshot_id"] != self.snapshot.snapshot_id:
            raise PaginationCursorError(
                f"Cursor belongs to snapshot {parsed['snapshot_id']!r}, not {self.snapshot.snapshot_id!r}."
            )
        position = parsed["position"]
        if position > len(self.snapshot._entries):
            raise PaginationCursorError("Cursor position is outside the snapshot.")
        if position < len(self.snapshot._entries) and not self.snapshot._entries[position].matched:
            raise PaginationCursorError("Cursor position does not point to a matching node.")
        boundary = parsed.get("boundary")
        if boundary is not None:
            if boundary["snapshot_id"] != self.snapshot.snapshot_id:
                raise PaginationCursorError("Cursor boundary belongs to another snapshot.")
            if boundary["end"] > position:
                raise PaginationCursorError("Cursor boundary is ahead of the cursor position.")
            expected = self.snapshot.boundary_at(position)
            if expected is None or boundary != _boundary_to_dict(expected):
                raise PaginationCursorError("Cursor boundary does not match the snapshot.")
        if validate_page_size:
            self._validate_page_size(parsed["page_size"])
        return parsed


@dataclass(frozen=True)
class ResumeResult:
    """Outcome of :any:`Paginator.recover` or :any:`PagedExporter.recover`.

    Attributes:
        cursor (dict): The normalized cursor which was supplied.
        boundary (Optional[SubtreeBoundary]): Last complete subtree boundary
            known by the cursor.  Always available, also for an expired
            snapshot.
        paginator (Optional[Paginator]): Usable paginator if the snapshot is
            still retained, otherwise ``None``.
    """

    cursor: dict
    boundary: Any
    paginator: Any

    @property
    def resumed(self):
        """Whether the original snapshot is still available."""
        return self.paginator is not None

    @property
    def expired(self):
        """Whether the snapshot has been reclaimed."""
        return self.paginator is None


class PagedExporter:
    r"""Paginated export which preserves snapshot paths and payloads.

    The export captures membership, sibling order, display names and payloads
    when it starts.  Later tree modifications do not alter an export in
    progress.

    Args:
        root: Root node of the tree.

    Keyword Args:
        page_size (int): Maximum number of matching nodes per page.
        filter\\_: Function called with every node to select exported nodes.
        pathfunc: Function called with every node to produce its display name.
        payloadfunc: Function called with every node to produce its payload.
            By default the node's public data attributes are captured.
        identityfunc: Function called with every node to capture an external
            identity for reconciliation boundaries.
        maxlevel (int): Maximum number of captured levels.
        store (SnapshotStore): Store for resumable exports.
        snapshot (TreeSnapshot): Pre-built snapshot, mainly for advanced use.
    """

    def __init__(
        self,
        root,
        page_size=10,
        filter_=None,
        pathfunc=None,
        payloadfunc=None,
        identityfunc=None,
        maxlevel=None,
        store=None,
        snapshot=None,
    ):
        if snapshot is None:
            snapshot = TreeSnapshot(
                root,
                filter_=filter_,
                pathfunc=pathfunc,
                payloadfunc=payloadfunc or _default_payload,
                identityfunc=identityfunc,
                maxlevel=maxlevel,
            )
        self.paginator = Paginator(root, page_size=page_size, store=store, snapshot=snapshot)

    @property
    def snapshot_id(self):
        """Identifier of the export snapshot."""
        return self.paginator.snapshot_id

    @property
    def snapshot(self):
        """The export snapshot."""
        return self.paginator.snapshot

    @property
    def store(self):
        """The snapshot store used by the export."""
        return self.paginator.store

    def export_page(self, cursor=None, page_size=None):
        """Return the next export page as a JSON-serializable dictionary."""
        page = self.paginator.page(cursor, page_size=page_size)
        items = []
        for item in page.items:
            items.append(
                {
                    "key": list(item.key),
                    "path": list(item.path),
                    "ancestor_path": list(item.ancestor_path),
                    "name": item.name,
                    "depth": item.depth,
                    "payload": item.payload,
                }
            )
        return {
            "snapshot_id": page.snapshot_id,
            "page_number": page.page_number,
            "items": items,
            "cursor": page.cursor,
            "boundary": _boundary_to_dict(page.boundary),
            "complete": page.complete,
            "skipped": page.skipped,
        }

    def export_pages(self, start=None, page_size=None):
        """Yield export pages, starting at cursor `start`."""
        cursor = start
        while True:
            page = self.export_page(cursor, page_size=page_size)
            yield page
            if page["complete"]:
                return
            cursor = page["cursor"]

    @classmethod
    def resume(cls, cursor, store, page_size=None):
        """Resume an export from `cursor` against the original snapshot store."""
        exporter = cls.__new__(cls)
        exporter.paginator = Paginator.resume(cursor, store, page_size=page_size)
        return exporter

    @classmethod
    def recover(cls, cursor, store, page_size=None):
        """Try to resume an export and report the outcome without raising on expiry."""
        result = Paginator.recover(cursor, store, page_size=page_size)
        exporter = None
        if result.paginator is not None:
            exporter = cls.__new__(cls)
            exporter.paginator = result.paginator
        return ResumeResult(cursor=result.cursor, boundary=result.boundary, paginator=exporter)


def _parse_cursor(cursor):
    if isinstance(cursor, str):
        try:
            cursor = json.loads(cursor)
        except json.JSONDecodeError as exc:
            raise PaginationCursorError("Cursor is not valid JSON.") from exc
    _validate_mapping(cursor, "cursor")
    if cursor.get("version") != _CURSOR_VERSION:
        raise PaginationCursorError(f"Unsupported cursor version {cursor.get('version')!r}.")
    snapshot_id = cursor.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise PaginationCursorError("Cursor snapshot_id must be a non-empty string.")
    position = cursor.get("position")
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        raise PaginationCursorError("Cursor position must be a non-negative integer.")
    page_number = cursor.get("page_number")
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 0:
        raise PaginationCursorError("Cursor page_number must be a non-negative integer.")
    page_size = cursor.get("page_size")
    if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size < 1:
        raise PaginationCursorError("Cursor page_size must be a positive integer.")
    boundary = cursor.get("boundary")
    if boundary is not None:
        boundary = _boundary_from_dict(boundary).to_dict()
        if boundary["snapshot_id"] != snapshot_id:
            raise PaginationCursorError("Cursor boundary belongs to another snapshot.")
        if boundary["end"] > position:
            raise PaginationCursorError("Cursor boundary is ahead of the cursor position.")
        if boundary["start"] > position:
            raise PaginationCursorError("Cursor boundary start is ahead of the cursor position.")
        _validate_ints(tuple(boundary["key"]), "boundary key")
        _validate_ints((boundary["start"], boundary["end"], boundary["size"], boundary["matched_count"]), "boundary")
    return {
        "version": _CURSOR_VERSION,
        "snapshot_id": snapshot_id,
        "position": position,
        "page_number": page_number,
        "page_size": page_size,
        "boundary": boundary,
    }


def _validate_mapping(value, name):
    if not isinstance(value, Mapping):
        raise PaginationCursorError(f"{name.capitalize()} must be a mapping or JSON object.")


def _validate_ints(values, name):
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PaginationCursorError(f"{name.capitalize()} must contain non-negative integers.")


def _boundary_to_dict(boundary):
    if boundary is None:
        return None
    data = boundary.to_dict()
    try:
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError):
        data["identity"] = _identity_to_json(data["identity"])
        try:
            json.dumps(data, allow_nan=False)
        except (TypeError, ValueError):
            data = dict(boundary.to_dict(), identity=None)
    return data


def _identity_to_json(identity):
    if identity is None or isinstance(identity, (str, int, float, bool)):
        return identity
    try:
        json.dumps(identity, allow_nan=False)
    except (TypeError, ValueError):
        return repr(identity)
    return identity


def _boundary_from_dict(data):
    return None if data is None else SubtreeBoundary.from_dict(data)


def _default_filter(node):
    return True


def _default_path_element(node):
    try:
        return node.name
    except AttributeError:
        return node


def _default_payload(node):
    data = {}
    try:
        attrs = vars(node)
    except TypeError:
        attrs = {}
    for key, value in attrs.items():
        if key not in _INTERNAL_ATTRS and not key.startswith("_"):
            data[key] = value
    for cls in type(node).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            if slot in _INTERNAL_ATTRS or slot.startswith("_") or slot in data:
                continue
            try:
                data[slot] = getattr(node, slot)
            except AttributeError:
                pass
    return data
