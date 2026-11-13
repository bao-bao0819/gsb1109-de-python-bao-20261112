"""
Asset inventory over trees with symbolic links (multiple entry points).

A theme directory may expose the same asset through several entries.
:any:`SymlinkNode` instances are entries referencing another tree node.

This module provides two statistics at once:

* the **entry count** -- how many entries are seen while walking the tree
  structure, so the same real asset is counted once per reachable path;
* the **asset count** -- how many distinct real assets exist, deduplicated by
  their asset identifier (a link contributes the identifier of its target).

The walk follows the actual parent/child structure. A link is an entry of its
target asset; it does not jump into the target's children, because the target
subtree is walked where it is attached. When a link points back to an asset on
the current walk path, the path leading to the loop is kept as a diagnostic and
expansion along that link stops. The looping link itself still counts as an
entry, and the same asset reached through another, acyclic path counts again.
Subtree subtotals do not necessarily add up to the global deduplicated count;
:any:`InventoryReport.subtree_discrepancies` lists the assets explaining the gap.
"""

from typing import NamedTuple

from .node.nodemixin import NodeMixin
from .node.symlinknodemixin import SymlinkNodeMixin

__all__ = ["AssetInventory", "InventoryReport", "LoopOccurrence", "SubtreeInventory", "path_names"]


class SubtreeInventory(NamedTuple):
    """Inventory numbers for one entry point (subtree root)."""

    root: NodeMixin
    entry_count: int
    asset_ids: tuple
    loops: tuple

    @property
    def asset_count(self):
        """Number of distinct asset identifiers reachable from this entry."""
        return len(self.asset_ids)


class LoopOccurrence(NamedTuple):
    """Diagnostic record of a link pointing back to an asset on the current path."""

    path: tuple
    target: NodeMixin

    @property
    def path_names(self):
        """Names of the nodes along :attr:`path`, including the looping link."""
        return path_names(self.path)


class InventoryReport:
    """
    Result of an :any:`AssetInventory` walk.

    Attributes:
        entry_count: Number of entries seen (an asset reachable via several
            paths counts once per path, including looping links).
        asset_count: Number of distinct real asset identifiers.
        subtrees: Per-entry-point :any:`SubtreeInventory` results.
        loops: All :any:`LoopOccurrence` diagnostics collected during the walk.
        entry_paths: Mapping of asset identifier to a tuple of entry paths,
            each path being a tuple of node names.
    """

    def __init__(self, entry_count, asset_ids, subtrees, loops, entry_paths):
        self.entry_count = entry_count
        self._asset_ids = asset_ids
        self.subtrees = tuple(subtrees)
        self.loops = tuple(loops)
        self.entry_paths = {asset_id: tuple(paths) for asset_id, paths in entry_paths.items()}

    @property
    def asset_count(self):
        """Number of distinct real asset identifiers across all entries."""
        return len(self._asset_ids)

    @property
    def asset_ids(self):
        """Sorted tuple of all distinct asset identifiers."""
        return tuple(sorted(self._asset_ids, key=_sort_key))

    @property
    def shared_assets(self):
        """
        Assets seen through more than one entry path.

        Returns tuples of ``(asset_id, entry_count)`` sorted by asset id.
        The gap between :attr:`entry_count` and :attr:`asset_count` is the sum
        of ``entry_count - 1`` over these assets.
        """
        counts = [(asset_id, len(paths)) for asset_id, paths in self.entry_paths.items()]
        return tuple(sorted((item for item in counts if item[1] > 1), key=lambda item: _sort_key(item[0])))

    def subtree_discrepancies(self):
        """
        Assets explaining why subtree subtotals exceed the global dedup count.

        For every asset reachable from more than one entry point, returns a
        ``(asset_id, subtree_ids)`` tuple, where ``subtree_ids`` are the asset
        identifiers of the entry points containing the asset::

            sum(subtree.asset_count for subtree in report.subtrees)
            - report.asset_count
            == sum(len(subtree_ids) - 1 for _, subtree_ids in discrepancies)
        """
        owners = {}
        for subtree in self.subtrees:
            owner_id = _identity_id(subtree.root)
            for asset_id in subtree.asset_ids:
                owners.setdefault(asset_id, []).append(owner_id)
        discrepancies = [(asset_id, tuple(owner_ids)) for asset_id, owner_ids in owners.items() if len(owner_ids) > 1]
        return tuple(sorted(discrepancies, key=lambda item: _sort_key(item[0])))

    def __repr__(self):
        return (
            f"InventoryReport(entry_count={self.entry_count}, asset_count={self.asset_count}, loops={len(self.loops)})"
        )


class AssetInventory:
    """
    Walk tree entries and count entries versus distinct real assets.

    Keyword Args:
        idattr (str): Node attribute holding the real asset identifier.
            Defaults to ``"name"``. :any:`SymlinkNode` attribute access is
            forwarded to its target, so the identifier of the linked asset is
            used automatically.

    Two entry points, the second one referencing an asset of the first::

        >>> from anytree import Node, SymlinkNode
        >>> themes = Node("themes")
        >>> asset_a = Node("assetA", parent=themes)
        >>> asset_b = Node("assetB", parent=asset_a)
        >>> other = Node("other")
        >>> link_a = SymlinkNode(asset_a, parent=other)

    The link is a second entry of ``assetA``, so the entry count exceeds the
    deduplicated asset count::

        >>> report = AssetInventory().analyze([themes, other])
        >>> report.entry_count
        5
        >>> report.asset_count
        4
        >>> report.shared_assets
        (('assetA', 2),)
        >>> report.entry_paths["assetA"]
        (('themes', 'assetA'), ('other', 'assetA'))

    Subtree subtotals count the shared asset once per entry point and therefore
    do not add up to the global count; the discrepancy is fully explained::

        >>> [(subtree.root.name, subtree.entry_count, subtree.asset_count)
        ...  for subtree in report.subtrees]
        [('themes', 3, 3), ('other', 2, 2)]
        >>> sum(subtree.asset_count for subtree in report.subtrees)
        5
        >>> report.subtree_discrepancies()
        (('assetA', ('themes', 'other')),)

    A link pointing back to an ancestor closes a loop. The looping path is kept
    as a diagnostic and expansion along that link stops, while the link entry
    itself still counts. A second, acyclic link to the same asset still adds an
    entry::

        >>> root = Node("root")
        >>> directory = Node("dir", parent=root)
        >>> loop_link = SymlinkNode(root, parent=directory)
        >>> alias = SymlinkNode(directory, parent=root)
        >>> cyclic = AssetInventory().analyze(root)
        >>> cyclic.entry_count
        4
        >>> cyclic.asset_count
        2
        >>> cyclic.shared_assets
        (('dir', 2), ('root', 2))
        >>> len(cyclic.loops)
        1
        >>> cyclic.loops[0].target is root
        True
        >>> cyclic.loops[0].path_names
        ('root', 'dir', 'root')
    """

    def __init__(self, idattr="name"):
        self.idattr = idattr

    def analyze(self, entries):
        """
        Analyze one entry node or an iterable of entry nodes.

        A single node is treated as the only entry point. Several nodes are
        treated as independent entry points (subtrees) of one dataset.

        Returns an :any:`InventoryReport`.
        """
        roots = (entries,) if isinstance(entries, NodeMixin) else tuple(entries)
        state = _WalkState(self.idattr)
        subtrees = [self._walk(root, state) for root in roots]
        return InventoryReport(
            entry_count=state.entry_count,
            asset_ids=state.asset_ids,
            subtrees=subtrees,
            loops=state.loops,
            entry_paths=state.entry_paths,
        )

    def _walk(self, root, state):
        local_ids = set()
        local_loops = []
        local_entries = 0
        active = []

        def visit(node, path):
            nonlocal local_entries
            current_path = (*path, node)
            state.entry_count += 1
            local_entries += 1

            resolved, loop_target = _resolve(node, active)
            asset_id = _asset_id(resolved, state.idattr)
            state.asset_ids.add(asset_id)
            local_ids.add(asset_id)
            state.entry_paths.setdefault(asset_id, []).append(path_names(current_path))

            if loop_target is not None:
                loop = LoopOccurrence(path=tuple(current_path), target=loop_target)
                state.loops.append(loop)
                local_loops.append(loop)
                return

            active.append(resolved)
            for child in node.children:
                visit(child, current_path)
            active.pop()

        visit(root, ())
        return SubtreeInventory(
            root=root,
            entry_count=local_entries,
            asset_ids=tuple(sorted(local_ids, key=_sort_key)),
            loops=tuple(local_loops),
        )


class _WalkState:
    """Accumulated counters shared by all entry points of one analysis."""

    def __init__(self, idattr):
        self.idattr = idattr
        self.entry_count = 0
        self.asset_ids = set()
        self.loops = []
        self.entry_paths = {}


def _resolve(node, active):
    """
    Return ``(resolved_identity, loop_target)`` for `node`.

    A plain node resolves to itself. A link resolves to its real target. If the
    target is one of the `active` assets on the current walk path, the second
    item is the asset the link points back to, signalling a loop.
    """
    if not isinstance(node, SymlinkNodeMixin):
        return node, None
    target = _follow_links(node)
    if isinstance(target, SymlinkNodeMixin) or any(target is ancestor for ancestor in active):
        return target, target
    return target, None


def _follow_links(link):
    """Follow a chain of links to the real target, stopping on self-references."""
    seen = set()
    node = link
    while isinstance(node, SymlinkNodeMixin):
        if id(node) in seen:
            return node
        seen.add(id(node))
        node = node.target
    return node


def _identity_id(node, idattr="name"):
    """Asset identifier of `node`, resolving links to their target first."""
    return _asset_id(_follow_links(node), idattr)


def _asset_id(node, idattr="name"):
    """Read the real asset identifier of `node`."""
    return getattr(node, idattr)


def path_names(path):
    """Return the names of the nodes in `path` as a tuple."""
    return tuple(getattr(node, "name", None) for node in path)


def _sort_key(value):
    """Stable ordering for identifiers of arbitrary, possibly mixed types."""
    return str(value)
