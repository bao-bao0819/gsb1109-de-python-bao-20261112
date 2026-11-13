"""Asset catalog subscription diffing."""

from typing import NamedTuple, Tuple

from .iterators import PreOrderIter


class Subscription:
    """
    Subscription condition matching catalog entries by hierarchy and path.

    Args:
        path: catalog path of the subscribed subtree root,
              either as string (e.g. ``"/root/assets"``) or as tuple of names
              (e.g. ``("root", "assets")``).
              The default matches the entire catalog.

    Keyword Args:
        maxlevel: maximum level relative to the subscribed subtree root.
                  ``1`` subscribes the root node only, ``2`` includes its children, and so on.
                  `None` subscribes the entire subtree.
        separator: path separator used to parse a string `path`.

    >>> sub = Subscription("/root/assets", maxlevel=2)
    >>> sub
    Subscription('/root/assets', maxlevel=2)
    >>> sub.matches(("root", "assets"))
    True
    >>> sub.matches(("root", "assets", "photos"))
    True
    >>> sub.matches(("root", "assets", "photos", "cat"))
    False
    >>> sub.matches(("root", "inbox"))
    False

    A subscription without path matches every entry:

    >>> Subscription().matches(("anything", "at", "all"))
    True
    """

    def __init__(self, path=(), maxlevel=None, separator="/"):
        if isinstance(path, str):
            path = tuple(part for part in path.split(separator) if part)
        self.path = tuple(path)
        self.maxlevel = maxlevel
        self.separator = separator

    def __repr__(self):
        path = self.separator + self.separator.join(self.path)
        if self.maxlevel is None:
            return f"{type(self).__name__}({path!r})"
        return f"{type(self).__name__}({path!r}, maxlevel={self.maxlevel!r})"

    def __eq__(self, other):
        if isinstance(other, Subscription):
            return self.path == other.path and self.maxlevel == other.maxlevel
        return NotImplemented

    def __hash__(self):
        return hash((self.path, self.maxlevel))

    def matches(self, names):
        """
        Check whether a catalog entry is subscribed.

        Args:
            names: tuple of node names from the catalog root to the entry,
                   e.g. ``("root", "assets", "photos")``.

        Returns:
            True if the entry is within the subscribed subtree and level range.
        """
        base = self.path
        if tuple(names[: len(base)]) != base:
            return False
        return self.maxlevel is None or len(names) - len(base) < self.maxlevel


class AssetEntry(NamedTuple):
    """Catalog entry of an asset: stable `id` and display `path`."""

    id: str
    path: str


class PathChange(NamedTuple):
    """Display path change of an asset: stable `id`, `old_path` and `new_path`."""

    id: str
    old_path: str
    new_path: str


class CatalogDiff(NamedTuple):
    """
    Minimal asset change list between two catalog versions.

    Attributes:
        added: tuple of :any:`AssetEntry` which newly entered the subscription.
        removed: tuple of :any:`AssetEntry` which left the subscription.
        moved: tuple of :any:`PathChange` of assets which are still subscribed,
               but whose display path changed.
    """

    added: Tuple[AssetEntry, ...]
    removed: Tuple[AssetEntry, ...]
    moved: Tuple[PathChange, ...]

    @property
    def empty(self):
        """Return True if there are no changes at all."""
        return not (self.added or self.removed or self.moved)


def catalogdiff(old_root, new_root, subscription=None, idattr="id", separator="/"):
    """
    Compare two versions of an asset catalog and derive the minimal change list for a subscription.

    Both catalog versions are compared as snapshots, keyed by the stable asset id
    (see `idattr`). Intermediate publish steps do not matter: moving a subtree first
    and renaming a node afterwards notifies exactly like publishing the final tree directly.

    Args:
        old_root: root node of the previous catalog version.
        new_root: root node of the new catalog version.

    Keyword Args:
        subscription: :any:`Subscription` condition matching entries by hierarchy and path.
                      `None` subscribes the entire catalog.
        idattr: name of the node attribute holding the stable asset id.
                Nodes without this attribute are not assets themselves,
                but still contribute to the display path of their descendants.
        separator: path separator used to render display paths.

    Returns:
        :any:`CatalogDiff` with the minimal change list:

        * `added`: assets which newly entered the subscription.
        * `removed`: assets which left the subscription, e.g. children of a subtree
          which was moved out of the subscribed area, even if their own fields are unchanged.
        * `moved`: assets which are still subscribed, but whose display path changed.

        An asset visible through multiple catalog entries (e.g. via :any:`SymlinkNode`)
        is reported by its first entry in tree order. It remains subscribed as long as
        any of its entries stays visible.

    Example:

    >>> from anytree import Node, RenderTree
    >>> old = Node("root", id="root")
    >>> assets = Node("assets", parent=old, id="assets")
    >>> photos = Node("photos", parent=assets, id="photos")
    >>> cat = Node("cat", parent=photos, id="cat")
    >>> dog = Node("dog", parent=photos, id="dog")
    >>> inbox = Node("inbox", parent=old, id="inbox")
    >>> print(RenderTree(old))
    Node('/root', id='root')
    ├── Node('/root/assets', id='assets')
    │   └── Node('/root/assets/photos', id='photos')
    │       ├── Node('/root/assets/photos/cat', id='cat')
    │       └── Node('/root/assets/photos/dog', id='dog')
    └── Node('/root/inbox', id='inbox')

    The new version moves the `photos` subtree to `inbox` and renames `cat` to `kitty`:

    >>> new = Node("root", id="root")
    >>> assets = Node("assets", parent=new, id="assets")
    >>> inbox = Node("inbox", parent=new, id="inbox")
    >>> photos = Node("photos", parent=inbox, id="photos")
    >>> kitty = Node("kitty", parent=photos, id="cat")
    >>> dog = Node("dog", parent=photos, id="dog")

    Subscribing the entire catalog reports display path changes only:

    >>> diff = catalogdiff(old, new)
    >>> diff.added
    ()
    >>> diff.removed
    ()
    >>> for change in diff.moved:
    ...     print(change)
    PathChange(id='photos', old_path='/root/assets/photos', new_path='/root/inbox/photos')
    PathChange(id='cat', old_path='/root/assets/photos/cat', new_path='/root/inbox/photos/kitty')
    PathChange(id='dog', old_path='/root/assets/photos/dog', new_path='/root/inbox/photos/dog')

    Subscribing `/root/assets` reports the moved out subtree as removed,
    including all children whose own fields did not change:

    >>> diff = catalogdiff(old, new, Subscription("/root/assets"))
    >>> for entry in diff.removed:
    ...     print(entry)
    AssetEntry(id='photos', path='/root/assets/photos')
    AssetEntry(id='cat', path='/root/assets/photos/cat')
    AssetEntry(id='dog', path='/root/assets/photos/dog')
    """
    old = _snapshot(old_root, subscription, idattr, separator)
    new = _snapshot(new_root, subscription, idattr, separator)
    added = tuple(AssetEntry(asset_id, path) for asset_id, path in new.items() if asset_id not in old)
    removed = tuple(AssetEntry(asset_id, path) for asset_id, path in old.items() if asset_id not in new)
    moved = tuple(
        PathChange(asset_id, old[asset_id], path)
        for asset_id, path in new.items()
        if asset_id in old and old[asset_id] != path
    )
    return CatalogDiff(added, removed, moved)


def _snapshot(root, subscription, idattr, separator):
    """Map asset id to its first subscribed display path in tree order."""
    entries = {}
    for node in PreOrderIter(root):
        asset_id = getattr(node, idattr, None)
        if asset_id is None:
            continue
        names = tuple(n.name for n in node.path)
        if subscription is not None and not subscription.matches(names):
            continue
        if asset_id not in entries:
            entries[asset_id] = separator + separator.join(names)
    return entries
