"""
Resolve legacy import records against a tree.

A legacy system only stored a few *ordered fragments* of the
organizational path of every record. Segments may be missing anywhere,
even in the middle, and department names may occur more than once in
the new tree. A record therefore often matches several nodes.

On top of the fragments, a few relations between legacy keys may be
known, for example that two records were located below the same
parent. Such relations can rule out combinations which look valid for
every single record, so the involved records have to be resolved
*jointly*.

The :any:`LegacyResolver` computes, for one import batch:

* which records have a unique placement (:any:`Resolution.resolved`),
* which records have to be considered jointly and which mappings are
  still feasible (:any:`Resolution.ambiguous`),
* which records contradict each other, i.e. no joint mapping exists
  (:any:`Resolution.conflicts`),
* which records do not match any node at all (:any:`Resolution.unmatched`).
"""

from .iterators.preorderiter import PreOrderIter

_MIN_CONSTRAINT_KEYS = 2

__all__ = [
    "ConflictGroup",
    "JointGroup",
    "LegacyRecord",
    "LegacyResolveError",
    "LegacyResolver",
    "RecordConstraint",
    "Resolution",
    "SameNode",
    "SameParent",
]


class LegacyResolveError(RuntimeError):
    """Legacy Resolve Error."""


class LegacyRecord:
    """
    A single legacy record.

    Args:
        key: Unique legacy key (e.g. the old employee id).
        fragments: Ordered path fragments known for this record.
            The fragments have to occur, in the given order, on the
            path from the root to the placement node. Segments may be
            missing anywhere. The last fragment is the name of the
            placement node itself.

    >>> LegacyRecord("E1", ("corp", "sales"))
    LegacyRecord('E1', ('corp', 'sales'))
    >>> LegacyRecord("E1", [])
    Traceback (most recent call last):
      ...
    anytree.legacyresolve.LegacyResolveError: LegacyRecord 'E1' has no fragments.
    """

    def __init__(self, key, fragments):
        fragments = tuple(fragments)
        if not fragments:
            msg = f"LegacyRecord {key!r} has no fragments."
            raise LegacyResolveError(msg)
        self.key = key
        self.fragments = fragments

    def __repr__(self):
        return f"LegacyRecord({self.key!r}, {self.fragments!r})"


class RecordConstraint:
    """
    Relation between the placements of several legacy records.

    Args:
        keys: Two or more legacy keys this constraint applies to.
    """

    relation = "constraint"

    def __init__(self, *keys):
        if len(keys) == 1 and not isinstance(keys[0], str):
            try:
                keys = tuple(keys[0])
            except TypeError:
                pass
        if len(keys) < _MIN_CONSTRAINT_KEYS:
            msg = f"{type(self).__name__} requires at least two keys."
            raise LegacyResolveError(msg)
        self.keys = tuple(keys)

    def consistent(self, mapping):
        """
        Return `True` if the (partial) `mapping` does not violate this constraint.

        Only keys present in `mapping` are considered, so the method can
        be used on partial mappings while searching.
        """
        raise NotImplementedError  # pragma: no cover

    def __repr__(self):
        return f"{type(self).__name__}{self.keys!r}"


class SameNode(RecordConstraint):
    """The records have to be placed on the very same node."""

    relation = "same node"

    def consistent(self, mapping):
        """Return `True` if all mapped records are placed on the same node."""
        nodes = [mapping[key] for key in self.keys if key in mapping]
        return all(node is nodes[0] for node in nodes[1:])


class SameParent(RecordConstraint):
    """The placements of the records have to share the same parent node."""

    relation = "same parent"

    def consistent(self, mapping):
        """Return `True` if the placements of all mapped records share their parent."""
        parents = [mapping[key].parent for key in self.keys if key in mapping]
        return all(parent is parents[0] for parent in parents[1:])


class JointGroup:
    """
    Records which have to be considered jointly.

    Attributes:
        keys: Legacy keys of the involved records.
        placements: All still feasible joint mappings (legacy key to node).
        truncated: `True` if not all feasible mappings are listed because
            `max_solutions` was exceeded.
    """

    def __init__(self, keys, placements, truncated=False):
        self.keys = tuple(keys)
        self.placements = list(placements)
        self.truncated = truncated

    @property
    def fixed(self):
        """Records mapped to the same node in every feasible placement."""
        fixed = {}
        for key in self.keys:
            node = self.placements[0][key]
            if all(placement[key] is node for placement in self.placements[1:]):
                fixed[key] = node
        return fixed

    @property
    def open(self):
        """Records with differing nodes across the feasible placements."""
        fixed = self.fixed
        return tuple(key for key in self.keys if key not in fixed)

    def __repr__(self):
        return f"JointGroup(keys={self.keys!r}, placements={len(self.placements)})"


class ConflictGroup:
    """
    Records whose constraints cannot be satisfied jointly.

    Attributes:
        keys: Legacy keys of the involved records.
        candidates: The mutually contradictory placement options per
            legacy key. No combination of them satisfies all constraints.
    """

    def __init__(self, keys, candidates):
        self.keys = tuple(keys)
        self.candidates = {key: tuple(candidates[key]) for key in self.keys}

    def __repr__(self):
        return f"ConflictGroup(keys={self.keys!r})"


class Resolution:
    """
    Result of :any:`LegacyResolver.resolve`.

    Attributes:
        resolved: Legacy key to node for every record with a unique
            placement. This includes records of a :any:`JointGroup` which
            are mapped to the same node in every feasible placement.
        ambiguous: List of :any:`JointGroup` with more than one feasible
            joint mapping.
        conflicts: List of :any:`ConflictGroup` without any feasible
            joint mapping.
        unmatched: Legacy key to :any:`LegacyRecord` for records without
            any candidate node.
    """

    def __init__(self, resolved, ambiguous, conflicts, unmatched, idattr):
        self.resolved = resolved
        self.ambiguous = ambiguous
        self.conflicts = conflicts
        self.unmatched = unmatched
        self.idattr = idattr

    @property
    def complete(self):
        """`True` if every record has a unique placement."""
        return not self.ambiguous and not self.conflicts and not self.unmatched

    def node_ids(self, mapping):
        """Return the stable node ids of a legacy key to node mapping."""
        return {key: getattr(node, self.idattr) for key, node in mapping.items()}

    @property
    def resolved_ids(self):
        """Stable node ids of the uniquely placed records."""
        return self.node_ids(self.resolved)

    def __repr__(self):
        return (
            f"Resolution(resolved={len(self.resolved)}, ambiguous={len(self.ambiguous)},"
            f" conflicts={len(self.conflicts)}, unmatched={len(self.unmatched)})"
        )


class LegacyResolver:
    """
    Resolve :any:`LegacyRecord` instances against a tree.

    Args:
        root: Root node of the tree to resolve against.
        pathattr (str): Name of the node attribute matched against the fragments.
        idattr (str): Name of the node attribute holding the stable id.

    Example:

    >>> from anytree import Node
    >>> corp = Node("corp", id=1)
    >>> north = Node("north", parent=corp, id=2)
    >>> south = Node("south", parent=corp, id=3)
    >>> sales_north = Node("sales", parent=north, id=4)
    >>> sales_south = Node("sales", parent=south, id=5)

    The department name `sales` exists twice. A record only knowing
    `corp` and `sales` has two candidate placements:

    >>> resolver = LegacyResolver(corp)
    >>> resolver.candidates(("corp", "sales"))
    [Node('/corp/north/sales', id=4), Node('/corp/south/sales', id=5)]

    Two such records are known to have been below the same parent.
    The constraint does not fully disambiguate them, both feasible
    joint mappings are reported:

    >>> alice = LegacyRecord("E1", ("corp", "sales"))
    >>> bob = LegacyRecord("E2", ("sales",))
    >>> resolution = resolver.resolve([alice, bob], constraints=[SameParent("E1", "E2")])
    >>> resolution.complete
    False
    >>> group = resolution.ambiguous[0]
    >>> group.keys
    ('E1', 'E2')
    >>> group.placements
    [{'E1': Node('/corp/north/sales', id=4), 'E2': Node('/corp/north/sales', id=4)}, \
{'E1': Node('/corp/south/sales', id=5), 'E2': Node('/corp/south/sales', id=5)}]

    A record with more fragments may still resolve uniquely:

    >>> carol = LegacyRecord("E3", ("corp", "north", "sales"))
    >>> resolution = resolver.resolve([alice, bob, carol], constraints=[SameParent("E1", "E2")])
    >>> resolution.resolved
    {'E3': Node('/corp/north/sales', id=4)}
    >>> resolution.resolved_ids
    {'E3': 4}
    """

    def __init__(self, root, pathattr="name", idattr="id"):
        super().__init__()
        self.root = root
        self.pathattr = pathattr
        self.idattr = idattr

    def candidates(self, fragments):
        """
        Return all nodes matching the ordered `fragments`.

        A node matches if the fragments occur, in the given order, on the
        path from the root to the node and the last fragment is the name
        of the node itself. Segments may be missing anywhere.
        """
        fragments = tuple(fragments)
        if not fragments:
            msg = "fragments must not be empty."
            raise LegacyResolveError(msg)
        matches = []
        for node in PreOrderIter(self.root):
            names = [str(getattr(item, self.pathattr, None)) for item in node.path]
            if names[-1] != str(fragments[-1]):
                continue
            if _is_subsequence(fragments[:-1], names[:-1]):
                matches.append(node)
        return matches

    def resolve(self, records, constraints=(), max_solutions=256):
        """
        Resolve `records` under the given `constraints`.

        Args:
            records: Iterable of :any:`LegacyRecord`.
            constraints: Iterable of :any:`RecordConstraint` relating
                legacy keys.
            max_solutions (int): Maximum number of joint mappings listed
                per :any:`JointGroup`.

        Returns:
            Resolution
        """
        bykey, constraints = self.__checked(records, constraints)
        constraints_by_key = {}
        for constraint in constraints:
            for key in constraint.keys:
                constraints_by_key.setdefault(key, []).append(constraint)

        resolution = Resolution({}, [], [], {}, self.idattr)
        for keys in self.__components(bykey, constraints):
            self.__resolve_component(keys, bykey, constraints_by_key, max_solutions, resolution)
        return resolution

    @staticmethod
    def __checked(records, constraints):
        bykey = {}
        for record in records:
            if not isinstance(record, LegacyRecord):
                msg = f"record {record!r} is not a LegacyRecord."
                raise LegacyResolveError(msg)
            if record.key in bykey:
                msg = f"duplicate legacy key {record.key!r}."
                raise LegacyResolveError(msg)
            bykey[record.key] = record
        constraints = list(constraints)
        for constraint in constraints:
            if not isinstance(constraint, RecordConstraint):
                msg = f"constraint {constraint!r} is not a RecordConstraint."
                raise LegacyResolveError(msg)
            for key in constraint.keys:
                if key not in bykey:
                    msg = f"constraint {constraint!r} references unknown legacy key {key!r}."
                    raise LegacyResolveError(msg)
        return bykey, constraints

    def __resolve_component(self, keys, bykey, constraints_by_key, max_solutions, resolution):
        candidates = {key: self.candidates(bykey[key].fragments) for key in keys}
        if len(keys) == 1 and not candidates[keys[0]]:
            resolution.unmatched[keys[0]] = bykey[keys[0]]
            return
        constraints = list(dict.fromkeys(constraint for key in keys for constraint in constraints_by_key.get(key, [])))
        placements = self.__placements(keys, candidates, constraints, max_solutions)
        if not placements:
            resolution.conflicts.append(ConflictGroup(keys, candidates))
        elif len(placements) == 1:
            resolution.resolved.update(placements[0])
        else:
            truncated = len(placements) > max_solutions
            group = JointGroup(keys, placements[:max_solutions], truncated=truncated)
            resolution.ambiguous.append(group)
            if not truncated:
                resolution.resolved.update(group.fixed)

    @staticmethod
    def __components(bykey, constraints):
        parent = {key: key for key in bykey}

        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for constraint in constraints:
            first, *rest = constraint.keys
            for key in rest:
                parent[find(key)] = find(first)

        components = {}
        for key in bykey:
            components.setdefault(find(key), []).append(key)
        return list(components.values())

    @staticmethod
    def __placements(keys, candidates, constraints, limit):
        placements = []
        constraints_by_key = {}
        for constraint in constraints:
            for key in constraint.keys:
                constraints_by_key.setdefault(key, []).append(constraint)
        assignment = {}

        def backtrack(index):
            if len(placements) > limit:
                return
            if index == len(keys):
                placements.append(dict(assignment))
                return
            key = keys[index]
            for node in candidates[key]:
                assignment[key] = node
                if all(constraint.consistent(assignment) for constraint in constraints_by_key.get(key, [])):
                    backtrack(index + 1)
                del assignment[key]

        backtrack(0)
        return placements


def _is_subsequence(fragments, names):
    """Return `True` if `fragments` occur in order within `names`."""
    if not fragments:
        return True
    index = 0
    for name in names:
        if name == str(fragments[index]):
            index += 1
            if index == len(fragments):
                return True
    return False
