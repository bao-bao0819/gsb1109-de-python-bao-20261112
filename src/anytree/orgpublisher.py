"""Publishing of organizational dimension tables as ordered trees.

A reorganization receives a batch of *final* parent-child relations and
sibling ordering constraints addressed by stable department identifiers.
The batch is only published when it forms one complete, ordered tree.

The publishing process:

1. Validate that the final relations describe exactly one connected tree
   (one root, no unknown parents, no cycles).
2. Detect contradictory explicit ordering constraints and report every
   conflicting entry via :any:`OrderConflictError`.
3. Order the children of every department so that all explicit constraints
   hold while inverting as few old sibling relationships as possible.
   Remaining ties are settled by the stable department identifier.
4. Submit the same batch twice, using two different operation sequences,
   and verify that both resulting trees are identical before handing the
   publication to queryers and exporters.

Example:

>>> from anytree import RenderTree
>>> from anytree.orgpublisher import publish_org_tree
>>> relations = {
...     "root": None,
...     "sales": "root",
...     "eng": "root",
...     "ui": "eng",
...     "api": "eng",
... }
>>> old = {
...     "root": None,
...     "sales": "root",
...     "eng": "root",
...     "ui": "eng",
...     "api": "eng",
... }
>>> publication = publish_org_tree(relations, [("eng", "sales")], old=old)
>>> print(RenderTree(publication.root))
AnyNode(id='root')
├── AnyNode(id='eng')
│   ├── AnyNode(id='ui')
│   └── AnyNode(id='api')
└── AnyNode(id='sales')

The two internally submitted arrangements are guaranteed to be identical:

>>> publication.verify()
True
"""

import json

from .iterators import PreOrderIter
from .node import AnyNode, LightNodeMixin, NodeMixin
from .node.exceptions import TreeError


class OrgStructureError(TreeError):
    """The final relations do not describe one complete tree."""


class OrderConflictError(OrgStructureError):
    """The explicit sibling ordering constraints cannot all be satisfied.

    Attributes:
        conflicts: Tuple of conflicting ``(before, after)`` constraint entries
            in the order they were submitted.
    """

    def __init__(self, conflicts, details=()):
        self.conflicts = tuple(conflicts)
        self.details = tuple(details)
        rendered = ", ".join(f"{before!r} -> {after!r}" for before, after in self.conflicts)
        message = f"Conflicting ordering constraints: [{rendered}]"
        if self.details:
            message = f"{message} ({'; '.join(self.details)})"
        super().__init__(message)


class Publication:
    """A published organization tree.

    Two independent arrangements of the same batch are kept side by side.
    Queryers read :attr:`root` while exporters read the very same structure.
    :meth:`verify` proves that both submission sequences produced identical
    organizations.

    Args:
        root: Root node of the primary arrangement.
        secondary_root: Root node of the arrangement built differently.
        id_attr: Attribute holding the stable department identifier.
    """

    def __init__(self, root, secondary_root, id_attr):
        self._root = root
        self._secondary_root = secondary_root
        self._id_attr = id_attr
        self._nodes = {getattr(node, id_attr): node for node in PreOrderIter(root)}

    @property
    def root(self):
        """Root node served to queryers."""
        return self._root

    @property
    def secondary_root(self):
        """Root node of the second submission of the same batch."""
        return self._secondary_root

    @property
    def nodes(self):
        """Mapping of stable department identifier to primary node."""
        return dict(self._nodes)

    @property
    def id_attr(self):
        """Attribute holding the stable department identifier."""
        return self._id_attr

    def __iter__(self):
        return iter(PreOrderIter(self._root))

    def __len__(self):
        return len(self._nodes)

    def _signature(self, root):
        signature = []
        for node in PreOrderIter(root):
            child_ids = tuple(getattr(child, self._id_attr) for child in node.children)
            signature.append((getattr(node, self._id_attr), child_ids))
        return tuple(signature)

    def verify(self):
        """Return `True` when both submissions expose the same organization."""
        return self._signature(self._root) == self._signature(self._secondary_root)

    def export(self, exporter):
        """Export the primary arrangement with a custom `exporter`."""
        return exporter.export(self._root)

    def to_dict(self):
        """Return the organization as a plain, recursively ordered dictionary."""
        return self._to_dict(self._root)

    def _to_dict(self, node):
        data = {
            key: value
            for key, value in node.__dict__.items()
            if key
            not in (
                "_NodeMixin__children",
                "_NodeMixin__parent",
                "_LightNodeMixin__children",
                "_LightNodeMixin__parent",
            )
        }
        children = [self._to_dict(child) for child in node.children]
        if children:
            data["children"] = children
        return data

    def to_json(self, **kwargs):
        """Return the organization as a JSON string."""
        params = {"indent": 2}
        params.update(kwargs)
        return json.dumps(self.to_dict(), **params)

    def write(self, filehandle, **kwargs):
        """Write the organization as JSON to `filehandle`."""
        params = {"indent": 2}
        params.update(kwargs)
        return json.dump(self.to_dict(), filehandle, **params)


def publish_org_tree(relations, ordering=(), *, old=None, nodecls=AnyNode, id_attr="id"):
    """Publish a batch of final relations and ordering constraints.

    Args:
        relations: Final parent-child relations, either a mapping of
            department identifier to parent identifier (``None`` for the
            root) or an iterable of ``(identifier, parent)`` pairs.
        ordering: Iterable of explicit ``(before, after)`` constraints
            requiring `before` to appear before `after` among siblings.
        old: Previous organization used to preserve old sibling order.
            May be a tree node, a relation mapping/pairs iterable, or `None`.
        nodecls: Node class used to build the published tree.
        id_attr: Attribute name storing the stable department identifier.

    Returns:
        Publication: The verified, ordered organization.

    Raises:
        OrgStructureError: The relations do not form one complete tree or
            reference unknown departments.
        OrderConflictError: The explicit ordering constraints contradict
            each other or cannot be expressed as sibling orders.
    """
    relation_map = _coerce_relations(relations)
    constraints = _coerce_constraints(ordering)
    _validate_relations(relation_map)
    _validate_constraint_ids(relation_map, constraints)
    conflicts, details = _find_conflicts(relation_map, constraints)
    if conflicts:
        raise OrderConflictError(conflicts, details)
    old_info = _old_sibling_info(old, id_attr)
    child_orders = _order_all_children(relation_map, constraints, old_info)
    primary = _build_arrangement(relation_map, child_orders, nodecls, id_attr, top_down=True)
    secondary = _build_arrangement(relation_map, child_orders, nodecls, id_attr, top_down=False)
    publication = Publication(primary, secondary, id_attr)
    if not publication.verify():  # pragma: no cover - defensive guard
        msg = "The two submissions of the batch produced different organizations."
        raise OrgStructureError(msg)
    return publication


def _coerce_relations(relations):
    if isinstance(relations, dict):
        return dict(relations)
    relation_map = {}
    for identifier, parent in relations:
        if identifier in relation_map:
            msg = f"Duplicate final relation for department {identifier!r}."
            raise OrgStructureError(msg)
        relation_map[identifier] = parent
    return relation_map


def _coerce_constraints(ordering):
    return tuple((before, after) for before, after in ordering)


def _validate_relations(relation_map):
    if not relation_map:
        msg = "Cannot publish an empty organization: no departments were supplied."
        raise OrgStructureError(msg)
    unknown_parents = sorted(
        {str(parent) for parent in relation_map.values() if parent is not None and parent not in relation_map}
    )
    if unknown_parents:
        rendered = ", ".join(unknown_parents)
        msg = f"Relations reference unknown parent departments: {rendered}."
        raise OrgStructureError(msg)
    cyclic = _cyclic_departments(relation_map)
    if cyclic:
        rendered = ", ".join(repr(identifier) for identifier in cyclic)
        msg = f"Relations contain a cycle involving departments: {rendered}."
        raise OrgStructureError(msg)
    roots = [identifier for identifier, parent in relation_map.items() if parent is None]
    if len(roots) != 1:
        rendered = ", ".join(repr(identifier) for identifier in roots)
        msg = f"An organization must have exactly one root, found {len(roots)}: {rendered}."
        raise OrgStructureError(msg)


def _cyclic_departments(relation_map):
    state = {}
    cyclic = []

    def visit(identifier):
        status = state.get(identifier)
        if status == "active":
            cyclic.append(identifier)
            return True
        if status == "done":
            return False
        state[identifier] = "active"
        parent = relation_map[identifier]
        if parent is not None and visit(parent):
            cyclic.append(identifier)
            return True
        state[identifier] = "done"
        return False

    for identifier in relation_map:
        visit(identifier)
    seen = set()
    return [identifier for identifier in cyclic if not (identifier in seen or seen.add(identifier))]


def _validate_constraint_ids(relation_map, constraints):
    known = set(relation_map)
    unknown = []
    for before, after in constraints:
        for identifier in (before, after):
            if identifier not in known and identifier not in unknown:
                unknown.append(identifier)
    if unknown:
        rendered = ", ".join(repr(identifier) for identifier in unknown)
        msg = f"Ordering constraints reference unknown departments: {rendered}."
        raise OrgStructureError(msg)


def _reachability(vertices, edges):
    reach = {vertex: set() for vertex in vertices}
    for source, target in edges:
        reach[source].add(target)
    pending = True
    while pending:
        pending = False
        for source in vertices:
            additions = set()
            for target in reach[source]:
                additions.update(reach[target])
            undiscovered = additions - reach[source]
            if undiscovered:
                reach[source].update(undiscovered)
                pending = True
    return reach


def _find_conflicts(relation_map, constraints):
    edges = []
    seen = set()
    for before, after in constraints:
        if (before, after) not in seen:
            seen.add((before, after))
            edges.append((before, after))
    vertices = set(relation_map)
    reach = _reachability(vertices, edges)
    conflicts = []
    cyclic_pairs = []
    cross_family_pairs = []
    for before, after in edges:
        reasons = []
        if before == after or before in reach[after]:
            reasons.append("cycle")
            cyclic_pairs.append((before, after))
        if relation_map[before] != relation_map[after]:
            reasons.append("not siblings")
            cross_family_pairs.append((before, after))
        if reasons and (before, after) not in conflicts:
            conflicts.append((before, after))
    details = []
    if cyclic_pairs:
        rendered = ", ".join(f"{before!r}->{after!r}" for before, after in cyclic_pairs)
        details.append(f"cyclic constraints: {rendered}")
    if cross_family_pairs:
        rendered = ", ".join(f"{before!r}->{after!r}" for before, after in cross_family_pairs)
        details.append(f"constraints between non-siblings: {rendered}")
    return conflicts, details


def _old_sibling_info(old, id_attr):
    """Map each old department id to ``(old_parent_id, sibling_index)``."""
    if old is None:
        return {}
    if isinstance(old, (NodeMixin, LightNodeMixin)):
        info = {}
        for node in PreOrderIter(old.root):
            for index, child in enumerate(node.children):
                info[getattr(child, id_attr)] = (getattr(node, id_attr), index)
        return info
    relation_map = _coerce_relations(old)
    _validate_relations(relation_map)
    insertion = {identifier: index for index, identifier in enumerate(relation_map)}
    grouped = {}
    for identifier, parent in relation_map.items():
        grouped.setdefault(parent, []).append(identifier)
    info = {}
    for parent, members in grouped.items():
        for index, identifier in enumerate(sorted(members, key=lambda member: insertion[member])):
            info[identifier] = (parent, index)
    return info


def _order_all_children(relation_map, constraints, old_info):
    grouped = {}
    for before, after in constraints:
        grouped.setdefault(relation_map[before], []).append((before, after))
    child_orders = {}
    for parent in relation_map:
        children = [identifier for identifier, owner in relation_map.items() if owner == parent]
        relevant_edges = grouped.get(parent, [])
        child_orders[parent] = _order_siblings(children, relevant_edges, old_info)
    return child_orders


def _prefer_old_order(left, right, old_info):
    """Return `True` when `left` should be picked before `right`."""
    left_info = old_info.get(left)
    right_info = old_info.get(right)
    if left_info is not None and right_info is not None and left_info[0] == right_info[0]:
        return left_info[1] < right_info[1]
    return left < right


def _order_siblings(children, edges, old_info):
    indegree = dict.fromkeys(children, 0)
    adjacency = {identifier: [] for identifier in children}
    seen = set()
    for before, after in edges:
        if (before, after) in seen:
            continue
        seen.add((before, after))
        adjacency[before].append(after)
        indegree[after] += 1
    available = sorted(identifier for identifier in children if indegree[identifier] == 0)
    ordered = []
    while available:
        chosen_index = 0
        for index in range(1, len(available)):
            if _prefer_old_order(available[index], available[chosen_index], old_info):
                chosen_index = index
        chosen = available.pop(chosen_index)
        ordered.append(chosen)
        for follower in adjacency[chosen]:
            indegree[follower] -= 1
            if indegree[follower] == 0:
                _insort(available, follower)
    return ordered


def _insort(ordered, value):
    for index, existing in enumerate(ordered):
        if value < existing:
            ordered.insert(index, value)
            return
    ordered.append(value)


def _preorder_sequence(relation_map, child_orders):
    root = next(identifier for identifier, parent in relation_map.items() if parent is None)
    sequence = []
    pending = [root]
    while pending:
        parent = pending.pop(0)
        sequence.append(parent)
        pending[0:0] = child_orders[parent]
    return sequence


def _build_arrangement(relation_map, child_orders, nodecls, id_attr, *, top_down):
    nodes = {identifier: nodecls(**{id_attr: identifier}) for identifier in relation_map}
    sequence = _preorder_sequence(relation_map, child_orders)
    root_id = sequence[0]
    if top_down:
        for parent in sequence:
            nodes[parent].children = [nodes[child] for child in child_orders[parent]]
    else:
        # Move every subtree out first, then re-attach bottom-up in the
        # final sibling order, mirroring detach/re-attach reorganizations.
        for parent in reversed(sequence):
            for child in child_orders[parent]:
                nodes[child].parent = None
        for parent in reversed(sequence):
            for child in child_orders[parent]:
                nodes[child].parent = nodes[parent]
    return nodes[root_id]
