"""
Temporal tree dimension for slowly changing parent relations.

Unlike the regular :any:`anytree.NodeMixin` tree, where a node has exactly one
current parent, this module keeps the full history of parent relations as
half-open effective intervals ``[valid_from, valid_to)``. Records may arrive
late and in any order; business time, not insertion time, decides which parent
applies.

The motivating use case is a slowly changing dimension such as a product
category tree. In June the category "tea" moved from "food" to "beverages", but
an order placed in April still has to roll up to "food" when queried at the
April business date:

>>> from datetime import date
>>> from anytree import Node
>>> from anytree.temporal import TemporalDimension

>>> catalog = TemporalDimension()
>>> food = Node("food")
>>> beverages = Node("beverages")
>>> tea = Node("tea")
>>> catalog.add_edge(tea, food, date(2026, 1, 1), date(2026, 6, 1))

The June migration is a late-arriving record. Inserting it after the fact does
not change how historical business dates resolve:

>>> catalog.add_edge(tea, beverages, date(2026, 6, 1))
>>> [node.name for node in catalog.ancestors_at(tea, date(2026, 4, 15))]
['food']
>>> [node.name for node in catalog.path_at(tea, date(2026, 6, 2))]
['beverages', 'tea']
>>> catalog.parent_at(tea, date(2026, 5, 31)).name
'food'

Updates that are legal on their own can combine into a cycle on the dates where
their intervals overlap. :any:`TemporalDimension.validate` lists the dates that
cannot be published together with the nodes involved, while unaffected dates
remain queryable in their respective versions - see
:any:`TemporalDimension.validate`.
"""

from collections import namedtuple

from .node.exceptions import LoopError, TreeError
from .node.lightnodemixin import LightNodeMixin
from .node.nodemixin import NodeMixin

REASON_CYCLE = "cycle"
REASON_AMBIGUOUS_PARENT = "ambiguous parent"

TemporalEdge = namedtuple("TemporalEdge", ("child", "parent", "valid_from", "valid_to"))
TemporalFailure = namedtuple("TemporalFailure", ("valid_from", "valid_to", "reason", "nodes"))


class TemporalTreeError(TreeError):
    """Base error for temporal tree dimension operations."""


class TemporalLoopError(TemporalTreeError, LoopError):
    """Temporal tree contains a cycle at a given point in business time."""

    def __init__(self, message, nodes):
        super().__init__(message)
        self.nodes = tuple(nodes)


class TemporalAmbiguityError(TemporalTreeError):
    """Node has more than one effective parent at the same point in business time."""

    def __init__(self, message, nodes):
        super().__init__(message)
        self.nodes = tuple(nodes)


class TemporalPublishError(TemporalTreeError):
    """Publishing the temporal tree dimension failed for some points in business time."""

    def __init__(self, result):
        self.result = result
        super().__init__(f"Cannot publish temporal tree on {len(result.invalid_dates)} interval(s).")


class TemporalValidationResult:
    """
    Outcome of :any:`TemporalDimension.validate`.

    Attributes:
        valid: `True` when every point in business time resolves to a forest.
        failures: Tuple of :any:`TemporalFailure` records, ordered by business time.
        invalid_dates: Tuple of merged ``(valid_from, valid_to)`` ranges that cannot
            be published. An open-ended range reports `None` as `valid_to`.
        nodes: Nodes involved in any failure, in first-seen order.
    """

    def __init__(self, failures):
        self.failures = tuple(failures)
        self.valid = not self.failures
        self.invalid_dates = self._merge_invalid_dates(self.failures)
        nodes = []
        seen = set()
        for failure in self.failures:
            for node in failure.nodes:
                if id(node) not in seen:
                    seen.add(id(node))
                    nodes.append(node)
        self.nodes = tuple(nodes)

    @staticmethod
    def _merge_invalid_dates(failures):
        ranges = {(failure.valid_from, failure.valid_to) for failure in failures}
        ordered = sorted(ranges, key=lambda item: (item[0], item[1] is None, item[1]))
        merged = []
        for valid_from, valid_to in ordered:
            if merged and merged[-1][1] == valid_from:
                merged[-1][1] = valid_to
            else:
                merged.append([valid_from, valid_to])
        return tuple((valid_from, valid_to) for valid_from, valid_to in merged)

    def raise_if_invalid(self):
        """Raise :any:`TemporalPublishError` when the validation found failures."""
        if not self.valid:
            raise TemporalPublishError(self)

    def __bool__(self):
        return self.valid

    def __repr__(self):
        state = "valid" if self.valid else "invalid"
        return f"TemporalValidationResult({state}, invalid_dates={self.invalid_dates!r})"


class TemporalDimension:
    """
    Tree dimension with parent relations versioned by effective intervals.

    Business time is any comparable value, typically :any:`datetime.date`.
    Intervals are half-open: a relation is effective on `valid_from` and ceases
    to be effective on `valid_to`. Pass `valid_to=None` for an open-ended
    relation. Records may be added in any order, including late arrivals that
    describe historical periods.
    """

    def __init__(self):
        self._nodes = {}
        self._edges = []

    @property
    def nodes(self):
        """All registered nodes in registration order."""
        return tuple(self._nodes.values())

    @property
    def edges(self):
        """All registered :any:`TemporalEdge` records in insertion order."""
        return tuple(self._edges)

    def add_node(self, node):
        """
        Register a standalone root `node`.

        Nodes passed to :any:`add_edge` are registered automatically, so this
        call is only needed for nodes without any parent relation.

        >>> from anytree import Node
        >>> dim = TemporalDimension()
        >>> root = dim.add_node(Node("root"))
        >>> dim.nodes
        (Node('/root'),)
        """
        self._check_node(node, "Node")
        self._register(node)
        return node

    def add_edge(self, child, parent, valid_from, valid_to=None):
        """
        Record that `child` is below `parent` on the interval ``[valid_from, valid_to)``.

        `parent=None` marks an explicit detachment interval where the child is a
        root. Records are accepted in any order and may arrive late. Overlapping
        intervals are stored as given and reported by :any:`validate` instead of
        being rejected eagerly.

        Args:
            child: Node whose parent is described.
            parent: New parent node, or `None` for a detachment interval.
            valid_from: First point in business time where the relation is effective (inclusive).
            valid_to: First point where the relation stops being effective (exclusive),
                or `None` for an open-ended interval.

        >>> from datetime import date
        >>> from anytree import Node
        >>> dim = TemporalDimension()
        >>> root = Node("root")
        >>> moved = Node("moved")
        >>> other = Node("other")
        >>> # late arrival: historical interval inserted after the current one
        >>> dim.add_edge(moved, other, date(2026, 6, 1))
        >>> dim.add_edge(moved, root, date(2026, 1, 1), date(2026, 6, 1))
        >>> dim.parent_at(moved, date(2026, 3, 1)).name
        'root'
        >>> dim.parent_at(moved, date(2026, 7, 1)).name
        'other'
        """
        self._check_node(child, "Child node")
        if parent is not None:
            self._check_node(parent, "Parent node")
        if valid_to is not None and not valid_from < valid_to:
            msg = f"valid_from {valid_from!r} must be before valid_to {valid_to!r}."
            raise ValueError(msg)
        self._register(child)
        if parent is not None:
            self._register(parent)
        self._edges.append(TemporalEdge(child, parent, valid_from, valid_to))

    def edges_of(self, node):
        """
        Return all :any:`TemporalEdge` records describing the parent of `node`.

        >>> from datetime import date
        >>> from anytree import Node
        >>> dim = TemporalDimension()
        >>> leaf = Node("leaf")
        >>> dim.add_edge(leaf, Node("old"), date(2026, 1, 1), date(2026, 6, 1))
        >>> dim.add_edge(leaf, Node("new"), date(2026, 6, 1))
        >>> len(dim.edges_of(leaf))
        2
        """
        self._require_known(node)
        return tuple(edge for edge in self._edges if edge.child is node)

    def parent_at(self, node, point):
        """
        Return the effective parent of `node` at business time `point`.

        Returns `None` when no relation covers `point`, meaning the node is a
        root in that version.

        Raises:
            TemporalAmbiguityError: multiple parent relations cover `point`.
        """
        self._require_known(node)
        active = [edge for edge in self._edges if edge.child is node and self._covers(edge, point)]
        if len(active) > 1:
            nodes = self._involved_nodes(active)
            msg = f"Node {node!r} has {len(active)} effective parents at {point!r}."
            raise TemporalAmbiguityError(msg, nodes)
        return active[0].parent if active else None

    def ancestors_at(self, node, point):
        """
        Return the complete ancestor chain of `node` at business time `point`.

        The result runs from the root of that version down to the immediate
        parent of `node`. A node without an effective parent returns an empty
        tuple.

        Raises:
            TemporalLoopError: the effective relations form a cycle at `point`.
            TemporalAmbiguityError: some node has multiple effective parents at `point`.
        """
        self._require_known(node)
        chain = []
        positions = {}
        current = node
        while True:
            parent = self.parent_at(current, point)
            if parent is None:
                break
            if id(parent) in positions:
                cycle = tuple(chain[positions[id(parent)] :])
                msg = f"Effective parent relations form a cycle at {point!r}."
                raise TemporalLoopError(msg, cycle)
            positions[id(parent)] = len(chain)
            chain.append(parent)
            current = parent
        chain.reverse()
        return tuple(chain)

    def path_at(self, node, point):
        """
        Return the full path from the version root down to `node` at business time `point`.

        >>> from datetime import date
        >>> from anytree import Node
        >>> dim = TemporalDimension()
        >>> root = Node("root")
        >>> child = Node("child", parent=root)
        >>> dim.add_edge(child, root, date(2026, 1, 1))
        >>> [item.name for item in dim.path_at(child, date(2026, 1, 1))]
        ['root', 'child']
        """
        return (*self.ancestors_at(node, point), node)

    def roots_at(self, point):
        """
        Return all nodes that are roots at business time `point`.

        >>> from datetime import date
        >>> from anytree import Node
        >>> dim = TemporalDimension()
        >>> root = Node("root")
        >>> child = Node("child")
        >>> dim.add_node(root)
        Node('/root')
        >>> dim.add_edge(child, root, date(2026, 1, 1))
        >>> [item.name for item in dim.roots_at(date(2026, 1, 1))]
        ['root']
        """
        return tuple(node for node in self._nodes.values() if self.parent_at(node, point) is None)

    def is_valid_at(self, point):
        """
        Return whether the relations effective at `point` form an acyclic forest.

        >>> from datetime import date
        >>> from anytree import Node
        >>> dim = TemporalDimension()
        >>> a = Node("a")
        >>> b = Node("b")
        >>> dim.add_edge(a, b, date(2026, 1, 1), date(2026, 2, 1))
        >>> dim.add_edge(b, a, date(2026, 1, 1), date(2026, 2, 1))
        >>> dim.is_valid_at(date(2026, 1, 15))
        False
        >>> dim.is_valid_at(date(2026, 3, 1))
        True
        """
        active = {idx for idx, edge in enumerate(self._edges) if self._covers(edge, point)}
        return not self._segment_failures(sorted(active), point, point)

    def validate(self):
        """
        Validate every effective interval and return a :any:`TemporalValidationResult`.

        A sweep over all interval endpoints partitions business time into
        segments with a constant set of effective relations. Each segment is
        checked for ambiguous (overlapping) parent assignments and for cycles.
        Dates without any failure stay publishable and queryable even when other
        dates fail.

        Two updates that are fine on their own may close a cycle where their
        intervals overlap. Here ``a -> c`` only overlaps the ``c -> b`` and
        ``b -> a`` relations in January, so January cannot be published while
        February still resolves cleanly:

        >>> from datetime import date
        >>> from anytree import Node
        >>> dim = TemporalDimension()
        >>> a = Node("a")
        >>> b = Node("b")
        >>> c = Node("c")
        >>> dim.add_edge(c, b, date(2026, 1, 1), date(2026, 3, 1))
        >>> dim.add_edge(b, a, date(2026, 1, 1))
        >>> dim.add_edge(a, c, date(2026, 1, 1), date(2026, 2, 1))
        >>> result = dim.validate()
        >>> result.valid
        False
        >>> result.invalid_dates
        ((datetime.date(2026, 1, 1), datetime.date(2026, 2, 1)),)
        >>> [node.name for node in result.nodes]
        ['c', 'b', 'a']
        >>> [node.name for node in dim.path_at(c, date(2026, 2, 15))]
        ['a', 'b', 'c']

        Querying an affected date raises instead of returning a broken chain:

        >>> dim.ancestors_at(c, date(2026, 1, 10))
        Traceback (most recent call last):
            ...
        anytree.temporal.TemporalLoopError: Effective parent relations form a cycle at datetime.date(2026, 1, 10).
        """
        events = {}
        for idx, edge in enumerate(self._edges):
            events.setdefault(edge.valid_from, []).append((idx, True))
            if edge.valid_to is not None:
                events.setdefault(edge.valid_to, []).append((idx, False))
        active = set()
        failures = []
        endpoints = sorted(events)
        for position, point in enumerate(endpoints):
            for idx, entering in events[point]:
                if entering:
                    active.add(idx)
                else:
                    active.discard(idx)
            if not active:
                continue
            segment_end = endpoints[position + 1] if position + 1 < len(endpoints) else None
            failures.extend(self._segment_failures(sorted(active), point, segment_end))
        failures.sort(key=lambda failure: (failure.valid_from, failure.reason, [repr(node) for node in failure.nodes]))
        return TemporalValidationResult(failures)

    def publish(self):
        """
        Validate the dimension and return the result when every date can be published.

        Raises:
            TemporalPublishError: at least one interval fails validation. The
                exception carries the full :any:`TemporalValidationResult`.
        """
        result = self.validate()
        result.raise_if_invalid()
        return result

    def _segment_failures(self, active, segment_start, segment_end):
        failures = []
        by_child = {}
        for idx in active:
            edge = self._edges[idx]
            by_child.setdefault(id(edge.child), []).append(edge)
        parent_map = {}
        for edges in by_child.values():
            child = edges[0].child
            if len(edges) > 1:
                failure = TemporalFailure(
                    segment_start,
                    segment_end,
                    REASON_AMBIGUOUS_PARENT,
                    tuple(self._involved_nodes(edges)),
                )
                failures.append(failure)
            else:
                parent_map[child] = edges[0].parent
        failures.extend(self._cycle_failures(parent_map, segment_start, segment_end))
        return failures

    @staticmethod
    def _cycle_failures(parent_map, segment_start, segment_end):
        failures = []
        gray = set()
        black = set()
        for start in parent_map:
            if start in gray or start in black:
                continue
            path = []
            positions = {}
            node = start
            while node is not None and node not in gray and node not in black:
                gray.add(node)
                positions[node] = len(path)
                path.append(node)
                node = parent_map.get(node)
            if node is not None and node in gray:
                cycle = tuple(path[positions[node] :])
                failures.append(TemporalFailure(segment_start, segment_end, REASON_CYCLE, cycle))
            gray.clear()
            black.update(path)
        return failures

    @staticmethod
    def _involved_nodes(edges):
        nodes = []
        seen = set()
        for edge in edges:
            for node in (edge.child, edge.parent):
                if node is not None and id(node) not in seen:
                    seen.add(id(node))
                    nodes.append(node)
        return nodes

    def _register(self, node):
        if id(node) not in self._nodes:
            self._nodes[id(node)] = node

    def _require_known(self, node):
        if id(node) not in self._nodes:
            msg = f"Node {node!r} is not part of this temporal dimension."
            raise TemporalTreeError(msg)

    @staticmethod
    def _check_node(node, label):
        if not isinstance(node, (NodeMixin, LightNodeMixin)):
            msg = f"{label} {node!r} is not of type 'NodeMixin'."
            raise TemporalTreeError(msg)

    @staticmethod
    def _covers(edge, point):
        if point < edge.valid_from:
            return False
        return edge.valid_to is None or point < edge.valid_to
