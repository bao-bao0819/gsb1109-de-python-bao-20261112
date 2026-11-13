"""Reconciliation of reports published against the same region tree.

Two teams publish reports on the same tree. Each report row refers to one tree
node and claims to cover the full leaf set of that node (a county is a leaf in
the region tree). Rows are atomic: a row can only be assigned to a reconciliation
group as a whole, it can never be split.

The finest common grouping is obtained from the leaf partition induced by both
reports: two leaves land in the same block exactly when they are covered by the
same row in *each* report. Every block that is covered by a reliable row on both
sides is one directly reconcilable common group.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .iterators import PreOrderIter
from .node.nodemixin import NodeMixin

MISSING = "missing"
OVERLAP = "overlap"
INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class ReportLine:
    """One report row.

    `row_id` identifies the row, `node` is the tree node it reports on, `value`
    is the reported figure and `complete=False` means the row admits it does not
    fully cover the leaf set of `node`.
    """

    row_id: str
    node: NodeMixin
    value: float = 0.0
    complete: bool = True


@dataclass(frozen=True)
class CoverageIssue:
    """A coverage defect of one report.

    `kind` is one of :data:`MISSING`, :data:`OVERLAP` or :data:`INCOMPLETE`,
    `leaves` are the affected leaf nodes and `row_ids` are the rows involved
    (empty for missing leaves, one id for an incomplete row, several ids for an
    overlap).
    """

    report: str
    kind: str
    leaves: Tuple[NodeMixin, ...]
    row_ids: Tuple[str, ...]


@dataclass(frozen=True)
class CommonGroup:
    """One directly reconcilable common group.

    `leaves` are the counties in the group, `node` is the tree node whose leaf
    set equals the group (``None`` when the group is a fragment of a node) and
    `left_rows` / `right_rows` are the source rows that sum into the group on
    each side. `left_value` / `right_value` are the values of those rows. When
    one atomic row spans several groups its value is repeated in each of them,
    so group values are not additive across groups; the row composition always
    is.
    """

    key: Tuple[str, ...]
    node: Optional[NodeMixin]
    leaves: FrozenSet[NodeMixin]
    left_rows: Tuple[ReportLine, ...]
    right_rows: Tuple[ReportLine, ...]
    left_value: float
    right_value: float


@dataclass(frozen=True)
class OneSidedBlock:
    """Leaves covered reliably by only one report, so they cannot be reconciled."""

    leaves: FrozenSet[NodeMixin]
    side: str
    rows: Tuple[ReportLine, ...]


@dataclass(frozen=True)
class ReconciliationResult:
    """Result of :func:`reconcile_reports`."""

    groups: Tuple[CommonGroup, ...]
    issues: Tuple[CoverageIssue, ...]
    one_sided: Tuple[OneSidedBlock, ...]
    uncovered: FrozenSet[NodeMixin]


def _leaf_index(scope: NodeMixin) -> Dict[NodeMixin, int]:
    return {node: index for index, node in enumerate(PreOrderIter(scope.root))}


def _sorted_leaves(leaves: FrozenSet[NodeMixin], index: Dict[NodeMixin, int]) -> Tuple[NodeMixin, ...]:
    return tuple(sorted(leaves, key=lambda leaf: index[leaf]))


def _name(node: NodeMixin) -> str:
    return node.name  # type: ignore[attr-defined]


def _check_scope(scope: NodeMixin, lines: List[ReportLine], report: str) -> None:
    scope_leaves = frozenset(scope.leaves)
    for line in lines:
        if line.node.root is not scope.root:
            raise ValueError(f"{report}: row {line.row_id!r} does not belong to the tree of {scope!r}")
        line_leaves = frozenset(line.node.leaves)
        if not line_leaves <= scope_leaves:
            outside = _sorted_leaves(line_leaves - scope_leaves, _leaf_index(scope))
            names = ", ".join(_name(node) for node in outside)
            raise ValueError(f"{report}: row {line.row_id!r} covers leaves outside the scope: {names}")


def validate_report(
    scope: NodeMixin, lines: List[ReportLine], report: str = "report"
) -> Tuple[Dict[NodeMixin, List[ReportLine]], List[CoverageIssue]]:
    """Validate one report against `scope`.

    Returns the leaf-to-rows coverage map and the detected issues. Leaves with
    no covering row are reported as :data:`MISSING`, leaves covered by more than
    one row as :data:`OVERLAP` and rows with ``complete=False`` as
    :data:`INCOMPLETE`.

    >>> from anytree import Node
    >>> root = Node("country")
    >>> province = Node("province", parent=root)
    >>> county_a = Node("county_a", parent=province)
    >>> county_b = Node("county_b", parent=province)
    >>> lines = [
    ...     ReportLine("r1", county_a),
    ...     ReportLine("r2", county_a),
    ...     ReportLine("r3", province, complete=False),
    ... ]
    >>> _coverage, issues = validate_report(province, lines, "team-x")
    >>> sorted(issue.kind for issue in issues)
    ['incomplete', 'overlap']
    >>> next(issue.leaves for issue in issues if issue.kind == "overlap")
    (Node('/country/province/county_a'),)
    """
    _check_scope(scope, lines, report)
    seen_ids: Set[str] = set()
    for line in lines:
        if line.row_id in seen_ids:
            raise ValueError(f"{report}: duplicate row id {line.row_id!r}")
        seen_ids.add(line.row_id)
    index = _leaf_index(scope)
    scope_leaves = frozenset(scope.leaves)
    covering: Dict[NodeMixin, List[ReportLine]] = {}
    issues: List[CoverageIssue] = []

    for line in lines:
        if not line.complete:
            issues.append(
                CoverageIssue(
                    report,
                    INCOMPLETE,
                    _sorted_leaves(frozenset(line.node.leaves), index),
                    (line.row_id,),
                )
            )
        for leaf in line.node.leaves:
            covering.setdefault(leaf, []).append(line)

    missing = scope_leaves - frozenset(covering)
    if missing:
        issues.append(CoverageIssue(report, MISSING, _sorted_leaves(missing, index), ()))

    overlap_clusters: Dict[Tuple[str, ...], Set[NodeMixin]] = {}
    for leaf, rows in covering.items():
        if len(rows) > 1:
            key = tuple(sorted(row.row_id for row in rows))
            overlap_clusters.setdefault(key, set()).add(leaf)
    for row_ids, leaves in overlap_clusters.items():
        issues.append(CoverageIssue(report, OVERLAP, _sorted_leaves(frozenset(leaves), index), row_ids))

    return covering, issues


def _reliable_rows(covering: Dict[NodeMixin, List[ReportLine]]) -> Dict[NodeMixin, ReportLine]:
    reliable = {}
    for leaf, rows in covering.items():
        if len(rows) == 1 and rows[0].complete:
            reliable[leaf] = rows[0]
    return reliable


def _group_node(leaves: FrozenSet[NodeMixin]) -> Optional[NodeMixin]:
    common = set.intersection(*(set(leaf.path) for leaf in leaves))
    ancestor = max(common, key=lambda node: node.depth)
    return ancestor if frozenset(ancestor.leaves) == leaves else None


def _group_key(node: Optional[NodeMixin], leaves: FrozenSet[NodeMixin], index: Dict[NodeMixin, int]) -> Tuple[str, ...]:
    if node is not None:
        return tuple(_name(part) for part in node.path)
    fragment = tuple(_name(node) for node in _sorted_leaves(leaves, index))
    return ("<fragment>", *fragment)


def reconcile_reports(
    scope: NodeMixin,
    left: List[ReportLine],
    right: List[ReportLine],
    left_name: str = "left",
    right_name: str = "right",
) -> ReconciliationResult:
    """Reconcile two reports covering the same `scope`.

    The finest common grouping is the leaf partition where two leaves share a
    block only when the same left row and the same right row cover them. Each
    block covered reliably on both sides becomes a :class:`CommonGroup`.

    >>> from anytree import Node
    >>> country = Node("country")
    >>> province = Node("province", parent=country)
    >>> city_a = Node("city_a", parent=province)
    >>> county_a1 = Node("county_a1", parent=city_a)
    >>> county_a2 = Node("county_a2", parent=city_a)
    >>> city_b = Node("city_b", parent=province)
    >>> county_b1 = Node("county_b1", parent=city_b)
    >>> county_b2 = Node("county_b2", parent=city_b)
    >>> left_rows = [ReportLine("L-province", province, value=300)]
    >>> right_rows = [
    ...     ReportLine("R-city_a", city_a, value=120),
    ...     ReportLine("R-city_b", city_b, value=180),
    ... ]
    >>> result = reconcile_reports(province, left_rows, right_rows)
    >>> [group.node.name for group in result.groups]
    ['city_a', 'city_b']
    >>> [(group.left_rows[0].row_id, group.right_rows[0].row_id) for group in result.groups]
    [('L-province', 'R-city_a'), ('L-province', 'R-city_b')]
    >>> result.issues
    ()
    """
    left_covering, left_issues = validate_report(scope, left, left_name)
    right_covering, right_issues = validate_report(scope, right, right_name)
    index = _leaf_index(scope)

    reliable_left = _reliable_rows(left_covering)
    reliable_right = _reliable_rows(right_covering)

    blocks: Dict[Tuple[Optional[str], Optional[str]], Set[NodeMixin]] = {}
    for leaf in scope.leaves:
        left_row = reliable_left.get(leaf)
        right_row = reliable_right.get(leaf)
        signature = (
            left_row.row_id if left_row is not None else None,
            right_row.row_id if right_row is not None else None,
        )
        blocks.setdefault(signature, set()).add(leaf)

    groups: List[CommonGroup] = []
    one_sided: List[OneSidedBlock] = []
    uncovered: Set[NodeMixin] = set()

    for signature, block_leaves in blocks.items():
        leaves = frozenset(block_leaves)
        left_id, right_id = signature
        left_rows = _rows_for(leaves, reliable_left)
        right_rows = _rows_for(leaves, reliable_right)
        if left_id is not None and right_id is not None:
            node = _group_node(leaves)
            groups.append(
                CommonGroup(
                    key=_group_key(node, leaves, index),
                    node=node,
                    leaves=leaves,
                    left_rows=left_rows,
                    right_rows=right_rows,
                    left_value=sum(row.value for row in left_rows),
                    right_value=sum(row.value for row in right_rows),
                )
            )
        elif left_id is not None or right_id is not None:
            one_sided.append(
                OneSidedBlock(
                    leaves=leaves,
                    side=left_name if left_id is not None else right_name,
                    rows=left_rows if left_id is not None else right_rows,
                )
            )
        else:
            uncovered.update(leaves)

    groups.sort(key=lambda group: min(index[leaf] for leaf in group.leaves))
    one_sided.sort(key=lambda block: min(index[leaf] for leaf in block.leaves))
    return ReconciliationResult(
        groups=tuple(groups),
        issues=tuple(left_issues + right_issues),
        one_sided=tuple(one_sided),
        uncovered=frozenset(uncovered),
    )


def _rows_for(leaves: FrozenSet[NodeMixin], reliable: Dict[NodeMixin, ReportLine]) -> Tuple[ReportLine, ...]:
    rows: Dict[str, ReportLine] = {}
    for leaf in leaves:
        row = reliable.get(leaf)
        if row is not None:
            rows[row.row_id] = row
    return tuple(rows.values())
