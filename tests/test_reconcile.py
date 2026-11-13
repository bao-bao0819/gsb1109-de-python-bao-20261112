"""Tests for report reconciliation."""

import pytest

from anytree import Node
from anytree.reconcile import (
    INCOMPLETE,
    MISSING,
    OVERLAP,
    CoverageIssue,
    ReportLine,
    reconcile_reports,
    validate_report,
)

from .helper import eq_


def _build_region_tree():
    country = Node("country")
    province_a = Node("province_a", parent=country)
    city_a = Node("city_a", parent=province_a)
    county_a1 = Node("county_a1", parent=city_a)
    county_a2 = Node("county_a2", parent=city_a)
    city_b = Node("city_b", parent=province_a)
    county_b1 = Node("county_b1", parent=city_b)
    county_b2 = Node("county_b2", parent=city_b)
    county_x = Node("county_x", parent=province_a)  # separately listed county
    province_b = Node("province_b", parent=country)
    city_c = Node("city_c", parent=province_b)
    county_c1 = Node("county_c1", parent=city_c)
    county_c2 = Node("county_c2", parent=city_c)
    city_d = Node("city_d", parent=province_b)
    county_d1 = Node("county_d1", parent=city_d)
    return country, province_a, province_b


def test_finest_common_grouping_mixed_granularity():
    """Province-level rows meet city and separately-listed-county rows."""
    country, province_a, province_b = _build_region_tree()
    left = [
        ReportLine("L-province_a", province_a, value=500),
        ReportLine("L-province_b", province_b, value=300),
    ]
    right = [
        ReportLine("R-city_a", province_a.children[0], value=200),
        ReportLine("R-city_b", province_a.children[1], value=250),
        ReportLine("R-county_x", province_a.children[2], value=50),
        ReportLine("R-province_b", province_b, value=300),
    ]

    result = reconcile_reports(country, left, right)

    eq_(result.issues, ())
    eq_([group.node.name for group in result.groups], ["city_a", "city_b", "county_x", "province_b"])
    eq_(
        [tuple(row.row_id for row in group.left_rows) for group in result.groups],
        [("L-province_a",), ("L-province_a",), ("L-province_a",), ("L-province_b",)],
    )
    eq_(
        [tuple(row.row_id for row in group.right_rows) for group in result.groups],
        [("R-city_a",), ("R-city_b",), ("R-county_x",), ("R-province_b",)],
    )
    # The province row is atomic and contributes its full value to every group it spans.
    eq_([group.left_value for group in result.groups], [500, 500, 500, 300])
    eq_([group.right_value for group in result.groups], [200, 250, 50, 300])
    eq_(result.one_sided, ())
    eq_(result.uncovered, frozenset())


def test_missing_leaves_are_listed_and_become_one_sided():
    _, province_a, _ = _build_region_tree()
    county_x = province_a.children[2]
    left = [
        ReportLine("L-city_a", province_a.children[0]),
        ReportLine("L-city_b", province_a.children[1]),
    ]
    right = [ReportLine("R-province_a", province_a)]

    result = reconcile_reports(province_a, left, right, left_name="team-a", right_name="team-b")

    (issue,) = result.issues
    eq_(isinstance(issue, CoverageIssue), True)
    eq_(issue.report, "team-a")
    eq_(issue.kind, MISSING)
    eq_(issue.leaves, (county_x,))
    eq_(issue.row_ids, ())

    eq_([group.node.name for group in result.groups], ["city_a", "city_b"])
    (block,) = result.one_sided
    eq_(block.side, "team-b")
    eq_(block.leaves, frozenset((county_x,)))
    eq_(tuple(row.row_id for row in block.rows), ("R-province_a",))


def test_overlap_reports_affected_leaves_and_rows():
    _, province_a, _ = _build_region_tree()
    city_a = province_a.children[0]
    county_x = province_a.children[2]
    left = [ReportLine("L-province_a", province_a)]
    right = [
        ReportLine("R-province_a", province_a),
        ReportLine("R-city_a", city_a),
    ]

    result = reconcile_reports(province_a, left, right, left_name="team-a", right_name="team-b")

    overlap_issues = [issue for issue in result.issues if issue.kind == OVERLAP]
    eq_(len(overlap_issues), 1)
    issue = overlap_issues[0]
    eq_(issue.report, "team-b")
    eq_(issue.leaves, city_a.leaves)
    eq_(issue.row_ids, ("R-city_a", "R-province_a"))

    # Overlapped leaves are not trustworthy on the right, so they stay one-sided.
    eq_(frozenset().union(*(block.leaves for block in result.one_sided)), frozenset(city_a.leaves))

    # The same pair of atomic rows covers both city_b and county_x, so they form
    # one fragment group instead of being split.
    eq_(len(result.groups), 1)
    group = result.groups[0]
    eq_(group.node, None)
    eq_(group.leaves, frozenset(province_a.children[1].leaves) | frozenset((county_x,)))
    eq_(tuple(row.row_id for row in group.left_rows), ("L-province_a",))
    eq_(tuple(row.row_id for row in group.right_rows), ("R-province_a",))


def test_incomplete_row_flags_full_leaf_range():
    _, province_a, _ = _build_region_tree()
    left = [ReportLine("L-province_a", province_a, complete=False)]
    right = [
        ReportLine("R-city_a", province_a.children[0]),
        ReportLine("R-city_b", province_a.children[1]),
        ReportLine("R-county_x", province_a.children[2]),
    ]

    result = reconcile_reports(province_a, left, right)

    (issue,) = result.issues
    eq_(issue.kind, INCOMPLETE)
    eq_(issue.leaves, province_a.leaves)
    eq_(issue.row_ids, ("L-province_a",))
    # No leaf has a reliable row on both sides.
    eq_(result.groups, ())
    eq_(len(result.one_sided), 3)


def test_uncovered_when_both_sides_unreliable():
    _, province_a, _ = _build_region_tree()
    county_x = province_a.children[2]
    left = [
        ReportLine("L-city_a", province_a.children[0]),
        ReportLine("L-city_b", province_a.children[1]),
    ]
    right = [
        ReportLine("R-city_a", province_a.children[0]),
        ReportLine("R-city_b", province_a.children[1]),
    ]

    result = reconcile_reports(province_a, left, right)

    eq_(result.uncovered, frozenset((county_x,)))
    missing_reports = {issue.report for issue in result.issues if issue.kind == MISSING}
    eq_(missing_reports, {"left", "right"})


def test_fragment_group_when_part_of_node_survives():
    country = Node("country")
    city = Node("city", parent=country)
    county_1 = Node("county_1", parent=city)
    county_2 = Node("county_2", parent=city)
    county_3 = Node("county_3", parent=city)

    left = [ReportLine("L-city", city)]
    right = [
        ReportLine("R-county_1", county_1),
        ReportLine("R-city", city),  # overlaps county_1, reliably covers county_2 and county_3
    ]

    result = reconcile_reports(city, left, right)

    (overlap,) = [issue for issue in result.issues if issue.kind == OVERLAP]
    eq_(overlap.leaves, (county_1,))
    (group,) = result.groups
    eq_(group.node, None)
    eq_(group.leaves, frozenset((county_2, county_3)))
    eq_(group.key, ("<fragment>", "county_2", "county_3"))
    eq_(tuple(row.row_id for row in group.left_rows), ("L-city",))
    eq_(tuple(row.row_id for row in group.right_rows), ("R-city",))


def test_row_outside_scope_is_rejected():
    _, province_a, province_b = _build_region_tree()
    lines = [ReportLine("L-province_b", province_b)]
    with pytest.raises(ValueError, match="outside the scope"):
        validate_report(province_a, lines, "team-a")


def test_row_from_other_tree_is_rejected():
    root_a = Node("root_a")
    root_b = Node("root_b")
    node_b = Node("node_b", parent=root_b)
    with pytest.raises(ValueError, match="does not belong to the tree"):
        validate_report(root_a, [ReportLine("foreign", node_b)], "team-a")


def test_duplicate_row_id_is_rejected():
    _, province_a, _ = _build_region_tree()
    lines = [
        ReportLine("dup", province_a.children[0]),
        ReportLine("dup", province_a.children[1]),
    ]
    with pytest.raises(ValueError, match="duplicate row id"):
        validate_report(province_a, lines, "team-a")
