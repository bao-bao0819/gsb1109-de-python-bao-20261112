# Tests for multi-row column header extraction (xlrd.header).

import sys
from unittest import TestCase

import xlrd
from xlrd import Book
from xlrd.biffh import XL_CELL_BLANK, XL_CELL_EMPTY, XL_CELL_TEXT, XLRDError
from xlrd.header import (
    REGION_INDEPENDENT_DUPLICATE,
    REGION_MERGED,
    ROLE_LABEL,
    ROLE_MERGED_SUBORDINATE,
    ROLE_VACANT,
    build_column_header_map,
    cell_address,
)
from xlrd.sheet import Colinfo, Sheet

from .helpers import from_sample


def make_sheet():
    """
    Build a sheet in memory with the exact shape of the two-sales-table
    problem, including every blank-cell variant.

    Row 0: 商品(A1:A2 merged) | 期初(B1:C1 merged) | 期末(D1:E1 merged)
           | F truly empty     | 数量(G) 数量(H) independent
    Row 1: (A merged blank)    | B=数量 C=金额       | D=数量 E=金额
    Column B is hidden.
    """
    book = Book()
    book.logfile = sys.stdout
    book.verbosity = 0
    book.ragged_rows = False
    book.formatting_info = True
    book.biff_version = 80
    book._sheet_visibility = [0]
    sheet = Sheet(book, 0, 'sales', 0)

    def text(rowx, colx, value):
        sheet.put_cell(rowx, colx, XL_CELL_TEXT, value, 0)

    text(0, 0, '商品')
    text(0, 1, '期初')
    text(0, 3, '期末')
    text(1, 1, '数量')
    text(1, 2, '金额')
    text(1, 3, '数量')
    text(1, 4, '金额')
    text(0, 6, '数量')
    text(0, 7, '数量')
    # Subordinate cell of the vertical 商品 merge: formatted blank.
    sheet.put_cell(1, 0, XL_CELL_BLANK, '', 0)
    # Column F header cells are never written: truly empty.
    sheet.put_cell(0, 5, XL_CELL_EMPTY, '', 0)
    sheet.put_cell(1, 5, XL_CELL_EMPTY, '', 0)
    sheet.merged_cells = [
        (0, 2, 0, 1),   # 商品 A1:A2
        (0, 1, 1, 3),   # 期初 B1:C1
        (0, 1, 3, 5),   # 期末 D1:E1
    ]
    sheet.tidy_dimensions()
    hidden = Colinfo()
    hidden.hidden = 1
    sheet.colinfo_map[1] = hidden
    return sheet


class TestColumnPaths(TestCase):

    def setUp(self):
        self.sheet = make_sheet()
        self.header_map = self.sheet.column_header_map(2)

    def test_paths_and_origins(self):
        by_col = dict(
            (col.colx, col) for col in self.header_map.columns
        )
        self.assertEqual(by_col[1].path, ('期初', '数量'))
        self.assertEqual(by_col[1].path_origins, ((0, 1), (1, 1)))
        self.assertEqual(by_col[1].path_addresses, ('B1', 'B2'))
        self.assertEqual(by_col[3].path, ('期末', '数量'))
        self.assertEqual(by_col[3].path_origins, ((0, 3), (1, 3)))
        self.assertEqual(by_col[3].path_addresses, ('D1', 'D2'))
        # A vertically merged label contributes one path component only.
        self.assertEqual(by_col[0].path, ('商品',))
        self.assertEqual(by_col[0].path_origins, ((0, 0),))
        self.assertEqual(by_col[0].path_addresses, ('A1',))

    def test_merged_parent_is_one_shared_label(self):
        by_col = dict(
            (col.colx, col) for col in self.header_map.columns
        )
        opening = [by_col[1], by_col[2]]
        # Both columns under 期初 must reference the same anchor label.
        self.assertEqual(
            opening[0].path_origins[0], opening[1].path_origins[0]
        )
        self.assertEqual(opening[0].levels[0].label,
                         opening[1].levels[0].label)
        self.assertEqual(
            opening[1].levels[0].role, ROLE_MERGED_SUBORDINATE
        )
        self.assertEqual(
            opening[1].levels[0].merged_range, (0, 1, 1, 3)
        )

    def test_truly_empty_vs_blank_vs_label_are_distinct_roles(self):
        by_col = dict(
            (col.colx, col) for col in self.header_map.columns
        )
        # Column F: truly empty (XL_CELL_EMPTY).
        vacant = by_col[5]
        self.assertEqual(vacant.path, ())
        self.assertTrue(
            all(level.role == ROLE_VACANT for level in vacant.levels)
        )
        self.assertEqual(
            [level.vacancy_kind for level in vacant.levels],
            ['empty', 'empty'],
        )
        # A2 is the subordinate of a merge: blank cell, merged role, but
        # never confused with a truly empty cell or an independent label.
        subordinate = by_col[0].levels[1]
        self.assertEqual(subordinate.role, ROLE_MERGED_SUBORDINATE)
        self.assertEqual(subordinate.merged_range, (0, 2, 0, 1))
        # G1/H1 are independent labels (even though text is identical).
        g_label = by_col[6].levels[0]
        h_label = by_col[7].levels[0]
        self.assertEqual(g_label.role, ROLE_LABEL)
        self.assertEqual(h_label.role, ROLE_LABEL)
        self.assertEqual(g_label.label.value, h_label.label.value)
        self.assertNotEqual(g_label.label.origin, h_label.label.origin)
        self.assertEqual(g_label.label.merged_range, None)

    def test_explicit_header_row_range(self):
        header_map = self.sheet.column_header_map((1, 2))
        by_col = dict((col.colx, col) for col in header_map.columns)
        self.assertEqual(by_col[1].path, ('数量',))
        self.assertEqual(by_col[1].path_origins, ((1, 1),))

    def test_invalid_header_rows(self):
        self.assertRaises(ValueError, self.sheet.column_header_map, 0)
        self.assertRaises(
            ValueError, self.sheet.column_header_map, (2, 1)
        )
        self.assertRaises(
            ValueError, self.sheet.column_header_map, (1, 2, 3)
        )

    def test_column_paths_helper(self):
        rows = list(self.sheet.column_paths(2))
        self.assertEqual(rows[1], (
            1, ('期初', '数量'), ((0, 1), (1, 1)),
        ))


class TestReverseLookup(TestCase):

    def setUp(self):
        self.sheet = make_sheet()
        self.header_map = self.sheet.column_header_map(2)

    def test_unique_path_locates_one_column_with_coordinates(self):
        match = self.header_map.locate(('期初', '数量'))
        self.assertTrue(match.unique)
        self.assertEqual(match.column.colx, 1)
        # Consumer can jump straight back to the source cells.
        rowx, colx = match.column.leaf_origin
        self.assertEqual((rowx, colx), (1, 1))
        self.assertEqual(self.sheet.cell_value(rowx, colx), '数量')
        self.assertEqual(cell_address(rowx, colx), 'B2')

    def test_same_named_leaf_does_not_collapse_to_first_hit(self):
        match = self.header_map.locate(('数量',))
        # Two independent single-level 数量 headers (G, H); full paths
        # disambiguate, and the bare name returns every hit.
        self.assertFalse(match.unique)
        self.assertEqual(
            sorted(col.colx for col in match.columns), [6, 7]
        )
        self.assertEqual(match.region.kind, REGION_INDEPENDENT_DUPLICATE)
        self.assertEqual(match.region.range, (0, 1, 6, 8))
        self.assertEqual(match.region.origins, ((0, 6), (0, 7)))
        self.assertRaises(XLRDError, lambda: match.column)

    def test_full_paths_disambiguate_same_named_branches(self):
        opening = self.header_map.locate(('期初', '数量'))
        closing = self.header_map.locate(('期末', '数量'))
        self.assertEqual(opening.column.colx, 1)
        self.assertEqual(closing.column.colx, 3)
        self.assertNotEqual(
            opening.column.path_origins, closing.column.path_origins
        )

    def test_merged_parent_ambiguity_reports_intact_range(self):
        match = self.header_map.locate(('期初',), prefix=True)
        self.assertEqual(
            sorted(col.colx for col in match.columns), [1, 2]
        )
        self.assertEqual(match.region.kind, REGION_MERGED)
        # The full B1:C1 merged range, never split into two parents.
        self.assertEqual(match.region.range, (0, 1, 1, 3))
        self.assertEqual(match.region.level_index, 0)

    def test_ambiguities_lists_each_fanout(self):
        queries = dict(
            (item.query, item.region) for item in self.header_map.ambiguities()
        )
        self.assertIn(('期初',), queries)
        self.assertIn(('期末',), queries)
        self.assertEqual(queries[('期初',)].kind, REGION_MERGED)
        self.assertEqual(queries[('期末',)].range, (0, 1, 3, 5))
        # Same-text independent cells are reported as a separate ambiguity.
        self.assertIn(('数量',), queries)
        self.assertEqual(
            queries[('数量',)].kind, REGION_INDEPENDENT_DUPLICATE
        )
        self.assertEqual(queries[('数量',)].range, (0, 1, 6, 8))

    def test_sheet_level_lookup_helper(self):
        match = self.sheet.locate_column_path(('期末', '金额'), 2)
        self.assertTrue(match.unique)
        self.assertEqual(match.column.colx, 4)
        missing = self.sheet.locate_column_path(('不存在',), 2)
        self.assertEqual(missing.columns, [])
        self.assertFalse(missing.unique)


class TestProjection(TestCase):

    def setUp(self):
        self.sheet = make_sheet()
        self.header_map = self.sheet.column_header_map(2)

    def test_hiding_column_keeps_remaining_path_in_its_group(self):
        projected = self.header_map.projection(include_hidden=False)
        self.assertEqual(
            [col.colx for col in projected.columns],
            [0, 2, 3, 4, 5, 6, 7],
        )
        # The 期初-数量 column (B) is gone ...
        self.assertEqual(
            projected.locate(('期初', '数量')).columns, []
        )
        # ... and the surviving 数量 must still resolve to 期末, not be
        # re-attached to the opening group.
        surviving = projected.locate(('期末', '数量'))
        self.assertTrue(surviving.unique)
        self.assertEqual(surviving.column.colx, 3)
        self.assertEqual(surviving.column.path_origins, ((0, 3), (1, 3)))

    def test_projection_keeps_original_coordinates(self):
        projected = self.header_map.projection(
            include_hidden=False, columns=[2, 3]
        )
        colx = [col.colx for col in projected.columns]
        self.assertEqual(colx, [2, 3])
        for column in projected.columns:
            original = self.header_map.column_at(column.colx)
            self.assertEqual(
                column.path_origins, original.path_origins
            )

    def test_build_excluding_hidden(self):
        header_map = build_column_header_map(
            self.sheet, 2, include_hidden=False
        )
        self.assertNotIn(1, [col.colx for col in header_map.columns])
        # The hidden flag is still visible on a full build.
        full = build_column_header_map(self.sheet, 2)
        self.assertTrue(full.column_at(1).hidden)


class TestRealWorkbook(TestCase):
    """End-to-end checks against a genuine BIFF8 .xls fixture."""

    def setUp(self):
        book = xlrd.open_workbook(
            from_sample('header_groups.xls'), formatting_info=True
        )
        self.sheet = book.sheet_by_name('sales')
        self.header_map = self.sheet.column_header_map(2)

    def test_fixture_roundtrip(self):
        by_col = dict(
            (col.colx, col) for col in self.header_map.columns
        )
        self.assertEqual(by_col[1].path, ('期初', '数量'))
        self.assertEqual(by_col[1].path_addresses, ('B1', 'B2'))
        self.assertEqual(by_col[3].path, ('期末', '数量'))
        self.assertEqual(by_col[3].path_addresses, ('D1', 'D2'))
        # F is truly empty in the real file; A2/C1/E1 are merged blanks.
        self.assertEqual(
            [level.vacancy_kind for level in by_col[5].levels],
            ['empty', 'empty'],
        )
        self.assertEqual(
            by_col[2].levels[0].role, ROLE_MERGED_SUBORDINATE
        )
        self.assertTrue(by_col[1].hidden)
        self.assertFalse(by_col[3].hidden)

    def test_fixture_projection_and_lookup(self):
        projected = self.header_map.projection(include_hidden=False)
        self.assertEqual(
            projected.locate(('期初', '数量')).columns, []
        )
        match = projected.locate(('期末', '数量'))
        self.assertTrue(match.unique)
        rowx, colx = match.column.leaf_origin
        self.assertEqual(self.sheet.cell_value(rowx, colx), '数量')
        # Data row stays reachable through the original column index.
        self.assertEqual(self.sheet.cell_value(2, colx), 3.0)

    def test_cross_group_wide_merge_stays_one_header(self):
        book = xlrd.open_workbook(
            from_sample('header_groups.xls'), formatting_info=True
        )
        sheet = book.sheet_by_name('wide_merge')
        header_map = sheet.column_header_map(2)
        match = header_map.locate(('跨组合计',), prefix=True)
        self.assertEqual(len(match.columns), 4)
        self.assertEqual(match.region.kind, REGION_MERGED)
        self.assertEqual(match.region.range, (0, 1, 0, 4))
        # Leaf 数量 columns remain two distinct columns underneath it.
        leaves = header_map.locate(('跨组合计', '数量'))
        self.assertEqual(
            sorted(col.colx for col in leaves.columns), [0, 2]
        )

    def test_merged_leaf_shared_anchor_reports_merged_region(self):
        book = Book()
        book.logfile = sys.stdout
        book.verbosity = 0
        book.ragged_rows = False
        book.formatting_info = True
        book.biff_version = 80
        book._sheet_visibility = [0]
        sheet = Sheet(book, 0, 'merged_leaf', 0)
        sheet.put_cell(0, 0, XL_CELL_TEXT, '数量', 0)
        sheet.put_cell(0, 1, XL_CELL_BLANK, '', 0)
        sheet.merged_cells = [(0, 1, 0, 2)]
        sheet.tidy_dimensions()

        header_map = sheet.column_header_map(1)
        left = header_map.column_at(0)
        right = header_map.column_at(1)
        # One shared anchor, not two independent same-named cells.
        self.assertEqual(left.path, right.path)
        self.assertEqual(left.path_origins, right.path_origins)
        self.assertEqual(
            right.levels[0].role, ROLE_MERGED_SUBORDINATE
        )
        match = header_map.locate(('数量',))
        self.assertFalse(match.unique)
        self.assertEqual(
            sorted(col.colx for col in match.columns), [0, 1]
        )
        self.assertEqual(match.region.kind, REGION_MERGED)
        self.assertEqual(match.region.range, (0, 1, 0, 2))
