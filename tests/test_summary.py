# Copyright (c) 2026
# This module is part of the xlrd package, which is released under a
# BSD-style licence.
import json
import unittest

import xlrd
from xlrd.biffh import (
    XL_CELL_BLANK, XL_CELL_BOOLEAN, XL_CELL_DATE, XL_CELL_EMPTY,
    XL_CELL_ERROR, XL_CELL_NUMBER, XL_CELL_TEXT,
)
from xlrd.summary import (
    REASON_BLANK, REASON_EMPTY, REASON_ERROR, REASON_OUT_OF_BOUNDS,
    REASON_TEXT, merge_summaries, summarize_sheet, summarize_workbook,
)

from .helpers import from_sample


class FakeSheet(object):
    """Minimal sheet-like object: grid of (ctype, value) tuples."""

    def __init__(self, name, grid):
        self.name = name
        self._grid = grid
        self.nrows = len(grid)

    def row_len(self, rowx):
        return len(self._grid[rowx])

    def cell_type(self, rowx, colx):
        return self._grid[rowx][colx][0]

    def cell_value(self, rowx, colx):
        return self._grid[rowx][colx][1]


NUM = XL_CELL_NUMBER
TXT = XL_CELL_TEXT
ERR = XL_CELL_ERROR
BLANK = XL_CELL_BLANK
EMPTY = XL_CELL_EMPTY
DATE = XL_CELL_DATE
BOOL = XL_CELL_BOOLEAN


class TestSummarizeSheet(unittest.TestCase):

    def test_exact_total_and_zero_booked_separately(self):
        sheet = FakeSheet('S1', [
            [(TXT, 'txn'), (NUM, 10.0)],
            [(TXT, 'a'), (NUM, 0.0)],
            [(TXT, 'b'), (NUM, 2.5)],
        ])
        summary = summarize_sheet(sheet, amount_col=1)
        self.assertEqual(summary.rows_scanned, 3)
        self.assertEqual(summary.exact_total, 12.5)
        self.assertEqual(summary.total_low, 12.5)
        self.assertEqual(summary.total_high, 12.5)
        self.assertTrue(summary.fully_determined)
        self.assertEqual(summary.zero_cells, 1)
        self.assertEqual(summary.error_cells, 0)

    def test_error_blank_empty_text_date_boolean_are_unresolved(self):
        sheet = FakeSheet('S2', [
            [(TXT, 'k1'), (ERR, 7)],
            [(TXT, 'k2'), (BLANK, '')],
            [(TXT, 'k3'), (EMPTY, '')],
            [(TXT, 'k4'), (TXT, 'n/a')],
            [(TXT, 'k5'), (DATE, 44000.0)],
            [(TXT, 'k6'), (BOOL, 1)],
            [(TXT, 'k7'), (NUM, 3.0)],
        ])
        summary = summarize_sheet(sheet, amount_col=1)
        reasons = [entry.reason for entry in summary.unresolved]
        self.assertEqual(
            sorted(reasons),
            sorted([REASON_ERROR, REASON_BLANK, REASON_EMPTY,
                    REASON_TEXT, 'date', 'boolean']),
        )
        self.assertEqual(summary.error_cells, 1)
        self.assertEqual(summary.blank_cells, 1)
        self.assertEqual(summary.empty_cells, 1)
        self.assertEqual(summary.exact_total, 3.0)
        self.assertFalse(summary.fully_determined)
        error_entry = summary.unresolved[0]
        self.assertEqual(error_entry.ref.error_text, '#DIV/0!')
        self.assertEqual(error_entry.ref.a1, 'B1')
        self.assertEqual(error_entry.ref.location, 'S2/S2!B1')

    def test_bounds_exclude_and_record_out_of_bounds(self):
        sheet = FakeSheet('S3', [
            [(TXT, 'a'), (NUM, 5.0)],
            [(TXT, 'b'), (NUM, 500.0)],
            [(TXT, 'c'), (NUM, -1.0)],
            [(TXT, 'd'), (NUM, 50.0)],
        ])
        summary = summarize_sheet(
            sheet, amount_col=1, lower=0.0, upper=100.0,
        )
        self.assertEqual(summary.exact_total, 55.0)
        self.assertEqual(len(summary.out_of_bounds), 2)
        self.assertEqual(
            [entry.reason for entry in summary.unresolved],
            [REASON_OUT_OF_BOUNDS, REASON_OUT_OF_BOUNDS],
        )
        self.assertEqual(summary.zero_cells, 0)
        with self.assertRaises(ValueError):
            summarize_sheet(sheet, amount_col=1, lower=10.0, upper=1.0)

    def test_boundary_values_are_inclusive(self):
        sheet = FakeSheet('S4', [
            [(TXT, 'a'), (NUM, 0.0)],
            [(TXT, 'b'), (NUM, 100.0)],
        ])
        summary = summarize_sheet(
            sheet, amount_col=1, lower=0.0, upper=100.0,
        )
        self.assertEqual(summary.exact_total, 100.0)
        self.assertEqual(summary.zero_cells, 1)
        self.assertEqual(summary.unresolved, [])

    def test_known_duplicates_counted_once(self):
        sheet = FakeSheet('S5', [
            [(TXT, 'T1'), (NUM, 10.0)],
            [(TXT, 'T1'), (NUM, 10.0)],
            [(TXT, 'T2'), (NUM, 5.0)],
        ])
        summary = summarize_sheet(sheet, amount_col=1, key_col=0)
        self.assertEqual(summary.exact_total, 15.0)
        self.assertEqual(len(summary.transactions), 2)
        self.assertEqual(summary.duplicate_rows_dropped, 1)
        group_t1 = summary.transactions[0]
        self.assertEqual(len(group_t1.members), 2)
        self.assertTrue(group_t1.exact)
        self.assertEqual(group_t1.amount_range.low, 10.0)

    def test_contradictory_duplicates_give_range_with_witnesses(self):
        sheet = FakeSheet('S6', [
            [(TXT, 'T1'), (NUM, 10.0)],
            [(TXT, 'T1'), (NUM, 12.0)],
            [(TXT, 'T1'), (NUM, 12.0)],
            [(TXT, 'T2'), (NUM, 4.0)],
        ])
        summary = summarize_sheet(sheet, amount_col=1, key_col=0)
        self.assertEqual(summary.total_low, 14.0)
        self.assertEqual(summary.total_high, 16.0)
        self.assertIsNone(summary.exact_total)
        self.assertEqual(len(summary.contradictory_groups), 1)
        group = summary.contradictory_groups[0]
        self.assertEqual(group.amount_range.low, 10.0)
        self.assertEqual(group.amount_range.high, 12.0)
        self.assertEqual(
            [ref.a1 for ref in group.amount_range.witness_low], ['B1'])
        self.assertEqual(
            [ref.a1 for ref in group.amount_range.witness_high],
            ['B2', 'B3'],
        )
        self.assertEqual(summary.duplicate_rows_dropped, 2)

    def test_tolerance_treats_near_duplicates_as_agreeing(self):
        sheet = FakeSheet('S7', [
            [(TXT, 'T1'), (NUM, 10.0)],
            [(TXT, 'T1'), (NUM, 10.005)],
        ])
        summary = summarize_sheet(
            sheet, amount_col=1, key_col=0, tolerance=0.01,
        )
        self.assertEqual(summary.exact_total, 10.0)
        self.assertEqual(summary.contradictory_groups, [])

    def test_duplicate_with_unresolved_member_still_ranges(self):
        sheet = FakeSheet('S8', [
            [(TXT, 'T1'), (NUM, 10.0)],
            [(TXT, 'T1'), (ERR, 7)],
            [(TXT, 'T1'), (NUM, 14.0)],
        ])
        summary = summarize_sheet(sheet, amount_col=1, key_col=0)
        self.assertEqual(summary.error_cells, 1)
        group = summary.transactions[0]
        self.assertTrue(group.contradiction)
        self.assertEqual(group.amount_range.low, 10.0)
        self.assertEqual(group.amount_range.high, 14.0)
        self.assertEqual(len(group.unresolved_members), 1)
        self.assertFalse(group.fully_known)

    def test_keyless_rows_are_never_merged(self):
        sheet = FakeSheet('S9', [
            [(EMPTY, ''), (NUM, 10.0)],
            [(EMPTY, ''), (NUM, 10.0)],
        ])
        summary = summarize_sheet(sheet, amount_col=1)
        self.assertEqual(summary.exact_total, 20.0)
        self.assertEqual(len(summary.transactions), 2)
        self.assertEqual(summary.duplicate_rows_dropped, 0)
        self.assertEqual(len(summary.key_issues), 0)

    def test_missing_key_is_flagged(self):
        sheet = FakeSheet('S10', [
            [(EMPTY, ''), (NUM, 10.0)],
            [(ERR, 7), (NUM, 20.0)],
        ])
        summary = summarize_sheet(sheet, amount_col=1, key_col=0)
        self.assertEqual(len(summary.key_issues), 2)
        self.assertEqual(summary.exact_total, 30.0)

    def test_header_name_resolution_and_skip(self):
        sheet = FakeSheet('S11', [
            [(TXT, 'ref'), (TXT, 'amount')],
            [(TXT, 'a'), (NUM, 7.0)],
            [(TXT, 'b'), (NUM, 8.0)],
        ])
        summary = summarize_sheet(
            sheet, amount_col='amount', key_col='ref', header_rows=1,
        )
        self.assertEqual(summary.rows_scanned, 2)
        self.assertEqual(summary.exact_total, 15.0)
        with self.assertRaises(ValueError):
            summarize_sheet(sheet, amount_col='missing', header_rows=1)

    def test_ragged_row_treated_as_empty(self):
        sheet = FakeSheet('S12', [
            [(TXT, 'a'), (NUM, 7.0)],
            [(TXT, 'b')],
        ])
        summary = summarize_sheet(sheet, amount_col=1)
        self.assertEqual(summary.exact_total, 7.0)
        self.assertEqual(summary.empty_cells, 1)
        self.assertEqual(summary.unresolved[0].reason, REASON_EMPTY)

    def test_row_range_partitions_one_sheet(self):
        sheet = FakeSheet('S13', [
            [(TXT, 'a'), (NUM, 1.0)],
            [(TXT, 'b'), (NUM, 2.0)],
            [(TXT, 'c'), (NUM, 4.0)],
        ])
        first = summarize_sheet(
            sheet, amount_col=1, row_range=(0, 2), partition='P1',
        )
        second = summarize_sheet(
            sheet, amount_col=1, row_range=(2, 3), partition='P2',
        )
        self.assertEqual(first.exact_total, 3.0)
        self.assertEqual(second.exact_total, 4.0)
        self.assertEqual(first.partition, 'P1')


class TestMerge(unittest.TestCase):

    def _two_partitions(self):
        left = FakeSheet('L', [
            [(TXT, 'T1'), (NUM, 10.0)],
            [(TXT, 'T2'), (NUM, 5.0)],
            [(TXT, 'Lonly'), (NUM, 2.0)],
        ])
        right = FakeSheet('R', [
            [(TXT, 'T1'), (NUM, 10.0)],
            [(TXT, 'T3'), (NUM, 8.0)],
        ])
        summary_left = summarize_sheet(
            left, amount_col=1, key_col=0, partition='PL')
        summary_right = summarize_sheet(
            right, amount_col=1, key_col=0, partition='PR')
        return summary_left, summary_right

    def test_cross_partition_duplicate_counted_once(self):
        left, right = self._two_partitions()
        merged = merge_summaries([left, right])
        # T1 appears in both partitions: counted once (10), not twice.
        self.assertEqual(merged.exact_total, 25.0)
        keys = [group.key for group in merged.transactions]
        self.assertEqual(sorted(keys), ['Lonly', 'T1', 'T2', 'T3'])
        t1 = next(group for group in merged.transactions
                  if group.key == 'T1')
        self.assertEqual(len(t1.members), 2)
        self.assertEqual(t1.amount_range.low, 10.0)
        self.assertEqual(merged.duplicate_rows_dropped, 1)
        # 3 + 2 distinct source cells; the two T1 rows are different
        # cells in different partitions, only the transaction is shared.
        self.assertEqual(merged.rows_scanned, 5)

    def test_naive_interval_addition_would_be_wrong(self):
        left, right = self._two_partitions()
        naive = left.amount_range + right.amount_range
        merged = merge_summaries([left, right])
        # Naive addition double counts T1 (10 + 10).
        self.assertEqual(naive.low, 35.0)
        self.assertEqual(merged.exact_total, 25.0)

    def test_cross_partition_contradiction_range_with_witnesses(self):
        left = FakeSheet('L', [
            [(TXT, 'T1'), (NUM, 10.0)],
        ])
        right = FakeSheet('R', [
            [(TXT, 'T1'), (NUM, 13.0)],
        ])
        summary_left = summarize_sheet(
            left, amount_col=1, key_col=0, partition='PL')
        summary_right = summarize_sheet(
            right, amount_col=1, key_col=0, partition='PR')
        merged = merge_summaries([summary_left, summary_right])
        self.assertEqual(merged.total_low, 10.0)
        self.assertEqual(merged.total_high, 13.0)
        group = merged.contradictory_groups[0]
        low_locs = [ref.location for ref
                    in group.amount_range.witness_low]
        high_locs = [ref.location for ref
                     in group.amount_range.witness_high]
        self.assertEqual(low_locs, ['PL/L!B1'])
        self.assertEqual(high_locs, ['PR/R!B1'])

    def test_keyless_groups_carried_over_not_merged(self):
        left = FakeSheet('L', [
            [(EMPTY, ''), (NUM, 10.0)],
        ])
        right = FakeSheet('R', [
            [(EMPTY, ''), (NUM, 10.0)],
        ])
        summary_left = summarize_sheet(left, amount_col=1, partition='PL')
        summary_right = summarize_sheet(right, amount_col=1, partition='PR')
        merged = merge_summaries([summary_left, summary_right])
        self.assertEqual(merged.exact_total, 20.0)
        self.assertEqual(len(merged.transactions), 2)

    def test_merge_preserves_unresolved_and_counts(self):
        left = FakeSheet('L', [
            [(TXT, 'T1'), (ERR, 7)],
            [(TXT, 'T2'), (NUM, 5.0)],
        ])
        right = FakeSheet('R', [
            [(TXT, 'T3'), (BLANK, '')],
            [(TXT, 'T4'), (NUM, 0.0)],
        ])
        summary_left = summarize_sheet(
            left, amount_col=1, key_col=0, partition='PL')
        summary_right = summarize_sheet(
            right, amount_col=1, key_col=0, partition='PR')
        merged = merge_summaries([summary_left, summary_right])
        self.assertEqual(merged.exact_total, 5.0)
        self.assertEqual(merged.error_cells, 1)
        self.assertEqual(merged.blank_cells, 1)
        self.assertEqual(merged.zero_cells, 1)
        self.assertEqual(len(merged.unresolved), 2)
        self.assertFalse(merged.fully_determined)

    def test_merge_keeps_out_of_bounds_excluded(self):
        left = FakeSheet('L', [
            [(TXT, 'T1'), (NUM, 999.0)],
            [(TXT, 'T2'), (NUM, 1.0)],
        ])
        right = FakeSheet('R', [
            [(TXT, 'T3'), (NUM, 2.0)],
        ])
        summary_left = summarize_sheet(
            left, amount_col=1, key_col=0, lower=0.0, upper=100.0,
            partition='PL')
        summary_right = summarize_sheet(
            right, amount_col=1, key_col=0, lower=0.0, upper=100.0,
            partition='PR')
        merged = merge_summaries([summary_left, summary_right])
        self.assertEqual(merged.exact_total, 3.0)
        self.assertEqual(len(merged.out_of_bounds), 1)

    def test_merge_via_method_and_empty_input(self):
        left, right = self._two_partitions()
        merged = left.merge(right, partition='custom')
        self.assertEqual(merged.partition, 'custom')
        self.assertEqual(merged.exact_total, 25.0)
        with self.assertRaises(ValueError):
            merge_summaries([])

    def test_remerging_an_input_does_not_double_book(self):
        left = FakeSheet('L', [
            [(TXT, 'T1'), (ERR, 7)],
            [(TXT, 'T2'), (NUM, 0.0)],
        ])
        right = FakeSheet('R', [
            [(TXT, 'T3'), (NUM, 5.0)],
        ])
        summary_left = summarize_sheet(
            left, amount_col=1, key_col=0, partition='PL')
        summary_right = summarize_sheet(
            right, amount_col=1, key_col=0, partition='PR')
        merged = merge_summaries([summary_left, summary_right])
        remerged = merge_summaries([merged, summary_left])
        self.assertEqual(remerged.error_cells, 1)
        self.assertEqual(remerged.zero_cells, 1)
        self.assertEqual(len(remerged.unresolved), 1)
        self.assertEqual(len(remerged.out_of_bounds), 0)
        self.assertEqual(remerged.exact_total, 5.0)
        self.assertEqual(remerged.rows_scanned, 3)
        keys = sorted(group.key for group in remerged.transactions)
        self.assertEqual(keys, ['T1', 'T2', 'T3'])


class TestSerialization(unittest.TestCase):

    def test_as_dict_is_json_serialisable(self):
        sheet = FakeSheet('J', [
            [(TXT, 'T1'), (NUM, 10.0)],
            [(TXT, 'T1'), (NUM, 12.0)],
            [(TXT, 'T2'), (ERR, 7)],
        ])
        summary = summarize_sheet(sheet, amount_col=1, key_col=0)
        payload = json.loads(summary.to_json())
        self.assertEqual(payload['totals']['low'], 10.0)
        self.assertEqual(payload['totals']['high'], 12.0)
        self.assertEqual(payload['counts']['error_cells'], 1)
        self.assertEqual(
            payload['counts']['contradictory_transactions'], 1)
        text = summary.report_text()
        self.assertIn('Total (possible range)', text)
        self.assertIn('low  witness', text)
        self.assertIn('high witness', text)


class TestRealWorkbook(unittest.TestCase):

    def test_error_cell_sample(self):
        book = xlrd.open_workbook(
            from_sample('formula_test_sjmachin.xls'), formatting_info=True)
        sheet = book.sheet_by_name('Sheet1')
        summary = summarize_sheet(sheet, amount_col=1, header_rows=1)
        self.assertEqual(summary.error_cells, 1)
        self.assertAlmostEqual(summary.exact_total,
                               0.14285714285714285)
        reasons = {entry.reason for entry in summary.unresolved}
        self.assertIn(REASON_ERROR, reasons)
        error_entry = next(
            entry for entry in summary.unresolved
            if entry.reason == REASON_ERROR)
        self.assertEqual(error_entry.ref.error_text, '#DIV/0!')
        self.assertEqual(error_entry.ref.rowx, 6)

    def test_numeric_sample_and_workbook_helper(self):
        book = xlrd.open_workbook(from_sample('profiles.xls'))
        sheet = book.sheet_by_name('PROFILEDEF')
        summary = summarize_sheet(sheet, amount_col=1, header_rows=1)
        expected = sum(
            sheet.cell_value(rowx, 1)
            for rowx in range(1, sheet.nrows)
        )
        self.assertEqual(summary.exact_total, expected)
        self.assertEqual(summary.unresolved, [])
        summaries = summarize_workbook(
            book, amount_col=1, sheet_names=['PROFILEDEF'], header_rows=1)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].exact_total, expected)


if __name__ == '__main__':
    unittest.main()
