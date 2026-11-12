# -*- coding: utf-8
"""Tests for worksheet-scoped defined name resolution (xlrd.names)."""

from struct import pack
from unittest import TestCase

import xlrd
from xlrd.biffh import XLRDError
from xlrd.book import Name
from xlrd.formula import Ref3D
from xlrd.names import (
    AreaReference, ExternalReference, NameDependency, NameResolution,
    REASON_CIRCULAR, REASON_EMPTY, REASON_ERROR, REASON_FUNCTION,
    REASON_RELATIVE, REASON_SELF_REF,
    resolve_name, resolve_named, resolve_visible_names,
)

from .helpers import from_sample


def make_name(index, name, scope=-1, hidden=0, builtin=0):
    nobj = Name()
    nobj.book = None
    nobj.name_index = index
    nobj.name = name
    nobj.scope = scope
    nobj.hidden = hidden
    nobj.builtin = builtin
    nobj.macro = 0
    nobj.binary = 0
    nobj.basic_formula_len = 0
    nobj.raw_formula = b''
    nobj.result = None
    nobj.stack = []
    return nobj


class FakeBook(object):
    """Minimal book needed by the name-selection helpers."""

    def __init__(self, names, sheet_names):
        self.name_obj_list = names
        self._sheet_names = sheet_names
        self.biff_version = 80
        for nobj in names:
            nobj.book = self


class NameSelectionTests(TestCase):

    def setUp(self):
        names = [
            make_name(0, 'Rate', scope=-1),
            make_name(1, 'Rate', scope=1),   # local shadow on sheet 1
            make_name(2, 'Rate', scope=2),   # local on sheet 2
            make_name(3, 'Secret', scope=-1, hidden=1),
            make_name(4, 'Print_Area', scope=0, builtin=1),
        ]
        self.book = FakeBook(names, ['S1', 'S2', 'S3'])

    def test_global_visible_everywhere(self):
        chosen = {r.name: r for r in resolve_visible_names(self.book, 0)}
        self.assertEqual(chosen['Rate'].scope, -1)

    def test_local_name_shadows_global(self):
        for sheetx, expected_scope in ((1, 1), (2, 2)):
            chosen = {r.name: r for r in resolve_visible_names(self.book, sheetx)}
            self.assertEqual(chosen['Rate'].scope, expected_scope)

    def test_other_sheet_locals_are_hidden(self):
        # From sheet 0 only the global Rate is visible, not the local copies.
        chosen = {r.name: r for r in resolve_visible_names(self.book, 0)}
        self.assertEqual(len(chosen), 3)
        self.assertEqual(chosen['Rate'].scope, -1)

    def test_filters(self):
        no_hidden = {r.name for r in
                     resolve_visible_names(self.book, 0, include_hidden=False)}
        self.assertNotIn('Secret', no_hidden)
        no_builtin = {r.name for r in
                      resolve_visible_names(self.book, 0, include_builtin=False)}
        self.assertNotIn('Print_Area', no_builtin)

    def test_resolve_named_prefers_local(self):
        resolution = resolve_named(self.book, 1, 'rate')  # case-insensitive
        self.assertEqual(resolution.scope, 1)

    def test_resolve_named_missing_raises(self):
        with self.assertRaises(XLRDError):
            resolve_named(self.book, 0, 'absent')


class NamesDemoTests(TestCase):

    def setUp(self):
        self.book = xlrd.open_workbook(from_sample('namesdemo.xls'))
        self.sheet = self.book.sheet_by_index(0)

    def resolution(self, name):
        return self.sheet.defined_name(name)

    def test_static_area_reference(self):
        res = self.resolution('Sales')
        self.assertEqual(res.kind, 'reference')
        self.assertTrue(res.complete)
        area = res.areas[0]
        self.assertEqual(area.sheet_names, ('Sheet3',))
        self.assertEqual(area.coords, (1, 2, 1, 14))

    def test_three_dimensional_area(self):
        res = self.resolution('Moscow')
        area = res.areas[0]
        self.assertEqual(area.sheet_names, ('Sheet1', 'Sheet2', 'Sheet3'))
        self.assertEqual(area.coords, (0, 10, 0, 26))
        with self.assertRaises(ValueError):
            area.sheet_name

    def test_union_keeps_both_areas(self):
        res = self.resolution('List')
        self.assertEqual(len(res.areas), 2)
        self.assertEqual(
            [a.coords for a in res.areas],
            [(8, 10, 0, 4), (9, 11, 2, 6)],
        )

    def test_union_iter_cells_deduplicates_overlap(self):
        res = self.resolution('List')
        cells = list(res.iter_cells())
        self.assertEqual(len(cells), len(set(cells)))
        # rectangle1 = 8 cells, rectangle2 = 8 cells, overlap = 2 cells
        self.assertEqual(len(cells), 14)

    def test_exact_duplicate_area_is_removed(self):
        sheet3 = self.book.sheet_by_index(2)
        res = sheet3.defined_name('Print_Titles')
        keys = [area.key() for area in res.areas]
        self.assertEqual(len(keys), len(set(keys)))
        rebuilt = [
            AreaReference(self.book, self.book.name_obj_list[21].result.value[0]),
            AreaReference(self.book, self.book.name_obj_list[21].result.value[0]),
        ]
        self.assertEqual(rebuilt[0], rebuilt[1])
        self.assertEqual(len({area.key() for area in rebuilt}), 1)

    def test_intersection(self):
        res = self.resolution('Intersection')
        self.assertEqual(len(res.areas), 1)
        self.assertEqual(res.areas[0].coords, (9, 10, 2, 4))

    def test_range_operator_envelopes(self):
        res = self.resolution('Union')  # rectangle1:rectangle2
        self.assertEqual(res.areas[0].coords, (8, 11, 0, 6))

    def test_dependencies_are_listed(self):
        res = self.resolution('all_local_ranges')
        deps = {(d.name, d.scope_sheet_name) for d in res.dependencies}
        self.assertEqual(deps, {
            ('LocalRange', 'Sheet1'),
            ('localRange', 'Sheet2'),
            ('Localrange', 'Sheet3'),
        })
        for dep in res.dependencies:
            self.assertIsInstance(dep, NameDependency)

    def test_transitive_dependencies(self):
        res = self.resolution('BottomLine')
        self.assertEqual(
            {d.name for d in res.dependencies},
            {'Profit', 'Year_Tot'},
        )

    def test_relative_reference_requires_base_cell(self):
        res = self.resolution('RelativePos')
        self.assertFalse(res.complete)
        self.assertTrue(
            any(reason == REASON_RELATIVE for reason, _ in res.unresolved_branches)
        )
        self.assertEqual(res.areas, [])

    def test_relative_reference_uses_base_cell(self):
        name_obj = self.book.name_obj_list[26]  # RelativePos
        res = resolve_name(self.book, name_obj, base_sheet=0, base_row=26,
                           base_col=2)
        self.assertTrue(res.complete)
        self.assertEqual(res.base_cell, (0, 26, 2))
        self.assertEqual(res.areas[0].coords, (26, 36, 2, 28))

    def test_partly_relative_reference(self):
        name_obj = self.book.name_obj_list[0]  # A1Z10, one relative row bound
        res = resolve_name(self.book, name_obj, base_sheet=2, base_row=26,
                           base_col=0)
        self.assertTrue(res.complete)
        self.assertEqual(res.areas[0].sheet_names, ('Sheet3',))
        self.assertEqual(res.areas[0].coords, (0, 10, 0, 26))

    def test_constant_names(self):
        number = self.resolution('PosInt')
        self.assertEqual(number.kind, 'constant')
        self.assertTrue(number.complete)
        self.assertEqual(number.constant, ('oNUM', 65535.0))
        text = self.resolution('ASCII_String')
        self.assertEqual(text.constant, ('oSTRG', 'ascii'))
        boolean = self.resolution('vrai')
        self.assertEqual(boolean.constant, ('oBOOL', 1))

    def test_local_builtin_name_scoped_to_its_sheet(self):
        sheet3 = self.book.sheet_by_index(2)
        res = sheet3.defined_name('Print_Area')
        self.assertTrue(res.builtin)
        self.assertEqual(res.scope_sheet_name, 'Sheet3')
        with self.assertRaises(XLRDError):
            self.sheet.defined_name('Print_Area')

    def test_result_is_named_resolution(self):
        self.assertIsInstance(self.resolution('Sales'), NameResolution)


class ExternalLinkTests(TestCase):

    def setUp(self):
        self.book = xlrd.open_workbook(from_sample('invalid_formula.xls'))
        self.sheet = self.book.sheet_by_index(0)

    def test_external_sheet_is_never_complete(self):
        res = self.sheet.defined_name('_1bs1_')
        self.assertEqual(res.kind, 'external')
        self.assertFalse(res.complete)
        self.assertEqual(res.areas, [])
        self.assertEqual(len(res.external_refs), 1)
        ref = res.external_refs[0]
        self.assertIsInstance(ref, ExternalReference)
        self.assertEqual(ref.kind, 'external_sheet')
        # Coordinates from the formula are preserved even though the
        # workbook they point into is unavailable.
        self.assertEqual((ref.rowxlo, ref.rowxhi, ref.colxlo, ref.colxhi),
                         (260, 370, 3, 4))

    def test_half_resolvable_union_is_partial(self):
        # 材料大类 has two external branches only in this fixture; ensure a
        # name mixing local and external branches would not look complete.
        res = self.sheet.defined_name('材料大类')
        self.assertFalse(res.complete)
        self.assertEqual(res.kind, 'external')
        self.assertTrue(
            all(ref.kind == 'external_sheet' for ref in res.external_refs)
        )

    def test_error_branches_are_reported(self):
        res = self.sheet.defined_name('_DAT1')
        self.assertEqual(res.kind, 'error')
        self.assertFalse(res.complete)
        reasons = {reason for reason, _ in res.unresolved_branches}
        self.assertIn(REASON_ERROR, reasons)

    def test_function_branch_marks_unresolved(self):
        res = self.sheet.defined_name('月份')  # MONTH(...)
        self.assertFalse(res.complete)
        reasons = {reason for reason, _ in res.unresolved_branches}
        self.assertIn(REASON_FUNCTION, reasons)

    def test_no_name_in_corrupt_book_reports_complete(self):
        resolutions = self.sheet.defined_names(include_builtin=False)
        self.assertTrue(resolutions)
        self.assertFalse(any(r.complete and r.kind == 'reference'
                             for r in resolutions))


def name_token(target_index):
    """BIFF8 tName token pointing at the 1-based name ``target_index``."""
    return b'\x23' + pack('<H', target_index + 1) + b'\x00\x00'


def formula_name(index, name, formula):
    nobj = make_name(index, name)
    nobj.raw_formula = formula
    nobj.basic_formula_len = len(formula)
    return nobj


class FormulaBranchTests(TestCase):

    def _book(self, names):
        book = FakeBook(names, ['S1'])
        book.biff_version = 80
        return book

    def test_self_reference_is_flagged(self):
        nobj = formula_name(0, 'Self', name_token(0))
        book = self._book([nobj])
        res = resolve_name(book, nobj)
        self.assertFalse(res.complete)
        reasons = {reason for reason, _ in res.unresolved_branches}
        self.assertIn(REASON_SELF_REF, reasons)

    def test_circular_pair_is_flagged(self):
        first = formula_name(0, 'Alpha', name_token(1))
        second = formula_name(1, 'Beta', name_token(0))
        book = self._book([first, second])
        res = resolve_name(book, first)
        self.assertFalse(res.complete)
        reasons = {reason for reason, _ in res.unresolved_branches}
        # A two-name cycle returns to the root name itself.
        self.assertIn(REASON_SELF_REF, reasons)
        self.assertEqual({d.name for d in res.dependencies}, {'Alpha', 'Beta'})

    def test_three_way_cycle_is_flagged(self):
        alpha = formula_name(0, 'Alpha', name_token(1))
        beta = formula_name(1, 'Beta', name_token(2))
        gamma = formula_name(2, 'Gamma', name_token(1))
        book = self._book([alpha, beta, gamma])
        res = resolve_name(book, alpha)
        self.assertFalse(res.complete)
        reasons = {reason for reason, _ in res.unresolved_branches}
        self.assertIn(REASON_CIRCULAR, reasons)

    def test_empty_formula_is_unresolved(self):
        nobj = make_name(0, 'Empty')  # zero-length formula, like an undefined label
        book = self._book([nobj])
        res = resolve_name(book, nobj)
        self.assertEqual(res.kind, 'unresolved')
        self.assertIn(REASON_EMPTY,
                      {reason for reason, _ in res.unresolved_branches})

    def test_unknown_token_is_handled_not_raised(self):
        nobj = make_name(0, 'Weird')
        nobj.raw_formula = b'\x26'  # tMemArea: not allowed in a name formula
        nobj.basic_formula_len = 1
        book = self._book([nobj])
        res = resolve_name(book, nobj)
        self.assertFalse(res.complete)
        self.assertTrue(res.unresolved_branches)


class PartialResolutionTests(TestCase):

    def _resolution(self, refs, externals):
        from xlrd.names import _Node
        book = FakeBook([], ['S1'])
        book.biff_version = 80
        name_obj = make_name(0, 'Mixed')
        name_obj.book = book
        node = _Node(refs=refs, externals=externals)
        return NameResolution(book, name_obj, node, base_cell=None)

    def test_partial_reference_is_incomplete(self):
        res = self._resolution(
            refs=[Ref3D((0, 1, 0, 2, 0, 2))],
            externals=[ExternalReference('external_sheet', 'other.xls')],
        )
        self.assertEqual(res.kind, 'partial_reference')
        self.assertFalse(res.complete)
        self.assertEqual(len(res.areas), 1)
        self.assertEqual(res.areas[0].coords, (0, 2, 0, 2))
        self.assertEqual(res.external_refs[0].kind, 'external_sheet')

    def test_overlapping_areas_dedup_cells_only(self):
        ref_a = Ref3D((0, 1, 0, 3, 0, 3))  # 3x3
        ref_b = Ref3D((0, 1, 2, 5, 2, 5))  # overlaps one cell
        res = self._resolution(refs=[ref_a, ref_b], externals=[])
        self.assertEqual(len(res.areas), 2)  # areas kept distinct
        cells = list(res.iter_cells())
        self.assertEqual(len(cells), len(set(cells)))
        self.assertEqual(len(cells), 9 + 9 - 1)


class RefErrorNameTests(TestCase):

    def setUp(self):
        self.book = xlrd.open_workbook(from_sample('issue20.xls'))

    def test_ref_error_name_is_incomplete(self):
        sheet = self.book.sheet_by_index(0)
        res = sheet.defined_name('Print_Area')
        self.assertEqual(res.kind, 'error')
        self.assertFalse(res.complete)
        self.assertEqual(res.areas, [])
        reasons = {reason for reason, _ in res.unresolved_branches}
        self.assertIn(REASON_ERROR, reasons)

    def test_local_string_constant_per_sheet(self):
        for sheetx, value in ((0, 'Sheet1'), (1, 'Sheet2'), (2, 'Sheet3')):
            sheet = self.book.sheet_by_index(sheetx)
            res = sheet.defined_name('Sheet_Title')
            self.assertEqual(res.scope, sheetx)
            self.assertEqual(res.kind, 'constant')
            self.assertTrue(res.complete)
            self.assertEqual(res.constant, ('oSTRG', value))


class FormulaNameWorkbookTests(TestCase):

    def setUp(self):
        self.book = xlrd.open_workbook(
            from_sample('formula_test_names.xls'))
        self.sheet = self.book.sheet_by_index(0)

    def test_constant_from_evaluated_function(self):
        # xlrd fully evaluates CHOOSE of constants; the function branch must
        # not hide the concrete string result.
        res = self.sheet.defined_name('testchoose')
        self.assertEqual(res.kind, 'constant')
        self.assertTrue(res.complete)
        self.assertEqual(res.constant, ('oSTRG', 'C'))

    def test_unevaluated_function_is_incomplete(self):
        res = self.sheet.defined_name('tfunc')  # ABS(...) -> oUNK
        self.assertFalse(res.complete)
        self.assertIn(REASON_FUNCTION,
                      {reason for reason, _ in res.unresolved_branches})
