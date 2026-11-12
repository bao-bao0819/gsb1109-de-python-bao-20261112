# -*- coding: utf-8 -*-
from unittest import TestCase

import xlrd
from xlrd import columnexport
from xlrd.columnexport import (
    cell_format_basis, export_columns, parse_number_format,
    section_has_percent, split_format_sections,
)

from .helpers import from_sample


class TestSectionSplitting(TestCase):

    def test_plain_sections(self):
        self.assertEqual(
            split_format_sections('0.00%;-0.00%;0.00'),
            ['0.00%', '-0.00%', '0.00'],
        )

    def test_semicolon_inside_quotes_is_not_a_separator(self):
        self.assertEqual(
            split_format_sections('0.00"a;b";0.00'),
            ['0.00"a;b"', '0.00'],
        )

    def test_none_format_string(self):
        self.assertEqual(split_format_sections(None), [])


class TestPercentDetection(TestCase):

    def test_genuine_marker(self):
        self.assertTrue(section_has_percent('0.00%'))
        self.assertTrue(section_has_percent('[>=1]0.0%'))

    def test_quoted_percent_is_literal(self):
        self.assertFalse(section_has_percent('0.00" %"'))
        self.assertFalse(section_has_percent('0.00"(%)"'))

    def test_escaped_percent_is_literal(self):
        self.assertFalse(section_has_percent('0.00\\%'))

    def test_quoted_literal_plus_real_marker(self):
        self.assertTrue(section_has_percent('0.00" share: "0.00%'))

    def test_bracketed_content_is_never_a_marker(self):
        self.assertFalse(section_has_percent('[Red]0.00'))


class TestParseNumberFormat(TestCase):

    def test_single_section_applies_to_all_numbers(self):
        sections = parse_number_format('0%')
        self.assertEqual(len(sections), 1)
        section = sections[0]
        self.assertEqual(section['condition'], 'all')
        self.assertEqual(section['index'], 0)
        self.assertTrue(section['is_percent'])
        self.assertIsNone(section['explicit_condition'])

    def test_positional_conditions(self):
        sections = parse_number_format('0.00%;-0.00%;0.00')
        self.assertEqual(
            [s['condition'] for s in sections],
            ['positive', 'negative', 'zero'],
        )
        self.assertTrue(sections[0]['is_percent'])
        self.assertTrue(sections[1]['is_percent'])
        self.assertFalse(sections[2]['is_percent'])

    def test_four_section_format_has_text_section(self):
        sections = parse_number_format('0%;-0%;0%;@')
        self.assertEqual(
            [s['condition'] for s in sections],
            ['positive', 'negative', 'zero', 'text'],
        )

    def test_explicit_conditions_take_precedence(self):
        sections = parse_number_format('[>=1]0.0%;[<1]0.00;General')
        self.assertEqual(sections[0]['explicit_condition'], '>=1')
        self.assertTrue(sections[0]['is_percent'])
        self.assertEqual(sections[1]['explicit_condition'], '<1')
        self.assertFalse(sections[1]['is_percent'])
        # The trailing section falls back to the positional convention.
        self.assertIsNone(sections[2]['explicit_condition'])
        self.assertEqual(sections[2]['condition'], 'zero')

    def test_colour_bracket_is_not_mistaken_for_a_condition(self):
        sections = parse_number_format('[Red]0.00')
        self.assertIsNone(sections[0]['explicit_condition'])
        self.assertEqual(sections[0]['condition'], 'all')

    def test_quoted_percent_section_is_not_scaled(self):
        section = parse_number_format('0.00" %"')[0]
        self.assertFalse(section['is_percent'])
        section = parse_number_format('0.00\\%')[0]
        self.assertFalse(section['is_percent'])

    def test_more_than_four_sections_does_not_crash(self):
        sections = parse_number_format('0%;-0%;0%;@;0.00')
        self.assertEqual(len(sections), 5)
        self.assertEqual(
            [s['condition'] for s in sections],
            ['positive', 'negative', 'zero', 'text', 'text'],
        )


class TestCellFormatBasis(TestCase):

    def setUp(self):
        self.book = xlrd.open_workbook(
            from_sample('percent_columns.xls'), formatting_info=True)
        self.sheet = self.book.sheet_by_index(0)

    def test_percent_cell_basis(self):
        basis = cell_format_basis(self.book, self.sheet, 0, 0)
        self.assertEqual(basis['coordinate'], 'A1')
        self.assertEqual(basis['value'], 0.1)
        self.assertEqual(basis['xf_index'], self.sheet.cell(0, 0).xf_index)
        self.assertEqual(basis['format_str'], '0%')
        self.assertEqual(basis['format_kind'], 'custom')
        self.assertTrue(basis['is_percent'])
        self.assertEqual(basis['section']['condition'], 'all')

    def test_plain_number_cell_basis(self):
        basis = cell_format_basis(self.book, self.sheet, 0, 1)
        self.assertEqual(basis['coordinate'], 'B1')
        self.assertFalse(basis['is_percent'])
        self.assertEqual(basis['section']['format_text'], '0.00')

    def test_quoted_percent_is_treated_as_plain_number(self):
        basis = cell_format_basis(self.book, self.sheet, 0, 3)
        self.assertEqual(basis['coordinate'], 'D1')
        self.assertFalse(basis['is_percent'])

    def test_escaped_percent_is_treated_as_plain_number(self):
        basis = cell_format_basis(self.book, self.sheet, 1, 3)
        self.assertEqual(basis['coordinate'], 'D2')
        self.assertFalse(basis['is_percent'])

    def test_real_marker_wins_over_quoted_one(self):
        basis = cell_format_basis(self.book, self.sheet, 2, 3)
        self.assertEqual(basis['coordinate'], 'D3')
        self.assertTrue(basis['is_percent'])

    def test_explicit_condition_selects_percent_section(self):
        basis = cell_format_basis(self.book, self.sheet, 0, 4)
        self.assertEqual(basis['coordinate'], 'E1')
        self.assertTrue(basis['is_percent'])
        self.assertEqual(basis['section']['explicit_condition'], '>=1')

    def test_explicit_condition_selects_plain_section(self):
        basis = cell_format_basis(self.book, self.sheet, 1, 4)
        self.assertEqual(basis['coordinate'], 'E2')
        self.assertFalse(basis['is_percent'])
        self.assertEqual(basis['section']['explicit_condition'], '<1')

    def test_zero_selects_the_non_percent_zero_section(self):
        basis = cell_format_basis(self.book, self.sheet, 2, 5)
        self.assertEqual(basis['coordinate'], 'F3')
        self.assertEqual(basis['section']['condition'], 'zero')
        self.assertFalse(basis['is_percent'])

    def test_negative_selects_negative_section(self):
        basis = cell_format_basis(self.book, self.sheet, 1, 5)
        self.assertEqual(basis['coordinate'], 'F2')
        self.assertEqual(basis['section']['condition'], 'negative')
        self.assertTrue(basis['is_percent'])

    def test_built_in_format_key_is_classified(self):
        # Repoint A1's XF at the built-in "0%" format (key 9, <= 163).
        xf_index = self.sheet.cell(0, 0).xf_index
        self.book.xf_list[xf_index].format_key = 9
        basis = cell_format_basis(self.book, self.sheet, 0, 0)
        self.assertEqual(basis['format_key'], 9)
        self.assertEqual(basis['format_kind'], 'built-in')
        self.assertEqual(basis['format_str'], '0%')
        self.assertTrue(basis['is_percent'])

    def test_built_in_non_percent_format_key(self):
        xf_index = self.sheet.cell(0, 1).xf_index
        self.book.xf_list[xf_index].format_key = 2  # built-in "0.00"
        basis = cell_format_basis(self.book, self.sheet, 0, 1)
        self.assertEqual(basis['format_kind'], 'built-in')
        self.assertFalse(basis['is_percent'])

    def test_text_cell_has_no_basis(self):
        book = xlrd.open_workbook(
            from_sample('Formate.xls'), formatting_info=True)
        sheet = book.sheet_by_name(u'Blätt1')
        self.assertIsNone(cell_format_basis(book, sheet, 0, 0))

    def test_no_formatting_info_returns_none(self):
        book = xlrd.open_workbook(from_sample('percent_columns.xls'))
        sheet = book.sheet_by_index(0)
        self.assertIsNone(cell_format_basis(book, sheet, 0, 0))


class TestColumnExport(TestCase):

    def setUp(self):
        self.book = xlrd.open_workbook(
            from_sample('percent_columns.xls'), formatting_info=True)
        self.exports = {
            export['column_letter']: export
            for export in export_columns(self.book)
        }

    def _record(self, column, coordinate):
        export = self.exports[column]
        for record in export['converted'] + export['ambiguous']:
            if record['coordinate'] == coordinate:
                return record
        raise AssertionError('no record for %s' % coordinate)

    def test_unique_percent_values_are_converted(self):
        export = self.exports['A']
        self.assertEqual(len(export['ambiguous']), 0)
        values = {r['coordinate']: r['exported_value']
                  for r in export['converted']}
        self.assertEqual(values, {'A1': 10.0, 'A2': 25.0})
        self.assertTrue(all(r['unit'] == 'percent' for r in export['converted']))

    def test_unique_plain_values_pass_through(self):
        export = self.exports['B']
        self.assertEqual(len(export['ambiguous']), 0)
        values = {r['coordinate']: r['exported_value']
                  for r in export['converted']}
        self.assertEqual(values, {'B1': 0.1, 'B2': 0.25})
        self.assertTrue(all(r['unit'] == 'number' for r in export['converted']))

    def test_shared_value_with_conflicting_units_is_ambiguous(self):
        export = self.exports['C']
        ambiguous = {r['coordinate']: r for r in export['ambiguous']}
        self.assertEqual(set(ambiguous), {'C1', 'C2'})
        for record in ambiguous.values():
            self.assertFalse(record['converted'])
            self.assertEqual(record['exported_value'], 0.1)
            self.assertEqual(record['original_value'], 0.1)
            self.assertEqual(record['unit'], 'ambiguous')
            self.assertIn('manual review', record['reason'])
        # The uniquely-formatted value in the same column still converts.
        converted = {r['coordinate']: r['exported_value']
                     for r in export['converted']}
        self.assertEqual(converted, {'C3': 20.0})

    def test_ambiguous_records_keep_coordinates_and_format_evidence(self):
        record = self._record('C', 'C1')
        self.assertEqual(record['xf_index'],
                         self.book.sheet_by_index(0).cell(0, 2).xf_index)
        self.assertEqual(record['format_str'], '0%')
        self.assertTrue(record['section_is_percent'])
        other = self._record('C', 'C2')
        self.assertFalse(other['section_is_percent'])

    def test_quoted_percent_unique_value_passes_through(self):
        record = self._record('D', 'D4')
        self.assertEqual(record['unit'], 'number')
        self.assertEqual(record['exported_value'], 0.3)
        self.assertFalse(record['section_is_percent'])

    def test_conditional_format_ambiguity_is_reported_with_coordinates(self):
        export = self.exports['E']
        ambiguous = {r['coordinate'] for r in export['ambiguous']}
        self.assertEqual(ambiguous, {'E1', 'E3'})
        converted = {r['coordinate']: r['exported_value']
                     for r in export['converted']}
        self.assertEqual(converted, {'E2': 0.5})

    def test_zero_section_ambiguity(self):
        export = self.exports['F']
        ambiguous = {r['coordinate'] for r in export['ambiguous']}
        self.assertEqual(ambiguous, {'F3', 'F4'})
        converted = sorted((r['coordinate'], r['exported_value'])
                           for r in export['converted'])
        self.assertEqual(converted, [('F1', 50.0), ('F2', -50.0)])

    def test_same_value_under_distinct_percent_xfs_still_converts(self):
        export = self.exports['G']
        self.assertEqual(len(export['ambiguous']), 0)
        records = {r['coordinate']: r for r in export['converted']}
        self.assertEqual(set(records), {'G1', 'G2'})
        self.assertTrue(all(r['unit'] == 'percent' for r in records.values()))
        self.assertTrue(all(r['exported_value'] == 10.0 for r in records.values()))
        # The two cells really do rely on distinct XF records.
        self.assertNotEqual(records['G1']['xf_index'], records['G2']['xf_index'])

    def test_export_without_formatting_info_is_empty(self):
        book = xlrd.open_workbook(from_sample('percent_columns.xls'))
        self.assertEqual(export_columns(book), [])

    def test_export_one_sheet_only(self):
        sheet = self.book.sheet_by_index(0)
        exports = export_columns(self.book, sheet)
        self.assertTrue(all(e['sheet_name'] == sheet.name for e in exports))
        self.assertEqual(
            sorted(e['column_letter'] for e in exports),
            ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        )


class TestModuleReExport(TestCase):

    def test_public_api_is_on_xlrd(self):
        self.assertIs(xlrd.parse_number_format, parse_number_format)
        self.assertIs(xlrd.export_columns, export_columns)
        self.assertIs(xlrd.cell_format_basis, cell_format_basis)

    def test_section_dict_construction_helpers(self):
        section = columnexport.NumberFormatSection(
            index=0, condition='all', format_text='0%',
            is_percent=True, explicit_condition=None)
        self.assertEqual(section['condition'], 'all')

    def test_result_dict_types_are_exported(self):
        self.assertIs(xlrd.CellFormatBasis, columnexport.CellFormatBasis)
        self.assertIs(xlrd.ColumnExport, columnexport.ColumnExport)
