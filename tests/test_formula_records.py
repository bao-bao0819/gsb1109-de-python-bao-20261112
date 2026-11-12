# -*- coding: utf-8 -*-
"""
Regression tests for the way ``Sheet.read`` consumes FORMULA result records.

A FORMULA record (BIFF opcode 0x0006/0x0206/0x0406) only carries the cached
result of the formula.  The 8-byte result field is either an IEEE-754 double,
or, when its last two bytes are ``0xFFFF``, a "special" value: a boolean, an
error code, an empty string, or a marker saying that the real text result
follows in a separate STRING record.

xlrd only ever sees these cached bytes, so on its own it cannot tell a
constant from a stale cache.  These tests therefore use two independent
sources of truth:

* the byte offset of every record in the synthesised stream, and
* the ``(rowx, colx)`` coordinates stored inside each cell record.

The fixtures deliberately put two adjacent formula cells whose cached text is
identical, so that attaching the wrong STRING record to the wrong FORMULA
would still "look right" by value alone.

Evidence per case (what can be proven, and from where):

==============================  ========================  ============================
Situation                       Provable value / source   Cache freshness
==============================  ========================  ============================
numeric FORMULA result          float, inline in FORMULA  unknown (no recalculation
                                                          evidence in the record)
boolean FORMULA result          0/1, special byte in      unknown
                                FORMULA result
error FORMULA result            error code, special byte  unknown
                                in FORMULA result
empty-string FORMULA result     '', special byte, no      unknown
                                STRING record
text FORMULA + trailing STRING  text, from the STRING     unknown; the FORMULA only
                                record that is the        promises a STRING, it does
                                FORMULA's immediate        not prove the text is fresh
                                successor
text FORMULA, ARRAY in between  text, STRING after the    unknown
                                skipped ARRAY
text FORMULA missing STRING     no value: parser raises   n/a
                                XLRDError instead of
                                grafting the neighbour
standalone BOOLERR              bool/error, from the      n/a (not a FORMULA cache)
                                BOOLERR record's own
                                coordinate
unreferenced coordinate         EMPTY, because no record  n/a
                                targets the coordinate
==============================  ========================  ============================

The freshness column is deliberately left unknown: nothing in a FORMULA or
STRING record proves the cached result still matches a recomputation of the
formula, so the tests never claim it.
"""

import io
from struct import Struct, pack
from unittest import TestCase

from xlrd.biffh import (
    XL_ARRAY,
    XL_BOOLERR,
    XL_CELL_BOOLEAN,
    XL_CELL_EMPTY,
    XL_CELL_ERROR,
    XL_CELL_NUMBER,
    XL_CELL_TEXT,
    XL_DIMENSION,
    XL_EOF,
    XL_FORMULA,
    XL_FORMULA3,
    XL_FORMULA4,
    XL_NUMBER,
    XL_STRING,
    XLRDError,
)
from xlrd.sheet import Sheet


SHARED_TEXT = 'shared text'

# Every FORMULA opcode routes through the same result-parsing branch.
FORMULA_OPCODES = (XL_FORMULA, XL_FORMULA3, XL_FORMULA4)


def record(opcode, body=b''):
    """Wrap a BIFF record body in its 4-byte header."""
    return pack('<HH', opcode, len(body)) + body


def formula_record(rowx, colx, result_bytes, flags=0, xf_index=15,
                   opcode=XL_FORMULA):
    """
    Build a FORMULA record.

    ``result_bytes`` is the raw 8-byte cached result: either a packed double
    or one of the special (boolean / error / empty-string / string) markers.
    """
    body = (
        pack('<HHH', rowx, colx, xf_index)
        + result_bytes
        + pack('<H', flags)
    )
    body += pack('<H', 0)  # cce: zero formula-token bytes follow
    return record(opcode, body)


def number_result(value):
    """8-byte numeric FORMULA result (no 0xFFFF marker in the tail)."""
    return pack('<d', value)


def special_result(first_byte, value=0):
    """
    8-byte special FORMULA result.

    Layout: type byte, reserved zero, value byte, reserved zero, then the
    0xFFFF tail that marks the whole field as a non-number.
    """
    return pack('<BBBBHH', first_byte, 0, value, 0, 0, 0xFFFF)


def string_record(text):
    """BIFF8 STRING record: char count, compression flag, character bytes."""
    chars = text.encode('latin_1')
    body = pack('<H', len(text)) + b'\x00' + chars  # 0x00 = compressed latin-1
    return record(XL_STRING, body)


def number_record(rowx, colx, value, xf_index=15):
    return record(XL_NUMBER, pack('<HHHd', rowx, colx, xf_index, value))


def boolerr_record(rowx, colx, value, is_err, xf_index=15):
    return record(XL_BOOLERR, pack('<HHHBB', rowx, colx, xf_index, value, is_err))


def array_record():
    body = pack('<HHBB', 1, 1, 0, 0)  # bounding range, values are not validated
    body += b'\x00'                   # array flags
    body += b'\x00' * 5               # reserved
    body += pack('<H', 0)             # token length
    return record(XL_ARRAY, body)


def dimension_record(nrows, ncols):
    # BIFF8 DIMENSIONS: 4 reserved bytes, then first/last row (int), pad,
    # first/last col (unsigned short).  xlrd unpacks body bytes [4:12].
    body = pack('<I', 0) + pack('<ixxH', nrows, ncols)
    return record(XL_DIMENSION, body)


def worksheet(cell_records, nrows, ncols):
    return [dimension_record(nrows, ncols)] + cell_records + [record(XL_EOF)]


def harness_offset(records, index):
    """Absolute byte offset of the record at ``index`` in ``records``."""
    return sum(len(chunk) for chunk in records[:index])


class FakeBook(object):
    """Minimal stand-in for :class:`xlrd.book.Book` for a single worksheet."""

    biff_version = 80
    encoding = 'latin_1'
    formatting_info = False
    ragged_rows = False
    verbosity = 0
    logfile = io.StringIO()
    _sheet_visibility = [0]

    def __init__(self, stream):
        self._stream = stream
        self._position = 0
        # XF slots default to "number", as a real Book's map does for the
        # slots reached by these fixtures.
        self._xf_index_to_xl_type_map = {0: XL_CELL_NUMBER, 15: XL_CELL_NUMBER}
        self.xf_list = []

    def get_record_parts(self):
        opcode, length = Struct('<HH').unpack(self._read(4))
        return opcode, length, self._read(length)

    def _read(self, size):
        chunk = self._stream[self._position:self._position + size]
        self._position += len(chunk)
        if len(chunk) != size:
            raise AssertionError('synthetic stream ended unexpectedly')
        return chunk


class FormulaRecordHarness(object):
    """
    Build a worksheet stream and parse it with the production ``Sheet.read``.

    ``expected_offsets`` maps a cell coordinate to the absolute byte offset at
    which its owning cell record begins in the stream; this is the coordinate
    half of the independent truth, checked against the offset-indexed layout.
    """

    def __init__(self, records, expected_offsets):
        self.expected_offsets = expected_offsets
        self.actual_offsets = self._index_record_offsets(records)
        stream = b''.join(records)
        book = FakeBook(stream)
        sheet = Sheet(book, position=0, name='fixture', number=0)
        sheet.read(book)
        self.sheet = sheet

    @staticmethod
    def _index_record_offsets(records):
        offsets = []
        position = 0
        for chunk in records:
            offsets.append(position)
            position += len(chunk)
        return offsets

    def cell(self, rowx, colx):
        return self.sheet.cell(rowx, colx)

    def assert_coordinate_offsets_are_distinct_records(self):
        seen = set()
        for coordinate, offset in self.expected_offsets.items():
            assert offset in self.actual_offsets, (coordinate, offset)
            assert offset not in seen, (coordinate, offset)
            seen.add(offset)
        assert len(seen) == len(self.expected_offsets)


class FormulaCachedResultTests(TestCase):

    def test_numeric_formula_cache_is_read_as_number(self):
        # Every FORMULA opcode reaches the same result-parsing branch.
        for opcode in FORMULA_OPCODES:
            formula = formula_record(
                0, 0, number_result(3.25), opcode=opcode
            )
            records = worksheet([formula], nrows=1, ncols=1)
            harness = FormulaRecordHarness(
                records, expected_offsets={(0, 0): len(records[0])}
            )
            # Coordinate truth and offset truth name the same physical
            # FORMULA record; the inline double needs no trailing STRING.
            self.assertEqual(harness.expected_offsets[(0, 0)],
                             harness.actual_offsets[1])
            cell = harness.cell(0, 0)
            self.assertEqual(cell.ctype, XL_CELL_NUMBER)
            self.assertEqual(cell.value, 3.25)

    def test_boolean_formula_cache_is_read_as_boolean(self):
        formula = formula_record(0, 0, special_result(1, value=1))
        records = worksheet([formula], nrows=1, ncols=1)
        harness = FormulaRecordHarness(
            records, expected_offsets={(0, 0): len(records[0])}
        )
        cell = harness.cell(0, 0)
        self.assertEqual(cell.ctype, XL_CELL_BOOLEAN)
        self.assertEqual(cell.value, 1)

    def test_error_formula_cache_is_read_as_error(self):
        formula = formula_record(0, 0, special_result(2, value=0x07))  # #DIV/0!
        records = worksheet([formula], nrows=1, ncols=1)
        harness = FormulaRecordHarness(
            records, expected_offsets={(0, 0): len(records[0])}
        )
        cell = harness.cell(0, 0)
        self.assertEqual(cell.ctype, XL_CELL_ERROR)
        self.assertEqual(cell.value, 0x07)

    def test_empty_string_formula_cache_is_text_without_string_record(self):
        # first_byte == 3 means a cached zero-length string; no STRING record.
        formula = formula_record(0, 0, special_result(3))
        records = worksheet([formula], nrows=1, ncols=1)
        harness = FormulaRecordHarness(
            records, expected_offsets={(0, 0): len(records[0])}
        )
        cell = harness.cell(0, 0)
        self.assertEqual(cell.ctype, XL_CELL_TEXT)
        self.assertEqual(cell.value, '')

    def test_string_record_is_attached_to_its_own_formula(self):
        first = formula_record(0, 0, special_result(0))
        first_string = string_record(SHARED_TEXT)
        second = formula_record(0, 1, special_result(0))
        second_string = string_record(SHARED_TEXT)
        records = worksheet(
            [first, first_string, second, second_string], nrows=1, ncols=2
        )
        offsets = {
            (0, 0): harness_offset(records, 1),
            (0, 1): harness_offset(records, 3),
        }
        harness = FormulaRecordHarness(records, offsets)
        harness.assert_coordinate_offsets_are_distinct_records()
        left = harness.cell(0, 0)
        right = harness.cell(0, 1)
        # Both cells legitimately hold the same cached text.
        self.assertEqual(left.ctype, XL_CELL_TEXT)
        self.assertEqual(right.ctype, XL_CELL_TEXT)
        self.assertEqual(left.value, SHARED_TEXT)
        self.assertEqual(right.value, SHARED_TEXT)
        # Adjacency truth: each STRING record begins at the exact byte right
        # after its own FORMULA, so binding follows record order/coordinates,
        # not a value comparison that could not tell the two texts apart.
        self.assertEqual(harness.actual_offsets[2],
                         harness.actual_offsets[1] + len(first))
        self.assertEqual(harness.actual_offsets[4],
                         harness.actual_offsets[3] + len(second))
        self.assertNotEqual(harness.actual_offsets[2],
                            harness.actual_offsets[4])

    def test_array_formula_skips_array_record_before_string(self):
        # An ARRAY record may sit between the FORMULA and its STRING record.
        formula = formula_record(0, 0, special_result(0))
        array = array_record()
        text = string_record(SHARED_TEXT)
        records = worksheet([formula, array, text], nrows=1, ncols=1)
        harness = FormulaRecordHarness(
            records, expected_offsets={(0, 0): harness_offset(records, 1)}
        )
        cell = harness.cell(0, 0)
        self.assertEqual(cell.ctype, XL_CELL_TEXT)
        self.assertEqual(cell.value, SHARED_TEXT)
        # Offset truth: record 2 really is the intermediate ARRAY, sitting
        # immediately after the FORMULA and before the STRING at record 3.
        self.assertEqual(harness.actual_offsets[2],
                         harness.expected_offsets[(0, 0)] + len(formula))

    def test_missing_trailing_string_does_not_consume_neighbour(self):
        # First FORMULA promises a STRING (first_byte 0) but none is written.
        # The next record belongs to a different coordinate; the parser must
        # fail rather than graft the neighbour's result onto the first cell.
        broken_formula = formula_record(0, 0, special_result(0))
        neighbour = number_record(0, 1, 9.5)
        records = worksheet([broken_formula, neighbour], nrows=1, ncols=2)
        offsets = {
            (0, 0): harness_offset(records, 1),
            (0, 1): harness_offset(records, 2),
        }
        with self.assertRaises(XLRDError):
            FormulaRecordHarness(records, offsets)

    def test_missing_string_before_identical_neighbour_is_not_silently_joined(self):
        # Same hazard, but the neighbour is itself a text FORMULA whose cached
        # text equals what the broken cell would have produced.  Comparing
        # values alone could not detect the mis-join; coordinates and record
        # offsets make it unambiguous, and the parser must still reject it.
        broken_formula = formula_record(0, 0, special_result(0))
        neighbour_formula = formula_record(0, 1, special_result(0))
        neighbour_string = string_record(SHARED_TEXT)
        records = worksheet(
            [broken_formula, neighbour_formula, neighbour_string],
            nrows=1,
            ncols=2,
        )
        offsets = {
            (0, 0): harness_offset(records, 1),
            (0, 1): harness_offset(records, 2),
        }
        with self.assertRaises(XLRDError):
            FormulaRecordHarness(records, offsets)

    def test_boolerr_record_is_independent_of_formula_cache(self):
        # A standalone BOOLERR cell carries its value in its own record,
        # not via a FORMULA cache, and lands at its own coordinate offset.
        boolean = boolerr_record(0, 0, 1, is_err=0)
        error = boolerr_record(0, 1, 0x2A, is_err=1)  # #N/A
        records = worksheet([boolean, error], nrows=1, ncols=2)
        offsets = {
            (0, 0): harness_offset(records, 1),
            (0, 1): harness_offset(records, 2),
        }
        harness = FormulaRecordHarness(records, offsets)
        harness.assert_coordinate_offsets_are_distinct_records()
        self.assertEqual(harness.cell(0, 0).ctype, XL_CELL_BOOLEAN)
        self.assertEqual(harness.cell(0, 0).value, 1)
        self.assertEqual(harness.cell(0, 1).ctype, XL_CELL_ERROR)
        self.assertEqual(harness.cell(0, 1).value, 0x2A)

    def test_unreferenced_cell_remains_empty(self):
        # Only (1, 1) holds a record, so the 2x2 grid has empty gap cells.
        # An unreferenced coordinate such as (0, 0) must stay empty rather
        # than inherit the neighbour formula's cached value.
        formula = formula_record(1, 1, number_result(7.0))
        records = worksheet([formula], nrows=2, ncols=2)
        harness = FormulaRecordHarness(
            records, expected_offsets={(1, 1): harness_offset(records, 1)}
        )
        self.assertEqual(harness.cell(0, 0).ctype, XL_CELL_EMPTY)
        self.assertEqual(harness.cell(0, 0).value, '')
        self.assertEqual(harness.cell(1, 1).ctype, XL_CELL_NUMBER)
        self.assertEqual(harness.cell(1, 1).value, 7.0)
