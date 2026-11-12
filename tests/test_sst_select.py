# Tests for selective reading of the shared string table (SST).
#
# Covers:
# * unpack_SST_table(wanted=...) returning wanted strings at their original
#   SST indexes and None for skipped strings (which are never decoded);
# * CONTINUE record boundaries falling in character data, in rich-text
#   formatting runs, in phonetic data, and exactly between two strings;
# * the sst_select argument of open_workbook, both as a set of SST indexes
#   and as a {sheet: columns} mapping;
# * stability of business output and of the decoded character count when
#   large amounts of unselected text are added.

import struct
from unittest import TestCase

import xlrd
from xlrd.book import unpack_SST_table
from xlrd.timemachine import UNICODE_LITERAL


def make_entry(text, rich_runs=(), phonetic=b'', wide=False):
    "Build one XLUnicodeRichExtendedString (as stored in the SST)."
    options = 0
    if wide:
        encoded = text.encode('utf-16-le')
        options |= 0x01
    else:
        encoded = text.encode('latin-1')
    if rich_runs:
        options |= 0x08
    if phonetic:
        options |= 0x04
    out = [struct.pack('<HB', len(text), options)]
    if rich_runs:
        out.append(struct.pack('<H', len(rich_runs)))
    if phonetic:
        out.append(struct.pack('<i', len(phonetic)))
    out.append(encoded)
    for ifrt, ifnt in rich_runs:
        out.append(struct.pack('<HH', ifrt, ifnt))
    out.append(phonetic)
    return b''.join(out)


def sst_header(nstrings):
    "SST record payload header: total and unique string counts."
    return struct.pack('<ii', nstrings, nstrings)


class TestUnpackSSTTableSelective(TestCase):

    def test_full_decode_unchanged(self):
        entries = [
            make_entry(UNICODE_LITERAL('alpha')),
            make_entry(UNICODE_LITERAL('wide16'), wide=True),
            make_entry(UNICODE_LITERAL('rich'), rich_runs=[(0, 1), (2, 3)]),
            make_entry(UNICODE_LITERAL('ph'), phonetic=b'\x01\x02\x03\x04'),
        ]
        datatab = [sst_header(4) + b''.join(entries)]
        strings, rt_runs = unpack_SST_table(datatab, 4)
        self.assertEqual(strings, ['alpha', 'wide16', 'rich', 'ph'])
        self.assertEqual(rt_runs, {2: [(0, 1), (2, 3)]})

    def test_selective_decode_and_stats(self):
        entries = [
            make_entry(UNICODE_LITERAL('alpha')),
            make_entry(UNICODE_LITERAL('beta-long-skipped')),
            make_entry(UNICODE_LITERAL('gamma')),
        ]
        datatab = [sst_header(3) + b''.join(entries)]
        stats = {}
        strings, rt_runs = unpack_SST_table(datatab, 3, wanted={0, 2}, stats=stats)
        # Wanted strings are returned at their original SST indexes.
        self.assertEqual(strings, ['alpha', None, 'gamma'])
        self.assertEqual(rt_runs, {})
        self.assertEqual(stats['decoded_chars'], len('alpha') + len('gamma'))
        self.assertEqual(stats['skipped_chars'], len('beta-long-skipped'))

    def test_skipped_strings_are_never_decoded(self):
        entries = [
            make_entry(UNICODE_LITERAL('wanted')),
            make_entry(UNICODE_LITERAL('x') * 10000),
            make_entry(UNICODE_LITERAL('y') * 20000, wide=True),
        ]
        datatab = [sst_header(3) + b''.join(entries)]
        decoded = []
        real_unicode = xlrd.book.unicode
        def counting_unicode(raw, encoding):
            decoded.append(len(raw))
            return real_unicode(raw, encoding)
        xlrd.book.unicode = counting_unicode
        try:
            strings, _ = unpack_SST_table(datatab, 3, wanted={0})
        finally:
            xlrd.book.unicode = real_unicode
        self.assertEqual(strings[0], 'wanted')
        self.assertEqual(strings[1:], [None, None])
        # Only the wanted string was turned into a Python string.
        self.assertEqual(decoded, [len('wanted')])

    def test_continue_in_char_data_with_encoding_switch(self):
        # First record ends after 2 compressed characters; the CONTINUE
        # record switches to 16-bit characters for the rest of the string.
        cjk = UNICODE_LITERAL('中文')
        rec0 = sst_header(2) + struct.pack('<HB', 4, 0x00) + b'ab'
        rec1 = b'\x01' + cjk.encode('utf-16-le') + make_entry(UNICODE_LITERAL('tail'))
        strings, _ = unpack_SST_table([rec0, rec1], 2)
        self.assertEqual(strings, ['ab' + cjk, 'tail'])
        # The first string spans a record boundary; skipping it must not
        # misalign the following string.
        stats = {}
        strings, _ = unpack_SST_table([rec0, rec1], 2, wanted={1}, stats=stats)
        self.assertEqual(strings, [None, 'tail'])
        self.assertEqual(stats['decoded_chars'], 4)
        self.assertEqual(stats['skipped_chars'], 4)

    def test_continue_in_wide_char_data(self):
        entry0 = make_entry(UNICODE_LITERAL('wxyz'), wide=True)
        # Split the UTF-16 character data after the first 2 characters.
        cut = 8 + 3 + 4
        rec0 = (sst_header(2) + entry0)[:cut]
        rest = (sst_header(2) + entry0)[cut:]
        rec1 = b'\x01' + rest + make_entry(UNICODE_LITERAL('end'))
        strings, _ = unpack_SST_table([rec0, rec1], 2)
        self.assertEqual(strings, ['wxyz', 'end'])

    def test_continue_in_rich_text_runs(self):
        entry0 = make_entry(UNICODE_LITERAL('rich'), rich_runs=[(0, 1), (2, 3)])
        entry1 = make_entry(UNICODE_LITERAL('next'))
        payload = sst_header(2) + entry0 + entry1
        # Split 2 bytes into the second 4-byte formatting run.
        cut = 8 + 3 + 2 + 4 + 4 + 2
        rec0, rec1 = payload[:cut], payload[cut:]
        strings, rt_runs = unpack_SST_table([rec0, rec1], 2)
        self.assertEqual(strings, ['rich', 'next'])
        self.assertEqual(rt_runs, {0: [(0, 1), (2, 3)]})
        # Skipping the rich-text string still leaves the next string aligned.
        strings, rt_runs = unpack_SST_table([rec0, rec1], 2, wanted={1})
        self.assertEqual(strings, [None, 'next'])
        self.assertEqual(rt_runs, {})

    def test_continue_in_phonetic_data(self):
        entry0 = make_entry(UNICODE_LITERAL('ph'), phonetic=b'ABCDEF')
        entry1 = make_entry(UNICODE_LITERAL('after'))
        payload = sst_header(2) + entry0 + entry1
        # Split 4 bytes into the 6 bytes of phonetic data.
        cut = 8 + 3 + 4 + 2 + 4
        rec0, rec1 = payload[:cut], payload[cut:]
        strings, _ = unpack_SST_table([rec0, rec1], 2)
        self.assertEqual(strings, ['ph', 'after'])
        strings, _ = unpack_SST_table([rec0, rec1], 2, wanted={1})
        self.assertEqual(strings, [None, 'after'])

    def test_continue_exactly_between_strings(self):
        entry0 = make_entry(UNICODE_LITERAL('first'))
        entry1 = make_entry(UNICODE_LITERAL('second'))
        rec0 = sst_header(2) + entry0
        rec1 = entry1
        strings, _ = unpack_SST_table([rec0, rec1], 2)
        self.assertEqual(strings, ['first', 'second'])
        strings, _ = unpack_SST_table([rec0, rec1], 2, wanted={1})
        self.assertEqual(strings, [None, 'second'])


# --- end-to-end tests with a hand-built BIFF8 workbook stream ---

def rec(code, data=b''):
    return struct.pack('<HH', code, len(data)) + data


def bof_record(stream_type):
    return rec(0x0809, struct.pack('<HHHHII', 0x0600, stream_type, 0x0DBB, 0x07CC, 0, 0))


def boundsheet_record(offset, name):
    encoded = name.encode('latin-1')
    data = struct.pack('<iBB', offset, 0, 0) + bytes([len(encoded), 0]) + encoded
    return rec(0x0085, data)


def labelsst_record(rowx, colx, sstindex):
    return rec(0x00FD, struct.pack('<HHHi', rowx, colx, 0, sstindex))


EOF_RECORD = rec(0x000A)


def build_workbook(cells, sst_chunks):
    """
    Build a minimal BIFF8 workbook stream.

    cells: list of (rowx, colx, sstindex) LABELSST cells.
    sst_chunks: list of SST/CONTINUE record payloads covering the
        8-byte SST header plus all string entries.
    """
    sheet_stream = b''.join([
        bof_record(0x0010),
        rec(0x0200, struct.pack('<iiHHH', 0, 10, 0, 10, 0)),
        b''.join(labelsst_record(*cell) for cell in cells),
        EOF_RECORD,
    ])
    globals_stream = b''.join([
        bof_record(0x0005),
        boundsheet_record(0, 'Sheet1'),  # offset patched below
        b''.join(rec(0x00FC if i == 0 else 0x003C, chunk)
                 for i, chunk in enumerate(sst_chunks)),
        EOF_RECORD,
    ])
    sheet_offset = len(globals_stream)
    globals_stream = b''.join([
        bof_record(0x0005),
        boundsheet_record(sheet_offset, 'Sheet1'),
        b''.join(rec(0x00FC if i == 0 else 0x003C, chunk)
                 for i, chunk in enumerate(sst_chunks)),
        EOF_RECORD,
    ])
    assert len(globals_stream) == sheet_offset
    return globals_stream + sheet_stream


# Selected business columns: 0 and 2. Column 1 holds unselected text.
SELECTED = {
    0: UNICODE_LITERAL('name-0'),
    2: UNICODE_LITERAL('note-0'),
}
LONG_UNSELECTED = UNICODE_LITERAL('LONG-') + 'y' * 250


def make_test_workbook(n_extra):
    entry0 = make_entry(SELECTED[0])
    entry1 = make_entry(LONG_UNSELECTED)
    entry2 = make_entry(SELECTED[2])
    extras = [make_entry(UNICODE_LITERAL('X') * (1000 + i)) for i in range(n_extra)]
    # The long unselected string spans a CONTINUE boundary: the first
    # record ends 100 bytes into its character data, and the CONTINUE
    # record (which starts with an option flags byte) carries the rest
    # plus all remaining entries.
    e1_hdr_len = 3
    rec0 = sst_header(3 + n_extra) + entry0 + entry1[:e1_hdr_len + 100]
    rec1 = b'\x00' + entry1[e1_hdr_len + 100:] + entry2 + b''.join(extras)
    cells = [
        (0, 0, 0),
        (0, 1, 1),
        (0, 2, 2),
    ] + [
        (1 + i, 1, 3 + i) for i in range(n_extra)
    ]
    return build_workbook(cells, [rec0, rec1])


class TestSelectiveSSTOpenWorkbook(TestCase):

    def check_business_output(self, book):
        sheet = book.sheet_by_index(0)
        for colx, expected in SELECTED.items():
            self.assertEqual(sheet.cell_value(0, colx), expected)
            self.assertEqual(sheet.cell_type(0, colx), xlrd.XL_CELL_TEXT)
        # Unselected column reads as empty.
        self.assertEqual(sheet.cell_type(0, 1), xlrd.XL_CELL_EMPTY)

    def test_select_by_columns(self):
        book = xlrd.open_workbook(
            file_contents=make_test_workbook(0),
            sst_select={0: {0, 2}},
        )
        self.check_business_output(book)
        self.assertEqual(
            book._sst_stats['decoded_chars'],
            len(SELECTED[0]) + len(SELECTED[2]),
        )
        self.assertEqual(book._sst_stats['skipped_chars'], len(LONG_UNSELECTED))

    def test_select_by_columns_with_sheet_name(self):
        book = xlrd.open_workbook(
            file_contents=make_test_workbook(0),
            sst_select={'Sheet1': [0, 2]},
        )
        self.check_business_output(book)

    def test_select_by_sst_indexes(self):
        book = xlrd.open_workbook(
            file_contents=make_test_workbook(0),
            sst_select={0, 2},
        )
        self.check_business_output(book)

    def test_default_decodes_everything(self):
        book = xlrd.open_workbook(file_contents=make_test_workbook(3))
        sheet = book.sheet_by_index(0)
        self.assertEqual(sheet.cell_value(0, 1), LONG_UNSELECTED)
        self.assertEqual(sheet.cell_value(2, 1), 'X' * 1001)
        self.assertEqual(
            book._sst_stats['decoded_chars'],
            len(SELECTED[0]) + len(SELECTED[2]) + len(LONG_UNSELECTED)
            + 1000 + 1001 + 1002,
        )

    def test_output_and_decode_count_stable_with_extra_unselected(self):
        outputs = []
        decoded_counts = []
        for n_extra in (0, 5, 50):
            book = xlrd.open_workbook(
                file_contents=make_test_workbook(n_extra),
                sst_select={0: {0, 2}},
            )
            sheet = book.sheet_by_index(0)
            outputs.append([sheet.cell_value(0, colx) for colx in (0, 2)])
            decoded_counts.append(book._sst_stats['decoded_chars'])
        # Adding lots of unselected text changes neither the business
        # output nor the amount of decoding performed.
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])
        self.assertEqual(decoded_counts[0], decoded_counts[1])
        self.assertEqual(decoded_counts[1], decoded_counts[2])
        self.assertEqual(decoded_counts[0], len(SELECTED[0]) + len(SELECTED[2]))
