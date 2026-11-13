# Copyright (c) 2026
# This module is part of the xlrd package, which is released under a
# BSD-style licence.
"""
Partition-wise financial summarisation on top of :mod:`xlrd`.

The module answers one question -- "what is the total of this amount
column?" -- without lying when the answer is not fully knowable:

* Numeric cells inside the configured ``lower``/``upper`` bounds contribute
  to the total; valid zeroes are counted separately.
* Error, blank/empty, text, date and boolean cells can never be mistaken
  for amounts.  Each one is booked individually as an *unresolved* source
  row, so the report always states which rows could not be valued.
* Rows that are known to describe the same original transaction (they
  share a key column value) are counted once.  Conflicting duplicate
  values do not raise: the transaction gets a possible amount range whose
  endpoints carry the source cells that witness them.
* Partition summaries can be merged again.  The merge recognises the same
  transaction across partitions instead of blindly adding two ranges.
"""

from __future__ import absolute_import

import json

from .biffh import (
    XL_CELL_BLANK, XL_CELL_BOOLEAN, XL_CELL_DATE, XL_CELL_EMPTY,
    XL_CELL_ERROR, XL_CELL_NUMBER, XL_CELL_TEXT, error_text_from_code,
)
from .book import colname
from .timemachine import xrange

#: ``(str, unicode)`` on Python 2, ``(str,)`` on Python 3.
string_types = (str, type(u''))

#: Reasons why a source row's amount could not be determined.
REASON_ERROR = 'error'
REASON_EMPTY = 'empty'
REASON_BLANK = 'blank'
REASON_TEXT = 'text'
REASON_DATE = 'date'
REASON_BOOLEAN = 'boolean'
REASON_OUT_OF_BOUNDS = 'out_of_bounds'
REASON_KEY_MISSING = 'key_missing'

_TYPE_NAMES = {
    XL_CELL_EMPTY: 'empty',
    XL_CELL_TEXT: 'text',
    XL_CELL_NUMBER: 'number',
    XL_CELL_DATE: 'date',
    XL_CELL_BOOLEAN: 'boolean',
    XL_CELL_ERROR: 'error',
    XL_CELL_BLANK: 'blank',
}


class CellRef(object):
    """Reference to one source cell inside a workbook partition."""

    __slots__ = (
        'partition', 'sheet', 'rowx', 'colx', 'ctype', 'value',
        'error_text',
    )

    def __init__(self, partition, sheet, rowx, colx, ctype, value):
        self.partition = partition
        self.sheet = sheet
        self.rowx = rowx
        self.colx = colx
        self.ctype = ctype
        self.value = value
        self.error_text = (
            error_text_from_code.get(value, '#UNKNOWN!')
            if ctype == XL_CELL_ERROR else None
        )

    @property
    def cell_type_name(self):
        return _TYPE_NAMES.get(self.ctype, 'unknown')

    @property
    def a1(self):
        """A1-style coordinate, e.g. ``B7`` (row indices are 1-based here)."""
        return '%s%d' % (colname(self.colx), self.rowx + 1)

    @property
    def location(self):
        """Human-readable location, e.g. ``P1/Sheet1!B7``."""
        return '%s/%s!%s' % (self.partition, self.sheet, self.a1)

    @property
    def identity(self):
        return (self.partition, self.sheet, self.rowx, self.colx)

    def as_dict(self):
        return {
            'partition': self.partition,
            'sheet': self.sheet,
            'rowx': self.rowx,
            'colx': self.colx,
            'a1': self.a1,
            'location': self.location,
            'cell_type': self.cell_type_name,
            'value': self.value,
            'error_text': self.error_text,
        }

    def __repr__(self):
        return 'CellRef(%r)' % self.location


class AmountRange(object):
    """
    Possible amount of one transaction or one total.

    ``low`` and ``high`` are inclusive bounds.  An exact amount is a range
    with ``low == high``.  ``witness_low``/``witness_high`` list the source
    cells that attain each endpoint -- essential when duplicate rows
    contradict each other.
    """

    __slots__ = ('low', 'high', 'witness_low', 'witness_high')

    def __init__(self, low, high, witness_low=None, witness_high=None):
        if low > high:
            raise ValueError(
                'AmountRange low (%r) is greater than high (%r)' % (low, high)
            )
        self.low = low
        self.high = high
        self.witness_low = list(witness_low or [])
        self.witness_high = list(witness_high or [])

    @classmethod
    def point(cls, value, witnesses=None):
        """An exact amount, i.e. a degenerate range ``[value, value]``."""
        return cls(value, value, witnesses, witnesses)

    @property
    def exact(self):
        return self.low == self.high

    def __add__(self, other):
        # Interval addition: used to build partition totals from
        # independent transactions.
        return AmountRange(
            self.low + other.low,
            self.high + other.high,
            self.witness_low + other.witness_low,
            self.witness_high + other.witness_high,
        )

    def as_dict(self):
        return {
            'low': self.low,
            'high': self.high,
            'exact': self.exact,
            'witness_low': [ref.as_dict() for ref in self.witness_low],
            'witness_high': [ref.as_dict() for ref in self.witness_high],
        }

    def __eq__(self, other):
        if not isinstance(other, AmountRange):
            return NotImplemented
        return self.low == other.low and self.high == other.high

    def __repr__(self):
        if self.exact:
            return 'AmountRange(exact=%r)' % self.low
        return 'AmountRange(low=%r, high=%r)' % (self.low, self.high)


class UnresolvedEntry(object):
    """A source row whose amount could not be determined."""

    __slots__ = ('ref', 'reason', 'detail')

    def __init__(self, ref, reason, detail=None):
        self.ref = ref
        self.reason = reason
        self.detail = detail

    def as_dict(self):
        return {
            'reason': self.reason,
            'detail': self.detail,
            'cell': self.ref.as_dict(),
        }

    def __repr__(self):
        return 'UnresolvedEntry(%r, %r)' % (self.reason, self.ref.location)


class TransactionGroup(object):
    """
    One original transaction, possibly backed by several duplicate rows.
    """

    __slots__ = (
        'token', 'key', 'identifiable', 'members', 'amount_range',
        'unresolved_members', 'contradiction',
    )

    def __init__(self, token, key, identifiable, members, amount_range,
                 unresolved_members, contradiction):
        #: internal hashable grouping token (unique for keyless rows)
        self.token = token
        #: display value of the shared key (``None`` for keyless rows)
        self.key = key
        #: whether the transaction can be recognised again on merge
        self.identifiable = identifiable
        #: every source :class:`CellRef` carrying this transaction
        self.members = members
        #: :class:`AmountRange` or ``None`` if no numeric witness exists
        self.amount_range = amount_range
        #: :class:`UnresolvedEntry` objects for non-numeric duplicate rows
        self.unresolved_members = unresolved_members
        #: True when numeric witnesses disagree beyond tolerance
        self.contradiction = contradiction

    @property
    def determined(self):
        return self.amount_range is not None

    @property
    def exact(self):
        return self.determined and self.amount_range.exact

    @property
    def fully_known(self):
        return self.exact and not self.unresolved_members

    def as_dict(self):
        return {
            'key': self.key,
            'identifiable': self.identifiable,
            'determined': self.determined,
            'exact': self.exact,
            'fully_known': self.fully_known,
            'contradiction': self.contradiction,
            'amount_range': (
                self.amount_range.as_dict()
                if self.amount_range is not None else None
            ),
            'members': [ref.as_dict() for ref in self.members],
            'unresolved_members': [
                entry.as_dict() for entry in self.unresolved_members
            ],
        }

    def __repr__(self):
        return 'TransactionGroup(key=%r, range=%r)' % (
            self.key, self.amount_range,
        )


def _resolve_column(sheet, column, header_rows):
    """Accept an integer column index or a header name (str)."""
    if isinstance(column, int):
        if column < 0:
            raise ValueError('column index must be >= 0, got %r' % column)
        return column
    if isinstance(column, string_types) and header_rows > 0:
        header_rowx = header_rows - 1
        for colx in xrange(sheet.row_len(header_rowx)):
            if sheet.cell_value(header_rowx, colx) == column:
                return colx
        raise ValueError(
            'column name %r not found in header row %d of sheet %r'
            % (column, header_rowx, sheet.name)
        )
    raise ValueError(
        'column must be an integer index, or a header name when '
        'header_rows > 0; got %r' % (column,)
    )


def _build_group(token, key, identifiable, records, tolerance):
    """
    Turn the raw records of one key into a :class:`TransactionGroup`.

    Each record is ``(cell_ref, numeric_value_or_None, unresolved_or_None)``.
    """
    members = [record[0] for record in records]
    unresolved_members = [
        record[2] for record in records if record[2] is not None
    ]
    numeric = [
        (record[0], record[1]) for record in records
        if record[1] is not None
    ]
    amount_range = None
    contradiction = False
    if numeric:
        values = [value for _, value in numeric]
        low_value = min(values)
        high_value = max(values)
        if high_value - low_value <= tolerance:
            # Duplicates agree (or a single witness): exact amount,
            # represented by the first numeric witness.
            first_ref = numeric[0][0]
            amount_range = AmountRange.point(low_value, [first_ref])
        else:
            contradiction = True
            witness_low = [
                ref for ref, value in numeric if value == low_value
            ]
            witness_high = [
                ref for ref, value in numeric if value == high_value
            ]
            amount_range = AmountRange(
                low_value, high_value, witness_low, witness_high,
            )
    return TransactionGroup(
        token, key, identifiable, members, amount_range,
        unresolved_members, contradiction,
    )


class PartitionSummary(object):
    """
    Result of summarising one partition (a sheet, a row range or a book).

    Totals are exposed as an :class:`AmountRange`: exact when every
    transaction is fully known, otherwise a possible interval.  Rows whose
    amount could not be determined are listed in :attr:`unresolved` and
    counted by type so nothing disappears silently.
    """

    __slots__ = (
        'partition', 'transactions', 'unresolved', 'key_issues',
        'out_of_bounds', 'rows_scanned', 'error_cells', 'blank_cells',
        'empty_cells', 'zero_cells', 'duplicate_rows_dropped',
        'lower', 'upper', 'tolerance',
    )

    def __init__(self, partition, lower=None, upper=None, tolerance=0.0):
        self.partition = partition
        self.transactions = []
        self.unresolved = []
        self.key_issues = []
        self.out_of_bounds = []
        self.rows_scanned = 0
        self.error_cells = 0
        self.blank_cells = 0
        self.empty_cells = 0
        self.zero_cells = 0
        self.duplicate_rows_dropped = 0
        self.lower = lower
        self.upper = upper
        self.tolerance = tolerance

    @property
    def contradictory_groups(self):
        return [
            group for group in self.transactions if group.contradiction
        ]

    @property
    def amount_range(self):
        """
        Total possible amount over the *determined* transactions.

        Undetermined rows are never guessed; inspect :attr:`unresolved`
        to see what may widen the real total.
        """
        total = AmountRange.point(0.0)
        for group in self.transactions:
            if group.determined:
                total = total + group.amount_range
        return total

    @property
    def total_low(self):
        return self.amount_range.low

    @property
    def total_high(self):
        return self.amount_range.high

    @property
    def exact_total(self):
        """Exact total number, or ``None`` if the total is an interval."""
        result = self.amount_range
        return result.low if result.exact else None

    @property
    def fully_determined(self):
        if self.unresolved:
            return False
        return all(group.fully_known for group in self.transactions)

    def merge(self, other, partition=None):
        """Merge with another summary (see :func:`merge_summaries`)."""
        return merge_summaries(
            [self, other],
            partition=partition or 'merged',
        )

    def as_dict(self):
        return {
            'partition': self.partition,
            'bounds': {'lower': self.lower, 'upper': self.upper},
            'tolerance': self.tolerance,
            'rows_scanned': self.rows_scanned,
            'totals': {
                'low': self.total_low,
                'high': self.total_high,
                'exact_total': self.exact_total,
                'fully_determined': self.fully_determined,
            },
            'counts': {
                'error_cells': self.error_cells,
                'blank_cells': self.blank_cells,
                'empty_cells': self.empty_cells,
                'zero_cells': self.zero_cells,
                'out_of_bounds': len(self.out_of_bounds),
                'duplicate_rows_dropped': self.duplicate_rows_dropped,
                'contradictory_transactions': len(self.contradictory_groups),
                'unresolved_rows': len(self.unresolved),
            },
            'transactions': [
                group.as_dict() for group in self.transactions
            ],
            'unresolved': [entry.as_dict() for entry in self.unresolved],
            'key_issues': [entry.as_dict() for entry in self.key_issues],
            'out_of_bounds': [ref.as_dict() for ref in self.out_of_bounds],
        }

    def to_json(self, **kwargs):
        return json.dumps(self.as_dict(), **kwargs)

    def report_text(self):
        """Plain-text report suitable for logs or console output."""
        lines = []
        lines.append('Partition: %s' % self.partition)
        if self.exact_total is not None:
            lines.append('Total (exact): %s' % self.exact_total)
        else:
            lines.append(
                'Total (possible range): %s .. %s'
                % (self.total_low, self.total_high)
            )
        lines.append(
            'Rows scanned: %d | zero cells: %d | error cells: %d | '
            'blank cells: %d | empty cells: %d'
            % (self.rows_scanned, self.zero_cells, self.error_cells,
               self.blank_cells, self.empty_cells)
        )
        lines.append(
            'Duplicate rows counted once: %d | contradictory transactions: '
            '%d | unresolved rows: %d'
            % (self.duplicate_rows_dropped,
               len(self.contradictory_groups), len(self.unresolved))
        )
        for group in self.contradictory_groups:
            amount_range = group.amount_range
            lines.append(
                '  Contradiction key=%r: %s .. %s'
                % (group.key, amount_range.low, amount_range.high)
            )
            for ref in amount_range.witness_low:
                lines.append('    low  witness: %s = %s'
                             % (ref.location, ref.value))
            for ref in amount_range.witness_high:
                lines.append('    high witness: %s = %s'
                             % (ref.location, ref.value))
        for entry in self.unresolved:
            lines.append('  Unresolved [%s]: %s (cell type=%s, value=%r)'
                         % (entry.reason, entry.ref.location,
                            entry.ref.cell_type_name, entry.ref.value))
        return '\n'.join(lines)

    def __repr__(self):
        return 'PartitionSummary(partition=%r, total=%r)' % (
            self.partition, self.amount_range,
        )


def summarize_sheet(sheet, amount_col, key_col=None, lower=None,
                    upper=None, header_rows=0, row_range=None,
                    partition=None, tolerance=0.0):
    """
    Summarise one :class:`~xlrd.sheet.Sheet` (or sheet-like object).

    :param sheet: an xlrd sheet; any object exposing ``name``, ``nrows``,
        ``row_len``, ``cell_type`` and ``cell_value`` works too.
    :param amount_col: index of the amount column, or its header name when
        ``header_rows`` is given.
    :param key_col: index of the column identifying the original
        transaction. Rows sharing a key value are treated as known
        duplicates of one transaction. ``None`` disables deduplication.
    :param lower: inclusive lower bound for acceptable amounts.
    :param upper: inclusive upper bound for acceptable amounts.
    :param header_rows: number of leading rows to skip (and to search for
        column names).
    :param row_range: optional ``(start_rowx, end_rowx)`` absolute row
        range; useful for splitting one sheet into several partitions.
    :param partition: name of the partition used in cell references;
        defaults to the sheet name.
    :param tolerance: maximum difference for duplicate numeric values to be
        considered agreeing (default ``0.0``).
    :rtype: :class:`PartitionSummary`
    """
    if lower is not None and upper is not None and lower > upper:
        raise ValueError('lower (%r) is greater than upper (%r)'
                         % (lower, upper))
    if tolerance < 0:
        raise ValueError('tolerance must be >= 0, got %r' % tolerance)

    amount_colx = _resolve_column(sheet, amount_col, header_rows)
    key_colx = None
    if key_col is not None:
        key_colx = _resolve_column(sheet, key_col, header_rows)
        if key_colx == amount_colx:
            raise ValueError('key_col and amount_col must differ')

    partition_name = partition if partition is not None else sheet.name
    summary = PartitionSummary(
        partition_name, lower=lower, upper=upper, tolerance=tolerance,
    )

    if row_range is None:
        start_rowx = header_rows
        end_rowx = sheet.nrows
    else:
        start_rowx, end_rowx = row_range
        start_rowx = max(start_rowx, header_rows)

    # key -> list of (CellRef, numeric value or None, UnresolvedEntry or None)
    records_by_token = {}
    order = []

    for rowx in xrange(start_rowx, end_rowx):
        if rowx >= sheet.nrows:
            break
        summary.rows_scanned += 1
        row_len = sheet.row_len(rowx)

        key = None
        identifiable = False
        if key_colx is not None and key_colx < row_len:
            key_ctype = sheet.cell_type(rowx, key_colx)
            if key_ctype not in (XL_CELL_EMPTY, XL_CELL_BLANK,
                                 XL_CELL_ERROR):
                key = sheet.cell_value(rowx, key_colx)
                identifiable = True
            else:
                key_ref = CellRef(
                    partition_name, sheet.name, rowx, key_colx,
                    key_ctype, sheet.cell_value(rowx, key_colx),
                )
                summary.key_issues.append(UnresolvedEntry(
                    key_ref, REASON_KEY_MISSING,
                    'key cell is %s' % key_ref.cell_type_name,
                ))

        if identifiable:
            token = key
        else:
            token = ('__no_key__', partition_name, sheet.name, rowx)
        if token not in records_by_token:
            records_by_token[token] = []
            order.append(token)

        if amount_colx >= row_len:
            # Ragged row: the amount position does not exist, equivalent
            # to an empty cell.
            ref = CellRef(partition_name, sheet.name, rowx, amount_colx,
                          XL_CELL_EMPTY, '')
            entry = UnresolvedEntry(ref, REASON_EMPTY,
                                    'amount column absent in ragged row')
            summary.empty_cells += 1
            summary.unresolved.append(entry)
            records_by_token[token].append((ref, None, entry))
            continue

        ctype = sheet.cell_type(rowx, amount_colx)
        value = sheet.cell_value(rowx, amount_colx)
        ref = CellRef(partition_name, sheet.name, rowx, amount_colx,
                      ctype, value)
        numeric_value = None
        entry = None

        if ctype == XL_CELL_NUMBER:
            if lower is not None and value < lower:
                entry = UnresolvedEntry(
                    ref, REASON_OUT_OF_BOUNDS,
                    'value %r below lower bound %r' % (value, lower),
                )
                summary.out_of_bounds.append(ref)
            elif upper is not None and value > upper:
                entry = UnresolvedEntry(
                    ref, REASON_OUT_OF_BOUNDS,
                    'value %r above upper bound %r' % (value, upper),
                )
                summary.out_of_bounds.append(ref)
            else:
                numeric_value = value
                if value == 0:
                    summary.zero_cells += 1
        elif ctype == XL_CELL_ERROR:
            summary.error_cells += 1
            entry = UnresolvedEntry(
                ref, REASON_ERROR,
                'error cell %s' % ref.error_text,
            )
        elif ctype == XL_CELL_EMPTY:
            summary.empty_cells += 1
            entry = UnresolvedEntry(ref, REASON_EMPTY)
        elif ctype == XL_CELL_BLANK:
            summary.blank_cells += 1
            entry = UnresolvedEntry(ref, REASON_BLANK)
        elif ctype == XL_CELL_TEXT:
            entry = UnresolvedEntry(ref, REASON_TEXT,
                                    'text is never coerced into an amount')
        elif ctype == XL_CELL_DATE:
            entry = UnresolvedEntry(ref, REASON_DATE,
                                    'date serial is not an amount')
        elif ctype == XL_CELL_BOOLEAN:
            entry = UnresolvedEntry(ref, REASON_BOOLEAN)
        else:  # pragma: no cover - defensive, xlrd defines 7 types
            entry = UnresolvedEntry(ref, _TYPE_NAMES.get(ctype, 'unknown'))

        if entry is not None:
            summary.unresolved.append(entry)
        records_by_token[token].append((ref, numeric_value, entry))

    for token in order:
        records = records_by_token[token]
        identifiable = not (
            isinstance(token, tuple) and token
            and token[0] == '__no_key__'
        )
        key = token if identifiable else None
        group = _build_group(
            token, key, identifiable, records, summary.tolerance,
        )
        summary.transactions.append(group)
        if len(records) > 1:
            summary.duplicate_rows_dropped += len(records) - 1

    return summary


def summarize_workbook(book, amount_col, key_col=None, sheet_names=None,
                       lower=None, upper=None, header_rows=0,
                       tolerance=0.0):
    """
    Summarise every (or selected) sheet of a workbook as separate partitions.

    :param book: an xlrd :class:`~xlrd.book.Book`.
    :param sheet_names: optional iterable of sheet names to include.
    :rtype: list of :class:`PartitionSummary`
    """
    if sheet_names is None:
        sheets = book.sheets()
    else:
        sheets = [book.sheet_by_name(name) for name in sheet_names]
    return [
        summarize_sheet(
            sheet, amount_col, key_col=key_col, lower=lower, upper=upper,
            header_rows=header_rows, tolerance=tolerance,
        )
        for sheet in sheets
    ]


def merge_summaries(summaries, partition='merged', tolerance=None):
    """
    Merge partition summaries, recognising transactions across partitions.

    Groups sharing an identifiable key are rebuilt from all of their source
    witnesses instead of adding the two partition ranges, so a transaction
    duplicated in both partitions is still counted once. Keyless groups
    cannot be matched and are carried over unchanged.

    :param summaries: iterable of :class:`PartitionSummary` objects.
    :param partition: name of the merged partition.
    :param tolerance: contradiction tolerance; defaults to the tolerance of
        the first input summary.
    :rtype: :class:`PartitionSummary`
    """
    summaries = list(summaries)
    if not summaries:
        raise ValueError('need at least one summary to merge')
    if tolerance is None:
        tolerance = summaries[0].tolerance

    lowers = {item.lower for item in summaries}
    uppers = {item.upper for item in summaries}
    merged = PartitionSummary(
        partition,
        lower=next(iter(lowers)) if len(lowers) == 1 else None,
        upper=next(iter(uppers)) if len(uppers) == 1 else None,
        tolerance=tolerance,
    )

    records_by_token = {}
    order = []
    seen_refs = set()
    seen_unresolved = set()
    seen_key_issues = set()
    seen_out_of_bounds = set()

    for summary in summaries:
        for group in summary.transactions:
            token = group.token if group.identifiable else (
                '__kept__', group.token, summary.partition,
            )
            if token not in records_by_token:
                records_by_token[token] = []
                order.append(token)
            for ref in group.members:
                if ref.identity in seen_refs:
                    continue
                seen_refs.add(ref.identity)
                # Re-apply the *source* partition's bounds so an
                # out-of-bounds number stays excluded after merging.
                numeric_value = None
                if ref.ctype == XL_CELL_NUMBER:
                    if (summary.lower is None
                            or ref.value >= summary.lower) and (
                            summary.upper is None
                            or ref.value <= summary.upper):
                        numeric_value = ref.value
                entry = None
                for unresolved in group.unresolved_members:
                    if unresolved.ref.identity == ref.identity:
                        entry = unresolved
                        break
                records_by_token[token].append(
                    (ref, numeric_value, entry)
                )

        # Deduplicate by source cell so re-merging an already merged
        # summary never books the same unresolved row twice.
        for item in summary.unresolved:
            if item.ref.identity not in seen_unresolved:
                seen_unresolved.add(item.ref.identity)
                merged.unresolved.append(item)
        for item in summary.key_issues:
            if item.ref.identity not in seen_key_issues:
                seen_key_issues.add(item.ref.identity)
                merged.key_issues.append(item)
        for ref in summary.out_of_bounds:
            if ref.identity not in seen_out_of_bounds:
                seen_out_of_bounds.add(ref.identity)
                merged.out_of_bounds.append(ref)

    for token in order:
        records = records_by_token[token]
        merged.rows_scanned += len(records)
        identifiable = not (
            isinstance(token, tuple) and token and token[0] == '__kept__'
        )
        key = token if identifiable else None
        group = _build_group(
            token, key, identifiable, records, tolerance,
        )
        merged.transactions.append(group)
        if len(records) > 1:
            merged.duplicate_rows_dropped += len(records) - 1
        # Derive cell counts from the deduplicated source records so the
        # merged report stays consistent even with overlapping inputs.
        for ref, numeric_value, _ in records:
            if ref.ctype == XL_CELL_ERROR:
                merged.error_cells += 1
            elif ref.ctype == XL_CELL_BLANK:
                merged.blank_cells += 1
            elif ref.ctype == XL_CELL_EMPTY:
                merged.empty_cells += 1
            if numeric_value is not None and ref.value == 0:
                merged.zero_cells += 1

    return merged


__all__ = [
    'AmountRange',
    'CellRef',
    'PartitionSummary',
    'TransactionGroup',
    'UnresolvedEntry',
    'merge_summaries',
    'summarize_sheet',
    'summarize_workbook',
    'REASON_BLANK',
    'REASON_BOOLEAN',
    'REASON_DATE',
    'REASON_EMPTY',
    'REASON_ERROR',
    'REASON_KEY_MISSING',
    'REASON_OUT_OF_BOUNDS',
    'REASON_TEXT',
]
