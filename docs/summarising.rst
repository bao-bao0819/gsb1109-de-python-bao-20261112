Partition-wise financial summaries
==================================

.. module:: xlrd.summary

:mod:`xlrd.summary` totals an amount column without hiding uncertainty.
Every row is booked exactly once, either as a transaction amount or as an
*unresolved* source row explaining why no amount could be determined.

Basic use
---------

::

    from xlrd import open_workbook
    from xlrd.summary import summarize_sheet, merge_summaries

    book = open_workbook("accounts.xls")
    sheet = book.sheet_by_name("January")
    summary = summarize_sheet(
        sheet,
        amount_col=2,          # column index, or header name with header_rows
        key_col=0,             # rows sharing this value are duplicate rows
        lower=0.0,             # inclusive amount bounds
        upper=1_000_000.0,
        header_rows=1,
    )
    print(summary.report_text())

The total is an :class:`~xlrd.summary.AmountRange`. When every transaction
is known exactly, ``summary.exact_total`` is a number; otherwise inspect
``total_low``/``total_high`` for the possible interval and
``summary.unresolved`` for the rows that could not be valued.

Cell accounting
---------------

The amount column's cell types are counted separately so they can never be
confused with real money:

* error cells (``error_cells``), blank cells (``blank_cells``) and empty
  cells (``empty_cells``) are each booked with their own reason; a ragged
  row missing the amount position counts as empty;
* a numeric ``0`` is a valid amount and counted in ``zero_cells``;
* text, date and boolean cells are never coerced into numbers;
* numbers outside ``lower``/``upper`` are excluded and listed in
  ``out_of_bounds`` (the bounds are inclusive).

Each unresolved entry carries a :class:`~xlrd.summary.CellRef` with the
sheet name, A1 coordinate, cell type and (for error cells) the Excel error
text such as ``#DIV/0!``.

Duplicate rows and contradictions
---------------------------------

When ``key_col`` is given, rows with the same key value are known copies of
one original transaction and are counted once
(``duplicate_rows_dropped`` records how many extra rows were folded in).
If the numeric witnesses disagree, the transaction receives a possible
amount range whose ``witness_low`` and ``witness_high`` attributes name the
exact source cells attaining each endpoint::

    for group in summary.contradictory_groups:
        print(group.key, group.amount_range.low, group.amount_range.high)
        print([ref.location for ref in group.amount_range.witness_high])

Rows without a usable key cannot be matched to anything and are always
treated as independent transactions; the missing key is recorded in
``key_issues``.

Merging partitions
------------------

Summaries built from different sheets or row ranges are merged with
:func:`~xlrd.summary.merge_summaries` (or ``summary.merge(other)``).
Transactions are matched by their key across partitions and rebuilt from
all witnesses, so a transaction duplicated in two partitions is still
counted once -- the two ranges are never simply added::

    merged = merge_summaries([january, february], partition="Q1")

Keyless transactions cannot be recognised across partitions and are carried
over unchanged. Unresolved rows, cell counts and out-of-bounds records are
aggregated into the merged report.

For machine-readable output, every result object exposes ``as_dict()`` and
the summary offers ``to_json()``.
