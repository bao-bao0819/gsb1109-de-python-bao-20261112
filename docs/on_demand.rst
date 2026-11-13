Loading worksheets on demand
=============================

.. currentmodule:: xlrd.book

This feature, new in version 0.7.1, is governed by the ``on_demand`` argument
to the :func:`~xlrd.open_workbook` function and allows saving memory and time by
loading only those sheets that the caller is interested in, and releasing sheets
when no longer required.

``on_demand=False`` (default):
  No change. :func:`~xlrd.open_workbook` loads global data
  and all sheets, releases resources no longer required (principally the
  :class:`str` or :class:`mmap.mmap` object containing the Workbook stream),
  and returns.

``on_demand=True`` and BIFF version < 5.0:
  A warning message is emitted,
  ``on_demand`` is recorded as ``False``, and the old process is followed.

``on_demand=True`` and BIFF version >= 5.0:
  :func:`~xlrd.open_workbook` loads global
  data and returns without releasing resources. At this stage, the only
  information available about sheets is :attr:`Book.nsheets` and
  :meth:`Book.sheet_names`.

:meth:`Book.sheet_by_name` and :meth:`Book.sheet_by_index` will load the
requested sheet if it is not already loaded.

:meth:`Book.sheets` will load all unloaded sheets.

The caller may save memory by calling
:meth:`Book.unload_sheet` when finished with the sheet.
This applies irrespective of the state of ``on_demand``.

The caller may re-load an unloaded sheet by calling :meth:`Book.sheet_by_name`
or :meth:`Book.sheet_by_index`, except if the required resources have been
released (which will
have happened automatically when ``on_demand`` is false). This is the only
case where an exception will be raised.

The caller may query the state of a sheet using :meth:`Book.sheet_loaded`.

:meth:`Book.release_resources` may used to save memory and close
any memory-mapped file before proceeding to examine already-loaded
sheets. Once resources are released, no further sheets can be loaded.

When using on-demand, it is advisable to ensure that
:meth:`Book.release_resources` is always called, even if an exception
is raised in your own code; otherwise if the input file has been
memory-mapped, the :class:`mmap.mmap` object will not be closed and you will
not be able to access the physical file until your Python process
terminates. This can be done by calling :meth:`Book.release_resources`
explicitly in the finally part of a try/finally block.

The Book object is also a context manager, so  you can wrap your code in a
``with`` statement that will make sure underlying resources are closed.

Budgeted, borrow-aware sheet cache
----------------------------------

An import job that works through many worksheets in turn can cap how many
sheets are resident at once by passing the ``sheet_cache_size`` argument to
:func:`~xlrd.open_workbook` (it only takes effect when ``on_demand=True``)::

    book = open_workbook(path, on_demand=True, sheet_cache_size=2)

Rather than loading a sheet and remembering to unload it, a reader *borrows*
a sheet through the :meth:`Book.borrow_sheet` context manager::

    with book.borrow_sheet(sheetx) as sheet:
        rows = [sheet.row_values(r) for r in range(sheet.nrows)]

The cache gives the following guarantees:

* Two readers borrowing the same sheet share one physical load; the sheet is
  only parsed once.
* A borrowed sheet is *pinned*. It is not evicted when another borrower
  finishes or under least-recently-used (LRU) pressure, so rows already handed
  to a reader stay valid for the whole ``with`` block.
* When the budget is full, sheets whose borrowers have all released them are
  retired LRU to make room.
* If every resident sheet is still borrowed, loading another sheet raises
  :class:`xlrd.sheet_cache.SheetBudgetExceeded`. The exception exposes
  ``sheetx``, ``blocking_sheets`` and ``blocking_sheet_names`` so the caller
  can see which in-use pages prevent the load. Nothing is unloaded and no row
  already delivered is affected; releasing a borrower and retrying succeeds.
* If the ``with`` body raises, the borrow is released automatically in the
  ``finally`` phase, so a reader that crashes does not pin the sheet and the
  next worksheet can still be brought into the cache.

Physical loads and evictions are counted. :meth:`Book.sheet_cache_info`
returns a snapshot of the budget, resident sheets, per-sheet borrower counts
and the ``loads``/``unloads``/``evictions``/``failed_loads`` counters;
:meth:`Book.reconcile_sheet_cache` checks that ``loads - unloads`` equals the
number of resident sheets (``balanced``). The cache itself is available as
:attr:`Book.sheet_cache` (``None`` for a fully-loaded, non-on-demand book).

When ``sheet_cache_size`` is omitted, every on-demand sheet stays loaded,
which preserves the pre-existing on-demand behaviour.

