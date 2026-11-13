# Copyright (c) 2005-2012 Stephen John Machin, Lingfo Pty Ltd
# This module is part of the xlrd package, which is released under a
# BSD-style licence.
"""
Budgeted, borrow-aware cache for :class:`~xlrd.book.Book` worksheets opened
in ``on_demand`` mode.

The cache keeps at most *max_loaded* sheets in memory at any time. A sheet is
*borrowed* by a reader (through :meth:`~xlrd.book.Book.borrow_sheet`); while a
sheet has at least one active borrower it is pinned and can never be evicted,
so rows already handed to a reader stay valid. Sheets whose borrowers have all
released them become eviction candidates and are retired least-recently-used
first when a new sheet needs a slot.

Every physical sheet parse and every eviction is counted, so callers can
reconcile memory behaviour through :meth:`BoundedSheetCache.reconcile`.
"""
from __future__ import print_function

import threading
from collections import OrderedDict
from contextlib import contextmanager

from .biffh import XLRDError


class SheetBudgetExceeded(XLRDError):
    """
    Raised when a sheet cannot be loaded because the cache budget is full and
    every cached sheet is pinned by an active borrower.

    No sheet is unloaded and no row already delivered to a caller is touched
    when this exception is raised; the caller may release a borrower and retry.
    """

    def __init__(self, message, sheetx, blocking_sheets, blocking_sheet_names):
        super(SheetBudgetExceeded, self).__init__(message)
        #: Index of the sheet that could not be loaded.
        self.sheetx = sheetx
        #: Indices of the sheets whose active borrowers block the load.
        self.blocking_sheets = blocking_sheets
        #: Names corresponding to :attr:`blocking_sheets`.
        self.blocking_sheet_names = blocking_sheet_names


class BoundedSheetCache(object):
    """
    Thread-safe LRU cache of loaded sheets bounded by a page budget.

    A *borrow* is a claim on a sheet for the duration of a read. Two readers
    borrowing the same sheet share one physical load, and the sheet stays
    loaded until every borrower has released it. A plain :meth:`acquire` (used
    by :meth:`~xlrd.book.Book.sheet_by_index`) returns a sheet without taking
    out a borrow, preserving the historical ``get_sheet`` calling convention;
    the returned object stays valid even if its cache slot is later retired
    LRU.
    """

    def __init__(self, book, max_loaded):
        if not isinstance(max_loaded, int) or isinstance(max_loaded, bool):
            raise TypeError("sheet_cache_size must be an int, got %r"
                            % type(max_loaded).__name__)
        if max_loaded < 1:
            raise ValueError("sheet_cache_size must be >= 1, got %d"
                             % max_loaded)
        self._book = book
        self._max_loaded = max_loaded
        # LRU order of loaded sheets, most recently used last.
        self._order = OrderedDict()
        # sheetx -> number of live borrowers (zero means evictable LRU).
        self._refcounts = {}
        # Reentrant because info()/reconcile() reuse the locked accessors.
        self._lock = threading.RLock()
        # Accounting counters (physical events only).
        self.loads = 0
        self.unloads = 0
        self.evictions = 0
        self.failed_loads = 0
        self._released = False

    @property
    def max_loaded(self):
        return self._max_loaded

    def is_loaded(self, sheetx):
        with self._lock:
            return self._book._sheet_list[sheetx] is not None

    def borrower_count(self, sheetx):
        with self._lock:
            return self._refcounts.get(sheetx, 0)

    def loaded_indices(self):
        with self._lock:
            return [sheetx for sheetx, sh in enumerate(self._book._sheet_list)
                    if sh is not None]

    def _names(self, indices):
        return [self._book._sheet_names[sheetx] for sheetx in indices]

    def _unload_locked(self, sheetx):
        self._book._sheet_list[sheetx] = None
        del self._order[sheetx]
        self._refcounts.pop(sheetx, None)
        self.unloads += 1
        self.evictions += 1

    def _free_slot_locked(self, sheetx):
        """Make one cache slot available, or name the sheets blocking it."""
        for candidate in self._order:
            if self._refcounts.get(candidate, 0) == 0:
                self._unload_locked(candidate)
                return
        blockers = list(self._order)
        self.failed_loads += 1
        raise SheetBudgetExceeded(
            "Sheet cache budget of %d loaded sheet(s) is exhausted; cannot "
            "load sheet %r. The following %d sheet(s) are held by active "
            "borrowers and cannot be unloaded: %r"
            % (self._max_loaded, self._book._sheet_names[sheetx],
               len(blockers), self._names(blockers)),
            sheetx, blockers, self._names(blockers),
        )

    def _load_locked(self, sheetx):
        sh = self._book._load_sheet(sheetx)
        self._order[sheetx] = True
        self._refcounts.setdefault(sheetx, 0)
        self.loads += 1
        return sh

    def acquire(self, sheetx):
        """
        Return the sheet, loading it through the budget if necessary.

        Unlike :meth:`borrow`, no borrower claim is taken, so the slot may
        later be retired under LRU pressure. The :class:`~xlrd.sheet.Sheet`
        object handed back keeps its rows regardless, matching the legacy
        ``get_sheet`` calling convention.
        """
        with self._lock:
            if self._released:
                raise XLRDError(
                    "Can't load sheets after releasing resources.")
            sh = self._book._sheet_list[sheetx]
            if sh is not None:
                self._order.move_to_end(sheetx)
                return sh
            if len(self._order) >= self._max_loaded:
                self._free_slot_locked(sheetx)
            try:
                return self._load_locked(sheetx)
            except Exception:
                self.failed_loads += 1
                raise

    @contextmanager
    def borrow(self, sheetx):
        """
        Lend a sheet to one reader for the duration of the ``with`` block.

        Nested or concurrent borrows of the same sheet share a single load;
        the sheet is only evictable once the last borrower releases it. If the
        block raises, the borrow is still released so that subsequent sheets
        can keep being loaded.
        """
        with self._lock:
            if self._released:
                raise XLRDError(
                    "Can't load sheets after releasing resources.")
            sh = self._book._sheet_list[sheetx]
            if sh is not None:
                self._order.move_to_end(sheetx)
            else:
                if len(self._order) >= self._max_loaded:
                    self._free_slot_locked(sheetx)
                try:
                    sh = self._load_locked(sheetx)
                except Exception:
                    self.failed_loads += 1
                    raise
            self._refcounts[sheetx] = self._refcounts.get(sheetx, 0) + 1
        try:
            yield sh
        finally:
            with self._lock:
                count = self._refcounts.get(sheetx, 0) - 1
                if count <= 0:
                    self._refcounts.pop(sheetx, None)
                else:
                    self._refcounts[sheetx] = count
                self._order.move_to_end(sheetx)

    def evict(self, sheetx):
        """Force a single sheet out; a borrowed sheet cannot be evicted."""
        with self._lock:
            if self._book._sheet_list[sheetx] is None:
                return
            if self._refcounts.get(sheetx, 0) > 0:
                raise XLRDError(
                    "Can't unload sheet %r: it has %d active borrower(s)."
                    % (self._book._sheet_names[sheetx],
                       self._refcounts[sheetx]))
            self._unload_locked(sheetx)

    def close(self):
        """
        Stop accepting new loads without evicting resident sheets.

        Already-loaded sheets stay available for examination, mirroring
        :meth:`~xlrd.book.Book.release_resources`; any later demand for a
        non-loaded sheet raises :class:`~xlrd.biffh.XLRDError`.
        """
        with self._lock:
            self._released = True

    def info(self):
        """Snapshot of cache state and counters for diagnostics/reporting."""
        with self._lock:
            return {
                "budget": self._max_loaded,
                "loaded": self.loaded_indices(),
                "borrowers": dict(self._refcounts),
                "loads": self.loads,
                "unloads": self.unloads,
                "evictions": self.evictions,
                "failed_loads": self.failed_loads,
            }

    def reconcile(self):
        """
        Reconcile load/unload counters against the sheets currently resident.

        :returns: A dict of counters with ``balanced`` true exactly when
                  ``loads - unloads`` equals the number of loaded sheets.
        """
        with self._lock:
            loaded = sum(1 for sh in self._book._sheet_list if sh is not None)
            active_borrows = sum(self._refcounts.values())
            return {
                "loads": self.loads,
                "unloads": self.unloads,
                "loaded": loaded,
                "active_borrows": active_borrows,
                "balanced": self.loads - self.unloads == loaded,
            }
