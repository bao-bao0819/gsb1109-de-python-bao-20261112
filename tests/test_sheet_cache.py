"""
Tests for the budgeted, borrow-aware on-demand sheet cache.

Covers: shared single load on concurrent borrows, pinning while borrowed,
LRU eviction of unborrowed sheets, budget-exhaustion reporting that preserves
delivered rows, load/unload reconciliation, recovery after a reader raises,
and backwards compatibility with the legacy on_demand/non-on_demand modes.
"""
import threading

import pytest

from xlrd import open_workbook, SheetBudgetExceeded, XLRDError

from .helpers import from_sample

SAMPLE = 'profiles.xls'
NSHEETS = 5


def open_book(cache_size):
    return open_workbook(
        from_sample(SAMPLE), on_demand=True, sheet_cache_size=cache_size)


def expected_first_row(sheetx):
    return open_workbook(from_sample(SAMPLE)) \
        .sheet_by_index(sheetx).row_values(0)


def row_signature(book, sheetx):
    sh = book.sheet_by_index(sheetx)
    return sh.name, sh.nrows, [sh.row_values(r) for r in range(sh.nrows)]


class TestBoundedCache(object):

    def test_budget_caps_loaded_sheets(self):
        book = open_book(2)
        for sheetx in range(NSHEETS):
            with book.borrow_sheet(sheetx):
                pass
            assert len(book.sheet_cache_info()["loaded"]) <= 2
        assert book.sheet_cache_info()["budget"] == 2

    def test_two_borrowers_load_sheet_once(self):
        book = open_book(2)
        with book.borrow_sheet(0):
            assert book.sheet_cache_info()["loads"] == 1
            with book.borrow_sheet(0):
                assert book.sheet_cache_info()["loads"] == 1
                assert book.sheet_cache.borrower_count(0) == 2
            assert book.sheet_cache.borrower_count(0) == 1
        assert book.sheet_cache.borrower_count(0) == 0
        assert book.sheet_cache_info()["loads"] == 1

    def test_borrowed_sheet_pinned_against_other_release(self):
        book = open_book(2)
        with book.borrow_sheet(0):
            with book.borrow_sheet(1):
                pass  # second borrower finishes; sheet 0 must survive
            assert book.sheet_loaded(0)
            assert book.sheet_cache.borrower_count(0) == 1

    def test_lru_evicts_unborrowed_sheet_on_pressure(self):
        book = open_book(2)
        with book.borrow_sheet(0):
            pass
        with book.borrow_sheet(1):
            pass
        info = book.sheet_cache_info()
        assert info["loaded"] == [0, 1]
        # Touching sheet 0 makes sheet 1 the least-recently-used.
        with book.borrow_sheet(0):
            pass
        with book.borrow_sheet(2):
            pass
        info = book.sheet_cache_info()
        assert info["loaded"] == [0, 2]
        assert 1 not in info["loaded"]
        assert info["unloads"] == 1 and info["evictions"] == 1

    def test_budget_exhausted_names_blockers_and_keeps_rows(self):
        book = open_book(2)
        delivered = []
        with book.borrow_sheet(0) as sh0:
            delivered.append([sh0.row_values(r) for r in range(sh0.nrows)])
            with book.borrow_sheet(1):
                with pytest.raises(SheetBudgetExceeded) as excinfo:
                    with book.borrow_sheet(2):
                        pytest.fail("third sheet must not fit in budget 2")
        err = excinfo.value
        assert err.sheetx == 2
        assert err.blocking_sheets == [0, 1]
        assert err.blocking_sheet_names == book.sheet_names()[:2]
        info = book.sheet_cache_info()
        assert info["failed_loads"] == 1
        # Nothing got unloaded and no physical load happened for sheet 2.
        assert info["unloads"] == 0
        assert info["loads"] == 2
        assert book.sheet_loaded(0) and book.sheet_loaded(1)
        assert not book.sheet_loaded(2)
        # Rows handed out before the refusal are still intact.
        assert delivered[0] == [
            book.sheet_by_index(0).row_values(r) for r in range(15)]

    def test_retry_succeeds_after_borrower_releases(self):
        book = open_book(1)
        with book.borrow_sheet(0):
            with pytest.raises(SheetBudgetExceeded):
                with book.borrow_sheet(1):
                    pass
        with book.borrow_sheet(1):  # slot freed by LRU eviction of sheet 0
            assert book.sheet_cache_info()["loaded"] == [1]

    def test_reconcile_load_unload_accounting(self):
        book = open_book(2)
        for sheetx in range(NSHEETS):
            with book.borrow_sheet(sheetx):
                pass
        report = book.reconcile_sheet_cache()
        assert report["balanced"]
        assert report["loads"] - report["unloads"] == report["loaded"]
        assert report["active_borrows"] == 0

    def test_reader_exception_releases_borrow_and_pipeline_continues(self):
        book = open_book(2)

        def flaky_import(sheetx):
            with book.borrow_sheet(sheetx) as sh:
                if sheetx == 1:
                    raise RuntimeError("simulated reader crash")
                return sh.nrows

        with pytest.raises(RuntimeError):
            flaky_import(1)
        # The dead borrower's claim is gone, so the sheet is evictable and the
        # next worksheet can still be brought into the warehouse.
        assert book.sheet_cache.borrower_count(1) == 0
        collected = {}
        for sheetx in range(NSHEETS):
            if sheetx == 1:
                continue
            with book.borrow_sheet(sheetx) as sh:
                collected[sheetx] = sh.nrows
        assert set(collected) == {0, 2, 3, 4}
        assert book.reconcile_sheet_cache()["balanced"]

    def test_full_rotation_delivers_every_sheet_rows_once_loaded(self):
        book = open_book(1)
        expected = {}
        for sheetx in range(NSHEETS):
            name, nrows, rows = row_signature(
                open_workbook(from_sample(SAMPLE)), sheetx)
            expected[sheetx] = (name, nrows, rows)
        for sheetx in range(NSHEETS):
            with book.borrow_sheet(sheetx) as sh:
                name, nrows, rows = expected[sheetx]
                assert sh.name == name
                assert sh.nrows == nrows
                assert [sh.row_values(r) for r in range(nrows)] == rows

    def test_sheets_helper_returns_all_objects_under_tight_budget(self):
        book = open_book(1)
        all_sheets = book.sheets()
        assert len(all_sheets) == NSHEETS
        # Every returned Sheet object keeps its rows even though only the last
        # slot remains resident in the budgeted cache.
        assert [sh.name for sh in all_sheets] == book.sheet_names()
        for sheetx, sh in enumerate(all_sheets):
            assert sh.nrows == 15
            assert sh.row_values(0) == expected_first_row(sheetx)

    def test_acquired_sheet_stays_valid_after_lru_eviction(self):
        book = open_book(1)
        first = book.sheet_by_index(0)
        first_rows = [first.row_values(r) for r in range(first.nrows)]
        # Loading every other sheet forces sheet 0 out of its single slot.
        for sheetx in range(1, NSHEETS):
            book.sheet_by_index(sheetx)
        assert not book.sheet_loaded(0)
        assert [first.row_values(r) for r in range(first.nrows)] == first_rows

    def test_unload_refused_while_borrowed(self):
        book = open_book(2)
        with book.borrow_sheet(0):
            with pytest.raises(XLRDError, match="active borrower"):
                book.unload_sheet(0)
            assert book.sheet_loaded(0)
        book.unload_sheet(0)
        assert not book.sheet_loaded(0)
        assert book.sheet_cache_info()["unloads"] == 1

    def test_concurrent_borrowers_share_single_load(self):
        book = open_book(1)
        barrier = threading.Barrier(4)
        results = []
        errors = []

        def reader():
            try:
                barrier.wait()
                with book.borrow_sheet(0) as sh:
                    results.append(sh.row_values(0))
                    # Hold the pin briefly so peers overlap.
                    barrier2.wait()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        barrier2 = threading.Barrier(4)
        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        info = book.sheet_cache_info()
        assert info["loads"] == 1
        assert len(results) == 4
        assert all(row == results[0] for row in results)
        assert book.sheet_cache.borrower_count(0) == 0

    def test_borrow_by_name_matches_index(self):
        book = open_book(2)
        names = book.sheet_names()
        with book.borrow_sheet(names[3]) as by_name, \
                book.borrow_sheet(3) as by_index:
            assert by_name is by_index
        assert book.sheet_cache_info()["loads"] == 1

    def test_unknown_sheet_name_raises(self):
        book = open_book(2)
        with pytest.raises(XLRDError, match="No sheet named"):
            with book.borrow_sheet("nope"):
                pass


class TestCacheLifecycle(object):

    def test_resident_sheets_survive_release_resources(self):
        book = open_book(2)
        with book.borrow_sheet(0):
            pass
        snapshot = [book.sheet_by_index(0).row_values(r) for r in range(15)]
        book.release_resources()
        assert book.sheet_loaded(0)
        assert [book.sheet_by_index(0).row_values(r)
                for r in range(15)] == snapshot
        with pytest.raises(XLRDError, match="releasing resources"):
            with book.borrow_sheet(1):
                pass

    def test_context_manager_closes_cache(self):
        with open_workbook(from_sample(SAMPLE), on_demand=True,
                           sheet_cache_size=2) as book:
            with book.borrow_sheet(0):
                pass
        with pytest.raises(XLRDError, match="releasing resources"):
            with book.borrow_sheet(1):
                pass

    @pytest.mark.parametrize("bad_size,exc", [
        (0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError),
    ])
    def test_invalid_cache_size_rejected(self, bad_size, exc):
        with pytest.raises(exc):
            open_workbook(from_sample(SAMPLE), on_demand=True,
                          sheet_cache_size=bad_size)


class TestBackwardsCompatibility(object):

    def test_legacy_on_demand_without_budget_keeps_all_sheets(self):
        book = open_workbook(from_sample(SAMPLE), on_demand=True)
        assert book.sheet_cache is not None
        for sheetx in range(NSHEETS):
            book.sheet_by_index(sheetx)
        info = book.sheet_cache_info()
        assert info["loaded"] == list(range(NSHEETS))
        assert info["unloads"] == 0
        assert book.reconcile_sheet_cache()["balanced"]

    def test_fully_loaded_book_has_no_eviction_cache(self):
        book = open_workbook(from_sample(SAMPLE))
        assert book.sheet_cache is None
        assert book.sheet_cache_info() is None
        assert book.reconcile_sheet_cache() is None
        with book.borrow_sheet(0) as sh:
            assert sh.nrows == 15

    def test_budget_ignored_without_on_demand(self):
        book = open_workbook(
            from_sample(SAMPLE), on_demand=False, sheet_cache_size=1)
        assert book.sheet_cache is None
        assert book.sheet_loaded(0) and book.sheet_loaded(NSHEETS - 1)
        assert len(book.sheets()) == NSHEETS

    def test_legacy_unload_still_works_without_cache(self):
        book = open_workbook(from_sample(SAMPLE))
        assert book.sheet_loaded(0)
        book.unload_sheet(0)
        assert not book.sheet_loaded(0)
