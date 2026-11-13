import pytest

import portion as P
from portion.gears import GearTable, Mismatch


# Published gears: negative values, gaps, and non-uniform spacing
# (consecutive gears differ by 7, 2, 1, 2, 3, 4 and 11).
GEARS = (-10, -3, -1, 0, 2, 5, 9, 20)


@pytest.fixture
def table():
    return GearTable(GEARS)


class TestGearTable:
    def test_gears_are_sorted_and_deduplicated(self):
        table = GearTable([5, -3, 5, 0, -3])
        assert table.gears == (-3, 0, 5)
        assert len(table) == 3
        assert list(table) == [-3, 0, 5]
        assert -3 in table
        assert 4 not in table

    def test_empty_table_is_rejected(self):
        with pytest.raises(ValueError):
            GearTable([])

    def test_read_with_gear_bounds(self, table):
        assert table.read(P.closed(-10, 20)) == frozenset(GEARS)
        # Adjacent gears do not differ by one, both ends are still read.
        assert table.read(P.closed(-3, -1)) == frozenset({-3, -1})
        assert table.read(P.open(-3, 5)) == frozenset({-1, 0, 2})
        assert table.read(P.closedopen(-3, 5)) == frozenset({-3, -1, 0, 2})
        assert table.read(P.openclosed(-3, 5)) == frozenset({-1, 0, 2, 5})

    def test_read_with_bounds_outside_the_table(self, table):
        # Bounds that are not gears are handled by magnitude comparison.
        assert table.read(P.closed(-9, 8)) == frozenset({-3, -1, 0, 2, 5})
        assert table.read(P.closed(-4.5, 6.5)) == frozenset({-3, -1, 0, 2, 5})
        assert table.read(P.open(2.5, 19.5)) == frozenset({5, 9})

    def test_read_empty_and_unbounded(self, table):
        assert table.read(P.empty()) == frozenset()
        assert table.read(P.openclosed(-P.inf, 0)) == frozenset({-10, -3, -1, 0})
        assert table.read(P.closedopen(5, P.inf)) == frozenset({5, 9, 20})
        assert table.read(P.open(-P.inf, P.inf)) == frozenset(GEARS)

    def test_read_rejects_non_intervals(self, table):
        with pytest.raises(TypeError):
            table.read(1)

    def test_to_interval(self, table):
        assert table.to_interval({-1, 2}) == P.singleton(-1) | P.singleton(2)
        assert table.to_interval(set()) == P.empty()
        assert table.to_interval(GEARS) == P.Interval(
            *(P.singleton(g) for g in GEARS)
        )

    def test_to_interval_rejects_unknown_gears(self, table):
        with pytest.raises(ValueError):
            table.to_interval({-1, 3})

    def test_discrete_operations(self, table):
        a = frozenset({-3, -1, 0})
        b = frozenset({0, 2, 5})
        assert table.union(a, b) == frozenset({-3, -1, 0, 2, 5})
        assert table.intersection(a, b) == frozenset({0})
        assert table.difference(a, b) == frozenset({-3, -1})
        assert table.complement(a) == frozenset({-10, 2, 5, 9, 20})

    def test_discrete_operations_reject_unknown_gears(self, table):
        with pytest.raises(ValueError):
            table.union({-1}, {3})
        with pytest.raises(ValueError):
            table.complement({1})


class TestCheck:
    def test_consistent_operations(self, table):
        cases = [
            (P.closed(-10, 20), P.closed(0, 9)),
            (P.closed(-4.5, 6.5), P.open(-3, 5)),
            (P.openclosed(-3, 2), P.closedopen(2, 20)),
            (P.singleton(0), P.closed(-1, 2)),
            (P.empty(), P.closed(-3, 5)),
            (P.open(-P.inf, 0), P.closed(0, P.inf)),
        ]
        for a, b in cases:
            results = table.check(a, b)
            assert set(results) == {"union", "intersection", "difference", "complement"}
            assert all(mismatch is None for mismatch in results.values())

    def test_consistent_complement_without_second_operand(self, table):
        results = table.check(P.closedopen(-3, 5))
        assert set(results) == {"complement"}
        assert results["complement"] is None

    def test_isolated_unattainable_point_is_not_reported(self, table):
        # Continuous intersection is the singleton [1], but 1 is not a
        # published gear, so the empty discrete intersection must not be
        # reported as a difference.
        a = P.closed(-3, 1)
        b = P.closed(1, 5)
        assert a & b == P.singleton(1)
        assert table.read(a & b) == frozenset()
        assert all(mismatch is None for mismatch in table.check(a, b).values())
        assert table.compare(P.singleton(1), frozenset()) is None

    def test_mismatch_reports_smallest_gear_and_memberships(self, table):
        continuous = P.closed(-3, 9)  # reads as {-3, -1, 0, 2, 5, 9}
        discrete = {-3, -1, 5, 9}  # missing 0 and 2

        mismatch = table.compare(continuous, discrete, operation="union")
        assert isinstance(mismatch, Mismatch)
        assert mismatch.operation == "union"
        assert mismatch.gear == 0
        assert mismatch.in_continuous
        assert not mismatch.in_discrete
        assert "0" in str(mismatch)

    def test_mismatch_with_extra_discrete_gear(self, table):
        continuous = P.closed(2, 9)  # reads as {2, 5, 9}
        discrete = {-10, 2, 5, 9}  # extra -10

        mismatch = table.compare(continuous, discrete, operation="difference")
        assert mismatch.gear == -10
        assert not mismatch.in_continuous
        assert mismatch.in_discrete

    def test_compare_rejects_invalid_inputs(self, table):
        with pytest.raises(TypeError):
            table.compare(1, {-1})
        with pytest.raises(ValueError):
            table.compare(P.closed(-1, 2), {3})

    def test_check_rejects_non_intervals(self, table):
        with pytest.raises(TypeError):
            table.check(1, P.closed(0, 1))
        with pytest.raises(TypeError):
            table.check(P.closed(0, 1), 2)
