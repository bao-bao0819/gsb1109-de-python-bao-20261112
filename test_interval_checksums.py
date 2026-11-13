"""interval_checksums 的单元测试与随机对拍.

参照实现: 对约束矩阵做分数(Fraction)高斯消元, 与图方法结果交叉验证.
"""

from __future__ import annotations

import random
from fractions import Fraction

from interval_checksums import (
    CONTRADICTORY,
    DETERMINED,
    UNDERDETERMINED,
    Record,
    analyze,
)


def brute_force(records, query_lo, query_hi):
    """返回 (status, query_total_or_None), 用精确分数线性代数求解."""
    positions = sorted(
        {p for rec in records for p in range(rec.lo, rec.hi + 1)}
        | set(range(query_lo, query_hi + 1))
    )
    index = {p: i for i, p in enumerate(positions)}
    rows = []
    for rec in records:
        row = [Fraction(0)] * len(positions)
        for p in range(rec.lo, rec.hi + 1):
            row[index[p]] = Fraction(1)
        rows.append((row, Fraction(rec.total)))

    # 高斯消元 -> 行阶梯, 同时判矛盾
    pivot_of_row = []
    r = 0
    for c in range(len(positions)):
        pivot = next(
            (k for k in range(r, len(rows)) if rows[k][0][c] != 0), None
        )
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        for k in range(len(rows)):
            if k != r and rows[k][0][c] != 0:
                factor = rows[k][0][c] / rows[r][0][c]
                rows[k] = (
                    [a - factor * b for a, b in zip(rows[k][0], rows[r][0])],
                    rows[k][1] - factor * rows[r][1],
                )
        pivot_of_row.append(c)
        r += 1
    for row, rhs in rows:
        if all(v == 0 for v in row) and rhs != 0:
            return CONTRADICTORY, None

    # 查询向量是否在行空间: 用主元行逐步消去
    query_vec = [Fraction(0)] * len(positions)
    for p in range(query_lo, query_hi + 1):
        query_vec[index[p]] = Fraction(1)
    total = Fraction(0)
    for row_idx, c in enumerate(pivot_of_row):
        if query_vec[c] != 0:
            factor = query_vec[c] / rows[row_idx][0][c]
            query_vec = [
                a - factor * b for a, b in zip(query_vec, rows[row_idx][0])
            ]
            total += factor * rows[row_idx][1]
    if all(v == 0 for v in query_vec):
        return DETERMINED, total
    return UNDERDETERMINED, None


def check_witnesses(records, report, query_lo, query_hi):
    """两份明细都必须满足全部校验, 且查询和不同."""
    for assignment, total in (
        (report.assignment_a, report.query_sum_a),
        (report.assignment_b, report.query_sum_b),
    ):
        for rec in records:
            assert (
                sum(assignment.get(p, 0) for p in range(rec.lo, rec.hi + 1))
                == rec.total
            ), f"明细不满足校验 {rec}"
        assert (
            sum(assignment.get(p, 0) for p in range(query_lo, query_hi + 1))
            == total
        )
    assert report.query_sum_a != report.query_sum_b


def test_determined_pure_addition():
    report = analyze([Record(1, 3, 10, "A"), Record(4, 5, 7, "B")], 1, 5)
    assert report.status == DETERMINED
    assert report.query_total == 17
    assert [t.coefficient for t in report.terms] == [1, 1]


def test_determined_with_subtraction():
    report = analyze([Record(1, 3, 10, "A"), Record(1, 2, 6, "B")], 3, 3)
    assert report.status == DETERMINED
    assert report.query_total == 4
    coeffs = {t.record.label: t.coefficient for t in report.terms}
    assert coeffs == {"A": 1, "B": -1}
    assert sum(t.contribution for t in report.terms) == 4


def test_single_point_query():
    report = analyze([Record(5, 5, -3, "A")], 5, 5)
    assert report.status == DETERMINED
    assert report.query_total == -3


def test_underdetermined_partial_overlap():
    # [1,2] 与 [2,3] 的校验无法决定 [1,3] 的和
    report = analyze([Record(1, 2, 5, "A"), Record(2, 3, 8, "B")], 1, 3)
    assert report.status == UNDERDETERMINED
    check_witnesses([Record(1, 2, 5), Record(2, 3, 8)], report, 1, 3)


def test_underdetermined_gap_in_coverage():
    report = analyze([Record(1, 2, 5, "A")], 1, 3)
    assert report.status == UNDERDETERMINED
    check_witnesses([Record(1, 2, 5)], report, 1, 3)


def test_underdetermined_no_records():
    report = analyze([], 3, 7)
    assert report.status == UNDERDETERMINED
    assert report.query_sum_a != report.query_sum_b


def test_contradictory_duplicate():
    report = analyze([Record(1, 2, 5, "A"), Record(1, 2, 6, "B")], 1, 2)
    assert report.status == CONTRADICTORY
    assert report.conflict_residual != 0
    lhs = sum(t.contribution for t in report.conflict_terms)
    assert lhs == report.conflict_residual


def test_contradictory_cycle():
    records = [Record(1, 2, 5, "A"), Record(3, 4, 5, "B"), Record(1, 4, 11, "C")]
    report = analyze(records, 1, 4)
    assert report.status == CONTRADICTORY
    # 组合系数对应的右端和即 residual: +C -A -B = 11-5-5 = 1 != 0
    assert sum(t.contribution for t in report.conflict_terms) != 0
    coeffs = {t.record.label: t.coefficient for t in report.conflict_terms}
    assert coeffs == {"A": -1, "B": -1, "C": 1}


def test_determined_via_cycle_consistent():
    # 三条记录自洽: [1,2]=5, [2,3]=5, [1,3]=10
    records = [Record(1, 2, 5, "A"), Record(2, 3, 5, "B"), Record(1, 3, 10, "C")]
    report = analyze(records, 1, 3)
    assert report.status == DETERMINED
    assert report.query_total == 10


def test_negative_and_zero_increments_allowed():
    records = [Record(1, 3, -4, "A"), Record(4, 4, 9, "B")]
    report = analyze(records, 1, 4)
    assert report.status == DETERMINED
    assert report.query_total == 5


def test_fuzz_against_brute_force():
    rng = random.Random(20261113)
    for trial in range(500):
        n_records = rng.randint(0, 5)
        records = []
        for i in range(n_records):
            lo = rng.randint(1, 8)
            hi = lo + rng.randint(0, 4)
            records.append(Record(lo, hi, rng.randint(-9, 9), f"R{i}"))
        qlo = rng.randint(1, 8)
        qhi = qlo + rng.randint(0, 4)

        report = analyze(records, qlo, qhi)
        expected_status, expected_total = brute_force(records, qlo, qhi)
        assert report.status == expected_status, (
            f"trial {trial}: {records} query=({qlo},{qhi}) "
            f"got {report.status}, want {expected_status}"
        )
        if report.status == DETERMINED:
            assert Fraction(report.query_total) == expected_total
            assert sum(t.contribution for t in report.terms) == report.query_total
        elif report.status == UNDERDETERMINED:
            check_witnesses(records, report, qlo, qhi)
        else:
            assert report.conflict_residual != 0
