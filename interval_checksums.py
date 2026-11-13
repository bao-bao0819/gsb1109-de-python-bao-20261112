"""区间流水号加总校验的可判定性分析.

背景: 归档服务保存了一批整数流水号区间 [lo, hi](含端点)的加总校验值,
区间可交叉、不保证连续覆盖; 每个位置的原始增量可正、可负、可为零.
给定查询区间 [L, R], 判定其总和是否被现有校验值唯一决定.

方法(前缀和 + 图连通性, 全部整数运算, 适合小批记录):

  记位置 p 的增量为 x[p], 前缀和 s[n] 为同一连续段内不超过 n 的增量之和.
  每条校验记录 "[lo, hi] 的总和为 t" 等价于约束

      s[hi] - s[lo-1] = t,

  即在前缀和节点 lo-1 与 hi 之间连一条权为 t 的边. 于是:

  * 矛盾: 某条边与已确定的前缀和差冲突 —— 沿环把各记录按 ±1 组合,
    推出 0 = d (d != 0), 说明原始校验互相矛盾;
  * 可确定: 节点 L-1 与 R 在同一连通分量 —— 任取一条路径逐项伸缩
    (telescoping) 即得查询和, 合成系数天然为 ±1;
  * 信息不足: 两节点不连通 —— 给 R 所在连通分量的前缀和整体 +1,
    所有校验值不变而查询和恰好 +1, 得到两份满足全部校验却得到
    不同查询和的整数明细.

复杂度: O(记录数 + 覆盖位置数), 无浮点误差.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

DETERMINED = "determined"
UNDERDETERMINED = "underdetermined"
CONTRADICTORY = "contradictory"


@dataclass(frozen=True)
class Record:
    """一条归档校验: 区间 [lo, hi](含端点)的增量总和为 total."""

    lo: int
    hi: int
    total: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"区间起点大于终点: [{self.lo}, {self.hi}]")

    def __str__(self) -> str:
        name = self.label or "记录"
        return f"{name}[{self.lo},{self.hi}]={self.total}"


@dataclass(frozen=True)
class Term:
    """合成项: 某条记录以 coefficient(+1 或 -1) 参与线性组合."""

    record: Record
    coefficient: int

    @property
    def contribution(self) -> int:
        return self.coefficient * self.record.total


@dataclass
class Report:
    """判定结果.

    status 为 DETERMINED 时: query_total 给出唯一总和, terms 给出合成明细;
    status 为 CONTRADICTORY 时: conflict_terms 中的记录按系数组合推出
    0 = conflict_residual, 说明校验值互相矛盾;
    status 为 UNDERDETERMINED 时: assignment_a / assignment_b 是两份满足
    全部校验的整数明细(位置 -> 增量), 但查询和分别为
    query_sum_a / query_sum_b, 二者不同.
    """

    status: str
    query: tuple[int, int]
    query_total: int | None = None
    terms: list[Term] = field(default_factory=list)
    conflict_terms: list[Term] = field(default_factory=list)
    conflict_residual: int = 0
    assignment_a: dict[int, int] = field(default_factory=dict)
    assignment_b: dict[int, int] = field(default_factory=dict)
    query_sum_a: int = 0
    query_sum_b: int = 0


def _merge_runs(intervals: list[tuple[int, int]]) -> list[list[int]]:
    """把覆盖到的整数位置合并成极大连续段 [lo, hi]."""
    runs: list[list[int]] = []
    for lo, hi in sorted(intervals):
        if runs and lo <= runs[-1][1] + 1:
            runs[-1][1] = max(runs[-1][1], hi)
        else:
            runs.append([lo, hi])
    return runs


def analyze(
    records: list[Record], query_lo: int, query_hi: int
) -> Report:
    """判定查询区间 [query_lo, query_hi] 的总和是否被 records 唯一决定."""
    if query_lo > query_hi:
        raise ValueError(f"查询区间起点大于终点: [{query_lo}, {query_hi}]")
    records = list(records)
    query = (query_lo, query_hi)

    runs = _merge_runs(
        [(rec.lo, rec.hi) for rec in records] + [query]
    )

    # 前缀和节点 n 表示 s[n]; 记录 [lo, hi] = t 是边 (lo-1, hi), 权为 t.
    # 邻接表元素: (邻接节点, 方向系数 ±1, 记录下标); 只含已处理的记录.
    adj: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    potential: dict[int, int] = {}  # 节点 -> 前缀和取值
    component: dict[int, int] = {}  # 节点 -> 连通分量代表元
    members: dict[int, set[int]] = {}  # 分量代表元 -> 节点集合

    def path_terms(src: int, dst: int) -> list[Term]:
        """同分量内从 src 到 dst 的路径, 返回合成项列表.

        路径上各项 contribution 之和恰为 s[dst] - s[src].
        """
        prev: dict[int, tuple[int, int, int] | None] = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                break
            for nxt, sign, idx in adj[node]:
                if nxt not in prev:
                    prev[nxt] = (node, sign, idx)
                    queue.append(nxt)
        terms: list[Term] = []
        node = dst
        while node != src:
            before, sign, idx = prev[node]  # type: ignore[misc]
            terms.append(Term(records[idx], sign))
            node = before
        terms.reverse()
        return terms

    # 逐条加入记录: 合并分量(平移前缀和)或检测矛盾.
    for idx, rec in enumerate(records):
        a, b = rec.lo - 1, rec.hi
        for node in (a, b):
            if node not in potential:
                potential[node] = 0
                component[node] = node
                members[node] = {node}
        root_a, root_b = component[a], component[b]
        if root_a == root_b:
            residual = potential[a] + rec.total - potential[b]
            if residual != 0:
                terms = path_terms(b, a) + [Term(rec, +1)]
                return Report(
                    status=CONTRADICTORY,
                    query=query,
                    conflict_terms=terms,
                    conflict_residual=residual,
                )
        else:
            # 平移较小分量并入较大分量, 使约束 s[b] - s[a] = total 成立.
            if len(members[root_a]) >= len(members[root_b]):
                shift = potential[a] + rec.total - potential[b]
                src_root, dst_root = root_b, root_a
            else:
                shift = potential[b] - rec.total - potential[a]
                src_root, dst_root = root_a, root_b
            for node in members[src_root]:
                potential[node] += shift
                component[node] = dst_root
            members[dst_root] |= members.pop(src_root)
        adj[a].append((b, +1, idx))
        adj[b].append((a, -1, idx))

    # 覆盖位置上的孤立前缀和节点(无任何记录经过)补 0, 各自成分量.
    for lo, hi in runs:
        for node in range(lo - 1, hi + 1):
            if node not in potential:
                potential[node] = 0
                component[node] = node

    src, dst = query_lo - 1, query_hi
    if component[src] == component[dst]:
        terms = path_terms(src, dst)
        total = sum(term.contribution for term in terms)
        return Report(
            status=DETERMINED, query=query, query_total=total, terms=terms
        )

    # 信息不足: 第二份明细把 dst 所在分量的前缀和整体 +1,
    # 所有校验(分量内部的差)不变, 查询和恰好 +1.
    def build_assignment(bump: int | None) -> dict[int, int]:
        assignment: dict[int, int] = {}
        for lo, hi in runs:
            for pos in range(lo, hi + 1):
                here = potential[pos] + (1 if component[pos] == bump else 0)
                prev = potential[pos - 1] + (
                    1 if component[pos - 1] == bump else 0
                )
                assignment[pos] = here - prev
        return assignment

    assignment_a = build_assignment(None)
    assignment_b = build_assignment(component[dst])
    sum_a = sum(assignment_a[p] for p in range(query_lo, query_hi + 1))
    sum_b = sum(assignment_b[p] for p in range(query_lo, query_hi + 1))
    return Report(
        status=UNDERDETERMINED,
        query=query,
        assignment_a=assignment_a,
        assignment_b=assignment_b,
        query_sum_a=sum_a,
        query_sum_b=sum_b,
    )


def _format_terms(terms: list[Term]) -> str:
    parts = []
    for term in terms:
        sign = "+" if term.coefficient > 0 else "-"
        parts.append(f"{sign} ({term.record})")
    return " ".join(parts)


def format_report(report: Report) -> str:
    """把判定结果格式化为人类可读的文本."""
    qlo, qhi = report.query
    head = f"查询区间 [{qlo}, {qhi}]: "
    if report.status == DETERMINED:
        lines = [head + f"可唯一确定, 总和 = {report.query_total}", "  合成明细:"]
        for term in report.terms:
            sign = "+" if term.coefficient > 0 else "-"
            lines.append(f"    {sign} {term.record}  (贡献 {term.contribution})")
        return "\n".join(lines)
    if report.status == CONTRADICTORY:
        return "\n".join(
            [
                head + "校验值互相矛盾",
                f"  组合 {_format_terms(report.conflict_terms)}",
                f"  推出 0 = {report.conflict_residual}, 不可能成立",
            ]
        )
    lines = [
        head + f"信息不足, 查询和不唯一 (可差出 {report.query_sum_b - report.query_sum_a})",
        "  两份均满足全部校验的明细 (位置: 增量):",
    ]
    for name, assignment, total in (
        ("甲", report.assignment_a, report.query_sum_a),
        ("乙", report.assignment_b, report.query_sum_b),
    ):
        body = ", ".join(f"{p}: {assignment[p]}" for p in sorted(assignment))
        lines.append(f"    明细{name} [{total}]: {body}")
    return "\n".join(lines)


if __name__ == "__main__":
    demos = [
        (
            "可确定(纯相加)",
            [Record(1, 3, 10, "A"), Record(4, 5, 7, "B")],
            (1, 5),
        ),
        (
            "可确定(含相减)",
            [Record(1, 3, 10, "A"), Record(1, 2, 6, "B")],
            (3, 3),
        ),
        (
            "信息不足",
            [Record(1, 2, 5, "A")],
            (1, 3),
        ),
        (
            "互相矛盾",
            [Record(1, 2, 5, "A"), Record(3, 4, 5, "B"), Record(1, 4, 11, "C")],
            (1, 4),
        ),
    ]
    for title, records, (qlo, qhi) in demos:
        print(f"=== {title} ===")
        for rec in records:
            print(f"  校验: {rec}")
        print(format_report(analyze(records, qlo, qhi)))
        print()
