"""补测档位选取（小规模可执行办法）。

模型：
- 4 个独立随机源 S1..S4，每个源有互斥的补测档位（成本, 补测后方差），档位 0 = 不补测。
- 共用的标定误差方差 CALIB_VAR 加到每个指标上，不随补测次数下降（方差下限）。
- 三个日报指标是源误差的线性组合：M_j = sum_i w[j][i] * e_i + 1.0 * e_calib
  独立源 => Var(M_j) = sum_i w[j][i]^2 * Var(e_i) + CALIB_VAR。
- 目标：预算内选一档/源，使 max_j Var(M_j)/TOL_j 最小。

办法：4 个源 × 3 档 = 81 种组合，直接穷举，过滤超预算，取最大比值最小者。
对照：贪心规则“按单列误差（源的基准方差）从大到小，买得起就买最高档”。
"""

from itertools import product

CALIB_VAR = 0.5  # 共用标定误差方差，任何档位都降不了

SOURCES = ["S1", "S2", "S3", "S4"]
TIERS = {  # 每源互斥档位: (成本, 补测后方差)
    "S1": [(0, 4.0), (3, 2.0), (5, 1.0)],
    "S2": [(0, 3.0), (2, 1.5), (4, 0.8)],
    "S3": [(0, 2.5), (2, 1.2), (3, 0.6)],
    "S4": [(0, 1.5), (1, 0.7), (2, 0.4)],
}
WEIGHTS = [  # 行 = 指标 M1/M2/M3，列 = 源 S1..S4
    [1.0, 0.3, 0.2, 0.1],
    [0.2, 1.0, 0.4, 0.2],
    [0.1, 0.3, 1.0, 0.8],
]
TOL = [2.0, 2.0, 1.5]  # 三个指标的容许方差
BUDGET = 7


def metric_variances(selection):
    """selection: {源: 档位下标} -> 三个指标的剩余方差（含标定下限）。"""
    variances = []
    for row in WEIGHTS:
        var = CALIB_VAR
        for source, weight in zip(SOURCES, row):
            var += weight**2 * TIERS[source][selection[source]][1]
        variances.append(var)
    return variances


def ratios(selection):
    return [v / t for v, t in zip(metric_variances(selection), TOL)]


def cost_of(selection):
    return sum(TIERS[s][selection[s]][0] for s in SOURCES)


def brute_force(budget):
    best = None
    for combo in product(range(3), repeat=4):
        selection = dict(zip(SOURCES, combo))
        cost = cost_of(selection)
        if cost > budget:
            continue
        key = (max(ratios(selection)), sum(ratios(selection)), cost)
        if best is None or key < best[0]:
            best = (key, selection)
    return best[1]


def greedy_by_column_error(budget):
    """按单列误差从大到小排序，依次买能负担的最高档（会选错的朴素规则）。"""
    selection = {s: 0 for s in SOURCES}
    remaining = budget
    order = sorted(SOURCES, key=lambda s: -TIERS[s][0][1])  # 基准方差降序
    for source in order:
        for tier in (2, 1):
            cost, _ = TIERS[source][tier]
            if cost <= remaining:
                selection[source] = tier
                remaining -= cost
                break
    return selection


def show(title, selection):
    variances = metric_variances(selection)
    ratio_list = ratios(selection)
    print(f"\n== {title} ==")
    print("选择:", {s: f"档位{t}" for s, t in selection.items()},
          f"| 总成本 {cost_of(selection)}")
    for j, (var, ratio) in enumerate(zip(variances, ratio_list), 1):
        print(f"  M{j}: 剩余方差 {var:.3f} (含标定下限 {CALIB_VAR}) "
              f"| 方差/容许 = {ratio:.3f}")
    print(f"  最大比值 = {max(ratio_list):.3f}")


if __name__ == "__main__":
    optimal = brute_force(BUDGET)
    greedy = greedy_by_column_error(BUDGET)
    show(f"最优选取（穷举 81 种，预算 {BUDGET}）", optimal)
    show(f"贪心对照（按单列误差从大到小，预算 {BUDGET}）", greedy)

    print("\n== 一批“按单列误差从大到小会选错”的记录 ==")
    print(f"{'记录':>4} {'预算':>4} {'贪心选择':<28} {'贪心最大比':>8} "
          f"{'最优选择':<28} {'最优最大比':>8} {'选错?':>4}")
    records = 0
    for record_id, budget in enumerate(range(4, 11), 1):
        opt = brute_force(budget)
        grd = greedy_by_column_error(budget)
        opt_max, grd_max = max(ratios(opt)), max(ratios(grd))
        wrong = grd_max > opt_max + 1e-9
        records += wrong
        fmt = lambda sel: ",".join(f"{s}:t{sel[s]}" for s in SOURCES)
        print(f"{record_id:>4} {budget:>4} {fmt(grd):<28} {grd_max:>8.3f} "
              f"{fmt(opt):<28} {opt_max:>8.3f} {'是' if wrong else '否':>4}")
    print(f"\n共 {records} 条记录中贪心规则选错（最大比值严格劣于最优）。")
