# -*- coding: utf-8 -*-
"""
合表列映射的判定与校准参考实现（判定直接落在 xlrd 原始行上）。

背景
----
历史分区按表头名拼接，但同名/近名列里混有错误单元格、文本编号、数值编号。
本模块给出三条不可绕过的规则：

1. 允许列重排与明确别名：候选映射是逻辑列之间的一一双射，列位置不参与身份；
   别名必须来自显式别名表，表头抽样只证明“表头存在、语义同类”，不证明数据可连。
2. 连接键按 xlrd 的 ctype 打类型标签，永不做隐式文本→数值转换：
   文本 "00123" 与数值 123.0 是两个键；前导零承载身份时显示相同也不能合并。
3. XL_CELL_ERROR / XL_CELL_EMPTY / XL_CELL_BLANK 不参与连接，进隔离清单，
   绝不与任何键（含彼此）相等。

校准
----
给定有限的、人工核对过的校准行（已知应 1:1 命中的行号对），对每个候选映射：
- 复算连接基数与扇出/丢失；
- 与校准期望逐对比对；
- 触发前导零守卫或错误单元格阈值即淘汰。
所有幸存映射解释一致的列可先交付；仍有分歧的列挂起并写明补充证据。

处理真实 .xls 时，FakeSheet 的位置直接换成 xlrd.open_workbook 得到的 Sheet：
两者都暴露 nrows / row_types(rowx) / row_values(rowx)，rowx 从 0 开始。
注意 open_workbook(ragged_rows=True) 时用 row_len(rowx) 兜底；
BLANK 单元格只有 formatting_info=True 时才会出现 ctype=6。
"""

from __future__ import print_function

from xlrd import (
    XL_CELL_BLANK,
    XL_CELL_BOOLEAN,
    XL_CELL_DATE,
    XL_CELL_EMPTY,
    XL_CELL_ERROR,
    XL_CELL_NUMBER,
    XL_CELL_TEXT,
    error_text_from_code,
)

HEADER_ROW = 0

# ---------------------------------------------------------------------------
# 1. 类型保持的键规范化：身份 = (类型标签, 值)，不做任何跨类型折叠
# ---------------------------------------------------------------------------

NULL_KEY = ("NULL",)
ERROR_PREFIX = "ERROR"


def typed_key(ctype, value):
    """合规键：保留 xlrd 单元格类型，文本编号与数值编号天然不相等。"""
    if ctype in (XL_CELL_EMPTY, XL_CELL_BLANK):
        return NULL_KEY
    if ctype == XL_CELL_TEXT:
        return ("T", value)
    if ctype == XL_CELL_NUMBER:
        return ("N", value)
    if ctype == XL_CELL_DATE:
        # DATE 在 xlrd 里也是 float，必须单独打标签，禁止与 NUMBER 互等。
        return ("D", value)
    if ctype == XL_CELL_BOOLEAN:
        return ("B", value)
    if ctype == XL_CELL_ERROR:
        return (ERROR_PREFIX, value)
    raise ValueError("unknown ctype %r" % (ctype,))


def naive_display_key(ctype, value):
    """反例：按显示值把文本编号强转数值，专门用于演示被守卫淘汰的做法。"""
    if ctype == XL_CELL_TEXT:
        try:
            return ("N", float(value))
        except ValueError:
            return ("T", value)
    return typed_key(ctype, value)


def is_joinable(key):
    """单列键或复合键：任一分量为空/错误即整行不可连接，转入隔离清单。"""
    if key and isinstance(key[0], tuple):
        return all(component[0] not in ("NULL", ERROR_PREFIX) for component in key)
    return key[0] not in ("NULL", ERROR_PREFIX)


def leading_zero_text(ctype, value):
    return ctype == XL_CELL_TEXT and len(value) > 1 and value[0] == "0" and value.isdigit()


# ---------------------------------------------------------------------------
# 2. 样表：与真实 xlrd.Sheet 同构的最小实现（row_types / row_values / nrows）
# ---------------------------------------------------------------------------


class FakeSheet(object):
    def __init__(self, name, header, rows):
        # rows: 不含表头，每个元素是 [(ctype, value), ...]
        self.name = name
        self._header = list(header)
        self._rows = rows
        self.nrows = len(rows) + 1
        self.ncols = len(header)

    def row_types(self, rowx):
        if rowx == HEADER_ROW:
            return [XL_CELL_TEXT] * self.ncols
        return [cell[0] for cell in self._rows[rowx - 1]]

    def row_values(self, rowx):
        if rowx == HEADER_ROW:
            return list(self._header)
        return [cell[1] for cell in self._rows[rowx - 1]]

    def row_len(self, rowx):
        if rowx == HEADER_ROW:
            return self.ncols
        return len(self._rows[rowx - 1])

    def header(self):
        return self.row_values(HEADER_ROW)


# ---------------------------------------------------------------------------
# 3. 列画像与前导零守卫
# ---------------------------------------------------------------------------


def cell_at(sheet, rowx, colx):
    """ragged_rows=True 时短行尾部缺格按 XL_CELL_EMPTY 处理。"""
    if colx >= sheet.row_len(rowx):
        return XL_CELL_EMPTY, ""
    return sheet.row_types(rowx)[colx], sheet.row_values(rowx)[colx]


def profile_columns(sheet, data_rows):
    """返回每列的 ctype 分布、错误码、以及是否检出前导零文本编号。"""
    profiles = {}
    for colx in range(sheet.ncols):
        ctypes = {}
        errors = []
        has_leading_zero = False
        for rowx in data_rows:
            ctype, value = cell_at(sheet, rowx, colx)
            ctypes[ctype] = ctypes.get(ctype, 0) + 1
            if ctype == XL_CELL_ERROR:
                errors.append((rowx, colx, error_text_from_code.get(value, value)))
            if leading_zero_text(ctype, value):
                has_leading_zero = True
        profiles[colx] = {
            "ctypes": ctypes,
            "errors": errors,
            "zero_sensitive": has_leading_zero,
        }
    return profiles


def assert_no_numeric_collapse(normalize, sheet, colx, data_rows):
    """前导零守卫：文本编号列经规范化后，不得与去零数值键重合。"""
    seen = {}
    for rowx in data_rows:
        ctype, value = cell_at(sheet, rowx, colx)
        key = normalize(ctype, value)
        if not is_joinable(key):
            continue
        signature = (ctype, value)
        if key in seen and seen[key] != signature:
            return False, (seen[key], signature, key)
        seen[key] = signature
    return True, None


# ---------------------------------------------------------------------------
# 4. 候选映射：显式别名 + 语义同类内允许重排，双射
# ---------------------------------------------------------------------------
# 逻辑列 -> 新分区允许的表头别名。别名表是唯一的配对白名单：
# 抽样表头只证明“表头存在、语义同类”，不在表里的表头（如“渠道”）永远不参与配对。
ALIASES = {
    "机构编号": ("组织代码", "客户编号"),  # 两个编号列表头同类，抽样阶段无法区分
    "客户号": ("客户编号", "组织代码"),
    "客户名称": ("客户名称",),
    "金额": ("金额(元)", "金额(万元)"),
    "备注": ("注释",),
}

# 语义类：同类内允许重排；跨类不允许配对（ID 列不得映射到 MEASURE 列）
SEMANTIC_CLASS = {
    "机构编号": "ID",
    "客户号": "ID",
    "客户名称": "TEXT",
    "金额": "MEASURE",
    "备注": "TEXT",
    "组织代码": "ID",
    "客户编号": "ID",
    "金额(元)": "MEASURE",
    "金额(万元)": "MEASURE",
    "注释": "TEXT",
}


def enumerate_mappings(header_h, header_n):
    """枚举所有满足：别名白名单、语义同类、双射（列重排）的列映射。"""
    n_index = {name: idx for idx, name in enumerate(header_n)}
    allowed = {}
    for h_colx, h_name in enumerate(header_h):
        targets = []
        for alias in ALIASES[h_name]:
            if alias in n_index and SEMANTIC_CLASS[alias] == SEMANTIC_CLASS[h_name]:
                targets.append(n_index[alias])
        if not targets:
            raise ValueError("列 %r 在新分区找不到合规别名目标" % h_name)
        allowed[h_colx] = targets

    mappings = []

    def assign(colxes, chosen):
        if not colxes:
            mappings.append(dict(chosen))
            return
        h_colx = colxes[0]
        for n_colx in allowed[h_colx]:
            if n_colx not in chosen.values():  # 双射：一个目标列只能被配对一次
                chosen[h_colx] = n_colx
                assign(colxes[1:], chosen)
                del chosen[h_colx]

    assign(list(range(len(header_h))), {})
    return mappings


# ---------------------------------------------------------------------------
# 5. 在给定映射下复算连接，并与校准行比对
# ---------------------------------------------------------------------------


def key_at(sheet, rowx, colxes, normalize):
    key = []
    for colx in colxes:
        ctype, value = cell_at(sheet, rowx, colx)
        key.append(normalize(ctype, value))
    return tuple(key)


def guard_violations(sheet_h, sheet_n, mapping, key_cols, normalize, data_h, data_n):
    """任一连接键列发生跨类型折叠（如前导零编号被转数值）即整体否决。"""
    violations = []
    for h_colx in key_cols:
        ok_h, evidence_h = assert_no_numeric_collapse(
            normalize, sheet_h, h_colx, data_h)
        if not ok_h:
            violations.append(("H", h_colx, evidence_h))
        n_colx = mapping[h_colx]
        ok_n, evidence_n = assert_no_numeric_collapse(
            normalize, sheet_n, n_colx, data_n)
        if not ok_n:
            violations.append(("N", n_colx, evidence_n))
    return violations


def evaluate_mapping(sheet_h, sheet_n, mapping, key_cols, calibration,
                     normalize=typed_key, error_quota=0.0):
    """
    calibration 是有限人工证据，只约束被核对过的行：
      positive:   [(h_rowx, n_rowx), ...] 必须且只能 1:1 命中（校准行）
      negative:   [(h_rowx, n_rowx), ...] 身份不同，禁止连接（如前导零两机构）
      quarantine: [("H"|"N", rowx), ...]  含错误/空键，必须进隔离清单
    未被核对的全量行允许出现额外命中——这正是基数分歧的来源。
    """
    data_h = range(HEADER_ROW + 1, sheet_h.nrows)
    data_n = range(HEADER_ROW + 1, sheet_n.nrows)
    # H 侧读自身列；N 侧读映射后的列，两侧键的分量顺序必须一致
    n_key_cols = [mapping[colx] for colx in key_cols]

    violations = guard_violations(
        sheet_h, sheet_n, mapping, key_cols, normalize, data_h, data_n)

    quarantine = []
    n_buckets = {}
    for rowx in data_n:
        key = key_at(sheet_n, rowx, n_key_cols, normalize)
        if not is_joinable(key):
            quarantine.append(("N", rowx, key))
            continue
        n_buckets.setdefault(key, []).append(rowx)

    matched_pairs = []
    fan_out = []
    dropped = []
    for rowx in data_h:
        key = key_at(sheet_h, rowx, key_cols, normalize)
        if not is_joinable(key):
            quarantine.append(("H", rowx, key))
            dropped.append(rowx)
            continue
        hits = n_buckets.get(key, [])
        if len(hits) == 1:
            matched_pairs.append((rowx, hits[0]))
        elif len(hits) > 1:
            fan_out.append((rowx, hits))
        else:
            dropped.append(rowx)

    actual = set(matched_pairs)
    positive = set(calibration["positive"])
    negative = set(calibration.get("negative", []))
    expected_quarantine = set(tuple(x) for x in calibration.get("quarantine", []))
    actual_quarantine = set((side, rowx) for side, rowx, _ in quarantine)

    positive_rows = {h for h, _ in positive}
    fan_out_rows = {h for h, _ in fan_out}

    checks = {
        "guard": not violations,
        "positive_pairs": positive <= actual,
        "no_fanout_on_calibration": positive_rows.isdisjoint(fan_out_rows),
        "negative_pairs": negative.isdisjoint(actual),
        "quarantine": expected_quarantine <= actual_quarantine,
    }
    calibration_ok = all(checks.values())

    total = max(1, len(data_h) + len(data_n))
    # 配额只约束校准清单之外的“意外”错误行；已人工确认应隔离的行不算违规
    excess_quarantine = actual_quarantine - expected_quarantine
    err_ratio = float(len(excess_quarantine)) / total

    return {
        "mapping": mapping,
        "cardinality": len(matched_pairs),
        "matched_pairs": sorted(matched_pairs),
        "fan_out": fan_out,
        "dropped": dropped,
        "quarantine": quarantine,
        "guard_violations": violations,
        "checks": checks,
        "calibration_ok": calibration_ok,
        "missing_expected": sorted(positive - actual),
        "forbidden_joined": sorted(negative & actual),
        "missing_quarantine": sorted(expected_quarantine - actual_quarantine),
        "error_ratio": err_ratio,
        "error_quota_violated": err_ratio > error_quota,
    }


# ---------------------------------------------------------------------------
# 6. 演示
# ---------------------------------------------------------------------------


def build_demo():
    # 历史分区：机构编号是文本编号（前导零有身份含义），客户号是数值编号
    header_h = ["机构编号", "客户号", "客户名称", "金额", "备注"]
    rows_h = [
        # r1 校准行：机构 "00123" 与 r2 的 "123" 是两家不同机构
        [(XL_CELL_TEXT, "00123"), (XL_CELL_NUMBER, 1001.0),
         (XL_CELL_TEXT, "甲"), (XL_CELL_NUMBER, 100.0), (XL_CELL_EMPTY, "")],
        [(XL_CELL_TEXT, "123"), (XL_CELL_NUMBER, 1002.0),
         (XL_CELL_TEXT, "乙"), (XL_CELL_NUMBER, 200.0), (XL_CELL_TEXT, "常规")],
        # r3：错误单元格污染编号列，必须隔离，不得参与连接
        [(XL_CELL_ERROR, 0x2A), (XL_CELL_NUMBER, 1003.0),
         (XL_CELL_TEXT, "丙"), (XL_CELL_NUMBER, 300.0), (XL_CELL_TEXT, "#N/A 行")],
        # 全量行：用于暴露单位歧义映射的基数差异
        [(XL_CELL_TEXT, "555"), (XL_CELL_NUMBER, 1004.0),
         (XL_CELL_TEXT, "丁"), (XL_CELL_NUMBER, 100.0), (XL_CELL_TEXT, "")],
    ]

    # 新分区：列重排 + 明确别名；金额同时存在元/万元两列
    header_n = ["客户名称", "客户编号", "组织代码", "金额(元)", "注释", "渠道", "金额(万元)"]
    rows_n = [
        [(XL_CELL_TEXT, "甲"), (XL_CELL_NUMBER, 1001.0), (XL_CELL_TEXT, "00123"),
         (XL_CELL_NUMBER, 100.0), (XL_CELL_EMPTY, ""), (XL_CELL_TEXT, "线上"),
         (XL_CELL_NUMBER, 100.0)],   # 校准行上两列数值恰好相同，单位歧义无法被抽样发现
        [(XL_CELL_TEXT, "乙"), (XL_CELL_NUMBER, 1002.0), (XL_CELL_TEXT, "123"),
         (XL_CELL_NUMBER, 200.0), (XL_CELL_TEXT, "常规"), (XL_CELL_TEXT, "线下"),
         (XL_CELL_NUMBER, 200.0)],
        [(XL_CELL_TEXT, "丙"), (XL_CELL_NUMBER, 1003.0), (XL_CELL_TEXT, "999"),
         (XL_CELL_NUMBER, 300.0), (XL_CELL_TEXT, "#N/A 行"), (XL_CELL_TEXT, "线上"),
         (XL_CELL_NUMBER, 0.03)],
        [(XL_CELL_TEXT, "丁"), (XL_CELL_NUMBER, 1004.0), (XL_CELL_TEXT, "555"),
         (XL_CELL_NUMBER, 100.0), (XL_CELL_EMPTY, ""), (XL_CELL_TEXT, "线上"),
         (XL_CELL_NUMBER, 0.01)],   # 万元列：正确映射能连，单位错映射连不上
        [(XL_CELL_TEXT, "戊"), (XL_CELL_NUMBER, 1005.0), (XL_CELL_TEXT, "777"),
         (XL_CELL_NUMBER, 90.0), (XL_CELL_TEXT, ""), (XL_CELL_TEXT, "线下"),
         (XL_CELL_NUMBER, 100.0)],  # 万元列又一个 100.0：单位错映射在此扇出
    ]
    return FakeSheet("H", header_h, rows_h), FakeSheet("N", header_n, rows_n)


def main():
    sheet_h, sheet_n = build_demo()
    data_h = range(HEADER_ROW + 1, sheet_h.nrows)
    data_n = range(HEADER_ROW + 1, sheet_n.nrows)

    print("== 列画像（前导零守卫输入） ==")
    for colx, profile in profile_columns(sheet_h, data_h).items():
        print("H 列%d %r: ctypes=%r zero_sensitive=%r errors=%r" % (
            colx, sheet_h.header()[colx], profile["ctypes"],
            profile["zero_sensitive"], profile["errors"]))

    print("\n== 反例：显示值强转数值会折叠身份 ==")
    trap_col = 0
    ok, evidence = assert_no_numeric_collapse(
        naive_display_key, sheet_h, trap_col, data_h)
    print("naive_display_key 通过前导零守卫: %r" % ok)
    if evidence:
        print("  折叠证据: %r 与 %r 被归并到键 %r" % evidence)
    ok, _ = assert_no_numeric_collapse(typed_key, sheet_h, trap_col, data_h)
    print("typed_key     通过前导零守卫: %r" % ok)

    key_cols = [0, 1, 3]  # 机构编号、客户号、金额 构成连接键
    # 有限人工校准证据（rowx 从 0 起，0 为表头）：
    # positive   只核对 r1<->r1、r2<->r2，必须 1:1；
    # negative   "00123" 与 "123" 是两家机构，这两对禁止连上；
    # quarantine H 表 r3 的编号是 #N/A 错误单元格，必须隔离、禁止参与连接。
    calibration = {
        "positive": [(1, 1), (2, 2)],
        "negative": [(1, 2), (2, 1)],
        "quarantine": [("H", 3)],
    }

    mappings = enumerate_mappings(sheet_h.header(), sheet_n.header())
    print("\n== 候选映射 %d 个（均通过抽样表头与双射约束） ==" % len(mappings))

    survivors = []
    for idx, mapping in enumerate(mappings):
        report = evaluate_mapping(sheet_h, sheet_n, mapping, key_cols, calibration)
        label = describe(mapping, sheet_h, sheet_n)
        print("\n[映射%d] %s" % (idx, label))
        print("  全量连接基数=%d 扇出=%r 丢失行=%r 隔离=%d 行" % (
            report["cardinality"], report["fan_out"],
            report["dropped"], len(report["quarantine"])))
        print("  各项校准=%r" % report["checks"])
        print("  校准通过=%r 缺正例=%r 违禁连接=%r 缺隔离=%r 错误配额违规=%r" % (
            report["calibration_ok"], report["missing_expected"],
            report["forbidden_joined"], report["missing_quarantine"],
            report["error_quota_violated"]))
        if report["calibration_ok"] and not report["error_quota_violated"]:
            survivors.append((mapping, report))

    print("\n== 对照：若把文本编号强转数值（naive_display_key），全部映射被守卫否决 ==")
    for idx, mapping in enumerate(mappings):
        report = evaluate_mapping(sheet_h, sheet_n, mapping, key_cols, calibration,
                                  normalize=naive_display_key)
        print("[映射%d] guard=%r 折叠证据=%r" % (
            idx, report["checks"]["guard"],
            [v[2] for v in report["guard_violations"]]))

    print("\n== 幸存映射 %d 个，求列级共识 ==" % len(survivors))
    consensus, contested = split_consensus(survivors)
    for h_colx, n_colx in consensus:
        print("  可先交付: %s -> %s（所有幸存解释一致）" % (
            sheet_h.header()[h_colx], sheet_n.header()[n_colx]))
    for h_colx, _ in contested:
        options = sorted(set(
            sheet_n.header()[mapping[h_colx]] for mapping, _ in survivors))
        print("  挂起: %s 在幸存映射中目标为 %r；各解释全量基数=%r，需补含单位差异的校准行" % (
            sheet_h.header()[h_colx], options,
            sorted(set(report["cardinality"] for _, report in survivors))))


def describe(mapping, sheet_h, sheet_n):
    return ", ".join(
        "%s->%s" % (sheet_h.header()[h], sheet_n.header()[n])
        for h, n in sorted(mapping.items())
    )


def split_consensus(survivors):
    if not survivors:
        return [], []
    first = survivors[0][0]
    consensus, contested = [], []
    for h_colx, n_colx in first.items():
        if all(mapping[h_colx] == n_colx for mapping, _ in survivors):
            consensus.append((h_colx, n_colx))
        else:
            contested.append((h_colx, None))
    return consensus, contested


if __name__ == "__main__":
    main()
