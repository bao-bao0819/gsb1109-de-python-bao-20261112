#!/usr/bin/env python
"""
结算表"日期列"审计演示（基于 xlrd 2.x 的真实语义）。

排查四件事：
1. 同一序列号在 1900/1904 两种 datemode 下相差 1462 天（约 4 年）的纪元分歧；
2. 纯时间（时刻）与时长（elapsed）被同一套日期单元格类型误认；
3. 两条导出文本完全相同、但正确分区不同的行，以及区分它们所需的最少源信息；
4. 1900 伪闰日与丢失工作簿来源后的可挽回范围。

运行：.venv/bin/python scripts/date_audit_demo.py
"""
import datetime

import xlrd
from xlrd import xldate
from xlrd.formatting import is_date_format_string
from xlrd.sheet import ctype_text
from xlrd.xldate import XLDateAmbiguous, XLDateNegative, XLDateTooLarge

EPOCH_OFFSET_DAYS = 1462  # 同一物理日期在两种 datemode 下的序列号平移量


class _QuietBook:
    verbosity = 0

    class logfile:
        @staticmethod
        def write(*_args):
            pass

        @staticmethod
        def flush():
            pass


def is_time_only_format(fmt_str):
    """日期类格式且剥掉方括号后不含 y/d（年/日）字符 => 时刻或时长格式。"""
    if not fmt_str or not is_date_format_string(_QuietBook(), fmt_str):
        return False
    return not any(ch in fmt_str.lower() for ch in ("y", "d"))


def classify(serial, ctype, fmt_str, datemode):
    """按 xlrd 的实际解码规则给一行归类。"""
    if ctype != xlrd.XL_CELL_DATE:
        return "非日期单元格，按数值/文本搬运，不应走日期解码"
    if serial < 0:
        return "负序列号：tuple API 报 XLDateNegative；datetime API 给纪元前日期"
    if datemode == 0 and 1.0 <= serial < 61.0:
        return "1900 伪闰日窗口（1<=serial<61）：tuple API 报 XLDateAmbiguous"
    if 0.0 <= serial < 1.0:
        return "纯时间（一天内的时刻），无日期部分"
    if is_time_only_format(fmt_str):
        return "时长（elapsed）：序列号>=1 但格式只含 h/m/s，不是日期"
    if serial >= xldate._XLDAYS_TOO_LARGE[datemode]:
        return "超界：tuple API 报 XLDateTooLarge"
    return "日期/日期时间，按 datemode 正常解码"


def safe_tuple(serial, datemode):
    try:
        return xldate.xldate_as_tuple(serial, datemode)
    except (XLDateAmbiguous, XLDateNegative, XLDateTooLarge) as exc:
        return type(exc).__name__


def safe_datetime(serial, datemode):
    try:
        return xldate.xldate_as_datetime(serial, datemode)
    except (OverflowError, XLDateTooLarge) as exc:
        return "%s: %s" % (type(exc).__name__, exc)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def main():
    section("1. 纪元分歧：同一序列号在两种 datemode 下的差（正常区间 1462 天）")
    print("datemode=0 纪元 1899-12-30；datemode=1 纪元 1904-01-01")
    print("工作簿 datemode 来自 DATEMODE 记录，缺失时 xlrd 默认 0")
    print("serial>=61 时差恒为 1462 天；serial 1..60 因 1900 伪闰日只有 1461 天\n")
    for serial in (0.0, 1.0, 60.0, 61.0, 38406.0):
        d0 = safe_datetime(serial, 0)
        d1 = safe_datetime(serial, 1)
        delta = (d1 - d0).days if isinstance(d0, datetime.datetime) and isinstance(
            d1, datetime.datetime
        ) else "-"
        print(
            "serial=%-8s mode0 -> %-22s mode1 -> %-22s 相差 %s 天"
            % (serial, str(d0), str(d1), delta)
        )

    section("2. 行分类：原始数值 + 单元格类型 + 显示格式 + datemode")
    rows = [
        # serial, ctype, fmt_str, datemode, 备注
        (38406.0, xlrd.XL_CELL_DATE, "yyyy-mm-dd", 0, "普通日期（1900 系统）"),
        (36944.0, xlrd.XL_CELL_DATE, "yyyy-mm-dd", 1, "同一物理日期（1904 系统）"),
        (0.75, xlrd.XL_CELL_DATE, "hh:mm:ss", 0, "时刻 18:00:00"),
        (0.75, xlrd.XL_CELL_DATE, "[h]:mm:ss", 0, "时长 18 小时"),
        (1.5, xlrd.XL_CELL_DATE, "hh:mm:ss", 0, "时刻只显示小数部分 12:00:00（同时落入伪闰日窗口）"),
        (1.5, xlrd.XL_CELL_DATE, "[h]:mm:ss", 0, "时长 36:00:00（>24h 时长与伪闰日窗口重叠）"),
        (60.0, xlrd.XL_CELL_DATE, "yyyy-mm-dd", 0, "伪闰日 1900-02-29"),
        (60.0, xlrd.XL_CELL_DATE, "yyyy-mm-dd", 1, "1904 系统下是 1904-03-01"),
        (38406.0, xlrd.XL_CELL_NUMBER, "General", 0, "数值单元格，从未被标记为日期"),
        (-1.0, xlrd.XL_CELL_DATE, "yyyy-mm-dd", 1, "负序列号"),
    ]
    for serial, ctype, fmt_str, datemode, note in rows:
        print(
            "\n%-28s serial=%-8s ctype=%-11s fmt=%-12s datemode=%d"
            % (note, serial, ctype_text[ctype], fmt_str, datemode)
        )
        print("   tuple API :", safe_tuple(serial, datemode))
        print("   datetime  :", safe_datetime(serial, datemode))
        print("   分区      :", classify(serial, ctype, fmt_str, datemode))

    section("3. 两条导出文本完全相同、正确分区不同的行")
    print("\n[对 A] 纪元孪生：文本都是 2005-02-23，原始序列号与 datemode 都不同")
    pair_a = [
        ("行A1", 38406.0, 0, "yyyy-mm-dd"),
        ("行A2", 38406.0 - EPOCH_OFFSET_DAYS, 1, "yyyy-mm-dd"),
    ]
    for name, serial, datemode, fmt_str in pair_a:
        dt = safe_datetime(serial, datemode)
        print(
            "  %s 导出文本=%s  原始 serial=%-7s datemode=%d  (%s)"
            % (name, dt.strftime("%Y-%m-%d"), serial, datemode, fmt_str)
        )
    print("  => 最少源信息：serial + datemode；只留文本则不可区分")

    print("\n[对 B] 时刻/时长孪生：serial 与文本都为 18:00:00，仅格式串不同")
    pair_b = [
        ("行B1", 0.75, 0, "hh:mm:ss", "时刻（每天 18:00）"),
        ("行B2", 0.75, 0, "[h]:mm:ss", "时长（18 个小时）"),
    ]
    for name, serial, datemode, fmt_str, meaning in pair_b:
        hh, mm = divmod(round(serial * 1440), 60)
        print(
            "  %s 导出文本=%02d:%02d:00  serial=%s  fmt=%-10s 含义=%s"
            % (name, hh, mm, serial, fmt_str, meaning)
        )
    print("  => 两行 ctype 都是 XL_CELL_DATE，类型救不了；")
    print("     最少源信息：format_str（方括号 [h]/[mm]）或该列的业务定义")

    section("4. 1900 伪闰日与来源丢失后的可挽回范围")
    print("\n伪闰日窗口（仅 datemode=0，serial 1..60）:")
    for serial in (59.0, 60.0, 61.0):
        print(
            "  serial=%-3s tuple=%-20s datetime=%s"
            % (serial, str(safe_tuple(serial, 0)), safe_datetime(serial, 0))
        )
    print("  serial 59 与 60 在 datetime API 下碰撞为同一天 1900-02-28；")
    print("  serial 60 即 Excel 虚构的 1900-02-29，无真实公历日期可映射，应隔离标记。")
    print("  serial >= 61 解码精确，不受伪闰日影响；datemode=1 无此窗口。")

    print("\n来源丢失的汇总行：")
    print("  - 只剩文本：serial/datemode 均不可恢复，只能回源系统重取；")
    print("  - 剩 serial、丢 datemode：每个 serial 都有两个候选（正常区间相差 1462 天，")
    print("    serial 1..60 受伪闰日影响为 1461 天），只能靠业务日期区间/同表兄弟行/文件元数据反推；")
    print("    0<=serial<1 的纯时间与纪元无关，仍可恢复；")
    print("  - 剩 serial+datemode、丢格式：日期解码仍精确，但时刻与时长（0<=serial<1）不可分，")
    print("    需要列的业务定义；")
    print("  - 负序列号/超界值：两个 xlrd API 行为不同，tuple 抛错、datetime 宽松，审计时必须固定 API。")


if __name__ == "__main__":
    main()
