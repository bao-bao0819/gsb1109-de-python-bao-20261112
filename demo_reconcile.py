"""cadastre_reconcile 的可执行对账演示（零第三方依赖）。

运行：python3 demo_reconcile.py
"""

from cadastre_reconcile import (
    PolygonRecord,
    analyze_matchings,
    attribute_candidates,
    diff_geometry,
    normalize_ring,
    presentation_diff,
    reconcile,
    ring_encoding_delta,
)

RING_A = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
RING_A_SHIFTED = [(2, 2), (0, 2), (0, 0), (2, 0), (2, 2)]  # 仅换起点，同向
RING_A_REVERSED = [(0, 2), (2, 2), (2, 0), (0, 0), (0, 2)]  # 真正反向且换起点
RING_B = [(10, 10), (12, 10), (12, 12), (10, 12), (10, 10)]
HOLE = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)]
RING_C_MOVED = [(0, 0), (3, 0), (3, 2), (0, 2), (0, 0)]  # 真实改动


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("1. 环规范化：换起点/反向被吸收")
    canon_a = normalize_ring(RING_A)
    canon_a2 = normalize_ring(RING_A_SHIFTED)
    canon_a3 = normalize_ring(RING_A_REVERSED)
    print("key 相等（移位/反向均被吸收）:",
          canon_a.key == canon_a2.key == canon_a3.key)
    delta1 = ring_encoding_delta(RING_A, RING_A_SHIFTED)
    delta2 = ring_encoding_delta(RING_A, RING_A_REVERSED)
    print("仅换起点 -> 反向:", delta1.reversed, " 位移:", delta1.shift)
    print("反向编码 -> 反向:", delta2.reversed, " 位移:", delta2.shift)

    section("2. part 重排与表达差异")
    old_multi = [[RING_A], [RING_B]]
    new_multi = [[RING_B], [RING_A_SHIFTED]]  # part 顺序交换 + 环换起点
    old_rec = PolygonRecord.build("M1", old_multi)
    new_rec = PolygonRecord.build("M2", new_multi)
    print("指纹相等（几何未改）:", old_rec.fingerprint == new_rec.fingerprint)
    print(presentation_diff(old_multi, new_multi))

    section("3. 真实几何改动与表达差异分离")
    old_hole = PolygonRecord.build("H1", [[RING_A, HOLE]])
    new_moved = PolygonRecord.build("H2", [[RING_C_MOVED, HOLE]])
    print("指纹相等:", old_hole.fingerprint == new_moved.fingerprint)
    print(diff_geometry(old_hole, new_moved))

    section("4. 多最大匹配：必然续接 vs 不可判定反例")
    old_records = [
        PolygonRecord.build("O1", [[RING_A]], {"name": "甲"}),
        PolygonRecord.build("O2", [[RING_A]], {"name": "乙"}),
        PolygonRecord.build("O3", [[RING_B]], {"name": "丙"}),
    ]
    new_records = [
        PolygonRecord.build("N1", [[RING_A_SHIFTED]], {"name": "乙"}),
        PolygonRecord.build("N2", [[RING_A]], {"name": "甲"}),
        PolygonRecord.build("N3", [[RING_B]], {"name": "丙"}),
    ]
    analysis = analyze_matchings(old_records, new_records)
    print("最大匹配数量:", len(analysis.all_max_matchings))
    for matching in analysis.all_max_matchings:
        print("  匹配:", sorted(matching, key=repr))
    print("必然续接（出现在每个最大匹配中）:", sorted(analysis.certain_pairs, key=repr))
    print("无法确定的边:", sorted(analysis.undecidable_edges, key=repr))

    section("5. 属性同名只是候选证据，不能裁决")
    candidates = attribute_candidates(old_records, new_records, "name")
    for oid, peers in sorted(candidates.items()):
        print(f"  {oid} 的同名候选: {sorted(peers, key=repr)}")
    print("注意：O1/N2 同名'甲'、O2/N1 同名'乙'，但交换配对同样是合法最大匹配，")
    print("几何判据无法区分 M_alpha={O1-N1,O2-N2} 与 M_beta={O1-N2,O2-N1}。")

    section("6. 完整对账结论")
    raw = {
        "O1": [[RING_A]], "O2": [[RING_A]], "O3": [[RING_B]],
        "N1": [[RING_A_SHIFTED]], "N2": [[RING_A]], "N3": [[RING_B]],
    }
    for verdict in reconcile(old_records, new_records, raw):
        print(verdict)


if __name__ == "__main__":
    main()
