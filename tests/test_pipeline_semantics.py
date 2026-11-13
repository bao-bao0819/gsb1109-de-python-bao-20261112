"""裁剪 -> 降采样 -> 方向翻转 局部变换链的语义验算。

链顺序（外角/corner 约定，局部像元索引为列、行）::

    结果变换 = 降采样变换 DS @ 局部翻转 L

供应方的局部翻转有两种原点约定：

* 外角原点（corner）：L 直接作用于像元外角索引，(0, 0) 是左上角外角。
* 像元中心原点（center）：C = L @ translation(0.5, 0.5)，消费方必须先做
  L = C @ translation(-0.5, -0.5) 还原（与 affine.loadsw 的处理一致）。

两类待区分的错误：

* 次序写反：L @ DS（应为 DS @ L）。
* 中心约定写反：拿到 C 却直接当 L 用，漏掉 -0.5 平移。

验算用图幅：6x5 母图、像元 2x3 世界单位、裁 4x4 窗口、2 倍降采样，
输出 2x2；母图像元值编码为 100*(行+1)+(列+1)，每个像元都带行列号。
"""

import numpy as np

from affine import Affine

# 母图与链参数 -------------------------------------------------------------
NX, NY = 6, 5
PX, PY = 2.0, 3.0
ORIGIN_X, ORIGIN_Y = 100.0, 50.0
COL0, ROW0 = 1, 1
WIN = 4
K = 2
COARSE = WIN // K

HALF = Affine.translation(0.5, 0.5)
NEG_HALF = Affine.translation(-0.5, -0.5)

# 外角约定：变换把 (列, 行) 外角索引映射到世界坐标
PARENT = Affine(PX, 0.0, ORIGIN_X, 0.0, -PY, ORIGIN_Y)
CROP = Affine(
    PX,
    0.0,
    ORIGIN_X + COL0 * PX,
    0.0,
    -PY,
    ORIGIN_Y - ROW0 * PY,
)
DS = Affine(K * PX, 0.0, CROP.c, 0.0, -K * PY, CROP.f)

# 方向 -> (x 轴向, y 轴向)，-1 表示该轴翻转
ORIENTATIONS = {
    "N": (1.0, 1.0),
    "W": (-1.0, 1.0),  # 左右镜像
    "E": (1.0, -1.0),  # 上下镜像
    "S": (-1.0, -1.0),  # 180 度
}


def labeled_parent():
    """返回带行列编号的母图，值 100*(行+1)+(列+1)。"""
    return np.array(
        [[100 * (r + 1) + (c + 1) for c in range(NX)] for r in range(NY)],
        dtype=int,
    )


def window_flip(sx, sy):
    """绕窗口外边翻转的局部变换（保持 footprint 落在原窗口上）。"""
    tx = float(COARSE) if sx < 0 else 0.0
    ty = float(COARSE) if sy < 0 else 0.0
    return Affine.translation(tx, ty) @ Affine(sx, 0.0, 0.0, 0.0, sy, 0.0)


def center_form(local):
    """外角约定变换 -> 像元中心约定变换（供应方视角）。"""
    return local @ HALF


def chain_corner(local):
    """正确链：外角约定局部变换。"""
    return DS @ local


def chain_center_supplied(center):
    """正确链：供应方给中心约定时先还原回外角约定。"""
    return DS @ (center @ NEG_HALF)


def chain_order_reversed(local):
    """错误链 A：变换次序写反。"""
    return local @ DS


def chain_center_misread(center):
    """错误链 B：中心约定被当成外角约定，漏掉 -0.5 平移。"""
    return DS @ center


def expected_dn(sx, sy, parent):
    """按方向语义直接重排得到的输出像元编号。

    输出像元中心对应粗像元块的中心细像元（K 为偶数时取块内第 K//2 个），
    与 resample() 反算像元中心的取法严格对齐。
    """
    mid = K // 2
    out = np.empty((COARSE, COARSE), dtype=int)
    for r in range(COARSE):
        for c in range(COARSE):
            src_c = COARSE - 1 - c if sx < 0 else c
            src_r = COARSE - 1 - r if sy < 0 else r
            out[r, c] = parent[ROW0 + K * src_r + mid, COL0 + K * src_c + mid]
    return out


def resample(transform, parent):
    """用给定输出变换反算每个输出像元中心对应的母图像元，越界记 -1。"""
    inv_parent = ~PARENT
    out = np.full((COARSE, COARSE), -1, dtype=int)
    for r in range(COARSE):
        for c in range(COARSE):
            wx, wy = transform @ (c + 0.5, r + 0.5)
            pc, pr = inv_parent @ (wx, wy)
            pc, pr = round(pc), round(pr)
            if 0 <= pc < NX and 0 <= pr < NY:
                out[r, c] = parent[pr, pc]
    return out


def envelope(transform):
    """输出栅格四角外包范围 (xmin, xmax, ymin, ymax)。"""
    pts = [transform @ p for p in ((0, 0), (COARSE, 0), (COARSE, COARSE), (0, COARSE))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def steps(transform):
    """相邻格点步向：列方向一步、行方向一步的世界坐标增量。"""
    return (
        tuple(np.array(transform @ (1, 0)) - np.array(transform @ (0, 0))),
        tuple(np.array(transform @ (0, 1)) - np.array(transform @ (0, 0))),
    )


def shift_at(transform, good, col, row):
    """候选变换相对正确变换在指定检查点的世界坐标偏移。"""
    return tuple(np.array(transform @ (col, row)) - np.array(good @ (col, row)))


def classify(good, candidate):
    """依据 K0=(0,0) 单点偏移量级判定错误类别。"""
    shift = shift_at(candidate, good, 0, 0)
    if np.allclose(shift, (0.0, 0.0), atol=1e-9):
        return "正确"
    half_cell = (good.a * 0.5, good.e * 0.5)
    if np.allclose(shift, half_cell, atol=1e-9):
        return "中心约定写反（偏移恰为半个粗像元）"
    return "变换次序写反（偏移取决于原点/窗口位置，量级远大于半像元）"


def test_correct_chain_matches_labeled_reference():
    """两种约定走正确链，结果变换一致且重排编号与方向语义完全吻合。"""
    parent = labeled_parent()
    for name, (sx, sy) in ORIENTATIONS.items():
        local = window_flip(sx, sy)
        center = center_form(local)
        good = chain_corner(local)
        assert chain_center_supplied(center).almost_equals(good)
        np.testing.assert_array_equal(
            resample(good, parent), expected_dn(sx, sy, parent)
        )


def test_four_corner_envelope_cannot_discriminate():
    """三类结果的四角外包尺寸完全相同，包络检查全部放行。"""
    local = window_flip(*ORIENTATIONS["W"])
    good = chain_corner(local)
    bug_order = chain_order_reversed(local)
    bug_center = chain_center_misread(center_form(local))
    extents = [envelope(t) for t in (good, bug_order, bug_center)]
    widths = {xmax - xmin for xmin, xmax, _, _ in extents}
    heights = {ymax - ymin for _, _, ymin, ymax in extents}
    assert widths == {K * PX * COARSE}
    assert heights == {K * PY * COARSE}


def test_minimal_checkpoints_classify_the_two_bugs():
    """K0 外角 + K1 内部带号像元即可区分；两类错误下步向均不变。"""
    parent = labeled_parent()
    # N 是恒等翻转：L=I 时 L@DS == DS@L，次序写反与正确链完全相同，
    # 属于对称样本盲区（见 test_symmetric_samples_hide_errors），此处排除。
    for name, (sx, sy) in ORIENTATIONS.items():
        local = window_flip(sx, sy)
        good = chain_corner(local)
        bug_order = chain_order_reversed(local)
        bug_center = chain_center_misread(center_form(local))
        good_dn = resample(good, parent)

        # 中心约定错误在任何方向（含 N）都留下半个粗像元的恒定偏移
        assert classify(good, bug_center).startswith("中心约定写反")
        assert shift_at(bug_center, good, 0, 0) == shift_at(bug_center, good, 1, 1)
        assert steps(bug_center) == steps(good)
        assert np.allclose(
            shift_at(bug_center, good, 0, 0),
            (good.a * 0.5, good.e * 0.5),
        )
        assert not np.array_equal(resample(bug_center, parent), good_dn)

        if name == "N":
            continue

        # 真正含翻转的方向：K0 单点偏移量级即可判定次序写反
        assert classify(good, bug_order).startswith("变换次序写反")
        assert shift_at(bug_order, good, 0, 0) == shift_at(bug_order, good, 1, 1)
        assert steps(bug_order) == steps(good)
        assert not np.array_equal(resample(bug_order, parent), good_dn)


def test_symmetric_samples_hide_errors():
    """对称样本上哪些错误观察不到。"""
    # 1. 均匀图幅：四种方向重排出来的编号完全一致，翻转在数据中不可见
    uniform = np.zeros((NY, NX), dtype=int)
    tables = {
        name: expected_dn(sx, sy, uniform) for name, (sx, sy) in ORIENTATIONS.items()
    }
    assert all(np.array_equal(t, tables["N"]) for t in tables.values())

    # 2. 中心对称的 4x4 窗口内容：180 度翻转与不翻转在数据上不可区分
    symmetric = np.tile(np.array([[1, 2], [2, 1]]), (2, 2))
    assert np.array_equal(symmetric, symmetric[::-1, ::-1])
    padded = np.pad(
        symmetric,
        ((ROW0, NY - WIN - ROW0), (COL0, NX - WIN - COL0)),
    )
    assert np.array_equal(expected_dn(1, 1, padded), expected_dn(-1, -1, padded))

    # 3. 恒等翻转（方向 N，L=I）：次序写反在数学上与正确链完全相同
    identity_flip = window_flip(1.0, 1.0)
    assert (DS @ identity_flip).almost_equals(identity_flip @ DS)

    # 4. 零原点 + 绕原点翻转：次序写反在数学上与正确链完全相同
    ds_zero = Affine(K * PX, 0.0, 0.0, 0.0, -K * PY, 0.0)
    mirror = Affine(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert (ds_zero @ mirror).almost_equals(mirror @ ds_zero)

    # 5. 半像元级偏移在粗容差/向外取整的包络检查下被放行
    local = window_flip(*ORIENTATIONS["W"])
    good = chain_corner(local)
    bug_center = chain_center_misread(center_form(local))
    tol = max(abs(good.a), abs(good.e)) * 0.5
    assert np.allclose(envelope(bug_center), envelope(good), atol=tol + 1e-9)


def test_step_vectors_catch_doubled_flip():
    """步向用于抓轴向符号错误；该错误在均匀样本中同样不可见。"""
    local_w = window_flip(*ORIENTATIONS["W"])
    local_s = window_flip(*ORIENTATIONS["S"])
    good = chain_corner(local_w)
    wrong = chain_corner(local_s)
    assert steps(good) != steps(wrong)
    uniform = np.zeros((NY, NX), dtype=int)
    assert np.array_equal(resample(good, uniform), resample(wrong, uniform))


def _print_report():
    parent = labeled_parent()
    print("母图（像元值=100*(行+1)+(列+1)）：")
    print(parent)
    print(f"\nPARENT={PARENT}\nCROP  ={CROP}\nDS    ={DS}")

    name = "W"
    sx, sy = ORIENTATIONS[name]
    local = window_flip(sx, sy)
    center = center_form(local)
    good = chain_corner(local)
    bug_order = chain_order_reversed(local)
    bug_center = chain_center_misread(center)

    print(f"\n===== 方向 {name}：局部翻转 L={local}")
    print(f"中心约定 C=L@T(+0.5)={center}")
    for label, transform in (
        ("正确 DS@L        ", good),
        ("次序反 L@DS      ", bug_order),
        ("中心误用 DS@C    ", bug_center),
    ):
        print(f"{label}: {transform}")
        print(f"  包络={envelope(transform)} 步向={steps(transform)}")
        print(f"  重采样编号=\n{resample(transform, parent)}")

    print("\n检查点世界坐标与偏移：")
    print(f"{'检查点':<10}{'正确':<22}{'次序反偏移':<18}{'中心误用偏移':<18}")
    for col, row in ((0, 0), (1, 0), (0, 1), (1, 1)):
        mark = (
            "K0 外角"
            if (col, row) == (0, 0)
            else "K1 内部"
            if (col, row) == (1, 1)
            else ""
        )
        print(
            f"({col},{row}) {mark:<6}"
            f"{good @ (col, row)!s:<22}"
            f"{shift_at(bug_order, good, col, row)!s:<18}"
            f"{shift_at(bug_center, good, col, row)!s:<18}"
        )

    print("\n错误判定：")
    print(f"  次序反 -> {classify(good, bug_order)}")
    print(f"  中心误 -> {classify(good, bug_center)}")

    print("\n对称样本盲区：")
    uniform = np.zeros_like(parent)
    print(
        "  均匀图幅下 N/W/E/S 输出编号全部相同：",
        all(
            np.array_equal(
                expected_dn(*ORIENTATIONS[a], uniform),
                expected_dn(*ORIENTATIONS[b], uniform),
            )
            for a in ORIENTATIONS
            for b in ORIENTATIONS
        ),
    )
    ds_zero = Affine(K * PX, 0.0, 0.0, 0.0, -K * PY, 0.0)
    mirror = Affine(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    print(
        "  零原点时 DS@M == M@DS：", (ds_zero @ mirror).almost_equals(mirror @ ds_zero)
    )


if __name__ == "__main__":
    _print_report()
