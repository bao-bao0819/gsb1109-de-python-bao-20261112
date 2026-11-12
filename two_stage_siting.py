#!/usr/bin/env python3
"""
两阶段最近仓站选择（球面粗筛 + WGS84 椭球 geodesic 精算）

判据概要
========
声明量
  * 椭球 (a, f)：旋转椭球，大地坐标 (phi, lambda) 下的线元为
        ds^2 = M(phi)^2 dphi^2 + N(phi)^2 cos^2 phi dlambda^2
    其中子午圈曲率半径 M = a(1-e^2)/(1-e^2 sin^2 phi)^(3/2)，
    卯酉圈曲率半径 N = a/(1-e^2 sin^2 phi)^(1/2)，e^2 = f(2-f)。
  * 适用纬度带 [phi_lo, phi_hi]：M、N 均随 |phi| 单调递增，故带内逐点成立
        Rmin = min_band M <= M(phi), N(phi) <= max_band N = Rmax。
  * 安全护栏：查询点与站点之间的“小圆劣弧”必须整体落在纬度带内，
    且球心角 sigma 不超过声明上限（默认 90 度）。不满足则强制精算。

球面量
  把大地经纬度当作单位球坐标，球心角 sigma 用 atan2/haversine 计算，
  单位球度量 dsigma^2 = dphi^2 + cos^2 phi dlambda^2。逐点夹逼：
        Rmin^2 dsigma^2 <= ds^2 <= Rmax^2 dsigma^2。
  * 下界：任意曲线 s >= Rmin * 球面像长 >= Rmin * sigma（大圆最短）。
  * 上界：把球面大圆弧抬回椭球，其长度 <= Rmax * sigma，测地线不更长。
  因此对每个站点有可证明包络
        L_i = Rmin * sigma_i <= s_i(geodesic) <= Rmax * sigma_i = U_i。

跳过规则（只有能证明不改变前 k 名时才跳过）
  1. 按 sigma 取球面排序的前 k 名，边界并列全部纳入候选集 C，护栏失败者纳入 C。
  2. 对 C 精算，令 t 为其中第 k 小的精算距离。
  3. 任一未精算站点 j，仅当 L_j > t + tol（tol 为浮点容差）时才可跳过；
     否则把 j 加入 C 重新精算并更新 t，循环到不再新增。
  4. 最终保留所有精算距离 <= t（含并列）的站点。
  被跳过者满足 s_j >= L_j > t >= 任一入选者距离，故不可能改变前 k 名；
  等号情形一律精算、一律保留。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from geopy.distance import ELLIPSOIDS, geodesic

EPS_REL = 1e-12  # 浮点并列容差（相对）
EPS_ABS = 1e-9  # 浮点并列容差（米）


@dataclass(frozen=True)
class Ellipsoid:
    name: str
    a: float  # 长半轴，米
    f: float  # 扁率

    @classmethod
    def from_geopy(cls, name: str) -> "Ellipsoid":
        # geopy 的 ELLIPSOIDS 以公里存储，这里统一换算成米
        a_km, _b, f = ELLIPSOIDS[name]
        return cls(name, a_km * 1000.0, f)

    @property
    def e2(self) -> float:
        return self.f * (2.0 - self.f)

    def meridian_radius(self, lat_deg: float) -> float:
        s = math.sin(math.radians(lat_deg))
        return self.a * (1.0 - self.e2) / (1.0 - self.e2 * s * s) ** 1.5

    def prime_vertical_radius(self, lat_deg: float) -> float:
        s = math.sin(math.radians(lat_deg))
        return self.a / math.sqrt(1.0 - self.e2 * s * s)


class LatitudeBand:
    """声明的适用纬度带及其可证明包络常数。"""

    def __init__(self, ellipsoid: Ellipsoid, lat_lo: float, lat_hi: float,
                 sigma_max_deg: float = 90.0):
        if not (-90.0 <= lat_lo < lat_hi <= 90.0):
            raise ValueError("纬度带需满足 -90 <= lat_lo < lat_hi <= 90")
        if not (0.0 < sigma_max_deg <= 180.0):
            raise ValueError("sigma_max 需在 (0, 180] 度内")
        self.ellipsoid = ellipsoid
        self.lat_lo = lat_lo
        self.lat_hi = lat_hi
        self.sigma_max = math.radians(sigma_max_deg)
        # M、N 为 |phi| 的偶函数且随 |phi| 单调；跨赤道时最小值在 |phi|=0
        min_abs_lat = 0.0 if lat_lo <= 0.0 <= lat_hi else min(
            abs(lat_lo), abs(lat_hi))
        max_abs_lat = max(abs(lat_lo), abs(lat_hi))
        self.r_min = ellipsoid.meridian_radius(min_abs_lat)
        self.r_max = ellipsoid.prime_vertical_radius(max_abs_lat)

    def arc_in_band(self, p: tuple[float, float], q: tuple[float, float],
                    sigma: float) -> bool:
        """判断 p->q 的球面小圆劣弧是否整体落在纬度带内且未超角距上限。"""
        if sigma > self.sigma_max:
            return False
        if sigma < 1e-14:
            return self.lat_lo <= p[0] <= self.lat_hi
        u = _unit_vector(*p)
        v = _unit_vector(*q)
        lats = [p[0], q[0]]
        # 大圆极点 n = u x v；纬度极值点 t ∝ z - (z·n)n（两个对跖顶点）
        nx, ny, nz = _cross(u, v)
        norm = math.hypot(math.hypot(nx, ny), nz)
        nx, ny, nz = nx / norm, ny / norm, nz / norm
        tv = (-nz * nx, -nz * ny, 1.0 - nz * nz)
        tv_norm = math.hypot(math.hypot(tv[0], tv[1]), tv[2])
        cos_sigma = math.cos(sigma)
        for sign in (1.0, -1.0):
            t = (sign * tv[0] / tv_norm, sign * tv[1] / tv_norm,
                 sign * tv[2] / tv_norm)
            # 顶点位于劣弧上的充要条件：与两端点的夹角都不超过 sigma
            if (_dot(t, u) >= cos_sigma - 1e-12
                    and _dot(t, v) >= cos_sigma - 1e-12):
                lats.append(math.degrees(math.asin(max(-1.0, min(1.0, t[2])))))
        return self.lat_lo <= min(lats) and max(lats) <= self.lat_hi

    def envelope(self, sigma: float) -> tuple[float, float]:
        return self.r_min * sigma, self.r_max * sigma


def _unit_vector(lat: float, lon: float) -> tuple[float, float, float]:
    phi = math.radians(lat)
    lam = math.radians(lon)
    return (math.cos(phi) * math.cos(lam),
            math.cos(phi) * math.sin(lam),
            math.sin(phi))


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def central_angle(p: tuple[float, float], q: tuple[float, float]) -> float:
    """单位球心角 sigma，atan2 形式，近距与对跖点均稳定。"""
    u = _unit_vector(*p)
    v = _unit_vector(*q)
    x = _dot(u, v)
    cx, cy, cz = _cross(u, v)
    y = math.hypot(math.hypot(cx, cy), cz)
    return math.atan2(y, x)


def _tie_tol(distance: float) -> float:
    return max(EPS_ABS, EPS_REL * abs(distance))


def top_k_with_ties(items: list, k: int, key) -> list:
    ordered = sorted(items, key=key)
    cutoff = key(ordered[k - 1])
    return [item for item in ordered if key(item) <= cutoff + _tie_tol(cutoff)]


@dataclass
class StationRow:
    sid: str
    point: tuple[float, float]
    sigma: float
    lower: float
    upper: float
    guarded: bool
    exact: float | None = None


def select_two_stage(query, stations, k: int, band: LatitudeBand):
    """返回 (入选站点行, 全部行, 精算次数)。stations 为 (sid, (lat, lon)) 列表。"""
    if not 1 <= k <= len(stations):
        raise ValueError("k 需在 [1, 站点数] 内")
    rows: list[StationRow] = []
    forced = set()
    for sid, point in stations:
        sigma = central_angle(query, point)
        lower, upper = band.envelope(sigma)
        guarded = band.arc_in_band(query, point, sigma)
        if not guarded:
            forced.add(sid)
        rows.append(StationRow(sid, point, sigma, lower, upper, guarded))

    spherical_top = top_k_with_ties(rows, k, key=lambda r: r.sigma)
    candidate_ids = {r.sid for r in spherical_top} | forced
    exact_calls = 0

    def evaluate(sid: str):
        nonlocal exact_calls
        row = next(r for r in rows if r.sid == sid)
        if row.exact is None:
            row.exact = geodesic(query, row.point,
                                 ellipsoid=band.ellipsoid.name).meters
            exact_calls += 1

    for sid in set(candidate_ids):
        evaluate(sid)

    while True:
        evaluated = [r for r in rows if r.exact is not None]
        ordered = sorted(evaluated, key=lambda r: r.exact)
        threshold = ordered[k - 1].exact
        added = False
        for row in rows:
            if row.exact is None and row.lower <= threshold + _tie_tol(threshold):
                evaluate(row.sid)
                added = True
        if not added:
            break

    evaluated = [r for r in rows if r.exact is not None]
    threshold = sorted(evaluated, key=lambda r: r.exact)[k - 1].exact
    selected = [r for r in evaluated
                if r.exact <= threshold + _tie_tol(threshold)]
    selected.sort(key=lambda r: r.exact)

    # 不变量：每个被跳过的站点都有严格大于门槛的可证明下界
    for row in rows:
        if row.exact is None:
            assert row.lower > threshold + _tie_tol(threshold), row.sid
    return selected, rows, exact_calls, threshold


def full_exact_selection(query, stations, k: int, ellipsoid_name: str):
    """全精算参照：所有站点都算 geodesic，前 k 名边界并列全部保留。"""
    rows = []
    for sid, point in stations:
        rows.append((sid, geodesic(query, point, ellipsoid=ellipsoid_name).meters))
    cutoff = sorted(d for _, d in rows)[k - 1]
    return {(sid, d) for sid, d in rows
            if d <= cutoff + _tie_tol(cutoff)}


def reversal_demo():
    print("=" * 78)
    print("实例：球面排名与 WGS84 geodesic 排名相反")
    print("-" * 78)
    query = (45.0, 0.0)
    station_a = (45.2003, 0.0)  # 子午向，dlat = 0.2003 度
    dlon = 0.2 / math.cos(math.radians(45.0))
    station_b = (45.0, dlon)  # 卯酉向，球面角距恰好 0.2 度
    for label, point in (("A 子午向", station_a), ("B 卯酉向", station_b)):
        sg = math.degrees(central_angle(query, point))
        from geopy.distance import great_circle
        print(f"{label}: 球面角距={sg:.6f} 度  great_circle="
              f"{great_circle(query, point).meters:11.3f} m  "
              f"geodesic={geodesic(query, point).meters:11.3f} m")
    print("球面：B 更近（0.2000 < 0.2003 度）；geodesic：A 更近"
          "（22260.09 < 22301.24 m），排名反转。")


def main():
    ellipsoid = Ellipsoid.from_geopy("WGS-84")
    band = LatitudeBand(ellipsoid, lat_lo=20.0, lat_hi=60.0,
                        sigma_max_deg=90.0)
    print("声明：椭球 %s，a=%.1f m，1/f=%.6f" %
          (ellipsoid.name, ellipsoid.a, 1.0 / ellipsoid.f))
    print("适用纬度带 [20, 60] 度；Rmin=M(20)=%.3f m，Rmax=N(60)=%.3f m，"
          "带宽 %.4f%%" %
          (band.r_min, band.r_max,
           100.0 * (band.r_max - band.r_min) / band.r_min))

    reversal_demo()

    print("=" * 78)
    print("本地样本：1 个查询点 + 12 个站点，k=3，边界并列全部保留")
    print("-" * 78)
    query = (45.0, 0.0)
    stations = [
        ("S01", (45.2003, 0.0)),               # 反转实例 A
        ("S02", (45.0, 0.2 / math.cos(math.radians(45.0)))),  # 反转实例 B
        ("S03", (45.05, 0.05)),
        ("S04", (44.90, -0.10)),
        ("S05", (45.30, 0.20)),
        ("S06", (44.70, 0.25)),
        ("S07", (45.10, -0.30)),
        ("S08", (44.50, -0.40)),
        ("S09", (45.50, 0.50)),
        ("S10", (44.20, 0.60)),
        # E1/E2 同坐标，且精算距离恰好落在第 3/4 名边界：并列必须全部保留
        ("E1", (45.0, 0.22)),
        ("E2", (45.0, 0.22)),
    ]
    k = 3
    selected, rows, exact_calls, threshold = select_two_stage(
        query, stations, k, band)
    reference = full_exact_selection(query, stations, k, ellipsoid.name)

    print(f"{'站点':>4} {'sigma(度)':>10} {'下界 L(m)':>13} "
          f"{'上界 U(m)':>13} {'精算 s(m)':>13}  状态")
    for r in sorted(rows, key=lambda r: r.sigma):
        status = "入选" if r in selected else (
            "已精算未入选" if r.exact is not None else "跳过(有证明)")
        exact_text = f"{r.exact:13.3f}" if r.exact is not None else f"{'-':>13}"
        guard = "" if r.guarded else " [护栏外强制精算]"
        print(f"{r.sid:>4} {math.degrees(r.sigma):10.6f} {r.lower:13.3f} "
              f"{r.upper:13.3f} {exact_text}  {status}{guard}")
    print("-" * 78)
    print(f"两阶段精算次数: {exact_calls} / {len(stations)}；"
          f"第 {k} 名门槛 t={threshold:.3f} m（含并列）")
    got = {(r.sid, r.exact) for r in selected}
    print("两阶段入选:", sorted(sid for sid, _ in got))
    print("全精算入选:", sorted(sid for sid, _ in reference))
    assert got == reference, "两阶段结果与全精算不一致"
    print("核对通过：两阶段入选集合与全精算完全一致，且所有跳过站点均满足 L > t。")

    print("=" * 78)
    print("随机压力核对：300 组随机样本，逐组与全精算比对入选集合")
    print("-" * 78)
    rng = random.Random(20261113)
    mismatches = 0
    call_counts = []
    station_counts = []
    guard_tripped = 0
    for trial in range(300):
        n = rng.randint(5, 20)
        q = (rng.uniform(25.0, 55.0), rng.uniform(-5.0, 5.0))
        sts = []
        for i in range(n):
            lat = rng.uniform(20.0, 60.0)
            lon = rng.uniform(-120.0, 120.0)  # 大跨度会触发角距/纬度带护栏
            sts.append((f"T{i:02d}", (lat, lon)))
        kk = rng.randint(1, min(4, n))
        sel, all_rows, calls, _ = select_two_stage(q, sts, kk, band)
        ref = full_exact_selection(q, sts, kk, ellipsoid.name)
        # 直接验证可证明包络：所有护栏内站点必须满足 L <= 精算值 <= U
        for r in all_rows:
            if r.guarded:
                exact = geodesic(q, r.point, ellipsoid=ellipsoid.name).meters
                assert r.lower - 1e-6 <= exact <= r.upper + 1e-6
        if {(r.sid, r.exact) for r in sel} != ref:
            mismatches += 1
        if any(not r.guarded for r in all_rows):
            guard_tripped += 1
        call_counts.append(calls)
        station_counts.append(n)
    total_calls = sum(call_counts)
    total_sites = sum(station_counts)
    print(f"不一致组数: {mismatches} / 300；护栏触发组数: {guard_tripped}")
    print(f"每组精算次数 min/mean/max = {min(call_counts)}/"
          f"{sum(call_counts) / len(call_counts):.2f}/{max(call_counts)}"
          f"（站点数 5-20）")
    print(f"合计精算 {total_calls} / {total_sites} 次，"
          f"节省 {100.0 * (1 - total_calls / total_sites):.1f}%")

    print("=" * 78)
    print("护栏自检：同纬度 25 度、经度 ±80 度的两点，大圆弧向赤道凹陷")
    p, q = (25.0, -80.0), (25.0, 80.0)
    sg = central_angle(p, q)
    print(f"球心角={math.degrees(sg):.2f} 度，弧在带内: "
          f"{band.arc_in_band(p, q, sg)}（应为 False，此类对强制精算）")

    print("=" * 78)
    print("跨赤道纬度带 [-10, 10] 自检：Rmin 必须取赤道 M(0)，逐对验证包络")
    eq_band = LatitudeBand(ellipsoid, lat_lo=-10.0, lat_hi=10.0)
    assert eq_band.r_min == ellipsoid.meridian_radius(0.0)
    rng2 = random.Random(99)
    eq_violations = 0
    for _ in range(50000):
        pp = (rng2.uniform(-10, 10), rng2.uniform(-5, 5))
        qq = (rng2.uniform(-10, 10), rng2.uniform(-5, 5))
        sg2 = central_angle(pp, qq)
        if not eq_band.arc_in_band(pp, qq, sg2):
            continue
        exact = geodesic(pp, qq, ellipsoid=ellipsoid.name).meters
        lo2, hi2 = eq_band.envelope(sg2)
        if not (lo2 - 1e-6 <= exact <= hi2 + 1e-6):
            eq_violations += 1
    print(f"5 万对护栏内样本包络违反数: {eq_violations}")


if __name__ == "__main__":
    main()
