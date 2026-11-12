"""离线地籍两版本对账判据（无稳定主键场景）。

限定前提
--------
1. 两个版本的坐标必须**逐位精确相等**（同一二进制 double 值）；本模块不做
   容差比较。若数据源仅文本格式不同，应在上游把坐标规范化为同一 double。
2. 环无自交，外环/内环结构合法。该前提下，同一环的点集只可能因"换起点"或
   "整体反向"而呈现不同序列，规范化后必然得到同一 key。
3. 属性仅作候选证据：同名不产生匹配边，只用于事后标注与人工复核分流。

核心判据（全部离线、确定性、可复算）
------------------------------------
* 环 key：闭合环去尾点后，在全部"循环移位 × 正/反向"中取字典序最小序列。
  key 相等 ⇔ 两环是同一几何环（仅起点/方向差异），与属性无关。
* 面指纹：一个 polygon = 多个 part，每个 part = 外环 + 若干内环；
  面指纹是其全部环 key 的多重集（Counter）。指纹相等 ⇔ 几何完全未改，
  part 重排、环起点变化、环反向都被吸收。
* 记录匹配图：旧记录 O_i 与新记录 N_j 之间，仅当指纹相等时连硬边。
  在该二分图上枚举**全部最大匹配**：
  - 出现在每个最大匹配中的边 = 必然续接（几何强制对应）；
  - 其余可达边 = 无法确定的对应，必须连同属性证据交人工判定；
  - 每个最大匹配中都 unmatched 的旧记录 = 灭失，新记录 = 新生。
* 指纹不等的记录之间不存在硬边；若两记录仍共享若干环 key，只输出
  "几何改动候选对"（次级证据，不是续接结论），并附环级增删清单。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Sequence

Point = tuple[float, float]
Ring = Sequence[Sequence[float]]
RingKey = tuple[Point, ...]


@dataclass(frozen=True)
class CanonicalRing:
    """环的规范化结果。key 是与起点、方向无关的环身份。"""

    key: RingKey
    length: int


def _ring_core(ring: Ring) -> list[Point]:
    points = [(float(x), float(y)) for x, y in (p[:2] for p in ring)]
    if len(points) > 1 and points[0] == points[-1]:
        return points[:-1]
    return points


def normalize_ring(ring: Ring) -> CanonicalRing:
    """把闭合（或首尾同点）坐标环规范化为与起点/方向无关的形式。

    >>> a = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    >>> b = [(1, 1), (0, 1), (0, 0), (1, 0), (1, 1)]  # 换起点且反向
    >>> normalize_ring(a).key == normalize_ring(b).key
    True
    """
    core = _ring_core(ring)
    n = len(core)
    if n == 0:
        return CanonicalRing((), 0)
    backward = list(reversed(core))
    key = min(
        min(tuple(core[i:] + core[:i]) for i in range(n)),
        min(tuple(backward[i:] + backward[:i]) for i in range(n)),
    )
    return CanonicalRing(key, n)


@dataclass(frozen=True)
class RingEncodingDelta:
    """同一环 key 的两种原始编码之间的差异（精确坐标前提）。

    reversed=False: 仅循环移位，shift 为新起点在旧序列中的下标；
    reversed=True : 先整体反向再循环移位。
    两环 key 不等时无法定义，调用 ring_encoding_delta 会抛 ValueError。
    """

    reversed: bool
    shift: int


def ring_encoding_delta(old_ring: Ring, new_ring: Ring) -> RingEncodingDelta:
    """判定同 key 两环是"换起点"还是"反向+换起点"，并给出位移。"""
    old_core = _ring_core(old_ring)
    new_core = _ring_core(new_ring)
    n = len(old_core)
    if len(new_core) != n:
        raise ValueError("rings have different vertex counts, keys cannot match")
    for shift in range(n):
        if new_core == old_core[shift:] + old_core[:shift]:
            return RingEncodingDelta(False, shift)
    reversed_core = list(reversed(old_core))
    for shift in range(n):
        if new_core == reversed_core[shift:] + reversed_core[:shift]:
            return RingEncodingDelta(True, shift)
    raise ValueError("ring keys differ; encoding delta is undefined")


@dataclass(frozen=True)
class Part:
    """单个 part：外环 key + 内环 key 集合。"""

    exterior: RingKey
    interiors: frozenset[RingKey]


@dataclass(frozen=True)
class PolygonRecord:
    """一条 polygon 记录的规范化几何。"""

    record_id: Hashable
    parts: tuple[Part, ...]
    fingerprint: frozenset[tuple[RingKey, int]]
    attrs: dict = field(default_factory=dict, compare=False)

    @classmethod
    def build(
        cls,
        record_id: Hashable,
        parts: Iterable[Iterable[Ring]],
        attrs: dict | None = None,
    ) -> "PolygonRecord":
        """parts: [[外环, 内环1, 内环2, ...], 第二个 part, ...]。"""
        norm_parts: list[Part] = []
        counter: Counter[RingKey] = Counter()
        for part in parts:
            rings = [normalize_ring(r) for r in part]
            if not rings:
                continue
            exterior = rings[0].key
            interiors = frozenset(r.key for r in rings[1:])
            norm_parts.append(Part(exterior, interiors))
            counter[exterior] += 1
            for key in interiors:
                counter[key] += 1
        return cls(
            record_id,
            tuple(norm_parts),
            frozenset(counter.items()),
            attrs or {},
        )


@dataclass(frozen=True)
class PresentationDiff:
    """几何相同前提下，表达层面的差异标注。"""

    start_shifted_rings: frozenset[RingKey]
    reversed_rings: frozenset[RingKey]
    parts_reordered: bool


def presentation_diff(
    old_raw: Sequence[Sequence[Ring]],
    new_raw: Sequence[Sequence[Ring]],
) -> PresentationDiff:
    """在指纹相等的两份原始几何间，区分起点变化 / 反向 / part 重排。

    入参是未经规范化的原始 part 环列表；调用前提是两者面指纹相等。
    """
    def collect(raw_parts):
        ring_by_key: dict[RingKey, Ring] = {}
        sigs: list[frozenset[tuple[RingKey, int]]] = []
        for part in raw_parts:
            sigs.append(
                frozenset(Counter(normalize_ring(r).key for r in part).items())
            )
            for ring in part:
                ring_by_key.setdefault(normalize_ring(ring).key, ring)
        return ring_by_key, sigs

    old_rings, old_sigs = collect(old_raw)
    new_rings, new_sigs = collect(new_raw)

    shifted: set[RingKey] = set()
    reversed_rings: set[RingKey] = set()
    for key, old_ring in old_rings.items():
        if key not in new_rings:
            continue
        delta = ring_encoding_delta(old_ring, new_rings[key])
        if delta.shift != 0:
            shifted.add(key)
        if delta.reversed:
            reversed_rings.add(key)
    parts_reordered = (
        Counter(old_sigs) == Counter(new_sigs) and old_sigs != new_sigs
    )
    return PresentationDiff(
        frozenset(shifted), frozenset(reversed_rings), parts_reordered
    )


@dataclass(frozen=True)
class GeometryChange:
    """真实几何改动的环级描述；改动环即不同 key，不做近似配对。"""

    removed_ring_keys: frozenset[RingKey]
    added_ring_keys: frozenset[RingKey]
    removed_counts: dict[RingKey, int]
    added_counts: dict[RingKey, int]
    shared_ring_keys: frozenset[RingKey]


def diff_geometry(
    old_record: PolygonRecord, new_record: PolygonRecord
) -> GeometryChange:
    """指纹不等时给出环级增删清单与共享环（次级证据）。"""
    old_counter = Counter(dict(old_record.fingerprint))
    new_counter = Counter(dict(new_record.fingerprint))
    removed = old_counter - new_counter
    added = new_counter - old_counter
    shared = frozenset(old_counter.keys() & new_counter.keys())
    return GeometryChange(
        frozenset(removed),
        frozenset(added),
        dict(removed),
        dict(added),
        shared,
    )


def _enum_max_matchings(
    left: Sequence[Hashable],
    adjacency: dict[Hashable, frozenset[Hashable]],
    cap: int,
) -> tuple[tuple[frozenset[tuple[Hashable, Hashable]], ...], bool]:
    """枚举二分图全部最大基数匹配；匹配数超 cap 时截断。"""
    results: list[frozenset[tuple]] = []
    best_size = 0
    truncated = False

    def visit(
        idx: int,
        used_right: set[Hashable],
        chosen: set[tuple[Hashable, Hashable]],
    ) -> None:
        nonlocal best_size, truncated
        if len(results) >= cap:
            truncated = True
            return
        if idx == len(left):
            size = len(chosen)
            if size > best_size:
                best_size = size
                results.clear()
                results.append(frozenset(chosen))
            elif size == best_size:
                results.append(frozenset(chosen))
            return
        options = adjacency.get(left[idx], frozenset()) - used_right
        if not options:
            visit(idx + 1, used_right, chosen)
            return
        for right in sorted(options, key=repr):
            used_right.add(right)
            chosen.add((left[idx], right))
            visit(idx + 1, used_right, chosen)
            chosen.discard((left[idx], right))
            used_right.discard(right)
        visit(idx + 1, used_right, chosen)  # 左节点弃权，参与最大基数比较

    visit(0, set(), set())
    max_len = max((len(m) for m in results), default=0)
    return tuple(m for m in results if len(m) == max_len), truncated


@dataclass(frozen=True)
class MatchingAnalysis:
    certain_pairs: frozenset[tuple[Hashable, Hashable]]
    undecidable_edges: frozenset[tuple[Hashable, Hashable]]
    always_unmatched_old: frozenset[Hashable]
    always_unmatched_new: frozenset[Hashable]
    all_max_matchings: tuple[frozenset[tuple[Hashable, Hashable]], ...]
    truncated: bool


def analyze_matchings(
    old_records: Sequence[PolygonRecord],
    new_records: Sequence[PolygonRecord],
    cap: int = 10_000,
) -> MatchingAnalysis:
    """以"面指纹相等"为唯一硬边，枚举最大匹配并划分必然/歧义对应。"""
    new_by_fp: dict[frozenset, list[PolygonRecord]] = {}
    for rec in new_records:
        new_by_fp.setdefault(rec.fingerprint, []).append(rec)

    adjacency: dict[Hashable, frozenset[Hashable]] = {}
    for old in old_records:
        adjacency[old.record_id] = frozenset(
            n.record_id for n in new_by_fp.get(old.fingerprint, ())
        )

    matchings, truncated = _enum_max_matchings(
        [r.record_id for r in old_records], adjacency, cap
    )
    if not matchings:
        matchings = (frozenset(),)

    edge_intersection = set(matchings[0])
    edge_union = set(matchings[0])
    matched_old: set[Hashable] = set()
    matched_new: set[Hashable] = set()
    for matching in matchings:
        edge_intersection &= matching
        edge_union |= matching
        for left, right in matching:
            matched_old.add(left)
            matched_new.add(right)

    old_ids = {r.record_id for r in old_records}
    new_ids = {r.record_id for r in new_records}
    return MatchingAnalysis(
        certain_pairs=frozenset(edge_intersection),
        undecidable_edges=frozenset(edge_union - edge_intersection),
        always_unmatched_old=frozenset(old_ids - matched_old),
        always_unmatched_new=frozenset(new_ids - matched_new),
        all_max_matchings=matchings,
        truncated=truncated,
    )


def attribute_candidates(
    old_records: Sequence[PolygonRecord],
    new_records: Sequence[PolygonRecord],
    attr_name: str,
) -> dict[Hashable, frozenset[Hashable]]:
    """同名属性候选对。返回值**不得**用于构造匹配边，只供人工复核排序。"""
    buckets: dict[object, list[Hashable]] = {}
    for rec in new_records:
        buckets.setdefault(rec.attrs.get(attr_name), []).append(rec.record_id)
    return {
        rec.record_id: frozenset(buckets.get(rec.attrs.get(attr_name), ()))
        for rec in old_records
    }


@dataclass(frozen=True)
class ReconcileVerdict:
    old_id: Hashable | None
    new_id: Hashable | None
    status: str
    detail: object = None
    attr_note: str = ""


def reconcile(
    old_records: Sequence[PolygonRecord],
    new_records: Sequence[PolygonRecord],
    raw_geometry: dict[Hashable, Sequence[Sequence[Ring]]] | None = None,
    attr_name: str = "name",
    cap: int = 10_000,
) -> list[ReconcileVerdict]:
    """完整对账：几何硬判据优先，表达差异/真实改动/歧义分层标注。

    status 取值：
      UNCHANGED                指纹相等且在所有最大匹配中强制配对
      AMBIGUOUS                指纹相等但存在多个最大匹配，边不唯一
      GEOMETRY_CHANGED_CANDIDATE 指纹不等但共享环 key（次级证据，需人工确认）
      DISAPPEARED / BIRTH      无任何硬边且无共享环的未匹配记录
    """
    analysis = analyze_matchings(old_records, new_records, cap)
    old_by_id = {r.record_id: r for r in old_records}
    new_by_id = {r.record_id: r for r in new_records}
    same_name = attribute_candidates(old_records, new_records, attr_name)
    verdicts: list[ReconcileVerdict] = []

    for oid, nid in sorted(analysis.certain_pairs, key=repr):
        presentation = None
        if raw_geometry is not None:
            presentation = presentation_diff(
                raw_geometry[oid], raw_geometry[nid]
            )
        note = (
            "attr_same_name"
            if nid in same_name.get(oid, frozenset())
            else "attr_diff_or_missing"
        )
        verdicts.append(
            ReconcileVerdict(oid, nid, "UNCHANGED", presentation, note)
        )

    for oid, nid in sorted(analysis.undecidable_edges, key=repr):
        note = (
            "attr_same_name_but_not_decisive"
            if nid in same_name.get(oid, frozenset())
            else "attr_evidence_only_must_manual_review"
        )
        verdicts.append(
            ReconcileVerdict(oid, nid, "AMBIGUOUS", None, note)
        )

    leftover_old = [
        old_by_id[oid] for oid in sorted(analysis.always_unmatched_old, key=repr)
    ]
    leftover_new = [
        new_by_id[nid] for nid in sorted(analysis.always_unmatched_new, key=repr)
    ]
    linked_old: set[Hashable] = set()
    linked_new: set[Hashable] = set()
    for old_rec in leftover_old:
        old_keys = set(dict(old_rec.fingerprint))
        for new_rec in leftover_new:
            new_keys = set(dict(new_rec.fingerprint))
            if old_keys & new_keys:
                linked_old.add(old_rec.record_id)
                linked_new.add(new_rec.record_id)
                verdicts.append(
                    ReconcileVerdict(
                        old_rec.record_id,
                        new_rec.record_id,
                        "GEOMETRY_CHANGED_CANDIDATE",
                        diff_geometry(old_rec, new_rec),
                        "ring_overlap_is_secondary_evidence",
                    )
                )
    for old_rec in leftover_old:
        if old_rec.record_id not in linked_old:
            verdicts.append(
                ReconcileVerdict(old_rec.record_id, None, "DISAPPEARED")
            )
    for new_rec in leftover_new:
        if new_rec.record_id not in linked_new:
            verdicts.append(
                ReconcileVerdict(None, new_rec.record_id, "BIRTH")
            )
    return verdicts
