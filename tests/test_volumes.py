"""Tests for order-preserving, size-aware shapefile volume splitting."""

from __future__ import annotations

import warnings
from fractions import Fraction
from pathlib import Path

import pytest

import shapefile
import shapefile_volumes as volumes
from shapefile_volumes import (
    ComponentBytes,
    ComponentLimits,
    VolumeOversizeError,
    _balanced_boundaries,
    _collect_oversize,
    _component_headers,
    _greedy_boundaries,
    _prefix_sums,
    plan_volumes,
    split_shapefile,
)

LIMIT = ComponentLimits(10**9, 10**9, 10**9)
COMPONENT_EXTS = (".shp", ".shx", ".dbf")


def _points(n, z=False, m=False):
    points = []
    for i in range(n):
        if z and m:
            points.append((float(i), float(i), i * 0.1, float(i * 2)))
        elif m:
            points.append((float(i), float(i), float(i * 2)))
        else:
            points.append((float(i), float(i)))
    return points


@pytest.fixture
def polygon_source(tmp_path):
    """Polygon source with multiple parts and multibyte attribute text."""
    source = tmp_path / "source"
    writer = shapefile.Writer(
        target=source, shapeType=shapefile.POLYGON, encoding="utf-8"
    )
    writer.field("biz", "C", size=10)
    writer.field("note", "C", size=24)
    keys = ["A", "A", "B", "C", "C", "C", "D", "E", "E"]
    for index, key in enumerate(keys):
        ring = [[(float(j), float(j + index)) for j in range(3 + index)]]
        ring[0].append(ring[0][0])
        writer.poly(ring)
        writer.record(key, f"记录{index}")
    writer.close()
    return source, keys


def _emit_and_check(plan, reader, target):
    plan.emit_volume(reader, 0, target)
    assignment = plan.volumes[0]
    for ext in COMPONENT_EXTS:
        actual = Path(str(target) + ext).stat().st_size
        predicted = getattr(assignment.file_sizes, ext[1:])
        assert actual == predicted


def test_single_volume_predicted_lengths_match_files(polygon_source, tmp_path):
    source, _ = polygon_source
    with shapefile.Reader(source) as reader:
        plan = plan_volumes(reader, LIMIT)
        assert plan.num_volumes == 1
        _emit_and_check(plan, reader, tmp_path / "out")


def test_shape_content_bytes_matches_every_shape_type(tmp_path):
    builders = {
        shapefile.POINT: lambda w: w.point(1.0, 2.0),
        shapefile.POINTM: lambda w: w.pointm(1.0, 2.0, 9.0),
        shapefile.POINTZ: lambda w: w.pointz(1.0, 2.0, 3.0, 9.0),
        shapefile.MULTIPOINT: lambda w: w.multipoint(_points(5)),
        shapefile.MULTIPOINTM: lambda w: w.multipointm(_points(5, m=True)),
        shapefile.MULTIPOINTZ: lambda w: w.multipointz(_points(5, z=True, m=True)),
        shapefile.POLYLINE: lambda w: w.line([_points(4), _points(3)]),
        shapefile.POLYLINEM: lambda w: w.linem(
            [_points(4, m=True), _points(3, m=True)]
        ),
        shapefile.POLYLINEZ: lambda w: w.linez(
            [_points(4, z=True, m=True), _points(3, z=True, m=True)]
        ),
        shapefile.POLYGON: lambda w: w.poly([_points(4), _points(5)]),
        shapefile.POLYGONM: lambda w: w.polym([_points(4, m=True), _points(5, m=True)]),
        shapefile.POLYGONZ: lambda w: w.polyz(
            [_points(4, z=True, m=True), _points(5, z=True, m=True)]
        ),
        shapefile.MULTIPATCH: lambda w: w.multipatch(
            [_points(4, z=True, m=True), _points(3, z=True, m=True)],
            [shapefile.RING, shapefile.OUTER_RING],
        ),
    }
    for shape_type, builder in builders.items():
        source = tmp_path / f"type_{shape_type}"
        writer = shapefile.Writer(target=source, shapeType=shape_type)
        writer.field("id", "N", size=6)
        builder(writer)
        writer.record(1)
        writer.close()

        with shapefile.Reader(source) as reader:
            plan = plan_volumes(reader, LIMIT)
            target = tmp_path / f"out_{shape_type}"
            plan.emit_volume(reader, 0, target)
            assignment = plan.volumes[0]
            for ext in COMPONENT_EXTS:
                actual = Path(str(target) + ext).stat().st_size
                assert actual == getattr(assignment.file_sizes, ext[1:]), (
                    shape_type,
                    ext,
                )


def test_null_shape_uses_four_content_bytes(tmp_path):
    source = tmp_path / "nulls"
    writer = shapefile.Writer(target=source, shapeType=shapefile.POINT)
    writer.field("id", "N", size=4)
    writer.null()
    writer.record(1)
    writer.close()
    with shapefile.Reader(source) as reader:
        shape = reader.shape(0)
        assert volumes.shape_content_bytes(shape) == 4
        plan = plan_volumes(reader, LIMIT)
        plan.emit_volume(reader, 0, tmp_path / "null_out")
        assert (tmp_path / "null_out.shp").stat().st_size == (
            plan.volumes[0].file_sizes.shp
        )


def test_consecutive_business_rows_stay_together(polygon_source):
    source, keys = polygon_source
    with shapefile.Reader(source) as reader:
        plan = plan_volumes(
            reader,
            ComponentLimits(shp=700, shx=400, dbf=400),
            business_key=lambda row: reader.record(row)[0],
        )
        for group in plan.groups:
            owners = {plan.volume_of_row(row) for row in group.source_rows}
            assert len(owners) == 1
        emitted_rows = [
            row for assignment in plan.volumes for row in assignment.source_rows
        ]
        assert emitted_rows == list(range(len(keys)))


def test_limits_are_respected_and_local_indices_contiguous(polygon_source):
    source, _ = polygon_source
    limits = ComponentLimits(shp=700, shx=400, dbf=400)
    with shapefile.Reader(source) as reader:
        plan = plan_volumes(reader, limits)
        for assignment in plan.volumes:
            assert assignment.file_sizes.shp <= limits.shp
            assert assignment.file_sizes.shx <= limits.shx
            assert assignment.file_sizes.dbf <= limits.dbf
            assert sorted(assignment.row_to_local_index.values()) == list(
                range(len(assignment.source_rows))
            )


def test_min_volume_count_is_greedy_optimal(polygon_source):
    source, _ = polygon_source
    limits = ComponentLimits(shp=500, shx=300, dbf=300)
    with shapefile.Reader(source) as reader:
        plan = plan_volumes(reader, limits)
        groups = plan.groups
        prefixes = _prefix_sums(groups)
        headers = _component_headers(plan.field_sizes)
        capacity = ComponentBytes(
            limits.shp - headers.shp,
            limits.shx - headers.shx,
            limits.dbf - headers.dbf,
        )
        greedy = _greedy_boundaries(groups, prefixes, capacity)
        assert plan.num_volumes == len(greedy) - 1


def test_balanced_packing_matches_brute_force():
    # Small synthetic groups: enumerate every contiguous partition and
    # confirm the planner reaches the minimum count and the minimum max ratio.
    groups = [
        volumes.Group(0, "a", (0,), ComponentBytes(50, 8, 10)),
        volumes.Group(1, "b", (1,), ComponentBytes(70, 8, 10)),
        volumes.Group(2, "c", (2,), ComponentBytes(30, 8, 10)),
        volumes.Group(3, "d", (3,), ComponentBytes(90, 8, 10)),
    ]
    limits = ComponentLimits(220, 200, 200)
    headers = ComponentBytes(100, 100, 65)
    capacity = ComponentBytes(120, 100, 135)
    assert not _collect_oversize(groups, capacity)
    prefixes = _prefix_sums(groups)
    greedy = _greedy_boundaries(groups, prefixes, capacity)
    num_volumes = len(greedy) - 1
    boundaries = _balanced_boundaries(
        groups, num_volumes, prefixes, headers, limits, capacity
    )

    feasible_partitions = []

    def block_fits(start, end):
        return all(
            prefixes[k][end] - prefixes[k][start] <= capacity.as_tuple()[k]
            for k in range(3)
        )

    def block_ratio(start, end):
        ratio = Fraction(0)
        for k in range(3):
            size = headers.as_tuple()[k] + (prefixes[k][end] - prefixes[k][start])
            ratio = max(ratio, Fraction(size, limits.as_tuple()[k]))
        return ratio

    def recurse(start, bounds, max_ratio, count):
        if start == len(groups):
            feasible_partitions.append((count, max_ratio))
            return
        for end in range(start + 1, len(groups) + 1):
            if block_fits(start, end):
                recurse(
                    end,
                    bounds + [end],
                    max(max_ratio, block_ratio(start, end)),
                    count + 1,
                )

    recurse(0, [], Fraction(0), 0)
    min_count = min(count for count, _ in feasible_partitions)
    best_ratio = min(
        ratio for count, ratio in feasible_partitions if count == min_count
    )
    assert num_volumes == min_count
    achieved = max(
        block_ratio(start, end) for start, end in zip(boundaries, boundaries[1:])
    )
    assert achieved == best_ratio


def test_oversize_group_names_offending_component(polygon_source):
    source, _ = polygon_source
    with shapefile.Reader(source) as reader:
        with pytest.raises(VolumeOversizeError) as exc_info:
            plan_volumes(
                reader,
                ComponentLimits(shp=120, shx=400, dbf=400),
                business_key=lambda row: reader.record(row)[0],
            )
    offending = {
        group.business_key: group.offending_components
        for group in exc_info.value.oversize_groups
    }
    assert all(components == ("shp",) for components in offending.values())
    assert offending  # at least one group reported


def test_dbf_only_oversize_is_reported(polygon_source):
    source, _ = polygon_source
    with shapefile.Reader(source) as reader:
        with pytest.raises(VolumeOversizeError) as exc_info:
            plan_volumes(
                reader,
                ComponentLimits(shp=10**9, shx=10**9, dbf=100),
            )
    assert all(
        "dbf" in group.offending_components for group in exc_info.value.oversize_groups
    )


def test_limit_smaller_than_header_is_rejected(polygon_source):
    source, _ = polygon_source
    with shapefile.Reader(source) as reader:
        with pytest.raises(ValueError):
            plan_volumes(reader, ComponentLimits(shp=50, shx=400, dbf=400))


def test_multibyte_encoding_uses_byte_width(tmp_path):
    for encoding, text in (("utf-8", "汉字α"), ("gbk", "测试数据")):
        source = tmp_path / f"src_{encoding}"
        writer = shapefile.Writer(
            target=source, shapeType=shapefile.POINT, encoding=encoding
        )
        writer.field("name", "C", size=12)
        writer.point(1.0, 2.0)
        writer.record(text)
        writer.point(3.0, 4.0)
        writer.record("short")
        writer.close()

        with shapefile.Reader(source, encoding=encoding) as reader:
            plan = plan_volumes(reader, LIMIT)
            assert plan.encoding.replace("-", "_") == encoding.replace("-", "_")
            target = tmp_path / f"out_{encoding}"
            plan.emit_volume(reader, 0, target)
            assert (
                Path(str(target) + ".dbf").stat().st_size
                == plan.volumes[0].file_sizes.dbf
            )
            with shapefile.Reader(str(target), encoding=encoding) as reread:
                assert reread.record(0)[0] == text


def test_text_overflow_is_flagged_by_re_encoding(tmp_path):
    source = tmp_path / "latin1"
    writer = shapefile.Writer(
        target=source, shapeType=shapefile.POINT, encoding="latin-1"
    )
    writer.field("c", "C", size=1)
    writer.point(0.0, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        writer.record("é")
    writer.close()

    reader = shapefile.Reader(source, encoding="utf-8", encodingErrors="replace")
    plan = plan_volumes(reader, LIMIT)
    assert plan.text_overflow_fields == {0: ("c",)}
    reader.close()


def test_plan_then_emit_keeps_identical_mapping(polygon_source, tmp_path):
    source, _ = polygon_source
    limits = ComponentLimits(shp=700, shx=400, dbf=400)
    with shapefile.Reader(source) as reader:
        plan = plan_volumes(reader, limits)
        written = plan.emit_all(reader, tmp_path / "later")

        for assignment in plan.volumes:
            base = written[assignment.volume_index]
            for ext in COMPONENT_EXTS:
                actual = Path(str(base) + ext).stat().st_size
                assert actual == getattr(assignment.file_sizes, ext[1:])
            with shapefile.Reader(str(base)) as volume_reader:
                for source_row, local in assignment.row_to_local_index.items():
                    assert (
                        volume_reader.record(local)[0] == reader.record(source_row)[0]
                    )


def test_per_volume_emit_matches_one_shot_emit(polygon_source, tmp_path):
    source, _ = polygon_source
    limits = ComponentLimits(shp=700, shx=400, dbf=400)
    with shapefile.Reader(source) as reader:
        plan = plan_volumes(reader, limits)

        one_shot = plan.emit_all(reader, tmp_path / "one_shot")
        per_volume = []
        for volume_index in range(plan.num_volumes):
            target = tmp_path / f"single_{volume_index:02d}"
            plan.emit_volume(reader, volume_index, target)
            per_volume.append(target)

        assert len(one_shot) == len(per_volume)
        for batch_target, single_target in zip(one_shot, per_volume):
            for ext in COMPONENT_EXTS:
                assert (
                    Path(str(batch_target) + ext).stat().st_size
                    == Path(str(single_target) + ext).stat().st_size
                )
            with shapefile.Reader(str(batch_target)) as batch_reader:
                with shapefile.Reader(str(single_target)) as single_reader:
                    assert len(batch_reader) == len(single_reader)
                    for local_index in range(len(batch_reader)):
                        assert batch_reader.record(local_index) == (
                            single_reader.record(local_index)
                        )
                        assert (
                            batch_reader.shape(local_index).points
                            == single_reader.shape(local_index).points
                        )


def test_split_shapefile_is_consistent_with_plan(polygon_source, tmp_path):
    source, _ = polygon_source
    limits = ComponentLimits(shp=700, shx=400, dbf=400)
    with shapefile.Reader(source) as reader:
        plan, written = split_shapefile(reader, tmp_path / "batch", limits)
        assert len(written) == plan.num_volumes
        for assignment in plan.volumes:
            base = written[assignment.volume_index]
            for ext in COMPONENT_EXTS:
                assert Path(str(base) + ext).stat().st_size == getattr(
                    assignment.file_sizes, ext[1:]
                )


def test_empty_and_single_row_sources(tmp_path):
    empty = tmp_path / "empty"
    writer = shapefile.Writer(target=empty, shapeType=shapefile.POINT)
    writer.field("id", "N", size=5)
    writer.close()
    with shapefile.Reader(empty) as reader:
        plan = plan_volumes(reader, LIMIT)
        assert plan.num_volumes == 0
        assert plan.row_mapping == []

    single = tmp_path / "single"
    writer = shapefile.Writer(target=single, shapeType=shapefile.POINT)
    writer.field("id", "N", size=5)
    writer.point(1.0, 2.0)
    writer.record(1)
    writer.close()
    with shapefile.Reader(single) as reader:
        plan = plan_volumes(reader, LIMIT)
        assert plan.num_volumes == 1
        assert plan.volume_of_row(0) == 0
        assert plan.local_index_of_row(0) == 0


def test_planning_is_deterministic(polygon_source):
    source, _ = polygon_source
    limits = ComponentLimits(shp=700, shx=400, dbf=400)
    with shapefile.Reader(source) as reader:
        first = plan_volumes(reader, limits)
        second = plan_volumes(reader, limits)
    assert [assignment.source_rows for assignment in first.volumes] == [
        assignment.source_rows for assignment in second.volumes
    ]
    assert first.row_mapping == second.row_mapping


def test_predicted_lengths_summary_is_verifiable(polygon_source):
    source, _ = polygon_source
    limits = ComponentLimits(shp=700, shx=400, dbf=400)
    with shapefile.Reader(source) as reader:
        plan = plan_volumes(reader, limits)
        summary = plan.predicted_lengths()
        assert len(summary) == plan.num_volumes
        for entry, assignment in zip(summary, plan.volumes):
            assert entry["shp_bytes"] == assignment.file_sizes.shp
            assert entry["dbf_limit"] == limits.dbf
            assert isinstance(entry["max_occupancy_ratio"], Fraction)


def test_assignments_cover_every_source_row_once(polygon_source):
    source, keys = polygon_source
    with shapefile.Reader(source) as reader:
        plan = plan_volumes(
            reader,
            ComponentLimits(shp=700, shx=400, dbf=400),
        )
    covered = [row for assignment in plan.volumes for row in assignment.source_rows]
    assert sorted(covered) == list(range(len(keys)))
    assert len(covered) == len(keys)
