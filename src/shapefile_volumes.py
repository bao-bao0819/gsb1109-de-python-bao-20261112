"""Order-preserving volume splitting for ESRI Shapefiles.

A batch of source rows is split into contiguous *volumes* while honouring
three independent byte limits, one per constituent file (``.shp``,
``.shx`` and ``.dbf``).  Rows sharing a business key are grouped first:
consecutive rows with the same key form an indivisible group that always
lands in a single volume.

The splitting rules are, in priority order:

1. Use the minimum possible number of volumes.
2. For that number of volumes, minimise the occupancy ratio of the fullest
   component across every volume.
3. Keep source order, and make the plan the single source of truth so that
   emitting volumes later reproduces the exact same row-to-volume mapping.

All predicted lengths are exact byte counts derived from the same binary
layout :mod:`shapefile` writes, so they can be verified against the files
produced by :meth:`VolumePlan.emit_volume`.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import cast

import shapefile
from shapefile import (
    MULTIPATCH,
    NULL,
    Point_shapeTypes,
    PointM_shapeTypes,
    PointZ_shapeTypes,
    Reader,
    Shape,
    Writer,
    _CanHaveParts_shapeTypes,
    _HasM_shapeTypes,
    _HasZ_shapeTypes,
)

# Fixed layout constants from the ESRI shapefile specification, matching
# the constants used by shapefile.Writer / ShpWriter.
SHP_HEADER_BYTES = 100
SHX_HEADER_BYTES = 100
SHP_RECORD_HEADER_BYTES = 8  # big-endian record number + content length
SHAPE_TYPE_INT_BYTES = 4
SHX_RECORD_BYTES = 8
BBOX_BYTES = 4 * 8
Z_OR_M_BOX_BYTES = 2 * 8
INT32_BYTES = 4
DOUBLE_BYTES = 8
DBF_HEADER_TERMINATOR_BYTES = 1
DBF_FIELD_DESCRIPTOR_BYTES = 32
DBF_DELETION_FLAG_BYTES = 1

BusinessKey = Hashable
KeyFn = Callable[[int], BusinessKey]


def _shape_at(reader: Reader, row_index: int) -> Shape:
    """Reads one shape, rejecting a spatial-filter ``None`` result.

    Volume planning never passes a bbox filter, so a present row always
    yields a concrete shape (including null shapes).
    """
    shape = reader.shape(row_index)
    if shape is None:
        raise shapefile.ShapefileException(
            f"Source row {row_index} unexpectedly returned no geometry."
        )
    return shape


def _record_at(reader: Reader, row_index: int) -> shapefile._Record:
    """Reads one dbf record, rejecting a deleted-record ``None`` result."""
    record = reader.record(row_index)
    if record is None:
        raise shapefile.ShapefileException(
            f"Source row {row_index} is marked as deleted in the .dbf and "
            "cannot be copied into a volume."
        )
    return record


@dataclass(frozen=True)
class ComponentLimits:
    """Maximum on-disk size in bytes for each constituent file."""

    shp: int
    shx: int
    dbf: int

    def as_tuple(self) -> tuple[int, int, int]:
        return self.shp, self.shx, self.dbf


@dataclass(frozen=True)
class ComponentBytes:
    """Exact byte occupancy of one unit (a row, a group or a volume)."""

    shp: int
    shx: int
    dbf: int

    def as_tuple(self) -> tuple[int, int, int]:
        return self.shp, self.shx, self.dbf

    def __add__(self, other: ComponentBytes) -> ComponentBytes:
        return ComponentBytes(
            self.shp + other.shp,
            self.shx + other.shx,
            self.dbf + other.dbf,
        )


@dataclass(frozen=True)
class OversizeGroup:
    """A group whose body cannot fit inside a single volume."""

    group_index: int
    business_key: BusinessKey
    source_rows: tuple[int, ...]
    required: ComponentBytes
    body_capacity: ComponentBytes
    offending_components: tuple[str, ...]


class VolumeOversizeError(shapefile.ShapefileException):
    """Raised when one or more indivisible groups exceed the given limits."""

    def __init__(self, oversize_groups: Sequence[OversizeGroup]):
        self.oversize_groups = list(oversize_groups)
        lines = []
        for item in oversize_groups:
            lines.append(
                "group "
                f"{item.group_index} (key={item.business_key!r}, "
                f"source rows {list(item.source_rows)}) needs "
                f"shp={item.required.shp}B, shx={item.required.shx}B, "
                f"dbf={item.required.dbf}B but a single volume provides "
                f"shp={item.body_capacity.shp}B, "
                f"shx={item.body_capacity.shx}B, "
                f"dbf={item.body_capacity.dbf}B; "
                f"oversized component(s): {', '.join(item.offending_components)}"
            )
        super().__init__("; ".join(lines))


@dataclass(frozen=True)
class Group:
    """An indivisible run of consecutive source rows."""

    index: int
    business_key: BusinessKey
    source_rows: tuple[int, ...]
    body: ComponentBytes


@dataclass(frozen=True)
class VolumeAssignment:
    """Which groups/rows one volume contains and its predicted file sizes."""

    volume_index: int
    group_indices: tuple[int, ...]
    source_rows: tuple[int, ...]
    row_to_local_index: dict[int, int]
    body: ComponentBytes
    file_sizes: ComponentBytes
    occupancy_ratios: dict[str, Fraction]


@dataclass
class VolumePlan:
    """The full, reproducible splitting plan.

    The plan stores explicit boundaries, so emitting volumes afterwards
    only replays them and cannot produce a different mapping.
    """

    limits: ComponentLimits
    encoding: str
    encoding_errors: str
    shape_type: int
    field_sizes: tuple[int, ...]
    groups: list[Group]
    volumes: list[VolumeAssignment]
    row_mapping: list[tuple[int, int]]
    text_overflow_fields: dict[int, tuple[str, ...]] = field(default_factory=dict)

    @property
    def num_volumes(self) -> int:
        return len(self.volumes)

    def predicted_lengths(self) -> list[dict[str, object]]:
        """Verifiable per-volume summary of predicted file lengths.

        Every entry pairs the exact byte prediction for ``.shp`` / ``.shx``
        / ``.dbf`` with its limit and occupancy ratio, so it can be checked
        directly against the emitted files.
        """
        summary = []
        for assignment in self.volumes:
            summary.append(
                {
                    "volume_index": assignment.volume_index,
                    "source_rows": assignment.source_rows,
                    "shp_bytes": assignment.file_sizes.shp,
                    "shx_bytes": assignment.file_sizes.shx,
                    "dbf_bytes": assignment.file_sizes.dbf,
                    "shp_limit": self.limits.shp,
                    "shx_limit": self.limits.shx,
                    "dbf_limit": self.limits.dbf,
                    "max_occupancy_ratio": max(assignment.occupancy_ratios.values()),
                }
            )
        return summary

    def volume_of_row(self, source_row: int) -> int:
        return self.row_mapping[source_row][0]

    def local_index_of_row(self, source_row: int) -> int:
        return self.row_mapping[source_row][1]

    def emit_volume(
        self,
        reader: Reader,
        volume_index: int,
        target: str | Path,
    ) -> Path:
        """Writes one planned volume and returns the target path.

        ``target`` is the file path *without* extension; ``.shp``, ``.shx``
        and ``.dbf`` are written alongside it.
        """
        assignment = self.volumes[volume_index]
        target_path = Path(target)
        writer = Writer(
            target=target_path,
            shapeType=self.shape_type,
            encoding=self.encoding,
            encodingErrors=self.encoding_errors,
        )
        try:
            for data_field in reader.data_fields:
                writer.dbf_writer.field(
                    data_field.name,
                    data_field.field_type,
                    data_field.size,
                    data_field.decimal,
                )
            for source_row in assignment.source_rows:
                writer.shape(_shape_at(reader, source_row))
                writer.record(*_record_at(reader, source_row))
        finally:
            writer.close()
        return target_path

    def emit_all(
        self,
        reader: Reader,
        target_prefix: str | Path,
    ) -> list[Path]:
        """Writes every planned volume as ``<prefix>_<index>`` (zero padded)."""
        prefix = Path(target_prefix)
        width = max(2, len(str(self.num_volumes - 1)))
        written = []
        for volume_index in range(self.num_volumes):
            target = prefix.with_name(f"{prefix.name}_{volume_index:0{width}d}")
            self.emit_volume(reader, volume_index, target)
            written.append(target)
        return written


def shape_content_bytes(shape: Shape) -> int:
    """Content bytes of one shape, excluding its 8-byte record header.

    Mirrors :meth:`shapefile.Shape.write_to_byte_stream` and
    :meth:`shapefile.ShpWriter._shp_record` exactly, including the leading
    4-byte shape type integer and every part / coordinate / Z / M block.
    """
    shape_type = shape.shapeType
    n_points = len(shape.points)
    total = SHAPE_TYPE_INT_BYTES

    if shape_type == NULL:
        return total

    if shape_type in Point_shapeTypes:
        total += 2 * DOUBLE_BYTES  # x, y
        if shape_type in PointZ_shapeTypes:
            total += DOUBLE_BYTES  # single Z
        if shape_type in PointM_shapeTypes:
            total += DOUBLE_BYTES  # single M
        return total

    # Every remaining shape type carries its own bounding box.
    total += BBOX_BYTES

    if shape_type in _CanHaveParts_shapeTypes:
        n_parts = len(shape.parts)
        total += INT32_BYTES  # number of parts
        total += n_parts * INT32_BYTES  # part start indices

    total += INT32_BYTES  # number of points

    if shape_type == MULTIPATCH:
        total += len(shape.partTypes) * INT32_BYTES

    total += n_points * 2 * DOUBLE_BYTES  # x/y coordinates

    if shape_type in _HasZ_shapeTypes:
        total += Z_OR_M_BOX_BYTES + n_points * DOUBLE_BYTES
    if shape_type in _HasM_shapeTypes:
        total += Z_OR_M_BOX_BYTES + n_points * DOUBLE_BYTES

    return total


def shp_record_bytes(shape: Shape) -> int:
    """Full on-disk bytes of one ``.shp`` record (header plus content)."""
    return SHP_RECORD_HEADER_BYTES + shape_content_bytes(shape)


def dbf_record_length(field_sizes: Sequence[int]) -> int:
    """Fixed width of one ``.dbf`` record, including the deletion flag."""
    return DBF_DELETION_FLAG_BYTES + sum(field_sizes)


def dbf_header_length(num_fields: int) -> int:
    """Bytes occupied by the ``.dbf`` header and field descriptors."""
    return DBF_FIELD_DESCRIPTOR_BYTES * num_fields + 32 + DBF_HEADER_TERMINATOR_BYTES


def _component_headers(field_sizes: Sequence[int]) -> ComponentBytes:
    return ComponentBytes(
        shp=SHP_HEADER_BYTES,
        shx=SHX_HEADER_BYTES,
        dbf=dbf_header_length(len(field_sizes)),
    )


def _row_body_bytes(shape: Shape) -> ComponentBytes:
    return ComponentBytes(
        shp=shp_record_bytes(shape),
        shx=SHX_RECORD_BYTES,
        dbf=0,  # record width is constant; added by the caller once.
    )


def _build_groups(
    reader: Reader,
    field_sizes: tuple[int, ...],
    key_fn: KeyFn,
) -> list[Group]:
    record_width = dbf_record_length(field_sizes)
    groups: list[Group] = []
    current_rows: list[int] = []
    current_key: BusinessKey | None = None
    current_body = ComponentBytes(0, 0, 0)

    def flush() -> None:
        if not current_rows:
            return
        groups.append(
            Group(
                index=len(groups),
                business_key=current_key,
                source_rows=tuple(current_rows),
                body=ComponentBytes(
                    shp=current_body.shp,
                    shx=current_body.shx,
                    dbf=len(current_rows) * record_width,
                ),
            )
        )

    for row_index in range(len(reader)):
        key = key_fn(row_index)
        row_body = _row_body_bytes(_shape_at(reader, row_index))
        if current_rows and key != current_key:
            flush()
            current_rows = []
            current_body = ComponentBytes(0, 0, 0)
        current_key = key
        current_rows.append(row_index)
        current_body = ComponentBytes(
            shp=current_body.shp + row_body.shp,
            shx=current_body.shx + row_body.shx,
            dbf=0,
        )
    flush()
    return groups


def _text_overflow_fields(
    reader: Reader,
) -> dict[int, tuple[str, ...]]:
    """Encodes C/M text with the declared encoding to flag field overflow.

    On-disk records stay fixed width (pyshp pads / truncates), but encoding
    is performed explicitly so multibyte text is measured in bytes rather
    than Python code points.
    """
    encoding = reader.encoding
    errors = reader.encodingErrors
    text_fields = [
        (index, data_field.name, data_field.size)
        for index, data_field in enumerate(reader.data_fields)
        if data_field.field_type in ("C", "M")
    ]
    overflows: dict[int, tuple[str, ...]] = {}
    for row_index in range(len(reader)):
        record = _record_at(reader, row_index)
        offending = []
        for field_index, field_name, size in text_fields:
            value = record[field_index]
            if value is None:
                continue
            encoded = str(value).encode(encoding, errors)
            if len(encoded) > size:
                offending.append(field_name)
        if offending:
            overflows[row_index] = tuple(offending)
    return overflows


def _prefix_sums(groups: Sequence[Group]) -> tuple[list[int], list[int], list[int]]:
    shp_prefix = [0]
    shx_prefix = [0]
    dbf_prefix = [0]
    for group in groups:
        shp_prefix.append(shp_prefix[-1] + group.body.shp)
        shx_prefix.append(shx_prefix[-1] + group.body.shx)
        dbf_prefix.append(dbf_prefix[-1] + group.body.dbf)
    return shp_prefix, shx_prefix, dbf_prefix


def _furthest_feasible_end(
    start: int,
    num_groups: int,
    prefixes: tuple[list[int], list[int], list[int]],
    body_capacity: ComponentBytes,
) -> int:
    """Largest exclusive end index whose block from ``start`` still fits."""
    end = num_groups
    for prefix, capacity in zip(prefixes, body_capacity.as_tuple()):
        limit = prefix[start] + capacity
        end = min(end, bisect_right(prefix, limit) - 1)
    return max(start, end)


def _greedy_boundaries(
    groups: Sequence[Group],
    prefixes: tuple[list[int], list[int], list[int]],
    body_capacity: ComponentBytes,
) -> list[int]:
    """Longest-prefix next-fit; optimal for ordered contiguous splitting."""
    boundaries = [0]
    position = 0
    total = len(groups)
    while position < total:
        end = _furthest_feasible_end(position, total, prefixes, body_capacity)
        if end == position:
            raise RuntimeError(
                "An indivisible group exceeds the volume limits; "
                "oversize groups must be detected before packing."
            )
        boundaries.append(end)
        position = end
    return boundaries


def _collect_oversize(
    groups: Sequence[Group],
    body_capacity: ComponentBytes,
) -> list[OversizeGroup]:
    oversize = []
    for group in groups:
        offending = []
        if group.body.shp > body_capacity.shp:
            offending.append("shp")
        if group.body.shx > body_capacity.shx:
            offending.append("shx")
        if group.body.dbf > body_capacity.dbf:
            offending.append("dbf")
        if offending:
            oversize.append(
                OversizeGroup(
                    group_index=group.index,
                    business_key=group.business_key,
                    source_rows=group.source_rows,
                    required=group.body,
                    body_capacity=body_capacity,
                    offending_components=tuple(offending),
                )
            )
    return oversize


def _block_ratio_pair(
    start: int,
    end: int,
    prefixes: tuple[list[int], list[int], list[int]],
    headers: ComponentBytes,
    limits: ComponentLimits,
) -> tuple[int, int]:
    """Max component ratio of one block as an unreduced ``(num, denom)``.

    Cross multiplication is used for comparison instead of constructing
    :class:`~fractions.Fraction`, avoiding repeated gcd work in the DP while
    keeping every comparison exact.
    """
    numerator = 0
    denominator = 1
    for prefix, header, limit in zip(prefixes, headers.as_tuple(), limits.as_tuple()):
        total_bytes = header + (prefix[end] - prefix[start])
        if total_bytes * denominator > numerator * limit:
            numerator, denominator = total_bytes, limit
    return numerator, denominator


def _max_ratio_pair(
    first: tuple[int, int],
    second: tuple[int, int],
) -> tuple[int, int]:
    """Exact max of two unreduced ratio pairs."""
    if first[0] * second[1] >= second[0] * first[1]:
        return first
    return second


def _balanced_boundaries(
    groups: Sequence[Group],
    num_volumes: int,
    prefixes: tuple[list[int], list[int], list[int]],
    headers: ComponentBytes,
    limits: ComponentLimits,
    body_capacity: ComponentBytes,
) -> list[int]:
    """DP minimising the fullest component ratio for the fixed volume count."""
    total = len(groups)

    # dp[j] = best (max ratio pair, parent) after the current number of
    # volumes.  Ratios stay unreduced (numerator, denominator) pairs and are
    # compared by exact cross multiplication.
    zero_ratio = (0, 1)
    dp: dict[int, tuple[tuple[int, int], int]] = {0: (zero_ratio, -1)}
    parents: list[dict[int, int]] = []
    for _ in range(num_volumes):
        next_dp: dict[int, tuple[tuple[int, int], int]] = {}
        step_parents: dict[int, int] = {}
        for start, (prior_ratio, _) in dp.items():
            max_end = _furthest_feasible_end(start, total, prefixes, body_capacity)
            for end in range(start + 1, max_end + 1):
                block_ratio = _block_ratio_pair(start, end, prefixes, headers, limits)
                candidate = _max_ratio_pair(prior_ratio, block_ratio)
                existing = next_dp.get(end)
                if existing is None or (
                    candidate[0] * existing[0][1] < existing[0][0] * candidate[1]
                ):
                    next_dp[end] = (candidate, start)
                    step_parents[end] = start
        dp = next_dp
        parents.append(step_parents)

    if total not in dp:
        raise RuntimeError(
            "Balanced packing failed despite feasible groups; "
            "the greedy volume count should always be achievable."
        )

    boundaries = [total]
    position = total
    for step in range(num_volumes - 1, -1, -1):
        position = parents[step][position]
        boundaries.append(position)
    boundaries.reverse()
    return boundaries


def _build_assignments(
    groups: Sequence[Group],
    boundaries: Sequence[int],
    headers: ComponentBytes,
    limits: ComponentLimits,
) -> tuple[list[VolumeAssignment], list[tuple[int, int]]]:
    assignments = []
    total_rows = sum(len(group.source_rows) for group in groups)
    row_mapping: list[tuple[int, int] | None] = [None] * total_rows
    for volume_index in range(len(boundaries) - 1):
        first_group = boundaries[volume_index]
        last_group = boundaries[volume_index + 1]
        group_indices = tuple(range(first_group, last_group))
        source_rows = tuple(
            row
            for group_index in group_indices
            for row in groups[group_index].source_rows
        )
        body = ComponentBytes(0, 0, 0)
        for group_index in group_indices:
            body = body + groups[group_index].body
        file_sizes = ComponentBytes(
            shp=headers.shp + body.shp,
            shx=headers.shx + body.shx,
            dbf=headers.dbf + body.dbf,
        )
        local_index_map = {
            row: local_index for local_index, row in enumerate(source_rows)
        }
        for local_index, row in enumerate(source_rows):
            row_mapping[row] = (volume_index, local_index)
        assignments.append(
            VolumeAssignment(
                volume_index=volume_index,
                group_indices=group_indices,
                source_rows=source_rows,
                row_to_local_index=local_index_map,
                body=body,
                file_sizes=file_sizes,
                occupancy_ratios={
                    "shp": Fraction(file_sizes.shp, limits.shp),
                    "shx": Fraction(file_sizes.shx, limits.shx),
                    "dbf": Fraction(file_sizes.dbf, limits.dbf),
                },
            )
        )
    return assignments, cast(list[tuple[int, int]], row_mapping)


def plan_volumes(
    reader: Reader,
    limits: ComponentLimits,
    business_key: KeyFn | None = None,
) -> VolumePlan:
    """Creates a deterministic, order-preserving splitting plan.

    ``business_key`` maps a source row index to a hashable key.  Consecutive
    rows returning the same key form an indivisible group.  When omitted,
    every source row is its own group.

    Raises :class:`VolumeOversizeError` if any single group is too large for
    one volume; the exception names each offending component explicitly.
    """
    if min(limits.as_tuple()) <= 0:
        raise ValueError("All component limits must be positive byte counts.")

    field_sizes = tuple(field.size for field in reader.data_fields)
    headers = _component_headers(field_sizes)
    body_capacity = ComponentBytes(
        shp=limits.shp - headers.shp,
        shx=limits.shx - headers.shx,
        dbf=limits.dbf - headers.dbf,
    )
    if min(body_capacity.as_tuple()) <= 0:
        raise ValueError(
            "Each component limit must exceed its fixed header size: "
            f"headers require shp={headers.shp}B, shx={headers.shx}B, "
            f"dbf={headers.dbf}B."
        )

    key_fn: KeyFn = business_key if business_key is not None else lambda row: row
    groups = _build_groups(reader, field_sizes, key_fn)

    oversize = _collect_oversize(groups, body_capacity)
    if oversize:
        raise VolumeOversizeError(oversize)

    prefixes = _prefix_sums(groups)
    greedy_boundaries = _greedy_boundaries(groups, prefixes, body_capacity)
    num_volumes = len(greedy_boundaries) - 1

    if num_volumes <= 1:
        boundaries = greedy_boundaries
    else:
        boundaries = _balanced_boundaries(
            groups,
            num_volumes,
            prefixes,
            headers,
            limits,
            body_capacity,
        )

    assignments, row_mapping = _build_assignments(groups, boundaries, headers, limits)
    text_overflows = _text_overflow_fields(reader)

    return VolumePlan(
        limits=limits,
        encoding=reader.encoding,
        encoding_errors=reader.encodingErrors,
        shape_type=reader.shapeType,
        field_sizes=field_sizes,
        groups=groups,
        volumes=assignments,
        row_mapping=row_mapping,
        text_overflow_fields=text_overflows,
    )


def split_shapefile(
    source: str | Path | Reader,
    target_prefix: str | Path,
    limits: ComponentLimits,
    business_key: KeyFn | None = None,
) -> tuple[VolumePlan, list[Path]]:
    """Plans and immediately emits every volume.

    The same :class:`VolumePlan` drives both steps, so the emitted files
    always match the predicted lengths and row mapping exactly.
    """
    close_reader = False
    if isinstance(source, Reader):
        reader = source
    else:
        reader = Reader(source)
        close_reader = True
    try:
        plan = plan_volumes(reader, limits, business_key)
        written = plan.emit_all(reader, target_prefix)
        return plan, written
    finally:
        if close_reader:
            reader.close()
