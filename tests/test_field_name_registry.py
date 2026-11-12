"""
Tests for the cross-partition FieldNameRegistry and the long-to-short dbf
field name mapping used when migrating long source field names into the
10 byte dbf field-name limit.
"""

import io
import json
import warnings

import pytest

import shapefile
from shapefile import Field, FieldMappingConflict, FieldMappingError, FieldNameRegistry


def test_short_names_are_truncated_at_code_point_boundaries():
    registry = FieldNameRegistry("utf-8")

    short = registry.short_name_for("人口普查总量数据字段甲")

    # 3 Chinese code points = 9 UTF-8 bytes; a fourth would be 12 bytes.
    assert short == "人口普"
    encoded = short.encode("utf-8")
    assert len(encoded) <= 10
    # Round-tripping proves the cut did not leave a dangling multi-byte char.
    assert encoded.decode("utf-8") == short


def test_colliding_long_names_get_distinct_stable_short_names():
    registry = FieldNameRegistry()

    first = registry.short_name_for("population_total_2025_census")
    second = registry.short_name_for("population_total_2025_estimate")

    assert first == "population"
    assert first != second
    assert len(second.encode("utf-8")) <= 10
    # Re-registering either name is deterministic and identity preserving.
    assert registry.short_name_for("population_total_2025_census") == first
    assert registry.short_name_for("population_total_2025_estimate") == second


def test_published_binding_keeps_its_identity_for_new_partitions():
    first_partition = FieldNameRegistry()
    published = first_partition.short_name_for("abcdefghijklmnop")
    assert published == "abcdefghij"

    # A later partition loads the already-published mapping and encounters a
    # different long name whose natural truncation is the same short name.
    later_partition = FieldNameRegistry.from_mapping(first_partition.to_mapping())
    new_short = later_partition.short_name_for("abcdefghijk_other")

    assert new_short != "abcdefghij"
    # The old column keeps its published identity.
    assert later_partition.short_name_for("abcdefghijklmnop") == "abcdefghij"


def test_conflict_for_names_the_minimum_incumbent_column():
    registry = FieldNameRegistry()
    registry.short_name_for("shared_prefix_xyz")

    assert registry.conflict_for("shared_prefix_abc") == "shared_prefix_xyz"
    assert registry.conflict_for("totally_different_name") is None
    # The long name's own binding is not reported as a conflict.
    assert registry.conflict_for("shared_prefix_xyz") is None


def test_from_mapping_rejects_two_long_names_on_one_short_name():
    with pytest.raises(FieldMappingConflict):
        FieldNameRegistry.from_mapping({"long_one": "same", "long_two": "same"})


def test_suffix_fallback_when_every_prefix_is_taken():
    registry = FieldNameRegistry()
    long_names = [f"sameprefix_field_{index}" for index in range(12)]

    short_names = [registry.short_name_for(name) for name in long_names]

    assert len(set(short_names)) == len(short_names)
    assert all(len(name.encode("utf-8")) <= 10 for name in short_names)
    assert short_names[0] == "sameprefix"


def test_registry_path_emits_no_truncation_warning_under_strict_writer():
    registry = FieldNameRegistry()
    dbf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("error", shapefile.PossibleDataLoss)
        writer = shapefile.Writer(dbf=dbf, strict=True, field_name_registry=registry)
        writer.field("population_total_2025_census", "N", size=10, decimal=2)
        writer.record(5)
        writer.close()


def test_bind_and_restore_preserve_field_type_size_and_decimal():
    registry = FieldNameRegistry()
    source_fields = [
        Field("population_total_2025_census", "N", 12, 2),
        Field("人口普查总量数据字段甲", "C", 20, 0),
        Field("measurement_date_field", "D", 8, 0),
    ]

    bound = registry.bind_fields(source_fields)
    assert [field.name for field in bound] != [field.name for field in source_fields]
    assert all(len(field.name.encode("utf-8")) <= 10 for field in bound)
    for source, bound_field in zip(source_fields, bound):
        assert bound_field.field_type == source.field_type
        assert bound_field.size == source.size
        assert bound_field.decimal == source.decimal

    restored = shapefile.restore_field_names(bound, registry.to_mapping())
    assert restored == source_fields


def test_restore_rejects_uncovered_and_inconsistent_mappings():
    fields = [Field("short", "C", 10, 0)]

    with pytest.raises(FieldMappingError):
        shapefile.restore_field_names(fields, {})

    with pytest.raises(FieldMappingError):
        shapefile.restore_field_names(
            fields, {"long_one": "short", "long_two": "short"}
        )


def test_writer_reader_round_trip_across_partitions(tmp_path):
    registry = FieldNameRegistry("utf-8")
    long_names = [
        "population_total_2025_census",
        "population_total_2025_estimate",
    ]

    for partition, long_name in enumerate(long_names):
        path = tmp_path / f"partition_{partition}.dbf"
        writer = shapefile.Writer(dbf=path, field_name_registry=registry)
        writer.field(long_name, "N", size=10, decimal=2)
        writer.record(partition + 1)
        writer.close()

    mapping_document = registry.to_document()
    assert mapping_document["encoding"] == "utf-8"

    for partition, long_name in enumerate(long_names):
        path = tmp_path / f"partition_{partition}.dbf"
        reader = shapefile.Reader(dbf=path)
        try:
            stored = reader.data_fields
            assert len(stored[0].name.encode("utf-8")) <= 10
            restored = reader.restore_fields(mapping_document["fields"])
            assert restored == [Field(long_name, "N", 10, 2)]
            assert reader.record(0)[0] == partition + 1
        finally:
            reader.close()

    # The document is self-contained and survives JSON serialisation.
    reloaded = FieldNameRegistry.from_document(
        json.loads(json.dumps(mapping_document, ensure_ascii=False))
    )
    assert reloaded.to_mapping() == registry.to_mapping()


def test_latin1_registry_counts_bytes_not_code_points():
    registry = FieldNameRegistry("latin1")
    name = "é" * 12  # each accented char is one latin1 byte but two utf8 bytes

    short = registry.short_name_for(name)

    assert short == "é" * 10
    assert len(short.encode("latin1")) == 10


def test_writer_rejects_registry_with_different_encoding():
    registry = FieldNameRegistry("utf-8")
    with pytest.raises(ValueError):
        shapefile.Writer(
            dbf=io.BytesIO(), encoding="latin1", field_name_registry=registry
        )


def test_mapping_tracks_on_disk_name_after_space_normalisation(tmp_path):
    # Outside strict mode PyShp replaces spaces with underscores; the
    # registry must map the long name to the normalised on-disk column.
    registry = FieldNameRegistry("utf-8")
    path = tmp_path / "spaces.dbf"
    with pytest.warns(shapefile.PossibleDataLoss):
        writer = shapefile.Writer(dbf=path, field_name_registry=registry)
        writer.field("long field name with spaces", "C", size=10)
        writer.record("v")
        writer.close()

    reader = shapefile.Reader(dbf=path)
    try:
        on_disk_name = reader.data_fields[0].name
        assert " " not in on_disk_name
        assert registry.to_mapping()["long field name with spaces"] == on_disk_name
        restored = reader.restore_fields(registry.to_mapping())
        assert restored[0].name == "long field name with spaces"
    finally:
        reader.close()


def test_normalised_name_colliding_with_published_column_raises(tmp_path):
    # "ab_cd_efgh..." publishes the short name "ab_cd_efgh".  A later name
    # whose truncation contains spaces at the same positions would normalise
    # to the same on-disk name, which must be rejected instead of breaking
    # the first column's identity.
    registry = FieldNameRegistry("utf-8")
    writer = shapefile.Writer(dbf=tmp_path / "first.dbf", field_name_registry=registry)
    writer.field("ab_cd_efghijklmnop", "C", size=10)
    writer.record("v")
    writer.close()

    colliding = shapefile.Writer(
        dbf=tmp_path / "second.dbf", field_name_registry=registry
    )
    colliding.field("some_other_valid_field", "C", size=10)
    with pytest.raises(FieldMappingConflict):
        with pytest.warns(shapefile.PossibleDataLoss):
            colliding.field("ab cd efghijklm", "C", size=10)
    colliding.record("ok")
    colliding.close()


def test_mapping_document_requires_fields_mapping():
    with pytest.raises(FieldMappingError):
        FieldNameRegistry.from_document({"version": 1, "encoding": "utf-8"})


def test_mapping_document_validates_encoding_and_value_types():
    with pytest.raises(FieldMappingError):
        FieldNameRegistry.from_document({"encoding": 123, "fields": {}})
    with pytest.raises(FieldMappingError):
        FieldNameRegistry.from_mapping({"long_name": 456})
