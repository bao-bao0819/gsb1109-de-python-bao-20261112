import json
import unittest

from geopy.extra.address_migration import (
    AddressMigrator,
    FieldMapping,
    LevelSpec,
    REASON_AMBIGUOUS_SPLIT,
    REASON_MISSING_FIELDS,
    SourceSpec,
    TargetSpec,
)
from geopy.location import Location


COUNTRY = LevelSpec("country", unit="country")
CITY = LevelSpec("city", unit="municipality")
DISTRICT = LevelSpec("district", unit="borough")


def build_source(**kwargs):
    defaults = dict(
        mappings=(
            FieldMapping("raw.data.id", "source_id", required=True),
            FieldMapping("raw.data.country", "country", required=True),
            FieldMapping("raw.data.city", "city", required=True),
            FieldMapping("raw.data.district", "district"),
            FieldMapping("address", "display"),
        ),
        levels=(COUNTRY, CITY, DISTRICT),
        address_field="address",
        raw_field="raw",
        latitude_field="latitude",
        longitude_field="longitude",
        packed_levels=(CITY, DISTRICT),
        display_separator=", ",
    )
    defaults.update(kwargs)
    return SourceSpec(**defaults)


def build_target(**kwargs):
    defaults = dict(
        mappings=(
            FieldMapping("source_id", "place_id", required=True),
            FieldMapping("country", "country_name"),
            FieldMapping("city", "city_name"),
            FieldMapping("district", "district_name"),
            FieldMapping("display", "formatted_address"),
        ),
        levels=(COUNTRY, CITY, DISTRICT),
        display_separator=", ",
    )
    defaults.update(kwargs)
    return TargetSpec(**defaults)


def make_row(
    row_id="p1",
    country="Atlantis",
    city="Springfield",
    district="Northgate",
    address="Springfield, Northgate",
    latitude=40.0,
    longitude=-70.0,
):
    return {
        "id": row_id,
        "address": address,
        "latitude": latitude,
        "longitude": longitude,
        "raw": {
            "data": {
                "id": row_id,
                "country": country,
                "city": city,
                "district": district,
            }
        },
    }


def build_migrator(**source_kwargs):
    return AddressMigrator(
        build_source(**source_kwargs), build_target()
    )


class AddressMigrationTestCase(unittest.TestCase):

    def test_migrate_row_with_unique_split(self):
        migrator = build_migrator(identity_field="id")
        report = migrator.migrate([make_row()])

        self.assertEqual(report.skipped_count, 0)
        self.assertEqual(report.migrated_count, 1)
        record = report.migrated[0]

        self.assertEqual(record.source_identity, "p1")
        self.assertEqual(record.address, "Atlantis, Springfield, Northgate")
        self.assertEqual(record.latitude, 40.0)
        self.assertEqual(record.longitude, -70.0)
        self.assertEqual(
            record.raw,
            {
                "place_id": "p1",
                "country_name": "Atlantis",
                "city_name": "Springfield",
                "district_name": "Northgate",
                "formatted_address": "Springfield, Northgate",
            },
        )
        self.assertEqual(
            record.admin_hierarchy,
            [
                {"level": "country", "value": "Atlantis",
                 "unit": "country"},
                {"level": "city", "value": "Springfield",
                 "unit": "municipality"},
                {"level": "district", "value": "Northgate",
                 "unit": "borough"},
            ],
        )

        location = record.to_location()
        self.assertIsInstance(location, Location)
        self.assertEqual(location.address, record.address)
        self.assertEqual(location.raw, record.raw)
        self.assertEqual(location.latitude, 40.0)
        self.assertEqual(location.longitude, -70.0)

    def test_split_uses_raw_evidence_even_when_order_differs(self):
        row = make_row(address="Northgate, Springfield")
        report = build_migrator(identity_field="id").migrate([row])

        self.assertEqual(report.migrated_count, 1)
        record = report.migrated[0]
        self.assertEqual(record.raw["city_name"], "Springfield")
        self.assertEqual(record.raw["district_name"], "Northgate")

    def test_ambiguous_split_is_not_migrated(self):
        row = make_row(
            city="Springfield",
            district="Springfield",
            address="Springfield, Springfield",
        )
        report = build_migrator(identity_field="id").migrate([row])

        self.assertEqual(report.migrated_count, 0)
        skipped = report.skipped[0]
        self.assertEqual(skipped.reason, REASON_AMBIGUOUS_SPLIT)
        self.assertEqual(
            sorted(skipped.missing_fields), ["city", "district"]
        )

    def test_token_count_mismatch_is_ambiguous(self):
        row = make_row(address="Springfield, Northgate, Extra")
        report = build_migrator(identity_field="id").migrate([row])

        self.assertEqual(report.migrated_count, 0)
        skipped = report.skipped[0]
        self.assertEqual(skipped.reason, REASON_AMBIGUOUS_SPLIT)
        self.assertEqual(
            sorted(skipped.missing_fields), ["city", "district"]
        )

    def test_missing_raw_evidence_listed_as_missing_field(self):
        row = make_row()
        del row["raw"]["data"]["district"]
        report = build_migrator(identity_field="id").migrate([row])

        self.assertEqual(report.migrated_count, 0)
        skipped = report.skipped[0]
        self.assertEqual(skipped.reason, REASON_MISSING_FIELDS)
        self.assertIn("district", skipped.missing_fields)

    def test_required_source_field_missing(self):
        row = make_row()
        del row["raw"]["data"]["id"]
        report = build_migrator(identity_field="id").migrate([row])

        self.assertEqual(report.migrated_count, 0)
        skipped = report.skipped[0]
        self.assertEqual(skipped.reason, REASON_MISSING_FIELDS)
        self.assertEqual(skipped.missing_fields, ("source_id",))

    def test_target_required_field_missing(self):
        source = build_source(identity_field="id", packed_levels=())
        target = build_target(
            mappings=(
                FieldMapping("source_id", "place_id", required=True),
                FieldMapping("country", "country_name", required=True),
                FieldMapping("city", "city_name"),
                FieldMapping("district", "district_name"),
                FieldMapping("district", "mandatory_new",
                             required=True),
                FieldMapping("display", "formatted_address"),
            )
        )
        row = make_row()
        del row["raw"]["data"]["district"]
        migrator = AddressMigrator(source, target)
        report = migrator.migrate([row])

        self.assertEqual(report.migrated_count, 0)
        skipped = report.skipped[0]
        self.assertEqual(skipped.reason, REASON_MISSING_FIELDS)
        self.assertIn("mandatory_new", skipped.missing_fields)

    def test_default_value_covers_optional_missing_field(self):
        source = build_source(
            identity_field="id",
            packed_levels=(),
            mappings=(
                FieldMapping("raw.data.id", "source_id", required=True),
                FieldMapping("raw.data.country", "country",
                             required=True),
                FieldMapping("raw.data.city", "city", required=True),
                FieldMapping("raw.data.district", "district",
                             default="Unknown District"),
                FieldMapping("address", "display"),
            ),
        )
        row = make_row()
        del row["raw"]["data"]["district"]
        report = AddressMigrator(source, build_target()).migrate([row])

        self.assertEqual(report.migrated_count, 1)
        self.assertEqual(
            report.migrated[0].raw["district_name"], "Unknown District"
        )

    def test_same_display_name_different_sites_are_not_merged(self):
        row_a = make_row(
            row_id="a",
            latitude=40.0,
            longitude=-70.0,
            address="Springfield, Northgate",
        )
        row_b = make_row(
            row_id="b",
            latitude=41.0,
            longitude=-71.0,
            address="Springfield, Northgate",
        )
        report = build_migrator(identity_field="id").migrate(
            [row_a, row_b]
        )

        self.assertEqual(report.migrated_count, 2)
        self.assertEqual(report.duplicate_count, 0)
        identities = {record.source_identity for record in
                      report.migrated}
        self.assertEqual(identities, {"a", "b"})

    def test_fingerprint_identity_distinguishes_different_raw(self):
        row_a = make_row(row_id="a", latitude=40.0, longitude=-70.0)
        row_b = make_row(row_id="b", latitude=41.0, longitude=-71.0)
        migrator = build_migrator()
        report = migrator.migrate([row_a, row_b])

        self.assertEqual(report.migrated_count, 2)
        self.assertNotEqual(
            report.migrated[0].source_identity,
            report.migrated[1].source_identity,
        )

    def test_repeated_migration_keeps_first_source_identity(self):
        migrator = build_migrator(identity_field="id")
        first = make_row()
        report = migrator.migrate([first])
        self.assertEqual(report.migrated_count, 1)

        changed = make_row(latitude=99.0, longitude=99.0)
        changed["raw"]["data"]["city"] = "New Springfield"
        report = migrator.migrate([changed])

        self.assertEqual(report.migrated_count, 0)
        self.assertEqual(report.duplicate_count, 1)
        self.assertEqual(report.duplicates, ["p1"])
        record = migrator.migrated[0]
        self.assertEqual(record.raw["city_name"], "Springfield")
        self.assertEqual(record.latitude, 40.0)
        self.assertEqual(record.longitude, -70.0)

    def test_identical_rows_are_duplicates_without_identity_field(self):
        migrator = build_migrator()
        migrator.migrate([make_row()])
        report = migrator.migrate([make_row()])

        self.assertEqual(report.migrated_count, 0)
        self.assertEqual(report.duplicate_count, 1)

    def test_state_roundtrip_preserves_first_run(self):
        migrator = build_migrator(identity_field="id")
        migrator.migrate([make_row()])
        state = json.loads(json.dumps(migrator.state_dict()))

        restored = build_migrator(identity_field="id")
        restored.load_state(state)
        changed = make_row(latitude=99.0, longitude=99.0)
        report = restored.migrate([changed])

        self.assertEqual(report.duplicate_count, 1)
        record = restored.migrated[0]
        self.assertEqual(record.latitude, 40.0)
        self.assertEqual(
            record.to_dict(), migrator.migrated[0].to_dict()
        )

    def test_location_instances_are_accepted(self):
        location = Location(
            "Springfield, Northgate",
            (40.0, -70.0),
            {
                "data": {
                    "id": "loc1",
                    "country": "Atlantis",
                    "city": "Springfield",
                    "district": "Northgate",
                }
            },
        )
        report = build_migrator(
            identity_field="raw.data.id"
        ).migrate([location])

        self.assertEqual(report.migrated_count, 1)
        record = report.migrated[0]
        self.assertEqual(record.source_identity, "loc1")
        self.assertEqual(record.longitude, -70.0)

    def test_missing_coordinates_are_reported_not_raised(self):
        row = make_row()
        row["latitude"] = None
        report = build_migrator(identity_field="id").migrate([row])

        self.assertEqual(report.migrated_count, 0)
        skipped = report.skipped[0]
        self.assertEqual(skipped.reason, REASON_MISSING_FIELDS)
        self.assertEqual(
            sorted(skipped.missing_fields),
            ["latitude", "longitude"],
        )

    def test_first_seen_order_is_preserved(self):
        rows = [make_row(row_id="z"), make_row(row_id="a"),
                make_row(row_id="m")]
        migrator = build_migrator(identity_field="id")
        migrator.migrate(rows)

        self.assertEqual(
            [record.source_identity for record in migrator.migrated],
            ["z", "a", "m"],
        )

    def test_point_field_coordinates(self):
        source = build_source(
            identity_field="id",
            point_field="point",
            latitude_field=None,
            longitude_field=None,
        )
        row = make_row()
        row["point"] = (row.pop("latitude"), row.pop("longitude"))
        report = AddressMigrator(source, build_target()).migrate([row])

        self.assertEqual(report.migrated_count, 1)
        record = report.migrated[0]
        self.assertEqual(record.latitude, 40.0)
        self.assertEqual(record.longitude, -70.0)


class SpecValidationTestCase(unittest.TestCase):

    def test_unknown_target_mapping_source_rejected(self):
        with self.assertRaises(ValueError):
            AddressMigrator(
                build_source(),
                build_target(mappings=(
                    FieldMapping("unknown", "x", required=True),
                )),
            )

    def test_unknown_target_level_rejected(self):
        with self.assertRaises(ValueError):
            AddressMigrator(
                build_source(),
                build_target(levels=(LevelSpec("planet"),)),
            )

    def test_coordinate_declaration_required(self):
        with self.assertRaises(ValueError):
            build_source(
                latitude_field=None,
                longitude_field=None,
                point_field=None,
            )

    def test_duplicate_mapping_targets_rejected(self):
        with self.assertRaises(ValueError):
            build_source(
                mappings=(
                    FieldMapping("raw.data.id", "dup"),
                    FieldMapping("raw.data.city", "dup"),
                )
            )

    def test_unknown_packed_level_rejected(self):
        with self.assertRaises(ValueError):
            build_source(packed_levels=("planet",))

    def test_empty_separator_rejected(self):
        with self.assertRaises(ValueError):
            build_source(display_separator="")


if __name__ == "__main__":
    unittest.main()
