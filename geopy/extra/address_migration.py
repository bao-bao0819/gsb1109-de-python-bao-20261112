"""
Offline migration of stored geocoder responses between two address schemas.

The old storage directory keeps both the parsed :class:`geopy.location.Location`
fields (display address, coordinates) and parts of the raw service response.
When the address source is replaced, those rows have to be replayed into the
new response schema without making any network calls. This module provides the
machinery for that; the caller declares everything that is specific to the two
service versions:

* the field mappings between the old row/raw payload and the new raw payload
  (:class:`FieldMapping`),
* the administrative hierarchy and the unit of every level
  (:class:`LevelSpec`, :class:`SourceSpec`, :class:`TargetSpec`).

Rows whose display string packs several administrative levels into one line
are only split automatically when the raw evidence allows a unique
decomposition. Rows that cannot be replayed directly are reported back with
the list of fields that are missing or ambiguous. Re-running the migration
never overwrites an already migrated record: the source identity observed on
the first run is preserved, and sites sharing a display name but belonging to
different administrative levels are never merged.

The module only depends on the standard library and therefore runs fully
offline.
"""

import collections.abc
import hashlib
import itertools
import json

from geopy.location import Location

__all__ = (
    "FieldMapping",
    "LevelSpec",
    "SourceSpec",
    "TargetSpec",
    "MigratedRecord",
    "SkippedRow",
    "MigrationReport",
    "AddressMigrator",
    "REASON_MISSING_FIELDS",
    "REASON_AMBIGUOUS_SPLIT",
)

REASON_MISSING_FIELDS = "missing_fields"
REASON_AMBIGUOUS_SPLIT = "ambiguous_split"

_NO_DEFAULT = object()


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _dig(obj, path):
    """Fetch a dotted ``path`` from nested mappings or objects."""
    current = obj
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, collections.abc.Mapping):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def _normalize_row(row):
    """Accept both plain dict rows and :class:`Location` instances."""
    if isinstance(row, Location):
        return {
            "address": row.address,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "raw": row.raw,
        }
    return row


class FieldMapping:
    """A single field mapping between the two response versions.

    :param str source: Dotted path of the field in the old stored row
        (including its embedded raw payload).
    :param str target: Name of the field in the new raw response.
    :param bool required: When ``True``, a row missing this value cannot be
        replayed and is reported with this field listed as missing.
    :param default: Optional value used when the source field is absent.
        A field with a default is never reported as missing.
    """

    __slots__ = ("source", "target", "required", "has_default", "default")

    def __init__(self, source, target, *, required=False, default=_NO_DEFAULT):
        self.source = source
        self.target = target
        self.required = required
        self.has_default = default is not _NO_DEFAULT
        self.default = default

    def __repr__(self):
        return "FieldMapping(%r -> %r, required=%r)" % (
            self.source, self.target, self.required,
        )


class LevelSpec:
    """An administrative hierarchy level.

    :param str name: Canonical name of the level (also used as a field name
        in the intermediate migration data).
    :param str unit: Administrative unit declared by the caller, e.g.
        ``"municipality"`` or ``"borough"``.
    """

    __slots__ = ("name", "unit")

    def __init__(self, name, *, unit=None):
        self.name = name
        self.unit = unit

    def __repr__(self):
        return "LevelSpec(%r, unit=%r)" % (self.name, self.unit)


class SourceSpec:
    """Description of the old stored response version.

    :param mappings: :class:`FieldMapping` objects reading values out of the
        old rows. Their ``target`` names populate the intermediate data.
    :param levels: Ordered :class:`LevelSpec` objects describing the old
        administrative hierarchy (broadest to narrowest).
    :param str address_field: Path of the display address field.
    :param str raw_field: Path of the embedded raw response field.
    :param str point_field: Optional path of a ``(latitude, longitude)``
        sequence in the old row.
    :param str latitude_field: Optional path of the latitude field.
    :param str longitude_field: Optional path of the longitude field.
    :param packed_levels: Levels that are concatenated into the single
        display address line, in display order. Items may be :class:`LevelSpec`
        objects or level names.
    :param str display_separator: Separator used inside the display line.
    :param str identity_field: Optional dotted path of the stable source
        identity. When omitted, a content fingerprint (including the resolved
        levels) is used instead.
    """

    def __init__(
        self,
        *,
        mappings,
        levels,
        address_field="address",
        raw_field="raw",
        point_field=None,
        latitude_field=None,
        longitude_field=None,
        packed_levels=(),
        display_separator=",",
        identity_field=None,
    ):
        self.mappings = tuple(mappings)
        self.levels = tuple(levels)
        self.address_field = address_field
        self.raw_field = raw_field
        self.point_field = point_field
        self.latitude_field = latitude_field
        self.longitude_field = longitude_field
        self.display_separator = display_separator
        self.identity_field = identity_field

        level_by_name = {level.name: level for level in self.levels}
        self._level_by_name = level_by_name
        self.packed_levels = tuple(
            self._resolve_level(item, level_by_name) for item in packed_levels
        )

        self._validate()

    @staticmethod
    def _resolve_level(item, level_by_name):
        if isinstance(item, LevelSpec):
            level = item
        else:
            level = level_by_name.get(item)
        if level is None or level.name not in level_by_name:
            raise ValueError("Unknown packed level: %r" % (item,))
        return level_by_name[level.name]

    def _validate(self):
        if not self.mappings:
            raise ValueError("SourceSpec requires at least one mapping")
        if not self.levels:
            raise ValueError("SourceSpec requires at least one level")
        if len(self._level_by_name) != len(self.levels):
            raise ValueError("SourceSpec level names must be unique")
        targets = [mapping.target for mapping in self.mappings]
        if len(set(targets)) != len(targets):
            raise ValueError("SourceSpec mapping targets must be unique")
        if not self.display_separator:
            raise ValueError("display_separator must not be empty")
        has_point = self.point_field is not None
        has_latlon = (
            self.latitude_field is not None
            and self.longitude_field is not None
        )
        if not has_point and not has_latlon:
            raise ValueError(
                "SourceSpec requires either point_field or both "
                "latitude_field and longitude_field"
            )


class TargetSpec:
    """Description of the new response version to replay rows into.

    :param mappings: :class:`FieldMapping` objects whose ``source`` names
        refer to the intermediate data produced from the :class:`SourceSpec`
        and whose ``target`` names are the new raw response fields.
    :param levels: Ordered :class:`LevelSpec` objects used to rebuild the
        display address. Their names must exist in the source hierarchy.
    :param str display_separator: Separator used when joining the target
        display address.
    """

    def __init__(self, *, mappings, levels, display_separator=", "):
        self.mappings = tuple(mappings)
        self.levels = tuple(levels)
        self.display_separator = display_separator
        if not self.mappings:
            raise ValueError("TargetSpec requires at least one mapping")
        if not self.levels:
            raise ValueError("TargetSpec requires at least one level")
        targets = [mapping.target for mapping in self.mappings]
        if len(set(targets)) != len(targets):
            raise ValueError("TargetSpec mapping targets must be unique")
        if not display_separator:
            raise ValueError("display_separator must not be empty")


class MigratedRecord:
    """A single successfully replayed row."""

    __slots__ = (
        "source_identity", "address", "latitude", "longitude",
        "raw", "_levels",
    )

    def __init__(self, source_identity, address, latitude, longitude,
                 raw, levels):
        self.source_identity = source_identity
        self.address = address
        self.latitude = latitude
        self.longitude = longitude
        self.raw = raw
        # levels: tuple of (level_name, value, unit)
        self._levels = tuple(levels)

    @property
    def admin_hierarchy(self):
        """Ordered list of ``{"level", "value", "unit"}`` dictionaries."""
        return [
            {"level": name, "value": value, "unit": unit}
            for name, value, unit in self._levels
        ]

    def to_location(self):
        """Rebuild a :class:`geopy.location.Location` from the record."""
        return Location(
            self.address, (self.latitude, self.longitude), self.raw
        )

    def to_dict(self):
        return {
            "source_identity": self.source_identity,
            "address": self.address,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "raw": self.raw,
            "levels": [
                {"level": name, "value": value, "unit": unit}
                for name, value, unit in self._levels
            ],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["source_identity"],
            data["address"],
            data["latitude"],
            data["longitude"],
            data["raw"],
            [
                (item["level"], item["value"], item["unit"])
                for item in data["levels"]
            ],
        )

    def __repr__(self):
        return "MigratedRecord(%r, %r)" % (
            self.source_identity, self.address,
        )

    def __eq__(self, other):
        if not isinstance(other, MigratedRecord):
            return NotImplemented
        return self.to_dict() == other.to_dict()


class SkippedRow:
    """A row that could not be replayed directly."""

    __slots__ = ("index", "source_identity", "reason",
                 "missing_fields", "detail")

    def __init__(self, index, source_identity, reason, missing_fields,
                 detail):
        self.index = index
        self.source_identity = source_identity
        self.reason = reason
        self.missing_fields = tuple(missing_fields)
        self.detail = detail

    def __repr__(self):
        return "SkippedRow(index=%r, reason=%r, missing_fields=%r)" % (
            self.index, self.reason, list(self.missing_fields),
        )


class MigrationReport:
    """Result of one :meth:`AddressMigrator.migrate` call."""

    def __init__(self):
        self.migrated = []
        self.skipped = []
        self.duplicates = []

    @property
    def migrated_count(self):
        return len(self.migrated)

    @property
    def skipped_count(self):
        return len(self.skipped)

    @property
    def duplicate_count(self):
        return len(self.duplicates)

    def __repr__(self):
        return (
            "MigrationReport(migrated=%d, skipped=%d, duplicates=%d)"
            % (self.migrated_count, self.skipped_count,
               self.duplicate_count)
        )


class AddressMigrator:
    """Offline, idempotent replayer of old address rows.

    :param SourceSpec source: Description of the old stored response.
    :param TargetSpec target: Description of the new response schema.
    """

    def __init__(self, source, target):
        self.source = source
        self.target = target
        self._migrated = {}
        self._validate_cross_references()

    def _validate_cross_references(self):
        canonical = {mapping.target for mapping in self.source.mappings}
        canonical.update(level.name for level in self.source.levels)
        source_level_names = {level.name for level in self.source.levels}

        for mapping in self.target.mappings:
            if mapping.source not in canonical:
                raise ValueError(
                    "Target mapping source %r is not produced by the "
                    "SourceSpec mappings or levels" % mapping.source
                )
        for level in self.target.levels:
            if level.name not in source_level_names:
                raise ValueError(
                    "Target level %r is not declared in SourceSpec"
                    % level.name
                )

    @property
    def migrated(self):
        """Migrated records in first-seen order."""
        return list(self._migrated.values())

    def migrate(self, rows):
        """Migrate an iterable of rows.

        Rows may be plain dictionaries or :class:`geopy.location.Location`
        instances. Repeated calls (or re-loaded state, see
        :meth:`load_state`) never replace an existing record: rows whose
        source identity was already migrated are reported as duplicates.
        """
        report = MigrationReport()
        for index, row in enumerate(rows):
            record, skipped, duplicate = self._migrate_one(row, index)
            if record is not None:
                report.migrated.append(record)
            elif duplicate is not None:
                report.duplicates.append(duplicate)
            else:
                report.skipped.append(skipped)
        return report

    def _migrate_one(self, row, index):
        row = _normalize_row(row)
        identity = self._resolve_identity(row, index)

        if identity in self._migrated:
            return None, None, identity

        canonical, extraction_missing = self._extract_canonical(row)
        if extraction_missing:
            return None, SkippedRow(
                index, identity, REASON_MISSING_FIELDS,
                extraction_missing,
                "Required source field has no value and no default",
            ), None

        split_reason, split_missing = self._resolve_packed_levels(
            row, canonical
        )
        if split_missing:
            return None, SkippedRow(
                index, identity, split_reason,
                split_missing,
                "Display string cannot be uniquely split into the "
                "declared levels using the raw evidence",
            ), None

        replay_missing = self._check_target_requirements(canonical)
        if replay_missing:
            return None, SkippedRow(
                index, identity, REASON_MISSING_FIELDS,
                replay_missing,
                "New response schema requires fields with no value",
            ), None

        try:
            latitude, longitude = self._extract_coordinates(row)
        except (TypeError, ValueError, IndexError):
            return None, SkippedRow(
                index, identity, REASON_MISSING_FIELDS,
                self._coordinate_fields(),
                "Coordinates cannot be read from the old row",
            ), None

        record = self._build_record(
            identity, row, canonical, latitude, longitude
        )
        self._migrated[identity] = record
        return record, None, None

    def _extract_canonical(self, row):
        canonical = {}
        missing = []
        for mapping in self.source.mappings:
            value = _dig(row, mapping.source)
            if _is_missing(value):
                if mapping.has_default:
                    value = mapping.default
                elif mapping.required:
                    missing.append(mapping.target)
                    continue
            if not _is_missing(value):
                canonical[mapping.target] = value
        return canonical, missing

    def _resolve_packed_levels(self, row, canonical):
        """Try to split the display line into the packed levels.

        Returns a ``(reason, missing_fields)`` tuple. On success both are
        empty/falsy. A missing address or raw evidence is reported as
        :data:`REASON_MISSING_FIELDS`; evidence that does not identify a
        unique decomposition is :data:`REASON_AMBIGUOUS_SPLIT`.
        """
        packed = self.source.packed_levels
        if not packed:
            return None, []

        packed_names = [level.name for level in packed]
        address = _dig(row, self.source.address_field)
        if _is_missing(address):
            return REASON_MISSING_FIELDS, packed_names

        evidence = {}
        for level in packed:
            value = canonical.get(level.name)
            if _is_missing(value):
                return REASON_MISSING_FIELDS, [level.name]
            evidence[level.name] = value.strip()

        tokens = [
            token.strip()
            for token in address.split(self.source.display_separator)
        ]
        if len(tokens) != len(packed) or any(not token for token in tokens):
            return REASON_AMBIGUOUS_SPLIT, packed_names

        valid_assignments = []
        for permutation in itertools.permutations(range(len(tokens))):
            assignment = {}
            for level_index, token_index in enumerate(permutation):
                level = packed[level_index]
                if tokens[token_index] != evidence[level.name]:
                    break
                assignment[level.name] = tokens[token_index]
            else:
                valid_assignments.append(assignment)

        if len(valid_assignments) != 1:
            return REASON_AMBIGUOUS_SPLIT, packed_names

        canonical.update(valid_assignments[0])
        return None, []

    def _check_target_requirements(self, canonical):
        missing = []
        for mapping in self.target.mappings:
            if mapping.required and _is_missing(
                canonical.get(mapping.source)
            ):
                missing.append(mapping.target)
        for level in self.target.levels:
            if _is_missing(canonical.get(level.name)):
                missing.append(level.name)
        return _dedupe(missing)

    def _build_record(self, identity, row, canonical, latitude, longitude):
        new_raw = {}
        for mapping in self.target.mappings:
            value = canonical.get(mapping.source)
            if not _is_missing(value):
                new_raw[mapping.target] = value

        address = self.target.display_separator.join(
            canonical[level.name] for level in self.target.levels
        )

        levels = [
            (
                level.name,
                canonical[level.name],
                level.unit,
            )
            for level in self.target.levels
        ]
        return MigratedRecord(
            identity, address, latitude, longitude, new_raw, levels
        )

    def _extract_coordinates(self, row):
        if self.source.point_field is not None:
            point = _dig(row, self.source.point_field)
            return float(point[0]), float(point[1])
        latitude = float(_dig(row, self.source.latitude_field))
        longitude = float(_dig(row, self.source.longitude_field))
        return latitude, longitude

    def _coordinate_fields(self):
        if self.source.point_field is not None:
            return [self.source.point_field]
        return [
            self.source.latitude_field,
            self.source.longitude_field,
        ]

    def _resolve_identity(self, row, index):
        if self.source.identity_field is not None:
            value = _dig(row, self.source.identity_field)
            if not _is_missing(value):
                return str(value)

        payload = {
            "address": _dig(row, self.source.address_field),
            "levels": self._identity_levels(row),
            "raw": _dig(row, self.source.raw_field),
        }
        serialized = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, default=str
        )
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
        return "fp-%s" % digest

    def _identity_levels(self, row):
        levels = {}
        for mapping in self.source.mappings:
            if mapping.target in self.source._level_by_name:
                value = _dig(row, mapping.source)
                if not _is_missing(value):
                    levels[mapping.target] = value
        return levels

    def state_dict(self):
        """Export the migration ledger as JSON-serializable data."""
        return {
            "records": [
                record.to_dict()
                for record in self._migrated.values()
            ],
        }

    def load_state(self, state):
        """Load a ledger previously produced by :meth:`state_dict`.

        Loaded records are treated as first-run results: their source
        identities are preserved and cannot be overwritten.
        """
        for data in state.get("records", []):
            record = MigratedRecord.from_dict(data)
            self._migrated.setdefault(record.source_identity, record)


def _dedupe(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
