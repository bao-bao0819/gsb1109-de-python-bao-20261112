""":class:`.OfflineReverseCache` provides offline cache-hit criteria for
batch reverse geocoding with geopy.

Every reverse geocoding result stored in the cache is annotated by its
source with:

- a *validity region* — the geographic area where the result holds,
  declared as a latitude/longitude box which may cross the antimeridian
  and whose edges may be declared open or closed by the source;
- an *expiry moment* — the instant after which the result must no
  longer be used;
- an *administrative division version* — the version of the
  administrative boundaries the result was computed against.

A cached answer for a target administrative level (e.g. ``"country"``
or ``"state"``) may only be released for a coordinate when **all**
currently valid records whose validity regions contain that coordinate
agree on the value for that level. If the applicable records disagree,
or none applies, the coordinate must be re-geocoded online.

Example::

    from datetime import datetime, timedelta, timezone

    from geopy.extra.reverse_cache import (
        CacheRecord, OfflineReverseCache, ValidityRegion,
    )

    cache = OfflineReverseCache()
    cache.add(CacheRecord(
        record_id="nominatim-1",
        source="nominatim",
        region=ValidityRegion(
            south=32.0, north=42.0, west=-125.0, east=-114.0,
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        admin_version="tiger-2026",
        admin={"country": "United States", "state": "California"},
    ))

    decision = cache.lookup(36.7, -119.7, admin_level="state")
    if decision:
        print(decision.value)  # "California"
    else:
        print(decision.reason)  # why the cache cannot answer

When a source retracts a record, :meth:`.OfflineReverseCache.retract`
reports exactly the previously answered points which must be
re-adjudicated, together with the reason each of them turned from
certain to unknown::

    for entry in cache.retract("nominatim-1"):
        print(entry.latitude, entry.longitude, entry.reason)
"""

from datetime import datetime, timezone

__all__ = (
    "CONFLICTING_RECORDS",
    "MISSING_ADMIN_LEVEL",
    "NO_VALID_RECORD",
    "CacheDecision",
    "CacheRecord",
    "OfflineReverseCache",
    "Rejudication",
    "ValidityRegion",
)

#: Miss reason: no valid (unexpired, not retracted) record's validity
#: region contains the coordinate.
NO_VALID_RECORD = "no_valid_record"

#: Miss reason: at least one applicable record does not provide the
#: requested administrative level, so agreement cannot be established.
MISSING_ADMIN_LEVEL = "missing_admin_level"

#: Miss reason: the applicable records disagree on the value of the
#: requested administrative level.
CONFLICTING_RECORDS = "conflicting_records"


def _normalize_longitude(longitude):
    """Normalize a longitude to the ``[-180, 180)`` range."""
    normalized = (longitude + 180.0) % 360.0 - 180.0
    # ``%`` may produce exactly -180.0 for inputs like 180.0; that is
    # the same meridian, which is what we want.
    return normalized


class ValidityRegion:
    """A latitude/longitude box where a reverse geocoding result holds,
    as declared by the source.

    Longitude ranges crossing the antimeridian are supported: a region
    with ``west=170`` and ``east=-170`` covers the 20-degree-wide strip
    around the 180th meridian. Declaring ``west=-180, east=180`` covers
    all longitudes.

    Each edge may independently be declared open or closed by the
    source via the ``include_*`` flags. Boundary comparisons are exact:
    a coordinate compares equal to a closed edge it sits on.
    """

    def __init__(self, south, north, west, east,
                 include_south=True, include_north=True,
                 include_west=True, include_east=True):
        south = float(south)
        north = float(north)
        west = float(west)
        east = float(east)
        if not -90.0 <= south <= north <= 90.0:
            raise ValueError(
                "expected -90 <= south <= north <= 90, got south=%r, "
                "north=%r" % (south, north)
            )
        self.south = south
        self.north = north
        self.include_south = bool(include_south)
        self.include_north = bool(include_north)
        self.include_west = bool(include_west)
        self.include_east = bool(include_east)

        # A declared span of a full turn (or more) covers every
        # longitude, e.g. west=-180, east=180.
        self._full_circle = (east - west) >= 360.0
        self._west = _normalize_longitude(west)
        self._east = _normalize_longitude(east)
        # After normalization, west > east means the range wraps
        # around the antimeridian.
        self._crosses_antimeridian = self._west > self._east

    @property
    def west(self):
        return self._west

    @property
    def east(self):
        return self._east

    @property
    def crosses_antimeridian(self):
        """Whether the longitude range wraps around the antimeridian."""
        return self._crosses_antimeridian

    def contains(self, latitude, longitude):
        """Whether the coordinate falls within this region, honoring
        the source-declared open/closed edges."""
        latitude = float(latitude)
        if latitude > self.south:
            lat_ok = latitude < self.north or (
                self.include_north and latitude == self.north
            )
        else:
            lat_ok = self.include_south and latitude == self.south and (
                latitude < self.north
                or (self.include_north and latitude == self.north)
            )
        if not lat_ok:
            return False

        longitude = _normalize_longitude(float(longitude))
        if self._full_circle:
            return True
        west_ok = longitude > self._west or (
            self.include_west and longitude == self._west
        )
        east_ok = longitude < self._east or (
            self.include_east and longitude == self._east
        )
        if self._crosses_antimeridian:
            return west_ok or east_ok
        return west_ok and east_ok

    def __repr__(self):
        return (
            "%s(south=%r, north=%r, west=%r, east=%r)" % (
                type(self).__name__, self.south, self.north,
                self._west, self._east,
            )
        )


class CacheRecord:
    """A single reverse geocoding result as annotated by its source.

    :param record_id: Unique identifier of the record.
    :param source: Name of the geocoding source which issued and
        annotated the result.
    :param region: :class:`.ValidityRegion` where the result holds.
    :param expires_at: Timezone-aware :class:`~datetime.datetime` of
        the expiry moment declared by the source. The record is valid
        strictly before this instant.
    :param admin_version: Version of the administrative divisions the
        result was computed against.
    :param admin: Mapping of administrative level names (e.g.
        ``"country"``, ``"state"``) to their values.
    """

    def __init__(self, record_id, source, region, expires_at,
                 admin_version, admin):
        if expires_at.tzinfo is None or expires_at.tzinfo.utcoffset(
                expires_at) is None:
            raise ValueError("expires_at must be timezone-aware")
        self.record_id = record_id
        self.source = source
        self.region = region
        self.expires_at = expires_at
        self.admin_version = admin_version
        self.admin = dict(admin)
        self.retracted = False

    def is_valid_at(self, moment):
        """Whether the record may be used at the given moment: not
        retracted and not yet expired."""
        return not self.retracted and moment < self.expires_at

    def __repr__(self):
        return "%s(record_id=%r, source=%r)" % (
            type(self).__name__, self.record_id, self.source,
        )


class CacheDecision:
    """The outcome of a cache lookup.

    A decision is truthy when it is a hit, i.e. the cached answer may
    be released for the requested coordinate and administrative level.
    """

    def __init__(self, hit, admin_level, value=None, reason=None,
                 record_ids=(), admin_versions=()):
        self.hit = bool(hit)
        self.admin_level = admin_level
        #: The agreed value for ``admin_level``; ``None`` on a miss.
        self.value = value
        #: On a miss, one of ``NO_VALID_RECORD``,
        #: ``MISSING_ADMIN_LEVEL`` or ``CONFLICTING_RECORDS``.
        self.reason = reason
        #: Ids of the valid records applicable to the coordinate.
        self.record_ids = tuple(record_ids)
        #: Administrative division versions of the applicable records.
        self.admin_versions = tuple(admin_versions)

    def __bool__(self):
        return self.hit

    def __repr__(self):
        if self.hit:
            return "%s(hit, %r=%r)" % (
                type(self).__name__, self.admin_level, self.value,
            )
        return "%s(miss, reason=%r)" % (type(self).__name__, self.reason)


class Rejudication:
    """A historical cache hit which must be re-adjudicated because a
    record was retracted.

    :ivar latitude: Latitude of the historical query point.
    :ivar longitude: Longitude of the historical query point.
    :ivar admin_level: The administrative level that was answered.
    :ivar previous_value: The value the cache previously released.
    :ivar reason: Why the point turned from certain to unknown; one of
        the miss reason constants.
    :ivar detail: Human-readable explanation mentioning the retracted
        record.
    """

    def __init__(self, latitude, longitude, admin_level, previous_value,
                 reason, detail):
        self.latitude = latitude
        self.longitude = longitude
        self.admin_level = admin_level
        self.previous_value = previous_value
        self.reason = reason
        self.detail = detail

    def __repr__(self):
        return "%s(%r, %r, reason=%r)" % (
            type(self).__name__, self.latitude, self.longitude,
            self.reason,
        )


class _QueryEvent:
    """A logged lookup, used to re-adjudicate historical hits."""

    __slots__ = ("latitude", "longitude", "admin_level", "hit", "value",
                 "record_ids")

    def __init__(self, latitude, longitude, admin_level, decision):
        self.latitude = latitude
        self.longitude = longitude
        self.admin_level = admin_level
        self.hit = decision.hit
        self.value = decision.value
        self.record_ids = decision.record_ids


class OfflineReverseCache:
    """Offline cache of annotated reverse geocoding results, providing
    cache-hit criteria for batch reverse lookups.

    Lookups are logged so that :meth:`retract` can report exactly the
    historical points affected by a source retraction.
    """

    def __init__(self):
        self._records = {}
        self._history = []

    def add(self, record):
        """Store a :class:`.CacheRecord` in the cache."""
        self._records[record.record_id] = record

    def get(self, record_id):
        """Return the record with the given id, or ``None``."""
        return self._records.get(record_id)

    def _evaluate(self, latitude, longitude, admin_level, at):
        applicable = [
            record for record in self._records.values()
            if record.region.contains(latitude, longitude)
            and record.is_valid_at(at)
        ]
        record_ids = tuple(record.record_id for record in applicable)
        if not applicable:
            return CacheDecision(
                False, admin_level, reason=NO_VALID_RECORD,
                record_ids=record_ids,
            )
        if any(admin_level not in record.admin for record in applicable):
            return CacheDecision(
                False, admin_level, reason=MISSING_ADMIN_LEVEL,
                record_ids=record_ids,
            )
        values = {record.admin[admin_level] for record in applicable}
        if len(values) != 1:
            return CacheDecision(
                False, admin_level, reason=CONFLICTING_RECORDS,
                record_ids=record_ids,
            )
        admin_versions = tuple(sorted({
            record.admin_version for record in applicable
        }))
        return CacheDecision(
            True, admin_level, value=values.pop(), record_ids=record_ids,
            admin_versions=admin_versions,
        )

    def lookup(self, latitude, longitude, admin_level, at=None):
        """Decide whether the cache may answer a reverse lookup for the
        coordinate at the given administrative level.

        The cache releases an answer only when every valid record whose
        validity region contains the coordinate agrees on the value for
        ``admin_level``.

        :param at: The moment of the lookup; defaults to now (UTC).
            Expired and retracted records are not applicable.
        :rtype: :class:`.CacheDecision`
        """
        if at is None:
            at = datetime.now(timezone.utc)
        latitude = float(latitude)
        longitude = float(longitude)
        decision = self._evaluate(latitude, longitude, admin_level, at)
        self._history.append(
            _QueryEvent(latitude, longitude, admin_level, decision)
        )
        return decision

    def lookup_many(self, points, admin_level, at=None):
        """Evaluate a batch of ``(latitude, longitude)`` pairs,
        returning one :class:`.CacheDecision` per point, in order.
        Points whose decision is falsy must be re-geocoded online.
        """
        return [
            self.lookup(latitude, longitude, admin_level, at=at)
            for latitude, longitude in points
        ]

    def retract(self, record_id, at=None):
        """Mark the record as retracted by its source and report the
        historical points which must be re-adjudicated.

        Only previously certain answers (cache hits) which involved the
        retracted record and can no longer be released are listed, each
        with the reason it turned from certain to unknown. Points whose
        answer is still backed by the remaining records are not listed.

        :param at: The moment of re-evaluation; defaults to now (UTC).
        :return: A list of :class:`.Rejudication`, one per distinct
            affected ``(latitude, longitude, admin_level)``.
        """
        record = self._records[record_id]
        if record.retracted:
            raise ValueError("record %r is already retracted" % record_id)
        record.retracted = True
        if at is None:
            at = datetime.now(timezone.utc)

        # Keep only the historical hits which involved the retracted
        # record, deduplicated by (point, level).
        hits = {}
        for event in self._history:
            if event.hit and record_id in event.record_ids:
                key = (event.latitude, event.longitude, event.admin_level)
                hits[key] = event

        report = []
        for (latitude, longitude, admin_level), event in hits.items():
            decision = self._evaluate(latitude, longitude, admin_level, at)
            if decision.hit:
                continue
            if decision.reason == NO_VALID_RECORD:
                detail = (
                    "record %r from source %r was retracted and no other "
                    "valid record covers this point"
                    % (record_id, record.source)
                )
            elif decision.reason == MISSING_ADMIN_LEVEL:
                detail = (
                    "record %r from source %r was retracted and the "
                    "remaining records do not all provide the %r level"
                    % (record_id, record.source, admin_level)
                )
            else:
                detail = (
                    "record %r from source %r was retracted and the "
                    "remaining records disagree on the %r level"
                    % (record_id, record.source, admin_level)
                )
            report.append(Rejudication(
                latitude, longitude, admin_level,
                previous_value=event.value,
                reason=decision.reason, detail=detail,
            ))
        return report
