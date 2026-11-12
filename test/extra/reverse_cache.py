from datetime import datetime, timedelta, timezone

import pytest

from geopy.extra.reverse_cache import (
    CONFLICTING_RECORDS,
    MISSING_ADMIN_LEVEL,
    NO_VALID_RECORD,
    CacheRecord,
    OfflineReverseCache,
    ValidityRegion,
)

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
FUTURE = NOW + timedelta(days=30)
PAST = NOW - timedelta(days=1)


def make_record(record_id="r1", source="src", region=None, expires_at=FUTURE,
                admin_version="v1", admin=None):
    if region is None:
        region = ValidityRegion(south=30.0, north=40.0,
                                west=-120.0, east=-110.0)
    if admin is None:
        admin = {"country": "US", "state": "California"}
    return CacheRecord(record_id=record_id, source=source, region=region,
                       expires_at=expires_at, admin_version=admin_version,
                       admin=admin)


class TestValidityRegion:
    def test_contains_interior(self):
        region = ValidityRegion(south=30.0, north=40.0,
                                west=-120.0, east=-110.0)
        assert region.contains(35.0, -115.0)
        assert not region.contains(45.0, -115.0)
        assert not region.contains(35.0, -100.0)

    def test_closed_edges_by_default(self):
        region = ValidityRegion(south=30.0, north=40.0,
                                west=-120.0, east=-110.0)
        assert region.contains(30.0, -115.0)
        assert region.contains(40.0, -115.0)
        assert region.contains(35.0, -120.0)
        assert region.contains(35.0, -110.0)

    def test_open_edges_per_source_declaration(self):
        region = ValidityRegion(south=30.0, north=40.0,
                                west=-120.0, east=-110.0,
                                include_south=False, include_north=False,
                                include_west=False, include_east=False)
        assert region.contains(35.0, -115.0)
        assert not region.contains(30.0, -115.0)
        assert not region.contains(40.0, -115.0)
        assert not region.contains(35.0, -120.0)
        assert not region.contains(35.0, -110.0)

    def test_mixed_edges(self):
        region = ValidityRegion(south=30.0, north=40.0,
                                west=-120.0, east=-110.0,
                                include_east=False)
        assert region.contains(35.0, -120.0)
        assert not region.contains(35.0, -110.0)

    def test_antimeridian_crossing(self):
        region = ValidityRegion(south=-10.0, north=10.0,
                                west=170.0, east=-170.0)
        assert region.crosses_antimeridian
        assert region.contains(0.0, 175.0)
        assert region.contains(0.0, -175.0)
        assert region.contains(0.0, 180.0)
        assert region.contains(0.0, -180.0)
        assert not region.contains(0.0, 0.0)
        assert not region.contains(0.0, 160.0)
        assert not region.contains(0.0, -160.0)

    def test_antimeridian_open_edges(self):
        region = ValidityRegion(south=-10.0, north=10.0,
                                west=170.0, east=-170.0,
                                include_west=False, include_east=False)
        assert region.contains(0.0, 180.0)
        assert not region.contains(0.0, 170.0)
        assert not region.contains(0.0, -170.0)

    def test_full_longitude_circle(self):
        region = ValidityRegion(south=-10.0, north=10.0,
                                west=-180.0, east=180.0)
        assert not region.crosses_antimeridian
        assert region.contains(0.0, 0.0)
        assert region.contains(0.0, 179.999)
        assert region.contains(0.0, -179.999)
        assert not region.contains(20.0, 0.0)

    def test_longitude_normalization(self):
        region = ValidityRegion(south=-10.0, north=10.0,
                                west=170.0, east=190.0)
        assert region.crosses_antimeridian
        assert region.contains(0.0, 185.0)
        assert not region.contains(0.0, 0.0)

    def test_invalid_latitude_range(self):
        with pytest.raises(ValueError):
            ValidityRegion(south=40.0, north=30.0, west=0.0, east=1.0)
        with pytest.raises(ValueError):
            ValidityRegion(south=-91.0, north=0.0, west=0.0, east=1.0)


class TestCacheRecord:
    def test_naive_expiry_rejected(self):
        with pytest.raises(ValueError):
            make_record(expires_at=datetime(2027, 1, 1))

    def test_validity_window(self):
        record = make_record()
        assert record.is_valid_at(NOW)
        assert not record.is_valid_at(FUTURE)
        assert not record.is_valid_at(FUTURE + timedelta(seconds=1))
        record.retracted = True
        assert not record.is_valid_at(NOW)


class TestLookup:
    def test_hit(self):
        cache = OfflineReverseCache()
        cache.add(make_record())
        decision = cache.lookup(35.0, -115.0, "state", at=NOW)
        assert decision
        assert decision.value == "California"
        assert decision.record_ids == ("r1",)
        assert decision.admin_versions == ("v1",)

    def test_miss_outside_all_regions(self):
        cache = OfflineReverseCache()
        cache.add(make_record())
        decision = cache.lookup(0.0, 0.0, "state", at=NOW)
        assert not decision
        assert decision.reason == NO_VALID_RECORD

    def test_expired_record_not_applicable(self):
        cache = OfflineReverseCache()
        cache.add(make_record(expires_at=PAST))
        decision = cache.lookup(35.0, -115.0, "state", at=NOW)
        assert not decision
        assert decision.reason == NO_VALID_RECORD

    def test_overlapping_records_agree(self):
        cache = OfflineReverseCache()
        cache.add(make_record("r1", admin_version="v1"))
        cache.add(make_record(
            "r2",
            region=ValidityRegion(south=34.0, north=36.0,
                                  west=-116.0, east=-114.0),
            admin_version="v2",
        ))
        decision = cache.lookup(35.0, -115.0, "state", at=NOW)
        assert decision
        assert decision.value == "California"
        assert set(decision.record_ids) == {"r1", "r2"}
        assert decision.admin_versions == ("v1", "v2")

    def test_overlapping_records_disagree(self):
        cache = OfflineReverseCache()
        cache.add(make_record("r1"))
        cache.add(make_record("r2", admin={"country": "US", "state": "Nevada"}))
        decision = cache.lookup(35.0, -115.0, "state", at=NOW)
        assert not decision
        assert decision.reason == CONFLICTING_RECORDS

    def test_agreement_on_other_level_does_not_release_target(self):
        cache = OfflineReverseCache()
        cache.add(make_record("r1"))
        cache.add(make_record("r2", admin={"country": "US", "state": "Nevada"}))
        # Both records agree on "country" but not on "state".
        assert cache.lookup(35.0, -115.0, "country", at=NOW)
        assert not cache.lookup(35.0, -115.0, "state", at=NOW)

    def test_record_missing_target_level(self):
        cache = OfflineReverseCache()
        cache.add(make_record("r1"))
        cache.add(make_record("r2", admin={"country": "US"}))
        decision = cache.lookup(35.0, -115.0, "state", at=NOW)
        assert not decision
        assert decision.reason == MISSING_ADMIN_LEVEL

    def test_lookup_many(self):
        cache = OfflineReverseCache()
        cache.add(make_record())
        decisions = cache.lookup_many(
            [(35.0, -115.0), (0.0, 0.0)], "state", at=NOW,
        )
        assert [bool(d) for d in decisions] == [True, False]
        assert decisions[0].value == "California"
        assert decisions[1].reason == NO_VALID_RECORD

    def test_antimeridian_hit(self):
        cache = OfflineReverseCache()
        cache.add(make_record(
            region=ValidityRegion(south=-10.0, north=10.0,
                                  west=170.0, east=-170.0),
            admin={"country": "NZ"},
        ))
        assert cache.lookup(0.0, 179.5, "country", at=NOW)
        assert cache.lookup(0.0, -179.5, "country", at=NOW)
        assert not cache.lookup(0.0, 0.0, "country", at=NOW)


class TestRetract:
    def test_only_point_covered_by_retracted_record(self):
        cache = OfflineReverseCache()
        cache.add(make_record("r1"))
        cache.lookup(35.0, -115.0, "state", at=NOW)
        cache.lookup(0.0, 0.0, "state", at=NOW)  # a miss, never certain

        report = cache.retract("r1", at=NOW)
        assert len(report) == 1
        entry = report[0]
        assert (entry.latitude, entry.longitude) == (35.0, -115.0)
        assert entry.admin_level == "state"
        assert entry.previous_value == "California"
        assert entry.reason == NO_VALID_RECORD
        assert "r1" in entry.detail

    def test_point_still_backed_by_other_record_not_listed(self):
        cache = OfflineReverseCache()
        cache.add(make_record("r1"))
        cache.add(make_record("r2"))
        cache.lookup(35.0, -115.0, "state", at=NOW)

        assert cache.retract("r1", at=NOW) == []
        # The remaining record still answers.
        assert cache.lookup(35.0, -115.0, "state", at=NOW)

    def test_remaining_records_conflict(self):
        cache = OfflineReverseCache()
        cache.add(make_record("r1"))
        cache.add(make_record("r2"))
        # r1 + r2 agree, so this is a hit.
        cache.lookup(35.0, -115.0, "state", at=NOW)
        # r3 arrives later and conflicts once r1 is gone.
        cache.add(make_record("r3",
                              admin={"country": "US", "state": "Nevada"}))

        report = cache.retract("r1", at=NOW)
        assert len(report) == 1
        assert report[0].reason == CONFLICTING_RECORDS
        assert "state" in report[0].detail

    def test_remaining_record_missing_level(self):
        cache = OfflineReverseCache()
        cache.add(make_record("r1"))
        cache.lookup(35.0, -115.0, "state", at=NOW)
        # r2 covers the point but does not provide the target level.
        cache.add(make_record("r2", admin={"country": "US"}))
        report = cache.retract("r1", at=NOW)
        assert len(report) == 1
        assert report[0].reason == MISSING_ADMIN_LEVEL

    def test_duplicate_queries_reported_once(self):
        cache = OfflineReverseCache()
        cache.add(make_record("r1"))
        cache.lookup(35.0, -115.0, "state", at=NOW)
        cache.lookup(35.0, -115.0, "state", at=NOW)
        report = cache.retract("r1", at=NOW)
        assert len(report) == 1

    def test_distinct_levels_reported_separately(self):
        cache = OfflineReverseCache()
        cache.add(make_record("r1"))
        cache.lookup(35.0, -115.0, "state", at=NOW)
        cache.lookup(35.0, -115.0, "country", at=NOW)
        report = cache.retract("r1", at=NOW)
        assert {(e.latitude, e.longitude, e.admin_level) for e in report} == {
            (35.0, -115.0, "state"),
            (35.0, -115.0, "country"),
        }

    def test_unknown_record(self):
        cache = OfflineReverseCache()
        with pytest.raises(KeyError):
            cache.retract("nope", at=NOW)

    def test_double_retract_rejected(self):
        cache = OfflineReverseCache()
        cache.add(make_record("r1"))
        cache.retract("r1", at=NOW)
        with pytest.raises(ValueError):
            cache.retract("r1", at=NOW)

    def test_retracted_record_not_applicable(self):
        cache = OfflineReverseCache()
        cache.add(make_record("r1"))
        cache.retract("r1", at=NOW)
        decision = cache.lookup(35.0, -115.0, "state", at=NOW)
        assert not decision
        assert decision.reason == NO_VALID_RECORD
