import json
from urllib.parse import parse_qs

import pytest

from geopy.adapters import AioHTTPAdapter
from geopy.exc import ConfigurationError, GeocoderServiceError
from geopy.extra.geonames_pagination import (
    Checkpoint,
    GeoNamesPaginator,
    PendingRange,
    ReplayAdapter,
    ReplayError,
)
from geopy.geocoders import GeoNames


def make_place(geoname_id, name="Springfield", lat="1.0", lng="2.0"):
    return {
        "geonameId": geoname_id,
        "name": name,
        "lat": lat,
        "lng": lng,
        "adminName1": "State",
        "countryName": "Country",
    }


def make_page(ids):
    return {"geonames": [make_place(geoname_id) for geoname_id in ids]}


def make_geocoder(transcript=None):
    def factory(*, proxies, ssl_context):
        return ReplayAdapter(
            transcript=transcript,
            mode="replay",
            proxies=proxies,
            ssl_context=ssl_context,
        )

    return GeoNames(username="dummy", adapter_factory=factory)


def make_paginator(geocoder, pages_by_start, **kwargs):
    paginator = GeoNamesPaginator(geocoder, **kwargs)
    for start, doc in pages_by_start.items():
        key = ReplayAdapter._key_for_url(paginator._build_url(start))
        geocoder.adapter._transcript[key] = doc
    return paginator


class TestReplayAdapter:
    def test_replay_returns_recorded_payload_without_network(self):
        adapter = ReplayAdapter(
            transcript={"q=1": {"geonames": []}},
            mode="replay",
            proxies={},
            ssl_context=None,
        )
        result = adapter.get_json(
            "http://example.test/searchJSON?q=1",
            timeout=1,
            headers={},
        )
        assert result == {"geonames": []}
        assert adapter.request_log == ["q=1"]

    def test_replay_isolates_returned_payload_from_transcript(self):
        adapter = ReplayAdapter(transcript={"q=1": {"v": [1]}}, mode="replay")
        adapter.get_json("http://x/?q=1", timeout=1, headers={})["v"].append(2)
        assert adapter.transcript() == {"q=1": {"v": [1]}}

    def test_unknown_request_raises_replay_error(self):
        adapter = ReplayAdapter(mode="replay")
        with pytest.raises(ReplayError):
            adapter.get_json("http://x/?missing=1", timeout=1, headers={})

    def test_key_is_independent_of_parameter_order(self):
        url_a = "http://x/?a=1&b=2"
        url_b = "http://x/?b=2&a=1"
        assert ReplayAdapter._key_for_url(url_a) == (
            ReplayAdapter._key_for_url(url_b)
        )

    def test_record_mode_requires_wrapped_factory(self):
        with pytest.raises(ValueError):
            ReplayAdapter(mode="record")

    def test_record_mode_captures_and_replays_responses(self):
        class StubAdapter(ReplayAdapter):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.mode = "stub"

            def get_json(self, url, *, timeout, headers):
                return {"geonames": [make_place(1)]}

        recorded = ReplayAdapter(
            mode="record",
            wrapped_factory=lambda **kwargs: StubAdapter(**kwargs),
        )
        recorded.get_json("http://x/?q=1", timeout=1, headers={})

        replay = ReplayAdapter(
            transcript=recorded.transcript(), mode="replay"
        )
        assert replay.get_json(
            "http://x/?q=1", timeout=1, headers={}
        ) == {"geonames": [make_place(1)]}

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            ReplayAdapter(mode="nope")


class TestPagination:
    def test_complete_run_under_declared_stable_ordering(self):
        geocoder = make_geocoder()
        paginator = make_paginator(
            geocoder,
            {0: make_page([1, 2]), 2: make_page([3])},
            page_size=2,
            ordering="population",
        )

        result = paginator.fetch_all()

        assert result.is_complete is True
        assert result.has_pending_ranges is False
        assert result.received_ids == ["1", "2", "3"]
        assert [loc.raw["geonameId"] for loc in result.locations] == [1, 2, 3]
        assert result.checkpoint.end_observed is True
        # Two requests: the full page and the short terminal page.
        assert len(geocoder.adapter.request_log) == 2

    def test_without_ordering_never_complete_even_with_end_evidence(self):
        geocoder = make_geocoder()
        paginator = make_paginator(
            geocoder,
            {0: make_page([1]), 1: make_page([])},
            page_size=2,
            ordering=None,
        )

        result = paginator.fetch_all()

        assert result.checkpoint.end_observed is True
        assert result.is_complete is False
        assert result.received_ids == ["1"]

    def test_max_pages_stops_before_end_without_end_evidence(self):
        geocoder = make_geocoder()
        paginator = make_paginator(
            geocoder,
            {0: make_page([1, 2]), 2: make_page([3, 4])},
            page_size=2,
            ordering="population",
            max_pages=1,
        )

        result = paginator.fetch_all()

        assert result.is_complete is False
        assert result.checkpoint.end_observed is False
        assert result.checkpoint.next_start == 2
        assert result.received_ids == ["1", "2"]

    def test_duplicate_address_text_is_reconciled_by_stable_id(self):
        page = {
            "geonames": [
                make_place(10, name="Springfield"),
                make_place(11, name="Springfield"),
            ]
        }
        geocoder = make_geocoder()
        paginator = make_paginator(
            geocoder,
            {0: page, 2: make_page([])},
            page_size=2,
            ordering="name",
        )

        result = paginator.fetch_all()

        assert result.is_complete is True
        assert {loc.raw["geonameId"] for loc in result.locations} == {10, 11}
        assert all(loc.address.startswith("Springfield") for loc in result.locations)

    def test_same_id_seen_again_after_shift_is_deduplicated(self):
        geocoder = make_geocoder()
        first = make_paginator(
            geocoder,
            {0: make_page([1, 2])},
            page_size=2,
            ordering="population",
            max_pages=1,
        )
        first.fetch_all()
        checkpoint = first.checkpoint

        # An insertion between page 0 and page 1 makes id 2 reappear.
        resumed_geocoder = make_geocoder()
        resumed = GeoNamesPaginator(
            resumed_geocoder,
            page_size=2,
            ordering="population",
            checkpoint=checkpoint,
        )
        for start, doc in {
            0: make_page([1, 2]),
            2: make_page([2, 3]),
            4: make_page([]),
        }.items():
            key = ReplayAdapter._key_for_url(resumed._build_url(start))
            resumed_geocoder.adapter._transcript[key] = doc

        result = resumed.fetch_all()

        ids = [loc.raw["geonameId"] for loc in result.locations]
        assert sorted(ids) == [1, 2, 3]
        assert len(ids) == len(set(ids))

    def test_missing_geoname_id_raises(self):
        geocoder = make_geocoder()
        broken_doc = {"geonames": [{"name": "no id", "lat": "1", "lng": "2"}]}
        paginator = make_paginator(
            geocoder,
            {0: broken_doc},
            page_size=2,
            ordering="population",
        )
        with pytest.raises(GeocoderServiceError):
            paginator.fetch_all()

    def test_unknown_ordering_rejected(self):
        geocoder = make_geocoder()
        with pytest.raises(ValueError):
            GeoNamesPaginator(
                geocoder,
                page_size=2,
                ordering="not-a-real-ordering",
            )

    def test_async_adapter_rejected(self):
        # AioHTTPAdapter creates its session lazily, so instantiating the
        # geocoder does not open any resource that needs closing.
        async_geocoder = GeoNames(
            username="dummy",
            adapter_factory=AioHTTPAdapter,
        )
        with pytest.raises(ConfigurationError):
            GeoNamesPaginator(
                async_geocoder,
                page_size=2,
                ordering="population",
            )

    def test_replay_without_record_raises(self):
        geocoder = make_geocoder()  # empty transcript
        paginator = GeoNamesPaginator(
            geocoder, page_size=2, ordering="population"
        )
        with pytest.raises(ReplayError):
            paginator.fetch_all()

    def test_list_params_become_repeated_query_parameters(self):
        geocoder = make_geocoder()
        paginator = GeoNamesPaginator(
            geocoder,
            query_params={"country": ["DE", "FR"]},
            page_size=2,
            ordering="population",
        )
        url = paginator._build_url(0)
        assert "country=DE" in url
        assert "country=FR" in url

    def test_service_error_status_propagates(self):
        geocoder = make_geocoder()
        paginator = make_paginator(
            geocoder,
            {0: {"status": {"value": 10, "message": "bad credentials"}}},
            page_size=2,
            ordering="population",
        )
        with pytest.raises(GeocoderServiceError):
            paginator.fetch_all()


class TestCheckpoint:
    def test_save_and_load_roundtrip(self, tmp_path):
        geocoder = make_geocoder()
        paginator = make_paginator(
            geocoder,
            {0: make_page([1, 2]), 2: make_page([3])},
            query_params={"country": "DE"},
            page_size=2,
            ordering="population",
        )
        paginator.fetch_all()
        path = tmp_path / "checkpoint.json"
        paginator.save_checkpoint(path)

        raw = json.loads(path.read_text())
        assert raw["query_params"] == {"country": "DE"}
        assert raw["ordering"] == "population"

        restored = Checkpoint.load(path)
        assert restored.observed_ids == ["1", "2", "3"]
        assert restored.seen_pages[0] == {"start": 0, "ids": ["1", "2"]}
        assert restored.end_observed is True

    def test_resume_with_unchanged_head_completes(self, tmp_path):
        geocoder = make_geocoder()
        first = make_paginator(
            geocoder,
            {0: make_page([1, 2]), 2: make_page([3, 4])},
            page_size=2,
            ordering="population",
            max_pages=1,
        )
        result = first.fetch_all()
        assert result.is_complete is False
        path = tmp_path / "checkpoint.json"
        first.save_checkpoint(path)

        resumed_geocoder = make_geocoder()
        resumed = GeoNamesPaginator.from_checkpoint(path, resumed_geocoder)
        for start, doc in {
            0: make_page([1, 2]),
            2: make_page([3]),
        }.items():
            key = ReplayAdapter._key_for_url(resumed._build_url(start))
            resumed_geocoder.adapter._transcript[key] = doc

        resumed_result = resumed.fetch_all()

        assert resumed_result.is_complete is True
        assert resumed_result.received_ids == ["1", "2", "3"]
        # The head page is re-read, then fetching continues at startRow=2.
        requested_starts = [
            parse_qs(key)["startRow"][0]
            for key in resumed_geocoder.adapter.request_log
        ]
        assert requested_starts == ["0", "2"]

    def test_head_movement_reports_range_and_reread_authenticates_it(self):
        geocoder = make_geocoder()
        first = make_paginator(
            geocoder,
            {0: make_page([1, 2, 3]), 3: make_page([4, 5])},
            page_size=3,
            ordering="population",
        )
        first.fetch_all()
        checkpoint = first.checkpoint

        # A new candidate is inserted at the head of the result list.
        resumed_geocoder = make_geocoder()
        resumed = GeoNamesPaginator(
            resumed_geocoder,
            page_size=3,
            ordering="population",
            checkpoint=checkpoint,
        )
        for start, doc in {
            0: make_page([9, 1, 2]),
            3: make_page([3, 4, 5]),
            6: make_page([]),
        }.items():
            key = ReplayAdapter._key_for_url(resumed._build_url(start))
            resumed_geocoder.adapter._transcript[key] = doc

        result = resumed.fetch_all()

        # Every formerly fetched offset was re-read, so the pending range
        # is authenticated and the run can complete.
        assert result.is_complete is True
        assert result.pending_ranges == []
        assert sorted(result.received_ids) == [
            "1", "2", "3", "4", "5", "9",
        ]

    def test_movement_at_offset_zero_is_authenticated_by_head_reread(self):
        geocoder = make_geocoder()
        first = make_paginator(
            geocoder,
            {0: make_page([1, 2]), 2: make_page([3])},
            page_size=2,
            ordering="population",
            max_pages=1,
        )
        first.fetch_all()
        checkpoint = first.checkpoint

        resumed_geocoder = make_geocoder()
        resumed = GeoNamesPaginator(
            resumed_geocoder,
            page_size=2,
            ordering="population",
            checkpoint=checkpoint,
            max_pages=1,
        )
        key = ReplayAdapter._key_for_url(resumed._build_url(0))
        # The very first element changed: divergence is offset 0, so the
        # full fetched prefix [0, 2) must be re-read.
        resumed_geocoder.adapter._transcript[key] = make_page([9, 1])

        result = resumed.fetch_all()

        # The re-read head page covers the whole fetched prefix [0, 2),
        # so that part is authenticated; the run stays incomplete only
        # because no end evidence exists yet.
        assert result.pending_ranges == []
        assert result.is_complete is False
        assert sorted(result.received_ids) == ["1", "2", "9"]
        assert result.checkpoint.next_start == 2

    def test_unauthenticated_range_is_kept_when_source_cannot_certify(self):
        geocoder = make_geocoder()
        first = make_paginator(
            geocoder,
            {0: make_page([1, 2, 3]), 3: make_page([4, 5])},
            page_size=3,
            ordering="population",
        )
        first.fetch_all()
        checkpoint = first.checkpoint

        resumed_geocoder = make_geocoder()
        resumed = GeoNamesPaginator(
            resumed_geocoder,
            page_size=3,
            ordering="population",
            checkpoint=checkpoint,
            max_pages=1,  # only the head re-read is allowed
        )
        key = ReplayAdapter._key_for_url(resumed._build_url(0))
        resumed_geocoder.adapter._transcript[key] = make_page([9, 1, 2])

        result = resumed.fetch_all()

        assert result.is_complete is False
        # Offsets [3, 5) were never re-fetched and stay pending.
        assert result.pending_ranges == [PendingRange(start=3, end=5)]
        assert result.checkpoint.end_observed is False

    def test_source_shrinking_clips_dangling_pending_range(self):
        geocoder = make_geocoder()
        first = make_paginator(
            geocoder,
            {0: make_page([1, 2, 3]), 3: make_page([4, 5])},
            page_size=3,
            ordering="population",
        )
        first.fetch_all()
        checkpoint = first.checkpoint

        resumed_geocoder = make_geocoder()
        resumed = GeoNamesPaginator(
            resumed_geocoder,
            page_size=3,
            ordering="population",
            checkpoint=checkpoint,
        )
        key = ReplayAdapter._key_for_url(resumed._build_url(0))
        # The dataset now contains a single row: the end evidence.
        resumed_geocoder.adapter._transcript[key] = make_page([1])

        result = resumed.fetch_all()

        # The pending range pointed past the new dataset length and is
        # clipped away once end-of-data is observed.
        assert result.pending_ranges == []
        assert result.is_complete is True
        assert result.received_ids == ["1", "2", "3", "4", "5"]

    def test_checkpoint_page_size_mismatch_rejected(self):
        geocoder = make_geocoder()
        checkpoint = Checkpoint(
            query_params={},
            page_size=5,
            ordering="population",
            next_start=0,
        )
        with pytest.raises(ValueError):
            GeoNamesPaginator(
                geocoder,
                page_size=10,
                ordering="population",
                checkpoint=checkpoint,
            )

    def test_checkpoint_ordering_mismatch_rejected(self):
        geocoder = make_geocoder()
        checkpoint = Checkpoint(
            query_params={},
            page_size=5,
            ordering="population",
            next_start=0,
        )
        with pytest.raises(ValueError):
            GeoNamesPaginator(
                geocoder,
                page_size=5,
                ordering="name",
                checkpoint=checkpoint,
            )

    def test_checkpoint_params_mismatch_rejected(self):
        geocoder = make_geocoder()
        checkpoint = Checkpoint(
            query_params={"country": "DE"},
            page_size=5,
            ordering="population",
            next_start=0,
        )
        with pytest.raises(ValueError):
            GeoNamesPaginator(
                geocoder,
                query_params={"country": "FR"},
                page_size=5,
                ordering="population",
                checkpoint=checkpoint,
            )


class TestPendingRange:
    def test_serialization_roundtrip(self):
        rng = PendingRange(start=3, end=7)
        assert PendingRange.from_dict(rng.to_dict()) == rng
