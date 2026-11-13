import pytest

from geopy.exc import (
    ConfigurationError,
    GeocoderQueryError,
    GeocoderUnavailable,
)
from geopy.extra.planner import (
    Plan,
    Source,
    Store,
    execute_plan,
    merge_evidence,
    normalize_address,
    plan_round,
)


def make_source(name, quota=10, regions=("DE",), cost=1.0, upstream=None):
    return Source(name=name, quota=quota, regions=regions, cost=cost,
                  upstream=upstream)


def make_store(store_id, address=None, region="DE", min_sources=1,
               address_key=None):
    return Store(store_id=store_id, address=address or store_id,
                 region=region, min_sources=min_sources,
                 address_key=address_key)


class FakeGeocoder:
    """An offline stand-in for a real geocoding service."""

    def __init__(self, results=None, error=None):
        self.results = results or {}
        self.error = error
        self.calls = []

    def geocode(self, address):
        self.calls.append(address)
        if self.error is not None:
            raise self.error
        return self.results.get(address)


class TestNormalizeAddress:
    def test_casefold_and_whitespace(self):
        assert normalize_address("  Hauptstr. 1,\n Berlin ") == \
            "hauptstr. 1, berlin"

    def test_equivalent_addresses_share_key(self):
        assert normalize_address("Main Street 1") == \
            normalize_address("main  street 1")


class TestValidation:
    def test_negative_quota(self):
        with pytest.raises(ConfigurationError):
            make_source("a", quota=-1)

    def test_negative_min_sources(self):
        with pytest.raises(ConfigurationError):
            make_store("s1", min_sources=-1)

    def test_duplicate_source_names(self):
        sources = [make_source("a"), make_source("a")]
        with pytest.raises(ConfigurationError):
            plan_round(sources, [make_store("s1")])

    def test_duplicate_store_ids(self):
        stores = [make_store("s1"), make_store("s1")]
        with pytest.raises(ConfigurationError):
            plan_round([make_source("a")], stores)


class TestPlanRound:
    def test_basic_coverage(self):
        plan = plan_round(
            [make_source("a")],
            [make_store("s1"), make_store("s2")],
        )
        assert plan.covered == {"s1", "s2"}
        assert plan.uncovered == frozenset()
        assert len(plan.requests) == 2
        assert plan.total_cost == 2.0

    def test_quota_limits_coverage(self):
        plan = plan_round(
            [make_source("a", quota=1)],
            [make_store("s1"), make_store("s2")],
        )
        assert len(plan.covered) == 1
        assert len(plan.requests) == 1
        assert plan.total_cost == 1.0

    def test_equivalent_addresses_share_one_request(self):
        stores = [
            make_store("s1", "Hauptstr. 1, Berlin"),
            make_store("s2", "hauptstr. 1,  berlin"),
        ]
        plan = plan_round([make_source("a", quota=1)], stores)
        assert plan.covered == {"s1", "s2"}
        assert len(plan.requests) == 1
        assert plan.total_cost == 1.0
        assert plan.requests[0].stores == ("s1", "s2")

    def test_explicit_address_key_sharing(self):
        stores = [
            make_store("s1", "Hauptstr. 1, 10115 Berlin", address_key="h1"),
            make_store("s2", "Hauptstraße 1, Berlin", address_key="h1"),
        ]
        plan = plan_round([make_source("a", quota=1)], stores)
        assert plan.covered == {"s1", "s2"}
        assert len(plan.requests) == 1

    def test_custom_normalize(self):
        stores = [make_store("s1", "alpha, x"), make_store("s2", "alpha, y")]
        plan = plan_round(
            [make_source("a", quota=1)],
            stores,
            normalize=lambda address: address.split(",")[0],
        )
        assert plan.covered == {"s1", "s2"}
        assert len(plan.requests) == 1

    def test_region_not_served(self):
        sources = [make_source("a", regions=("DE",))]
        stores = [make_store("s1", region="FR")]
        plan = plan_round(sources, stores)
        assert plan.covered == frozenset()
        assert plan.uncovered == {"s1"}
        assert plan.requests == ()

    def test_min_independent_sources(self):
        sources = [make_source("a"), make_source("b")]
        stores = [make_store("s1", min_sources=2)]
        plan = plan_round(sources, stores)
        assert plan.covered == {"s1"}
        assert len(plan.requests) == 2

    def test_aliases_of_same_upstream_are_not_independent(self):
        sources = [
            make_source("a1", upstream="upstream-a"),
            make_source("a2", upstream="upstream-a"),
        ]
        stores = [make_store("s1", min_sources=2)]
        plan = plan_round(sources, stores)
        assert plan.covered == frozenset()
        assert plan.uncovered == {"s1"}
        # No requests: two aliases can never satisfy the requirement,
        # and the cheapest plan covering nothing requests nothing.
        assert plan.requests == ()
        assert plan.total_cost == 0.0

    def test_cheapest_alias_is_chosen(self):
        sources = [
            make_source("expensive-alias", cost=5.0, upstream="upstream-a"),
            make_source("cheap-alias", cost=1.0, upstream="upstream-a"),
            make_source("other", cost=2.0, upstream="upstream-b"),
        ]
        stores = [make_store("s1", min_sources=2)]
        plan = plan_round(sources, stores)
        assert plan.covered == {"s1"}
        assert plan.total_cost == 3.0
        assert {request.source for request in plan.requests} == \
            {"cheap-alias", "other"}

    def test_lowest_cost_among_equal_coverage(self):
        sources = [
            make_source("cheap", cost=1.0),
            make_source("pricey", cost=9.0),
        ]
        stores = [make_store("s1")]
        plan = plan_round(sources, stores)
        assert plan.covered == {"s1"}
        assert plan.total_cost == 1.0
        assert [request.source for request in plan.requests] == ["cheap"]

    def test_coverage_beats_cost(self):
        sources = [
            make_source("cheap", quota=1, cost=1.0),
            make_source("pricey", quota=2, cost=100.0),
        ]
        stores = [make_store("s1"), make_store("s2")]
        plan = plan_round(sources, stores)
        # Covering both stores (cost 101) wins over covering a single
        # store for a cost of 1.
        assert plan.covered == {"s1", "s2"}
        assert plan.total_cost == 101.0
        assert {request.source for request in plan.requests} == \
            {"cheap", "pricey"}

    def test_existing_evidence_counts_towards_requirement(self):
        sources = [make_source("b", upstream="upstream-b")]
        stores = [make_store("s1", min_sources=2)]
        plan = plan_round(sources, stores,
                          evidence={"s1": {"upstream-a"}})
        assert plan.covered == {"s1"}
        assert len(plan.requests) == 1
        assert plan.requests[0].source == "b"

    def test_store_satisfied_by_evidence_needs_no_requests(self):
        sources = [make_source("a")]
        stores = [make_store("s1", min_sources=1)]
        plan = plan_round(sources, stores, evidence={"s1": {"upstream-a"}})
        assert plan.covered == {"s1"}
        assert plan.requests == ()
        assert plan.total_cost == 0.0

    def test_zero_min_sources_is_trivially_covered(self):
        plan = plan_round(
            [make_source("a")],
            [make_store("s1", min_sources=0)],
        )
        assert plan.covered == {"s1"}
        assert plan.requests == ()

    def test_source_without_quota_is_not_used(self):
        sources = [make_source("a", quota=0)]
        plan = plan_round(sources, [make_store("s1")])
        assert plan.uncovered == {"s1"}
        assert plan.requests == ()

    def test_expected_evidence(self):
        sources = [make_source("a", upstream="ua"),
                   make_source("b", upstream="ub")]
        stores = [make_store("s1", min_sources=2)]
        plan = plan_round(sources, stores, evidence={"s1": {"u0"}})
        # The existing evidence counts, so a single additional
        # independent upstream is sufficient (and cheapest).
        assert len(plan.requests) == 1
        assert plan.expected_evidence == {"s1": {"ua"}}

    def test_shared_request_serves_only_matching_region(self):
        sources = [make_source("a", regions=("DE",), quota=1)]
        stores = [
            make_store("s1", "Main St 1", region="DE", address_key="k"),
            make_store("s2", "Main St 1", region="FR", address_key="k"),
        ]
        plan = plan_round(sources, stores)
        assert plan.covered == {"s1"}
        assert plan.uncovered == {"s2"}
        assert plan.requests[0].stores == ("s1",)

    def test_plan_is_deterministic(self):
        sources = [make_source("b"), make_source("a")]
        stores = [make_store("s1"), make_store("s2")]
        first = plan_round(sources, stores)
        second = plan_round(list(reversed(sources)), stores)
        assert first == second
        assert isinstance(first, Plan)


class TestExecutePlan:
    def test_successful_round_collects_evidence(self):
        sources = [make_source("a", upstream="ua"),
                   make_source("b", upstream="ub")]
        stores = [make_store("s1", "addr-1", min_sources=2)]
        plan = plan_round(sources, stores)
        geocoders = {
            "a": FakeGeocoder({"addr-1": "loc-a"}),
            "b": FakeGeocoder({"addr-1": "loc-b"}),
        }
        result = execute_plan(plan, sources, geocoders)
        assert result.evidence == {"s1": {"ua", "ub"}}
        assert not result.temporary_failures
        assert not result.permanent_failures
        assert result.locations[("a", "addr-1")] == "loc-a"

    def test_callable_backend(self):
        sources = [make_source("a")]
        stores = [make_store("s1", "addr-1")]
        plan = plan_round(sources, stores)
        result = execute_plan(plan, sources, {"a": lambda query: "loc"})
        assert result.evidence == {"s1": {"a"}}

    def test_none_result_is_not_evidence(self):
        sources = [make_source("a")]
        stores = [make_store("s1", "addr-1")]
        plan = plan_round(sources, stores)
        result = execute_plan(plan, sources, {"a": FakeGeocoder()})
        assert result.evidence == {}
        assert result.not_found == (("a", "addr-1"),)

    def test_permanent_failure(self):
        sources = [make_source("a")]
        stores = [make_store("s1", "addr-1")]
        plan = plan_round(sources, stores)
        geocoders = {"a": FakeGeocoder(error=GeocoderQueryError("bad"))}
        result = execute_plan(plan, sources, geocoders)
        assert result.evidence == {}
        assert ("a", "addr-1") in result.permanent_failures

    def test_temporary_failure_allows_replanning_with_evidence(self):
        sources = [
            make_source("flaky", upstream="u1"),
            make_source("steady", upstream="u2"),
        ]
        stores = [make_store("s1", "addr-1", min_sources=2)]

        # First round: the flaky source is temporarily unavailable.
        plan = plan_round(sources, stores)
        assert len(plan.requests) == 2
        geocoders = {
            "flaky": FakeGeocoder(error=GeocoderUnavailable("down")),
            "steady": FakeGeocoder({"addr-1": "loc"}),
        }
        result = execute_plan(plan, sources, geocoders)
        assert ("flaky", "addr-1") in result.temporary_failures
        assert result.evidence == {"s1": {"u2"}}

        # Re-plan: the valid evidence already obtained participates, so
        # only the missing independent source is requested again.
        evidence = merge_evidence(result.evidence)
        replan = plan_round(sources, stores, evidence=evidence)
        assert len(replan.requests) == 1
        assert replan.requests[0].source == "flaky"
        assert replan.covered == {"s1"}

        # Second round: the flaky source recovers.
        recovered = {"flaky": FakeGeocoder({"addr-1": "loc"})}
        second = execute_plan(replan, sources, recovered)
        evidence = merge_evidence(evidence, second.evidence)
        assert evidence == {"s1": {"u1", "u2"}}

        # A third round needs no requests at all.
        final = plan_round(sources, stores, evidence=evidence)
        assert final.requests == ()
        assert final.covered == {"s1"}

    def test_equivalent_addresses_fail_and_succeed_together(self):
        sources = [make_source("a", quota=1)]
        stores = [
            make_store("s1", "Main St 1"),
            make_store("s2", "main  st 1"),
        ]
        plan = plan_round(sources, stores)
        assert len(plan.requests) == 1
        geocoder = FakeGeocoder({"Main St 1": "loc"})
        result = execute_plan(plan, sources, {"a": geocoder})
        assert geocoder.calls == ["Main St 1"]
        assert result.evidence == {"s1": {"a"}, "s2": {"a"}}


class TestMergeEvidence:
    def test_merge_unions_upstreams(self):
        merged = merge_evidence(
            {"s1": {"u1"}, "s2": {"u2"}},
            {"s1": {"u2", "u3"}},
            None,
            {},
        )
        assert merged == {"s1": {"u1", "u2", "u3"}, "s2": {"u2"}}
