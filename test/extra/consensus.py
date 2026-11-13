import random

import pytest

from geopy.distance import Distance, geodesic
from geopy.extra.consensus import find_consensus
from geopy.location import Location


def make_location(lat, lon, address="Somewhere", raw=None):
    return Location(address, (lat, lon), raw if raw is not None else {})


# Three sources reporting candidates for one address in Berlin.
BERLIN_A = make_location(52.52000, 13.40500, "Berlin A")
BERLIN_B = make_location(52.52010, 13.40510, "Berlin B")
BERLIN_C = make_location(52.51990, 13.40490, "Berlin C")


def test_all_sources_agree():
    result = find_consensus(
        [("nominatim", BERLIN_A), ("photon", BERLIN_B), ("arcgis", BERLIN_C)],
        max_distance=1.0,
    )
    assert not result.is_ambiguous
    assert result.source_count == 3
    assert result.confirmed is not None
    assert result.confirmed.sources == {"nominatim", "photon", "arcgis"}
    assert result.confirmed.indices == (0, 1, 2)


def test_max_distance_accepts_distance_instance():
    result = find_consensus(
        [("a", BERLIN_A), ("b", BERLIN_B)],
        max_distance=Distance(1),
    )
    assert result.source_count == 2
    assert result.confirmed is not None


def test_chain_does_not_merge():
    # A--B and B--C are each within the threshold, but A--C is not:
    # the three must not collapse into a single group.
    west = make_location(52.0, 13.0000)
    middle = make_location(52.0, 13.0070)  # ~480 m from each end
    east = make_location(52.0, 13.0140)     # ~960 m from west
    assert geodesic(west.point, middle.point).kilometers < 0.5
    assert geodesic(middle.point, east.point).kilometers < 0.5
    assert geodesic(west.point, east.point).kilometers > 0.5

    result = find_consensus(
        [("a", west), ("b", middle), ("c", east)],
        max_distance=0.5,
    )
    assert result.source_count == 2
    for group in result.groups:
        assert len(group.members) == 2


def test_one_candidate_per_source():
    # Source "a" reports two candidates; only one may join the group.
    far = make_location(48.0, 11.0, "Munich")
    result = find_consensus(
        [("a", BERLIN_A), ("a", far), ("b", BERLIN_B)],
        max_distance=1.0,
    )
    assert result.source_count == 2
    assert result.confirmed is not None
    assert result.confirmed.sources == {"a", "b"}
    locations = [member.location for member in result.confirmed.members]
    assert far not in locations


def test_source_of_callable():
    def source_of(location):
        return location.raw["source"]

    candidates = [
        make_location(52.52000, 13.40500, raw={"source": "x"}),
        make_location(52.52010, 13.40510, raw={"source": "y"}),
    ]
    result = find_consensus(candidates, 1.0, source_of=source_of)
    assert result.source_count == 2
    assert result.confirmed.sources == {"x", "y"}


def test_pairs_required_without_source_of():
    with pytest.raises(TypeError):
        find_consensus([BERLIN_A, BERLIN_B], 1.0)


def test_ambiguous_groups_in_different_jurisdictions():
    # Two disjoint clusters, each confirmed by all three sources.
    def jurisdiction_of(location):
        return location.raw["state"]

    cluster1 = [
        ("s1", make_location(52.5200, 13.4050, raw={"state": "Berlin"})),
        ("s2", make_location(52.5201, 13.4051, raw={"state": "Berlin"})),
        ("s3", make_location(52.5199, 13.4049, raw={"state": "Berlin"})),
    ]
    cluster2 = [
        ("s1", make_location(48.1351, 11.5820, raw={"state": "Bavaria"})),
        ("s2", make_location(48.1352, 11.5821, raw={"state": "Bavaria"})),
        ("s3", make_location(48.1350, 11.5819, raw={"state": "Bavaria"})),
    ]
    candidates = cluster1 + cluster2
    result = find_consensus(
        candidates, 1.0, jurisdiction_of=jurisdiction_of,
    )
    assert result.is_ambiguous
    assert result.confirmed is None
    assert result.source_count == 3
    assert len(result.groups) == 2
    # Original response indices are preserved for tracing.
    assert {group.indices for group in result.groups} == {
        (0, 1, 2), (3, 4, 5),
    }
    assert {next(iter(group.jurisdiction)) for group in result.groups} == {
        "Berlin", "Bavaria",
    }


def test_same_jurisdiction_groups_are_not_ambiguous():
    def jurisdiction_of(location):
        return location.raw["state"]

    # Two equally good groups, both in Berlin: confirmable.
    candidates = [
        ("s1", make_location(52.5200, 13.4050, raw={"state": "Berlin"})),
        ("s1", make_location(52.5201, 13.4051, raw={"state": "Berlin"})),
        ("s2", make_location(52.52005, 13.40505, raw={"state": "Berlin"})),
    ]
    result = find_consensus(
        candidates, 1.0, jurisdiction_of=jurisdiction_of,
    )
    assert not result.is_ambiguous
    assert result.confirmed is not None
    assert result.source_count == 2
    assert len(result.groups) == 2


def test_input_order_does_not_change_result():
    def jurisdiction_of(location):
        return location.raw.get("state")

    candidates = [
        ("s1", make_location(52.5200, 13.4050, "A1", {"state": "Berlin"})),
        ("s2", make_location(52.5201, 13.4051, "A2", {"state": "Berlin"})),
        ("s3", make_location(52.5199, 13.4049, "A3", {"state": "Berlin"})),
        ("s1", make_location(48.1351, 11.5820, "B1", {"state": "Bavaria"})),
        ("s2", make_location(48.1352, 11.5821, "B2", {"state": "Bavaria"})),
        ("s3", make_location(48.1350, 11.5819, "B3", {"state": "Bavaria"})),
        ("s1", make_location(52.5200, 13.4052, "A4", {"state": "Berlin"})),
    ]
    reference = find_consensus(
        candidates, 1.0, jurisdiction_of=jurisdiction_of,
    )
    rng = random.Random(20261112)
    for _ in range(20):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        result = find_consensus(
            shuffled, 1.0, jurisdiction_of=jurisdiction_of,
        )
        assert result.is_ambiguous == reference.is_ambiguous
        assert result.source_count == reference.source_count
        # Canonical group identity ignores input order and indices.
        assert [g._key() for g in result.groups] == [
            g._key() for g in reference.groups
        ]
        if reference.confirmed is not None:
            assert result.confirmed._key() == reference.confirmed._key()


def test_empty_input():
    result = find_consensus([], 1.0)
    assert result.groups == ()
    assert result.source_count == 0
    assert not result.is_ambiguous
    assert result.confirmed is None


def test_centroid():
    result = find_consensus(
        [("a", make_location(52.0, 13.0)), ("b", make_location(52.0, 13.001))],
        max_distance=1.0,
    )
    lat, lon = result.confirmed.centroid
    assert lat == pytest.approx(52.0)
    assert lon == pytest.approx(13.0005)
