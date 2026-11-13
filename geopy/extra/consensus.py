"""Build a consensus location from candidates returned by multiple geocoders.

When the same address is geocoded by several independent sources, each
source may return a few candidate :class:`geopy.location.Location` items.
This module picks the group of candidates which is best confirmed by the
sources, under the business rules:

* any two candidates merged into one group must be within a given
  ellipsoid (geodesic) distance of each other -- pairwise, so a chain of
  individually-close candidates whose ends are far apart does *not*
  collapse into a single point;
* each source contributes at most one candidate to a group;
* the best groups cover the largest possible number of distinct sources.

If several best groups fall into different jurisdictions (e.g. different
administrative areas), the result is ambiguous: all of them are reported
together with the original response indices of their members, so the
caller can trace every candidate back to the raw geocoder response.

The search is exhaustive and the tie-breaking is canonical (based on the
candidate coordinates, addresses and source keys, never on the input
order), so reordering the input sequence never changes the confirmable
result.

Example::

    >>> from geopy.extra.consensus import find_consensus
    >>> result = find_consensus(
    ...     [("nominatim", loc1), ("photon", loc2), ("arcgis", loc3)],
    ...     max_distance=0.5,  # kilometers
    ... )
    >>> if result.confirmed is not None:
    ...     point = result.confirmed.centroid
    ... else:
    ...     for group in result.groups:
    ...         print(group.indices, group.jurisdiction)
"""

from geopy.distance import Distance, geodesic
from geopy.location import Location

__all__ = (
    "Candidate",
    "CandidateGroup",
    "ConsensusResult",
    "find_consensus",
)


class Candidate:
    """A single geocoder response item taking part in the consensus search.

    Wraps a :class:`geopy.location.Location` with its source key and the
    item's index in the original response sequence.
    """

    __slots__ = ("location", "source", "index")

    def __init__(self, location, source, index):
        self.location = location
        self.source = source
        self.index = index

    @property
    def latitude(self):
        return self.location.latitude

    @property
    def longitude(self):
        return self.location.longitude

    def _key(self):
        # Canonical, input-order-independent identity used for sorting and
        # tie-breaking. The index is deliberately excluded.
        return (
            _sortable(self.source),
            self.location.latitude,
            self.location.longitude,
            self.location.address,
        )

    def __repr__(self):
        return "Candidate(%r, %r, index=%r)" % (
            self.location, self.source, self.index)


class CandidateGroup:
    """A compatible set of candidates, at most one per source."""

    __slots__ = ("members", "jurisdiction")

    def __init__(self, members, jurisdiction=None):
        self.members = tuple(members)
        self.jurisdiction = jurisdiction

    @property
    def sources(self):
        """The distinct source keys contributing to this group."""
        return frozenset(member.source for member in self.members)

    @property
    def indices(self):
        """Original response indices of the members, sorted ascending."""
        return tuple(sorted(member.index for member in self.members))

    @property
    def centroid(self):
        """The mean ``(latitude, longitude)`` of the member locations."""
        n = len(self.members)
        return (
            sum(member.latitude for member in self.members) / n,
            sum(member.longitude for member in self.members) / n,
        )

    def _key(self):
        return tuple(member._key() for member in self.members)

    def __len__(self):
        return len(self.members)

    def __eq__(self, other):
        return (
            isinstance(other, CandidateGroup) and
            self._key() == other._key()
        )

    def __repr__(self):
        return "CandidateGroup(members=%r)" % (self.members,)


class ConsensusResult:
    """The outcome of :func:`find_consensus`.

    .. attribute:: groups

        All best (most source-complete) compatible groups, in canonical
        order. Empty when there were no candidates.

    .. attribute:: source_count

        The number of distinct sources covered by each best group.

    .. attribute:: is_ambiguous

        ``True`` when the best groups fall into more than one
        jurisdiction, so no single group can be confirmed.

    .. attribute:: confirmed

        The single confirmed :class:`CandidateGroup`, or ``None`` when
        the result is ambiguous or empty.
    """

    __slots__ = ("groups", "source_count", "is_ambiguous", "confirmed")

    def __init__(self, groups, source_count, is_ambiguous, confirmed):
        self.groups = groups
        self.source_count = source_count
        self.is_ambiguous = is_ambiguous
        self.confirmed = confirmed

    def __repr__(self):
        return (
            "ConsensusResult(source_count=%r, is_ambiguous=%r, groups=%r)"
            % (self.source_count, self.is_ambiguous, self.groups)
        )


def _sortable(key):
    # Allow mixing arbitrary hashable source keys (e.g. str and int)
    # in comparisons.
    return (type(key).__name__, str(key))


def _normalize_candidates(candidates, source_of):
    normalized = []
    for index, item in enumerate(candidates):
        if source_of is None:
            if isinstance(item, Location):
                raise TypeError(
                    "when `source_of` is not given, each candidate must be "
                    "a (source, location) pair; got a bare Location: %r"
                    % (item,)
                )
            try:
                source, location = item
            except (TypeError, ValueError):
                raise TypeError(
                    "when `source_of` is not given, each candidate must be "
                    "a (source, location) pair; got %r" % (item,)
                )
        else:
            location = item
            source = source_of(location)
        normalized.append(Candidate(location, source, index))
    return normalized


def _max_kilometers(max_distance):
    if isinstance(max_distance, Distance):
        return max_distance.kilometers
    return float(max_distance)


def _measure_kilometers(measure, candidate_a, candidate_b):
    value = measure(candidate_a.location.point, candidate_b.location.point)
    if isinstance(value, Distance):
        return value.kilometers
    return float(value)


def _compatible_pairs(candidates, max_kilometers, measure):
    compatible = [set() for _ in candidates]
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            if a.source == b.source:
                continue
            if _measure_kilometers(measure, a, b) <= max_kilometers:
                compatible[i].add(j)
                compatible[j].add(i)
    return compatible


def _maximum_cliques(candidates, compatible):
    """Find all largest cliques of pairwise-compatible candidates.

    Returns a list of tuples of positions into ``candidates``. The search
    is order-independent: candidates are visited in canonical order and
    every largest clique is collected, never just the first one found.
    """
    order = sorted(range(len(candidates)), key=lambda i: candidates[i]._key())
    best = []
    best_size = 0

    def search(pos, chosen, used_sources):
        nonlocal best, best_size
        remaining_sources = {
            candidates[order[k]].source for k in range(pos, len(order))
        } - used_sources
        if len(chosen) + len(remaining_sources) < best_size:
            return
        if pos == len(order):
            if len(chosen) > best_size:
                best = [tuple(sorted(chosen))]
                best_size = len(chosen)
            elif len(chosen) == best_size:
                clique = tuple(sorted(chosen))
                if clique not in best:
                    best.append(clique)
            return
        i = order[pos]
        candidate = candidates[i]
        # Option 1: skip this candidate.
        search(pos + 1, chosen, used_sources)
        # Option 2: take this candidate, if allowed.
        if (candidate.source not in used_sources and
                all(j in compatible[i] for j in chosen)):
            search(pos + 1, chosen + (i,), used_sources | {candidate.source})

    search(0, (), frozenset())
    return best


def find_consensus(candidates, max_distance, *, source_of=None,
                   jurisdiction_of=None, distance=None):
    """Find the most source-complete compatible candidate group(s).

    :param candidates:
        An iterable of candidates. When ``source_of`` is given, each item
        must be a :class:`geopy.location.Location`; otherwise each item
        must be a ``(source, location)`` pair. ``source`` may be any
        hashable identifying the geocoder the candidate came from.

    :param max_distance:
        The largest allowed ellipsoid distance between any two members of
        a group: a :class:`geopy.distance.Distance` instance or a number
        of kilometers.

    :param source_of:
        Optional callable mapping a :class:`geopy.location.Location` to
        its source key.

    :param jurisdiction_of:
        Optional callable mapping a :class:`geopy.location.Location` to a
        hashable jurisdiction key (e.g. an administrative area). When
        given, best groups whose members' jurisdiction key sets differ
        make the result ambiguous. When not given, all groups are
        considered to share one jurisdiction.

    :param distance:
        The distance measure to use, called with two
        :class:`geopy.point.Point` arguments. Defaults to
        :class:`geopy.distance.geodesic` (the WGS-84 ellipsoid).

    :rtype: ConsensusResult
    """
    measure = distance if distance is not None else geodesic
    normalized = _normalize_candidates(candidates, source_of)
    if not normalized:
        return ConsensusResult((), 0, False, None)

    max_kilometers = _max_kilometers(max_distance)
    compatible = _compatible_pairs(normalized, max_kilometers, measure)
    cliques = _maximum_cliques(normalized, compatible)

    groups = []
    for clique in cliques:
        members = tuple(sorted(
            (normalized[i] for i in clique),
            key=lambda candidate: candidate._key(),
        ))
        if jurisdiction_of is not None:
            jurisdiction = frozenset(
                jurisdiction_of(member.location) for member in members
            )
        else:
            jurisdiction = None
        groups.append(CandidateGroup(members, jurisdiction))

    groups.sort(key=lambda group: group._key())
    source_count = len(groups[0].sources)

    jurisdictions = {group.jurisdiction for group in groups}
    is_ambiguous = len(jurisdictions) > 1
    confirmed = None if is_ambiguous else groups[0]

    return ConsensusResult(tuple(groups), source_count, is_ambiguous,
                           confirmed)
