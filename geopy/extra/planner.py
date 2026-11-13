""":mod:`geopy.extra.planner` helps to enrich a directory of stores
(e.g. a list of shop addresses) through multiple geocoding services at
once.

Each geocoding *source* has a request quota, a set of regions it can
serve, a cost per request and an upstream identity. Every store requires
confirmation from a minimum number of *independent* sources: two sources
which are aliases of the same upstream count as a single piece of
evidence.

:func:`plan_round` computes a single round of geocoding requests such
that the number of stores meeting their requirement is maximized. Among
all plans achieving that coverage, the cheapest one is chosen.
Equivalent addresses (as confirmed by the caller, see
:func:`normalize_address` and :attr:`Store.address_key`) share a single
request within the same source, so quota and cost are only consumed
once.

:func:`execute_plan` runs a plan against geocoder-like objects. When a
service responds with a temporary failure (e.g.
:class:`geopy.exc.GeocoderUnavailable`), the evidence already obtained
is kept and can be fed back into :func:`plan_round` via its ``evidence``
argument to plan the next round::

    evidence = {}
    for _ in range(max_rounds):
        plan = plan_round(sources, stores, evidence=evidence)
        if not plan.requests:
            break
        result = execute_plan(plan, sources, geocoders)
        evidence = merge_evidence(evidence, result.evidence)
"""

from dataclasses import dataclass, field
from typing import (
    AbstractSet,
    Dict,
    FrozenSet,
    Mapping,
    Optional,
    Tuple,
)

from geopy.exc import (
    ConfigurationError,
    GeocoderRateLimited,
    GeocoderServiceError,
    GeocoderTimedOut,
    GeocoderUnavailable,
)

__all__ = (
    "Plan",
    "PlannedRequest",
    "RoundResult",
    "Source",
    "Store",
    "execute_plan",
    "merge_evidence",
    "normalize_address",
    "plan_round",
)

#: Exceptions which indicate a temporary failure of a geocoding service:
#: the request may succeed when retried in a later round.
TEMPORARY_ERRORS = (GeocoderTimedOut, GeocoderUnavailable, GeocoderRateLimited)


def normalize_address(address):
    """Default address normalization used to detect equivalent addresses.

    Two addresses are considered equivalent (and thus share a single
    request within the same source) when their normalized forms are
    equal. The default normalization case-folds the address and
    collapses all whitespace. Callers may pass their own ``normalize``
    callable to :func:`plan_round` or set :attr:`Store.address_key`
    explicitly.
    """
    return " ".join(str(address).casefold().split())


@dataclass(frozen=True)
class Source:
    """A geocoding source which can be queried for store addresses.

    :param name: Unique name of the source.
    :param quota: Maximum number of requests this source may serve in a
        single round. Equivalent addresses share a single request.
    :param regions: Regions (jurisdictions) the source can serve. A
        store can only be assigned to a source serving its region.
    :param cost: Cost of a single request to this source.
    :param upstream: Identity of the upstream data provider. Sources
        sharing the same upstream are aliases: they do not count as
        independent evidence for a store. Defaults to ``name``.
    """

    name: str
    quota: int
    regions: AbstractSet[str]
    cost: float = 1.0
    upstream: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            raise ConfigurationError("Source name must not be empty")
        if self.quota < 0:
            raise ConfigurationError("Source quota must not be negative")
        if self.cost < 0:
            raise ConfigurationError("Source cost must not be negative")
        object.__setattr__(self, "regions", frozenset(self.regions))
        if self.upstream is None:
            object.__setattr__(self, "upstream", self.name)


@dataclass(frozen=True)
class Store:
    """A single entry of the store directory to be geocoded.

    :param store_id: Unique identifier of the store.
    :param address: The address to geocode.
    :param region: Region (jurisdiction) the store belongs to.
    :param min_sources: Minimum number of independent sources (distinct
        upstreams) required to consider this store confirmed.
    :param address_key: Explicit equivalence key for the address. Stores
        with equal keys share a single request within the same source.
        When not given, ``normalize(address)`` is used.
    """

    store_id: str
    address: str
    region: str
    min_sources: int = 1
    address_key: Optional[str] = None

    def __post_init__(self):
        if self.min_sources < 0:
            raise ConfigurationError("Store min_sources must not be negative")


@dataclass(frozen=True)
class PlannedRequest:
    """A single geocoding request of a plan.

    ``stores`` contains the ids of all stores which this request
    provides evidence for. Equivalent addresses are merged, so a single
    request may serve multiple stores.
    """

    source: str
    address: str
    address_key: str
    stores: Tuple[str, ...]
    cost: float


@dataclass(frozen=True)
class Plan:
    """The result of :func:`plan_round`.

    :param requests: The requests to perform in this round.
    :param covered: Ids of stores which will meet their requirement,
        assuming all planned requests succeed. Stores already satisfied
        by pre-existing evidence are included.
    :param uncovered: Ids of stores which cannot meet their requirement
        in this round.
    :param total_cost: Total cost of all planned requests.
    :param expected_evidence: Mapping of store id to the set of
        upstreams this plan would confirm if all requests succeed.
        Pre-existing evidence is not included.
    """

    requests: Tuple[PlannedRequest, ...]
    covered: FrozenSet[str]
    uncovered: FrozenSet[str]
    total_cost: float
    expected_evidence: Dict[str, FrozenSet[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundResult:
    """The result of :func:`execute_plan`.

    :param evidence: Mapping of store id to the set of upstreams
        successfully confirmed in this round. Pass it (merged with
        earlier rounds via :func:`merge_evidence`) as the ``evidence``
        argument of :func:`plan_round` to re-plan.
    :param locations: Mapping of ``(source, address_key)`` to the
        geocoded location for each successful request.
    :param temporary_failures: Mapping of ``(source, address_key)`` to
        the exception for requests which failed temporarily and may be
        retried in a later round.
    :param permanent_failures: Mapping of ``(source, address_key)`` to
        the exception for requests which failed permanently.
    :param not_found: ``(source, address_key)`` pairs for which the
        service responded but found no matching location. Such responses
        do not count as evidence.
    """

    evidence: Dict[str, FrozenSet[str]]
    locations: Dict[Tuple[str, str], object]
    temporary_failures: Dict[Tuple[str, str], Exception]
    permanent_failures: Dict[Tuple[str, str], Exception]
    not_found: Tuple[Tuple[str, str], ...]


def merge_evidence(*evidences):
    """Merge evidence mappings of earlier rounds into a single mapping
    of store id to a set of confirmed upstreams, suitable for the
    ``evidence`` argument of :func:`plan_round`.
    """
    merged = {}
    for evidence in evidences:
        if not evidence:
            continue
        for store_id, upstreams in evidence.items():
            merged.setdefault(store_id, set()).update(upstreams)
    return merged


def _subsets(keys, size):
    """Yield ``(subset, union)`` pairs for all subsets of exactly
    ``size`` elements of ``keys``, where ``keys`` is a list of
    ``(address_key, store_set)`` pairs. ``subset`` contains indices into
    ``keys`` and ``union`` is the union of the chosen store sets.

    Subsets containing an element which adds no new stores are skipped:
    they are dominated by the corresponding smaller, cheaper subset.
    """
    if size == 0:
        yield (), frozenset()
        return

    def recurse(start, chosen, union):
        if len(chosen) + (len(keys) - start) < size:
            return
        if len(chosen) == size:
            yield tuple(chosen), union
            return
        for index in range(start, len(keys)):
            store_set = keys[index][1]
            if store_set <= union:
                continue
            yield from recurse(index + 1, chosen + [index], union | store_set)

    yield from recurse(0, [], frozenset())


def plan_round(sources, stores, evidence=None, *, normalize=None):
    """Plan a single round of geocoding requests.

    :param sources: An iterable of :class:`Source` describing the
        available geocoding sources and their quotas for this round.
    :param stores: An iterable of :class:`Store` to geocode.
    :param evidence: An optional mapping of store id to an iterable of
        upstream names already confirmed in earlier rounds. Valid
        evidence obtained so far fully counts towards each store's
        requirement, so re-planning after temporary failures only
        requests what is still missing.
    :param normalize: An optional callable used to compute address
        equivalence keys, defaults to :func:`normalize_address`.
    :rtype: :class:`Plan`

    The plan maximizes the number of stores meeting their
    ``min_sources`` requirement; among all plans achieving that
    coverage, the one with the lowest total cost is chosen.
    """
    normalize = normalize or normalize_address
    sources = list(sources)
    stores = list(stores)
    evidence = evidence or {}

    source_names = [source.name for source in sources]
    if len(set(source_names)) != len(source_names):
        raise ConfigurationError("Source names must be unique")
    store_ids = [store.store_id for store in stores]
    if len(set(store_ids)) != len(store_ids):
        raise ConfigurationError("Store ids must be unique")

    count = len(stores)
    keys = [
        store.address_key if store.address_key is not None else normalize(store.address)
        for store in stores
    ]
    required = [store.min_sources for store in stores]
    base = [set(evidence.get(store_id, ())) for store_id in store_ids]

    usable = [source for source in sources if source.quota > 0]

    # For each usable source, the address keys it could be requested for,
    # mapped to the set of store indices the request would serve.
    source_keys = []
    source_stores = []
    for source in usable:
        by_key = {}
        for index, store in enumerate(stores):
            if store.region in source.regions:
                by_key.setdefault(keys[index], set()).add(index)
        entries = [(key, frozenset(indices)) for key, indices in by_key.items()]
        source_keys.append(entries)
        source_stores.append(frozenset().union(*(s for _, s in entries)) if entries
                             else frozenset())

    # Process sources serving the most stores first, so that good
    # incumbents are found early in the search.
    order = sorted(
        range(len(usable)),
        key=lambda i: (-len(source_stores[i]), usable[i].cost, usable[i].name),
    )
    usable = [usable[i] for i in order]
    source_keys = [source_keys[i] for i in order]
    source_stores = [source_stores[i] for i in order]

    # remaining[i][t] is the set of upstreams of sources[i:] which may
    # still serve store t. Used for the optimistic coverage bound.
    remaining = [[set() for _ in range(count)] for _ in range(len(usable) + 1)]
    for i in range(len(usable) - 1, -1, -1):
        upstream = usable[i].upstream
        for t in range(count):
            remaining[i][t] = set(remaining[i + 1][t])
            if t in source_stores[i]:
                remaining[i][t].add(upstream)

    upstreams = [set(upstreams_t) for upstreams_t in base]
    choices = [None] * len(usable)
    best = {"coverage": -1, "cost": 0.0, "choices": None}

    def visit(i, cost):
        possible = 0
        satisfied = 0
        for t in range(count):
            if len(upstreams[t]) >= required[t]:
                satisfied += 1
                possible += 1
            elif len(upstreams[t] | remaining[i][t]) >= required[t]:
                possible += 1
        if possible < best["coverage"]:
            return
        if possible == best["coverage"] and best["choices"] is not None \
                and cost >= best["cost"]:
            return
        if i == len(usable):
            if satisfied > best["coverage"] or (
                satisfied == best["coverage"]
                and (best["choices"] is None or cost < best["cost"])
            ):
                best["coverage"] = satisfied
                best["cost"] = cost
                best["choices"] = list(choices)
            return
        source = usable[i]
        entries = source_keys[i]
        max_size = min(source.quota, len(entries))
        for size in range(max_size, -1, -1):
            for subset, union in _subsets(entries, size):
                upstream = source.upstream
                changed = []
                for t in union:
                    if upstream not in upstreams[t]:
                        upstreams[t].add(upstream)
                        changed.append(t)
                choices[i] = subset
                visit(i + 1, cost + size * source.cost)
                choices[i] = None
                for t in changed:
                    upstreams[t].discard(upstream)

    visit(0, 0.0)

    # Reconstruct the chosen plan.
    final_upstreams = [set(upstreams_t) for upstreams_t in base]
    requests = []
    total_cost = 0.0
    for i, source in enumerate(usable):
        subset = best["choices"][i]
        if not subset:
            continue
        entries = source_keys[i]
        for entry_index in subset:
            key, store_set = entries[entry_index]
            served = sorted(store_set)
            representative = stores[served[0]].address
            requests.append(PlannedRequest(
                source=source.name,
                address=representative,
                address_key=key,
                stores=tuple(store_ids[t] for t in served),
                cost=source.cost,
            ))
            total_cost += source.cost
            for t in served:
                final_upstreams[t].add(source.upstream)

    requests.sort(key=lambda request: (request.source, request.address_key))

    covered = set()
    uncovered = set()
    expected = {}
    for t, store_id in enumerate(store_ids):
        if len(final_upstreams[t]) >= required[t]:
            covered.add(store_id)
        else:
            uncovered.add(store_id)
        gained = final_upstreams[t] - base[t]
        if gained:
            expected[store_id] = frozenset(gained)

    return Plan(
        requests=tuple(requests),
        covered=frozenset(covered),
        uncovered=frozenset(uncovered),
        total_cost=total_cost,
        expected_evidence=expected,
    )


def execute_plan(plan, sources, geocoders):
    """Execute a plan against geocoder-like objects.

    :param plan: A :class:`Plan` returned by :func:`plan_round`.
    :param sources: The sources the plan was computed for, either as an
        iterable of :class:`Source` or as a mapping of source name to
        :class:`Source`.
    :param geocoders: A mapping of source name to a geocoder. Each
        geocoder may either be a callable accepting an address or expose
        a ``geocode`` method (such as the geocoder classes of geopy).
    :rtype: :class:`RoundResult`

    Requests failing with a temporary error (see ``TEMPORARY_ERRORS``)
    are reported in :attr:`RoundResult.temporary_failures`; the evidence
    already obtained remains valid and can participate in re-planning
    via :func:`plan_round`'s ``evidence`` argument.
    """
    if isinstance(sources, Mapping):
        by_name = dict(sources)
    else:
        by_name = {source.name: source for source in sources}

    evidence = {}
    locations = {}
    temporary_failures = {}
    permanent_failures = {}
    not_found = []

    for request in plan.requests:
        backend = geocoders[request.source]
        geocode = getattr(backend, "geocode", None)
        if geocode is None:
            geocode = backend
        try:
            location = geocode(request.address)
        except TEMPORARY_ERRORS as error:
            temporary_failures[(request.source, request.address_key)] = error
            continue
        except GeocoderServiceError as error:
            permanent_failures[(request.source, request.address_key)] = error
            continue
        if location is None:
            not_found.append((request.source, request.address_key))
            continue
        locations[(request.source, request.address_key)] = location
        upstream = by_name[request.source].upstream
        for store_id in request.stores:
            evidence.setdefault(store_id, set()).add(upstream)

    return RoundResult(
        evidence={store_id: frozenset(upstreams)
                  for store_id, upstreams in evidence.items()},
        locations=locations,
        temporary_failures=temporary_failures,
        permanent_failures=permanent_failures,
        not_found=tuple(not_found),
    )
