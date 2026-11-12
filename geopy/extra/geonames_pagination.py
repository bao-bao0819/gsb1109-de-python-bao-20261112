"""Offline-replayable, checkpointable pagination for the GeoNames web service.

GeoNames' search endpoint (``/searchJSON``) pages through the
``startRow``/``maxRows`` parameters but gives no snapshot guarantee:
between two requests candidates may be inserted into or removed from the
result set, and the ``name`` field is not unique. This module therefore

- reconciles pages by the source's stable identifier (the ``geonameId``
  field) instead of by address text,
- distinguishes *"these IDs have been observed"* from *"the whole query
  range has been read"* -- completeness requires both a stable ordering
  declared by the caller and an end-of-data evidence observed in a
  response,
- persists checkpoints which, on resume, re-read the first page and
  detect that the head of the result list has moved. The offsets that
  must be re-read are reported as pending ranges; ranges the live source
  cannot certify (for instance because they were not re-fetched yet) are
  kept open.

All HTTP traffic goes through a regular geopy adapter.
:class:`ReplayAdapter` makes a captured transcript of responses
replayable offline, so the very same pagination code runs in tests and
disaster-recovery jobs without touching the network.

Example::

    from geopy.geocoders import GeoNames
    from geopy.extra.geonames_pagination import GeoNamesPaginator

    geocoder = GeoNames(username="myuser")
    paginator = GeoNamesPaginator(
        geocoder,
        query_params={"country": "DE", "featureClass": "P"},
        page_size=100,
        ordering="relevance",
    )
    result = paginator.fetch_all()
    # result.is_complete is only True when the declared ordering was
    # stable, the terminal page was observed and no pending range remains.
    paginator.save_checkpoint("geonames-de.json")

    # Later, possibly in another process:
    paginator = GeoNamesPaginator.from_checkpoint("geonames-de.json", geocoder)
    result = paginator.fetch_all()

.. versionadded:: 2.6
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse

from geopy.adapters import BaseAsyncAdapter, BaseSyncAdapter
from geopy.exc import ConfigurationError, GeocoderServiceError
from geopy.geocoders.base import DEFAULT_SENTINEL
from geopy.location import Location
from geopy.util import logger

__all__ = (
    "Checkpoint",
    "GeoNamesPaginator",
    "PaginationResult",
    "PendingRange",
    "ReplayAdapter",
    "ReplayError",
)

# GeoNames' own per-request cap for maxRows.
_MAX_ROWS_LIMIT = 1000

# Values accepted by the GeoNames ``orderBy`` parameter. Stability is
# never inferred from the wire response: an ordering is only treated as
# stable once the caller explicitly declares it.
_KNOWN_ORDERINGS = frozenset(
    (
        "population",
        "elevation",
        "name",
        "relevance",
        "distance",
    )
)


class ReplayError(GeocoderServiceError):
    """Raised when a recorded response cannot be replayed.

    This happens when a request made in ``replay`` mode was not captured
    during the preceding ``record`` run, or when a recorded payload is
    malformed.
    """


class ReplayAdapter(BaseSyncAdapter):
    """A sync adapter serving pre-recorded JSON responses.

    Two modes are supported:

    - ``record``: requests are forwarded to a wrapped real adapter and
      every parsed JSON response is stored under a key derived from the
      request's query parameters;
    - ``replay``: no network access happens -- responses are looked up by
      the same key. A request that wasn't recorded raises
      :class:`ReplayError`.

    The transcript returned by :meth:`transcript` only contains parsed
    JSON, so it can be persisted as-is and replayed later offline.

    :param transcript: mapping of normalized query strings to parsed JSON
        responses; when ``None`` an empty transcript is created.
    :param wrapped_factory: factory accepting ``proxies`` and
        ``ssl_context`` keyword arguments, producing the adapter used for
        real requests in ``record`` mode.
    :param str mode: either ``"record"`` or ``"replay"``.
    """

    def __init__(
        self,
        *,
        transcript=None,
        wrapped_factory=None,
        mode="replay",
        proxies=None,
        ssl_context=None,
    ):
        super().__init__(proxies=proxies or {}, ssl_context=ssl_context)
        if mode not in ("record", "replay"):
            raise ValueError("mode must be 'record' or 'replay'")
        self.mode = mode
        self._transcript = dict(transcript or {})
        self._request_log = []  # type: List[str]
        if mode == "record":
            if wrapped_factory is None:
                raise ValueError(
                    "wrapped_factory is required when recording responses"
                )
            self._wrapped = wrapped_factory(
                proxies=proxies, ssl_context=ssl_context
            )
        else:
            self._wrapped = None

    @staticmethod
    def _key_for_url(url):
        parsed = urlparse(url)
        # Sorting the pairs makes the key independent of parameter order,
        # while parse_qsl keeps repeated keys (e.g. several ``country``).
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        return urlencode(sorted(pairs))

    def transcript(self):
        """Return a copy of the recorded request/response transcript."""
        return json.loads(json.dumps(self._transcript))

    @property
    def request_log(self):
        """Normalized query keys in the order the adapter was called."""
        return list(self._request_log)

    def get_json(self, url, *, timeout, headers):
        key = self._key_for_url(url)
        self._request_log.append(key)
        if self.mode == "replay":
            try:
                payload = self._transcript[key]
            except KeyError:
                raise ReplayError(
                    "No recorded response for request: %s" % url
                )
            return json.loads(json.dumps(payload))
        response = self._wrapped.get_json(
            url, timeout=timeout, headers=headers
        )
        self._transcript[key] = response
        return json.loads(json.dumps(response))

    def get_text(self, url, *, timeout, headers):  # pragma: no cover
        raise NotImplementedError(
            "ReplayAdapter only supports JSON responses"
        )


@dataclass
class PendingRange:
    """A half-open ``[start, end)`` interval of ``startRow`` offsets.

    A pending range means the offsets inside it have not been certified
    against the current source contents and must be re-read. Ranges are
    created unauthenticated; successfully re-fetching every offset in the
    range authenticates and removes it.
    """

    start: int
    end: int

    def to_dict(self):
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data):
        return cls(start=data["start"], end=data["end"])


@dataclass
class Checkpoint:
    """A serializable resume point for one parameterized query.

    A checkpoint never claims completeness on its own: it only records
    which offsets were requested, which stable IDs were observed and
    which ranges still need (re-)reading. Whether the query range is
    complete is decided by :class:`GeoNamesPaginator` from the
    end-of-data evidence.
    """

    query_params: Dict[str, Any]
    page_size: int
    ordering: Optional[str]
    next_start: int
    observed_ids: List[str] = field(default_factory=list)
    seen_pages: List[Dict[str, Any]] = field(default_factory=list)
    pending_ranges: List[PendingRange] = field(default_factory=list)
    end_observed: bool = False

    def to_dict(self):
        return {
            "query_params": self.query_params,
            "page_size": self.page_size,
            "ordering": self.ordering,
            "next_start": self.next_start,
            "observed_ids": list(self.observed_ids),
            "seen_pages": [
                {"start": page["start"], "ids": list(page["ids"])}
                for page in self.seen_pages
            ],
            "pending_ranges": [rng.to_dict() for rng in self.pending_ranges],
            "end_observed": self.end_observed,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            query_params=dict(data["query_params"]),
            page_size=data["page_size"],
            ordering=data.get("ordering"),
            next_start=data["next_start"],
            observed_ids=list(data.get("observed_ids", ())),
            seen_pages=[
                {
                    "start": page["start"],
                    "ids": list(page["ids"]),
                }
                for page in data.get("seen_pages", ())
            ],
            pending_ranges=[
                PendingRange.from_dict(item)
                for item in data.get("pending_ranges", ())
            ],
            end_observed=data.get("end_observed", False),
        )

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fp:
            return cls.from_dict(json.load(fp))


@dataclass
class PaginationResult:
    """Outcome of a (possibly resumed) pagination run.

    :ivar locations: locations fetched during this run, in source order
        and de-duplicated by ``geonameId``.
    :ivar received_ids: stable IDs observed across all runs (including
        checkpoint history). This only proves that these IDs were
        received -- never that the query range is complete.
    :ivar is_complete: ``True`` only when the caller declared a stable
        ordering, end-of-data was observed and no pending range remains.
    :ivar checkpoint: the checkpoint reflecting the state after this run.
    :ivar pending_ranges: offsets still awaiting certification; non-empty
        means the source could not (yet) authenticate the whole range.
    """

    locations: List[Location]
    received_ids: List[str]
    is_complete: bool
    checkpoint: Checkpoint
    pending_ranges: List[PendingRange]

    @property
    def has_pending_ranges(self):
        return bool(self.pending_ranges)


def _place_id(place):
    geoname_id = place.get("geonameId")
    if geoname_id is None:
        raise GeocoderServiceError(
            "GeoNames response contains a place without a geonameId; "
            "reconciliation by stable identifier is impossible: %r" % (place,)
        )
    return str(geoname_id)


def _parse_location(place):
    latitude = place.get("lat")
    longitude = place.get("lng")
    if latitude and longitude:
        latitude = float(latitude)
        longitude = float(longitude)
    else:
        return None
    placename = place.get("name")
    state = place.get("adminName1")
    country = place.get("countryName")
    address = ", ".join(
        part for part in (placename, state, country) if part
    )
    return Location(address, (latitude, longitude), place)


class GeoNamesPaginator:
    """Page through a GeoNames query with ID-based reconciliation.

    :param geocoder: a :class:`geopy.geocoders.GeoNames` instance.
    :param query_params: extra parameters for the ``/searchJSON`` request
        (e.g. ``{"country": "DE"}``). List or tuple values produce
        repeated query parameters, matching GeoNames conventions.
    :param int page_size: value of the ``maxRows`` parameter, clamped to
        the service's limit of 1000.
    :param str ordering: stable ordering declared by the caller, sent as
        the ``orderBy`` parameter. ``None`` declares no stability, in
        which case a run can never be marked complete.
    :param checkpoint: an optional :class:`Checkpoint` to resume from.
    :param int max_pages: safety cap on the number of requests per run.
    """

    def __init__(
        self,
        geocoder,
        *,
        query_params=None,
        page_size=100,
        ordering=None,
        checkpoint=None,
        max_pages=None,
    ):
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if ordering is not None and ordering not in _KNOWN_ORDERINGS:
            raise ValueError(
                "Unknown ordering %r; known orderings: %s"
                % (ordering, sorted(_KNOWN_ORDERINGS))
            )
        if isinstance(geocoder.adapter, BaseAsyncAdapter):
            raise ConfigurationError(
                "GeoNamesPaginator drives synchronous paging and requires "
                "a sync adapter (such as URLLibAdapter or ReplayAdapter); "
                "an async adapter was provided."
            )
        self.geocoder = geocoder
        self.query_params = dict(query_params or {})
        self.page_size = min(page_size, _MAX_ROWS_LIMIT)
        self.ordering = ordering
        self.max_pages = max_pages

        if checkpoint is not None:
            self._validate_checkpoint(checkpoint)
            self.checkpoint = checkpoint
        else:
            self.checkpoint = Checkpoint(
                query_params=dict(self.query_params),
                page_size=self.page_size,
                ordering=self.ordering,
                next_start=0,
            )

    def _validate_checkpoint(self, checkpoint):
        if checkpoint.page_size != self.page_size:
            raise ValueError(
                "Checkpoint page_size %r does not match the paginator's %r"
                % (checkpoint.page_size, self.page_size)
            )
        if checkpoint.ordering != self.ordering:
            raise ValueError(
                "Checkpoint ordering %r does not match the paginator's %r"
                % (checkpoint.ordering, self.ordering)
            )
        if _freeze_params(checkpoint.query_params) != _freeze_params(
            self.query_params
        ):
            raise ValueError(
                "Checkpoint query_params %r do not match %r"
                % (checkpoint.query_params, self.query_params)
            )

    def _build_url(self, start):
        params = [("username", self.geocoder.username)]
        for key, value in self.query_params.items():
            if isinstance(value, (list, tuple)):
                for item in value:
                    params.append((key, item))
            else:
                params.append((key, value))
        if self.ordering is not None:
            params.append(("orderBy", self.ordering))
        params.append(("maxRows", self.page_size))
        params.append(("startRow", start))
        return "?".join((self.geocoder.api, urlencode(params)))

    def _request_page(self, start, timeout):
        url = self._build_url(start)
        logger.debug("%s pagination: %s", self.__class__.__name__, url)
        return self.geocoder._call_geocoder(url, lambda doc: doc, timeout=timeout)

    def fetch_all(self, *, timeout=DEFAULT_SENTINEL):
        """Read pages until end-of-data or the safety cap is reached.

        On resume the first page is always re-read and compared with the
        recorded one, so movement at the head of the result set is
        detected before any new data is trusted.
        """
        checkpoint = self.checkpoint
        locations = []  # type: List[Location]
        emitted = set()  # type: set
        observed = set(checkpoint.observed_ids)
        covered = []  # type: List  # list of (start, end) intervals
        pages_fetched = 0
        pending = list(checkpoint.pending_ranges)

        start, reached_end = self._authenticate_head(
            timeout=timeout,
            locations=locations,
            emitted=emitted,
            observed=observed,
            covered=covered,
            pending=pending,
        )
        pages_fetched += 1 if checkpoint.seen_pages else 0

        while not reached_end and (
            self.max_pages is None or pages_fetched < self.max_pages
        ):
            doc = self._request_page(start, timeout)
            page_ids, row_count = self._ingest(
                doc,
                start,
                observed=observed,
                emitted=emitted,
                locations=locations,
                covered=covered,
            )
            pages_fetched += 1
            checkpoint.seen_pages.append(
                {"start": start, "ids": page_ids}
            )
            start += row_count
            checkpoint.next_start = start
            if row_count < self.page_size:
                # A short (including empty) page is the end-of-data
                # evidence, valid only under a caller-declared ordering.
                checkpoint.end_observed = True
                reached_end = True
                break

        checkpoint.observed_ids = sorted(observed)
        checkpoint.pending_ranges = self._reconcile_pending(
            pending, covered, checkpoint
        )

        return PaginationResult(
            locations=locations,
            received_ids=list(checkpoint.observed_ids),
            is_complete=self._evaluate_completeness(checkpoint),
            checkpoint=checkpoint,
            pending_ranges=list(checkpoint.pending_ranges),
        )

    def _authenticate_head(
        self,
        *,
        timeout,
        locations,
        emitted,
        observed,
        covered,
        pending,
    ):
        """Re-read page zero on resume and detect head-of-list movement.

        Returns a ``(start_row, reached_end)`` tuple: the offset to
        continue fetching from and whether the re-read head page already
        constitutes the end of the current result set.
        """
        checkpoint = self.checkpoint
        if not checkpoint.seen_pages:
            return 0, False

        doc = self._request_page(0, timeout)
        page_ids, row_count = self._ingest(
            doc,
            0,
            observed=observed,
            emitted=emitted,
            locations=locations,
            covered=covered,
        )
        recorded_first = checkpoint.seen_pages[0]["ids"]
        divergence = _first_divergence(recorded_first, page_ids)
        head_record = {"start": 0, "ids": page_ids}
        reached_end = row_count < self.page_size

        if divergence is None:
            # The head of the list still matches: old page records and
            # the previous end evidence stay valid.
            checkpoint.seen_pages[0] = head_record
            if reached_end:
                checkpoint.end_observed = True
            return checkpoint.next_start, reached_end

        logger.warning(
            "GeoNames pagination: first page moved at offset %s; "
            "the prefix must be re-read",
            divergence,
        )
        pending.append(
            PendingRange(
                start=divergence,
                end=max(checkpoint.next_start, divergence + 1),
            )
        )
        # The old layout and its end evidence describe a different
        # ordering now; only the freshly verified head record survives.
        checkpoint.seen_pages = [head_record]
        checkpoint.end_observed = reached_end
        # The furthest offset actually fetched in the current layout.
        # The main loop advances it further; when the head page was
        # short it equals the new dataset length, which allows pending
        # ranges to be clipped against it.
        checkpoint.next_start = row_count

        if divergence == 0:
            # The head page itself is the first re-read page and it has
            # already been ingested above.
            return row_count, reached_end
        return divergence, reached_end

    def _ingest(
        self,
        doc,
        start,
        *,
        observed,
        emitted,
        locations,
        covered,
    ):
        self.geocoder._raise_for_error(doc)
        places = doc.get("geonames", [])
        page_ids = [_place_id(place) for place in places]
        covered.append((start, start + len(places)))
        for place, geoname_id in zip(places, page_ids):
            observed.add(geoname_id)
            if geoname_id in emitted:
                # Re-reading after an insertion may surface the same
                # stable ID again at a shifted offset.
                continue
            emitted.add(geoname_id)
            location = _parse_location(place)
            if location is not None:
                locations.append(location)
        return page_ids, len(places)

    def _reconcile_pending(self, pending, covered, checkpoint):
        """Drop the parts of pending ranges certified by this run.

        Offsets covered by a page fetched in this run are authenticated.
        When end-of-data was observed, the dataset's current length is
        known, so ranges pointing past it are clipped away. Whatever
        remains is kept pending: the source could not certify it.
        """
        merged_covered = _merge_intervals(covered)
        remaining = []  # type: List[PendingRange]
        for rng in pending:
            end = rng.end
            if checkpoint.end_observed:
                end = min(end, checkpoint.next_start)
            if end <= rng.start:
                continue
            for uncovered_start, uncovered_end in _uncovered_parts(
                rng.start, end, merged_covered
            ):
                remaining.append(
                    PendingRange(
                        start=uncovered_start,
                        end=uncovered_end,
                    )
                )
        return _merge_ranges(remaining)

    def _evaluate_completeness(self, checkpoint):
        if self.ordering is None:
            logger.info(
                "GeoNames pagination: no stable ordering declared; "
                "the run cannot be considered complete"
            )
            return False
        if not checkpoint.end_observed:
            logger.info(
                "GeoNames pagination: no end-of-data evidence observed; "
                "the run cannot be considered complete"
            )
            return False
        if checkpoint.pending_ranges:
            logger.info(
                "GeoNames pagination: unauthenticated ranges remain; "
                "the run cannot be considered complete"
            )
            return False
        return True

    def save_checkpoint(self, path):
        """Persist the current checkpoint to ``path``."""
        self.checkpoint.save(path)

    @classmethod
    def from_checkpoint(cls, path, geocoder, *, max_pages=None):
        """Reconstruct a paginator from a checkpoint file."""
        checkpoint = Checkpoint.load(path)
        return cls(
            geocoder,
            query_params=checkpoint.query_params,
            page_size=checkpoint.page_size,
            ordering=checkpoint.ordering,
            checkpoint=checkpoint,
            max_pages=max_pages,
        )


def _first_divergence(recorded, current):
    """Return the first offset where two ordered ID sequences differ.

    ``None`` means the recorded sequence is still a prefix of the current
    one (i.e. the head of the list has not moved).
    """
    for offset in range(min(len(recorded), len(current))):
        if recorded[offset] != current[offset]:
            return offset
    if len(recorded) > len(current):
        return len(current)
    return None


def _merge_intervals(intervals):
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _merge_ranges(ranges):
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda rng: (rng.start, rng.end))
    merged = [
        PendingRange(start=ordered[0].start, end=ordered[0].end)
    ]
    for rng in ordered[1:]:
        last = merged[-1]
        if rng.start <= last.end:
            last.end = max(last.end, rng.end)
        else:
            merged.append(PendingRange(start=rng.start, end=rng.end))
    return merged


def _uncovered_parts(start, end, covered):
    """Yield the sub-intervals of ``[start, end)`` not in ``covered``."""
    cursor = start
    for covered_start, covered_end in covered:
        if covered_end <= cursor:
            continue
        if covered_start >= end:
            break
        if covered_start > cursor:
            yield cursor, min(covered_start, end)
        cursor = max(cursor, covered_end)
        if cursor >= end:
            break
    if cursor < end:
        yield cursor, end


def _freeze_params(params):
    frozen = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, (list, tuple)):
            frozen.append((key, tuple(value)))
        else:
            frozen.append((key, value))
    return tuple(frozen)
