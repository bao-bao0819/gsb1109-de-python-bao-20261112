"""Offline status comparison of the sync and async rate limiters.

These tests drive both :class:`.RateLimiter` and
:class:`.AsyncRateLimiter` against an in-memory fake geocoder (no network)
along one and the same reply timeline. A single source row may end up in
one of these terminal states:

* a genuine "no such address" reply -- the wrapped function returns
  ``None`` on the first try;
* an address found only after one or more retries;
* retries exhausted -- the configured ``return_value_on_exception`` is
  handed back instead of the exception;
* cancellation while waiting for a retry (async only).

The key ambiguity exercised here is that both "definitively not found"
and "could not be confirmed" surface as the Python value ``None``. The
returned value alone cannot recover the reason, so the call history
(number of attempts, quota consumed and the exception kind) has to be
kept alongside the row to tell the two histories apart.
"""

import asyncio
from unittest.mock import patch

import pytest

from geopy.exc import GeocoderQuotaExceeded, GeocoderServiceError
from geopy.extra.rate_limiter import AsyncRateLimiter, RateLimiter
from geopy.location import Location

MIN_DELAY = 1.0
ERROR_WAIT = 5.0
MAX_RETRIES = 2

# Source rows of the address backfill batch.
ROW_NOT_FOUND = "nonexistent-address"
ROW_FOUND_LATER = "retry-then-found-address"
ROW_EXHAUSTED = "always-failing-address"
ROW_CANCELLED = "cancelled-while-waiting-address"

FOUND_LOCATION = Location(
    "123 Retry Lane, Retrytown", (52.0, 13.0), {"source": "offline-fake"}
)
FALLBACK = object()


class FakeGeocoder:
    """Scripted offline geocoder.

    ``timeline`` maps a query string to the list of outcomes of its
    successive attempts. An outcome is either an exception instance
    (raised) or a plain return value (``None`` means "no such address").
    Every invocation is recorded together with a monotonically increasing
    attempt number, so the per-row call history can be asserted on.
    """

    def __init__(self, timeline):
        self._timeline = {key: list(values) for key, values in timeline.items()}
        self.calls = []
        self._clock = 0.0

    def _next_clock(self):
        # Each attempt lands in its own rate-limiter request slot.
        self._clock += MIN_DELAY
        return self._clock

    def __call__(self, query):
        index = sum(1 for call in self.calls if call[0] == query)
        self.calls.append((query, index))
        outcome = self._timeline[query][index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def is_async(request):
    return request.param


@pytest.fixture
def run(is_async):
    """Call the rate limiter transparently for sync or async mode."""
    if is_async:

        async def run(limiter, query):
            return await limiter(query)

    else:

        async def run(limiter, query):
            return limiter(query)

    return run


@pytest.fixture
def make_limiter(is_async, request):
    patchers = []

    def make_limiter(geocoder, **overrides):
        cls = AsyncRateLimiter if is_async else RateLimiter
        if is_async:
            # AsyncRateLimiter awaits the wrapped function, so wrap the
            # synchronous fake in a coroutine function.
            sync_call = geocoder.__call__

            async def func(query):
                return sync_call(query)
        else:
            func = geocoder

        kwargs = dict(
            min_delay_seconds=MIN_DELAY,
            max_retries=MAX_RETRIES,
            error_wait_seconds=ERROR_WAIT,
            return_value_on_exception=FALLBACK,
        )
        kwargs.update(overrides)
        limiter = cls(func, **kwargs)
        # Fully offline: freeze the clock and turn sleeps into no-ops.
        limiter._clock = geocoder._next_clock
        patcher = patch.object(type(limiter), "_sleep")
        patchers.append(patcher)
        mock_sleep = patcher.start()
        if is_async:

            async def noop(_seconds):  # pragma: no cover
                return None
        else:

            def noop(_seconds):  # pragma: no cover
                return None

        mock_sleep.side_effect = noop
        return limiter

    def cleanup():
        for patcher in patchers:
            patcher.stop()

    request.addfinalizer(cleanup)
    return make_limiter


def make_geocoder():
    return FakeGeocoder(
        {
            # Definitive "no such address": a single successful empty reply.
            ROW_NOT_FOUND: [None],
            # Transient errors, then the address is found.
            ROW_FOUND_LATER: [
                GeocoderServiceError("transient-1"),
                GeocoderServiceError("transient-2"),
                FOUND_LOCATION,
            ],
            # Never recovers: every attempt is a quota error.
            ROW_EXHAUSTED: [GeocoderQuotaExceeded("quota")] * (MAX_RETRIES + 1),
            # Cancelled mid-flight: the first attempt errors, the wait after
            # it is interrupted before the second attempt can run.
            ROW_CANCELLED: [
                GeocoderServiceError("transient-before-cancel"),
                FOUND_LOCATION,
            ],
        }
    )


def attempts_for(geocoder, query):
    return [index for q, index in geocoder.calls if q == query]


async def test_timeline_attempts_quota_and_submittable_rows(
    is_async, make_limiter, run
):
    geocoder = make_geocoder()
    limiter = make_limiter(geocoder)

    not_found = await run(limiter, ROW_NOT_FOUND)
    found = await run(limiter, ROW_FOUND_LATER)
    exhausted = await run(limiter, ROW_EXHAUSTED)

    # Attempt counts per source row.
    assert attempts_for(geocoder, ROW_NOT_FOUND) == [0]
    assert attempts_for(geocoder, ROW_FOUND_LATER) == [0, 1, 2]
    assert attempts_for(geocoder, ROW_EXHAUSTED) == [0, 1, 2]

    # Every attempt consumes one request slot, hence quota:
    # 1 + 3 + 3 = 7 wrapped calls for the three finished rows.
    assert len(geocoder.calls) == 7

    # Returned values along the same timeline.
    assert not_found is None
    assert found is FOUND_LOCATION
    assert exhausted is FALLBACK

    # Only rows carrying a concrete Location are submittable. The exhausted
    # row yields the fallback instead and must be held back.
    submittable = [
        query
        for query, value in (
            (ROW_NOT_FOUND, not_found),
            (ROW_FOUND_LATER, found),
            (ROW_EXHAUSTED, exhausted),
        )
        if isinstance(value, Location)
    ]
    assert submittable == [ROW_FOUND_LATER]


async def test_none_is_indistinguishable_without_call_history(
    is_async, make_limiter, run
):
    """Two rows end up as Python ``None`` for different reasons.

    ``ROW_NOT_FOUND`` is confirmed missing, whereas a row whose retries are
    exhausted with the default ``return_value_on_exception=None`` cannot be
    confirmed. The values compare equal; only the call history separates
    "checked, no such address" from "not yet confirmable".
    """
    geocoder = make_geocoder()
    limiter = make_limiter(geocoder, return_value_on_exception=None)

    not_found = await run(limiter, ROW_NOT_FOUND)
    unconfirmed = await run(limiter, ROW_EXHAUSTED)

    assert not_found is None
    assert unconfirmed is None
    assert not_found == unconfirmed

    history_not_found = attempts_for(geocoder, ROW_NOT_FOUND)
    history_unconfirmed = attempts_for(geocoder, ROW_EXHAUSTED)

    # The public return values do not recover the reason; the histories do.
    assert history_not_found == [0]
    assert history_unconfirmed == [0, 1, 2]
    assert history_not_found != history_unconfirmed

    # Downstream classification keyed on the source row history.
    def classify(query):
        return (
            "confirmed-not-found"
            if attempts_for(geocoder, query) == [0]
            else "unconfirmed"
        )

    assert classify(ROW_NOT_FOUND) == "confirmed-not-found"
    assert classify(ROW_EXHAUSTED) == "unconfirmed"


async def test_cancelled_retry_leaves_row_unconfirmed(is_async, make_limiter):
    if not is_async:
        pytest.skip("Cancellation only applies to AsyncRateLimiter")

    geocoder = make_geocoder()
    limiter = make_limiter(geocoder)

    async def blocking_sleep(_seconds):
        # Park inside the error backoff until the task is cancelled.
        await asyncio.Event().wait()

    with patch.object(AsyncRateLimiter, "_sleep", side_effect=blocking_sleep):
        task = asyncio.ensure_future(limiter(ROW_CANCELLED))
        # Let the first (failing) attempt and the backoff run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # Only the first attempt happened; the retry never went out.
    assert attempts_for(geocoder, ROW_CANCELLED) == [0]
    assert len(geocoder.calls) == 1


async def test_sync_and_async_share_the_terminal_state_table(
    is_async, make_limiter, run
):
    """The same reply timeline yields identical terminal states in both modes."""
    geocoder = make_geocoder()
    limiter = make_limiter(geocoder)

    results = {
        ROW_NOT_FOUND: await run(limiter, ROW_NOT_FOUND),
        ROW_FOUND_LATER: await run(limiter, ROW_FOUND_LATER),
        ROW_EXHAUSTED: await run(limiter, ROW_EXHAUSTED),
    }

    # Sync and async must agree on both values and per-row attempt counts.
    assert results == {
        ROW_NOT_FOUND: None,
        ROW_FOUND_LATER: FOUND_LOCATION,
        ROW_EXHAUSTED: FALLBACK,
    }
    assert attempts_for(geocoder, ROW_NOT_FOUND) == [0]
    assert attempts_for(geocoder, ROW_FOUND_LATER) == [0, 1, 2]
    assert attempts_for(geocoder, ROW_EXHAUSTED) == [0, 1, 2]


async def test_no_swallow_re_raises_after_exhaustion(is_async, make_limiter, run):
    geocoder = make_geocoder()
    limiter = make_limiter(geocoder, swallow_exceptions=False)

    with pytest.raises(GeocoderQuotaExceeded):
        await run(limiter, ROW_EXHAUSTED)

    # max_retries + 1 attempts were still consumed before propagating.
    assert attempts_for(geocoder, ROW_EXHAUSTED) == [0, 1, 2]
