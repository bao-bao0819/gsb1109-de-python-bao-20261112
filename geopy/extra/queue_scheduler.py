""":class:`.QueueScheduler` allows sharing a single geocoder quota between
multiple independent request queues (for example, a nightly address
backfill and an interactive single-address lookup) while keeping the
scheduling decisions fully deterministic and testable.

The design goals are:

* The geocoder quota is modeled as an *interval budget* keyed by a
  source account: queues declaring the same source key share the same
  budget, so at most one request per ``min_interval_seconds`` is sent
  for that account, retries included.
* A long (backfill) queue is guaranteed an explicit minimum share of
  the dispatch slots, so an always-full interactive queue can never
  completely starve it.
* Requests may carry a deadline. A request that expires while queued
  is dropped without ever touching the interval budget, i.e. it cannot
  occupy the next sendable point in time. Requests that have already
  been sent (including their retries) have already deducted the budget
  and keep counting towards it.
* Time is read from a pluggable clock. :class:`.ManualClock` is an
  advanceable clock which makes the interleaving of multiple queues
  fully reproducible in tests -- no real map service and no real
  sleeping are required.

A typical setup::

    from geopy.extra.queue_scheduler import (
        ManualClock, QueueScheduler, SystemClock,
    )

    clock = SystemClock()
    scheduler = QueueScheduler(
        lambda request: geolocator.geocode(request.payload),
        min_interval_seconds=1.0,
        min_share=0.2,
        priority_queue="realtime",
        clock=clock,
    )

    # Nightly backfill and interactive lookups use the same account,
    # therefore they share the same interval budget.
    scheduler.enqueue(
        "backfill", "paris", source="account-a",
        deadline=clock.now() + 3600,
    )
    scheduler.enqueue("realtime", "berlin", source="account-a")

    while True:
        scheduler.dispatch_due()
        event_at = scheduler.next_event_at()
        if event_at is None:
            break
        # With a SystemClock, sleep until ``event_at``; with a
        # ManualClock, simply call ``clock.advance_to(event_at)``.
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from timeit import default_timer
from typing import Any, Deque, Dict, List, Optional

from geopy.exc import GeocoderServiceError

__all__ = (
    "GeocoderRequest",
    "ManualClock",
    "QueueScheduler",
    "RequestStatus",
    "SharedIntervalBudget",
    "SystemClock",
)


class RequestStatus(Enum):
    """Lifecycle states of a :class:`.GeocoderRequest`."""

    QUEUED = "queued"
    DISPATCHED = "dispatched"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class GeocoderRequest:
    """A single geocoding job moving through the queues.

    :param int request_id: Unique id assigned by the scheduler.
    :param str queue: Queue name the request belongs to.
    :param str source: Source account key. Queues using the same
        key share the same interval budget.
    :param payload: Opaque value handed over to the scheduler handler.
    :param float enqueued_at: Clock time at which the request was enqueued.
    :param deadline: Absolute clock time after which the request must
        not be sent anymore. ``None`` means no deadline.
    :param float not_before: Earliest absolute clock time at which
        the request may be dispatched (used for retry back-off).
    """

    request_id: int
    queue: str
    source: str
    payload: Any
    enqueued_at: float
    deadline: Optional[float] = None
    not_before: float = 0.0
    attempts: int = 0
    status: RequestStatus = RequestStatus.QUEUED
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[BaseException] = None
    history: List[float] = field(default_factory=list)

    @property
    def is_final(self):
        return self.status in (
            RequestStatus.SUCCEEDED,
            RequestStatus.FAILED,
            RequestStatus.EXPIRED,
        )


class SystemClock:
    """Clock backed by :func:`timeit.default_timer` for production use."""

    def now(self):
        return default_timer()

    def advance(self, seconds):  # pragma: no cover - real time cannot be moved
        raise RuntimeError(
            "SystemClock cannot be advanced; use ManualClock in tests"
        )

    def advance_to(self, when):  # pragma: no cover - see ``advance``
        raise RuntimeError(
            "SystemClock cannot be advanced; use ManualClock in tests"
        )


class ManualClock:
    """Advanceable clock for deterministic, sleep-free testing.

    Time only moves when :meth:`advance` or :meth:`advance_to` is
    called, so the exact dispatch order of multiple queues can be
    asserted against fixed timestamps.
    """

    def __init__(self, start=0.0):
        self._now = float(start)

    def now(self):
        return self._now

    def advance(self, seconds):
        if seconds < 0:
            raise ValueError("Cannot move the clock backwards")
        self._now += float(seconds)
        return self._now

    def advance_to(self, when):
        if when < self._now:
            raise ValueError("Cannot move the clock backwards")
        self._now = float(when)
        return self._now


class SharedIntervalBudget:
    """An interval-based quota shared by every queue of one source account.

    A budget becomes available again ``interval_seconds`` after the
    previous deduction. Every deduction -- including deductions made
    for retries -- increments :attr:`deducted_count`, which is the
    number of requests actually charged to the upstream geocoder.
    """

    def __init__(self, interval_seconds, clock):
        assert interval_seconds >= 0
        self.interval_seconds = float(interval_seconds)
        self._clock = clock
        self._next_available_at = None
        self.deducted_count = 0

    @property
    def next_available_at(self):
        """Absolute time of the next sendable point, or ``None`` if unused."""
        return self._next_available_at

    def is_available(self, now=None):
        if now is None:
            now = self._clock.now()
        return self._next_available_at is None or self._next_available_at <= now

    def deduct(self, now=None):
        """Charge one request slot and return the next available time."""
        if now is None:
            now = self._clock.now()
        if self._next_available_at is not None and now < self._next_available_at:
            raise RuntimeError(
                "Cannot deduct a budget before it becomes available"
            )
        self._next_available_at = now + self.interval_seconds
        self.deducted_count += 1
        return self._next_available_at


class _QueueCounters:
    def __init__(self, source):
        self.source = source
        self.queued = 0
        self.dispatched = 0
        self.succeeded = 0
        self.failed = 0
        self.expired = 0

    def as_dict(self):
        return {
            "source": self.source,
            "queued": self.queued,
            "dispatched": self.dispatched,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "expired": self.expired,
        }


class QueueScheduler:
    """Multi-queue scheduler sharing interval budgets per source account.

    Each unique ``source`` key gets its own :class:`.SharedIntervalBudget`.
    Queues that enqueue requests under the same source key therefore
    share one geocoder quota, while different sources get independent
    budgets. A queue and its source may have the same name (the source
    defaults to the queue name), which makes single-account setups
    straightforward.

    Among the queues that currently have a dispatchable request, the
    next one is chosen with weighted fairness (smallest
    ``dispatched_count / weight``). The priority queue -- typically the
    interactive one -- receives the remainder of the capacity while
    every other queue is guaranteed at least ``min_share`` of the
    dispatch slots, which protects a long backfill queue from starving.

    The scheduler never sleeps itself. :meth:`dispatch_due` sends
    everything that is sendable at the current clock time and returns;
    :meth:`next_event_at` tells when it is worth calling again.
    """

    def __init__(
        self,
        handler,
        *,
        min_interval_seconds=1.0,
        min_share=0.2,
        priority_queue=None,
        clock=None,
        retry_attempts=0,
        retry_error_wait_seconds=None,
        retry_on=(GeocoderServiceError,),
    ):
        """
        :param callable handler:
            Function called with a :class:`.GeocoderRequest` for every
            dispatch. Its return value is stored on the request.

        :param float min_interval_seconds:
            Minimum spacing between two dispatches charged to the same
            source account. Retries are spaced by the same budget.

        :param float min_share:
            Minimum fraction of dispatch slots guaranteed to each
            non-priority queue while the priority queue is active.
            Must be strictly between 0 and 1.

        :param str priority_queue:
            Name of the latency-sensitive (interactive) queue. It may
            consume up to ``1 - min_share`` of the slots. When ``None``
            all active queues share the slots equally.

        :param clock:
            Clock instance with a ``now()`` method. Defaults to
            :class:`.SystemClock`; pass a :class:`.ManualClock` in
            tests to make scheduling deterministic.

        :param int retry_attempts:
            Number of retries after a failed dispatch. ``0`` disables
            retries. Every retry deducts the source budget again.

        :param float retry_error_wait_seconds:
            Back-off before a retry. Defaults to
            ``min_interval_seconds`` and must not be smaller than it.

        :param tuple retry_on:
            Exception types that trigger a retry. Other exceptions
            fail the request immediately after its single deduction.
        """
        assert min_interval_seconds >= 0
        assert 0.0 < min_share < 1.0
        assert retry_attempts >= 0
        if retry_error_wait_seconds is None:
            retry_error_wait_seconds = min_interval_seconds
        assert retry_error_wait_seconds >= min_interval_seconds

        self._handler = handler
        self.min_interval_seconds = float(min_interval_seconds)
        self.min_share = float(min_share)
        self.priority_queue = priority_queue
        self.clock = clock if clock is not None else SystemClock()
        self.retry_attempts = retry_attempts
        self.retry_error_wait_seconds = float(retry_error_wait_seconds)
        self._retry_on = tuple(retry_on)

        self._lock = threading.Lock()
        self._queues = {}  # type: Dict[str, Deque[GeocoderRequest]]
        self._queue_source = {}  # type: Dict[str, str]
        self._budgets = {}  # type: Dict[str, SharedIntervalBudget]
        self._counters = {}  # type: Dict[str, _QueueCounters]
        self._requests = {}  # type: Dict[int, GeocoderRequest]
        self._next_request_id = 0

    def enqueue(self, queue, payload, *, source=None, deadline=None):
        """Add a request to ``queue``.

        :param str queue: Queue name, e.g. ``"backfill"``.
        :param payload: Opaque value passed to the handler.
        :param str source: Source account key. Defaults to ``queue``.
            Queues using the same source share one interval budget.
        :param deadline: Absolute clock-time deadline. A request whose
            deadline has already passed is recorded as expired without
            entering the queue or touching the budget.
        :rtype: GeocoderRequest
        """
        if source is None:
            source = queue
        now = self.clock.now()
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._ensure_queue(queue, source)
            request = GeocoderRequest(
                request_id=request_id,
                queue=queue,
                source=source,
                payload=payload,
                enqueued_at=now,
                deadline=deadline,
                not_before=now,
            )
            self._requests[request_id] = request
            if deadline is not None and now > deadline:
                request.status = RequestStatus.EXPIRED
                request.completed_at = now
                self._counters[queue].expired += 1
                return request
            self._queues[queue].append(request)
            self._counters[queue].queued += 1
            return request

    def enqueue_many(self, queue, payloads, *, source=None, deadline=None):
        """Enqueue an iterable of payloads for one queue at once."""
        return [
            self.enqueue(queue, payload, source=source, deadline=deadline)
            for payload in payloads
        ]

    def get_request(self, request_id):
        """Look up a request by the id returned by :meth:`enqueue`."""
        return self._requests[request_id]

    def next_event_at(self):
        """Earliest clock time at which another dispatch may happen.

        ``None`` is returned when no live queued requests remain.
        Expired requests are purged as a side effect.
        """
        with self._lock:
            now = self.clock.now()
            event_at = None
            for queue, pending in self._queues.items():
                self._purge_expired(queue, now)
                if not pending:
                    continue
                head = pending[0]
                source = self._queue_source[queue]
                ready_at = max(
                    head.not_before,
                    self._budgets[source].next_available_at or now,
                )
                event_at = ready_at if event_at is None else min(event_at, ready_at)
            return event_at

    def dispatch_due(self):
        """Dispatch every request sendable at the current clock time.

        Returns the requests dispatched by this call (in dispatch
        order), including requests that were re-dispatched as retries.
        Expired head requests are dropped without budget deductions.
        """
        dispatched = []
        while True:
            with self._lock:
                now = self.clock.now()
                queue = self._pick_queue(now)
                if queue is None:
                    return dispatched
                request = self._queues[queue][0]
                source = self._queue_source[queue]
                budget = self._budgets[source]
                budget.deduct(now)
                request.attempts += 1
                request.history.append(now)
                request.status = RequestStatus.DISPATCHED
                self._counters[queue].dispatched += 1

            try:
                result = self._handler(request)
            except Exception as error:
                self._handle_failure(request, error)
            else:
                self._handle_success(request, result)
            dispatched.append(request)

    def run_until_idle(self, *, max_iterations=10000):
        """Advance a :class:`.ManualClock` and dispatch until all queues drain.

        Raises :class:`RuntimeError` if more than ``max_iterations``
        rounds were needed, which would indicate a scheduling bug.
        Returns the full list of dispatched requests in order.
        """
        all_dispatched = []
        iterations = 0
        while True:
            all_dispatched.extend(self.dispatch_due())
            event_at = self.next_event_at()
            if event_at is None:
                return all_dispatched
            iterations += 1
            if iterations > max_iterations:
                raise RuntimeError("Scheduler did not become idle")
            now = self.clock.now()
            if event_at > now:
                self.clock.advance_to(event_at)

    def stats(self):
        """Return per-queue counters and the per-source budget state."""
        with self._lock:
            now = self.clock.now()
            for queue in list(self._queues):
                self._purge_expired(queue, now)
            return {
                "queues": {
                    queue: self._counters[queue].as_dict()
                    for queue in sorted(self._counters)
                },
                "budgets": {
                    source: {
                        "deducted": budget.deducted_count,
                        "next_available_at": budget.next_available_at,
                    }
                    for source, budget in sorted(self._budgets.items())
                },
            }

    def _ensure_queue(self, queue, source):
        if queue not in self._queues:
            self._queues[queue] = deque()
            self._queue_source[queue] = source
            self._counters[queue] = _QueueCounters(source)
        if source not in self._budgets:
            self._budgets[source] = SharedIntervalBudget(
                self.min_interval_seconds, self.clock
            )

    def _purge_expired(self, queue, now):
        """Drop expired head requests; they never deduct the budget."""
        pending = self._queues[queue]
        counters = self._counters[queue]
        while pending:
            head = pending[0]
            if head.status is RequestStatus.DISPATCHED:
                # In-flight requests keep their slot even if their
                # deadline passes while the handler is running.
                return
            if head.deadline is None or now <= head.deadline:
                return
            pending.popleft()
            head.status = RequestStatus.EXPIRED
            head.completed_at = now
            counters.queued -= 1
            counters.expired += 1

    def _eligible_queues(self, now):
        eligible = []
        for queue, pending in self._queues.items():
            self._purge_expired(queue, now)
            if not pending:
                continue
            head = pending[0]
            # The head is currently being handled by another thread
            # (possible with a zero interval); it must not be dispatched
            # a second time before its retry/finalization.
            if head.status is RequestStatus.DISPATCHED:
                continue
            if head.not_before > now:
                continue
            source = self._queue_source[queue]
            if not self._budgets[source].is_available(now):
                continue
            eligible.append(queue)
        return eligible

    def _weights(self, active):
        if self.priority_queue is None:
            equal = 1.0 / len(active)
            return {queue: equal for queue in active}

        others = [q for q in active if q != self.priority_queue]
        weights = {}
        if self.priority_queue in active:
            weights[self.priority_queue] = 1.0 - self.min_share
            remaining = self.min_share
        else:
            remaining = 1.0
        if others:
            share = remaining / len(others)
            for queue in others:
                weights[queue] = share
        return weights

    def _pick_queue(self, now):
        active = self._eligible_queues(now)
        if not active:
            return None
        weights = self._weights(active)

        def fairness_key(queue):
            ratio = self._counters[queue].dispatched / weights[queue]
            is_priority = 0 if queue == self.priority_queue else 1
            # Smallest normalized dispatch count wins; the priority
            # queue leads exact ties, and the name remains a final
            # deterministic tie-breaker.
            return ratio, is_priority, queue

        return min(active, key=fairness_key)

    def _handle_success(self, request, result):
        with self._lock:
            pending = self._queues[request.queue]
            if pending and pending[0] is request:
                pending.popleft()
                counters = self._counters[request.queue]
                counters.queued -= 1
                counters.succeeded += 1
            request.status = RequestStatus.SUCCEEDED
            request.completed_at = self.clock.now()
            request.result = result

    def _handle_failure(self, request, error):
        with self._lock:
            now = self.clock.now()
            retryable = isinstance(error, self._retry_on)
            if retryable and request.attempts <= self.retry_attempts:
                # Stays at the queue head; its next dispatch will
                # deduct the shared budget again.
                request.status = RequestStatus.QUEUED
                request.not_before = now + self.retry_error_wait_seconds
                request.error = error
                return
            pending = self._queues[request.queue]
            if pending and pending[0] is request:
                pending.popleft()
                counters = self._counters[request.queue]
                counters.queued -= 1
                counters.failed += 1
            request.status = RequestStatus.FAILED
            request.completed_at = now
            request.error = error
