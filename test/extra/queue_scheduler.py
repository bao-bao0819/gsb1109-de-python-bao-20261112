import threading

import pytest

from geopy.exc import GeocoderServiceError
from geopy.extra.queue_scheduler import (
    ManualClock,
    QueueScheduler,
    RequestStatus,
    SharedIntervalBudget,
    SystemClock,
)


def make_scheduler(clock, handler=None, **kwargs):
    if handler is None:
        def handler(request):
            return "result-of-%s" % request.payload
    return QueueScheduler(handler, clock=clock, **kwargs)


class TestManualClock:
    def test_advance_and_advance_to(self):
        clock = ManualClock(start=10)
        assert clock.now() == 10
        assert clock.advance(2.5) == 12.5
        assert clock.advance_to(15) == 15
        assert clock.now() == 15

    @pytest.mark.parametrize("kwargs", [{"seconds": -1}, {"when": 1}])
    def test_cannot_move_backwards(self, kwargs):
        clock = ManualClock(start=5)
        with pytest.raises(ValueError):
            if "seconds" in kwargs:
                clock.advance(kwargs["seconds"])
            else:
                clock.advance_to(kwargs["when"])

    def test_system_clock_cannot_be_advanced(self):
        with pytest.raises(RuntimeError):
            SystemClock().advance(1)
        with pytest.raises(RuntimeError):
            SystemClock().advance_to(1)


class TestSharedIntervalBudget:
    def test_first_deduction_is_immediate(self):
        clock = ManualClock()
        budget = SharedIntervalBudget(interval_seconds=2, clock=clock)
        assert budget.is_available()
        assert budget.next_available_at is None
        assert budget.deduct() == 2
        assert budget.deducted_count == 1
        assert not budget.is_available()
        clock.advance_to(2)
        assert budget.is_available()

    def test_cannot_deduct_before_available(self):
        clock = ManualClock()
        budget = SharedIntervalBudget(interval_seconds=5, clock=clock)
        budget.deduct()
        with pytest.raises(RuntimeError):
            budget.deduct()

    def test_repeated_deductions_are_spaced(self):
        clock = ManualClock()
        budget = SharedIntervalBudget(interval_seconds=1, clock=clock)
        for expected_time in range(5):
            clock.advance_to(expected_time)
            assert budget.deduct() == expected_time + 1
        assert budget.deducted_count == 5


class TestSharedSourceBudget:
    def test_two_queues_of_same_source_share_one_budget(self):
        clock = ManualClock()
        scheduler = make_scheduler(
            clock, min_interval_seconds=1, min_share=0.25
        )
        scheduler.enqueue("backfill", "b1")
        scheduler.enqueue("realtime", "r1", source="backfill")

        sent = scheduler.dispatch_due()
        assert [r.payload for r in sent] == ["b1"]

        clock.advance(1)
        sent = scheduler.dispatch_due()
        assert [r.payload for r in sent] == ["r1"]

        stats = scheduler.stats()
        assert stats["budgets"] == {
            "backfill": {"deducted": 2, "next_available_at": 2}
        }

    def test_different_sources_have_independent_budgets(self):
        clock = ManualClock()
        scheduler = make_scheduler(clock, min_interval_seconds=10)
        scheduler.enqueue("queue-a", "a1", source="account-a")
        scheduler.enqueue("queue-b", "b1", source="account-b")

        sent = scheduler.dispatch_due()
        assert {r.payload for r in sent} == {"a1", "b1"}
        assert {r.history[0] for r in sent} == {0}

        stats = scheduler.stats()["budgets"]
        assert stats["account-a"]["deducted"] == 1
        assert stats["account-b"]["deducted"] == 1
        assert stats["account-a"]["next_available_at"] == 10
        assert stats["account-b"]["next_available_at"] == 10

    def test_interleaved_order_across_queues_is_deterministic(self):
        clock = ManualClock()
        scheduler = make_scheduler(clock, min_interval_seconds=1)
        scheduler.enqueue_many("alpha", ["a1", "a2"], source="acct")
        scheduler.enqueue_many("beta", ["b1", "b2"], source="acct")

        order = []
        for tick in range(4):
            clock.advance_to(tick)
            order.extend(r.payload for r in scheduler.dispatch_due())

        assert order == ["a1", "b1", "a2", "b2"]
        sent_requests = [scheduler.get_request(rid) for rid in range(4)]
        assert [r.history for r in sent_requests] == [
            [0],
            [2],
            [1],
            [3],
        ]
        assert scheduler.stats()["budgets"]["acct"]["deducted"] == 4


class TestMinimumShare:
    def test_backfill_keeps_minimum_share_under_priority_load(self):
        clock = ManualClock()
        scheduler = make_scheduler(
            clock,
            min_interval_seconds=1,
            min_share=0.25,
            priority_queue="realtime",
        )
        scheduler.enqueue_many("backfill", range(16), source="acct")
        scheduler.enqueue_many("realtime", range(16), source="acct")

        sent = scheduler.run_until_idle()
        order = [r.queue for r in sent]

        while "realtime" in order and order[-1] == "backfill":
            order.pop()
        backfill_slots = [i for i, queue in enumerate(order) if queue == "backfill"]
        gaps = [b - a for a, b in zip(backfill_slots, backfill_slots[1:])]

        # While both queues are full, backfill receives one slot out
        # of every four and never waits more than four slots again.
        assert max(gaps) == 4
        assert all(gap == 4 for gap in gaps[:5])
        assert scheduler.stats()["budgets"]["acct"]["deducted"] == 32

    def test_equal_shares_without_priority_queue(self):
        clock = ManualClock()
        scheduler = make_scheduler(clock, min_interval_seconds=1)
        scheduler.enqueue_many("alpha", range(4), source="acct")
        scheduler.enqueue_many("beta", range(4), source="acct")

        order = [r.queue for r in scheduler.run_until_idle()]
        assert order == [
            "alpha", "beta", "alpha", "beta",
            "alpha", "beta", "alpha", "beta",
        ]

    def test_priority_takes_leftover_capacity_when_backfill_idle(self):
        clock = ManualClock()
        scheduler = make_scheduler(
            clock,
            min_interval_seconds=1,
            min_share=0.25,
            priority_queue="realtime",
        )
        scheduler.enqueue_many("realtime", range(3), source="acct")

        sent = scheduler.run_until_idle()
        assert [r.payload for r in sent] == [0, 1, 2]
        assert [r.history[0] for r in sent] == [0, 1, 2]


class TestDeadlines:
    def test_request_expired_before_enqueue_never_touches_budget(self):
        clock = ManualClock(start=10)
        scheduler = make_scheduler(clock, min_interval_seconds=1)
        expired = scheduler.enqueue(
            "backfill", "late", deadline=5
        )
        live = scheduler.enqueue("backfill", "live")

        assert expired.status is RequestStatus.EXPIRED
        assert expired.completed_at == 10
        assert scheduler.dispatch_due() == [live]
        assert scheduler.stats()["queues"]["backfill"]["expired"] == 1
        assert scheduler.stats()["budgets"]["backfill"]["deducted"] == 1

    def test_expired_head_does_not_occupy_next_send_point(self):
        clock = ManualClock()
        scheduler = make_scheduler(clock, min_interval_seconds=1)
        stale = scheduler.enqueue("backfill", "stale", deadline=0.5)
        fresh = scheduler.enqueue("backfill", "fresh")

        clock.advance_to(2)
        sent = scheduler.dispatch_due()

        assert stale.status is RequestStatus.EXPIRED
        assert [r.payload for r in sent] == ["fresh"]
        # The budget is charged for the first time at t=2, not at t=0.
        assert fresh.history == [2]
        budget = scheduler.stats()["budgets"]["backfill"]
        assert budget == {"deducted": 1, "next_available_at": 3}

    def test_next_event_ignores_expired_requests(self):
        clock = ManualClock()
        scheduler = make_scheduler(clock, min_interval_seconds=1)
        scheduler.enqueue("backfill", "stale", deadline=0.5)
        scheduler.enqueue("backfill", "fresh")

        clock.advance_to(2)
        assert scheduler.next_event_at() == 2
        scheduler.run_until_idle()
        assert scheduler.next_event_at() is None

    def test_deadline_does_not_affect_already_sent_request(self):
        clock = ManualClock()
        scheduler = make_scheduler(clock, min_interval_seconds=1)
        request = scheduler.enqueue("backfill", "sent", deadline=10)
        scheduler.dispatch_due()
        assert request.status is RequestStatus.SUCCEEDED

        clock.advance_to(20)
        scheduler.dispatch_due()
        assert request.is_final
        assert scheduler.stats()["budgets"]["backfill"]["deducted"] == 1


class TestRetries:
    def test_each_retry_deducts_the_shared_budget(self):
        clock = ManualClock()
        attempts = []

        def handler(request):
            attempts.append(clock.now())
            if len(attempts) < 3:
                raise GeocoderServiceError("boom")
            return "ok"

        scheduler = make_scheduler(
            clock,
            handler=handler,
            min_interval_seconds=1,
            retry_attempts=2,
            retry_error_wait_seconds=1,
        )
        request = scheduler.enqueue("backfill", "x")

        for tick in range(3):
            clock.advance_to(tick)
            scheduler.dispatch_due()

        assert request.attempts == 3
        assert request.history == [0, 1, 2]
        assert request.status is RequestStatus.SUCCEEDED
        assert request.result == "ok"
        assert scheduler.stats()["budgets"]["backfill"]["deducted"] == 3

    def test_retry_backoff_holds_the_queue_head(self):
        clock = ManualClock()
        scheduler = make_scheduler(
            clock,
            handler=lambda req: (_ for _ in ()).throw(GeocoderServiceError()),
            min_interval_seconds=1,
            retry_attempts=1,
            retry_error_wait_seconds=1,
        )
        head = scheduler.enqueue("backfill", "head")
        scheduler.enqueue("backfill", "second")

        scheduler.dispatch_due()
        assert head.attempts == 1
        assert head.status is RequestStatus.QUEUED
        assert scheduler.next_event_at() == 1

        clock.advance_to(1)
        scheduler.dispatch_due()
        assert head.attempts == 2
        assert head.status is RequestStatus.FAILED
        assert isinstance(head.error, GeocoderServiceError)

        clock.advance_to(2)
        scheduler.dispatch_due()
        second = scheduler.get_request(1)
        assert second.history == [2]
        # Two attempts of the failed request plus one new request.
        assert scheduler.stats()["budgets"]["backfill"]["deducted"] == 3

    def test_non_retryable_error_fails_immediately(self):
        clock = ManualClock()

        def handler(request):
            raise ValueError("not a geocoder error")

        scheduler = make_scheduler(
            clock,
            handler=handler,
            min_interval_seconds=1,
            retry_attempts=3,
        )
        request = scheduler.enqueue("backfill", "x")
        scheduler.dispatch_due()

        assert request.status is RequestStatus.FAILED
        assert request.attempts == 1
        assert isinstance(request.error, ValueError)
        assert scheduler.stats()["budgets"]["backfill"]["deducted"] == 1

    def test_retry_consumes_slots_visible_to_other_queues(self):
        clock = ManualClock()
        outcomes = {"flaky": 0}

        def handler(request):
            if request.payload == "flaky":
                outcomes["flaky"] += 1
                if outcomes["flaky"] == 1:
                    raise GeocoderServiceError("transient")
            return "ok"

        scheduler = make_scheduler(
            clock,
            handler=handler,
            min_interval_seconds=1,
            retry_attempts=1,
            priority_queue="realtime",
            min_share=0.25,
        )
        flaky = scheduler.enqueue("backfill", "flaky", source="acct")
        other = scheduler.enqueue("realtime", "other", source="acct")

        scheduler.dispatch_due()  # t=0: realtime sent
        clock.advance_to(1)
        scheduler.dispatch_due()  # t=1: flaky attempt 1 fails
        clock.advance_to(2)
        scheduler.dispatch_due()  # t=2: flaky retry succeeds

        assert flaky.history == [1, 2]
        assert other.history == [0]
        assert scheduler.stats()["budgets"]["acct"]["deducted"] == 3


class TestSchedulerFlow:
    def test_run_until_idle_drains_mixed_queues(self):
        clock = ManualClock()
        scheduler = make_scheduler(
            clock,
            handler=lambda req: None,
            min_interval_seconds=0.5,
            min_share=0.3,
            priority_queue="realtime",
        )
        scheduler.enqueue_many("backfill", ["b1", "b2"], source="acct")
        scheduler.enqueue("realtime", "r1", source="acct")
        scheduler.enqueue("solo", "s1", source="other-acct", deadline=100)

        dispatched = scheduler.run_until_idle()

        assert len(dispatched) == 4
        assert [r.payload for r in dispatched] == ["r1", "s1", "b1", "b2"]
        assert [r.history[0] for r in dispatched] == [0, 0, 0.5, 1.0]
        assert scheduler.next_event_at() is None
        stats = scheduler.stats()
        assert stats["queues"]["backfill"]["succeeded"] == 2
        assert stats["queues"]["realtime"]["succeeded"] == 1
        assert stats["queues"]["solo"]["succeeded"] == 1
        assert stats["queues"]["solo"]["expired"] == 0
        assert stats["budgets"]["acct"]["deducted"] == 3
        assert stats["budgets"]["other-acct"]["deducted"] == 1

    def test_invalid_construction_arguments(self):
        with pytest.raises(AssertionError):
            QueueScheduler(lambda req: None, min_share=0)
        with pytest.raises(AssertionError):
            QueueScheduler(lambda req: None, min_share=1)
        with pytest.raises(AssertionError):
            QueueScheduler(
                lambda req: None,
                min_interval_seconds=2,
                retry_error_wait_seconds=1,
            )
        with pytest.raises(AssertionError):
            QueueScheduler(lambda req: None, retry_attempts=-1)


class TestThreadSafety:
    def test_inflight_head_is_not_dispatched_twice(self):
        clock = ManualClock()
        entered = threading.Event()
        release = threading.Event()

        def handler(request):
            entered.set()
            assert release.wait(timeout=5)
            return "ok"

        scheduler = make_scheduler(
            clock, handler=handler, min_interval_seconds=0
        )
        scheduler.enqueue("backfill", "x")

        worker = threading.Thread(target=scheduler.dispatch_due)
        worker.start()
        assert entered.wait(timeout=5)

        # The handler is still running; another dispatch round must
        # not pick the same in-flight head, even with a zero interval.
        assert scheduler.dispatch_due() == []
        stats = scheduler.stats()
        assert stats["queues"]["backfill"]["dispatched"] == 1
        assert stats["budgets"]["backfill"]["deducted"] == 1

        release.set()
        worker.join(timeout=5)
        request = scheduler.get_request(0)
        assert request.status is RequestStatus.SUCCEEDED
        assert request.attempts == 1

    def test_parallel_drain_dispatches_each_request_once(self):
        clock = ManualClock()
        n_requests = 50
        n_workers = 8
        scheduler = make_scheduler(
            clock, handler=lambda req: None, min_interval_seconds=0
        )
        scheduler.enqueue_many("backfill", range(n_requests))

        def worker():
            while scheduler.next_event_at() is not None:
                scheduler.dispatch_due()

        threads = [threading.Thread(target=worker) for _ in range(n_workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert scheduler.next_event_at() is None
        stats = scheduler.stats()
        assert stats["budgets"]["backfill"]["deducted"] == n_requests
        assert stats["queues"]["backfill"]["succeeded"] == n_requests
        assert all(
            scheduler.get_request(rid).attempts == 1
            for rid in range(n_requests)
        )
