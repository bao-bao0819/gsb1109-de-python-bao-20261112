"""Tests for the inspection work-order schedule analysis."""

import datetime
import io
from pathlib import Path

import pytest

from inspection_schedule import (
    ScheduleDataError,
    advance_hours,
    minimum_corrections,
    parse_calendar,
    simulate,
)

FIXTURE = Path(__file__).resolve().parent.parent / "test_files" / "inspection_workorders.ics"

BATCH_START = datetime.datetime(2026, 11, 9, 8, 0, 0, tzinfo=datetime.timezone.utc)


def _calendar(body: str) -> str:
    return (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//Inspection//Work Orders//EN\r\n"
        "X-BATCH-START:20260101T000000Z\r\n"
        f"{body}"
        "END:VCALENDAR\r\n"
    )


def test_parse_extracts_work_orders_and_extension_fields():
    batch = parse_calendar(FIXTURE)

    assert batch.batch_start == BATCH_START
    assert len(batch.work_orders) == 9

    wo_a = batch.work_order("WO-A")
    assert wo_a.summary == "Pressure vessel inspection - chain one"
    assert wo_a.planned_hours == 4
    assert wo_a.actual_hours == 6
    assert wo_a.delay_hours == 2
    assert wo_a.is_delayed is True

    wo_b = batch.work_order("WO-B")
    assert wo_b.is_delayed is False
    assert wo_b.delay_hours == 0
    # Comma-separated predecessor lists survive vobject's text decoding.
    assert batch.work_order("WO-H").predecessor_uids == ("WO-B", "WO-F")


def test_dependency_closure_excludes_unclosed_orders():
    batch = parse_calendar(FIXTURE)

    # WO-G names a missing predecessor; WO-I is chained behind WO-G.
    assert batch.open_uids == {"WO-G", "WO-I"}
    assert "WO-G" not in batch.closed_uids
    assert "WO-I" not in batch.closed_uids
    # Only closed orders carry schedulable predecessor edges.
    assert set(batch.predecessors) == set(batch.work_orders)
    assert batch.predecessors["WO-G"] == []
    assert "WO-G" not in simulate(batch).early_finish


def test_baseline_schedule_and_parallel_critical_chains():
    batch = parse_calendar(FIXTURE)
    result = simulate(batch)

    assert result.acceptance_offset_hours == 13
    assert result.acceptance_at == BATCH_START + datetime.timedelta(hours=13)
    assert result.critical_chain is None  # two parallel chains, no unique chain
    assert result.critical_chains == [
        ["WO-A", "WO-B", "WO-H"],
        ["WO-E", "WO-F", "WO-H"],
    ]
    # Unlimited parallel workstations: independent roots all start at hour 0.
    assert result.early_start["WO-A"] == 0
    assert result.early_start["WO-E"] == 0
    assert result.early_start["WO-H"] == 10
    assert result.early_finish["WO-H"] == 13


def test_correcting_one_parallel_chain_does_not_advance_acceptance():
    batch = parse_calendar(FIXTURE)

    assert advance_hours(batch, ("WO-A",)) == 0
    assert simulate(batch, ("WO-A",)).critical_chains == [["WO-E", "WO-F", "WO-H"]]

    assert advance_hours(batch, ("WO-E",)) == 0
    assert simulate(batch, ("WO-E",)).critical_chains == [["WO-A", "WO-B", "WO-H"]]

    # A delayed but non-critical order never moves the terminal either.
    assert advance_hours(batch, ("WO-J",)) == 0


def test_both_parallel_chains_must_be_corrected_together():
    batch = parse_calendar(FIXTURE)

    result = simulate(batch, ("WO-A", "WO-E"))
    assert result.acceptance_offset_hours == 11
    assert advance_hours(batch, ("WO-A", "WO-E")) == 2
    # Once balanced, a single chain is critical again.
    assert result.critical_chain == ["WO-A", "WO-B", "WO-H"]
    assert result.durations["WO-A"] == 4  # planned replaces actual
    assert result.durations["WO-B"] == 4  # untouched order keeps actual hours


def test_minimum_corrections_search():
    batch = parse_calendar(FIXTURE)

    zero = minimum_corrections(batch, 0)
    assert zero.achieved is True
    assert zero.minimal_corrections == ()

    one_hour = minimum_corrections(batch, 1)
    assert one_hour.achieved is True
    assert one_hour.minimal_corrections == ("WO-A", "WO-E")
    assert one_hour.corrected.acceptance_offset_hours == 11

    two_hours = minimum_corrections(batch, 2)
    assert two_hours.minimal_corrections == ("WO-A", "WO-E")

    # No single correction can advance the parallel batch, proving minimality.
    for single in batch.delayed_closed_uids:
        assert advance_hours(batch, (single,)) == 0

    impossible = minimum_corrections(batch, 3)
    assert impossible.achieved is False
    assert impossible.minimal_corrections is None
    assert impossible.corrected is None
    assert impossible.best_advance_hours == 2


def test_single_chain_needs_only_one_correction():
    ics = _calendar(
        "BEGIN:VTODO\r\nUID:R1\r\nX-PLANNED-HOURS:2\r\nX-ACTUAL-HOURS:5\r\nEND:VTODO\r\n"
        "BEGIN:VTODO\r\nUID:R2\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\n"
        "X-PREDECESSORS:R1\r\nEND:VTODO\r\n"
        "BEGIN:VTODO\r\nUID:R3\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\nEND:VTODO\r\n"
    )
    batch = parse_calendar(ics)

    assert simulate(batch).critical_chain == ["R1", "R2"]
    assert simulate(batch).acceptance_offset_hours == 6

    outcome = minimum_corrections(batch, 3)
    assert outcome.achieved is True
    assert outcome.minimal_corrections == ("R1",)
    assert outcome.corrected.acceptance_offset_hours == 3


@pytest.mark.parametrize("separator", [",", ";", " "])
def test_predecessor_separators_are_all_supported(separator):
    preds = separator.join(["R1", "R2"])
    ics = _calendar(
        "BEGIN:VTODO\r\nUID:R1\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\nEND:VTODO\r\n"
        "BEGIN:VTODO\r\nUID:R2\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\nEND:VTODO\r\n"
        "BEGIN:VTODO\r\nUID:R3\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\n"
        f"X-PREDECESSORS:{preds}\r\nEND:VTODO\r\n"
    )
    batch = parse_calendar(ics)
    assert batch.work_order("R3").predecessor_uids == ("R1", "R2")
    assert batch.closed_uids == {"R1", "R2", "R3"}


def test_folded_predecessor_line_is_unfolded():
    ics = _calendar(
        "BEGIN:VTODO\r\nUID:R1\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\nEND:VTODO\r\n"
        "BEGIN:VTODO\r\nUID:R2\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\nEND:VTODO\r\n"
        "BEGIN:VTODO\r\nUID:R3\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\n"
        "X-PREDECESSORS:R1,\r\n R2\r\nEND:VTODO\r\n"
    )
    batch = parse_calendar(ics)
    assert batch.work_order("R3").predecessor_uids == ("R1", "R2")


def test_stream_and_path_inputs_are_equivalent():
    from_path = parse_calendar(FIXTURE)
    from_stream = parse_calendar(io.StringIO(FIXTURE.read_text(encoding="utf-8")))
    assert from_stream.closed_uids == from_path.closed_uids
    assert simulate(from_stream).acceptance_offset_hours == simulate(from_path).acceptance_offset_hours


def test_missing_predecessor_keeps_chain_open_but_roots_still_publish():
    ics = _calendar(
        "BEGIN:VTODO\r\nUID:ROOT\r\nX-PLANNED-HOURS:2\r\nX-ACTUAL-HOURS:2\r\nEND:VTODO\r\n"
        "BEGIN:VTODO\r\nUID:ORPHAN\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\n"
        "X-PREDECESSORS:GHOST\r\nEND:VTODO\r\n"
    )
    batch = parse_calendar(ics)
    assert batch.closed_uids == {"ROOT"}
    assert batch.open_uids == {"ORPHAN"}
    result = simulate(batch)
    assert set(result.early_finish) == {"ROOT"}
    assert result.acceptance_offset_hours == 2


def test_error_conditions():
    with pytest.raises(ScheduleDataError):
        parse_calendar("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nEND:VCALENDAR\r\n")

    missing_hours = _calendar(
        "BEGIN:VTODO\r\nUID:X\r\nX-ACTUAL-HOURS:1\r\nEND:VTODO\r\n"
    )
    with pytest.raises(ScheduleDataError):
        parse_calendar(missing_hours)

    duplicate = _calendar(
        "BEGIN:VTODO\r\nUID:X\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\nEND:VTODO\r\n"
        "BEGIN:VTODO\r\nUID:X\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\nEND:VTODO\r\n"
    )
    with pytest.raises(ScheduleDataError):
        parse_calendar(duplicate)

    cyclic = _calendar(
        "BEGIN:VTODO\r\nUID:A\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\n"
        "X-PREDECESSORS:B\r\nEND:VTODO\r\n"
        "BEGIN:VTODO\r\nUID:B\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:1\r\n"
        "X-PREDECESSORS:A\r\nEND:VTODO\r\n"
    )
    with pytest.raises(ScheduleDataError):
        parse_calendar(cyclic)

    negative = _calendar(
        "BEGIN:VTODO\r\nUID:A\r\nX-PLANNED-HOURS:-1\r\nX-ACTUAL-HOURS:1\r\nEND:VTODO\r\n"
    )
    with pytest.raises(ScheduleDataError):
        parse_calendar(negative)


def test_correcting_unknown_or_open_order_is_rejected():
    batch = parse_calendar(FIXTURE)
    with pytest.raises(ScheduleDataError):
        simulate(batch, ("WO-G",))
    with pytest.raises(ScheduleDataError):
        simulate(batch, ("NOPE",))


def test_negative_required_advance_is_rejected():
    batch = parse_calendar(FIXTURE)
    with pytest.raises(ScheduleDataError):
        minimum_corrections(batch, -1)


def test_iso_offset_batch_start_is_accepted():
    ics = (
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:x\r\n"
        "X-BATCH-START:2026-01-01T08:00:00+02:00\r\n"
        "BEGIN:VTODO\r\nUID:A\r\nX-PLANNED-HOURS:1\r\nX-ACTUAL-HOURS:3\r\nEND:VTODO\r\n"
        "END:VCALENDAR\r\n"
    )
    batch = parse_calendar(ics)
    assert batch.batch_start.utcoffset() == datetime.timedelta(hours=2)
    result = simulate(batch)
    assert result.acceptance_at == batch.batch_start + datetime.timedelta(hours=3)
