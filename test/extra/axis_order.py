import pytest

from geopy.extra.axis_order import (
    NORMAL,
    SWAPPED,
    AxisOrderDiagnosis,
    Constraint,
    StationRecord,
    TrustedPoint,
    diagnose_axis_order,
    format_report,
    reconstruct,
)
from geopy.point import Point

BERLIN = Point(52.5, 13.4)


def test_range_forces_normal_when_second_column_exceeds_90():
    records = [
        ("P1", "A", 48.0, 120.0),
        ("P1", "B", 49.0, 121.0),
    ]
    diagnosis = diagnose_axis_order(records)
    assert diagnosis.is_unique
    assert diagnosis.forced == {"P1": NORMAL}
    assert diagnosis.ambiguous == ()
    assert diagnosis.explanations[0].orders == {"P1": NORMAL}
    assert diagnosis.explanations[0].points[0] == Point(48.0, 120.0)


def test_range_forces_swapped_when_first_column_exceeds_90():
    records = [("P2", "A", 100.0, 48.0)]
    diagnosis = diagnose_axis_order(records)
    assert diagnosis.forced == {"P2": SWAPPED}
    assert diagnosis.explanations[0].points[0] == Point(48.0, 100.0)


def test_both_columns_in_range_yield_two_explanations_without_evidence():
    records = [("P3", "B", 52.5, 13.4)]
    diagnosis = diagnose_axis_order(records)
    assert diagnosis.is_consistent
    assert not diagnosis.is_unique
    assert len(diagnosis.explanations) == 2
    assert diagnosis.ambiguous == ("P3",)
    orders = {
        explanation.orders["P3"]
        for explanation in diagnosis.explanations
    }
    assert orders == {NORMAL, SWAPPED}
    points = {
        tuple(explanation.points[0]) for explanation in diagnosis.explanations
    }
    assert points == {tuple(Point(52.5, 13.4)), tuple(Point(13.4, 52.5))}


def test_trusted_coordinate_resolves_ambiguous_partition():
    records = [("P4", "BER", 52.5, 13.4)]
    diagnosis = diagnose_axis_order(
        records, trusted={"BER": BERLIN}, tolerance_km=1.0)
    assert diagnosis.is_unique
    assert diagnosis.forced == {"P4": NORMAL}
    assert diagnosis.explanations[0].points[0] == BERLIN


def test_trusted_coordinate_detects_swapped_partition():
    records = [("P5", "BER", 13.4, 52.5)]
    diagnosis = diagnose_axis_order(
        records, trusted=[TrustedPoint("BER", BERLIN)], tolerance_km=1.0)
    assert diagnosis.forced == {"P5": SWAPPED}
    assert diagnosis.explanations[0].points[0] == Point(52.5, 13.4)


def test_cross_partition_station_gives_two_mirror_explanations():
    records = [
        ("P6", "MAD", 40.0, -3.0),
        ("P7", "MAD", -3.0, 40.0),
    ]
    diagnosis = diagnose_axis_order(records, tolerance_km=1.0)
    assert len(diagnosis.explanations) == 2
    assert set(diagnosis.forced) == set()
    assert set(diagnosis.ambiguous) == {"P6", "P7"}
    order_pairs = {
        (explanation.orders["P6"], explanation.orders["P7"])
        for explanation in diagnosis.explanations
    }
    assert order_pairs == {(NORMAL, SWAPPED), (SWAPPED, NORMAL)}
    for explanation in diagnosis.explanations:
        assert explanation.points[0] == explanation.points[1]


def test_cross_partition_plus_trusted_coordinate_pins_single_explanation():
    records = [
        ("P6", "MAD", 40.0, -3.0),
        ("P7", "MAD", -3.0, 40.0),
    ]
    diagnosis = diagnose_axis_order(
        records, trusted={"MAD": Point(40.0, -3.0)}, tolerance_km=1.0)
    assert diagnosis.is_unique
    assert diagnosis.forced == {"P6": NORMAL, "P7": SWAPPED}


def test_cross_partition_tolerance_is_respected():
    records = [
        ("P8", "S", 40.0, -3.0),
        ("P9", "S", -3.0, 40.001),
    ]
    tight = diagnose_axis_order(records, tolerance_km=0.05)
    assert len(tight.explanations) == 0
    assert tight.conflict is not None

    loose = diagnose_axis_order(records, tolerance_km=1.0)
    assert len(loose.explanations) == 2


def test_contradiction_locates_unsatisfiable_trusted_constraint():
    records = [("P10", "X", 50.0, 10.0)]
    diagnosis = diagnose_axis_order(
        records, trusted={"X": Point(0.0, 0.0)}, tolerance_km=1.0)
    assert not diagnosis.is_consistent
    assert diagnosis.explanations == ()
    conflict = diagnosis.conflict
    assert isinstance(conflict, tuple) and len(conflict) == 1
    assert conflict[0].kind == "trusted"
    assert conflict[0].records == (
        StationRecord("P10", "X", 50.0, 10.0, 0),
    )


def test_contradiction_locates_cross_partition_tolerance_violation():
    records = [
        ("P8", "S", 40.0, -3.0),
        ("P9", "S", -3.0, 40.001),
    ]
    diagnosis = diagnose_axis_order(records, tolerance_km=0.05)
    assert not diagnosis.is_consistent
    conflict = diagnosis.conflict
    assert len(conflict) == 1
    assert conflict[0].kind == "cross_partition"
    assert {record.partition for record in conflict[0].records} == {"P8", "P9"}


def test_contradiction_with_two_range_constraints_reports_minimal_pair():
    # Record 0 is only range-valid when swapped (first column exceeds 90),
    # record 1 is only range-valid when normal (second column exceeds 90).
    # A partition has a single axis order, so both cannot hold together;
    # each constraint on its own is satisfiable, hence a minimal pair.
    records = [
        ("P13", "a", 100.0, 50.0),
        ("P13", "b", 50.0, 100.0),
    ]
    diagnosis = diagnose_axis_order(records)
    assert not diagnosis.is_consistent
    conflict = diagnosis.conflict
    assert len(conflict) == 2
    assert {constraint.kind for constraint in conflict} == {"range"}
    assert {
        record.index
        for constraint in conflict
        for record in constraint.records
    } == {0, 1}


def test_reported_conflict_core_is_inclusion_minimal():
    from geopy.extra.axis_order import (
        _build_constraints,
        _constraint_masks,
    )

    records = [
        ("P8", "S", 40.0, -3.0),
        ("P9", "S", -3.0, 40.001),
    ]
    diagnosis = diagnose_axis_order(records, tolerance_km=0.05)
    core = diagnosis.conflict
    assert core

    raw = [
        StationRecord(partition, station, first, second, index)
        for index, (partition, station, first, second) in enumerate(records)
    ]
    partition_index = {"P8": 0, "P9": 1}
    built = _build_constraints(raw, {}, partition_index, 0.05)
    masks = _constraint_masks(built, 4)
    constraint_by_detail = {
        constraint.detail: mask
        for (constraint, _holds), mask in zip(built, masks)
    }
    core_masks = [constraint_by_detail[item.detail] for item in core]

    def conjunct(indices):
        combined = (1 << 4) - 1
        for index in indices:
            combined &= core_masks[index]
        return combined

    assert conjunct(range(len(core))) == 0
    for dropped in range(len(core)):
        remaining = [i for i in range(len(core)) if i != dropped]
        assert conjunct(remaining) != 0


def test_conflict_core_excludes_unrelated_satisfiable_constraints():
    records = [
        ("P10", "X", 100.0, 50.0),
        ("P11", "Y", 52.5, 13.4),
    ]
    diagnosis = diagnose_axis_order(
        records,
        trusted={"X": Point(50.0, 10.0), "Y": BERLIN},
        tolerance_km=1.0,
    )
    assert not diagnosis.is_consistent
    involved_partitions = {
        record.partition
        for constraint in diagnosis.conflict
        for record in constraint.records
    }
    assert involved_partitions == {"P10"}


def test_reconstruct_returns_none_for_illegal_latitude():
    record = StationRecord("P", "S", 100.0, 50.0, 0)
    assert reconstruct(record, NORMAL) is None
    assert reconstruct(record, SWAPPED) == Point(50.0, 100.0)


def test_duplicate_station_within_one_partition_must_stay_consistent():
    # Both orders are range-valid here.  In normal order the two points are
    # 0.1 degree of latitude apart near the equator (~11.1 km); after
    # swapping they are 0.1 degree of longitude apart at latitude 40
    # (~8.5 km).  A 10 km tolerance therefore forces "swapped".
    records = [
        ("P12", "DUP", 0.0, 40.0),
        ("P12", "DUP", 0.1, 40.0),
    ]
    diagnosis = diagnose_axis_order(records, tolerance_km=10.0)
    assert diagnosis.is_unique
    assert diagnosis.forced == {"P12": SWAPPED}
    explanation = diagnosis.explanations[0]
    assert explanation.points[0] == Point(40.0, 0.0)
    assert explanation.points[1] == Point(40.0, 0.1)


def test_input_validation():
    with pytest.raises(ValueError):
        diagnose_axis_order([], tolerance_km=1.0)
    with pytest.raises(ValueError):
        diagnose_axis_order([("P", "S", 1.0, 2.0)], tolerance_km=-1.0)


def test_format_report_covers_all_outcomes():
    ambiguous = diagnose_axis_order([("P3", "B", 52.5, 13.4)])
    report = format_report(ambiguous)
    assert "Viable explanations: 2" in report
    assert "P3" in report

    conflict = diagnose_axis_order(
        [("P10", "X", 50.0, 10.0)],
        trusted={"X": Point(0.0, 0.0)},
    )
    conflict_report = format_report(conflict)
    assert conflict_report.startswith("CONTRADICTION")
    assert "trusted" in conflict_report
    assert "P10" in conflict_report


def test_full_batch_diagnosis_mixes_forced_ambiguous_and_resolved_partitions():
    records = [
        ("normal_by_range", "A1", 48.0, 120.0),
        ("swapped_by_range", "A2", 120.0, 48.0),
        ("ambiguous", "BER", 13.4, 52.5),
        ("linked_a", "MAD", 40.0, -3.0),
        ("linked_b", "MAD", -3.0, 40.0),
    ]
    diagnosis = diagnose_axis_order(
        records,
        trusted={"BER": BERLIN, "MAD": Point(40.0, -3.0)},
        tolerance_km=1.0,
    )
    assert diagnosis.is_unique
    assert diagnosis.forced == {
        "normal_by_range": NORMAL,
        "swapped_by_range": SWAPPED,
        "ambiguous": SWAPPED,
        "linked_a": NORMAL,
        "linked_b": SWAPPED,
    }
    assert isinstance(diagnosis, AxisOrderDiagnosis)
    assert all(
        isinstance(constraint, Constraint)
        for constraint in diagnosis.conflict or ()
    )
