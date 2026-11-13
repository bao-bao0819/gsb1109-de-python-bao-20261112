import pytest
from uncertainties import ufloat

from uncertainties.temperature_migration import (
    AbsoluteReading,
    CalibrationConflictError,
    CalibrationRegistry,
    DifferenceRecord,
    ProvenanceError,
    convert_absolute,
    convert_then_subtract,
    cross_check_difference,
    migrate_difference,
)


@pytest.fixture
def registry():
    registry = CalibrationRegistry()
    # Legacy batch A: Fahrenheit-like column, slope and zero with errors.
    registry.register_batch(
        "A", "degF", slope=2.0, slope_std=0.1, zero=30.0, zero_std=5.0
    )
    # Legacy batch B: another unit with its own calibration.
    registry.register_batch(
        "B", "degR", slope=1.0, slope_std=0.05, zero=273.15, zero_std=1.0
    )
    return registry


def test_absolute_conversion_applies_shared_calibration(registry):
    reading = AbsoluteReading(batch_id="A", value=10.0)
    converted = convert_absolute(reading, registry)

    assert converted.nominal_value == pytest.approx(50.0)
    # slope * value and zero both contribute: sqrt((0.1*10)**2 + 5**2)
    assert converted.std_dev == pytest.approx((1.0 + 25.0) ** 0.5)


def test_same_batch_difference_cancels_zero_completely(registry):
    high = AbsoluteReading(batch_id="A", value=10.0)
    low = AbsoluteReading(batch_id="A", value=4.0)
    record = DifferenceRecord(
        record_id="rec-same",
        raw_difference=6.0,
        source_batch_ids=("A", "A"),
        high=high,
        low=low,
    )

    path_a = convert_then_subtract(high, low, registry)
    path_b = migrate_difference(record, registry)

    assert path_a.nominal_value == pytest.approx(12.0)
    assert path_b.nominal_value == pytest.approx(12.0)
    # Only the shared slope applied to the raw gap remains: 0.1 * 6.
    assert path_b.std_dev == pytest.approx(0.6)
    assert path_a.std_dev == pytest.approx(0.6)
    # The zero (5.0) and its uncertainty leave no trace at all.
    assert path_a.nominal_value == path_b.nominal_value
    assert path_a.std_dev == path_b.std_dev


def test_shared_parameters_beat_treating_readings_as_independent(registry):
    high = AbsoluteReading(batch_id="A", value=10.0)
    low = AbsoluteReading(batch_id="A", value=4.0)

    shared = convert_then_subtract(high, low, registry)

    # What the difference would look like with separate (uncorrelated)
    # slope/zero draws per endpoint:
    slope_h, zero_h = ufloat(2.0, 0.1), ufloat(30.0, 5.0)
    slope_l, zero_l = ufloat(2.0, 0.1), ufloat(30.0, 5.0)
    independent = (slope_h * 10.0 + zero_h) - (slope_l * 4.0 + zero_l)

    assert shared.std_dev == pytest.approx(0.6)
    assert independent.std_dev == pytest.approx(51.16**0.5)
    assert shared.std_dev < independent.std_dev


def test_cross_batch_difference_keeps_both_zeros(registry):
    high = AbsoluteReading(batch_id="A", value=10.0)  # -> 50
    low = AbsoluteReading(batch_id="B", value=200.0)  # -> 473.15
    record = DifferenceRecord(
        record_id="rec-cross",
        raw_difference=-190.0,
        source_batch_ids=("A", "B"),
        high=high,
        low=low,
    )

    result = migrate_difference(record, registry)

    assert result.nominal_value == pytest.approx(50.0 - 473.15)
    # Both zeros survive: (0.1*10)^2 + 5^2 + (0.05*200)^2 + 1^2
    assert result.std_dev == pytest.approx(127.0**0.5)


def test_cross_batch_difference_requires_endpoint_readings(registry):
    record = DifferenceRecord(
        record_id="rec-cross-no-endpoints",
        raw_difference=-190.0,
        source_batch_ids=("A", "B"),
    )

    with pytest.raises(ProvenanceError, match="endpoint"):
        migrate_difference(record, registry)


def test_difference_without_source_batches_is_refused(registry):
    high = AbsoluteReading(batch_id="A", value=10.0)
    low = AbsoluteReading(batch_id="A", value=4.0)
    record = DifferenceRecord(
        record_id="rec-orphan",
        raw_difference=6.0,
        source_batch_ids=None,
        high=high,
        low=low,
    )

    with pytest.raises(ProvenanceError, match="source batches"):
        migrate_difference(record, registry)
    with pytest.raises(ProvenanceError, match="source batches"):
        cross_check_difference(record, registry)


def test_declared_batches_must_match_endpoint_batches(registry):
    high = AbsoluteReading(batch_id="A", value=10.0)
    low = AbsoluteReading(batch_id="B", value=200.0)
    record = DifferenceRecord(
        record_id="rec-mismatch",
        raw_difference=-190.0,
        source_batch_ids=("A", "A"),
        high=high,
        low=low,
    )

    with pytest.raises(ProvenanceError, match="do not match"):
        migrate_difference(record, registry)


def test_cross_check_accepts_consistent_complete_record(registry):
    high = AbsoluteReading(batch_id="A", value=10.0)
    low = AbsoluteReading(batch_id="A", value=4.0)
    record = DifferenceRecord(
        record_id="rec-ok",
        raw_difference=6.0,
        source_batch_ids=("A", "A"),
        high=high,
        low=low,
    )

    result = cross_check_difference(record, registry)

    assert result.consistent is True
    assert result.nominal_difference == pytest.approx(0.0)
    assert result.std_difference == pytest.approx(0.0)
    assert (
        result.converted_then_subtracted.nominal_value
        == result.migrated_from_source.nominal_value
    )


def test_cross_check_flags_tampered_raw_difference(registry):
    high = AbsoluteReading(batch_id="A", value=10.0)
    low = AbsoluteReading(batch_id="A", value=4.0)
    record = DifferenceRecord(
        record_id="rec-tampered",
        raw_difference=9.0,  # does not match the 10 - 4 endpoints
        source_batch_ids=("A", "A"),
        high=high,
        low=low,
    )

    result = cross_check_difference(record, registry)

    assert result.consistent is False
    assert result.nominal_difference == pytest.approx(2.0 * 3.0)


def test_cross_check_requires_complete_record(registry):
    record = DifferenceRecord(
        record_id="rec-incomplete",
        raw_difference=6.0,
        source_batch_ids=("A", "A"),
    )

    with pytest.raises(ProvenanceError, match="complete"):
        cross_check_difference(record, registry)


def test_batch_can_only_hold_one_parameter_set(registry):
    # Re-registering the very same parameters is harmless.
    again = registry.register_batch(
        "A", "degF", slope=2.0, slope_std=0.1, zero=30.0, zero_std=5.0
    )
    assert again is registry.params("A")

    with pytest.raises(CalibrationConflictError, match="already registered"):
        registry.register_batch(
            "A", "degF", slope=2.1, slope_std=0.1, zero=30.0, zero_std=5.0
        )


def test_unknown_batch_lookup(registry):
    reading = AbsoluteReading(batch_id="Z", value=1.0)
    with pytest.raises(KeyError, match="unknown calibration batch"):
        convert_absolute(reading, registry)
    assert "A" in registry
    assert "Z" not in registry


def test_ufloat_raw_readings_are_supported(registry):
    reading = AbsoluteReading(batch_id="A", value=ufloat(10.0, 0.2))
    converted = convert_absolute(reading, registry)

    assert converted.nominal_value == pytest.approx(50.0)
    # Reading noise is scaled by the slope in addition to calibration noise.
    expected_var = (0.1 * 10.0) ** 2 + 5.0**2 + (2.0 * 0.2) ** 2
    assert converted.std_dev == pytest.approx(expected_var**0.5)
