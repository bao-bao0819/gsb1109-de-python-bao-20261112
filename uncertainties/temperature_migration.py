"""
Migration of legacy temperature-warehouse data into the new warehouse.

The legacy warehouse stores each column in its own unit.  The new
warehouse uses one unified unit, so every legacy value must go through
the calibration relation::

    T_new = slope * T_raw + zero

Both calibration coefficients carry an uncertainty and all readings of
the same calibration batch share one single set of coefficients.

Two kinds of columns are migrated:

* absolute temperature readings, and
* temperature differences between two endpoint readings.

For a difference whose endpoints share a calibration batch, the
calibration zero cancels when the endpoints are converted first and
subtracted afterwards.  The zero does not cancel when the endpoints
come from different batches.  A legacy difference whose source batches
are not recorded cannot be migrated honestly: it must not be turned
into a fabricated independent reading.

Every complete record is checked along two independent paths:

1. "convert then subtract": convert both endpoints and take the
   difference, and
2. "migrate the difference from its declared source": migrate the
   legacy difference column itself using the calibration parameters of
   its source batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import NamedTuple, Optional, Union

from uncertainties import UFloat, ufloat

RawValue = Union[float, UFloat]

__all__ = [
    "CalibrationConflictError",
    "ProvenanceError",
    "CalibrationParams",
    "CalibrationRegistry",
    "LegacyColumn",
    "AbsoluteReading",
    "DifferenceRecord",
    "CrossCheckResult",
    "convert_absolute",
    "convert_then_subtract",
    "migrate_difference",
    "cross_check_difference",
]


class ProvenanceError(ValueError):
    """A difference cannot be migrated because its source batches are unknown."""


class CalibrationConflictError(ValueError):
    """A batch is registered twice with incompatible calibration parameters."""


@dataclass(frozen=True)
class CalibrationParams:
    """Calibration coefficients of one batch, with their uncertainties."""

    source_unit: str
    slope: UFloat
    zero: UFloat

    @classmethod
    def create(
        cls,
        source_unit: str,
        slope: float,
        slope_std: float,
        zero: float,
        zero_std: float,
    ) -> "CalibrationParams":
        return cls(
            source_unit=source_unit,
            slope=ufloat(slope, slope_std),
            zero=ufloat(zero, zero_std),
        )


def _same_params(first: CalibrationParams, second: CalibrationParams) -> bool:
    return (
        first.source_unit == second.source_unit
        and first.slope.nominal_value == second.slope.nominal_value
        and first.slope.std_dev == second.slope.std_dev
        and first.zero.nominal_value == second.zero.nominal_value
        and first.zero.std_dev == second.zero.std_dev
    )


class CalibrationRegistry:
    """
    Holds the calibration parameters of every known batch.

    A batch owns exactly one slope/zero pair: every reading of the
    batch is converted through the same uncertain coefficients, which
    preserves their correlations automatically.
    """

    def __init__(self) -> None:
        self._batches: dict[str, CalibrationParams] = {}

    def register_batch(
        self,
        batch_id: str,
        source_unit: str,
        slope: float,
        slope_std: float,
        zero: float,
        zero_std: float,
    ) -> CalibrationParams:
        params = CalibrationParams.create(source_unit, slope, slope_std, zero, zero_std)
        existing = self._batches.get(batch_id)
        if existing is not None:
            if not _same_params(existing, params):
                msg = (
                    f"calibration batch {batch_id!r} is already registered "
                    "with different parameters"
                )
                raise CalibrationConflictError(msg)
            return existing
        self._batches[batch_id] = params
        return params

    def params(self, batch_id: str) -> CalibrationParams:
        try:
            return self._batches[batch_id]
        except KeyError:
            msg = f"unknown calibration batch {batch_id!r}"
            raise KeyError(msg) from None

    def __contains__(self, batch_id: object) -> bool:
        return batch_id in self._batches


@dataclass(frozen=True)
class LegacyColumn:
    """A legacy column: its own unit and the calibration batch it belongs to."""

    name: str
    unit: str
    batch_id: str


@dataclass(frozen=True)
class AbsoluteReading:
    """One raw absolute temperature reading of a legacy column."""

    batch_id: str
    value: RawValue
    column: Optional[str] = None


@dataclass(frozen=True)
class DifferenceRecord:
    """
    A raw temperature-difference column of one record.

    ``source_batch_ids`` is the ordered pair ``(high_batch, low_batch)``
    of the two endpoint readings.  ``None`` means the legacy data does
    not say where the endpoints come from; such a record cannot be
    migrated.  ``high`` and ``low`` hold the endpoint readings when the
    record is complete (required for cross-batch differences).
    """

    record_id: str
    raw_difference: RawValue
    source_batch_ids: Optional[tuple[str, str]] = None
    high: Optional[AbsoluteReading] = None
    low: Optional[AbsoluteReading] = None


def convert_absolute(reading: AbsoluteReading, registry: CalibrationRegistry) -> UFloat:
    """Convert one absolute reading to the unified unit."""
    params = registry.params(reading.batch_id)
    return params.slope * reading.value + params.zero


def convert_then_subtract(
    high: AbsoluteReading,
    low: AbsoluteReading,
    registry: CalibrationRegistry,
) -> UFloat:
    """Migration path A: convert both endpoints first, then subtract."""
    return convert_absolute(high, registry) - convert_absolute(low, registry)


def _check_endpoint_provenance(record: DifferenceRecord) -> tuple[str, str]:
    if record.source_batch_ids is None:
        msg = (
            f"difference record {record.record_id!r} does not declare its "
            "source batches; refusing to treat it as an independent reading"
        )
        raise ProvenanceError(msg)

    high_batch, low_batch = record.source_batch_ids
    if record.high is None or record.low is None:
        msg = f"difference record {record.record_id!r} is missing endpoint readings"
        raise ProvenanceError(msg)
    if record.high.batch_id != high_batch or record.low.batch_id != low_batch:
        msg = (
            f"endpoint batches of record {record.record_id!r} do not match "
            "its declared source batches"
        )
        raise ProvenanceError(msg)
    return high_batch, low_batch


def migrate_difference(
    record: DifferenceRecord, registry: CalibrationRegistry
) -> UFloat:
    """
    Migration path B: migrate the legacy difference column from the
    calibration parameters of its declared source batches.

    * Endpoints of the same batch: the shared zero cancels, so the
      migrated difference is simply the slope applied to the raw
      difference.
    * Endpoints of different batches: the zeros do not cancel and the
      scalar raw difference is insufficient.  The endpoint readings of
      the complete record are required and the result is built from
      them.
    * Records without declared source batches are refused instead of
      being fabricated into independent readings.
    """
    high_batch, low_batch = _check_endpoint_provenance(record)
    raw_delta: RawValue = record.raw_difference

    if high_batch == low_batch:
        params = registry.params(high_batch)
        return params.slope * raw_delta

    if record.high is None or record.low is None:
        msg = (
            f"cross-batch difference record {record.record_id!r} requires "
            "both endpoint readings because the calibration zeros do not "
            "cancel"
        )
        raise ProvenanceError(msg)
    return convert_then_subtract(record.high, record.low, registry)


class CrossCheckResult(NamedTuple):
    """Outcome of comparing the two migration paths on one complete record."""

    record_id: str
    converted_then_subtracted: UFloat
    migrated_from_source: UFloat
    nominal_difference: float
    std_difference: float
    consistent: bool


def cross_check_difference(
    record: DifferenceRecord,
    registry: CalibrationRegistry,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> CrossCheckResult:
    """
    Cross-check the two migration paths on one complete record.

    Path A converts the two endpoint readings and subtracts them.
    Path B migrates the difference column from its declared source.
    Both the nominal values and the propagated standard deviations
    must agree within the given tolerances.
    """
    if record.high is None or record.low is None:
        msg = (
            f"difference record {record.record_id!r} is not complete: "
            "both endpoint readings are required for cross-checking"
        )
        raise ProvenanceError(msg)

    path_a = convert_then_subtract(record.high, record.low, registry)
    path_b = migrate_difference(record, registry)

    nominal_difference = abs(path_a.nominal_value - path_b.nominal_value)
    std_difference = abs(path_a.std_dev - path_b.std_dev)
    consistent = isclose(
        path_a.nominal_value,
        path_b.nominal_value,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    ) and isclose(
        path_a.std_dev,
        path_b.std_dev,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )

    return CrossCheckResult(
        record_id=record.record_id,
        converted_then_subtracted=path_a,
        migrated_from_source=path_b,
        nominal_difference=nominal_difference,
        std_difference=std_difference,
        consistent=consistent,
    )
