"""Tests for batch migration of ambiguous supplier geotransform tuples."""

import pytest

from affine import Affine
from affine.migration import (
    Convention,
    ImageRecord,
    InconsistentEvidenceError,
    LedgerEntry,
    MigrationResult,
    PendingImage,
    SeamRecord,
    interpret,
    migrate,
)


def test_interpret_gdal_order():
    coeffs = (100.0, 2.0, 0.0, 200.0, 0.0, -2.0)
    assert interpret(coeffs, Convention.GDAL) == Affine(2, 0, 100, 0, -2, 200)


def test_interpret_affine_order():
    coeffs = (2.0, 0.0, 100.0, 0.0, -2.0, 200.0)
    assert interpret(coeffs, Convention.AFFINE) == Affine(2, 0, 100, 0, -2, 200)


def test_ledger_identifies_gdal_batch():
    images = [
        ImageRecord("img1", "batch1", (100.0, 2.0, 0.0, 200.0, 0.0, -2.0)),
        ImageRecord("img2", "batch1", (300.0, 2.0, 0.0, 400.0, 0.0, -2.0)),
    ]
    ledger = [LedgerEntry("img1", (0.0, 0.0), (100.0, 200.0))]
    result = migrate(images, ledger)
    assert result.migrated["img1"] == Affine(2, 0, 100, 0, -2, 200)
    assert result.migrated["img2"] == Affine(2, 0, 300, 0, -2, 400)
    assert not result.pending
    assert result.conventions == {"batch1": frozenset({Convention.GDAL})}


def test_ledger_identifies_affine_batch():
    images = [ImageRecord("img1", "batch1", (2.0, 0.0, 100.0, 0.0, -2.0, 200.0))]
    ledger = [LedgerEntry("img1", (0.0, 0.0), (100.0, 200.0))]
    result = migrate(images, ledger)
    assert result.migrated["img1"] == Affine(2, 0, 100, 0, -2, 200)
    assert not result.pending
    assert result.conventions == {"batch1": frozenset({Convention.AFFINE})}


def test_ambiguous_batch_keeps_pending_with_discriminating_pixel():
    images = [ImageRecord("img1", "batch1", (0.0, 1.0, 0.0, 0.0, 0.0, 1.0))]
    result = migrate(images)
    assert not result.migrated
    pending = result.pending["img1"]
    assert isinstance(pending, PendingImage)
    world_affine = pending.world_by_convention[Convention.AFFINE]
    world_gdal = pending.world_by_convention[Convention.GDAL]
    assert world_affine != world_gdal
    assert pending.candidates[Convention.AFFINE] @ pending.pixel == world_affine
    assert pending.candidates[Convention.GDAL] @ pending.pixel == world_gdal
    assert result.conventions == {
        "batch1": frozenset({Convention.AFFINE, Convention.GDAL})
    }


def test_equivalent_interpretations_are_not_conflicts():
    # t0 == t1 == t2 and t3 == t4 == t5: both orderings give the same
    # six coefficients, so the image migrates even with no evidence.
    images = [ImageRecord("img1", "batch1", (1.0, 1.0, 1.0, 5.0, 5.0, 5.0))]
    result = migrate(images)
    assert result.migrated["img1"] == Affine(1, 1, 1, 5, 5, 5)
    assert not result.pending


def test_seam_within_batch_fixes_convention():
    images = [
        ImageRecord("img1", "batch1", (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
        ImageRecord("img2", "batch1", (5.0, 1.0, 0.0, 5.0, 0.0, 1.0)),
    ]
    seams = [SeamRecord("img1", (0.0, 0.0), "img2", (0.0, 0.0))]
    result = migrate(images, seams=seams)
    assert result.migrated["img1"] == Affine(0, 1, 0, 0, 0, 1)
    assert result.migrated["img2"] == Affine(5, 1, 0, 5, 0, 1)
    assert not result.pending
    assert result.conventions == {"batch1": frozenset({Convention.AFFINE})}


def test_seam_across_batches_links_conventions():
    images = [
        ImageRecord("img1", "batch1", (100.0, 2.0, 0.0, 200.0, 0.0, -2.0)),
        ImageRecord("img2", "batch2", (0.0, 3.0, 50.0, 0.0, 4.0, 60.0)),
    ]
    # The ledger entry forces batch1 to GDAL order; the seam then
    # propagates the decision and forces batch2 to Affine order.
    ledger = [LedgerEntry("img1", (0.0, 0.0), (100.0, 200.0))]
    seams = [SeamRecord("img1", (-25.0, 70.0), "img2", (0.0, 0.0))]
    result = migrate(images, ledger, seams)
    assert result.migrated["img1"] == Affine(2, 0, 100, 0, -2, 200)
    assert result.migrated["img2"] == Affine(0, 3, 50, 0, 4, 60)
    assert not result.pending
    assert result.conventions == {
        "batch1": frozenset({Convention.GDAL}),
        "batch2": frozenset({Convention.AFFINE}),
    }


def test_batches_without_links_are_solved_independently():
    images = [
        ImageRecord("img1", "batch1", (100.0, 2.0, 0.0, 200.0, 0.0, -2.0)),
        ImageRecord("img2", "batch2", (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    ]
    ledger = [LedgerEntry("img1", (0.0, 0.0), (100.0, 200.0))]
    result = migrate(images, ledger)
    assert result.migrated["img1"] == Affine(2, 0, 100, 0, -2, 200)
    assert set(result.pending) == {"img2"}


def test_contradictory_evidence_raises():
    images = [ImageRecord("img1", "batch1", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))]
    ledger = [LedgerEntry("img1", (0.0, 0.0), (9.0, 9.0))]
    with pytest.raises(InconsistentEvidenceError):
        migrate(images, ledger)


def test_duplicate_image_id_raises():
    images = [
        ImageRecord("img1", "batch1", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
        ImageRecord("img1", "batch1", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    ]
    with pytest.raises(ValueError, match="Duplicate image id"):
        migrate(images)


def test_unknown_image_in_evidence_raises():
    images = [ImageRecord("img1", "batch1", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))]
    with pytest.raises(ValueError, match="unknown image"):
        migrate(images, [LedgerEntry("nope", (0.0, 0.0), (0.0, 0.0))])
    with pytest.raises(ValueError, match="unknown image"):
        migrate(images, seams=[SeamRecord("img1", (0.0, 0.0), "nope", (0.0, 0.0))])


def test_image_record_requires_six_coefficients():
    with pytest.raises(ValueError, match="Expected 6 coefficients"):
        ImageRecord("img1", "batch1", (1.0, 0.0, 0.0))


def test_result_type():
    result = migrate([ImageRecord("img1", "batch1", (1.0, 1.0, 1.0, 5.0, 5.0, 5.0))])
    assert isinstance(result, MigrationResult)
