"""Migrate supplier geotransform tuples with ambiguous coefficient order.

Suppliers deliver the six coefficients of a geotransform as bare tuples
with no column names.  Historical batches mix two orderings:

* the GDAL ``GetGeoTransform()`` order ``(c, a, b, f, d, e)``, and
* the ``Affine`` constructor order ``(a, b, c, d, e, f)``.

Within one batch the convention is fixed, but which one applies is
unknown.  Two kinds of evidence disambiguate a batch:

* ledger entries giving known world coordinates for a few pixels, and
* seam records stating that a pixel of one image and a pixel of another
  image occupy the same world location.

The migration solves each batch jointly against all evidence.  An image
is migrated to the standard ``Affine`` coefficient order only when every
feasible interpretation of its batch yields the same six coefficients.
The remaining images are kept pending together with one pixel whose
world coordinates distinguish the candidate interpretations.  Images
whose six numbers happen to make both orderings equivalent are migrated
like any other resolved image and never flagged as conflicts.
"""

from __future__ import annotations

from enum import Enum
from itertools import product
from typing import Iterable, Sequence

from attrs import define, field

import affine
from affine import Affine, AffineError

__all__ = [
    "Convention",
    "ImageRecord",
    "InconsistentEvidenceError",
    "LedgerEntry",
    "MigrationResult",
    "PendingImage",
    "SeamRecord",
    "interpret",
    "migrate",
]


class Convention(Enum):
    """Coefficient ordering convention of a bare six-number tuple."""

    AFFINE = "affine"
    GDAL = "gdal"


class InconsistentEvidenceError(AffineError):
    """No coefficient ordering satisfies all evidence of a batch."""


def interpret(coeffs: Sequence[float], convention: Convention) -> Affine:
    """Build a transform from a bare tuple under a given convention.

    Parameters
    ----------
    coeffs : sequence of float
        The six supplier numbers with no column names.
    convention : Convention
        ``Convention.GDAL`` for GDAL ``GetGeoTransform()`` order
        ``(c, a, b, f, d, e)``, ``Convention.AFFINE`` for the ``Affine``
        constructor order ``(a, b, c, d, e, f)``.

    Returns
    -------
    Affine
    """
    if convention is Convention.GDAL:
        return Affine.from_gdal(*coeffs)
    return Affine(*coeffs)


@define(frozen=True)
class ImageRecord:
    """One supplier image: a batch tag and six nameless coefficients.

    Parameters
    ----------
    image_id : str
        Unique identifier of the image.
    batch : str
        Identifier of the delivery batch.  All images of a batch share
        one coefficient ordering convention.
    coeffs : tuple of float
        The six coefficients, in whichever order the batch used.
    """

    image_id: str
    batch: str
    coeffs: tuple[float, ...] = field(converter=tuple)

    @coeffs.validator
    def _check_coeffs(self, attribute, value):
        if len(value) != 6:
            raise ValueError(f"Expected 6 coefficients, found {len(value)}")


@define(frozen=True)
class LedgerEntry:
    """A known world coordinate for one pixel of one image.

    Parameters
    ----------
    image_id : str
        Image the pixel belongs to.
    pixel : tuple of float
        Pixel coordinates ``(col, row)``.
    world : tuple of float
        Known world coordinates ``(x, y)`` of the pixel.
    """

    image_id: str
    pixel: tuple[float, float] = field(converter=tuple)
    world: tuple[float, float] = field(converter=tuple)


@define(frozen=True)
class SeamRecord:
    """A pair of pixels from two images that coincide in the world.

    Parameters
    ----------
    image_a, image_b : str
        The two images, possibly from different batches.
    pixel_a, pixel_b : tuple of float
        Pixel coordinates ``(col, row)`` in each image that map to the
        same world location.
    """

    image_a: str
    pixel_a: tuple[float, float] = field(converter=tuple)
    image_b: str
    pixel_b: tuple[float, float] = field(converter=tuple)


@define(frozen=True)
class PendingImage:
    """An image kept back because its coefficient order is ambiguous.

    Attributes
    ----------
    image_id : str
        Identifier of the image.
    batch : str
        Identifier of the delivery batch.
    pixel : tuple of float
        A pixel whose world coordinates differ between the candidate
        interpretations.
    world_by_convention : dict
        World coordinates of ``pixel`` under each feasible convention.
    candidates : dict
        Candidate transforms keyed by convention.
    """

    image_id: str
    batch: str
    pixel: tuple[float, float]
    world_by_convention: dict[Convention, tuple[float, float]]
    candidates: dict[Convention, Affine]


@define(frozen=True)
class MigrationResult:
    """Outcome of a migration.

    Attributes
    ----------
    migrated : dict
        Image id to ``Affine`` in standard constructor order, for images
        where every feasible interpretation yields the same six
        coefficients.
    pending : dict
        Image id to :class:`PendingImage` for images whose coefficient
        order remains ambiguous.
    conventions : dict
        Batch id to the frozenset of conventions consistent with all
        evidence.
    """

    migrated: dict[str, Affine]
    pending: dict[str, PendingImage]
    conventions: dict[str, frozenset[Convention]]


_PROBE_PIXELS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))


def _discriminating_pixel(transforms: list[Affine]) -> tuple[float, float]:
    """Return a pixel where the candidate transforms disagree the most."""
    best_pixel = _PROBE_PIXELS[0]
    best_spread = -1.0
    for pixel in _PROBE_PIXELS:
        points = [transform @ pixel for transform in transforms]
        spread = max(
            max(abs(p[0] - q[0]), abs(p[1] - q[1]))
            for i, p in enumerate(points)
            for q in points[i + 1 :]
        )
        if spread > best_spread:
            best_pixel = pixel
            best_spread = spread
    return best_pixel


def migrate(
    images: Iterable[ImageRecord],
    ledger: Iterable[LedgerEntry] = (),
    seams: Iterable[SeamRecord] = (),
    precision: float | None = None,
) -> MigrationResult:
    """Migrate supplier tuples to standard ``Affine`` coefficient order.

    Every batch is solved jointly against the ledger entries and seam
    records touching it.  Seam records may link batches, in which case
    the linked batches are solved together.  An image is migrated only
    when all conventions consistent with the evidence produce the same
    six coefficients for it; otherwise it is kept pending with one
    pixel that distinguishes the candidate interpretations.

    Parameters
    ----------
    images : iterable of ImageRecord
        The images to migrate.
    ledger : iterable of LedgerEntry, optional
        Known world coordinates for a few pixels.
    seams : iterable of SeamRecord, optional
        Pixel pairs of two images that coincide in the world.
    precision : float, optional
        Absolute tolerance for world coordinate comparisons.  Defaults
        to ``affine.EPSILON``.

    Returns
    -------
    MigrationResult

    Raises
    ------
    ValueError
        If image ids are duplicated or evidence references unknown
        images.
    InconsistentEvidenceError
        If no convention satisfies all evidence of a batch.
    """
    if precision is None:
        precision = affine.EPSILON

    ledger = list(ledger)
    seams = list(seams)

    by_id: dict[str, ImageRecord] = {}
    for image in images:
        if image.image_id in by_id:
            raise ValueError(f"Duplicate image id {image.image_id!r}")
        by_id[image.image_id] = image

    for entry in ledger:
        if entry.image_id not in by_id:
            raise ValueError(f"Ledger entry references unknown image {entry.image_id!r}")
    for seam in seams:
        for image_id in (seam.image_a, seam.image_b):
            if image_id not in by_id:
                raise ValueError(f"Seam record references unknown image {image_id!r}")

    batches: dict[str, list[str]] = {}
    for image in by_id.values():
        batches.setdefault(image.batch, []).append(image.image_id)

    parent = {batch: batch for batch in batches}

    def find(batch: str) -> str:
        while parent[batch] != batch:
            parent[batch] = parent[parent[batch]]
            batch = parent[batch]
        return batch

    for seam in seams:
        root_a = find(by_id[seam.image_a].batch)
        root_b = find(by_id[seam.image_b].batch)
        if root_a != root_b:
            parent[root_a] = root_b

    components: dict[str, list[str]] = {}
    for batch in batches:
        components.setdefault(find(batch), []).append(batch)

    candidates = {
        image.image_id: {
            convention: interpret(image.coeffs, convention)
            for convention in Convention
        }
        for image in by_id.values()
    }

    def consistent(assignment: dict[str, Convention], entries, links) -> bool:
        for entry in entries:
            transform = candidates[entry.image_id][
                assignment[by_id[entry.image_id].batch]
            ]
            wx, wy = transform @ entry.pixel
            if abs(wx - entry.world[0]) >= precision or abs(wy - entry.world[1]) >= precision:
                return False
        for seam in links:
            transform_a = candidates[seam.image_a][
                assignment[by_id[seam.image_a].batch]
            ]
            transform_b = candidates[seam.image_b][
                assignment[by_id[seam.image_b].batch]
            ]
            wa = transform_a @ seam.pixel_a
            wb = transform_b @ seam.pixel_b
            if abs(wa[0] - wb[0]) >= precision or abs(wa[1] - wb[1]) >= precision:
                return False
        return True

    feasible_conventions: dict[str, frozenset[Convention]] = {}
    for component_batches in components.values():
        component = set(component_batches)
        entries = [e for e in ledger if by_id[e.image_id].batch in component]
        links = [s for s in seams if by_id[s.image_a].batch in component]
        assignments = (
            dict(zip(component_batches, combo))
            for combo in product(Convention, repeat=len(component_batches))
        )
        feasible = [a for a in assignments if consistent(a, entries, links)]
        if not feasible:
            raise InconsistentEvidenceError(
                "No coefficient ordering satisfies all evidence of "
                f"batches {sorted(component)!r}"
            )
        for batch in component_batches:
            feasible_conventions[batch] = frozenset(
                assignment[batch] for assignment in feasible
            )

    migrated: dict[str, Affine] = {}
    pending: dict[str, PendingImage] = {}
    for image in by_id.values():
        transforms = {
            convention: candidates[image.image_id][convention]
            for convention in feasible_conventions[image.batch]
        }
        if len(set(transforms.values())) == 1:
            migrated[image.image_id] = next(iter(transforms.values()))
        else:
            pixel = _discriminating_pixel(list(transforms.values()))
            pending[image.image_id] = PendingImage(
                image_id=image.image_id,
                batch=image.batch,
                pixel=pixel,
                world_by_convention={
                    convention: transform @ pixel
                    for convention, transform in transforms.items()
                },
                candidates=transforms,
            )

    return MigrationResult(
        migrated=migrated, pending=pending, conventions=feasible_conventions
    )
