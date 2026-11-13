# coding=utf-8

"""
Offline quality control for derived metrics that display a zero uncertainty.

This package propagates uncertainties with a first-order (linear) expansion.
The reported :attr:`AffineScalarFunc.std_dev` is the root-sum-square of the
first derivatives times the source standard deviations.  A derived metric can
therefore print ``+/-0`` for three very different reasons:

1. **True constant** -- the metric does not depend on any uncertain source, so
   it genuinely never moves.
2. **Same-source cancellation** -- uncertain sources are present, but the same
   lineage is subtracted from (or otherwise algebraically cancels) itself,
   e.g. ``x - x`` or ``x ** 2 - x ** 2``.  The result is exactly invariant at
   every order, even though each operand carries an uncertainty.
3. **Stationary point** -- the first derivative happens to be zero at the
   nominal value, but higher-order terms still make the metric fluctuate,
   e.g. ``x ** 2`` or ``umath.cos(x)`` evaluated at ``x = 0``.  The linear
   rule reports zero while the real value moves quadratically.

A displayed zero alone cannot distinguish these cases.  The tools in this
module re-evaluate a metric on an offline perturbation ensemble built from the
*independent leaf sources* and classify the zero.  Linked (same-source)
perturbations stay linked because :func:`audit_zero` rebuilds the whole
calculation through ``recipe`` for every ensemble member: the same leaf
argument is referenced everywhere it occurs, instead of being replaced by
independent copies.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

from uncertainties.core import (
    UFloat,
    Variable,
    nominal_value,
    std_dev,
    ufloat,
)

__all__ = [
    "ZeroKind",
    "PerturbationPoint",
    "ZeroAudit",
    "audit_zero",
    "perturbation_ensemble",
    "active_sources",
    "ZERO_DISPLAY_TOL",
]

# Default tolerance below which a displayed standard deviation, or a
# perturbation spread, is treated as exactly zero.
ZERO_DISPLAY_TOL = 1e-12

# Hard cap on the number of joint (full-factorial) corner evaluations, so that
# a metric with many sources does not trigger an exponential explosion.
MAX_JOINT_POINTS = 4096


class ZeroKind(Enum):
    """Classification of a metric whose displayed uncertainty is zero."""

    #: The metric depends on no uncertain source and never moves.
    CONSTANT = "true_constant"

    #: Uncertain sources exist but cancel algebraically (same lineage); the
    #: metric is invariant under every allowed perturbation.
    SAME_SOURCE_CANCELLATION = "same_source_cancellation"

    #: The first derivative is zero at the nominal point, but allowed
    #: perturbations move the value through higher-order terms.
    STATIONARY_POINT = "stationary_point"

    #: The displayed uncertainty is not zero, so there is nothing to audit.
    NONZERO = "nonzero_uncertainty"


@dataclass(frozen=True)
class PerturbationPoint:
    """One evaluation of the audited metric on the perturbation ensemble.

    ``deltas`` maps each leaf source tag to the signed shift applied to its
    nominal value (sources left at their nominal value are mapped to 0.0).
    ``value`` is the metric evaluated at that point.
    """

    deltas: dict
    value: float


@dataclass(frozen=True)
class ZeroAudit:
    """Result of auditing a metric that displays a zero uncertainty."""

    kind: ZeroKind
    is_alert: bool
    nominal_value: float
    displayed_std_dev: float
    active_source_tags: tuple
    tolerance: float
    spread: float
    points: tuple = field(default_factory=tuple)
    detail: str = ""

    @property
    def label(self):
        """Human-readable, report-friendly name of the classification."""
        return {
            ZeroKind.CONSTANT: "true constant",
            ZeroKind.SAME_SOURCE_CANCELLATION: "same-source cancellation",
            ZeroKind.STATIONARY_POINT: "stationary-point zero",
            ZeroKind.NONZERO: "non-zero uncertainty",
        }[self.kind]

    def as_dict(self):
        """Return a plain, JSON-serialisable summary of the audit."""
        return {
            "kind": self.kind.value,
            "label": self.label,
            "is_alert": self.is_alert,
            "nominal_value": self.nominal_value,
            "displayed_std_dev": self.displayed_std_dev,
            "active_source_tags": list(self.active_source_tags),
            "tolerance": self.tolerance,
            "perturbation_spread": self.spread,
            "detail": self.detail,
            "points": [
                {"deltas": dict(point.deltas), "value": point.value}
                for point in self.points
            ],
        }

    def report(self):
        """Return a compact, multi-line text block for report consumers."""
        lines = [
            "zero-uncertainty audit",
            "  kind            : {} ({})".format(self.kind.value, self.label),
            "  alert           : {}".format("RAISED" if self.is_alert else "none"),
            "  nominal value   : {:.12g}".format(self.nominal_value),
            "  displayed std   : {:.12g}".format(self.displayed_std_dev),
            "  active sources  : {}".format(
                ", ".join(self.active_source_tags) or "(none)"
            ),
            "  perturb spread  : {:.12g} (tol={:.3g})".format(
                self.spread, self.tolerance
            ),
            "  detail          : {}".format(self.detail),
        ]
        return "\n".join(lines)


def _leaf_specs(sources):
    """Return ``(tag, nominal_value, std_dev)`` specs for leaf sources."""
    specs = []
    for index, source in enumerate(sources):
        tag = getattr(source, "tag", None) or "source_{}".format(index)
        specs.append((tag, nominal_value(source), std_dev(source)))
    return specs


def _validate_sources(metric, leaves, sources):
    """Ensure explicit ``sources`` match the active leaves of ``metric``.

    Same-source linkage is represented by reusing the same leaf
    :class:`Variable` object.  Correlation built behind an opaque factory that
    hides its primitives cannot be perturbed through a public leaf, so it is
    rejected here with guidance rather than silently producing an empty
    ensemble.
    """
    leaf_ids = {id(leaf) for leaf in leaves}
    source_ids = {id(source) for source in sources}

    missing = leaf_ids - source_ids
    if missing:
        raise ValueError(
            "The metric depends on uncertain source variable(s) that are not "
            "included in 'sources'. Express linked inputs by reusing the same "
            "leaf Variable object in the recipe (e.g. 'lambda z: z - z') and "
            "pass that leaf in 'sources', instead of building correlation "
            "inside an opaque helper that hides its primitive variables."
        )

    unused = source_ids - leaf_ids
    if unused:
        raise ValueError(
            "'sources' contains variable(s) that do not reach the metric; "
            "pass only its independent leaf sources."
        )


def active_sources(metric):
    """Return the uncertain leaf :class:`Variable` objects of ``metric``.

    A leaf counts as active when its own standard deviation is strictly
    positive.  Coefficients that collapsed to zero (as in ``x - x``) still
    leave the leaf present in the linear part, so such a metric is *not*
    treated as a constant.
    """
    if not isinstance(metric, UFloat):
        return ()

    linear_part = metric._linear_part
    if not linear_part.expanded():
        linear_part.expand()

    return tuple(
        variable
        for variable in linear_part.linear_combo
        if isinstance(variable, Variable) and variable._std_dev > 0
    )


def perturbation_ensemble(recipe, sources, *, max_joint_points=MAX_JOINT_POINTS):
    """Evaluate ``recipe`` over an offline perturbation ensemble.

    ``sources`` are the independent leaf numbers with uncertainty (typically
    :class:`Variable` objects) that feed the metric.  ``recipe`` must accept
    exactly one fresh argument per source, in the same order, and rebuild the
    *entire* derived calculation (including any correlated intermediates).
    Rebuilding through ``recipe`` for every ensemble member is what keeps a
    shared lineage shared: referencing the same argument twice reproduces
    ``x - x`` exactly instead of silently substituting two independent draws.

    The ensemble always contains the nominal point and the two single-axis
    ``+/- sigma`` points of every active source (these are required to catch
    metrics such as ``a ** 2 + b ** 2`` at the origin, whose joint corners are
    all equal).  When affordable, all joint sign-combination corners are added.

    Return a tuple of :class:`PerturbationPoint`.
    """
    specs = _leaf_specs(sources)
    tags = [tag for tag, _, _ in specs]
    active_index = [i for i, (_, _, sigma) in enumerate(specs) if sigma > 0]

    # Each row is a signed, per-source shift expressed in absolute units.
    shift_rows = [[0.0] * len(specs)]

    for i in active_index:
        sigma = specs[i][2]
        for sign in (-1.0, 1.0):
            row = [0.0] * len(specs)
            row[i] = sign * sigma
            shift_rows.append(row)

    if active_index and 2 ** len(active_index) <= max_joint_points:
        for signs in itertools.product((-1.0, 1.0), repeat=len(active_index)):
            row = [0.0] * len(specs)
            for i, sign in zip(active_index, signs):
                row[i] = sign * specs[i][2]
            shift_rows.append(row)

    # For a single active source the joint corners repeat the single-axis
    # points; duplicate shifts are dropped so each ensemble point is unique.
    unique_rows = []
    seen = set()
    for row in shift_rows:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    points = []
    for shifts in unique_rows:
        fresh_inputs = [
            ufloat(nominal + shift, sigma, tag=tag)
            for (tag, nominal, sigma), shift in zip(specs, shifts)
        ]
        value = float(nominal_value(recipe(*fresh_inputs)))
        points.append(
            PerturbationPoint(
                deltas={tag: shift for tag, shift in zip(tags, shifts)},
                value=value,
            )
        )

    return tuple(points)


def audit_zero(metric, recipe, sources=None, *, tolerance=ZERO_DISPLAY_TOL):
    """Classify why ``metric`` displays a zero uncertainty.

    Parameters
    ----------
    metric:
        The derived number with uncertainty as produced by the regular
        calculation.  A plain Python number is accepted as a constant.
    recipe:
        Callable that rebuilds the metric from one fresh leaf value per
        source, in the same order as ``sources`` (see
        :func:`perturbation_ensemble`).  It must reproduce the same data
        lineage, including same-source references and any correlated
        intermediates.  Reusing the same argument inside ``recipe`` keeps a
        linked perturbation linked; replacing it with a newly created
        independent :func:`ufloat` breaks that relationship.
    sources:
        Sequence of independent leaf numbers with uncertainty fed into
        ``recipe`` and ``metric``.  When ``None`` (the default), the active
        leaf :class:`Variable` objects of ``metric`` are used.  Explicit
        sources are checked to match those leaves exactly.
    tolerance:
        Absolute tolerance: standard deviations and perturbation spreads at or
        below it are treated as exactly zero.

    Returns
    -------
    ZeroAudit
        A structured verdict.  ``is_alert`` is ``True`` for the two
        misleading zeros (same-source cancellation and stationary point) and
        ``False`` for genuine constants and for metrics that are not zero.

    Example
    -------
    >>> import warnings
    >>> warnings.simplefilter("ignore")
    >>> from uncertainties import ufloat
    >>> from uncertainties.quality import audit_zero
    >>> displacement = ufloat(0.0, 0.05, tag="d")
    >>> squared = displacement ** 2
    >>> audit = audit_zero(squared, lambda d: d ** 2)
    >>> audit.kind.value
    'stationary_point'
    >>> audit.is_alert
    True
    """
    displayed = float(std_dev(metric))
    nominal = float(nominal_value(metric))
    leaves = active_sources(metric)
    active_tags = tuple(
        getattr(leaf, "tag", None) or "leaf_{}".format(index)
        for index, leaf in enumerate(leaves)
    )

    if sources is None:
        sources = list(leaves)
    else:
        _validate_sources(metric, leaves, sources)

    if displayed > tolerance:
        return ZeroAudit(
            kind=ZeroKind.NONZERO,
            is_alert=False,
            nominal_value=nominal,
            displayed_std_dev=displayed,
            active_source_tags=active_tags,
            tolerance=tolerance,
            spread=0.0,
            points=(),
            detail=(
                "The linear uncertainty is non-zero, so no zero-uncertainty "
                "audit applies."
            ),
        )

    if not leaves:
        return ZeroAudit(
            kind=ZeroKind.CONSTANT,
            is_alert=False,
            nominal_value=nominal,
            displayed_std_dev=displayed,
            active_source_tags=(),
            tolerance=tolerance,
            spread=0.0,
            points=(),
            detail=(
                "No active uncertain source reaches this metric: the zero is "
                "a genuine, exact constant."
            ),
        )

    points = perturbation_ensemble(recipe, sources)
    values = [point.value for point in points]
    spread = max(values) - min(values)

    if spread <= tolerance:
        kind = ZeroKind.SAME_SOURCE_CANCELLATION
        detail = (
            "Active uncertain sources are present, but every linked "
            "perturbation leaves the value unchanged (spread <= tolerance). "
            "The zero comes from algebraic cancellation of the same data "
            "lineage, not from the inputs being exact."
        )
    else:
        kind = ZeroKind.STATIONARY_POINT
        detail = (
            "The first derivative vanishes at the nominal point, so linear "
            "propagation prints zero, yet allowed perturbations move the "
            "value (spread > tolerance) through higher-order terms. The "
            "metric really fluctuates."
        )

    return ZeroAudit(
        kind=kind,
        is_alert=True,
        nominal_value=nominal,
        displayed_std_dev=displayed,
        active_source_tags=active_tags,
        tolerance=tolerance,
        spread=spread,
        points=points,
        detail=detail,
    )
