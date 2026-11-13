# coding=utf-8

"""
Offline quality-control samples for derived metrics that print a zero
uncertainty.

These tests pin down the three distinct meanings of a displayed ``+/-0``:

* a genuine constant (no uncertain source reaches the metric);
* a same-source cancellation (uncertain inputs are present, but the same data
  lineage cancels itself and never moves);
* a stationary-point zero (the first derivative vanishes at the nominal value
  while higher-order terms still make the metric fluctuate).

Perturbations are applied to the independent leaf sources and linked
perturbations stay linked because every recipe rebuilds the whole calculation
from the same leaf argument.
"""

import json
from math import cos, isclose, sqrt

import pytest

from uncertainties import ufloat
from uncertainties.quality import (
    ZeroKind,
    active_sources,
    audit_zero,
    perturbation_ensemble,
)

try:
    import uncertainties.umath as um
except ImportError:  # pragma: no cover - umath ships with the package
    um = None


pytestmark = [
    pytest.mark.skipif(um is None, reason="uncertainties.umath required"),
    pytest.mark.filterwarnings(
        "ignore:Using UFloat objects with std_dev==0:UserWarning"
    ),
]


def value_at(audit, **deltas):
    """Return the ensemble value for the point with the given source deltas."""
    for point in audit.points:
        if all(
            isclose(point.deltas.get(name, 0.0), shift)
            for name, shift in deltas.items()
        ):
            return point.value
    raise AssertionError("no ensemble point with deltas %r" % deltas)


# (label, metric, recipe, expected kind, alert?, expected spread)
CLASSIFICATION_CASES = [
    (
        "plain float is a true constant",
        2.0,
        lambda: 2.0,
        ZeroKind.CONSTANT,
        False,
        0.0,
    ),
    (
        "exact zero-sigma variable is a true constant",
        ufloat(5.0, 0.0, tag="c"),
        lambda c: c,
        ZeroKind.CONSTANT,
        False,
        0.0,
    ),
    (
        "x - x is a same-source cancellation",
        (lambda x: x - x)(ufloat(0.5, 0.1, tag="x")),
        lambda x: x - x,
        ZeroKind.SAME_SOURCE_CANCELLATION,
        True,
        0.0,
    ),
    (
        "(2x+1)-(2x+1) cancels the same lineage",
        (lambda x: (2 * x + 1) - (2 * x + 1))(ufloat(0.5, 0.1, tag="x")),
        lambda x: (2 * x + 1) - (2 * x + 1),
        ZeroKind.SAME_SOURCE_CANCELLATION,
        True,
        0.0,
    ),
    (
        "x**2 - x**2 cancels at every order",
        (lambda x: x**2 - x**2)(ufloat(0.0, 0.1, tag="x")),
        lambda x: x**2 - x**2,
        ZeroKind.SAME_SOURCE_CANCELLATION,
        True,
        0.0,
    ),
    (
        "squaring a near-zero displacement is a stationary point",
        (lambda d: d**2)(ufloat(0.0, 0.05, tag="d")),
        lambda d: d**2,
        ZeroKind.STATIONARY_POINT,
        True,
        0.05**2,
    ),
    (
        "cubing at the origin is a stationary point",
        (lambda x: x**3)(ufloat(0.0, 0.1, tag="x")),
        lambda x: x**3,
        ZeroKind.STATIONARY_POINT,
        True,
        0.002,
    ),
    (
        "cos at zero is a stationary point",
        (lambda x: um.cos(x))(ufloat(0.0, 0.1, tag="x")),
        lambda x: um.cos(x),
        ZeroKind.STATIONARY_POINT,
        True,
        1.0 - cos(0.1),
    ),
]


@pytest.mark.parametrize(
    "label, metric, recipe, expected_kind, expected_alert, expected_spread",
    CLASSIFICATION_CASES,
    ids=[case[0] for case in CLASSIFICATION_CASES],
)
def test_zero_classification(
    label, metric, recipe, expected_kind, expected_alert, expected_spread
):
    audit = audit_zero(metric, recipe)

    assert audit.kind is expected_kind
    assert audit.is_alert is expected_alert
    assert isclose(audit.spread, expected_spread, rel_tol=1e-12, abs_tol=1e-15)

    if expected_kind is ZeroKind.CONSTANT:
        assert audit.active_source_tags == ()
        assert audit.points == ()
    else:
        # A misleading zero always traces back to at least one active source.
        assert len(audit.active_source_tags) >= 1


def test_squared_near_zero_displays_zero_but_really_moves():
    """The headline measurement case: squared near-zero displacement."""
    displacement = ufloat(0.0, 0.05, tag="d")
    squared = displacement**2

    # The regular (linear) report shows no uncertainty...
    assert squared.std_dev == 0.0

    audit = audit_zero(squared, lambda d: d**2)

    # ...but the offline audit proves the value is not actually constant.
    assert audit.kind is ZeroKind.STATIONARY_POINT
    assert audit.is_alert is True
    assert audit.nominal_value == 0.0
    assert isclose(value_at(audit, d=0.05), 0.0025)
    assert isclose(value_at(audit, d=-0.05), 0.0025)
    assert isclose(audit.spread, 0.0025)


def test_same_source_cancellation_is_invariant_under_all_perturbations():
    x = ufloat(0.5, 0.1, tag="x")
    audit = audit_zero(x - x, lambda value: value - value)

    assert audit.kind is ZeroKind.SAME_SOURCE_CANCELLATION
    assert len(audit.points) == 3  # nominal + the two +/- sigma corners
    assert all(point.value == 0.0 for point in audit.points)
    # The uncertain leaf is still visible, so this is not mistaken for a
    # genuine constant.
    assert audit.active_source_tags == ("x",)


def test_linked_perturbation_keeps_shared_lineage():
    """Reusing one leaf keeps the two operands perfectly correlated."""
    z = ufloat(1.0, 0.1, tag="z")

    direct = audit_zero(z - z, lambda value: value - value)
    assert direct.kind is ZeroKind.SAME_SOURCE_CANCELLATION

    # A separate affine object built from the same primitive still cancels.
    shared = audit_zero(z - (z + 0.0), lambda value: value - (value + 0.0))
    assert shared.kind is ZeroKind.SAME_SOURCE_CANCELLATION
    assert all(point.value == 0.0 for point in shared.points)


def test_independent_leaves_do_not_falsely_cancel():
    """Equal nominal values on distinct leaves keep a non-zero uncertainty."""
    u = ufloat(1.0, 0.1, tag="u")
    v = ufloat(1.0, 0.1, tag="v")

    audit = audit_zero(u - v, lambda a, b: a - b)

    assert audit.kind is ZeroKind.NONZERO
    assert audit.is_alert is False
    assert isclose(audit.displayed_std_dev, sqrt(2) * 0.1)


def test_partial_lineage_cancellation_leaves_the_other_source():
    """(x + z) - x cancels x but leaves z, so it is not a zero at all."""
    x = ufloat(2.0, 0.1, tag="x")
    z = ufloat(3.0, 0.2, tag="z")

    audit = audit_zero((x + z) - x, lambda a, b: (a + b) - a)

    assert audit.kind is ZeroKind.NONZERO
    assert audit.nominal_value == 3.0
    assert isclose(audit.displayed_std_dev, 0.2)


def test_multisource_stationary_metric_needs_single_axis_points():
    """a**2 + b**2 at the origin: joint corners alone all read 0.005."""
    a = ufloat(0.0, 0.05, tag="a")
    b = ufloat(0.0, 0.05, tag="b")

    audit = audit_zero(a**2 + b**2, lambda x, y: x**2 + y**2)

    assert audit.kind is ZeroKind.STATIONARY_POINT
    # Single-axis perturbations prove the value moves away from the nominal.
    assert isclose(value_at(audit, a=0.05, b=0.0), 0.0025)
    assert isclose(value_at(audit, a=0.0, b=-0.05), 0.0025)
    # The full joint corner reaches the largest value.
    assert isclose(value_at(audit, a=0.05, b=0.05), 0.005)
    assert isclose(audit.spread, 0.005)


def test_ensemble_contains_nominal_and_both_signs_per_source():
    x = ufloat(0.0, 0.1, tag="x")
    points = perturbation_ensemble(lambda value: value**3, [x])

    shifts = {point.deltas["x"] for point in points}
    assert shifts == {-0.1, 0.0, 0.1}


def test_audit_dict_is_json_serialisable():
    x = ufloat(0.0, 0.1, tag="x")
    audit = audit_zero(x**3, lambda value: value**3)

    payload = audit.as_dict()
    decoded = json.loads(json.dumps(payload))

    assert decoded["kind"] == "stationary_point"
    assert decoded["is_alert"] is True
    assert isclose(decoded["perturbation_spread"], 0.002)
    assert len(decoded["points"]) == len(audit.points)


def test_report_names_the_zero_kind_for_consumers():
    d = ufloat(0.0, 0.05, tag="d")
    text = audit_zero(d**2, lambda value: value**2).report()

    assert "stationary_point" in text
    assert "RAISED" in text


def test_explicit_sources_must_match_the_metric_leaves():
    u = ufloat(1.0, 0.1, tag="u")
    extra = ufloat(9.0, 0.2, tag="extra")

    with pytest.raises(ValueError, match="do not reach"):
        audit_zero(u - u, lambda value: value - value, sources=[u, extra])

    with pytest.raises(ValueError, match="not included in 'sources'"):
        audit_zero(u - u, lambda value: value - value, sources=[])


def test_active_sources_exclude_exact_leaves():
    exact = ufloat(1.0, 0.0, tag="exact")
    noisy = ufloat(1.0, 0.1, tag="noisy")

    assert active_sources(exact) == ()
    assert active_sources(noisy - noisy) == (noisy,)
    # A zero-sigma leaf does not turn a stationary metric into a constant.
    assert {id(leaf) for leaf in active_sources(noisy**2)} == {id(noisy)}
