.. index:: quality control

==============================
Zero-Uncertainty Quality Checks
==============================

.. module:: uncertainties.quality

This package propagates uncertainties with a **first-order (linear)**
expansion.  The reported :attr:`~uncertainties.UFloat.std_dev` is the
root-sum-square of the first derivatives times the source standard
deviations.  A derived metric can therefore print ``+/-0`` for three
different reasons, and a displayed zero alone cannot tell them apart:

1. **True constant** -- the metric depends on no uncertain source and
   genuinely never moves.
2. **Same-source cancellation** -- uncertain sources are present, but the
   same data lineage cancels itself (e.g. ``x - x``).  The value is exactly
   invariant, at every order, even though each operand carries an
   uncertainty.
3. **Stationary point** -- the first derivative is zero at the nominal value,
   but higher-order terms still make the metric fluctuate (e.g. ``x ** 2`` or
   ``cos(x)`` at ``x = 0``).  The linear rule reports zero while the real
   value moves quadratically.

The :func:`audit_zero` function re-evaluates a metric on an offline
perturbation ensemble built from the independent leaf sources and returns a
:class:`ZeroAudit` verdict that distinguishes these cases.

Basic usage
===========

``recipe`` rebuilds the whole calculation from one fresh leaf value per
source.  Reusing the same argument inside ``recipe`` is what keeps a linked
(same-source) perturbation linked:

>>> from uncertainties import ufloat
>>> from uncertainties.quality import audit_zero
>>> d = ufloat(0.0, 0.05, tag="d")
>>> audit_zero(d ** 2, lambda value: value ** 2).kind.value
'stationary_point'
>>> x = ufloat(0.5, 0.1, tag="x")
>>> audit_zero(x - x, lambda value: value - value).kind.value
'same_source_cancellation'
>>> audit_zero(2.0, lambda: 2.0).kind.value
'true_constant'
>>> audit_zero(d ** 2, lambda value: value ** 2).is_alert
True

Only the two misleading zeros raise an alert; genuine constants and metrics
with a non-zero uncertainty do not.

QC sample catalog
=================

The table lists achievable inputs and the metric values observed on the
perturbation ensemble.  The spread is the difference between the largest and
smallest perturbed value; a zero spread with active sources is the signature
of an algebraic cancellation, whereas a non-zero spread with a displayed
standard deviation of zero is the signature of a stationary point.

.. list-table::
   :header-rows: 1
   :widths: 18 22 12 12 22 14

   * - Category
     - Input / recipe
     - Nominal value
     - Displayed std
     - Perturbed value (corner)
     - Perturbation spread
   * - True constant
     - ``2.0``
     - 2.0
     - 0
     - 2.0 (no sources)
     - 0
   * - Same-source subtract
     - ``x - x``, x = 0.5 +/- 0.1
     - 0.0
     - 0
     - 0.0 at x = 0.4 and x = 0.6
     - 0
   * - Same-source power
     - ``x**2 - x**2``, x = 0 +/- 0.1
     - 0.0
     - 0
     - 0.0 at x = +/- 0.1
     - 0
   * - Stationary (square)
     - ``d**2``, d = 0 +/- 0.05
     - 0.0
     - 0
     - 0.0025 at d = +/- 0.05
     - 0.0025
   * - Stationary (cube)
     - ``x**3``, x = 0 +/- 0.1
     - 0.0
     - 0
     - +/- 0.001 at x = +/- 0.1
     - 0.002
   * - Stationary (cosine)
     - ``cos(x)``, x = 0 +/- 0.1
     - 1.0
     - 0
     - 0.995004 at x = +/- 0.1
     - ~0.004996
   * - Stationary (2 sources)
     - ``a**2 + b**2``, a, b = 0 +/- 0.05
     - 0.0
     - 0
     - 0.0025 single-axis; 0.005 joint corner
     - 0.005

The two "stationary" rows are the alerting cases a report consumer should
treat as *a real fluctuation hidden behind a zero*, not an exact value.

Keeping linked perturbations linked
====================================

Correlation is represented by reusing the same leaf :class:`~uncertainties.Variable`
object.  Because ``recipe`` rebuilds the calculation for every ensemble
member, referencing the same argument twice reproduces an identity such as
``x - x`` exactly.  Replacing it with a newly created, independent
:func:`~uncertainties.ufloat` would break that relationship.  Equal nominal
values on *distinct* leaves do not cancel and keep a non-zero uncertainty:

>>> u = ufloat(1.0, 0.1, tag="u")
>>> v = ufloat(1.0, 0.1, tag="v")
>>> audit_zero(u - v, lambda a, b: a - b).kind.value
'nonzero_uncertainty'

Multi-source stationary metrics
================================

For ``a ** 2 + b ** 2`` evaluated at the origin, every joint sign-combination
corner happens to read the same value (0.005), so the ensemble also includes
single-axis ``+/- sigma`` points.  These move the metric to 0.0025 and expose
the hidden second-order fluctuation, which joint corners alone would miss.

API
===

.. autoclass:: ZeroKind
   :members:

.. autoclass:: ZeroAudit
   :members:

.. autofunction:: audit_zero

.. autofunction:: perturbation_ensemble

.. autofunction:: active_sources
