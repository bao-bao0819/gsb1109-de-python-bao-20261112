# coding=utf-8

"""
Main module for the uncertainties package, with internal functions.
"""

# The idea behind this module is to replace the result of mathematical
# operations by a local approximation of the defining function.  For
# example, sin(0.2+/-0.01) becomes the affine function
# (AffineScalarFunc object) whose nominal value is sin(0.2) and
# whose variations are given by sin(0.2+delta) = 0.98...*delta.
# Uncertainties can then be calculated by using this local linear
# approximation of the original function.

from __future__ import division  # Many analytical derivatives depend on this

from builtins import str, zip, range, object
import functools
from math import sqrt, isfinite, fsum, isnan  # Optimization: no attribute look-up
from warnings import warn

import copy
import collections

from uncertainties.formatting import format_ufloat
from uncertainties.parsing import str_to_number_with_uncert
from . import ops
from uncertainties.ops import (
    _wrap,
    set_doc,
    nan_if_exception,
    modified_operators,
    modified_ops_with_reflection,
)

# Attributes that are always exported (some other attributes are
# exported only if the NumPy module is available...):
__all__ = [
    # All sub-modules and packages are not imported by default,
    # in particular because NumPy might be unavailable.
    "ufloat",  # Main function: returns a number with uncertainty
    "ufloat_fromstr",  # Important function: returns a number with uncertainty
    # Uniform access to nominal values and standard deviations:
    "nominal_value",
    "std_dev",
    # Utility functions (more are exported if NumPy is present):
    "covariance_matrix",
    # Class for testing whether an object is a number with
    # uncertainty.  Not usually created by users (except through the
    # Variable subclass), but possibly manipulated by external code
    # ['derivatives()' method, etc.].
    "UFloat",
    "Variable",
    # Wrapper for allowing non-pure-Python function to handle
    # quantitities with uncertainties:
    "wrap",
    # used internally and will be removed by linter if not here
    "nan_if_exception",
    "modified_operators",
    "modified_ops_with_reflection",
    "correlated_values",
    "correlated_values_norm",
    "correlation_matrix",
    "CovarianceConflict",
    "CovarianceCompletion",
    "covariance_completion_bounds",
    "partial_covariance_bounds",
    "covariance_bounds",
    "linear_combination_variance_bounds",
    "partial_covariance_variance_bounds",
    "variance_bounds",
]

###############################################################################
## Definitions that depend on the availability of NumPy:


try:
    import numpy
except ImportError:
    numpy = None


def correlated_values(nom_values, covariance_mat, tags=None):
    """
    Return numbers with uncertainties (AffineScalarFunc objects)
    that correctly reproduce the given covariance matrix, and have
    the given (float) values as their nominal value.

    The correlated_values_norm() function returns the same result,
    but takes a correlation matrix instead of a covariance matrix.

    The list of values and the covariance matrix must have the
    same length, and the matrix must be a square (symmetric) one.

    The numbers with uncertainties returned depend on newly
    created, independent variables (Variable objects).

    nom_values -- sequence with the nominal (real) values of the
    numbers with uncertainties to be returned.

    covariance_mat -- full covariance matrix of the returned numbers with
    uncertainties. For example, the first element of this matrix is the
    variance of the first number with uncertainty. This matrix must be a
    NumPy array-like (list of lists, NumPy array, etc.).

    tags -- if 'tags' is not None, it must list the tag of each new
    independent variable.

    This function raises NotImplementedError if numpy cannot be
    imported.
    """
    if numpy is None:
        msg = (
            "uncertainties was not able to import numpy so "
            "correlated_values is unavailable."
        )
        raise NotImplementedError(msg)
    # !!! It would in principle be possible to handle 0 variance
    # variables by first selecting the sub-matrix that does not contain
    # such variables (with the help of numpy.ix_()), and creating
    # them separately.

    std_devs = numpy.sqrt(numpy.diag(covariance_mat))

    # For numerical stability reasons, we go through the correlation
    # matrix, because it is insensitive to any change of scale in the
    # quantities returned. However, care must be taken with 0 variance
    # variables: calculating the correlation matrix cannot be simply done
    # by dividing by standard deviations. We thus use specific
    # normalization values, with no null value:
    norm_vector = std_devs.copy()
    norm_vector[norm_vector == 0] = 1

    return correlated_values_norm(
        # !! The following zip() is a bit suboptimal: correlated_values()
        # separates back the nominal values and the standard deviations:
        list(zip(nom_values, std_devs)),
        covariance_mat / norm_vector / norm_vector[:, numpy.newaxis],
        tags,
    )


def correlated_values_norm(values_with_std_dev, correlation_mat, tags=None):
    """
    Return correlated values like correlated_values(), but takes
    instead as input:

    - nominal (float) values along with their standard deviation, and
    - a correlation matrix (i.e. a normalized covariance matrix).

    values_with_std_dev -- sequence of (nominal value, standard
    deviation) pairs. The returned, correlated values have these
    nominal values and standard deviations.

    correlation_mat -- correlation matrix between the given values, except
    that any value with a 0 standard deviation must have its correlations
    set to 0, with a diagonal element set to an arbitrary value (something
    close to 0-1 is recommended, for a better numerical precision).  When
    no value has a 0 variance, this is the covariance matrix normalized by
    standard deviations, and thus a symmetric matrix with ones on its
    diagonal.  This matrix must be an NumPy array-like (list of lists,
    NumPy array, etc.).

    tags -- like for correlated_values().

    This function raises NotImplementedError if numpy cannot be
    imported.
    """
    if numpy is None:
        msg = (
            "uncertainties was not able to import numpy so "
            "correlated_values_norm is unavailable."
        )
        raise NotImplementedError(msg)

    # If no tags were given, we prepare tags for the newly created
    # variables:
    if tags is None:
        tags = (None,) * len(values_with_std_dev)

    (nominal_values, std_devs) = numpy.transpose(values_with_std_dev)

    # We diagonalize the correlation matrix instead of the
    # covariance matrix, because this is generally more stable
    # numerically. In fact, the covariance matrix can have
    # coefficients with arbitrary values, through changes of units
    # of its input variables. This creates numerical instabilities.
    #
    # The covariance matrix is diagonalized in order to define
    # the independent variables that model the given values:
    (variances, transform) = numpy.linalg.eigh(correlation_mat)

    # Numerical errors might make some variances negative: we set
    # them to zero:
    variances[variances < 0] = 0.0

    # Creation of new, independent variables:

    # We use the fact that the eigenvectors in 'transform' are
    # special: 'transform' is unitary: its inverse is its transpose:

    variables = tuple(
        # The variables represent "pure" uncertainties:
        Variable(0, sqrt(variance), tag)
        for (variance, tag) in zip(variances, tags)
    )

    # The coordinates of each new uncertainty as a function of the
    # new variables must include the variable scale (standard deviation):
    transform *= std_devs[:, numpy.newaxis]

    # Representation of the initial correlated values:
    values_funcs = tuple(
        AffineScalarFunc(value, LinearCombination(dict(zip(variables, coords))))
        for (coords, value) in zip(transform, nominal_values)
    )

    return values_funcs


def correlation_matrix(nums_with_uncert):
    """
    Return the correlation matrix of the given sequence of
    numbers with uncertainties, as a NumPy array of floats.

    This function raises NotImplementedError if numpy cannot be
    imported.
    """
    if numpy is None:
        msg = (
            "uncertainties was not able to import numpy so "
            "correlation_matrix is unavailable."
        )
        raise NotImplementedError(msg)

    cov_mat = numpy.array(covariance_matrix(nums_with_uncert))

    std_devs = numpy.sqrt(cov_mat.diagonal())

    return cov_mat / std_devs / std_devs[numpy.newaxis].T


class CovarianceConflict(ValueError):
    """
    Raised when the known entries of a partial covariance matrix cannot
    belong to any positive-semidefinite covariance matrix.

    The offending columns are available as ``conflict_columns`` (a tuple
    of column-label tuples), ``conflicts`` (a tuple of index tuples) and
    ``reasons``.  A diagonal entry is reported as a one-column conflict;
    a pair of incompatible off-diagonal entries is reported as the
    corresponding two-column conflict.  If a fully specified 3x3 matrix
    has a negative determinant, the conflict involves all three columns.
    """

    def __init__(self, conflicts, columns=None):
        if columns is None:
            columns = (0, 1, 2)
        self.columns = tuple(columns)
        self.conflicts = tuple(tuple(indices) for indices, _reason in conflicts)
        self.reasons = tuple(reason for _indices, reason in conflicts)
        self.conflict_columns = tuple(
            tuple(self.columns[index] for index in indices)
            for indices in self.conflicts
        )
        described = ", ".join(
            "%s (%s)" % (labels, reason)
            for labels, reason in zip(self.conflict_columns, self.reasons)
        )
        super().__init__("Conflicting covariance entries: " + described)


def _covariance_missing(value):
    """Return True for the supported spellings of a missing entry."""

    if value is None:
        return True
    try:
        return bool(isnan(value))
    except TypeError:
        return False


def _covariance_tolerance(first, second, rtol, atol):
    scale = max(abs(first), abs(second))
    return atol + rtol * scale


def _covariance_rank(matrix):
    """
    Rank of a small matrix, with a scale-aware tolerance.

    This avoids requiring NumPy for the diagnostics attached to a
    covariance completion.
    """

    rows = [[float(value) for value in row] for row in matrix]
    scale = max(abs(value) for row in rows for value in row)
    if scale == 0:
        return 0
    tolerance = 1e-9 * scale
    rank = 0
    for column in range(len(rows)):
        pivot = max(range(rank, len(rows)), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) <= tolerance:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        for row in range(len(rows)):
            if row == rank:
                continue
            factor = rows[row][column] / pivot_value
            if factor:
                for other_column in range(column, len(rows)):
                    rows[row][other_column] -= factor * rows[rank][other_column]
        rank += 1
    return rank


class CovarianceCompletion:
    """
    Result of :func:`covariance_completion_bounds`.

    The attributes most useful to callers are:

    ``completion_interval``
        The closed interval of values that the missing off-diagonal
        covariance can take while the completed 3x3 matrix remains
        positive semidefinite.

    ``variance_min`` and ``variance_max``
        The attainable extrema of the variance of the requested linear
        combination.  They are ``None`` when no weights were supplied.

    ``completion_at_min`` and ``completion_at_max``
        The missing covariance value that realizes each variance
        extremum.  When the variance does not depend on the missing
        entry, the two interval endpoints are reported.  For a fully
        specified matrix these are ``None``.

    ``covariance_at_min`` and ``covariance_at_max``
        The corresponding fully completed 3x3 covariance matrices.

    ``missing_pair``
        The pair of zero-based indices whose covariance was missing, or
        ``None`` for a fully specified matrix.

    ``completion_interval`` is ``(None, None)`` for a fully specified
    matrix because there is nothing to complete.
    """

    __slots__ = (
        "columns",
        "missing_pair",
        "completion_interval",
        "weights",
        "variance_min",
        "variance_max",
        "completion_at_min",
        "completion_at_max",
        "rank_lower",
        "rank_upper",
        "is_fixed",
        "is_constant",
        "lower",
        "upper",
        "_variances",
        "_known_covariances",
        "_atol",
        "_rtol",
    )

    def __init__(
        self,
        columns,
        missing_pair,
        completion_interval,
        variances,
        known_covariances,
        weights=None,
        variance_min=None,
        variance_max=None,
        completion_at_min=None,
        completion_at_max=None,
        rank_lower=0,
        rank_upper=0,
        is_fixed=False,
        is_constant=False,
        atol=0.0,
        rtol=1e-12,
    ):
        self.columns = tuple(columns)
        self.missing_pair = missing_pair
        self.completion_interval = tuple(completion_interval)
        self._variances = tuple(variances)
        self._known_covariances = dict(known_covariances)
        self.weights = None if weights is None else tuple(weights)
        self.lower, self.upper = completion_interval
        self.variance_min = variance_min
        self.variance_max = variance_max
        self.completion_at_min = completion_at_min
        self.completion_at_max = completion_at_max
        self.rank_lower = rank_lower
        self.rank_upper = rank_upper
        self.is_fixed = is_fixed
        self.is_constant = is_constant
        self._atol = atol
        self._rtol = rtol
        for name in (
            "lower",
            "upper",
            "variance_min",
            "variance_max",
            "completion_at_min",
            "completion_at_max",
        ):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError("covariance bounds must be finite")

    @property
    def variance_interval(self):
        """Attainable ``(minimum, maximum)`` variance, or ``None``."""

        if self.variance_min is None:
            return None
        return (self.variance_min, self.variance_max)

    @property
    def completion_lower(self):
        """Lower endpoint of the feasible missing covariance."""

        return self.completion_interval[0]

    @property
    def completion_upper(self):
        """Upper endpoint of the feasible missing covariance."""

        return self.completion_interval[1]

    @property
    def rank_at_min(self):
        """Rank of the completion that gives the minimum variance."""

        if self.weights is None:
            return self.rank_lower
        if self.completion_at_min == self.completion_lower:
            return self.rank_lower
        return self.rank_upper

    @property
    def rank_at_max(self):
        """Rank of the completion that gives the maximum variance."""

        if self.weights is None:
            return self.rank_upper
        if self.completion_at_max == self.completion_upper:
            return self.rank_upper
        return self.rank_lower

    @property
    def rank_deficient_at_min(self):
        return self.rank_at_min < 3

    @property
    def rank_deficient_at_max(self):
        return self.rank_at_max < 3

    def completed_covariance(self, covariance=None):
        """
        Return the full covariance matrix for a chosen missing entry.

        If ``covariance`` is omitted, the lower endpoint of the feasible
        interval is used.  The returned matrix is a list of lists.
        """

        if covariance is None and self.missing_pair is not None:
            covariance = self.completion_lower

        matrix = [
            [self._variances[row] if row == column else 0.0 for column in range(3)]
            for row in range(3)
        ]
        for (row, column), value in self._known_covariances.items():
            matrix[row][column] = value
            matrix[column][row] = value
        if self.missing_pair is not None:
            row, column = self.missing_pair
            covariance = float(covariance)
            matrix[row][column] = covariance
            matrix[column][row] = covariance
        return matrix

    def variance_for(self, covariance):
        """Variance of the requested combination for a chosen completion."""

        if self.weights is None:
            raise ValueError("Variance bounds require a weight for each column.")

        total = fsum(
            weight * weight * variance
            for weight, variance in zip(self.weights, self._variances)
        )
        total += 2 * fsum(
            self.weights[row] * self.weights[column] * value
            for (row, column), value in self._known_covariances.items()
        )
        if self.missing_pair is not None:
            row, column = self.missing_pair
            total += 2 * self.weights[row] * self.weights[column] * float(covariance)
        return total

    @property
    def covariance_at_min(self):
        """Completed matrix attaining ``variance_min`` (or ``None``)."""

        if self.variance_min is None:
            return None
        if self.completion_at_min is None:
            return self.completed_covariance()
        return self.completed_covariance(self.completion_at_min)

    @property
    def covariance_at_max(self):
        """Completed matrix attaining ``variance_max`` (or ``None``)."""

        if self.variance_max is None:
            return None
        if self.completion_at_max is None:
            return self.completed_covariance()
        return self.completed_covariance(self.completion_at_max)

    def as_dict(self):
        """Return a plain, serializable summary of the result."""

        return {
            "columns": self.columns,
            "missing_pair": self.missing_pair,
            "missing_columns": (
                None
                if self.missing_pair is None
                else tuple(self.columns[index] for index in self.missing_pair)
            ),
            "completion_interval": self.completion_interval,
            "completion_at_min": self.completion_at_min,
            "completion_at_max": self.completion_at_max,
            "variance_min": self.variance_min,
            "variance_max": self.variance_max,
            "rank_lower": self.rank_lower,
            "rank_upper": self.rank_upper,
            "is_fixed": self.is_fixed,
            "is_constant": self.is_constant,
        }

    def __iter__(self):
        yield self.variance_min
        yield self.variance_max
        yield self.completion_at_min
        yield self.completion_at_max

    def __repr__(self):
        return (
            "CovarianceCompletion(columns=%r, missing_pair=%r, "
            "completion_interval=%r, variance_interval=%r)"
            % (
                self.columns,
                self.missing_pair,
                self.completion_interval,
                self.variance_interval,
            )
        )


def covariance_completion_bounds(
    covariance,
    weights=None,
    *,
    columns=None,
    atol=0.0,
    rtol=1e-12,
):
    """
    Bound a linear-combination variance over all positive-semidefinite
    completions of a 3x3 covariance matrix with one missing off-diagonal.

    ``covariance`` must be a 3x3 array-like.  Its diagonal contains the
    variances (the squares of the supplied standard deviations).  Exactly
    one off-diagonal pair may be missing; represent a missing entry with
    ``None`` or ``NaN``.  A missing mirrored entry is filled from the
    value that is present.  Supplying a complete matrix is also accepted;
    its missing interval then collapses to the known value.

    ``weights`` contains the three coefficients of the linear aggregate
    ``sum(weights[i] * X[i])``.  The variance of this aggregate is affine
    in the missing covariance, so its extrema occur at the two endpoints
    of the feasible interval.  If ``weights`` is ``None``, only the
    feasible completion interval is returned.

    ``columns`` gives optional labels used when reporting a conflict.

    The full 3x3 matrix is positive semidefinite if and only if all its
    principal minors are nonnegative.  With one missing entry, the
    pairwise constraints on the known entries are checked first, then the
    remaining determinant inequality gives a closed interval for the
    missing value.  A zero-variance column is handled by the Schur
    complement limit rather than by dividing by zero; rank-deficient but
    compatible inputs are therefore accepted.

    Returns a :class:`CovarianceCompletion`.  If the known entries cannot
    be completed, :class:`CovarianceConflict` is raised and its
    ``conflict_columns`` attribute identifies the offending columns.
    """

    if columns is None:
        columns = (0, 1, 2)
    else:
        columns = tuple(columns)
        if len(columns) != 3:
            raise ValueError("columns must contain exactly three labels")

    try:
        rows = [list(row) for row in covariance]
    except TypeError:
        raise TypeError("covariance must be a 3x3 array-like object") from None
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("covariance must be a 3x3 array-like object")

    conflicts = []

    variances = []
    for index in range(3):
        value = rows[index][index]
        if _covariance_missing(value):
            raise ValueError("the covariance diagonal cannot be missing")
        value = float(value)
        if value < 0:
            if value < -atol:
                conflicts.append(((index,), "negative variance"))
                continue
            value = 0.0
        variances.append(value)

    if conflicts:
        raise CovarianceConflict(conflicts, columns)

    known_covariances = {}
    missing_pairs = []
    for row in range(3):
        for column in range(row + 1, 3):
            first = rows[row][column]
            second = rows[column][row]
            first_missing = _covariance_missing(first)
            second_missing = _covariance_missing(second)

            if first_missing and second_missing:
                missing_pairs.append((row, column))
                continue
            if first_missing:
                value = float(second)
            elif second_missing:
                value = float(first)
            else:
                first = float(first)
                second = float(second)
                tolerance = _covariance_tolerance(first, second, rtol, atol)
                if abs(first - second) > tolerance:
                    conflicts.append(((row, column), "asymmetric entries"))
                    continue
                value = (first + second) / 2.0
            known_covariances[(row, column)] = value

    if len(missing_pairs) > 1:
        raise ValueError(
            "exactly one off-diagonal covariance pair must be missing; "
            "missing pairs are %r" % (tuple(missing_pairs),)
        )
    missing_pair = missing_pairs[0] if missing_pairs else None

    for (row, column), value in known_covariances.items():
        bound = sqrt(variances[row] * variances[column])
        tolerance = atol + rtol * max(bound, abs(value))
        if abs(value) > bound + tolerance:
            conflicts.append(
                ((row, column), "outside the pairwise PSD bound %g" % bound)
            )

    if conflicts:
        raise CovarianceConflict(conflicts, columns)

    if missing_pair is None:
        lower = upper = None

        determinant = (
            variances[0] * variances[1] * variances[2]
            + 2
            * known_covariances[(0, 1)]
            * known_covariances[(0, 2)]
            * known_covariances[(1, 2)]
            - variances[0] * known_covariances[(1, 2)] ** 2
            - variances[1] * known_covariances[(0, 2)] ** 2
            - variances[2] * known_covariances[(0, 1)] ** 2
        )
        determinant_scale = max(
            variances[0] * variances[1] * variances[2],
            variances[0] * known_covariances[(1, 2)] ** 2,
            variances[1] * known_covariances[(0, 2)] ** 2,
            variances[2] * known_covariances[(0, 1)] ** 2,
        )
        determinant_tolerance = atol + rtol * determinant_scale
        if determinant < -determinant_tolerance:
            raise CovarianceConflict(
                [((0, 1, 2), "the three known pairs are jointly infeasible")],
                columns,
            )
    else:
        missing_row, missing_column = missing_pair
        third = 3 - missing_row - missing_column

        third_to_missing_row = known_covariances[
            (min(third, missing_row), max(third, missing_row))
        ]
        third_to_missing_column = known_covariances[
            (min(third, missing_column), max(third, missing_column))
        ]

        pairwise_bound = sqrt(variances[missing_row] * variances[missing_column])

        if variances[third] > 0:
            left = variances[third] * variances[missing_row] - third_to_missing_row**2
            right = (
                variances[third] * variances[missing_column]
                - third_to_missing_column**2
            )
            schur_scale = max(
                variances[third] * variances[missing_row],
                third_to_missing_row**2,
                variances[third] * variances[missing_column],
                third_to_missing_column**2,
            )
            schur_tolerance = atol + rtol * schur_scale
            if left < -schur_tolerance or right < -schur_tolerance:
                conflicts.append(
                    (
                        (third, missing_row if left < 0 else missing_column),
                        "the Schur complement is not positive semidefinite",
                    )
                )
                raise CovarianceConflict(conflicts, columns)
            left = max(left, 0.0)
            right = max(right, 0.0)
            center = third_to_missing_row * third_to_missing_column / variances[third]
            half_width = sqrt(left * right) / variances[third]
            lower = center - half_width
            upper = center + half_width
        else:
            # A zero-variance column must be uncorrelated with every other
            # column.  What remains is the 2x2 covariance block.
            tolerance = atol
            zero_conflicts = []
            if abs(third_to_missing_row) > tolerance:
                zero_conflicts.append(
                    ((third, missing_row), "zero-variance column is correlated")
                )
            if abs(third_to_missing_column) > tolerance:
                zero_conflicts.append(
                    ((third, missing_column), "zero-variance column is correlated")
                )
            if zero_conflicts:
                raise CovarianceConflict(zero_conflicts, columns)
            lower = -pairwise_bound
            upper = pairwise_bound

        lower = max(lower, -pairwise_bound)
        upper = min(upper, pairwise_bound)
        interval_tolerance = atol + rtol * max(abs(lower), abs(upper), pairwise_bound)
        if lower > upper + interval_tolerance:
            raise CovarianceConflict(
                [
                    (
                        (0, 1, 2),
                        "the pairwise constraints leave no feasible completion",
                    )
                ],
                columns,
            )
        if lower > upper:
            lower = upper = (lower + upper) / 2.0

    if missing_pair is not None:
        fixed_tolerance = atol + rtol * max(abs(lower), abs(upper))
        is_fixed = abs(upper - lower) <= fixed_tolerance
    else:
        is_fixed = True

    lower_matrix = [
        [variances[row] if row == column else 0.0 for column in range(3)]
        for row in range(3)
    ]
    for (row, column), value in known_covariances.items():
        lower_matrix[row][column] = value
        lower_matrix[column][row] = value
    if missing_pair is not None:
        row, column = missing_pair
        lower_matrix[row][column] = lower
        lower_matrix[column][row] = lower
    rank_lower = _covariance_rank(lower_matrix)

    if missing_pair is not None:
        upper_matrix = [row[:] for row in lower_matrix]
        row, column = missing_pair
        upper_matrix[row][column] = upper
        upper_matrix[column][row] = upper
        rank_upper = _covariance_rank(upper_matrix)
    else:
        rank_upper = rank_lower

    if weights is None:
        return CovarianceCompletion(
            columns,
            missing_pair,
            (lower, upper) if missing_pair is not None else (None, None),
            variances,
            known_covariances,
            rank_lower=rank_lower,
            rank_upper=rank_upper,
            is_fixed=is_fixed,
            atol=atol,
            rtol=rtol,
        )

    weights = tuple(float(weight) for weight in weights)
    if len(weights) != 3:
        raise ValueError("weights must contain exactly three coefficients")

    constant_term = fsum(
        weight * weight * variance for weight, variance in zip(weights, variances)
    )
    constant_term += 2 * fsum(
        weights[row] * weights[column] * value
        for (row, column), value in known_covariances.items()
    )

    if missing_pair is None:
        return CovarianceCompletion(
            columns,
            None,
            (None, None),
            variances,
            known_covariances,
            weights=weights,
            variance_min=constant_term,
            variance_max=constant_term,
            rank_lower=rank_lower,
            rank_upper=rank_upper,
            is_fixed=True,
            is_constant=True,
            atol=atol,
            rtol=rtol,
        )

    row, column = missing_pair
    slope = 2.0 * weights[row] * weights[column]

    value_lower = constant_term + slope * lower
    value_upper = constant_term + slope * upper
    variance_tolerance = atol + rtol * max(
        abs(constant_term), abs(value_lower), abs(value_upper)
    )

    def clean_variance(value):
        if value < 0 and value >= -variance_tolerance:
            return 0.0
        return value

    value_lower = clean_variance(value_lower)
    value_upper = clean_variance(value_upper)

    if slope > 0:
        variance_min, completion_at_min = value_lower, lower
        variance_max, completion_at_max = value_upper, upper
    elif slope < 0:
        variance_min, completion_at_min = value_upper, upper
        variance_max, completion_at_max = value_lower, lower
    else:
        variance_min = variance_max = constant_term
        completion_at_min, completion_at_max = lower, upper

    if variance_min > variance_max:
        variance_min, variance_max = variance_max, variance_min
        completion_at_min, completion_at_max = (
            completion_at_max,
            completion_at_min,
        )

    return CovarianceCompletion(
        columns,
        missing_pair,
        (lower, upper),
        variances,
        known_covariances,
        weights=weights,
        variance_min=variance_min,
        variance_max=variance_max,
        completion_at_min=completion_at_min,
        completion_at_max=completion_at_max,
        rank_lower=rank_lower,
        rank_upper=rank_upper,
        is_fixed=is_fixed,
        is_constant=(slope == 0.0),
        atol=atol,
        rtol=rtol,
    )


def partial_covariance_bounds(
    std_devs,
    covariances,
    weights=None,
    *,
    columns=None,
    atol=0.0,
    rtol=1e-12,
):
    """
    Convenience wrapper for a supplier table of standard deviations and
    a partial dictionary of pairwise covariances.

    ``std_devs`` is a sequence of three standard deviations; its squares
    form the covariance diagonal.  ``covariances`` maps zero-based column
    pairs, such as ``(0, 1)``, to their covariance.  Exactly one of the
    three off-diagonal pairs must be absent (or mapped to ``None``/``NaN``).
    Keys may be given in either order.

    The remaining arguments are those of
    :func:`covariance_completion_bounds`.
    """

    std_devs = tuple(float(value) for value in std_devs)
    if len(std_devs) != 3:
        raise ValueError("std_devs must contain exactly three values")
    if any(value < 0 for value in std_devs):
        raise ValueError("standard deviations cannot be negative")

    matrix = [[None, None, None] for _ in range(3)]
    for index, value in enumerate(std_devs):
        matrix[index][index] = value * value

    supplied = {}
    for key, value in dict(covariances).items():
        try:
            row, column = key
        except (TypeError, ValueError):
            raise ValueError(
                "covariance keys must be (row, column) index pairs"
            ) from None
        row = int(row)
        column = int(column)
        if row == column or not (0 <= row < 3 and 0 <= column < 3):
            raise ValueError("covariance keys must be distinct indices in 0..2")
        pair = (min(row, column), max(row, column))
        previous = supplied.get(pair)
        if pair in supplied:
            previous_missing = _covariance_missing(previous)
            value_missing = _covariance_missing(value)
            if previous_missing and value_missing:
                value = None
            elif previous_missing:
                value = value
            elif value_missing:
                value = previous
            else:
                tolerance = _covariance_tolerance(previous, value, rtol, atol)
                if abs(previous - value) > tolerance:
                    raise ValueError(
                        "covariance pair %r is supplied twice with different values"
                        % (pair,)
                    )
                value = (previous + value) / 2.0
        supplied[pair] = value

    for (row, column), value in supplied.items():
        matrix[row][column] = value
        matrix[column][row] = value

    return covariance_completion_bounds(
        matrix,
        weights,
        columns=columns,
        atol=atol,
        rtol=rtol,
    )


# Descriptive aliases for the same completion operation.  They are kept
# as assignments (rather than wrappers) so that their signatures and
# docstrings cannot drift from the implementation.
covariance_bounds = covariance_completion_bounds
linear_combination_variance_bounds = covariance_completion_bounds
partial_covariance_variance_bounds = partial_covariance_bounds
variance_bounds = covariance_completion_bounds


########################################
class LinearCombination(object):
    """
    Linear combination of Variable differentials.

    The linear_combo attribute can change formally, but its value
    always remains the same. Typically, the linear combination can
    thus be expanded.

    The expanded form of linear_combo is a mapping from Variables to
    the coefficient of their differential.
    """

    # ! Invariant: linear_combo is represented internally exactly as
    # the linear_combo argument to __init__():
    __slots__ = "linear_combo"

    def __init__(self, linear_combo):
        """
        linear_combo can be modified by the object, during its
        lifetime. This allows the object to change its internal
        representation over time (for instance by expanding the linear
        combination and replacing the original expression with the
        expanded one).

        linear_combo -- if linear_combo is a dict, then it represents
        an expanded linear combination and must map Variables to the
        coefficient of their differential. Otherwise, it should be a
        list of (coefficient, LinearCombination) pairs (that
        represents a linear combination expression).
        """

        self.linear_combo = linear_combo

    def __bool__(self):
        """
        Return True only if the linear combination is non-empty, i.e. if
        the linear combination contains any term.
        """
        return bool(self.linear_combo)

    def expanded(self):
        """
        Return True if and only if the linear combination is expanded.
        """
        return isinstance(self.linear_combo, dict)

    def expand(self):
        """
        Expand the linear combination.

        The expansion is a collections.defaultdict(float).

        This should only be called if the linear combination is not
        yet expanded.
        """

        # The derivatives are built progressively by expanding each
        # term of the linear combination until there is no linear
        # combination to be expanded.

        # Final derivatives, constructed progressively:
        derivatives = collections.defaultdict(float)

        while self.linear_combo:  # The list of terms is emptied progressively
            # One of the terms is expanded or, if no expansion is
            # needed, simply added to the existing derivatives.
            #
            # Optimization note: since Python's operations are
            # left-associative, a long sum of Variables can be built
            # such that the last term is essentially a Variable (and
            # not a NestedLinearCombination): popping from the
            # remaining terms allows this term to be quickly put in
            # the final result, which limits the number of terms
            # remaining (and whose size can temporarily grow):
            (main_factor, main_expr) = self.linear_combo.pop()

            # print "MAINS", main_factor, main_expr

            if main_expr.expanded():
                for var, factor in main_expr.linear_combo.items():
                    derivatives[var] += main_factor * factor

            else:  # Non-expanded form
                for factor, expr in main_expr.linear_combo:
                    # The main_factor is applied to expr:
                    self.linear_combo.append((main_factor * factor, expr))

            # print "DERIV", derivatives

        self.linear_combo = derivatives

    def __getstate__(self):
        # Not false, otherwise __setstate__() will not be called:
        return (self.linear_combo,)

    def __setstate__(self, state):
        (self.linear_combo,) = state


class AffineScalarFunc(object):
    """
    Affine functions that support basic mathematical operations
    (addition, etc.).  Such functions can for instance be used for
    representing the local (linear) behavior of any function.

    This class can also be used to represent constants.

    The variables of affine scalar functions are Variable objects.

    AffineScalarFunc objects include facilities for calculating the
    'error' on the function, from the uncertainties on its variables.

    Main attributes and methods:

    - nominal_value, std_dev: value at the origin / nominal value, and
      standard deviation.  The standard deviation can be NaN or infinity.

    - n, s: abbreviations for nominal_value and std_dev.

    - error_components(): error_components()[x] is the error due to
      Variable x.

    - derivatives: derivatives[x] is the (value of the) derivative
      with respect to Variable x.  This attribute is a Derivatives
      dictionary whose keys are the Variable objects on which the
      function depends. The values are the numerical values of the
      derivatives.

      All the Variable objects on which the function depends are in
      'derivatives'.

    - std_score(x): position of number x with respect to the
      nominal value, in units of the standard deviation.
    """

    # To save memory in large arrays:
    __slots__ = ("_nominal_value", "_linear_part")

    # !! Fix for mean() in NumPy 1.8.0:
    class dtype(object):
        type = staticmethod(lambda value: value)

    #! The code could be modified in order to accommodate for non-float
    # nominal values.  This could for instance be done through
    # the operator module: instead of delegating operations to
    # float.__*__ operations, they could be delegated to
    # operator.__*__ functions (while taking care of properly handling
    # reverse operations: __radd__, etc.).

    def __init__(self, nominal_value, linear_part):
        """
        nominal_value -- value of the function when the linear part is
        zero.

        linear_part -- LinearCombination that describes the linear
        part of the AffineScalarFunc.
        """

        # ! A technical consistency requirement is that the
        # linear_part can be nested inside a NestedLinearCombination
        # (because this is how functions on AffineScalarFunc calculate
        # their result: by constructing nested expressions for them).

        # Defines the value at the origin:

        # Only float-like values are handled.  One reason is that it
        # does not make sense for a scalar function to be affine to
        # not yield float values.  Another reason is that it would not
        # make sense to have a complex nominal value, here (it would
        # not be handled correctly at all): converting to float should
        # be possible.

        self._nominal_value = float(nominal_value)

        # In order to have a linear execution time for long sums, the
        # _linear_part is generally left as is (otherwise, each
        # successive term would expand to a linearly growing sum of
        # terms: efficiently handling such terms [so, without copies]
        # is not obvious, when the algorithm should work for all
        # functions beyond sums).
        if not isinstance(linear_part, LinearCombination):
            linear_part = LinearCombination(linear_part)
        self._linear_part = linear_part

    # The following prevents the 'nominal_value' attribute from being
    # modified by the user:
    @property
    def nominal_value(self):
        "Nominal value of the random number."
        return self._nominal_value

    # Abbreviation (for formulas, etc.):
    n = nominal_value

    ############################################################

    # Making derivatives a property gives the user a clean syntax,
    # which is consistent with derivatives becoming a dictionary.
    @property
    def derivatives(self):
        """
        Return a mapping from each Variable object on which the function
        (self) depends to the value of the derivative with respect to
        that variable.

        This mapping should not be modified.

        Derivative values are always floats.

        This mapping is cached, for subsequent calls.
        """
        warn(
            f"{self.__class__.__name__}.derivatives() is deprecated. It will "
            f"be removed in a future release.",
            FutureWarning,
            stacklevel=2,
        )

        if not self._linear_part.expanded():
            self._linear_part.expand()
            # Attempts to get the contribution of a variable that the
            # function does not depend on raise a KeyError:
            self._linear_part.linear_combo.default_factory = None

        return self._linear_part.linear_combo

    ########################################

    # Uncertainties handling:

    def error_components(self):
        """
        Individual components of the standard deviation of the affine
        function (in absolute value), returned as a dictionary with
        Variable objects as keys. The returned variables are the
        independent variables that the affine function depends on.

        This method assumes that the derivatives contained in the
        object take scalar values (and are not a tuple, like what
        math.frexp() returns, for instance).
        """
        warn(
            f"{self.__class__.__name__}.error_components() is currently an "
            f"instance method. This method is deprecated. In a future release it will "
            f"be replaced with an instance property by the same name. It will be "
            f"accessed by {self.__class__.__name__}.error_components (with no "
            f"parentheses).",
            FutureWarning,
            stacklevel=2,
        )

        # Calculation of the variance:
        error_components = {}

        for variable, derivative in self.derivatives.items():
            # print "TYPE", type(variable), type(derivative)

            # Individual standard error due to variable:

            # 0 is returned even for a NaN derivative (in this case no
            # multiplication by the derivative is performed): an exact
            # variable obviously leads to no uncertainty in the
            # functions that depend on it.
            if variable._std_dev == 0:
                # !!! Shouldn't the errors always be floats, as a
                # convention of this module?
                error_components[variable] = 0
            else:
                error_components[variable] = abs(derivative * variable._std_dev)

        return error_components

    @property
    def std_dev(self):
        """
        Standard deviation of the affine function.

        This method assumes that the function returns scalar results.

        This returned standard deviation depends on the current
        standard deviations [std_dev] of the variables (Variable
        objects) involved.
        """
        #! It would be possible to not allow the user to update the
        # std dev of Variable objects, in which case AffineScalarFunc
        # objects could have a pre-calculated or, better, cached
        # std_dev value (in fact, many intermediate AffineScalarFunc do
        # not need to have their std_dev calculated: only the final
        # AffineScalarFunc returned to the user does).
        return float(sqrt(sum(delta**2 for delta in self.error_components().values())))

    # Abbreviation (for formulas, etc.):
    s = std_dev

    def __repr__(self):
        # Not putting spaces around "+/-" helps with arrays of
        # Variable, as each value with an uncertainty is a
        # block of signs (otherwise, the standard deviation can be
        # mistaken for another element of the array).

        std_dev = self.std_dev  # Optimization, since std_dev is calculated

        # A zero standard deviation is printed because otherwise,
        # ufloat_fromstr() does not correctly parse back the value
        # ("1.23" is interpreted as "1.23(1)"):

        if std_dev:
            std_dev_str = repr(std_dev)
        else:
            std_dev_str = "0"

        return "%r+/-%s" % (self.nominal_value, std_dev_str)

    def __str__(self):
        # An empty format string and str() usually return the same
        # string
        # (http://docs.python.org/2/library/string.html#format-specification-mini-language):
        return self.format("")

    @set_doc(format_ufloat.__doc__)
    def __format__(self, format_spec):
        return format_ufloat(self, format_spec)

    @set_doc("""
        Return the same result as self.__format__(format_spec), or
        equivalently as the format(self, format_spec) of Python 2.6+.

        This method is meant to be used for formatting numbers with
        uncertainties in Python < 2.6, with '... %s ...' %
        num.format('.2e').
        """)
    def format(self, format_spec):
        return format_ufloat(self, format_spec)

    def std_score(self, value):
        """
        Return 'value' - nominal value, in units of the standard
        deviation.

        Raises a ValueError exception if the standard deviation is zero.
        """
        try:
            # The ._nominal_value is a float: there is no integer division,
            # here:
            return (value - self._nominal_value) / self.std_dev
        except ZeroDivisionError:
            raise ValueError("The standard deviation is zero: undefined result")

    def __deepcopy__(self, memo):
        """
        Hook for the standard copy module.

        The returned AffineScalarFunc is a completely fresh copy,
        which is fully independent of any variable defined so far.
        New variables are specially created for the returned
        AffineScalarFunc object.
        """
        return AffineScalarFunc(self._nominal_value, copy.deepcopy(self._linear_part))

    def __getstate__(self):
        """
        Hook for the pickle module.

        The slot attributes of the parent classes are returned, as
        well as those of the __dict__ attribute of the object (if
        any).
        """

        # In general (case where this class is subclassed), data
        # attribute are stored in two places: possibly in __dict_, and
        # in slots. Data from both locations is returned by this
        # method.

        all_attrs = {}

        # Support for subclasses that do not use __slots__ (except
        # through inheritance): instances have a __dict__
        # attribute. The keys in this __dict__ are shadowed by the
        # slot attribute names (reference:
        # http://stackoverflow.com/questions/15139067/attribute-access-in-python-first-slots-then-dict/15139208#15139208).
        # The method below not only preserves this behavior, but also
        # saves the full contents of __dict__. This is robust:
        # unpickling gives back the original __dict__ even if __dict__
        # contains keys that are shadowed by slot names:

        try:
            all_attrs["__dict__"] = self.__dict__
        except AttributeError:
            pass

        # All the slot attributes are gathered.

        # Classes that do not define __slots__ have the __slots__ of
        # one of their parents (the first parent with their own
        # __slots__ in MRO). This is why the slot names are first
        # gathered (with repetitions removed, in general), and their
        # values obtained later.

        all_slots = set()

        for cls in type(self).mro():
            # In the diamond inheritance pattern, some parent classes
            # may not have __slots__:
            slot_names = getattr(cls, "__slots__", ())

            # Slot names can be given in various forms (string,
            # sequence, iterable):
            if isinstance(slot_names, str):
                all_slots.add(slot_names)  # Single name
            else:
                all_slots.update(slot_names)

        # The slot values are stored:
        for name in all_slots:
            try:
                # !! It might happen that '__dict__' is itself a slot
                # name. In this case, its value is saved
                # again. Alternatively, the loop could be done on
                # all_slots - {'__dict__'}:
                all_attrs[name] = getattr(self, name)
            except AttributeError:
                pass  # Undefined slot attribute

        return all_attrs

    def __setstate__(self, data_dict):
        """
        Hook for the pickle module.
        """
        for name, value in data_dict.items():
            # Contrary to the default __setstate__(), this does not
            # necessarily save to the instance dictionary (because the
            # instance might contain slots):
            setattr(self, name, value)


ops.add_arithmetic_ops(AffineScalarFunc)
ops.add_comparative_ops(AffineScalarFunc)
to_affine_scalar = AffineScalarFunc._to_affine_scalar

# Nicer name, for users: isinstance(ufloat(...), UFloat) is
# True. Also: isinstance(..., UFloat) is the test for "is this a
# number with uncertainties from the uncertainties package?":
UFloat = AffineScalarFunc


def wrap(f, derivatives_args=None, derivatives_kwargs=None):
    """Wrap a function f into one that accepts Variables.

    The function f must return a float or integer value.  The returned
    wrapped function will return values with both uncertainties and
    correlations, but can be used as a drop-in replacement for the
    original function.

    Arguments:
    ----------
    derivatives_args: list or iterable
           list or tupleof derivative functionss or None with respect to
           the positional arguments of `f`.  See Note 1.
    derivatives_kwargs: dictionary
           dict of derivative functionss or None with respect to the
           keyword arguments of `f`.  See Note 1.

    Notes:
    -------
    1.  Each function must be the partial derivative of f with respect to the
        corresponding positional parameters, and must have the same signature
        as ``f``. `derivative_args` hold derivitative functions for positional
        arguments (include `*varargs`), while  `derivative_kwargs` holds
        derivitative functions for keyword arguments (include `**kwargs`). If an
        entry is `None` or not supplied, and if the argument value isa numeric
        Variable, a numerical derivative will be used. Non-numeric are ignored.
    2.  If derivatives are meaningless or the function is not function is not
        differentiable, the derivative funcion should return NaN for values
        for which the the function is not differentiable.

    Example:
    --------
    To wrap `sin`, one could do
       >>> from uncertainties import wrap, umath
       >>> import math
       >>> usin_a = wrap(math.sin)   # uses numerical derivative
       >>> usin_b = wrap(math.sin, [math.cos])  # use analytic derivative
       >>> usin_c = umath.sin        # builtin, same as usin_2

    These will all give the same result.
    """
    return _wrap(
        AffineScalarFunc,
        f,
        derivatives_args=derivatives_args,
        derivatives_kwargs=derivatives_kwargs,
    )


###############################################################################


class NegativeStdDev(Exception):
    """Raise for a negative standard deviation"""

    pass


class Variable(AffineScalarFunc):
    """
    Representation of a float-like scalar Variable with its uncertainty.

    Variables are independent from each other, but correlations between them
    are handled through the AffineScalarFunc class.
    """

    # To save memory in large arrays:
    __slots__ = ("_std_dev", "tag")

    def __init__(self, value, std_dev, tag=None):
        """
        The nominal value and the standard deviation of the variable
        are set.

        The value is converted to float.

        The standard deviation std_dev can be NaN. It should normally
        be a float or an integer.

        'tag' is a tag that the user can associate to the variable.  This
        is useful for tracing variables.

        The meaning of the nominal value is described in the main
        module documentation.
        """

        #! The value, std_dev, and tag are assumed by __copy__() not to
        # be copied.  Either this should be guaranteed here, or __copy__
        # should be updated.

        # Only float-like values are handled.  One reason is that the
        # division operator on integers would not produce a
        # differentiable functions: for instance, Variable(3, 0.1)/2
        # has a nominal value of 3/2 = 1, but a "shifted" value
        # of 3.1/2 = 1.55.
        value = float(value)

        # If the variable changes by dx, then the value of the affine
        # function that gives its value changes by 1*dx:

        # ! Memory cycles are created.  However, they are garbage
        # collected, if possible.  Using a weakref.WeakKeyDictionary
        # takes much more memory.  Thus, this implementation chooses
        # more cycles and a smaller memory footprint instead of no
        # cycles and a larger memory footprint.
        super(Variable, self).__init__(value, LinearCombination({self: 1.0}))

        self.std_dev = std_dev  # Assignment through a Python property

        self.tag = tag

    @property
    def std_dev(self):
        return self._std_dev

    # Standard deviations can be modified (this is a feature).
    # AffineScalarFunc objects that depend on the Variable have their
    # std_dev automatically modified (recalculated with the new
    # std_dev of their Variables):
    @std_dev.setter
    def std_dev(self, std_dev):
        # We force the error to be float-like.  Since it is considered
        # as a standard deviation, it must be either positive or NaN:
        # (Note: if NaN < 0 is False, there is no need to test
        # separately for NaN. But this is not guaranteed, even if it
        # should work on most platforms.)
        if std_dev < 0 and isfinite(std_dev):
            raise NegativeStdDev("The standard deviation cannot be negative")

        self._std_dev = float(std_dev)

    # The following method is overridden so that we can represent the tag:
    def __repr__(self):
        num_repr = super(Variable, self).__repr__()

        if self.tag is None:
            return num_repr
        else:
            return "< %s = %s >" % (self.tag, num_repr)

    def __hash__(self):
        # All Variable objects are by definition independent
        # variables, so they never compare equal; therefore, their
        # id() are allowed to differ
        # (http://docs.python.org/reference/datamodel.html#object.__hash__):
        return id(self)

    def __copy__(self):
        """
        Hook for the standard copy module.
        """

        # !!!!!! The comment below might not be valid anymore now that
        # Variables do not contain derivatives anymore.

        # This copy implicitly takes care of the reference of the
        # variable to itself (in self.derivatives): the new Variable
        # object points to itself, not to the original Variable.

        # Reference: http://www.doughellmann.com/PyMOTW/copy/index.html

        #! The following assumes that the arguments to Variable are
        # *not* copied upon construction, since __copy__ is not supposed
        # to copy "inside" information:
        return Variable(self.nominal_value, self.std_dev, self.tag)

    def __deepcopy__(self, memo):
        """
        Hook for the standard copy module.

        A new variable is created.
        """

        # This deep copy implicitly takes care of the reference of the
        # variable to itself (in self.derivatives): the new Variable
        # object points to itself, not to the original Variable.

        # Reference: http://www.doughellmann.com/PyMOTW/copy/index.html

        return self.__copy__()


###############################################################################

# Utilities


def nominal_value(x):
    """
    Return the nominal value of x if it is a quantity with
    uncertainty (i.e., an AffineScalarFunc object); otherwise, returns
    x unchanged.

    This utility function is useful for transforming a series of
    numbers, when only some of them generally carry an uncertainty.
    """

    if isinstance(x, AffineScalarFunc):
        return x.nominal_value
    else:
        return x


def std_dev(x):
    """
    Return the standard deviation of x if it is a quantity with
    uncertainty (i.e., an AffineScalarFunc object); otherwise, returns
    the float 0.

    This utility function is useful for transforming a series of
    numbers, when only some of them generally carry an uncertainty.
    """

    if isinstance(x, AffineScalarFunc):
        return x.std_dev
    else:
        return 0.0


def covariance_matrix(nums_with_uncert):
    """
    Return a matrix that contains the covariances between the given
    sequence of numbers with uncertainties (AffineScalarFunc objects).
    The resulting matrix implicitly depends on their ordering in
    'nums_with_uncert'.

    The covariances are floats (never int objects).

    The returned covariance matrix is the exact linear approximation
    result, if the nominal values of the numbers with uncertainties
    and of their variables are their mean.  Otherwise, the returned
    covariance matrix should be close to its linear approximation
    value.

    The returned matrix is a list of lists.
    """
    # See PSI.411 in EOL's notes.

    covariance_matrix = []
    for i1, expr1 in enumerate(nums_with_uncert, 1):
        derivatives1 = expr1.derivatives  # Optimization
        vars1 = set(derivatives1)  # !! Python 2.7+: viewkeys() would work
        coefs_expr1 = []

        for expr2 in nums_with_uncert[:i1]:
            derivatives2 = expr2.derivatives  # Optimization
            coefs_expr1.append(
                sum(
                    (
                        (derivatives1[var] * derivatives2[var] * var._std_dev**2)
                        # var is a variable common to both numbers with
                        # uncertainties:
                        for var in vars1.intersection(derivatives2)
                    ),
                    # The result is always a float (sum() with no terms
                    # returns an integer):
                    0.0,
                )
            )

        covariance_matrix.append(coefs_expr1)

    # We symmetrize the matrix:
    for i, covariance_coefs in enumerate(covariance_matrix):
        covariance_coefs.extend(
            [covariance_matrix[j][i] for j in range(i + 1, len(covariance_matrix))]
        )

    return covariance_matrix


def ufloat_fromstr(representation, tag=None):
    """
    Create an uncertainties Variable from a string representation.
    Several representation formats are supported.

    Arguments:
    ----------
    representation: string
        string representation of a value with uncertainty
    tag:   string or `None`
        optional tag for tracing and organizing Variables ['None']

    Returns:
    --------
    uncertainties Variable.

    Notes:
    --------
    1. Invalid representations raise a ValueError.

    2. Using the form "nominal(std)" where "std" is an integer creates
       a Variable with "std" giving the least significant digit(s).
       That is, "1.25(3)" is the same as `ufloat(1.25, 0.03)`,
       while "1.25(3.)" is the same as `ufloat(1.25, 3.)`

    3. If the representation does not contain an uncertainty, an
       uncertainty of 1 in the least significant digit is assigned to
       the nominal value. For nominal values corresponding to "nan", an
       uncertainty of 1 is assigned.

    Examples:
    -----------

    >>> from uncertainties import ufloat_fromstr
    >>> x = ufloat_fromstr("12.58+/-0.23")  # = ufloat(12.58, 0.23)
    >>> x = ufloat_fromstr("12.58 ± 0.23")  # = ufloat(12.58, 0.23)
    >>> x = ufloat_fromstr("3.85e5 +/- 2.3e4")  # = ufloat(3.8e5, 2.3e4)
    >>> x = ufloat_fromstr("(38.5 +/- 2.3)e4")  # = ufloat(3.8e5, 2.3e4)
    >>> x = ufloat_fromstr("38.5 +- 10%") # = ufloat(38.5, 3.85) (percentage uncertainty)

    >>> x = ufloat_fromstr("72.1(2.2)")  # = ufloat(72.1, 2.2)
    >>> x = ufloat_fromstr("72.15(4)")  # = ufloat(72.15, 0.04)
    >>> x = ufloat_fromstr("680(41)e-3")  # = ufloat(0.68, 0.041)
    >>> x = ufloat_fromstr("23.2")  # = ufloat(23.2, 0.1)
    >>> x = ufloat_fromstr("23.29")  # = ufloat(23.29, 0.01)
    >>> x = ufloat_fromstr("nan")  # = ufloat(numpy.nan, 1.0)

    >>> x = ufloat_fromstr("680.3(nan)") # = ufloat(680.3, numpy.nan)
    """
    (nom, std) = str_to_number_with_uncert(representation.strip())
    return ufloat(nom, std, tag)


def ufloat(nominal_value, std_dev=None, tag=None):
    """
    Create an uncertainties Variable

    Arguments:
    ----------
    nominal_value: float
        nominal value of Variable
    std_dev:   float or `None`
        standard error of Variable, or `None` if not available [`None`]
    tag:   string or `None`
        optional tag for tracing and organizing Variables ['None']

    Returns:
    --------
    uncertainties Variable

    Examples
    ----------
    >>> from uncertainties import ufloat
    >>> a = ufloat(5, 0.2)
    >>> b = ufloat(1000, 30, tag='kilo')


    Notes:
    --------
    1. `nominal_value` is typically interpreted as `mean` or `central value`
    2. `std_dev` is typically interpreted as `standard deviation` or the
        1-sigma level uncertainty.
    3. The returned Variable will have attributes `nominal_value`, `std_dev`,
       and `tag` which match the input values.
    """
    if std_dev == 0:
        warn(
            "Using UFloat objects with std_dev==0 may give unexpected results.",
            stacklevel=2,
        )
    return Variable(nominal_value, std_dev, tag=tag)


# Deprecated UFloat methods


def deprecation_wrapper(func, msg):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        warn(msg, FutureWarning, stacklevel=2)
        return func(*args, **kwargs)

    return wrapped


deprecated_methods = [
    "__floordiv__",
    "__mod__",
    "__abs__",
    "__trunc__",
    "__lt__",
    "__gt__",
    "__le__",
    "__ge__",
]

for method_name in deprecated_methods:
    message = (
        f"AffineScalarFunc.{method_name}() is deprecated. It will be removed in a future "
        f"release."
    )
    setattr(
        AffineScalarFunc,
        method_name,
        deprecation_wrapper(getattr(AffineScalarFunc, method_name), message),
    )
