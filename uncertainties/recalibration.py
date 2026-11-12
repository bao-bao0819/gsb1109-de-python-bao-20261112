# !! Whenever the documentation below is updated, the documentation in
# doc/ should be checked for consistency.

"""
Revision of published uncertainty tables after the recalibration of an
error source.

A published calibration table stores, for each output column, the
*signed* sensitivities (partial derivatives) of the column to each
original error source, along with the covariance matrix of the sources
(often diagonal, for independent sources).  When an instrument is
recalibrated, one error source gets a new standard deviation and new
covariances with the other sources, while the nominal readings do not
move.  This module recomputes the column uncertainties and produces an
itemized account of every covariance change between columns, broken
down by the pair of error sources each change comes from.

Because the change is tracked per source pair, a column whose own
variance is strictly unchanged can still show a changed covariance
with another column, and any subset of columns exported on its own
still carries the full explanation of which source pair each
difference comes from.

This module requires NumPy.
"""

import itertools

try:
    import numpy
except ImportError:
    numpy = None


__all__ = ["recalibrate", "RecalibrationReport", "SourcePairContribution"]


class SourcePairContribution:
    """
    Contribution of a single pair of error sources to the change of
    the covariance matrix between columns.

    Attributes:

    - source_pair -- (name, name) of the two error sources involved
      (both names are the same for a variance change of a single
      source).

    - matrix -- square NumPy array giving, for every pair of columns,
      how much of the covariance change comes from this source pair.
      The sum of the matrices of all the contributions of a
      RecalibrationReport is its full delta_covariance.
    """

    def __init__(self, source_pair, matrix):
        self.source_pair = source_pair
        self.matrix = matrix

    def __repr__(self):
        return "SourcePairContribution(source_pair={!r})".format(self.source_pair)


def _require_numpy():
    if numpy is None:
        msg = (
            "uncertainties was not able to import numpy so "
            "recalibrate is unavailable."
        )
        raise NotImplementedError(msg)


def _resolve_index(name_or_index, names, kind):
    """
    Return the integer index of 'name_or_index' in 'names', accepting
    either a name (string) or an integer index.  'kind' is used in
    error messages ("column" or "error source").
    """
    if isinstance(name_or_index, str):
        try:
            return names.index(name_or_index)
        except ValueError:
            msg = "unknown {} name: {!r}".format(kind, name_or_index)
            raise KeyError(msg)
    index = int(name_or_index)
    if not 0 <= index < len(names):
        msg = "{} index out of range: {!r}".format(kind, name_or_index)
        raise IndexError(msg)
    return index


def recalibrate(
    sensitivities,
    source_covariance,
    updated_source,
    new_std_dev=None,
    new_covariances=None,
    nominal_values=None,
    column_names=None,
    source_names=None,
):
    """
    Return a RecalibrationReport describing how the uncertainties and
    covariances of published table columns change when one error
    source is recalibrated.

    sensitivities -- (n_columns, n_sources) array-like of the *signed*
    sensitivities of each published column to each original error
    source, as retained in the published table.

    source_covariance -- original (n_sources, n_sources) covariance
    matrix of the error sources.  A one-dimensional sequence of
    standard deviations is also accepted and means independent
    sources.

    updated_source -- name (or index) of the recalibrated error
    source.

    new_std_dev -- new standard deviation of the recalibrated source
    (None keeps the original standard deviation).

    new_covariances -- mapping {other source name or index: new
    covariance} giving the new covariances between the recalibrated
    source and the other sources.  Covariances not mentioned keep
    their original value.

    nominal_values -- optional sequence of the published nominal
    readings (one per column).  They are carried over unchanged: a
    recalibration of error sources does not move the nominal values.

    column_names, source_names -- optional names used in the report
    (defaults are "column 0", ... and "source 0", ...).

    This function raises NotImplementedError if numpy cannot be
    imported.
    """
    _require_numpy()

    sens = numpy.asarray(sensitivities, dtype=float)
    if sens.ndim != 2:
        msg = "sensitivities must be a 2-dimensional (columns, sources) array"
        raise ValueError(msg)
    (n_columns, n_sources) = sens.shape

    source_cov = numpy.asarray(source_covariance, dtype=float)
    if source_cov.ndim == 1:
        # A sequence of standard deviations means independent sources:
        source_cov = numpy.diag(source_cov**2)
    if source_cov.shape != (n_sources, n_sources):
        msg = (
            "source_covariance must have shape ({0}, {0}) to match the "
            "{0} error sources of the sensitivities".format(n_sources)
        )
        raise ValueError(msg)
    if not numpy.allclose(source_cov, source_cov.T, rtol=0, atol=0):
        msg = "source_covariance must be symmetric"
        raise ValueError(msg)

    if column_names is None:
        column_names = ["column {}".format(i) for i in range(n_columns)]
    elif len(column_names) != n_columns:
        msg = "column_names must have one name per column"
        raise ValueError(msg)
    else:
        column_names = list(column_names)

    if source_names is None:
        source_names = ["source {}".format(i) for i in range(n_sources)]
    elif len(source_names) != n_sources:
        msg = "source_names must have one name per error source"
        raise ValueError(msg)
    else:
        source_names = list(source_names)

    if nominal_values is not None:
        nominal_values = list(nominal_values)
        if len(nominal_values) != n_columns:
            msg = "nominal_values must have one value per column"
            raise ValueError(msg)

    k = _resolve_index(updated_source, source_names, "error source")

    new_source_cov = source_cov.copy()
    if new_std_dev is not None:
        if new_std_dev < 0:
            msg = "the new standard deviation cannot be negative"
            raise ValueError(msg)
        new_source_cov[k, k] = float(new_std_dev) ** 2
    for (other_source, covariance) in (new_covariances or {}).items():
        j = _resolve_index(other_source, source_names, "error source")
        if j == k:
            msg = (
                "the new variance of the recalibrated source must be "
                "given through new_std_dev, not new_covariances"
            )
            raise ValueError(msg)
        new_source_cov[k, j] = new_source_cov[j, k] = covariance

    # The recalibrated source covariance matrix must remain a valid
    # (positive semi-definite) covariance matrix:
    eigenvalues = numpy.linalg.eigvalsh(new_source_cov)
    tolerance = 100 * numpy.finfo(float).eps * max(1.0, abs(eigenvalues).max())
    if eigenvalues.min() < -tolerance:
        msg = (
            "the recalibrated source covariance matrix is not positive "
            "semi-definite (smallest eigenvalue: {!r})".format(eigenvalues.min())
        )
        raise ValueError(msg)

    old_column_cov = sens @ source_cov @ sens.T
    new_column_cov = sens @ new_source_cov @ sens.T

    # The change of covariance between columns is decomposed into one
    # contribution per pair of error sources, so that every entry of
    # the change can be attributed to the source pair it comes from.
    # Only the row and column of the recalibrated source can change:
    delta_source_cov = new_source_cov - source_cov
    contributions = []
    if delta_source_cov[k, k] != 0:
        contributions.append(
            SourcePairContribution(
                (source_names[k], source_names[k]),
                delta_source_cov[k, k] * numpy.outer(sens[:, k], sens[:, k]),
            )
        )
    for j in range(n_sources):
        if j != k and delta_source_cov[k, j] != 0:
            contributions.append(
                SourcePairContribution(
                    (source_names[k], source_names[j]),
                    delta_source_cov[k, j]
                    * (
                        numpy.outer(sens[:, k], sens[:, j])
                        + numpy.outer(sens[:, j], sens[:, k])
                    ),
                )
            )

    return RecalibrationReport(
        column_names=column_names,
        source_names=source_names,
        nominal_values=nominal_values,
        updated_source=source_names[k],
        old_source_covariance=source_cov,
        new_source_covariance=new_source_cov,
        old_covariance=old_column_cov,
        new_covariance=new_column_cov,
        contributions=contributions,
    )


class RecalibrationReport:
    """
    Result of recalibrate(): revised column uncertainties and the
    itemized changes of the covariances between columns.

    Attributes:

    - column_names, source_names -- names of the published columns and
      of the original error sources.

    - nominal_values -- published nominal readings (unchanged by the
      recalibration), or None if they were not provided.

    - updated_source -- name of the recalibrated error source.

    - old_source_covariance, new_source_covariance -- covariance
      matrices of the error sources, before and after recalibration.

    - old_covariance, new_covariance -- covariance matrices between
      the columns, before and after recalibration (NumPy arrays).

    - old_std_devs, new_std_devs -- standard uncertainty of each
      column, before and after recalibration (NumPy arrays).

    - contributions -- list of SourcePairContribution objects giving,
      for each changed pair of error sources, the part of the column
      covariance change that comes from this source pair.  Their
      matrices sum to delta_covariance.
    """

    def __init__(
        self,
        column_names,
        source_names,
        nominal_values,
        updated_source,
        old_source_covariance,
        new_source_covariance,
        old_covariance,
        new_covariance,
        contributions,
    ):
        self.column_names = column_names
        self.source_names = source_names
        self.nominal_values = nominal_values
        self.updated_source = updated_source
        self.old_source_covariance = old_source_covariance
        self.new_source_covariance = new_source_covariance
        self.old_covariance = old_covariance
        self.new_covariance = new_covariance
        self.contributions = contributions

    @property
    def old_std_devs(self):
        return numpy.sqrt(numpy.diag(self.old_covariance))

    @property
    def new_std_devs(self):
        return numpy.sqrt(numpy.diag(self.new_covariance))

    @property
    def delta_covariance(self):
        """
        Change of the covariance matrix between columns
        (new_covariance - old_covariance).  A column can have a zero
        diagonal change (unchanged variance) while its covariance with
        another column does change.
        """
        return self.new_covariance - self.old_covariance

    @property
    def delta_std_devs(self):
        return self.new_std_devs - self.old_std_devs

    def subset(self, columns):
        """
        Return a new RecalibrationReport restricted to the given
        columns (names or indices), for exporting only part of the
        table.

        The source-pair contributions are restricted along with the
        covariance matrices, so the changes visible in the subset are
        still fully explained by the source pairs they come from.
        """
        indices = [
            _resolve_index(column, self.column_names, "column") for column in columns
        ]
        nominal_values = (
            None
            if self.nominal_values is None
            else [self.nominal_values[i] for i in indices]
        )
        return RecalibrationReport(
            column_names=[self.column_names[i] for i in indices],
            source_names=self.source_names,
            nominal_values=nominal_values,
            updated_source=self.updated_source,
            old_source_covariance=self.old_source_covariance,
            new_source_covariance=self.new_source_covariance,
            old_covariance=self.old_covariance[numpy.ix_(indices, indices)],
            new_covariance=self.new_covariance[numpy.ix_(indices, indices)],
            contributions=[
                SourcePairContribution(
                    contribution.source_pair,
                    contribution.matrix[numpy.ix_(indices, indices)],
                )
                for contribution in self.contributions
            ],
        )

    def explain(self, column_a, column_b):
        """
        Explain the change of covariance between two columns (names or
        indices): return a list of (source pair, change) tuples, one
        per pair of error sources that contributes to the change,
        followed by a ("total", total change) tuple.  The individual
        changes sum to the total change of covariance between the two
        columns.
        """
        i = _resolve_index(column_a, self.column_names, "column")
        j = _resolve_index(column_b, self.column_names, "column")
        explanation = [
            (contribution.source_pair, contribution.matrix[i, j])
            for contribution in self.contributions
            if contribution.matrix[i, j] != 0
        ]
        explanation.append(("total", self.delta_covariance[i, j]))
        return explanation

    def to_text(self, columns=None):
        """
        Return an itemized text report of the revised uncertainties
        and of the covariance changes between columns.

        columns -- optional subset of columns (names or indices) to
        report on; the covariance changes in the subset are still
        broken down by the pair of error sources they come from.
        """
        report = self if columns is None else self.subset(columns)
        lines = []
        lines.append(
            "Recalibration of error source {!r}: "
            "nominal values unchanged".format(report.updated_source)
        )
        lines.append("")
        lines.append("Standard uncertainties of the columns:")
        for (i, name) in enumerate(report.column_names):
            lines.append(
                "  {}: {:.6g} -> {:.6g} (change {:+.6g})".format(
                    name,
                    report.old_std_devs[i],
                    report.new_std_devs[i],
                    report.delta_std_devs[i],
                )
            )
        lines.append("")
        lines.append("Covariance changes between columns:")
        any_change = False
        n_columns = len(report.column_names)
        for (i, j) in itertools.combinations(range(n_columns), 2):
            delta = report.delta_covariance[i, j]
            if delta == 0:
                continue
            any_change = True
            lines.append(
                "  {} vs {}: {:+.6g} -> {:+.6g} (change {:+.6g})".format(
                    report.column_names[i],
                    report.column_names[j],
                    report.old_covariance[i, j],
                    report.new_covariance[i, j],
                    delta,
                )
            )
            for (source_pair, change) in report.explain(
                report.column_names[i], report.column_names[j]
            )[:-1]:
                lines.append(
                    "    from sources {} x {}: {:+.6g}".format(
                        source_pair[0], source_pair[1], change
                    )
                )
        if not any_change:
            lines.append("  (no covariance change between the columns)")
        return "\n".join(lines)
