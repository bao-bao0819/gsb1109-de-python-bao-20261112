# coding=utf-8

"""
Grouped variance budgets for quantities with uncertainties.

The total variance of a quantity is decomposed into signed contributions
from each source and from pairs of sources.  Contributions are then
aggregated into group-local contributions and cross-group contributions.
Because the decomposition keeps the individual signed pair terms, groups
can later be merged or relabelled without changing the total variance.
"""

from math import fsum, isfinite, sqrt
from collections.abc import Mapping

from uncertainties.core import AffineScalarFunc

__all__ = [
    "UNCLASSIFIED",
    "Contribution",
    "VarianceBudget",
    "variance_budget",
    "error_budget",
    "uncertainty_budget",
    "grouped_variance",
    "merge_groups",
]

UNCLASSIFIED = None


class Contribution(object):
    """A signed variance contribution from one source or one source pair."""

    __slots__ = ("source_a", "source_b", "group_a", "group_b", "value")

    def __init__(self, source_a, source_b, group_a, group_b, value):
        self.source_a = source_a
        self.source_b = source_b
        self.group_a = group_a
        self.group_b = group_b
        self.value = float(value)

    @property
    def is_within_group(self):
        return self.group_a == self.group_b

    @property
    def is_between_groups(self):
        return self.group_a != self.group_b

    @property
    def is_diagonal(self):
        return self.source_a is self.source_b

    def as_dict(self):
        return {
            "source_a": self.source_a,
            "source_b": self.source_b,
            "group_a": self.group_a,
            "group_b": self.group_b,
            "value": self.value,
        }

    def __repr__(self):
        return (
            "Contribution(source_a={!r}, source_b={!r}, group_a={!r}, "
            "group_b={!r}, value={!r})"
        ).format(
            self.source_a,
            self.source_b,
            self.group_a,
            self.group_b,
            self.value,
        )


def _is_sequence_of_numbers(value):
    if isinstance(value, (str, bytes, Mapping)):
        return False
    try:
        for item in value:
            float(item)
    except (TypeError, ValueError):
        return False
    return True


def _is_group_collection(value):
    if isinstance(value, (str, bytes, Mapping)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _source_group(source, groups):
    if source in groups:
        return groups[source]
    tag = getattr(source, "tag", None)
    if tag is not None and tag in groups:
        return groups[tag]
    return UNCLASSIFIED


def _groups_for_labels(groups, labels):
    if groups is None:
        return tuple(UNCLASSIFIED for _ in labels)

    if callable(groups):
        return tuple(groups(label) for label in labels)

    if isinstance(groups, Mapping):
        labels_set = set(labels)
        membership_style = not labels_set.intersection(groups) and all(
            _is_group_collection(value) and not isinstance(value, (str, bytes))
            for value in groups.values()
        )
        if membership_style:
            assignments = {}
            for group, members in groups.items():
                if isinstance(members, (str, bytes)):
                    members = (members,)
                for member in members:
                    assignments[member] = group
            return tuple(_source_group(label, assignments) for label in labels)
        return tuple(_source_group(label, groups) for label in labels)

    if _is_sequence_of_numbers(groups):
        raise TypeError("group labels cannot be numbers")

    assignments = tuple(groups)
    if len(assignments) != len(labels):
        raise ValueError(
            "groups must contain one group label per source "
            "({} groups for {} sources)".format(len(assignments), len(labels))
        )
    return assignments


def _normalise_sensitivity(sensitivity, sources, covariance):
    if isinstance(sensitivity, AffineScalarFunc):
        linear_part = sensitivity._linear_part
        if not linear_part.expanded():
            linear_part.expand()
            linear_part.linear_combo.default_factory = None
        derivatives = linear_part.linear_combo
        labels = tuple(derivatives)
        values = tuple(float(derivatives[label]) for label in labels)
        if sources is not None:
            labels = tuple(sources)
            values = tuple(float(derivatives.get(label, 0.0)) for label in labels)
        if covariance is None:
            covariance = tuple(
                tuple(
                    float(label.std_dev) ** 2 if row == column else 0.0
                    for column, _ in enumerate(labels)
                )
                for row, label in enumerate(labels)
            )
        return labels, values, covariance

    if isinstance(sensitivity, Mapping):
        labels = tuple(sensitivity)
        values = tuple(float(value) for value in sensitivity.values())
    else:
        try:
            values = tuple(float(value) for value in sensitivity)
        except (TypeError, ValueError):
            raise TypeError(
                "sensitivity must be an AffineScalarFunc, a mapping of source "
                "to sensitivity, or a sequence of sensitivities"
            )
        labels = tuple(range(len(values)))

    if sources is not None:
        sources = tuple(sources)
        if len(sources) != len(values):
            raise ValueError(
                "sources must contain one label per source ({} labels for {} "
                "sensitivities)".format(len(sources), len(values))
            )
        labels = sources

    return labels, values, covariance


def _covariance_matrix(covariance, labels):
    n_sources = len(labels)
    if covariance is None:
        raise ValueError("a covariance matrix is required for explicit sensitivities")

    if hasattr(covariance, "shape") and hasattr(covariance, "ndim"):
        covariance = covariance.tolist()

    if isinstance(covariance, Mapping):
        matrix = [[0.0] * n_sources for _ in range(n_sources)]
        index_by_label = {label: index for index, label in enumerate(labels)}
        assigned_pairs = {}

        def index_for(key):
            if key in index_by_label:
                return index_by_label[key]
            if isinstance(key, int) and 0 <= key < n_sources:
                return key
            raise KeyError("unknown covariance source {!r}".format(key))

        for key, value in covariance.items():
            if isinstance(key, tuple) and len(key) == 2:
                row, column = (index_for(part) for part in key)
                pair = frozenset((row, column))
                value = float(value)
                if pair in assigned_pairs and assigned_pairs[pair] != value:
                    raise ValueError(
                        "contradictory covariance entries for {!r}".format(key)
                    )
                assigned_pairs[pair] = value
                matrix[row][column] = value
                matrix[column][row] = value
            else:
                index = index_for(key)
                matrix[index][index] = float(value)
        return tuple(tuple(row) for row in matrix)

    if hasattr(covariance, "__iter__") and not isinstance(covariance, (str, bytes)):
        values = list(covariance)
    else:
        raise TypeError("covariance must be a covariance matrix or a mapping")

    if len(values) == n_sources and _is_sequence_of_numbers(values):
        return tuple(
            tuple(
                float(values[row]) if row == column else 0.0
                for column in range(n_sources)
            )
            for row in range(n_sources)
        )

    if len(values) != n_sources:
        raise ValueError(
            "covariance must have one row per source ({} rows for {} sources)".format(
                len(values), n_sources
            )
        )

    matrix = []
    for row in values:
        row_values = tuple(float(value) for value in row)
        if len(row_values) != n_sources:
            raise ValueError(
                "covariance must be square ({} columns for {} sources)".format(
                    len(row_values), n_sources
                )
            )
        matrix.append(row_values)
    return tuple(matrix)


class VarianceBudget(object):
    """
    A signed, regroupable variance budget.

    The atomic decomposition is::

        variance = sum_i sensitivity_i**2 * covariance[i][i]
                 + sum_{i<j} sensitivity_i * sensitivity_j
                            * (covariance[i][j] + covariance[j][i])

    Each diagonal and each source-pair term is kept separately.  Group
    aggregates are always recomputed from these atomic terms, so merging
    or renaming groups transfers the corresponding cross terms into the
    merged group and never changes the total variance.

    ``within`` contains one signed contribution per group.  ``between``
    contains one signed contribution per unordered pair of distinct
    groups.  Sources without a group assignment belong to the
    ``UNCLASSIFIED`` group (``None``) and are still included in both
    mappings.
    """

    __slots__ = (
        "_sources",
        "_sensitivities",
        "_covariance",
        "_groups",
        "_diagonal_terms",
        "_pair_terms",
    )

    def __init__(self, sensitivity, covariance=None, groups=None, sources=None):
        labels, values, covariance = _normalise_sensitivity(
            sensitivity, sources, covariance
        )
        if len(set(labels)) != len(labels):
            raise ValueError(
                "source labels must be unique so that each source has one group"
            )
        covariance = _covariance_matrix(covariance, labels)
        self._sources = tuple(labels)
        self._sensitivities = tuple(values)
        self._covariance = covariance
        self._groups = _groups_for_labels(groups, self._sources)

        self._diagonal_terms = tuple(
            (index, self._sensitivities[index] ** 2 * covariance[index][index])
            for index in range(len(self._sources))
        )
        pair_terms = []
        for first in range(len(self._sources)):
            for second in range(first + 1, len(self._sources)):
                value = (
                    self._sensitivities[first]
                    * self._sensitivities[second]
                    * (covariance[first][second] + covariance[second][first])
                )
                pair_terms.append((first, second, value))
        self._pair_terms = tuple(pair_terms)

        for group in self._groups:
            hash(group)

    @classmethod
    def _from_parts(cls, sources, sensitivities, covariance, groups):
        budget = object.__new__(cls)
        budget._sources = tuple(sources)
        budget._sensitivities = tuple(sensitivities)
        budget._covariance = tuple(tuple(row) for row in covariance)
        budget._groups = tuple(groups)
        diagonal_terms = tuple(
            (
                index,
                budget._sensitivities[index] ** 2 * budget._covariance[index][index],
            )
            for index in range(len(budget._sources))
        )
        pair_terms = []
        for first in range(len(budget._sources)):
            for second in range(first + 1, len(budget._sources)):
                value = (
                    budget._sensitivities[first]
                    * budget._sensitivities[second]
                    * (
                        budget._covariance[first][second]
                        + budget._covariance[second][first]
                    )
                )
                pair_terms.append((first, second, value))
        budget._diagonal_terms = diagonal_terms
        budget._pair_terms = tuple(pair_terms)
        return budget

    @property
    def sources(self):
        """Labels of the sources in covariance-matrix order."""
        return self._sources

    @property
    def sensitivities(self):
        """Metric sensitivity (partial derivative) for each source."""
        return self._sensitivities

    @property
    def covariance(self):
        """The covariance matrix used by the budget."""
        return self._covariance

    @property
    def source_groups(self):
        """Mapping from source label to its current group."""
        return dict(zip(self._sources, self._groups))

    @property
    def groups(self):
        """Group labels present in the budget, in source order."""
        seen = []
        for group in self._groups:
            if group not in seen:
                seen.append(group)
        return tuple(seen)

    @property
    def total_variance(self):
        """Propagated variance, obtained by summing all signed atomic terms."""
        return fsum(value for _, _, value in self._atomic_terms())

    variance = total_variance
    total = total_variance

    @property
    def propagation_variance(self):
        """Variance obtained directly from sensitivity^T * covariance * sensitivity."""
        terms = []
        for row, sensitivity in enumerate(self._sensitivities):
            row_sum = fsum(
                self._covariance[row][column] * self._sensitivities[column]
                for column in range(len(self._sources))
            )
            terms.append(sensitivity * row_sum)
        return fsum(terms)

    @property
    def variance_discrepancy(self):
        """Difference between the grouped and direct propagation variances."""
        return self.total_variance - self.propagation_variance

    @property
    def std_dev(self):
        """Standard deviation corresponding to ``total_variance``."""
        variance = self.total_variance
        return sqrt(variance) if variance >= 0 else float("nan")

    @property
    def within(self):
        """Signed group-local contributions, keyed by group label."""
        order = self.groups
        sums = {group: [] for group in order}
        for index, value in self._diagonal_terms:
            sums[self._groups[index]].append(value)
        for first, second, value in self._pair_terms:
            first_group = self._groups[first]
            second_group = self._groups[second]
            if first_group == second_group:
                sums[first_group].append(value)
        return {group: fsum(values) for group, values in sums.items()}

    within_groups = within

    @property
    def between(self):
        """Signed cross-group contributions, keyed by an unordered group pair."""
        order = {group: index for index, group in enumerate(self.groups)}
        sums = {}
        for first, second, value in self._pair_terms:
            first_group = self._groups[first]
            second_group = self._groups[second]
            if first_group == second_group:
                continue
            if order[first_group] > order[second_group]:
                first_group, second_group = second_group, first_group
            key = (first_group, second_group)
            sums.setdefault(key, []).append(value)
        return {key: fsum(values) for key, values in sums.items()}

    between_groups = between
    cross_group_contributions = between

    def within_rows(self):
        """The within-group contributions as sorted report rows."""
        return sorted(self.within.items(), key=lambda item: str(item[0]))

    def between_rows(self):
        """The between-group contributions as sorted report rows."""
        return sorted(self.between.items(), key=lambda item: str(item[0]))

    def between_contribution(self, first_group, second_group):
        """Return a cross-group contribution independently of key order."""
        if first_group == second_group:
            return self.within.get(first_group, 0.0)
        order = {group: index for index, group in enumerate(self.groups)}
        if order[first_group] > order[second_group]:
            first_group, second_group = second_group, first_group
        return self.between.get((first_group, second_group), 0.0)

    @property
    def group_totals(self):
        """Each group's within contribution plus its cross contributions."""
        totals = {group: [value] for group, value in self.within.items()}
        for (first, second), value in self.between.items():
            totals.setdefault(first, [])
            totals.setdefault(second, [])
            totals[first].append(value)
            totals[second].append(value)
        return {group: fsum(values) for group, values in totals.items()}

    @property
    def group_sum(self):
        """Sum of all within-group and between-group entries."""
        return fsum(list(self.within.values()) + list(self.between.values()))

    @property
    def unclassified_sources(self):
        """Sources that have no group assignment."""
        return tuple(
            source
            for source, group in zip(self._sources, self._groups)
            if group is UNCLASSIFIED
        )

    @property
    def unclassified_within(self):
        """Group-local contribution of unclassified sources."""
        return self.within.get(UNCLASSIFIED, 0.0)

    @property
    def unclassified_between(self):
        """Cross contributions between unclassified sources and any group."""
        return fsum(value for key, value in self.between.items() if UNCLASSIFIED in key)

    @property
    def unclassified(self):
        """All variance contributions involving an unclassified source."""
        return self.unclassified_within + self.unclassified_between

    unclassified_contribution = unclassified

    def _atomic_terms(self):
        terms = []
        for index, value in self._diagonal_terms:
            source = self._sources[index]
            terms.append((source, source, value))
        for first, second, value in self._pair_terms:
            terms.append((self._sources[first], self._sources[second], value))
        return terms

    @property
    def contributions(self):
        """Signed atomic contributions, one per source and source pair."""
        terms = []
        for index, value in self._diagonal_terms:
            source = self._sources[index]
            group = self._groups[index]
            terms.append(Contribution(source, source, group, group, value))
        for first, second, value in self._pair_terms:
            terms.append(
                Contribution(
                    self._sources[first],
                    self._sources[second],
                    self._groups[first],
                    self._groups[second],
                    value,
                )
            )
        return tuple(terms)

    entries = contributions

    def __iter__(self):
        return iter(self.contributions)

    def __len__(self):
        return len(self._sources) + len(self._pair_terms)

    def regroup_sources(self, groups):
        """Return a new budget with a new source-to-group assignment."""
        if isinstance(groups, Mapping):
            labels_set = set(self._sources)
            membership_style = not labels_set.intersection(groups) and all(
                _is_group_collection(value) and not isinstance(value, (str, bytes))
                for value in groups.values()
            )
            if membership_style:
                assignments = {}
                for group, members in groups.items():
                    if isinstance(members, (str, bytes)):
                        members = (members,)
                    for member in members:
                        assignments[member] = group
                new_groups = tuple(
                    _source_group(source, assignments) for source in self._sources
                )
            else:
                new_groups = tuple(
                    _source_group(source, groups) if source in groups else group
                    for source, group in zip(self._sources, self._groups)
                )
        elif callable(groups):
            new_groups = tuple(groups(source) for source in self._sources)
        else:
            new_groups = tuple(groups)
            if len(new_groups) != len(self._sources):
                raise ValueError(
                    "groups must contain one group label per source "
                    "({} groups for {} sources)".format(
                        len(new_groups), len(self._sources)
                    )
                )
        return self._from_parts(
            self._sources, self._sensitivities, self._covariance, new_groups
        )

    def regroup(self, groups=None, *, by="auto", merge=None, new_group=None):
        """
        Return a new budget with relabelled or merged groups.

        ``groups`` is normally a mapping from current group labels to new
        group labels.  This is what is needed to merge two groups, for
        example with ``{"A": "AB", "B": "AB"}``.  A source-to-group
        mapping can also be supplied explicitly with ``by="source"``.
        ``merge=(group_a, group_b)`` is a shorthand for merging two groups.
        """
        if merge is not None:
            if len(tuple(merge)) != 2:
                raise ValueError("merge must contain exactly two groups")
            return self.merge_groups(*merge, new_group=new_group)

        if groups is None:
            return self

        if by not in ("auto", "group", "source"):
            raise ValueError("by must be 'auto', 'group' or 'source'")

        if callable(groups) or not isinstance(groups, Mapping):
            return self.regroup_sources(groups)

        current_groups = set(self.groups)
        mapping_keys = set(groups)
        treat_as_group = by == "group" or (
            by == "auto" and mapping_keys.issubset(current_groups)
        )
        if treat_as_group:
            new_groups = tuple(groups.get(group, group) for group in self._groups)
            return self._from_parts(
                self._sources, self._sensitivities, self._covariance, new_groups
            )

        return self.regroup_sources(groups)

    regrouped = regroup
    regrouped_budget = regroup

    def merge_groups(self, first_group, second_group, new_group=None):
        """
        Return a new budget with two groups replaced by one group.

        The old cross contribution between the two groups becomes part of
        the new group-local contribution.  Cross contributions with every
        other group are combined additively.
        """
        if first_group not in self.groups or second_group not in self.groups:
            raise ValueError("both groups must exist in the budget")
        if first_group == second_group:
            return self
        if new_group is None:
            if isinstance(first_group, str) and isinstance(second_group, str):
                new_group = "{}+{}".format(first_group, second_group)
            else:
                new_group = (first_group, second_group)
        new_groups = tuple(
            new_group if group in (first_group, second_group) else group
            for group in self._groups
        )
        return self._from_parts(
            self._sources, self._sensitivities, self._covariance, new_groups
        )

    merge = merge_groups

    def as_dict(self):
        """Return the reportable budget as ordinary dictionaries."""
        return {
            "total_variance": self.total_variance,
            "std_dev": self.std_dev,
            "within": dict(self.within),
            "between": dict(self.between),
            "unclassified": self.unclassified,
            "unclassified_sources": self.unclassified_sources,
        }

    def check(self, rel_tol=1e-12, abs_tol=1e-15):
        """Raise ``AssertionError`` if the grouped ledger does not balance."""
        difference = self.group_sum - self.total_variance
        scale = max(abs(self.total_variance), abs_tol)
        if abs(difference) > max(abs_tol, rel_tol * scale):
            raise AssertionError(
                "group contributions do not sum to the propagated variance: "
                "{} != {}".format(self.group_sum, self.total_variance)
            )
        if not isfinite(difference):
            raise AssertionError("variance budget contains non-finite entries")
        return True

    def __repr__(self):
        return ("VarianceBudget(sources={}, groups={}, total_variance={!r})").format(
            len(self._sources), len(self.groups), self.total_variance
        )


def variance_budget(sensitivity, covariance=None, groups=None, sources=None):
    """
    Build a regroupable variance budget from sources and sensitivities.

    Parameters
    ----------
    sensitivity:
        Either an :class:`AffineScalarFunc` result (its derivatives are
        used as sensitivities, and independent variable variances as the
        covariance), a mapping of source labels to sensitivities, or a
        sequence of sensitivities.
    covariance:
        Covariance matrix in source order, a mapping of source-pair keys
        to covariances (diagonal entries may use a single key), or
        ``None`` when ``sensitivity`` is an ``AffineScalarFunc``.
    groups:
        Mapping of source labels (or variable tags) to group labels.
        Sources without an entry are assigned to ``UNCLASSIFIED``.  A
        reverse mapping of group labels to source collections, a sequence
        aligned with the sources, and a callable taking a source are also
        accepted.
    sources:
        Optional source labels for a sensitivity sequence or an
        ``AffineScalarFunc``.

    The returned budget keeps every signed diagonal and source-pair
    covariance term.  Its ``within`` mapping contains group-local terms
    and its ``between`` mapping contains cross-group terms.  Their sum is
    the propagated variance, including negative covariance terms and
    sources with no group assignment.
    """
    return VarianceBudget(sensitivity, covariance, groups, sources)


error_budget = variance_budget
uncertainty_budget = variance_budget
grouped_variance = variance_budget


def merge_groups(budget, first_group, second_group, new_group=None):
    """Return ``budget`` with two groups merged into one."""
    if not isinstance(budget, VarianceBudget):
        raise TypeError("budget must be a VarianceBudget")
    return budget.merge_groups(first_group, second_group, new_group=new_group)
