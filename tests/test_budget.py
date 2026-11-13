import math

import pytest

from uncertainties import (
    UNCLASSIFIED,
    VarianceBudget,
    correlation_matrix,
    correlated_values,
    covariance_matrix,
    ufloat,
    variance_budget,
)
from uncertainties.budget import Contribution


def _budget_matrix_square(budget, sensitivities, covariance):
    return sum(
        sensitivities[row] * covariance[row][column] * sensitivities[column]
        for row in range(len(sensitivities))
        for column in range(len(sensitivities))
    )


def test_single_covariance_entry_matches_full_matrix():
    sensitivities = [2.0, -1.0, 0.5]
    covariance = {
        ("a", "a"): 0.4,
        ("b", "b"): 0.2,
        ("c", "c"): 0.3,
        ("a", "b"): -0.1,
    }
    matrix_budget = variance_budget(
        sensitivities,
        [
            [0.4, -0.1, 0.0],
            [-0.1, 0.2, 0.0],
            [0.0, 0.0, 0.3],
        ],
        sources=["a", "b", "c"],
        groups=["A", "B", "C"],
    )
    mapping_budget = variance_budget(
        sensitivities, covariance, sources=["a", "b", "c"], groups=["A", "B", "C"]
    )

    assert math.isclose(mapping_budget.total_variance, matrix_budget.total_variance)
    assert math.isclose(
        mapping_budget.between[("A", "B")], matrix_budget.between[("A", "B")]
    )

    with pytest.raises(ValueError):
        variance_budget(
            sensitivities,
            {("a", "b"): -0.1, ("b", "a"): 0.2},
            sources=["a", "b", "c"],
        )


def test_group_totals_and_aliases():
    sensitivity = [1.0, 1.0, 0.5]
    covariance = [
        [0.4, 0.1, 0.0],
        [0.1, 0.2, 0.0],
        [0.0, 0.0, 0.3],
    ]
    budget = variance_budget(sensitivity, covariance, groups=["A", "B", None])

    totals = budget.group_totals
    assert math.isclose(
        math.fsum(totals.values()),
        budget.total_variance + math.fsum(budget.between.values()),
    )
    assert math.isclose(totals["A"], budget.within["A"] + budget.between[("A", "B")])
    assert math.isclose(totals[None], budget.within[None], abs_tol=1e-15)

    assert budget.within_groups == budget.within
    assert budget.between_groups == budget.between
    assert dict(budget.within_rows()) == budget.within
    assert dict(budget.between_rows()) == budget.between
    assert budget.regrouped_budget({"A": "X"}) is not None


def test_affine_without_groups_is_all_unclassified():
    x = ufloat(1.0, 0.1, "x")
    y = ufloat(2.0, 0.2, "y")
    result = 3.0 * x + 2.0 * y

    budget = variance_budget(result)
    assert budget.groups == (None,)
    assert set(budget.unclassified_sources) == {x, y}
    assert math.isclose(budget.within[None], result.std_dev**2)
    assert math.isclose(budget.group_sum, result.std_dev**2)
    assert budget.check()


def test_signed_within_and_between_contributions():
    sensitivities = [2.0, -1.5, 0.5]
    covariance = [
        [1.0, -0.6, 0.2],
        [-0.6, 0.8, -0.1],
        [0.2, -0.1, 0.3],
    ]
    budget = variance_budget(
        sensitivities, covariance, groups=["process_a", "process_b", "process_c"]
    )

    expected = _budget_matrix_square(budget, sensitivities, covariance)
    assert math.isclose(budget.total_variance, expected)
    assert math.isclose(budget.group_sum, expected)
    assert math.isclose(budget.variance_discrepancy, 0.0)

    assert math.isclose(budget.within["process_a"], 2.0**2 * 1.0)
    assert math.isclose(budget.within["process_b"], (-1.5) ** 2 * 0.8)
    assert math.isclose(budget.within["process_c"], 0.5**2 * 0.3)

    between = budget.between_contribution
    assert math.isclose(between("process_a", "process_b"), 2 * 2.0 * -1.5 * -0.6)
    assert math.isclose(between("process_a", "process_c"), 2 * 2.0 * 0.5 * 0.2)
    assert math.isclose(between("process_b", "process_c"), 2 * -1.5 * 0.5 * -0.1)
    assert math.isclose(
        between("process_b", "process_a"), between("process_a", "process_b")
    )
    assert budget.check()


def test_unclassified_sources_remain_in_the_budget():
    sensitivities = [1.0, 2.0, 0.5]
    covariance = [
        [0.4, 0.1, 0.0],
        [0.1, 0.9, -0.2],
        [0.0, -0.2, 0.25],
    ]
    budget = variance_budget(
        sensitivities,
        covariance,
        groups={"a": "A", "b": "B"},
        sources=["a", "b", "unassigned"],
    )

    assert budget.unclassified_sources == ("unassigned",)
    assert math.isclose(budget.unclassified_within, 0.5**2 * 0.25)
    assert math.isclose(
        budget.unclassified_between,
        2 * 1.0 * 0.5 * 0.0 + 2 * 2.0 * 0.5 * -0.2,
    )
    assert math.isclose(
        budget.unclassified, budget.unclassified_within + budget.unclassified_between
    )
    assert math.isclose(budget.group_sum, budget.total_variance)
    assert math.isclose(budget.within[UNCLASSIFIED], budget.unclassified_within)


def test_negative_cross_terms_are_retained():
    sensitivity = [1.0, 1.0]
    covariance = [[0.1, -0.09], [-0.09, 0.1]]
    budget = variance_budget(sensitivity, covariance, groups=["A", "B"])

    assert budget.between[("A", "B")] < 0
    assert math.isclose(budget.between[("A", "B")], 2 * -0.09)
    assert math.isclose(budget.total_variance, 0.1 + 0.1 - 2 * 0.09)
    assert math.isclose(
        sum(budget.within.values()) + sum(budget.between.values()),
        budget.total_variance,
    )


def test_merge_moves_between_term_into_merged_group():
    sensitivities = [1.0, 2.0, 0.5, 1.5]
    covariance = [
        [1.0, 0.3, 0.1, -0.2],
        [0.3, 0.8, -0.1, 0.0],
        [0.1, -0.1, 0.6, 0.05],
        [-0.2, 0.0, 0.05, 0.4],
    ]
    budget = variance_budget(sensitivities, covariance, groups=["A", "B", "C", "A"])
    merged = budget.merge_groups("A", "B", new_group="AB")

    assert math.isclose(merged.total_variance, budget.total_variance)
    assert math.isclose(
        merged.within["AB"],
        budget.within["A"] + budget.within["B"] + budget.between[("A", "B")],
    )
    assert math.isclose(
        merged.between[("AB", "C")],
        budget.between[("A", "C")] + budget.between[("B", "C")],
    )
    assert ("A", "B") not in merged.between
    assert math.isclose(merged.group_sum, budget.group_sum)
    assert merged.check()

    relabelled = budget.regroup({"A": "AB", "B": "AB"})
    assert math.isclose(relabelled.within["AB"], merged.within["AB"])
    assert math.isclose(relabelled.group_sum, budget.total_variance)


def test_budget_from_affine_result_and_covariance_matrix():
    x = ufloat(10.0, 1.0, "x")
    y = ufloat(5.0, 0.8, "y")
    common = ufloat(0.0, 0.5, "common")
    result = 2.0 * x + 3.0 * y + 4.0 * common

    budget = variance_budget(result, groups={"x": "A", "y": "B", "common": "shared"})
    assert math.isclose(budget.total_variance, result.std_dev**2)
    assert math.isclose(budget.group_sum, result.std_dev**2)

    variables = [x, y, common]
    covariance = covariance_matrix(variables)
    covariance_budget = variance_budget(
        result,
        covariance,
        groups={"x": "A", "y": "B", "common": "shared"},
        sources=variables,
    )
    assert math.isclose(covariance_budget.total_variance, result.std_dev**2)


def test_budget_from_correlated_values():
    variables = correlated_values(
        [1.0, 2.0, 3.0],
        [[0.04, -0.012, 0.0], [-0.012, 0.09, 0.015], [0.0, 0.015, 0.02]],
        tags=["a", "b", "c"],
    )
    result = 3.0 * variables[0] - 2.0 * variables[1] + variables[2]
    budget = variance_budget(result, groups={"a": "first", "b": "first", "c": "second"})

    assert math.isclose(budget.total_variance, result.std_dev**2)
    assert math.isclose(budget.group_sum, result.std_dev**2)
    assert math.isclose(
        correlation_matrix(list(variables))[0, 1],
        -0.012 / math.sqrt(0.04 * 0.09),
    )


def test_api_aliases_and_entry_ledger():
    sensitivity = {"a": 2.0, "b": -1.0}
    covariance = {("a", "a"): 0.5, ("b", "b"): 0.2, ("a", "b"): -0.1}
    budget = variance_budget(sensitivity, covariance, groups={"a": "A", "b": "B"})

    assert isinstance(budget, VarianceBudget)
    assert math.isclose(budget.total, budget.total_variance)
    assert math.isclose(budget.variance, budget.total_variance)
    assert budget.entries

    entries = budget.contributions
    assert len(entries) == 3
    assert all(isinstance(entry, Contribution) for entry in entries)
    assert math.isclose(
        math.fsum(entry.value for entry in entries), budget.total_variance
    )
    assert math.isclose(
        math.fsum(budget.within.values()) + math.fsum(budget.between.values()),
        budget.total_variance,
    )


def test_invalid_inputs():
    with pytest.raises(ValueError):
        variance_budget([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]], groups=["A"])

    with pytest.raises(ValueError):
        variance_budget([1.0], [[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError):
        variance_budget([1.0, 2.0], [[1.0]])

    with pytest.raises(ValueError):
        variance_budget([1.0], [[1.0]], groups=["A"]).merge_groups("A", "B")

    with pytest.raises(ValueError):
        variance_budget([1.0], [[1.0]], groups=["A"]).regroup(["A", "B"])
