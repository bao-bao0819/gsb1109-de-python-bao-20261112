"""Regroupable variance budget for a weighted production report.

Run with: python examples/variance_budget_report.py
"""

from uncertainties import variance_budget


def report(budget, title):
    print(title)
    print(
        "  variance: {:.6f}   std. dev.: {:.4f}".format(
            budget.total_variance, budget.std_dev
        )
    )
    print("  within-group contributions")
    for group, value in budget.within.items():
        print("    {!r:>12}: {:+0.6f}".format(group, value))
    print("  between-group contributions")
    for (first, second), value in budget.between.items():
        print("    {!r:>8} <-> {!r:<8}: {:+0.6f}".format(first, second, value))
    print("  unclassified (still booked): {:+0.6f}".format(budget.unclassified))
    print("  sum of all entries: {:+0.6f}".format(budget.group_sum))
    print()


def main():
    sensitivities = {
        "press_rate": 1.8,
        "press_scrap": -0.6,
        "dryer_rate": 1.2,
        "dryer_scrap": -0.4,
        "packaging_rate": 0.9,
        "shared_shift": 1.0,
    }
    covariance = {
        ("press_rate", "press_rate"): 0.36,
        ("press_scrap", "press_scrap"): 0.08,
        ("dryer_rate", "dryer_rate"): 0.20,
        ("dryer_scrap", "dryer_scrap"): 0.05,
        ("packaging_rate", "packaging_rate"): 0.12,
        ("shared_shift", "shared_shift"): 0.25,
        ("press_rate", "dryer_rate"): 0.06,
        ("press_rate", "shared_shift"): -0.04,
        ("dryer_rate", "shared_shift"): 0.03,
        ("packaging_rate", "shared_shift"): 0.02,
        ("press_rate", "packaging_rate"): -0.03,
    }

    budget = variance_budget(
        sensitivities,
        covariance,
        groups={
            "press_rate": "press",
            "press_scrap": "press",
            "dryer_rate": "dryer",
            "dryer_scrap": "dryer",
            "packaging_rate": "packaging",
        },
    )
    report(budget, "Before regrouping")
    report(budget.merge_groups("press", "dryer"), "After merging press and dryer")
    print("unclassified sources:", budget.unclassified_sources)


if __name__ == "__main__":
    main()
