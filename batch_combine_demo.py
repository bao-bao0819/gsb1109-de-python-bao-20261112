"""
Demonstration: combine correlated batch means with source identities.

Scenario
--------
Five batches measure the same quantity.

* B1, B2, B3 use calibration artefact CAL-A: its calibration error is a
  single shared error source, so these three batches are correlated.
* B4 and B5 are duplicate submissions of the *same* sample: they share both
  calibration artefact CAL-B and the sample-preparation error SMP-X and carry
  no independent repeat noise, which makes them perfectly correlated.  Each
  also has the smallest reported marginal uncertainty, so a naive
  inverse-variance average would treat them as two precise, independent
  results and overstate the accuracy.

Run with:  python batch_combine_demo.py
"""

import numpy as np

from uncertainties import correlation_matrix

from uncertainties.batch_combine import (
    BatchResult,
    ErrorSource,
    combine_batches,
    format_report,
)


def main():
    batches = [
        BatchResult("B1", 10.00, private_std_dev=0.20),
        BatchResult("B2", 10.20, private_std_dev=0.25),
        BatchResult("B3", 9.80, private_std_dev=0.40),
        BatchResult("B4", 10.10, private_std_dev=0.0),
        BatchResult("B5", 10.10, private_std_dev=0.0),
    ]

    error_sources = [
        ErrorSource("CAL-A", 0.50, ("B1", "B2", "B3")),
        ErrorSource("CAL-B", 0.40, ("B4", "B5")),
        ErrorSource("SMP-X", 0.15, ("B4", "B5")),
    ]

    result = combine_batches(batches, error_sources)

    print(format_report(result))
    print()

    marginal_std = np.sqrt(np.diag(result["covariance"]))
    print("reported marginal standard deviations:")
    for batch_id, std in zip(result["batch_ids"], marginal_std):
        print(f"  {batch_id}: {std:.4f}")
    print()

    corr = np.array(correlation_matrix(result["measurements"]))
    print("correlation matrix (1.0000 = fully shared errors):")
    header = "       " + "".join(f"{bid:>8s}" for bid in result["batch_ids"])
    print(header)
    for batch_id, row in zip(result["batch_ids"], corr):
        print(f"{batch_id:>6s} " + "".join(f"{value:8.4f}" for value in row))
    print()

    weights = result["weights"]
    pair_weight = weights[3] + weights[4]
    print(
        "B4 and B5 are perfectly correlated: together they carry "
        f"{pair_weight:.4f} of the total weight, exactly what a single"
    )
    print("submission would carry - the duplicate row adds no precision.")
    print()

    naive_precision = 1.0 / np.sum(1.0 / marginal_std**2)
    naive_weights = (1.0 / marginal_std**2) / np.sum(
        1.0 / marginal_std**2
    )
    print(
        "naive inverse-variance average (correlations ignored) would claim"
        f" std dev {np.sqrt(naive_precision):.4f}"
    )
    print(
        "  with weights "
        + ", ".join(
            f"{bid}={w:.4f}"
            for bid, w in zip(result["batch_ids"], naive_weights)
        )
    )
    print(
        "the source-aware combination correctly reports "
        f"{result['combined_std_dev']:.4f} instead."
    )


if __name__ == "__main__":
    main()
