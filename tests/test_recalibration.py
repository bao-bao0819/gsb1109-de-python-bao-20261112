import pytest

import uncertainties
from uncertainties import recalibrate, RecalibrationReport


try:
    import numpy as np
except ImportError:
    np = None

pytestmark = pytest.mark.skipif(np is None, reason="numpy is not available")


# Published table: three columns, three error sources, with signed
# sensitivities of each column to each source:
SENSITIVITIES = [
    [1.0, 0.5, 0.0],  # column "A"
    [0.0, 1.0, -1.0],  # column "B"
    [2.0, 0.0, 1.0],  # column "C"
]
SOURCE_NAMES = ["pressure", "temperature", "drift"]
COLUMN_NAMES = ["A", "B", "C"]
# Originally independent error sources:
SOURCE_STDS = [0.1, 0.2, 0.3]


def make_report(**kwargs):
    args = dict(
        sensitivities=SENSITIVITIES,
        source_covariance=SOURCE_STDS,
        updated_source="pressure",
        new_std_dev=0.15,
        new_covariances={"temperature": 0.01, "drift": -0.005},
        nominal_values=[10.0, 20.0, 30.0],
        column_names=COLUMN_NAMES,
        source_names=SOURCE_NAMES,
    )
    args.update(kwargs)
    return recalibrate(**args)


def test_nominal_values_unchanged():
    report = make_report()
    assert report.nominal_values == [10.0, 20.0, 30.0]


def test_revised_covariance_matches_source_update():
    report = make_report()
    sens = np.array(SENSITIVITIES)
    new_source_cov = np.array(
        [
            [0.15**2, 0.01, -0.005],
            [0.01, 0.2**2, 0.0],
            [-0.005, 0.0, 0.3**2],
        ]
    )
    assert np.allclose(report.new_source_covariance, new_source_cov)
    assert np.allclose(report.new_covariance, sens @ new_source_cov @ sens.T)
    assert np.allclose(
        report.old_covariance, sens @ np.diag(np.array(SOURCE_STDS) ** 2) @ sens.T
    )
    assert np.allclose(report.new_std_devs, np.sqrt(np.diag(report.new_covariance)))


def test_contributions_sum_to_delta_covariance():
    report = make_report()
    total = sum((c.matrix for c in report.contributions), np.zeros((3, 3)))
    assert np.allclose(total, report.delta_covariance)
    # The recalibrated source is involved in every source pair:
    for contribution in report.contributions:
        assert "pressure" in contribution.source_pair


def test_variance_unchanged_but_covariance_changed():
    # Column "A" only depends on "temperature", column "B" on both
    # recalibrated "pressure" and "temperature": the variance of "A"
    # cannot change, but its covariance with "B" does through the new
    # pressure-temperature covariance.
    report = recalibrate(
        sensitivities=[[0.0, 1.0], [1.0, 1.0]],
        source_covariance=[0.1, 0.2],
        updated_source=0,
        new_covariances={1: 0.01},
        column_names=["A", "B"],
    )
    assert report.delta_covariance[0, 0] == 0  # variance of A unchanged
    assert report.delta_std_devs[0] == 0
    assert report.delta_covariance[0, 1] == pytest.approx(0.01)
    explanation = report.explain("A", "B")
    # The change is fully attributed to the pressure/temperature pair:
    (source_pair, change), (label, total) = explanation
    assert source_pair == ("source 0", "source 1")
    assert change == pytest.approx(0.01)
    assert label == "total"
    assert total == pytest.approx(0.01)


def test_signed_sensitivities_give_signed_changes():
    # The new pressure-drift covariance is negative; column "B" has a
    # negative sensitivity to drift, column "C" a positive one:
    report = make_report()
    assert report.delta_covariance[1, 2] != 0
    # B x C change comes from the pressure x temperature and the
    # pressure x drift pairs:
    explanation = report.explain("B", "C")
    by_pair = dict(explanation[:-1])
    # sens[B, pressure]*sens[C, drift] + sens[B, drift]*sens[C, pressure]
    # = 0*1 + (-1)*2 = -2, times -0.005:
    assert by_pair[("pressure", "drift")] == pytest.approx((-2.0) * -0.005)
    # sens[B, pressure]*sens[C, temperature]
    # + sens[B, temperature]*sens[C, pressure] = 0*0 + 1*2 = 2, times 0.01:
    assert by_pair[("pressure", "temperature")] == pytest.approx(2.0 * 0.01)
    (label, total) = explanation[-1]
    assert label == "total"
    assert total == pytest.approx(sum(by_pair.values()))


def test_subset_export_keeps_source_pair_attribution():
    report = make_report()
    sub = report.subset(["B", "C"])
    assert isinstance(sub, RecalibrationReport)
    assert sub.column_names == ["B", "C"]
    assert sub.nominal_values == [20.0, 30.0]
    # The subset covariance change is the corresponding block of the
    # full change:
    assert np.allclose(sub.delta_covariance, report.delta_covariance[1:, 1:])
    # ... and it is still fully explained by the source pairs:
    total = sum((c.matrix for c in sub.contributions), np.zeros((2, 2)))
    assert np.allclose(total, sub.delta_covariance)
    explanation = sub.explain("B", "C")
    (label, explained_total) = explanation[-1]
    assert label == "total"
    assert explained_total == pytest.approx(report.delta_covariance[1, 2])
    partial = sum(change for (_, change) in explanation[:-1])
    assert partial == pytest.approx(explained_total)


def test_subset_by_indices_and_single_column():
    report = make_report()
    sub = report.subset([2])
    assert sub.delta_covariance.shape == (1, 1)
    assert sub.delta_covariance[0, 0] == pytest.approx(report.delta_covariance[2, 2])


def test_to_text_mentions_changes_and_sources():
    text = make_report().to_text()
    assert "pressure" in text
    assert "nominal values unchanged" in text
    assert "B vs C" in text
    # Subset export still explains the origin of the difference:
    sub_text = make_report().to_text(columns=["B", "C"])
    assert "from sources pressure x drift" in sub_text
    assert "A vs" not in sub_text


def test_to_text_without_any_covariance_change():
    report = recalibrate(
        sensitivities=[[1.0], [2.0]],
        source_covariance=[0.1],
        updated_source=0,
        new_std_dev=0.2,
    )
    # A single source: only variances change, no covariance *change*
    # breakdown by pairs is needed, but covariances do change here:
    assert report.delta_covariance[0, 1] != 0
    text = report.to_text()
    assert "column 0 vs column 1" in text


def test_invalid_inputs():
    with pytest.raises(ValueError):
        make_report(new_std_dev=-0.1)
    with pytest.raises(KeyError):
        make_report(updated_source="unknown source")
    with pytest.raises(ValueError):
        # The new covariances make the source matrix non positive
        # semi-definite:
        make_report(new_covariances={"temperature": 1.0, "drift": 0.0})
    with pytest.raises(ValueError):
        make_report(new_covariances={"pressure": 0.01})
    with pytest.raises(ValueError):
        recalibrate(
            sensitivities=[1.0, 2.0],  # not 2-dimensional
            source_covariance=[0.1],
            updated_source=0,
        )
    with pytest.raises(ValueError):
        recalibrate(
            sensitivities=[[1.0, 2.0]],
            source_covariance=[[1.0, 0.1], [0.2, 1.0]],  # not symmetric
            updated_source=0,
        )


def test_accessible_from_package_root():
    assert uncertainties.recalibrate is recalibrate
    assert "recalibrate" in uncertainties.__all__
