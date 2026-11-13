import math

import pytest

from uncertainties import ufloat, correlated_values
from uncertainties.leading import (
    LeadChain,
    LeadEvidence,
    LeadRegistry,
    lead_margin,
    minimal_leading_evidence,
    pair_uncertainty,
    robust_leading_pairs,
)


def shared_bias_batches():
    """
    Three measurements that share one bias term.  Two of them have the
    same nominal value and are tied.
    """
    bias = ufloat(0.0, 2.0, "shared bias")
    noise_a = ufloat(0.0, 0.5, "noise A")
    noise_b = ufloat(0.0, 0.5, "noise B")
    noise_c = ufloat(0.0, 0.5, "noise C")
    return {
        "A": 10.0 + bias + noise_a,
        "B": 8.0 + bias + noise_b,
        "C": 10.0 + bias + noise_c,
    }


def test_pair_uncertainty_cancels_shared_terms():
    measurements = shared_bias_batches()
    leader, laggard = measurements["A"], measurements["B"]

    # The shared bias cancels in the difference: only the two
    # independent 0.5-standard-deviation noise terms remain.
    assert pair_uncertainty(leader, laggard) == pytest.approx(math.sqrt(0.5))
    assert pair_uncertainty(leader, laggard) == pytest.approx(
        (leader - laggard).std_dev
    )

    # Judging from the marginal error bars instead would assume the
    # measurements independent and overstate the difference error.
    marginal_difference_std = math.sqrt(leader.std_dev**2 + laggard.std_dev**2)
    assert marginal_difference_std > pair_uncertainty(leader, laggard)


def test_shared_bias_changes_the_verdict():
    measurements = shared_bias_batches()
    registry = LeadRegistry(measurements, k=1.0)

    # Correlated criterion: margin 2 - sqrt(0.5) > 0, so A leads B.
    assert registry.is_leading("A", "B")
    assert registry.margin("A", "B") == pytest.approx(2.0 - math.sqrt(0.5))

    # The marginal error bars would reject the same pair:
    naive_margin = (
        measurements["A"].nominal_value
        - measurements["B"].nominal_value
        - math.sqrt(measurements["A"].std_dev ** 2 + measurements["B"].std_dev ** 2)
    )
    assert naive_margin < 0.0

    # The reverse pair never leads.
    assert not registry.is_leading("B", "A")


def test_tied_batches_have_no_edge():
    measurements = shared_bias_batches()
    registry = LeadRegistry(measurements, k=1.0)

    # A and C share the same nominal value: no margin can be positive.
    assert not registry.is_leading("A", "C")
    assert not registry.is_leading("C", "A")
    assert registry.margin("A", "C") == pytest.approx(-math.sqrt(0.5))
    assert all(pair.leader != "C" or pair.laggard != "A" for pair in registry.pairs())


def test_robust_leading_pairs_returns_margins():
    measurements = shared_bias_batches()
    pairs = robust_leading_pairs(measurements, k=1.0)

    assert pairs == [
        LeadEvidence("A", "B", pytest.approx(2.0 - math.sqrt(0.5))),
        LeadEvidence("C", "B", pytest.approx(2.0 - math.sqrt(0.5))),
    ]
    assert all(pair.margin > 0.0 for pair in pairs)


def test_k_controls_strictness():
    high = ufloat(5.0, 10.0)
    low = ufloat(4.0, 10.0)
    measurements = {"high": high, "low": low}

    # With k = 0 the nominal gap alone establishes the lead.
    assert LeadRegistry(measurements, k=0.0).is_leading("high", "low")
    assert lead_margin(high, low, 0.0) == pytest.approx(1.0)

    # With k = 1 the uncertainty dominates the gap.
    assert not LeadRegistry(measurements, k=1.0).is_leading("high", "low")
    assert lead_margin(high, low, 1.0) == pytest.approx(1.0 - math.sqrt(200.0))

    with pytest.raises(ValueError):
        LeadRegistry(measurements, k=-1.0)


def test_exact_measurements_have_zero_uncertainty():
    registry = LeadRegistry({"high": 5.0, "low": 2.0}, k=3.0)
    assert pair_uncertainty(5.0, 2.0) == 0.0
    assert registry.margin("high", "low") == pytest.approx(3.0)
    assert registry.is_leading("high", "low")


def test_transitive_reduction_keeps_minimum_evidence():
    measurements = {
        "A": ufloat(20.0, 0.1),
        "B": ufloat(12.0, 0.1),
        "C": ufloat(5.0, 0.1),
    }
    registry = LeadRegistry(measurements, k=1.0)

    direct_pairs = {(pair.leader, pair.laggard) for pair in registry.pairs()}
    assert direct_pairs == {("A", "B"), ("A", "C"), ("B", "C")}

    # A -> C is recoverable through A -> B -> C and is dropped from
    # the stored evidence.
    minimal = minimal_leading_evidence(measurements, k=1.0)
    minimal_pairs = {(pair.leader, pair.laggard) for pair in minimal}
    assert minimal_pairs == {("A", "B"), ("B", "C")}
    assert registry.evidence() == minimal


def test_prove_omitted_pair_uses_shortest_chain_and_true_margin():
    measurements = {
        "A": ufloat(20.0, 0.1),
        "B": ufloat(12.0, 0.1),
        "C": ufloat(5.0, 0.1),
    }
    registry = LeadRegistry(measurements, k=1.0)

    chain = registry.prove("A", "C")
    assert isinstance(chain, LeadChain)
    assert chain.chain == ("A", "B", "C")
    assert chain.hops == (("A", "B"), ("B", "C"))
    assert chain.direct is True
    assert len(chain.hop_margins) == 2
    assert all(margin > 0.0 for margin in chain.hop_margins)

    # The endpoint margin is recomputed from the endpoints themselves,
    # not assembled from the hop margins.
    assert chain.endpoint_margin == pytest.approx(registry.margin("A", "C"))
    assert chain.endpoint_margin != pytest.approx(sum(chain.hop_margins))

    direct = registry.prove("A", "B")
    assert direct.chain == ("A", "B")
    assert direct.hop_margins == (direct.endpoint_margin,)


def test_shortest_chain_breaks_ties_lexicographically():
    measurements = {
        "A": ufloat(20.0, 0.1),
        "B": ufloat(12.0, 0.1),
        "E": ufloat(12.0, 0.1),
        "D": ufloat(0.0, 0.1),
    }
    registry = LeadRegistry(measurements, k=1.0)

    stored = {(pair.leader, pair.laggard) for pair in registry.evidence()}
    # The direct A -> D edge is redundant; B and E are tied and have
    # no edge between them.
    assert ("A", "D") not in stored
    assert ("B", "E") not in stored
    assert ("E", "B") not in stored

    chain = registry.prove("A", "D")
    assert chain.chain == ("A", "B", "D")
    assert chain.endpoint_margin == pytest.approx(registry.margin("A", "D"))


def test_prove_rejects_unknown_or_unprovable_pairs():
    measurements = {
        "A": ufloat(20.0, 0.1),
        "B": ufloat(12.0, 0.1),
        "C": ufloat(5.0, 0.1),
    }
    registry = LeadRegistry(measurements, k=1.0)

    with pytest.raises(KeyError):
        registry.prove("A", "Z")
    with pytest.raises(KeyError):
        registry.prove("Z", "A")
    with pytest.raises(KeyError):
        registry.prove("C", "A")
    with pytest.raises(KeyError):
        registry.prove("A", "A")


def test_correlated_values_round_trip():
    # Build correlated measurements through a covariance matrix and
    # check that the shared-error structure yields the same margin as
    # an explicit shared-variable construction.
    explicit = shared_bias_batches()
    names = ["A", "B", "C"]
    nominals = [explicit[name].nominal_value for name in names]

    from uncertainties import covariance_matrix

    cov_mat = covariance_matrix([explicit[name] for name in names])
    rebuilt = dict(zip(names, correlated_values(nominals, cov_mat)))

    explicit_registry = LeadRegistry(explicit, k=1.0)
    rebuilt_registry = LeadRegistry(rebuilt, k=1.0)

    for leader in names:
        for laggard in names:
            if leader == laggard:
                continue
            assert rebuilt_registry.margin(leader, laggard) == pytest.approx(
                explicit_registry.margin(leader, laggard)
            )

    assert {(pair.leader, pair.laggard) for pair in rebuilt_registry.pairs()} == {
        (pair.leader, pair.laggard) for pair in explicit_registry.pairs()
    }
