"""
Tests for partitioned persistence of measurements with uncertainties.

The scenario modeled here is an experiment run with two instruments over two
days.  Both instruments share a calibration bias (one common source of
uncertainty), while each measurement also has its own independent noise.
The measurements are written to separate partitions and read back in both
possible orders:

- the shared calibration bias must keep a single identity across partitions,
  so it cancels out in differences and shows up as a covariance between the
  original results of the two instruments;
- two error sources that merely use the same tag must remain independent;
- any single partition must still be readable on its own, with its marginal
  uncertainties fully reconstructed.
"""

import io
import math
import pickle

import pytest

import uncertainties.core as uncert_core
from uncertainties import ufloat, covariance_matrix
from uncertainties.persistence import (
    VariableRegistry,
    write_partition,
    read_partition,
)


# Standard deviation of the calibration bias shared by both instruments:
BIAS_STD = 0.10
A_NOISE_STD = 0.05
B_NOISE_STD = 0.07


def build_experiment():
    """
    Return (bias, instrument A results, instrument B results).

    Each instrument has one result per day, built from the same calibration
    bias and from instrument-specific independent noise.
    """
    bias = ufloat(0.0, BIAS_STD, tag="calibration_bias")

    a_day1 = 1.00 + bias + ufloat(0.0, A_NOISE_STD, tag="A:noise:day1")
    a_day2 = 1.50 + bias + ufloat(0.0, A_NOISE_STD, tag="A:noise:day2")

    b_day1 = 2.00 + bias + ufloat(0.0, B_NOISE_STD, tag="B:noise:day1")
    b_day2 = 2.50 + bias + ufloat(0.0, B_NOISE_STD, tag="B:noise:day2")

    return bias, (a_day1, a_day2), (b_day1, b_day2)


def write_partitions(partitions):
    """
    Write the given named partitions with one shared write registry.

    Returns a mapping partition name -> binary blob.
    """
    registry = VariableRegistry()
    blobs = {}

    for name, measurement in partitions.items():
        blob = io.BytesIO()
        write_partition(measurement, blob, registry=registry)
        blobs[name] = blob.getvalue()

    return blobs


def read_partitions_in_order(blobs, order):
    """
    Read the partition blobs in the given order with one fresh registry.
    """
    registry = VariableRegistry()
    read_back = {}

    for name in order:
        read_back[name] = read_partition(
            io.BytesIO(blobs[name]), registry=registry
        )

    return read_back


def test_shared_bias_cancels_and_covariance_recovered():
    """
    Main scenario: two instruments, two days, one shared calibration bias.
    """
    _bias, (a1, a2), (b1, b2) = build_experiment()

    blobs = write_partitions({"A": [a1, a2], "B": [b1, b2]})

    for order in (("A", "B"), ("B", "A")):
        read_back = read_partitions_in_order(blobs, order)
        (ra1, ra2), (rb1, rb2) = read_back["A"], read_back["B"]

        # Nominal values are preserved:
        assert ra1.nominal_value == pytest.approx(1.00)
        assert ra2.nominal_value == pytest.approx(1.50)
        assert rb1.nominal_value == pytest.approx(2.00)
        assert rb2.nominal_value == pytest.approx(2.50)

        # Marginal uncertainties: bias variance + instrument noise variance:
        assert ra1.std_dev == pytest.approx(
            math.sqrt(BIAS_STD**2 + A_NOISE_STD**2)
        )
        assert rb1.std_dev == pytest.approx(
            math.sqrt(BIAS_STD**2 + B_NOISE_STD**2)
        )

        # The shared bias is the only common error source, so the
        # covariance between original results of the two instruments is
        # exactly its variance:
        assert covariance_matrix([ra1, rb1])[0][1] == pytest.approx(
            BIAS_STD**2
        )
        assert covariance_matrix([ra2, rb2])[0][1] == pytest.approx(
            BIAS_STD**2
        )

        # Correlations internal to one partition are preserved too:
        assert covariance_matrix([ra1, ra2])[0][1] == pytest.approx(
            BIAS_STD**2
        )

        # Difference between the two original results: the shared bias
        # cancels, only the two independent noises remain:
        direct_diff = ra1 - rb1
        assert direct_diff.nominal_value == pytest.approx(-1.00)
        assert direct_diff.std_dev == pytest.approx(
            math.sqrt(A_NOISE_STD**2 + B_NOISE_STD**2)
        )

        # Difference between the two days, for each instrument: the bias
        # cancels there as well and only the day noises remain:
        a_day_diff = ra2 - ra1
        assert a_day_diff.nominal_value == pytest.approx(0.50)
        assert a_day_diff.std_dev == pytest.approx(
            math.sqrt(2) * A_NOISE_STD
        )

        b_day_diff = rb2 - rb1
        assert b_day_diff.nominal_value == pytest.approx(0.50)
        assert b_day_diff.std_dev == pytest.approx(
            math.sqrt(2) * B_NOISE_STD
        )

        # Difference of the day-to-day changes: only the four independent
        # noise terms contribute (the bias variance drops out):
        full_diff = a_day_diff - b_day_diff
        assert full_diff.nominal_value == pytest.approx(0.0)
        assert full_diff.std_dev == pytest.approx(
            math.sqrt(2 * A_NOISE_STD**2 + 2 * B_NOISE_STD**2)
        )


def test_read_order_does_not_change_result():
    """
    Reading A then B and B then A must give identical statistics.
    """
    _bias, a_results, b_results = build_experiment()
    blobs = write_partitions({"A": a_results, "B": b_results})

    ab = read_partitions_in_order(blobs, ("A", "B"))
    ba = read_partitions_in_order(blobs, ("B", "A"))

    values_ab = ab["A"] + ab["B"]
    values_ba = ba["A"] + ba["B"]

    covariance_ab = covariance_matrix(values_ab)
    covariance_ba = covariance_matrix(values_ba)

    assert len(covariance_ab) == len(covariance_ba)
    for row_ab, row_ba in zip(covariance_ab, covariance_ba):
        assert row_ab == pytest.approx(row_ba)

    for value_ab, value_ba in zip(values_ab, values_ba):
        assert value_ab.nominal_value == pytest.approx(value_ba.nominal_value)
        assert value_ab.std_dev == pytest.approx(value_ba.std_dev)


def test_same_tag_keeps_distinct_error_sources_independent():
    """
    Two distinct variables with the same tag must not be merged.
    """
    first = ufloat(1.0, 0.10, tag="duplicated_tag")
    second = ufloat(2.0, 0.20, tag="duplicated_tag")

    blobs = write_partitions({"first": first, "second": second})
    read_back = read_partitions_in_order(blobs, ("first", "second"))
    first_back, second_back = read_back["first"], read_back["second"]

    # Independent error sources: no covariance, quadrature combination:
    assert covariance_matrix([first_back, second_back])[0][1] == 0.0
    assert first_back.std_dev == pytest.approx(0.10)
    assert second_back.std_dev == pytest.approx(0.20)
    assert (first_back - second_back).std_dev == pytest.approx(
        math.sqrt(0.10**2 + 0.20**2)
    )


def test_single_partition_restores_marginal_uncertainty():
    """
    A partition read on its own keeps its internal correlations and margins.
    """
    _bias, (a1, a2), _b = build_experiment()
    blobs = write_partitions({"A": [a1, a2]})

    # Default registry: the partition is fully self-contained:
    a1_back, a2_back = read_partition(io.BytesIO(blobs["A"]))

    assert a1_back.nominal_value == pytest.approx(1.00)
    assert a2_back.nominal_value == pytest.approx(1.50)
    assert a1_back.std_dev == pytest.approx(
        math.sqrt(BIAS_STD**2 + A_NOISE_STD**2)
    )
    assert a2_back.std_dev == pytest.approx(
        math.sqrt(BIAS_STD**2 + A_NOISE_STD**2)
    )

    # The two days share the bias inside the single partition:
    assert covariance_matrix([a1_back, a2_back])[0][1] == pytest.approx(
        BIAS_STD**2
    )

    # A lone variable is also fully reconstructed:
    lone = ufloat(3.0, 0.25, tag="lone")
    lone_blob = io.BytesIO()
    write_partition(lone, lone_blob)
    lone_back = read_partition(io.BytesIO(lone_blob.getvalue()))

    assert lone_back.nominal_value == pytest.approx(3.0)
    assert lone_back.std_dev == pytest.approx(0.25)
    assert lone_back.tag == "lone"


def test_partition_files_on_disk(tmp_path):
    """
    Partitions can be written to and read from separate files.
    """
    _bias, (a1, a2), (b1, b2) = build_experiment()

    write_registry = VariableRegistry()
    path_a = tmp_path / "instrument_a.part"
    path_b = tmp_path / "instrument_b.part"

    write_partition([a1, a2], path_a, registry=write_registry)
    write_partition([b1, b2], path_b, registry=write_registry)

    # Simulate a separate reading program: a brand new registry merges the
    # partitions solely from the keys stored in the files:
    read_registry = VariableRegistry()
    ra1, ra2 = read_partition(path_a, registry=read_registry)
    rb1, rb2 = read_partition(path_b, registry=read_registry)

    assert covariance_matrix([ra1, rb1])[0][1] == pytest.approx(BIAS_STD**2)
    assert (ra1 - rb1).std_dev == pytest.approx(
        math.sqrt(A_NOISE_STD**2 + B_NOISE_STD**2)
    )
    assert (ra2 - ra1).std_dev == pytest.approx(
        math.sqrt(2) * A_NOISE_STD
    )
    assert (rb2 - rb1).std_dev == pytest.approx(
        math.sqrt(2) * B_NOISE_STD
    )


def test_nested_containers_round_trip():
    """
    Identity survives arbitrary nesting inside picklable containers.
    """
    bias = ufloat(0.0, BIAS_STD, tag="calibration_bias")
    table = {
        "instrument": "A",
        "days": (1.0 + bias, 2.0 + bias + ufloat(0.0, 0.05)),
        "repetitions": [ufloat(0.5, 0.02, tag="rep")],
    }

    blob = io.BytesIO()
    write_partition(table, blob)
    table_back = read_partition(io.BytesIO(blob.getvalue()))

    assert table_back["instrument"] == "A"
    day1, day2 = table_back["days"]

    # The bias is shared between the two days inside the nested structure:
    assert covariance_matrix([day1, day2])[0][1] == pytest.approx(BIAS_STD**2)
    assert day1.std_dev == pytest.approx(BIAS_STD)
    assert table_back["repetitions"][0].tag == "rep"


@pytest.mark.parametrize("protocol", [0, 1, 2, pickle.HIGHEST_PROTOCOL])
def test_legacy_protocols_are_upgraded(protocol):
    """
    Any requested protocol works; old protocols are upgraded transparently.
    """
    bias = ufloat(0.0, BIAS_STD, tag="bias")
    derived = bias + ufloat(0.0, A_NOISE_STD)

    blob = io.BytesIO()
    write_partition(derived, blob, protocol=protocol)
    derived_back = read_partition(io.BytesIO(blob.getvalue()))

    assert derived_back.nominal_value == pytest.approx(0.0)
    assert derived_back.std_dev == pytest.approx(
        math.sqrt(BIAS_STD**2 + A_NOISE_STD**2)
    )


def test_partition_is_not_plain_pickle():
    """
    A partition must be read through read_partition(), not pickle.load().
    """
    blob = io.BytesIO()
    write_partition(ufloat(1.0, 0.1), blob)

    with pytest.raises(RuntimeError):
        pickle.loads(blob.getvalue())


def test_standard_pickling_semantics_unchanged():
    """
    Ordinary pickle behavior is untouched by the partition machinery.
    """
    x = ufloat(1.0, 0.1, tag="x")
    derived = 2 * x

    # Correlation inside one regular pickle blob is preserved:
    x_back, derived_back = pickle.loads(pickle.dumps((x, derived)))
    assert (derived_back - 2 * x_back).std_dev == pytest.approx(0.0)

    # Separate regular pickle blobs stay independent, as documented:
    first = pickle.loads(pickle.dumps(x))
    second = pickle.loads(pickle.dumps(x))
    assert covariance_matrix([first, second])[0][1] == 0.0


class TaggedVariable(uncert_core.Variable):
    """Variable subclass carrying extra state, defined at module level."""

    __slots__ = ("extra",)


def test_variable_subclasses_keep_their_pickle_representation():
    """
    Subclassed variables are not registry-merged and keep their extra state.
    """
    special = TaggedVariable(1.0, 0.1, tag="special")
    special.extra = "carried over"

    blob = io.BytesIO()
    write_partition(special, blob)
    special_back = read_partition(io.BytesIO(blob.getvalue()))

    assert type(special_back) is TaggedVariable
    assert special_back.extra == "carried over"
    assert special_back.tag == "special"
    assert special_back.std_dev == pytest.approx(0.1)

    # Two separately written subclassed variables stay independent, since
    # they do not take part in the registry identity:
    other = TaggedVariable(2.0, 0.2, tag="special")
    other_blob = io.BytesIO()
    write_partition(other, other_blob)

    registry = VariableRegistry()
    first_back = read_partition(io.BytesIO(blob.getvalue()), registry=registry)
    second_back = read_partition(
        io.BytesIO(other_blob.getvalue()), registry=registry
    )
    assert covariance_matrix([first_back, second_back])[0][1] == 0.0
