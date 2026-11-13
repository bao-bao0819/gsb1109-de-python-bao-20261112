import itertools

import pytest

import portion as P


class IntInterval(P.AbstractDiscreteInterval):
    _step = 1


D = P.create_api(IntInterval)


def expand(observations, readings):
    """Expand readings by the frequency of their observation."""
    return [
        reading
        for (_, frequency), reading in zip(observations, readings)
        for _ in range(frequency)
    ]


class TestKthSmallestValues:
    def test_single_observation(self):
        observations = [(P.closed(0, 10), 3)]
        for k in (1, 2, 3):
            assert P.kth_smallest_values(observations, k) == P.closed(0, 10)

    def test_singletons(self):
        observations = [(P.singleton(1), 1), (P.singleton(3), 1)]
        assert P.kth_smallest_values(observations, 1) == P.singleton(1)
        assert P.kth_smallest_values(observations, 2) == P.singleton(3)

    def test_frequency_is_repeated_as_a_whole(self):
        observations = [(P.closed(0, 2), 2), (P.singleton(10), 1)]
        assert P.kth_smallest_values(observations, 1) == P.closed(0, 2)
        assert P.kth_smallest_values(observations, 2) == P.closed(0, 2)
        assert P.kth_smallest_values(observations, 3) == P.singleton(10)

    def test_open_end_below_detection_limit(self):
        observations = [(P.open(-P.inf, 5), 1), (P.singleton(7), 1)]
        values = P.kth_smallest_values(observations, 1)
        assert values == P.open(-P.inf, 5)
        assert 5 not in values
        assert P.kth_smallest_values(observations, 2) == P.singleton(7)

    def test_open_bound_is_preserved(self):
        observations = [(P.open(0, 10), 1), (P.singleton(20), 1)]
        values = P.kth_smallest_values(observations, 1)
        assert values == P.open(0, 10)
        assert 0 not in values
        assert 10 not in values

    def test_internal_hole_is_preserved(self):
        observations = [
            (P.closed(0, 1) | P.closed(9, 10), 1),
            (P.singleton(5), 1),
        ]
        values = P.kth_smallest_values(observations, 1)
        assert values == P.closed(0, 1) | P.singleton(5)
        assert not values.atomic

        values = P.kth_smallest_values(observations, 2)
        assert values == P.singleton(5) | P.closed(9, 10)

    def test_frequency_cannot_be_split(self):
        # The two copies of the second observation must take the same value,
        # so rank 3 can never fall on 5: the copies either both lie below
        # (and the 3rd smallest is in [0, 1]) or both lie above (in [9, 10]).
        observations = [
            (P.singleton(0), 1),
            (P.closed(0, 1) | P.closed(9, 10), 2),
            (P.singleton(5), 1),
        ]
        assert P.kth_smallest_values(observations, 1) == P.singleton(0)
        assert P.kth_smallest_values(observations, 2) == P.closed(0, 1) | P.singleton(
            5
        )
        assert P.kth_smallest_values(observations, 3) == P.closed(0, 1) | P.closed(
            9, 10
        )
        assert P.kth_smallest_values(observations, 4) == P.singleton(5) | P.closed(9, 10)

    def test_result_uses_observation_class(self):
        observations = [(D.closed(0, 2), 1), (D.closed(1, 3), 1)]
        values = P.kth_smallest_values(observations, 2)
        assert isinstance(values, IntInterval)
        assert values == D.closed(1, 3)

    def test_discrete_observations(self):
        observations = [(D.open(0, 3), 1)]
        assert P.kth_smallest_values(observations, 1) == D.closed(1, 2)

    def test_exposed_in_created_api(self):
        assert D.kth_smallest_values is P.kth_smallest_values
        assert D.check_kth_smallest is P.check_kth_smallest
        assert D.KthSmallestCheck is P.KthSmallestCheck

    @pytest.mark.parametrize("k", [0, -1, 3, 1.0, "1", True])
    def test_invalid_k(self, k):
        with pytest.raises(ValueError):
            P.kth_smallest_values([(P.closed(0, 1), 2)], k)

    @pytest.mark.parametrize("frequency", [0, -1, 1.5, "2", True])
    def test_invalid_frequency(self, frequency):
        with pytest.raises(ValueError):
            P.kth_smallest_values([(P.closed(0, 1), frequency)], 1)

    def test_empty_observation_interval(self):
        with pytest.raises(ValueError):
            P.kth_smallest_values([(P.empty(), 1)], 1)

    def test_no_observation(self):
        with pytest.raises(ValueError):
            P.kth_smallest_values([], 1)

    def test_not_a_pair(self):
        with pytest.raises(TypeError):
            P.kth_smallest_values([P.closed(0, 1)], 1)
        with pytest.raises(TypeError):
            P.kth_smallest_values([(1, 1)], 1)


class TestCheckKthSmallest:
    def test_reachable(self):
        observations = [
            (P.singleton(0), 1),
            (P.closed(0, 1) | P.closed(9, 10), 2),
            (P.singleton(5), 1),
        ]
        check = P.check_kth_smallest(observations, 3, 0.5)
        assert check.reachable
        assert check
        assert check.reason is None
        assert len(check.readings) == len(observations)
        for (interval, _), reading in zip(observations, check.readings):
            assert reading in interval
        assert sorted(expand(observations, check.readings))[2] == 0.5

    def test_reachable_witness_uses_whole_frequency(self):
        observations = [(P.closed(0, 10), 2), (P.singleton(1), 1)]
        check = P.check_kth_smallest(observations, 2, 4)
        assert check.reachable
        # One reading per observation: the frequency-2 observation contributes
        # twice the same value.
        assert len(check.readings) == 2
        assert expand(observations, check.readings).count(check.readings[0]) == 2
        assert sorted(expand(observations, check.readings))[1] == 4

    def test_unreachable_value_not_a_reading(self):
        observations = [(P.singleton(1), 1), (P.singleton(3), 1)]
        check = P.check_kth_smallest(observations, 1, 2)
        assert not check.reachable
        assert not check
        assert check.readings is None
        assert "not a possible reading" in check.reason

    def test_unreachable_too_many_below(self):
        observations = [(P.singleton(0), 2), (P.closed(5, 10), 1)]
        check = P.check_kth_smallest(observations, 2, 7)
        assert not check.reachable
        assert "necessarily smaller" in check.reason

    def test_unreachable_too_many_above(self):
        observations = [(P.closed(0, 5), 1), (P.singleton(10), 2)]
        check = P.check_kth_smallest(observations, 2, 3)
        assert not check.reachable
        assert "necessarily greater" in check.reason

    def test_unreachable_frequency_cannot_be_split(self):
        observations = [
            (P.singleton(0), 1),
            (P.closed(0, 1) | P.closed(9, 10), 2),
            (P.singleton(5), 1),
        ]
        check = P.check_kth_smallest(observations, 3, 5)
        assert not check.reachable
        assert "cannot be split" in check.reason
        assert "skips rank 3" in check.reason

    def test_discrete_witness(self):
        observations = [(D.open(0, 3), 1)]
        check = P.check_kth_smallest(observations, 1, 2)
        assert check.reachable
        assert check.readings == (2,)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            P.check_kth_smallest([(P.closed(0, 1), 1)], 2, 0)
        with pytest.raises(ValueError):
            P.check_kth_smallest([(P.closed(0, 1), 0)], 1, 0)


class TestBruteForce:
    @pytest.mark.parametrize(
        "observations",
        [
            [(D.closed(0, 2), 2), (D.closed(1, 3), 1)],
            [(D.closed(0, 1) | D.closed(3, 4), 2), (D.singleton(2), 1)],
            [
                (D.closed(0, 1) | D.closed(3, 4), 2),
                (D.singleton(2), 1),
                (D.closed(0, 4), 1),
            ],
            [(D.open(0, 3), 3), (D.closed(2, 4), 2)],
        ],
    )
    def test_against_exhaustive_enumeration(self, observations):
        domains = [list(P.iterate(interval, 1)) for interval, _ in observations]
        n = sum(frequency for _, frequency in observations)

        expected = {k: set() for k in range(1, n + 1)}
        for readings in itertools.product(*domains):
            sample = sorted(expand(observations, readings))
            for k in range(1, n + 1):
                expected[k].add(sample[k - 1])

        for k in range(1, n + 1):
            values = P.kth_smallest_values(observations, k)
            assert set(P.iterate(values, 1)) == expected[k]
            for value in expected[k]:
                check = P.check_kth_smallest(observations, k, value)
                assert check.reachable
                assert sorted(expand(observations, check.readings))[k - 1] == value
            for value in set(range(-1, 6)) - expected[k]:
                assert not P.check_kth_smallest(observations, k, value).reachable
