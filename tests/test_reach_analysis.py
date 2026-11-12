import random
import unittest

from cachetools import LFUCache, LRUCache

from analysis.bruteforce import brute_force_bounds
from analysis.demo import reversal_log, trace
from analysis.model import AccessLog, Gap, log_from_pair, make_log
from analysis.policies import LFU, LRU, evaluate_sequence
from analysis.reach import reachable_bounds, verdict


class PolicySimulationTest(unittest.TestCase):
    def test_lru_matches_cachetools(self):
        rng = random.Random(11)
        for capacity in range(1, 6):
            for _ in range(50):
                sequence = [rng.randrange(5) for _ in range(rng.randint(1, 40))]
                mine, real = LRU(capacity), LRUCache(capacity)
                for key in sequence:
                    was_present = key in real
                    miss = mine.access(key)
                    if was_present:
                        real[key]  # a hit only reads and refreshes recency
                    else:
                        real[key] = None
                    self.assertEqual(miss, not was_present)
                    self.assertEqual(set(mine.contents()), set(real))
                    self.assertEqual(mine.contents(), tuple(real._LRUCache__order))

    def test_lfu_matches_cachetools(self):
        rng = random.Random(23)
        for capacity in range(1, 6):
            for _ in range(50):
                sequence = [rng.randrange(5) for _ in range(rng.randint(1, 40))]
                mine, real = LFU(capacity), LFUCache(capacity)
                for key in sequence:
                    was_present = key in real
                    miss = mine.access(key)
                    if was_present:
                        real[key]  # a hit only reads, incrementing count once
                    else:
                        real[key] = None
                    self.assertEqual(miss, not was_present)
                    self.assertEqual(set(mine.contents()), set(real))
                    links = real._LFUCache__links
                    for present in mine.contents():
                        self.assertEqual(mine._count[present], links[present].count)
                if capacity >= 2 and len(mine.contents()) == capacity:
                    victim = min(mine._data, key=mine._count.__getitem__)
                    real_victim = next(iter(real._LFUCache__root.next.keys))
                    self.assertEqual(victim, real_victim)

    def test_lfu_tie_breaks_lru(self):
        policy = LFU(3)
        for key in (1, 2, 3):
            policy.access(key)
        for key in (3, 2, 1):
            policy.access(key)
        self.assertTrue(policy.access(4))  # evicts least-recently-used at count 2
        self.assertEqual(set(policy.contents()), {2, 1, 4})

    def test_evaluate_sequence(self):
        sequence = (1, 1, 2, 3, 1)
        self.assertEqual(evaluate_sequence(sequence, 2), (4, 3))

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            LRU(0)
        with self.assertRaises(ValueError):
            LFU(0)


class AccessLogTest(unittest.TestCase):
    def test_completion_count_and_values(self):
        log = AccessLog([1, Gap(2, (2, 3)), 4, Gap(1, (4, 5))])
        self.assertEqual(log.gap_count(), 2)
        self.assertEqual(log.completion_count(), 2**2 * 2)
        completions = set(log.completions())
        self.assertEqual(len(completions), 8)
        for sequence in completions:
            self.assertEqual(sequence[0], 1)
            self.assertEqual(sequence[3], 4)
            self.assertIn(sequence[1], (2, 3))
            self.assertIn(sequence[2], (2, 3))
            self.assertIn(sequence[4], (4, 5))

    def test_zero_length_gap_adds_nothing(self):
        log = AccessLog([1, Gap(0, ()), 2])
        self.assertEqual(set(log.completions()), {(1, 2)})

    def test_invalid_gap(self):
        with self.assertRaises(ValueError):
            AccessLog([Gap(1, ())])
        with self.assertRaises(ValueError):
            AccessLog([Gap(-1, (1,))])

    def test_builders(self):
        log = make_log((7, 8), {0: (1, (1, 2)), 2: (1, (8,))})
        self.assertEqual(
            set(log.completions()),
            {(1, 7, 8, 8), (2, 7, 8, 8)},
        )
        other = log_from_pair((7, 8), [(-1, 1, (1,)), (1, 1, (8,))])
        self.assertEqual(set(other.completions()), {(1, 7, 8, 8)})


class ReachableBoundsTest(unittest.TestCase):
    def test_reversal_bounds_and_witnesses(self):
        log = reversal_log()
        bounds = reachable_bounds(log, 2)
        self.assertEqual((bounds.min_diff, bounds.max_diff), (-1, 1))
        # Both bounds are witnessed on a concrete completion of one log.
        completions = set(log.completions())
        self.assertIn(bounds.min_sequence, completions)
        self.assertIn(bounds.max_sequence, completions)
        # Opposite conclusions, sharing the observed prefix.
        self.assertEqual(bounds.max_sequence[:4], (1, 1, 2, 3))
        self.assertEqual(bounds.min_sequence[:4], (1, 1, 2, 3))
        self.assertEqual(bounds.max_sequence, (1, 1, 2, 3, 1))
        self.assertEqual(bounds.min_sequence, (1, 1, 2, 3, 2))
        # Replayed counts realize the advertised differences.
        self.assertEqual(bounds.max_counts, (4, 3))
        self.assertEqual(bounds.min_counts, (3, 4))
        self.assertEqual(bounds.max_counts[0] - bounds.max_counts[1], bounds.max_diff)
        self.assertEqual(bounds.min_counts[0] - bounds.min_counts[1], bounds.min_diff)
        self.assertIn("reversal", verdict(bounds))

    def test_trace_reports_misses(self):
        rows = trace((1, 1, 2, 3, 1), 2)
        self.assertEqual(
            [row["lru_miss"] for row in rows], [True, False, True, True, True]
        )
        self.assertEqual(
            [row["lfu_miss"] for row in rows], [True, False, True, True, False]
        )

    def test_fully_observed_log_is_a_single_point(self):
        log = AccessLog([1, 2, 3, 1, 2, 3, 1])
        bounds = reachable_bounds(log, 2)
        self.assertEqual(bounds.min_diff, bounds.max_diff)
        self.assertEqual(bounds.min_sequence, bounds.max_sequence)
        self.assertEqual(bounds.min_counts, bounds.max_counts)

    def test_lfu_never_worse_log(self):
        log = AccessLog([1, 1, Gap(2, (2, 3)), 1])
        bounds = reachable_bounds(log, 2)
        self.assertEqual((bounds.min_diff, bounds.max_diff), (0, 1))
        self.assertIn("never worse", verdict(bounds))

    def test_empty_log(self):
        bounds = reachable_bounds(AccessLog([]), 2)
        self.assertEqual((bounds.min_diff, bounds.max_diff), (0, 0))
        self.assertEqual(bounds.min_counts, (0, 0))

    def test_invalid_arguments(self):
        with self.assertRaises(TypeError):
            reachable_bounds([1, 2], 2)
        with self.assertRaises(ValueError):
            reachable_bounds(AccessLog([1]), 0)

    def test_matches_brute_force_on_random_logs(self):
        rng = random.Random(99)
        for _ in range(100):
            capacity = rng.randint(1, 4)
            keys = list(range(rng.randint(2, 5)))
            events = []
            for _ in range(rng.randint(1, 6)):
                if rng.random() < 0.4:
                    events.append(
                        Gap(
                            rng.randint(0, 3),
                            rng.sample(keys, rng.randint(1, len(keys))),
                        )
                    )
                else:
                    events.append(rng.choice(keys))
            log = AccessLog(events)
            if log.completion_count() > 3000:
                continue
            bounds = reachable_bounds(log, capacity)
            brute = brute_force_bounds(log, capacity)
            self.assertEqual(bounds.min_diff, brute.min_diff)
            self.assertEqual(bounds.max_diff, brute.max_diff)
            self.assertEqual(bounds.min_counts, brute.min_counts)
            self.assertEqual(bounds.max_counts, brute.max_counts)


if __name__ == "__main__":
    unittest.main()
