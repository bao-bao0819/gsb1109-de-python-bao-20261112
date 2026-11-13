import itertools
import random
import unittest

from geopy.distance import geodesic, great_circle
from geopy.point import Point
from geopy.trajectory import (
    InfeasibleTrajectory,
    TrajectoryBounds,
    TrajectorySegment,
    trajectory_length_bounds,
)


def brute_force_bounds(frames, segments=None, distance=geodesic):
    """Enumerate every candidate sequence; reference implementation."""
    if segments is None:
        segments = [TrajectorySegment()] * (len(frames) - 1)
    lengths = []
    for sequence in itertools.product(
        *[range(len(frame)) for frame in frames]
    ):
        total = 0.0
        for index in range(len(sequence) - 1):
            edge = distance(
                frames[index][sequence[index]],
                frames[index + 1][sequence[index + 1]],
            ).km
            if edge > segments[index].max_distance:
                total = None
                break
            total += edge
        if total is not None:
            lengths.append((total, sequence))
    return lengths


def path_length(points, distance=geodesic):
    return sum(
        distance(a, b).km for a, b in zip(points, points[1:])
    )


class TrajectorySegmentTestCase(unittest.TestCase):

    def test_unconstrained_segment_has_infinite_budget(self):
        segment = TrajectorySegment()
        self.assertEqual(segment.max_distance, float("inf"))
        self.assertEqual(
            TrajectorySegment(time=1.0).max_distance, float("inf")
        )
        self.assertEqual(
            TrajectorySegment(speed_limit=50.0).max_distance,
            float("inf"),
        )

    def test_budget_is_speed_times_time(self):
        segment = TrajectorySegment(time=0.5, speed_limit=80.0)
        self.assertEqual(segment.max_distance, 40.0)


class TrajectoryLengthBoundsTestCase(unittest.TestCase):

    def test_single_candidate_chain_matches_direct_sum(self):
        frames = [
            [(0.0, 0.0)],
            [(1.0, 0.0)],
            [(2.0, 0.0)],
        ]
        expected = (
            geodesic(frames[0][0], frames[1][0]).km
            + geodesic(frames[1][0], frames[2][0]).km
        )
        result = trajectory_length_bounds(frames)
        self.assertAlmostEqual(result.min_km, expected)
        self.assertAlmostEqual(result.max_km, expected)
        self.assertEqual(result.min_sequence, (0, 0, 0))
        self.assertEqual(result.max_sequence, (0, 0, 0))

    def test_single_timestamp_has_zero_length(self):
        point = (10.0, 20.0)
        result = trajectory_length_bounds([[point]])
        self.assertEqual(result.min_km, 0.0)
        self.assertEqual(result.max_km, 0.0)
        self.assertEqual(result.min_sequence, (0,))
        self.assertEqual(result.min_points, (Point(point),))

    def test_shared_endpoint_coupling_beats_greedy_nearest(self):
        # Greedily taking the nearest candidate at each segment picks X,
        # but picking Y (identical to the final fix) yields the global
        # minimum because the shared endpoint can only occupy one spot.
        start = (0.0, 0.0)
        near = (0.0, 0.05)
        far = (0.1, 0.05)
        frames = [[start], [near, far], [far]]
        result = trajectory_length_bounds(frames)

        greedy = (
            geodesic(start, near).km + geodesic(near, far).km
        )
        self.assertEqual(result.min_sequence, (0, 1, 0))
        self.assertLess(result.min_km, greedy)
        self.assertAlmostEqual(
            result.min_km, geodesic(start, far).km
        )
        # The longest route goes via the near candidate.
        self.assertEqual(result.max_sequence, (0, 0, 0))
        self.assertAlmostEqual(result.max_km, greedy)

    def test_brute_force_agreement_with_speed_limits(self):
        rng = random.Random(1109)
        for _ in range(60):
            frame_count = rng.randint(2, 4)
            frames = [
                [
                    (rng.uniform(-5, 5), rng.uniform(-5, 5))
                    for _ in range(rng.randint(1, 3))
                ]
                for _ in range(frame_count)
            ]
            segments = []
            for _ in range(frame_count - 1):
                if rng.random() < 0.3:
                    segments.append(TrajectorySegment())
                else:
                    segments.append(
                        TrajectorySegment(
                            time=rng.uniform(0.1, 2.0),
                            speed_limit=rng.uniform(5.0, 400.0),
                        )
                    )
            feasible = brute_force_bounds(frames, segments)
            if not feasible:
                with self.assertRaises(InfeasibleTrajectory) as caught:
                    trajectory_length_bounds(frames, segments)
                dead_index = caught.exception.timestamp_index
                # Prefix ending just before the dead timestamp is feasible.
                self.assertTrue(
                    brute_force_bounds(
                        frames[:dead_index], segments[:dead_index - 1]
                    )
                )
                self.assertFalse(
                    brute_force_bounds(
                        frames[:dead_index + 1], segments[:dead_index]
                    )
                )
                continue
            result = trajectory_length_bounds(frames, segments)
            expected_min = min(length for length, _ in feasible)
            expected_max = max(length for length, _ in feasible)
            self.assertAlmostEqual(result.min_km, expected_min, places=8)
            self.assertAlmostEqual(result.max_km, expected_max, places=8)
            for index in range(frame_count - 1):
                edge = geodesic(
                    result.min_points[index], result.min_points[index + 1]
                ).km
                self.assertLessEqual(edge, segments[index].max_distance)

    def test_returned_points_attain_reported_bounds(self):
        frames = [
            [(0.0, 0.0), (0.0, 0.2)],
            [(0.0, 0.1), (1.0, 0.0)],
            [(0.0, 0.2), (1.0, 0.1)],
            [(0.5, 0.5)],
        ]
        result = trajectory_length_bounds(frames)
        self.assertIsInstance(result, TrajectoryBounds)
        self.assertAlmostEqual(
            result.min_km, path_length(result.min_points)
        )
        self.assertAlmostEqual(
            result.max_km, path_length(result.max_points)
        )
        self.assertLessEqual(result.min_km, result.max_km)
        for index, candidate in enumerate(result.min_sequence):
            self.assertEqual(
                result.min_points[index],
                Point(frames[index][candidate]),
            )

    def test_brute_force_agreement_on_small_grids(self):
        rng = random.Random(20261113)
        for _ in range(40):
            frame_count = rng.randint(2, 5)
            frames = [
                [
                    (rng.uniform(-30, 30), rng.uniform(-30, 30))
                    for _ in range(rng.randint(1, 3))
                ]
                for _ in range(frame_count)
            ]
            result = trajectory_length_bounds(frames)
            feasible = brute_force_bounds(frames)
            expected_min = min(length for length, _ in feasible)
            expected_max = max(length for length, _ in feasible)
            self.assertAlmostEqual(result.min_km, expected_min, places=8)
            self.assertAlmostEqual(result.max_km, expected_max, places=8)
            self.assertAlmostEqual(
                path_length(result.min_points), expected_min, places=8
            )
            self.assertAlmostEqual(
                path_length(result.max_points), expected_max, places=8
            )

    def test_speed_limit_prunes_unreachable_connections(self):
        # The far candidate is ~111 km away, unreachable in 0.5 h at 50 km/h.
        frames = [
            [(0.0, 0.0)],
            [(0.0, 0.1), (1.0, 0.0)],
            [(0.0, 0.2)],
        ]
        segments = [
            TrajectorySegment(time=0.5, speed_limit=50.0),
            TrajectorySegment(time=2.0, speed_limit=200.0),
        ]
        result = trajectory_length_bounds(frames, segments)
        self.assertEqual(result.min_sequence[1], 0)
        self.assertEqual(result.max_sequence[1], 0)
        for index in range(len(frames) - 1):
            edge = geodesic(
                result.min_points[index], result.min_points[index + 1]
            ).km
            self.assertLessEqual(edge, segments[index].max_distance)

    def test_segments_accepted_as_plain_tuples(self):
        frames = [
            [(0.0, 0.0)],
            [(0.0, 0.1), (1.0, 0.0)],
            [(0.0, 0.2)],
        ]
        tuple_result = trajectory_length_bounds(
            frames, [(0.5, 50.0), (2.0, 200.0)]
        )
        object_result = trajectory_length_bounds(
            frames,
            [
                TrajectorySegment(0.5, 50.0),
                TrajectorySegment(2.0, 200.0),
            ],
        )
        self.assertEqual(
            tuple_result.min_sequence, object_result.min_sequence
        )
        self.assertAlmostEqual(
            tuple_result.min_km, object_result.min_km
        )

    def test_none_segment_imposes_no_constraint(self):
        frames = [[(0.0, 0.0)], [(0.0, 0.1), (1.0, 0.0)]]
        result = trajectory_length_bounds(
            frames, [TrajectorySegment()]
        )
        unconstrained = trajectory_length_bounds(frames)
        self.assertEqual(
            result.max_sequence, unconstrained.max_sequence
        )

    def test_infeasible_reports_earliest_dead_timestamp(self):
        frames = [
            [(0.0, 0.0)],
            [(0.0, 0.1)],
            [(10.0, 10.0)],
        ]
        segments = [
            TrajectorySegment(time=1.0, speed_limit=200.0),
            TrajectorySegment(time=0.1, speed_limit=10.0),
        ]
        with self.assertRaises(InfeasibleTrajectory) as caught:
            trajectory_length_bounds(frames, segments)
        self.assertEqual(caught.exception.timestamp_index, 2)
        self.assertEqual(caught.exception.failed_at, 1)

    def test_infeasible_at_first_segment(self):
        frames = [[(0.0, 0.0)], [(1.0, 0.0)]]
        with self.assertRaises(InfeasibleTrajectory) as caught:
            trajectory_length_bounds(
                frames, [TrajectorySegment(time=0.1, speed_limit=1.0)]
            )
        self.assertEqual(caught.exception.timestamp_index, 1)
        self.assertEqual(caught.exception.failed_at, 0)

    def test_partial_candidate_death_then_total_failure(self):
        # At timestamp 1 only candidate 0 survives; timestamp 2 is solely
        # reachable from the dead candidate 1, so the prefix dies at index 2.
        frames = [
            [(0.0, 0.0)],
            [(0.0, 0.05), (5.0, 5.0)],
            [(5.0, 5.05)],
        ]
        segments = [
            TrajectorySegment(time=1.0, speed_limit=20.0),
            TrajectorySegment(time=1.0, speed_limit=20.0),
        ]
        with self.assertRaises(InfeasibleTrajectory) as caught:
            trajectory_length_bounds(frames, segments)
        self.assertEqual(caught.exception.timestamp_index, 2)

    def test_missing_gap_feasible_and_infeasible(self):
        frames = [[(0.0, 0.0)], [(0.0, 1.0)]]
        gap_km = geodesic(frames[0][0], frames[1][0]).km
        feasible = trajectory_length_bounds(
            frames,
            [TrajectorySegment(time=gap_km / 100.0, speed_limit=100.0)],
        )
        self.assertAlmostEqual(feasible.min_km, gap_km)
        with self.assertRaises(InfeasibleTrajectory):
            trajectory_length_bounds(
                frames,
                [TrajectorySegment(time=gap_km / 200.0, speed_limit=100.0)],
            )

    def test_point_objects_accepted(self):
        frames = [
            [Point(0.0, 0.0)],
            [Point(0.0, 0.1), Point(1.0, 0.0)],
            [Point(0.0, 0.2)],
        ]
        result = trajectory_length_bounds(frames)
        self.assertAlmostEqual(
            result.min_km,
            geodesic(Point(0.0, 0.0), Point(0.0, 0.1)).km
            + geodesic(Point(0.0, 0.1), Point(0.0, 0.2)).km,
        )

    def test_custom_distance_class(self):
        frames = [
            [(0.0, 0.0)],
            [(0.0, 0.1), (1.0, 0.0)],
            [(0.0, 0.2)],
        ]
        result = trajectory_length_bounds(frames, distance=great_circle)
        expected_min = (
            great_circle(frames[0][0], frames[1][0]).km
            + great_circle(frames[1][0], frames[2][0]).km
        )
        self.assertAlmostEqual(result.min_km, expected_min)
        self.assertAlmostEqual(
            result.max_km,
            path_length(result.max_points, distance=great_circle),
        )

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            trajectory_length_bounds([])
        with self.assertRaises(ValueError):
            trajectory_length_bounds([[(0.0, 0.0)], []])
        frames = [[(0.0, 0.0)], [(0.0, 0.1)], [(0.0, 0.2)]]
        with self.assertRaises(ValueError):
            trajectory_length_bounds(frames, [TrajectorySegment()])
        with self.assertRaises(ValueError):
            trajectory_length_bounds(
                frames,
                [TrajectorySegment(), TrajectorySegment(),
                 TrajectorySegment()],
            )


if __name__ == "__main__":
    unittest.main()
