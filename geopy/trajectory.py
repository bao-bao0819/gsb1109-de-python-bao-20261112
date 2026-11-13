"""
Bounds on the total geodesic length of a trajectory sampled at discrete
GPS times, where each timestamp offers several discrete candidate
coordinates instead of a single fix.

At every GPS timestamp exactly one of its candidate coordinates must be
chosen. The length of the resulting trajectory is the sum of the
:class:`.geodesic` distances between the chosen coordinates of every pair
of consecutive timestamps, so simply summing the nearest pair of each
segment is invalid: the shared endpoint can only occupy one position.

Segments with missing observations are supported through explicit
:class:`TrajectorySegment` entries carrying the elapsed time and a speed
limit. A connection between two candidates is reachable only when the
geodesic distance does not exceed ``speed * time``; connections violating
the limit are discarded before the dynamic program runs.
"""

from dataclasses import dataclass
from math import inf

from geopy.distance import geodesic
from geopy.point import Point

__all__ = [
    "InfeasibleTrajectory",
    "TrajectorySegment",
    "TrajectoryBounds",
    "trajectory_length_bounds",
]


@dataclass(frozen=True)
class TrajectorySegment:
    """
    A transition between two consecutive GPS timestamps.

    :param time: Elapsed time of the segment. Any positive numeric unit is
        acceptable, provided ``speed_limit`` uses the reciprocal unit
        (e.g. hours and km/h, seconds and km/s). ``None`` means the segment
        is a regular observed interval without a reachability constraint.
    :param float speed_limit: Maximum reachable speed in distance units per
        time unit. ``None`` means no speed constraint applies.
    """

    time: object = None
    speed_limit: object = None

    @property
    def max_distance(self):
        """Maximum reachable geodesic distance in kilometers, or infinity."""
        if self.time is None or self.speed_limit is None:
            return inf
        return float(self.time) * float(self.speed_limit)


@dataclass(frozen=True)
class TrajectoryBounds:
    """
    Result of :func:`trajectory_length_bounds`.

    :var float min_km: Shortest achievable total length in kilometers.
    :var float max_km: Longest achievable total length in kilometers.
    :ivar min_sequence: Candidate sequence attaining ``min_km``. Each
        element is the index of the chosen coordinate at that timestamp.
    :ivar max_sequence: Candidate sequence attaining ``max_km``. Each
        element is the index of the chosen coordinate at that timestamp.
    :ivar min_points: :class:`~geopy.point.Point` objects of the shortest
        sequence in timestamp order.
    :ivar max_points: :class:`~geopy.point.Point` objects of the longest
        sequence in timestamp order.
    """

    min_km: float
    max_km: float
    min_sequence: tuple
    max_sequence: tuple
    min_points: tuple
    max_points: tuple


class InfeasibleTrajectory(ValueError):
    """
    Raised when no candidate sequence can satisfy every reachability
    constraint.

    :ivar int timestamp_index: Earliest timestamp at which feasibility is
        lost, i.e. whose candidates cannot extend any feasible prefix
        ending at the previous timestamp.
    :ivar int failed_at: Position of the failing segment in the input
        sequence (the segment from ``timestamp_index - 1`` to
        ``timestamp_index``).
    """

    def __init__(self, message, timestamp_index, failed_at=None):
        super().__init__(message)
        self.timestamp_index = timestamp_index
        self.failed_at = (
            timestamp_index - 1 if failed_at is None else failed_at
        )


def _as_point(candidate):
    if isinstance(candidate, Point):
        return candidate
    return Point(candidate)


def _normalize_candidates(timestamps):
    frames = []
    for index, candidates in enumerate(timestamps):
        points = tuple(
            _as_point(candidate) for candidate in (candidates or ())
        )
        if not points:
            raise ValueError(
                "Timestamp %d has no candidate coordinates" % index
            )
        frames.append(points)
    return frames


def _normalize_segments(segments, frame_count):
    if segments is None:
        return tuple(TrajectorySegment() for _ in range(frame_count - 1))
    normalized = tuple(
        segment if isinstance(segment, TrajectorySegment)
        else TrajectorySegment(*segment)
        for segment in segments
    )
    if len(normalized) != frame_count - 1:
        raise ValueError(
            "Expected %d segments for %d timestamps, got %d"
            % (frame_count - 1, frame_count, len(normalized))
        )
    return normalized


def _rebuild(parents, frame_count, last_index):
    sequence = [0] * frame_count
    sequence[-1] = last_index
    candidate_index = last_index
    for frame_index in range(frame_count - 1, 0, -1):
        candidate_index = parents[frame_index - 1][candidate_index]
        sequence[frame_index - 1] = candidate_index
    return tuple(sequence)


def trajectory_length_bounds(timestamps, segments=None, distance=geodesic):
    """
    Compute the shortest and longest achievable total trajectory length.

    Each element of ``timestamps`` is a sequence of candidate coordinates
    at one GPS time; every coordinate accepted by :class:`~geopy.point.Point`
    (such as ``(lat, lon)`` tuples) is allowed. A candidate is picked at
    every timestamp, and the total length is the sum of geodesic distances
    between consecutive picks.

    ``segments`` describes the interval between each pair of consecutive
    timestamps: one entry per interval, either a :class:`TrajectorySegment`
    or a plain ``(time, speed_limit)`` tuple. Pairs whose geodesic distance
    exceeds ``time * speed_limit`` are treated as unreachable and excluded
    from the optimization. ``None`` entries (or omitting the argument)
    impose no constraint.

    Dynamic programming over the candidate grid finds both global extrema
    in ``O(sum(len(frame_i) * len(frame_i-1)))`` time while honoring the
    shared-endpoint coupling that rules out summing per-segment nearest
    values.

        >>> from geopy.trajectory import trajectory_length_bounds
        >>> timestamps = [
        ...     [(0.0, 0.0)],
        ...     [(0.0, 0.1), (1.0, 0.0)],
        ...     [(0.0, 0.2)],
        ... ]
        >>> result = trajectory_length_bounds(timestamps)
        >>> [round(value, 3) for value in (result.min_km, result.max_km)]
        [22.264, 223.368]
        >>> result.min_sequence, result.max_sequence
        ((0, 0, 0), (0, 1, 0))

    :param timestamps: Candidate coordinates per GPS timestamp.
    :param segments: Optional per-interval reachability constraints.
    :param distance: Distance class used for measurements; defaults to
        :class:`.geodesic`.
    :rtype: TrajectoryBounds
    :raises InfeasibleTrajectory: if the earliest timestamp whose
        candidates cannot be reached from any feasible prefix leaves the
        problem without a solution.
    :raises ValueError: if a timestamp has no candidates or the number of
        segments does not match the number of intervals.
    """
    frames = _normalize_candidates(timestamps)
    if not frames:
        raise ValueError("At least one timestamp with candidates is required")
    segments = _normalize_segments(segments, len(frames))

    if len(frames) == 1:
        point = frames[0][0]
        return TrajectoryBounds(
            min_km=0.0,
            max_km=0.0,
            min_sequence=(0,),
            max_sequence=(0,),
            min_points=(point,),
            max_points=(point,),
        )

    min_costs = [0.0 for _ in frames[0]]
    max_costs = [0.0 for _ in frames[0]]
    min_parents = []
    max_parents = []

    for frame_index in range(1, len(frames)):
        previous = frames[frame_index - 1]
        current = frames[frame_index]
        segment = segments[frame_index - 1]
        next_min = [inf] * len(current)
        next_max = [-inf] * len(current)
        min_parent = [-1] * len(current)
        max_parent = [-1] * len(current)

        for current_index, current_point in enumerate(current):
            for previous_index, previous_point in enumerate(previous):
                edge = distance(previous_point, current_point).km
                if edge > segment.max_distance:
                    continue
                min_value = min_costs[previous_index] + edge
                if min_value < next_min[current_index]:
                    next_min[current_index] = min_value
                    min_parent[current_index] = previous_index
                max_value = max_costs[previous_index] + edge
                if max_value > next_max[current_index]:
                    next_max[current_index] = max_value
                    max_parent[current_index] = previous_index

        if all(parent < 0 for parent in min_parent):
            raise InfeasibleTrajectory(
                "No reachable connection into timestamp %d: every "
                "candidate is excluded by the segment %d reachability "
                "constraint"
                % (frame_index, frame_index - 1),
                timestamp_index=frame_index,
                failed_at=frame_index - 1,
            )

        min_costs = next_min
        max_costs = next_max
        min_parents.append(min_parent)
        max_parents.append(max_parent)

    best_min_index = min(range(len(frames[-1])), key=min_costs.__getitem__)
    best_max_index = max(range(len(frames[-1])), key=max_costs.__getitem__)
    min_sequence = _rebuild(min_parents, len(frames), best_min_index)
    max_sequence = _rebuild(max_parents, len(frames), best_max_index)

    return TrajectoryBounds(
        min_km=min_costs[best_min_index],
        max_km=max_costs[best_max_index],
        min_sequence=min_sequence,
        max_sequence=max_sequence,
        min_points=tuple(
            frames[index][candidate]
            for index, candidate in enumerate(min_sequence)
        ),
        max_points=tuple(
            frames[index][candidate]
            for index, candidate in enumerate(max_sequence)
        ),
    )
