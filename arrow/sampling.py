"""
Provides stratified uniform sampling of real instants over a time range.

The range is partitioned into strata of one local hour each.  On days when
clocks are turned back, the repeated local hour (for example the two
``01:00`` hours on a fall-back day) yields two independent strata, while
local hours that are skipped when clocks spring forward yield no stratum at
all.

The caller supplies a total budget, optional per-stratum lower and upper
bounds and a seed.  The budget is distributed across strata as integer
quotas, proportionally to the true (UTC) duration of each stratum, using
largest remainders with ties broken by UTC order.  Sampling is deterministic
for a given seed, so a global allocation table can be replayed per day and
still yields the same sample IDs.

"""

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from datetime import timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union

from dateutil.tz import datetime_exists, resolve_imaginary

from arrow.arrow import Arrow

__all__ = [
    "HourStratum",
    "Sample",
    "SamplingPlan",
    "StratumShortfall",
    "daily_ranges",
    "replay_range",
    "stratified_plan",
    "stratified_sample",
]

_MICROSECONDS_PER_DAY = 24 * 60 * 60 * 1_000_000


@dataclass(frozen=True)
class HourStratum:
    """One local-hour stratum of a sampling plan.

    :ivar key: stable identifier of the stratum, derived from the local hour
        and fold, e.g. ``"2026-11-01T01:00#1"``.
    :ivar local_start: start of the local hour occurrence (fold-aware).
    :ivar utc_start: effective start of the stratum in UTC, clipped to the
        sampled range.
    :ivar utc_end: effective end of the stratum in UTC, clipped to the
        sampled range.
    :ivar fold: the fold of the local hour occurrence (``1`` for the second
        occurrence of a repeated hour on fall-back days).

    """

    key: str
    local_start: Arrow
    utc_start: Arrow
    utc_end: Arrow
    fold: int

    @property
    def duration(self) -> timedelta:
        """The true (UTC) duration of the stratum."""

        return self.utc_end - self.utc_start


@dataclass(frozen=True)
class Sample:
    """A single sampled instant.

    :ivar sample_id: stable identifier, ``"<stratum key>#<rank>"``.
    :ivar stratum_key: key of the stratum the sample was drawn from.
    :ivar instant: the sampled instant, in the timezone of the sampled range.

    """

    sample_id: str
    stratum_key: str
    instant: Arrow


@dataclass(frozen=True)
class StratumShortfall:
    """Unmet demand of a stratum when the budget cannot be fully placed.

    :ivar stratum_key: key of the stratum.
    :ivar missing: number of samples the stratum would have received without
        its upper bound, minus the number it actually received.

    """

    stratum_key: str
    missing: int


def _total_microseconds(delta: timedelta) -> int:
    return (
        delta.days * _MICROSECONDS_PER_DAY
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def _as_aware_arrow(value: Union[Arrow, dt_datetime], name: str) -> Arrow:
    if isinstance(value, Arrow):
        if value.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware")
        return value
    if isinstance(value, dt_datetime):
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError(f"{name} must be timezone-aware")
        return Arrow.fromdatetime(value)
    raise TypeError(f"{name} must be an Arrow or datetime, got {type(value)!r}")


def _build_strata(start: Arrow, end: Arrow) -> List[HourStratum]:
    """Partition ``[start, end)`` into local-hour strata.

    Repeated local hours (fall-back) yield one stratum per fold; skipped
    local hours (spring-forward) yield none.  Strata are clipped to the
    range and returned in UTC order.

    """

    tz = start.tzinfo
    range_start = start.to(timezone.utc)
    range_end = end.to(timezone.utc)
    one_hour = timedelta(hours=1)

    strata: List[HourStratum] = []
    day = start.date()
    last_day = end.date()

    while day <= last_day:
        for hour in range(24):
            naive = dt_datetime.combine(day, dt_time(hour))
            seen_utc_start: Optional[Arrow] = None
            for fold in (0, 1):
                aware = naive.replace(tzinfo=tz, fold=fold)
                if not datetime_exists(aware):
                    # skipped local hour (spring-forward): no stratum
                    continue
                utc_start = Arrow.fromdatetime(aware.astimezone(timezone.utc))
                if seen_utc_start is not None and utc_start == seen_utc_start:
                    # non-ambiguous hour: fold=1 maps to the same instant
                    continue
                seen_utc_start = utc_start
                utc_end = utc_start + one_hour
                eff_start = max(utc_start, range_start)
                eff_end = min(utc_end, range_end)
                if eff_start < eff_end:
                    strata.append(
                        HourStratum(
                            key=f"{day.isoformat()}T{hour:02d}:00#{fold}",
                            local_start=Arrow.fromdatetime(aware),
                            utc_start=eff_start,
                            utc_end=eff_end,
                            fold=fold,
                        )
                    )
        day += timedelta(days=1)

    strata.sort(key=lambda stratum: stratum.utc_start)
    return strata


def _allocate(
    durations: List[int],
    budget: int,
    minimum: int,
    maximum: Optional[int],
) -> List[int]:
    """Distribute ``budget`` across strata proportionally to ``durations``.

    Each stratum receives at least ``minimum`` and at most ``maximum`` (if
    given).  The remainder after the minimums is handed out with largest
    remainders; ties are broken by stratum order (UTC order).  All arithmetic
    is exact integer arithmetic, so the result is deterministic.

    """

    count = len(durations)
    cap = budget if maximum is None else maximum
    allocation = [minimum] * count
    remaining = budget - minimum * count
    active = [i for i in range(count) if durations[i] > 0 and allocation[i] < cap]

    while remaining > 0 and active:
        total_duration = sum(durations[i] for i in active)
        floors = {}
        remainders = {}
        for i in active:
            numerator = remaining * durations[i]
            floors[i] = numerator // total_duration
            remainders[i] = numerator - floors[i] * total_duration
        leftover = remaining - sum(floors.values())
        # largest remainders first, ties broken by UTC (i.e. index) order
        by_remainder = sorted(active, key=lambda i: (-remainders[i], i))
        extra = set(by_remainder[:leftover])
        still_active = []
        for i in active:
            give = floors[i] + (1 if i in extra else 0)
            give = min(give, cap - allocation[i])
            if give > 0:
                allocation[i] += give
                remaining -= give
            if allocation[i] < cap:
                still_active.append(i)
        active = still_active

    return allocation


def _stratum_seed(seed: int, key: str) -> int:
    digest = hashlib.blake2b(f"{seed}|{key}".encode("utf-8"), digest_size=16)
    return int.from_bytes(digest.digest(), "big")


def _sample_stratum(stratum: HourStratum, count: int, seed: int) -> List[Sample]:
    if count <= 0:
        return []
    rng = random.Random(_stratum_seed(seed, stratum.key))
    span = _total_microseconds(stratum.duration)
    offsets = sorted(rng.sample(range(span), count))
    tz = stratum.local_start.tzinfo
    samples = []
    for rank, offset in enumerate(offsets):
        instant = (stratum.utc_start + timedelta(microseconds=offset)).to(tz)
        samples.append(
            Sample(
                sample_id=f"{stratum.key}#{rank:04d}",
                stratum_key=stratum.key,
                instant=instant,
            )
        )
    return samples


def _shortfalls(
    strata: List[HourStratum],
    durations: List[int],
    budget: int,
    minimum: int,
    allocation: List[int],
) -> Tuple[StratumShortfall, ...]:
    if sum(allocation) >= budget:
        return ()
    # budget could not be fully placed: compare against the uncapped ideal
    ideal = _allocate(durations, budget, minimum, None)
    return tuple(
        StratumShortfall(stratum_key=strata[i].key, missing=ideal[i] - allocation[i])
        for i in range(len(strata))
        if ideal[i] > allocation[i]
    )


class SamplingPlan:
    """A stratified sampling plan over a time range.

    Holds the strata, the global allocation table and the shortfalls, and
    produces deterministic samples.  Use :meth:`samples_between` together
    with :meth:`daily_ranges` to replay the same global allocation table as
    per-day tasks and obtain identical sample IDs.

    """

    def __init__(
        self,
        start: Arrow,
        end: Arrow,
        strata: List[HourStratum],
        allocation: Dict[str, int],
        shortfalls: Tuple[StratumShortfall, ...],
        seed: int,
    ) -> None:
        self._start = start
        self._end = end
        self._strata = tuple(strata)
        self._allocation = dict(allocation)
        self._shortfalls = shortfalls
        self._seed = seed

    @property
    def start(self) -> Arrow:
        """The start of the planned range."""

        return self._start

    @property
    def end(self) -> Arrow:
        """The end of the planned range."""

        return self._end

    @property
    def seed(self) -> int:
        """The seed the plan was created with."""

        return self._seed

    @property
    def strata(self) -> Tuple[HourStratum, ...]:
        """The strata of the plan, in UTC order."""

        return self._strata

    @property
    def allocation(self) -> Dict[str, int]:
        """The global allocation table, mapping stratum keys to quotas."""

        return dict(self._allocation)

    @property
    def shortfalls(self) -> Tuple[StratumShortfall, ...]:
        """Strata whose upper bound prevented the budget from being met."""

        return self._shortfalls

    def daily_ranges(self) -> List[Tuple[Arrow, Arrow]]:
        """Split the planned range into per-day task ranges."""

        return daily_ranges(self._start, self._end)

    def samples(self) -> List[Sample]:
        """Draw all samples of the plan, in UTC order."""

        return self.samples_between(self._start, self._end)

    def samples_between(
        self, start: Union[Arrow, dt_datetime], end: Union[Arrow, dt_datetime]
    ) -> List[Sample]:
        """Replay the global allocation table for a sub-range.

        Strata are selected from the plan (not rebuilt), so replaying the
        per-day ranges returned by :meth:`daily_ranges` yields exactly the
        same sample IDs as :meth:`samples`.

        """

        window_start = _as_aware_arrow(start, "start").to(timezone.utc)
        window_end = _as_aware_arrow(end, "end").to(timezone.utc)
        samples: List[Sample] = []
        for stratum in self._strata:
            if stratum.utc_start < window_end and stratum.utc_end > window_start:
                count = self._allocation.get(stratum.key, 0)
                for sample in _sample_stratum(stratum, count, self._seed):
                    instant_utc = sample.instant.to(timezone.utc)
                    if window_start <= instant_utc < window_end:
                        samples.append(sample)
        return samples


def stratified_plan(
    start: Union[Arrow, dt_datetime],
    end: Union[Arrow, dt_datetime],
    budget: int,
    min_per_stratum: int = 0,
    max_per_stratum: Optional[int] = None,
    seed: int = 0,
) -> SamplingPlan:
    """Create a stratified sampling plan for the range ``[start, end)``.

    :param start: aware start of the range; its timezone defines the local
        hours used as strata.
    :param end: aware end of the range (converted to ``start``'s timezone).
    :param budget: total number of samples to distribute.
    :param min_per_stratum: lower bound of samples each stratum receives.
    :param max_per_stratum: optional upper bound of samples per stratum.
    :param seed: seed making the sampled instants deterministic.

    Quotas are integers distributed proportionally to the true (UTC)
    duration of each stratum; ties are broken by UTC order.  If the budget
    exceeds what the upper bounds allow, the plan's ``shortfalls`` list the
    unmet demand per stratum.

    """

    start_arrow = _as_aware_arrow(start, "start")
    end_arrow = _as_aware_arrow(end, "end").to(start_arrow.tzinfo)
    if not start_arrow < end_arrow:
        raise ValueError("start must be before end")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if min_per_stratum < 0:
        raise ValueError("min_per_stratum must be non-negative")
    if max_per_stratum is not None and max_per_stratum < min_per_stratum:
        raise ValueError("max_per_stratum must be >= min_per_stratum")

    strata = _build_strata(start_arrow, end_arrow)
    if min_per_stratum * len(strata) > budget:
        raise ValueError(
            f"budget {budget} cannot cover the minimum of "
            f"{min_per_stratum} for {len(strata)} strata"
        )

    durations = [_total_microseconds(stratum.duration) for stratum in strata]
    quotas = _allocate(durations, budget, min_per_stratum, max_per_stratum)
    allocation = {stratum.key: quota for stratum, quota in zip(strata, quotas)}
    shortfalls = _shortfalls(strata, durations, budget, min_per_stratum, quotas)

    return SamplingPlan(
        start=start_arrow,
        end=end_arrow,
        strata=strata,
        allocation=allocation,
        shortfalls=shortfalls,
        seed=seed,
    )


def stratified_sample(
    start: Union[Arrow, dt_datetime],
    end: Union[Arrow, dt_datetime],
    budget: int,
    min_per_stratum: int = 0,
    max_per_stratum: Optional[int] = None,
    seed: int = 0,
) -> List[Sample]:
    """Draw stratified uniform samples from the range ``[start, end)``.

    Equivalent to ``stratified_plan(...).samples()``; see
    :func:`stratified_plan` for the parameters.

    """

    return stratified_plan(
        start,
        end,
        budget,
        min_per_stratum=min_per_stratum,
        max_per_stratum=max_per_stratum,
        seed=seed,
    ).samples()


def daily_ranges(
    start: Union[Arrow, dt_datetime], end: Union[Arrow, dt_datetime]
) -> List[Tuple[Arrow, Arrow]]:
    """Split the range ``[start, end)`` into per-day task ranges.

    Boundaries fall on local midnights in ``start``'s timezone, so every
    local-hour stratum lies entirely inside exactly one task range.

    """

    start_arrow = _as_aware_arrow(start, "start")
    end_arrow = _as_aware_arrow(end, "end").to(start_arrow.tzinfo)
    if not start_arrow < end_arrow:
        raise ValueError("start must be before end")

    tz = start_arrow.tzinfo
    ranges = []
    current = start_arrow
    while current < end_arrow:
        next_day = current.date() + timedelta(days=1)
        midnight = resolve_imaginary(dt_datetime.combine(next_day, dt_time.min, tz))
        boundary = Arrow.fromdatetime(midnight)
        ranges.append((current, min(boundary, end_arrow)))
        current = boundary
    return ranges


def replay_range(
    start: Union[Arrow, dt_datetime],
    end: Union[Arrow, dt_datetime],
    allocation: Dict[str, int],
    seed: int = 0,
) -> List[Sample]:
    """Replay a global allocation table for the range ``[start, end)``.

    Strata are rebuilt from the range and their quotas looked up in
    ``allocation``.  Replaying the per-day ranges of :func:`daily_ranges`
    with the ``allocation`` of a :class:`SamplingPlan` yields the same
    sample IDs as the plan itself.

    """

    start_arrow = _as_aware_arrow(start, "start")
    end_arrow = _as_aware_arrow(end, "end").to(start_arrow.tzinfo)
    if not start_arrow < end_arrow:
        raise ValueError("start must be before end")

    samples: List[Sample] = []
    for stratum in _build_strata(start_arrow, end_arrow):
        if stratum.key not in allocation:
            raise KeyError(
                f"allocation table has no entry for stratum {stratum.key!r}"
            )
        samples.extend(_sample_stratum(stratum, allocation[stratum.key], seed))
    return samples
