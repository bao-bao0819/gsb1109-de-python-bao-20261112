"""
Provides the :class:`Arrow <arrow.arrow.Arrow>` class, an enhanced ``datetime``
replacement.

"""

import calendar
import re
import sys
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from datetime import timedelta, timezone
from datetime import tzinfo as dt_tzinfo
from math import trunc
from time import struct_time
from typing import (
    Any,
    ClassVar,
    Dict,
    Final,
    Generator,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

from dateutil import tz as dateutil_tz
from dateutil.relativedelta import relativedelta

from arrow import formatter, locales, parser, util
from arrow.constants import DEFAULT_LOCALE, DEHUMANIZE_LOCALES
from arrow.locales import TimeFrameLiteral

TZ_EXPR = Union[dt_tzinfo, str]

_T_FRAMES = Literal[
    "year",
    "years",
    "month",
    "months",
    "day",
    "days",
    "hour",
    "hours",
    "minute",
    "minutes",
    "second",
    "seconds",
    "microsecond",
    "microseconds",
    "week",
    "weeks",
    "quarter",
    "quarters",
]

_BOUNDS = Literal["[)", "()", "(]", "[]"]

# A single relative-shift rule, e.g. {"months": 1} or {"days": -3}.
_SHIFT_RULE = Mapping[str, int]

_GRANULARITY = Literal[
    "auto",
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "quarter",
    "year",
]


class Arrow:
    """An :class:`Arrow <arrow.arrow.Arrow>` object.

    Implements the ``datetime`` interface, behaving as an aware ``datetime`` while implementing
    additional functionality.

    :param year: the calendar year.
    :param month: the calendar month.
    :param day: the calendar day.
    :param hour: (optional) the hour. Defaults to 0.
    :param minute: (optional) the minute, Defaults to 0.
    :param second: (optional) the second, Defaults to 0.
    :param microsecond: (optional) the microsecond. Defaults to 0.
    :param tzinfo: (optional) A timezone expression.  Defaults to UTC.
    :param fold: (optional) 0 or 1, used to disambiguate repeated wall times. Defaults to 0.

    .. _tz-expr:

    Recognized timezone expressions:

        - A ``tzinfo`` object.
        - A ``str`` describing a timezone, similar to 'US/Pacific', or 'Europe/Berlin'.
        - A ``str`` in ISO 8601 style, as in '+07:00'.
        - A ``str``, one of the following:  'local', 'utc', 'UTC'.

    Usage::

        >>> import arrow
        >>> arrow.Arrow(2013, 5, 5, 12, 30, 45)
        <Arrow [2013-05-05T12:30:45+00:00]>

    """

    resolution: ClassVar[timedelta] = dt_datetime.resolution
    min: ClassVar["Arrow"]
    max: ClassVar["Arrow"]

    _ATTRS: Final[List[str]] = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
    ]
    _ATTRS_PLURAL: Final[List[str]] = [f"{a}s" for a in _ATTRS]
    _MONTHS_PER_QUARTER: Final[int] = 3
    _MONTHS_PER_YEAR: Final[int] = 12
    _SECS_PER_MINUTE: Final[int] = 60
    _SECS_PER_HOUR: Final[int] = 60 * 60
    _SECS_PER_DAY: Final[int] = 60 * 60 * 24
    _SECS_PER_WEEK: Final[int] = 60 * 60 * 24 * 7
    _SECS_PER_MONTH: Final[float] = 60 * 60 * 24 * 30.5
    _SECS_PER_QUARTER: Final[float] = 60 * 60 * 24 * 30.5 * 3
    _SECS_PER_YEAR: Final[int] = 60 * 60 * 24 * 365

    _SECS_MAP: Final[Mapping[TimeFrameLiteral, float]] = {
        "second": 1.0,
        "minute": _SECS_PER_MINUTE,
        "hour": _SECS_PER_HOUR,
        "day": _SECS_PER_DAY,
        "week": _SECS_PER_WEEK,
        "month": _SECS_PER_MONTH,
        "quarter": _SECS_PER_QUARTER,
        "year": _SECS_PER_YEAR,
    }

    _datetime: dt_datetime

    def __init__(
        self,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
        tzinfo: Optional[TZ_EXPR] = None,
        **kwargs: Any,
    ) -> None:
        if tzinfo is None:
            tzinfo = timezone.utc
        # detect that tzinfo is a pytz object (issue #626)
        elif (
            isinstance(tzinfo, dt_tzinfo)
            and hasattr(tzinfo, "localize")
            and hasattr(tzinfo, "zone")
            and tzinfo.zone
        ):
            tzinfo = parser.TzinfoParser.parse(tzinfo.zone)
        elif isinstance(tzinfo, str):
            tzinfo = parser.TzinfoParser.parse(tzinfo)

        fold = kwargs.get("fold", 0)

        self._datetime = dt_datetime(
            year, month, day, hour, minute, second, microsecond, tzinfo, fold=fold
        )

    # factories: single object, both original and from datetime.

    @classmethod
    def now(cls, tzinfo: Optional[dt_tzinfo] = None) -> "Arrow":
        """Constructs an :class:`Arrow <arrow.arrow.Arrow>` object, representing "now" in the given
        timezone.

        :param tzinfo: (optional) a ``tzinfo`` object. Defaults to local time.

        Usage::

            >>> arrow.now('Asia/Baku')
            <Arrow [2019-01-24T20:26:31.146412+04:00]>

        """

        if tzinfo is None:
            tzinfo = dt_datetime.now().astimezone().tzinfo

        dt = dt_datetime.now(tzinfo)

        return cls(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            dt.tzinfo,
            fold=getattr(dt, "fold", 0),
        )

    @classmethod
    def utcnow(cls) -> "Arrow":
        """Constructs an :class:`Arrow <arrow.arrow.Arrow>` object, representing "now" in UTC
        time.

        Usage::

            >>> arrow.utcnow()
            <Arrow [2019-01-24T16:31:40.651108+00:00]>

        """

        dt = dt_datetime.now(timezone.utc)

        return cls(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            dt.tzinfo,
            fold=getattr(dt, "fold", 0),
        )

    @classmethod
    def fromtimestamp(
        cls,
        timestamp: Union[int, float, str],
        tzinfo: Optional[TZ_EXPR] = None,
    ) -> "Arrow":
        """Constructs an :class:`Arrow <arrow.arrow.Arrow>` object from a timestamp, converted to
        the given timezone.

        :param timestamp: an ``int`` or ``float`` timestamp, or a ``str`` that converts to either.
        :param tzinfo: (optional) a ``tzinfo`` object.  Defaults to local time.

        """

        if tzinfo is None:
            tzinfo = dt_datetime.now().astimezone().tzinfo
        elif isinstance(tzinfo, str):
            tzinfo = parser.TzinfoParser.parse(tzinfo)

        if not util.is_timestamp(timestamp):
            raise ValueError(f"The provided timestamp {timestamp!r} is invalid.")

        timestamp = util.normalize_timestamp(float(timestamp))
        dt = dt_datetime.fromtimestamp(timestamp, tzinfo)

        return cls(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            dt.tzinfo,
            fold=getattr(dt, "fold", 0),
        )

    @classmethod
    def utcfromtimestamp(cls, timestamp: Union[int, float, str]) -> "Arrow":
        """Constructs an :class:`Arrow <arrow.arrow.Arrow>` object from a timestamp, in UTC time.

        :param timestamp: an ``int`` or ``float`` timestamp, or a ``str`` that converts to either.

        """

        if not util.is_timestamp(timestamp):
            raise ValueError(f"The provided timestamp {timestamp!r} is invalid.")

        timestamp = util.normalize_timestamp(float(timestamp))
        dt = dt_datetime.fromtimestamp(timestamp, timezone.utc)

        return cls(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            timezone.utc,
            fold=getattr(dt, "fold", 0),
        )

    @classmethod
    def fromdatetime(cls, dt: dt_datetime, tzinfo: Optional[TZ_EXPR] = None) -> "Arrow":
        """Constructs an :class:`Arrow <arrow.arrow.Arrow>` object from a ``datetime`` and
        optional replacement timezone.

        :param dt: the ``datetime``
        :param tzinfo: (optional) A :ref:`timezone expression <tz-expr>`.  Defaults to ``dt``'s
            timezone, or UTC if naive.

        Usage::

            >>> dt
            datetime.datetime(2021, 4, 7, 13, 48, tzinfo=tzfile('/usr/share/zoneinfo/US/Pacific'))
            >>> arrow.Arrow.fromdatetime(dt)
            <Arrow [2021-04-07T13:48:00-07:00]>

        """

        if tzinfo is None:
            if dt.tzinfo is None:
                tzinfo = timezone.utc
            else:
                tzinfo = dt.tzinfo

        return cls(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tzinfo,
            fold=getattr(dt, "fold", 0),
        )

    @classmethod
    def fromdate(cls, date: dt_date, tzinfo: Optional[TZ_EXPR] = None) -> "Arrow":
        """Constructs an :class:`Arrow <arrow.arrow.Arrow>` object from a ``date`` and optional
        replacement timezone.  All time values are set to 0.

        :param date: the ``date``
        :param tzinfo: (optional) A :ref:`timezone expression <tz-expr>`.  Defaults to UTC.

        """

        if tzinfo is None:
            tzinfo = timezone.utc

        return cls(date.year, date.month, date.day, tzinfo=tzinfo)

    @classmethod
    def strptime(
        cls, date_str: str, fmt: str, tzinfo: Optional[TZ_EXPR] = None
    ) -> "Arrow":
        """Constructs an :class:`Arrow <arrow.arrow.Arrow>` object from a date string and format,
        in the style of ``datetime.strptime``.  Optionally replaces the parsed timezone.

        :param date_str: the date string.
        :param fmt: the format string using datetime format codes.
        :param tzinfo: (optional) A :ref:`timezone expression <tz-expr>`.  Defaults to the parsed
            timezone if ``fmt`` contains a timezone directive, otherwise UTC.

        Usage::

            >>> arrow.Arrow.strptime('20-01-2019 15:49:10', '%d-%m-%Y %H:%M:%S')
            <Arrow [2019-01-20T15:49:10+00:00]>

        """

        dt = dt_datetime.strptime(date_str, fmt)
        if tzinfo is None:
            tzinfo = dt.tzinfo

        return cls(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tzinfo,
            fold=getattr(dt, "fold", 0),
        )

    @classmethod
    def fromordinal(cls, ordinal: int) -> "Arrow":
        """Constructs an :class:`Arrow <arrow.arrow.Arrow>` object corresponding
            to the Gregorian Ordinal.

        :param ordinal: an ``int`` corresponding to a Gregorian Ordinal.

        Usage::

            >>> arrow.fromordinal(737741)
            <Arrow [2020-11-12T00:00:00+00:00]>

        """

        util.validate_ordinal(ordinal)
        dt = dt_datetime.fromordinal(ordinal)
        return cls(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            dt.tzinfo,
            fold=getattr(dt, "fold", 0),
        )

    # factories: ranges and spans

    @classmethod
    def range(
        cls,
        frame: _T_FRAMES,
        start: Union["Arrow", dt_datetime],
        end: Union["Arrow", dt_datetime, None] = None,
        tz: Optional[TZ_EXPR] = None,
        limit: Optional[int] = None,
    ) -> Generator["Arrow", None, None]:
        """Returns an iterator of :class:`Arrow <arrow.arrow.Arrow>` objects, representing
        points in time between two inputs.

        :param frame: The timeframe.  Can be any ``datetime`` property (day, hour, minute...).
        :param start: A datetime expression, the start of the range.
        :param end: (optional) A datetime expression, the end of the range.
        :param tz: (optional) A :ref:`timezone expression <tz-expr>`.  Defaults to
            ``start``'s timezone, or UTC if ``start`` is naive.
        :param limit: (optional) A maximum number of tuples to return.

        **NOTE**: The ``end`` or ``limit`` must be provided.  Call with ``end`` alone to
        return the entire range.  Call with ``limit`` alone to return a maximum # of results from
        the start.  Call with both to cap a range at a maximum # of results.

        **NOTE**: ``tz`` internally **replaces** the timezones of both ``start`` and ``end`` before
        iterating.  As such, either call with naive objects and ``tz``, or aware objects from the
        same timezone and no ``tz``.

        Supported frame values: year, quarter, month, week, day, hour, minute, second, microsecond.

        Recognized datetime expressions:

            - An :class:`Arrow <arrow.arrow.Arrow>` object.
            - A ``datetime`` object.

        Usage::

            >>> start = datetime(2013, 5, 5, 12, 30)
            >>> end = datetime(2013, 5, 5, 17, 15)
            >>> for r in arrow.Arrow.range('hour', start, end):
            ...     print(repr(r))
            ...
            <Arrow [2013-05-05T12:30:00+00:00]>
            <Arrow [2013-05-05T13:30:00+00:00]>
            <Arrow [2013-05-05T14:30:00+00:00]>
            <Arrow [2013-05-05T15:30:00+00:00]>
            <Arrow [2013-05-05T16:30:00+00:00]>

        **NOTE**: Unlike Python's ``range``, ``end`` *may* be included in the returned iterator::

            >>> start = datetime(2013, 5, 5, 12, 30)
            >>> end = datetime(2013, 5, 5, 13, 30)
            >>> for r in arrow.Arrow.range('hour', start, end):
            ...     print(repr(r))
            ...
            <Arrow [2013-05-05T12:30:00+00:00]>
            <Arrow [2013-05-05T13:30:00+00:00]>

        """

        _, frame_relative, relative_steps = cls._get_frames(frame)

        tzinfo = cls._get_tzinfo(start.tzinfo if tz is None else tz)

        start = cls._get_datetime(start).replace(tzinfo=tzinfo)
        end, limit = cls._get_iteration_params(end, limit)
        end = cls._get_datetime(end).replace(tzinfo=tzinfo)

        current = cls.fromdatetime(start)
        original_day = start.day
        day_is_clipped = False
        i = 0

        while current <= end and i < limit:
            i += 1
            yield current

            values = [getattr(current, f) for f in cls._ATTRS]
            current = cls(*values, tzinfo=tzinfo).shift(  # type: ignore[misc]
                check_imaginary=True, **{frame_relative: relative_steps}
            )

            if frame in ["month", "quarter", "year"] and current.day < original_day:
                day_is_clipped = True

            if day_is_clipped and not cls._is_last_day_of_month(current):
                current = current.replace(day=original_day)

    def span(
        self,
        frame: _T_FRAMES,
        count: int = 1,
        bounds: _BOUNDS = "[)",
        exact: bool = False,
        week_start: int = 1,
    ) -> Tuple["Arrow", "Arrow"]:
        """Returns a tuple of two new :class:`Arrow <arrow.arrow.Arrow>` objects, representing the timespan
        of the :class:`Arrow <arrow.arrow.Arrow>` object in a given timeframe.

        :param frame: the timeframe.  Can be any ``datetime`` property (day, hour, minute...).
        :param count: (optional) the number of frames to span.
        :param bounds: (optional) a ``str`` of either '()', '(]', '[)', or '[]' that specifies
            whether to include or exclude the start and end values in the span. '(' excludes
            the start, '[' includes the start, ')' excludes the end, and ']' includes the end.
            If the bounds are not specified, the default bound '[)' is used.
        :param exact: (optional) whether to have the start of the timespan begin exactly
            at the time specified by ``start`` and the end of the timespan truncated
            so as not to extend beyond ``end``.
        :param week_start: (optional) only used in combination with the week timeframe. Follows isoweekday() where
            Monday is 1 and Sunday is 7.

        Supported frame values: year, quarter, month, week, day, hour, minute, second.

        Usage::

            >>> arrow.utcnow()
            <Arrow [2013-05-09T03:32:36.186203+00:00]>

            >>> arrow.utcnow().span('hour')
            (<Arrow [2013-05-09T03:00:00+00:00]>, <Arrow [2013-05-09T03:59:59.999999+00:00]>)

            >>> arrow.utcnow().span('day')
            (<Arrow [2013-05-09T00:00:00+00:00]>, <Arrow [2013-05-09T23:59:59.999999+00:00]>)

            >>> arrow.utcnow().span('day', count=2)
            (<Arrow [2013-05-09T00:00:00+00:00]>, <Arrow [2013-05-10T23:59:59.999999+00:00]>)

            >>> arrow.utcnow().span('day', bounds='[]')
            (<Arrow [2013-05-09T00:00:00+00:00]>, <Arrow [2013-05-10T00:00:00+00:00]>)

            >>> arrow.utcnow().span('week')
            (<Arrow [2021-02-22T00:00:00+00:00]>, <Arrow [2021-02-28T23:59:59.999999+00:00]>)

            >>> arrow.utcnow().span('week', week_start=6)
            (<Arrow [2021-02-20T00:00:00+00:00]>, <Arrow [2021-02-26T23:59:59.999999+00:00]>)

        """

        util.validate_bounds(bounds)

        frame_absolute, frame_relative, relative_steps = self._get_frames(frame)

        if frame_absolute == "week":
            if not 1 <= week_start <= 7:
                raise ValueError("week_start argument must be between 1 and 7.")
            attr = "day"
        elif frame_absolute == "quarter":
            attr = "month"
        else:
            attr = frame_absolute

        floor = self
        if not exact:
            index = self._ATTRS.index(attr)
            frames = self._ATTRS[: index + 1]

            values = [getattr(self, f) for f in frames]

            for _ in range(3 - len(values)):
                values.append(1)

            floor = self.__class__(*values, tzinfo=self.tzinfo)  # type: ignore[misc]

            if frame_absolute == "week":
                # if week_start is greater than self.isoweekday() go back one week by setting delta = 7
                delta = 7 if week_start > self.isoweekday() else 0
                floor = floor.shift(days=-(self.isoweekday() - week_start) - delta)
            elif frame_absolute == "quarter":
                floor = floor.shift(months=-((self.month - 1) % 3))

        ceil = floor.shift(
            check_imaginary=True, **{frame_relative: count * relative_steps}
        )

        if bounds[0] == "(":
            floor = floor.shift(microseconds=+1)

        if bounds[1] == ")":
            ceil = ceil.shift(microseconds=-1)

        return floor, ceil

    def floor(self, frame: _T_FRAMES, **kwargs: Any) -> "Arrow":
        """Returns a new :class:`Arrow <arrow.arrow.Arrow>` object, representing the "floor"
        of the timespan of the :class:`Arrow <arrow.arrow.Arrow>` object in a given timeframe.
        Equivalent to the first element in the 2-tuple returned by
        :func:`span <arrow.arrow.Arrow.span>`.

        :param frame: the timeframe.  Can be any ``datetime`` property (day, hour, minute...).
        :param week_start: (optional) only used in combination with the week timeframe. Follows isoweekday() where
            Monday is 1 and Sunday is 7.

        Usage::

            >>> arrow.utcnow().floor('hour')
            <Arrow [2013-05-09T03:00:00+00:00]>

            >>> arrow.utcnow().floor('week', week_start=7)
            <Arrow [2021-02-21T00:00:00+00:00]>

        """

        return self.span(frame, **kwargs)[0]

    def ceil(self, frame: _T_FRAMES, **kwargs: Any) -> "Arrow":
        """Returns a new :class:`Arrow <arrow.arrow.Arrow>` object, representing the "ceiling"
        of the timespan of the :class:`Arrow <arrow.arrow.Arrow>` object in a given timeframe.
        Equivalent to the second element in the 2-tuple returned by
        :func:`span <arrow.arrow.Arrow.span>`.

        :param frame: the timeframe.  Can be any ``datetime`` property (day, hour, minute...).
        :param week_start: (optional) only used in combination with the week timeframe. Follows isoweekday() where
            Monday is 1 and Sunday is 7.

        Usage::

            >>> arrow.utcnow().ceil('hour')
            <Arrow [2013-05-09T03:59:59.999999+00:00]>

            >>> arrow.utcnow().ceil('week', week_start=7)
            <Arrow [2021-02-27T23:59:59.999999+00:00]>

        """

        return self.span(frame, **kwargs)[1]

    @classmethod
    def span_range(
        cls,
        frame: _T_FRAMES,
        start: dt_datetime,
        end: dt_datetime,
        tz: Optional[TZ_EXPR] = None,
        limit: Optional[int] = None,
        bounds: _BOUNDS = "[)",
        exact: bool = False,
    ) -> Iterable[Tuple["Arrow", "Arrow"]]:
        """Returns an iterator of tuples, each :class:`Arrow <arrow.arrow.Arrow>` objects,
        representing a series of timespans between two inputs.

        :param frame: The timeframe.  Can be any ``datetime`` property (day, hour, minute...).
        :param start: A datetime expression, the start of the range.
        :param end: (optional) A datetime expression, the end of the range.
        :param tz: (optional) A :ref:`timezone expression <tz-expr>`.  Defaults to
            ``start``'s timezone, or UTC if ``start`` is naive.
        :param limit: (optional) A maximum number of tuples to return.
        :param bounds: (optional) a ``str`` of either '()', '(]', '[)', or '[]' that specifies
            whether to include or exclude the start and end values in each span in the range. '(' excludes
            the start, '[' includes the start, ')' excludes the end, and ']' includes the end.
            If the bounds are not specified, the default bound '[)' is used.
        :param exact: (optional) whether to have the first timespan start exactly
            at the time specified by ``start`` and the final span truncated
            so as not to extend beyond ``end``.

        **NOTE**: The ``end`` or ``limit`` must be provided.  Call with ``end`` alone to
        return the entire range.  Call with ``limit`` alone to return a maximum # of results from
        the start.  Call with both to cap a range at a maximum # of results.

        **NOTE**: ``tz`` internally **replaces** the timezones of both ``start`` and ``end`` before
        iterating.  As such, either call with naive objects and ``tz``, or aware objects from the
        same timezone and no ``tz``.

        Supported frame values: year, quarter, month, week, day, hour, minute, second, microsecond.

        Recognized datetime expressions:

            - An :class:`Arrow <arrow.arrow.Arrow>` object.
            - A ``datetime`` object.

        **NOTE**: Unlike Python's ``range``, ``end`` will *always* be included in the returned
        iterator of timespans.

        Usage:

            >>> start = datetime(2013, 5, 5, 12, 30)
            >>> end = datetime(2013, 5, 5, 17, 15)
            >>> for r in arrow.Arrow.span_range('hour', start, end):
            ...     print(r)
            ...
            (<Arrow [2013-05-05T12:00:00+00:00]>, <Arrow [2013-05-05T12:59:59.999999+00:00]>)
            (<Arrow [2013-05-05T13:00:00+00:00]>, <Arrow [2013-05-05T13:59:59.999999+00:00]>)
            (<Arrow [2013-05-05T14:00:00+00:00]>, <Arrow [2013-05-05T14:59:59.999999+00:00]>)
            (<Arrow [2013-05-05T15:00:00+00:00]>, <Arrow [2013-05-05T15:59:59.999999+00:00]>)
            (<Arrow [2013-05-05T16:00:00+00:00]>, <Arrow [2013-05-05T16:59:59.999999+00:00]>)
            (<Arrow [2013-05-05T17:00:00+00:00]>, <Arrow [2013-05-05T17:59:59.999999+00:00]>)

        """

        tzinfo = cls._get_tzinfo(start.tzinfo if tz is None else tz)
        start = cls.fromdatetime(start, tzinfo).span(frame, exact=exact)[0]
        end = cls.fromdatetime(end, tzinfo)
        _range = cls.range(frame, start, end, tz, limit)
        if not exact:
            for r in _range:
                yield r.span(frame, bounds=bounds, exact=exact)

        for r in _range:
            floor, ceil = r.span(frame, bounds=bounds, exact=exact)
            if ceil > end:
                ceil = end
                if bounds[1] == ")":
                    ceil += relativedelta(microseconds=-1)
            if floor == end:
                break
            elif floor + relativedelta(microseconds=-1) == end:
                break
            yield floor, ceil

    @classmethod
    def interval(
        cls,
        frame: _T_FRAMES,
        start: dt_datetime,
        end: dt_datetime,
        interval: int = 1,
        tz: Optional[TZ_EXPR] = None,
        bounds: _BOUNDS = "[)",
        exact: bool = False,
    ) -> Iterable[Tuple["Arrow", "Arrow"]]:
        """Returns an iterator of tuples, each :class:`Arrow <arrow.arrow.Arrow>` objects,
        representing a series of intervals between two inputs.

        :param frame: The timeframe.  Can be any ``datetime`` property (day, hour, minute...).
        :param start: A datetime expression, the start of the range.
        :param end: (optional) A datetime expression, the end of the range.
        :param interval: (optional) Time interval for the given time frame.
        :param tz: (optional) A timezone expression.  Defaults to UTC.
        :param bounds: (optional) a ``str`` of either '()', '(]', '[)', or '[]' that specifies
            whether to include or exclude the start and end values in the intervals. '(' excludes
            the start, '[' includes the start, ')' excludes the end, and ']' includes the end.
            If the bounds are not specified, the default bound '[)' is used.
        :param exact: (optional) whether to have the first timespan start exactly
            at the time specified by ``start`` and the final interval truncated
            so as not to extend beyond ``end``.

        Supported frame values: year, quarter, month, week, day, hour, minute, second

        Recognized datetime expressions:

            - An :class:`Arrow <arrow.arrow.Arrow>` object.
            - A ``datetime`` object.

        Recognized timezone expressions:

            - A ``tzinfo`` object.
            - A ``str`` describing a timezone, similar to 'US/Pacific', or 'Europe/Berlin'.
            - A ``str`` in ISO 8601 style, as in '+07:00'.
            - A ``str``, one of the following:  'local', 'utc', 'UTC'.

        Usage:

            >>> start = datetime(2013, 5, 5, 12, 30)
            >>> end = datetime(2013, 5, 5, 17, 15)
            >>> for r in arrow.Arrow.interval('hour', start, end, 2):
            ...     print(r)
            ...
            (<Arrow [2013-05-05T12:00:00+00:00]>, <Arrow [2013-05-05T13:59:59.999999+00:00]>)
            (<Arrow [2013-05-05T14:00:00+00:00]>, <Arrow [2013-05-05T15:59:59.999999+00:00]>)
            (<Arrow [2013-05-05T16:00:00+00:00]>, <Arrow [2013-05-05T17:59:59.999999+00:0]>)
        """
        if interval < 1:
            raise ValueError("interval has to be a positive integer")

        spanRange = iter(
            cls.span_range(frame, start, end, tz, bounds=bounds, exact=exact)
        )
        while True:
            try:
                intvlStart, intvlEnd = next(spanRange)
                for _ in range(interval - 1):
                    try:
                        _, intvlEnd = next(spanRange)
                    except StopIteration:
                        continue
                yield intvlStart, intvlEnd
            except StopIteration:
                return

    # representations

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.__str__()}]>"

    def __str__(self) -> str:
        return self._datetime.isoformat()

    def __format__(self, formatstr: str) -> str:
        if len(formatstr) > 0:
            return self.format(formatstr)

        return str(self)

    def __hash__(self) -> int:
        return self._datetime.__hash__()

    # attributes and properties

    def __getattr__(self, name: str) -> Any:
        if name == "week":
            return self.isocalendar()[1]

        if name == "quarter":
            return int((self.month - 1) / self._MONTHS_PER_QUARTER) + 1

        if not name.startswith("_"):
            value: Optional[Any] = getattr(self._datetime, name, None)

            if value is not None:
                return value

        return cast(int, object.__getattribute__(self, name))

    @property
    def tzinfo(self) -> dt_tzinfo:
        """Gets the ``tzinfo`` of the :class:`Arrow <arrow.arrow.Arrow>` object.

        Usage::

            >>> arw=arrow.utcnow()
            >>> arw.tzinfo
            tzutc()

        """

        # In Arrow, `_datetime` cannot be naive.
        return cast(dt_tzinfo, self._datetime.tzinfo)

    @property
    def datetime(self) -> dt_datetime:
        """Returns a datetime representation of the :class:`Arrow <arrow.arrow.Arrow>` object.

        Usage::

            >>> arw=arrow.utcnow()
            >>> arw.datetime
            datetime.datetime(2019, 1, 24, 16, 35, 27, 276649, tzinfo=tzutc())

        """

        return self._datetime

    @property
    def naive(self) -> dt_datetime:
        """Returns a naive datetime representation of the :class:`Arrow <arrow.arrow.Arrow>`
        object.

        Usage::

            >>> nairobi = arrow.now('Africa/Nairobi')
            >>> nairobi
            <Arrow [2019-01-23T19:27:12.297999+03:00]>
            >>> nairobi.naive
            datetime.datetime(2019, 1, 23, 19, 27, 12, 297999)

        """

        return self._datetime.replace(tzinfo=None)

    def timestamp(self) -> float:
        """Returns a timestamp representation of the :class:`Arrow <arrow.arrow.Arrow>` object, in
        UTC time.

        Usage::

            >>> arrow.utcnow().timestamp()
            1616882340.256501

        """

        return self._datetime.timestamp()

    @property
    def int_timestamp(self) -> int:
        """Returns an integer timestamp representation of the :class:`Arrow <arrow.arrow.Arrow>` object, in
        UTC time.

        Usage::

            >>> arrow.utcnow().int_timestamp
            1548260567

        """

        return int(self.timestamp())

    @property
    def float_timestamp(self) -> float:
        """Returns a floating-point timestamp representation of the :class:`Arrow <arrow.arrow.Arrow>`
        object, in UTC time.

        Usage::

            >>> arrow.utcnow().float_timestamp
            1548260516.830896

        """

        return self.timestamp()

    @property
    def fold(self) -> int:
        """Returns the ``fold`` value of the :class:`Arrow <arrow.arrow.Arrow>` object."""

        return self._datetime.fold

    @property
    def ambiguous(self) -> bool:
        """Indicates whether the :class:`Arrow <arrow.arrow.Arrow>` object is a repeated wall time in the current
        timezone.

        """

        return dateutil_tz.datetime_ambiguous(self._datetime)

    @property
    def imaginary(self) -> bool:
        """Indicates whether the :class: `Arrow <arrow.arrow.Arrow>` object exists in the current timezone."""

        return not dateutil_tz.datetime_exists(self._datetime)

    # mutation and duplication.

    def clone(self) -> "Arrow":
        """Returns a new :class:`Arrow <arrow.arrow.Arrow>` object, cloned from the current one.

        Usage:

            >>> arw = arrow.utcnow()
            >>> cloned = arw.clone()

        """

        return self.fromdatetime(self._datetime)

    def replace(self, **kwargs: Any) -> "Arrow":
        """Returns a new :class:`Arrow <arrow.arrow.Arrow>` object with attributes updated
        according to inputs.

        Use property names to set their value absolutely::

            >>> import arrow
            >>> arw = arrow.utcnow()
            >>> arw
            <Arrow [2013-05-11T22:27:34.787885+00:00]>
            >>> arw.replace(year=2014, month=6)
            <Arrow [2014-06-11T22:27:34.787885+00:00]>

        You can also replace the timezone without conversion, using a
        :ref:`timezone expression <tz-expr>`::

            >>> arw.replace(tzinfo=tz.tzlocal())
            <Arrow [2013-05-11T22:27:34.787885-07:00]>

        """

        absolute_kwargs = {}

        for key, value in kwargs.items():
            if key in self._ATTRS:
                absolute_kwargs[key] = value
            elif key in ["week", "quarter"]:
                raise ValueError(f"Setting absolute {key} is not supported.")
            elif key not in ["tzinfo", "fold"]:
                raise ValueError(f"Unknown attribute: {key!r}.")

        current = self._datetime.replace(**absolute_kwargs)

        tzinfo = kwargs.get("tzinfo")

        if tzinfo is not None:
            tzinfo = self._get_tzinfo(tzinfo)
            current = current.replace(tzinfo=tzinfo)

        fold = kwargs.get("fold")

        if fold is not None:
            current = current.replace(fold=fold)

        return self.fromdatetime(current)

    def shift(self, check_imaginary: bool = True, **kwargs: Any) -> "Arrow":
        """Returns a new :class:`Arrow <arrow.arrow.Arrow>` object with attributes updated
        according to inputs.

        Parameters:
        check_imaginary (bool): If True (default), will check for and resolve
        imaginary times (like during DST transitions). If False, skips this check.


        Use pluralized property names to relatively shift their current value:

        >>> import arrow
        >>> arw = arrow.utcnow()
        >>> arw
        <Arrow [2013-05-11T22:27:34.787885+00:00]>
        >>> arw.shift(years=1, months=-1)
        <Arrow [2014-04-11T22:27:34.787885+00:00]>

        Day-of-the-week relative shifting can use either Python's weekday numbers
        (Monday = 0, Tuesday = 1 .. Sunday = 6) or using dateutil.relativedelta's
        day instances (MO, TU .. SU).  When using weekday numbers, the returned
        date will always be greater than or equal to the starting date.

        Using the above code (which is a Saturday) and asking it to shift to Saturday:

        >>> arw.shift(weekday=5)
        <Arrow [2013-05-11T22:27:34.787885+00:00]>

        While asking for a Monday:

        >>> arw.shift(weekday=0)
        <Arrow [2013-05-13T22:27:34.787885+00:00]>

        """

        relative_kwargs = {}
        additional_attrs = ["weeks", "quarters", "weekday"]

        for key, value in kwargs.items():
            if key in self._ATTRS_PLURAL or key in additional_attrs:
                relative_kwargs[key] = value
            else:
                supported_attr = ", ".join(self._ATTRS_PLURAL + additional_attrs)
                raise ValueError(
                    f"Invalid shift time frame. Please select one of the following: {supported_attr}."
                )

        # core datetime does not support quarters, translate to months.
        relative_kwargs.setdefault("months", 0)
        relative_kwargs["months"] += (
            relative_kwargs.pop("quarters", 0) * self._MONTHS_PER_QUARTER
        )

        current = self._datetime + relativedelta(**relative_kwargs)

        # If check_imaginary is True, perform the check for imaginary times (DST transitions)
        if check_imaginary and not dateutil_tz.datetime_exists(current):
            current = dateutil_tz.resolve_imaginary(current)

        return self.fromdatetime(current)

    @classmethod
    def reverse_shift(
        cls,
        target: "Arrow",
        start: "Arrow",
        end: "Arrow",
        shifts: Sequence[_SHIFT_RULE],
        fold: Optional[int] = None,
        check_imaginary: bool = True,
    ) -> List[Tuple["Arrow", "Arrow"]]:
        """Reverses a sequence of relative :meth:`shift` calls.

        Given a ``target`` due date, a bounded interval of candidate billing
        (source) dates ``start``–``end`` and an ordered list of shift rules,
        returns every maximal, closed interval of source datetimes inside the
        bounds for which applying the shifts **in order** lands on the target's
        calendar day in the target's timezone.

        This correctly handles month-end clamping, where several distinct source
        days collapse onto the same resulting day (e.g. Jan 29/30/31 all shift to
        Feb 29 in a leap year), which makes a naive subtraction of months invalid.
        It also handles interleaved month and day rules, since the rules are
        replayed in their original order rather than combined and negated.

        :param target: an :class:`Arrow` whose calendar day (and timezone) is the
            desired result of applying ``shifts``.
        :param start: the earliest candidate source datetime to consider (inclusive).
        :param end: the latest candidate source datetime to consider (inclusive).
        :param shifts: an ordered sequence of shift keyword mappings, applied in
            order, e.g. ``[{"days": 1}, {"months": 1}]``.
        :param fold: (optional) ``0`` or ``1`` to additionally require that the
            resulting wall time carry that ``fold`` (relevant for ambiguous DST
            times); defaults to ``None`` to accept either fold.
        :param check_imaginary: (optional) forwarded to each :meth:`shift` call;
            when ``True`` (default) non-existent DST wall times are resolved.
        :returns: a list of disjoint, ordered ``(start, end)`` :class:`Arrow`
            tuples, each fully inside the supplied bounds and inclusive on both
            ends. An empty list means no source datetime in the interval works.

        Usage::

            >>> import arrow
            >>> due = arrow.get(2020, 2, 29)
            >>> lo, hi = arrow.get(2020, 1, 1), arrow.get(2020, 2, 15)
            >>> arrow.Arrow.reverse_shift(due, lo, hi, [{"months": 1}])
            [(<Arrow [2020-01-29T00:00:00+00:00]>, <Arrow [2020-01-31T23:59:59.999999+00:00]>)]

        The search runs over source *day blocks* (a constant number of shift
        evaluations per block via edge probes and logarithmic binary searches), and
        only falls back to a whole-second sweep -- never a microsecond sweep -- on
        the rare blocks containing a DST/fold transition, a deleted calendar day or
        a month-clamp plus time-of-day discontinuity, so a long input interval is
        handled in roughly linear time in the number of days rather than its
        microsecond length.
        """

        if start > end:
            raise ValueError("'start' must be on or before 'end'.")
        if fold is not None and fold not in (0, 1):
            raise ValueError("'fold' must be None, 0 or 1.")

        tzinfo = target.tzinfo
        target_date = target.date()

        def shift_all(source: "Arrow") -> "Arrow":
            current = source
            for rule in shifts:
                current = current.shift(check_imaginary=check_imaginary, **rule)
            return current.to(tzinfo)

        def matches(source: "Arrow") -> bool:
            result = shift_all(source)
            if result.date() != target_date:
                return False
            return fold is None or getattr(result.datetime, "fold", 0) == fold

        # Split the source interval into absolute day blocks. Month/year clamping
        # (e.g. source days 29/30/31 onto one short month) only changes value at a
        # wall-day boundary; inside a fixed-offset block the result instant is
        # normally monotone and the matching set is one interval. Blocks that can be
        # non-monotone -- DST/fold, a deleted calendar day, or a clamp plus
        # time-of-day mix -- are detected and solved with an exact whole-second
        # scan. The interval is therefore never enumerated microsecond by microsecond.
        target_ambiguous = cls._day_has_ambiguity(tzinfo, target_date)
        # The target wall day maps to a (possibly non-24-hour) UTC instant
        # interval bounded by its two local midnights, which handles DST gaps,
        # repeats and zones that delete a whole day.
        target_day_start_utc = cls(
            target_date.year,
            target_date.month,
            target_date.day,
            tzinfo=tzinfo,
        ).to(timezone.utc).datetime
        next_wall_day = target_date + timedelta(days=1)
        target_day_end_utc = cls(
            next_wall_day.year,
            next_wall_day.month,
            next_wall_day.day,
            tzinfo=tzinfo,
        ).to(timezone.utc).datetime
        segments: List[Tuple["Arrow", "Arrow"]] = []

        # Build strictly-increasing absolute day blocks from source midnights.
        # Wall .shift(days=1) is unreliable where a zone deletes a whole calendar
        # day (e.g. Pacific/Apia, 2011-12-30): two wall labels then denote the same
        # instants. Construct each midnight directly, de-duplicate by UTC instant
        # and sort, so a deleted day collapses into one correct, non-overlapping
        # block instead of being revisited.
        source_tz = start.tzinfo
        midnight_by_instant: Dict[dt_datetime, "Arrow"] = {}
        has_deleted_day = False
        cursor_date = start.date()
        last_date = end.date() + timedelta(days=1)
        while cursor_date <= last_date:
            # Normalize through UTC: in a zone that deletes a day, constructing the
            # ghost wall label yields the old offset; the round trip returns the
            # canonical label of that same instant (the following real midnight),
            # so the deleted day collapses onto its successor instead of producing
            # a non-canonical, overlapping block.
            midnight = cls(
                cursor_date.year,
                cursor_date.month,
                cursor_date.day,
                tzinfo=source_tz,
            ).to(timezone.utc).to(source_tz)
            if midnight.date() != cursor_date:
                has_deleted_day = True
            midnight_by_instant.setdefault(
                midnight.to(timezone.utc).datetime, midnight
            )
            cursor_date += timedelta(days=1)
        midnights = [
            midnight_by_instant[instant]
            for instant in sorted(midnight_by_instant)
        ]
        # A deleted calendar day makes even sub-day block joins non-trivial; such
        # zones are historical one-offs (e.g. Pacific/Apia, 2011-12-30), so always
        # using the exact scan there has no practical performance cost.
        deleted_day_risk = has_deleted_day

        # Month/year/quarter rules clamp at short month ends. Paired with a
        # time-of-day rule that can cross midnight inside one block, the clamp can
        # make the result instant jump backward, so such blocks use the exact scan.
        clamp_frames = {"years", "quarters", "months"}
        time_of_day_frames = {"hours", "minutes", "seconds"}
        present_frames = {key for rule in shifts for key in rule}
        clamp_mix = bool(present_frames & clamp_frames) and bool(
            present_frames & time_of_day_frames
        )

        for block_start, next_midnight in zip(midnights, midnights[1:]):
            block_last = cls.fromdatetime(
                next_midnight.to(timezone.utc).datetime - timedelta(microseconds=1)
            ).to(source_tz)
            day_start = max(block_start, start)
            day_end = min(block_last, end)
            if day_start > day_end:
                continue

            source_transition = day_start.utcoffset() != day_end.utcoffset()
            fold_transition = fold is not None and target_ambiguous
            if (
                source_transition
                or fold_transition
                or deleted_day_risk
                or clamp_mix
            ):
                # These blocks can have a non-contiguous or non-monotone preimage
                # (DST/fold, a deleted wall day, or a month-clamp + time-of-day
                # mix); solve them with the exact whole-second scan. Calendar-only
                # billing rules on ordinary zones stay on the fast path below.
                segments.extend(
                    cls._match_segments_scan(day_start, day_end, matches)
                )
            else:
                # Fixed-offset block. A clamp + time-of-day mix can still jump
                # backward at the single wall-midnight crossing of the sub-day
                # shift; the monotone binary path verifies its result and signals
                # fallback to the exact scan if the assumption is violated.
                monotone_segments = cls._match_segments_monotone(
                    day_start,
                    day_end,
                    shift_all,
                    matches,
                    target_day_start_utc,
                    target_day_end_utc,
                )
                if monotone_segments is None:
                    segments.extend(
                        cls._match_segments_scan(day_start, day_end, matches)
                    )
                else:
                    segments.extend(monotone_segments)

        return cls._merge_segments(segments)

    @classmethod
    def _match_segments_monotone(
        cls,
        day_start: "Arrow",
        day_end: "Arrow",
        shift_all: Any,
        matches: Any,
        target_day_start_utc: dt_datetime,
        target_day_end_utc: dt_datetime,
    ) -> Optional[List[Tuple["Arrow", "Arrow"]]]:
        """Solves one block whose shifted result is monotone in absolute instant.

        The target wall day maps to the UTC instant interval
        ``[target_day_start_utc, target_day_end_utc)`` (not always 24 hours). With
        a fixed source offset inside the block, the shift adds a constant instant
        delta, so sources landing in that interval form one interval found with two
        lower-bound binary searches -- exact even when month clamping makes the
        result *wall* date non-monotone. Returns ``None`` if endpoint verification
        detects that the monotone assumption does not hold, so the caller can fall
        back to the exact whole-second scan.
        """
        span_us = (day_end.datetime - day_start.datetime) // cls.resolution

        def at(offset_us: int) -> "Arrow":
            # Add an elapsed instant to the block's own wall label: instant-correct
            # within one block, and keeps a consistent tzinfo for Arrow ordering
            # (a UTC round trip would re-label times across a DST gap inconsistently).
            return cls.fromdatetime(
                day_start.datetime + timedelta(microseconds=offset_us)
            )

        def result_utc(offset_us: int) -> dt_datetime:
            return cast(
                dt_datetime,
                shift_all(at(offset_us)).to(timezone.utc).datetime,
            )

        # Fast edge probes for the common case where the block does not span more
        # than one target-day boundary: both edges match => whole block, both miss
        # => nothing. Multi-day blocks (the wanted day can be interior) fall
        # through to the instant-range overlap test below.
        start_matches = matches(day_start)
        end_matches = matches(day_end)
        if start_matches and end_matches:
            return [(day_start, day_end)]

        first_instant = result_utc(0)
        last_instant = result_utc(span_us)
        spans_multiple_days = (
            last_instant - first_instant
        ) >= timedelta(hours=23)
        if start_matches == end_matches and not spans_multiple_days:
            return []

        # A monotone block's result range is bounded by its two endpoint instants.
        if (
            last_instant < target_day_start_utc
            or first_instant >= target_day_end_utc
        ):
            return []

        def first_at_or_after(instant: dt_datetime) -> int:
            lo, hi = 0, span_us + 1
            while lo < hi:
                mid = (lo + hi) // 2
                if result_utc(mid) >= instant:
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        first = first_at_or_after(target_day_start_utc)
        after_last = first_at_or_after(target_day_end_utc)
        if first >= after_last:
            return []

        seg_start = at(first)
        seg_end = at(after_last - 1)

        # Defense in depth: confirm the binary-derived boundaries against the real
        # predicate; any contradiction means the instant was not monotone here.
        if not (matches(seg_start) and matches(seg_end)):
            return None
        if first > 0 and matches(at(first - 1)):
            return None
        if after_last <= span_us and matches(at(after_last)):
            return None
        return [(seg_start, seg_end)]

    @classmethod
    def _match_segments_scan(
        cls,
        day_start: "Arrow",
        day_end: "Arrow",
        matches: Any,
    ) -> List[Tuple["Arrow", "Arrow"]]:
        """Returns the matching closed sub-intervals inside one source block.

        Used for DST-transition or fold-ambiguous blocks where the shifted result
        need not be monotone across the block. The block is swept one whole wall
        second at a time to locate matching runs, then each run's two boundary
        seconds are refined to microsecond precision. Inside a single wall second
        the non-microsecond fields are constant and a microsecond rule can carry
        into the next second at most once, so the predicate is monotone there and a
        binary search gives an exact edge. Instant (not wall) arithmetic keeps this
        correct across skipped or repeated local times; the block is enumerated by
        seconds, never microseconds.
        """
        span_us = (day_end.datetime - day_start.datetime) // cls.resolution

        def at(offset_us: int) -> "Arrow":
            # Add an elapsed instant to the block's own wall label: instant-correct
            # within one block, and keeps a consistent tzinfo for Arrow ordering
            # (a UTC round trip would re-label times across a DST gap inconsistently).
            return cls.fromdatetime(
                day_start.datetime + timedelta(microseconds=offset_us)
            )

        total_seconds = span_us // 1_000_000

        def first_match(second: int) -> Optional[int]:
            """First matching microsecond offset within wall ``second``, if any."""
            cell_start = second * 1_000_000
            lo, hi = cell_start, min(cell_start + 999_999, span_us)
            if not matches(at(hi)):
                return None
            while lo < hi:
                mid = (lo + hi) // 2
                if matches(at(mid)):
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        def last_match(second: int) -> Optional[int]:
            """Last matching microsecond offset within wall ``second``, if any."""
            cell_start = second * 1_000_000
            lo, hi = cell_start, min(cell_start + 999_999, span_us)
            if not matches(at(lo)):
                return None
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if matches(at(mid)):
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        runs: List[Tuple[int, int]] = []
        run_start: Optional[int] = None
        previous = False
        for second in range(total_seconds + 1):
            current_matches = matches(at(second * 1_000_000))
            if current_matches and not previous:
                run_start = second
            if previous and not current_matches:
                runs.append((cast(int, run_start), second - 1))
            previous = current_matches
        if previous:
            runs.append((cast(int, run_start), total_seconds))

        segments: List[Tuple["Arrow", "Arrow"]] = []
        for first_second, last_second in runs:
            # A run found from microsecond-0 samples may start partway through the
            # preceding wall second or end partway through the following one.
            start_offset: Optional[int] = None
            if first_second > 0:
                start_offset = first_match(first_second - 1)
            if start_offset is None:
                start_offset = first_match(first_second)
            end_offset: Optional[int] = None
            if last_second < total_seconds:
                end_offset = last_match(last_second + 1)
            if end_offset is None:
                end_offset = last_match(last_second)

            if start_offset is None or end_offset is None:
                continue
            segments.append((at(start_offset), at(end_offset)))
        return segments

    @staticmethod
    def _day_has_ambiguity(tzinfo: dt_tzinfo, day: dt_date) -> bool:
        """Whether ``day`` in ``tzinfo`` contains a repeated (ambiguous) wall time.

        A fall-back stretches the wall day beyond 24 hours; comparing the first
        and last wall readings (using the fold-1 interpretation of the latter)
        detects it in constant time for any transition size.
        """
        first = dt_datetime(day.year, day.month, day.day, tzinfo=tzinfo)
        last = dt_datetime(
            day.year, day.month, day.day, 23, 59, 59, 999999, tzinfo=tzinfo
        ).replace(fold=1)
        elapsed = last.astimezone(timezone.utc) - first.astimezone(timezone.utc)
        return elapsed > timedelta(hours=24)

    @staticmethod
    def _merge_segments(
        segments: List[Tuple["Arrow", "Arrow"]],
    ) -> List[Tuple["Arrow", "Arrow"]]:
        """Merges ordered, touching or overlapping closed segments.

        Adjacency is judged by absolute instant, not wall arithmetic, so segments
        separated only by a skipped local time (e.g. a deleted calendar day) still
        merge when no valid source instant lies between them.
        """
        merged: List[Tuple["Arrow", "Arrow"]] = []
        for seg_start, seg_end in sorted(segments, key=lambda item: item[0]):
            if merged:
                previous_end_utc = merged[-1][1].to(timezone.utc).datetime
                start_utc = seg_start.to(timezone.utc).datetime
                if start_utc <= previous_end_utc + timedelta(microseconds=1):
                    merged[-1] = (merged[-1][0], max(merged[-1][1], seg_end))
                    continue
            merged.append((seg_start, seg_end))
        return merged

    def to(self, tz: TZ_EXPR) -> "Arrow":
        """Returns a new :class:`Arrow <arrow.arrow.Arrow>` object, converted
        to the target timezone.

        :param tz: A :ref:`timezone expression <tz-expr>`.

        Usage::

            >>> utc = arrow.utcnow()
            >>> utc
            <Arrow [2013-05-09T03:49:12.311072+00:00]>

            >>> utc.to('US/Pacific')
            <Arrow [2013-05-08T20:49:12.311072-07:00]>

            >>> utc.to(tz.tzlocal())
            <Arrow [2013-05-08T20:49:12.311072-07:00]>

            >>> utc.to('-07:00')
            <Arrow [2013-05-08T20:49:12.311072-07:00]>

            >>> utc.to('local')
            <Arrow [2013-05-08T20:49:12.311072-07:00]>

            >>> utc.to('local').to('utc')
            <Arrow [2013-05-09T03:49:12.311072+00:00]>

        """

        if not isinstance(tz, dt_tzinfo):
            tz = parser.TzinfoParser.parse(tz)

        dt = self._datetime.astimezone(tz)

        return self.__class__(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            dt.tzinfo,
            fold=getattr(dt, "fold", 0),
        )

    # string output and formatting

    def format(
        self, fmt: str = "YYYY-MM-DD HH:mm:ssZZ", locale: str = DEFAULT_LOCALE
    ) -> str:
        """Returns a string representation of the :class:`Arrow <arrow.arrow.Arrow>` object,
        formatted according to the provided format string. For a list of formatting values,
        see :ref:`supported-tokens`

        :param fmt: the format string.
        :param locale: the locale to format.

        Usage::

            >>> arrow.utcnow().format('YYYY-MM-DD HH:mm:ss ZZ')
            '2013-05-09 03:56:47 -00:00'

            >>> arrow.utcnow().format('X')
            '1368071882'

            >>> arrow.utcnow().format('MMMM DD, YYYY')
            'May 09, 2013'

            >>> arrow.utcnow().format()
            '2013-05-09 03:56:47 -00:00'

        """

        return formatter.DateTimeFormatter(locale).format(self._datetime, fmt)

    def humanize(
        self,
        other: Union["Arrow", dt_datetime, None] = None,
        locale: str = DEFAULT_LOCALE,
        only_distance: bool = False,
        granularity: Union[_GRANULARITY, List[_GRANULARITY]] = "auto",
    ) -> str:
        """Returns a localized, humanized representation of a relative difference in time.

        :param other: (optional) an :class:`Arrow <arrow.arrow.Arrow>` or ``datetime`` object.
            Defaults to now in the current :class:`Arrow <arrow.arrow.Arrow>` object's timezone.
        :param locale: (optional) a ``str`` specifying a locale.  Defaults to 'en-us'.
        :param only_distance: (optional) returns only time difference eg: "11 seconds" without "in" or "ago" part.
        :param granularity: (optional) defines the precision of the output. Set it to strings 'second', 'minute',
                           'hour', 'day', 'week', 'month' or 'year' or a list of any combination of these strings

        Usage::

            >>> earlier = arrow.utcnow().shift(hours=-2)
            >>> earlier.humanize()
            '2 hours ago'

            >>> later = earlier.shift(hours=4)
            >>> later.humanize(earlier)
            'in 4 hours'

        """

        locale_name = locale
        locale = locales.get_locale(locale)

        if other is None:
            utc = dt_datetime.now(timezone.utc).replace(tzinfo=timezone.utc)
            dt = utc.astimezone(self._datetime.tzinfo)

        elif isinstance(other, Arrow):
            dt = other._datetime

        elif isinstance(other, dt_datetime):
            if other.tzinfo is None:
                dt = other.replace(tzinfo=self._datetime.tzinfo)
            else:
                dt = other.astimezone(self._datetime.tzinfo)

        else:
            raise TypeError(
                f"Invalid 'other' argument of type {type(other).__name__!r}. "
                "Argument must be of type None, Arrow, or datetime."
            )

        if isinstance(granularity, list) and len(granularity) == 1:
            granularity = granularity[0]

        _delta = int(round((self._datetime - dt).total_seconds()))
        sign = -1 if _delta < 0 else 1
        delta_second = diff = abs(_delta)

        try:
            if granularity == "auto":
                if diff < 10:
                    return locale.describe("now", only_distance=only_distance)

                if diff < self._SECS_PER_MINUTE:
                    seconds = sign * delta_second
                    return locale.describe(
                        "seconds", seconds, only_distance=only_distance
                    )

                elif diff < self._SECS_PER_MINUTE * 2:
                    return locale.describe("minute", sign, only_distance=only_distance)

                elif diff < self._SECS_PER_HOUR:
                    minutes = sign * max(delta_second // self._SECS_PER_MINUTE, 2)
                    return locale.describe(
                        "minutes", minutes, only_distance=only_distance
                    )

                elif diff < self._SECS_PER_HOUR * 2:
                    return locale.describe("hour", sign, only_distance=only_distance)

                elif diff < self._SECS_PER_DAY:
                    hours = sign * max(delta_second // self._SECS_PER_HOUR, 2)
                    return locale.describe("hours", hours, only_distance=only_distance)

                calendar_diff = (
                    relativedelta(dt, self._datetime)
                    if self._datetime < dt
                    else relativedelta(self._datetime, dt)
                )
                calendar_months = (
                    calendar_diff.years * self._MONTHS_PER_YEAR + calendar_diff.months
                )

                # For months, if more than 2 weeks, count as a full month
                if calendar_diff.days > 14:
                    calendar_months += 1

                calendar_months = min(calendar_months, self._MONTHS_PER_YEAR)

                if diff < self._SECS_PER_DAY * 2:
                    return locale.describe("day", sign, only_distance=only_distance)

                elif diff < self._SECS_PER_WEEK:
                    days = sign * max(delta_second // self._SECS_PER_DAY, 2)
                    return locale.describe("days", days, only_distance=only_distance)

                elif calendar_months >= 1 and diff < self._SECS_PER_YEAR:
                    if calendar_months == 1:
                        return locale.describe(
                            "month", sign, only_distance=only_distance
                        )
                    else:
                        months = sign * calendar_months
                        return locale.describe(
                            "months", months, only_distance=only_distance
                        )

                elif diff < self._SECS_PER_WEEK * 2:
                    return locale.describe("week", sign, only_distance=only_distance)

                elif diff < self._SECS_PER_MONTH:
                    weeks = sign * max(delta_second // self._SECS_PER_WEEK, 2)
                    return locale.describe("weeks", weeks, only_distance=only_distance)

                elif diff < self._SECS_PER_YEAR * 2:
                    return locale.describe("year", sign, only_distance=only_distance)

                else:
                    years = sign * max(delta_second // self._SECS_PER_YEAR, 2)
                    return locale.describe("years", years, only_distance=only_distance)

            elif isinstance(granularity, str):
                granularity = cast(TimeFrameLiteral, granularity)  # type: ignore[assignment]

                if granularity == "second":
                    delta = sign * float(delta_second)
                    if abs(delta) < 2:
                        return locale.describe("now", only_distance=only_distance)
                elif granularity == "minute":
                    delta = sign * delta_second / self._SECS_PER_MINUTE
                elif granularity == "hour":
                    delta = sign * delta_second / self._SECS_PER_HOUR
                elif granularity == "day":
                    delta = sign * delta_second / self._SECS_PER_DAY
                elif granularity == "week":
                    delta = sign * delta_second / self._SECS_PER_WEEK
                elif granularity == "month":
                    delta = sign * delta_second / self._SECS_PER_MONTH
                elif granularity == "quarter":
                    delta = sign * delta_second / self._SECS_PER_QUARTER
                elif granularity == "year":
                    delta = sign * delta_second / self._SECS_PER_YEAR
                else:
                    raise ValueError(
                        "Invalid level of granularity. "
                        "Please select between 'second', 'minute', 'hour', 'day', 'week', 'month', 'quarter' or 'year'."
                    )

                if trunc(abs(delta)) != 1:
                    granularity += "s"  # type: ignore[assignment]
                return locale.describe(granularity, delta, only_distance=only_distance)

            else:
                if not granularity:
                    raise ValueError(
                        "Empty granularity list provided. "
                        "Please select one or more from 'second', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'."
                    )

                timeframes: List[Tuple[TimeFrameLiteral, float]] = []

                def gather_timeframes(_delta: float, _frame: TimeFrameLiteral) -> float:
                    if _frame in granularity:
                        value = sign * _delta / self._SECS_MAP[_frame]
                        _delta %= self._SECS_MAP[_frame]
                        if trunc(abs(value)) != 1:
                            timeframes.append(
                                (cast(TimeFrameLiteral, _frame + "s"), value)
                            )
                        else:
                            timeframes.append((_frame, value))
                    return _delta

                delta = float(delta_second)
                frames: Tuple[TimeFrameLiteral, ...] = (
                    "year",
                    "quarter",
                    "month",
                    "week",
                    "day",
                    "hour",
                    "minute",
                    "second",
                )
                for frame in frames:
                    delta = gather_timeframes(delta, frame)

                if len(timeframes) < len(granularity):
                    raise ValueError(
                        "Invalid level of granularity. "
                        "Please select between 'second', 'minute', 'hour', 'day', 'week', 'month', 'quarter' or 'year'."
                    )

                return locale.describe_multi(timeframes, only_distance=only_distance)

        except KeyError as e:
            raise ValueError(
                f"Humanization of the {e} granularity is not currently translated in the {locale_name!r} locale. "
                "Please consider making a contribution to this locale."
            )

    def dehumanize(self, input_string: str, locale: str = "en_us") -> "Arrow":
        """Returns a new :class:`Arrow <arrow.arrow.Arrow>` object, that represents
        the time difference relative to the attributes of the
        :class:`Arrow <arrow.arrow.Arrow>` object.

        :param timestring: a ``str`` representing a humanized relative time.
        :param locale: (optional) a ``str`` specifying a locale.  Defaults to 'en-us'.

        Usage::

                >>> arw = arrow.utcnow()
                >>> arw
                <Arrow [2021-04-20T22:27:34.787885+00:00]>
                >>> earlier = arw.dehumanize("2 days ago")
                >>> earlier
                <Arrow [2021-04-18T22:27:34.787885+00:00]>

                >>> arw = arrow.utcnow()
                >>> arw
                <Arrow [2021-04-20T22:27:34.787885+00:00]>
                >>> later = arw.dehumanize("in a month")
                >>> later
                <Arrow [2021-05-18T22:27:34.787885+00:00]>

        """

        # Create a locale object based off given local
        locale_obj = locales.get_locale(locale)

        # Check to see if locale is supported
        normalized_locale_name = locale.lower().replace("_", "-")

        if normalized_locale_name not in DEHUMANIZE_LOCALES:
            raise ValueError(
                f"Dehumanize does not currently support the {locale} locale, please consider making a contribution to add support for this locale."
            )

        current_time = self.fromdatetime(self._datetime)

        # Create an object containing the relative time info
        time_object_info = dict.fromkeys(
            ["seconds", "minutes", "hours", "days", "weeks", "months", "years"], 0
        )

        # Create an object representing if unit has been seen
        unit_visited = dict.fromkeys(
            ["now", "seconds", "minutes", "hours", "days", "weeks", "months", "years"],
            False,
        )

        # Create a regex pattern object for numbers
        num_pattern = re.compile(r"\d+")

        # Search input string for each time unit within locale
        for unit, unit_object in locale_obj.timeframes.items():
            # Need to check the type of unit_object to create the correct dictionary
            if isinstance(unit_object, Mapping):
                strings_to_search = unit_object
            else:
                strings_to_search = {unit: str(unit_object)}

            # Search for any matches that exist for that locale's unit.
            # Needs to cycle all through strings as some locales have strings that
            # could overlap in a regex match, since input validation isn't being performed.
            for time_delta, time_string in strings_to_search.items():
                # Replace {0} with regex \d representing digits
                search_string = str(time_string)
                search_string = search_string.format(r"\d+")

                # Create search pattern and find within string
                pattern = re.compile(rf"(^|\b|\d){search_string}")
                match = pattern.search(input_string)

                # If there is no match continue to next iteration
                if not match:
                    continue

                match_string = match.group()
                num_match = num_pattern.search(match_string)

                # If no number matches
                # Need for absolute value as some locales have signs included in their objects
                if not num_match:
                    change_value = (
                        1 if not time_delta.isnumeric() else abs(int(time_delta))
                    )
                else:
                    change_value = int(num_match.group())

                # No time to update if now is the unit
                if unit == "now":
                    unit_visited[unit] = True
                    continue

                # Add change value to the correct unit (incorporates the plurality that exists within timeframe i.e second v.s seconds)
                time_unit_to_change = str(unit)
                time_unit_to_change += (
                    "s" if (str(time_unit_to_change)[-1] != "s") else ""
                )
                time_object_info[time_unit_to_change] = change_value
                unit_visited[time_unit_to_change] = True

        # Assert error if string does not modify any units
        if not any([True for k, v in unit_visited.items() if v]):
            raise ValueError(
                "Input string not valid. Note: Some locales do not support the week granularity in Arrow. "
                "If you are attempting to use the week granularity on an unsupported locale, this could be the cause of this error."
            )

        # Sign logic
        future_string = locale_obj.future
        future_string = future_string.format(".*")
        future_pattern = re.compile(rf"^{future_string}$")
        future_pattern_match = future_pattern.findall(input_string)

        past_string = locale_obj.past
        past_string = past_string.format(".*")
        past_pattern = re.compile(rf"^{past_string}$")
        past_pattern_match = past_pattern.findall(input_string)

        # If a string contains the now unit, there will be no relative units, hence the need to check if the now unit
        # was visited before raising a ValueError
        if past_pattern_match:
            sign_val = -1
        elif future_pattern_match:
            sign_val = 1
        elif unit_visited["now"]:
            sign_val = 0
        else:
            raise ValueError(
                "Invalid input String. String does not contain any relative time information. "
                "String should either represent a time in the future or a time in the past. "
                "Ex: 'in 5 seconds' or '5 seconds ago'."
            )

        time_changes = {k: sign_val * v for k, v in time_object_info.items()}

        return current_time.shift(check_imaginary=True, **time_changes)

    # query functions

    def is_between(
        self,
        start: "Arrow",
        end: "Arrow",
        bounds: _BOUNDS = "()",
    ) -> bool:
        """Returns a boolean denoting whether the :class:`Arrow <arrow.arrow.Arrow>` object is between
        the start and end limits.

        :param start: an :class:`Arrow <arrow.arrow.Arrow>` object.
        :param end: an :class:`Arrow <arrow.arrow.Arrow>` object.
        :param bounds: (optional) a ``str`` of either '()', '(]', '[)', or '[]' that specifies
            whether to include or exclude the start and end values in the range. '(' excludes
            the start, '[' includes the start, ')' excludes the end, and ']' includes the end.
            If the bounds are not specified, the default bound '()' is used.

        Usage::

            >>> start = arrow.get(datetime(2013, 5, 5, 12, 30, 10))
            >>> end = arrow.get(datetime(2013, 5, 5, 12, 30, 36))
            >>> arrow.get(datetime(2013, 5, 5, 12, 30, 27)).is_between(start, end)
            True

            >>> start = arrow.get(datetime(2013, 5, 5))
            >>> end = arrow.get(datetime(2013, 5, 8))
            >>> arrow.get(datetime(2013, 5, 8)).is_between(start, end, '[]')
            True

            >>> start = arrow.get(datetime(2013, 5, 5))
            >>> end = arrow.get(datetime(2013, 5, 8))
            >>> arrow.get(datetime(2013, 5, 8)).is_between(start, end, '[)')
            False

        """

        util.validate_bounds(bounds)

        if not isinstance(start, Arrow):
            raise TypeError(
                f"Cannot parse start date argument type of {type(start)!r}."
            )

        if not isinstance(end, Arrow):
            raise TypeError(f"Cannot parse end date argument type of {type(start)!r}.")

        include_start = bounds[0] == "["
        include_end = bounds[1] == "]"

        target_ts = self.float_timestamp
        start_ts = start.float_timestamp
        end_ts = end.float_timestamp

        return (
            (start_ts <= target_ts <= end_ts)
            and (include_start or start_ts < target_ts)
            and (include_end or target_ts < end_ts)
        )

    # datetime methods

    def date(self) -> dt_date:
        """Returns a ``date`` object with the same year, month and day.

        Usage::

            >>> arrow.utcnow().date()
            datetime.date(2019, 1, 23)

        """

        return self._datetime.date()

    def time(self) -> dt_time:
        """Returns a ``time`` object with the same hour, minute, second, microsecond.

        Usage::

            >>> arrow.utcnow().time()
            datetime.time(12, 15, 34, 68352)

        """

        return self._datetime.time()

    def timetz(self) -> dt_time:
        """Returns a ``time`` object with the same hour, minute, second, microsecond and
        tzinfo.

        Usage::

            >>> arrow.utcnow().timetz()
            datetime.time(12, 5, 18, 298893, tzinfo=tzutc())

        """

        return self._datetime.timetz()

    def astimezone(self, tz: Optional[dt_tzinfo]) -> dt_datetime:
        """Returns a ``datetime`` object, converted to the specified timezone.

        :param tz: a ``tzinfo`` object.

        Usage::

            >>> pacific=arrow.now('US/Pacific')
            >>> nyc=arrow.now('America/New_York').tzinfo
            >>> pacific.astimezone(nyc)
            datetime.datetime(2019, 1, 20, 10, 24, 22, 328172, tzinfo=tzfile('/usr/share/zoneinfo/America/New_York'))

        """

        return self._datetime.astimezone(tz)

    def utcoffset(self) -> Optional[timedelta]:
        """Returns a ``timedelta`` object representing the whole number of minutes difference from
        UTC time.

        Usage::

            >>> arrow.now('US/Pacific').utcoffset()
            datetime.timedelta(-1, 57600)

        """

        return self._datetime.utcoffset()

    def dst(self) -> Optional[timedelta]:
        """Returns the daylight savings time adjustment.

        Usage::

            >>> arrow.utcnow().dst()
            datetime.timedelta(0)

        """

        return self._datetime.dst()

    def timetuple(self) -> struct_time:
        """Returns a ``time.struct_time``, in the current timezone.

        Usage::

            >>> arrow.utcnow().timetuple()
            time.struct_time(tm_year=2019, tm_mon=1, tm_mday=20, tm_hour=15, tm_min=17, tm_sec=8, tm_wday=6, tm_yday=20, tm_isdst=0)

        """

        return self._datetime.timetuple()

    def utctimetuple(self) -> struct_time:
        """Returns a ``time.struct_time``, in UTC time.

        Usage::

            >>> arrow.utcnow().utctimetuple()
            time.struct_time(tm_year=2019, tm_mon=1, tm_mday=19, tm_hour=21, tm_min=41, tm_sec=7, tm_wday=5, tm_yday=19, tm_isdst=0)

        """

        return self._datetime.utctimetuple()

    def toordinal(self) -> int:
        """Returns the proleptic Gregorian ordinal of the date.

        Usage::

            >>> arrow.utcnow().toordinal()
            737078

        """

        return self._datetime.toordinal()

    def weekday(self) -> int:
        """Returns the day of the week as an integer (0-6).

        Usage::

            >>> arrow.utcnow().weekday()
            5

        """

        return self._datetime.weekday()

    def isoweekday(self) -> int:
        """Returns the ISO day of the week as an integer (1-7).

        Usage::

            >>> arrow.utcnow().isoweekday()
            6

        """

        return self._datetime.isoweekday()

    def isocalendar(self) -> Tuple[int, int, int]:
        """Returns a 3-tuple, (ISO year, ISO week number, ISO weekday).

        Usage::

            >>> arrow.utcnow().isocalendar()
            (2019, 3, 6)

        """

        return self._datetime.isocalendar()

    def isoformat(self, sep: str = "T", timespec: str = "auto") -> str:
        """Returns an ISO 8601 formatted representation of the date and time.

        Usage::

            >>> arrow.utcnow().isoformat()
            '2019-01-19T18:30:52.442118+00:00'

        """

        return self._datetime.isoformat(sep, timespec)

    def ctime(self) -> str:
        """Returns a ctime formatted representation of the date and time.

        Usage::

            >>> arrow.utcnow().ctime()
            'Sat Jan 19 18:26:50 2019'

        """

        return self._datetime.ctime()

    def strftime(self, format: str) -> str:
        """Formats in the style of ``datetime.strftime``.

        :param format: the format string.

        Usage::

            >>> arrow.utcnow().strftime('%d-%m-%Y %H:%M:%S')
            '23-01-2019 12:28:17'

        """

        return self._datetime.strftime(format)

    def for_json(self) -> str:
        """Serializes for the ``for_json`` protocol of simplejson.

        Usage::

            >>> arrow.utcnow().for_json()
            '2019-01-19T18:25:36.760079+00:00'

        """

        return self.isoformat()

    # math

    def __add__(self, other: Any) -> "Arrow":
        if isinstance(other, (timedelta, relativedelta)):
            return self.fromdatetime(self._datetime + other, self._datetime.tzinfo)

        return NotImplemented

    def __radd__(self, other: Union[timedelta, relativedelta]) -> "Arrow":
        return self.__add__(other)

    @overload
    def __sub__(self, other: Union[timedelta, relativedelta]) -> "Arrow":
        pass  # pragma: no cover

    @overload
    def __sub__(self, other: Union[dt_datetime, "Arrow"]) -> timedelta:
        pass  # pragma: no cover

    def __sub__(self, other: Any) -> Union[timedelta, "Arrow"]:
        if isinstance(other, (timedelta, relativedelta)):
            return self.fromdatetime(self._datetime - other, self._datetime.tzinfo)

        elif isinstance(other, dt_datetime):
            return self._datetime - other

        elif isinstance(other, Arrow):
            return self._datetime - other._datetime

        return NotImplemented

    def __rsub__(self, other: Any) -> timedelta:
        if isinstance(other, dt_datetime):
            return other - self._datetime

        return NotImplemented

    # comparisons

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, (Arrow, dt_datetime)):
            return False

        return self._datetime == self._get_datetime(other)

    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, (Arrow, dt_datetime)):
            return True

        return not self.__eq__(other)

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, (Arrow, dt_datetime)):
            return NotImplemented

        return self._datetime > self._get_datetime(other)

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, (Arrow, dt_datetime)):
            return NotImplemented

        return self._datetime >= self._get_datetime(other)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, (Arrow, dt_datetime)):
            return NotImplemented

        return self._datetime < self._get_datetime(other)

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, (Arrow, dt_datetime)):
            return NotImplemented

        return self._datetime <= self._get_datetime(other)

    # internal methods
    @staticmethod
    def _get_tzinfo(tz_expr: Optional[TZ_EXPR]) -> dt_tzinfo:
        """Get normalized tzinfo object from various inputs."""
        if tz_expr is None:
            return timezone.utc
        if isinstance(tz_expr, dt_tzinfo):
            return tz_expr
        else:
            try:
                return parser.TzinfoParser.parse(tz_expr)
            except parser.ParserError:
                raise ValueError(f"{tz_expr!r} not recognized as a timezone.")

    @classmethod
    def _get_datetime(
        cls, expr: Union["Arrow", dt_datetime, int, float, str]
    ) -> dt_datetime:
        """Get datetime object from a specified expression."""
        if isinstance(expr, Arrow):
            return expr.datetime
        elif isinstance(expr, dt_datetime):
            return expr
        elif util.is_timestamp(expr):
            timestamp = float(expr)
            return cls.utcfromtimestamp(timestamp).datetime
        else:
            raise ValueError(f"{expr!r} not recognized as a datetime or timestamp.")

    @classmethod
    def _get_frames(cls, name: _T_FRAMES) -> Tuple[str, str, int]:
        """Finds relevant timeframe and steps for use in range and span methods.

        Returns a 3 element tuple in the form (frame, plural frame, step), for example ("day", "days", 1)

        """
        if name in cls._ATTRS:
            return name, f"{name}s", 1
        elif name[-1] == "s" and name[:-1] in cls._ATTRS:
            return name[:-1], name, 1
        elif name in ["week", "weeks"]:
            return "week", "weeks", 1
        elif name in ["quarter", "quarters"]:
            return "quarter", "months", 3
        else:
            supported = ", ".join(
                [
                    "year(s)",
                    "month(s)",
                    "day(s)",
                    "hour(s)",
                    "minute(s)",
                    "second(s)",
                    "microsecond(s)",
                    "week(s)",
                    "quarter(s)",
                ]
            )
            raise ValueError(
                f"Range or span over frame {name} not supported. Supported frames: {supported}."
            )

    @classmethod
    def _get_iteration_params(cls, end: Any, limit: Optional[int]) -> Tuple[Any, int]:
        """Sets default end and limit values for range method."""
        if end is None:
            if limit is None:
                raise ValueError("One of 'end' or 'limit' is required.")

            return cls.max, limit

        else:
            if limit is None:
                return end, sys.maxsize
            return end, limit

    @staticmethod
    def _is_last_day_of_month(date: "Arrow") -> bool:
        """Returns a boolean indicating whether the datetime is the last day of the month."""
        return cast(int, date.day) == calendar.monthrange(date.year, date.month)[1]


Arrow.min = Arrow.fromdatetime(dt_datetime.min)
Arrow.max = Arrow.fromdatetime(dt_datetime.max)
