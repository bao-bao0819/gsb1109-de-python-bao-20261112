"""Analytic scheduling of :meth:`Arrow.humanize` re-renders.

A board row caches the localized string produced by :meth:`Arrow.humanize`.
Instead of polling every second (which cannot scale to events months away) or
re-rendering on a fixed timer (which wastes work when the text is unchanged),
this module computes the *next absolute instant* at which the cached string can
possibly change for a fixed event and a reference time moving forward.

Why a render-driven search is safe
-----------------------------------

``humanize`` rounds the difference to whole seconds and builds its string from
that rounded delta, its sign and, in ``granularity="auto"`` mode, calendar
components. Every bundled locale renderer is a closed-form function of the
timeframe name, the sign and the truncated integer count, so inside a time
range where (a) the UTC offset is constant and (b) the sign of the delta is
constant, moving the reference forward only increases ``abs(delta)`` and the
rendered string never repeats: the predicate "the render differs from the
segment's first render" is therefore monotonic and binary-searchable.

The two things that break that monotonicity are sign flips (the reference
crossing the event) and UTC-offset switches (DST). Both are known analytically:
the event is given, and offset transitions are located with an exponential then
binary search. They split the timeline into monotonic segments. Each segment is
then searched with a binary search over whole seconds, so the cost is
``O(log horizon + transitions * log segment)`` render calls -- never a
per-second walk from now to a month away.

Locales with rendering logic outside the audited closed-form set, or with
missing timeframe templates, cannot be given a static guarantee and are reported
with the concrete reason by :func:`locale_safety_report`.
"""

from dataclasses import dataclass
from datetime import datetime as dt_datetime
from datetime import timedelta, timezone
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from arrow.constants import DEFAULT_LOCALE
from arrow.locales import get_locale

if TYPE_CHECKING:
    from arrow.arrow import Arrow

_TimeLike = Union["Arrow", dt_datetime, None]
_Granularity = str
_GRANULARITY_FRAMES: Tuple[str, ...] = (
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "quarter",
    "year",
)
# Auto mode never emits a quarter description.
_AUTO_FRAMES: Tuple[str, ...] = (
    "now",
    "second",
    "seconds",
    "minute",
    "minutes",
    "hour",
    "hours",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
)
_RENDER_HOOKS: Tuple[str, ...] = (
    "describe",
    "describe_multi",
    "_format_timeframe",
    "_format_relative",
)

# Classes that define the humanize rendering hooks shipped with arrow. Each
# definition was audited and depends solely on the timeframe, the sign of
# ``delta`` and its truncated absolute value (Korean and Thai additionally
# branch on the signed delta, which is itself determined by the absolute
# difference). A hook defined by any other class cannot be given a static
# guarantee. Subclasses are resolved through the MRO, so e.g. PolishLocale is
# covered because its hook is defined on SlavicBaseLocale.
_AUDITED_HOOK_OWNERS: frozenset = frozenset(
    {
        "Locale",
        "EnglishLocale",
        "FinnishLocale",
        "KoreanLocale",
        "SlavicBaseLocale",
        "GermanBaseLocale",
        "ArabicLocale",
        "IcelandicLocale",
        "CzechLocale",
        "SlovakLocale",
        "HebrewLocale",
        "HungarianLocale",
        "ThaiLocale",
        "LaotianLocale",
        "EstonianLocale",
        "CroatianLocale",
        "MalteseLocale",
        "SerbianLocale",
        "LuxembourgishLocale",
        "ZuluLocale",
        "SinhalaLocale",
        "AmharicLocale",
    }
)

# Locales whose renderer raises for a zero delta in some timeframe, so the
# instant the reference crosses the event itself is not renderable upstream.
_ZERO_DELTA_RAISE_OWNERS: frozenset = frozenset({"IcelandicLocale"})


class UnsafeBoundaryError(RuntimeError):
    """Raised when no safe re-render instant can be guaranteed.

    The message always states the concrete reason, e.g. a locale hook that is
    not statically audited or a candidate that failed render verification.
    """


@dataclass(frozen=True)
class LocaleSafety:
    """Whether a refresh term can be derived for one locale/granularity pair."""

    locale: str
    locale_class: str
    granularity: str
    only_distance: bool
    safe: bool
    reason: str


def _coerce(value: _TimeLike, event_tz: object) -> "Arrow":
    from arrow.arrow import Arrow

    if value is None:
        utc = dt_datetime.now(timezone.utc).replace(tzinfo=timezone.utc)
        return Arrow.fromdatetime(utc.astimezone(event_tz))  # type: ignore[arg-type]
    if isinstance(value, Arrow):
        return value
    if isinstance(value, dt_datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=event_tz)  # type: ignore[arg-type]
        return Arrow.fromdatetime(value)
    raise TypeError(
        f"Invalid argument of type {type(value).__name__!r}; "
        "expected None, Arrow or datetime."
    )


def _normalize(
    event: _TimeLike, reference: _TimeLike, horizon: _TimeLike
) -> Tuple["Arrow", "Arrow", "Arrow"]:
    """Coerces inputs and puts reference/horizon into the event's zone.

    ``humanize`` re-zones the reference into the event's tzinfo before it
    subtracts, so every scheduling computation happens in that same zone.
    """
    event_arrow = _coerce(event, timezone.utc)
    reference_arrow = _coerce(reference, event_arrow.tzinfo).to(event_arrow.tzinfo)
    horizon_arrow = (
        _coerce(horizon, event_arrow.tzinfo).to(event_arrow.tzinfo)
        if horizon is not None
        else reference_arrow.shift(years=2)
    )
    return event_arrow, reference_arrow, horizon_arrow


def _render(
    event: "Arrow",
    reference: "Arrow",
    locale: str,
    granularity: Union[str, Sequence[str]],
    only_distance: bool,
) -> Optional[str]:
    """Renders the row, returning ``None`` if the locale itself raises."""
    try:
        return event.humanize(
            reference,
            locale=locale,
            granularity=granularity,  # type: ignore[arg-type]
            only_distance=only_distance,
        )
    except ValueError:
        # Missing translations (and the Icelandic zero-delta bug) surface as
        # ValueError. Treat an exception as a distinct, non-matching output.
        return None


def _at_timestamp(event: "Arrow", timestamp: int) -> "Arrow":
    """Builds an event-zoned Arrow at an absolute whole-second timestamp."""
    from arrow.arrow import Arrow

    utc_instant = dt_datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return Arrow.fromdatetime(utc_instant).to(event.tzinfo)


def _hook_owners(locale_obj: object) -> Set[str]:
    """Names of the classes defining the render hooks for this locale (MRO)."""
    owners: Set[str] = set()
    for hook in _RENDER_HOOKS:
        for cls in type(locale_obj).mro():
            if hook in cls.__dict__:
                owners.add(cls.__name__)
                break
    return owners


def _assert_audited_locale(locale_name: str) -> None:
    locale_obj = get_locale(locale_name)
    unaudited = _hook_owners(locale_obj) - _AUDITED_HOOK_OWNERS
    if unaudited:
        raise UnsafeBoundaryError(
            f"Locale {locale_name!r} ({type(locale_obj).__name__}) defines "
            f"humanize hook(s) {sorted(unaudited)!r} outside the audited "
            "closed-form set, so no safe re-render term can be derived."
        )


def _first_renderable(
    event: "Arrow",
    reference: "Arrow",
    horizon: "Arrow",
    locale: str,
    granularity: Union[str, Sequence[str]],
    only_distance: bool,
) -> Optional["Arrow"]:
    """First instant at/after the reference whose render does not raise.

    Used when the current render itself errors (e.g. the Icelandic zero-delta
    defect, which only affects a narrow delta range). Returns ``None`` if the
    pair is unrenderable across the whole horizon (a missing translation).
    """
    lo = reference.int_timestamp
    hi = horizon.int_timestamp
    end_render = _render(
        event, _at_timestamp(event, hi), locale, granularity, only_distance
    )
    if end_render is None:
        return None
    while lo + 1 < hi:
        middle = (lo + hi) // 2
        if _render(
            event, _at_timestamp(event, middle), locale, granularity, only_distance
        ) is None:
            lo = middle
        else:
            hi = middle
    return _at_timestamp(event, hi)


def _offset_transitions(
    tzinfo: object,
    reference: "Arrow",
    horizon: "Arrow",
) -> List["Arrow"]:
    """All UTC-offset change instants within the search window.

    Fixed offsets (``datetime.timezone``, dateutil ``tzutc``/``tzoffset``) have
    none. Named zones are scanned with an exponential then binary probe, so a
    transition months away costs O(log horizon) offset evaluations rather than
    a per-second walk. Only ``utcoffset`` matters: abbreviation changes with no
    offset change cannot affect ``humanize``.
    """
    if isinstance(tzinfo, timezone):
        return []

    transitions: List["Arrow"] = []
    start = reference.to("UTC")
    limit = horizon.to("UTC")
    cursor = start
    step = timedelta(hours=1)
    maximum_step = timedelta(days=30)
    while cursor < limit:
        baseline = cursor.to(tzinfo).utcoffset()  # type: ignore[arg-type]
        probe = min(cursor + step, limit)
        if probe.to(tzinfo).utcoffset() == baseline:  # type: ignore[arg-type]
            if probe >= limit:
                break
            cursor = probe
            step = min(step * 2, maximum_step)
            continue

        # Narrow the bracketed change to the exact whole second.
        low, high = cursor, probe
        while high - low > timedelta(seconds=1):
            middle = low + (high - low) / 2
            if middle.to(tzinfo).utcoffset() == baseline:  # type: ignore[arg-type]
                low = middle
            else:
                high = middle
        transition = high.floor("second").to(reference.tzinfo)
        if transition.int_timestamp > reference.int_timestamp:
            transitions.append(transition)
        cursor = probe
        step = timedelta(hours=1)
    return transitions


def _first_change_in_segment(
    event: "Arrow",
    segment_start: "Arrow",
    segment_end: "Arrow",
    locale: str,
    granularity: Union[str, Sequence[str]],
    only_distance: bool,
) -> Optional["Arrow"]:
    """Earliest instant inside one monotonic segment whose render changes.

    The segment has constant UTC offset and constant event/reference sign, so
    the rendered string never repeats as the reference advances. A binary
    search over whole seconds therefore finds the first differing render in
    ``O(log segment_length)`` calls.
    """
    start_text = _render(event, segment_start, locale, granularity, only_distance)
    end_text = _render(event, segment_end, locale, granularity, only_distance)
    if end_text == start_text:
        return None

    lo = segment_start.int_timestamp
    hi = segment_end.int_timestamp
    while lo + 1 < hi:
        middle = (lo + hi) // 2
        middle_text = _render(
            event,
            _at_timestamp(event, middle),
            locale,
            granularity,
            only_distance,
        )
        if middle_text == start_text:
            lo = middle
        else:
            hi = middle
    return _at_timestamp(event, hi)


def _validate_granularity(
    granularity: Union[str, Sequence[str]],
) -> Union[str, Tuple[str, ...]]:
    if isinstance(granularity, list) and len(granularity) == 1:
        granularity = granularity[0]
    if isinstance(granularity, list):
        if not granularity:
            raise ValueError(
                "Empty granularity list provided. Please select one or more "
                "from 'second', 'minute', 'hour', 'day', 'week', 'month', "
                "'quarter', 'year'."
            )
        invalid = [frame for frame in granularity if frame not in _GRANULARITY_FRAMES]
        if invalid:
            raise ValueError(f"Invalid granularity frame(s): {invalid!r}.")
        return tuple(granularity)
    if granularity == "auto" or granularity in _GRANULARITY_FRAMES:
        return granularity
    raise ValueError(
        "Invalid level of granularity. Please select between 'second', "
        "'minute', 'hour', 'day', 'week', 'month', 'quarter' or 'year'."
    )


def next_humanize_refresh(
    event: _TimeLike,
    reference: _TimeLike = None,
    *,
    locale: str = DEFAULT_LOCALE,
    granularity: Union[_Granularity, Sequence[str]] = "auto",
    only_distance: bool = False,
    horizon: _TimeLike = None,
) -> Optional["Arrow"]:
    """Returns the next instant the cached ``humanize`` text must be refreshed.

    The reference represents the board clock and is assumed to move forward.
    The returned :class:`Arrow` is the first instant strictly after
    ``reference`` (and no later than ``horizon``) at which
    ``event.humanize(reference, ...)`` yields a different string. ``None`` is
    returned when the text is stable for the whole horizon.

    :param event: the fixed event being humanized.
    :param reference: the board clock. Defaults to now in the event's zone,
        matching ``humanize(other=None)``.
    :param locale: locale name, identical to ``humanize``.
    :param granularity: ``"auto"``, a single frame, or a list of frames.
    :param only_distance: mirrors the ``humanize`` flag; some locales use a
        separate template table when it is set.
    :param horizon: how far ahead to search. Defaults to two years.
    :raises UnsafeBoundaryError: if the locale logic cannot be guaranteed.
    :raises ValueError: if the granularity is invalid or untranslated.
    """
    event_arrow, reference_arrow, horizon_arrow = _normalize(
        event, reference, horizon
    )
    granularity = _validate_granularity(granularity)
    _assert_audited_locale(locale)

    reference_ts = reference_arrow.int_timestamp
    horizon_ts = horizon_arrow.int_timestamp
    if horizon_ts <= reference_ts:
        return None

    # A None render at the reference is a transient locale defect (e.g.
    # Icelandic at a zero delta). The first instant that renders again is itself
    # the next visible change; if nothing renders, the pair is unsupported.
    base_text = _render(
        event_arrow, reference_arrow, locale, granularity, only_distance
    )
    if base_text is None:
        first_renderable = _first_renderable(
            event_arrow,
            reference_arrow,
            horizon_arrow,
            locale,
            granularity,
            only_distance,
        )
        if first_renderable is None:
            raise ValueError(
                f"Humanization with granularity {granularity!r} does not render "
                f"for locale {locale!r}; see locale_safety_report() for the reason."
            )
        return first_renderable

    # Sign flips and offset switches are the only non-monotonic boundaries.
    break_ts = {horizon_ts}
    if reference_ts < event_arrow.int_timestamp < horizon_ts:
        break_ts.add(event_arrow.int_timestamp)
    for transition in _offset_transitions(
        event_arrow.tzinfo, reference_arrow, horizon_arrow
    ):
        ts = transition.int_timestamp
        if reference_ts < ts < horizon_ts:
            break_ts.add(ts)

    previous = reference_arrow
    for boundary_ts in sorted(break_ts):
        segment_end = _at_timestamp(event_arrow, boundary_ts)
        start_text = _render(
            event_arrow, previous, locale, granularity, only_distance
        )
        end_text = _render(
            event_arrow, segment_end, locale, granularity, only_distance
        )
        if start_text != base_text:
            # A boundary between two segments already changed the text.
            return previous
        if end_text != base_text:
            found = _first_change_in_segment(
                event_arrow,
                previous,
                segment_end,
                locale,
                granularity,
                only_distance,
            )
            if found is not None:
                return found
        previous = segment_end
    return None


def iter_humanize_refresh(
    event: _TimeLike,
    reference: _TimeLike = None,
    *,
    locale: str = DEFAULT_LOCALE,
    granularity: Union[_Granularity, Sequence[str]] = "auto",
    only_distance: bool = False,
    horizon: _TimeLike = None,
) -> List["Arrow"]:
    """Returns every refresh instant up to ``horizon`` as a sorted list.

    Each step re-derives the next boundary from the previous refresh point with
    a logarithmic search, so chains spanning months stay O(number of rendered
    text changes * log distance) rather than scanning wall-clock time.
    """
    event_arrow, reference_arrow, horizon_arrow = _normalize(
        event, reference, horizon
    )

    refreshes: List["Arrow"] = []
    cursor = reference_arrow
    while True:
        next_refresh = next_humanize_refresh(
            event_arrow,
            cursor,
            locale=locale,
            granularity=granularity,
            only_distance=only_distance,
            horizon=horizon_arrow,
        )
        if next_refresh is None:
            return refreshes
        refreshes.append(next_refresh)
        cursor = next_refresh


def locale_safety_report(
    locale: Optional[str] = None, *, only_distance: bool = False
) -> List[LocaleSafety]:
    """Audits which (locale, granularity) pairs admit a safe refresh term.

    A pair is unsafe when its timeframe templates are missing or empty (the
    renderer itself raises ``ValueError``) or when a rendering hook is defined
    outside the audited closed-form set. Known upstream defects are reported
    with their concrete cause. The ``reason`` field always states why a pair is
    or is not schedulable.
    """
    from arrow import locales

    if locale is None:
        names = sorted(locales._locale_map)
    else:
        names = [locale.lower().replace("_", "-")]

    report: List[LocaleSafety] = []
    frames = ("auto",) + _GRANULARITY_FRAMES
    for name in names:
        locale_obj = locales.get_locale(name)
        class_name = type(locale_obj).__name__
        templates = locale_obj.timeframes
        distance_templates = getattr(locale_obj, "timeframes_only_distance", None)
        unaudited = _hook_owners(locale_obj) - _AUDITED_HOOK_OWNERS
        zero_delta_owner = _hook_owners(locale_obj) & _ZERO_DELTA_RAISE_OWNERS
        for frame in frames:
            if frame == "auto":
                required = list(_AUTO_FRAMES)
            elif frame == "second":
                required = ["now", "second", "seconds"]
            else:
                required = [frame, frame + "s"]
            missing: List[str] = []
            for key in required:
                table = (
                    distance_templates
                    if only_distance and distance_templates is not None
                    else templates
                )
                value = table.get(key)
                if value is None or value == "":
                    missing.append(key)
            if missing:
                safe = False
                reason = (
                    "missing/empty timeframe template(s) "
                    f"{missing!r}; humanize raises ValueError"
                )
            elif unaudited:
                safe = False
                reason = (
                    "rendering hook(s) "
                    f"{sorted(unaudited)!r} are outside the audited closed-form set"
                )
            elif zero_delta_owner and not only_distance:
                safe = True
                reason = (
                    "schedulable, but humanize raises ValueError exactly when "
                    "the reference reaches the event (delta == 0) due to "
                    f"{sorted(zero_delta_owner)!r}"
                )
            else:
                safe = True
                reason = "renderer is a closed-form function of sign and count"
            report.append(
                LocaleSafety(
                    locale=name,
                    locale_class=class_name,
                    granularity=frame,
                    only_distance=only_distance,
                    safe=safe,
                    reason=reason,
                )
            )
    return report
