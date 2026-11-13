from datetime import datetime, timezone

import pytest

import arrow
from arrow import humanize_schedule
from arrow.humanize_schedule import (
    UnsafeBoundaryError,
    locale_safety_report,
    next_humanize_refresh,
    iter_humanize_refresh,
)


def _plus(instant, seconds):
    return humanize_schedule._at_timestamp(instant, instant.int_timestamp + seconds)


def _render(event, instant, **kwargs):
    try:
        return event.humanize(instant, **kwargs)
    except ValueError:
        return None


def _brute_first_change(event, reference, window_seconds, **kwargs):
    """Reference ground truth: an actual per-second walk over a short window."""
    base = _render(event, reference, **kwargs)
    cursor = reference
    end = _plus(reference, window_seconds)
    while cursor.int_timestamp < end.int_timestamp:
        cursor = _plus(cursor, 1)
        if _render(event, cursor, **kwargs) != base:
            return cursor
    return None


@pytest.mark.parametrize(
    "granularity",
    [
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "quarter",
        "year",
        ["hour", "minute"],
        ["year", "month", "day"],
    ],
)
@pytest.mark.parametrize("only_distance", [False, True])
def test_fixed_granularity_matches_brute_force(granularity, only_distance):
    event = arrow.get(2026, 5, 17, 9, 12, 30, tzinfo="UTC")
    for offset in (-9000, -3600, -61, 61, 3600, 9000):
        reference = _plus(event, offset)
        horizon = _plus(reference, 3 * 3600)
        kwargs = dict(
            locale="en-us", granularity=granularity, only_distance=only_distance
        )
        assert next_humanize_refresh(
            event, reference, horizon=horizon, **kwargs
        ) == _brute_first_change(event, reference, 3 * 3600, **kwargs)


def test_auto_ladder_thresholds():
    event = arrow.get(2026, 4, 1, 12, 0, 0, tzinfo="UTC")
    # The "just now" window ends exactly at a 10-second distance.
    reference = event
    refresh = next_humanize_refresh(
        event, reference, horizon=_plus(reference, 86400)
    )
    assert refresh == _plus(event, 10)
    assert event.humanize(_plus(event, 9)) == "just now"
    assert event.humanize(_plus(event, 10)) != "just now"


def test_future_to_past_sign_flip():
    event = arrow.get(2026, 6, 15, 12, 0, 0, tzinfo="UTC")
    reference = _plus(event, -2)
    chain = iter_humanize_refresh(
        event,
        reference,
        horizon=_plus(event, 2),
        granularity="second",
    )
    # Boundaries sit on the two instants either side of the event itself.
    assert chain == [_plus(event, -1), _plus(event, 2)]
    # The second boundary is the first render on the past side of the event.
    assert event.humanize(_plus(event, 2), granularity="second").endswith("ago")
    assert event.humanize(_plus(event, -2), granularity="second").startswith("in")


def test_cross_month_auto_schedule():
    # Jan 31 -> the auto month/day branches must track calendar month rollover.
    event = arrow.get(2026, 1, 31, 12, 0, 0, tzinfo="UTC")
    reference = event.shift(days=-10)
    horizon = event.shift(months=3)
    chain = iter_humanize_refresh(event, reference, horizon=horizon)

    cursor = reference
    previous = event.humanize(reference)
    for boundary in chain:
        # Rendering stays constant between two adjacent boundaries.
        midpoint = _plus(cursor, (boundary.int_timestamp - cursor.int_timestamp) // 2)
        assert event.humanize(midpoint) == previous
        assert event.humanize(_plus(boundary, -1)) == previous
        assert event.humanize(boundary) != previous
        cursor = boundary
        previous = event.humanize(boundary)
    assert event.humanize(horizon) == previous


@pytest.mark.parametrize(
    "zone",
    [
        "America/New_York",
        "Europe/Paris",
        "Australia/Lord_Howe",
        "Asia/Kathmandu",
    ],
)
def test_dst_transitions_are_exhaustive(zone):
    # Event in spring, windows straddling both the spring-forward and
    # fall-back offset switches.
    event = arrow.get(2026, 3, 31, 18, 30, 0, tzinfo=zone)
    windows = (
        (
            arrow.get(2026, 3, 7, 12, tzinfo=zone),
            arrow.get(2026, 3, 9, 12, tzinfo=zone),
        ),
        (
            arrow.get(2026, 10, 31, 12, tzinfo=zone),
            arrow.get(2026, 11, 2, 12, tzinfo=zone),
        ),
    )
    for start, end in windows:
        cursor = start
        previous = event.humanize(cursor, granularity="hour")
        while cursor < end:
            boundary = next_humanize_refresh(
                event, cursor, horizon=end, granularity="hour"
            )
            if boundary is None:
                probe = cursor
                while probe < end:
                    probe = _plus(probe, 1)
                    assert event.humanize(probe, granularity="hour") == previous
                break
            probe = cursor
            while probe < boundary:
                probe = _plus(probe, 1)
                if probe < boundary:
                    assert event.humanize(probe, granularity="hour") == previous
            assert event.humanize(boundary, granularity="hour") != previous
            cursor = boundary
            previous = event.humanize(cursor, granularity="hour")


def test_distant_event_uses_logarithmic_not_linear_work():
    event = arrow.get(2027, 1, 15, 12, 0, 0, tzinfo="UTC")
    reference = arrow.get(2026, 1, 15, 12, 0, 0, tzinfo="UTC")  # one year away

    calls = 0
    original = humanize_schedule._render

    def counting_render(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    humanize_schedule._render = counting_render
    try:
        refresh = next_humanize_refresh(
            event, reference, granularity="day"
        )
    finally:
        humanize_schedule._render = original

    # A year is ~365 days; a per-second scan would need millions of renders.
    assert calls < 60
    assert refresh == _plus(reference, 1)


def test_returns_none_when_text_is_stable():
    event = arrow.get(2026, 1, 1, tzinfo="UTC")
    # Six months away truncates to 0 whole years; a one-hour horizon cannot
    # move it into a different year bucket.
    reference = event.shift(months=-6)
    assert event.humanize(reference, granularity="year") == "in 0 years"
    assert (
        next_humanize_refresh(
            event,
            reference,
            horizon=reference.shift(hours=1),
            granularity="year",
        )
        is None
    )


def test_locale_specific_plurals_match_renderer():
    event = arrow.get(2026, 2, 10, 8, 0, 0, tzinfo="UTC")
    for locale in ("ru", "pl", "ar", "he", "ko", "th", "cs", "de"):
        reference = _plus(event, -5 * 3600)
        horizon = _plus(reference, 6 * 3600)
        kwargs = dict(locale=locale, granularity="hour")
        assert next_humanize_refresh(
            event, reference, horizon=horizon, **kwargs
        ) == _brute_first_change(event, reference, 6 * 3600, **kwargs)


def test_unsupported_granularity_raises_value_error():
    event = arrow.get(2026, 1, 1, tzinfo="UTC")
    with pytest.raises(ValueError):
        next_humanize_refresh(event, granularity="nonsense")
    with pytest.raises(ValueError):
        next_humanize_refresh(event, granularity=[])


def test_missing_locale_template_is_reported_unsafe():
    report = {
        item.granularity: item for item in locale_safety_report("de-de")
    }
    assert report["quarter"].safe is False
    assert "quarter" in report["quarter"].reason

    event = arrow.get(2026, 1, 1, tzinfo="UTC")
    with pytest.raises(ValueError):
        next_humanize_refresh(event, granularity="quarter", locale="de-de")


def test_every_bundled_hook_owner_is_audited():
    from arrow import locales

    for name, locale_cls in locales._locale_map.items():
        owners = humanize_schedule._hook_owners(locale_cls())
        assert owners <= humanize_schedule._AUDITED_HOOK_OWNERS, name


def test_icelandic_zero_delta_is_transient():
    event = arrow.get(2026, 6, 1, tzinfo="UTC")
    # Inside the trunc-to-zero day region humanize raises, but it recovers.
    reference = event
    refresh = next_humanize_refresh(
        event,
        reference,
        horizon=_plus(event, 2 * 86400),
        locale="is",
        granularity="day",
    )
    assert refresh is not None
    assert event.humanize(refresh, locale="is", granularity="day") is not None

    report = {
        item.granularity: item
        for item in locale_safety_report("is")
    }
    assert report["second"].safe is True
    assert "delta == 0" in report["second"].reason


def test_naive_datetime_reference_is_assumed_event_zone():
    event = arrow.get(2026, 6, 1, 12, 0, 0, tzinfo="UTC")
    naive = datetime(2026, 6, 1, 11, 59, 55)
    refresh = next_humanize_refresh(
        event, naive, granularity="second"
    )
    assert refresh == arrow.get(2026, 6, 1, 11, 59, 56, tzinfo="UTC")


def test_invalid_reference_type():
    event = arrow.get(2026, 1, 1, tzinfo="UTC")
    with pytest.raises(TypeError):
        next_humanize_refresh(event, "not a date")
