"""Round-trip audit of Arrow's humanized relative-time field for two locales.

The event "age" field is rendered by ``Arrow.humanize`` as a localized relative
phrase and, for a subset of locales, turned back into a timestamp by
``Arrow.dehumanize`` relative to the same collection instant.

Two locales that Arrow actually ships are compared:

* ``en-us`` -- present in ``DEHUMANIZE_LOCALES`` (reverse parsing supported).
* ``ko``    -- has a full ``humanize`` translation but is intentionally absent
               from ``DEHUMANIZE_LOCALES`` (reverse parsing unsupported).

For every sampled event the phrase is generated from a known instant and then
interpreted again relative to the fixed collection moment. The audit separates
two fundamentally different failure modes:

* precision loss -- the locale *can* reverse parse, but the phrase only pins
  down a time window, so distinct original business days collapse onto the
  same phrase (and therefore onto the same reconstructed instant);
* unsupported reverse parsing -- ``dehumanize`` refuses the locale outright,
  so no event can ever be reconstructed, regardless of phrase granularity.

Run it directly: ``.venv/bin/python humanize_roundtrip_audit.py``.
"""

from collections import defaultdict
from typing import Dict, List, NamedTuple, Set, Tuple

import arrow
from arrow.constants import DEHUMANIZE_LOCALES

# Fixed collection instant ("now" for every event), anchored near a midnight so
# that the 10-second "just now" window straddles two business days.
COLLECTION = arrow.Arrow(2026, 11, 13, 0, 0, 5, tzinfo="UTC")

# (label, locale name, reverse-parsing support expected by the audit)
LOCALES: Tuple[Tuple[str, str, bool], ...] = (
    ("English (US)", "en-us", True),
    ("Korean", "ko", False),
)

# Known event instants expressed as offsets from the collection instant.
# Offsets deliberately include the sub-10-second "now" band straddling
# midnight, minute/hour granularities, and day-scale events on different
# business days.
OFFSETS: Tuple[Tuple[str, int], ...] = (
    ("8s in the past, previous business day", -8),
    ("3s in the future, current business day", 3),
    ("30s in the past, previous business day", -30),
    ("45m in the past, previous business day", -45 * 60),
    ("5h in the past, previous business day", -5 * 3600),
    ("1d in the past, previous business day", -86400),
    ("1.5d in the past, two business days back", -129600),
    ("2d in the past, two business days back", -2 * 86400),
    ("1d in the future, next business day", 86400),
    ("1.5d in the future, next business day", 129600),
)


def _business_day(event: arrow.Arrow) -> str:
    return event.strftime("%Y-%m-%d")


class EventRow(NamedTuple):
    label: str
    event: arrow.Arrow
    day: str
    phrase: str
    detail: str


class AuditResult(NamedTuple):
    locale: str
    supported: bool
    sampled: int
    distinct_phrases: int
    collisions: Dict[str, List[str]]
    precision_loss_events: int
    unsupported_events: int


def audit_locale(
    label: str, locale: str, reverse_supported: bool
) -> AuditResult:
    print("=" * 78)
    print(f"Locale: {label} ({locale!r})")
    print(f"Collection instant: {COLLECTION.isoformat()}")
    supported = locale in DEHUMANIZE_LOCALES
    print(f"dehumanize supported by Arrow: {supported}")
    assert supported is reverse_supported, (
        f"support expectation mismatch for {locale}"
    )
    print("-" * 78)

    rows: List[EventRow] = []
    phrase_to_days: Dict[str, Set[str]] = defaultdict(set)
    phrase_to_events: Dict[str, List[str]] = defaultdict(list)
    precision_loss_events = 0
    unsupported_events = 0

    for event_label, offset in OFFSETS:
        event = COLLECTION.shift(seconds=offset)
        phrase = event.humanize(COLLECTION, locale=locale)
        business_day = _business_day(event)
        phrase_to_days[phrase].add(business_day)
        phrase_to_events[phrase].append(event_label)

        if supported:
            reconstructed = COLLECTION.dehumanize(phrase, locale=locale)
            exact = reconstructed == event
            same_day = _business_day(reconstructed) == business_day
            outcome = (
                "exact"
                if exact
                else "same business day"
                if same_day
                else "WRONG DAY"
            )
            if not exact:
                precision_loss_events += 1
            detail = f"-> {reconstructed.isoformat()} [{outcome}]"
        else:
            try:
                COLLECTION.dehumanize(phrase, locale=locale)
            except ValueError as exc:
                unsupported_events += 1
                detail = f"-> ValueError: {str(exc).split('.')[0]}"
            else:  # pragma: no cover - defensive, contradicts the whitelist
                detail = "-> unexpectedly reconstructed"

        rows.append(
            EventRow(
                label=event_label,
                event=event,
                day=business_day,
                phrase=phrase,
                detail=detail,
            )
        )

    for row in rows:
        print(
            f"{row.event.isoformat()} (day {row.day})\n"
            f"    phrase={row.phrase!r}\n"
            f"    {row.label}\n"
            f"    reverse: {row.detail}"
        )

    print("-" * 78)
    print("Distinct business days sharing an identical phrase:")
    collisions = 0
    for phrase, days in sorted(
        phrase_to_days.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        tag = "COLLISION" if len(days) > 1 else "single day"
        print(f"  [{tag}] {phrase!r} -> days {sorted(days)}")
        if len(days) > 1:
            collisions += 1
            for source in phrase_to_events[phrase]:
                print(f"        source event: {source}")

    print("-" * 78)
    print("Tally (counts kept separate by failure mode):")
    print(f"  sampled events:                   {len(OFFSETS)}")
    print(f"  distinct phrases produced:        {len(phrase_to_days)}")
    print(f"  phrases spanning >1 business day: {collisions}")
    if supported:
        print(
            "  precision-loss events (reverse parsed, not exact): "
            f"{precision_loss_events}"
        )
        print("  unsupported-reverse events:                        0")
    else:
        print("  precision-loss events (reverse parsed, not exact): 0 (not counted;")
        print("        parsing never happens, so precision cannot be assessed)")
        print(
            "  unsupported-reverse events:                        "
            f"{unsupported_events}"
        )
    print()

    return AuditResult(
        locale=locale,
        supported=supported,
        sampled=len(OFFSETS),
        distinct_phrases=len(phrase_to_days),
        collisions={
            phrase: sorted(days)
            for phrase, days in phrase_to_days.items()
            if len(days) > 1
        },
        precision_loss_events=precision_loss_events,
        unsupported_events=unsupported_events,
    )


def main() -> None:
    results = {
        locale: audit_locale(label, locale, reverse_supported)
        for label, locale, reverse_supported in LOCALES
    }

    en = results["en-us"]
    ko = results["ko"]

    # English: reverse parsing works, but at least one phrase is shared by
    # events on different business days, so it cannot uniquely locate a day.
    assert en.supported is True
    assert en.unsupported_events == 0
    assert en.precision_loss_events > 0
    assert en.collisions, "expected at least one cross-day phrase collision"
    assert en.collisions["just now"] == ["2026-11-12", "2026-11-13"]

    # Korean: every sampled event is blocked at the locale gate; precision loss
    # is deliberately not counted because no phrase is ever reverse parsed.
    assert ko.supported is False
    assert ko.unsupported_events == ko.sampled
    assert ko.precision_loss_events == 0
    assert ko.collisions, "Korean day-level phrases also collapse across days"

    print("Summary: en-us phrases lose precision (a time window, not a unique day);")
    print("ko reverse parsing is unsupported entirely, counted separately.")


if __name__ == "__main__":
    main()
