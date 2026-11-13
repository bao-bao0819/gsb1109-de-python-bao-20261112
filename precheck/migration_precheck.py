"""Migration pre-check for moving the date importer from ``datetime.strptime``
to :mod:`arrow`.

The importer currently feeds a finite, explicitly registered set of field
syntaxes to the standard-library parser.  Before switching those formats to
Arrow, every registered syntax must be compared on three layers:

1. acceptance domain -- which field strings each parser accepts or rejects;
2. converted instant -- the POSIX moment after a reference timezone is attached;
3. business day -- the civil calendar date (year, month, day).

A second concern covers batches whose formats omit the year.  ``strptime``
silently fills a missing year with its 1900 default, but some batches shipped
without recording any default.  For those batches every datetime-valid year is
swept and the certifiable range is reported instead of assuming a default.

Compatibility evidence rule
---------------------------
A field on which both parsers merely *succeed* is not, by itself, evidence of
compatibility: the instant and business-day layers are still diffed, and
both-success rows are explicitly excluded from the acceptance-evidence count.
Only shared rejections and fully equal three-layer results count as agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, TypedDict

from arrow.parser import DateTimeParser, ParserError

# datetime supports years 1..9999; this is the full certifiable sweep range.
MIN_YEAR = 1
MAX_YEAR = 9999
ALL_YEARS: Sequence[int] = range(MIN_YEAR, MAX_YEAR + 1)
REF_TZ = timezone.utc


class Rejected(Exception):
    """Raised when a parser refuses a field (any failure mode, normalized)."""


@dataclass(frozen=True)
class Case:
    """One minimal field that reproduces a single difference class.

    :param case_id: stable identifier of the difference class.
    :param layer: which comparison layer the class belongs to.
    :param field: the shortest field string that reproduces the difference.
    :param legacy_fmt: registered standard-library format directive.
    :param arrow_fmt: Arrow token string proposed for the migration.
    :param claim: expected acceptance relation, used as a test oracle.
    :param note: human-readable explanation of the divergence.
    """

    case_id: str
    layer: str
    field: str
    legacy_fmt: str
    arrow_fmt: str
    claim: str  # "arrow_only" | "legacy_only" | "both_accept_diff" | "both_reject"
    note: str


@dataclass
class Verdict:
    """Three-layer result of parsing one field with both parsers."""

    case: Case
    legacy_status: str
    arrow_status: str
    legacy_dt: Optional[datetime] = None
    arrow_dt: Optional[datetime] = None
    legacy_error: Optional[str] = None
    arrow_error: Optional[str] = None

    @property
    def relation(self) -> str:
        if self.legacy_status == "OK" and self.arrow_status == "OK":
            return "both_accept"
        if self.legacy_status == "REJ" and self.arrow_status == "REJ":
            return "both_reject"
        if self.arrow_status == "OK":
            return "arrow_only"
        return "legacy_only"

    @property
    def instant_equal(self) -> Optional[bool]:
        """None unless both parsers accepted the field."""
        if self.relation != "both_accept":
            return None
        assert self.legacy_dt is not None and self.arrow_dt is not None
        return _epoch_us(self.legacy_dt) == _epoch_us(self.arrow_dt)

    @property
    def business_day_equal(self) -> Optional[bool]:
        """None unless both parsers accepted the field."""
        if self.relation != "both_accept":
            return None
        assert self.legacy_dt is not None and self.arrow_dt is not None
        return _business_day(self.legacy_dt) == _business_day(self.arrow_dt)

    @property
    def awareness_equal(self) -> Optional[bool]:
        """Whether both results are naive or both are timezone-aware.

        A naive ``strptime`` result beside an Arrow ``ZoneInfo`` result shares a
        UTC instant only after the importer attaches its reference zone; the
        raw awareness difference must still block an automatic migration.
        """
        if self.relation != "both_accept":
            return None
        assert self.legacy_dt is not None and self.arrow_dt is not None
        return (self.legacy_dt.tzinfo is None) == (self.arrow_dt.tzinfo is None)


class Batch:
    """A registered import batch, optionally missing a recorded default year."""

    batch_id: str
    field: str
    legacy_fmt: str
    arrow_fmt: str
    default_year: Optional[int]
    # Builder used to re-express the field once an explicit year is supplied,
    # together with the Arrow/strptime formats of that completed field.
    complete_field: Optional[Callable[[int], str]]
    complete_arrow_fmt: Optional[str]
    complete_legacy_fmt: Optional[str]
    note: str

    def __init__(
        self,
        batch_id: str,
        field: str,
        legacy_fmt: str,
        arrow_fmt: str,
        default_year: Optional[int],
        complete_field: Optional[Callable[[int], str]] = None,
        complete_arrow_fmt: Optional[str] = None,
        complete_legacy_fmt: Optional[str] = None,
        note: str = "",
    ) -> None:
        self.batch_id = batch_id
        self.field = field
        self.legacy_fmt = legacy_fmt
        self.arrow_fmt = arrow_fmt
        self.default_year = default_year
        self.complete_field = complete_field
        self.complete_arrow_fmt = complete_arrow_fmt
        self.complete_legacy_fmt = complete_legacy_fmt
        self.note = note


class Certification(TypedDict, total=False):
    """Result of sweeping a year-less batch over every explicit year."""

    batch_id: str
    field: str
    default_year: Optional[int]
    note: str
    bare_legacy: str
    bare_arrow: str
    arrow_accepted_years: List[int]
    legacy_accepted_years: List[int]
    sets_agree: bool
    certifiable_years: List[int]
    certifiable_min: Optional[int]
    certifiable_max: Optional[int]
    certifiable_count: int
    legacy_only_years: List[int]
    arrow_only_years: List[int]


def _epoch_us(dt: datetime) -> int:
    """POSIX microseconds; naive datetimes are interpreted in the reference tz."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=REF_TZ)
    return int(round(dt.timestamp() * 1_000_000))


def _business_day(dt: datetime) -> Tuple[int, int, int]:
    return dt.year, dt.month, dt.day


def legacy_parse(field_text: str, fmt: str) -> datetime:
    """Standard-library parsing, normalized to :class:`Rejected`."""
    try:
        return datetime.strptime(field_text, fmt)
    except (ValueError, TypeError) as exc:
        raise Rejected(type(exc).__name__) from exc


def arrow_parse(field_text: str, fmt: str) -> datetime:
    """Arrow parsing, normalized to :class:`Rejected`.

    ``ParserMatchError`` is a ``ParserError``; out-of-range component errors
    surface as plain ``ValueError`` from the underlying datetime constructor.
    """
    try:
        return DateTimeParser().parse(field_text, fmt)
    except (ParserError, ValueError) as exc:
        raise Rejected(type(exc).__name__) from exc


def run_case(case: Case) -> Verdict:
    """Parse one minimal field on both sides and assemble the three-layer verdict."""
    try:
        legacy_dt = legacy_parse(case.field, case.legacy_fmt)
        legacy_status, legacy_error = "OK", None
    except Rejected as exc:
        legacy_dt, legacy_status, legacy_error = None, "REJ", str(exc)

    try:
        arrow_dt = arrow_parse(case.field, case.arrow_fmt)
        arrow_status, arrow_error = "OK", None
    except Rejected as exc:
        arrow_dt, arrow_status, arrow_error = None, "REJ", str(exc)

    return Verdict(
        case=case,
        legacy_status=legacy_status,
        arrow_status=arrow_status,
        legacy_dt=legacy_dt,
        arrow_dt=arrow_dt,
        legacy_error=legacy_error,
        arrow_error=arrow_error,
    )


# ---------------------------------------------------------------------------
# Registered difference classes, each backed by the shortest reproducing field.
# ---------------------------------------------------------------------------

CASES: List[Case] = [
    # -- acceptance domain: Arrow accepts what strptime rejects --------------
    Case(
        "D01", "domain", "2026-01-01 24:00", "%Y-%m-%d %H:%M", "YYYY-MM-DD HH:mm",
        "arrow_only",
        "Hour 24 is end-of-day midnight: Arrow rolls to next day 00:00; "
        "strptime %H only accepts 00..23.",
    ),
    Case(
        "D02", "domain", "2026-01-01 00:00:00.1234567",
        "%Y-%m-%d %H:%M:%S.%f", "YYYY-MM-DD HH:mm:ss.S",
        "arrow_only",
        "%f consumes at most six digits (rest is unconverted data); Arrow S "
        "takes arbitrary precision and rounds to microseconds.",
    ),
    Case(
        "D03", "domain", "2026-01-01 23:59:59.9999999",
        "%Y-%m-%d %H:%M:%S.%f", "YYYY-MM-DD HH:mm:ss.S",
        "arrow_only",
        "Seven-digit fraction rounds up in Arrow, carrying the instant and the "
        "business day onto the next calendar day; strptime rejects the field.",
    ),
    Case(
        "D04", "domain", "2026-01-01 00:00:00 CET",
        "%Y-%m-%d %H:%M:%S %Z", "YYYY-MM-DD HH:mm:ss ZZZ",
        "arrow_only",
        "strptime %Z only recognizes a fixed tzname set (UTC/GMT and local "
        "names); Arrow ZZZ resolves CET through ZoneInfo.",
    ),
    Case(
        "D05", "domain", "2026-01-01 America/New_York",
        "%Y-%m-%d %Z", "YYYY-MM-DD ZZZ",
        "arrow_only",
        "IANA zone names are outside the strptime %Z vocabulary; Arrow ZZZ "
        "accepts them and attaches a ZoneInfo object.",
    ),
    Case(
        "D06", "domain", "+02", "%z", "Z",
        "arrow_only",
        "strptime %z requires at least HHMM; Arrow Z/ZZ accept an hours-only "
        "offset and build a fixed timedelta timezone.",
    ),
    Case(
        "D07", "domain", " 2026-01-01", "%Y-%m-%d", "YYYY-MM-DD",
        "arrow_only",
        "strptime is anchored and demands a full match; Arrow searches inside "
        "the string and tolerates surrounding whitespace or text.",
    ),
    Case(
        "D08", "domain", "2026-W01", "%G-W%V", "W",
        "arrow_only",
        "ISO %G must be paired with a weekday directive in strptime; Arrow W "
        "treats a missing weekday as day 1 of the ISO week.",
    ),
    Case(
        "D09", "domain", "15th", "%d", "Do",
        "arrow_only",
        "Ordinal day suffixes have no strptime directive; Arrow Do parses "
        "'15th', '2nd', ... via the locale ordinal regex.",
    ),
    Case(
        "D10", "domain", "1767225600", "%s", "X",
        "arrow_only",
        "Unix epoch seconds are not a strptime directive on this platform; "
        "Arrow X converts the timestamp directly in UTC.",
    ),
    Case(
        "D11", "domain", "2026-01-01 00:30 PM", "%Y-%m-%d %I:%M %p",
        "YYYY-MM-DD hh:mm A",
        "arrow_only",
        "12-hour %I rejects hour 00; Arrow accepts it with a meridiem and "
        "maps midnight to 12.",
    ),
    Case(
        "D12", "domain", "2026-01-01 00:00:00,5",
        "%Y-%m-%d %H:%M:%S.%f", "YYYY-MM-DD HH:mm:ss,S",
        "arrow_only",
        "strptime sub-seconds require a '.'; Arrow treats the comma as a "
        "literal separator before the S token.",
    ),
    # -- acceptance domain: strptime accepts what Arrow rejects --------------
    Case(
        "D13", "domain", "060", "%j", "DDDD",
        "legacy_only",
        "Bare day-of-year: strptime defaults the year to 1900; Arrow refuses "
        "DDD/DDDD without an explicit YYYY component.",
    ),
    Case(
        "D14", "domain", "366", "%j", "DDDD",
        "legacy_only",
        "Bare day 366 rolls 1900 forward to 1901-01-01 in strptime; Arrow "
        "rejects day-of-year fields that carry no year at all.",
    ),
    Case(
        "D15", "domain", "0", "%w", "d",
        "legacy_only",
        "strptime %w encodes Sunday as 0; Arrow numeric weekday d is [1-7] "
        "Monday-based, so 0 is outside its grammar.",
    ),
    Case(
        "D16", "domain", "2026-01-01T00:00:00+0200",
        "%Y-%m-%dT%H:%M:%S%z", "YYYY-MM-DDTHH:mm:ssZZ",
        "legacy_only",
        "One %z directive accepts offsets with and without a colon; Arrow ZZ "
        "requires the colon shape (use token Z for the no-colon shape, or "
        "register both formats).",
    ),
    Case(
        "D17", "domain", "0001-060", "%Y-%j", "YYYY-DDDD",
        "legacy_only",
        "Date-range floor on Arrow's day-of-year token: Arrow rebuilds an "
        "unpadded '{year}-{doy}' string and calls strptime('%Y-%j') internally, "
        "which demands a four-digit year. Years 1..999 parse in stdlib but fail "
        "the Arrow DDD/DDDD path, so its effective floor is year 1000.",
    ),
    # -- both accept, but instant or business day diverge --------------------
    Case(
        "I01", "instant", "03-09", "%m-%d", "MM-DD",
        "both_accept_diff",
        "Missing year: strptime injects 1900, Arrow injects year 1. Same "
        "month/day, different instant and different business-day year.",
    ),
    Case(
        "I02", "instant", "14:30", "%H:%M", "HH:mm",
        "both_accept_diff",
        "Missing date components: strptime anchors at 1900-01-01, Arrow at "
        "0001-01-01.",
    ),
    Case(
        "I03", "instant", "Monday", "%A", "dddd",
        "both_accept_diff",
        "Weekday name alone: strptime takes the first Monday on/after its "
        "1900-01-01 anchor (1900-01-01); Arrow uses the 1970 epoch anchor "
        "(1970-01-05). Same weekday, different instant and business day.",
    ),
    Case(
        "I04", "instant", "1", "%u", "d",
        "both_accept_diff",
        "Numeric weekday alone: strptime anchors at 1900-01-01. Arrow's single "
        "d token matches [1-7] but has no component handler, so the digit is "
        "ignored and the empty parts fall back to year 0001.",
    ),
    Case(
        "I05", "instant", "2026-01-01 GMT", "%Y-%m-%d %Z", "YYYY-MM-DD ZZZ",
        "both_accept_diff",
        "strptime recognizes GMT but returns a naive datetime; Arrow attaches "
        "ZoneInfo('GMT'). Equal UTC instant, but tz-awareness differs, so the "
        "instant layer must not be treated as structurally identical.",
    ),
]


# ---------------------------------------------------------------------------
# Batches, including ones that recorded no default year.
# ---------------------------------------------------------------------------

BATCHES: List[Batch] = [
    Batch(
        "B-DEFAULT-1900", "12-31", "%m-%d", "MM-DD", default_year=1900,
        note="Recorded default 1900; migration must supply the year explicitly "
             "because Arrow has no strptime-style 1900 fallback.",
    ),
    Batch(
        "B-NOYEAR-MD", "03-09", "%m-%d", "MM-DD", default_year=None,
        complete_field=lambda y: f"{y:04d}-03-09",
        complete_arrow_fmt="YYYY-MM-DD", complete_legacy_fmt="%Y-%m-%d",
        note="Month-day batch with no default year on record.",
    ),
    Batch(
        "B-NOYEAR-LEAP", "02-29", "%m-%d", "MM-DD", default_year=None,
        complete_field=lambda y: f"{y:04d}-02-29",
        complete_arrow_fmt="YYYY-MM-DD", complete_legacy_fmt="%Y-%m-%d",
        note="Leap-day batch with no default year on record.",
    ),
    Batch(
        "B-NOYEAR-DOY", "060", "%j", "DDDD", default_year=None,
        complete_field=lambda y: f"{y:04d}-060",
        complete_arrow_fmt="YYYY-DDDD", complete_legacy_fmt="%Y-%j",
        note="Day-of-year batch with no default year; Arrow rejects the bare field.",
    ),
    Batch(
        "B-NOYEAR-DOY366", "366", "%j", "DDDD", default_year=None,
        complete_field=lambda y: f"{y:04d}-366",
        complete_arrow_fmt="YYYY-DDDD", complete_legacy_fmt="%Y-%j",
        note="Day 366 batch: valid on leap years, rolls over otherwise.",
    ),
]


def _bounds(years: List[int]) -> str:
    """Render a sorted year list as an inclusive min..max span, or 'none'."""
    if not years:
        return "none"
    return f"{years[0]}..{years[-1]}"


def certify_batch(batch: Batch) -> Certification:
    """Sweep every datetime-valid year for a year-less batch.

    The field as stored is parsed first; then each year is injected and the
    completed field is swept through *both* parsers.  The returned certifiable
    range rests on explicit years, never on an assumed default.
    """
    result: Certification = {
        "batch_id": batch.batch_id,
        "field": batch.field,
        "default_year": batch.default_year,
        "note": batch.note,
    }

    try:
        legacy_dt = legacy_parse(batch.field, batch.legacy_fmt)
        result["bare_legacy"] = f"OK {_business_day(legacy_dt)}"
    except Rejected as exc:
        result["bare_legacy"] = f"REJ {exc}"
    try:
        arrow_dt = arrow_parse(batch.field, batch.arrow_fmt)
        result["bare_arrow"] = f"OK {_business_day(arrow_dt)}"
    except Rejected as exc:
        result["bare_arrow"] = f"REJ {exc}"

    if batch.complete_field is None:
        return result

    build_field = batch.complete_field
    arrow_fmt = batch.complete_arrow_fmt
    legacy_fmt = batch.complete_legacy_fmt
    assert arrow_fmt is not None and legacy_fmt is not None

    arrow_years: List[int] = []
    legacy_years: List[int] = []
    for year in ALL_YEARS:
        completed = build_field(year)
        try:
            arrow_parse(completed, arrow_fmt)
            arrow_years.append(year)
        except Rejected:
            pass
        try:
            legacy_parse(completed, legacy_fmt)
            legacy_years.append(year)
        except Rejected:
            pass

    common = [y for y in arrow_years if y in set(legacy_years)]
    legacy_only_years = [y for y in legacy_years if y not in set(arrow_years)]
    arrow_only_years = [y for y in arrow_years if y not in set(legacy_years)]

    result.update(
        {
            "arrow_accepted_years": arrow_years,
            "legacy_accepted_years": legacy_years,
            "sets_agree": arrow_years == legacy_years,
            # A batch is certifiable only where BOTH parsers accept an explicit
            # year; parser-specific bands are reported separately, never merged.
            "certifiable_years": common,
            "certifiable_min": common[0] if common else None,
            "certifiable_max": common[-1] if common else None,
            "certifiable_count": len(common),
            "legacy_only_years": legacy_only_years,
            "arrow_only_years": arrow_only_years,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "-"
    suffix = "" if dt.tzinfo is not None else " (naive)"
    return dt.isoformat() + suffix


def collect_verdicts() -> List[Verdict]:
    return [run_case(case) for case in CASES]


def render_report(verdicts: Optional[List[Verdict]] = None) -> str:
    """Render the deterministic pre-check report written to ``report.txt``."""
    verdicts = verdicts if verdicts is not None else collect_verdicts()
    lines: List[str] = []
    lines.append("ARROW MIGRATION PRE-CHECK")
    lines.append(f"reference timezone for naive instants: {REF_TZ}")
    lines.append(f"swept year range for year-less batches: {MIN_YEAR}..{MAX_YEAR}")
    lines.append("=" * 78)

    domain = [v for v in verdicts if v.case.layer == "domain"]
    instant = [v for v in verdicts if v.case.layer == "instant"]

    lines.append("")
    lines.append("[1] ACCEPTANCE-DOMAIN DIVERGENCES")
    for verdict in domain:
        lines.append(f"  {verdict.case.case_id} field={verdict.case.field!r}")
        lines.append(
            f"      legacy {verdict.case.legacy_fmt!r} -> {verdict.legacy_status}"
            f"{(' ' + verdict.legacy_error) if verdict.legacy_error else ''}"
        )
        lines.append(
            f"      arrow  {verdict.case.arrow_fmt!r} -> {verdict.arrow_status}"
            f"{(' ' + verdict.arrow_error) if verdict.arrow_error else ''}"
        )
        lines.append(f"      relation={verdict.relation}  {verdict.case.note}")

    lines.append("")
    lines.append("[2] INSTANT / [3] BUSINESS-DAY DIVERGENCES (BOTH PARSERS ACCEPT)")
    for verdict in instant:
        lines.append(f"  {verdict.case.case_id} field={verdict.case.field!r}")
        lines.append(f"      legacy -> {_fmt_dt(verdict.legacy_dt)}")
        lines.append(f"      arrow  -> {_fmt_dt(verdict.arrow_dt)}")
        lines.append(
            f"      instant_equal={verdict.instant_equal} "
            f"business_day_equal={verdict.business_day_equal} "
            f"awareness_equal={verdict.awareness_equal}"
        )
        assert verdict.legacy_dt is not None and verdict.arrow_dt is not None
        lines.append(f"      legacy day={_business_day(verdict.legacy_dt)} "
                     f"arrow day={_business_day(verdict.arrow_dt)}")
        lines.append(f"      {verdict.case.note}")

    lines.append("")
    lines.append("[4] MISSING-YEAR BATCH CERTIFICATION (no assumed default)")
    for batch in BATCHES:
        cert = certify_batch(batch)
        lines.append(f"  {cert['batch_id']} field={cert['field']!r} "
                     f"recorded_default_year={cert['default_year']}")
        lines.append(f"      bare field  legacy={cert['bare_legacy']} "
                     f"arrow={cert['bare_arrow']}")
        if "certifiable_min" in cert:
            rejected_sample = [
                y for y in range(MIN_YEAR, MIN_YEAR + 12)
                if y not in cert["arrow_accepted_years"]
            ][:6]
            lines.append(
                f"      explicit-year sweep: {cert['certifiable_count']} years "
                f"certifiable by BOTH parsers, range "
                f"{cert['certifiable_min']}..{cert['certifiable_max']}; "
                f"raw accepted sets agree: {cert['sets_agree']}"
            )
            if not cert["sets_agree"]:
                lo = cert["legacy_only_years"]
                ao = cert["arrow_only_years"]
                lines.append(
                    f"      range disagreement: legacy-only years "
                    f"{_bounds(lo)} ({len(lo)}), arrow-only years "
                    f"{_bounds(ao)} ({len(ao)})"
                )
            lines.append(
                "      first rejected years (if any): "
                f"{rejected_sample or 'none'}"
            )
        lines.append(f"      {cert['note']}")

    lines.append("")
    lines.append("[5] COMPATIBILITY-EVIDENCE LEDGER")
    both_accept = [v for v in verdicts if v.relation == "both_accept"]
    both_reject = [v for v in verdicts if v.relation == "both_reject"]
    arrow_only = [v for v in verdicts if v.relation == "arrow_only"]
    legacy_only = [v for v in verdicts if v.relation == "legacy_only"]
    diff_when_accept = [
        v for v in both_accept
        if v.instant_equal is False
        or v.business_day_equal is False
        or v.awareness_equal is False
    ]
    lines.append(f"  arrow-only acceptance classes : {len(arrow_only)} "
                 f"({', '.join(v.case.case_id for v in arrow_only)})")
    lines.append(f"  legacy-only acceptance classes: {len(legacy_only)} "
                 f"({', '.join(v.case.case_id for v in legacy_only)})")
    lines.append(f"  both reject (hard agreement)   : {len(both_reject)}")
    lines.append(f"  both accept but layers differ  : {len(diff_when_accept)} "
                 f"({', '.join(v.case.case_id for v in diff_when_accept)})")
    lines.append(
        "  policy: both-success rows are NOT counted as compatibility evidence;"
    )
    lines.append(
        "          a syntax is certified only when both sides reject together,"
    )
    lines.append(
        "          or both accept with identical epoch instant and business day."
    )
    lines.append("=" * 78)
    return "\n".join(lines) + "\n"


def main() -> None:  # pragma: no cover - exercised via the generated report
    report = render_report()
    print(report, end="")
    report_path = Path(__file__).resolve().parent / "report.txt"
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
