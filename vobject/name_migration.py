"""Migration between old-style and new-style structured name directories.

The *new* directory stores the five components of a vCard ``N`` property
(family, given, additional, prefix, suffix) as separate columns, and each
column may hold several values.  The *old* directory only has one set of
nullable scalar columns: every component is either missing (``None``, the
column is absent/NULL), empty (``""``) or a single string.

This module provides:

``upgrade`` / ``upgrade_all``
    Map old records to the new schema.  Only records that need no guessing
    are mapped; scalars containing ``,``, ``;`` or ``\\`` are ambiguous
    (a comma may be a vCard value list or a literal comma, a semicolon may
    be a leaked component separator, a backslash may hide exporter-specific
    escapes) and are reported instead of being silently interpreted.

``downgrade`` / ``export_for_old_system``
    Map new records back to old scalars so the legacy system can read
    them.  Multi-valued components are joined with ``,`` which is lossy,
    so ``find_conflicts`` reports any pair of distinct source names that
    become indistinguishable after the downgrade, listing the conflicting
    components and the original values.

``sync_from_old_system``
    Apply rows written back by the old system.  Rows that were not touched
    since the export are recovered with their original (possibly
    multi-valued) semantics from the export snapshot; edited rows are
    re-upgraded under the no-guessing rules.

``ExportConvention`` / ``format_line`` / ``parse_line`` / ``infer_convention``
    Exporters disagree about how a missing column, an empty column and an
    escaped semicolon are written.  A convention captures one set of
    choices; ``infer_convention`` picks the convention that matches
    business-provided sample pairs of (export line, expected record).

A small command line interface ties the pieces together, see
``python3 -m vobject.name_migration --help``.
"""

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .base import str_
from .vcard import NAME_ORDER, Name, serializeFields, splitFields

__all__ = [
    "COMPONENTS",
    "OldName",
    "NewName",
    "AmbiguousComponent",
    "NeedsGuessError",
    "upgrade",
    "upgrade_all",
    "downgrade",
    "NameConflict",
    "find_conflicts",
    "ExportSnapshot",
    "export_for_old_system",
    "SyncResult",
    "sync_from_old_system",
    "ExportConvention",
    "CANDIDATE_CONVENTIONS",
    "format_line",
    "parse_line",
    "AmbiguousConventionError",
    "matching_conventions",
    "infer_convention",
    "to_vcard_name",
    "from_vcard_name",
    "new_name_to_n_value",
    "new_name_from_n_value",
    "main",
]

COMPONENTS = NAME_ORDER  # ("family", "given", "additional", "prefix", "suffix")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OldName:
    """One old-directory record: five nullable scalar columns.

    ``None`` means the column is missing (absent/NULL), ``""`` means the
    column is present but empty; the two are kept distinct.
    """

    family: Optional[str] = None
    given: Optional[str] = None
    additional: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Optional[str]]) -> "OldName":
        return cls(**{c: mapping.get(c) for c in COMPONENTS})

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {c: getattr(self, c) for c in COMPONENTS}


@dataclass(frozen=True)
class NewName:
    """One new-directory record: five columns, each a tuple of values."""

    family: Tuple[str, ...] = ()
    given: Tuple[str, ...] = ()
    additional: Tuple[str, ...] = ()
    prefix: Tuple[str, ...] = ()
    suffix: Tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Iterable[str]]) -> "NewName":
        return cls(**{c: tuple(mapping.get(c) or ()) for c in COMPONENTS})

    def as_dict(self) -> Dict[str, List[str]]:
        return {c: list(getattr(self, c)) for c in COMPONENTS}


# ---------------------------------------------------------------------------
# Upgrade (old -> new), no guessing
# ---------------------------------------------------------------------------

#: Characters that make an old scalar ambiguous: a comma may be a vCard
#: value-list separator or literal data, a semicolon may be a leaked
#: component separator, a backslash may hide exporter-specific escapes.
AMBIGUOUS_CHARS = {
    ",": "comma may separate multiple values or be literal data",
    ";": "semicolon may be a leaked component separator or literal data",
    "\\": "backslash escapes depend on the exporter convention",
}


@dataclass(frozen=True)
class AmbiguousComponent:
    """One component of an old record that cannot be mapped safely."""

    component: str
    value: str
    reason: str


class NeedsGuessError(ValueError):
    """Raised when an old record cannot be upgraded without guessing."""

    def __init__(self, issues: Sequence[AmbiguousComponent]):
        self.issues = tuple(issues)
        summary = "; ".join(f"{i.component}={i.value!r}: {i.reason}" for i in self.issues)
        super().__init__(f"record needs guessing: {summary}")


def _upgrade_scalar(component: str, value: Optional[str]) -> Tuple[str, ...]:
    if value is None:
        return ()
    for char, reason in AMBIGUOUS_CHARS.items():
        if char in value:
            raise NeedsGuessError([AmbiguousComponent(component, value, reason)])
    return (value,)


def upgrade(old: OldName) -> NewName:
    """Map one old record to the new schema.

    Missing columns become empty value lists, empty columns become a single
    empty value, and plain scalars become single values.  Raises
    :class:`NeedsGuessError` listing every ambiguous component if any
    scalar cannot be mapped without guessing.
    """
    issues = []
    values = {}
    for component in COMPONENTS:
        scalar = getattr(old, component)
        try:
            values[component] = _upgrade_scalar(component, scalar)
        except NeedsGuessError as err:
            issues.extend(err.issues)
    if issues:
        raise NeedsGuessError(issues)
    return NewName(**values)


@dataclass(frozen=True)
class UpgradeResult:
    migrated: Mapping[Any, NewName]
    skipped: Mapping[Any, Tuple[AmbiguousComponent, ...]]


def upgrade_all(records: Mapping[Any, OldName]) -> UpgradeResult:
    """Upgrade every record that needs no guessing; report the rest."""
    migrated: Dict[Any, NewName] = {}
    skipped: Dict[Any, Tuple[AmbiguousComponent, ...]] = {}
    for key, old in records.items():
        try:
            migrated[key] = upgrade(old)
        except NeedsGuessError as err:
            skipped[key] = err.issues
    return UpgradeResult(migrated=migrated, skipped=skipped)


# ---------------------------------------------------------------------------
# Downgrade (new -> old) and conflict detection
# ---------------------------------------------------------------------------


def downgrade(new: NewName) -> OldName:
    """Map one new record to old scalar columns for the legacy system.

    An empty value list becomes a missing column, a single value becomes
    the scalar, and multiple values are joined with ``,`` (the vCard text
    list convention).  The join is lossy: use :func:`find_conflicts` to
    detect names that collapse onto the same old record.
    """
    scalars = {}
    for component in COMPONENTS:
        values = getattr(new, component)
        if not values:
            scalars[component] = None
        elif len(values) == 1:
            scalars[component] = values[0]
        else:
            scalars[component] = ",".join(values)
    return OldName(**scalars)


@dataclass(frozen=True)
class NameConflict:
    """Distinct source names that are indistinguishable after downgrade."""

    downgraded: OldName
    components: Tuple[str, ...]  # components on which the sources differ
    sources: Tuple[NewName, ...]  # the distinct original names
    keys: Tuple[Any, ...]  # record keys of the sources, in input order


def find_conflicts(records: Mapping[Any, NewName]) -> List[NameConflict]:
    """Report groups of distinct new records sharing one downgraded form."""
    groups: Dict[OldName, List[Tuple[Any, NewName]]] = {}
    for key, new in records.items():
        groups.setdefault(downgrade(new), []).append((key, new))
    conflicts = []
    for old, members in groups.items():
        distinct: List[NewName] = []
        keys: List[Any] = []
        for key, new in members:
            if new not in distinct:
                distinct.append(new)
            keys.append(key)
        if len(distinct) < 2:
            continue
        differing = tuple(
            c for c in COMPONENTS if len({getattr(n, c) for n in distinct}) > 1
        )
        conflicts.append(
            NameConflict(
                downgraded=old,
                components=differing,
                sources=tuple(distinct),
                keys=tuple(keys),
            )
        )
    return conflicts


# ---------------------------------------------------------------------------
# Round trip: export a snapshot, sync what the old system writes back
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportSnapshot:
    """Remembers, per record key, the exported old row and its source."""

    originals: Mapping[Any, Tuple[OldName, NewName]]


def export_for_old_system(records: Mapping[Any, NewName]) -> ExportSnapshot:
    """Downgrade every record and keep a snapshot for the write-back sync."""
    return ExportSnapshot(
        originals={key: (downgrade(new), new) for key, new in records.items()}
    )


@dataclass(frozen=True)
class SyncResult:
    restored: Mapping[Any, NewName]  # untouched rows, original semantics
    remapped: Mapping[Any, NewName]  # edited/new rows, re-upgraded
    skipped: Mapping[Any, Tuple[AmbiguousComponent, ...]]  # edits needing guesses
    removed: Tuple[Any, ...]  # exported keys absent from the write-back


def sync_from_old_system(
    written_back: Mapping[Any, OldName], snapshot: ExportSnapshot
) -> SyncResult:
    """Merge rows written back by the old system into the new directory.

    A row identical to the exported one was not touched, so the original
    new record is recovered as-is.  Any other row is treated as an edit
    (or an addition) and upgraded under the no-guessing rules.
    """
    restored: Dict[Any, NewName] = {}
    remapped: Dict[Any, NewName] = {}
    skipped: Dict[Any, Tuple[AmbiguousComponent, ...]] = {}
    for key, old in written_back.items():
        original = snapshot.originals.get(key)
        if original is not None and old == original[0]:
            restored[key] = original[1]
            continue
        try:
            remapped[key] = upgrade(old)
        except NeedsGuessError as err:
            skipped[key] = err.issues
    removed = tuple(key for key in snapshot.originals if key not in written_back)
    return SyncResult(restored=restored, remapped=remapped, skipped=skipped, removed=removed)


# ---------------------------------------------------------------------------
# Export conventions: missing vs empty columns, escaped semicolons
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportConvention:
    """How one exporter writes the five scalar columns as a single line.

    ``null_token``
        Token written for a missing column (``None``).  If ``None`` itself,
        missing columns are written exactly like empty ones and the
        distinction is lost on this export.
    ``escape_style``
        ``"backslash"`` writes a literal semicolon as ``\\;`` (and ``\\``
        for a backslash); ``"quote"`` wraps values containing ``;`` or
        ``"`` in double quotes, doubling inner quotes.
    ``drop_trailing_missing``
        Whether trailing missing columns are omitted from the line.
    """

    name: str
    null_token: Optional[str] = None
    escape_style: str = "backslash"
    drop_trailing_missing: bool = False


#: Conventions tried by :func:`infer_convention`, most specific first.
CANDIDATE_CONVENTIONS = (
    ExportConvention("sql-dump", null_token="\\N", escape_style="backslash"),
    ExportConvention("sql-dump-quoted", null_token="\\N", escape_style="quote"),
    ExportConvention("vcard-backslash", null_token=None, escape_style="backslash"),
    ExportConvention("csv-quoted", null_token=None, escape_style="quote"),
    ExportConvention(
        "trailing-omitted",
        null_token=None,
        escape_style="backslash",
        drop_trailing_missing=True,
    ),
)


def _escape_backslash(value: str) -> str:
    return value.replace("\\", "\\\\").replace(";", "\\;")


def _split_backslash_raw(line: str) -> List[str]:
    """Split on unescaped semicolons, keeping escape sequences raw."""
    fields: List[str] = []
    buf: List[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line):
            buf.append(line[index : index + 2])
            index += 2
        elif char == ";":
            fields.append("".join(buf))
            buf = []
            index += 1
        else:
            buf.append(char)
            index += 1
    fields.append("".join(buf))
    return fields


def _unescape_backslash(field: str) -> str:
    out: List[str] = []
    index = 0
    while index < len(field):
        char = field[index]
        if char == "\\":
            if index + 1 >= len(field):
                raise ValueError(f"dangling backslash in {field!r}")
            nxt = field[index + 1]
            if nxt not in ("\\", ";"):
                raise ValueError(f"unsupported escape \\{nxt} in {field!r}")
            out.append(nxt)
            index += 2
        else:
            out.append(char)
            index += 1
    return "".join(out)


def _escape_quote(value: str) -> str:
    if '"' in value or ";" in value:
        return '"' + value.replace('"', '""') + '"'
    return value


def _split_quote(line: str) -> List[str]:
    reader = csv.reader(io.StringIO(line), delimiter=";", strict=True)
    try:
        return next(reader)
    except csv.Error as err:
        raise ValueError(f"cannot parse quoted line {line!r}: {err}") from err


def format_line(old: OldName, convention: ExportConvention) -> str:
    """Render one old record as an export line under ``convention``."""
    escape = _escape_backslash if convention.escape_style == "backslash" else _escape_quote
    values = [getattr(old, c) for c in COMPONENTS]
    if convention.drop_trailing_missing:
        while values and values[-1] is None:
            values.pop()
    tokens = []
    for value in values:
        if value is None:
            tokens.append(convention.null_token if convention.null_token is not None else "")
        else:
            tokens.append(escape(value))
    return ";".join(tokens)


def parse_line(line: str, convention: ExportConvention) -> OldName:
    """Parse one export line into an old record under ``convention``.

    Short lines are padded with missing columns; lines with more than five
    columns are rejected.
    """
    if convention.escape_style == "backslash":
        raw_fields = _split_backslash_raw(line)
        fields: List[Optional[str]] = [
            None
            if convention.null_token is not None and raw == convention.null_token
            else _unescape_backslash(raw)
            for raw in raw_fields
        ]
    else:
        quoted = _split_quote(line)
        fields = [
            None if convention.null_token is not None and value == convention.null_token else value
            for value in quoted
        ]
    if len(fields) > len(COMPONENTS):
        raise ValueError(f"expected at most {len(COMPONENTS)} columns, got {len(fields)}: {line!r}")
    values: List[Optional[str]] = list(fields) + [None] * (len(COMPONENTS) - len(fields))
    return OldName(**dict(zip(COMPONENTS, values)))


class AmbiguousConventionError(ValueError):
    """Raised when samples match more than one candidate convention."""

    def __init__(self, matches: Sequence[ExportConvention]):
        self.matches = tuple(matches)
        names = ", ".join(c.name for c in self.matches)
        super().__init__(f"samples match more than one convention: {names}")


def matching_conventions(
    samples: Iterable[Tuple[str, OldName]],
    candidates: Iterable[ExportConvention] = CANDIDATE_CONVENTIONS,
) -> List[ExportConvention]:
    """Return every candidate consistent with all (line, record) samples."""
    pairs = list(samples)
    matches = []
    for convention in candidates:
        try:
            consistent = all(
                parse_line(line, convention) == old
                and format_line(old, convention) == line
                for line, old in pairs
            )
        except ValueError:
            consistent = False
        if consistent:
            matches.append(convention)
    return matches


def infer_convention(
    samples: Iterable[Tuple[str, OldName]],
    candidates: Iterable[ExportConvention] = CANDIDATE_CONVENTIONS,
) -> ExportConvention:
    """Pick the single convention matching business-provided samples.

    Raises :class:`ValueError` if no candidate matches, or
    :class:`AmbiguousConventionError` if the samples cannot distinguish
    several candidates.
    """
    pairs = list(samples)
    matches = matching_conventions(pairs, candidates)
    if not matches:
        raise ValueError(f"no known convention matches {len(pairs)} sample(s)")
    if len(matches) > 1:
        raise AmbiguousConventionError(matches)
    return matches[0]


# ---------------------------------------------------------------------------
# Bridge to vobject's native vCard N handling
# ---------------------------------------------------------------------------


def to_vcard_name(new: NewName) -> Name:
    """Build a :class:`vobject.vcard.Name` from a new-directory record."""
    kwargs = {}
    for component in COMPONENTS:
        values = getattr(new, component)
        if len(values) == 1:
            kwargs[component] = values[0]
        else:
            kwargs[component] = list(values)
    return Name(**kwargs)


def from_vcard_name(name: Name) -> NewName:
    """Build a new-directory record from a :class:`vobject.vcard.Name`.

    An empty vCard component has no values; a string is one value; a list
    of strings maps to that many values.
    """
    kwargs = {}
    for component in COMPONENTS:
        value = getattr(name, component)
        if isinstance(value, (list, tuple)):
            kwargs[component] = tuple(str_(v) for v in value)
        elif value in ("", None):
            kwargs[component] = ()
        else:
            kwargs[component] = (str_(value),)
    return NewName(**kwargs)


def new_name_to_n_value(new: NewName) -> str:
    """Serialize a new record as a vCard ``N`` property value (lossless)."""
    return serializeFields(to_vcard_name(new), NAME_ORDER)


def new_name_from_n_value(value: str) -> NewName:
    """Parse a vCard ``N`` property value into a new-directory record."""
    fields = splitFields(value)
    fields = fields + [""] * (len(COMPONENTS) - len(fields))
    kwargs = {}
    for component, field_value in zip(COMPONENTS, fields):
        if isinstance(field_value, list):
            kwargs[component] = tuple(field_value)
        elif field_value == "":
            kwargs[component] = ()
        else:
            kwargs[component] = (field_value,)
    return NewName(**kwargs)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

DEFAULT_NULL_TOKEN = "\\N"


def _load_old_csv(path: str, null_token: Optional[str]) -> Dict[str, OldName]:
    records: Dict[str, OldName] = {}
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [c for c in COMPONENTS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s): {', '.join(missing)}")
        for index, row in enumerate(reader):
            key = row.get("id") or str(index)
            values = {}
            for component in COMPONENTS:
                raw = row.get(component)
                if raw is None or (null_token is not None and raw == null_token):
                    values[component] = None
                else:
                    values[component] = raw
            records[key] = OldName(**values)
    return records


def _dump_old_csv(records: Mapping[Any, OldName], path: str, null_token: Optional[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", *COMPONENTS])
        for key, old in records.items():
            row = [key]
            for component in COMPONENTS:
                value = getattr(old, component)
                if value is None:
                    row.append(null_token if null_token is not None else "")
                else:
                    row.append(value)
            writer.writerow(row)


def _load_new_json(path: str) -> Dict[str, NewName]:
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    return {key: NewName.from_mapping(value) for key, value in data.items()}


def _dump_new_json(records: Mapping[Any, NewName], path: str) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump({str(key): new.as_dict() for key, new in records.items()}, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _load_snapshot(path: str) -> ExportSnapshot:
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    originals = {
        key: (OldName.from_mapping(entry["old"]), NewName.from_mapping(entry["new"]))
        for key, entry in data.items()
    }
    return ExportSnapshot(originals=originals)


def _dump_snapshot(snapshot: ExportSnapshot, path: str) -> None:
    data = {
        str(key): {"old": old.as_dict(), "new": new.as_dict()}
        for key, (old, new) in snapshot.originals.items()
    }
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _issues_as_json(issues: Mapping[Any, Tuple[AmbiguousComponent, ...]]) -> Dict[str, List[Dict[str, str]]]:
    return {
        str(key): [
            {"component": i.component, "value": i.value, "reason": i.reason}
            for i in issue_list
        ]
        for key, issue_list in issues.items()
    }


def _conflicts_as_json(conflicts: Sequence[NameConflict]) -> List[Dict[str, Any]]:
    return [
        {
            "downgraded": conflict.downgraded.as_dict(),
            "components": list(conflict.components),
            "sources": [source.as_dict() for source in conflict.sources],
            "keys": [str(key) for key in conflict.keys],
        }
        for conflict in conflicts
    ]


def _cmd_upgrade(args: argparse.Namespace) -> int:
    records = _load_old_csv(args.old_csv, args.null_token)
    result = upgrade_all(records)
    _dump_new_json(result.migrated, args.new_json)
    if result.skipped:
        json.dump(_issues_as_json(result.skipped), sys.stderr, indent=2, ensure_ascii=False)
        sys.stderr.write("\n")
        print(f"{len(result.skipped)} record(s) need guessing, skipped", file=sys.stderr)
    print(f"upgraded {len(result.migrated)} record(s) -> {args.new_json}")
    return 0


def _cmd_downgrade(args: argparse.Namespace) -> int:
    records = _load_new_json(args.new_json)
    conflicts = find_conflicts(records)
    if conflicts:
        report = _conflicts_as_json(conflicts)
        if args.conflicts:
            with open(args.conflicts, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
        else:
            json.dump(report, sys.stderr, indent=2, ensure_ascii=False)
            sys.stderr.write("\n")
        print(f"{len(conflicts)} downgrade conflict(s) detected", file=sys.stderr)
    snapshot = export_for_old_system(records)
    old_rows = {key: old for key, (old, _) in snapshot.originals.items()}
    _dump_old_csv(old_rows, args.old_csv, args.null_token)
    _dump_snapshot(snapshot, args.snapshot)
    print(f"downgraded {len(old_rows)} record(s) -> {args.old_csv}")
    return 2 if conflicts else 0


def _cmd_sync(args: argparse.Namespace) -> int:
    written_back = _load_old_csv(args.old_csv, args.null_token)
    snapshot = _load_snapshot(args.snapshot)
    result = sync_from_old_system(written_back, snapshot)
    merged: Dict[str, NewName] = {}
    merged.update(result.restored)
    merged.update(result.remapped)
    _dump_new_json(merged, args.new_json)
    if result.skipped:
        json.dump(_issues_as_json(result.skipped), sys.stderr, indent=2, ensure_ascii=False)
        sys.stderr.write("\n")
    print(
        f"synced {len(result.restored)} untouched, {len(result.remapped)} edited, "
        f"{len(result.skipped)} skipped, {len(result.removed)} removed"
    )
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    with open(args.samples, encoding="utf-8") as stream:
        raw = json.load(stream)
    samples = [(entry["line"], OldName.from_mapping(entry["record"])) for entry in raw]
    matches = matching_conventions(samples)
    if not matches:
        print("no known convention matches the samples", file=sys.stderr)
        return 1
    for convention in matches:
        print(convention.name)
    if len(matches) > 1:
        print("warning: samples do not distinguish these conventions", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="n_migrate",
        description="Migrate structured names between the old scalar directory "
        "and the new multi-valued directory.",
    )
    parser.add_argument(
        "--null-token",
        default=DEFAULT_NULL_TOKEN,
        help="CSV token standing for a missing column (default: %(default)r; "
        "use an empty string to disable)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upgrade_parser = subparsers.add_parser("upgrade", help="old CSV -> new JSON (no guessing)")
    upgrade_parser.add_argument("old_csv")
    upgrade_parser.add_argument("new_json")
    upgrade_parser.set_defaults(func=_cmd_upgrade)

    downgrade_parser = subparsers.add_parser(
        "downgrade", help="new JSON -> old CSV plus a snapshot for sync"
    )
    downgrade_parser.add_argument("new_json")
    downgrade_parser.add_argument("old_csv")
    downgrade_parser.add_argument("--snapshot", required=True)
    downgrade_parser.add_argument("--conflicts", help="write the conflict report here")
    downgrade_parser.set_defaults(func=_cmd_downgrade)

    sync_parser = subparsers.add_parser("sync", help="merge old-system write-back")
    sync_parser.add_argument("old_csv")
    sync_parser.add_argument("snapshot")
    sync_parser.add_argument("new_json")
    sync_parser.set_defaults(func=_cmd_sync)

    infer_parser = subparsers.add_parser(
        "infer", help="infer the export convention from sample pairs"
    )
    infer_parser.add_argument("samples", help="JSON list of {line, record} objects")
    infer_parser.set_defaults(func=_cmd_infer)

    args = parser.parse_args(argv)
    if args.null_token == "":
        args.null_token = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
