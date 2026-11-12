"""
Round-trip editing of vCard ORG paths through a standardization dictionary.

An ORG property is an ordered multi-level path (``ORG:Company;Dept;Team``)
and one card may carry several ORG properties, one per appointment.  This
module captures every appointment's original path, its escaped literal and
its source position, so that after standardizing department names (and
editing individual levels) the card can still be exported for the original
system:

* levels the user did not touch are exported with their original escaped
  literal, even when the dictionary maps several old names to one standard
  name;
* an edited level is translated back through the inverted dictionary;
* if a new standard name has several possible back-translations, only that
  one spot is marked as a pending choice -- other appointments on the same
  card are never rewritten.
"""

import io

from .base import VObjectError, backslashEscape, getLogicalLines, parseLine
from .icalendar import stringToTextValues

__all__ = [
    "OrgRecord",
    "PendingChoice",
    "PendingChoiceError",
    "capture_orgs",
    "export_orgs",
    "apply_orgs",
    "invert_dictionary",
    "split_escaped",
    "standardize_orgs",
]


def split_escaped(raw, separator=";"):
    """
    Split a raw (still backslash-escaped) ORG value into raw segments.

    Escaped separators (``\\;``) do not split, and escape sequences are
    kept verbatim inside each segment.
    """
    segments = []
    current = []
    i = 0
    while i < len(raw):
        char = raw[i]
        if char == "\\" and i + 1 < len(raw):
            current.append(raw[i : i + 2])
            i += 2
        elif char == separator:
            segments.append("".join(current))
            current = []
            i += 1
        else:
            current.append(char)
            i += 1
    segments.append("".join(current))
    return segments


def _unescape(segment):
    """Remove backslash escaping from one raw segment."""
    return stringToTextValues(segment)[0]


def _raw_of(line):
    """
    Rebuild an escaped literal for a parsed ORG ContentLine.

    Only used when the original source text is not available; prefer
    capturing from the source so the literal is preserved byte-for-byte.
    """
    value = line.value
    if isinstance(value, str):
        return backslashEscape(value)
    fields = []
    for item in value:
        if isinstance(item, (list, tuple)):
            fields.append(",".join(backslashEscape(part) for part in item))
        else:
            fields.append(backslashEscape(item))
    return ";".join(fields)


def _scan_source(source):
    """
    Return [(line_number, raw_value), ...] for every ORG line in source.
    """
    entries = []
    for logical_line, line_number in getLogicalLines(io.StringIO(source), allowQP=False):
        name, _params, value, _group = parseLine(logical_line, line_number)
        if name.upper() == "ORG":
            entries.append((line_number, value))
    return entries


class PendingChoice:
    """
    One edited level whose standard name maps back to several old names.

    Only this spot is undecided; resolve it with
    ``record.resolve(level_index, chosen_old_name)``.
    """

    def __init__(self, position, level_index, standard_name, candidates):
        self.position = position
        self.level_index = level_index
        self.standard_name = standard_name
        self.candidates = tuple(candidates)

    def __repr__(self):
        return f"<PendingChoice org#{self.position} level {self.level_index}: {self.standard_name!r} -> {self.candidates!r}>"

    def __eq__(self, other):
        try:
            return (
                self.position == other.position
                and self.level_index == other.level_index
                and self.standard_name == other.standard_name
                and self.candidates == other.candidates
            )
        except AttributeError:
            return False


class PendingChoiceError(VObjectError):
    """
    Raised by export_orgs when edited levels need a back-translation choice.

    The ``choices`` attribute lists every PendingChoice; each one names the
    appointment position and level index so callers can resolve exactly
    those spots and export again.
    """

    def __init__(self, choices):
        super().__init__(f"{len(choices)} edited ORG level(s) need a back-translation choice: {choices}")
        self.choices = list(choices)


class OrgRecord:
    """
    One ORG appointment captured from a vCard.

    @ivar position:
        0-based order of this ORG property among the card's ORG properties.
    @ivar line_number:
        Line number in the source stream, or None if unknown.
    @ivar raw_value:
        The original escaped literal of the whole ORG value.
    @ivar original_levels:
        Tuple of unescaped levels as originally parsed (the original path).
    @ivar standard_levels:
        Levels after applying the standardization dictionary.
    @ivar levels:
        Working copy the caller edits; starts as a copy of standard_levels.
    @ivar resolutions:
        Chosen old names for ambiguous edited levels, {level_index: name}.
    @ivar pending:
        PendingChoice per level index, populated by export().
    """

    def __init__(self, position, raw_value, line_number=None):
        self.position = position
        self.line_number = line_number
        self.raw_value = raw_value
        self.original_segments = tuple(split_escaped(raw_value))
        self.original_levels = tuple(_unescape(segment) for segment in self.original_segments)
        self.standard_levels = list(self.original_levels)
        self.levels = list(self.standard_levels)
        self.resolutions = {}
        self.pending = {}

    def standardize(self, dictionary):
        """
        Map every original level through the {old name: standard name}
        dictionary; unmapped levels keep their original value.
        """
        self.standard_levels = [dictionary.get(level, level) for level in self.original_levels]
        self.levels = list(self.standard_levels)
        self.resolutions = {}
        self.pending = {}
        return self

    def edited_indices(self):
        """Indices of levels whose working value differs from the standard."""
        return [
            i for i, value in enumerate(self.levels) if i >= len(self.standard_levels) or value != self.standard_levels[i]
        ]

    def resolve(self, level_index, original_name):
        """
        Choose one of the pending candidates for an ambiguous edited level.
        """
        choice = self.pending.get(level_index)
        if choice is None:
            raise ValueError(f"Level {level_index} of ORG #{self.position} has no pending choice")
        if original_name not in choice.candidates:
            raise ValueError(f"{original_name!r} is not a candidate for level {level_index} of ORG #{self.position}")
        self.resolutions[level_index] = original_name
        del self.pending[level_index]
        return self

    def export(self, reverse):
        """
        Build the escaped ORG value for the original system.

        Unchanged levels reuse their original escaped literal.  Edited
        levels are translated back through reverse ({standard: [old, ...]});
        a standard name with several candidates is recorded in self.pending
        and raises nothing here -- export_orgs collects them.
        """
        self.pending = {}
        segments = []
        for i, value in enumerate(self.levels):
            if i in self.resolutions:
                segments.append(backslashEscape(self.resolutions[i]))
            elif i < len(self.standard_levels) and value == self.standard_levels[i] and i < len(self.original_segments):
                segments.append(self.original_segments[i])
            else:
                candidates = reverse.get(value)
                if candidates is None:
                    segments.append(backslashEscape(value))
                elif len(candidates) == 1:
                    segments.append(backslashEscape(candidates[0]))
                else:
                    self.pending[i] = PendingChoice(self.position, i, value, candidates)
                    segments.append(None)
        if self.pending:
            return None
        return ";".join(segments)


def capture_orgs(card, source=None):
    """
    Capture one OrgRecord per ORG property of a parsed VCARD component.

    If source (the text the card was parsed from) is given, each record's
    escaped literal and line number come from the source itself; otherwise
    the literal is rebuilt from the parsed (unescaped) value.
    """
    lines = card.contents.get("org", [])
    scanned = _scan_source(source) if source is not None else None
    if scanned is not None and len(scanned) != len(lines):
        raise ValueError(f"Source has {len(scanned)} ORG lines but the card has {len(lines)}")
    records = []
    for position, line in enumerate(lines):
        if scanned is not None:
            line_number, raw_value = scanned[position]
        else:
            line_number = getattr(line, "lineNumber", None)
            raw_value = _raw_of(line)
        records.append(OrgRecord(position, raw_value, line_number=line_number))
    return records


def standardize_orgs(card, dictionary, source=None):
    """
    Capture the card's ORG appointments and standardize every level.

    Returns the list of OrgRecords; edit record.levels, then export.
    """
    records = capture_orgs(card, source=source)
    for record in records:
        record.standardize(dictionary)
    return records


def invert_dictionary(dictionary):
    """
    Invert {old name: standard name} into {standard name: [old name, ...]}.
    """
    reverse = {}
    for old_name, standard_name in dictionary.items():
        reverse.setdefault(standard_name, []).append(old_name)
    return reverse


def export_orgs(records, dictionary):
    """
    Export every record as an escaped ORG value, in source order.

    Raises PendingChoiceError if any edited level maps back to several old
    names; only those spots are marked, and no record is rewritten until
    every choice is resolved.
    """
    reverse = invert_dictionary(dictionary)
    values = []
    pending = []
    for record in records:
        values.append(record.export(reverse))
        pending.extend(record.pending.values())
    if pending:
        raise PendingChoiceError(pending)
    return values


def apply_orgs(card, records, dictionary):
    """
    Export the records and write the values back onto the card's ORG lines,
    so serializing the card reproduces the original system format.
    """
    values = export_orgs(records, dictionary)
    lines = card.contents.get("org", [])
    if len(lines) != len(records):
        raise ValueError(f"Card has {len(lines)} ORG lines but {len(records)} records were captured")
    for line, value in zip(lines, values):
        line.isNative = False
        line.encoded = True
        line.value = value
    return card
