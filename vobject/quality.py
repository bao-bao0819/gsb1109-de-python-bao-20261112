"""Directory quality reporting for vCard (VCF) streams.

:func:`vobject.base.readComponents` aborts an entire stream at the first
unparseable line, so a single corrupt card hides every record behind it.
The scanner in this module isolates damage at the record level instead:

* a card whose envelope (``BEGIN:VCARD`` ... ``END:VCARD``) is recognized but
  that contains unparseable or structurally invalid lines is reported as
  *damaged* while every property line that did parse still contributes to the
  field statistics;
* a card with a recognizable ``UID`` keeps its identity even when damaged, so
  it can be attributed as another (broken) version of a known contact;
* input that cannot be tied to any card envelope is *unreadable*: only the
  number of such records is known, never their contents.

Legal attributions of the uncertain records define the bound range:

* a damaged card whose UID matches an already known contact is fixed as
  another version of that contact, so its parseable fields complement the
  contact's fields under every attribution;
* a damaged card carrying an unrecognized UID cannot be merged into a known
  contact that already has a different UID. It may stand alone as a new
  contact, be treated as a corrupted version of a known contact that has no
  UID itself, or be discarded as invalid junk;
* a damaged card without a UID and an unreadable record may be discarded as
  junk, merged into any known contact, or treated as a new contact. An
  unreadable record can never establish a UID.

Because the true fate of the uncertain records cannot be observed, field
coverage is reported as reachable lower/upper bounds rather than a single
number. Both bounds correspond to concrete, legal attributions of the
uncertain records; :meth:`QualityReport.format_text` describes one such
attribution for each bound.
"""

import io

from .base import (
    Component,
    VObjectError,
    basestring,
    getBehavior,
    getLogicalLines,
    textLineToContentLine,
)

# Contact properties whose coverage is tracked, in report display order.
TRACKED_FIELDS = ("FN", "N", "TEL", "EMAIL", "ADR", "ORG", "UID")

# A line that fails to parse could have been any property -- except UID: a
# UID that cannot be read cannot establish identity and therefore never counts
# toward UID coverage.
POSSIBLE_FROM_UNKNOWN_LINE = tuple(name for name in TRACKED_FIELDS if name != "UID")

STATUS_OK = "ok"
STATUS_DAMAGED = "damaged"
STATUS_UNREADABLE = "unreadable"


class RecordResult:
    """Outcome of scanning one card-sized chunk of the input.

    @ivar index: 1-based position of the record in the scan.
    @ivar status: L{STATUS_OK}, L{STATUS_DAMAGED} or L{STATUS_UNREADABLE}.
    @ivar start_line: first source line covered by the record.
    @ivar end_line: last source line covered by the record.
    @ivar uid: recognized UID value, or None when no UID could be read.
    @ivar fields_present: tracked property names found on parseable lines.
    @ivar unknown_line_count: number of lines that could not be parsed; the
        property names they carried are unknowable.
    @ivar total_lines: number of logical lines consumed by the record.
    @ivar errors: human-readable descriptions of the damage found.
    """

    def __init__(
        self,
        index,
        status,
        start_line,
        end_line,
        uid=None,
        fields_present=None,
        unknown_line_count=0,
        total_lines=0,
        errors=None,
    ):
        self.index = index
        self.status = status
        self.start_line = start_line
        self.end_line = end_line
        self.uid = uid
        self.fields_present = frozenset(fields_present or ())
        self.unknown_line_count = unknown_line_count
        self.total_lines = total_lines
        self.errors = list(errors or ())

    def __repr__(self):
        return (
            f"<RecordResult #{self.index} {self.status} lines={self.start_line}-{self.end_line} "
            f"uid={self.uid!r} fields={sorted(self.fields_present)} unknown={self.unknown_line_count}>"
        )


class _RawCard:
    """Accumulator for a card while its lines are being streamed in."""

    def __init__(self, index, start_line, component_name):
        self.index = index
        self.start_line = start_line
        self.end_line = None
        self.component_name = component_name
        self.version_line = None
        self.vlines = []
        self.bad_lines = []
        self.fields_present = set()
        self.uid = None
        self.errors = []

    @property
    def total_lines(self):
        return len(self.vlines) + len(self.bad_lines) + 2


class _JunkRun:
    """Accumulator for a run of lines outside any card envelope."""

    def __init__(self, index, start_line):
        self.index = index
        self.start_line = start_line
        self.end_line = start_line
        self.total_lines = 0
        self.unparseable_lines = 0

    def add(self, line_number, unparseable):
        self.end_line = line_number
        self.total_lines += 1
        if unparseable:
            self.unparseable_lines += 1


class _Scanner:
    """State machine that splits a logical-line stream into records."""

    def __init__(self):
        self.records = []
        self._card = None
        self._junk = None
        self._next_index = 1

    def feed(self, line, line_number):
        try:
            vline = textLineToContentLine(line, line_number)
        except VObjectError as exc:
            if self._card is not None:
                self._card.bad_lines.append((line_number, line, str(exc)))
            else:
                self._junk_run(line_number).add(line_number, unparseable=True)
            return

        name = vline.name
        if name == "BEGIN":
            self._flush_junk()
            if self._card is not None:
                # A new envelope starts before the open one was closed: the
                # open card is structurally corrupt and must be abandoned so
                # that scanning can continue with the new envelope.
                self._card.errors.append(
                    f"BEGIN:{vline.value} on line {line_number} starts before the open "
                    f"{self._card.component_name} card was closed"
                )
                self._emit_card(damaged=True, end_line=line_number - 1)
            self._card = _RawCard(self._next_index, line_number, vline.value)
            self._next_index += 1
        elif name == "END":
            if self._card is None:
                self._junk_run(line_number).add(line_number, unparseable=False)
            elif vline.value.upper() == self._card.component_name.upper():
                self._card.end_line = line_number
                self._emit_card(damaged=False)
            else:
                self._card.errors.append(
                    f"END:{vline.value} on line {line_number} does not match "
                    f"BEGIN:{self._card.component_name}"
                )
                self._emit_card(damaged=True, end_line=line_number)
        else:
            if self._card is None:
                self._junk_run(line_number).add(line_number, unparseable=False)
                return
            self._card.vlines.append(vline)
            if name == "VERSION":
                self._card.version_line = vline
            if name in TRACKED_FIELDS:
                self._card.fields_present.add(name)
            if name == "UID" and self._card.uid is None:
                value = vline.value
                if isinstance(value, bytes):
                    value = value.decode("utf-8", "replace")
                value = value.strip()
                if value:
                    self._card.uid = value

    def finish(self):
        if self._card is not None:
            self._card.errors.append(
                f"component {self._card.component_name} starting on line "
                f"{self._card.start_line} was never closed"
            )
            self._emit_card(damaged=True, end_line=self._card.end_line)
        self._flush_junk()

    def _junk_run(self, line_number):
        if self._junk is None:
            self._junk = _JunkRun(self._next_index, line_number)
            self._next_index += 1
        return self._junk

    def _flush_junk(self):
        if self._junk is None:
            return
        junk = self._junk
        self._junk = None
        self.records.append(
            RecordResult(
                index=junk.index,
                status=STATUS_UNREADABLE,
                start_line=junk.start_line,
                end_line=junk.end_line,
                unknown_line_count=junk.unparseable_lines,
                total_lines=junk.total_lines,
                errors=["input outside any recognizable card envelope"],
            )
        )

    def _emit_card(self, damaged, end_line=None):
        card = self._card
        self._card = None
        status = STATUS_DAMAGED if damaged or card.bad_lines or card.errors else STATUS_OK
        if card.bad_lines:
            for line_number, raw_line, message in card.bad_lines:
                card.errors.append(f"unparseable line {line_number}: {raw_line!r} ({message})")
        if status == STATUS_OK:
            # All lines parsed; make sure vobject itself can build and natively
            # transform the component, so subtle value corruption is still
            # classified as damage rather than silently counted as valid.
            try:
                component = Component(card.component_name)
                for vline in card.vlines:
                    component.add(vline)
                if card.version_line is not None:
                    component.setBehaviorFromVersionLine(card.version_line)
                else:
                    behavior = getBehavior(component.name)
                    if behavior is not None:
                        component.setBehavior(behavior)
                component.transformChildrenToNative()
            except VObjectError as exc:
                status = STATUS_DAMAGED
                card.errors.append(f"native transformation failed: {exc}")
        self.records.append(
            RecordResult(
                index=card.index,
                status=status,
                start_line=card.start_line,
                end_line=end_line if end_line is not None else (card.end_line or card.start_line),
                uid=card.uid,
                fields_present=card.fields_present,
                unknown_line_count=len(card.bad_lines),
                total_lines=card.total_lines,
                errors=card.errors,
            )
        )


class QualityReport:
    """Field-coverage statistics with reachable bounds under uncertain records.

    @ivar records: L{RecordResult} instances in input order.
    @ivar total_records: number of card-sized chunks observed.
    @ivar ok_count: number of fully parseable, structurally valid cards.
    @ivar damaged_count: cards with a recognized envelope but some damage.
    @ivar unreadable_count: chunks with no recognizable card envelope.
    @ivar determinate_contacts: contacts established with certainty, i.e. ok
        cards merged by UID (ok cards without a UID each count separately).
    @ivar contact_lower_bound: fewest distinct contacts consistent with the
        data (every uncertain record is a duplicate of a known contact or
        discarded junk).
    @ivar contact_upper_bound: most distinct contacts consistent with the
        data (each unattributed record is a new contact, except records that
        share an unrecognized UID, which must remain one contact).
    @ivar field_coverage: mapping of tracked field name to a dictionary with
        C{lower_count}, C{upper_count}, C{lower_ratio} and C{upper_ratio}.
    @ivar lower_bound_example: one legal attribution realizing the lower bound.
    @ivar upper_bound_example: one legal attribution realizing the upper bound.
    """

    def __init__(self, records):
        self.records = list(records)
        self.total_records = len(self.records)
        self.ok_count = sum(1 for record in self.records if record.status == STATUS_OK)
        self.damaged_count = sum(1 for record in self.records if record.status == STATUS_DAMAGED)
        self.unreadable_count = sum(1 for record in self.records if record.status == STATUS_UNREADABLE)

        self._entities = {}
        self._possible_entities = {}
        self._free_uncertain = []
        self._uidless_entities = []
        self.lower_bound_example = None
        self.upper_bound_example = None

        self._build_entities()
        self.determinate_contacts = len(self._entities)
        self.contact_lower_bound = self.determinate_contacts
        self.contact_upper_bound = len(self._possible_entities)
        self.field_coverage = self._compute_coverage()

    def _build_entities(self):
        # Certain entities: ok cards merged by UID when present.
        for record in self.records:
            if record.status != STATUS_OK:
                continue
            key = record.uid if record.uid is not None else f"#{record.index}"
            self._entities.setdefault(key, set()).update(record.fields_present)
            self._possible_entities.setdefault(key, set()).update(record.fields_present)

        ok_uids = {key for key in self._entities if not key.startswith("#")}
        self._uidless_entities = [key for key in self._entities if key.startswith("#")]

        for record in self.records:
            if record.status == STATUS_OK:
                continue
            if record.status == STATUS_DAMAGED and record.uid is not None and record.uid in ok_uids:
                # Identity is established by the recognizable UID: the damaged
                # card is another version of the known contact. Its parsed
                # fields complement the existing ones in every attribution.
                self._entities[record.uid].update(record.fields_present)
                self._possible_entities[record.uid].update(record.fields_present)
                if record.unknown_line_count:
                    self._possible_entities[record.uid].update(POSSIBLE_FROM_UNKNOWN_LINE)
            else:
                self._free_uncertain.append(record)

        # Optimistic world: maximize the distinct contact population.
        # Records that share an unrecognized UID must be the same contact
        # (distinct contacts cannot share a UID); every remaining record is
        # its own new contact.
        possible_new = {}
        for record in self._free_uncertain:
            key = record.uid if record.uid is not None else f"#{record.index}"
            fields = possible_new.setdefault(key, set(record.fields_present))
            fields.update(record.fields_present)
            if record.status == STATUS_UNREADABLE or record.unknown_line_count:
                fields.update(POSSIBLE_FROM_UNKNOWN_LINE)
        self._possible_entities.update(possible_new)

        self._build_bound_examples(ok_uids)

    def _build_bound_examples(self, ok_uids):
        if not self._free_uncertain:
            return
        record = self._free_uncertain[0]
        location = f"record #{record.index} (lines {record.start_line}-{record.end_line})"

        # Lower bound: the record must add zero contacts. A damaged card with
        # an unrecognized UID cannot merge into a contact that has a different
        # UID, so pick a known contact without a UID when that is the only
        # legal anchor; otherwise the record is discarded as junk.
        if record.uid is not None and record.uid not in ok_uids:
            if self._uidless_entities:
                anchor = self._uidless_entities[0].lstrip("#")
                self.lower_bound_example = (
                    f"lower bound: {location} (uid={record.uid}) is attributed as a corrupted version "
                    f"of the known uid-less contact #{anchor}, so it adds no new contact"
                )
            else:
                self.lower_bound_example = (
                    f"lower bound: {location} (uid={record.uid}) is discarded as an invalid record, "
                    f"so it adds no new contact"
                )
        elif self._entities:
            anchor = next(iter(ok_uids), None) or next(iter(self._entities))
            self.lower_bound_example = (
                f"lower bound: {location} is attributed as a corrupted duplicate version of the "
                f"known contact {anchor}, so it adds no new contact"
            )
        else:
            self.lower_bound_example = (
                f"lower bound: {location} is attributed as discarded non-contact junk, so it adds no new contact"
            )
        self.upper_bound_example = (
            f"upper bound: {location} is attributed as a separate, previously unknown contact, "
            f"so it adds one new contact"
        )

    def _compute_coverage(self):
        upper_total = self.contact_upper_bound
        coverage = {}
        for field_name in TRACKED_FIELDS:
            # Lower numerator: fields established with certainty. Parseable
            # fields of a UID-matched damaged card already complemented their
            # known entity, so they survive even the most pessimistic
            # attribution; fields of unattributed records do not, since the
            # record itself could be discarded junk.
            lower_count = sum(1 for fields in self._entities.values() if field_name in fields)
            upper_count = sum(1 for fields in self._possible_entities.values() if field_name in fields)
            denominator = upper_total if upper_total else 1
            coverage[field_name] = {
                "lower_count": lower_count,
                "upper_count": upper_count,
                "lower_ratio": lower_count / denominator if upper_total else 0.0,
                "upper_ratio": upper_count / denominator if upper_total else 0.0,
            }
        return coverage

    @staticmethod
    def _percent(ratio):
        return f"{ratio * 100:.1f}%"

    def format_text(self):
        """Return a human-readable rendering of the report."""
        lines = [
            "vCard directory quality report",
            f"  records scanned     : {self.total_records} "
            f"(ok: {self.ok_count}, damaged: {self.damaged_count}, unreadable: {self.unreadable_count})",
            f"  determinate contacts: {self.determinate_contacts}",
            f"  contact count bounds: {self.contact_lower_bound} .. {self.contact_upper_bound}",
        ]
        if self.contact_upper_bound:
            lines.append(
                f"  field coverage (fielded contacts, share of the maximum population of "
                f"{self.contact_upper_bound} contacts):"
            )
            for field_name in TRACKED_FIELDS:
                entry = self.field_coverage[field_name]
                lines.append(
                    f"    {field_name:<5}: {entry['lower_count']}..{entry['upper_count']} "
                    f"({self._percent(entry['lower_ratio'])}..{self._percent(entry['upper_ratio'])})"
                )
        lines.append("  records:")
        for record in self.records:
            detail = f"uid={record.uid}" if record.uid else "no recognizable uid"
            fields = ",".join(sorted(record.fields_present)) or "-"
            lines.append(
                f"    #{record.index} {record.status:<10} lines {record.start_line}-{record.end_line} "
                f"[{detail}] fields={fields} unknown_lines={record.unknown_line_count}"
            )
        if self.lower_bound_example:
            lines.append(f"  {self.lower_bound_example}")
        if self.upper_bound_example:
            lines.append(f"  {self.upper_bound_example}")
        return "\n".join(lines)

    def __str__(self):
        return self.format_text()


def scan_vcard_quality(stream_or_string, allow_qp=True):
    """Scan a VCF stream and compute its bounded directory quality report.

    @param stream_or_string: a string or text stream containing vCards.
    @param allow_qp: enable vCard 2.1 quoted-printable unfolding, matching
        L{vobject.base.readComponents}.
    @return: a L{QualityReport}.
    """
    if isinstance(stream_or_string, basestring):
        stream = io.StringIO(stream_or_string)
    else:
        stream = stream_or_string

    scanner = _Scanner()
    for line, line_number in getLogicalLines(stream, allowQP=allow_qp):
        scanner.feed(line, line_number)
    scanner.finish()
    return QualityReport(scanner.records)
