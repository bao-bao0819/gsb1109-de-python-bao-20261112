"""Build a unified-directory changeset from source-specific vCard exports.

The two source address books only guarantee that a UID is unique *within* that
book, so the same bare UID in different books may refer to different people.
Every identity here is therefore a *scoped UID*, the pair ``(source, uid)``.

Cards are collapsed into one directory entry only when a human-confirmed
cross-source link connects their scoped UIDs. The policy encoded here is:

* Telephone numbers -- including a shared office switchboard -- are never
  evidence that two cards describe the same person, so they are never used to
  infer a match. A switchboard number shared by two cards is recorded in
  :attr:`Changeset.ignored_evidence` instead of linking them.
* Confirmed links are transitively closed. If the resulting component would
  place two *valid* UIDs from the same source under one person, the whole
  component is held for review and every source card that explains the
  conflict is retained (see :class:`ReviewGroup`).
* A later import that changes a phone number can only update the entry the
  card was already confirmed to belong to. Matching is by scoped UID, never by
  the new number, so an unrecognised card never hijacks an existing person.
"""

from argparse import ArgumentParser
from collections import defaultdict
from io import IOBase

import vobject
from .base import VObjectError, readComponents

# Properties that carry identity bookkeeping rather than contact data. They are
# managed explicitly and never compared as ordinary contact fields.
_META_PROPERTIES = ("VERSION", "UID", "X-SOURCE")


class DirectoryMergeError(VObjectError):
    """Raised when an input cannot be turned into a directory changeset."""


class ScopedUID(tuple):
    """A UID together with the address book that guarantees its uniqueness."""

    __slots__ = ()

    def __new__(cls, source, uid):
        if not source:
            raise DirectoryMergeError("A scoped UID requires a non-empty source")
        if not uid:
            raise DirectoryMergeError("A scoped UID requires a non-empty uid")
        return super().__new__(cls, (str(source), str(uid)))

    @property
    def source(self):
        return self[0]

    @property
    def uid(self):
        return self[1]

    def __str__(self):
        return f"{self.source}:{self.uid}"

    @staticmethod
    def parse(value):
        """Build a :class:`ScopedUID` from a tuple, ScopedUID, or ``source:uid``."""
        if isinstance(value, ScopedUID):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            return ScopedUID(value[0], value[1])
        if isinstance(value, str) and ":" in value:
            source, uid = value.split(":", 1)
            return ScopedUID(source, uid)
        raise DirectoryMergeError(f"Cannot interpret {value!r} as a scoped UID")


class _UnionFind:
    """Tiny disjoint-set structure used to close confirmed links transitively."""

    def __init__(self, nodes):
        self._parent = {node: node for node in nodes}

    def find(self, node):
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != node:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, left, right):
        self._parent[self.find(left)] = self.find(right)

    def components(self):
        grouped = defaultdict(list)
        for node in self._parent:
            grouped[self.find(node)].append(node)
        return list(grouped.values())


class ReviewGroup:
    """A set of confirmed cards held back because the links are contradictory.

    :ivar uids: every scoped UID caught up in the contradictory component.
    :ivar cards: the parsed source vCards that explain the conflict, retained
        verbatim so a human can adjudicate.
    :ivar reasons: human-readable explanations, one per offending source.
    """

    def __init__(self, uids, cards, reasons):
        self.uids = set(uids)
        self.cards = list(cards)
        self.reasons = list(reasons)

    def __repr__(self):
        return f"<ReviewGroup uids={sorted(map(str, self.uids))} reasons={self.reasons!r}>"


class PersonChange:
    """One unified directory entry and how it differs from the previous import.

    :ivar entry_id: stable identifier for this unified person.
    :ivar uids: scoped UIDs merged under this person.
    :ivar card: the merged vCard written to the unified directory.
    :ivar status: ``created``, ``updated`` or ``unchanged`` relative to
        ``previous``.
    :ivar changes: mapping of property name to ``{"old": [...], "new": [...]}``
        containing only properties whose values changed.
    """

    def __init__(self, entry_id, uids, card, status="created", changes=None):
        self.entry_id = entry_id
        self.uids = set(uids)
        self.card = card
        self.status = status
        self.changes = changes or {}

    def __repr__(self):
        return f"<PersonChange {self.entry_id} {self.status} uids={sorted(map(str, self.uids))}>"


class Changeset:
    """The result of reconciling one batch of vCards.

    :ivar entries: merged, non-contradictory directory entries.
    :ivar review: :class:`ReviewGroup` items held for human verification.
    :ivar unmatched: source cards that carried no valid UID and could not be
        placed or confirmed.
    :ivar ignored_evidence: descriptions of similarities (e.g. a shared
        switchboard) deliberately not treated as identity evidence.
    """

    def __init__(self, entries=None, review=None, unmatched=None, ignored_evidence=None):
        self.entries = entries or []
        self.review = review or []
        self.unmatched = unmatched or []
        self.ignored_evidence = ignored_evidence or []

    def unified_cards(self):
        """Return the merged vCard components in stable entry order."""
        return [change.card for change in self.entries]

    def serialize(self):
        """Serialize every merged entry as a single vCard stream."""
        return "".join(card.serialize() for card in self.unified_cards())


def _normalize_phone(value):
    """Reduce a phone value to digits for switchboard comparison."""
    return "".join(ch for ch in str(value) if ch.isdigit())


def _card_source(card):
    source = card.getChildValue("x-source")
    if source:
        return str(source)
    uid_lines = card.contents.get("uid", [])
    for line in uid_lines:
        if "SOURCE" in line.params:
            return str(line.params["SOURCE"][0])
    return None


def read_source_cards(stream, source=None):
    """Parse vCards and pair each with the address book it came from.

    ``source`` tags every card in the stream. When omitted, the source is read
    from each card's ``X-SOURCE`` line or its UID ``SOURCE`` parameter.
    """
    if isinstance(stream, IOBase):
        cards = list(readComponents(stream))
    else:
        cards = list(readComponents(str(stream)))
    tagged = []
    for card in cards:
        card_source = source or _card_source(card)
        if not card_source:
            raise DirectoryMergeError(
                "Card has no source; pass source= or add an X-SOURCE/UID;SOURCE= marker"
            )
        tagged.append((card_source, card))
    return tagged


def _valid_uids(source, card):
    uids = []
    for line in card.contents.get("uid", []):
        value = str(line.value).strip()
        if value:
            uids.append(ScopedUID(source, value))
    return sorted(set(uids))


def _property_signature(line):
    """Comparable, order-independent fingerprint of a contact property."""
    params = tuple(sorted((key, tuple(values)) for key, values in line.params.items()))
    value = line.value
    if isinstance(value, (list, tuple)):
        value = tuple(str(part) for part in value)
    else:
        value = str(value)
    return line.name, params, value


def _merge_cards(records, entry_id):
    """Merge the parsed cards of one clean identity component into a vCard."""
    merged = vobject.vCard()
    ordered = sorted(records, key=lambda record: record[0])

    for scoped_uid in sorted({scoped for scoped, _ in records}):
        uid_line = merged.add("uid")
        uid_line.value = scoped_uid.uid
        uid_line.source_param = scoped_uid.source

    seen = set()
    for _, card in ordered:
        for line in card.getChildren():
            if line.name in _META_PROPERTIES:
                continue
            signature = _property_signature(line)
            if signature in seen:
                continue
            seen.add(signature)
            merged.add(line.duplicate(line))

    if "fn" not in merged.contents:
        raise DirectoryMergeError(f"Entry {entry_id} cannot be merged: every vCard needs an FN")
    return merged


def _value_index(entries):
    """Map each scoped UID to the previously committed entry containing it."""
    index = {}
    for change in entries:
        for scoped_uid in change.uids:
            index[scoped_uid] = change
    return index


def _diff_properties(old_card, new_card):
    """Return only the contact properties whose value sets differ."""
    def by_name(card):
        grouped = defaultdict(set)
        for line in card.getChildren():
            if line.name in _META_PROPERTIES:
                continue
            grouped[line.name].add(_property_signature(line))
        return grouped

    old_values = by_name(old_card)
    new_values = by_name(new_card)
    changes = {}
    for name in sorted(set(old_values) | set(new_values)):
        before = {sig[2] for sig in old_values.get(name, set())}
        after = {sig[2] for sig in new_values.get(name, set())}
        if before != after:
            changes[name] = {"old": sorted(before), "new": sorted(after)}
    return changes


def _switchboard_overlaps(records, switchboard_numbers):
    """Find cross-source cards that share only a switchboard number."""
    if not switchboard_numbers:
        return []
    wanted = {_normalize_phone(number) for number in switchboard_numbers}
    holders = defaultdict(set)
    for scoped_uid, card in records:
        for line in card.contents.get("tel", []):
            if _normalize_phone(line.value) in wanted:
                holders[_normalize_phone(line.value)].add(scoped_uid.source)
    notes = []
    for number, sources in sorted(holders.items()):
        if len(sources) > 1:
            notes.append(
                f"switchboard {number} shared by sources {sorted(sources)} was not used as identity evidence"
            )
    return notes


def build_changeset(tagged_cards, confirmations=(), switchboard_numbers=None, previous=None):
    """Reconcile tagged vCards into a unified-directory :class:`Changeset`.

    :param tagged_cards: iterable of ``(source, vcard)`` pairs, as produced by
        :func:`read_source_cards`.
    :param confirmations: human-confirmed links; each item is a pair of scoped
        UID specs (tuples, :class:`ScopedUID`, or ``"source:uid"`` strings).
        Links are applied transitively.
    :param switchboard_numbers: phone values known to be an office switchboard.
        They are retained on cards but never used as identity evidence.
    :param previous: an earlier :class:`Changeset` (or list of
        :class:`PersonChange`) used to classify entries as created/updated.
    :raises DirectoryMergeError: if a confirmation references an unknown UID.
    """
    records = []
    unmatched = []
    for source, card in tagged_cards:
        uids = _valid_uids(source, card)
        if not uids:
            unmatched.append(card)
            continue
        for scoped_uid in uids:
            records.append((scoped_uid, card))

    uid_to_card = {}
    for scoped_uid, card in records:
        uid_to_card.setdefault(scoped_uid, card)

    links = [
        (ScopedUID.parse(left), ScopedUID.parse(right))
        for left, right in confirmations
    ]
    unknown = sorted(
        {str(uid) for pair in links for uid in pair if uid not in uid_to_card}
    )
    if unknown:
        raise DirectoryMergeError(f"Confirmations reference unknown scoped UIDs: {', '.join(unknown)}")

    previous_entries = previous.entries if isinstance(previous, Changeset) else (previous or [])
    previous_by_uid = _value_index(previous_entries)

    union_find = _UnionFind(uid_to_card.keys())
    # Identity once confirmed lives on in the directory, so scoped UIDs that
    # shared a prior entry stay merged even when the caller does not restate
    # every confirmation on a later import.
    for change in previous_entries:
        present = sorted(uid for uid in change.uids if uid in uid_to_card)
        for scoped_uid in present[1:]:
            union_find.union(present[0], scoped_uid)
    for left, right in links:
        union_find.union(left, right)

    entries = []
    review = []
    emitted = 0
    for component in sorted(union_find.components(), key=lambda group: tuple(map(str, group))):
        component_uids = set(component)
        per_source = defaultdict(list)
        for scoped_uid in component:
            per_source[scoped_uid.source].append(scoped_uid)

        reasons = [
            f"source {source!r} contributes multiple valid UIDs {sorted(uid.uid for uid in uids)} to one person"
            for source, uids in sorted(per_source.items())
            if len(uids) > 1
        ]
        if reasons:
            cards = []
            seen_cards = set()
            for scoped_uid in sorted(component_uids):
                card = uid_to_card[scoped_uid]
                if id(card) not in seen_cards:
                    seen_cards.add(id(card))
                    cards.append(card)
            review.append(ReviewGroup(component_uids, cards, reasons))
            continue

        emitted += 1
        entry_id = f"entry-{emitted:04d}"
        entry_records = [(scoped_uid, uid_to_card[scoped_uid]) for scoped_uid in component]
        merged = _merge_cards(entry_records, entry_id)

        prior = None
        for scoped_uid in component_uids:
            prior = previous_by_uid.get(scoped_uid)
            if prior is not None:
                break
        if prior is None:
            status, changes = "created", {}
        else:
            changes = _diff_properties(prior.card, merged)
            status = "updated" if changes else "unchanged"
        entries.append(PersonChange(entry_id, component_uids, merged, status, changes))

    ignored = _switchboard_overlaps(records, switchboard_numbers or [])
    return Changeset(entries=entries, review=review, unmatched=unmatched, ignored_evidence=ignored)


def get_arguments():
    parser = ArgumentParser(description="Merge source-specific vCard exports into a unified directory changeset")
    parser.add_argument("source", help="Address book identifier the imported cards belong to")
    parser.add_argument("vcard_file", help="vCard (.vcf) stream to import")
    parser.add_argument(
        "-c",
        "--confirm",
        action="append",
        default=[],
        metavar="SOURCE:UID=SOURCE:UID",
        help="Confirmed cross-source link, may be repeated",
    )
    parser.add_argument(
        "-s",
        "--switchboard",
        action="append",
        default=[],
        metavar="NUMBER",
        help="Office switchboard number that is never identity evidence",
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    with open(args.vcard_file) as stream:
        tagged = read_source_cards(stream, source=args.source)
    confirmations = []
    for link in args.confirm:
        left, right = link.split("=", 1)
        confirmations.append((ScopedUID.parse(left), ScopedUID.parse(right)))
    changeset = build_changeset(tagged, confirmations, switchboard_numbers=args.switchboard)
    for group in changeset.review:
        print(f"HELD FOR REVIEW: {'; '.join(group.reasons)}")
    print(changeset.serialize(), end="")


if __name__ == "__main__":
    main()
