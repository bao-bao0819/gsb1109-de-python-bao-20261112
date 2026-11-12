"""Per-exchange vCard projection and receipt collection.

An :class:`ExchangeSession` produces a *projection* of local vCards that is
safe to hand to an external partner: only properties belonging to authorized
groups are included, and authorized groups are renamed to codenames that are
valid for this exchange only.  Values, labels and identifiers of unauthorized
groups never appear in the exchange file.

The partner returns *receipts* (vCards referencing the codenames).  Receipts
may arrive in any order and the same receipt may be sent more than once.
Receipts that resolve deterministically are applied back to the original
cards idempotently; receipts carrying expired or unknown codenames, and
receipts that contradict an already-applied receipt under the same codename,
are quarantined instead of being applied.
"""

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from .base import Component, ContentLine, newFromBehavior, readComponents

# Quarantine reasons.
REASON_SESSION_EXPIRED = "session-expired"
REASON_UNKNOWN_CODENAME = "unknown-codename"
REASON_CONFLICT = "conflict"

# Ungrouped structural properties handled explicitly during projection.
_STRUCTURAL = ("VERSION", "UID")


@dataclass
class ReceiptRecord:
    """One receipt entry: all values seen for one codenamed group+property."""

    card_codename: Optional[str]
    name: str
    values: Tuple[str, ...]
    group: Optional[str] = None
    group_codename: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class CollectionReport:
    """Outcome of one :meth:`ExchangeSession.collect` call."""

    applied: List[ReceiptRecord] = field(default_factory=list)
    duplicates: List[ReceiptRecord] = field(default_factory=list)
    quarantined: List[ReceiptRecord] = field(default_factory=list)

    @property
    def clean(self):
        """True when nothing had to be quarantined."""
        return not self.quarantined


class ExchangeSession:
    """One projection + receipt-collection cycle for a set of vCards.

    ``authorized_groups`` names the vCard groups (RFC 2425 group prefixes,
    matched case-insensitively) the partner may see.  ``valid_until`` bounds
    how long receipts are accepted; ``now`` may be injected for testing.
    """

    def __init__(
        self,
        authorized_groups: Iterable[str],
        *,
        include_ungrouped: bool = True,
        valid_until: Optional[datetime] = None,
        now: Optional[Callable[[], datetime]] = None,
        token: Optional[str] = None,
    ):
        self.authorized_groups = frozenset(g.lower() for g in authorized_groups)
        self.include_ungrouped = include_ungrouped
        self.valid_until = valid_until
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._token = token or secrets.token_hex(4)
        self._closed = False

        # real group (lowercased) -> codename, and back to the real group
        self.group_codenames: Dict[str, str] = {}
        self._real_groups: Dict[str, str] = {}
        self._groups_by_codename: Dict[str, str] = {}

        # original UID -> card codename, codename -> original card
        self.card_codenames: Dict[str, str] = {}
        self._card_keys: Dict[str, str] = {}
        self._cards_by_codename: Dict[str, Component] = {}

        # (card codename, real group, property) -> accepted values
        self._applied: Dict[Tuple[str, str, str], Tuple[str, ...]] = {}
        self.quarantine: List[ReceiptRecord] = []

        self._group_counter = 0
        self._card_counter = 0

    # ------------------------------------------------------------------
    # session state

    @property
    def expired(self):
        """True once the session is closed or past ``valid_until``."""
        if self._closed:
            return True
        if self.valid_until is None:
            return False
        moment = self._now()
        until = self.valid_until
        if until.tzinfo is None and moment.tzinfo is not None:
            moment = moment.replace(tzinfo=None)
        elif until.tzinfo is not None and moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment > until

    def close(self):
        """Invalidate every codename issued by this session."""
        self._closed = True

    @property
    def applied(self):
        """Mapping of applied receipt keys to their accepted values."""
        return dict(self._applied)

    # ------------------------------------------------------------------
    # codenames

    def group_codename(self, group: str) -> str:
        """Return this exchange's codename for an authorized group."""
        key = group.lower()
        codename = self.group_codenames.get(key)
        if codename is None:
            self._group_counter += 1
            codename = f"g{self._token}{self._group_counter}"
            self.group_codenames[key] = codename
            self._real_groups[key] = group
            self._groups_by_codename[codename.lower()] = key
        return codename

    def card_codename(self, card: Component) -> str:
        """Return this exchange's codename for a card."""
        uid = card.getChildValue("uid")
        key = uid if uid is not None else f"object:{id(card)}"
        codename = self._card_keys.get(key)
        if codename is None:
            self._card_counter += 1
            codename = f"card-{self._token}-{self._card_counter}"
            self._card_keys[key] = codename
            self._cards_by_codename[codename] = card
            if uid is not None:
                self.card_codenames[uid] = codename
        return codename

    # ------------------------------------------------------------------
    # projection

    def project_card(self, card: Component) -> Component:
        """Build the exchange-safe projection of one vCard.

        Only ungrouped structural properties (subject to
        ``include_ungrouped``) and properties of authorized groups are
        carried over; authorized groups are renamed to their codename and
        the card's UID is replaced by a per-exchange card codename.
        """
        projected = newFromBehavior("vcard", "3.0")
        projected.add("version").value = card.getChildValue("version") or "3.0"
        projected.add("uid").value = self.card_codename(card)
        for child in card.getChildren():
            if isinstance(child, Component):
                continue
            name = child.name.upper()
            if child.group is None:
                if name in _STRUCTURAL or not self.include_ungrouped:
                    continue
                projected.add(child.duplicate(child))
            else:
                if child.group.lower() not in self.authorized_groups:
                    continue
                line = child.duplicate(child)
                line.group = self.group_codename(child.group)
                projected.add(line)
        return projected

    def project(self, cards: Iterable[Component]) -> str:
        """Serialize the projections of ``cards`` into one exchange file.

        Validation is skipped: a projection may legitimately omit required
        properties (e.g. FN when ungrouped properties are excluded).
        """
        return "".join(self.project_card(card).serialize(validate=False) for card in cards)

    # ------------------------------------------------------------------
    # receipt collection

    def collect(self, source: Union[str, Component, Iterable[Component]]) -> CollectionReport:
        """Collect receipts returned by the partner.

        ``source`` is a serialized vCard stream, one card, or an iterable of
        cards.  Deterministically matching receipts are applied back to the
        original cards; duplicates are idempotent no-ops; expired, unknown or
        contradicting codenames are quarantined.
        """
        if isinstance(source, Component):
            cards = [source]
        elif isinstance(source, str):
            cards = list(readComponents(source))
        else:
            cards = list(source)
        report = CollectionReport()
        for card in cards:
            self._collect_card(card, report)
        return report

    def _collect_card(self, card: Component, report: CollectionReport):
        card_codename = card.getChildValue("uid")
        pending: Dict[Tuple[str, str], List] = {}
        for child in card.getChildren():
            if isinstance(child, Component) or not child.group:
                continue
            key = (child.group, child.name.upper())
            pending.setdefault(key, []).append(child.value)
        for (group_codename, name), values in pending.items():
            self._collect_entry(card_codename, group_codename, name, values, report)

    def _collect_entry(self, card_codename, group_codename, name, values, report):
        record = ReceiptRecord(
            card_codename=card_codename,
            name=name,
            values=tuple(sorted(str(v) for v in values)),
            group_codename=group_codename,
        )
        group_key = self._groups_by_codename.get(group_codename.lower())
        if self.expired:
            record.reason = REASON_SESSION_EXPIRED
        elif card_codename not in self._cards_by_codename or group_key is None:
            record.reason = REASON_UNKNOWN_CODENAME
        if record.reason is not None:
            self.quarantine.append(record)
            report.quarantined.append(record)
            return

        record.group = self._real_groups[group_key]
        applied_key = (card_codename, group_key, name)
        accepted = self._applied.get(applied_key)
        if accepted is None:
            self._applied[applied_key] = record.values
            self._apply(card_codename, record.group, name, values)
            report.applied.append(record)
        elif accepted == record.values:
            report.duplicates.append(record)
        else:
            record.reason = REASON_CONFLICT
            self.quarantine.append(record)
            report.quarantined.append(record)

    def _apply(self, card_codename, group, name, values):
        """Write receipt values back onto the original card, idempotently."""
        card = self._cards_by_codename[card_codename]
        for line in list(card.contents.get(name.lower(), [])):
            if (line.group or "").lower() == group.lower():
                card.remove(line)
        for value in values:
            card.add(ContentLine(name, [], value, group=group))
