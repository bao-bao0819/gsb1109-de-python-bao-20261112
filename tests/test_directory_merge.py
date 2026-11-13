"""Tests for building a unified-directory changeset from source-specific vCards."""

import io

import pytest
import vobject

from vobject.base import ParseError
from vobject.directory_merge import (
    Changeset,
    DirectoryMergeError,
    ScopedUID,
    build_changeset,
    read_source_cards,
)

BOOK_A = (
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "UID:a1\r\n"
    "FN:Alice Arden\r\n"
    "N:Arden;Alice\r\n"
    "TEL;TYPE=CELL:+100\r\n"
    "TEL:+555-0100\r\n"
    "END:VCARD\r\n"
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "UID:a2\r\n"
    "FN:Carol Carr\r\n"
    "TEL:+555-0100\r\n"
    "END:VCARD\r\n"
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "FN:No Identity\r\n"
    "END:VCARD\r\n"
)

BOOK_B = (
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "UID:b1\r\n"
    "FN:Alice Arden\r\n"
    "N:Arden;Alice\r\n"
    "TEL;TYPE=CELL:+200\r\n"
    "END:VCARD\r\n"
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "UID:b2\r\n"
    "FN:Carol Carr\r\n"
    "TEL:+555-0100\r\n"
    "END:VCARD\r\n"
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "UID:b3\r\n"
    "FN:Alice Twin\r\n"
    "END:VCARD\r\n"
)


def _tagged():
    return read_source_cards(io.StringIO(BOOK_A), "A") + read_source_cards(io.StringIO(BOOK_B), "B")


def test_scoped_uid_is_source_specific():
    left = ScopedUID("A", "x")
    right = ScopedUID("B", "x")
    assert left != right
    assert (left.source, left.uid) == ("A", "x")
    assert str(left) == "A:x"
    assert ScopedUID.parse("A:x") == left
    assert ScopedUID.parse(("B", "x")) == right
    with pytest.raises(DirectoryMergeError):
        ScopedUID("", "x")
    with pytest.raises(DirectoryMergeError):
        ScopedUID.parse("no-colon")


def test_read_source_cards_requires_source():
    with pytest.raises(DirectoryMergeError):
        read_source_cards(io.StringIO("BEGIN:VCARD\r\nVERSION:3.0\r\nFN:X\r\nEND:VCARD\r\n"))
    tagged = read_source_cards(
        io.StringIO("BEGIN:VCARD\r\nVERSION:3.0\r\nX-SOURCE:Z\r\nUID:z1\r\nFN:X\r\nEND:VCARD\r\n")
    )
    assert tagged[0][0] == "Z"


def test_confirmed_cross_source_cards_merge_into_one_entry():
    changeset = build_changeset(
        _tagged(),
        [("A:a1", "B:b1"), ("A:a2", "B:b2")],
        switchboard_numbers=["+555-0100"],
    )
    # a1/b1 and a2/b2 are confirmed pairs; the unlinked b3 stays its own entry.
    assert len(changeset.entries) == 3
    alice = next(change for change in changeset.entries if {str(u) for u in change.uids} == {"A:a1", "B:b1"})
    assert alice.status == "created"

    serialized = alice.card.serialize()
    # Both source-specific UIDs are preserved, each scoped to its book.
    assert "UID;SOURCE=A:a1" in serialized
    assert "UID;SOURCE=B:b1" in serialized
    # Distinct phone numbers are both retained; identical N/FN are not duplicated.
    assert serialized.count("TEL;TYPE=CELL:+100") == 1
    assert serialized.count("TEL;TYPE=CELL:+200") == 1
    assert serialized.count("FN:Alice Arden") == 1
    assert serialized.count("N:Arden;Alice") == 1

    # The output is still a valid, re-parseable vCard stream.
    reparsed = list(vobject.readComponents(io.StringIO(changeset.serialize())))
    assert len(reparsed) == 3


def test_switchboard_is_never_identity_evidence():
    # Carol's two cards share only the switchboard number and are not confirmed.
    changeset = build_changeset(_tagged(), switchboard_numbers=["+555-0100"])
    carol_uids = {scoped for change in changeset.entries for scoped in change.uids if scoped.uid in ("a2", "b2")}
    assert {str(u) for u in carol_uids} == {"A:a2", "B:b2"}
    assert not any(len(change.uids) > 1 for change in changeset.entries)
    assert changeset.ignored_evidence
    # The number itself is still kept on the retained cards.
    carol = next(change for change in changeset.entries if any(str(u) == "A:a2" for u in change.uids))
    assert any("555-0100" in str(line.value) for line in carol.card.tel_list)


def test_transitive_links_are_closed():
    changeset = build_changeset(_tagged(), [("A:a1", "B:b1"), ("B:b1", "B:b3")])
    # b1 and b3 are two valid UIDs in source B landing on one person: held, not merged.
    review = changeset.review[0]
    assert {str(u) for u in review.uids} == {"A:a1", "B:b1", "B:b3"}
    assert any("'B'" in reason for reason in review.reasons)
    # Every card that explains the conflict is retained.
    retained_uids = {uid.value for card in review.cards for uid in card.uid_list}
    assert {"a1", "b1", "b3"} <= set(retained_uids)
    # The conflicted UIDs are absent from the emitted entries.
    emitted = {str(u) for change in changeset.entries for u in change.uids}
    assert not {"A:a1", "B:b1", "B:b3"} & emitted


def test_same_source_two_valid_uids_holds_whole_group():
    changeset = build_changeset(
        _tagged(),
        [("A:a1", "B:b1"), ("A:a1", "A:a2")],
    )
    assert len(changeset.review) == 1
    review = changeset.review[0]
    assert {str(u) for u in review.uids} == {"A:a1", "A:a2", "B:b1"}
    assert any("'A'" in reason and "a1" in reason and "a2" in reason for reason in review.reasons)
    # Unrelated people are still emitted and entry ids stay gap-free.
    assert [change.entry_id for change in changeset.entries] == ["entry-0001", "entry-0002"]


def test_card_without_uid_is_unmatched():
    changeset = build_changeset(_tagged())
    assert [card.fn.value for card in changeset.unmatched] == ["No Identity"]


def test_unknown_confirmation_uid_is_rejected():
    with pytest.raises(DirectoryMergeError):
        build_changeset(_tagged(), [("A:a1", "B:missing")])


def test_second_import_updates_only_confirmed_person():
    first = build_changeset(
        _tagged(),
        [("A:a1", "B:b1"), ("A:a2", "B:b2")],
        switchboard_numbers=["+555-0100"],
    )

    # Alice changes her mobile in book A; confirmations are not restated.
    book_a_v2 = BOOK_A.replace("TEL;TYPE=CELL:+100", "TEL;TYPE=CELL:+999-NEW")
    tagged_v2 = read_source_cards(io.StringIO(book_a_v2), "A") + read_source_cards(io.StringIO(BOOK_B), "B")
    second = build_changeset(tagged_v2, switchboard_numbers=["+555-0100"], previous=first)

    alice = next(change for change in second.entries if any(str(u) == "A:a1" for u in change.uids))
    carol = next(change for change in second.entries if any(str(u) == "A:a2" for u in change.uids))
    assert alice.status == "updated"
    assert alice.changes["TEL"]["old"] == ["+100", "+200", "+555-0100"]
    assert alice.changes["TEL"]["new"] == ["+200", "+555-0100", "+999-NEW"]
    assert carol.status == "unchanged"
    assert carol.changes == {}
    # Identity survived without restating the confirmation.
    assert {str(u) for u in alice.uids} == {"A:a1", "B:b1"}

    # A brand-new card that happens to reuse Alice's new number must not join her.
    book_c = (
        "BEGIN:VCARD\r\nVERSION:3.0\r\nUID:c9\r\nFN:Sam Stranger\r\n"
        "TEL;TYPE=CELL:+999-NEW\r\nEND:VCARD\r\n"
    )
    tagged_v3 = tagged_v2 + read_source_cards(io.StringIO(book_c), "C")
    third = build_changeset(tagged_v3, previous=first)
    stranger = next(change for change in third.entries if any(str(u) == "C:c9" for u in change.uids))
    assert {str(u) for u in stranger.uids} == {"C:c9"}
    assert all("C:c9" not in {str(u) for u in change.uids} for change in third.entries if change is not stranger)


def test_serialized_changeset_round_trips():
    changeset = build_changeset(_tagged(), [("A:a1", "B:b1")])
    stream = changeset.serialize()
    cards = list(vobject.readComponents(io.StringIO(stream)))
    assert {card.uid_list[0].params["SOURCE"][0] for card in cards} <= {"A", "B"}


def test_changeset_defaults_are_independent():
    assert Changeset().entries == []
    assert Changeset().review == []
    first, second = Changeset(), Changeset()
    first.entries.append("x")
    assert second.entries == []
    # malformed input still raises the library parse error rather than passing silently
    with pytest.raises(ParseError):
        read_source_cards(io.StringIO("BEGIN:VCARD\r\nno-colon-here\r\n"), "A")
