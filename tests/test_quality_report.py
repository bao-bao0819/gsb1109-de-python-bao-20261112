"""Tests for bounded directory quality reporting over continuous VCF input."""

import io

import pytest

from vobject.quality import (
    POSSIBLE_FROM_UNKNOWN_LINE,
    STATUS_DAMAGED,
    STATUS_OK,
    STATUS_UNREADABLE,
    TRACKED_FIELDS,
    scan_vcard_quality,
)

# One continuous stream mixing every record class the report must distinguish:
#   #1 ok card for Alice (UID, FN, N, TEL)
#   #2 damaged duplicate version of Alice: same recognizable UID, a legal EMAIL
#      that complements record #1, plus one unparseable contact line
#   #3 ok card for Bob, no UID (FN, N)
#   #4 damaged card whose UID matches no ok card (Dave), with an unparseable
#      line, so its identity is only possible, not determinate
#   #5 fully unreadable run outside any card envelope
CONTINUOUS_VCF = (
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "UID:alice-uid\r\n"
    "FN:Alice\r\n"
    "N:Anderson;Alice\r\n"
    "TEL:+1 555 1111\r\n"
    "END:VCARD\r\n"
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "UID:alice-uid\r\n"
    "EMAIL:alice@example.com\r\n"
    "TEL/BAD:+1 555 0000\r\n"
    "END:VCARD\r\n"
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "FN:Bob\r\n"
    "N:Brown;Bob\r\n"
    "END:VCARD\r\n"
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "UID:dave-uid\r\n"
    "FN:Dave\r\n"
    "TEL:+1 555 9999\r\n"
    "X-BAD/SLASH:TRUE\r\n"
    "END:VCARD\r\n"
    "@@@ totally unreadable record, line one @@@\r\n"
    "!!! totally unreadable record, line two !!!\r\n"
)


def _entry(report, field_name):
    return report.field_coverage[field_name]


def test_record_classification_and_counts():
    report = scan_vcard_quality(io.StringIO(CONTINUOUS_VCF))

    assert [record.status for record in report.records] == [
        STATUS_OK,
        STATUS_DAMAGED,
        STATUS_OK,
        STATUS_DAMAGED,
        STATUS_UNREADABLE,
    ]
    assert report.total_records == 5
    assert report.ok_count == 2
    assert report.damaged_count == 2
    assert report.unreadable_count == 1

    # The damaged duplicate keeps a recognizable UID; the unreadable run has
    # none, and the other damaged card has a UID with no ok counterpart.
    assert report.records[1].uid == "alice-uid"
    assert report.records[3].uid == "dave-uid"
    assert report.records[4].uid is None
    assert report.records[4].unknown_line_count == 2


def test_determinate_contacts_and_bounds():
    report = scan_vcard_quality(io.StringIO(CONTINUOUS_VCF))

    # Only the two ok cards are certain: Alice (one UID) and Bob (no UID).
    assert report.determinate_contacts == 2
    # The two unattributed records (#4 damaged, #5 unreadable) could be
    # duplicates/junk or each a new contact; the UID-matched duplicate #2 is
    # already fixed to Alice and never adds a contact.
    assert report.contact_lower_bound == 2
    assert report.contact_upper_bound == 4


def test_damaged_card_legal_fields_still_count():
    report = scan_vcard_quality(io.StringIO(CONTINUOUS_VCF))

    # Record #2 is damaged but its EMAIL line parsed and its UID anchors it to
    # the known Alice contact, so EMAIL coverage is certain even though the
    # contact line itself (TEL/BAD) never parses.
    assert report.records[1].fields_present == {"UID", "EMAIL"}
    assert report.records[1].unknown_line_count == 1
    assert _entry(report, "EMAIL")["lower_count"] == 1
    assert _entry(report, "EMAIL")["upper_count"] == 3

    # Alice already had TEL; the broken contact line cannot add anything.
    assert _entry(report, "TEL")["lower_count"] == 1
    assert _entry(report, "TEL")["upper_count"] == 3


def test_field_coverage_bounds():
    report = scan_vcard_quality(io.StringIO(CONTINUOUS_VCF))
    # Four possible contacts is the upper-bound population.
    assert report.contact_upper_bound == 4

    expected = {
        "FN": (2, 4),
        "N": (2, 4),
        "TEL": (1, 3),
        "EMAIL": (1, 3),
        "ADR": (0, 3),
        "ORG": (0, 3),
        "UID": (1, 2),
    }
    for field_name in TRACKED_FIELDS:
        entry = _entry(report, field_name)
        assert (entry["lower_count"], entry["upper_count"]) == expected[field_name]
        assert entry["lower_ratio"] == pytest.approx(entry["lower_count"] / 4)
        assert entry["upper_ratio"] == pytest.approx(entry["upper_count"] / 4)
        assert entry["lower_ratio"] <= entry["upper_ratio"]

    # An unparseable line can never establish identity: UID stays bounded at
    # the two parsed UIDs rather than the full population.
    assert "UID" not in POSSIBLE_FROM_UNKNOWN_LINE


def test_bound_attribution_examples_are_legal():
    report = scan_vcard_quality(io.StringIO(CONTINUOUS_VCF))

    assert report.lower_bound_example is not None
    assert report.upper_bound_example is not None
    # One concrete attribution per bound, referencing an actual record.
    assert "record #4" in report.lower_bound_example
    # Dave's UID matches no ok card, so it cannot merge into Alice (whose UID
    # differs); the legal zero-contact attribution is the uid-less Bob (#3).
    assert "contact #3" in report.lower_bound_example
    assert "record #4" in report.upper_bound_example
    assert "new contact" in report.upper_bound_example
    # Both example attributions must be realizable: the bounds they imply
    # match the numeric report.
    assert report.contact_lower_bound == report.determinate_contacts
    assert report.contact_upper_bound - report.contact_lower_bound == 2


def test_unrecognized_uid_card_cannot_merge_when_all_contacts_have_uid():
    # Every ok card carries a UID, so a damaged card with a different UID has
    # no legal merge target: the lower-bound attribution must discard it.
    raw = (
        "BEGIN:VCARD\r\nVERSION:3.0\r\nUID:alice-uid\r\nFN:Alice\r\nEND:VCARD\r\n"
        "BEGIN:VCARD\r\nVERSION:3.0\r\nUID:carol-uid\r\nFN:Carol\r\nX-BAD/SLASH:1\r\nEND:VCARD\r\n"
    )
    report = scan_vcard_quality(io.StringIO(raw))

    assert report.determinate_contacts == 1
    assert report.contact_lower_bound == 1
    assert report.contact_upper_bound == 2
    assert "record #2" in report.lower_bound_example
    assert "discarded" in report.lower_bound_example
    # Carol's parsed FN still reaches the upper-bound population.
    assert _entry(report, "FN") == {
        "lower_count": 1,
        "upper_count": 2,
        "lower_ratio": 0.5,
        "upper_ratio": 1.0,
    }


def test_clean_stream_has_collapsed_bounds():
    clean = (
        "BEGIN:VCARD\r\nVERSION:3.0\r\nUID:u1\r\nFN:Alice\r\nEND:VCARD\r\n"
        "BEGIN:VCARD\r\nVERSION:3.0\r\nUID:u2\r\nFN:Bob\r\nTEL:123\r\nEND:VCARD\r\n"
    )
    report = scan_vcard_quality(io.StringIO(clean))

    assert report.ok_count == 2
    assert report.damaged_count == 0
    assert report.unreadable_count == 0
    assert report.contact_lower_bound == report.contact_upper_bound == 2
    for field_name in TRACKED_FIELDS:
        entry = _entry(report, field_name)
        assert entry["lower_count"] == entry["upper_count"]
        assert entry["lower_ratio"] == pytest.approx(entry["upper_ratio"])
    assert _entry(report, "FN")["lower_ratio"] == pytest.approx(1.0)
    assert _entry(report, "TEL")["lower_ratio"] == pytest.approx(0.5)


def test_unclosed_card_does_not_swallow_following_cards():
    raw = (
        "BEGIN:VCARD\r\nVERSION:3.0\r\nUID:lost\r\nFN:Lost\r\n"
        "BEGIN:VCARD\r\nVERSION:3.0\r\nUID:kept\r\nFN:Kept\r\nEND:VCARD\r\n"
    )
    report = scan_vcard_quality(io.StringIO(raw))

    assert [record.status for record in report.records] == [STATUS_DAMAGED, STATUS_OK]
    assert report.records[0].uid == "lost"
    assert report.records[1].uid == "kept"
    assert report.determinate_contacts == 1
    assert (report.contact_lower_bound, report.contact_upper_bound) == (1, 2)


def test_only_unreadable_input():
    report = scan_vcard_quality("@@@ junk @@@\r\n!!! more junk !!!\r\n")

    assert report.total_records == 1
    assert report.records[0].status == STATUS_UNREADABLE
    assert report.determinate_contacts == 0
    assert report.contact_lower_bound == 0
    assert report.contact_upper_bound == 1
    # Nothing about the contents is observable, so every field starts at zero.
    for field_name in TRACKED_FIELDS:
        assert _entry(report, field_name)["lower_count"] == 0
    # And an unreadable record can never contribute a UID.
    assert _entry(report, "UID")["upper_count"] == 0
    assert _entry(report, "FN")["upper_count"] == 1


def test_report_renders_text():
    report = scan_vcard_quality(io.StringIO(CONTINUOUS_VCF))
    text = report.format_text()

    assert "determinate contacts: 2" in text
    assert "contact count bounds: 2 .. 4" in text
    assert "damaged" in text
    assert "unreadable" in text
    assert "FN   : 2..4" in text
    assert "UID  : 1..2" in text
