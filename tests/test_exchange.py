from datetime import datetime, timezone

import vobject
from vobject.exchange import (
    REASON_CONFLICT,
    REASON_SESSION_EXPIRED,
    REASON_UNKNOWN_CODENAME,
    ExchangeSession,
)

SOURCE = (
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "UID:real-uid-123\r\n"
    "FN:Meister Berger\r\n"
    "N:Berger;Meister;;;\r\n"
    "home.TEL;type=voice:+49 3581 123456\r\n"
    "home.LABEL:Hufenshlagel 1234\r\n"
    "work.TEL;type=fax:+49 3581 654321\r\n"
    "work.X-INTERNAL-ID:secret-42\r\n"
    "EMAIL:mb@goerlitz.de\r\n"
    "END:VCARD\r\n"
)


def make_card():
    return vobject.readOne(SOURCE)


def make_session(**kwargs):
    return ExchangeSession(["home"], token="c0ffee", **kwargs)


def make_receipt(session, tel_values, card_codename=None, group=None):
    group = group or session.group_codenames["home"]
    card_codename = card_codename or session.card_codenames["real-uid-123"]
    lines = ["BEGIN:VCARD", "VERSION:3.0", f"UID:{card_codename}"]
    lines += [f"{group}.TEL:{value}" for value in tel_values]
    lines.append("END:VCARD")
    return "\r\n".join(lines) + "\r\n"


def home_tels(card):
    return [line.value for line in card.contents["tel"] if (line.group or "").lower() == "home"]


def test_projection_hides_unauthorized_groups():
    session = make_session()
    out = session.project([make_card()])
    assert "+49 3581 123456" in out
    assert "Hufenshlagel 1234" in out
    # unauthorized group values, labels and identifiers are gone
    assert "654321" not in out
    assert "secret-42" not in out
    assert "X-INTERNAL-ID" not in out
    assert "work" not in out.lower()


def test_projection_uses_exchange_scoped_codenames():
    session = make_session()
    out = session.project([make_card()])
    codename = session.group_codenames["home"]
    assert f"{codename}.TEL" in out
    assert f"{codename}.LABEL" in out
    assert "home." not in out
    # the real UID is replaced by a per-exchange card codename
    assert "real-uid-123" not in out
    assert session.card_codenames["real-uid-123"] in out
    # codenames are only valid for this exchange
    other = ExchangeSession(["home"])
    assert other.project([make_card()]) != out
    assert other.group_codenames["home"] != codename


def test_projection_can_drop_ungrouped_properties():
    session = make_session(include_ungrouped=False)
    out = session.project([make_card()])
    assert "mb@goerlitz.de" not in out
    assert "FN" not in out
    assert "VERSION:3.0" in out
    assert "+49 3581 123456" in out


def test_collect_applies_receipt_back_to_original_card():
    card = make_card()
    session = make_session()
    session.project([card])
    report = session.collect(make_receipt(session, ["+49 3581 999"]))
    assert report.clean
    assert len(report.applied) == 1
    assert report.applied[0].group == "home"
    assert home_tels(card) == ["+49 3581 999"]


def test_collect_is_idempotent_regardless_of_order():
    card = make_card()
    session = make_session()
    session.project([card])
    receipt = make_receipt(session, ["+49 3581 111", "+49 3581 222"])
    first = session.collect(receipt)
    assert len(first.applied) == 1
    assert sorted(home_tels(card)) == ["+49 3581 111", "+49 3581 222"]
    # same receipt returned again, lines in a different order
    reshuffled = make_receipt(session, ["+49 3581 222", "+49 3581 111"])
    second = session.collect(reshuffled)
    assert second.clean
    assert not second.applied
    assert len(second.duplicates) == 1
    assert sorted(home_tels(card)) == ["+49 3581 111", "+49 3581 222"]


def test_conflicting_receipt_is_quarantined():
    card = make_card()
    session = make_session()
    session.project([card])
    session.collect(make_receipt(session, ["+49 3581 111"]))
    report = session.collect(make_receipt(session, ["+49 3581 666"]))
    assert not report.clean
    assert report.quarantined[0].reason == REASON_CONFLICT
    # the first, unambiguous receipt stays applied
    assert home_tels(card) == ["+49 3581 111"]


def test_expired_session_quarantines_receipts():
    card = make_card()
    now = datetime(2026, 11, 13, tzinfo=timezone.utc)
    session = make_session(
        valid_until=datetime(2026, 11, 12, tzinfo=timezone.utc),
        now=lambda: now,
    )
    session.project([card])
    report = session.collect(make_receipt(session, ["+49 3581 999"]))
    assert report.quarantined[0].reason == REASON_SESSION_EXPIRED
    assert home_tels(card) == ["+49 3581 123456"]


def test_closed_session_quarantines_receipts():
    card = make_card()
    session = make_session()
    session.project([card])
    session.close()
    report = session.collect(make_receipt(session, ["+49 3581 999"]))
    assert report.quarantined[0].reason == REASON_SESSION_EXPIRED
    assert home_tels(card) == ["+49 3581 123456"]


def test_foreign_exchange_codenames_are_quarantined():
    card = make_card()
    session = make_session()
    other = ExchangeSession(["home"])
    session.project([card])
    other.project([make_card()])
    # a receipt issued for a different exchange must not be applied here
    report = session.collect(make_receipt(other, ["+49 3581 999"]))
    assert report.quarantined[0].reason == REASON_UNKNOWN_CODENAME
    assert home_tels(card) == ["+49 3581 123456"]
    # same for an unknown group codename on a known card codename
    report = session.collect(make_receipt(session, ["x"], group="zz9"))
    assert report.quarantined[0].reason == REASON_UNKNOWN_CODENAME
    assert home_tels(card) == ["+49 3581 123456"]
