import io

import pytest

import vobject
from vobject.vcard_merge import (
    AmbiguousGroupError,
    merge_vcards,
    merge_vcards_with_report,
)


def card(*lines):
    body = "".join(line + "\r\n" for line in lines)
    return "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Alice Example\r\n" + body + "END:VCARD\r\n"


def reimport(text):
    return vobject.readOne(io.StringIO(text))


def pairs(text):
    """Return a list of (group, name, value) for grouped lines, in order."""
    card = reimport(text)
    return [(line.group, line.name, line.value) for line in card.lines() if line.group is not None]


def label_for(text, number):
    card = reimport(text)
    for line in card.lines():
        if line.name.upper() == "TEL" and line.value == number:
            group = line.group
            for other in card.lines():
                if other.group == group and other.name.upper() == "X-ABLABEL":
                    return other.value
    return None


BASE_TWO_PHONES = card(
    "item1.TEL;TYPE=CELL:+1-555-0100",
    "item1.X-ABLabel:mobile",
    "item2.TEL;TYPE=WORK:+1-555-0200",
    "item2.X-ABLabel:office",
)


def test_base_case_has_two_independent_phone_groups():
    assert pairs(BASE_TWO_PHONES) == [
        ("item1", "TEL", "+1-555-0100"),
        ("item2", "TEL", "+1-555-0200"),
        ("item1", "X-ABLABEL", "mobile"),
        ("item2", "X-ABLABEL", "office"),
    ]


def test_label_change_and_renumbered_group_with_changed_number():
    # Left changes item1's label only.  Right swaps the group numbers and,
    # in the group that used to be item1, changes the number keeping its label.
    left = card(
        "item1.TEL;TYPE=CELL:+1-555-0100",
        "item1.X-ABLabel:personal",
        "item2.TEL;TYPE=WORK:+1-555-0200",
        "item2.X-ABLabel:office",
    )
    right = card(
        "item2.TEL;TYPE=CELL:+1-555-0100",
        "item2.X-ABLabel:mobile",
        "item1.TEL;TYPE=WORK:+1-555-0300",
        "item1.X-ABLabel:office",
    )

    merged = merge_vcards(BASE_TWO_PHONES, left, right)
    text = merged.serialize()

    assert label_for(text, "+1-555-0100") == "personal"
    assert label_for(text, "+1-555-0300") == "office"
    assert label_for(text, "+1-555-0200") is None
    assert len(pairs(text)) == 4


def test_merge_is_symmetric_in_content():
    left = card(
        "item1.TEL;TYPE=CELL:+1-555-0100",
        "item1.X-ABLabel:personal",
        "item2.TEL;TYPE=WORK:+1-555-0200",
        "item2.X-ABLabel:office",
    )
    right = card(
        "item2.TEL;TYPE=CELL:+1-555-0100",
        "item2.X-ABLabel:mobile",
        "item1.TEL;TYPE=WORK:+1-555-0300",
        "item1.X-ABLabel:office",
    )
    first = merge_vcards(BASE_TWO_PHONES, left, right).serialize()
    second = merge_vcards(BASE_TWO_PHONES, right, left).serialize()

    def normalized(text):
        return sorted((line.group, line.name, line.value) for line in reimport(text).lines() if line.group)

    assert normalized(first) == normalized(second)


def test_reimport_keeps_labels_with_original_phone_numbers():
    left = card(
        "item1.TEL;TYPE=CELL:+1-555-0100",
        "item1.X-ABLabel:personal",
        "item2.TEL;TYPE=WORK:+1-555-0200",
        "item2.X-ABLabel:office",
    )
    right = card(
        "item2.TEL;TYPE=CELL:+1-555-0100",
        "item2.X-ABLabel:mobile",
        "item1.TEL;TYPE=WORK:+1-555-0300",
        "item1.X-ABLabel:office",
    )

    text = merge_vcards(BASE_TWO_PHONES, left, right).serialize()
    assert label_for(text, "+1-555-0100") == "personal"
    assert label_for(text, "+1-555-0300") == "office"


def test_identical_base_groups_leave_choice_to_caller():
    base = card(
        "item1.TEL:+1-555-0100",
        "item1.X-ABLabel:home",
        "item2.TEL:+1-555-0100",
        "item2.X-ABLabel:home",
    )
    # Left changes the label in the group it calls item2.  The two base
    # groups are indistinguishable, so either pairing is possible.
    left = card(
        "item1.TEL:+1-555-0100",
        "item1.X-ABLabel:home",
        "item2.TEL:+1-555-0100",
        "item2.X-ABLabel:work",
    )
    right = card(
        "item1.TEL:+1-555-0100",
        "item1.X-ABLabel:home",
        "item2.TEL:+1-555-0100",
        "item2.X-ABLabel:home",
    )

    report = merge_vcards_with_report(base, left, right)
    assert report.ambiguities
    assert report.ambiguities[0].side == "left"
    assert set(report.ambiguities[0].candidates) == {"item1", "item2"}
    # The default keeps the group-number based correspondence.
    assert len(pairs(report.serialize())) == 4


def test_caller_can_resolve_identical_groups_through_choices():
    base = card(
        "item1.TEL:+1-555-0100",
        "item1.X-ABLabel:home",
        "item2.TEL:+1-555-0100",
        "item2.X-ABLabel:home",
    )
    left = card(
        "item1.TEL:+1-555-0100",
        "item1.X-ABLabel:home",
        "item2.TEL:+1-555-0100",
        "item2.X-ABLabel:work",
    )

    report = merge_vcards_with_report(
        base, left, base, choices={("left", "item2"): "item1"}
    )
    assert [line.value for line in report.vcard.lines() if line.name == "X-ABLABEL"] == ["work", "home"]


def test_caller_resolver_callable_receives_ambiguity():
    base = card(
        "item1.TEL:+1-555-0100",
        "item1.X-ABLabel:home",
        "item2.TEL:+1-555-0100",
        "item2.X-ABLabel:home",
    )
    left = card(
        "item1.TEL:+1-555-0100",
        "item1.X-ABLabel:home",
        "item2.TEL:+1-555-0100",
        "item2.X-ABLabel:work",
    )
    seen = []

    def resolver(ambiguity):
        seen.append(ambiguity)
        return "item2"

    result = merge_vcards(base, left, base, resolver=resolver)
    assert seen and seen[0].candidates == ("item1", "item2")
    assert [line.value for line in result.lines() if line.name == "X-ABLABEL"] == ["home", "work"]


def test_strict_mode_refuses_to_guess_identical_groups():
    base = card(
        "item1.TEL:+1-555-0100",
        "item1.X-ABLabel:home",
        "item2.TEL:+1-555-0100",
        "item2.X-ABLabel:home",
    )
    left = card(
        "item1.TEL:+1-555-0100",
        "item1.X-ABLabel:home",
        "item2.TEL:+1-555-0100",
        "item2.X-ABLabel:work",
    )
    with pytest.raises(AmbiguousGroupError):
        merge_vcards(base, left, base, strict=True)


def test_new_group_from_one_side_is_kept_and_labelled():
    base = card("item1.TEL:+1-555-0100", "item1.X-ABLabel:mobile")
    left = base
    right = card(
        "item1.TEL:+1-555-0100",
        "item1.X-ABLabel:mobile",
        "item2.TEL:+1-555-0200",
        "item2.X-ABLabel:office",
    )

    merged = merge_vcards(base, left, right)
    text = merged.serialize()
    assert label_for(text, "+1-555-0100") == "mobile"
    assert label_for(text, "+1-555-0200") == "office"


def test_properties_outside_groups_also_merge():
    left = card("item1.TEL:+1-555-0100", "item1.X-ABLabel:mobile", "NOTE:left note")
    right = card("item1.TEL:+1-555-0100", "item1.X-ABLabel:mobile", "NOTE:right note")
    base = card("item1.TEL:+1-555-0100", "item1.X-ABLabel:mobile")
    report = merge_vcards_with_report(base, left, right)
    assert len(report.conflicts) == 1
    assert report.conflicts[0].property == "note"
    # Left wins deterministically; both changes remain visible to the caller.
    assert "left note" in report.serialize()


def test_both_sides_change_different_parts_of_same_group():
    base = card(
        "item1.TEL;TYPE=CELL:+1-555-0100",
        "item1.X-ABLabel:mobile",
        "item2.TEL;TYPE=WORK:+1-555-0200",
        "item2.X-ABLabel:office",
    )
    left = card(
        "item1.TEL;TYPE=CELL:+1-555-0999",
        "item1.X-ABLabel:mobile",
        "item2.TEL;TYPE=WORK:+1-555-0200",
        "item2.X-ABLabel:office",
    )
    right = card(
        "item1.TEL;TYPE=CELL:+1-555-0100",
        "item1.X-ABLabel:work",
        "item2.TEL;TYPE=WORK:+1-555-0200",
        "item2.X-ABLabel:office",
    )

    report = merge_vcards_with_report(base, left, right)
    text = report.serialize()
    assert label_for(text, "+1-555-0999") == "work"
    assert label_for(text, "+1-555-0200") == "office"
    assert not report.conflicts


def test_delete_versus_edit_reports_conflict_and_keeps_edit():
    base = card(
        "item1.TEL:+1-555-0100",
        "item1.X-ABLabel:mobile",
        "item2.TEL:+1-555-0200",
        "item2.X-ABLabel:office",
    )
    left = card("item1.TEL:+1-555-0100", "item1.X-ABLabel:mobile")
    right = card(
        "item1.TEL:+1-555-0100",
        "item1.X-ABLabel:mobile",
        "item2.TEL:+1-555-0299",
        "item2.X-ABLabel:office",
    )

    report = merge_vcards_with_report(base, left, right)
    assert report.conflicts
    assert label_for(report.serialize(), "+1-555-0299") == "office"


def test_inputs_are_not_modified():
    base = reimport(BASE_TWO_PHONES)
    before = base.serialize()
    left = reimport(BASE_TWO_PHONES)
    right = reimport(BASE_TWO_PHONES)
    merge_vcards(base, left, right)
    assert base.serialize() == before


def test_package_exports_and_vcard_module_reexport():
    assert vobject.merge_vcards is merge_vcards
    assert callable(vobject.vcard.merge_vcards)
    assert callable(vobject.vcard.merge_vcards_with_report)
    merged = vobject.vcard.merge_vcards(BASE_TWO_PHONES, BASE_TWO_PHONES, BASE_TWO_PHONES)
    assert merged.serialize() == reimport(BASE_TWO_PHONES).serialize()


def test_cli_reads_three_files(tmp_path):
    from vobject.vcard_merge import main

    base = tmp_path / "base.vcf"
    left = tmp_path / "left.vcf"
    right = tmp_path / "right.vcf"
    base.write_text(BASE_TWO_PHONES, encoding="utf-8")
    left.write_text(
        BASE_TWO_PHONES.replace("item1.X-ABLabel:mobile", "item1.X-ABLabel:personal"),
        encoding="utf-8",
    )
    right.write_text(
        card(
            "item2.TEL;TYPE=CELL:+1-555-0100",
            "item2.X-ABLabel:mobile",
            "item1.TEL;TYPE=WORK:+1-555-0300",
            "item1.X-ABLabel:office",
        ),
        encoding="utf-8",
    )
    output = tmp_path / "merged.vcf"

    assert main([str(base), str(left), str(right), "-o", str(output)]) == 0
    assert label_for(output.read_text(encoding="utf-8"), "+1-555-0100") == "personal"
    assert label_for(output.read_text(encoding="utf-8"), "+1-555-0300") == "office"
