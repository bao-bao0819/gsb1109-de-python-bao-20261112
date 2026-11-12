import pytest

import vobject
from vobject.org_roundtrip import (
    PendingChoiceError,
    apply_orgs,
    capture_orgs,
    export_orgs,
    invert_dictionary,
    split_escaped,
    standardize_orgs,
)

CARD = (
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "FN:Test User\r\n"
    "ORG:Example Corp\\, HQ;Old Sales Dept;Team A\r\n"
    "ORG:Example Corp\\, HQ;Old Sales Department;Team B\r\n"
    "END:VCARD\r\n"
)

# Two old departments map to the same standard name.
DICTIONARY = {
    "Old Sales Dept": "Sales Division",
    "Old Sales Department": "Sales Division",
    "Team A": "Alpha Team",
    "Old Marketing": "Marketing Division",
    "Temp Office": "Shared Office",
    "Hot Desk": "Shared Office",
}


def read_card():
    return vobject.readOne(CARD)


def standardize():
    return standardize_orgs(read_card(), DICTIONARY, source=CARD)


def test_split_escaped_keeps_literals():
    assert split_escaped(r"Example Corp\, HQ;Sales\;West;Team A") == ["Example Corp\\, HQ", "Sales\\;West", "Team A"]
    assert split_escaped("only") == ["only"]
    assert split_escaped("a;;b") == ["a", "", "b"]


def test_capture_preserves_path_literal_and_position():
    records = capture_orgs(read_card(), source=CARD)
    assert len(records) == 2
    assert [r.position for r in records] == [0, 1]
    assert [r.line_number for r in records] == [4, 5]
    assert records[0].raw_value == "Example Corp\\, HQ;Old Sales Dept;Team A"
    assert records[1].raw_value == "Example Corp\\, HQ;Old Sales Department;Team B"
    assert records[0].original_levels == ("Example Corp, HQ", "Old Sales Dept", "Team A")
    assert records[1].original_levels == ("Example Corp, HQ", "Old Sales Department", "Team B")


def test_capture_without_source_rebuilds_literal():
    records = capture_orgs(read_card())
    assert records[0].raw_value == "Example Corp\\, HQ;Old Sales Dept;Team A"
    assert records[0].original_levels == ("Example Corp, HQ", "Old Sales Dept", "Team A")


def test_standardize_maps_levels():
    records = standardize()
    assert records[0].levels == ["Example Corp, HQ", "Sales Division", "Alpha Team"]
    assert records[1].levels == ["Example Corp, HQ", "Sales Division", "Team B"]


def test_unchanged_levels_keep_original_values():
    # Many-to-one dictionary, no edits: each appointment exports its own
    # original literal, byte for byte.
    records = standardize()
    assert export_orgs(records, DICTIONARY) == [
        "Example Corp\\, HQ;Old Sales Dept;Team A",
        "Example Corp\\, HQ;Old Sales Department;Team B",
    ]


def test_unique_back_translation_rewrites_only_that_level():
    records = standardize()
    records[0].levels[1] = "Marketing Division"
    values = export_orgs(records, DICTIONARY)
    # Only the edited level changes; untouched levels keep their literals.
    assert values[0] == "Example Corp\\, HQ;Old Marketing;Team A"
    # The other appointment is not rewritten.
    assert values[1] == "Example Corp\\, HQ;Old Sales Department;Team B"


def test_new_standard_name_without_back_translation_exports_as_is():
    records = standardize()
    records[1].levels[2] = "Brand New Team"
    values = export_orgs(records, DICTIONARY)
    assert values[1] == "Example Corp\\, HQ;Old Sales Department;Brand New Team"


def test_ambiguous_back_translation_marks_only_that_spot():
    records = standardize()
    records[0].levels[2] = "Shared Office"
    with pytest.raises(PendingChoiceError) as excinfo:
        export_orgs(records, DICTIONARY)
    choices = excinfo.value.choices
    assert len(choices) == 1
    choice = choices[0]
    assert choice.position == 0
    assert choice.level_index == 2
    assert choice.standard_name == "Shared Office"
    assert choice.candidates == ("Temp Office", "Hot Desk")
    # The pending spot is on the edited record only.
    assert set(records[0].pending) == {2}
    assert records[1].pending == {}


def test_resolving_pending_choice_exports_without_touching_other_appointment():
    records = standardize()
    records[0].levels[2] = "Shared Office"
    with pytest.raises(PendingChoiceError):
        export_orgs(records, DICTIONARY)
    records[0].resolve(2, "Hot Desk")
    values = export_orgs(records, DICTIONARY)
    assert values[0] == "Example Corp\\, HQ;Old Sales Dept;Hot Desk"
    assert values[1] == "Example Corp\\, HQ;Old Sales Department;Team B"


def test_resolve_rejects_unknown_candidate():
    records = standardize()
    records[0].levels[2] = "Shared Office"
    with pytest.raises(PendingChoiceError):
        export_orgs(records, DICTIONARY)
    with pytest.raises(ValueError):
        records[0].resolve(2, "Old Sales Dept")
    with pytest.raises(ValueError):
        records[0].resolve(1, "Hot Desk")


def test_apply_orgs_round_trips_through_serialize():
    card = read_card()
    records = standardize_orgs(card, DICTIONARY, source=CARD)
    records[0].levels[1] = "Marketing Division"
    apply_orgs(card, records, DICTIONARY)
    serialized = card.serialize()
    assert "ORG:Example Corp\\, HQ;Old Marketing;Team A\r\n" in serialized
    assert "ORG:Example Corp\\, HQ;Old Sales Department;Team B\r\n" in serialized


def test_invert_dictionary_preserves_candidate_order():
    assert invert_dictionary(DICTIONARY)["Sales Division"] == ["Old Sales Dept", "Old Sales Department"]
    assert invert_dictionary(DICTIONARY)["Shared Office"] == ["Temp Office", "Hot Desk"]
