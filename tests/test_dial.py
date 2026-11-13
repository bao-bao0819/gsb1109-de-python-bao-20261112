"""Tests for PBX dial sequence export from vCard TEL properties."""

import io

import pytest

import vobject
from vobject.dial import AmbiguousDial, DialError, DialSequence, PbxRules, export_dial_sequences


def parse_card(lines):
    body = "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Test Contact\r\n" + "\r\n".join(lines) + "\r\nEND:VCARD\r\n"
    return vobject.readOne(io.StringIO(body))


def export(lines, rules):
    return export_dial_sequences(parse_card(lines), rules)


def test_outside_prefix_is_prepended_to_plain_number():
    rules = PbxRules(outside_prefix="9")
    result = export(["TEL:5550100"], rules)
    assert result.is_clear
    assert len(result.sequences) == 1
    sequence = result.sequences[0]
    assert isinstance(sequence, DialSequence)
    assert sequence.sequence == "95550100"
    assert sequence.number == "5550100"
    assert sequence.extension is None


def test_grouped_x_extension_disambiguates_number_and_extension():
    rules = PbxRules(outside_prefix="9", extension_lengths=(4,),
                     home_country_code="1", local_area_codes=("617",))
    result = export([
        "work.TEL:+1-617-555-0100",
        "work.X-EXTENSION:1234",
    ], rules)
    assert result.is_clear
    sequence = result.sequences[0]
    assert sequence.sequence == "95550100p1234"
    assert sequence.number == "5550100"
    assert sequence.extension == "1234"
    assert any("X-EXTENSION" in source for source in sequence.sources)


def test_ungrouped_x_extension_is_not_associated_with_tel():
    rules = PbxRules(outside_prefix="9")
    result = export([
        "TEL:5550100",
        "X-EXTENSION:1234",
    ], rules)
    assert result.is_clear
    sequence = result.sequences[0]
    assert sequence.sequence == "95550100"
    assert sequence.extension is None


def test_tel_uri_ext_and_postd_parameters():
    rules = PbxRules(outside_prefix="9", home_country_code="1")
    result = export([r"TEL;VALUE=uri:tel:+1-555-555-0100\;ext=4321\;postd=pp2"], rules)
    assert result.is_clear
    sequence = result.sequences[0]
    assert sequence.sequence == "95555550100pp2p4321"
    assert sequence.number == "5555550100"
    assert sequence.postdial == "pp2"
    assert sequence.extension == "4321"
    assert any("ext=" in source for source in sequence.sources)
    assert any("postd=" in source for source in sequence.sources)


def test_tel_ext_parameter_disambiguates_without_bare_number_alternative():
    rules = PbxRules(outside_prefix="9", extension_lengths=(4,))
    result = export(["TEL;EXT=1234:55501001234"], rules)
    assert result.is_clear
    sequence = result.sequences[0]
    assert sequence.sequence == "95550100p1234"
    assert sequence.extension == "1234"
    assert any("EXT=" in source for source in sequence.sources)


def test_inline_text_extension_marker():
    rules = PbxRules(outside_prefix="9", extension_lengths=(4,))
    result = export(["TEL:555 0100 ext. 1234"], rules)
    assert result.is_clear
    sequence = result.sequences[0]
    assert sequence.sequence == "95550100p1234"
    assert sequence.extension == "1234"
    assert any("inline extension marker" in source for source in sequence.sources)


def test_compact_inline_extension_marker():
    rules = PbxRules(outside_prefix="9")
    result = export(["TEL:5550100x1234"], rules)
    sequence = result.sequences[0]
    assert sequence.sequence == "95550100p1234"
    assert sequence.extension == "1234"


def test_letter_control_between_digit_runs_is_postdial():
    rules = PbxRules(outside_prefix="9")
    result = export(["TEL:5550100p1234w99"], rules)
    sequence = result.sequences[0]
    assert sequence.number == "5550100"
    assert sequence.postdial == "p1234w99"
    assert sequence.sequence == "95550100p1234w99"


def test_pause_and_wait_markers_are_not_number_digits():
    rules = PbxRules(outside_prefix="9")
    result = export([r"TEL:5550100\,12\;34"], rules)
    assert result.is_clear
    sequence = result.sequences[0]
    assert sequence.number == "5550100"
    assert sequence.postdial == "p12w34"
    assert sequence.sequence == "95550100p12w34"


def test_custom_pause_and_wait_characters():
    rules = PbxRules(outside_prefix="9", pause_char=",", wait_char=";")
    result = export([r"TEL:5550100\,12\;34"], rules)
    sequence = result.sequences[0]
    assert sequence.sequence == "95550100,12;34"


def test_ambiguous_digit_run_returns_conflicting_strings_and_sources():
    rules = PbxRules(outside_prefix="9", extension_lengths=(4,))
    result = export(["TEL:55501001234"], rules)
    assert not result.is_clear
    assert result.sequences == []
    ambiguity = result.ambiguities[0]
    assert isinstance(ambiguity, AmbiguousDial)
    assert ambiguity.sequences == ["955501001234", "95550100p1234"]
    full_sources, split_sources = ambiguity.interpretations
    assert full_sources[0] == "955501001234"
    assert split_sources[0] == "95550100p1234"
    assert any("subscriber number" in source for source in full_sources[1])
    assert any("last 4 TEL digits" in source for source in split_sources[1])


def test_internal_extension_versus_outside_call_is_ambiguous():
    rules = PbxRules(outside_prefix="9", extension_lengths=(4,))
    result = export(["TEL:1234"], rules)
    assert not result.is_clear
    assert result.ambiguities[0].sequences == ["91234", "1234"]


def test_interpretations_that_dial_identically_collapse():
    rules = PbxRules(outside_prefix="9", extension_lengths=(4,))
    result = export([
        "work.TEL;EXT=1234:5550100",
        "work.X-EXTENSION:1234",
    ], rules)
    assert result.is_clear
    assert len(result.sequences) == 1
    sequence = result.sequences[0]
    assert sequence.sequence == "95550100p1234"
    assert any("TEL parameter EXT=" in source for source in sequence.sources)
    assert any("X-EXTENSION" in source for source in sequence.sources)


def test_conflicting_explicit_extension_sources_go_to_manual_review():
    rules = PbxRules(outside_prefix="9", extension_lengths=(4,))
    result = export([
        "work.TEL;EXT=1234:5550100",
        "work.X-EXTENSION:9999",
    ], rules)
    assert not result.is_clear
    assert result.ambiguities[0].sequences == ["95550100p1234", "95550100p9999"]


def test_local_area_code_and_home_country_code_stripping():
    rules = PbxRules(outside_prefix="9", home_country_code="1", local_area_codes=("617", "415"))
    national = export(["TEL:6175550100"], rules).sequences[0]
    assert national.sequence == "95550100"
    international = export([r"TEL;VALUE=uri:tel:+1-617-555-0100"], rules).sequences[0]
    assert international.sequence == "95550100"
    untouched = export(["TEL:2125550100"], rules).sequences[0]
    assert untouched.sequence == "92125550100"


def test_trunk_prefix_in_parentheses_is_removed_from_international_numbers():
    rules = PbxRules(outside_prefix="9")
    result = export(["TEL:+49 (0)30 5555 0000"], rules).sequences[0]
    assert result.number == "493055550000"
    assert result.sequence == "9493055550000"


def test_vanity_letters_map_to_keypad_digits():
    rules = PbxRules(outside_prefix="9")
    result = export(["TEL:1-800-FLOWERS"], rules).sequences[0]
    assert result.sequence == "918003569377"


def test_multiple_tel_lines_keep_document_order():
    rules = PbxRules(outside_prefix="9")
    result = export([
        "TEL:5550100",
        "TEL:5550200",
    ], rules)
    assert [item.sequence for item in result.sequences] == ["95550100", "95550200"]


def test_round_trip_preserves_dial_sequence():
    rules = PbxRules(outside_prefix="9", home_country_code="1")
    card = parse_card([
        r"TEL:5550100\,12",
        r"TEL;VALUE=uri:tel:+1-555-555-0100\;ext=4321",
    ])
    before = export_dial_sequences(card, rules)
    reparsed = vobject.readOne(card.serialize())
    after = export_dial_sequences(reparsed, rules)
    assert [item.sequence for item in after.sequences] == [item.sequence for item in before.sequences]
    assert all("p" not in item.number and "w" not in item.number for item in after.sequences)


def test_invalid_telephone_values_raise():
    rules = PbxRules(outside_prefix="9")
    with pytest.raises(DialError):
        export(["TEL:555!0100"], rules)
    with pytest.raises(DialError):
        export(["TEL:---"], rules)


def test_invalid_rules_raise():
    with pytest.raises(DialError):
        PbxRules(outside_prefix="9,")
    with pytest.raises(DialError):
        PbxRules(extension_lengths=(0, -2))
    with pytest.raises(DialError):
        export_dial_sequences(parse_card(["TEL:5550100"]), ("9",))


def test_grouped_extension_property_rejects_control_markers():
    rules = PbxRules(outside_prefix="9")
    with pytest.raises(DialError):
        export([
            "work.TEL:5550100",
            r"work.X-EXTENSION:1234\,99",
        ], rules)
