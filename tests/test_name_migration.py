"""Tests for vobject.name_migration."""

import json

import pytest

from vobject.name_migration import (
    CANDIDATE_CONVENTIONS,
    AmbiguousConventionError,
    NeedsGuessError,
    NewName,
    OldName,
    downgrade,
    export_for_old_system,
    find_conflicts,
    format_line,
    from_vcard_name,
    infer_convention,
    main,
    matching_conventions,
    new_name_from_n_value,
    new_name_to_n_value,
    parse_line,
    sync_from_old_system,
    to_vcard_name,
    upgrade,
    upgrade_all,
)
from vobject.vcard import Name

# ---------------------------------------------------------------------------
# upgrade: old -> new, no guessing
# ---------------------------------------------------------------------------


def test_upgrade_maps_missing_empty_and_scalar():
    old = OldName(family="Smith", given=None, additional="", prefix="Dr.", suffix=None)
    new = upgrade(old)
    assert new == NewName(
        family=("Smith",), given=(), additional=("",), prefix=("Dr.",), suffix=()
    )


@pytest.mark.parametrize("value", ["Smith,Jones", "a;b", "back\\slash", "x,y;z\\"])
def test_upgrade_rejects_ambiguous_scalars(value):
    with pytest.raises(NeedsGuessError) as excinfo:
        upgrade(OldName(family=value))
    (issue,) = excinfo.value.issues
    assert issue.component == "family"
    assert issue.value == value
    assert issue.reason


def test_upgrade_collects_all_ambiguous_components():
    with pytest.raises(NeedsGuessError) as excinfo:
        upgrade(OldName(family="a,b", suffix="x;y"))
    assert {i.component for i in excinfo.value.issues} == {"family", "suffix"}


def test_upgrade_all_splits_migrated_and_skipped():
    records = {
        "ok": OldName(family="Smith"),
        "bad": OldName(given="a,b"),
    }
    result = upgrade_all(records)
    assert result.migrated == {"ok": NewName(family=("Smith",))}
    assert list(result.skipped) == ["bad"]
    assert result.skipped["bad"][0].component == "given"


# ---------------------------------------------------------------------------
# downgrade and round trip
# ---------------------------------------------------------------------------


def test_downgrade_rules():
    assert downgrade(NewName()) == OldName()
    assert downgrade(NewName(family=("Smith",))) == OldName(family="Smith")
    assert downgrade(NewName(additional=("",))) == OldName(additional="")
    assert downgrade(NewName(given=("John", "Jack"))) == OldName(given="John,Jack")


@pytest.mark.parametrize(
    "old",
    [
        OldName(),
        OldName(family="Smith", given="John", additional="A.", prefix="Dr.", suffix="III"),
        OldName(family="", given=None, additional="", prefix=None, suffix=""),
        OldName(family="van der Berg", given="Anne Marie"),
    ],
)
def test_round_trip_clean_records(old):
    assert downgrade(upgrade(old)) == old


# ---------------------------------------------------------------------------
# conflict detection
# ---------------------------------------------------------------------------


def test_find_conflicts_reports_components_and_originals():
    records = {
        "one": NewName(family=("Smith,Jones",)),
        "two": NewName(family=("Smith", "Jones")),
        "three": NewName(family=("Other",)),
    }
    (conflict,) = find_conflicts(records)
    assert conflict.downgraded == OldName(family="Smith,Jones")
    assert conflict.components == ("family",)
    assert set(conflict.sources) == {
        NewName(family=("Smith,Jones",)),
        NewName(family=("Smith", "Jones")),
    }
    assert set(conflict.keys) == {"one", "two"}


def test_find_conflicts_ignores_identical_sources():
    records = {
        "a": NewName(given=("John", "Jack")),
        "b": NewName(given=("John", "Jack")),
    }
    assert find_conflicts(records) == []


# ---------------------------------------------------------------------------
# export snapshot and write-back sync
# ---------------------------------------------------------------------------


def test_sync_restores_untouched_rows_with_original_semantics():
    records = {
        "multi": NewName(given=("John", "Jack")),  # downgrades lossily to "John,Jack"
        "plain": NewName(family=("Smith",)),
    }
    snapshot = export_for_old_system(records)
    written_back = {key: old for key, (old, _) in snapshot.originals.items()}
    result = sync_from_old_system(written_back, snapshot)
    assert result.restored == records
    assert result.remapped == {}
    assert result.skipped == {}
    assert result.removed == ()


def test_sync_reupgrades_edits_and_reports_problems():
    records = {"keep": NewName(family=("Smith",)), "edit": NewName(family=("Old",))}
    snapshot = export_for_old_system(records)
    written_back = {
        "edit": OldName(family="New", given="Jane"),  # edited
        "bad": OldName(family="a,b"),  # added by the old system, ambiguous
    }
    result = sync_from_old_system(written_back, snapshot)
    assert result.remapped["edit"] == NewName(family=("New",), given=("Jane",))
    assert list(result.skipped) == ["bad"]
    assert result.removed == ("keep",)


# ---------------------------------------------------------------------------
# export conventions
# ---------------------------------------------------------------------------


def _convention(name):
    return next(c for c in CANDIDATE_CONVENTIONS if c.name == name)


def test_format_and_parse_sql_dump():
    convention = _convention("sql-dump")
    old = OldName(family="Doe;Jr", given="John", additional=None, prefix="", suffix="III")
    line = format_line(old, convention)
    assert line == r"Doe\;Jr;John;\N;;III"
    assert parse_line(line, convention) == old


def test_format_and_parse_quoted():
    convention = _convention("csv-quoted")
    old = OldName(family="a;b", given='say "hi"', additional="", prefix="", suffix="")
    line = format_line(old, convention)
    assert line == '"a;b";"say ""hi""";;;'
    assert parse_line(line, convention) == old


def test_convention_without_null_token_collapses_missing_to_empty():
    convention = _convention("csv-quoted")
    old = OldName(family="Smith", given=None)
    assert parse_line(format_line(old, convention), convention) == OldName(
        family="Smith", given="", additional="", prefix="", suffix=""
    )


def test_trailing_missing_columns_are_omitted():
    convention = _convention("trailing-omitted")
    old = OldName(family="Smith", given="John")
    assert format_line(old, convention) == "Smith;John"
    assert parse_line("Smith;John", convention) == old


def test_parse_rejects_extra_columns_and_bad_escapes():
    convention = _convention("vcard-backslash")
    with pytest.raises(ValueError):
        parse_line("a;b;c;d;e;f", convention)
    with pytest.raises(ValueError):
        parse_line("dangling\\", convention)


def test_missing_and_empty_stay_distinct_with_null_token():
    convention = _convention("sql-dump")
    parsed = parse_line(r"Smith;\N;;\N;III", convention)
    assert parsed == OldName(family="Smith", given=None, additional="", prefix=None, suffix="III")


def test_infer_convention_from_business_samples():
    samples = [
        (r"Doe\;Jr;John;\N;;III", OldName(family="Doe;Jr", given="John", additional=None, prefix="", suffix="III")),
        (r"\N;\N;\N;\N;\N", OldName()),
    ]
    assert infer_convention(samples).name == "sql-dump"


def test_infer_convention_ambiguous_samples():
    samples = [("Smith;John;;;", OldName(family="Smith", given="John", additional="", prefix="", suffix=""))]
    matches = matching_conventions(samples)
    assert len(matches) > 1
    with pytest.raises(AmbiguousConventionError) as excinfo:
        infer_convention(samples)
    assert excinfo.value.matches == tuple(matches)


def test_infer_convention_no_match():
    with pytest.raises(ValueError):
        infer_convention([("a;b;c;d;e;f", OldName(family="a"))])


# ---------------------------------------------------------------------------
# vCard bridge
# ---------------------------------------------------------------------------


def test_vcard_name_round_trip():
    new = NewName(family=("Smith",), given=("John", "Jack"), suffix=("III",))
    name = to_vcard_name(new)
    assert name.family == "Smith"
    assert name.given == ["John", "Jack"]
    assert name.suffix == "III"
    assert from_vcard_name(name) == new


def test_n_value_serialization_is_lossless():
    new = NewName(family=("a,b",), given=("John", "Jack"), suffix=("x;y",))
    value = new_name_to_n_value(new)
    assert value == r"a\,b;John,Jack;;;x\;y"
    assert new_name_from_n_value(value) == new


def test_n_value_parsing_matches_vobject_semantics():
    name = Name(family="Smith", given=["John", "Jack"])
    from vobject.vcard import NAME_ORDER, serializeFields

    assert new_name_to_n_value(NewName(family=("Smith",), given=("John", "Jack"))) == serializeFields(
        name, NAME_ORDER
    )


# ---------------------------------------------------------------------------
# command line interface
# ---------------------------------------------------------------------------


def test_cli_upgrade_downgrade_sync(tmp_path, capsys):
    old_csv = tmp_path / "old.csv"
    old_csv.write_text(
        "id,family,given,additional,prefix,suffix\n"
        "1,Smith,John,\\N,,III\n"
        "2,\"a,b\",Jane,\\N,\\N,\\N\n",
        encoding="utf-8",
    )
    new_json = tmp_path / "new.json"
    assert main(["upgrade", str(old_csv), str(new_json)]) == 0
    data = json.loads(new_json.read_text(encoding="utf-8"))
    assert data == {
        "1": {"family": ["Smith"], "given": ["John"], "additional": [], "prefix": [""], "suffix": ["III"]}
    }
    assert "need guessing" in capsys.readouterr().err

    # old system reads a downgrade of the migrated record, then writes back
    export_csv = tmp_path / "export.csv"
    snapshot_json = tmp_path / "snapshot.json"
    assert main(["downgrade", str(new_json), str(export_csv), "--snapshot", str(snapshot_json)]) == 0
    lines = export_csv.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id,family,given,additional,prefix,suffix"
    assert lines[1] == "1,Smith,John,\\N,,III"

    # untouched write-back recovers the original record
    synced_json = tmp_path / "synced.json"
    assert main(["sync", str(export_csv), str(snapshot_json), str(synced_json)]) == 0
    assert json.loads(synced_json.read_text(encoding="utf-8")) == data


def test_cli_downgrade_reports_conflicts(tmp_path, capsys):
    new_json = tmp_path / "new.json"
    new_json.write_text(
        json.dumps(
            {
                "a": {"family": ["Smith,Jones"]},
                "b": {"family": ["Smith", "Jones"]},
            }
        ),
        encoding="utf-8",
    )
    old_csv = tmp_path / "old.csv"
    snapshot_json = tmp_path / "snapshot.json"
    conflicts_json = tmp_path / "conflicts.json"
    exit_code = main(
        [
            "downgrade",
            str(new_json),
            str(old_csv),
            "--snapshot",
            str(snapshot_json),
            "--conflicts",
            str(conflicts_json),
        ]
    )
    assert exit_code == 2
    (report,) = json.loads(conflicts_json.read_text(encoding="utf-8"))
    assert report["components"] == ["family"]
    assert report["downgraded"]["family"] == "Smith,Jones"
    assert {tuple(s["family"]) for s in report["sources"]} == {("Smith,Jones",), ("Smith", "Jones")}
    assert "conflict" in capsys.readouterr().err
