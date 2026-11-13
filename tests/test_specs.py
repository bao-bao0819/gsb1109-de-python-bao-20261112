from copy import deepcopy

import pytest

from anytree import AnyNode, RenderTree
from anytree.exporter import DictExporter
from anytree.importer import DictImporter
from anytree.specs import (
    DEFINITIONS_ATTR,
    OVERRIDES_ATTR,
    SPEC_ATTR,
    SPEC_ID_ATTR,
    from_definitions,
    is_modern,
    to_definitions,
)

from .helper import eq_


def _legacy_tree():
    return {
        "name": "catalog",
        "children": [
            {"name": "cat-a", "spec_id": "S1", "voltage": 230, "pin": 2},
            {"name": "cat-b", "spec_id": "S1", "voltage": 230, "pin": 2},
            {
                "name": "group",
                "children": [
                    {"name": "cat-c", "spec_id": "S1", "voltage": 110, "pin": 2},
                    {"name": "cat-d", "spec_id": "S1", "voltage": 230},
                    {"name": "cat-e", "spec_id": "S1", "voltage": 230, "pin": None},
                ],
            },
            {"name": "cat-x", "voltage": 230, "pin": 2},
            {"name": "cat-y", "voltage": 230, "pin": 2},
        ],
    }


def test_merge_compatible_copies_into_definition():
    """Same stable id and compatible content becomes one shared definition."""
    modern = to_definitions(_legacy_tree())
    eq_(modern[DEFINITIONS_ATTR], {"S1": {"voltage": 230, "pin": 2}})

    cat_a = modern["root"]["children"][0]
    cat_b = modern["root"]["children"][1]
    eq_(cat_a, {"name": "cat-a", SPEC_ATTR: "S1"})
    eq_(cat_b, {"name": "cat-b", SPEC_ATTR: "S1"})


def test_explicit_override_stays_on_original_node():
    """A diverging value is kept locally instead of corrupting the definition."""
    modern = to_definitions(_legacy_tree())
    cat_c = modern["root"]["children"][2]["children"][0]
    eq_(cat_c, {"name": "cat-c", SPEC_ATTR: "S1", OVERRIDES_ATTR: {"voltage": 110}})


def test_default_inheritance_versus_explicit_empty_value():
    """Missing attribute inherits; explicit None blocks inheritance."""
    modern = to_definitions(_legacy_tree())
    group = modern["root"]["children"][2]["children"]
    cat_d, cat_e = group[1], group[2]

    # cat-d omits `pin`: pure reference, inherits the default later.
    eq_(cat_d, {"name": "cat-d", SPEC_ATTR: "S1"})
    # cat-e sets pin=None explicitly: the empty value is an override.
    eq_(cat_e, {"name": "cat-e", SPEC_ATTR: "S1", OVERRIDES_ATTR: {"pin": None}})

    materialized = from_definitions(modern)
    leaves = {
        node["name"]: node
        for node in materialized["children"][2]["children"]
    }
    eq_(leaves["cat-d"]["pin"], 2)
    eq_(leaves["cat-e"]["pin"], None)
    assert "pin" in leaves["cat-e"]


def test_copies_without_stable_id_are_never_shared():
    """Identical-looking nodes without a stable id stay inline and independent."""
    legacy = _legacy_tree()
    modern = to_definitions(deepcopy(legacy))
    cat_x = modern["root"]["children"][3]
    cat_y = modern["root"]["children"][4]

    eq_(cat_x, {"name": "cat-x", "voltage": 230, "pin": 2})
    eq_(cat_y, {"name": "cat-y", "voltage": 230, "pin": 2})
    assert SPEC_ATTR not in cat_x and SPEC_ATTR not in cat_y
    assert modern[DEFINITIONS_ATTR] == {"S1": {"voltage": 230, "pin": 2}}

    cat_x["voltage"] = 999
    assert cat_y["voltage"] == 230


def test_roundtrip_is_reversible():
    """Modern -> legacy -> tree reproduces the effective attributes per node."""
    legacy = _legacy_tree()
    modern = to_definitions(deepcopy(legacy))
    expanded = from_definitions(modern)
    root = DictImporter().import_(expanded)

    # cat-d omitted `pin` in the legacy source; as a default inheritor it
    # effectively receives the shared value, so the re-exported legacy view
    # carries pin=2. Every other node matches its source exactly.
    expected = deepcopy(legacy)
    expected["children"][2]["children"][1]["pin"] = 2
    eq_(DictExporter().export(root), expected)

    # Fully specified copies are restored byte-for-byte, including the id.
    exact = DictExporter().export(root)["children"][0]
    eq_(exact, legacy["children"][0])


def test_input_is_not_mutated():
    legacy = _legacy_tree()
    snapshot = deepcopy(legacy)
    to_definitions(legacy)
    eq_(legacy, snapshot)


def test_global_change_reaches_only_inheritors():
    """Editing the shared definition changes inheritors, not local overrides."""
    modern = to_definitions(_legacy_tree())
    modern[DEFINITIONS_ATTR]["S1"]["voltage"] = 999
    modern[DEFINITIONS_ATTR]["S1"]["pin"] = 9

    materialized = from_definitions(modern)
    by_name = {}
    for child in materialized["children"]:
        for node in [child, *child.get("children", [])]:
            by_name[node["name"]] = node

    eq_(by_name["cat-a"]["voltage"], 999)
    eq_(by_name["cat-a"]["pin"], 9)
    eq_(by_name["cat-c"]["voltage"], 110)
    eq_(by_name["cat-c"]["pin"], 9)
    eq_(by_name["cat-d"]["voltage"], 999)
    eq_(by_name["cat-d"]["pin"], 9)
    eq_(by_name["cat-e"]["voltage"], 999)
    eq_(by_name["cat-e"]["pin"], None)


def test_export_legacy_format_reflects_effective_attributes():
    """Each imported category node exposes the attributes it actually sees."""
    modern = to_definitions(_legacy_tree())
    root = DictImporter().import_(from_definitions(modern))
    nodes = [repr(node) for _, _, node in RenderTree(root)]
    assert "AnyNode(name='cat-c', pin=2, spec_id='S1', voltage=110)" in nodes
    assert "AnyNode(name='cat-d', pin=2, spec_id='S1', voltage=230)" in nodes
    assert "AnyNode(name='cat-e', pin=None, spec_id='S1', voltage=230)" in nodes
    assert "AnyNode(name='cat-x', pin=2, voltage=230)" in nodes


def test_modern_format_idempotent_and_detected():
    modern = to_definitions(_legacy_tree())
    assert is_modern(modern)
    assert not is_modern(_legacy_tree())

    again = to_definitions(modern)
    eq_(again, modern)
    again[DEFINITIONS_ATTR]["S1"]["voltage"] = 1
    assert modern[DEFINITIONS_ATTR]["S1"]["voltage"] == 230


def test_unknown_spec_id_raises():
    modern = to_definitions(_legacy_tree())
    modern["root"]["children"][0][SPEC_ATTR] = "MISSING"
    with pytest.raises(KeyError):
        from_definitions(modern)


def test_overrides_without_reference_raises():
    modern = to_definitions(_legacy_tree())
    modern["root"]["children"][3][OVERRIDES_ATTR] = {"voltage": 1}
    with pytest.raises(ValueError):
        from_definitions(modern)


def test_invalid_definitions_raise():
    modern = to_definitions(_legacy_tree())
    modern[DEFINITIONS_ATTR]["S1"] = [("voltage", 230)]
    with pytest.raises(ValueError):
        from_definitions(modern)

    modern = to_definitions(_legacy_tree())
    del modern["root"]
    with pytest.raises(ValueError):
        from_definitions(modern)


def test_unhashable_spec_id_raises():
    legacy = {"name": "root", "children": [{"name": "bad", SPEC_ID_ATTR: ["S1"], "voltage": 1}]}
    with pytest.raises(TypeError):
        to_definitions(legacy)


def test_custom_id_and_node_attributes():
    legacy = {
        "code": "root",
        "children": [
            {"code": "a", "sku": "P1", "weight": 5},
            {"code": "b", "sku": "P1", "weight": 5},
            {"code": "c", "sku": "P1", "weight": 7},
        ],
    }
    modern = to_definitions(legacy, spec_id_attr="sku", node_attrs=("code",))
    eq_(modern[DEFINITIONS_ATTR], {"P1": {"weight": 5}})
    eq_(modern["root"]["children"][0], {"code": "a", SPEC_ATTR: "P1"})
    eq_(modern["root"]["children"][2], {"code": "c", SPEC_ATTR: "P1", OVERRIDES_ATTR: {"weight": 7}})
    eq_(from_definitions(modern), legacy)


def test_strict_value_equality():
    """True and 1 are distinct values and must not be merged as compatible."""
    legacy = {
        "name": "root",
        "children": [
            {"name": "a", "spec_id": "S1", "flag": True},
            {"name": "b", "spec_id": "S1", "flag": 1},
        ],
    }
    modern = to_definitions(legacy)
    eq_(modern["root"]["children"][0], {"name": "a", SPEC_ATTR: "S1"})
    eq_(modern["root"]["children"][1], {"name": "b", SPEC_ATTR: "S1", OVERRIDES_ATTR: {"flag": 1}})
    eq_(from_definitions(modern), legacy)


def test_none_override_roundtrip_through_real_nodes():
    legacy = {
        "name": "root",
        "children": [
            {"name": "a", "spec_id": "S1", "color": "red", "note": "x"},
            {"name": "b", "spec_id": "S1", "color": "red", "note": None},
        ],
    }
    modern = to_definitions(legacy)
    root = DictImporter().import_(from_definitions(modern))
    a_node, b_node = root.children
    assert isinstance(a_node, AnyNode)
    eq_(a_node.note, "x")
    eq_(b_node.note, None)
    assert hasattr(b_node, "note")
    eq_(DictExporter().export(root), legacy)


def test_explicit_none_spec_id_is_preserved():
    """An explicit spec_id=None is data and must not be treated as missing."""
    legacy = {"name": "root", "children": [{"name": "a", "spec_id": None, "voltage": 1}]}
    modern = to_definitions(deepcopy(legacy))
    eq_(modern[DEFINITIONS_ATTR], {})
    eq_(modern["root"]["children"][0], {"name": "a", "spec_id": None, "voltage": 1})
    assert "_spec_meta" not in modern
    eq_(from_definitions(modern), legacy)


def test_tree_without_any_spec_has_no_meta_and_is_unchanged():
    legacy = {"name": "root", "children": [{"name": "a", "voltage": 1}]}
    modern = to_definitions(deepcopy(legacy))
    eq_(modern[DEFINITIONS_ATTR], {})
    assert "_spec_meta" not in modern
    eq_(from_definitions(modern), legacy)
