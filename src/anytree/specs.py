"""Reversible migration between inline spec copies and shared spec definitions.

The legacy dictionary format stores a full copy of a specification on every
category node, even when several nodes describe the very same spec.  The new
format keeps each specification exactly once in a top-level ``definitions``
table and lets nodes reference it by a stable spec id, optionally carrying
explicit local overrides.

Two representations are supported:

* **Legacy format** -- plain node dictionaries, each holding a stable spec id
  and a flat inline copy of the spec attributes alongside its own category
  attributes::

      {"name": "cat-a", "spec_id": "S1", "voltage": 230, "pin": 2,
       "children": [...]}

* **New format** -- shared definitions plus references::

      {"definitions": {"S1": {"voltage": 230, "pin": 2}},
       "root": {"name": "cat-a", "spec": "S1", "children": [...]}}

Migration rules:

* The first occurrence of a stable spec id provides the canonical shared
  definition.  Later copies merge in where they are compatible: an attribute
  equal to the definition is inherited and dropped from the node.
* A diverging value stays on the original node as an explicit
  ``spec_overrides`` entry.  Attributes the definition does not know stay
  local the same way.
* A legacy copy that omits an attribute the definition provides becomes a
  default inheritor of that attribute (缺省继承).  An explicit ``None`` is
  different: it is an *empty value* override that blocks inheritance and is
  materialized back as ``None`` (显式空值).
* Nodes without a stable spec id are never shared, even when an identically
  named or identical-looking sibling exists.  They stay inline.

The migration works on plain dictionaries, so it composes with
:any:`anytree.exporter.DictExporter` and :any:`anytree.importer.DictImporter`
and stays reversible:

>>> from anytree.exporter import DictExporter
>>> from anytree.importer import DictImporter
>>> from anytree.specs import to_definitions, from_definitions
>>> legacy = {
...     "name": "catalog",
...     "children": [
...         {"name": "cat-a", "spec_id": "S1", "voltage": 230, "pin": 2},
...         {"name": "cat-b", "spec_id": "S1", "voltage": 230, "pin": 2},
...         {"name": "cat-c", "spec_id": "S1", "voltage": 110, "pin": 2},
...         {"name": "cat-x", "voltage": 230},
...     ],
... }
>>> modern = to_definitions(legacy)
>>> modern["definitions"]
{'S1': {'voltage': 230, 'pin': 2}}
>>> modern["root"]["children"][0]
{'name': 'cat-a', 'spec': 'S1'}
>>> modern["root"]["children"][2]
{'name': 'cat-c', 'spec': 'S1', 'spec_overrides': {'voltage': 110}}
>>> modern["root"]["children"][3]
{'name': 'cat-x', 'voltage': 230}
>>> roundtrip = DictExporter().export(DictImporter().import_(from_definitions(modern)))
>>> roundtrip == legacy
True
"""

SPEC_ID_ATTR = "spec_id"
SPEC_ATTR = "spec"
OVERRIDES_ATTR = "spec_overrides"
DEFINITIONS_ATTR = "definitions"
ROOT_ATTR = "root"
CHILDREN_ATTR = "children"
META_ATTR = "_spec_meta"
META_ID_ATTR = "spec_id_attr"

_STRUCTURAL_KEYS = (CHILDREN_ATTR, SPEC_ATTR, OVERRIDES_ATTR, DEFINITIONS_ATTR, META_ATTR)
DEFAULT_NODE_ATTRS = ("name",)
_MISSING = object()


def _spec_content(node, node_attrs):
    """Return the inline spec attributes of a legacy `node` dictionary."""
    reserved = set(_STRUCTURAL_KEYS) | set(node_attrs)
    return {key: value for key, value in node.items() if key not in reserved}


def _equal(left, right):
    """Compare spec values strictly, so ``True`` is not treated as ``1``."""
    return type(left) is type(right) and left == right


def _iter_nodes(node):
    """Yield `node` and all descendant dictionaries in pre-order."""
    yield node
    for child in node.get(CHILDREN_ATTR, ()):
        yield from _iter_nodes(child)


def is_modern(data):
    """Return `True` if `data` uses the shared-definition format."""
    return isinstance(data, dict) and DEFINITIONS_ATTR in data


def to_definitions(data, spec_id_attr=SPEC_ID_ATTR, node_attrs=DEFAULT_NODE_ATTRS):
    """Migrate legacy inline spec copies to shared definitions.

    Args:
        data: Legacy tree dictionary (the format consumed by
            :any:`DictImporter`) or an already migrated document, which is
            returned as a deep copy unchanged.
        spec_id_attr: Name of the attribute holding the stable spec id.
        node_attrs: Category attributes that identify the node itself and must
            never be merged into a spec definition (e.g. ``"name"``).  Every
            other non-structural attribute is treated as spec content.

    Returns:
        A new-format document with a top-level ``definitions`` table and a
        ``root`` tree whose spec-bearing nodes reference the definitions.

    Raises:
        TypeError: A stable spec id is unhashable and cannot be used as a
            definition key.
    """
    if is_modern(data):
        return _deep_copy(data)

    root = _deep_copy(data)
    node_attrs = tuple(node_attrs)
    definitions = {}
    has_specs = False

    for node in _iter_nodes(root):
        spec_id = node.pop(spec_id_attr, _MISSING)
        if spec_id is _MISSING:
            continue
        if spec_id is None:
            node[spec_id_attr] = spec_id
            continue
        has_specs = True
        if not _hashable(spec_id):
            raise TypeError(f"spec id {spec_id!r} must be hashable")

        content = _spec_content(node, node_attrs)
        if spec_id not in definitions:
            definitions[spec_id] = content
        reference = definitions[spec_id]
        overrides = {
            key: value
            for key, value in content.items()
            if key not in reference or not _equal(reference[key], value)
        }
        _replace_with_reference(node, spec_id, overrides, node_attrs)

    document = {DEFINITIONS_ATTR: definitions, ROOT_ATTR: root}
    if has_specs:
        document[META_ATTR] = {META_ID_ATTR: spec_id_attr}
    return document


def _replace_with_reference(node, spec_id, overrides, node_attrs):
    """Turn an inline legacy `node` into a spec reference in place."""
    kept = {key: node[key] for key in node_attrs if key in node}
    children = node.pop(CHILDREN_ATTR, None)
    node.clear()
    node.update(kept)
    node[SPEC_ATTR] = spec_id
    if overrides:
        node[OVERRIDES_ATTR] = overrides
    if children is not None:
        node[CHILDREN_ATTR] = children


def from_definitions(data):
    """Expand a shared-definition document back to legacy inline dictionaries.

    Each referencing node receives the materialized attributes it actually
    sees: the shared definition amended with its explicit overrides.  An
    override of ``None`` materializes as an explicit ``None`` value and blocks
    the inherited default.  Nodes without a ``spec`` reference are passed
    through untouched.

    Args:
        data: New-format document as produced by :func:`to_definitions`.

    Returns:
        A legacy tree dictionary as consumed by :any:`DictImporter`.

    Raises:
        ValueError: The document is malformed, e.g. overrides without a spec
            reference or a non-dictionary definition.
        KeyError: A node references an unknown spec id.
    """
    if not is_modern(data):
        return _deep_copy(data)

    definitions = data[DEFINITIONS_ATTR]
    _validate_definitions(definitions)
    if ROOT_ATTR not in data:
        raise ValueError(f"modern document is missing {ROOT_ATTR!r}")
    meta = data.get(META_ATTR, {})
    if not isinstance(meta, dict):
        raise ValueError(f"{META_ATTR!r} must be a dictionary")
    id_attr = meta.get(META_ID_ATTR, SPEC_ID_ATTR)

    root = _deep_copy(data[ROOT_ATTR])
    for node in _iter_nodes(root):
        _materialize(node, definitions, id_attr)
    return root


def _validate_definitions(definitions):
    if not isinstance(definitions, dict):
        raise ValueError(f"{DEFINITIONS_ATTR!r} must be a dictionary")
    for spec_id, spec in definitions.items():
        if not isinstance(spec, dict):
            raise ValueError(f"definition {spec_id!r} must be a dictionary")


def _materialize(node, definitions, id_attr):
    spec_id = node.pop(SPEC_ATTR, None)
    overrides = node.pop(OVERRIDES_ATTR, None)
    if spec_id is None:
        if overrides is not None:
            raise ValueError(f"{OVERRIDES_ATTR!r} requires a {SPEC_ATTR!r} reference")
        return
    if spec_id not in definitions:
        raise KeyError(f"unknown spec id {spec_id!r}")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError(f"{OVERRIDES_ATTR} of spec {spec_id!r} must be a dictionary")

    children = node.pop(CHILDREN_ATTR, None)
    own = dict(node)
    node.clear()
    node.update(definitions[spec_id])
    if overrides:
        node.update(overrides)
    node.update(own)
    node[id_attr] = spec_id
    if children is not None:
        node[CHILDREN_ATTR] = children


def _hashable(value):
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _deep_copy(value):
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value
