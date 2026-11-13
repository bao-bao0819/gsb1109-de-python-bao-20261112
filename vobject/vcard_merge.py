"""
Three-way merging of vCards that keeps vCard property groups together.

A vCard property group is the optional prefix before a property name, for
example ``item1.TEL`` and ``item1.X-ABLabel``.  Address Book uses these
groups to associate a telephone number with its label.  A group prefix is
only an association key; one side may renumber groups without changing any
property.

This module compares two exports with their common base.  Groups are
paired using properties that did not change: a group whose label changed
can still be recognized by its unchanged telephone number, and a group
whose number changed can be recognized by its unchanged label.  Properties
of a paired group are merged as a unit, so a label remains attached to the
telephone number it originally described.

If two base groups have exactly the same contents, an unchanged property
cannot tell them apart.  That correspondence is optional and is reported
as a L{GroupAmbiguity}; a caller supplied resolver (or an explicit
``choices`` mapping) can decide it.  Without a resolver a deterministic
fallback based on the group number is used, unless ``strict=True`` makes
the merge raise L{AmbiguousGroupError}.
"""

import difflib
import inspect
import io
import os
import sys
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass, field

from . import base
from .base import Component

__all__ = [
    "AmbiguousGroupError",
    "GroupAmbiguity",
    "MergeConflict",
    "MergeReport",
    "VCardMergeError",
    "merge",
    "merge_vcard",
    "merge_vcards",
    "merge_vcards_detailed",
    "merge_vcards_report",
    "merge_vcards_with_report",
    "three_way_merge",
]


class VCardMergeError(base.VObjectError):
    """Raised for invalid input or a failed merge."""


class AmbiguousGroupError(VCardMergeError):
    """Raised in strict mode when a group cannot be traced to a unique base group."""

    def __init__(self, ambiguities):
        self.ambiguities = list(ambiguities)
        message = "; ".join(str(item) for item in self.ambiguities)
        super().__init__(message or "ambiguous vCard group correspondence")


@dataclass(frozen=True)
class GroupAmbiguity:
    """
    A group on one side that could have come from several groups in the base.

    @ivar side: ``"left"`` or ``"right"``.
    @ivar group: the group prefix used on that side, or ``""``.
    @ivar candidates: group prefixes in the base that could be the source.
    @ivar reason: a short explanation.
    @ivar resolved: true if a caller supplied choice selected one candidate.
    """

    side: str
    group: str
    candidates: tuple
    reason: str = "more than one base group has the same unchanged properties"
    resolved: bool = False

    @property
    def candidate_ids(self):
        """Alias for L{candidates} (the values are group prefixes, not objects)."""
        return self.candidates

    def __str__(self):
        side = {"left": "left", "right": "right"}.get(self.side, self.side)
        return f"{side} group {self.group!r} could correspond to base groups {list(self.candidates)!r}"


@dataclass
class MergeConflict:
    """A property changed differently by the two sides."""

    location: str
    property: str
    base: list = field(default_factory=list)
    left: list = field(default_factory=list)
    right: list = field(default_factory=list)

    def __str__(self):
        return f"conflict for {self.property} in {self.location}"


@dataclass
class MergeReport:
    """
    Detailed result of a merge.

    The merged vCard is available as C{vcard}; conflicts and unresolved (or
    caller-resolved) group ambiguities are kept as C{conflicts} and
    C{ambiguities}.  C{unresolved} contains only the latter that no
    caller choice resolved.
    """

    vcard: Component
    conflicts: list = field(default_factory=list)
    ambiguities: list = field(default_factory=list)
    group_map: dict = field(default_factory=dict)

    @property
    def merged(self):
        return self.vcard

    @property
    def unresolved(self):
        return [ambiguity for ambiguity in self.ambiguities if not ambiguity.resolved]

    @property
    def unresolved_ambiguities(self):
        return self.unresolved

    def serialize(self, *args, **kwargs):
        return self.vcard.serialize(*args, **kwargs)

    def __getattr__(self, name):
        if name == "vcard":
            raise AttributeError(name)
        return getattr(self.vcard, name)


@dataclass
class _Group:
    key: str
    name: str
    lines: list
    keys: list
    counts: Counter

    @classmethod
    def from_lines(cls, group, lines):
        keys = [_canonical_line(line) for line in lines]
        return cls(
            key=(group or "").lower(),
            name=group or "",
            lines=list(lines),
            keys=keys,
            counts=Counter(keys),
        )

    @property
    def exact_key(self):
        return tuple(sorted(self.keys))

    def common_with(self, other):
        return sum((self.counts & other.counts).values())


@dataclass
class _Edit:
    start: int
    end: int
    keys: list
    lines: list
    side: str


# ---------------------------------------------------------------------------
# Input handling and line helpers
# ---------------------------------------------------------------------------


def _as_vcard(value):
    """Return a private copy of a vCard given a component or readable source."""
    if isinstance(value, Component):
        if value.name.upper() != "VCARD":
            raise VCardMergeError(f"expected a VCARD, got {value.name}")
        return value.duplicate(value)
    if isinstance(value, bytes):
        return _as_vcard(io.StringIO(value.decode("utf-8")))
    if isinstance(value, os.PathLike):
        return _path_vcard(value)
    if hasattr(value, "read"):
        card = base.readOne(value)
        if card.name.upper() != "VCARD":
            raise VCardMergeError(f"expected a VCARD, got {card.name}")
        return card
    if isinstance(value, str):
        if "BEGIN:VCARD" in value.upper() or "\n" in value or "\r" in value:
            return _as_vcard(io.StringIO(value))
        return _path_vcard(value)
    raise TypeError("expected a VCARD component, path, string, or readable stream")


def _path_vcard(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return _as_vcard(stream)
    except OSError as error:
        raise VCardMergeError(f"could not read vCard {path!s}: {error}") from error


def _clone_line(line):
    return line.duplicate(line)


def _canonical_line(line):
    clone = _clone_line(line)
    clone.group = None
    clone.lineNumber = None
    # Parameter values such as TYPE=CELL,WORK and TYPE=WORK,CELL have the
    # same meaning.  Normalize their order so that a side which only
    # reordered parameters is not mistaken for a changed property.
    clone.params = {name: sorted(values) for name, values in clone.params.items()}
    return clone.serialize(lineLength=1_000_000, validate=False).rstrip("\r\n")


def _split_groups(component):
    global_lines = []
    groups = {}
    for line in component.lines():
        if line.group is None:
            global_lines.append(line)
        else:
            groups.setdefault(line.group.lower(), []).append(line)
    groups = {key: _Group.from_lines(lines[0].group, lines) for key, lines in groups.items()}
    return global_lines, groups


def _same_keys(left, right):
    return sorted(_canonical_line(line) for line in left) == sorted(_canonical_line(line) for line in right)


# ---------------------------------------------------------------------------
# Group correspondence
# ---------------------------------------------------------------------------


def _resolver_parameters(resolver):
    try:
        signature = inspect.signature(resolver)
    except (TypeError, ValueError):
        return None
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is parameter.empty
    ]
    if any(parameter.kind == parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
        return 1
    return len(parameters)


def _invoke_resolver(resolver, ambiguity):
    count = _resolver_parameters(resolver)
    if count == 0:
        return resolver()
    if count is None:
        return resolver(ambiguity)
    if count == 1:
        return resolver(ambiguity)
    if count == 2:
        return resolver(ambiguity.candidates, ambiguity.group)
    if count == 3:
        return resolver(ambiguity.side, ambiguity.group, ambiguity.candidates)
    raise VCardMergeError("group resolver must take zero to three arguments")


def _mapping_choice(choices, side, group):
    if choices is None:
        return None
    if callable(choices):
        return choices(side, group)
    if (side, group) in choices:
        return choices[(side, group)]
    for key in ((side, group.lower()), (side, group.upper()), (side.lower(), group)):
        if key in choices:
            return choices[key]
    side_choices = choices.get(side)
    if isinstance(side_choices, dict):
        for key in (group, group.lower(), group.upper()):
            if key in side_choices:
                return side_choices[key]
    return None


def _resolve_candidate(value, candidates):
    if value is None:
        return None
    if isinstance(value, bool):
        raise VCardMergeError(f"group choice {value!r} is not a group")
    if isinstance(value, int):
        if 0 <= value < len(candidates):
            return candidates[value]
        raise VCardMergeError(f"group choice index {value} is out of range")
    if isinstance(value, str):
        for candidate in candidates:
            if candidate.name.lower() == value.lower():
                return candidate
        raise VCardMergeError(f"group choice {value!r} is not one of the candidate groups")
    if isinstance(value, _Group):
        for candidate in candidates:
            if candidate is value:
                return candidate
        raise VCardMergeError("group choice is not one of the candidate groups")
    raise VCardMergeError(f"could not understand group choice {value!r}")


def _ambiguous(side, side_group, candidates, options, ambiguities):
    """Choose one candidate, recording the ambiguity for the caller."""
    ambiguity = GroupAmbiguity(
        side=side,
        group=side_group.name,
        candidates=tuple(candidate.name for candidate in candidates),
    )

    choice = _mapping_choice(options.get("choices"), side, side_group.name)
    if choice is not None:
        selected = _resolve_candidate(choice, candidates)
        if selected is not None:
            ambiguities.append(
                GroupAmbiguity(
                    side=ambiguity.side,
                    group=ambiguity.group,
                    candidates=ambiguity.candidates,
                    reason=ambiguity.reason,
                    resolved=True,
                )
            )
            return selected

    resolver = options.get("resolver")
    if resolver is not None:
        selected = _resolve_candidate(_invoke_resolver(resolver, ambiguity), candidates)
        if selected is not None:
            ambiguities.append(
                GroupAmbiguity(
                    side=ambiguity.side,
                    group=ambiguity.group,
                    candidates=ambiguity.candidates,
                    reason=ambiguity.reason,
                    resolved=True,
                )
            )
            return selected

    if options.get("strict"):
        ambiguities.append(ambiguity)
        raise AmbiguousGroupError(ambiguities)

    ambiguities.append(ambiguity)
    same_id = [candidate for candidate in candidates if candidate.key == side_group.key]
    if len(same_id) == 1:
        return same_id[0]
    return candidates[0]


def _candidates_for(side_group, base_groups):
    """
    Return the base groups that could be the source of C{side_group}.

    Exact contents are the strongest evidence.  Otherwise unchanged
    properties are used as anchors, preferring the groups sharing the largest
    number of properties.  Only after both fail is the group prefix used.
    """
    if not base_groups:
        return []

    exact = [base_group for base_group in base_groups if base_group.exact_key == side_group.exact_key]
    if exact:
        return exact

    anchored = [base_group for base_group in base_groups if base_group.common_with(side_group) > 0]
    if anchored:
        best = max(base_group.common_with(side_group) for base_group in anchored)
        return [base_group for base_group in anchored if base_group.common_with(side_group) == best]

    same_name = [base_group for base_group in base_groups if base_group.key == side_group.key]
    return same_name


def _pair_groups(side_name, side_groups, base_groups, options, ambiguities, group_map, phase, matches=None):
    """
    Pair groups on one side with groups in the base.

    C{phase} selects which side groups are considered: ``"changed"`` means
    groups whose content is not exactly present in the base, ``"unchanged"``
    means the rest, and C{None} means all groups.  Changed groups are
    processed first by L{_merge_cards}, because their unchanged properties
    identify their source.

    C{matches} is the per-side match table carried between phases.  It is
    updated in place and returned together with the groups in this phase
    that have no base counterpart.
    """
    side_groups = list(side_groups.values()) if isinstance(side_groups, dict) else list(side_groups)
    if matches is None:
        matches = {}
    base_exact = {base_group.exact_key for base_group in base_groups}

    if phase == "changed":
        targets = [side_group for side_group in side_groups if side_group.exact_key not in base_exact]
    elif phase == "unchanged":
        targets = [side_group for side_group in side_groups if side_group.exact_key in base_exact]
    else:
        targets = list(side_groups)

    unmatched = []

    for side_group in targets:
        available = [base_group for base_group in base_groups if id(base_group) not in matches]
        candidates = _candidates_for(side_group, available)
        if not candidates:
            unmatched.append(side_group)
            continue

        if phase == "unchanged":
            same_id = [candidate for candidate in candidates if candidate.key == side_group.key]
            base_group = same_id[0] if same_id else candidates[0]
        elif len(candidates) == 1:
            base_group = candidates[0]
        else:
            base_group = _ambiguous(side_name, side_group, candidates, options, ambiguities)

        matches[id(base_group)] = side_group
        group_map[(side_name, side_group.name)] = base_group.name

    return matches, unmatched


def _pair_both_sides(left_groups, right_groups, base_groups, options, ambiguities, group_map):
    """
    Pair each side independently with the base groups.

    The two exports are independent descendants of the base, so both may
    refer to the same base group (for example when both edited the same
    telephone entry).  Base groups are therefore not consumed across sides;
    they are only consumed within one side.

    Within a side, changed groups are paired before unchanged groups.  This
    matters when the base has identical groups: an unchanged group has the
    same content as every one of them, while a changed group's unchanged
    properties identify its actual source.
    """
    left_matches, left_unmatched = _pair_groups_for_side(
        "left", left_groups, base_groups, options, ambiguities, group_map
    )
    right_matches, right_unmatched = _pair_groups_for_side(
        "right", right_groups, base_groups, options, ambiguities, group_map
    )
    return left_matches, left_unmatched, right_matches, right_unmatched


def _pair_groups_for_side(side_name, side_groups, base_groups, options, ambiguities, group_map):
    matches = {}
    unmatched = []
    for phase in ("changed", "unchanged"):
        updated_matches, phase_unmatched = _pair_groups(
            side_name,
            side_groups,
            base_groups,
            options,
            ambiguities,
            group_map,
            phase,
            matches,
        )
        matches.update(updated_matches)
        unmatched.extend(phase_unmatched)
    return matches, unmatched


# ---------------------------------------------------------------------------
# Three-way line merging
# ---------------------------------------------------------------------------


def _edits(base_keys, side_keys, side_lines, side_name):
    edits = []
    matcher = difflib.SequenceMatcher(None, base_keys, side_keys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        edits.append(_Edit(i1, i2, side_keys[j1:j2], side_lines[j1:j2], side_name))
    return edits


def _overlaps(left, right):
    if left.start == left.end and right.start == right.end:
        return left.start == right.start
    if left.start == left.end:
        return right.start <= left.start < right.end
    if right.start == right.end:
        return left.start <= right.start < left.end
    return max(left.start, right.start) < min(left.end, right.end)


def _merge_ordered_lists(base_lines, left_lines, right_lines, location, property_name, conflicts):
    """
    Merge ordered sequences of same-named properties.

    Non-overlapping edits from the two sides are combined.  Identical
    overlapping edits are applied once.  Differing overlapping edits are
    recorded as conflicts and the left side wins, giving a deterministic
    merged vCard rather than silently dropping a change.
    """
    base_keys = [_canonical_line(line) for line in base_lines]
    left_keys = [_canonical_line(line) for line in left_lines]
    right_keys = [_canonical_line(line) for line in right_lines]

    if left_keys == right_keys:
        return list(left_lines)
    if left_keys == base_keys:
        return list(right_lines)
    if right_keys == base_keys:
        return list(left_lines)

    if not base_keys:
        # Both sides independently introduced this property.  There is no
        # common ordering or value to merge, so they conflict.
        conflicts.append(
            MergeConflict(
                location=location,
                property=property_name,
                base=[],
                left=list(left_keys),
                right=list(right_keys),
            )
        )
        return list(left_lines)

    left_edits = _edits(base_keys, left_keys, left_lines, "left")
    right_edits = _edits(base_keys, right_keys, right_lines, "right")
    selected = list(left_edits)

    for right_edit in right_edits:
        conflicting = None
        for left_edit in left_edits:
            if not _overlaps(left_edit, right_edit):
                continue
            if left_edit.keys == right_edit.keys:
                conflicting = left_edit
                break
            conflicts.append(
                MergeConflict(
                    location=location,
                    property=property_name,
                    base=list(base_keys[left_edit.start : left_edit.end]),
                    left=list(left_edit.keys),
                    right=list(right_edit.keys),
                )
            )
            conflicting = left_edit
            break
        if conflicting is None:
            selected.append(right_edit)

    selected.sort(key=lambda edit: (edit.start, edit.end))
    result = []
    cursor = 0
    for edit in selected:
        if edit.start < cursor:
            continue
        result.extend(base_lines[cursor:edit.start])
        result.extend(edit.lines)
        cursor = edit.end
    result.extend(base_lines[cursor:])
    return result


def _merge_named_lines(base_lines, left_lines, right_lines, location, conflicts):
    """Merge a list of properties by name, preserving each name's edit order."""
    base_by_name = {}
    left_by_name = {}
    right_by_name = {}
    order = []
    for lines, table in ((base_lines, base_by_name), (left_lines, left_by_name), (right_lines, right_by_name)):
        for line in lines:
            name = line.name.lower()
            if name not in order:
                order.append(name)
            table.setdefault(name, []).append(line)

    merged = []
    for name in order:
        base_values = base_by_name.get(name, [])
        left_values = left_by_name.get(name)
        right_values = right_by_name.get(name)

        if name not in base_by_name:
            # Property added by one or both sides.
            if left_values is None:
                merged.extend(right_values or [])
                continue
            if right_values is None:
                merged.extend(left_values)
                continue
        else:
            left_values = left_values or []
            right_values = right_values or []

        merged.extend(_merge_ordered_lists(base_values, left_values, right_values, location, name, conflicts))
    return merged


# ---------------------------------------------------------------------------
# Group and card merging
# ---------------------------------------------------------------------------


def _merge_paired_group(base_group, left_group, right_group, conflicts):
    merged = _merge_named_lines(
        base_group.lines,
        left_group.lines,
        right_group.lines,
        f"group {base_group.name}",
        conflicts,
    )
    return merged


def _new_group_name(desired, used, number):
    candidate = desired or f"item{number}"
    if candidate.lower() not in used:
        used.add(candidate.lower())
        return candidate
    number += 1
    while True:
        candidate = f"item{number}"
        number += 1
        if candidate.lower() not in used:
            used.add(candidate.lower())
            return candidate


def _normalize_options(options):
    aliases = {
        "group_resolver": "resolver",
        "ambiguity_resolver": "resolver",
        "ambiguous_group_resolver": "resolver",
        "on_ambiguous_group": "resolver",
        "on_ambiguity": "resolver",
        "group_choices": "choices",
        "group_map": "choices",
        "raise_on_ambiguity": "strict",
    }
    normalized = dict(options)
    for alias, canonical in aliases.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized.pop(alias)
    normalized.setdefault("choices", None)
    normalized.setdefault("resolver", None)
    normalized.setdefault("strict", False)
    return normalized


def _merge_cards(base_card, left_card, right_card, options):
    conflicts = []
    ambiguities = []
    group_map = {}

    base_global, base_groups_map = _split_groups(base_card)
    left_global, left_groups_map = _split_groups(left_card)
    right_global, right_groups_map = _split_groups(right_card)
    base_groups = list(base_groups_map.values())

    left_matches, left_unmatched, right_matches, right_unmatched = _pair_both_sides(
        left_groups_map, right_groups_map, base_groups, options, ambiguities, group_map
    )

    merged_global = _merge_named_lines(base_global, left_global, right_global, "vcard", conflicts)

    output_groups = []
    used_names = set()
    for base_group in base_groups:
        left_group = left_matches.get(id(base_group))
        right_group = right_matches.get(id(base_group))

        if left_group is None and right_group is None:
            continue
        if left_group is None:
            if _same_keys(base_group.lines, right_group.lines):
                continue
            conflicts.append(
                MergeConflict(
                    location=f"group {base_group.name}",
                    property="group",
                    base=list(base_group.keys),
                    left=[],
                    right=list(right_group.keys),
                )
            )
            merged_lines = list(right_group.lines)
        elif right_group is None:
            if _same_keys(base_group.lines, left_group.lines):
                continue
            conflicts.append(
                MergeConflict(
                    location=f"group {base_group.name}",
                    property="group",
                    base=list(base_group.keys),
                    left=list(left_group.keys),
                    right=[],
                )
            )
            merged_lines = list(left_group.lines)
        else:
            merged_lines = _merge_paired_group(base_group, left_group, right_group, conflicts)

        if not merged_lines:
            continue
        name = _new_group_name(base_group.name, used_names, len(used_names) + 1)
        output_groups.append((name, merged_lines))

    additions = []
    consumed_right = set()
    for left_group in left_unmatched:
        right_same = None
        for right_group in right_unmatched:
            if id(right_group) in consumed_right:
                continue
            if right_group.exact_key == left_group.exact_key:
                right_same = right_group
                consumed_right.add(id(right_group))
                break
        additions.append((left_group, right_same))
    for right_group in right_unmatched:
        if id(right_group) not in consumed_right:
            additions.append((None, right_group))

    for left_group, right_group in additions:
        if left_group is not None and right_group is not None:
            lines = list(left_group.lines)
            name_hint = left_group.name or right_group.name
        elif left_group is not None:
            lines = list(left_group.lines)
            name_hint = left_group.name
        else:
            lines = list(right_group.lines)
            name_hint = right_group.name
        if not lines:
            continue
        name = _new_group_name(name_hint, used_names, len(used_names) + 1)
        output_groups.append((name, lines))

    version = (
        left_card.getChildValue("version")
        or right_card.getChildValue("version")
        or base_card.getChildValue("version")
        or "3.0"
    )
    merged = base.newFromBehavior("vcard", version)
    merged.contents = {}
    for line in merged_global:
        _append_line(merged, line, None)
    for name, lines in output_groups:
        for line in lines:
            _append_line(merged, line, name)

    return MergeReport(vcard=merged, conflicts=conflicts, ambiguities=ambiguities, group_map=group_map)


def _append_line(component, line, group):
    clone = _clone_line(line)
    clone.group = group
    clone.parentBehavior = component.behavior
    component.contents.setdefault(clone.name.lower(), []).append(clone)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_vcards_with_report(base_card, left_card, right_card, **options):
    """
    Merge two vCard exports relative to a common base and return a L{MergeReport}.

    C{base_card}, C{left_card} and C{right_card} may each be a VCARD component,
    a path, a string containing a vCard, or a readable stream.  They are not
    modified.

    Keyword options:

    @param choices:
        Mapping or callable used to decide ambiguous group correspondences.
        It may be keyed by C{(side, group)} or be a nested
        C{{"left": {{group: base_group}}, "right": ...}} mapping.
    @param resolver:
        Callable taking a L{GroupAmbiguity} (or its candidates/group) and
        returning a base group prefix, a candidate index, or a group object.
    @param strict:
        If true, raise L{AmbiguousGroupError} instead of falling back when a
        group is ambiguous and no resolver supplies a choice.
    """
    normalized = _normalize_options(options)
    cards = [_as_vcard(card) for card in (base_card, left_card, right_card)]
    return _merge_cards(*cards, normalized)


def merge_vcards(base_card, left_card, right_card, report=None, return_report=False, **options):
    """
    Merge two vCard exports relative to a common base and return the VCARD.

    When C{return_report} is true the L{MergeReport} is returned instead.
    An optional C{report} argument receives diagnostics: pass a mutable
    list to collect L{MergeConflict} objects, a two-element tuple of lists
    for conflicts and ambiguities, or a L{MergeReport} to update.
    """
    result = merge_vcards_with_report(base_card, left_card, right_card, **options)
    if report is not None:
        if isinstance(report, MergeReport):
            report.conflicts.extend(result.conflicts)
            report.ambiguities.extend(result.ambiguities)
        elif isinstance(report, tuple) and len(report) == 2:
            report[0].extend(result.conflicts)
            report[1].extend(result.ambiguities)
        else:
            report.extend(result.conflicts)
    if return_report:
        return result
    return result.vcard


def merge_vcards_report(base_card, left_card, right_card, **options):
    """Alias for L{merge_vcards_with_report}."""
    return merge_vcards_with_report(base_card, left_card, right_card, **options)


def merge_vcards_detailed(base_card, left_card, right_card, **options):
    """Alias for L{merge_vcards_with_report}."""
    return merge_vcards_with_report(base_card, left_card, right_card, **options)


def merge_vcard(base_card, left_card, right_card, **options):
    """Alias for L{merge_vcards}."""
    return merge_vcards(base_card, left_card, right_card, **options)


def merge(base_card, left_card, right_card, **options):
    """Short alias for L{merge_vcards}."""
    return merge_vcards(base_card, left_card, right_card, **options)


def three_way_merge(base_card, left_card, right_card, **options):
    """Alias for L{merge_vcards}."""
    return merge_vcards(base_card, left_card, right_card, **options)


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------


def _read_cli(path):
    with open(path, encoding="utf-8") as stream:
        return _as_vcard(stream)


def _package_version():
    from . import VERSION

    return VERSION


def main(argv=None):
    parser = ArgumentParser(description="Three-way merge of vCard files using vCard item groups")
    parser.add_argument("-V", "--version", action="version", version=_package_version())
    parser.add_argument("base", nargs="?", help="common ancestor vCard")
    parser.add_argument("left", nargs="?", help="first changed vCard")
    parser.add_argument("right", nargs="?", help="second changed vCard")
    parser.add_argument("-o", "--output", help="write the merged vCard here instead of stdout")
    parser.add_argument("--strict", action="store_true", help="fail on ambiguous group correspondence")
    parser.add_argument("--report", action="store_true", help="print conflicts and ambiguities to stderr")
    args = parser.parse_args(argv)
    if not args.base or not args.left or not args.right:
        parser.error("the following arguments are required: base, left, right")

    report = merge_vcards_with_report(
        _read_cli(args.base), _read_cli(args.left), _read_cli(args.right), strict=args.strict
    )
    text = report.vcard.serialize()
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
    else:
        sys.stdout.write(text)
    if args.report:
        for conflict in report.conflicts:
            print(str(conflict), file=sys.stderr)
        for ambiguity in report.ambiguities:
            print(str(ambiguity), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
