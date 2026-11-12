# -*- coding: utf-8 -*-
# Copyright (c) 2005-2012 Stephen John Machin, Lingfo Pty Ltd
# This module is part of the xlrd package, which is released under a
# BSD-style licence.
"""
Column export helpers.

The raw numeric value stored by Excel does not carry a unit.  The same
underlying float (e.g. ``0.1``) may mean a ratio (ten percent) under one
cell format and an ordinary quantity (zero point one) under another.
Only the XF record attached to each cell, and the number format string
it points at, tell the two apart.

This module never tries to reproduce what Excel displays.  It parses a
number format string into its semicolon-separated sections, records the
condition under which each section applies and -- crucially -- tells a
real percent marker (which scales the stored value by 100) apart from a
percent sign that is merely literal text (quoted or escaped).

Public entry points:

* :func:`parse_number_format` -- parse one format string into sections
* :func:`cell_format_basis` -- the full format basis of one cell
  (XF index, format key, section text and percent-scaling flag)
* :func:`export_columns` -- group numeric cells by column and original
  value, auto-convert only the records whose unit meaning is unique, and
  keep the rest with their original coordinates for manual review
"""

from __future__ import print_function

from .biffh import XL_CELL_NUMBER
from .formula import cellname, colname
from .timemachine import *

#: Excel applies the sections of a number format in this order.
#: A single-section format applies to every number; a two-section format
#: splits on zero; a three-section format adds an explicit zero section;
#: a four-section format adds a leading text section.
SECTION_CONDITIONS = ('positive', 'negative', 'zero', 'text')

#: Stored value is multiplied by this when a genuine percent marker is
#: present.  Excel stores ``10%`` as the float ``0.1``.
PERCENT_FACTOR = 100.0

#: Scale marker meaning "percent"; every unquoted/unescaped occurrence
#: in a numeric section scales the stored value by 100.
PERCENT_CHAR = UNICODE_LITERAL('%')


def split_format_sections(format_str):
    """
    Split a number format string on semicolons that are not hidden inside
    quoted text, square brackets or backslash escapes.

    :param format_str: the raw format string attached to a ``FORMAT`` record
    :returns: a list of section strings (possibly empty for a missing section)
    """
    if format_str is None:
        return []
    sections = []
    current = []
    in_quotes = False
    in_brackets = 0
    escaped = False
    for ch in format_str:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == UNICODE_LITERAL('\\'):
            current.append(ch)
            escaped = True
            continue
        if ch == UNICODE_LITERAL('"'):
            current.append(ch)
            in_quotes = not in_quotes
            continue
        if in_quotes:
            current.append(ch)
            continue
        if ch == UNICODE_LITERAL('['):
            in_brackets += 1
            current.append(ch)
            continue
        if ch == UNICODE_LITERAL(']') and in_brackets:
            in_brackets -= 1
            current.append(ch)
            continue
        if in_brackets:
            current.append(ch)
            continue
        if ch == UNICODE_LITERAL(';'):
            sections.append(UNICODE_LITERAL('').join(current))
            current = []
            continue
        current.append(ch)
    sections.append(UNICODE_LITERAL('').join(current))
    return sections


def section_has_percent(section):
    """
    Return true if the section contains a genuine percent marker.

    A percent sign counts as literal text -- and therefore does **not**
    scale the stored value -- when it is enclosed in double quotes
    (``"%"``) or introduced by a backslash escape (``\\%``).  Percent
    signs inside square brackets do not occur in valid Excel formats
    (brackets hold colour/condition/locale markers), but they are treated
    as non-marker text as well.
    """
    in_quotes = False
    in_brackets = 0
    escaped = False
    for ch in section:
        if escaped:
            escaped = False
            continue
        if ch == UNICODE_LITERAL('\\'):
            escaped = True
            continue
        if ch == UNICODE_LITERAL('"'):
            in_quotes = not in_quotes
            continue
        if in_quotes:
            continue
        if ch == UNICODE_LITERAL('['):
            in_brackets += 1
            continue
        if ch == UNICODE_LITERAL(']') and in_brackets:
            in_brackets -= 1
            continue
        if in_brackets:
            continue
        if ch == PERCENT_CHAR:
            return True
    return False


def _section_condition(index, nsections, section):
    """
    Resolve the applicability condition of one section.

    Explicit ``[<=n]`` / ``[>n]`` style conditions take precedence over
    the positional positive/negative/zero convention.
    """
    explicit = _explicit_condition(section)
    if explicit is not None:
        return explicit
    if nsections <= 1:
        return 'all'
    if nsections == 2:
        return ('positive', 'negative')[index]
    if nsections == 3:
        return ('positive', 'negative', 'zero')[index]
    if index < len(SECTION_CONDITIONS):
        return SECTION_CONDITIONS[index]
    # Excel ignores sections beyond the fourth; keep a positional text label.
    return SECTION_CONDITIONS[-1]


def _explicit_condition(section):
    # Conditions live in square brackets and start with a comparison
    # operator, e.g. "[>=100]", "[Red][<=0]".  Brackets holding a bare
    # colour name or a locale id (starts with "$" or holds a digit-only
    # language code) are not conditions.
    stripped = section.lstrip()
    while stripped.startswith(UNICODE_LITERAL('[')):
        end = stripped.find(UNICODE_LITERAL(']'))
        if end < 0:
            break
        content = stripped[1:end].strip()
        if content and content[0] in UNICODE_LITERAL('<>='):
            return content
        stripped = stripped[end + 1:].lstrip()
    return None


def parse_number_format(format_str):
    """
    Parse a number format string into its conditional sections.

    :param format_str: the raw format string (``None`` is tolerated for
      the locale-dependent built-in slots whose strings xlrd does not know)
    :returns: a list of :class:`NumberFormatSection` dictionaries, one per
      semicolon-separated section, in file order
    """
    raw_sections = split_format_sections(format_str)
    nsections = len(raw_sections)
    result = []
    for index, raw in enumerate(raw_sections):
        condition = _section_condition(index, nsections, raw)
        result.append(NumberFormatSection(
            index=index,
            condition=condition,
            format_text=raw,
            is_percent=section_has_percent(raw),
            explicit_condition=_explicit_condition(raw),
        ))
    return result


class NumberFormatSection(dict):
    """
    One semicolon-separated section of a number format string.

    Keys:

    * ``index`` -- zero-based position of the section in the format string
    * ``condition`` -- ``all``, ``positive``, ``negative``, ``zero``,
      ``text`` or an explicit condition such as ``">=100"``
    * ``format_text`` -- the raw section text, exactly as stored
    * ``is_percent`` -- true if the section has a real percent marker
      (quoted/escaped percent signs do not count)
    * ``explicit_condition`` -- the bracketed comparison condition, or
      ``None`` when the section relies on its positional condition
    """


def select_section(sections, value):
    """
    Return the section that Excel would use for *value*, without doing
    any display formatting.
    """
    if not sections:
        return None
    explicit = [s for s in sections if s['explicit_condition'] is not None]
    if explicit:
        for section in explicit:
            if _condition_matches(section['explicit_condition'], value):
                return section
        # Excel falls back to the final non-conditional numeric section
        # when no explicit condition matches.
        for section in reversed(sections):
            non_conditional = section['explicit_condition'] is None
            if non_conditional and section['condition'] != 'text':
                return section
        return explicit[-1]
    if len(sections) == 1:
        return sections[0]
    if value > 0:
        wanted = 'positive'
    elif value < 0:
        wanted = 'negative'
    else:
        wanted = 'zero'
    for section in sections:
        if section['condition'] == wanted:
            return section
    # Two-section formats reuse the positive section for zero.
    if value == 0:
        for section in sections:
            if section['condition'] == 'positive':
                return section
    return sections[0]


def _condition_matches(condition, value):
    # Conditions are simple comparisons: >n, >=n, <n, <=n, =n, <>n.
    for operator in ('>=', '<=', '<>', '>', '<', '='):
        if condition.startswith(operator):
            operand = condition[len(operator):].strip()
            try:
                bound = float(operand)
            except ValueError:
                return False
            if operator == '>':
                return value > bound
            if operator == '<':
                return value < bound
            if operator == '>=':
                return value >= bound
            if operator == '<=':
                return value <= bound
            if operator == '=':
                return value == bound
            return value != bound
    return False


def cell_format_basis(book, sheet, rowx, colx):
    """
    Resolve the full format basis of a numeric cell.

    :returns: a :class:`CellFormatBasis` dictionary describing the XF
      actually in effect (cell-level XF, else row/column/default XF),
      the ``FORMAT`` record it points at, the section selected for the
      stored value and its percent-scaling flag.  Returns ``None`` when
      the cell is not numeric or no XF information is available.
    """
    cell = sheet.cell(rowx, colx)
    if cell.ctype != XL_CELL_NUMBER:
        return None
    if cell.xf_index is None:
        return None
    xf_index = cell.xf_index
    xf = book.xf_list[xf_index]
    format_key = xf.format_key
    format_obj = book.format_map.get(format_key)
    format_str = format_obj.format_str if format_obj is not None else None
    sections = parse_number_format(format_str)
    section = select_section(sections, cell.value)
    is_percent = bool(section and section['is_percent'])
    return CellFormatBasis(
        sheet_name=sheet.name,
        coordinate=cellname(rowx, colx),
        rowx=rowx,
        colx=colx,
        value=cell.value,
        xf_index=xf_index,
        format_key=format_key,
        format_kind='built-in' if format_key <= 163 else 'custom',
        format_str=format_str,
        section=section,
        is_percent=is_percent,
    )


class CellFormatBasis(dict):
    """
    The format evidence behind one numeric cell.

    Keys:

    * ``coordinate`` -- A1-style coordinate, kept on every record so that
      ambiguous exports can be traced back to the source cell
    * ``xf_index`` / ``format_key`` -- the XF record actually applied and
      the key of the ``FORMAT`` record it references
    * ``format_str`` -- the full number format string
    * ``section`` -- the selected :class:`NumberFormatSection`
    * ``is_percent`` -- whether the selected section scales by 100
    * ``value`` -- the raw stored numeric value
    """


def export_columns(book, sheet=None):
    """
    Export numeric columns with format evidence and safe percent conversion.

    Cells are grouped first by column, then by stored value.  A group is
    auto-converted only when every cell in it has a unique unit meaning:
    all effective XFs select sections with the same percent flag.  When a
    single stored value appears both under a percent format and an
    ordinary numeric format, the group is ambiguous and every member is
    returned unconverted, carrying its original coordinate and the full
    format basis for manual review.

    :param book: a workbook opened with ``formatting_info=True``
    :param sheet: the sheet to export; defaults to every sheet in the book
    :returns: a list of :class:`ColumnExport` dictionaries, one per column
      that contains at least one numeric cell
    """
    sheets = [sheet] if sheet is not None else book.sheets()
    exports = []
    for target in sheets:
        exports.extend(_export_sheet(book, target))
    return exports


def _export_sheet(book, sheet):
    columns = {}
    for rowx in range(sheet.nrows):
        for colx in range(sheet.row_len(rowx)):
            basis = cell_format_basis(book, sheet, rowx, colx)
            if basis is None:
                continue
            columns.setdefault(colx, []).append(basis)
    result = []
    for colx in sorted(columns):
        records = columns[colx]
        result.append(_build_column_export(sheet, colx, records))
    return result


def _build_column_export(sheet, colx, records):
    groups = {}
    order = []
    for basis in records:
        key = basis['value']
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(basis)

    converted = []
    ambiguous = []
    for value in order:
        members = groups[value]
        percent_flags = frozenset(m['is_percent'] for m in members)
        if len(percent_flags) == 1:
            is_percent = next(iter(percent_flags))
            for member in members:
                exported = _exported_record(member, is_percent)
                converted.append(exported)
        else:
            for member in members:
                ambiguous.append(_ambiguous_record(member))

    return ColumnExport(
        sheet_name=sheet.name,
        colx=colx,
        column_letter=colname(colx),
        converted=converted,
        ambiguous=ambiguous,
    )


def _exported_record(basis, is_percent):
    record = _common_record(basis)
    record['unit'] = 'percent' if is_percent else 'number'
    record['converted'] = is_percent
    if is_percent:
        record['exported_value'] = basis['value'] * PERCENT_FACTOR
    else:
        record['exported_value'] = basis['value']
    return record


def _ambiguous_record(basis):
    record = _common_record(basis)
    record['unit'] = 'ambiguous'
    record['converted'] = False
    record['exported_value'] = basis['value']
    record['reason'] = (
        "stored value appears under both percent and non-percent formats; "
        "manual review required"
    )
    return record


def _common_record(basis):
    section = basis['section']
    return {
        'sheet_name': basis['sheet_name'],
        'coordinate': basis['coordinate'],
        'rowx': basis['rowx'],
        'colx': basis['colx'],
        'original_value': basis['value'],
        'xf_index': basis['xf_index'],
        'format_key': basis['format_key'],
        'format_kind': basis['format_kind'],
        'format_str': basis['format_str'],
        'section_index': section['index'] if section else None,
        'section_condition': section['condition'] if section else None,
        'section_text': section['format_text'] if section else None,
        'section_is_percent': section['is_percent'] if section else False,
    }


class ColumnExport(dict):
    """
    The export of one numeric column.

    Keys:

    * ``column_letter`` / ``colx`` / ``sheet_name`` -- column identity
    * ``converted`` -- records whose unit meaning is unique; percent
      records carry ``exported_value = original_value * 100`` while plain
      numbers pass through unchanged
    * ``ambiguous`` -- records whose stored value is shared by percent and
      non-percent formats; the raw value and coordinate are preserved for
      manual checking
    """
