# -*- coding: utf-8 -*-
# Copyright (c) 2005-2013 Stephen John Machin, Lingfo Pty Ltd
# This module is part of the xlrd package, which is released under a
# BSD-style licence.

"""
Multi-row (hierarchical) column header support.

Given a block of header rows on a :class:`~xlrd.sheet.Sheet`, this module
produces, for every column, a *column path* (the header labels from the
outermost to the innermost level) together with the *original cell
coordinates* of every label in that path.

Design rules used throughout this module:

* A merged cell is one label whose identity is the coordinate of its
  top-left (anchor) cell.  Subordinate cells inside the merged region do
  not become labels of their own; they reference the anchor label and keep
  the full merged range for reporting.
* A truly empty cell (``XL_CELL_EMPTY``) and a formatted blank cell
  (``XL_CELL_BLANK``) are vacancies, not labels.  They are recorded per
  level but never contribute a component to a column path.
* Two independent cells holding the same text are two different labels,
  because their coordinates differ.  Label identity is the anchor
  coordinate, never the text alone, so a name lookup never collapses to
  the first hit.
* Hiding a column only removes that column from a projected view; the
  surviving columns keep their original column index and coordinates, so
  their paths can never be re-attached to another group.

Merged-cell ranges and hidden-column flags are only populated when the
workbook is opened with ``open_workbook(..., formatting_info=True)``.
Without formatting information the builder still works, but every header
cell is treated as independent/visible.
"""

from __future__ import print_function

from .biffh import XL_CELL_BLANK, XL_CELL_EMPTY, XLRDError
from .timemachine import UNICODE_LITERAL, xrange

_COL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _colname(colx):
    """Zero-based column index to Excel column name (``0 -> 'A'``)."""
    name = UNICODE_LITERAL('')
    while True:
        quot, rem = divmod(colx, 26)
        name = _COL_LETTERS[rem] + name
        if not quot:
            return name
        colx = quot - 1


def cell_address(rowx, colx):
    """Return the A1-style address, e.g. ``'B2'``, of a zero-based position."""
    return _colname(colx) + str(rowx + 1)

#: The header position carries a label.
ROLE_LABEL = 'label'

#: The header position is a non-anchor cell covered by a merged region.
ROLE_MERGED_SUBORDINATE = 'merged_subordinate'

#: The header position holds no label (empty or formatted blank).
ROLE_VACANT = 'vacant'

#: The one-to-many region is a single merged cell range.
REGION_MERGED = 'merged'

#: The one-to-many region is the bounding box of several independent,
#: same-named cells (no merged range covers them all).
REGION_INDEPENDENT_DUPLICATE = 'independent_duplicate'


def _normalise_header_rows(header_rows):
    """Accept a row count or an explicit ``(rlo, rhi)`` range."""
    if isinstance(header_rows, (tuple, list)):
        if len(header_rows) != 2:
            raise ValueError(
                "header_rows range must have exactly two items: %r"
                % (header_rows,)
            )
        rlo, rhi = header_rows
        if not (0 <= rlo < rhi):
            raise ValueError(
                "header_rows range must satisfy 0 <= rlo < rhi: %r"
                % (header_rows,)
            )
        return rlo, rhi
    count = int(header_rows)
    if count <= 0:
        raise ValueError(
            "header_rows count must be a positive integer: %r" % (header_rows,)
        )
    return 0, count


class HeaderLabel(object):
    """
    One labelled header node.

    A label's identity is :attr:`origin` (the anchor coordinate).  The same
    object is shared by every column covered by a merged region, so a merged
    parent can never be mistaken for two separate parent headers.
    """

    __slots__ = ('value', 'origin', 'merged_range', 'cell_type')

    def __init__(self, value, origin, merged_range, cell_type):
        #: Text (or raw cell value) shown by the label.
        self.value = value
        #: ``(rowx, colx)`` of the anchor / independent cell.
        self.origin = origin
        #: ``(rlo, rhi, clo, chi)`` if the label anchors a merged region,
        #: otherwise ``None``.
        self.merged_range = merged_range
        #: Original xlrd cell type of the anchor cell.
        self.cell_type = cell_type

    @property
    def key(self):
        """Identity key: coordinates first, text second."""
        return self.origin

    def __repr__(self):
        return "HeaderLabel(%r@%r%s)" % (
            self.value, self.origin,
            " merged=%r" % (self.merged_range,) if self.merged_range else "",
        )


class HeaderLevel(object):
    """
    What occupies one header-row position of a column.

    The three roles keep truly blank cells, merged-region subordinates and
    independent labels apart.
    """

    __slots__ = ('rowx', 'colx', 'role', 'label', 'cell_type',
                 'vacancy_kind', 'merged_range')

    def __init__(self, rowx, colx, role, cell_type,
                 label=None, vacancy_kind=None, merged_range=None):
        self.rowx = rowx
        self.colx = colx
        #: One of :data:`ROLE_LABEL`, :data:`ROLE_MERGED_SUBORDINATE`
        #: and :data:`ROLE_VACANT`.
        self.role = role
        self.cell_type = cell_type
        #: The governing :class:`HeaderLabel` (anchor label for subordinates).
        self.label = label
        #: ``'empty'`` for ``XL_CELL_EMPTY``, ``'blank'`` for
        #: ``XL_CELL_BLANK``, otherwise ``None``.
        self.vacancy_kind = vacancy_kind
        #: Merged range covering this position, if any.
        self.merged_range = merged_range

    @property
    def is_vacant(self):
        return self.role == ROLE_VACANT

    def __repr__(self):
        return "HeaderLevel(%r, %r, %s)" % (self.rowx, self.colx, self.role)


class ColumnHeader(object):
    """The resolved header information for a single source column."""

    __slots__ = ('colx', 'hidden', 'levels', 'path', 'path_origins',
                 'path_addresses')

    def __init__(self, colx, hidden, levels):
        self.colx = colx
        #: True when a ``COLINFO`` record marks the column hidden.
        self.hidden = hidden
        #: One :class:`HeaderLevel` per requested header row (vacancies
        #: included, in physical order).
        self.levels = levels

        path = []
        origins = []
        addresses = []
        last_origin = None
        for level in levels:
            if level.role == ROLE_VACANT or level.label is None:
                continue
            # A vertically merged label reappears on subordinate rows with
            # the same origin; it is one path component, not several.
            if level.label.origin == last_origin:
                continue
            path.append(level.label.value)
            origins.append(level.label.origin)
            addresses.append(cell_address(*level.label.origin))
            last_origin = level.label.origin
        #: Tuple of label values, outermost header first.
        self.path = tuple(path)
        #: Tuple of ``(rowx, colx)`` anchors, parallel to :attr:`path`.
        self.path_origins = tuple(origins)
        #: Tuple of A1 addresses, parallel to :attr:`path`.
        self.path_addresses = tuple(addresses)

    @property
    def leaf_level(self):
        """The last labelled level (the innermost header cell)."""
        for level in reversed(self.levels):
            if level.role != ROLE_VACANT and level.label is not None:
                return level
        return None

    @property
    def leaf_origin(self):
        level = self.leaf_level
        return level.label.origin if level is not None else None

    def __repr__(self):
        return "ColumnHeader(colx=%d, path=%r)" % (self.colx, self.path)


class HeaderNode(object):
    """
    A node in the header tree.

    Children are keyed by label origin, so a merged anchor is a single node
    shared by all the columns underneath it, while same-named independent
    cells produce distinct sibling nodes.
    """

    __slots__ = ('label', 'parent', 'children', 'columns')

    def __init__(self, label, parent):
        #: ``None`` for the synthetic root.
        self.label = label
        self.parent = parent
        #: Ordered list of child :class:`HeaderNode`.
        self.children = []
        #: Source column indexes beneath this node.
        self.columns = []

    @property
    def path(self):
        pieces = []
        node = self
        while node.label is not None:
            pieces.append(node.label.value)
            node = node.parent
        pieces.reverse()
        return tuple(pieces)

    @property
    def is_ambiguous(self):
        """True when a lookup of this node's path hits more than one column."""
        return len(self.columns) > 1

    def __repr__(self):
        return "HeaderNode(%r, cols=%r)" % (
            self.label.value if self.label else None, self.columns,
        )


class AmbiguousRegion(object):
    """
    The smallest region explaining why a path maps to several columns.

    For a shared merged parent this is the intact merged range; for
    same-named independent cells it is the bounding box of their anchor
    cells, explicitly flagged so the two cases stay separable.
    """

    __slots__ = ('kind', 'range', 'level_index', 'origins')

    def __init__(self, kind, range_, level_index, origins):
        #: :data:`REGION_MERGED` or :data:`REGION_INDEPENDENT_DUPLICATE`.
        self.kind = kind
        #: ``(rlo, rhi, clo, chi)``; the merged range or the bounding box.
        self.range = range_
        #: Index inside the column path where the fan-out starts.
        self.level_index = level_index
        #: Distinct anchor coordinates participating in the fan-out.
        self.origins = tuple(origins)

    def __repr__(self):
        return "AmbiguousRegion(%s, %r)" % (self.kind, self.range)


class HeaderMatch(object):
    """Result of looking a column path up in a :class:`ColumnHeaderMap`."""

    __slots__ = ('query', 'columns', 'region')

    def __init__(self, query, columns, region):
        #: The path tuple that was searched for.
        self.query = tuple(query)
        #: Every matching :class:`ColumnHeader` (never just the first).
        self.columns = columns
        #: ``None`` when unique, otherwise an :class:`AmbiguousRegion`.
        self.region = region

    @property
    def unique(self):
        return len(self.columns) == 1

    @property
    def column(self):
        """The single match; raises if the path is one-to-many."""
        if len(self.columns) != 1:
            raise XLRDError(
                "Header path %r is not unique: %d columns match"
                % (self.query, len(self.columns))
            )
        return self.columns[0]

    def __repr__(self):
        return "HeaderMatch(%r, %d match(es))" % (self.query, len(self.columns))


class ColumnHeaderMap(object):
    """
    Column paths, coordinates, tree and reverse-lookup index for a header
    block of one sheet.

    Build instances with :meth:`xlrd.sheet.Sheet.column_header_map` or
    :func:`build_column_header_map`.  A projected view (for example hiding
    columns) is produced by :meth:`projection` and shares the immutable
    per-column objects, keeping original coordinates intact.
    """

    def __init__(self, sheet, rlo, rhi, columns):
        self._sheet = sheet
        #: First header row index (inclusive).
        self.header_rowx_lo = rlo
        #: Last header row index (exclusive).
        self.header_rowx_hi = rhi
        #: All :class:`ColumnHeader` objects in source-column order.
        self.columns = columns
        self._by_colx = dict((column.colx, column) for column in columns)
        self.root = self._build_tree(columns)

    # -- construction -----------------------------------------------------

    @staticmethod
    def _build_tree(columns):
        root = HeaderNode(None, None)
        for column in columns:
            node = root
            if column.colx not in node.columns:
                node.columns.append(column.colx)
            for origin in column.path_origins:
                child = None
                for candidate in node.children:
                    if candidate.label.origin == origin:
                        child = candidate
                        break
                if child is None:
                    level = None
                    for candidate_level in column.levels:
                        if (candidate_level.label is not None
                                and candidate_level.label.origin == origin):
                            level = candidate_level
                            break
                    child = HeaderNode(level.label, node)
                    node.children.append(child)
                if column.colx not in child.columns:
                    child.columns.append(column.colx)
                node = child
        return root

    # -- access -----------------------------------------------------------

    @property
    def visible_columns(self):
        """Columns not marked hidden, in source-column order."""
        return [column for column in self.columns if not column.hidden]

    def column_at(self, colx):
        """Return the :class:`ColumnHeader` for a source column index."""
        return self._by_colx[colx]

    def paths(self):
        """Iterate ``(colx, path, path_origins)`` for every column."""
        for column in self.columns:
            yield column.colx, column.path, column.path_origins

    def projection(self, include_hidden=True, columns=None):
        """
        Return a narrowed view of this map.

        :param include_hidden: when False, hidden columns are dropped.
        :param columns: optional explicit iterable of source column indexes
          to keep; hidden columns are still filtered when ``include_hidden``
          is False.

        The view reuses the original :class:`ColumnHeader` objects, so every
        surviving column keeps its source column index and cell coordinates;
        dropping a column can never move another column's path onto it.
        """
        if columns is None:
            selected = list(self.columns)
        else:
            wanted = set(columns)
            selected = [
                column for column in self.columns if column.colx in wanted
            ]
        if not include_hidden:
            selected = [column for column in selected if not column.hidden]
        return ColumnHeaderMap(
            self._sheet, self.header_rowx_lo, self.header_rowx_hi, selected
        )

    # -- reverse lookup ---------------------------------------------------

    def locate(self, path, prefix=False):
        """
        Find columns by a header path (tuple of label values).

        Every match is returned.  When the path is unique,
        :attr:`HeaderMatch.column` yields the column whose
        :attr:`~ColumnHeader.path_origins` points back at the source cells.
        When several columns match, :attr:`HeaderMatch.region` describes the
        minimal merged range or duplicate-cell bounding box responsible.

        :param prefix: when True, the query may name only the outermost
          levels (any node whose path starts with the query matches).
        """
        query = tuple(path)
        if prefix:
            matched = [
                column for column in self.columns
                if column.path[:len(query)] == query
            ]
        else:
            matched = [
                column for column in self.columns if column.path == query
            ]
        region = self._ambiguity_region(matched) if len(matched) > 1 \
            else None
        return HeaderMatch(query, matched, region)

    def ambiguities(self):
        """
        Every header path that resolves to more than one column, each paired
        with the smallest region causing the one-to-many mapping.

        Two kinds are covered: fan-out beneath a shared (usually merged)
        parent label, and same-text independent cells whose full paths are
        equal but whose anchor coordinates differ.
        """
        result = []
        seen_queries = set()

        def add_match(query, columns):
            if query in seen_queries or len(columns) <= 1:
                return
            seen_queries.add(query)
            region = self._ambiguity_region(columns)
            result.append(HeaderMatch(query, columns, region))

        def walk(node):
            if node.label is not None:
                columns = [self._by_colx[colx] for colx in node.columns]
                add_match(node.path, columns)
            for child in node.children:
                walk(child)

        walk(self.root)

        # Same-text independent labels share no tree node, so group columns
        # by the text path explicitly.  Columns whose whole origin tuple is
        # identical already appear above as a merged-label fan-out.
        groups = {}
        for column in self.columns:
            if not column.path:
                continue
            groups.setdefault(column.path, []).append(column)
        for query in sorted(groups):
            columns = groups[query]
            origin_sets = set(column.path_origins for column in columns)
            if len(origin_sets) > 1:
                add_match(query, columns)

        return result

    def _ambiguity_region(self, columns):
        """Smallest merged range or bounding box explaining a fan-out."""
        max_depth = max((len(column.path_origins) for column in columns),
                        default=0)

        for level_index in xrange(max_depth):
            origins = self._unique_in_order([
                column.path_origins[level_index]
                for column in columns
                if level_index < len(column.path_origins)
            ])
            if len(origins) <= 1:
                continue
            # The columns diverge at this level.  When they still share one
            # merged parent label at the level above, that intact merged
            # range is the smallest region explaining the fan-out; it must
            # not be split into two parent headers.
            if level_index > 0:
                parent_origins = self._unique_in_order([
                    column.path_origins[level_index - 1]
                    for column in columns
                    if level_index - 1 < len(column.path_origins)
                ])
                if len(parent_origins) == 1:
                    parent_level = self._level_at_path_index(
                        columns[0], level_index - 1
                    )
                    if (parent_level is not None
                            and parent_level.merged_range is not None):
                        return AmbiguousRegion(
                            REGION_MERGED, parent_level.merged_range,
                            level_index - 1, parent_origins,
                        )
            return AmbiguousRegion(
                REGION_INDEPENDENT_DUPLICATE,
                self._bounding_box(origins), level_index, origins,
            )

        # Every label anchor is shared, e.g. a merged leaf spans several
        # data columns: the deepest label's merged range is the region.
        deepest = max(
            (len(column.path_origins) for column in columns), default=0
        )
        if deepest:
            last_level = self._level_at_path_index(columns[0], deepest - 1)
            if last_level is not None and last_level.merged_range is not None:
                return AmbiguousRegion(
                    REGION_MERGED, last_level.merged_range, deepest - 1,
                    self._unique_in_order([
                        column.path_origins[deepest - 1]
                        for column in columns
                    ]),
                )
        leaf_origins = self._unique_in_order([
            column.leaf_origin for column in columns
            if column.leaf_origin is not None
        ])
        return AmbiguousRegion(
            REGION_INDEPENDENT_DUPLICATE,
            self._bounding_box(leaf_origins), max(deepest - 1, 0),
            leaf_origins,
        )

    def _level_at_path_index(self, column, path_index):
        target_origin = column.path_origins[path_index]
        for level in column.levels:
            if level.label is not None and level.label.origin == target_origin:
                return level
        return None

    @staticmethod
    def _bounding_box(origins):
        rows = [rowx for rowx, _colx in origins]
        cols = [colx for _rowx, colx in origins]
        return min(rows), max(rows) + 1, min(cols), max(cols) + 1

    @staticmethod
    def _unique_in_order(items):
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def __repr__(self):
        return "ColumnHeaderMap(%d columns, rows %d:%d)" % (
            len(self.columns), self.header_rowx_lo, self.header_rowx_hi,
        )


def _safe_cell(sheet, rowx, colx):
    """Return ``(ctype, value)`` even for ragged / out-of-span cells."""
    try:
        return sheet.cell_type(rowx, colx), sheet.cell_value(rowx, colx)
    except IndexError:
        return XL_CELL_EMPTY, UNICODE_LITERAL('')


def _column_hidden(sheet, colx):
    colinfo = sheet.colinfo_map.get(colx)
    return bool(colinfo is not None and getattr(colinfo, 'hidden', 0))


def build_column_header_map(sheet, header_rows, colx_range=None,
                            include_hidden=True):
    """
    Build the :class:`ColumnHeaderMap` for a sheet.

    :param sheet: a :class:`~xlrd.sheet.Sheet`.
    :param header_rows: either a positive number of header rows counted from
      row 0, or an explicit ``(first_rowx, last_rowx)`` range (exclusive).
    :param colx_range: optional ``(first_colx, last_colx)`` range (exclusive);
      defaults to all of the sheet's columns.
    :param include_hidden: when False, hidden columns are left out of the
      initial view (they remain recoverable by building again with True and
      projecting afterwards).

    Merged ranges and the hidden flag require
    ``open_workbook(..., formatting_info=True)``.
    """
    rlo, rhi = _normalise_header_rows(header_rows)
    if colx_range is None:
        clo, chi = 0, sheet.ncols
    else:
        clo, chi = colx_range
        if not (0 <= clo < chi):
            raise ValueError(
                "colx_range must satisfy 0 <= clo < chi: %r"
                % (colx_range,)
            )

    # Index every cell covered by a merged region.
    coverage = {}
    empty_anchors = set()
    for rng in sheet.merged_cells:
        mrlo, mrhi, mclo, mchi = rng
        anchor = (mrlo, mclo)
        anchor_type, anchor_value = _safe_cell(sheet, anchor[0], anchor[1])
        if anchor_type in (XL_CELL_EMPTY, XL_CELL_BLANK) \
                and anchor_value == UNICODE_LITERAL(''):
            empty_anchors.add(anchor)
        for rowx in xrange(mrlo, mrhi):
            for colx in xrange(mclo, mchi):
                coverage[(rowx, colx)] = rng

    labels_by_anchor = {}

    def anchor_label(rng):
        anchor = (rng[0], rng[2])
        label = labels_by_anchor.get(anchor)
        if label is None:
            ctype, value = _safe_cell(sheet, anchor[0], anchor[1])
            label = HeaderLabel(value, anchor, rng, ctype)
            labels_by_anchor[anchor] = label
        return label

    columns = []
    for colx in xrange(clo, chi):
        hidden = _column_hidden(sheet, colx)
        if hidden and not include_hidden:
            continue
        levels = []
        for rowx in xrange(rlo, rhi):
            rng = coverage.get((rowx, colx))
            ctype, value = _safe_cell(sheet, rowx, colx)
            if rng is not None:
                anchor = (rng[0], rng[2])
                if anchor in empty_anchors:
                    # A merged region whose own anchor carries no text: keep
                    # the range on record, but no empty label enters a path.
                    if (rowx, colx) == anchor:
                        levels.append(HeaderLevel(
                            rowx, colx, ROLE_VACANT, ctype,
                            vacancy_kind=('blank' if ctype == XL_CELL_BLANK
                                          else 'empty'),
                            merged_range=rng,
                        ))
                    else:
                        levels.append(HeaderLevel(
                            rowx, colx, ROLE_MERGED_SUBORDINATE, ctype,
                            merged_range=rng,
                        ))
                elif (rowx, colx) == anchor:
                    levels.append(HeaderLevel(
                        rowx, colx, ROLE_LABEL, ctype,
                        label=anchor_label(rng), merged_range=rng,
                    ))
                else:
                    levels.append(HeaderLevel(
                        rowx, colx, ROLE_MERGED_SUBORDINATE, ctype,
                        label=anchor_label(rng), merged_range=rng,
                    ))
            elif ctype in (XL_CELL_EMPTY, XL_CELL_BLANK) \
                    and value == UNICODE_LITERAL(''):
                levels.append(HeaderLevel(
                    rowx, colx, ROLE_VACANT, ctype,
                    vacancy_kind=('blank' if ctype == XL_CELL_BLANK
                                  else 'empty'),
                ))
            else:
                # Independent label: its own coordinate is its identity, even
                # when another independent cell holds the same text.
                origin = (rowx, colx)
                label = labels_by_anchor.get(origin)
                if label is None:
                    label = HeaderLabel(value, origin, None, ctype)
                    labels_by_anchor[origin] = label
                levels.append(HeaderLevel(
                    rowx, colx, ROLE_LABEL, ctype, label=label,
                ))
        columns.append(ColumnHeader(colx, hidden, levels))

    return ColumnHeaderMap(sheet, rlo, rhi, columns)
