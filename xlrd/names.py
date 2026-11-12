"""
Resolution of defined names (the ``NAME`` records of a workbook) into the
concrete worksheet areas that they refer to.

The :class:`~xlrd.formula.Operand` machinery in :mod:`xlrd.formula` evaluates
name formulas but only keeps a coarse success/failure flag
(:attr:`~xlrd.book.Name.any_err`); an external link and a genuine formula error
are indistinguishable, and the list of *other* names that a name depends on is
not exposed.  This module parses the same name-formula tokens independently and
keeps each branch of the formula separate, so that a name whose formula is only
half resolvable is never reported as complete.

The public entry points are :meth:`xlrd.sheet.Sheet.defined_names`,
:meth:`xlrd.sheet.Sheet.defined_name` and
:meth:`xlrd.book.Book.defined_names_for_sheet`.
"""

from __future__ import print_function

import struct
from struct import unpack

from .biffh import (
    XLRDError, unpack_string_update_pos, unpack_unicode_update_pos,
)
from .formula import (
    FormulaError, Ref3D, func_defs,
    get_cell_addr, get_cell_range_addr,
    get_externsheet_local_range, get_externsheet_local_range_b57,
    oBOOL, oNUM, oSTRG, okind_dict, onames, szdict,
)
from .timemachine import *

# Reasons recorded on unresolved branches.
REASON_EXTERNAL = 'external_reference'
REASON_ADDIN = 'addin_function'
REASON_DELETED_SHEET = 'deleted_sheet'
REASON_MACRO_SHEET = 'macro_sheet'
REASON_FUNCTION = 'unevaluated_function'
REASON_RELATIVE = 'relative_reference_without_base_cell'
REASON_CIRCULAR = 'circular_name_reference'
REASON_SELF_REF = 'self_reference'
REASON_ERROR = 'error_value'
REASON_EMPTY = 'empty_or_undefined_formula'
REASON_UNHANDLED = 'unhandled_formula_token'
REASON_MACRO = 'macro_name'
REASON_BINARY = 'binary_data'

# Negative sheet indices produced by get_externsheet_local_range*(), mapped to
# the reason why no local sheet could be resolved.
_NEGATIVE_SHEET_REASONS = {
    -2: REASON_DELETED_SHEET,
    -3: REASON_MACRO_SHEET,
    -101: REASON_ERROR,
    -102: REASON_ERROR,
    -103: REASON_ERROR,
}

__all__ = [
    'AreaReference',
    'ExternalReference',
    'NameDependency',
    'NameResolution',
    'REASON_ADDIN',
    'REASON_BINARY',
    'REASON_CIRCULAR',
    'REASON_DELETED_SHEET',
    'REASON_EMPTY',
    'REASON_ERROR',
    'REASON_EXTERNAL',
    'REASON_FUNCTION',
    'REASON_MACRO_SHEET',
    'REASON_MACRO',
    'REASON_RELATIVE',
    'REASON_SELF_REF',
    'REASON_UNHANDLED',
    'resolve_name',
    'resolve_named',
    'resolve_visible_names',
]


class AreaReference(object):
    """
    A rectangular cell area reached through a defined name.

    A single area may span several worksheets (a 3D reference such as
    ``Sheet1:Sheet3!$A$1:$Z$10``); :attr:`sheet_indices` then holds more than
    one entry and the row/column limits apply to every one of those sheets.

    Row and column limits are zero-based and half-open, matching the
    conventions of :class:`~xlrd.sheet.Sheet`: an area of
    ``rowxlo=0, rowxhi=2, colxlo=0, colxhi=1`` covers the two cells
    ``A1`` and ``A2``.
    """

    __slots__ = (
        'sheet_indices', 'sheet_names', 'rowxlo', 'rowxhi', 'colxlo', 'colxhi',
    )

    def __init__(self, book, ref3d):
        self.sheet_indices = tuple(range(ref3d.shtxlo, ref3d.shtxhi))
        self.sheet_names = tuple(book._sheet_names[ix] for ix in self.sheet_indices)
        self.rowxlo = ref3d.rowxlo
        self.rowxhi = ref3d.rowxhi
        self.colxlo = ref3d.colxlo
        self.colxhi = ref3d.colxhi

    @property
    def sheet_index(self):
        """The single sheet index, or :class:`ValueError` for a 3D area."""
        if len(self.sheet_indices) != 1:
            raise ValueError("Area spans %d sheets" % len(self.sheet_indices))
        return self.sheet_indices[0]

    @property
    def sheet_name(self):
        """The single sheet name, or :class:`ValueError` for a 3D area."""
        if len(self.sheet_names) != 1:
            raise ValueError("Area spans %d sheets" % len(self.sheet_names))
        return self.sheet_names[0]

    @property
    def coords(self):
        """``(rowxlo, rowxhi, colxlo, colxhi)``."""
        return self.rowxlo, self.rowxhi, self.colxlo, self.colxhi

    def key(self):
        return (
            self.sheet_indices, self.rowxlo, self.rowxhi,
            self.colxlo, self.colxhi,
        )

    def iter_cells(self):
        """
        Yield ``(sheet_index, rowx, colx)`` for every cell in the area.

        Take care with names such as ``Print_Titles`` which can cover whole
        rows or columns (tens of thousands of cells).
        """
        for sheetx in self.sheet_indices:
            for rowx in range(self.rowxlo, self.rowxhi):
                for colx in range(self.colxlo, self.colxhi):
                    yield sheetx, rowx, colx

    def __eq__(self, other):
        return isinstance(other, AreaReference) and self.key() == other.key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return "AreaReference(sheets=%r, rows=(%d, %d), cols=(%d, %d))" % (
            list(self.sheet_names), self.rowxlo, self.rowxhi,
            self.colxlo, self.colxhi,
        )


class ExternalReference(object):
    """
    A branch of a name formula that points outside the workbook.

    :ivar kind: one of ``"external_sheet"`` (a 3D reference into another
       workbook), ``"external_name"`` (a name defined in another workbook) or
       ``"addin"`` (an add-in function).
    :ivar description: human-readable detail, including the external name
       index when the target document is not available.
    :ivar rowxlo: row limit recovered from the formula, or ``None`` when no
       cell coordinates are available (an external *name*).
    """

    __slots__ = ('kind', 'description', 'rowxlo', 'rowxhi', 'colxlo', 'colxhi')

    def __init__(self, kind, description,
                 rowxlo=None, rowxhi=None, colxlo=None, colxhi=None):
        self.kind = kind
        self.description = description
        self.rowxlo = rowxlo
        self.rowxhi = rowxhi
        self.colxlo = colxlo
        self.colxhi = colxhi

    def key(self):
        return (
            self.kind, self.description, self.rowxlo, self.rowxhi,
            self.colxlo, self.colxhi,
        )

    def __eq__(self, other):
        return isinstance(other, ExternalReference) and self.key() == other.key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return "ExternalReference(kind=%r, description=%r)" % (
            self.kind, self.description,
        )


class NameDependency(object):
    """
    Another defined name referred to by a name formula.

    :ivar name_index: index of the target in
       :attr:`~xlrd.book.Book.name_obj_list`.
    :ivar name: the spelling of the target name.
    :ivar scope: its scope sheet index, or ``-1`` for a workbook-global name.
    :ivar scope_sheet_name: sheet name for a local name, else ``None``.
    """

    __slots__ = ('name_index', 'name', 'scope', 'scope_sheet_name')

    def __init__(self, book, name_obj):
        self.name_index = name_obj.name_index
        self.name = name_obj.name
        self.scope = name_obj.scope
        if name_obj.scope >= 0:
            self.scope_sheet_name = book._sheet_names[name_obj.scope]
        else:
            self.scope_sheet_name = None

    def key(self):
        return self.name_index

    def __eq__(self, other):
        return isinstance(other, NameDependency) and self.key() == other.key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        if self.scope_sheet_name is None:
            where = 'workbook'
        else:
            where = self.scope_sheet_name
        return "NameDependency(%r, scope=%r)" % (self.name, where)


class NameResolution(object):
    """
    The resolved meaning of one defined name as seen from one worksheet.

    :ivar name: the spelling of the name.
    :ivar scope: scope sheet index, or ``-1`` for a workbook-global name.
    :ivar scope_sheet_name: sheet name for a local name, else ``None``.
    :ivar hidden: whether the name is hidden in the Excel UI.
    :ivar builtin: whether the name is a built-in such as ``Print_Area``.
    :ivar kind: coarse classification, one of ``"reference"`` (fully resolved
       cell areas), ``"partial_reference"`` (some areas resolved, but at least
       one branch is external or unresolvable), ``"constant"`` (a number,
       string or boolean rather than cells), ``"external"``, ``"error"``,
       ``"unresolved"``, ``"macro"`` or ``"binary"``.
    :ivar areas: fully resolved :class:`AreaReference` objects, with exact
       duplicates removed; overlapping areas are both retained, and
       :meth:`iter_cells` takes care of visiting each cell only once.
    :ivar external_refs: :class:`ExternalReference` branches.
    :ivar unresolved_branches: ``(reason, description)`` tuples for branches
       that could not be turned into cell areas.
    :ivar dependencies: :class:`NameDependency` objects for every other name
       mentioned by the formula (transitively resolved names included).
    :ivar formula_text: the reconstituted formula text when available.
    :ivar constant: ``(kind_name, value)`` for a constant name, else ``None``.
    :ivar base_cell: ``(sheet_index, rowx, colx)`` used to resolve relative
       references, or ``None``.
    """

    __slots__ = (
        'name', 'scope', 'scope_sheet_name', 'hidden', 'builtin', 'kind',
        'areas', 'external_refs', 'unresolved_branches', 'dependencies',
        'formula_text', 'constant', 'base_cell',
    )

    def __init__(self, book, name_obj, node, base_cell):
        self.name = name_obj.name
        self.scope = name_obj.scope
        self.scope_sheet_name = (
            book._sheet_names[name_obj.scope] if name_obj.scope >= 0 else None
        )
        self.hidden = bool(name_obj.hidden)
        self.builtin = bool(name_obj.builtin)
        self.formula_text = getattr(name_obj.result, 'text', None)
        self.base_cell = base_cell
        self.external_refs = sorted(node.externals, key=lambda ref: ref.key())
        self.unresolved_branches = sorted(set(node.unresolved + node.errors))
        self.dependencies = sorted(
            (NameDependency(book, book.name_obj_list[ix]) for ix in node.depends),
            key=lambda dep: dep.key(),
        )
        seen = set()
        self.areas = []
        for ref3d in node.refs:
            if ref3d.shtxlo < 0:
                continue
            area = AreaReference(book, ref3d)
            if area.key() in seen:
                continue
            seen.add(area.key())
            self.areas.append(area)
        self.constant = self._constant(name_obj)
        self.kind = self._classify(name_obj)

    @staticmethod
    def _constant(name_obj):
        result = name_obj.result
        if result is None or result.value is None:
            return None
        if result.kind in (oNUM, oSTRG, oBOOL):
            return okind_dict[result.kind], result.value
        return None

    def _classify(self, name_obj):
        if name_obj.macro:
            return 'macro'
        if name_obj.binary:
            return 'binary'
        reasons = {reason for reason, _ in self.unresolved_branches}
        only_functions = bool(reasons) and reasons <= {REASON_FUNCTION}
        blocked = bool(self.external_refs) or bool(
            reasons - {REASON_FUNCTION}
        )
        # When every unparsed branch is a function, xlrd's own evaluator may
        # still have produced a concrete scalar (e.g. CHOOSE/IF of constants).
        if self.constant is not None and not blocked:
            return 'constant'
        if self.areas:
            return 'reference' if not blocked and not only_functions \
                else 'partial_reference'
        if self.external_refs and not reasons:
            return 'external'
        if any(reason == REASON_ERROR for reason, _ in self.unresolved_branches):
            return 'error'
        return 'unresolved'

    @property
    def complete(self):
        """
        ``True`` only when every branch of the formula was resolved.

        An external link, an error value, a circular reference or a function
        that cannot be evaluated statically keeps the name incomplete, even
        when :attr:`areas` already contains part of the target range.  The one
        exception is a pure scalar constant (no cell areas) that xlrd itself
        already reduced to a value despite the presence of function tokens.
        """
        if self.external_refs:
            return False
        reasons = {reason for reason, _ in self.unresolved_branches}
        if self.constant is not None and not self.areas \
                and not reasons - {REASON_FUNCTION}:
            return True
        if reasons:
            return False
        return bool(self.areas) or self.constant is not None

    @property
    def resolvable(self):
        """``True`` when the name unambiguously resolves to cells or a value."""
        return self.complete and self.kind in ('reference', 'constant')

    def iter_cells(self):
        """
        Yield each ``(sheet_index, rowx, colx)`` covered by the name once.

        Overlapping areas are merged at the cell level, so a union never
        reports a shared cell twice.
        """
        seen = set()
        for area in self.areas:
            for triple in area.iter_cells():
                if triple in seen:
                    continue
                seen.add(triple)
                yield triple

    def __repr__(self):
        return "NameResolution(name=%r, kind=%r, complete=%r, areas=%r)" % (
            self.name, self.kind, self.complete, self.areas,
        )


class _Node(object):
    """Internal stack element used while parsing a name formula."""

    __slots__ = (
        'refs', 'externals', 'errors', 'unresolved', 'depends', 'scalar',
    )

    def __init__(self, refs=None, externals=None, errors=None,
                 unresolved=None, depends=None, scalar=False):
        self.refs = list(refs or [])
        self.externals = list(externals or [])
        self.errors = list(errors or [])
        self.unresolved = list(unresolved or [])
        self.depends = set(depends or [])
        self.scalar = scalar

    def absorb(self, other):
        self.refs.extend(other.refs)
        self.externals.extend(other.externals)
        self.errors.extend(other.errors)
        self.unresolved.extend(other.unresolved)
        self.depends.update(other.depends)

    @classmethod
    def merged(cls, left, right, scalar=False):
        node = cls(scalar=scalar)
        node.absorb(left)
        node.absorb(right)
        return node


class _NameResolver(object):

    def __init__(self, book, base_sheet, base_row, base_col, root_index=-1):
        self.book = book
        self.base_sheet = base_sheet
        self.base_row = base_row
        self.base_col = base_col
        self._visiting = set()
        self._root_index = root_index

    def resolve(self, name_obj):
        node = self._parse(name_obj)
        node.refs = [self._apply_base(ref3d) for ref3d in node.refs]
        return node

    def _parse(self, name_obj):
        if name_obj.macro:
            return _Node(unresolved=[(REASON_MACRO, name_obj.name)])
        if name_obj.binary:
            return _Node(unresolved=[(REASON_BINARY, name_obj.name)])
        idx = name_obj.name_index
        if idx in self._visiting:
            reason = REASON_SELF_REF if idx == self._root_index else REASON_CIRCULAR
            return _Node(unresolved=[(reason, name_obj.name)])
        self._visiting.add(idx)
        try:
            node = self._parse_tokens(name_obj)
        finally:
            self._visiting.discard(idx)
        return node

    def _parse_tokens(self, name_obj):
        bk = self.book
        data = name_obj.raw_formula
        fmlalen = name_obj.basic_formula_len
        bv = bk.biff_version
        if fmlalen == 0:
            return _Node(unresolved=[(REASON_EMPTY, name_obj.name)])
        sztab = szdict[bv]
        stack = []
        pos = 0
        try:
            while 0 <= pos < fmlalen:
                op = BYTES_ORD(data[pos])
                opcode = op & 0x1f
                optype = (op & 0x60) >> 5
                opx = opcode + 32 if optype else opcode
                oname = onames[opx]
                sz = sztab[opx]
                if sz == -2:
                    raise FormulaError(
                        'Unexpected token 0x%02x ("%s"); biff_version=%d'
                        % (op, oname, bv)
                    )
                new_pos = self._handle_token(
                    data, pos, op, opcode, optype, opx, sz, bv, stack,
                )
                if sz == -1:  # variable-length token such as tStr
                    sz = new_pos - pos
                if sz <= 0:
                    raise FormulaError('Token size not set for opcode 0x%02x' % op)
                pos = new_pos
        except (FormulaError, IndexError, struct.error) as exc:
            return _Node(errors=[(REASON_UNHANDLED, str(exc))])
        if len(stack) != 1:
            return _Node(errors=[(REASON_ERROR, 'formula did not yield one result')])
        return stack[0]

    def _handle_token(self, data, pos, op, opcode, optype, opx, sz, bv, stack):
        # Simple scalar / error leaves ------------------------------------------------
        if not optype:
            if 0x03 <= opcode <= 0x0e:  # arithmetic, comparison, concat
                right = stack.pop()
                left = stack.pop()
                stack.append(_Node.merged(left, right, scalar=True))
                return pos + sz
            if opcode == 0x0f:  # tIsect
                stack.append(self._intersect(stack.pop(), stack.pop()))
                return pos + sz
            if opcode == 0x10:  # tList (union)
                right = stack.pop()
                left = stack.pop()
                node = _Node.merged(left, right)
                stack.append(node)
                return pos + sz
            if opcode == 0x11:  # tRange
                stack.append(self._enclosing_range(stack.pop(), stack.pop()))
                return pos + sz
            if 0x12 <= opcode <= 0x14:  # tUplus, tUminus, tPercent
                return pos + sz
            if opcode == 0x15:  # tParen: cosmetic only
                return pos + sz
            if opcode == 0x16:  # tMissArg
                stack.append(_Node(scalar=True))
                return pos + sz
            if opcode == 0x17:  # tStr
                if bv <= 70:
                    _, newpos = unpack_string_update_pos(
                        data, pos + 1, self.book.encoding, lenlen=1)
                else:
                    _, newpos = unpack_unicode_update_pos(
                        data, pos + 1, lenlen=1)
                stack.append(_Node(scalar=True))
                return newpos
            if opcode == 0x19:  # tAttr, variable size
                subop, nc = unpack('<BH', data[pos + 1:pos + 4])
                if subop == 0x04:  # Choose
                    actual_sz = nc * 2 + 6
                else:
                    actual_sz = 4
                if subop == 0x10 and stack:  # tAttrSum wraps one argument
                    inner = stack.pop()
                    inner.unresolved.append((REASON_FUNCTION, 'SUM'))
                    stack.append(inner)
                return pos + actual_sz
            if 0x1c <= opcode <= 0x1f:  # tErr, tBool, tInt, tNum
                if opcode == 0x1c:  # tErr
                    stack.append(_Node(errors=[(REASON_ERROR, onames[opx])]))
                else:
                    stack.append(_Node(scalar=True))
                return pos + sz
            if opcode in (0x04, 0x05):  # tRef / tArea without reference flags
                stack.append(_Node(unresolved=[(
                    REASON_ERROR, 'reference token without usable value')]))
                return pos + sz
            raise FormulaError('Unhandled opcode 0x%02x in name formula' % op)

        if opx == 0x20:  # tArray
            stack.append(_Node(unresolved=[(REASON_FUNCTION, 'array formula')]))
        elif opx == 0x21 or opx == 0x22:  # tFunc / tFuncVar
            stack.append(self._function(data, pos, opx, bv, stack))
        elif opx == 0x23:  # tName
            stack.append(self._local_name(data, pos, bv))
        elif opx == 0x24:  # tRef
            stack.append(self._cell_ref(data, pos, bv))
        elif opx == 0x25:  # tArea
            stack.append(self._area_ref(data, pos, bv))
        elif opx == 0x29:  # tMemFunc: no stack effect
            pass
        elif opx in (0x2f, 0x30):  # tRefErr / tAreaErr
            stack.append(_Node(errors=[(REASON_ERROR, onames[opx])]))
        elif opx == 0x39:  # tNameX
            stack.append(self._external_name(data, pos, bv))
        elif opx == 0x3a:  # tRef3d
            stack.append(self._ref3d(data, pos, bv, area=False))
        elif opx == 0x3b:  # tArea3d
            stack.append(self._ref3d(data, pos, bv, area=True))
        elif opx in (0x3c, 0x3d):  # tRefErr3d / tAreaErr3d
            stack.append(_Node(errors=[(REASON_ERROR, onames[opx])]))
        else:
            raise FormulaError('Unhandled opcode 0x%02x (%s)' % (op, onames[opx]))
        return pos + sz

    def _function(self, data, pos, opx, bv, stack):
        if opx == 0x21:  # tFunc
            nb = 1 + int(bv >= 40)
            funcx = unpack('<' + ' BH'[nb], data[pos + 1:pos + 1 + nb])[0]
            nargs = None
        else:  # tFuncVar
            nb = 1 + int(bv >= 40)
            nargs, funcx = unpack('<B' + ' BH'[nb], data[pos + 1:pos + 2 + nb])
            nargs = nargs % 128
        attrs = func_defs.get(funcx)
        fname = attrs[0] if attrs else 'FUNC%d' % funcx
        if nargs is None:
            nargs = attrs[1] if attrs else 0
        args = []
        for _ in range(max(nargs, 0)):
            if stack:
                args.append(stack.pop())
        node = _Node(unresolved=[(REASON_FUNCTION, fname)])
        for arg in args:
            node.absorb(arg)
        return node

    def _local_name(self, data, pos, bv):
        tgtnamex = unpack('<H', data[pos + 1:pos + 3])[0] - 1
        return self._name_by_index(tgtnamex)

    def _name_by_index(self, tgtnamex):
        bk = self.book
        if not 0 <= tgtnamex < len(bk.name_obj_list):
            return _Node(errors=[
                (REASON_ERROR, 'name index %d out of range' % tgtnamex)])
        target = bk.name_obj_list[tgtnamex]
        if target.macro:
            return _Node(unresolved=[(REASON_MACRO, target.name)], depends={tgtnamex})
        if target.binary:
            return _Node(unresolved=[(REASON_BINARY, target.name)], depends={tgtnamex})
        node = self._parse(target)
        node.depends.add(tgtnamex)
        return node

    def _external_name(self, data, pos, bv):
        dodgy = 0
        shx1 = -4
        if bv >= 80:
            refx, tgtnamex = unpack('<HH', data[pos + 1:pos + 5])
            tgtnamex -= 1
            origrefx = refx
            shx1, shx2 = get_externsheet_local_range(self.book, refx)
        else:
            refx, tgtnamex = unpack('<hxxxxxxxxH', data[pos + 1:pos + 13])
            tgtnamex -= 1
            origrefx = refx
            if refx > 0:
                refx -= 1
            elif refx < 0:
                refx = -refx - 1
            else:
                dodgy = 1
            if not dodgy:
                if origrefx > 0:
                    shx1 = shx2 = -4
                else:
                    exty = self.book._externsheet_type_b57[refx]
                    shx1 = shx2 = (-1, -1) if exty == 4 else (-666, -666)
        if tgtnamex == self._root_index:
            return _Node(errors=[(REASON_SELF_REF, 'name #%d' % tgtnamex)])
        if dodgy or shx1 < -1:
            desc = 'external name #%d via externsheet #%d' % (
                tgtnamex, origrefx)
            return _Node(externals=[ExternalReference(
                'addin' if shx1 == -5 else 'external_name', desc)])
        return self._name_by_index(tgtnamex)

    def _cell_ref(self, data, pos, bv):
        rowx, colx, row_rel, col_rel = get_cell_addr(
            data, pos + 1, bv, reldelta=1)
        relflags = (1, 1, row_rel, row_rel, col_rel, col_rel)
        coords = (0, 1, rowx, rowx + 1, colx, colx + 1)
        return _Node(refs=[Ref3D(coords + relflags)])

    def _area_ref(self, data, pos, bv):
        (row1, col1, r1rel, c1rel), (row2, col2, r2rel, c2rel) = \
            get_cell_range_addr(data, pos + 1, bv, reldelta=1)
        relflags = (1, 1, r1rel, r2rel, c1rel, c2rel)
        coords = (0, 1, row1, row2 + 1, col1, col2 + 1)
        return _Node(refs=[Ref3D(coords + relflags)])

    def _ref3d(self, data, pos, bv, area):
        if bv >= 80:
            refx = unpack('<H', data[pos + 1:pos + 3])[0]
            if area:
                (r1, c1, r1rel, c1rel), (r2, c2, r2rel, c2rel) = \
                    get_cell_range_addr(data, pos + 3, bv, reldelta=1)
            else:
                r1, c1, r1rel, c1rel = get_cell_addr(
                    data, pos + 3, bv, reldelta=1)
                r2, c2, r2rel, c2rel = r1, c1, r1rel, c1rel
            shx1, shx2 = get_externsheet_local_range(self.book, refx)
        else:
            raw_extshtx, raw_shx1, raw_shx2 = unpack(
                '<hxxxxxxxxhh', data[pos + 1:pos + 15])
            if area:
                (r1, c1, r1rel, c1rel), (r2, c2, r2rel, c2rel) = \
                    get_cell_range_addr(data, pos + 15, bv, reldelta=1)
            else:
                r1, c1, r1rel, c1rel = get_cell_addr(
                    data, pos + 15, bv, reldelta=1)
                r2, c2, r2rel, c2rel = r1, c1, r1rel, c1rel
            shx1, shx2 = get_externsheet_local_range_b57(
                self.book, raw_extshtx, raw_shx1, raw_shx2)
        coords = (shx1, shx2 + 1, r1, r2 + 1, c1, c2 + 1)
        relflags = (0, 0, r1rel, r2rel, c1rel, c2rel)
        return self._sheet_node(shx1, coords, relflags, refx=refx if bv >= 80 else None)

    def _sheet_node(self, shx1, coords, relflags, refx=None):
        if shx1 >= 0:
            return _Node(refs=[Ref3D(coords + relflags)])
        if shx1 == -4:
            desc = 'external workbook' + (
                ' via externsheet #%d' % refx if refx is not None else '')
            return _Node(externals=[ExternalReference(
                'external_sheet', desc,
                coords[2], coords[3], coords[4], coords[5])])
        if shx1 == -5:
            return _Node(externals=[ExternalReference('addin', 'add-in function')])
        if shx1 == -1:
            return _Node(unresolved=[(REASON_FUNCTION, 'sheet-unspecified reference')])
        reason = _NEGATIVE_SHEET_REASONS.get(shx1, REASON_ERROR)
        return _Node(errors=[(reason, 'sheet index %d' % shx1)])

    def _intersect(self, right, left):
        node = _Node.merged(left, right)
        if left.refs and right.refs and len(left.refs) == 1 \
                and len(right.refs) == 1:
            a, b = left.refs[0], right.refs[0]
            if a.shtxlo == b.shtxlo and a.shtxhi == b.shtxhi \
                    and (a.relflags or ()) == (b.relflags or ()):
                coords = (
                    a.shtxlo, a.shtxhi,
                    max(a.rowxlo, b.rowxlo), min(a.rowxhi, b.rowxhi),
                    max(a.colxlo, b.colxlo), min(a.colxhi, b.colxhi),
                )
                if coords[3] > coords[2] and coords[5] > coords[4]:
                    node.refs = [Ref3D(coords + a.relflags)]
                else:
                    node.errors.append((REASON_ERROR, 'empty intersection'))
        return node

    def _enclosing_range(self, right, left):
        node = _Node.merged(left, right)
        if left.refs and right.refs and len(left.refs) == 1 \
                and len(right.refs) == 1:
            a, b = left.refs[0], right.refs[0]
            if a.shtxlo == b.shtxlo and a.shtxhi == b.shtxhi \
                    and (a.relflags or ()) == (b.relflags or ()):
                coords = (
                    a.shtxlo, a.shtxhi,
                    min(a.rowxlo, b.rowxlo), max(a.rowxhi, b.rowxhi),
                    min(a.colxlo, b.colxlo), max(a.colxhi, b.colxhi),
                )
                node.refs = [Ref3D(coords + a.relflags)]
        return node

    def _apply_base(self, ref3d):
        flags = ref3d.relflags or (0, 0, 0, 0, 0, 0)
        if not any(flags):
            return ref3d
        if self.base_row is None or self.base_col is None or self.base_sheet is None:
            # The caller (_flag_relative_without_base) records these.
            return ref3d
        shtxlo, shtxhi = ref3d.shtxlo, ref3d.shtxhi
        if flags[0] or flags[1]:
            shtxlo = self.base_sheet
            shtxhi = self.base_sheet + 1
        rowxlo = ref3d.rowxlo + (self.base_row if flags[2] else 0)
        rowxhi = ref3d.rowxhi + (self.base_row if flags[3] else 0)
        colxlo = ref3d.colxlo + (self.base_col if flags[4] else 0)
        colxhi = ref3d.colxhi + (self.base_col if flags[5] else 0)
        return Ref3D((shtxlo, shtxhi, rowxlo, rowxhi, colxlo, colxhi))


def resolve_name(book, name_obj, base_sheet=None, base_row=None, base_col=None):
    """
    Resolve one :class:`~xlrd.book.Name` object.

    :param base_sheet: sheet index of the caller's cell. Required for
       sheet-relative references.
    :param base_row: zero-based row of the caller's cell. Required when the
       formula contains row-relative references.
    :param base_col: zero-based column of the caller's cell. Required when the
       formula contains column-relative references.
    """
    resolver = _NameResolver(
        book, base_sheet, base_row, base_col,
        root_index=name_obj.name_index,
    )
    node = resolver.resolve(name_obj)
    _flag_relative_without_base(node, base_sheet, base_row, base_col)
    base_cell = None
    if base_sheet is not None and base_row is not None and base_col is not None:
        base_cell = (base_sheet, base_row, base_col)
    return NameResolution(book, name_obj, node, base_cell)


def _flag_relative_without_base(node, base_sheet, base_row, base_col):
    missing_base = base_sheet is None or base_row is None or base_col is None
    if not missing_base:
        return
    relative = []
    for ref3d in node.refs:
        if any(ref3d.relflags):
            relative.append(
                (REASON_RELATIVE, str(Ref3D(ref3d.coords + ref3d.relflags)))
            )
    node.unresolved.extend(relative)
    node.refs = [ref3d for ref3d in node.refs if not any(ref3d.relflags)]


def _select_visible(book, sheet_index, include_hidden=True, include_builtin=True):
    """Return the Name objects usable from ``sheet_index``, definition order."""
    chosen = {}
    ordered = []
    for name_obj in book.name_obj_list:
        scope = name_obj.scope
        if scope != -1 and scope != sheet_index:
            continue
        if not include_hidden and name_obj.hidden:
            continue
        if not include_builtin and name_obj.builtin:
            continue
        key = name_obj.name.lower()
        if key in chosen:
            # A local definition wins over the workbook-global one.
            if chosen[key].scope == -1 and scope == sheet_index:
                chosen[key] = name_obj
            continue
        chosen[key] = name_obj
        ordered.append(key)
    return [chosen[key] for key in ordered]


def resolve_named(book, sheet_index, name, base_row=None, base_col=None):
    """
    Resolve the spelling ``name`` as it is seen from worksheet ``sheet_index``.

    A name local to the worksheet wins over a workbook-global name with the
    same case-insensitive spelling.

    :raises xlrd.biffh.XLRDError: no visible name matches.
    """
    key = name.lower()
    for name_obj in book.name_obj_list:
        if name_obj.name.lower() != key:
            continue
        if name_obj.scope == sheet_index:
            target = name_obj
            break
    else:
        target = None
        for name_obj in book.name_obj_list:
            if name_obj.name.lower() == key and name_obj.scope == -1:
                target = name_obj
                break
    if target is None:
        raise XLRDError(
            "No defined name %r is visible from sheet index %d"
            % (name, sheet_index)
        )
    return resolve_name(
        book, target, base_sheet=sheet_index,
        base_row=base_row, base_col=base_col,
    )


def resolve_visible_names(book, sheet_index, base_row=None, base_col=None,
                          include_hidden=True, include_builtin=True):
    """
    Resolve every name usable from the worksheet at ``sheet_index``.

    A workbook-global name is visible from every sheet; a name local to the
    given sheet shadows any global name with the same (case-insensitive)
    spelling.  Names local to other sheets, macro-sheet names and invalid
    names are excluded.

    Returns a list of :class:`NameResolution` objects in workbook definition
    order.
    """
    return [
        resolve_name(book, name_obj, base_sheet=sheet_index,
                     base_row=base_row, base_col=base_col)
        for name_obj in _select_visible(
            book, sheet_index, include_hidden, include_builtin)
    ]
