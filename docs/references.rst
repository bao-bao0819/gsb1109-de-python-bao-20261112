Named references, constants, formulas, and macros
=================================================

.. currentmodule:: xlrd.book


A name is used to refer to a cell, a group of cells, a constant
value, a formula, or a macro. Usually the scope of a name is global
across the whole workbook. However it can be local to a worksheet.
For example, if the sales figures are in different cells in
different sheets, the user may define the name "Sales" in each
sheet. There are built-in names, like "Print_Area" and
"Print_Titles"; these two are naturally local to a sheet.

To inspect the names with a user interface like MS Excel, OOo Calc,
or Gnumeric, click on Insert -> Names -> Define. This will show the global
names, plus those local to the currently selected sheet.

A :class:`Book` object provides two dictionaries (:attr:`Book.name_map` and
:attr:`Book.name_and_scope_map`) and a list (:attr:`Book.name_obj_list`) which
allow various ways of accessing the :class:`Name` objects.
There is one :class:`Name` object for each `NAME` record found in the workbook.
:class:`Name` objects have many attributes, several of which are relevant only
when ``obj.macro`` is ``1``.

In the examples directory you will find ``namesdemo.xls`` which
showcases the many different ways that names can be used, and
``xlrdnamesAPIdemo.py`` which offers 3 different queries for inspecting
the names in your files, and shows how to extract whatever a name is
referring to. There is currently one "convenience method",
:meth:`Name.cell`, which extracts the value in the case where the name
refers to a single cell. The source code for :meth:`Name.cell` is an extra
source of information on how the :class:`Name` attributes hang together.

.. note::

  Name information is *not* extracted from files older than
  Excel 5.0 (``Book.biff_version < 50``).

Resolving the names usable from a worksheet
-------------------------------------------

The dictionaries above give access to the raw :class:`Name` objects, but they
do not tell you which names are actually usable from a given worksheet, what
cell areas a name points at once its formula is expanded, or which other names
it depends on.  The name-resolution API answers those questions while keeping
unresolvable branches separate from the ones that did resolve.

Call :meth:`~xlrd.sheet.Sheet.defined_names` on a sheet to resolve every name
visible from it, or :meth:`~xlrd.sheet.Sheet.defined_name` to resolve one
spelling.  Workbook-global names are visible from every sheet; a name local to
the sheet shadows any global name with the same case-insensitive spelling.
The same calls are available on the workbook as
:meth:`~xlrd.book.Book.defined_names_for_sheet` and
:meth:`~xlrd.book.Book.defined_name_for_sheet`.

Each result is a :class:`~xlrd.names.NameResolution`:

* ``areas`` lists the fully resolved
  :class:`~xlrd.names.AreaReference` objects, including every member of a
  union and every sheet of a 3D reference.  Exact duplicate areas are removed,
  and :meth:`~xlrd.names.NameResolution.iter_cells` visits each cell of an
  overlapping union only once.
* ``dependencies`` lists the other defined names mentioned by the formula,
  transitively.
* ``external_refs`` lists branches that point into another workbook or an
  add-in, and ``unresolved_branches`` lists error values, circular references
  and functions that cannot be evaluated statically.
* ``complete`` is only ``True`` when *every* branch resolved.  A name whose
  formula is only half available therefore never looks complete: check
  ``complete`` (or ``resolvable``) before trusting ``areas`` as the whole
  story; ``kind`` is ``"partial_reference"`` when some areas resolved but
  other branches did not.

A relative reference is resolved against the cell that uses the name.  Pass
that cell's zero-based row and column as ``base_rowx`` and ``base_colx``;
without them the relative branch is reported under ``unresolved_branches``
instead of being guessed.

.. code-block:: python

    book = xlrd.open_workbook("namesdemo.xls")
    sheet = book.sheet_by_name("Sheet3")
    for resolution in sheet.defined_names():
        if resolution.complete:
            print(resolution.name, [a.coords for a in resolution.areas])
        else:
            print(resolution.name, "is not fully usable:",
                  resolution.external_refs, resolution.unresolved_branches)
