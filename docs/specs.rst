Spec Definitions
================

The :any:`anytree.specs` module provides a reversible migration between the
legacy dictionary format, where the same specification is copied into every
category node, and the shared-definition format, where each specification is
stored once and referenced by a stable spec id.

It operates on plain dictionaries and therefore composes with
:any:`anytree.exporter.DictExporter` and
:any:`anytree.importer.DictImporter`.

.. automodule:: anytree.specs
