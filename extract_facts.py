#!/usr/bin/env python3
"""Extract a fact table from cell_kinds_biff8.xls under 4 open modes
(ragged_rows x formatting_info), then:

  1. verify per-column existence and original coordinates in each mode;
  2. project to cell_value only and show which distinct BIFF sources collide
     (with hex of two different source records as evidence);
  3. cross-tabulate missing-value rates and state which mode combinations
     are directly comparable.
"""
import csv
import json
import os
import struct

import xlrd
from xlrd import XL_CELL_BLANK, XL_CELL_EMPTY

XLS = "cell_kinds_biff8.xls"
RECMAP = "cell_kinds_biff8.records.json"
OUTDIR = "out"

CTYPE = {
    0: "empty", 1: "text", 2: "number", 3: "date", 4: "bool", 5: "error",
    6: "blank",
}

with open(XLS, "rb") as f:
    RAW = f.read()
with open(RECMAP) as f:
    RECORDS = json.load(f)

os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- extract
def extract(ragged_rows, formatting_info):
    bk = xlrd.open_workbook(
        file_contents=RAW,
        ragged_rows=ragged_rows,
        formatting_info=formatting_info,
    )
    sh = bk.sheet_by_index(0)
    rows = []
    for rowx in range(sh.nrows):
        for colx in range(sh.row_len(rowx)):
            ct = sh.cell_type(rowx, colx)
            rows.append({
                "rowx": rowx,
                "colx": colx,
                "ctype": CTYPE[ct],
                "cell_value": repr(sh.cell_value(rowx, colx)),
                "xf_index": sh.cell_xf_index(rowx, colx) if formatting_info else "",
            })
    return sh, rows

combos = [
    ("rect_fmt0", False, False),
    ("rect_fmt1", False, True),
    ("ragged_fmt0", True, False),
    ("ragged_fmt1", True, True),
]

tables = {}
sheets = {}
for name, ragged, fmt in combos:
    sh, rows = extract(ragged, fmt)
    tables[name] = rows
    sheets[name] = sh
    path = os.path.join(OUTDIR, "facts_%s.csv" % name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rowx", "colx", "ctype",
                                          "cell_value", "xf_index"])
        w.writeheader()
        w.writerows(rows)

print("=== fact-table shape per mode ===")
for name, _, _ in combos:
    sh = sheets[name]
    lens = [sh.row_len(r) for r in range(sh.nrows)]
    print("%-12s nrows=%d ncols=%d row_lengths=%s total_cells=%d"
          % (name, sh.nrows, sh.ncols, lens, sum(lens)))

# -------------------------------------------- 1. existence & coordinates
print("\n=== 1. column existence per row (colx sets) ===")
for name, _, _ in combos:
    per_row = {}
    for r in tables[name]:
        per_row.setdefault(r["rowx"], []).append(r["colx"])
    print("%-12s %s" % (name, {k: v for k, v in sorted(per_row.items())}))

# original coordinates: every emitted (rowx, colx) that corresponds to a real
# BIFF record must sit at exactly that record's coordinates.
record_coords = {(r["rowx"], r["colx"]): r["name"]
                 for r in RECORDS if r["rowx"] is not None
                 and r["name"] not in ("STRING('')",)}
print("\ncoordinate check: emitted cells whose (rowx,colx) has a BIFF record")
for name, _, _ in combos:
    coords = {(r["rowx"], r["colx"]) for r in tables[name]}
    hits = {c: record_coords[c] for c in sorted(coords & set(record_coords))}
    phantom = sorted(coords - set(record_coords))
    print("%-12s record-backed=%d filled/no-record=%s"
          % (name, len(hits), phantom))

# -------------------------------------------- 2. cell_value-only projection
print("\n=== 2. projection to cell_value only (rect_fmt1) ===")
proj = {}
for r in tables["rect_fmt1"]:
    proj.setdefault(r["cell_value"], []).append((r["rowx"], r["colx"], r["ctype"]))
for val, cells in sorted(proj.items()):
    print("cell_value %-8s <- %s" % (val, cells))

def hex_at(offset, length):
    return " ".join("%02x" % b for b in RAW[offset:offset + length])

blank_rec = next(r for r in RECORDS if r["name"] == "BLANK" and r["rowx"] == 2)
formula_rec = next(r for r in RECORDS if r["name"] == "FORMULA(str)")
string_rec = next(r for r in RECORDS if r["name"] == "STRING('')")
print("\nevidence: two DIFFERENT source records, same cell_value ''")
print("  src1 (2,1) BLANK   @%d: %s" % (blank_rec["offset"],
      hex_at(blank_rec["offset"], blank_rec["length"])))
print("  src2 (3,1) FORMULA @%d: %s" % (formula_rec["offset"],
      hex_at(formula_rec["offset"], formula_rec["length"])))
print("       (3,1) STRING  @%d: %s" % (string_rec["offset"],
      hex_at(string_rec["offset"], string_rec["length"])))
hole = [r for r in RECORDS if r["rowx"] == 4 and r["colx"] == 1]
print("  src3 (4,1) never written: records covering (4,1) = %s" % hole)

# -------------------------------------------- 3. missing-rate cross-check
print("\n=== 3. missing-value rates per mode ===")
print("%-12s %7s %9s %9s   %s" % ("mode", "cells", "miss(val)", "miss(EMPTY)",
                                  "rates val/EMPTY"))
for name, _, _ in combos:
    rows = tables[name]
    n = len(rows)
    mv = sum(1 for r in rows if r["cell_value"] == "''")
    me = sum(1 for r in rows if r["ctype"] == "empty")
    mb = sum(1 for r in rows if r["ctype"] == "blank")
    print("%-12s %7d %5d (%5.1f%%) %4d (%5.1f%%)   blank_ctype=%d"
          % (name, n, mv, 100.0 * mv / n, me, 100.0 * me / n, mb))
