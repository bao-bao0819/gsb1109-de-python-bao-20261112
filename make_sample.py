#!/usr/bin/env python3
"""Build a minimal raw-BIFF8 .xls sample that distinguishes four cell kinds:

  A. cell never written        -> no BIFF record at all for that (row, col)
  B. formatted blank           -> BLANK record (0x0201) carrying an XF index
  C. numeric zero              -> NUMBER record (0x0203) with value 0.0
  D. formula cached ''         -> FORMULA record (0x0006, string-result flag)
                                  followed by a STRING record (0x0207) of length 0

The file is a bare BIFF8 stream (no OLE2 wrapper). xlrd accepts this:
book.py biff2_8_load() treats any file not starting with the OLE signature
as a raw BIFF stream ("got this one at the antique store").

Sheet layout (0-based coords), 3 columns A/B/C, 7 rows:

  row 0: A0 "kind"           B0 "value"          C0 "note"
  row 1: A1 "zero"           B1 NUMBER 0.0       C1 "num_record"
  row 2: A2 "blank_fmt"      B2 BLANK (xf=1)     C2 "blank_record"
  row 3: A3 "formula_empty"  B3 FORMULA+STRING "" C3 "formula_record"
  row 4: A4 "never_written"  (B4: no record)     C4 "no_record"
  row 5: A5 "trailing_hole"  (B5,C5: no record)
  row 6: A6 "blank_trailing" B6 BLANK (xf=1)     (C6: no record)

Row 2 vs row 4 give an interior BLANK record vs an interior hole (same
position kind, different source). Row 6 gives a *trailing* BLANK, which is
where ragged_rows and formatting_info interact. Row 5 gives trailing holes.
"""
import json
import struct
import sys

OUT = "cell_kinds_biff8.xls"
RECORD_MAP = "cell_kinds_biff8.records.json"

# --- record helpers -------------------------------------------------------

def rec(opcode, data):
    return struct.pack("<HH", opcode, len(data)) + data

def bof(stream_type):
    # BIFF8 BOF: version 0x0600, stream type, build, year, flags, min version
    return rec(0x0809, struct.pack("<HHHHII", 0x0600, stream_type,
                                   0x0DBB, 0x07CC, 0x00000041, 0x00000006))

EOF = rec(0x000A, b"")

def codepage(cp=0x04E4):  # 1252 -> xlrd encoding 'cp1252'
    return rec(0x0042, struct.pack("<H", cp))

def boundsheet(abs_pos, name):
    nb = name.encode("latin1")
    data = struct.pack("<IBB", abs_pos, 0x00, 0x00)  # visible, worksheet
    data += struct.pack("<BB", len(nb), 0x00) + nb   # compressed unicode name
    return rec(0x0085, data)

def sst(strings):
    data = struct.pack("<II", len(strings), len(strings))
    for s in strings:
        sb = s.encode("latin1")
        data += struct.pack("<HB", len(sb), 0x00) + sb  # XLUnicodeString, 8-bit
    return rec(0x00FC, data)

def xf(is_style):
    # BIFF8 XF, 20 bytes: font, fmt, pkd_type_par, align1, rot, align2, used,
    # brdbkg1, brdbkg2, brdbkg3
    if is_style:
        pkd = 0x0004 | (0x0FFF << 4)   # is_style, parent = 0xFFF
    else:
        pkd = 0x0001                   # cell XF, locked, parent style 0
    return rec(0x00E0, struct.pack("<HHHBBBBIiH", 0, 0, pkd, 0, 0, 0, 0,
                                   0, 0, 0))

def dimensions(rlo, rhi, clo, chi):
    return rec(0x0200, struct.pack("<IIHHH", rlo, rhi, clo, chi, 0))

def labelsst(rowx, colx, xf_index, sst_index):
    return rec(0x00FD, struct.pack("<HHHi", rowx, colx, xf_index, sst_index))

def number(rowx, colx, xf_index, value):
    return rec(0x0203, struct.pack("<HHHd", rowx, colx, xf_index, value))

def blank(rowx, colx, xf_index):
    return rec(0x0201, struct.pack("<HHH", rowx, colx, xf_index))

def formula_empty_string(rowx, colx, xf_index):
    # result field: byte0=0 (string result follows in STRING record),
    # bytes 6:8 = 0xFFFF marks a "special" (non-numeric) cached result.
    result = b"\x00\x00\x00\x00\x00\x00\xff\xff"
    flags = 0x0000
    chn = b"\x00\x00\x00\x00"
    tokens = b"\x17\x00\x00"  # ptgStr, empty string (="" )
    data = struct.pack("<HHH", rowx, colx, xf_index) + result \
        + struct.pack("<H", flags) + chn + struct.pack("<H", len(tokens)) + tokens
    return rec(0x0006, data)

def string_record(s):
    sb = s.encode("latin1")
    return rec(0x0207, struct.pack("<HB", len(sb), 0x00) + sb)

# --- shared strings ---------------------------------------------------------

SST = [
    "kind", "value", "note",                 # 0,1,2  header
    "zero", "num_record",                    # 3,4
    "blank_fmt", "blank_record",             # 5,6
    "formula_empty", "formula_record",       # 7,8
    "never_written", "no_record",            # 9,10
    "trailing_hole",                         # 11
    "blank_trailing",                        # 12
]

# --- assemble the stream, tracking every record's offset --------------------

record_map = []   # (offset, opcode, name, rowx_or_None, colx_or_None)
stream = bytearray()

def emit(blob, name, opcode, rowx=None, colx=None):
    record_map.append({
        "offset": len(stream),
        "opcode": "0x%04X" % opcode,
        "name": name,
        "rowx": rowx,
        "colx": colx,
        "length": len(blob),
    })
    stream.extend(blob)

# workbook globals (BOUNDSHEET position patched afterwards)
emit(bof(0x0005), "BOF(globals)", 0x0809)
emit(codepage(), "CODEPAGE", 0x0042)
bs_offset = len(stream)
emit(boundsheet(0, "evidence"), "BOUNDSHEET", 0x0085)   # pos patched below
emit(sst(SST), "SST", 0x00FC)
emit(xf(is_style=True), "XF[0] style", 0x00E0)
emit(xf(is_style=False), "XF[1] cell", 0x00E0)
emit(EOF, "EOF(globals)", 0x000A)

sheet_pos = len(stream)
# patch BOUNDSHEET lbPlyPos (first 4 bytes of its data)
struct.pack_into("<I", stream, bs_offset + 4, sheet_pos)

# worksheet substream
emit(bof(0x0010), "BOF(worksheet)", 0x0809)
emit(dimensions(0, 7, 0, 3), "DIMENSIONS", 0x0200)

emit(labelsst(0, 0, 1, 0),  "LABELSST", 0x00FD, 0, 0)
emit(labelsst(0, 1, 1, 1),  "LABELSST", 0x00FD, 0, 1)
emit(labelsst(0, 2, 1, 2),  "LABELSST", 0x00FD, 0, 2)

emit(labelsst(1, 0, 1, 3),  "LABELSST", 0x00FD, 1, 0)
emit(number(1, 1, 1, 0.0),  "NUMBER(0.0)", 0x0203, 1, 1)   # kind C: zero
emit(labelsst(1, 2, 1, 4),  "LABELSST", 0x00FD, 1, 2)

emit(labelsst(2, 0, 1, 5),  "LABELSST", 0x00FD, 2, 0)
emit(blank(2, 1, 1),        "BLANK", 0x0201, 2, 1)          # kind B: formatted blank
emit(labelsst(2, 2, 1, 6),  "LABELSST", 0x00FD, 2, 2)

emit(labelsst(3, 0, 1, 7),  "LABELSST", 0x00FD, 3, 0)
emit(formula_empty_string(3, 1, 1), "FORMULA(str)", 0x0006, 3, 1)  # kind D
emit(string_record(""),     "STRING('')", 0x0207, 3, 1)
emit(labelsst(3, 2, 1, 8),  "LABELSST", 0x00FD, 3, 2)

emit(labelsst(4, 0, 1, 9),  "LABELSST", 0x00FD, 4, 0)
# (4,1): kind A -- deliberately no record
emit(labelsst(4, 2, 1, 10), "LABELSST", 0x00FD, 4, 2)

emit(labelsst(5, 0, 1, 11), "LABELSST", 0x00FD, 5, 0)
# (5,1),(5,2): trailing holes -- no records

emit(labelsst(6, 0, 1, 12), "LABELSST", 0x00FD, 6, 0)
emit(blank(6, 1, 1),        "BLANK", 0x0201, 6, 1)          # trailing blank
# (6,2): trailing hole -- no record

emit(EOF, "EOF(worksheet)", 0x000A)

with open(OUT, "wb") as f:
    f.write(bytes(stream))
with open(RECORD_MAP, "w") as f:
    json.dump(record_map, f, indent=2)

print("wrote %s (%d bytes), %d records" % (OUT, len(stream), len(record_map)))
for r in record_map:
    coord = "" if r["rowx"] is None else " r%d c%d" % (r["rowx"], r["colx"])
    print("  @%4d  %-8s %-16s len=%-3d%s" % (r["offset"], r["opcode"],
                                             r["name"], r["length"], coord))
