"""
Regenerate tests/samples/percent_columns.xls.

openpyxl cannot write the legacy .xls format, so this runs in two steps
from the repository root:

    python tests/fixture_generators/make_percent_columns.py
    soffice --headless --convert-to xls \\
        --outdir tests/samples tests/samples/percent_columns.xlsx
    rm tests/samples/percent_columns.xlsx
"""
import openpyxl

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "pct"

# Column A: ratios under genuine percent formats
ws['A1'] = 0.1
ws['A1'].number_format = '0%'
ws['A2'] = 0.25
ws['A2'].number_format = '0.00%'

# Column B: ordinary numbers, same underlying 0.1 as A1 but different unit
ws['B1'] = 0.1
ws['B1'].number_format = '0.00'
ws['B2'] = 0.25
ws['B2'].number_format = '0.00'

# Column C: mixed column; 0.1 appears under both percent and plain
# formats -> ambiguous, 0.2 has a unique percent meaning
ws['C1'] = 0.1
ws['C1'].number_format = '0%'
ws['C2'] = 0.1
ws['C2'].number_format = '0.00'
ws['C3'] = 0.2
ws['C3'].number_format = '0.00%'

# Column D: percent signs that are literal text, not scaling markers
ws['D1'] = 0.1
ws['D1'].number_format = '0.00" %"'
ws['D2'] = 0.1
ws['D2'].number_format = '0.00\\%'
# quoted percent plus a genuine marker -> still a ratio
ws['D3'] = 0.1
ws['D3'].number_format = '0.00" share: "0.00%'
# unique value whose only percent sign is quoted: unit meaning is unique
# (a plain number), so it passes through automatically without scaling
ws['D4'] = 0.3
ws['D4'].number_format = '0.00" %"'

# Column E: explicit conditional sections
ws['E1'] = 2.0
ws['E1'].number_format = '[>=1]0.0%;[<1]0.00'
ws['E2'] = 0.5
ws['E2'].number_format = '[>=1]0.0%;[<1]0.00'
# 2.0 again under a plain format -> the 2.0 group becomes ambiguous
ws['E3'] = 2.0
ws['E3'].number_format = '0.00'

# Column F: positional sections, the zero section has no percent marker
ws['F1'] = 0.5
ws['F1'].number_format = '0.00%;-0.00%;0.00'
ws['F2'] = -0.5
ws['F2'].number_format = '0.00%;-0.00%;0.00'
ws['F3'] = 0
ws['F3'].number_format = '0.00%;-0.00%;0.00'
# zero under a percent format -> the 0 group is now ambiguous
ws['F4'] = 0
ws['F4'].number_format = '0%'

# Column G: the same stored value under two distinct percent XFs; the unit
# meaning is unique (a ratio in both), so both records auto-convert.
ws['G1'] = 0.1
ws['G1'].number_format = '0%'
ws['G2'] = 0.1
ws['G2'].number_format = '0.00%'

wb.save('tests/samples/percent_columns.xlsx')
