"""
Regenerate tests/samples/header_groups.xls.

Development-only helper: the checked-in .xls is the fixture the test suite
actually uses, so running the tests needs neither openpyxl nor LibreOffice.
Use this script only when the fixture itself must be rebuilt.

Requirements: openpyxl and a LibreOffice (soffice) binary able to export the
BIFF8 ``MS Excel 97`` filter.  Override the binary location with the
``SOFFICE`` environment variable.
"""

import os
import subprocess
import sys
import tempfile

import openpyxl

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'samples', 'header_groups.xls')


def build_workbook(path):
    wb = openpyxl.Workbook()
    sales = wb.active
    sales.title = 'sales'

    # Two-level headers: merged parents 期初 / 期末, each with a 数量 leaf.
    sales['A1'] = '商品'
    sales.merge_cells('A1:A2')
    sales['B1'] = '期初'
    sales.merge_cells('B1:C1')
    sales['B2'] = '数量'
    sales['C2'] = '金额'
    sales['D1'] = '期末'
    sales.merge_cells('D1:E1')
    sales['D2'] = '数量'
    sales['E2'] = '金额'
    # Two independent, same-named single-level headers (no merge, no parent).
    sales['G1'] = '数量'
    sales['H1'] = '数量'
    # Column F is never written: truly empty cells in the header block.
    sales['A3'] = '甲'
    sales['B3'] = 1
    sales['C3'] = 2
    sales['D3'] = 3
    sales['E3'] = 4
    sales['G3'] = 5
    sales['H3'] = 6
    # Hide the 期初-数量 column.
    sales.column_dimensions['B'].hidden = True

    wide = wb.create_sheet('wide_merge')
    # One merged parent spanning four leaf columns: it must remain a single
    # header, never be split into two parent headers.
    wide['A1'] = '跨组合计'
    wide.merge_cells('A1:D1')
    for col, name in zip('ABCD', ('数量', '金额', '数量', '金额')):
        wide['%s2' % col] = name
    wide['A3'] = 'row'

    wb.save(path)


def main():
    soffice = os.environ.get('SOFFICE', 'soffice')
    tmpdir = tempfile.mkdtemp(prefix='xlrd-header-fixture-')
    xlsx_path = os.path.join(tmpdir, 'header_groups.xlsx')
    build_workbook(xlsx_path)
    subprocess.check_call([
        soffice, '--headless', '--convert-to', 'xls', xlsx_path,
        '--outdir', tmpdir,
    ])
    with open(os.path.join(tmpdir, 'header_groups.xls'), 'rb') as src:
        data = src.read()
    with open(OUT_PATH, 'wb') as dst:
        dst.write(data)
    print('wrote %s (%d bytes)' % (OUT_PATH, len(data)))


if __name__ == '__main__':
    sys.exit(main())
