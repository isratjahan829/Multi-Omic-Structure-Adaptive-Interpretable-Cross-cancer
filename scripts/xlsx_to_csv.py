"""Convert an exported .xlsx cohort sheet to the CSV the loader expects.

Excel caps a worksheet at 16 384 columns, so a merged TCGA table saved as
.xlsx is silently truncated — usually mid-way through the CNA block. This
script reports how much survived so you know what you are working with, and
writes a CSV that `mosaic.data.loader` can read directly.

    python scripts/xlsx_to_csv.py Data.xlsx --out ./real --sheet BLCA_merged_full
"""

from __future__ import annotations

import argparse
import collections
import csv
import os

EXCEL_COLUMN_LIMIT = 16384


def convert(src: str, sheet: str, dest: str) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(src, read_only=True)
    ws = wb[sheet] if sheet in wb.sheetnames else wb[wb.sheetnames[0]]
    rows = ws.iter_rows(values_only=True)

    header = [h for h in next(rows) if h is not None]
    counts = collections.Counter(str(h).split("_")[0] for h in header)
    print(f"[{ws.title}] {ws.max_row - 1} rows x {len(header)} columns")
    for prefix in ("MUT", "CNA", "RNA", "EXP", "METH"):
        if counts.get(prefix):
            print(f"    {prefix}_*: {counts[prefix]}")
    if len(header) >= EXCEL_COLUMN_LIMIT:
        print(f"    !! {len(header)} columns hits Excel's {EXCEL_COLUMN_LIMIT} "
              "limit -- this sheet is TRUNCATED. Export the original CSV "
              "instead if you can.")

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    written = 0
    with open(dest, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            if row[0] is None:
                continue
            writer.writerow(["" if v is None else v for v in row[: len(header)]])
            written += 1
    print(f"    wrote {dest} ({written} rows)")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--sheet", default=None,
                    help="sheet name; defaults to every *_merged_full sheet")
    ap.add_argument("--out", default="./real")
    ap.add_argument("--cohort", default=None,
                    help="output name; defaults to the sheet's prefix")
    args = ap.parse_args()

    import openpyxl
    wb = openpyxl.load_workbook(args.xlsx, read_only=True)
    sheets = [args.sheet] if args.sheet else [
        s for s in wb.sheetnames if s.endswith("_merged_full")] or wb.sheetnames[:1]

    for sheet in sheets:
        cohort = args.cohort or sheet.replace("_merged_full", "")
        convert(args.xlsx, sheet,
                os.path.join(args.out, f"{cohort}_merged_full.csv"))


if __name__ == "__main__":
    main()
