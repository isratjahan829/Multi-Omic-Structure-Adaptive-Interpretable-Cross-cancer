"""Verify the raw Drive downloads are real cBioPortal exports, not HTML.

A Drive interstitial or quota page is served with HTTP 200, so the only
trustworthy check is the content itself: every one of these files must start
with a '#' metadata comment or a tab-separated header row, and must be large
enough to hold what it claims to hold.

    python -I scripts/verify_raw.py --raw ./raw

Exit status is non-zero if any file fails, so this can gate the merge step.
"""

from __future__ import annotations

import argparse
import os
import sys

EXPECTED = [
    # (filename, minimum plausible size in bytes)
    ("blca_data_clinical_patient.txt", 50_000),
    ("blca_clinical_sample.txt", 20_000),
    ("blca_data_mutations.txt", 10_000_000),
    ("blca_data_cna.txt", 5_000_000),
    ("blca_data_mrna_seq_v2_rsem.txt", 10_000_000),
    ("lihc_data_clinical_patient.txt", 50_000),
    ("lihc_clinical_sample.txt", 20_000),
    ("lihc_data_mutations.txt", 5_000_000),
    ("lihc_data_cna.txt", 5_000_000),
    ("lihc_data_mrna_seq_v2_rsem.txt", 10_000_000),
    ("thca_data_clinical_patient.txt", 50_000),
    ("thca_clinical_sample.txt", 20_000),
    ("thca_data_mutations.txt", 2_000_000),
    ("thca_data_cna.txt", 5_000_000),
    ("thca_data_mrna_seq_v2_rsem.txt", 10_000_000),
    ("lung_data_clinical_patient.txt", 50_000),
    ("lung_data_clinical_sample.txt", 20_000),
    ("lung_data_mutations.txt", 10_000_000),
    ("lung_data_cna.txt", 5_000_000),
    ("lung_data_mrna_seq_v2_rsem.txt", 10_000_000),
]

HTML_MARKERS = ("<!doctype", "<html", "<head", "<meta", "<script")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="./raw")
    args = ap.parse_args()

    failures = []
    print(f"{'file':<38} {'MB':>8}  {'cols':>5}  verdict")
    for name, floor in EXPECTED:
        path = os.path.join(args.raw, name)
        if not os.path.isfile(path):
            failures.append(f"{name}: missing")
            print(f"{name:<38} {'-':>8}  {'-':>5}  MISSING")
            continue
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(4096).decode("utf-8", "replace")
        first = head.split("\n", 1)[0]
        low = first.lstrip().lower()

        problems = []
        if any(low.startswith(m) for m in HTML_MARKERS) or "<html" in head.lower():
            problems.append("looks like HTML")
        if size < floor:
            problems.append(f"only {size} bytes (< {floor})")
        if not first.startswith("#") and "\t" not in first:
            problems.append("first line is neither a '#' comment nor tab separated")

        ncols = first.count("\t") + 1
        verdict = "OK" if not problems else "FAIL: " + "; ".join(problems)
        print(f"{name:<38} {size / 1e6:8.1f}  {ncols:>5}  {verdict}")
        if problems:
            failures.append(f"{name}: {'; '.join(problems)}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  ", f)
        return 1
    print("\nall 20 raw files verified as tab-separated cBioPortal text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
