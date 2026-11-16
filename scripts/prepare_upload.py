"""Shrink the cohort tables so they can be moved around.

A merged TCGA cohort is ~46 000 columns wide, most of them mutation and
copy-number calls that are zero for nearly every patient. The analysis keeps at
most a couple of thousand genes anyway, so carrying all of them costs bandwidth
and buys nothing.

This script keeps every clinical column and the top-N genes per omic block,
ranked by an **unsupervised** statistic — alteration frequency for mutation and
copy number, variance for continuous blocks. No label, no survival time and no
train/test split is consulted, so this is a pre-filter of the feature space, not
a selection step, and it does not leak. Say so in the methods anyway: "an
unsupervised variance/frequency pre-filter retained the top N genes per omic
block".

    python scripts/prepare_upload.py --root ./data --out ./data_small --top 6000

Output is parquet by default (`--format csv.gz` if you need text), which the
loader reads directly.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd

PREFIXES = {
    "mut": ["MUT_"],
    "cna": ["CNA_"],
    "exp": ["RNA_", "EXP_", "MRNA_", "GEXP_"],
    "met": ["METH_", "MET_"],
}


def classify(columns: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {k: [] for k in PREFIXES}
    out["clinical"] = []
    for c in columns:
        for block, prefs in PREFIXES.items():
            if any(c.startswith(p) for p in prefs):
                out[block].append(c)
                break
        else:
            out["clinical"].append(c)
    return {k: v for k, v in out.items() if v}


def rank_genes(df: pd.DataFrame, cols: List[str], block: str) -> np.ndarray:
    """Higher is better. Unsupervised: no labels are touched."""
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    with np.errstate(invalid="ignore"):
        if block == "mut":
            return np.nan_to_num(sub.abs().gt(0).mean().values)
        if block == "cna":
            return np.nan_to_num(sub.abs().gt(0.5).mean().values)
        return np.nan_to_num(sub.std().values)


def downcast(df: pd.DataFrame, blocks: Dict[str, List[str]]) -> pd.DataFrame:
    """int8 for the discrete blocks, float32 for the continuous ones."""
    for block, cols in blocks.items():
        if block == "clinical" or not cols:
            continue
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        if block in ("mut", "cna"):
            df[cols] = sub.fillna(0).round().clip(-2, 2).astype(np.int8)
        else:
            df[cols] = sub.astype(np.float32)
    return df


def process(path: str, dest_dir: str, top: int, fmt: str) -> str:
    name = os.path.basename(path)
    df = pd.read_csv(path, low_memory=False)
    blocks = classify(df.columns.tolist())

    keep = list(blocks.get("clinical", []))
    report = []
    for block, cols in blocks.items():
        if block == "clinical":
            continue
        if len(cols) <= top:
            chosen = cols
        else:
            scores = rank_genes(df, cols, block)
            chosen = [cols[i] for i in np.argsort(-scores)[:top]]
        keep.extend(chosen)
        report.append(f"{block}: {len(cols)} -> {len(chosen)}")

    out = downcast(df[keep].copy(), blocks)
    stem = name.replace(".csv", "")
    if fmt == "parquet":
        dest = os.path.join(dest_dir, stem + ".parquet")
        out.to_parquet(dest, compression="zstd", index=False)
    else:
        dest = os.path.join(dest_dir, stem + ".csv.gz")
        out.to_csv(dest, index=False, compression="gzip")

    before = os.path.getsize(path) / 1e6
    after = os.path.getsize(dest) / 1e6
    print(f"{name}: {df.shape[1]} -> {out.shape[1]} columns  "
          f"({', '.join(report)})")
    print(f"    {before:.0f} MB -> {after:.0f} MB  ({before / max(after, 1e-9):.0f}x)"
          f"  {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory with the cohort CSVs")
    ap.add_argument("--out", default="./data_small")
    ap.add_argument("--top", type=int, default=6000,
                    help="genes kept per omic block (the model uses ~2000)")
    ap.add_argument("--format", choices=["parquet", "csv.gz"], default="parquet")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "**", "*_merged_full.csv"),
                             recursive=True))
    if not files:
        raise SystemExit(f"no *_merged_full.csv under {args.root}")
    os.makedirs(args.out, exist_ok=True)

    total = 0.0
    for f in files:
        total += os.path.getsize(process(f, args.out, args.top, args.format)) / 1e6
    print(f"\n{len(files)} cohorts, {total:.0f} MB total in {args.out}")
    print("Zip that directory and upload it.")


if __name__ == "__main__":
    main()
