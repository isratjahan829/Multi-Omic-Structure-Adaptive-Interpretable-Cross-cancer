"""Build {COHORT}_merged_full.parquet from raw cBioPortal study exports.

The Drive folder holds the raw per-study files, not the merged wide tables the
pipeline reads:

    <cohort>_data_clinical_patient*.txt   '#'-commented header, one row/patient
    <cohort>_clinical_sample*.txt         '#'-commented header, one row/sample
    <cohort>_data_mutations*.txt          MAF: one row per mutation call
    <cohort>_data_cna*.txt                Hugo_Symbol + one column per sample
    <cohort>_data_mrna_seq_v2_rsem*.txt   Hugo_Symbol + one column per sample

This script turns them into one row per patient with MUT_/CNA_/RNA_ prefixed
gene columns plus the clinical columns, which is exactly what
mosaic.data.loader expects.

Two things are deliberate:

* **Gene pre-filtering is unsupervised.** Genes are ranked by mutation
  frequency (mutation block) or by variance (continuous blocks) across *all*
  patients, with no reference to any label or to the train/test split. Picking
  genes by their association with the outcome here would leak the label into
  every fold downstream, which is the single easiest way to produce a
  beautiful, meaningless result. The model's own fold-wise selection then
  narrows further using training rows only.
* **Parquet, not CSV.** A 3000-gene x 3-block wide table in CSV is hundreds of
  megabytes per cohort and loses dtypes; parquet with int8/float32 keeps it
  small enough to move around and read back exactly.

Usage
-----
    python scripts/merge_cbioportal.py --raw ./raw --out ./data_real \
        --cohorts blca lihc thca lung --genes-per-block 3000
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Mutation calls that do not change the protein carry little signal for a
# gene-level binary indicator and mostly add noise.
SILENT = {
    "Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "IGR",
    "RNA", "Splice_Region",
}


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def find_file(raw: str, cohort: str, *patterns: str) -> Optional[str]:
    """First file matching any pattern, searched case-insensitively.

    The Drive exports carry inconsistent suffixes -- "-1", "-2", " (1)" --
    so matching has to be loose rather than exact.
    """
    for pat in patterns:
        hits = [
            p for p in glob.glob(os.path.join(raw, "**", "*"), recursive=True)
            if os.path.isfile(p)
            and cohort.lower() in os.path.basename(p).lower()
            and re.search(pat, os.path.basename(p), re.I)
        ]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def read_commented(path: str) -> pd.DataFrame:
    """Read a cBioPortal clinical file, skipping its '#' metadata rows."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        skip = 0
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break
    return pd.read_csv(path, sep="\t", skiprows=skip, low_memory=False)


def patient_of(barcode: str) -> str:
    """TCGA-AB-1234-01A-11D-1234-08 -> TCGA-AB-1234."""
    parts = str(barcode).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(barcode)


# --------------------------------------------------------------------------- #
# Per-block builders
# --------------------------------------------------------------------------- #
def build_mutation_block(path: str, keep: int) -> pd.DataFrame:
    """Gene-level binary mutation matrix, patients x genes."""
    usecols = ["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"]
    chunks = []
    for chunk in pd.read_csv(path, sep="\t", comment="#", usecols=lambda c: c in usecols,
                             chunksize=500_000, low_memory=False):
        if "Variant_Classification" in chunk.columns:
            chunk = chunk[~chunk["Variant_Classification"].isin(SILENT)]
        chunks.append(chunk[["Hugo_Symbol", "Tumor_Sample_Barcode"]])
    maf = pd.concat(chunks, ignore_index=True)
    maf["patient"] = maf["Tumor_Sample_Barcode"].map(patient_of)
    maf = maf.dropna(subset=["Hugo_Symbol"])

    # One row per (patient, gene); duplicates collapse to a single 1.
    pairs = maf[["patient", "Hugo_Symbol"]].drop_duplicates()
    freq = pairs["Hugo_Symbol"].value_counts()
    genes = freq.head(keep).index.tolist()
    pairs = pairs[pairs["Hugo_Symbol"].isin(genes)]

    mat = (pairs.assign(v=np.int8(1))
                .pivot_table(index="patient", columns="Hugo_Symbol", values="v",
                             fill_value=0, aggfunc="max")
                .astype(np.int8))
    mat.columns = [f"MUT_{g}" for g in mat.columns]
    return mat


def build_matrix_block(path: str, prefix: str, keep: int,
                       log_transform: bool) -> pd.DataFrame:
    """Wide gene x sample file -> patients x genes, top `keep` by variance."""
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else df.columns[0]
    df = df[df[gene_col].notna()]
    drop = [c for c in ("Entrez_Gene_Id", "Cytoband") if c in df.columns]
    sample_cols = [c for c in df.columns if c not in drop + [gene_col]]

    # Collapse duplicate gene symbols by their mean.
    values = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    values.insert(0, "__gene__", df[gene_col].astype(str).values)
    values = values.groupby("__gene__").mean()

    if log_transform:
        values = np.log2(values.clip(lower=0) + 1.0)

    var = values.var(axis=1, skipna=True)
    genes = var.sort_values(ascending=False).head(keep).index
    sub = values.loc[genes].T                      # samples x genes
    sub.index = [patient_of(s) for s in sub.index]
    sub = sub.groupby(level=0).mean()              # collapse multi-sample patients
    sub.columns = [f"{prefix}{g}" for g in sub.columns]
    return sub.astype(np.float32)


# --------------------------------------------------------------------------- #
def merge_cohort(raw: str, cohort: str, keep: int) -> Optional[pd.DataFrame]:
    pat_path = find_file(raw, cohort, r"clinical_patient")
    smp_path = find_file(raw, cohort, r"clinical_sample")
    mut_path = find_file(raw, cohort, r"mutation")
    cna_path = find_file(raw, cohort, r"(^|_)cna")
    rna_path = find_file(raw, cohort, r"mrna|rsem|expression")

    print(f"\n[{cohort.upper()}]")
    for label, path in (("clinical_patient", pat_path), ("clinical_sample", smp_path),
                        ("mutations", mut_path), ("cna", cna_path), ("mrna", rna_path)):
        print(f"  {label:<18} {os.path.basename(path) if path else '-- MISSING --'}")

    if pat_path is None:
        print(f"  !! no clinical_patient file for {cohort}; skipping cohort")
        return None

    clinical = read_commented(pat_path)
    id_col = "PATIENT_ID" if "PATIENT_ID" in clinical.columns else clinical.columns[0]
    clinical[id_col] = clinical[id_col].astype(str).str.strip()
    clinical = clinical.drop_duplicates(subset=[id_col]).set_index(id_col)

    if smp_path is not None:
        sample = read_commented(smp_path)
        if "PATIENT_ID" in sample.columns:
            sample["PATIENT_ID"] = sample["PATIENT_ID"].astype(str).str.strip()
            # Keep the first sample per patient, preferring a primary tumour.
            if "SAMPLE_TYPE" in sample.columns:
                sample["_prio"] = (~sample["SAMPLE_TYPE"].astype(str)
                                   .str.contains("Primary", case=False, na=False)).astype(int)
                sample = sample.sort_values("_prio").drop(columns="_prio")
            sample = sample.drop_duplicates(subset=["PATIENT_ID"]).set_index("PATIENT_ID")
            new = [c for c in sample.columns if c not in clinical.columns]
            clinical = clinical.join(sample[new], how="left")

    blocks: Dict[str, pd.DataFrame] = {}
    if mut_path:
        blocks["mut"] = build_mutation_block(mut_path, keep)
    if cna_path:
        blocks["cna"] = build_matrix_block(cna_path, "CNA_", keep, log_transform=False)
    if rna_path:
        blocks["rna"] = build_matrix_block(rna_path, "RNA_", keep, log_transform=True)
    for name, blk in blocks.items():
        print(f"  block {name:<4} {blk.shape[0]:>5} patients x {blk.shape[1]:>5} genes")

    if not blocks:
        print(f"  !! no omic block for {cohort}; skipping cohort")
        return None

    # Patients with at least one omic block AND clinical data.
    omic_patients = sorted(set().union(*[set(b.index) for b in blocks.values()]))
    keep_patients = [p for p in omic_patients if p in clinical.index]
    print(f"  patients with clinical + >=1 omic block: {len(keep_patients)}")
    if not keep_patients:
        print("  !! patient identifiers did not match between clinical and omics")
        return None

    out = clinical.loc[keep_patients].copy()
    for blk in blocks.values():
        out = out.join(blk.reindex(keep_patients), how="left")

    # CANCER_TYPE_ACRONYM must name the *cohort as loaded*, because
    # mosaic.data.loader builds the tissue-of-origin label vocabulary from
    # DataConfig.cohorts and matches it against this column. The clinical
    # export usually ships its own acronym already (TCGA PanCanAtlas Cancer
    # Type Acronym), and for a merged study such as "lung" that column is a
    # mixture (LUAD / LUSC) rather than the cohort name -- so it is renamed
    # out of the way instead of being dropped or silently reused.
    for col in ("PATIENT_ID", "CANCER_TYPE_ACRONYM"):
        if col in out.columns:
            out = out.rename(columns={col: f"TCGA_{col}"})
    out.insert(0, "PATIENT_ID", out.index)
    out.insert(1, "CANCER_TYPE_ACRONYM", cohort.upper())
    out = out.reset_index(drop=True)

    native = out.get("TCGA_CANCER_TYPE_ACRONYM")
    if native is not None:
        mix = native.astype(str).value_counts().to_dict()
        print(f"  study's own acronym column: {mix} "
              f"(kept as TCGA_CANCER_TYPE_ACRONYM; label uses {cohort.upper()})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory holding the raw exports")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cohorts", nargs="+", required=True)
    ap.add_argument("--genes-per-block", type=int, default=3000)
    ap.add_argument("--csv", action="store_true",
                    help="also write CSV (large; parquet is written either way)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    written = []
    for cohort in args.cohorts:
        df = merge_cohort(args.raw, cohort, args.genes_per_block)
        if df is None:
            continue
        path = os.path.join(args.out, f"{cohort.upper()}_merged_full.parquet")
        df.to_parquet(path, index=False, compression="zstd")
        size = os.path.getsize(path) / 1e6
        print(f"  wrote {path}  shape={df.shape}  {size:.1f} MB")
        if args.csv:
            cpath = os.path.join(args.out, f"{cohort.upper()}_merged_full.csv")
            df.to_csv(cpath, index=False)
            print(f"  wrote {cpath}  {os.path.getsize(cpath) / 1e6:.1f} MB")
        written.append((cohort.upper(), df.shape))

    print("\nsummary")
    for name, shape in written:
        print(f"  {name:<6} {shape[0]:>5} patients x {shape[1]:>6} columns")
    if not written:
        raise SystemExit("no cohort could be merged -- check --raw and --cohorts")


if __name__ == "__main__":
    main()
