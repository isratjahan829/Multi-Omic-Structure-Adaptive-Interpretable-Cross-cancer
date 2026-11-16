"""Cohort loading and harmonisation.

The merged TCGA tables are wide (10^4-10^5 columns) and heterogeneous across
cohorts, so loading is done in two passes: a cheap header scan that decides
which columns are worth materialising, then a typed read of just those.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import DataConfig

SURVIVAL_EVENT_TOKENS = ("DECEASED", "PROGRESSION", "RECURRED", "EVENT")


@dataclass
class OmicBlock:
    """One omic modality for one cohort, aligned to a gene symbol axis."""

    name: str
    genes: List[str]
    values: np.ndarray  # (n_patients, n_genes), float32

    def subset(self, genes: List[str]) -> np.ndarray:
        """Return values re-indexed onto `genes`; absent genes become NaN."""
        pos = {g: i for i, g in enumerate(self.genes)}
        out = np.full((self.values.shape[0], len(genes)), np.nan, dtype=np.float32)
        idx_src, idx_dst = [], []
        for j, g in enumerate(genes):
            i = pos.get(g)
            if i is not None:
                idx_src.append(i)
                idx_dst.append(j)
        if idx_src:
            out[:, idx_dst] = self.values[:, idx_src]
        return out


@dataclass
class Cohort:
    name: str
    patients: List[str]
    blocks: Dict[str, OmicBlock]
    clinical: pd.DataFrame  # raw clinical columns, patient-indexed
    time: np.ndarray        # survival time in months, float32
    event: np.ndarray       # 1 = event observed, 0 = censored
    label: np.ndarray       # integer class label, -1 = missing

    def __len__(self) -> int:
        return len(self.patients)


# --------------------------------------------------------------------------- #
# Header inspection
# --------------------------------------------------------------------------- #
# The merged tables may arrive as CSV, gzipped CSV or parquet. Trying only
# the configured template is how a run once failed to find real parquet data
# and silently continued on the synthetic fallback instead.
COHORT_SUFFIXES = (".csv", ".parquet", ".pq", ".csv.gz")


def cohort_path(cohort: str, cfg: DataConfig) -> Optional[str]:
    """Locate a cohort's merged table, whatever extension it was written with."""
    template = cfg.file_template.format(cohort=cohort)
    stem = template
    for suf in COHORT_SUFFIXES:
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            break

    candidates = [template] + [stem + suf for suf in COHORT_SUFFIXES]
    for name in candidates:
        path = os.path.join(cfg.root, name)
        if os.path.exists(path):
            return path

    # Last resort: a case-insensitive match anywhere under root, which covers
    # exports that lower-case the cohort name.
    for folder, _, files in os.walk(cfg.root):
        for name in files:
            low = name.lower()
            if (low.startswith(f"{cohort.lower()}_merged_full")
                    and low.endswith(COHORT_SUFFIXES)):
                return os.path.join(folder, name)
    return None


def read_table(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a merged cohort table, dispatching on its extension."""
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        return pd.read_parquet(path, columns=usecols)
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def table_columns(path: str) -> List[str]:
    """Column names only, without reading the payload."""
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema.names)
    return pd.read_csv(path, nrows=0).columns.tolist()


def scan_columns(path: str, cfg: DataConfig) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Map every omic column to (block, gene) without reading the payload.

    Returns
    -------
    blocks : {block_name: {column_name: gene_symbol}}
    other  : columns that are not omic (clinical / identifiers)
    """
    header = table_columns(path)
    blocks: Dict[str, Dict[str, str]] = {k: {} for k in cfg.omic_prefixes}
    other: List[str] = []
    for col in header:
        matched = False
        for block, prefixes in cfg.omic_prefixes.items():
            for pref in prefixes:
                if col.startswith(pref):
                    blocks[block][col] = col[len(pref):]
                    matched = True
                    break
            if matched:
                break
        if not matched:
            other.append(col)
    return {k: v for k, v in blocks.items() if v}, other


# --------------------------------------------------------------------------- #
# Survival / label parsing
# --------------------------------------------------------------------------- #
def _clean_strings(values) -> pd.Series:
    """Upper-cased plain-object string Series, whatever the input dtype.

    Parquet hands back pandas extension arrays ("string", "large_string"),
    where ``astype(str)`` turns pd.NA into a float nan rather than the text
    "nan". Anything downstream that does ``t in s`` then raises
    ``TypeError: argument of type 'float' is not iterable``. Converting
    through object with an explicit fill keeps every path on real strings.
    """
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    return (s.astype(object)
             .where(s.notna(), "")
             .map(lambda v: "" if v is None else str(v))
             .str.upper())


def parse_survival(df: pd.DataFrame, endpoint: str) -> Tuple[np.ndarray, np.ndarray]:
    """cBioPortal encodes status as '1:DECEASED' / '0:LIVING'."""
    status_col, months_col = f"{endpoint}_STATUS", f"{endpoint}_MONTHS"
    if status_col not in df.columns or months_col not in df.columns:
        n = len(df)
        return np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)

    raw = _clean_strings(df[status_col])
    event = np.zeros(len(df), dtype=np.float32)
    lead = raw.str.extract(r"^\s*(\d)\s*:", expand=False)
    has_lead = lead.notna()
    event[has_lead.values] = lead[has_lead].astype(float).values
    if (~has_lead).any():
        token = raw[~has_lead].apply(
            lambda s: float(any(t in s for t in SURVIVAL_EVENT_TOKENS))
        )
        event[(~has_lead).values] = token.values
    event[raw.isin(["NAN", "", "NA", "[NOT AVAILABLE]"]).values] = np.nan

    time = pd.to_numeric(df[months_col], errors="coerce").astype(np.float32).values
    return time, event


_STAGE_MAP = {
    "STAGE I": 0, "STAGE IA": 0, "STAGE IB": 0, "STAGE IC": 0,
    "STAGE II": 1, "STAGE IIA": 1, "STAGE IIB": 1, "STAGE IIC": 1,
    "STAGE III": 2, "STAGE IIIA": 2, "STAGE IIIB": 2, "STAGE IIIC": 2,
    "STAGE IV": 3, "STAGE IVA": 3, "STAGE IVB": 3, "STAGE IVC": 3,
}


def parse_stage(series: pd.Series) -> np.ndarray:
    s = _clean_strings(series).str.strip()
    s = s.str.replace(r"^STAGE\s+0.*$", "STAGE I", regex=True)
    return s.map(_STAGE_MAP).fillna(-1).astype(int).values


def parse_grade(series: pd.Series) -> np.ndarray:
    s = _clean_strings(series).str.strip()
    out = np.full(len(s), -1, dtype=int)
    out[s.str.contains("LOW|G1|G2", regex=True, na=False)] = 0
    out[s.str.contains("HIGH|G3|G4", regex=True, na=False)] = 1
    return out


def build_labels(clinical: pd.DataFrame, cfg: DataConfig,
                 vocab: Optional[Dict[str, int]] = None
                 ) -> Tuple[np.ndarray, Dict[str, int]]:
    """Integer class labels plus the string->index vocabulary used."""
    target = cfg.class_target
    if target == "stage":
        col = "AJCC_PATHOLOGIC_TUMOR_STAGE"
        if col not in clinical.columns:
            return np.full(len(clinical), -1), {}
        return parse_stage(clinical[col]), {"I": 0, "II": 1, "III": 2, "IV": 3}
    if target == "grade":
        return parse_grade(clinical.get("GRADE", pd.Series(dtype=str))), {
            "low": 0, "high": 1
        }

    col = {"cohort": cfg.cohort_col, "subtype": "SUBTYPE"}.get(target, target)
    if col not in clinical.columns:
        return np.full(len(clinical), -1), {}
    vals = _clean_strings(clinical[col]).str.strip()
    if vocab is None:
        vocab = {v: i for i, v in enumerate(sorted(vals[vals != "nan"].unique()))}
    return vals.map(vocab).fillna(-1).astype(int).values, vocab


# --------------------------------------------------------------------------- #
# Omic value normalisation
# --------------------------------------------------------------------------- #
def _coerce_block(values: np.ndarray, block: str) -> np.ndarray:
    """Put every modality on a comparable, sign-preserving scale."""
    v = values.astype(np.float32)
    if block == "mut":
        # Some dumps store counts rather than 0/1; binarise.
        return (np.nan_to_num(v, nan=0.0) > 0).astype(np.float32)
    if block == "cna":
        # GISTIC calls in {-2,-1,0,1,2}; keep sign, squash magnitude to [-1,1].
        return np.clip(np.nan_to_num(v, nan=0.0) / 2.0, -1.0, 1.0)
    if block == "met":
        return np.clip(np.nan_to_num(v, nan=np.nan), 0.0, 1.0)
    return v  # expression: left raw, standardised later on train folds only


def load_cohort(cohort: str, cfg: DataConfig,
                label_vocab: Optional[Dict[str, int]] = None) -> Cohort:
    path = cohort_path(cohort, cfg)
    if path is None:
        raise FileNotFoundError(
            f"no merged table for cohort {cohort!r} under {cfg.root!r} "
            f"(looked for {cfg.file_template.format(cohort=cohort)} and the "
            f"same stem with extensions {', '.join(COHORT_SUFFIXES)}). "
            "Set DataConfig.root to the dataset directory."
        )

    blocks_map, other_cols = scan_columns(path, cfg)
    usecols = other_cols + [c for m in blocks_map.values() for c in m]
    df = read_table(path, usecols=usecols)

    # One row per patient; if a patient contributed several samples keep the
    # primary tumour, else the first occurrence.
    if "SAMPLE_TYPE" in df.columns:
        df["_prio"] = (~_clean_strings(df["SAMPLE_TYPE"]).str.contains(
            "Primary", case=False, na=False)).astype(int)
        df = df.sort_values("_prio").drop(columns="_prio")
    if cfg.id_col in df.columns:
        df = df.drop_duplicates(subset=cfg.id_col, keep="first").reset_index(drop=True)
        patients = df[cfg.id_col].astype(str).tolist()
    else:
        patients = [f"{cohort}-{i}" for i in range(len(df))]

    blocks: Dict[str, OmicBlock] = {}
    for name, colmap in blocks_map.items():
        cols = list(colmap)
        vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        blocks[name] = OmicBlock(name, [colmap[c] for c in cols],
                                 _coerce_block(vals, name))

    clinical = df[[c for c in other_cols if c in df.columns]].copy()
    clinical.index = patients
    time, event = parse_survival(clinical, cfg.survival_endpoint)
    label, _ = build_labels(clinical, cfg, label_vocab)

    return Cohort(cohort, patients, blocks, clinical, time, event, label)


def load_all(cfg: DataConfig) -> Dict[str, Cohort]:
    """Load every cohort with a shared label vocabulary across cohorts."""
    vocab: Optional[Dict[str, int]] = None
    if cfg.class_target in ("cohort",):
        vocab = {c: i for i, c in enumerate(cfg.cohorts)}
    return {c: load_cohort(c, cfg, vocab) for c in cfg.cohorts}
