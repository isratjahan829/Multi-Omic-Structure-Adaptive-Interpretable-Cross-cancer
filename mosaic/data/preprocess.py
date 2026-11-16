"""Fold-safe preprocessing.

Every statistic used here (gene selection, means, variances, scalers, survival
bin edges) is estimated on training indices only and then applied to the held
out split. Nothing in this module ever looks at validation or test rows while
fitting -- that is the single most common source of inflated numbers in
multi-omics papers, so the split is enforced by the API itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import DataConfig
from .loader import Cohort, _clean_strings


@dataclass
class PanCohortData:
    """All cohorts stacked into one pan-cancer matrix set."""

    patients: List[str]
    cohort_id: np.ndarray            # (N,) index into cohort_names
    cohort_names: List[str]
    blocks: Dict[str, np.ndarray]    # block -> (N, G) on the shared gene axis
    block_mask: Dict[str, np.ndarray]  # block -> (N,) 1 if measured
    genes: List[str]
    clinical: np.ndarray             # (N, C) encoded, already imputed
    clinical_names: List[str]
    time: np.ndarray
    event: np.ndarray
    label: np.ndarray

    def __len__(self) -> int:
        return len(self.patients)


@dataclass
class FittedPreprocessor:
    genes: List[str]
    block_names: List[str]
    clinical_names: List[str]
    center: Dict[str, np.ndarray] = field(default_factory=dict)
    scale: Dict[str, np.ndarray] = field(default_factory=dict)
    clin_center: Optional[np.ndarray] = None
    clin_scale: Optional[np.ndarray] = None
    surv_bins: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- #
# Gene selection (train rows only)
# --------------------------------------------------------------------------- #
def _score_genes(values: np.ndarray, block: str, cfg: DataConfig) -> np.ndarray:
    """Higher is better. NaNs are treated as unmeasured, not as zeros."""
    with np.errstate(invalid="ignore"):
        if block == "mut":
            return np.nanmean(np.abs(values) > 0, axis=0)
        if block == "cna":
            return np.nanmean(np.abs(values) > 0.25, axis=0)
        return np.nanstd(values, axis=0)


def select_genes(cohorts: Dict[str, Cohort], train_idx: Dict[str, np.ndarray],
                 cfg: DataConfig) -> List[str]:
    """Union of per-block top-K genes, ranked by cross-cohort consistency.

    A gene is kept when it is informative in *several* cohorts, which is what
    makes the learned gene graph transferable rather than cohort-specific.
    """
    per_block: Dict[str, Dict[str, List[float]]] = {}
    for cname, cohort in cohorts.items():
        rows = train_idx[cname]
        for bname, block in cohort.blocks.items():
            if len(rows) == 0:
                continue
            scores = _score_genes(block.values[rows], bname, cfg)
            if bname == "mut":
                keep = scores >= cfg.min_mut_freq
            else:
                thr = np.nanquantile(scores, cfg.var_quantile) if np.isfinite(
                    scores).any() else 0.0
                keep = scores >= thr
            store = per_block.setdefault(bname, {})
            order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
            rank = np.empty(len(order), dtype=np.float64)
            rank[order] = np.arange(len(order))
            rank = 1.0 - rank / max(len(order) - 1, 1)  # 1 = best
            for g_i, gene in enumerate(block.genes):
                if keep[g_i]:
                    store.setdefault(gene, []).append(float(rank[g_i]))

    selected: List[str] = []
    for bname, store in per_block.items():
        ranked = sorted(
            store.items(),
            key=lambda kv: (-len(kv[1]), -float(np.mean(kv[1]))),
        )
        selected.extend([g for g, _ in ranked[: cfg.n_genes_per_block]])

    seen, genes = set(), []
    for g in selected:
        if g not in seen:
            seen.add(g)
            genes.append(g)
        if len(genes) >= cfg.max_genes:
            break
    return genes


# --------------------------------------------------------------------------- #
# Label leakage
# --------------------------------------------------------------------------- #
# Clinical columns that are the label's own source, or a near-deterministic
# restatement of it. Leaving any of these in the feature matrix makes the task
# trivial: an earlier run of this pipeline scored a perfect 1.000 for every
# model, baselines included, purely because AJCC_PATHOLOGIC_TUMOR_STAGE was
# both the stage label and a clinical feature. Matching is by substring on the
# upper-cased column name, so one-hot children of a leaky column go too.
LEAKY_FOR_TARGET: Dict[str, Tuple[str, ...]] = {
    "stage": (
        "AJCC", "TUMOR_STAGE", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE",
        "CLIN_T_STAGE", "CLIN_N_STAGE", "CLIN_M_STAGE", "STAGE_EVENT",
    ),
    "grade": (
        "GRADE",
    ),
    "cohort": (
        "CANCER_TYPE", "SUBTYPE", "ONCOTREE", "ICD_O", "ICD_10",
        "TISSUE_SOURCE_SITE", "TUMOR_TISSUE_SITE", "PRIMARY_SITE",
        "ORGAN_OF_ORIGIN", "TISSUE_PROSPECTIVE", "TISSUE_RETROSPECTIVE",
    ),
    "subtype": (
        "SUBTYPE", "CANCER_TYPE", "ONCOTREE", "ICD_O", "ICD_10",
        "TISSUE_SOURCE_SITE", "TUMOR_TISSUE_SITE", "PRIMARY_SITE",
        "MOLECULAR_SUBTYPE", "MRNA_CLUSTER", "METHYLATION_CLUSTER",
    ),
}


def leaky_columns(cfg: DataConfig) -> List[str]:
    """Clinical columns that must not be used as features for cfg.class_target.

    Returns the configured clinical column names that match a leaky pattern,
    so a caller can both filter them and report exactly what was dropped.
    """
    patterns = LEAKY_FOR_TARGET.get(str(cfg.class_target).lower(), ())
    if not patterns:
        return []
    cols = list(cfg.clinical_numeric) + list(cfg.clinical_categorical)
    return [c for c in cols
            if any(p in c.upper() for p in patterns)]


def _is_leaky(name: str, cfg: DataConfig) -> bool:
    patterns = LEAKY_FOR_TARGET.get(str(cfg.class_target).lower(), ())
    up = str(name).upper()
    return any(p in up for p in patterns)


# --------------------------------------------------------------------------- #
# Clinical encoding
# --------------------------------------------------------------------------- #
def encode_clinical(cohorts: Dict[str, Cohort], cfg: DataConfig
                    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    frames = []
    for cname, cohort in cohorts.items():
        sub = cohort.clinical.copy()
        sub["__cohort__"] = cname
        frames.append(sub)
    allc = pd.concat(frames, axis=0)

    num_cols = [c for c in cfg.clinical_numeric
                if c in allc.columns and not _is_leaky(c, cfg)]
    num = allc[num_cols].apply(pd.to_numeric, errors="coerce")

    cat_cols = [c for c in cfg.clinical_categorical
                if c in allc.columns and not _is_leaky(c, cfg)]
    cat = allc[cat_cols].apply(lambda s: _clean_strings(s).str.strip())
    cat = cat.replace({"NAN": np.nan, "[NOT AVAILABLE]": np.nan, "": np.nan})
    dummies = pd.get_dummies(cat, dummy_na=False, dtype=np.float32)

    mat = pd.concat([num, dummies], axis=1)
    # Second pass: a one-hot child inherits its parent's name, but guard
    # against any column that slipped through under a different spelling.
    safe = [c for c in mat.columns if not _is_leaky(c, cfg)]
    mat = mat[safe]
    names = mat.columns.tolist()
    cohort_of_row = allc["__cohort__"].values
    return mat.to_numpy(np.float32), names, cohort_of_row


def assemble(cohorts: Dict[str, Cohort], genes: List[str],
             cfg: DataConfig) -> PanCohortData:
    """Project every cohort onto the shared gene axis and concatenate."""
    clin_mat, clin_names, _ = encode_clinical(cohorts, cfg)

    block_names = sorted({b for c in cohorts.values() for b in c.blocks})
    patients, cohort_id, per_block, per_mask = [], [], {b: [] for b in block_names}, {
        b: [] for b in block_names}
    time, event, label = [], [], []
    cohort_names = list(cohorts)

    for ci, cname in enumerate(cohort_names):
        cohort = cohorts[cname]
        n = len(cohort)
        patients.extend([f"{cname}|{p}" for p in cohort.patients])
        cohort_id.extend([ci] * n)
        time.append(cohort.time)
        event.append(cohort.event)
        label.append(cohort.label)
        for b in block_names:
            if b in cohort.blocks:
                per_block[b].append(cohort.blocks[b].subset(genes))
                per_mask[b].append(np.ones(n, dtype=np.float32))
            else:
                per_block[b].append(np.full((n, len(genes)), np.nan, np.float32))
                per_mask[b].append(np.zeros(n, dtype=np.float32))

    return PanCohortData(
        patients=patients,
        cohort_id=np.asarray(cohort_id, dtype=np.int64),
        cohort_names=cohort_names,
        blocks={b: np.concatenate(v, axis=0) for b, v in per_block.items()},
        block_mask={b: np.concatenate(v, axis=0) for b, v in per_mask.items()},
        genes=genes,
        clinical=clin_mat,
        clinical_names=clin_names,
        time=np.concatenate(time).astype(np.float32),
        event=np.concatenate(event).astype(np.float32),
        label=np.concatenate(label).astype(np.int64),
    )


# --------------------------------------------------------------------------- #
# Fit / transform
# --------------------------------------------------------------------------- #
def fit_preprocessor(data: PanCohortData, train_idx: np.ndarray,
                     cfg: DataConfig, n_bins: int) -> FittedPreprocessor:
    pp = FittedPreprocessor(genes=data.genes,
                            block_names=sorted(data.blocks),
                            clinical_names=data.clinical_names)

    for b, mat in data.blocks.items():
        tr = mat[train_idx]
        if b in ("mut", "cna", "met"):
            # Already on a fixed, interpretable scale; only fill gaps.
            pp.center[b] = np.zeros(mat.shape[1], np.float32)
            pp.scale[b] = np.ones(mat.shape[1], np.float32)
        else:
            mu = np.nanmean(tr, axis=0)
            sd = np.nanstd(tr, axis=0)
            pp.center[b] = np.nan_to_num(mu, nan=0.0).astype(np.float32)
            pp.scale[b] = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0).astype(
                np.float32)

    ctr = data.clinical[train_idx]
    pp.clin_center = np.nan_to_num(np.nanmean(ctr, axis=0), nan=0.0).astype(np.float32)
    sd = np.nanstd(ctr, axis=0)
    pp.clin_scale = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0).astype(np.float32)

    # Discrete-time survival bins: quantiles of observed event times on train.
    t = data.time[train_idx]
    e = data.event[train_idx]
    obs = t[(e > 0) & np.isfinite(t)]
    if obs.size >= n_bins:
        edges = np.quantile(obs, np.linspace(0, 1, n_bins + 1)[1:-1])
    else:
        finite = t[np.isfinite(t)]
        hi = float(finite.max()) if finite.size else 1.0
        edges = np.linspace(0, hi, n_bins + 1)[1:-1]
    pp.surv_bins = np.unique(edges).astype(np.float32)
    return pp


def transform(data: PanCohortData, pp: FittedPreprocessor
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the node-feature tensor.

    Returns
    -------
    x     : (N, G, 2*B) node features -- value channel and measured-mask
            channel per omic block.
    clin  : (N, C) standardised clinical matrix.
    ybin  : (N,) discrete survival bin index (-1 when time is missing).
    """
    n, g = len(data), len(data.genes)
    b = len(pp.block_names)
    x = np.zeros((n, g, 2 * b), dtype=np.float32)
    for i, name in enumerate(pp.block_names):
        mat = data.blocks[name]
        measured = np.isfinite(mat)
        vals = np.where(measured, mat, 0.0)
        vals = (vals - pp.center[name]) / pp.scale[name]
        vals = np.where(measured, vals, 0.0)
        x[:, :, i] = np.clip(vals, -8.0, 8.0)
        x[:, :, b + i] = measured.astype(np.float32) * data.block_mask[name][:, None]

    clin = np.where(np.isfinite(data.clinical), data.clinical, np.nan)
    clin = np.where(np.isnan(clin), pp.clin_center, clin)
    clin = ((clin - pp.clin_center) / pp.clin_scale).astype(np.float32)
    clin = np.clip(clin, -8.0, 8.0)

    ybin = np.digitize(data.time, pp.surv_bins).astype(np.int64)
    ybin[~np.isfinite(data.time)] = -1
    return x, clin, ybin


def stratification_key(data: PanCohortData) -> np.ndarray:
    """Cohort x event x label -- keeps every fold comparable."""
    lab = np.where(data.label >= 0, data.label, 99)
    ev = np.nan_to_num(data.event, nan=0.0).astype(int)
    key = data.cohort_id * 1000 + lab * 10 + ev
    # Collapse strata that are too small for k-fold to handle.
    uniq, counts = np.unique(key, return_counts=True)
    rare = set(uniq[counts < 6].tolist())
    return np.asarray([data.cohort_id[i] if key[i] in rare else key[i]
                       for i in range(len(key))])


def valid_mask(data: PanCohortData, need_label: bool, need_surv: bool) -> np.ndarray:
    m = np.ones(len(data), dtype=bool)
    if need_label:
        m &= data.label >= 0
    if need_surv:
        m &= np.isfinite(data.time) & np.isfinite(data.event) & (data.time > 0)
    return m
