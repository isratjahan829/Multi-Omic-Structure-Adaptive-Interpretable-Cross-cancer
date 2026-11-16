"""Overfitting diagnostics.

Four checks, in the order a sceptical reviewer applies them:

1. **generalisation gap** — the same metrics on train, validation and test.
   A gap is not by itself a failure; an *unreported* gap is;
2. **learning curves** — where train and validation separate during training,
   which tells you whether early stopping fired in the right place;
3. **label permutation test** — retrain on shuffled labels. If the pipeline
   leaks, performance stays above chance. This is the only check that can
   actually prove the absence of leakage, and it is the one most papers omit;
4. **sample-size curve** — performance against the number of training patients,
   which says whether the model is data-limited or capacity-limited.

Everything returns plain DataFrames so the tables go straight into a paper.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import classification_metrics, concordance_index


# --------------------------------------------------------------------------- #
def generalisation_gap(exp, tensors, model, splits: Dict[str, np.ndarray],
                       anchor_idx: np.ndarray) -> pd.DataFrame:
    """One row per split, plus an explicit train-minus-test gap row."""
    rows = {}
    for name, sel in splits.items():
        pred = exp.predict(model, tensors, sel, anchor_idx=anchor_idx)
        y = tensors.label.numpy()[sel]
        t = tensors.time.numpy()[sel]
        e = tensors.event.numpy()[sel]
        row = classification_metrics(y, pred["prob"])
        row["c_index"] = concordance_index(t, pred["risk"], e)
        row["n"] = len(sel)
        rows[name] = row

    df = pd.DataFrame(rows).T
    if "train" in df.index and "test" in df.index:
        gap = df.loc["train"] - df.loc["test"]
        gap["n"] = np.nan
        # Log loss is a cost: a positive gap there means test is *better*.
        df.loc["gap (train - test)"] = gap
    return df


def learning_curves(model) -> pd.DataFrame:
    """Per-evaluation train and validation metrics recorded during training."""
    hist = getattr(model, "_history", None)
    if not hist:
        return pd.DataFrame()
    return pd.DataFrame(hist).set_index("epoch")


# --------------------------------------------------------------------------- #
def permutation_test(exp, tensors, fit_idx: np.ndarray, val_idx: np.ndarray,
                     test_idx: np.ndarray, n_classes: int,
                     n_permutations: int = 5, seed: int = 0,
                     verbose: bool = True) -> pd.DataFrame:
    """Retrain on shuffled labels and shuffled survival; collect the null.

    Labels are permuted *within the training and validation folds only* — the
    test labels stay intact, so a model that has genuinely learned nothing must
    score at chance (balanced accuracy 1/K, C-index 0.5). Anything above that
    is leakage or an evaluation bug.
    """
    import copy

    import torch

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    original_label = tensors.label.clone()
    original_time = tensors.time.clone()
    original_event = tensors.event.clone()
    trainable = np.concatenate([fit_idx, val_idx])

    try:
        for p in range(n_permutations):
            perm = rng.permutation(trainable)
            tensors.label[trainable] = original_label[perm]
            tensors.time[trainable] = original_time[perm]
            tensors.event[trainable] = original_event[perm]

            torch.manual_seed(seed + p)
            model = exp.train_fold(tensors, fit_idx, val_idx, n_classes,
                                   verbose=False)
            # Restore the truth before scoring the untouched test split.
            tensors.label.copy_(original_label)
            tensors.time.copy_(original_time)
            tensors.event.copy_(original_event)

            pred = exp.predict(model, tensors, test_idx, anchor_idx=fit_idx)
            y = tensors.label.numpy()[test_idx]
            m = classification_metrics(y, pred["prob"])
            row = {
                "permutation": p + 1,
                "balanced_accuracy": m.get("balanced_accuracy", np.nan),
                "auroc": m.get("auroc", np.nan),
                "c_index": concordance_index(tensors.time.numpy()[test_idx],
                                             pred["risk"],
                                             tensors.event.numpy()[test_idx]),
            }
            rows.append(row)
            if verbose:
                print(f"  permutation {p + 1}/{n_permutations}: "
                      f"bacc={row['balanced_accuracy']:.3f} "
                      f"c-index={row['c_index']:.3f}")
    finally:
        tensors.label.copy_(original_label)
        tensors.time.copy_(original_time)
        tensors.event.copy_(original_event)

    return pd.DataFrame(rows).set_index("permutation")


def permutation_p_value(observed: float, null: Sequence[float]) -> float:
    """One-sided permutation p with the standard +1 correction.

    With B permutations the smallest attainable p is 1/(B+1); report B so the
    reader can see the resolution limit rather than a misleading `p = 0`.
    """
    null = np.asarray([v for v in null if np.isfinite(v)], dtype=float)
    if null.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((1 + (null >= observed).sum()) / (null.size + 1))


# --------------------------------------------------------------------------- #
def sample_size_curve(exp, tensors, train_idx: np.ndarray, val_idx: np.ndarray,
                      test_idx: np.ndarray, n_classes: int,
                      fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
                      seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Test performance as a function of training-set size.

    A curve that is still rising at 100 % means more patients would help; a
    curve that has flattened while the train-test gap is wide means the model,
    not the data, is the limit.
    """
    import torch

    rng = np.random.default_rng(seed)
    rows = []
    for frac in fractions:
        k = max(int(frac * len(train_idx)), 20)
        sub = rng.choice(train_idx, size=min(k, len(train_idx)), replace=False)
        torch.manual_seed(seed)
        model = exp.train_fold(tensors, sub, val_idx, n_classes, verbose=False)

        out = {}
        for name, sel in (("train", sub), ("test", test_idx)):
            pred = exp.predict(model, tensors, sel, anchor_idx=sub)
            m = classification_metrics(tensors.label.numpy()[sel], pred["prob"])
            out[f"{name}_balanced_accuracy"] = m.get("balanced_accuracy", np.nan)
            out[f"{name}_c_index"] = concordance_index(
                tensors.time.numpy()[sel], pred["risk"],
                tensors.event.numpy()[sel])
        out["n_train"] = len(sub)
        out["fraction"] = frac
        rows.append(out)
        if verbose:
            print(f"  n_train={len(sub):4d}: test bacc="
                  f"{out['test_balanced_accuracy']:.3f} "
                  f"test c-index={out['test_c_index']:.3f}")
    return pd.DataFrame(rows).set_index("n_train")


# --------------------------------------------------------------------------- #
def parameter_report(model, n_train: int) -> Dict[str, float]:
    """Capacity versus data, and where the parameters actually sit."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_part: Dict[str, int] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            by_part[name.split(".")[0]] = by_part.get(name.split(".")[0], 0) + p.numel()
    return {
        "trainable_parameters": total,
        "training_patients": n_train,
        "parameters_per_patient": total / max(n_train, 1),
        **{f"params[{k}]": v for k, v in sorted(by_part.items(),
                                                key=lambda kv: -kv[1])},
    }
