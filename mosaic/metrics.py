"""Evaluation metrics and statistical tests.

Reviewers at high-impact venues will ask for interval estimates and paired
tests, not point estimates, so everything here returns a bootstrap confidence
interval and the paired comparisons are implemented directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score,
                             f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


def _multiclass_brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Mean squared error between the predicted simplex and the one-hot truth.

    sklearn's brier_score_loss is binary only, so the multiclass form is
    written out: mean over patients of the squared distance between the
    predicted distribution and the indicator of the observed class.
    """
    onehot = np.zeros_like(prob)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(((prob - onehot) ** 2).sum(axis=1).mean())


def _one_vs_rest_counts(y_true: np.ndarray, pred: np.ndarray, cls: int):
    """(tn, fp, fn, tp) for one class against all the others."""
    pos_t, pos_p = y_true == cls, pred == cls
    tp = float((pos_t & pos_p).sum())
    fp = float((~pos_t & pos_p).sum())
    fn = float((pos_t & ~pos_p).sum())
    tn = float((~pos_t & ~pos_p).sum())
    return tn, fp, fn, tp


def _macro_specificity(y_true: np.ndarray, pred: np.ndarray, n_cls: int) -> float:
    """Unweighted mean of per-class true-negative rates.

    Clinically this is the complement of the false-alarm rate, which a
    reviewer will ask for alongside recall; it is averaged over the classes
    that actually occur so an absent class cannot inflate it.
    """
    vals = []
    for c in range(n_cls):
        if not (y_true == c).any():
            continue
        tn, fp, _, _ = _one_vs_rest_counts(y_true, pred, c)
        if tn + fp > 0:
            vals.append(tn / (tn + fp))
    return float(np.mean(vals)) if vals else float("nan")


def _macro_npv(y_true: np.ndarray, pred: np.ndarray, n_cls: int) -> float:
    """Unweighted mean of per-class negative predictive values."""
    vals = []
    for c in range(n_cls):
        if not (y_true == c).any():
            continue
        tn, _, fn, _ = _one_vs_rest_counts(y_true, pred, c)
        if tn + fn > 0:
            vals.append(tn / (tn + fn))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    mask = y_true >= 0
    y_true, prob = y_true[mask], prob[mask]
    if y_true.size == 0:
        return {}
    prob = np.clip(np.nan_to_num(prob, nan=1.0 / prob.shape[1]), 1e-12, 1.0)
    prob = prob / prob.sum(1, keepdims=True)
    pred = prob.argmax(1)
    n_cls = prob.shape[1]
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted",
                                      zero_division=0)),
        "precision_macro": float(precision_score(y_true, pred, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro",
                                           zero_division=0)),
        "specificity_macro": float(_macro_specificity(y_true, pred, n_cls)),
        "npv_macro": float(_macro_npv(y_true, pred, n_cls)),
        "kappa": float(cohen_kappa_score(y_true, pred)),
        "kappa_quadratic": (float(cohen_kappa_score(y_true, pred, weights="quadratic"))
                            if n_cls > 2 else float("nan")),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(set(y_true)) > 1 else 0.0,
        "log_loss": float(log_loss(y_true, prob, labels=list(range(n_cls)))),
        "brier": float(_multiclass_brier(y_true, prob)),
        "top2_accuracy": float(np.mean(
            [y in np.argsort(-p)[:2] for y, p in zip(y_true, prob)])),
    }
    present = np.unique(y_true)
    if len(present) > 1:
        try:
            if prob.shape[1] == 2:
                out["auroc"] = float(roc_auc_score(y_true, prob[:, 1]))
                out["auprc"] = float(average_precision_score(y_true, prob[:, 1]))
            else:
                sub = prob[:, present] / prob[:, present].sum(1, keepdims=True)
                remap = {c: i for i, c in enumerate(present)}
                y = np.asarray([remap[v] for v in y_true])
                out["auroc"] = float(roc_auc_score(y, sub, multi_class="ovr",
                                                   average="macro"))
                onehot = np.eye(len(present))[y]
                out["auprc"] = float(average_precision_score(onehot, sub,
                                                             average="macro"))
        except ValueError:
            pass
    return out


# --------------------------------------------------------------------------- #
# Survival
# --------------------------------------------------------------------------- #
def concordance_index(time: np.ndarray, risk: np.ndarray,
                      event: np.ndarray) -> float:
    """Harrell's C-index. Higher risk should mean shorter survival."""
    keep = np.isfinite(time) & np.isfinite(risk) & np.isfinite(event)
    time, risk, event = time[keep], risk[keep], event[keep]
    n = time.size
    if n < 2:
        return float("nan")

    order = np.argsort(time)
    time, risk, event = time[order], risk[order], event[order]
    conc = disc = tied = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        later = time > time[i]
        if not later.any():
            continue
        r_j = risk[later]
        conc += float((r_j < risk[i]).sum())
        disc += float((r_j > risk[i]).sum())
        tied += float((r_j == risk[i]).sum())
    denom = conc + disc + tied
    return float("nan") if denom == 0 else (conc + 0.5 * tied) / denom


def time_dependent_auc(time: np.ndarray, risk: np.ndarray, event: np.ndarray,
                       horizons: Sequence[float] = (12, 36, 60)
                       ) -> Dict[str, float]:
    """Cumulative/dynamic AUC at fixed horizons (months).

    Patients censored before the horizon carry no label and are dropped, which
    is the simple (uncorrected) estimator; report it alongside the C-index.
    """
    out: Dict[str, float] = {}
    for h in horizons:
        label = np.full(time.shape, -1)
        label[(time <= h) & (event == 1)] = 1
        label[time > h] = 0
        m = label >= 0
        if m.sum() > 10 and len(np.unique(label[m])) == 2:
            out[f"auc@{int(h)}m"] = float(roc_auc_score(label[m], risk[m]))
    return out


def logrank_test(time: np.ndarray, event: np.ndarray, group: np.ndarray
                 ) -> Tuple[float, float]:
    """Two-group log-rank statistic and p-value (chi-square, 1 df)."""
    from scipy import stats

    keep = np.isfinite(time) & np.isfinite(event)
    time, event, group = time[keep], event[keep], group[keep]
    times = np.unique(time[event == 1])
    o_minus_e = 0.0
    var = 0.0
    for t in times:
        at_risk = time >= t
        n = at_risk.sum()
        n1 = (at_risk & (group == 1)).sum()
        d = ((time == t) & (event == 1)).sum()
        d1 = ((time == t) & (event == 1) & (group == 1)).sum()
        if n < 2 or n1 == 0 or n1 == n:
            continue
        exp1 = d * n1 / n
        o_minus_e += d1 - exp1
        var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if var <= 0:
        return 0.0, 1.0
    chi2 = o_minus_e ** 2 / var
    return float(chi2), float(stats.chi2.sf(chi2, df=1))


def km_curve(time: np.ndarray, event: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimate; returns (times, survival probability)."""
    keep = np.isfinite(time) & np.isfinite(event)
    time, event = time[keep], event[keep]
    order = np.argsort(time)
    time, event = time[order], event[order]
    ts, s, surv = [0.0], 1.0, [1.0]
    for t in np.unique(time[event == 1]):
        n = (time >= t).sum()
        d = ((time == t) & (event == 1)).sum()
        if n > 0:
            s *= (1 - d / n)
            ts.append(float(t))
            surv.append(float(s))
    return np.asarray(ts), np.asarray(surv)


# --------------------------------------------------------------------------- #
# Uncertainty and inference
# --------------------------------------------------------------------------- #
def bootstrap_ci(fn, *arrays, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """(point estimate, lower, upper) from a stratified bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    point = fn(*arrays)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(*[a[idx] for a in arrays])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def paired_bootstrap_test(metric_fn, y, pred_a, pred_b, n_boot: int = 2000,
                          seed: int = 0) -> Dict[str, float]:
    """Two-sided paired bootstrap on the metric difference (a - b)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    obs = metric_fn(y, pred_a) - metric_fn(y, pred_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            diffs.append(metric_fn(y[idx], pred_a[idx]) - metric_fn(y[idx], pred_b[idx]))
        except Exception:
            continue
    diffs = np.asarray(diffs)
    if diffs.size == 0:
        return {"delta": float(obs), "p_value": float("nan")}
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"delta": float(obs), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(p, 1.0))}


def evaluate_predictions(y_true: np.ndarray,
                         prob: Optional[np.ndarray],
                         time: np.ndarray,
                         risk: Optional[np.ndarray],
                         event: np.ndarray,
                         seed: int = 0) -> Dict[str, float]:
    """Score one model's predictions with the same panel used for MOSAIC-GNN.

    Every baseline is scored through this one function so the comparison table
    cannot drift: a classifier with no risk head simply contributes no survival
    metrics (and vice versa), and the missing cells are genuinely missing
    rather than silently zero. Patients with no label, or with no usable
    follow-up time, are dropped per task rather than for the whole row.
    """
    out: Dict[str, float] = {}

    if prob is not None:
        prob = np.asarray(prob, dtype=float)
        lab_ok = np.asarray(y_true) >= 0
        if lab_ok.sum() > 1:
            out.update(classification_metrics(np.asarray(y_true)[lab_ok],
                                              prob[lab_ok]))
            try:
                out["ece"] = expected_calibration_error(
                    prob[lab_ok], np.asarray(y_true)[lab_ok])
            except Exception:
                pass

    if risk is not None:
        risk = np.asarray(risk, dtype=float).ravel()
        tt = np.asarray(time, dtype=float)
        ee = np.nan_to_num(np.asarray(event, dtype=float), nan=0.0)
        ok = np.isfinite(tt) & (tt > 0) & np.isfinite(risk)
        if ok.sum() > 1:
            tt, ee, rr = tt[ok], ee[ok], risk[ok]
            out["c_index"] = concordance_index(tt, rr, ee)
            out.update(time_dependent_auc(tt, rr, ee))
            median = np.median(rr)
            group = (rr > median).astype(int)
            chi2, p = logrank_test(tt, ee, group)
            out["logrank_chi2"], out["logrank_p"] = chi2, p

    return out


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up procedure.

    Comparing one model against twenty baselines on three metrics is sixty
    tests; at alpha = 0.05 roughly three of them come out "significant" by
    chance alone. Returns (q_values, rejected) in the input order, with
    q-values made monotone so a smaller raw p can never get a larger q.
    """
    p = np.asarray(list(p_values), dtype=float)
    out_q = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out_q, np.zeros(p.shape, dtype=bool)

    idx = np.flatnonzero(finite)
    sub = p[idx]
    order = np.argsort(sub)
    ranked = sub[order]
    m = ranked.size
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]      # enforce monotonicity
    q = np.clip(q, 0.0, 1.0)

    q_sub = np.empty_like(q)
    q_sub[order] = q
    out_q[idx] = q_sub
    return out_q, np.nan_to_num(out_q, nan=1.0) <= alpha


def expected_calibration_error(prob: np.ndarray, y: np.ndarray,
                               n_bins: int = 10) -> float:
    conf = prob.max(1)
    pred = prob.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def summarise(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))
            if arr.size > 1 else 0.0, "n": int(arr.size)}
