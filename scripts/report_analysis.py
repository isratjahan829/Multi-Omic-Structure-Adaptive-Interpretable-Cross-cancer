"""Derived analyses for the CSE400B report.

Everything in this module is computed from the CSV and JSON files that the
executed runs wrote. Nothing is typed in and nothing is estimated: each
function states the formula it applies, takes measured quantities as input,
and returns the result together with enough intermediate values that a
reader can repeat the arithmetic by hand.

The derivations are:

  * descriptive statistics of the assembled cohorts, including the Shannon
    entropy of the class distribution and the imbalance ratio;
  * reconstruction of each class's 2x2 contingency table from the exported
    support, sensitivity, specificity and precision, and from it the
    negative predictive value, Youden's J, the per-class Matthews
    correlation and a Wilson score interval for sensitivity;
  * an internal consistency audit that recomputes the pooled macro metrics
    from the per-class table and compares them with the pooled values the
    run reported;
  * absolute and relative generalisation gaps for every metric;
  * the rank of the proposed model against the comparator panel on every
    metric;
  * the standard error and the standardised effect of every paired
    bootstrap difference, recovered from the reported interval, together
    with an independent recomputation of the Benjamini-Hochberg q-values;
  * the ablation effects expressed in units of the measured noise floor;
  * z-scores of the observed metrics against the permutation null;
  * a least-squares fit of held-out performance on the logarithm of the
    training-set size;
  * leave-one-cancer-out transfer summarised by event count, with a
    Fisher z interval for the correlation and a Benjamini-Hochberg
    correction over the 32 log-rank tests;
  * the share of attribution carried by each omic block.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List, Optional, Sequence

Z95 = 1.959963984540054          # two-sided normal quantile at 95 %
NOISE_FLOOR = 0.01               # measured run-to-run spread, Section 6.8


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def read_csv(path: str) -> Optional[List[dict]]:
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows or None


def read_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def num(v) -> Optional[float]:
    """Parse a cell as a float; blank, '-' and NaN all become None."""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "-", "—", "nan", "NaN", "None"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return None if math.isnan(f) else f


def mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sd(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def quantile(xs: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolation quantile, the same convention as numpy's."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (pos - lo) * (xs[hi] - xs[lo])


# --------------------------------------------------------------------------- #
# Statistics computed from first principles
# --------------------------------------------------------------------------- #
def pearson(xs: Sequence[float], ys: Sequence[float]) -> dict:
    """Pearson correlation with the two-sided p-value of its t statistic.

        r = sum (x-x̄)(y-ȳ) / sqrt( sum (x-x̄)² · sum (y-ȳ)² )
        t = r · sqrt( (n-2) / (1-r²) ),  df = n-2
    """
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return {"n": n, "r": None, "p": None, "ci_low": None, "ci_high": None}
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    if sxx <= 0 or syy <= 0:
        return {"n": n, "r": None, "p": None, "ci_low": None, "ci_high": None}
    r = sxy / math.sqrt(sxx * syy)
    r = max(-0.999999999, min(0.999999999, r))
    t = r * math.sqrt((n - 2) / (1 - r * r))
    p = _t_sf(abs(t), n - 2) * 2
    # Fisher z interval: z = artanh r, SE = 1/sqrt(n-3)
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3) if n > 3 else None
    lo = math.tanh(z - Z95 * se) if se else None
    hi = math.tanh(z + Z95 * se) if se else None
    return {"n": n, "r": r, "p": p, "t": t, "z": z, "se_z": se,
            "ci_low": lo, "ci_high": hi}


def _t_sf(t: float, df: int) -> float:
    """Upper tail of Student's t, by the incomplete beta function."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b), by the continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 300,
            eps: float = 3e-16) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def wilson(k: float, n: float, z: float = Z95) -> Dict[str, Optional[float]]:
    """Wilson score interval for a proportion.

        centre = (k + z²/2) / (n + z²)
        half   = z/(n + z²) · sqrt( k(n-k)/n + z²/4 )
    """
    if not n or n <= 0:
        return {"low": None, "high": None}
    k = max(0.0, min(float(n), float(k)))
    den = n + z * z
    centre = (k + z * z / 2.0) / den
    half = (z / den) * math.sqrt(k * (n - k) / n + z * z / 4.0)
    return {"low": max(0.0, centre - half), "high": min(1.0, centre + half)}


def benjamini_hochberg(pvalues: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Step-up BH q-values, in the input order.

        q_(i) = min_{j >= i} ( p_(j) · m / j )
    """
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    out: List[Optional[float]] = [None] * len(pvalues)
    if m == 0:
        return out
    order = sorted(idx, key=lambda i: pvalues[i])
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, min(1.0, pvalues[i] * m / rank))
        out[i] = running
    return out


def linfit(xs: Sequence[float], ys: Sequence[float]) -> dict:
    """Ordinary least squares y = a + b·x with the coefficient of
    determination R² = 1 - SS_res / SS_tot."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None]
    n = len(pairs)
    if n < 2:
        return {"n": n, "slope": None, "intercept": None, "r2": None}
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    if sxx <= 0:
        return {"n": n, "slope": None, "intercept": None, "r2": None}
    b = sum((p[0] - mx) * (p[1] - my) for p in pairs) / sxx
    a = my - b * mx
    ss_tot = sum((p[1] - my) ** 2 for p in pairs)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in pairs)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return {"n": n, "slope": b, "intercept": a, "r2": r2,
            "ss_res": ss_res, "ss_tot": ss_tot}


# --------------------------------------------------------------------------- #
# 1. Cohort descriptive statistics
# --------------------------------------------------------------------------- #
def cohort_descriptives(runs: str) -> Optional[dict]:
    rows = read_csv(os.path.join(runs, "table1_cohorts.csv"))
    if not rows:
        return None
    key = "cohort" if "cohort" in rows[0] else list(rows[0])[0]
    pts = [num(r["patients"]) for r in rows]
    ev = [num(r["events"]) for r in rows]
    er = [num(r["event_rate"]) for r in rows]
    fu = [num(r.get("median_OS_months")) for r in rows]
    lab = [num(r.get("labelled")) for r in rows]
    total = sum(p for p in pts if p is not None)
    # Shannon entropy of the class distribution, in nats:
    #     H = -sum p_k ln p_k,  normalised by ln K (its maximum)
    probs = [p / total for p in pts if p]
    H = -sum(p * math.log(p) for p in probs if p > 0)
    K = len(probs)
    gini = 1 - sum(p * p for p in probs)
    big = max((p, r[key]) for p, r in zip(pts, rows) if p)
    small = min((p, r[key]) for p, r in zip(pts, rows) if p)
    hi_er = max((e, r[key]) for e, r in zip(er, rows) if e is not None)
    lo_er = min((e, r[key]) for e, r in zip(er, rows) if e is not None)
    return {
        "n_cohorts": len(rows),
        "patients_total": total,
        "events_total": sum(e for e in ev if e is not None),
        "labelled_total": sum(x for x in lab if x is not None),
        "pooled_event_rate": (sum(e for e in ev if e is not None) / total
                              if total else None),
        "patients_min": small[0], "patients_min_cohort": small[1],
        "patients_max": big[0], "patients_max_cohort": big[1],
        "patients_median": quantile(pts, 0.5),
        "patients_q1": quantile(pts, 0.25),
        "patients_q3": quantile(pts, 0.75),
        "imbalance_ratio": big[0] / small[0] if small[0] else None,
        "entropy_nats": H,
        "entropy_max": math.log(K) if K else None,
        "entropy_ratio": H / math.log(K) if K > 1 else None,
        "gini": gini,
        "event_rate_min": lo_er[0], "event_rate_min_cohort": lo_er[1],
        "event_rate_max": hi_er[0], "event_rate_max_cohort": hi_er[1],
        "event_rate_median": quantile(er, 0.5),
        "followup_median": quantile(fu, 0.5),
        "followup_min": min(f for f in fu if f is not None) if any(
            f is not None for f in fu) else None,
        "followup_max": max(f for f in fu if f is not None) if any(
            f is not None for f in fu) else None,
        "n_below_100": sum(1 for p in pts if p is not None and p < 100),
        "n_events_below_20": sum(1 for e in ev if e is not None and e < 20),
        "n_events_below_60": sum(1 for e in ev if e is not None and e < 60),
    }


# --------------------------------------------------------------------------- #
# 2. Per-class contingency tables and derived metrics
# --------------------------------------------------------------------------- #
def per_class_derived(runs: str, n_total: Optional[float] = None) -> Optional[dict]:
    rows = read_csv(os.path.join(runs, "table2b_per_class.csv"))
    if not rows:
        return None
    first = list(rows[0])[0]
    support = [num(r["support"]) for r in rows]
    n_sum = sum(s for s in support if s is not None)
    N = n_total or n_sum
    out = []
    for r in rows:
        n_c = num(r["support"]) or 0.0
        sens = num(r["recall"])
        spec = num(r["specificity"])
        prec = num(r["precision"])
        f1 = num(r["f1"])
        mpp = num(r.get("mean_predicted_prob"))
        neg = N - n_c
        # Reconstruct the one-versus-rest 2x2 table from the exported rates:
        #   TP = sens·n_c,  FN = n_c - TP,  TN = spec·(N - n_c),  FP = (N-n_c) - TN
        tp = sens * n_c if sens is not None else None
        fn = n_c - tp if tp is not None else None
        tn = spec * neg if spec is not None else None
        fp = neg - tn if tn is not None else None
        npv = (tn / (tn + fn)) if (tn is not None and fn is not None
                                   and (tn + fn) > 0) else None
        youden = (sens + spec - 1) if (sens is not None
                                       and spec is not None) else None
        bacc_c = ((sens + spec) / 2) if (sens is not None
                                         and spec is not None) else None
        mcc_c = None
        if None not in (tp, fp, fn, tn):
            den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc_c = ((tp * tn - fp * fn) / den) if den > 0 else 0.0
        ci = wilson(tp if tp is not None else 0.0, n_c) if n_c else {
            "low": None, "high": None}
        out.append({
            "class": r[first], "support": n_c, "sensitivity": sens,
            "specificity": spec, "precision": prec, "f1": f1,
            "npv": npv, "youden": youden, "bacc": bacc_c, "mcc": mcc_c,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sens_ci_low": ci["low"], "sens_ci_high": ci["high"],
            "mean_predicted_prob": mpp,
        })
    return {"rows": out, "n_classes": len(out), "support_sum": n_sum,
            "n_used": N,
            "macro": {
                "sensitivity": mean([r["sensitivity"] for r in out]),
                "specificity": mean([r["specificity"] for r in out]),
                "precision": mean([r["precision"] for r in out]),
                "f1": mean([r["f1"] for r in out]),
                "npv": mean([r["npv"] for r in out]),
                "youden": mean([r["youden"] for r in out]),
                "mcc": mean([r["mcc"] for r in out]),
            },
            "weighted_f1": (sum((r["f1"] or 0) * (r["support"] or 0)
                                for r in out) / n_sum) if n_sum else None,
            "n_zero_recall": sum(1 for r in out
                                 if r["sensitivity"] == 0),
            }


def class_names(runs: str) -> Optional[dict]:
    """Recover which cohort each class index is, and verify the recovery.

    The per-class export is indexed by the label encoder's integer class,
    which is not readable. The encoder sorts its labels, so index i should be
    the i-th cohort alphabetically. That is a guess until it is checked, so it
    is checked: each class's support must equal that cohort's count in the
    per-cohort export. The function returns the mapping together with how
    many classes confirmed it, and the caller reports both rather than
    assuming the mapping is right.
    """
    t1 = read_csv(os.path.join(runs, "table1_cohorts.csv"))
    pc = per_class_derived(runs)
    t2c = read_csv(os.path.join(runs, "table2c_per_cohort.csv"))
    if not (t1 and pc):
        return None
    k1 = "cohort" if "cohort" in t1[0] else list(t1[0])[0]
    names = sorted(r[k1] for r in t1)
    if len(names) != pc["n_classes"]:
        return {"mapping": None, "confirmed": 0, "contradicted": 0,
                "unchecked": pc["n_classes"],
                "reason": f"{len(names)} cohorts but {pc['n_classes']} "
                          "classes, so the orders cannot correspond"}
    counts = {}
    if t2c:
        key = list(t2c[0])[0]
        counts = {r[key]: num(r.get("n")) for r in t2c}
    mapping, confirmed, contradicted, unchecked = {}, 0, 0, 0
    for i, row in enumerate(pc["rows"]):
        name = names[i]
        mapping[str(row["class"])] = name
        n = counts.get(name)
        if n is None:
            unchecked += 1
        elif abs((row["support"] or 0) - n) < 0.5:
            confirmed += 1
        else:
            contradicted += 1
    return {"mapping": (mapping if contradicted == 0 else None),
            "confirmed": confirmed, "contradicted": contradicted,
            "unchecked": unchecked, "n_classes": pc["n_classes"],
            "reason": None if contradicted == 0 else
                      f"{contradicted} classes contradict the alphabetical "
                      "order, so the mapping is not reported"}


def single_fold_panel(runs: str) -> Optional[Dict[str, float]]:
    rows = read_csv(os.path.join(runs, "table2_single_fold.csv"))
    if not rows:
        return None
    first = list(rows[0])[0]
    return {r[first]: num(r["value"]) for r in rows}


def consistency_audit(runs: str) -> Optional[dict]:
    """Recompute the pooled macro metrics from the per-class table and
    compare them with the pooled values the run itself reported."""
    panel = single_fold_panel(runs)
    pc = per_class_derived(runs)
    if not panel or not pc:
        return None
    checks = [
        ("balanced_accuracy", "mean of the per-class sensitivities",
         pc["macro"]["sensitivity"], panel.get("balanced_accuracy")),
        ("recall_macro", "mean of the per-class sensitivities",
         pc["macro"]["sensitivity"], panel.get("recall_macro")),
        ("precision_macro", "mean of the per-class precisions",
         pc["macro"]["precision"], panel.get("precision_macro")),
        ("specificity_macro", "mean of the per-class specificities",
         pc["macro"]["specificity"], panel.get("specificity_macro")),
        ("f1_macro", "mean of the per-class F1 scores",
         pc["macro"]["f1"], panel.get("f1_macro")),
        ("f1_weighted", "support-weighted mean of the per-class F1 scores",
         pc["weighted_f1"], panel.get("f1_weighted")),
        ("npv_macro", "mean of the per-class NPVs reconstructed from the "
                      "2x2 tables", pc["macro"]["npv"],
         panel.get("npv_macro")),
    ]
    rows = []
    worst = 0.0
    for name, how, recomputed, reported in checks:
        diff = (abs(recomputed - reported)
                if (recomputed is not None and reported is not None)
                else None)
        if diff is not None:
            worst = max(worst, diff)
        rows.append({"metric": name, "how": how, "recomputed": recomputed,
                     "reported": reported, "abs_diff": diff})
    return {"rows": rows, "max_abs_diff": worst,
            "support_sum": pc["support_sum"], "n_classes": pc["n_classes"]}


# --------------------------------------------------------------------------- #
# 3. Generalisation gap, absolute and relative
# --------------------------------------------------------------------------- #
GAP_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
               "precision_macro", "recall_macro", "specificity_macro",
               "npv_macro", "kappa", "kappa_quadratic", "mcc", "auroc",
               "auprc", "top2_accuracy", "c_index", "log_loss", "brier"]


def gap_derived(runs: str) -> Optional[dict]:
    rows = read_csv(os.path.join(runs, "tableS1_generalisation_gap.csv"))
    if not rows:
        return None
    first = list(rows[0])[0]
    by = {r[first]: r for r in rows}
    if "train" not in by or "test" not in by:
        return None
    out = []
    for m in GAP_METRICS:
        tr, va, te = (num(by["train"].get(m)),
                      num(by.get("validation", {}).get(m)),
                      num(by["test"].get(m)))
        if tr is None or te is None:
            continue
        absolute = tr - te
        relative = (absolute / tr * 100.0) if tr not in (None, 0) else None
        out.append({"metric": m, "train": tr, "validation": va, "test": te,
                    "gap": absolute, "gap_pct": relative})
    disc = [r for r in out if r["metric"] not in ("log_loss", "brier")]
    return {"rows": out,
            "n": num(by["train"].get("n")),
            "n_test": num(by["test"].get("n")),
            "mean_abs_gap": mean([abs(r["gap"]) for r in disc]),
            "max_gap_metric": max(disc, key=lambda r: r["gap"])["metric"]
            if disc else None,
            "max_gap": max(r["gap"] for r in disc) if disc else None}


# --------------------------------------------------------------------------- #
# 4. Rank of the proposed model against the comparator panel
# --------------------------------------------------------------------------- #
HIGHER_IS_BETTER = {
    "accuracy": True, "balanced_accuracy": True, "f1_macro": True,
    "f1_weighted": True, "precision_macro": True, "recall_macro": True,
    "specificity_macro": True, "npv_macro": True, "auroc": True,
    "auprc": True, "mcc": True, "kappa": True, "kappa_quadratic": True,
    "top2_accuracy": True, "c_index": True, "auc@12m": True,
    "auc@36m": True, "auc@60m": True, "logrank_chi2": True,
    "log_loss": False, "brier": False, "ece": False,
}


def rank_table(runs: str, mosaic: Dict[str, float],
               metrics: Sequence[str]) -> Optional[dict]:
    rows = read_csv(os.path.join(runs, "table3_comparison_full.csv"))
    if not rows:
        return None
    first = list(rows[0])[0]
    out = []
    for m in metrics:
        vals = [(r[first], num(r.get(m))) for r in rows]
        vals = [(k, v) for k, v in vals if v is not None]
        own = mosaic.get(m)
        if own is None or not vals:
            continue
        hib = HIGHER_IS_BETTER.get(m, True)
        allv = vals + [("MOSAIC-GNN", own)]
        allv.sort(key=lambda kv: kv[1], reverse=hib)
        rank = 1 + [k for k, _ in allv].index("MOSAIC-GNN")
        best_name, best_val = allv[0]
        if best_name == "MOSAIC-GNN" and len(allv) > 1:
            best_name, best_val = allv[1]
        out.append({"metric": m, "mosaic": own, "rank": rank,
                    "of": len(allv), "best_model": best_name,
                    "best_value": best_val,
                    "delta_to_best": own - best_val,
                    "direction": "higher is better" if hib
                                 else "lower is better"})
    wins = sum(1 for r in out if r["rank"] == 1)
    return {"rows": out, "n_metrics": len(out), "n_rank_1": wins,
            "median_rank": quantile([r["rank"] for r in out], 0.5),
            "best_rank": min((r["rank"] for r in out), default=None),
            "worst_rank": max((r["rank"] for r in out), default=None)}


# --------------------------------------------------------------------------- #
# 5. Paired bootstrap: standard error, standardised effect, BH recomputed
# --------------------------------------------------------------------------- #
def paired_derived(runs: str) -> Optional[dict]:
    rows = read_csv(os.path.join(runs, "table4_paired_tests.csv"))
    if not rows:
        return None
    out = []
    for r in rows:
        rec = {"model": r.get("model")}
        for key in ("bacc", "cindex"):
            d = num(r.get(f"{key}_delta"))
            lo = num(r.get(f"{key}_ci_low"))
            hi = num(r.get(f"{key}_ci_high"))
            p = num(r.get(f"{key}_p"))
            q = num(r.get(f"{key}_q"))
            # The interval is delta ± z·SE, so SE = (hi - lo) / (2z) and the
            # standardised effect is z_eff = delta / SE.
            se = ((hi - lo) / (2 * Z95)) if (lo is not None
                                             and hi is not None) else None
            zeff = (d / se) if (d is not None and se not in (None, 0)) else None
            rec.update({f"{key}_delta": d, f"{key}_ci_low": lo,
                        f"{key}_ci_high": hi, f"{key}_p": p, f"{key}_q": q,
                        f"{key}_se": se, f"{key}_z": zeff,
                        f"{key}_excludes_zero": (
                            None if (lo is None or hi is None)
                            else not (lo <= 0 <= hi)),
                        f"{key}_sig": str(r.get(f"{key}_sig", "")).strip()
                        in ("True", "true", "1")})
        out.append(rec)
    # Independent recomputation of the Benjamini-Hochberg q-values. The run
    # corrected within each metric, so the family is the set of tests on one
    # metric; recomputing it that way reproduces the stored q-values, which
    # is the check reported in the text.
    worst, sizes = 0.0, {}
    for key in ("bacc", "cindex"):
        idx = [i for i, rec in enumerate(out)
               if rec.get(f"{key}_p") is not None]
        qs = benjamini_hochberg([out[i][f"{key}_p"] for i in idx])
        sizes[key] = len(idx)
        for i, q in zip(idx, qs):
            out[i][f"{key}_q_recomputed"] = q
            stored = out[i].get(f"{key}_q")
            if stored is not None and q is not None:
                worst = max(worst, abs(stored - q))

    # Sensitivity check: correcting over the pooled family of both metrics is
    # stricter, because the family is larger. Does any verdict change?
    pooled_keys, pooled_p = [], []
    for i, rec in enumerate(out):
        for key in ("bacc", "cindex"):
            if rec.get(f"{key}_p") is not None:
                pooled_keys.append((i, key))
                pooled_p.append(rec[f"{key}_p"])
    pooled_q = benjamini_hochberg(pooled_p)
    flips, pooled_worst = [], 0.0
    for (i, key), q in zip(pooled_keys, pooled_q):
        out[i][f"{key}_q_pooled"] = q
        stored = out[i].get(f"{key}_q")
        if stored is not None and q is not None:
            pooled_worst = max(pooled_worst, abs(stored - q))
            if (stored < 0.05) != (q < 0.05):
                flips.append((out[i]["model"], key, stored, q))
    return {"rows": out, "family_sizes": sizes,
            "q_max_abs_diff": worst,
            "pooled_family_size": len(pooled_p),
            "pooled_q_max_abs_diff": pooled_worst,
            "pooled_verdict_flips": flips,
            "n_models": len(out)}


# --------------------------------------------------------------------------- #
# 6. Ablation in units of the measured noise floor
# --------------------------------------------------------------------------- #
def ablation_derived(runs: str, floor: float = NOISE_FLOOR) -> Optional[dict]:
    """Ablation effects, each expressed in units of the measured noise floor.

    The fold spread of every arm comes from cohort_ablation.json, which holds
    the mean, the standard deviation and the fold count that the ablation run
    itself wrote; nothing here is recomputed from the model checkpoints.
    """
    tab = read_csv(os.path.join(runs, "table5_ablation.csv"))
    summ = read_json(os.path.join(runs, "cohort_ablation.json")) or {}
    if not tab:
        return None
    first = list(tab[0])[0]
    by = {r[first]: r for r in tab}
    full = num(by.get("full", {}).get("balanced_accuracy"))

    def spread(arm: str, metric: str) -> dict:
        s = (summ.get(arm) or {}).get(metric)
        return s if isinstance(s, dict) else {}

    arms = []
    for name, r in by.items():
        if name == "full":
            continue
        b = num(r.get("balanced_accuracy"))
        rec = {"arm": name, "balanced_accuracy": b,
               "auroc": num(r.get("auroc")),
               "c_index": num(r.get("c_index")),
               "delta": (b - full) if (b is not None and full is not None)
                        else None}
        for m in ("balanced_accuracy", "auroc", "c_index"):
            s = spread(name, m)
            rec[f"{m}_sd"] = s.get("std")
            rec[f"{m}_n"] = s.get("n")
        if rec["delta"] is not None:
            rec["floors"] = abs(rec["delta"]) / floor
            rec["outside"] = abs(rec["delta"]) > floor
        arms.append(rec)
    arms.sort(key=lambda a: (a["delta"] is None, a["delta"]))
    fs = spread("full", "balanced_accuracy")
    secs = (summ.get("full") or {}).get("seconds")
    return {"full": full, "arms": arms, "floor": floor,
            "full_sd": fs.get("std"), "full_folds": fs.get("n"),
            "n_outside": sum(1 for a in arms if a.get("outside")),
            "n_inside": sum(1 for a in arms if a.get("outside") is False),
            "seconds_mean": (secs or {}).get("mean")
            if isinstance(secs, dict) else secs}


# --------------------------------------------------------------------------- #
# 7. Permutation null: z-scores and the attainable p-value
# --------------------------------------------------------------------------- #
def permutation_derived(runs: str) -> Optional[dict]:
    tab = read_csv(os.path.join(runs, "tableS2_permutation_test.csv"))
    if not tab:
        return None
    first = list(tab[0])[0]
    rows = []
    B = None
    for r in tab:
        obs, mu, s = (num(r.get("observed")), num(r.get("null_mean")),
                      num(r.get("null_sd")))
        B = num(r.get("n_permutations")) or B
        rows.append({
            "metric": r[first], "observed": obs, "null_mean": mu,
            "null_sd": s, "null_max": num(r.get("null_max")),
            "p_value": num(r.get("p_value")),
            # z = (observed - null mean) / null SD
            "z": ((obs - mu) / s) if (obs is not None and mu is not None
                                      and s not in (None, 0)) else None,
            "excess": (obs - mu) if (obs is not None and mu is not None)
                      else None,
        })
    draws = read_csv(os.path.join(runs, "tableS2_permutation_null.csv"))
    per = []
    if draws:
        k = list(draws[0])[0]
        for r in draws:
            per.append({"permutation": r[k],
                        "balanced_accuracy": num(r.get("balanced_accuracy")),
                        "auroc": num(r.get("auroc")),
                        "c_index": num(r.get("c_index"))})
    return {"rows": rows, "draws": per, "B": B,
            "p_floor": (1.0 / (B + 1)) if B else None,
            "chance_bacc": None}


# --------------------------------------------------------------------------- #
# 8. Sample-size curve: least squares on the logarithm of n
# --------------------------------------------------------------------------- #
def samplesize_derived(runs: str) -> Optional[dict]:
    rows = read_csv(os.path.join(runs, "tableS3_sample_size.csv"))
    if not rows:
        return None
    n = [num(r["n_train"]) for r in rows]
    logn = [math.log2(v) for v in n if v]
    out = {"rows": [], "n_points": len(rows)}
    for col in ("test_balanced_accuracy", "test_c_index",
                "train_balanced_accuracy", "train_c_index"):
        ys = [num(r[col]) for r in rows]
        fit = linfit(logn, ys)
        out[col] = {"first": ys[0], "last": ys[-1],
                    "gain": (ys[-1] - ys[0]) if None not in (ys[0], ys[-1])
                            else None,
                    "slope_per_doubling": fit["slope"], "r2": fit["r2"],
                    "intercept": fit["intercept"]}
    for r, l in zip(rows, logn):
        out["rows"].append({
            "n_train": num(r["n_train"]), "log2n": l,
            "fraction": num(r["fraction"]),
            "train_balanced_accuracy": num(r["train_balanced_accuracy"]),
            "train_c_index": num(r["train_c_index"]),
            "test_balanced_accuracy": num(r["test_balanced_accuracy"]),
            "test_c_index": num(r["test_c_index"]),
            "gap_bacc": (num(r["train_balanced_accuracy"])
                         - num(r["test_balanced_accuracy"])),
            "gap_cindex": (num(r["train_c_index"])
                           - num(r["test_c_index"])),
        })
    return out


# --------------------------------------------------------------------------- #
# 9. Leave-one-cancer-out transfer
# --------------------------------------------------------------------------- #
def loco_derived(runs: str) -> Optional[dict]:
    tab = read_csv(os.path.join(runs, "table6_loco.csv"))
    t1 = read_csv(os.path.join(runs, "table1_cohorts.csv"))
    if not tab or not t1:
        return None
    k1 = "cohort" if "cohort" in t1[0] else list(t1[0])[0]
    info = {r[k1]: r for r in t1}
    first = list(tab[0])[0]
    rows = []
    for r in tab:
        name = r[first]
        i = info.get(name, {})
        rows.append({"cohort": name, "c_index": num(r.get("c_index")),
                     "logrank_p": num(r.get("logrank_p")),
                     "patients": num(i.get("patients")),
                     "events": num(i.get("events")),
                     "event_rate": num(i.get("event_rate"))})
    qs = benjamini_hochberg([r["logrank_p"] for r in rows])
    for r, q in zip(rows, qs):
        r["logrank_q"] = q

    # The classification column of this table is not reported as a result,
    # and this is the evidence for why. In a leave-one-cancer-out fold the
    # held-out class never appears in training, so a softmax head should be
    # unable to assign it at all. The exported single-class recall says
    # otherwise, and the size of the discrepancy is quantified here rather
    # than left as a remark: implied correct = recall * patients, because in
    # a single-class fold balanced accuracy reduces to that class's recall.
    cls = []
    for r, src in zip(rows, tab):
        b = num(src.get("balanced_accuracy"))
        if b is None:
            continue
        cls.append({"cohort": r["cohort"], "bacc": b,
                    "patients": r["patients"],
                    "implied_correct": (b * r["patients"]
                                        if r["patients"] else None)})
    cls.sort(key=lambda c: -c["bacc"])
    anomaly = {
        "n": len(cls), "rows": cls,
        "median": quantile([c["bacc"] for c in cls], 0.5),
        "n_above_90": sum(1 for c in cls if c["bacc"] > 0.9),
        "n_exactly_1": sum(1 for c in cls if c["bacc"] >= 0.999999),
        "n_above_half": sum(1 for c in cls if c["bacc"] > 0.5),
        "worst": min((c["bacc"] for c in cls), default=None),
    }
    ci = [r["c_index"] for r in rows]
    groups = {}
    for label, lo in (("all", 0), ("events_ge_20", 20), ("events_ge_60", 60)):
        sub = [r for r in rows
               if r["events"] is not None and r["events"] >= lo]
        groups[label] = {
            "n": len(sub),
            "median": quantile([r["c_index"] for r in sub], 0.5),
            "mean": mean([r["c_index"] for r in sub]),
            "sd": sd([r["c_index"] for r in sub]),
            "n_above_half": sum(1 for r in sub
                                if r["c_index"] is not None
                                and r["c_index"] > 0.5),
            "n_p_lt_05": sum(1 for r in sub if r["logrank_p"] is not None
                             and r["logrank_p"] < 0.05),
            "n_q_lt_05": sum(1 for r in sub if r["logrank_q"] is not None
                             and r["logrank_q"] < 0.05),
            "corr_event_rate": pearson([r["event_rate"] for r in sub],
                                       [r["c_index"] for r in sub]),
            "corr_patients": pearson([r["patients"] for r in sub],
                                     [r["c_index"] for r in sub]),
        }
    return {"rows": sorted(rows, key=lambda r: (r["c_index"] is None,
                                                r["c_index"])),
            "n_cohorts": len(rows),
            "median": quantile(ci, 0.5), "q1": quantile(ci, 0.25),
            "q3": quantile(ci, 0.75),
            "min": min(c for c in ci if c is not None),
            "max": max(c for c in ci if c is not None),
            "mean": mean(ci), "sd": sd(ci),
            "n_above_half": sum(1 for c in ci if c is not None and c > 0.5),
            "n_p_lt_05": sum(1 for r in rows if r["logrank_p"] is not None
                             and r["logrank_p"] < 0.05),
            "n_q_lt_05": sum(1 for r in rows if r["logrank_q"] is not None
                             and r["logrank_q"] < 0.05),
            "classification_anomaly": anomaly,
            "groups": groups}


# --------------------------------------------------------------------------- #
# 10. Attribution by omic block
# --------------------------------------------------------------------------- #
def interpretation_derived(runs: str) -> Optional[dict]:
    rows = read_csv(os.path.join(runs, "table7_gene_importance.csv"))
    if not rows:
        return None
    blocks = {"ig_mut": 0.0, "ig_cna": 0.0, "ig_exp": 0.0}
    for r in rows:
        for b in blocks:
            v = num(r.get(b))
            if v is not None:
                blocks[b] += abs(v)
    tot = sum(blocks.values())
    gates = [num(r.get("gate_mean")) for r in rows]
    mods = read_csv(os.path.join(runs, "table8_module_importance.csv"))
    sizes = [num(r.get("size")) for r in (mods or [])]
    att = [num(r.get("attention")) for r in (mods or [])]
    return {
        "n_genes_listed": len(rows),
        "blocks": {b: {"abs_sum": v,
                       "share": (v / tot if tot else None)}
                   for b, v in blocks.items()},
        "block_total": tot,
        "gate_mean": mean(gates), "gate_min": min(g for g in gates
                                                  if g is not None),
        "gate_max": max(g for g in gates if g is not None),
        "top_genes": [r[list(rows[0])[0]] for r in rows[:10]],
        "n_modules_listed": len(mods or []),
        "module_size_mean": mean(sizes),
        "module_attention_top": max(att) if att else None,
        "module_attention_sum_top": sum(a for a in att if a is not None),
    }


# --------------------------------------------------------------------------- #
# 11. Per-cohort behaviour on the held-out fold
# --------------------------------------------------------------------------- #
def per_cohort_derived(runs: str) -> Optional[dict]:
    rows = read_csv(os.path.join(runs, "table2c_per_cohort.csv"))
    t1 = read_csv(os.path.join(runs, "table1_cohorts.csv"))
    if not rows:
        return None
    first = list(rows[0])[0]
    k1 = ("cohort" if t1 and "cohort" in t1[0]
          else (list(t1[0])[0] if t1 else None))
    info = {r[k1]: r for r in (t1 or [])} if k1 else {}
    out = []
    for r in rows:
        name = r[first]
        out.append({
            "cohort": name, "n": num(r.get("n")),
            "accuracy": num(r.get("accuracy")),
            "balanced_accuracy": num(r.get("balanced_accuracy")),
            "f1_macro": num(r.get("f1_macro")),
            "f1_weighted": num(r.get("f1_weighted")),
            "mcc": num(r.get("mcc")), "kappa": num(r.get("kappa")),
            "top2_accuracy": num(r.get("top2_accuracy")),
            "log_loss": num(r.get("log_loss")),
            "brier": num(r.get("brier")),
            "c_index": num(r.get("c_index")),
            "events": num(info.get(name, {}).get("events")),
            "event_rate": num(info.get(name, {}).get("event_rate")),
        })
    acc = [r["accuracy"] for r in out]
    ci = [r["c_index"] for r in out]
    return {
        "rows": sorted(out, key=lambda r: (r["accuracy"] is None,
                                           -(r["accuracy"] or 0))),
        "n_cohorts": len(out),
        "accuracy": {"mean": mean(acc), "sd": sd(acc),
                     "median": quantile(acc, 0.5),
                     "min": min(a for a in acc if a is not None),
                     "max": max(a for a in acc if a is not None)},
        "c_index": {"mean": mean(ci), "sd": sd(ci),
                    "median": quantile(ci, 0.5),
                    "min": min(c for c in ci if c is not None),
                    "max": max(c for c in ci if c is not None)},
        "n_acc_above_90": sum(1 for a in acc if a is not None and a >= 0.9),
        "n_cindex_above_half": sum(1 for c in ci if c is not None and c > 0.5),
        "corr_n_accuracy": pearson([r["n"] for r in out], acc),
        "corr_n_cindex": pearson([r["n"] for r in out], ci),
        "corr_eventrate_cindex": pearson([r["event_rate"] for r in out], ci),
    }


# --------------------------------------------------------------------------- #
# 12. Runtime and model size
# --------------------------------------------------------------------------- #
def cost_derived(runs: str) -> Optional[dict]:
    pr = None
    for name in ("cohort_parameter_report.json",
                 "stage_binary_parameter_report.json"):
        pr = pr or read_json(os.path.join(runs, name))
    summ = read_json(os.path.join(runs, "cohort_ablation.json")) or {}
    s = (summ.get("full") or {}).get("seconds")
    secs = ([s["mean"]] if isinstance(s, dict) and s.get("mean") is not None
            else ([s] if isinstance(s, (int, float)) else []))
    blocks = {}
    for k, v in (pr or {}).items():
        if k.startswith("params[") and k.endswith("]"):
            blocks[k[7:-1]] = v
    tot = (pr or {}).get("trainable_parameters")
    return {
        "trainable_parameters": tot,
        "training_patients": (pr or {}).get("training_patients"),
        "parameters_per_patient": (pr or {}).get("parameters_per_patient"),
        "blocks": sorted(((k, v, (v / tot if tot else None))
                          for k, v in blocks.items()),
                         key=lambda kv: -kv[1]),
        "ablation_seconds_per_fold": secs,
        "ablation_seconds_mean": mean(secs),
    }


# --------------------------------------------------------------------------- #
# Everything, for one run directory
# --------------------------------------------------------------------------- #
def analyse(runs: str) -> dict:
    panel = single_fold_panel(runs) or {}
    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
               "precision_macro", "recall_macro", "specificity_macro",
               "npv_macro", "auroc", "auprc", "mcc", "kappa",
               "kappa_quadratic", "top2_accuracy", "log_loss", "brier",
               "ece", "c_index", "auc@12m", "auc@36m", "auc@60m",
               "logrank_chi2"]
    return {
        "runs": runs,
        "panel": panel,
        "descriptives": cohort_descriptives(runs),
        "per_class": per_class_derived(runs),
        "class_names": class_names(runs),
        "consistency": consistency_audit(runs),
        "gap": gap_derived(runs),
        "ranks": rank_table(runs, panel, metrics),
        "paired": paired_derived(runs),
        "ablation": ablation_derived(runs),
        "permutation": permutation_derived(runs),
        "samplesize": samplesize_derived(runs),
        "loco": loco_derived(runs),
        "interpretation": interpretation_derived(runs),
        "per_cohort": per_cohort_derived(runs),
        "cost": cost_derived(runs),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_pan_cohort")
    a = ap.parse_args()
    res = analyse(a.runs)
    for k, v in res.items():
        if k in ("runs", "panel"):
            print(k, "->", str(v)[:200])
            continue
        print(f"--- {k}: {'absent' if v is None else 'ok'}")
        if isinstance(v, dict):
            for kk, vv in v.items():
                if kk in ("rows", "draws", "arms", "blocks"):
                    print(f"    {kk}: {len(vv) if hasattr(vv,'__len__') else vv}"
                          " entries")
                else:
                    print(f"    {kk} = {str(vv)[:160]}")
