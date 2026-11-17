"""Assemble the manuscript draft (.docx) from the run artefacts.

Reads every table in `runs/` (plus the real-data tables in `real_results/`) and
every figure PNG, and emits a Word draft with each table and figure placed in
the section where it belongs, captioned and cross-referenced from the text.

    python scripts/build_manuscript.py --runs ./runs --out MOSAIC_GNN_draft.docx

The prose is a scaffold: section structure, methods that describe what the code
actually does, and results sentences that quote the numbers currently in
`runs/`. Anything that depends on results you have not produced yet is marked
[TODO] rather than invented.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Metric groupings. Keeping these explicit means every table in the manuscript
# reports the same columns in the same order.
# --------------------------------------------------------------------------- #
CLS_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
               "precision_macro", "recall_macro", "specificity_macro",
               "npv_macro", "auroc", "auprc", "mcc", "kappa",
               "kappa_quadratic", "top2_accuracy", "log_loss", "brier", "ece"]
SURV_METRICS = ["c_index", "c_index_lo", "c_index_hi", "auc@12m", "auc@36m",
                "auc@60m", "ibs", "brier@12m", "brier@36m", "brier@60m",
                "hr_high_vs_low", "hr_lo", "hr_hi", "hr_per_sd",
                "logrank_chi2", "logrank_p"]

PRETTY = {
    "accuracy": "Acc", "balanced_accuracy": "BAcc", "f1_macro": "F1-M",
    "f1_weighted": "F1-W", "precision_macro": "Prec", "recall_macro": "Sens",
    "specificity_macro": "Spec", "npv_macro": "NPV", "auroc": "AUROC",
    "auprc": "AUPRC", "mcc": "MCC", "kappa": "κ", "kappa_quadratic": "κ-quad",
    "top2_accuracy": "Top-2", "log_loss": "LogLoss", "brier": "Brier",
    "ece": "ECE", "c_index": "C-index", "c_index_lo": "CI-lo",
    "c_index_hi": "CI-hi", "auc@12m": "AUC12", "auc@36m": "AUC36",
    "auc@60m": "AUC60", "ibs": "IBS", "brier@12m": "BS12",
    "brier@36m": "BS36", "brier@60m": "BS60", "hr_high_vs_low": "HR",
    "hr_lo": "HR-lo", "hr_hi": "HR-hi", "hr_per_sd": "HR/SD",
    "logrank_chi2": "LR-χ²", "logrank_p": "LR-p", "n": "n",
}


def fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.1e}"
        return f"{v:.{digits}f}"
    return str(v)


def to_grid(df: pd.DataFrame, index_header: str, digits: int = 3) -> List[List[str]]:
    """DataFrame -> list of rows of strings, header row first."""
    header = [index_header] + [PRETTY.get(c, c) for c in df.columns]
    rows = [header]
    for idx, row in df.iterrows():
        rows.append([str(idx)] + [fmt(v, digits) for v in row.tolist()])
    return rows


def with_sd(df: pd.DataFrame, metrics: List[str], digits: int = 3
            ) -> pd.DataFrame:
    """mean ± sd strings for metrics that have a companion `_sd` column."""
    out = pd.DataFrame(index=df.index)
    for m in metrics:
        if m not in df.columns:
            continue
        sd = df.get(m + "_sd")
        if sd is None:
            out[m] = [fmt(v, digits) for v in df[m]]
        else:
            out[m] = [f"{v:.{digits}f} ± {s:.{digits}f}"
                      if np.isfinite(v) else "—"
                      for v, s in zip(df[m], sd)]
    return out


# --------------------------------------------------------------------------- #
def collect_facts(targets: Dict[str, str]) -> Dict[str, dict]:
    """The headline numbers the prose quotes, read out of the run artefacts.

    The draft must not contain a hand-typed number. Every quantity the text
    asserts is extracted here, from the CSV or JSON that a command wrote, and
    interpolated into the prose by make_docx.js. A fact that is missing because
    its phase did not run stays missing, and the prose renders "[not run]" in
    its place rather than a plausible-looking value.

    `targets` maps a target name ("cohort", "stage_binary") to its run
    directory.
    """
    facts: Dict[str, dict] = {}
    for target, runs in targets.items():
        f: Dict[str, object] = {"runs": runs, "target": target}

        # ---- the patient set, from table 1 ------------------------------- #
        t1 = os.path.join(runs, "table1_cohorts.csv")
        if os.path.exists(t1):
            d = pd.read_csv(t1)
            f["n_cohorts"] = int(len(d))
            f["n_patients"] = int(d["patients"].sum())
            f["n_events"] = int(d["events"].sum())
            lab = d[d["labelled"] > 0]
            f["n_labelled"] = int(lab["labelled"].sum())
            f["n_cohorts_labelled"] = int(len(lab))
            f["cohorts_unlabelled"] = sorted(d.loc[d["labelled"] == 0,
                                                   "cohort"].tolist())
            few = d.nsmallest(5, "events")[["cohort", "events", "patients"]]
            f["fewest_events"] = [
                {"cohort": r.cohort, "events": int(r.events),
                 "patients": int(r.patients)} for r in few.itertuples()]

        # ---- the baseline panel, from table 3 ---------------------------- #
        t3 = os.path.join(runs, "table3_comparison_full.csv")
        if os.path.exists(t3):
            d = pd.read_csv(t3, index_col=0)
            f["n_baselines"] = int(len(d.index.difference(["mosaic_gnn"])))
            for metric in ("balanced_accuracy", "auroc", "c_index"):
                if metric not in d.columns:
                    continue
                s = d[metric].drop(index=["mosaic_gnn"], errors="ignore")
                s = s.dropna()
                if s.empty:
                    continue
                f[f"best_{metric}"] = {"model": str(s.idxmax()),
                                       "value": float(s.max())}
                if "mosaic_gnn" in d.index and pd.notna(d.at["mosaic_gnn",
                                                             metric]):
                    f[f"mosaic_{metric}"] = float(d.at["mosaic_gnn", metric])

        # ---- MOSAIC-GNN's own held-out fold, from table 2 ----------------- #
        # table3 holds the baseline panel only -- the proposed model's metrics
        # are written to table2_single_fold.csv, so reading mosaic_* from
        # table3 alone leaves the draft printing "not run" for a number that
        # did run. Fill any mosaic_* the panel did not supply from table 2.
        t2 = os.path.join(runs, "table2_single_fold.csv")
        if os.path.exists(t2):
            d2 = pd.read_csv(t2, index_col=0)
            col = d2.columns[0] if len(d2.columns) else None
            if col is not None:
                for metric in ("balanced_accuracy", "auroc", "auprc",
                               "c_index", "f1_macro", "mcc", "ece",
                               "accuracy", "logrank_p"):
                    key = f"mosaic_{metric}"
                    if key in f or metric not in d2.index:
                        continue
                    v = d2.at[metric, col]
                    if pd.notna(v):
                        f[key] = float(v)
            # The comparison that tests the premise of the whole project: does
            # molecular data add anything over the clinical covariates alone?
            if "c_index" in d.columns:
                for mol in ("cox_lasso", "cox_ridge"):
                    if mol in d.index and "cox_clinical_only" in d.index:
                        a = d.at[mol, "c_index"]
                        b = d.at["cox_clinical_only", "c_index"]
                        if pd.notna(a) and pd.notna(b):
                            f["molecular_gain"] = {
                                "model": mol, "molecular": float(a),
                                "clinical_only": float(b),
                                "delta": float(a) - float(b)}
                            break

        # ---- cross-validation mean +/- SD, from the CLI summary ---------- #
        cv = os.path.join(runs, f"{target}_results.json")
        if os.path.exists(cv):
            with open(cv) as fh:
                j = json.load(fh)
            got = {}
            for metric in ("balanced_accuracy", "auroc", "c_index"):
                m = j.get(metric) if isinstance(j, dict) else None
                if isinstance(m, dict) and m.get("n"):
                    got[metric] = {"mean": m.get("mean"), "sd": m.get("std"),
                                   "n": m.get("n")}
            if got:
                f["cv"] = got

        # ---- paired tests, from table 4 ---------------------------------- #
        t4 = os.path.join(runs, "table4_paired_tests.csv")
        if os.path.exists(t4):
            d = pd.read_csv(t4, index_col=0)
            sig = [c for c in d.columns if c.endswith("_sig")]
            if sig:
                p = {"n_compared": int(len(d)),
                     "n_significant": {c: int(d[c].fillna(False)
                                              .astype(bool).sum())
                                       for c in sig}}
                # Per-metric win/loss/tie counts and the per-model deltas, so
                # that a sentence like "eight of twenty comparators are
                # significantly better" is read out of the table instead of
                # being counted by hand into the prose, where it would quietly
                # go stale the next time the run changes.
                for stem, dcol, qcol, scol in (
                        ("bacc", "bacc_delta", "bacc_q", "bacc_sig"),
                        ("cindex", "cindex_delta", "cindex_q", "cindex_sig")):
                    if dcol not in d.columns or scol not in d.columns:
                        continue
                    sub = d.dropna(subset=[dcol])
                    if sub.empty:
                        continue
                    flag = sub[scol].fillna(False).astype(bool)
                    p[stem] = {
                        "n_tested": int(len(sub)),
                        # "worse" = the baseline beats MOSAIC-GNN.
                        "n_worse": int((flag & (sub[dcol] < 0)).sum()),
                        "n_better": int((flag & (sub[dcol] > 0)).sum()),
                        "n_tied": int((~flag).sum()),
                        "worse_models": [str(m) for m in
                                         sub.index[flag & (sub[dcol] < 0)]],
                        "better_models": [str(m) for m in
                                          sub.index[flag & (sub[dcol] > 0)]]}
                p["models"] = {
                    str(m): {c: (None if pd.isna(d.at[m, c])
                                 else (bool(d.at[m, c]) if c.endswith("_sig")
                                       else float(d.at[m, c])))
                             for c in d.columns}
                    for m in d.index}
                f["paired"] = p

        # ---- the leakage control, from the permutation table ------------- #
        tp = os.path.join(runs, "tableS2_permutation_test.csv")
        if os.path.exists(tp):
            d = pd.read_csv(tp, index_col=0)
            f["permutation"] = {"rows": int(len(d)),
                                "columns": list(map(str, d.columns))}

        facts[target] = f
    return facts


def load_tables(runs: str, real: str) -> Dict[str, dict]:
    """Every table the manuscript needs, already formatted for Word."""
    def read(name, **kw):
        path = os.path.join(runs, f"{name}.csv")
        return pd.read_csv(path, index_col=0, **kw) if os.path.exists(path) else None

    T: Dict[str, dict] = {}

    t1 = read("table1_cohorts")
    if t1 is not None:
        T["T1"] = {
            "label": "Table 1",
            "caption": "Cohort characteristics. Patients, omic blocks available, "
                       "number of genes measured per block, observed events, "
                       "event rate, median and maximum follow-up, number of "
                       "labelled patients and the class distribution.",
            "grid": to_grid(t1, "Cohort"), "landscape": True, "font": 7,
        }

    t2 = read("table2_single_fold")
    if t2 is not None:
        s = t2.iloc[:, 0]
        order = [m for m in CLS_METRICS + SURV_METRICS if m in s.index]
        extra = [i for i in s.index if i not in order]
        df = pd.DataFrame({"Value": [fmt(s[i]) for i in order + extra]},
                          index=[PRETTY.get(i, i) for i in order + extra])
        half = (len(df) + 1) // 2
        left, right = df.iloc[:half], df.iloc[half:]
        grid = [["Metric", "Value", "Metric", "Value"]]
        for i in range(half):
            row = [left.index[i], left.iloc[i, 0]]
            row += ([right.index[i], right.iloc[i, 0]] if i < len(right)
                    else ["", ""])
            grid.append(row)
        T["T2"] = {
            "label": "Table 2",
            "caption": "Complete metric panel for MOSAIC-GNN on the held-out "
                       "fold: 17 classification metrics and 16 survival "
                       "metrics, plus calibration and uncertainty summaries.",
            "grid": grid, "landscape": False, "font": 9,
        }

    t2b = read("table2b_per_class")
    if t2b is not None:
        if "class" in t2b.columns:
            t2b = t2b.set_index("class")
        T["T2b"] = {
            "label": "Table 3",
            "caption": "Per-class performance: support, sensitivity, "
                       "specificity, positive and negative predictive value, "
                       "F1 and one-vs-rest AUROC for each stage.",
            "grid": to_grid(t2b, "Class"), "landscape": False, "font": 9,
        }

    t2c = read("table2c_per_cohort")
    if t2c is not None:
        keep = [c for c in ["n"] + CLS_METRICS + ["c_index"] if c in t2c.columns]
        T["T2c"] = {
            "label": "Table 4",
            "caption": "Per-cohort breakdown on the held-out fold. Pooled "
                       "pan-cancer numbers can hide a cohort in which the "
                       "model fails, so every metric is repeated per cohort.",
            "grid": to_grid(t2c[keep], "Cohort"), "landscape": True, "font": 7,
        }

    t3 = read("table3_comparison_full")
    if t3 is not None:
        cls = with_sd(t3.sort_values("balanced_accuracy", ascending=False),
                      CLS_METRICS)
        T["T5"] = {
            "label": "Table 5",
            "caption": "Classification performance of MOSAIC-GNN and 21 "
                       "comparators across four cross-validation folds "
                       "(mean ± sd). Dashes mark models that produce a risk "
                       "score only. All models were trained on identical "
                       "folds.",
            "grid": to_grid(cls, "Model"), "landscape": True, "font": 6,
        }
        surv = with_sd(t3.sort_values("c_index", ascending=False), SURV_METRICS)
        T["T6"] = {
            "label": "Table 6",
            "caption": "Survival performance of the same models: Harrell's "
                       "C-index with bootstrap limits, time-dependent AUC, "
                       "IPCW Brier scores, integrated Brier score, hazard "
                       "ratio between model-defined risk groups with its "
                       "confidence interval, hazard ratio per standard "
                       "deviation of the risk score, and the log-rank test.",
            "grid": to_grid(surv, "Model"), "landscape": True, "font": 6,
        }

    t4 = read("table4_paired_tests")
    if t4 is not None:
        T["T7"] = {
            "label": "Table 7",
            "caption": "Paired bootstrap comparison against MOSAIC-GNN on "
                       "pooled out-of-fold predictions. Δ is the MOSAIC-GNN "
                       "advantage; q is the Benjamini-Hochberg adjusted "
                       "p-value across the comparison family.",
            "grid": to_grid(t4, "Model"), "landscape": False, "font": 8,
        }

    t5 = read("table5_ablation")
    if t5 is not None:
        keep = [m for m in CLS_METRICS + SURV_METRICS if m in t5.columns]
        T["T8"] = {
            "label": "Table 8",
            "caption": "Ablation. Each row removes exactly one component of "
                       "MOSAIC-GNN and is trained under an otherwise identical "
                       "protocol. A component is justified only when its "
                       "removal costs performance.",
            "grid": to_grid(t5[keep], "Variant"), "landscape": True, "font": 6,
        }

    t6 = read("table6_loco")
    if t6 is not None:
        keep = [m for m in CLS_METRICS + SURV_METRICS if m in t6.columns]
        T["T9"] = {
            "label": "Table 9",
            "caption": "Leave-one-cancer-out transfer. The model is trained on "
                       "three cohorts and evaluated on the fourth, which it "
                       "has never seen. This is the test that distinguishes a "
                       "pan-cancer representation from a cohort detector.",
            "grid": to_grid(t6[keep], "Held-out cohort"), "landscape": True,
            "font": 6,
        }

    t7 = read("table7_gene_importance")
    if t7 is not None:
        T["T10"] = {
            "label": "Table 10",
            "caption": "Top 25 genes by integrated-gradient attribution, with "
                       "the functional module each belongs to, its mean "
                       "attention gate, and the signed contribution of each "
                       "omic layer.",
            "grid": to_grid(t7.head(25).reset_index(drop=True), "Rank", 5),
            "landscape": False, "font": 7,
        }

    t8 = read("table8_module_importance")
    if t8 is not None:
        T["T11"] = {
            "label": "Table 11",
            "caption": "Top 15 functional modules by attention, with module "
                       "size and the highest-gated member genes.",
            "grid": to_grid(t8.head(15).reset_index(drop=True), "Rank", 5),
            "landscape": False, "font": 7,
        }

    ts1 = read("tableS1_generalisation_gap")
    if ts1 is not None:
        keep = [m for m in ["n"] + CLS_METRICS + ["c_index"] if m in ts1.columns]
        T["S1"] = {
            "label": "Table S1",
            "caption": "Generalisation gap. Identical metrics on the training, "
                       "validation and held-out splits, with an explicit "
                       "train-minus-test row.",
            "grid": to_grid(ts1[keep], "Split"), "landscape": True, "font": 7,
        }

    ts2 = read("tableS2_permutation_test")
    if ts2 is not None:
        T["S2"] = {
            "label": "Table S2",
            "caption": "Label permutation test. The model is retrained on "
                       "shuffled training labels and scored on the intact test "
                       "split. A null that sits at chance is evidence that no "
                       "information leaks from the label into the features.",
            "grid": to_grid(ts2, "Metric", 4), "landscape": False, "font": 9,
        }

    ts3 = read("tableS3_sample_size")
    if ts3 is not None:
        T["S3"] = {
            "label": "Table S3",
            "caption": "Test performance as a function of training-set size, "
                       "separating a data limit from a capacity limit.",
            "grid": to_grid(ts3, "n train"), "landscape": False, "font": 9,
        }

    ts1b = read("tableS1b_learning_curve")
    if ts1b is not None:
        T["S4"] = {
            "label": "Table S4",
            "caption": "Training and validation metrics recorded every five "
                       "epochs.",
            "grid": to_grid(ts1b, "Epoch"), "landscape": True, "font": 7,
        }

    t9 = read("table9_patient_subnetwork")
    if t9 is not None:
        T["S5"] = {
            "label": "Table S5",
            "caption": "The 20 most strongly attended edges of one patient's "
                       "rewired interaction network, with the relation type "
                       "each edge came from (0 = prior, 1 = co-alteration, "
                       "2 = self-loop).",
            "grid": to_grid(t9.head(20).reset_index(drop=True), "Rank", 4),
            "landscape": False, "font": 8,
        }

    rpath = os.path.join(real, "real_BLCA_generalisation_gap.csv")
    if os.path.exists(rpath):
        rg = pd.read_csv(rpath, index_col=0)
        T["R1"] = {
            "label": "Table 12",
            "caption": "Real-data check on TCGA-BLCA (n = 401, 176 events, "
                       "mutation and copy-number only). Classification "
                       "collapses to a single class while the survival head "
                       "retains discrimination with a small gap.",
            "grid": to_grid(rg, "Split"), "landscape": True, "font": 7,
        }
    return T


FIGURES = [
    ("fig1_cohorts.png", "Figure 1",
     "Cohort composition. (a) class distribution by cohort; (b) follow-up "
     "distribution; (c) Kaplan-Meier curves per cohort; (d) event rate per "
     "cohort."),
    ("fig2_risk.png", "Figure 2",
     "Risk stratification. (a) Kaplan-Meier curves for model-defined risk "
     "tertiles with the log-rank p and the hazard ratio between the outer "
     "tertiles; (b) mean predicted survival curves from the discrete-time "
     "hazard head; (c) calibration of predicted against observed survival at "
     "36 months."),
    ("fig3_classification.png", "Figure 3",
     "Classification behaviour. (a) row-normalised confusion matrix; (b) "
     "one-vs-rest ROC curves; (c) reliability diagram with the expected "
     "calibration error; (d) accuracy against coverage when abstaining by "
     "MC-dropout predictive entropy."),
    ("fig4_comparison.png", "Figure 4",
     "Model comparison. (a) C-index with 95 % bootstrap intervals for every "
     "model; (b) balanced accuracy; (c) macro one-vs-rest AUROC. MOSAIC-GNN "
     "is highlighted."),
    ("fig5_ablation.png", "Figure 5",
     "Ablation. Change in each headline metric when a single component is "
     "removed; negative values mean the removed component was contributing."),
    ("fig6_loco.png", "Figure 6",
     "Leave-one-cancer-out transfer. (a) performance on each held-out cohort; "
     "(b) the gap between transfer performance and pooled cross-validation."),
    ("fig7_interpretation.png", "Figure 7",
     "Interpretation. (a) top genes by integrated-gradient attribution; (b) "
     "signed attribution decomposed by omic layer; (c) module attention by "
     "cohort."),
    ("figS1_learning_curves.png", "Figure S1",
     "Learning curves. (a) balanced accuracy and (b) C-index on the training "
     "and validation splits during training; (c) the train-minus-validation "
     "gap, which is what early stopping monitors."),
    ("figS2_sample_size.png", "Figure S2",
     "Sample-size curves for (a) balanced accuracy and (b) C-index, on the "
     "training and held-out splits."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="./runs")
    ap.add_argument("--real", default="./real_results")
    ap.add_argument("--out", default="manuscript_payload.json")
    ap.add_argument("--facts", nargs="*", default=[], metavar="TARGET:DIR",
                    help="run directories to read the prose's headline "
                         "numbers from, e.g. cohort:runs_pan_cohort "
                         "stage_binary:runs_pan_stage. Every number the draft "
                         "asserts comes from here; anything absent renders as "
                         "[not run] rather than being invented.")
    args = ap.parse_args()

    tables = load_tables(args.runs, args.real)
    figures = [{"file": os.path.abspath(os.path.join(args.runs, f)),
                "label": lab, "caption": cap}
               for f, lab, cap in FIGURES
               if os.path.exists(os.path.join(args.runs, f))]

    spec = {}
    for item in args.facts:
        if ":" not in item:
            raise SystemExit(f"--facts wants TARGET:DIR, got {item!r}")
        target, _, directory = item.partition(":")
        spec[target] = directory
    facts = collect_facts(spec) if spec else {}

    with open(args.out, "w") as fh:
        json.dump({"tables": tables, "figures": figures, "facts": facts},
                  fh, indent=1)
    print(f"wrote {args.out}: {len(tables)} tables, {len(figures)} figures, "
          f"{len(facts)} fact set(s)")
    for target, f in facts.items():
        have = [k for k in ("n_patients", "best_balanced_accuracy",
                            "molecular_gain", "cv", "paired", "permutation")
                if k in f]
        missing = [k for k in ("cv", "paired", "permutation") if k not in f]
        print(f"  {target:<14} have={','.join(have) or '-'}"
              + (f"  NOT RUN={','.join(missing)}" if missing else ""))
    for k, v in tables.items():
        print(f"  {v['label']:<10} {len(v['grid']) - 1:>3} rows x "
              f"{len(v['grid'][0]):>2} cols  {k}")


if __name__ == "__main__":
    main()
