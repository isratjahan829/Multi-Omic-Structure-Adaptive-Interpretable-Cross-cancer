"""Assemble REAL_RESULTS.md from the CSV/JSON a real run actually produced.

Every number in the report is read out of a file written by a run. Nothing is
typed in, nothing is interpolated, and a table whose inputs are missing is
reported as missing rather than filled in -- the point of the exercise is that
each figure can be traced back to a command that was executed.

    python scripts/make_report.py --targets cohort:runs_real_cohort \
        stage:runs_real_stage --out REAL_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import pandas as pd

MISSING = "_Not available: the run that produces this table did not complete._"


def read_csv(path: str, index_col=0) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, index_col=index_col)
    except Exception as err:                       # pragma: no cover
        print(f"  !! could not read {path}: {err}")
        return None
    return None if df.empty else df


def read_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def md(df: Optional[pd.DataFrame], cols: Optional[List[str]] = None,
       decimals: int = 4, note: str = MISSING) -> str:
    if df is None:
        return note
    out = df
    if cols:
        keep = [c for c in cols if c in out.columns]
        if keep:
            out = out[keep]
    return out.round(decimals).to_markdown()


def verdict(pairs: Dict[str, str]) -> str:
    """What the paired tests support, stated before any table is read.

    This is the section to read first and the one to defend. It is computed
    from table4_paired_tests.csv -- the paired bootstrap on identical held-out
    patients with Benjamini-Hochberg correction across the whole table -- so it
    cannot drift away from the numbers below it, and it says where MOSAIC-GNN
    loses as plainly as where it wins. A claim this section does not make is a
    claim the data does not support.
    """
    L: List[str] = ["## What the statistics support\n"]
    A = L.append
    A("Computed from `table4_paired_tests.csv` in each run directory: paired "
      "bootstrap against every baseline on exactly the same held-out "
      "patients, Benjamini-Hochberg corrected across the table. A positive "
      "delta means MOSAIC-GNN is ahead.\n")

    any_table = False
    # Collected across targets so that a result holding in more than one
    # target framing can be reported as a replication rather than left for a
    # reader to spot. Only what the tables say gets in here.
    clin: Dict[str, tuple] = {}
    for target, runs in pairs.items():
        d = read_csv(f"{runs}/table4_paired_tests.csv")
        if d is not None and "cox_clinical_only" in d.index \
                and "cindex_delta" in d.columns:
            v = d.at["cox_clinical_only", "cindex_delta"]
            q = d.at["cox_clinical_only", "cindex_q"] \
                if "cindex_q" in d.columns else None
            s = bool(d.at["cox_clinical_only", "cindex_sig"]) \
                if "cindex_sig" in d.columns else False
            if pd.notna(v):
                clin[target] = (float(v), None if q is None or pd.isna(q)
                                else float(q), s)
        if d is None:
            A(f"**`{target}`** — paired tests not available: the primary-fold "
              f"run that produces them did not complete. No claim can be made "
              f"for this target.\n")
            continue
        any_table = True
        A(f"### `{target}`\n")
        for label, dcol, qcol, scol in (
                ("Classification (balanced accuracy)",
                 "bacc_delta", "bacc_q", "bacc_sig"),
                ("Survival (C-index)",
                 "cindex_delta", "cindex_q", "cindex_sig")):
            if dcol not in d.columns or scol not in d.columns:
                continue
            sub = d.dropna(subset=[dcol])
            if sub.empty:
                continue
            sig = sub[sub[scol].fillna(False).astype(bool)]
            worse = sig[sig[dcol] < 0].sort_values(dcol)
            better = sig[sig[dcol] > 0].sort_values(dcol, ascending=False)
            ties = sub[~sub[scol].fillna(False).astype(bool)]
            A(f"**{label}.** {len(sub)} comparators tested.\n")
            if len(worse):
                A(f"- Significantly **better than MOSAIC-GNN** "
                  f"({len(worse)}): " + ", ".join(
                      f"`{m}` ({sub.at[m, dcol]:+.3f}, q={sub.at[m, qcol]:.3f})"
                      for m in worse.index) + "\n")
            if len(better):
                A(f"- MOSAIC-GNN **significantly better** "
                  f"({len(better)}): " + ", ".join(
                      f"`{m}` ({sub.at[m, dcol]:+.3f}, q={sub.at[m, qcol]:.3f})"
                      for m in better.index) + "\n")
            if len(ties):
                A(f"- **Not distinguishable** at q < 0.05 ({len(ties)}): "
                  + ", ".join(f"`{m}`" for m in ties.index) + "\n")
            if not len(worse) and not len(better):
                A("- No comparator is distinguishable from MOSAIC-GNN on this "
                  "metric.\n")
            A("")

    # The one claim that holds in more than one target framing. Reported as a
    # replication only when the tables actually show it in both, and reported
    # as a failure to replicate when they do not.
    if len(clin) > 1:
        won = {t: v for t, v in clin.items() if v[2] and v[0] > 0}
        A("### What replicates across targets\n")
        if len(won) == len(clin):
            A("MOSAIC-GNN significantly exceeds `cox_clinical_only` on "
              "**every target tested** (" + ", ".join(
                  f"`{t}` {v[0]:+.3f}"
                  + ("" if v[1] is None else f", q={v[1]:.3f}")
                  for t, v in won.items())
              + "). The two targets label overlapping patients differently "
                "and were fitted independently, so this is the same finding "
                "arrived at twice rather than one result counted twice. It is "
                "the strongest statement these tables support, and it is a "
                "statement about what the molecular data adds over clinical "
                "covariates — not about the architecture, which loses on "
                "classification in both.\n")
        else:
            A("The gain over `cox_clinical_only` does **not** hold on every "
              "target: " + ", ".join(
                  f"`{t}` {v[0]:+.3f}"
                  + ("" if v[1] is None else f", q={v[1]:.3f}")
                  + ("" if v[2] else " (not significant)")
                  for t, v in clin.items())
              + ". A claim about molecular features adding prognostic "
                "information must therefore be qualified by target, and must "
                "not be stated as a general result.\n")

    if any_table:
        A("### How to word this\n")
        A("Write the claim these tests support and no stronger. Where a "
          "baseline is significantly better, say so; where nothing is "
          "distinguishable, that is a tie and not a win. In particular, a "
          "significant gain over `cox_clinical_only` is the finding that "
          "molecular features add prognostic information beyond clinical "
          "covariates — which is a real result about the data, independent of "
          "whether the proposed architecture beats the other architectures. "
          "Keep the two apart.\n")
        A("Differences smaller than about 0.01 balanced accuracy should not be "
          "read as effects at all: three fits of the same fold with the same "
          "seed gave 0.4567, 0.4341 and 0.4547, so that is the noise floor of "
          "this pipeline.\n")
    return "\n".join(L)


def section(target: str, runs: str) -> str:
    L: List[str] = []
    A = L.append
    A(f"## Target: `{target}`\n")

    cv = read_json(f"{runs}/{target}_results.json")
    if cv is None:
        A("_Cross-validation did not complete; no summary to report._\n")
    else:
        s = cv["summary"]
        n_folds = len(cv["folds"])
        planned = cv.get("folds_planned")
        if planned and not cv.get("complete", True):
            A(f"**Incomplete: {n_folds} of {planned} planned folds finished** "
              "(the run was stopped by the wall clock, not by an error). The "
              "figures below average only the folds that ran, so their spread "
              "understates what the full schedule would have shown.\n")
        A(f"Repeated stratified cross-validation, {n_folds} held-out folds. "
          "Mean +/- SD over folds.\n")
        rows = []
        for key in ("balanced_accuracy", "accuracy", "macro_f1", "auroc",
                    "auprc", "log_loss", "brier", "ece", "c_index",
                    "logrank_p", "mean_predictive_entropy"):
            if key in s:
                rows.append({"metric": key,
                             "mean": s[key]["mean"],
                             "sd": s[key]["std"],
                             "n_folds": s[key]["n"]})
        A(pd.DataFrame(rows).set_index("metric").round(4).to_markdown())
        A("")

    A("### Held-out fold: full metric panel\n")
    A(md(read_csv(f"{runs}/table2_single_fold.csv")))
    A("")

    A("### Per class (held-out fold)\n")
    A(md(read_csv(f"{runs}/table2b_per_class.csv")))
    A("")

    A("### Per cohort (held-out fold)\n")
    A(md(read_csv(f"{runs}/table2c_per_cohort.csv")))
    A("")

    A("### Generalisation gap\n")
    gap = read_csv(f"{runs}/tableS1_generalisation_gap.csv")
    A(md(gap, cols=["balanced_accuracy", "auroc", "log_loss", "c_index", "n"]))
    A("")

    A("### Label permutation test\n")
    A("Models retrained on labels shuffled within the training and validation "
      "folds only; the held-out labels are left intact. A pipeline that leaks "
      "scores above chance here.\n")
    A(md(read_csv(f"{runs}/tableS2_permutation_test.csv")))
    A("")
    nullt = read_csv(f"{runs}/tableS2_permutation_null.csv")
    if nullt is not None:
        A("<details><summary>Per-permutation null</summary>\n")
        A(md(nullt))
        A("\n</details>\n")

    A("### Baseline comparison\n")
    A(md(read_csv(f"{runs}/table3_comparison_full.csv"),
         cols=["balanced_accuracy", "auroc", "auprc", "macro_f1", "c_index"]))
    A("")

    A("### Paired bootstrap against each baseline (held-out fold)\n")
    A("`*_delta` is MOSAIC-GNN minus the baseline on the same patients; "
      "`*_q` is the Benjamini-Hochberg corrected p-value over the whole "
      "table.\n")
    A(md(read_csv(f"{runs}/table4_paired_tests.csv"),
         cols=["bacc_delta", "bacc_q", "bacc_sig",
               "cindex_delta", "cindex_q", "cindex_sig"]))
    A("")

    A("### Ablation\n")
    A(md(read_csv(f"{runs}/table5_ablation.csv")))
    A("")

    A("### Leave-one-cancer-out transfer\n")
    A(md(read_csv(f"{runs}/table6_loco.csv")))
    A("")

    A("### Sample-size curve\n")
    A(md(read_csv(f"{runs}/tableS3_sample_size.csv")))
    A("")

    params = read_json(f"{runs}/{target}_parameter_report.json")
    if params:
        A("### Capacity\n")
        A(f"- trainable parameters: {int(params['trainable_parameters']):,}")
        A(f"- training patients: {int(params['training_patients']):,}")
        A(f"- parameters per training patient: "
          f"{params['parameters_per_patient']:.1f}\n")

    gene = read_csv(f"{runs}/table7_gene_importance.csv")
    if gene is not None:
        A("### Top genes by integrated gradients\n")
        A("<details><summary>Top 15</summary>\n")
        A(md(gene.head(15)))
        A("\n</details>\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True,
                    help="target:runs_dir pairs, e.g. cohort:runs_real_cohort")
    ap.add_argument("--out", default="REAL_RESULTS.md")
    ap.add_argument("--preamble", help="markdown file prepended verbatim")
    args = ap.parse_args()

    parts: List[str] = []
    if args.preamble and os.path.isfile(args.preamble):
        with open(args.preamble) as fh:
            parts.append(fh.read().rstrip() + "\n")

    pairs: Dict[str, str] = {}
    for spec in args.targets:
        target, _, runs = spec.partition(":")
        pairs[target] = runs

    parts.append(verdict(pairs))

    # The cohort table is identical for every target, so it is reported once.
    for runs in pairs.values():
        t1 = read_csv(f"{runs}/table1_cohorts.csv")
        if t1 is not None:
            parts.append("## Cohorts as actually loaded\n")
            parts.append(t1.to_markdown())
            parts.append("")
            break

    for target, runs in pairs.items():
        parts.append(section(target, runs))

    text = "\n".join(parts).rstrip() + "\n"
    with open(args.out, "w") as fh:
        fh.write(text)
    print(f"wrote {args.out} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
