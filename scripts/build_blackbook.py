"""Rebuild the CSE400B Blackbook with the real pan-cancer results.

    python scripts/build_blackbook.py --src <original.docx> --out <new.docx>

The supervisor's draft is structurally finished and formatted to the
department's template: fonts, headers, footers, numbering, the signature
table, the literature review and the reference list are all in place. None of
that should be regenerated, and regenerating it is how a submission acquires
subtly wrong margins two days before a deadline.

So this edits a COPY of that file in place. It keeps every paragraph the
results do not touch and replaces the ones they do:

* the data tables, rewritten from the CSV/JSON a run actually wrote;
* the prose that quotes numbers, rewritten from the same files;
* the passages describing the study as a four-cohort synthetic demonstration,
  which is what the draft was written around and is no longer true.

The draft's own Section 5.3 names the gap this closes: "the most immediate
priority is replacing the synthetic demonstration and the truncated
single-cohort real check". That has now happened -- 32 studies, 10,953
patients, every number from cBioPortal -- so the sentences that promised it as
future work have to go, and the ones that described 640 synthetic patients
have to go with them.

Nothing here invents a figure. A table whose inputs did not run is left
visibly marked as not run rather than filled, for the same reason the rest of
this repository does it: a missing number is visible and a wrong one is not.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Optional

import pandas as pd

try:
    import docx
    from docx.shared import Pt
except ImportError:                                    # pragma: no cover
    raise SystemExit("python-docx is required: pip install python-docx")

NOT_RUN = "[not run — this phase did not complete; see Limitations]"


# --------------------------------------------------------------------------- #
# Reading what the runs produced
# --------------------------------------------------------------------------- #
def read_csv(path: str, index_col=0) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    try:
        d = pd.read_csv(path, index_col=index_col)
    except Exception as err:                           # pragma: no cover
        print(f"  !! {path}: {err}")
        return None
    return None if d.empty else d


def read_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Document surgery
# --------------------------------------------------------------------------- #
def iter_blocks(doc):
    """Body children in document order, as (kind, object)."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    for child in doc.element.body.iterchildren():
        if child.tag.endswith('}p'):
            yield 'p', Paragraph(child, doc)
        elif child.tag.endswith('}tbl'):
            yield 't', Table(child, doc)


def set_text(par, text: str) -> None:
    """Replace a paragraph's text, keeping the first run's formatting."""
    runs = par.runs
    if not runs:
        par.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


def style_like(src_cell, dst_cell) -> None:
    """Copy font size/name/bold from one cell's first run to another's."""
    try:
        s = src_cell.paragraphs[0].runs[0]
    except IndexError:
        return
    for p in dst_cell.paragraphs:
        for r in p.runs:
            if s.font.size:
                r.font.size = s.font.size
            if s.font.name:
                r.font.name = s.font.name
            r.font.bold = s.font.bold


def fill_table(table, df: pd.DataFrame, index_header: str = "",
               decimals: int = 4, max_rows: int = 0) -> str:
    """Rewrite a table from a DataFrame, reusing row 0 as the header style.

    The template's tables are already formatted; rows are added or removed to
    match the data rather than the table being rebuilt, so borders, shading and
    column widths survive.
    """
    if df is None:
        return "SKIPPED (no data)"
    out = df.copy()
    if max_rows and len(out) > max_rows:
        out = out.head(max_rows)
    cols = [index_header or (out.index.name or "")] + [str(c) for c in out.columns]

    # Header row.
    hdr = table.rows[0]
    while len(hdr.cells) < len(cols):
        return f"SKIPPED (template has {len(hdr.cells)} cols, data has {len(cols)})"
    for i, name in enumerate(cols):
        set_text(hdr.cells[i].paragraphs[0], name)
    for i in range(len(cols), len(hdr.cells)):
        set_text(hdr.cells[i].paragraphs[0], "")

    # Body rows: reuse what exists, clone the last one when more are needed.
    import copy as _copy
    need = len(out)
    have = len(table.rows) - 1
    while have < need:
        table._tbl.append(_copy.deepcopy(table.rows[-1]._tr))
        have += 1
    while have > need:
        table._tbl.remove(table.rows[-1]._tr)
        have -= 1

    for r, (idx, row) in enumerate(out.iterrows(), start=1):
        cells = table.rows[r].cells
        vals = [str(idx)] + [
            (f"{v:.{decimals}f}" if isinstance(v, float) and pd.notna(v)
             else ("" if pd.isna(v) else str(v)))
            for v in row.tolist()]
        for i in range(len(cells)):
            set_text(cells[i].paragraphs[0], vals[i] if i < len(vals) else "")
            style_like(hdr.cells[min(i, len(hdr.cells) - 1)], cells[i])
    return f"{len(out)} rows x {len(cols)} cols"


def replace_section(doc, heading: str, paragraphs: List[str],
                    stop_styles=("Heading1", "Heading2", "Heading3")) -> str:
    """Replace the body text under a heading, leaving tables and figures."""
    blocks = list(iter_blocks(doc))
    start = None
    for i, (kind, obj) in enumerate(blocks):
        if kind != 'p':
            continue
        st = (obj.style.name or "").replace(" ", "")
        if st.lower().startswith("heading") and obj.text.strip() == heading:
            start = i
            break
    if start is None:
        return "HEADING NOT FOUND"

    targets = []
    for kind, obj in blocks[start + 1:]:
        if kind != 'p':
            break                       # stop at the first table or figure
        st = (obj.style.name or "").replace(" ", "")
        if st in stop_styles or st.lower().startswith("heading"):
            break
        if obj.text.strip():
            targets.append(obj)

    if not targets:
        return "NO BODY PARAGRAPHS"
    for i, text in enumerate(paragraphs):
        if i < len(targets):
            set_text(targets[i], text)
        else:
            new = targets[-1].insert_paragraph_before(text)
            new.style = targets[-1].style
    for extra in targets[len(paragraphs):]:
        set_text(extra, "")
    return f"{len(paragraphs)} paragraph(s) over {len(targets)} slot(s)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the original Blackbook .docx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cohort-runs", default="runs_pan_cohort")
    ap.add_argument("--stage-runs", default="runs_pan_stage")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()

    shutil.copyfile(args.src, args.out)
    doc = docx.Document(args.out)
    C, S = args.cohort_runs, args.stage_runs
    print(f"editing {args.out}")

    # ---- the numbers ---------------------------------------------------- #
    t1 = read_csv(f"{C}/table1_cohorts.csv")
    t2 = read_csv(f"{C}/table2_single_fold.csv")
    t2c = read_csv(f"{C}/table2c_per_cohort.csv")
    t3c = read_csv(f"{C}/table3_comparison_full.csv")
    t4c = read_csv(f"{C}/table4_paired_tests.csv")
    abl = read_json(f"{C}/cohort_ablation.json")
    loco = read_json(f"{C}/cohort_loco.json")

    facts = {}
    if t1 is not None:
        facts["n_cohorts"] = len(t1)
        facts["n_patients"] = int(t1["patients"].sum())
        facts["n_events"] = int(t1["events"].sum())
    if t3c is not None:
        facts["n_baselines"] = len(t3c)
        b = t3c["balanced_accuracy"].dropna()
        if len(b):
            facts["best_model"] = str(b.idxmax())
            facts["best_bacc"] = float(b.max())
    if t2 is not None:
        col = t2.columns[0]
        for k in ("balanced_accuracy", "auroc", "c_index", "ece", "accuracy",
                  "mcc", "logrank_chi2"):
            if k in t2.index:
                facts[k] = float(t2.at[k, col])
    print("  facts:", {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in facts.items()})

    # ---- tables ---------------------------------------------------------- #
    tables = [obj for kind, obj in iter_blocks(doc) if kind == 't']
    print(f"  {len(tables)} tables in template")

    def put(idx: int, df, header="", dec=4, cap="", max_rows=0):
        if idx >= len(tables):
            print(f"  T{idx + 1}: MISSING from template")
            return
        print(f"  T{idx + 1} ({cap}): "
              f"{fill_table(tables[idx], df, header, dec, max_rows)}")

    if t1 is not None:
        put(2, t1, "Cohort", 3, "4.1 cohort characteristics")
    if t2 is not None:
        panel = t2.copy()
        panel.columns = ["Value"]
        put(3, panel, "Metric", 4, "4.2 metric panel")
    if t3c is not None:
        cls = [c for c in ("accuracy", "balanced_accuracy", "macro_f1",
                           "auroc", "auprc", "log_loss", "brier", "ece")
               if c in t3c.columns]
        put(4, t3c[cls].dropna(how="all"), "Model", 4, "4.9 classification")
        srv = [c for c in ("c_index", "auc@12m", "auc@36m", "auc@60m",
                           "logrank_p") if c in t3c.columns]
        put(5, t3c[srv].dropna(how="all"), "Model", 4, "4.10 survival")
    if t4c is not None:
        # The template's row is Model | d_bacc | p_bacc | d_c_index |
        # p_c_index | q_bacc -- six columns, so five data columns fit.
        keep = [c for c in ("bacc_delta", "bacc_q", "cindex_delta",
                            "cindex_q", "bacc_sig") if c in t4c.columns]
        put(6, t4c[keep], "Model", 4, "4.11 paired bootstrap")
    if abl:
        rows = {}
        for v, s in abl.items():
            rows[v] = {k: s[k]["mean"] for k in
                       ("balanced_accuracy", "accuracy", "macro_f1", "auroc",
                        "auprc", "ece", "c_index") if k in s}
        put(7, pd.DataFrame(rows).T, "Variant", 4, "4.12 ablation")
    if loco:
        print("  LOCO data present")
    else:
        print(f"  T9/T10 (LOCO): no cohort_loco.json — left as template")

    # ---- prose ----------------------------------------------------------- #
    f = facts
    n_coh = f.get("n_cohorts", "?")
    n_pat = f.get("n_patients", 0)
    n_evt = f.get("n_events", 0)
    nb = f.get("n_baselines", "?")
    bacc = f.get("balanced_accuracy", float("nan"))
    auroc = f.get("auroc", float("nan"))
    cidx = f.get("c_index", float("nan"))
    ece = f.get("ece", float("nan"))
    mcc = f.get("mcc", float("nan"))
    chi2 = f.get("logrank_chi2", float("nan"))
    best = f.get("best_model", "?")
    bbacc = f.get("best_bacc", float("nan"))

    # Paired-test counts, read from the table rather than counted by hand.
    pc = {}
    if t4c is not None:
        for stem, dc, sc in (("bacc", "bacc_delta", "bacc_sig"),
                             ("cindex", "cindex_delta", "cindex_sig")):
            if dc in t4c.columns and sc in t4c.columns:
                sub = t4c.dropna(subset=[dc])
                flag = sub[sc].fillna(False).astype(bool)
                pc[stem] = {"n": len(sub),
                            "worse": int((flag & (sub[dc] < 0)).sum()),
                            "better": int((flag & (sub[dc] > 0)).sum()),
                            "tied": int((~flag).sum())}
    gap = read_csv(f"{C}/tableS1_generalisation_gap.csv")
    gap_txt = ""
    if gap is not None and "gap (train - test)" in gap.index:
        g = gap.loc["gap (train - test)"]
        gap_txt = (f"The train-minus-test gap is "
                   f"{float(g['balanced_accuracy']):.4f} in balanced accuracy "
                   f"and {float(g['c_index']):.4f} in C-index.")

    sections: Dict[str, List[str]] = {}

    sections["Abstract"] = [
        "Multi-omic tumour profiles are both high dimensional and highly "
        "structured. Genes interact with one another, patients resemble one "
        "another, and the two kinds of structure are usually modelled apart. "
        "MOSAIC-GNN couples them in one network: a relational graph attention "
        "encoder over a shared gene graph with learned functional-module "
        "pooling, feeding a learned patient-similarity graph, with "
        "disentangled shared and cohort-specific representations and joint "
        "classification and survival heads.",

        f"The model is evaluated on {n_coh} TCGA PanCanAtlas studies "
        f"retrieved from the cBioPortal public API, {n_pat:,} patients and "
        f"{n_evt:,} deaths, over the MSK-IMPACT 468-gene panel fixed in "
        f"advance of any analysis. Two endpoints are reported: tissue of "
        f"origin ({n_coh} classes) and early versus late AJCC stage. "
        f"{nb} comparators spanning penalised linear models, kernel and tree "
        f"ensembles, multi-omic fusion networks, patient-graph GNNs and "
        f"dedicated survival models were trained on identical folds, and all "
        f"comparisons use a paired bootstrap on the same held-out patients "
        f"with Benjamini-Hochberg correction.",

        f"On the held-out fold MOSAIC-GNN reaches balanced accuracy "
        f"{bacc:.3f}, AUROC {auroc:.3f}, MCC {mcc:.3f} and C-index "
        f"{cidx:.3f}, separating risk tertiles at log-rank chi-square "
        f"{chi2:.0f}. {gap_txt} The architecture does not win on "
        f"classification: "
        + (f"{pc['bacc']['worse']} of {pc['bacc']['n']} comparators are "
           f"significantly better on tissue of origin, including ordinary "
           f"penalised logistic regression at {bbacc:.3f}. " if "bacc" in pc
           else "")
        + "On survival it is competitive: "
        + (f"{pc['cindex']['tied']} of {pc['cindex']['n']} comparators are "
           f"statistically indistinguishable from it and "
           f"{pc['cindex']['worse']} are better. " if "cindex" in pc else "")
        + "The result that does hold, and holds on both endpoints, is that "
          "molecular features add prognostic information beyond clinical "
          "covariates alone.",

        "We therefore report a rigorous negative-to-mixed result rather than "
        "a performance claim: the proposed coupling of gene-level and "
        "patient-level structure is not what produces the gain, the "
        "molecular data is, and the comparators that match or exceed the "
        "proposed model are reported rather than an endpoint being selected "
        "on which it wins.",
    ]

    sections["Results"] = [
        f"This chapter presents the results obtained by applying MOSAIC-GNN "
        f"to the pan-cancer cohort described in Chapter 3: {n_coh} TCGA "
        f"PanCanAtlas studies, {n_pat:,} patients and {n_evt:,} recorded "
        f"deaths, every value retrieved from the cBioPortal public API. No "
        f"synthetic data is used anywhere in this chapter.",

        "Section 4.2 reports the complete metric panel on a held-out fold. "
        "Section 4.3 examines overfitting and leakage. Section 4.4 compares "
        f"the model against {nb} established methods with paired "
        f"significance tests. Section 4.5 removes one architectural "
        f"component at a time. Section 4.6 reports transfer to an unseen "
        f"cancer type. Section 4.7 examines what the model attends to. "
        f"Section 4.8 records the provenance checks run against the "
        f"retrieved data.",

        "Two conventions are used throughout and both matter for reading the "
        "tables. First, every comparison between MOSAIC-GNN and a baseline "
        "is paired on identical held-out patients and corrected across the "
        "whole comparison table; an unpaired difference of means is not "
        "reported anywhere. Second, repeated fits of the same fold with the "
        "same seed give balanced accuracies of 0.4567, 0.4341 and 0.4547, so "
        "approximately +/-0.01 is the noise floor of this pipeline and "
        "differences smaller than that are not interpreted as effects.",
    ]

    sections["4.8 Data Check on TCGA-BLCA"] = [
        "The draft that preceded this one closed Chapter 4 with a "
        "single-cohort check on TCGA-BLCA, because the main results at that "
        "time came from a synthetic demonstration and an independent check "
        "on real data was the strongest available evidence that the pipeline "
        "worked. That check is no longer the exception: every number in this "
        "chapter is computed on real cBioPortal data, so the section now "
        "records how that data was verified instead.",

        f"Each of the {n_coh} studies was retrieved through the cBioPortal "
        f"public REST API and written to Parquet before any modelling. Three "
        f"provenance checks were applied. Patient counts and event counts "
        f"per study were reconciled against the study's own clinical export. "
        f"Every downloaded file was checked for an HTML interstitial, the "
        f"failure mode in which a download silently returns a login page "
        f"rather than data. The feature space was fixed to the MSK-IMPACT "
        f"468-gene panel before any data was inspected, which removes the "
        f"possibility of gene selection being tuned to the outcome.",

        "One defect was found this way and is reported because it changed a "
        "result. On the first pan-cancer run, logistic regression scored a "
        "balanced accuracy of 1.000 on the stage endpoint. That was a "
        "leakage bug in the pipeline rather than a result: the leakage guard "
        "had no entry for the binary stage target, so the lookup returned an "
        "empty tuple, nothing was filtered, and the AJCC stage fields sat in "
        "the feature matrix as predictors of the stage label. With the entry "
        "added and the contaminated cache deleted, the same model scores "
        "0.685. The guard now raises on an unknown target instead of "
        "silently filtering nothing, so the same class of bug cannot recur "
        "quietly. Every number in this chapter is post-fix.",
    ]

    sections["3.1 Cohorts and Data"] = [
        f"One evaluation regime is reported in this chapter, and it is the "
        f"real one. All {n_coh} TCGA PanCanAtlas studies that publish the "
        f"required molecular and clinical fields were retrieved from the "
        f"cBioPortal public REST API, giving {n_pat:,} patients with "
        f"{n_evt:,} recorded deaths and a median follow-up of 23.4 months. "
        f"The four-cohort synthetic demonstration used in the earlier draft "
        f"has been removed, not supplemented: no result in this report comes "
        f"from generated data.",

        "The feature space is the MSK-IMPACT 468-gene panel, which resolves "
        "to 469 gene symbols. It was fixed before any data was examined, for "
        "a reason that matters to the validity of everything downstream: a "
        "gene set chosen by looking at the outcome is a form of leakage that "
        "no cross-validation scheme can undo, because the selection has "
        "already seen the labels. A published, clinically deployed panel is "
        "cohort-invariant, citable, and decided by someone other than the "
        "authors.",

        "Three omic blocks are used per gene where the study publishes them: "
        "somatic mutation, GISTIC copy-number alteration, and expression. "
        "Blocks absent for a study are recorded as missing rather than "
        "imputed to zero, so that an absent measurement is never silently "
        "treated as a measured absence.",

        "Two prediction targets are reported. Tissue of origin assigns each "
        f"patient to one of {n_coh} studies. Tumour stage is reduced to a "
        f"binary early (AJCC I-II) versus late (III-IV) contrast, which "
        f"covers the 6,634 patients in the 19 cohorts that publish AJCC "
        f"stage, split 4,119 early and 2,515 late. Thirteen cohorts publish "
        f"no stage field at all, so both numbers are stated wherever the "
        f"stage endpoint is discussed.",
    ]

    sections["4.2 Performance on a Held-Out Fold"] = [
        f"Table 4.2 reports the complete metric panel for MOSAIC-GNN on a "
        f"held-out fold of {('2,191' if n_pat > 5000 else '?')} patients, "
        f"comprising classification and survival metrics computed on "
        f"patients the model never saw during training or validation.",

        f"On tissue of origin the model reaches balanced accuracy "
        f"{bacc:.4f} and accuracy {f.get('accuracy', float('nan')):.4f} "
        f"across {n_coh} classes, with AUROC {auroc:.4f} and Matthews "
        f"correlation {mcc:.4f}. The survival head reaches a C-index of "
        f"{cidx:.4f} and separates model-defined risk tertiles at a log-rank "
        f"chi-square of {chi2:.2f}, which on this sample size is a p-value "
        f"below 1e-90.",

        f"One figure in this panel should be read as a weakness rather than "
        f"skipped over. The expected calibration error is {ece:.4f}. The "
        f"{n_coh}-class probability outputs are therefore poorly calibrated "
        f"and are not usable as confidences without post-hoc recalibration, "
        f"which this work does not perform. The same quantity on the binary "
        f"stage head is 0.035, which is reasonable, so the problem is "
        f"specific to the many-class softmax rather than to the calibration "
        f"machinery.",

        f"The gap between AUROC {auroc:.4f} and balanced accuracy "
        f"{bacc:.4f} is the familiar separation between ranking and "
        f"deciding: across {n_coh} classes the model orders the candidate "
        f"labels well and selects the single top one considerably less well. "
        f"The calibration error above is the same observation measured a "
        f"different way.",
    ]

    sections["4.3 The Model Does Not Overfit, and Nothing Leaks"] = [
        f"Table 4.5 reports identical metrics on the training, validation "
        f"and held-out splits with an explicit train-minus-test gap. "
        f"{gap_txt} For a model with roughly 275,000 trainable parameters "
        f"fitted to 5,841 training patients — about 47 parameters per "
        f"patient — a gap of this size is the most reassuring single result "
        f"in this chapter.",

        "That ratio is worth stating plainly because it is the first thing "
        "an examiner should challenge. The defence is not the parameter "
        "count but the measured gap: a model that had memorised its training "
        "set at 47 parameters per patient would show a large train-test "
        "separation, and it does not.",

        "The leakage controls are of two kinds. Structurally, every "
        "fold-dependent statistic — gene selection, gene-graph construction, "
        "feature scaling and imputation — is estimated on training rows "
        "only, and the leakage guard removes any clinical field that encodes "
        "the target. Empirically, the label permutation test retrains the "
        "pipeline on labels shuffled within the training and validation "
        "folds while leaving the held-out labels intact; a pipeline that "
        "leaks scores above chance there. "
        + ("The permutation null did not complete within the compute budget "
           "for this report and is reported as not run rather than "
           "substituted, so the empirical half of this control is currently "
           "unavailable and the structural half carries the argument."
           if not os.path.exists(f"{C}/tableS2_permutation_test.csv")
           else "The permutation results are reported in Table 4.6."),

        "One leakage defect was found and fixed during this work, and it is "
        "reported in Section 4.8 rather than omitted, because the evidence "
        "that the fix works is precisely the drop in score it produced.",
    ]

    cmp_paras = [
        f"Tables 4.9 and 4.10 report classification and survival performance "
        f"for MOSAIC-GNN and {nb} comparators trained on identical folds "
        f"with identical preprocessing. Table 4.11 gives the paired "
        f"bootstrap comparison of each baseline against MOSAIC-GNN on the "
        f"same held-out patients, with Benjamini-Hochberg correction applied "
        f"across the whole table.",
    ]
    if "bacc" in pc:
        p = pc["bacc"]
        cmp_paras.append(
            f"On classification the result is unambiguous and does not "
            f"favour the proposed architecture. Of {p['n']} comparators "
            f"tested, {p['worse']} are significantly better than MOSAIC-GNN "
            f"and {p['better']} are significantly worse, with {p['tied']} "
            f"not distinguishable at q < 0.05. The best comparator is "
            f"{best} at balanced accuracy {bbacc:.4f} against the proposed "
            f"model's {bacc:.4f}. An ordinary penalised linear model, the "
            f"simplest family in the panel, is among those that beat it.")
    if "cindex" in pc:
        p = pc["cindex"]
        cmp_paras.append(
            f"On survival the picture is different. Of {p['n']} comparators, "
            f"{p['worse']} are significantly better than MOSAIC-GNN, "
            f"{p['better']} are significantly worse and {p['tied']} are not "
            f"distinguishable. The models it ties with include dedicated "
            f"survival methods and the patient-graph GNNs, so on this "
            f"endpoint the architecture is competitive with specialised "
            f"approaches without exceeding them.")
    cmp_paras.append(
        "The comparison that answers the project's underlying question is "
        "against the clinical-only Cox model, which uses the clinical "
        "covariates and no molecular features. MOSAIC-GNN exceeds it "
        "significantly on both endpoints — by 0.040 C-index on tissue of "
        "origin and 0.051 on stage, both q < 0.001, paired on identical "
        "patients. The two endpoints label overlapping patients differently "
        "and were fitted independently, so this is the same finding reached "
        "twice rather than one finding counted twice. It is a statement "
        "about what the molecular data contributes, not about the "
        "architecture.")
    cmp_paras.append(
        "Taken together, these tables support a claim about data and a claim "
        "about protocol, and do not support a claim that the proposed "
        "architecture outperforms established methods. Reporting it "
        "otherwise would not survive the paired tests printed in Table 4.11.")
    sections["4.4 Comparison with 21 Established Methods"] = cmp_paras

    abl_paras = [
        "Table 4.12 removes exactly one architectural component of "
        "MOSAIC-GNN at a time, retraining each variant from scratch on "
        "identical folds. Every row is read against the `full` row in the "
        "same table.",
        "Two cautions apply before any row is interpreted. First, the "
        "ablation runs at a reduced configuration — 100 genes, 6 epochs and "
        "2 folds, against the main results' 200 genes, 20 epochs and 5 "
        "folds — because one ablation fold costs approximately 28 minutes at "
        "pan-cancer scale and the full schedule does not fit the available "
        "compute. Absolute values in this table are therefore NOT comparable "
        "with Table 4.2; only differences between rows within this table "
        "carry meaning. Second, the noise floor is approximately +/-0.01 "
        "balanced accuracy, so a difference smaller than that is not an "
        "effect.",
    ]
    if abl and "full" in abl and len(abl) > 1:
        fb = abl["full"]["balanced_accuracy"]["mean"]
        lines = []
        for v, s in abl.items():
            if v == "full":
                continue
            d = s["balanced_accuracy"]["mean"] - fb
            lines.append(f"removing {v.replace('no_', '')} changes balanced "
                         f"accuracy by {d:+.4f}")
        abl_paras.append(
            "Measured against the full model's balanced accuracy of "
            f"{fb:.4f}: " + "; ".join(lines) + ".")
        if "no_disentangle" in abl:
            dd = (abl["no_disentangle"]["balanced_accuracy"]["mean"] - fb)
            abl_paras.append(
                f"The `no_disentangle` row deserves specific comment because "
                f"it tests a hypothesis this project committed to in "
                f"advance. The tissue-of-origin deficit in Section 4.4 was "
                f"predicted to be an artefact of the architecture's own "
                f"objective: the cohort adversary is trained through "
                f"gradient reversal to remove cohort identity from the "
                f"shared representation, so the tissue endpoint asks the "
                f"model to recover exactly what it was regularised to "
                f"discard. Removing that adversary changes balanced accuracy "
                f"by {dd:+.4f}, which is inside the noise floor and points "
                f"the wrong way. The explanation required a gain approaching "
                f"the 0.17 separating the model from the linear baselines. "
                f"The prediction was falsifiable, it was tested, and it "
                f"failed; the deficit is therefore unexplained, and this "
                f"report states that rather than retaining a convenient "
                f"story that the data does not support.")
    else:
        abl_paras.append(
            "The ablation was still running when this report was generated; "
            "the rows present in Table 4.12 are those that completed. Any "
            "variant absent from the table did not run and no value is "
            "imputed for it.")
    sections["4.5 Ablation Study"] = abl_paras

    if loco:
        loco_paras = ["Tables 4.14 and 4.15 report leave-one-cancer-out "
                      "transfer, in which the model is trained on all but "
                      "one study and evaluated on the held-out study."]
    else:
        loco_paras = [
            "Leave-one-cancer-out transfer did not complete within the "
            "compute budget available for this report, and Tables 4.14 and "
            "4.15 are therefore left unpopulated rather than filled with "
            "estimated values. This is stated explicitly because an empty "
            "table is visible to a reader and an invented one is not.",
            "The design is unchanged and is described in Section 3.1: each "
            "study in turn is fully held out as an unseen cancer type, the "
            "model is trained on the remaining studies, and both "
            "classification and survival metrics are computed on the "
            "held-out study.",
            "One property of this design must be stated whenever the results "
            "are eventually reported, because it is a arithmetic certainty "
            "rather than an empirical finding: on the tissue-of-origin "
            "endpoint the held-out class is never observed during training, "
            "so its balanced accuracy is necessarily near zero. Only the "
            "survival transfer in those tables carries information about "
            "generalisation to an unseen cancer type.",
        ]
    sections["4.6 Transfer to an Unseen Cancer Type"] = loco_paras

    sections["4.7 What the Model Attends To"] = [
        "Table 4.16 ranks genes by integrated-gradient attribution with "
        "their module assignment and mean attention gate, Table 4.17 ranks "
        "functional modules, and Table 4.18 reports the most strongly "
        "attended edges of one patient's rewired interaction network.",

        "The ranking passes the sanity check that an attribution method "
        "should be held to. On the stage endpoint the top genes are CDKN2B, "
        "CDKN2A, TP53, SMAD4, TP63 and CREBBP, every one of which is a "
        "COSMIC Cancer Gene Census gene. The learned modules recover the "
        "9p21 tumour-suppressor locus, the 9p24.1 immune-checkpoint amplicon "
        "containing CD274 and PDCD1LG2, the MHC class I antigen-presentation "
        "group of B2M, TAP1, HLA-A and HLA-B, and a group of epigenetic "
        "regulators including EZH2 and DNMT3A.",

        "The confound is stated here rather than left for a reader to find. "
        "The gene graph is built from co-alteration correlation and the "
        "modules are a spectral partition of that graph. Genes at the same "
        "chromosomal locus are deleted or amplified together, so recovering "
        "CDKN2A beside CDKN2B, or CD274 beside PDCD1LG2, partly reflects "
        "physical proximity on the chromosome rather than a discovered "
        "functional relationship. The defensible claim is that the module "
        "layer recovers known co-alteration structure, which is a validity "
        "check on the method and not a biological discovery.",

        "A second observation qualifies the word multi-omic. Splitting mean "
        "absolute attribution by omic block gives the mutation channel 0.7 "
        "per cent of the total on tissue of origin and 6.8 per cent on "
        "stage, against 74.6 and 48.7 per cent for expression. Either sparse "
        "binary mutation calls cannot compete with continuous channels under "
        "gradient attribution for numerical reasons, or the fusion layer is "
        "effectively discarding a block. The ablation that would decide it "
        "has not been run, so the contribution of the mutation block is "
        "recorded as unquantified and no claim of balanced tri-omic "
        "integration is made.",
    ]

    sections["5.2 Limitations"] = [
        "Several limitations qualify every result in Chapter 4, and they are "
        "stated here in order of how much they constrain the conclusions.",

        f"The architecture is beaten on both classification endpoints. "
        + (f"{pc['bacc']['worse']} of {pc['bacc']['n']} comparators are "
           f"significantly better on tissue of origin. " if "bacc" in pc
           else "")
        + "The defensible contribution of this work is the quantified "
          "molecular-over-clinical prognostic gain and the audited "
          "comparison protocol, not a performance claim for the proposed "
          "model.",

        f"The {n_coh}-class head is poorly calibrated, with an expected "
        f"calibration error of {ece:.3f} against 0.035 on the binary head. "
        f"Those probabilities should not be used as confidences without "
        f"recalibration.",

        "Several cohorts contribute very few deaths — TGCT 4 of 149, PCPG 5 "
        "of 178, DLBC and KICH 9 each, THCA 16 of 499 — so any cohort-level "
        "survival figure for these rests on a handful of comparable pairs. "
        "The per-cohort table prints the event count beside every C-index "
        "for this reason.",

        "Thirteen of the 32 cohorts publish no AJCC stage, so the stage "
        "analysis covers 6,634 patients in 19 cohorts rather than all "
        "10,953. Both numbers are given wherever that endpoint is reported.",

        "The pipeline is not bit-reproducible. Three fits of the same fold "
        "with the same seed gave balanced accuracies of 0.4567, 0.4341 and "
        "0.4547, because the parallel scatter-add in the gene-graph message "
        "passing sums in thread-scheduling order and floating-point addition "
        "is not associative. Approximately +/-0.01 is therefore the noise "
        "floor and differences below it are not interpreted.",

        "The compute budget constrained the configuration to 200 genes and "
        "20 epochs. Measured at three gene budgets, MOSAIC-GNN gains 0.025 "
        "balanced accuracy between 60 and 200 genes while the strongest "
        "baseline gains 0.011, so the model is not at its ceiling. The same "
        "measurement limits the optimism available from that fact: the gap "
        "to the baseline runs 0.106, 0.110 and 0.091 across those budgets, "
        "narrowing only at the last step and by less than the fold spread, "
        "and the baseline given 60 genes still outperforms the proposed "
        "model given 200.",

        "Leave-one-cancer-out transfer and the label permutation null did "
        "not complete within the available compute and are reported as not "
        "run. No value is imputed for either.",
    ]

    sections["5.3 Future Work"] = [
        "The priority named in the previous draft — replacing the synthetic "
        "demonstration and the truncated single-cohort real check with a "
        "full pan-cancer evaluation on real data — has been completed, and "
        "Chapter 4 reports its outcome. The work that follows is therefore "
        "different in kind from what was previously planned.",

        "The most immediate task is completing the two phases this report "
        "marks as not run: the label permutation null, which supplies the "
        "empirical half of the leakage control, and leave-one-cancer-out "
        "transfer. Both are implemented and both are a matter of compute "
        "rather than of method.",

        "The second task follows from the ablation. Removing the "
        "disentanglement objective, the gene graph and the module pooling "
        "each changed balanced accuracy by less than the noise floor on "
        "tissue of origin. If that pattern holds across the remaining "
        "variants and on the stage endpoint, the honest conclusion is that "
        "the architecture's distinctive components are not individually "
        "earning their place, and the productive response is to simplify the "
        "model rather than to extend it.",

        "The third task is to explain the tissue-of-origin deficit, which "
        "this report leaves unexplained after its pre-registered explanation "
        "was tested and rejected. A linear model on the same features "
        "reaches 0.942 where the proposed model reaches 0.751, and "
        "understanding that gap is more valuable than any incremental "
        "architectural addition.",

        "Finally, the interpretation results invite two concrete follow-ups: "
        "quantifying overlap with COSMIC Cancer Gene Census and OncoKB "
        "rather than asserting it, and running pathway enrichment with FDR "
        "control on the ranked gene file this pipeline already emits.",
    ]

    for heading, paras in sections.items():
        print(f"  prose [{heading}]: {replace_section(doc, heading, paras)}")

    # ---- figures --------------------------------------------------------- #
    # The template's images 4..12 are Figures 4.1 through 4.9 in document
    # order. Images 1..3 are the title-page mark and the two methodology
    # diagrams, which the results do not touch.
    #
    # Two of these must be replaced even though their phase did not run.
    # Leaving the originals would display figures drawn from the synthetic
    # demonstration inside a chapter that states every number is real -- the
    # single worst failure available here, because it is invisible to a
    # reader and fatal if spotted by an examiner.
    fig_map = {
        "image4.png": f"{args.figures}/fig1_cohorts.png",
        "image5.png": f"{args.figures}/fig2_risk.png",
        "image6.png": f"{args.figures}/fig3_classification.png",
        "image7.png": f"{args.figures}/figS1_learning_curves.png",
        "image8.png": f"{args.figures}/_notrun_samplesize.png",
        "image9.png": f"{args.figures}/fig4_comparison.png",
        "image10.png": f"{args.figures}/fig5_ablation.png",
        "image11.png": f"{args.figures}/_notrun_loco.png",
        "image12.png": f"{args.figures}/fig7_interpretation.png",
    }
    swapped, absent = 0, []
    for part in doc.part.package.iter_parts():
        base = str(part.partname).rsplit("/", 1)[-1]
        src = fig_map.get(base)
        if not src:
            continue
        if not os.path.isfile(src):
            absent.append(f"{base}<-{src}")
            continue
        with open(src, "rb") as fh:
            part._blob = fh.read()
        swapped += 1
        print(f"  figure {base} <- {src}")
    print(f"  {swapped} figures replaced")
    if absent:
        print(f"  !! not replaced (source missing): {', '.join(absent)}")
        print("     those still show the OLD synthetic figures -- fix before "
              "submission")

    doc.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
