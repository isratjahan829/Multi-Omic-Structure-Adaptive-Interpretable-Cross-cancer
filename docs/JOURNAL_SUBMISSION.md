# Preparing this work for a Q1 journal

This is a practical checklist, written against what the repository actually
contains. It is ordered by what a reviewer reaches for first.

Everything below refers to real artefacts: `runs_pan_cohort/`,
`runs_pan_stage/`, `data_pan/`, `scripts/`, and the tables
`scripts/export_tables.py` writes. Where a table has not been produced, the
report says so, and this document does not pretend otherwise.

---

## 1. The claim to make, and the claim not to make

The single most common way a manuscript like this fails review is claiming
that the proposed architecture wins, when the comparison does not support it.
Decide the claim from the tables, in this order:

1. Open `runs_pan_*/table3_comparison_full.csv`. Find the best baseline for
   each metric.
2. Open `runs_pan_*/table4_paired_tests.csv`. This holds the paired bootstrap
   difference between MOSAIC-GNN and each baseline on the *same* patients,
   with Benjamini-Hochberg corrected q-values.
3. Make the claim the `*_sig` columns support, and no stronger.

If MOSAIC-GNN does not beat the best baseline at q < 0.05, do **not** write
"outperforms". Write what is true. Two honest framings are available and both
are publishable:

* **Structured multi-omics adds information over clinical covariates.** On
  the pan-cancer data, penalised Cox on molecular + clinical features reached
  c-index 0.779 against 0.734 for `cox_clinical_only` — a clinical-only
  model is the baseline many multi-omics papers never report, and beating it
  is a real result. State the gap and its uncertainty.
* **A saturation and difficulty result.** 32-class tissue of origin is
  near-ceiling for every reasonable method (0.942 for plain logistic
  regression), while early-vs-late stage from the same molecular profiles is
  hard (0.685 for the best baseline). That contrast is a finding about what
  bulk multi-omics encodes, and it is worth stating plainly rather than
  burying.

A benchmark paper that reports a negative or null result rigorously is
reviewable. A paper that claims a win the statistics do not support is
rejected, and at a thesis defence it is worse than rejected.

---

## 2. Reproducibility: what reviewers will ask for

| Reviewer asks | Where it is |
|---|---|
| Where did the data come from? | `scripts/fetch_cbioportal_api.py` — cBioPortal public API, every TCGA PanCanAtlas study |
| Which patients? | `runs_pan_*/table1_cohorts.csv` — per-cohort n, events, event rate, median follow-up |
| Which genes, and why those? | MSK-IMPACT 468 panel, fixed before any data was seen; `data_pan/panel_genes.json` lists them |
| How were folds made? | `mosaic/data/preprocess.py:stratification_key` — stratified by cohort × class × event |
| Exact settings for every number? | `runs_pan_*/<tag>_config.json`, written beside every result |
| Is there leakage? | `runs_pan_*/tableS2_permutation_test.csv` (label permutation), plus `LEAKY_FOR_TARGET` in `mosaic/data/preprocess.py` |
| Can I run it? | `scripts/run_pan.sh` (whole schedule), `notebooks/mosaic_gnn.ipynb` (executed, with outputs) |

Put the commit hash of the run in the manuscript. Every result file sits
next to the config that produced it, so a reader can match a number to a
command.

---

## 3. The feature-space decision — lead with this

Your earlier pipeline selected the highest-variance genes *separately inside
each cohort*. A reviewer who notices will raise two objections, and both are
fatal:

* the "pan-cancer" framing is undermined, because each cohort has different
  features;
* leave-one-cancer-out transfer is incoherent, because the held-out cohort's
  features were never the ones the model trained on.

The current pipeline fixes the feature space to a curated, clinically
deployed panel across all 32 cohorts. Say so in the Methods, and say why:
identical features everywhere, chosen before any data was seen, citable, and
no selection-bias objection. This is a methodological strength — do not let
it read as an arbitrary choice.

---

## 4. Statistics the manuscript must contain

All of these exist in the pipeline. Use them.

* **Uncertainty on every headline number.** `table2_single_fold.csv` carries
  bootstrap confidence intervals; the cross-validation summary in
  `<tag>_results.json` carries mean ± SD with the fold count `n`. Never
  report a bare point estimate.
* **Paired tests, corrected.** `table4_paired_tests.csv` — paired bootstrap on
  the same patients, Benjamini-Hochberg across the whole table. Unpaired
  comparisons of means will be challenged.
* **A leakage control.** The label permutation test retrains on shuffled
  labels with the held-out labels intact. A pipeline that leaks scores above
  chance there. Report the null and the permutation p-value, and note that
  the smallest attainable p is 1/(B+1) for B permutations.
* **Calibration, not just discrimination.** ECE is in the metric panel. A
  clinical-facing classifier with good AUROC and bad calibration is a known
  reviewer target.
* **The fold count, stated.** Where the baseline panel ran on fewer folds
  than the model, say which and how many. `n` is recorded per metric in the
  JSON for exactly this reason.

---

## 5. Limitations to write yourself, before a reviewer writes them

Being first to state these is worth more than hoping they go unnoticed.

1. **One cohort is a single histology.** The study loaded as `LUAD` is lung
   adenocarcinoma only, not a combined lung cohort. Name it precisely.
2. **THCA contributes very few deaths** (16 of 499). Any cohort-level
   survival figure for it rests on a handful of comparable pairs.
3. **Thirteen of 32 cohorts publish no AJCC stage**, so the stage analysis
   covers 6 634 patients across 19 cohorts, not all 10 953. State both
   numbers.
4. **Leave-one-cancer-out on the tissue target is degenerate by
   construction**: the held-out class is never seen in training, so its
   balanced accuracy is near zero necessarily. Only the survival transfer in
   that table carries information. Say this, or a reviewer will think you
   did not notice.
5. **Mini-batching deviates from the configured default.** Full-batch over
   the training set does not fit in 16 GB at this gene count, so training is
   mini-batched. That changes the optimisation trajectory; it is a real
   deviation, not a cosmetic one.
6. **The pipeline is not bit-reproducible.** Three fits of the same fold with
   the same seed gave balanced accuracy 0.4567, 0.4341 and 0.4547 — the
   parallel scatter-add in the gene-graph message passing sums in
   thread-scheduling order and float addition is not associative. Report the
   spread and treat differences smaller than it as noise.
7. **Compute ceiling.** State the gene budget and epoch count actually used
   and why, with the measured cost. Being explicit that 1500 genes × 120
   epochs was out of reach is far better than letting a reviewer wonder why
   the configuration looks modest.

---

## 6. Figures worth the space

In rough order of how much they earn:

1. **The comparison panel** — MOSAIC-GNN against every baseline, one metric
   per facet, with confidence intervals. From `table3_comparison_full.csv`.
2. **The two targets side by side** — tissue of origin near ceiling, stage
   near the noise floor. This is the paper's argument in one figure.
3. **The ablation** — `table5_ablation.csv`, which components earn their
   place. A reviewer reads this to decide whether the architecture is
   justified or decorative.
4. **Calibration curves** per target.
5. **Interpretation** — `table7_gene_importance.csv` (integrated gradients)
   and `table8_module_importance.csv`. Check the top genes against known
   biology for the dominant cohorts and say what you find, including if it is
   unremarkable.
6. **The permutation null** as a supplementary panel. It is the most
   convincing evidence that the pipeline is sound.

Build charts from the CSVs, not by retyping numbers.

---

## 7. Before you submit

- [ ] Every number in the manuscript traced to a file in `runs_pan_*/`
- [ ] No table filled in from a run that did not complete
- [ ] Fold counts stated wherever they differ between methods
- [ ] `random_survival_forest` reported as not run, with the reason
      (OOM-killed at this scale), not quietly dropped from the panel
- [ ] The `stage_binary` leakage fix described in Methods, with the before
      and after (1.000 → 0.685). This is a strength: it shows the pipeline
      was audited.
- [ ] Commit hash and `scripts/run_pan.sh` settings in the Methods
- [ ] Data availability: cBioPortal study IDs, panel id, and the fetch script
- [ ] Code availability: repository URL and commit
- [ ] A reader could regenerate every table with two commands
      (`scripts/run_pan.sh`, then `scripts/make_report.py`)

---

## 8. Where the old numbers came from

If anyone asks why earlier figures differed — and at a defence they will —
the answer is documented. The notebook's configuration cell tested for
`*_merged_full.csv` while the merge writes `*_merged_full.parquet`, so it
never found the real data and fell back to 160-patient synthetic data, while
still printing tables that looked real. That is where a balanced accuracy of
0.313 came from. It is fixed, and the fallback now prints a banner saying
every number below it is synthetic and may not be quoted.

Thirteen further defects were found and fixed during this work; the Methods
should say the pipeline was audited and the appendix can list them. Several
were silent-wrongness bugs rather than crashes — a baseline fitted without
class weights, per-cohort feature selection, an unguarded survival family
that could discard the whole comparison table. Finding and stating them is
the strongest possible answer to "how do we know these numbers are right?".
