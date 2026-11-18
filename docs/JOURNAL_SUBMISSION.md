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

### Those tables now exist, so here is the answer

This section used to end with "if MOSAIC-GNN does not beat the best baseline,
do not write outperforms". Both runs have since completed and the question is
settled. **MOSAIC-GNN does not win on classification on either target.** Do
not write "outperforms". The numbers, from `table4_paired_tests.csv`:

| | Baselines significantly better | MOSAIC-GNN better | Not distinguishable |
|---|---|---|---|
| `cohort`, balanced accuracy | **18 of 20** | 2 | 0 |
| `stage_binary`, balanced accuracy | **8 of 20** | 2 | 10 |
| `cohort`, c-index | **0 of 10** | 2 | 8 |
| `stage_binary`, c-index | 1 of 10 | 1 | 8 |

Read the two halves of that table differently. On classification the
architecture loses, decisively on tissue of origin. On survival it is never
significantly beaten on `cohort` and loses only to `cox_lasso` on stage. The
survival side is where it is defensible.

### The claims the tables support

* **Structured multi-omics adds information over clinical covariates — and
  this replicates.** MOSAIC-GNN significantly exceeds `cox_clinical_only` on
  **both** targets: +0.040 on `cohort` and +0.051 on `stage_binary`, both
  q < 0.001, paired on identical held-out patients. Penalised Cox shows the
  same gap independently (0.779 against 0.734). A clinical-only model is the
  baseline many multi-omics papers never report, and clearing it twice on
  independently fitted targets is the strongest result in this work. **Lead
  with this.**
* **A saturation and difficulty result.** 32-class tissue of origin is
  near-ceiling for *conventional classifiers* (0.942 for plain logistic
  regression), while early-vs-late stage from the same molecular profiles is
  hard (0.685 for the best baseline). That contrast is a finding about what
  bulk multi-omics encodes.

  Word this carefully. An earlier draft said tissue of origin "cannot
  separate architectures". That is false and the tables disprove it —
  MOSAIC-GNN sits 19 points below logistic regression there, which is a very
  clear separation. What is true is that the *conventional* classifiers are
  bunched within a hundredth of each other, so the endpoint cannot rank
  **them**.
* **Informative predictive uncertainty**, offered as a secondary claim with
  its limits attached. Selective prediction under MC dropout lifts accuracy
  from 0.58 at full coverage to about 0.68 at 35–40% coverage. No baseline in
  the panel emits calibrated uncertainty, so no paired test covers this. It
  was measured on the reduced notebook fit only, has not been reproduced at
  manuscript settings, and has not been compared against an
  uncertainty-equipped baseline such as a deep ensemble. Claim it at exactly
  that strength or leave it out.

### Two things a reviewer will find if you do not declare them

* **The cohort loss is unexplained, and the obvious explanation has been
  tested and rejected.** The natural story is that the architecture's cohort
  adversary, trained through gradient reversal to strip cohort identity from
  the shared representation, makes the `cohort` endpoint ask for exactly what
  the model was regularised to discard. We committed to that explanation,
  named `no_disentangle` as its test, and ran it: removing the adversary
  moved balanced accuracy by **−0.0068** (0.5772 → 0.5705 over two folds),
  inside the ±0.01 noise floor and in the wrong direction. The explanation
  required something near the 0.17 gap to the linear baselines.

  Write it this way round. A pre-registered hypothesis that was tested and
  abandoned is a strength in a defence; the same hypothesis asserted without
  the test is the sentence an examiner will attack, and rightly, because it
  reads as an excuse for a bad result. Do not reintroduce the adversary story
  as the reason for the cohort deficit — it is not supported.
* **The mutation block contributes almost nothing.** Mean absolute integrated
  gradients split by omic block give mutation 0.7% on `cohort` and 6.8% on
  `stage_binary`, against 74.6% and 48.7% for expression. Either sparse
  binary calls cannot compete with continuous channels in gradient
  attribution, or the fusion layer is discarding a block. Do not claim
  balanced tri-omic integration on this evidence.

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
| Exact settings for every number? | `runs_pan_*/<tag>_config.json` for the manuscript run; `<tag>_<command>_config.json` for each later phase (ablate, loco) |
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
8. **The 32-class head is badly calibrated.** ECE 0.273 on `cohort` against
   0.035 on `stage_binary`, with mean predictive entropy 1.896 nats. Those
   probabilities are not usable as confidences without post-hoc
   recalibration, which this work does not perform. Declaring this is
   especially necessary because the manuscript claims informative
   uncertainty on the binary head — a reviewer who finds good calibration
   claimed in one place and ECE 0.273 in a table elsewhere will distrust
   both.
9. **The architecture loses on both classification endpoints.** 18 of 20
   comparators are significantly better on `cohort` and 8 of 20 on
   `stage_binary`. Put this in the abstract, not only in a table. A reviewer
   who discovers it after reading a positive abstract will reject on
   framing even if every number is correct.

---

## 6. Figures worth the space

> **Before you build any figure, read `runs/README.md`.** The `runs/`
> directory holds synthetic smoke-test output, and five of its tracked PNGs
> carry exactly the filenames `build_manuscript.py` reads — `fig1_cohorts`,
> `fig2_risk`, `fig3_classification`, `figS1_learning_curves`,
> `figS2_sample_size`. Building with `--runs runs` yields a manuscript whose
> prose quotes real results and whose figures come from invented data, with
> nothing in the output flagging it. Build with `--runs runs_pan_cohort`.
>
> The pan-cancer runs produced **no** figures, so the current draft marks
> every one as not run. That is the correct state: a missing figure is
> visible, a wrong figure is not. The only figures from real data are in
> `deliverables/notebook_figures/`, and they come from the reduced notebook
> configuration — not quotable as manuscript results.

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
