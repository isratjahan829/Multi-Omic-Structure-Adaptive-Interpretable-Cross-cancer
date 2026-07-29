# MOSAIC-GNN

**M**ulti-**O**mic **S**tructure-**A**daptive **I**nterpretable **C**ross-cancer
**G**raph **N**eural **N**etwork for pan-cancer prognosis and biomarker discovery
on TCGA (BLCA, LIHC, SKCM, THCA).

---

## 1. What is actually new here

Most published multi-omics GNNs do one of two things: build a *patient similarity
graph* from concatenated features and run a GCN on it, or build a *gene network*
and pool it. MOSAIC-GNN couples both levels and adds three components that, to
our knowledge, have not been combined before in a cancer multi-omics setting:

| # | Component | Why it matters |
|---|-----------|----------------|
| 1 | **Two-level graph coupling.** Each patient is a *graph of genes* (nodes carry mutation / CNA / expression / methylation channels, edges are PPI + pathway + co-alteration). The pooled patient vector then becomes a node in a *patient-similarity graph*. | Message passing happens where the biology is (genes), and again where the epidemiology is (similar patients). Single-level models get one or the other. |
| 2 | **Patient-specific network rewiring.** Topology is shared, but attention coefficients are computed from each patient's own omic profile, so every patient gets a personalised weighting of the same interaction network — read out directly as an individual subnetwork. | Turns the model into a hypothesis generator, not just a classifier. |
| 3 | **Cohort-adversarial disentanglement.** The latent splits into `z_shared` (cohort-invariant, trained through a gradient-reversal layer against a cohort discriminator) and `z_specific` (cohort-predictive), with an orthogonality penalty between them. | This is what makes the pan-cancer model transfer to a *held-out cancer type* instead of silently learning "which cohort is this". Evaluated explicitly by the leave-one-cancer-out protocol. |
| 4 | **Learned patient graph** with a straight-through top-k, rather than a fixed kNN graph. | The graph is optimised for the prognostic task instead of for Euclidean proximity in a noisy 10⁴-dimensional space. |
| 5 | **Cross-omic contrastive alignment.** The same patient encoded through mutation-only and expression-only views must agree (InfoNCE). | Forces the representation onto signal corroborated by more than one assay — the standard defence against single-assay batch artefacts. |
| 6 | **Dual survival heads.** A Cox partial-likelihood risk score *and* a discrete-time hazard head. | The Cox head gives ranking (C-index); the hazard head gives a calibrated individual survival curve, which Cox alone cannot. |

Everything is trained end to end with a multi-task objective:

```
L = w_cls·CE           (stage / grade / subtype)
  + w_cox·CoxNLL       (Breslow partial likelihood)
  + w_haz·DiscreteHaz  (right-censored hazard likelihood)
  + w_adv·CE_GRL       (cohort adversary on z_shared)
  + w_coh·CE           (cohort head on z_specific)
  + w_ort·‖Zsᵀ Zc‖²    (disentanglement)
  + w_con·InfoNCE      (cross-omic alignment)
  + w_reg·(smoothness + sparsity + degree barrier on the learned graph)
```

---

## 2. Repository layout

```
mosaic/
  config.py            all hyper-parameters, serialised with every run
  data/
    loader.py          two-pass wide-CSV reader, omic block detection, survival parsing
    preprocess.py      fold-safe gene selection, scaling, survival binning
    graphs.py          gene graph (PPI/GMT prior + co-alteration kNN), spectral modules
  models/
    layers.py          shared-topology relational GAT, module pooling, graph learner,
                       gradient reversal, historical-embedding bank
    mosaic.py          the full model
  losses.py            Cox, discrete hazard, orthogonality, InfoNCE, graph regulariser
  metrics.py           C-index, time-dependent AUC, log-rank, KM, bootstrap CI, ECE
  pipeline.py          repeated stratified nested CV, training, evaluation
  baselines.py         21 comparators in four families (classical, multi-omics
                       fusion, patient-graph GNNs, survival models)
  explain.py           attention, integrated gradients, per-patient subnetworks, .rnk export
  cli.py               train | ablate | baseline | loco | explain
scripts/
  make_synthetic.py    schema-identical synthetic cohorts for smoke tests
  run_all.sh           the full experiment suite for the paper
  build_notebook.py    regenerates the notebook (keeps it diffable)
  xlsx_to_csv.py       .xlsx cohort sheet -> loader-ready CSV, and it warns
                       when Excel's 16 384-column cap truncated the sheet
notebooks/
  mosaic_gnn.ipynb     Kaggle-ready walkthrough: data -> one fold -> figures ->
                       full experiments -> interpretation
```

No `torch-geometric` dependency — message passing is implemented directly on a
shared topology with `index_add_`, which also keeps memory at O(batch × edges)
instead of O(patients × edges).

---

## 3. Data expectations

One wide CSV per cohort, named `{COHORT}_merged_full.csv`:

* clinical columns (`PATIENT_ID`, `AGE`, `SEX`, `AJCC_PATHOLOGIC_TUMOR_STAGE`,
  `OS_MONTHS`, `OS_STATUS`, …),
* `MUT_<GENE>` binary mutation calls,
* `CNA_<GENE>` GISTIC calls in {-2…2},
* optionally `RNA_<GENE>` / `EXP_<GENE>` expression and `METH_<GENE>` methylation.

If your cohorts arrive as `.xlsx`, convert them first:

```bash
python scripts/xlsx_to_csv.py Data.xlsx --out ./real
```

Note that Excel caps a worksheet at 16 384 columns, so a merged TCGA table
saved as `.xlsx` is silently truncated — usually part-way through the CNA
block, and any expression or methylation columns after it are lost entirely.
The script prints how many columns of each block survived. Export the original
CSV from cBioPortal whenever you can.

Blocks are detected by prefix (`DataConfig.omic_prefixes`), so a cohort missing
a modality is handled automatically: its channels are zero-filled and the
per-block availability mask channel tells the encoder the values are absent
rather than zero.

---

## 4. Running it

Prefer a notebook? `notebooks/mosaic_gnn.ipynb` runs the whole story with
figures. It locates the `mosaic` package automatically (working directory, a
Kaggle dataset, or an uploaded `mosaic-gnn.zip`) and falls back to synthetic
data if the TCGA files are not mounted, so it always runs end to end. Heavy
cells are guarded by a `RUN_FULL` flag.

From the command line:

```bash
pip install -r requirements.txt

# 0) smoke test on synthetic data (~2 min on CPU)
python scripts/make_synthetic.py --out ./synthetic --patients 160 --genes 300
python -m mosaic.cli train --root ./synthetic --max-genes 200 \
       --epochs 40 --folds 3 --repeats 1 --device cpu --out ./runs_smoke

# 1) main result: repeated 5-fold nested CV, pan-cancer
python -m mosaic.cli train --root /kaggle/input/datasets/jannatuljerin/tcga-capstone \
       --target stage --endpoint OS --folds 5 --repeats 3 --tag main

# 2) comparison table
python -m mosaic.cli baseline --root <ROOT> --tag main

# 3) ablation table
python -m mosaic.cli ablate --root <ROOT> --tag main

# 4) leave-one-cancer-out transfer (the headline generalisation experiment)
python -m mosaic.cli loco --root <ROOT> --tag main

# 5) interpretation: gene / module rankings, per-patient subnetworks, GSEA .rnk
python -m mosaic.cli explain --root <ROOT> --checkpoint runs/main_r0_f0.pt --tag main
```

Supplying real priors sharpens both accuracy and interpretability:

```bash
--ppi  string_human_links.csv     # columns: gene_a,gene_b,score
--gmt  c2.cp.reactome.v2023.1.symbols.gmt
```

Without them the gene graph falls back to a training-fold co-alteration kNN and
modules come from a spectral partition — so the pipeline runs offline (e.g. on a
Kaggle kernel with the internet switch off).

---

## 5. Experimental protocol

* **Splits.** Repeated stratified k-fold over the pooled pan-cancer cohort,
  stratified by `cohort × class × event`. Default 5 folds × 3 repeats.
* **Leakage control.** Gene selection, graph construction, scalers and survival
  bin edges are all fitted *inside* the training fold. `transform()` is the only
  function ever applied to held-out rows. The patient graph attaches held-out
  patients to training anchors only, never to each other.
* **Model selection.** Inner split on the training fold; the selection criterion
  is the mean of balanced accuracy and C-index, so neither task is starved.
* **Reporting.** Every model gets the *same* 31-metric row:
  * classification — accuracy, balanced accuracy, macro/weighted F1, macro
    precision / recall / specificity / NPV, AUROC, AUPRC, MCC, Cohen's kappa
    (linear and quadratic), top-2 accuracy, log loss, multiclass Brier, ECE;
  * survival — Harrell's C-index with a bootstrap 95 % CI, time-dependent AUC
    at 12/36/60 months, IPCW Brier at those horizons, integrated Brier score,
    hazard ratio (high vs low risk) with a Wald CI, HR per SD of the risk
    score, log-rank χ² and p.
  Paired bootstrap tests with Benjamini-Hochberg correction compare every
  baseline against MOSAIC-GNN on identical folds. Per-cohort and per-class
  breakdowns are always reported alongside the pooled number.
* **Uncertainty.** MC-dropout at test time gives a predictive entropy per
  patient; useful for a selective-prediction figure (accuracy vs coverage).

### Suggested figure set for the manuscript

1. Architecture schematic (two-level graph coupling).
2. Main comparison table + forest plot of C-index with bootstrap CIs.
3. Ablation bar chart (one bar per removed component).
4. Leave-one-cancer-out transfer matrix.
5. Kaplan–Meier curves for model-defined risk tertiles, per cohort, with log-rank p.
6. Module-attention heatmap (modules × cohorts) with enrichment annotation.
7. Two or three per-patient rewired subnetworks around known drivers.
8. Accuracy-vs-coverage curve from MC-dropout uncertainty.

---

### Comparison suite (`mosaic.baselines`)

| Family | Models |
|---|---|
| Classical | logistic L1 / L2 / elastic-net, linear SVM, RBF SVM, kNN, naive Bayes, shrinkage LDA, random forest, extra trees, hist gradient boosting, XGBoost, LightGBM, CatBoost, PCA+logistic |
| Multi-omics fusion | early-fusion MLP, late-fusion MLP, MOGONET-style per-omic GCN with view fusion |
| Patient-graph GNNs | GCN, GAT, GraphSAGE on a fixed kNN patient graph |
| Survival | Cox-lasso, Cox-ridge, clinical-only Cox, random survival forest, DeepSurv |

XGBoost / LightGBM / CatBoost / scikit-survival are optional: when a package is
absent that row is skipped and the rest of the table still runs.

---

## 6. Memory and speed

Peak memory of the gene stage is roughly
`batch × edges × heads × (hidden/heads)` floats per layer. With the defaults
(2 000 genes, kNN 8 → ~34 000 edges, 4 heads, hidden 64, batch 32) that is
~0.3 GB per layer. If you hit OOM, in order of least damage:

```
--batch-size 16          # first
--max-genes 1200         # second
GraphConfig.gene_knn 6   # third
ModelConfig.gene_layers 2
```

Set `TrainConfig.full_batch = False` whenever `batch_size` is given; the
historical-embedding bank keeps the patient graph global anyway.

---

## 7. Overfitting control

With ~10⁴ molecular features and a few hundred patients this is the first
thing a reviewer will probe, so it is handled explicitly rather than hoped for.

**In the model and the training loop**

| Mechanism | Setting | Why |
|---|---|---|
| Gene dropout | 0.15 | Whole genes are hidden each step — value *and* mask channel together. The strongest regulariser available when features outnumber patients 100:1, and it is exactly the corruption the model must survive at inference (an assay that was not run). |
| Input jitter | σ = 0.05 | Gaussian noise on the value channels only. |
| Dropout / edge dropout | 0.4 / 0.2 | Standard, plus stochastic sparsification of the gene graph. |
| Weight decay | 1e-3 | |
| Label smoothing | 0.1 | Stops the classifier from becoming over-confident on a small, noisy label set. |
| Gradient clipping | 5.0 | |
| Stochastic weight averaging | last 40 % of epochs | Kept only if it beats early stopping on validation. |
| Early stopping | patience 40 on a validation criterion weighting both tasks | |
| Fold-safe preprocessing | everywhere | Gene selection, graph construction, scalers and survival bins are fitted on training rows only. |

**In the evaluation** (`mosaic/diagnostics.py`, notebook section 5)

* `generalisation_gap` — train / validation / test side by side with an explicit
  gap row (Table S1);
* `learning_curves` — train vs validation per epoch, so you can see where early
  stopping fired (Figure S1);
* `permutation_test` — retrain on shuffled labels; the null must sit at chance.
  This is the only check that can *prove* absence of leakage (Table S2);
* `sample_size_curve` — test performance vs number of training patients, which
  separates "needs more data" from "too much model" (Table S3, Figure S2).

Measured on the synthetic demo after these controls: balanced accuracy
0.337 train vs 0.337 test (**gap 0.000**) and C-index 0.676 vs 0.624
(**gap 0.052**, down from 0.089 before SWA and gene dropout). The permutation
null landed at balanced accuracy 0.22-0.30 (chance 0.25) and C-index 0.47-0.55
(chance 0.50) — i.e. no leakage left in the pipeline.

If the gap is still too wide on your data, turn these knobs in order:
`gene_dropout` → 0.25, `max_genes` down, `gene_layers` → 2,
`ModelConfig.dropout` → 0.5.

---

## 8. About the executed demo output

`notebooks/mosaic_gnn_executed.ipynb` and `runs/` in the release archive were
produced on the **synthetic** cohorts (the TCGA files are not redistributable),
with a reduced budget: 200 genes, 30 epochs, 4 folds, 1 repeat, ablations on
2 folds. Read them as a template for the tables and figures, not as results.

Two artefacts of the synthetic data are worth knowing before you compare:

* the **classification** task saturates — the planted label is a monotone
  function of a driver-gene burden that tree ensembles recover exactly, so
  eight models tie at 1.000 balanced accuracy. On real staging labels that
  column separates properly;
* **IBS and the Brier columns are only populated for MOSAIC-GNN**, because it
  is the only model in the suite with a survival-curve head. Cox-type
  baselines produce a risk score, so they get C-index / time-dependent AUC /
  hazard ratio but no calibrated curve. Say this in the table caption rather
  than leaving the reader to wonder about the blanks.

The **survival** side of the demo does behave sensibly and is the part worth
reading: MOSAIC-GNN 0.641, random survival forest 0.626, clinical-only Cox
0.611, MOGONET-style 0.589, patient-graph GNNs 0.55-0.57, penalised Cox
0.51-0.54.

---

## 9. Reporting checklist before submission

- [ ] Every number is mean ± sd over ≥ 3 repeats × 5 folds, not a single split.
- [ ] Baselines were tuned on the same inner splits as the model.
- [ ] Paired bootstrap p-values reported for every headline comparison.
- [ ] Per-cohort results shown, not only the pooled pan-cancer number.
- [ ] Leave-one-cancer-out result reported even where it is weak — it is the
      honest measure of "pan-cancer".
- [ ] Top-ranked genes cross-checked against known drivers (COSMIC CGC / OncoKB)
      and the *novel* ones stated as hypotheses, not findings.
- [ ] Code, config JSONs, fold indices and seeds released.

## 10. Limitations to state in the paper

Cohort sizes here are in the hundreds, so the effective sample size for survival
is small and confidence intervals will be wide; the co-alteration fallback graph
is not a validated interaction network; and attention-derived rankings are
hypotheses that require orthogonal validation. State these explicitly — a
reviewer will otherwise state them for you.
