# MOSAIC-GNN on real TCGA data

Every figure below was produced by a command that ran to completion on this
machine. Nothing is estimated, interpolated or carried over from the earlier
synthetic-demo run; where a table is marked as unavailable, the run that
produces it did not finish and no substitute number has been put in its place.

## Data provenance

Raw cBioPortal study exports (clinical patient, clinical sample, mutations,
CNA, mRNA RSEM for each of four studies -- 20 files, 818 MB) were downloaded
and then checked individually: a Google Drive interstitial or quota page is
served with HTTP 200, so size alone proves nothing and each file was verified
to begin with a `#` metadata comment or a tab-separated header rather than
HTML (`scripts/verify_raw.py`). All 20 passed.

`scripts/merge_cbioportal.py` built one row per patient per cohort, keeping
the 3000 most-mutated genes (mutation block) and the 3000 highest-variance
genes (CNA and expression blocks). That pre-filter is unsupervised -- it never
consults a label or a fold -- and the model's own fold-wise selection then
narrows to 1500 genes using training rows only.

Two provenance notes that affect interpretation:

* The study loaded as `LUNG` carries `LUAD` in its own
  `CANCER_TYPE_ACRONYM` column for all of its patients, so it is lung
  **adenocarcinoma**, not a combined LUAD + LUSC cohort. It is labelled
  `LUNG` here because that is the cohort name the pipeline was asked to load.
* `THCA` contributes very few survival events, so its cohort-level C-index is
  estimated from a small number of comparable pairs and should be read with
  that in mind. The per-cohort table reports the event count alongside.

## What was run

`python -m mosaic.cli {train,baseline,loco,ablate}` with
`--max-genes 1500 --epochs 120 --folds 5 --repeats 1 --device cpu`, on both
the `cohort` (tissue of origin) and `stage` (four-class AJCC) targets, each
into its own output directory, followed by `scripts/export_tables.py` for the
diagnostics and the manuscript tables.

### Deviations from the requested settings

* **Mini-batching.** The configuration's default is `full_batch = True`. A
  full-batch step over ~740 training patients x 1500 genes x 3 relations does
  not fit in this machine's 16 GB, so training used `--batch-size 32`, the
  same batch size the project's notebook uses on CPU. This changes the
  optimisation trajectory and therefore the numbers; it is not a cosmetic
  choice and is recorded here for that reason.
* Any reduction in ablation folds, permutation count or diagnostic scope that
  was needed to fit the available compute is stated at the point where that
  table appears, with the setting actually used.

### Measured cost, and what it ruled out

One model fit at these settings measured **over 180 s/epoch** on this
four-core CPU container (idle machine, four threads, 1500 genes, 27682 gene
graph edges), i.e. roughly 4.5 to 6 hours per fit. The complete schedule is
about 64 fits per target -- five cross-validation folds, five baseline folds,
four leave-one-cancer-out folds, fifteen diagnostic retrains and forty
ablation folds -- or some 290 hours for the two targets. It therefore could
not be run, and the settings were kept rather than reduced, on the explicit
instruction that missing tables are preferable to altered ones.

Work was ordered so that the metrics named in the brief exist for both
targets as early as possible: the primary fold and its diagnostics first, then
the baseline panels, then the paired tests reusing the saved checkpoints, and
only then cross-validation, transfer, the permutation null and the ablation.
Each table below states whether it was produced; those that are absent are
absent because the run did not reach them, and no figure has been supplied in
their place.

### One comparison to make carefully

The single held-out-fold tables and the cross-validation summary do **not**
train on the same number of patients. The single-fold diagnostics split the
training set three ways (fit = 985, validation = 493), while `run_cv` splits
it two ways (fit = 739, validation = 739). The single-fold model therefore
sees about a third more training data than each cross-validation fold, so a
small gap between the two is expected and is not evidence of instability.

### Three defects fixed before anything could run

These were genuine faults in the committed code, not environment problems:

1. `mosaic.cli baseline` called `run_baseline_suite` without its required
   `block_names` argument and treated the returned `(metrics, predictions)`
   tuple as a metrics dict. The command could not execute at all.
2. The GAT patient-graph baseline scored attention through a dense
   `(N, N, 2H)` tensor of concatenated node pairs -- 3.50 GB per layer at
   N = 1848, H = 128, retained by autograd -- and was OOM-killed. Rewritten
   using the fact that a linear map of `[h_i ; h_j]` separates into a source
   and a destination term; verified identical to the original to float32
   precision (max absolute difference 2.4e-07), so no baseline number is
   affected by the change.
3. `logistic_l1` used `solver="liblinear"`, which cannot fit three or more
   classes. Both targets here are four-class, so it raised on every fold and
   was silently dropped from the comparison panel.

`merge_cbioportal.py` also crashed on every cohort (it inserted a
`CANCER_TYPE_ACRONYM` column the clinical export already provides), and
`diagnostics.learning_curves` read a `model._history` that nothing wrote.

## How to read the two targets

Tissue of origin and tumour stage are not two difficulties of one task; they
are different tasks, and the contrast between them is the substantive result.

Tissue of origin is close to trivially separable from bulk molecular profiles:
in a preliminary check at only 200 genes, ordinary logistic regression, LDA,
random forest, XGBoost and LightGBM all reached balanced accuracy 1.000.
A ceiling that high means the `cohort` target cannot discriminate between
methods -- every reasonable model saturates it -- so a tie there is the
expected outcome and is reported as a property of the benchmark rather than
as evidence for the architecture.

Four-class stage classification from the same molecular profiles is genuinely
hard, and is expected to stay modest. A modest stage number sitting beside a
near-perfect tissue number is a coherent finding about what bulk multi-omics
does and does not encode, and neither figure has been tuned toward the other.

Two structural caveats:

* **Leave-one-cancer-out on the `cohort` target is degenerate by
  construction.** Training on three cohorts and testing on the fourth means
  the held-out class was never observed, so its balanced accuracy is near
  zero necessarily, not as a finding. Only the survival transfer (C-index) in
  that table carries information.
* **The clinical feature block is not leakage-filtered for `cohort`.** The
  guard removes cancer-type-like column names, but stage and grade coding
  vocabularies differ sharply between these studies, so a classifier can read
  tissue partly from which clinical fields are populated. The `no_clinical`
  row of the ablation table is the direct measurement of how much of the
  tissue-of-origin result that explains, which is why it is worth reading
  before the headline number.
