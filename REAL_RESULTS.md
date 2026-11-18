# MOSAIC-GNN on pan-cancer TCGA data

Every figure in this report came out of a command that ran to completion on
this machine. Nothing is estimated, interpolated, or carried over from an
earlier run. Where a table is marked unavailable, the phase that produces it
did not finish, and no substitute number has been put in its place. A reader
who finds a number here can find the file it came from and the config that
was written beside it.

## Data provenance

The dataset is **every TCGA PanCanAtlas study in cBioPortal**: 32 studies,
**10 953 patients**, **3 543 deaths**, median follow-up 23.4 months.
`scripts/fetch_cbioportal_api.py` pulls it from the cBioPortal public API
(`/studies`, `/molecular-profiles`, `/molecular-data/fetch`,
`/mutations/fetch`, `/clinical-data`) and writes one parquet file per study
with one row per patient.

### The feature space is fixed before any data is seen

All three omic blocks are restricted to the **MSK-IMPACT 468 gene panel**
(469 genes after symbol resolution): mutation, discrete GISTIC copy number,
and mRNA expression. Every one of the 32 cohorts therefore has **the same
features** — 468 CNA, 469 expression, 469 mutation.

This is a deliberate correction to an earlier design, and it is the single
most consequential methodological decision in the work. The earlier pipeline
selected the highest-variance genes separately inside each cohort. That makes
the phrase "pan-cancer" unsupportable, because the cohorts do not share a
feature space, and it makes leave-one-cancer-out transfer incoherent, because
the held-out cohort's features are not the ones the model was trained on. A
curated, clinically deployed panel fixed in advance is citable, identical
everywhere, and immune to the selection-bias objection.

### The fetch was validated against an independent path

The API fetch was checked against the original file-export merge for BLCA,
which was built from downloaded cBioPortal study exports through a completely
separate code path: **411 patients against 411, 181 events against 181**, and
a TP53 mutation frequency of **0.487** against the roughly 0.48 reported in
the TCGA bladder cancer literature. Two independent routes to the same
numbers is the reason to trust the other 31 studies.

## What was run, and at what settings

```
python -m mosaic.cli {baseline,train,ablate,loco} \
    --root ./data_pan --cohorts <all 32> --target {cohort,stage_binary} \
    --max-genes 200 --epochs 20 --folds 5 --repeats 1 \
    --batch-size 32 --device cpu
```

`scripts/run_pan.sh` is that whole schedule, restart-safe. Every phase that
finished wrote a marker; every result file sits next to the `*_config.json`
that produced it.

### The targets

| Target | Classes | Labelled patients | Cohorts |
|---|---|---|---|
| `cohort` | 32 (tissue of origin) | 10 953 | 32 |
| `stage_binary` | 2 (early I–II vs late III–IV) | 6 634 (4 119 early, 2 515 late) | 19 |

Thirteen studies publish no AJCC stage at all — CESC, DLBC, GBM, LAML, LGG,
OV, PCPG, PRAD, SARC, THYM, UCEC, UCS, UVM — so the stage analysis covers
6 634 patients across 19 cohorts, not all 10 953. Both numbers belong in any
sentence about the stage result.

Four-class AJCC stage was **replaced** by the early/late collapse, and the
reason is not convenience: the sub-stage vocabularies differ between studies
(IIA in one is not recorded the same way in another), the four classes are
badly unbalanced, and the prognostically meaningful boundary is I/II against
III/IV. The four-class variant remains available as `--target stage`.

### Deviations from the originally requested settings

The brief asked for `--max-genes 1500 --epochs 120`. That was not reachable
on this machine, and the gap is not small:

| Configuration | Measured |
|---|---|
| 1500 genes, 3 layers, batch 32, knn 12 | 240.0 s/epoch |
| 200 genes, fit = 5 841 patients, batch 32 | 137.9 s/epoch |

At 240 s/epoch, 120 epochs is 8 hours for **one** fit, and the full schedule
is over a hundred fits across two targets — several weeks of continuous
compute on four cores, in a container that is rebuilt roughly every
forty-five minutes. The settings were therefore reduced to 200 genes and 20
epochs, which is stated everywhere the numbers appear rather than left for a
reader to discover. A reduced configuration honestly labelled is worth more
than a prescribed configuration that never produced a number.

Two further deviations:

* **Mini-batching.** The config default is `full_batch = True`. A full-batch
  step over ~5 800 training patients does not fit in 16 GB, so training is
  mini-batched at 32. This changes the optimisation trajectory. It is a real
  deviation, not a cosmetic one.
* **The baseline panel ran on one outer fold of five**
  (`MOSAIC_BASELINE_FOLDS=1`). Twenty-four models over 10 953 patients is the
  slowest phase in the schedule, and evaluating it on one fold left compute
  for MOSAIC-GNN's own cross-validation, without which the work has no
  headline result. Every baseline row is therefore a single fold with `n = 1`
  and `std = 0`, on the **same** held-out patients as fold 0 of the model's
  cross-validation, so the paired comparison is still like-for-like. Where
  this report shows a baseline beside a cross-validated model number, the
  fold counts differ and the table says so.

### One model that is reported as not run

`random_survival_forest` is **absent from both panels, and this is reported
rather than quietly omitted**. At pan-cancer scale it was OOM-killed by the
kernel. Bounding it to `max_depth=12`, `max_features="sqrt"` and two threads
did not save it, and a `SIGKILL` cannot be caught — so each attempt destroyed
the entire 24-model panel rather than leaving one blank row. It is skipped by
name (`MOSAIC_SKIP_BASELINES`) so that the other 23 models survive. Twenty-three
survival and classification baselines ran; this one did not, for that reason.

## Reading the two targets

Tissue of origin and tumour stage are not two difficulties of one task. They
are different tasks, and the contrast between them is the substantive result.

**Tissue of origin is close to saturated.** Plain L2 logistic regression on
the 468-gene panel reaches balanced accuracy 0.942 and AUROC 0.999 across 32
classes. Twelve further models land between 0.895 and 0.941. When the
simplest linear model in the panel is within a hundredth of the best, the
target cannot separate methods, and a tie there is a property of the benchmark
rather than evidence about any architecture.

**Early-versus-late stage from the same molecular profiles is hard.** The best
model in the panel reaches 0.685 balanced accuracy. Nothing in the panel
approaches the tissue numbers. That contrast — near-ceiling tissue, modest
stage, same patients, same genes, same pipeline — is a finding about what bulk
multi-omics encodes, and it is stated plainly here rather than buried.

### Where MOSAIC-GNN loses on tissue of origin, and the explanation that failed

On the `cohort` target MOSAIC-GNN reaches balanced accuracy **0.751** against
logistic regression's **0.942**. That is not a narrow miss; it is a loss of
nineteen points to the simplest model in the panel, and no reading of the
table makes it anything else.

An earlier version of this document proposed a mechanical cause. The
architecture contains a cohort adversary trained through gradient reversal,
whose purpose is to remove cohort-identifying information from the shared
representation so that what remains transfers across cancer types. Scoring
that model on cohort classification, the argument ran, measures how well it
recovers precisely the information it was regularised to discard — so the
deficit would be an artefact of the objective rather than a weakness of the
model.

That prediction was stated as falsifiable, the test was named in advance, and
**the test has now been run and the prediction failed.**

| `cohort` ablation, 2 folds | balanced accuracy | AUROC | C-index |
|---|---|---|---|
| `full` | 0.5772 ± 0.0057 | 0.9635 | 0.7344 |
| `no_disentangle` | 0.5705 ± 0.0080 | 0.9620 | 0.7372 |
| **difference** | **−0.0068** | −0.0015 | +0.0028 |

Removing the adversary entirely recovers **nothing**. Classification is
fractionally worse, and −0.007 sits inside the ±0.01 noise floor, so the
correct reading is no effect. The explanation required a gain approaching the
0.17 separating this model from the linear baselines; the measured change is
roughly twenty-five times smaller and points the wrong way.

**So the cohort classification deficit is unexplained.** It is not an
artefact of the disentanglement objective. The adversary is not what is
costing the model those nineteen points, and any sentence suggesting
otherwise has been removed from this report rather than softened.

Two statements survive, and only these two. First, `cohort` remains a weak
benchmark for separating *conventional* classifiers, which bunch within a
hundredth of each other — but it separates this architecture from them
perfectly well, and unfavourably. Second, a hypothesis that was committed to
in advance, tested against the obvious experiment, and abandoned on the
result is worth more at an examination than one that was never checked. What
would not have been defensible is leaving the claim standing because it was
convenient.

### Calibration on the 32-class head is poor

Expected calibration error on the `cohort` target is **0.273**, against 0.035
on `stage_binary`. The 32-class probability outputs are badly calibrated and
should not be read as confidences; a clinical application would need
post-hoc recalibration before using them for anything. This is a genuine
weakness of the run, it is listed in Limitations, and it is not softened
anywhere in the write-up. Mean predictive entropy of 1.896 nats over 32
classes says the same thing from the other direction: the head is diffuse.

The binary head is a different matter — ECE 0.035 at 0.594 balanced accuracy
is reasonable, which suggests the problem is the 32-way softmax under a
disentanglement penalty rather than the calibration machinery itself.

### The one number that was a leak, and how it was caught

On the first pan-cancer `stage_binary` run, logistic regression scored
**balanced accuracy 1.000**. That was a defect in this repository, not a
result: `LEAKY_FOR_TARGET` had no `stage_binary` entry, the lookup returned an
empty tuple, the leakage guard filtered nothing, and
`AJCC_PATHOLOGIC_TUMOR_STAGE` together with `PATH_T/N/M_STAGE` sat in the
feature matrix as predictors of the stage label.

With the entry added and the contaminated cache deleted, the same model scores
**0.685**. The drop from 1.000 to 0.685 is the evidence that the leak was real
and that the fix works. Unknown targets now raise instead of silently
filtering nothing, so the same class of bug cannot recur quietly.

Every number in this report is post-fix. A perfect score on a hard clinical
endpoint is a leak, never a result.

### What the leakage guard removes

For `cohort`: `CANCER_TYPE`, `SUBTYPE`, `ONCOTREE`, `ICD_O`, `ICD_10`,
`TISSUE_SOURCE_SITE`, `TUMOR_TISSUE_SITE`, `PRIMARY_SITE`, `ORGAN_OF_ORIGIN`
and the tissue-provenance columns. For `stage_binary`: the AJCC and
pathological/clinical T, N and M stage columns listed above. The dropped
columns are reported by name at load time, not merely filtered.

A residual caveat the guard cannot fix: clinical coding vocabularies differ
sharply between these 32 studies, so a classifier can read tissue partly from
**which** clinical fields are populated rather than from their values. The
`no_clinical` row of the ablation table is the direct measurement of how much
of the tissue result that explains, and it should be read before the headline
tissue number.

## What the model attends to, and the confound in reading it

The integrated-gradient ranking on the stage target puts canonical drivers at
the top: **CDKN2B, CDKN2A, TP53, SMAD4, TP63, CREBBP**, then MGA, EZH2,
RECQL4, AGO2, RAC1. Every one of the top six is a Cancer Gene Census gene.
That is the right sanity check to run on an attribution method, and it passes:
the model is not keying on arbitrary features.

The learned functional modules are also coherent rather than arbitrary:

| Module | Genes | Reading |
|---|---|---|
| 028 | CDKN2B, CDKN2A, ALOX12B | the 9p21 tumour-suppressor locus |
| 032 | CD274, JAK2, PDCD1LG2 | the 9p24.1 immune-checkpoint amplicon (PD-L1, PD-L2) |
| 014 | B2M, TAP1, HLA-B, HLA-A | MHC class I antigen presentation |
| 001 | EZH2, DNMT3A | epigenetic regulators |

**The confound, stated before anyone else states it.** The gene graph is built
from co-alteration correlation, and the modules are a spectral partition of
that graph. Genes that sit at the same locus are deleted or amplified
together, so recovering CDKN2A with CDKN2B, or CD274 with PDCD1LG2, partly
reflects physical proximity on the chromosome rather than a discovered
functional relationship. These modules are **correct**, and they are also
**partly predictable from the construction**. The honest claim is that the
module layer recovers known co-alteration structure — which is a validity
check on the method, not a biological discovery.

The attribution ranking is a set of hypotheses. A reviewer will expect the
overlap with COSMIC Cancer Gene Census and OncoKB quantified rather than
asserted, and pathway enrichment run on `runs_pan_stage/stage_binary_genes.rnk`
with FDR reported. Neither has been done here, and neither is claimed.

### The mutation channel is very nearly ignored

Splitting mean absolute integrated gradients by omic block gives a result that
has to be reported, because it qualifies the word "multi-omic" in the title:

| Target | Mutation | Copy number | Expression |
|---|---|---|---|
| `cohort` | 0.00036 (**0.7%**) | 0.01201 (24.7%) | 0.03627 (74.6%) |
| `stage_binary` | 0.00389 (**6.8%**) | 0.02551 (44.5%) | 0.02788 (48.7%) |

On tissue of origin the mutation block carries under one percent of total
attribution — roughly a hundredfold less than expression. On stage it rises to
seven percent, still the smallest of the three by a wide margin. Whatever the
model is integrating, it is overwhelmingly copy number and expression.

Two readings are available and the data here does not separate them. The
benign one is that binary mutation calls over a 468-gene panel are extremely
sparse, so most patients carry no signal in most of those features and a
continuous channel will dominate any gradient-based attribution for simple
numerical reasons. The unwelcome one is that the fusion layer is effectively
discarding a block, in which case the mutation channel is contributing
parameters and compute without contributing information.

Distinguishing them needs an ablation that drops the mutation block and
measures what happens to held-out performance. That ablation has not been run.
Until it is, the defensible statement is narrow: **gradient attribution is
dominated by copy number and expression, and the contribution of the mutation
block is unquantified.** A paper claiming balanced tri-omic integration on
this evidence would be overclaiming, and the draft does not make that claim.

## Honest limitations

1. **`LUAD` is lung adenocarcinoma only**, not a combined lung cohort. LUSC
   is loaded separately. Name it precisely in any sentence about lung.
2. **Several cohorts contribute almost no deaths**: TGCT 4 of 149, PCPG 5 of
   178, DLBC and KICH 9 each, THCA 16 of 499. Any cohort-level survival number
   for these rests on a handful of comparable pairs. The per-cohort table
   prints the event count beside every C-index for exactly this reason.
3. **Leave-one-cancer-out classification on the `cohort` target is a
   degenerate metric, and not in the way this document previously claimed.**
   Earlier revisions asserted that the held-out class is never seen in
   training, so its balanced accuracy must be "near zero by arithmetic". The
   first LOCO cohort returned **0.989**, which falsifies that reasoning
   outright.

   What is actually true is worse for the metric, not better. When the
   held-out test set contains exactly one class, `balanced_accuracy_score`
   reduces to the recall of that single class — scikit-learn drops the
   predicted-but-absent classes as undefined — and Matthews correlation is
   **0.0000 by construction**, which the run confirms. So the number is not a
   multiclass accuracy at all and carries no information about transfer in
   either direction, high or low.

   Separately, and still unexplained: the model assigned the correct
   held-out cohort to 91 of 92 ACC patients despite that label never
   appearing in training. That should not be possible from a softmax head
   given no positive examples, so it indicates something in the LOCO path
   that has not been identified. **No leave-one-cancer-out classification
   number is reported as a result in this work** until it is. The survival
   transfer in the same table is unaffected by the single-class degeneracy
   and is the half that carries information.

4. **The pipeline is not bit-reproducible.** Three fits of the same fold with
   the same seed gave balanced accuracy 0.4567, 0.4341 and 0.4547. The
   parallel scatter-add in the gene-graph message passing sums in
   thread-scheduling order, and float addition is not associative. A spread of
   roughly ±0.01 is therefore the noise floor: differences smaller than it
   must not be read as effects. This is reported rather than hidden behind a
   single lucky seed.
5. **Compute ceiling, and the reported numbers are not at the ceiling.** 200
   genes and 20 epochs, for the measured reasons above. The notebook fitted
   the same target at 120 genes and 6 epochs, which gives a second point on
   the curve:

   Three budgets have now been fitted on `stage_binary`, for MOSAIC-GNN and
   for the strongest baseline:

   | Genes / epochs | MOSAIC-GNN | `lightgbm` | Gap |
   |---|---|---|---|
   | 60 / 3 | 0.5687 | 0.6745 | 0.106 |
   | 120 / 6 | 0.5717 | 0.6820 | 0.110 |
   | 200 / 20 | 0.5938 | 0.6851 | 0.091 |

   Both families improve with more genes and the model improves faster —
   0.025 against 0.011 from 60 to 200 — so these results do understate what
   the architecture would reach with more compute. That is the favourable
   half and it is a measurement, not an assertion.

   The other half is in the same table. The gap runs 0.106, 0.110, 0.091: it
   narrows only at the final step and by less than the ±0.019 fold spread,
   so extrapolating the model past the baseline is not supported. Stated
   bluntly, because a reviewer will compute it: **`lightgbm` at 60 genes
   (0.6745) beats MOSAIC-GNN at 200 genes (0.5938) by 0.081** — the baseline
   with a third of the features outperforms the proposed architecture with
   all of them. No framing of the compute ceiling survives that sentence, and
   it is written here rather than left to be discovered.

   Note the two columns are not strictly comparable: the reduced figures are
   2-fold cross-validation means and the 200-gene figure is a single held-out
   fold. Compare the two CV rows with each other; treat the manuscript row as
   indicative.
6. **The 32-class head is badly calibrated.** ECE 0.273 on `cohort`, against
   0.035 on `stage_binary`. The multi-class probabilities are not usable as
   confidences without post-hoc recalibration, which this work does not do.
7. **MOSAIC-GNN is beaten on both classification targets**, by a wide margin
   on `cohort` (0.751 against 0.942) and by a significant margin on
   `stage_binary` by eight of twenty comparators. The survival side is where
   the architecture is competitive. A write-up that leads with the
   architecture rather than with what the molecular data adds over clinical
   covariates would be overclaiming, and the abstract is worded accordingly.

## The pipeline was audited, and it needed to be

Thirteen defects were found and fixed in the course of this work. Several were
**silent wrongness** rather than crashes, which is the dangerous kind:

* per-cohort gene selection, which made "pan-cancer" unsupportable;
* the `stage_binary` leakage above, which produced a perfect score;
* `hist_gradient_boosting` fitted without `class_weight="balanced"` — the only
  classifier in the panel missing it — scoring 0.196 against 0.931 after the
  fix;
* an unguarded survival family running last, where one exception discarded
  every classifier already fitted in that panel;
* a dense N×N patient graph (960 MB per instance, one per omic block in
  MOGONET) that made OOM certain at this scale; the sparse rewrite was
  verified identical to float precision (adjacency 0.0, GCN/SAGE 0.0,
  GAT 3.6e-07);
* resume checkpoints that restored the NumPy generator but not torch's, so
  dropout diverged after a restart.

The earlier balanced accuracy of 0.313 that appears in older drafts came from
a notebook whose configuration cell tested for `*_merged_full.csv` while the
merge writes `*_merged_full.parquet`. It never found the real data, fell back
to 160 synthetic patients, and printed tables that looked real. The fallback
now prints a banner stating that every number below it is synthetic and may
not be quoted — and that banner has already caught one mis-execution during
this work, within seconds, before any figure was built from it.

Stating all of this is the strongest available answer to "how do we know these
numbers are right?".

## What the statistics support

Computed from `table4_paired_tests.csv` in each run directory: paired bootstrap against every baseline on exactly the same held-out patients, Benjamini-Hochberg corrected across the table. A positive delta means MOSAIC-GNN is ahead.

### `cohort`

**Classification (balanced accuracy).** 20 comparators tested.

- Significantly **better than MOSAIC-GNN** (18): `logistic_l2` (-0.190, q=0.000), `lightgbm` (-0.189, q=0.000), `mogonet_style` (-0.187, q=0.000), `xgboost` (-0.187, q=0.000), `late_fusion_mlp` (-0.184, q=0.000), `logistic_elasticnet` (-0.183, q=0.000), `logistic_l1` (-0.182, q=0.000), `hist_gradient_boosting` (-0.179, q=0.000), `linear_svm` (-0.175, q=0.000), `svm_rbf` (-0.173, q=0.000), `patient_sage` (-0.164, q=0.000), `random_forest` (-0.162, q=0.000), `lda_shrinkage` (-0.158, q=0.000), `mlp_early_fusion` (-0.146, q=0.000), `patient_gat` (-0.146, q=0.000), `patient_gcn` (-0.144, q=0.000), `extra_trees` (-0.144, q=0.000), `pca_logistic` (-0.116, q=0.000)

- MOSAIC-GNN **significantly better** (2): `naive_bayes` (+0.208, q=0.000), `knn` (+0.061, q=0.000)


**Survival (C-index).** 10 comparators tested.

- MOSAIC-GNN **significantly better** (2): `cox_clinical_only` (+0.040, q=0.000), `patient_gcn` (+0.023, q=0.020)

- **Not distinguishable** at q < 0.05 (8): `mlp_early_fusion`, `late_fusion_mlp`, `mogonet_style`, `patient_gat`, `patient_sage`, `cox_lasso`, `cox_ridge`, `deepsurv`


### `stage_binary`

**Classification (balanced accuracy).** 20 comparators tested.

- Significantly **better than MOSAIC-GNN** (8): `lightgbm` (-0.098, q=0.000), `hist_gradient_boosting` (-0.085, q=0.000), `xgboost` (-0.075, q=0.000), `mogonet_style` (-0.048, q=0.010), `extra_trees` (-0.044, q=0.014), `logistic_l1` (-0.039, q=0.027), `linear_svm` (-0.038, q=0.042), `logistic_l2` (-0.038, q=0.044)

- MOSAIC-GNN **significantly better** (2): `naive_bayes` (+0.067, q=0.000), `knn` (+0.057, q=0.004)

- **Not distinguishable** at q < 0.05 (10): `logistic_elasticnet`, `svm_rbf`, `lda_shrinkage`, `random_forest`, `pca_logistic`, `mlp_early_fusion`, `late_fusion_mlp`, `patient_gcn`, `patient_gat`, `patient_sage`


**Survival (C-index).** 10 comparators tested.

- Significantly **better than MOSAIC-GNN** (1): `cox_lasso` (-0.028, q=0.000)

- MOSAIC-GNN **significantly better** (1): `cox_clinical_only` (+0.051, q=0.000)

- **Not distinguishable** at q < 0.05 (8): `mlp_early_fusion`, `late_fusion_mlp`, `mogonet_style`, `patient_gcn`, `patient_gat`, `patient_sage`, `cox_ridge`, `deepsurv`


### What replicates across targets

MOSAIC-GNN significantly exceeds `cox_clinical_only` on **every target tested** (`cohort` +0.040, q=0.000, `stage_binary` +0.051, q=0.000). The two targets label overlapping patients differently and were fitted independently, so this is the same finding arrived at twice rather than one result counted twice. It is the strongest statement these tables support, and it is a statement about what the molecular data adds over clinical covariates — not about the architecture, which loses on classification in both.

### How to word this

Write the claim these tests support and no stronger. Where a baseline is significantly better, say so; where nothing is distinguishable, that is a tie and not a win. In particular, a significant gain over `cox_clinical_only` is the finding that molecular features add prognostic information beyond clinical covariates — which is a real result about the data, independent of whether the proposed architecture beats the other architectures. Keep the two apart.

Differences smaller than about 0.01 balanced accuracy should not be read as effects at all: three fits of the same fold with the same seed gave 0.4567, 0.4341 and 0.4547, so that is the noise floor of this pipeline.

## Cohorts as actually loaded

| cohort   |   patients | blocks        | genes_per_block   |   events |   event_rate |   median_OS_months |   labelled |
|:---------|-----------:|:--------------|:------------------|---------:|-------------:|-------------------:|-----------:|
| ACC      |         92 | cna, exp, mut | 468, 469, 469     |       33 |        0.359 |               39.3 |         92 |
| BLCA     |        411 | cna, exp, mut | 468, 469, 469     |      181 |        0.44  |               17.6 |        411 |
| BRCA     |       1084 | cna, exp, mut | 468, 469, 469     |      151 |        0.139 |               27   |       1084 |
| CESC     |        297 | cna, exp, mut | 468, 469, 469     |       68 |        0.229 |               20.9 |        297 |
| CHOL     |         36 | cna, exp, mut | 468, 469, 469     |       18 |        0.5   |               21.2 |         36 |
| COADREAD |        594 | cna, exp, mut | 468, 469, 469     |      120 |        0.202 |               21.1 |        594 |
| DLBC     |         48 | cna, exp, mut | 468, 469, 469     |        9 |        0.188 |               26.7 |         48 |
| ESCA     |        182 | cna, exp, mut | 468, 468, 469     |       76 |        0.418 |               12.9 |        182 |
| GBM      |        585 | cna, exp, mut | 468, 469, 469     |      478 |        0.817 |               12.2 |        585 |
| HNSC     |        523 | cna, exp, mut | 468, 469, 469     |      219 |        0.419 |               21.2 |        523 |
| KICH     |         65 | cna, exp, mut | 468, 469, 469     |        9 |        0.138 |               73.9 |         65 |
| KIRC     |        512 | cna, exp, mut | 468, 469, 469     |      170 |        0.332 |               38.8 |        512 |
| KIRP     |        283 | cna, exp, mut | 468, 469, 469     |       41 |        0.145 |               25.2 |        283 |
| LAML     |        200 | cna, exp, mut | 468, 446, 469     |      133 |        0.665 |               12   |        200 |
| LGG      |        514 | cna, exp, mut | 468, 469, 469     |      125 |        0.243 |               22.3 |        514 |
| LIHC     |        372 | cna, exp, mut | 468, 469, 469     |      131 |        0.352 |               19.6 |        372 |
| LUAD     |        566 | cna, exp, mut | 468, 469, 469     |      186 |        0.329 |               21.5 |        566 |
| LUSC     |        487 | cna, exp, mut | 468, 469, 469     |      210 |        0.431 |               21.5 |        487 |
| MESO     |         87 | cna, exp, mut | 468, 469, 469     |       74 |        0.851 |               16.9 |         87 |
| OV       |        585 | cna, exp, mut | 468, 468, 469     |      342 |        0.585 |               33.3 |        585 |
| PAAD     |        184 | cna, exp, mut | 468, 469, 469     |       99 |        0.538 |               15.3 |        184 |
| PCPG     |        178 | cna, exp, mut | 468, 469, 469     |        5 |        0.028 |               24.9 |        178 |
| PRAD     |        494 | cna, exp, mut | 468, 469, 469     |       10 |        0.02  |               30.4 |        494 |
| SARC     |        255 | cna, exp, mut | 468, 469, 469     |       97 |        0.38  |               31.1 |        255 |
| SKCM     |        442 | cna, exp, mut | 468, 469, 469     |      213 |        0.482 |               41.6 |        442 |
| STAD     |        440 | cna, exp, mut | 468, 468, 469     |      171 |        0.389 |               14   |        440 |
| TGCT     |        149 | cna, exp, mut | 468, 469, 469     |        4 |        0.027 |               41.4 |        149 |
| THCA     |        499 | cna, exp, mut | 468, 469, 469     |       16 |        0.032 |               31   |        499 |
| THYM     |        123 | cna, exp, mut | 468, 469, 469     |        9 |        0.073 |               40.7 |        123 |
| UCEC     |        529 | cna, exp, mut | 468, 469, 469     |       87 |        0.164 |               30   |        529 |
| UCS      |         57 | cna, exp, mut | 468, 469, 469     |       35 |        0.614 |               20.1 |         57 |
| UVM      |         80 | cna, exp, mut | 468, 469, 469     |       23 |        0.287 |               25.8 |         80 |

## Target: `cohort`

_Cross-validation did not complete; no summary to report._

### Held-out fold: full metric panel

|                         |    value |
|:------------------------|---------:|
| accuracy                |   0.78   |
| balanced_accuracy       |   0.7512 |
| f1_macro                |   0.7218 |
| f1_weighted             |   0.7855 |
| precision_macro         |   0.7167 |
| recall_macro            |   0.7512 |
| specificity_macro       |   0.9929 |
| npv_macro               |   0.9928 |
| kappa                   |   0.77   |
| kappa_quadratic         |   0.8092 |
| mcc                     |   0.7705 |
| log_loss                |   1.0025 |
| brier                   |   0.4065 |
| top2_accuracy           |   0.8941 |
| auroc                   |   0.9846 |
| auprc                   |   0.7909 |
| ece                     |   0.273  |
| c_index                 |   0.7756 |
| auc@12m                 |   0.8034 |
| auc@36m                 |   0.8102 |
| auc@60m                 |   0.8093 |
| logrank_chi2            | 419.334  |
| logrank_p               |   0      |
| mean_predictive_entropy |   1.8961 |

### Per class (held-out fold)

|    |   support |   precision |   recall |     f1 |   specificity |   mean_predicted_prob |
|---:|----------:|------------:|---------:|-------:|--------------:|----------------------:|
|  0 |        19 |      0.8182 |   0.9474 | 0.878  |        0.9982 |                0.0241 |
|  1 |        82 |      0.7746 |   0.6707 | 0.719  |        0.9924 |                0.0275 |
|  2 |       217 |      0.9579 |   0.8387 | 0.8943 |        0.9959 |                0.0525 |
|  3 |        59 |      0.9636 |   0.8983 | 0.9298 |        0.9991 |                0.0256 |
|  4 |         7 |      0.1429 |   0.4286 | 0.2143 |        0.9918 |                0.0393 |
|  5 |       119 |      0.8    |   0.8067 | 0.8033 |        0.9884 |                0.0398 |
|  6 |        10 |      0.9091 |   1      | 0.9524 |        0.9995 |                0.032  |
|  7 |        36 |      0.5179 |   0.8056 | 0.6304 |        0.9875 |                0.0313 |
|  8 |       117 |      0.875  |   0.7778 | 0.8235 |        0.9937 |                0.0324 |
|  9 |       105 |      0.9368 |   0.8476 | 0.89   |        0.9971 |                0.0348 |
| 10 |        13 |      0.2222 |   0.3077 | 0.2581 |        0.9936 |                0.0299 |
| 11 |       102 |      0.8687 |   0.8431 | 0.8557 |        0.9938 |                0.0346 |
| 12 |        57 |      0.6964 |   0.6842 | 0.6903 |        0.992  |                0.0274 |
| 13 |        40 |      0.9714 |   0.85   | 0.9067 |        0.9995 |                0.0232 |
| 14 |       103 |      0.8981 |   0.9417 | 0.9194 |        0.9947 |                0.033  |
| 15 |        74 |      0.8553 |   0.8784 | 0.8667 |        0.9948 |                0.0288 |
| 16 |       113 |      0.6838 |   0.708  | 0.6957 |        0.9822 |                0.0373 |
| 17 |        98 |      0.7471 |   0.6633 | 0.7027 |        0.9895 |                0.0316 |
| 18 |        17 |      0.2708 |   0.7647 | 0.4    |        0.9839 |                0.03   |
| 19 |       117 |      0.7754 |   0.9145 | 0.8392 |        0.9851 |                0.0385 |
| 20 |        37 |      0.4333 |   0.7027 | 0.5361 |        0.9842 |                0.0301 |
| 21 |        35 |      0.5652 |   0.7429 | 0.642  |        0.9907 |                0.0261 |
| 22 |        99 |      1      |   1      | 1      |        1      |                0.0362 |
| 23 |        51 |      0.5526 |   0.4118 | 0.4719 |        0.9921 |                0.026  |
| 24 |        89 |      0.8267 |   0.6966 | 0.7561 |        0.9938 |                0.0321 |
| 25 |        88 |      0.7059 |   0.4091 | 0.518  |        0.9929 |                0.0298 |
| 26 |        29 |      0.8889 |   0.8276 | 0.8571 |        0.9986 |                0.0217 |
| 27 |       100 |      0.8557 |   0.83   | 0.8426 |        0.9933 |                0.0332 |
| 28 |        25 |      0.4848 |   0.64   | 0.5517 |        0.9922 |                0.0249 |
| 29 |       106 |      0.7596 |   0.7453 | 0.7524 |        0.988  |                0.0306 |
| 30 |        11 |      0.3333 |   0.4545 | 0.3846 |        0.9954 |                0.033  |
| 31 |        16 |      0.8421 |   1      | 0.9143 |        0.9986 |                0.0228 |

### Per cohort (held-out fold)

|          |   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   precision_macro |   recall_macro |   specificity_macro |   npv_macro |   kappa |   kappa_quadratic |   mcc |   log_loss |   brier |   top2_accuracy |   c_index |   n |
|:---------|-----------:|--------------------:|-----------:|--------------:|------------------:|---------------:|--------------------:|------------:|--------:|------------------:|------:|-----------:|--------:|----------------:|----------:|----:|
| ACC      |     0.9474 |              0.9474 |     0.4865 |        0.973  |            0.5    |         0.4737 |                 nan |           0 |       0 |                 0 |     0 |     0.3594 |  0.1298 |          1      |    0.7442 |  19 |
| BLCA     |     0.6707 |              0.6707 |     0.0892 |        0.8029 |            0.1111 |         0.0745 |                 nan |           0 |       0 |                 0 |     0 |     1.1366 |  0.4838 |          0.878  |    0.6962 |  82 |
| BRCA     |     0.8387 |              0.8387 |     0.0608 |        0.9123 |            0.0667 |         0.0559 |                 nan |           0 |       0 |                 0 |     0 |     0.994  |  0.3841 |          0.8986 |    0.6523 | 217 |
| CESC     |     0.8983 |              0.8983 |     0.1893 |        0.9464 |            0.2    |         0.1797 |                 nan |           0 |       0 |                 0 |     0 |     0.654  |  0.2435 |          0.9153 |    0.662  |  59 |
| COADREAD |     0.8067 |              0.8067 |     0.0687 |        0.893  |            0.0769 |         0.0621 |                 nan |           0 |       0 |                 0 |     0 |     1.0709 |  0.3941 |          0.8908 |    0.7458 | 119 |
| DLBC     |     1      |              1      |     1      |        1      |            1      |         1      |                 nan |         nan |     nan |               nan |     0 |     0.3185 |  0.094  |          1      |    0.6923 |  10 |
| ESCA     |     0.8056 |              0.8056 |     0.1487 |        0.8923 |            0.1667 |         0.1343 |                 nan |           0 |       0 |                 0 |     0 |     1.0485 |  0.4373 |          0.8889 |    0.6638 |  36 |
| GBM      |     0.7778 |              0.7778 |     0.125  |        0.875  |            0.1429 |         0.1111 |                 nan |           0 |       0 |                 0 |     0 |     1.0158 |  0.4461 |          0.9829 |    0.5372 | 117 |
| HNSC     |     0.8476 |              0.8476 |     0.1019 |        0.9175 |            0.1111 |         0.0942 |                 nan |           0 |       0 |                 0 |     0 |     0.8061 |  0.2998 |          0.8857 |    0.6947 | 105 |
| KICH     |     0.3077 |              0.3077 |     0.0941 |        0.4706 |            0.2    |         0.0615 |                 nan |           0 |       0 |                 0 |     0 |     1.9694 |  0.8578 |          0.6923 |    0.9474 |  13 |
| KIRC     |     0.8431 |              0.8431 |     0.1144 |        0.9149 |            0.125  |         0.1054 |                 nan |           0 |       0 |                 0 |     0 |     0.8477 |  0.3327 |          0.902  |    0.7831 | 102 |
| KIRP     |     0.6842 |              0.6842 |     0.0903 |        0.8125 |            0.1111 |         0.076  |                 nan |           0 |       0 |                 0 |     0 |     1.0708 |  0.4667 |          0.9123 |    0.8732 |  57 |
| LAML     |     0.85   |              0.85   |     0.4595 |        0.9189 |            0.5    |         0.425  |                 nan |           0 |       0 |                 0 |     0 |     0.4967 |  0.1901 |          0.925  |    0.4644 |  40 |
| LGG      |     0.9417 |              0.9417 |     0.2425 |        0.97   |            0.25   |         0.2354 |                 nan |           0 |       0 |                 0 |     0 |     0.683  |  0.25   |          0.9806 |    0.7506 | 103 |
| LIHC     |     0.8784 |              0.8784 |     0.1871 |        0.9353 |            0.2    |         0.1757 |                 nan |           0 |       0 |                 0 |     0 |     0.6777 |  0.2898 |          0.973  |    0.6728 |  74 |
| LUAD     |     0.708  |              0.708  |     0.0754 |        0.829  |            0.0909 |         0.0644 |                 nan |           0 |       0 |                 0 |     0 |     1.3702 |  0.5783 |          0.885  |    0.6382 | 113 |
| LUSC     |     0.6633 |              0.6633 |     0.0997 |        0.7975 |            0.125  |         0.0829 |                 nan |           0 |       0 |                 0 |     0 |     1.3362 |  0.5654 |          0.8673 |    0.5784 |  98 |
| MESO     |     0.7647 |              0.7647 |     0.1733 |        0.8667 |            0.2    |         0.1529 |                 nan |           0 |       0 |                 0 |     0 |     1.1082 |  0.4741 |          0.8824 |    0.681  |  17 |
| OV       |     0.9145 |              0.9145 |     0.1911 |        0.9554 |            0.2    |         0.1829 |                 nan |           0 |       0 |                 0 |     0 |     0.8695 |  0.3505 |          0.9573 |    0.5878 | 117 |
| PAAD     |     0.7027 |              0.7027 |     0.1376 |        0.8254 |            0.1667 |         0.1171 |                 nan |           0 |       0 |                 0 |     0 |     0.9584 |  0.4016 |          0.9459 |    0.6273 |  37 |
| PCPG     |     0.7429 |              0.7429 |     0.2842 |        0.8525 |            0.3333 |         0.2476 |                 nan |           0 |       0 |                 0 |     0 |     0.9414 |  0.42   |          0.9714 |    0.7931 |  35 |
| PRAD     |     1      |              1      |     1      |        1      |            1      |         1      |                 nan |         nan |     nan |               nan |     0 |     0.3389 |  0.0958 |          1      |    0.5128 |  99 |
| SARC     |     0.4118 |              0.4118 |     0.0972 |        0.5833 |            0.1667 |         0.0686 |                 nan |           0 |       0 |                 0 |     0 |     1.6292 |  0.7342 |          0.7255 |    0.6104 |  51 |
| SKCM     |     0.6966 |              0.6966 |     0.0684 |        0.8212 |            0.0833 |         0.0581 |                 nan |           0 |       0 |                 0 |     0 |     1.2937 |  0.4656 |          0.7865 |    0.6732 |  89 |
| STAD     |     0.4091 |              0.4091 |     0.0387 |        0.5806 |            0.0667 |         0.0273 |                 nan |           0 |       0 |                 0 |     0 |     1.7052 |  0.7136 |          0.6705 |    0.5995 |  88 |
| TGCT     |     0.8276 |              0.8276 |     0.1811 |        0.9057 |            0.2    |         0.1655 |                 nan |           0 |       0 |                 0 |     0 |     0.6607 |  0.2589 |          0.8621 |  nan      |  29 |
| THCA     |     0.83   |              0.83   |     0.1134 |        0.9071 |            0.125  |         0.1038 |                 nan |           0 |       0 |                 0 |     0 |     0.9256 |  0.3674 |          0.9    |    0.9459 | 100 |
| THYM     |     0.64   |              0.64   |     0.1301 |        0.7805 |            0.1667 |         0.1067 |                 nan |           0 |       0 |                 0 |     0 |     1.3892 |  0.581  |          0.8    |    0.8056 |  25 |
| UCEC     |     0.7453 |              0.7453 |     0.1423 |        0.8541 |            0.1667 |         0.1242 |                 nan |           0 |       0 |                 0 |     0 |     1.1535 |  0.5122 |          0.8774 |    0.657  | 106 |
| UCS      |     0.4545 |              0.4545 |     0.1562 |        0.625  |            0.25   |         0.1136 |                 nan |           0 |       0 |                 0 |     0 |     1.5385 |  0.7166 |          0.6364 |    0.5278 |  11 |
| UVM      |     1      |              1      |     1      |        1      |            1      |         1      |                 nan |         nan |     nan |               nan |     0 |     0.2507 |  0.0636 |          1      |    0.5102 |  16 |

### Generalisation gap

|                    |   balanced_accuracy |   auroc |   log_loss |   c_index |    n |
|:-------------------|--------------------:|--------:|-----------:|----------:|-----:|
| train              |              0.8143 |  0.9927 |     0.7962 |    0.7932 | 5841 |
| validation         |              0.7769 |  0.9878 |     0.8648 |    0.7678 | 2921 |
| test               |              0.7519 |  0.986  |     0.8909 |    0.7754 | 2191 |
| gap (train - test) |              0.0624 |  0.0067 |    -0.0947 |    0.0178 |  nan |

### Label permutation test

Models retrained on labels shuffled within the training and validation folds only; the held-out labels are left intact. A pipeline that leaks scores above chance here.

|                   |   observed |   null_mean |   null_sd |   null_max |   n_permutations |   p_value |
|:------------------|-----------:|------------:|----------:|-----------:|-----------------:|----------:|
| balanced_accuracy |     0.5937 |      0.0164 |    0.0046 |     0.0194 |                4 |       0.2 |
| auroc             |     0.9693 |      0.4869 |    0.0447 |     0.5474 |                4 |       0.2 |
| c_index           |     0.7348 |      0.5055 |    0.0441 |     0.5594 |                4 |       0.2 |

<details><summary>Per-permutation null</summary>

|   permutation |   balanced_accuracy |   auroc |   c_index |
|--------------:|--------------------:|--------:|----------:|
|             1 |              0.0194 |  0.5474 |    0.5594 |
|             2 |              0.0175 |  0.4759 |    0.4517 |
|             3 |              0.0096 |  0.44   |    0.5012 |
|             4 |              0.0191 |  0.4842 |    0.5096 |

</details>

### Baseline comparison

|                        |   balanced_accuracy |    auroc |    auprc |   c_index |
|:-----------------------|--------------------:|---------:|---------:|----------:|
| cox_clinical_only      |            nan      | nan      | nan      |    0.7342 |
| cox_lasso              |            nan      | nan      | nan      |    0.7793 |
| cox_ridge              |            nan      | nan      | nan      |    0.7695 |
| deepsurv               |            nan      | nan      | nan      |    0.7576 |
| extra_trees            |              0.8954 |   0.9959 |   0.9391 |  nan      |
| hist_gradient_boosting |              0.9314 |   0.9994 |   0.9664 |  nan      |
| knn                    |              0.6907 |   0.9689 |   0.8369 |  nan      |
| late_fusion_mlp        |              0.9354 |   0.9993 |   0.9702 |    0.7652 |
| lda_shrinkage          |              0.9094 |   0.9906 |   0.9219 |  nan      |
| lightgbm               |              0.941  |   0.9994 |   0.9714 |  nan      |
| linear_svm             |              0.9269 |   0.9978 |   0.9597 |  nan      |
| logistic_elasticnet    |              0.9352 |   0.9991 |   0.966  |  nan      |
| logistic_l1            |              0.9338 |   0.9988 |   0.9594 |  nan      |
| logistic_l2            |              0.9416 |   0.9989 |   0.9698 |  nan      |
| mlp_early_fusion       |              0.8981 |   0.9974 |   0.9424 |    0.7641 |
| mogonet_style          |              0.9393 |   0.9991 |   0.9745 |    0.7592 |
| naive_bayes            |              0.5437 |   0.7781 |   0.369  |  nan      |
| patient_gat            |              0.8977 |   0.9977 |   0.9415 |    0.7544 |
| patient_gcn            |              0.8954 |   0.9977 |   0.9446 |    0.7512 |
| patient_sage           |              0.916  |   0.9979 |   0.9571 |    0.7622 |
| pca_logistic           |              0.8683 |   0.9939 |   0.9036 |  nan      |
| random_forest          |              0.914  |   0.9948 |   0.944  |  nan      |
| svm_rbf                |              0.9254 |   0.9971 |   0.9493 |  nan      |
| xgboost                |              0.9389 |   0.9994 |   0.969  |  nan      |

### Paired bootstrap against each baseline (held-out fold)

`*_delta` is MOSAIC-GNN minus the baseline on the same patients; `*_q` is the Benjamini-Hochberg corrected p-value over the whole table.

| model                  |   bacc_delta |   bacc_q | bacc_sig   |   cindex_delta |   cindex_q | cindex_sig   |
|:-----------------------|-------------:|---------:|:-----------|---------------:|-----------:|:-------------|
| logistic_l2            |      -0.1897 |        0 | True       |       nan      |   nan      | False        |
| logistic_l1            |      -0.1819 |        0 | True       |       nan      |   nan      | False        |
| logistic_elasticnet    |      -0.1833 |        0 | True       |       nan      |   nan      | False        |
| linear_svm             |      -0.175  |        0 | True       |       nan      |   nan      | False        |
| svm_rbf                |      -0.1735 |        0 | True       |       nan      |   nan      | False        |
| knn                    |       0.0612 |        0 | True       |       nan      |   nan      | False        |
| naive_bayes            |       0.2082 |        0 | True       |       nan      |   nan      | False        |
| lda_shrinkage          |      -0.1575 |        0 | True       |       nan      |   nan      | False        |
| random_forest          |      -0.1621 |        0 | True       |       nan      |   nan      | False        |
| extra_trees            |      -0.1435 |        0 | True       |       nan      |   nan      | False        |
| pca_logistic           |      -0.1165 |        0 | True       |       nan      |   nan      | False        |
| xgboost                |      -0.1871 |        0 | True       |       nan      |   nan      | False        |
| lightgbm               |      -0.1891 |        0 | True       |       nan      |   nan      | False        |
| mlp_early_fusion       |      -0.1463 |        0 | True       |         0.0099 |     0.2714 | False        |
| late_fusion_mlp        |      -0.1836 |        0 | True       |         0.0088 |     0.2825 | False        |
| hist_gradient_boosting |      -0.1795 |        0 | True       |       nan      |   nan      | False        |
| mogonet_style          |      -0.1874 |        0 | True       |         0.0148 |     0.132  | False        |
| patient_gcn            |      -0.1436 |        0 | True       |         0.0229 |     0.02   | True         |
| patient_gat            |      -0.1458 |        0 | True       |         0.0196 |     0.0733 | False        |
| patient_sage           |      -0.1641 |        0 | True       |         0.0118 |     0.2033 | False        |
| cox_lasso              |     nan      |      nan | False      |        -0.0053 |     0.3811 | False        |
| cox_ridge              |     nan      |      nan | False      |         0.0046 |     0.529  | False        |
| cox_clinical_only      |     nan      |      nan | False      |         0.0398 |     0      | True         |
| deepsurv               |     nan      |      nan | False      |         0.0164 |     0.105  | False        |

### Ablation

|                      |   balanced_accuracy |   auroc |   c_index |
|:---------------------|--------------------:|--------:|----------:|
| full                 |              0.5772 |  0.9635 |    0.7344 |
| no_disentangle       |              0.5705 |  0.962  |    0.7372 |
| no_gene_graph        |              0.56   |  0.962  |    0.7383 |
| no_module_pool       |              0.5653 |  0.9614 |    0.7352 |
| no_patient_graph     |              0.5644 |  0.9606 |    0.7355 |
| no_contrast          |              0.5694 |  0.9617 |    0.7359 |
| no_clinical          |              0.1178 |  0.6941 |    0.5904 |
| no_clinical_residual |              0.3987 |  0.9262 |    0.7346 |

### Leave-one-cancer-out transfer

_Not available: the run that produces this table did not complete._

### Sample-size curve

_Not available: the run that produces this table did not complete._

### Capacity

- trainable parameters: 275,190
- training patients: 5,841
- parameters per training patient: 47.1

### Top genes by integrated gradients

<details><summary>Top 15</summary>

| gene    | module     |   gate_mean |   gate_std |   ig_abs |   ig_cna |   ig_exp |   ig_mut |
|:--------|:-----------|------------:|-----------:|---------:|---------:|---------:|---------:|
| CDKN2B  | MODULE_030 |      0.3814 |     0.0625 |   0.21   |   0.0957 |  -0.0703 |   0      |
| PPP2R1A | MODULE_033 |      0.3083 |     0.0585 |   0.1546 |   0.0177 |  -0.1346 |   0      |
| GNAS    | MODULE_046 |      0.3894 |     0.0981 |   0.1535 |   0.037  |  -0.1046 |  -0.0005 |
| APC     | MODULE_014 |      0.2798 |     0.0309 |   0.1361 |  -0.0205 |   0.0843 |  -0.0005 |
| NFE2L2  | MODULE_021 |      0.3025 |     0.0871 |   0.124  |   0.0147 |  -0.041  |  -0.0002 |
| DNAJB1  | MODULE_010 |      0.2968 |     0.0742 |   0.1215 |  -0.0084 |   0.0313 |  -0.0002 |
| AKT1    | MODULE_033 |      0.363  |     0.0932 |   0.1201 |   0.0049 |  -0.0589 |   0      |
| NOTCH3  | MODULE_010 |      0.3292 |     0.0644 |   0.1167 |  -0.0078 |   0.0084 |   0.0009 |
| YES1    | MODULE_010 |      0.3741 |     0.0871 |   0.1092 |   0.0126 |  -0.0129 |  -0.0001 |
| PAK5    | MODULE_014 |      0.2201 |     0.0296 |   0.1061 |  -0.0072 |   0.0903 |  -0.0001 |
| AGO2    | MODULE_045 |      0.505  |     0.1414 |   0.0843 |  -0.0051 |  -0.0379 |  -0.0003 |
| NOTCH1  | MODULE_006 |      0.2153 |     0.0348 |   0.0824 |   0.005  |  -0.0596 |  -0.0001 |
| CDKN2A  | MODULE_030 |      0.2218 |     0.0749 |   0.08   |   0.0457 |  -0.0132 |  -0.0003 |
| MAPK1   | MODULE_061 |      0.4048 |     0.0916 |   0.0786 |   0.0118 |  -0.0544 |  -0.0006 |
| PIK3R1  | MODULE_054 |      0.2637 |     0.0508 |   0.0767 |  -0.0084 |   0.0371 |  -0      |

</details>

## Target: `stage_binary`

_Cross-validation did not complete; no summary to report._

### Held-out fold: full metric panel

|                         |    value |
|:------------------------|---------:|
| accuracy                |   0.6136 |
| balanced_accuracy       |   0.594  |
| f1_macro                |   0.5931 |
| f1_weighted             |   0.615  |
| precision_macro         |   0.5926 |
| recall_macro            |   0.594  |
| specificity_macro       |   0.594  |
| npv_macro               |   0.5926 |
| kappa                   |   0.1865 |
| kappa_quadratic         | nan      |
| mcc                     |   0.1866 |
| log_loss                |   0.6554 |
| brier                   |   0.4629 |
| top2_accuracy           |   1      |
| auroc                   |   0.6374 |
| auprc                   |   0.4922 |
| ece                     |   0.0348 |
| c_index                 |   0.7228 |
| auc@12m                 |   0.7536 |
| auc@36m                 |   0.737  |
| auc@60m                 |   0.7356 |
| logrank_chi2            | 258.135  |
| logrank_p               |   0      |
| mean_predictive_entropy |   0.6691 |

### Per class (held-out fold)

|    |   support |   precision |   recall |     f1 |   specificity |   mean_predicted_prob |
|---:|----------:|------------:|---------:|-------:|--------------:|----------------------:|
|  0 |       821 |      0.6929 |   0.676  | 0.6843 |        0.5119 |                0.5306 |
|  1 |       504 |      0.4924 |   0.5119 | 0.5019 |        0.676  |                0.4694 |

### Per cohort (held-out fold)

|          |   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   precision_macro |   recall_macro |   specificity_macro |   npv_macro |    kappa |   kappa_quadratic |      mcc |   log_loss |    brier |   top2_accuracy |    auroc |    auprc |   c_index |   n |
|:---------|-----------:|--------------------:|-----------:|--------------:|------------------:|---------------:|--------------------:|------------:|---------:|------------------:|---------:|-----------:|---------:|----------------:|---------:|---------:|----------:|----:|
| ACC      |     0.7895 |              0.7143 |     0.7286 |        0.7624 |            0.875  |         0.7143 |              0.7143 |      0.875  |   0.4865 |               nan |   0.5669 |     0.557  |   0.3692 |               1 |   0.8333 |   0.7997 |    0.6739 |  20 |
| BLCA     |     0.5802 |              0.5895 |     0.5653 |        0.5942 |            0.578  |         0.5895 |              0.5895 |      0.578  |   0.1568 |               nan |   0.1672 |     0.6937 |   0.5    |               1 |   0.5448 |   0.6887 |    0.5516 |  81 |
| BRCA     |     0.6244 |              0.5589 |     0.5492 |        0.64   |            0.5498 |         0.5589 |              0.5589 |      0.5498 |   0.1061 |               nan |   0.1083 |     0.6344 |   0.4434 |               1 |   0.619  |   0.3662 |    0.7146 | 217 |
| CESC     |   nan      |            nan      |   nan      |      nan      |          nan      |       nan      |            nan      |    nan      | nan      |               nan | nan      |   nan      | nan      |             nan | nan      | nan      |    0.4856 |  60 |
| COADREAD |     0.5862 |              0.5759 |     0.576  |        0.5839 |            0.5776 |         0.5759 |              0.5759 |      0.5776 |   0.153  |               nan |   0.1534 |     0.6692 |   0.4767 |               1 |   0.6    |   0.5479 |    0.5989 | 119 |
| ESCA     |     0.5625 |              0.583  |     0.5625 |        0.5625 |            0.583  |         0.583  |              0.583  |      0.583  |   0.1547 |               nan |   0.166  |     0.6848 |   0.4918 |               1 |   0.587  |   0.499  |    0.4822 |  37 |
| GBM      |   nan      |            nan      |   nan      |      nan      |          nan      |       nan      |            nan      |    nan      | nan      |               nan | nan      |   nan      | nan      |             nan | nan      | nan      |    0.5533 | 117 |
| HNSC     |     0.764  |              0.688  |     0.678  |        0.7697 |            0.6709 |         0.688  |              0.688  |      0.6709 |   0.3571 |               nan |   0.3586 |     0.5884 |   0.3971 |               1 |   0.7058 |   0.8419 |    0.6422 | 104 |
| KICH     |     0.6923 |              0.6    |     0.5667 |        0.6205 |            0.8333 |         0.6    |              0.6    |      0.8333 |   0.2353 |               nan |   0.3651 |     0.658  |   0.4654 |               1 |   0.6    |   0.5776 |    0.8696 |  13 |
| KIRC     |     0.6176 |              0.5884 |     0.5888 |        0.6102 |            0.5956 |         0.5884 |              0.5884 |      0.5956 |   0.1818 |               nan |   0.1838 |     0.6609 |   0.4679 |               1 |   0.6317 |   0.5165 |    0.6991 | 102 |
| KIRP     |     0.7255 |              0.6134 |     0.6186 |        0.7176 |            0.6273 |         0.6134 |              0.6134 |      0.6273 |   0.2388 |               nan |   0.2402 |     0.5741 |   0.3855 |               1 |   0.7571 |   0.4785 |    0.6842 |  57 |
| LAML     |   nan      |            nan      |   nan      |      nan      |          nan      |       nan      |            nan      |    nan      | nan      |               nan | nan      |   nan      | nan      |             nan | nan      | nan      |    0.4599 |  40 |
| LGG      |   nan      |            nan      |   nan      |      nan      |          nan      |       nan      |            nan      |    nan      | nan      |               nan | nan      |   nan      | nan      |             nan | nan      | nan      |    0.7574 | 103 |
| LIHC     |     0.6714 |              0.5268 |     0.5235 |        0.6449 |            0.5374 |         0.5268 |              0.5268 |      0.5374 |   0.0607 |               nan |   0.0633 |     0.653  |   0.4611 |               1 |   0.4892 |   0.2778 |    0.5702 |  74 |
| LUAD     |     0.5882 |              0.6386 |     0.5547 |        0.6242 |            0.5938 |         0.6386 |              0.6386 |      0.5938 |   0.1856 |               nan |   0.2281 |     0.6785 |   0.4837 |               1 |   0.654  |   0.3729 |    0.5979 | 113 |
| LUSC     |     0.4536 |              0.5788 |     0.4383 |        0.4966 |            0.5523 |         0.5788 |              0.5788 |      0.5523 |   0.0802 |               nan |   0.1283 |     0.6892 |   0.495  |               1 |   0.73   |   0.3844 |    0.6294 |  98 |
| MESO     |     0.4118 |              0.35   |     0.3561 |        0.434  |            0.3714 |         0.35   |              0.35   |      0.3714 |  -0.2687 |               nan |  -0.2777 |     0.782  |   0.5844 |               1 |   0.2667 |   0.5892 |    0.7154 |  17 |
| OV       |   nan      |            nan      |   nan      |      nan      |          nan      |       nan      |            nan      |    nan      | nan      |               nan | nan      |   nan      | nan      |             nan | nan      | nan      |    0.5257 | 117 |
| PAAD     |     0.5135 |              0.4314 |     0.3833 |        0.6207 |            0.4794 |         0.4314 |              0.4314 |      0.4794 |  -0.0439 |               nan |  -0.0752 |     0.6954 |   0.5016 |               1 |   0.4314 |   0.1384 |    0.6139 |  37 |
| PCPG     |   nan      |            nan      |   nan      |      nan      |          nan      |       nan      |            nan      |    nan      | nan      |               nan | nan      |   nan      | nan      |             nan | nan      | nan      |  nan      |  35 |
| PRAD     |   nan      |            nan      |   nan      |      nan      |          nan      |       nan      |            nan      |    nan      | nan      |               nan | nan      |   nan      | nan      |             nan | nan      | nan      |    0.6644 |  99 |
| SARC     |   nan      |            nan      |   nan      |      nan      |          nan      |       nan      |            nan      |    nan      | nan      |               nan | nan      |   nan      | nan      |             nan | nan      | nan      |    0.525  |  51 |
| SKCM     |     0.5823 |              0.5783 |     0.5755 |        0.5775 |            0.582  |         0.5783 |              0.5783 |      0.582  |   0.1577 |               nan |   0.1603 |     0.6986 |   0.5046 |               1 |   0.552  |   0.5176 |    0.552  |  89 |
| STAD     |     0.5476 |              0.5612 |     0.5466 |        0.544  |            0.5633 |         0.5612 |              0.5612 |      0.5633 |   0.1177 |               nan |   0.1245 |     0.6991 |   0.5052 |               1 |   0.5739 |   0.5964 |    0.4816 |  88 |
| TGCT     |     0.75   |              0.4286 |     0.4286 |        0.75   |            0.4286 |         0.4286 |              0.4286 |      0.4286 |  -0.1429 |               nan |  -0.1429 |     0.6072 |   0.415  |               1 |   0.8214 |   0.3667 |  nan      |  29 |
| THCA     |     0.7    |              0.5685 |     0.5479 |        0.6371 |            0.685  |         0.5685 |              0.5685 |      0.685  |   0.1681 |               nan |   0.2252 |     0.5994 |   0.4097 |               1 |   0.6992 |   0.5524 |    0.8841 | 100 |
| THYM     |   nan      |            nan      |   nan      |      nan      |          nan      |       nan      |            nan      |    nan      | nan      |               nan | nan      |   nan      | nan      |             nan | nan      | nan      |    0.7    |  25 |
| UCEC     |   nan      |            nan      |   nan      |      nan      |          nan      |       nan      |            nan      |    nan      | nan      |               nan | nan      |   nan      | nan      |             nan | nan      | nan      |    0.7441 | 106 |
| UCS      |   nan      |            nan      |   nan      |      nan      |          nan      |       nan      |            nan      |    nan      | nan      |               nan | nan      |   nan      | nan      |             nan | nan      | nan      |    0.4595 |  11 |
| UVM      |   nan      |            nan      |   nan      |      nan      |          nan      |       nan      |            nan      |    nan      | nan      |               nan | nan      |   nan      | nan      |             nan | nan      | nan      |    0.8056 |  16 |

### Generalisation gap

|                    |   balanced_accuracy |   auroc |   log_loss |   c_index |    n |
|:-------------------|--------------------:|--------:|-----------:|----------:|-----:|
| train              |              0.5923 |  0.641  |     0.6472 |    0.736  | 5841 |
| validation         |              0.5828 |  0.6347 |     0.6475 |    0.7287 | 2921 |
| test               |              0.5874 |  0.6377 |     0.6524 |    0.7212 | 2191 |
| gap (train - test) |              0.0049 |  0.0033 |    -0.0052 |    0.0148 |  nan |

### Label permutation test

Models retrained on labels shuffled within the training and validation folds only; the held-out labels are left intact. A pipeline that leaks scores above chance here.

_Not available: the run that produces this table did not complete._

### Baseline comparison

|                        |   balanced_accuracy |    auroc |    auprc |   c_index |
|:-----------------------|--------------------:|---------:|---------:|----------:|
| cox_clinical_only      |            nan      | nan      | nan      |    0.6697 |
| cox_lasso              |            nan      | nan      | nan      |    0.7485 |
| cox_ridge              |            nan      | nan      | nan      |    0.7372 |
| deepsurv               |            nan      | nan      | nan      |    0.722  |
| extra_trees            |              0.6312 |   0.7346 |   0.6226 |  nan      |
| hist_gradient_boosting |              0.6723 |   0.7465 |   0.6349 |  nan      |
| knn                    |              0.5303 |   0.6295 |   0.4858 |  nan      |
| late_fusion_mlp        |              0.6194 |   0.6783 |   0.5582 |    0.7349 |
| lda_shrinkage          |              0.5984 |   0.6606 |   0.5105 |  nan      |
| lightgbm               |              0.6851 |   0.7532 |   0.6292 |  nan      |
| linear_svm             |              0.6255 |   0.6518 |   0.502  |  nan      |
| logistic_elasticnet    |              0.6227 |   0.668  |   0.5153 |  nan      |
| logistic_l1            |              0.6269 |   0.6753 |   0.5219 |  nan      |
| logistic_l2            |              0.6255 |   0.655  |   0.5058 |  nan      |
| mlp_early_fusion       |              0.6118 |   0.6612 |   0.5419 |    0.7217 |
| mogonet_style          |              0.635  |   0.6909 |   0.5656 |    0.7319 |
| naive_bayes            |              0.5208 |   0.546  |   0.4078 |  nan      |
| patient_gat            |              0.6105 |   0.6646 |   0.5381 |    0.7282 |
| patient_gcn            |              0.6076 |   0.6508 |   0.5204 |    0.7208 |
| patient_sage           |              0.6085 |   0.6533 |   0.5308 |    0.7223 |
| pca_logistic           |              0.6098 |   0.6536 |   0.5303 |  nan      |
| random_forest          |              0.6188 |   0.7439 |   0.6212 |  nan      |
| svm_rbf                |              0.5914 |   0.6706 |   0.5462 |  nan      |
| xgboost                |              0.6626 |   0.756  |   0.6331 |  nan      |

### Paired bootstrap against each baseline (held-out fold)

`*_delta` is MOSAIC-GNN minus the baseline on the same patients; `*_q` is the Benjamini-Hochberg corrected p-value over the whole table.

| model                  |   bacc_delta |   bacc_q | bacc_sig   |   cindex_delta |   cindex_q | cindex_sig   |
|:-----------------------|-------------:|---------:|:-----------|---------------:|-----------:|:-------------|
| logistic_l2            |      -0.0381 |   0.044  | True       |       nan      |   nan      | False        |
| logistic_l1            |      -0.0395 |   0.0275 | True       |       nan      |   nan      | False        |
| logistic_elasticnet    |      -0.0354 |   0.0527 | False      |       nan      |   nan      | False        |
| linear_svm             |      -0.0381 |   0.0422 | True       |       nan      |   nan      | False        |
| svm_rbf                |      -0.004  |   0.775  | False      |       nan      |   nan      | False        |
| knn                    |       0.0571 |   0.004  | True       |       nan      |   nan      | False        |
| naive_bayes            |       0.0665 |   0      | True       |       nan      |   nan      | False        |
| lda_shrinkage          |      -0.011  |   0.5147 | False      |       nan      |   nan      | False        |
| random_forest          |      -0.0314 |   0.0785 | False      |       nan      |   nan      | False        |
| extra_trees            |      -0.0439 |   0.0143 | True       |       nan      |   nan      | False        |
| hist_gradient_boosting |      -0.0849 |   0      | True       |       nan      |   nan      | False        |
| pca_logistic           |      -0.0224 |   0.2088 | False      |       nan      |   nan      | False        |
| xgboost                |      -0.0753 |   0      | True       |       nan      |   nan      | False        |
| lightgbm               |      -0.0978 |   0      | True       |       nan      |   nan      | False        |
| mlp_early_fusion       |      -0.0244 |   0.2088 | False      |        -0.001  |     0.973  | False        |
| late_fusion_mlp        |      -0.032  |   0.0785 | False      |        -0.0141 |     0.3725 | False        |
| mogonet_style          |      -0.0476 |   0.01   | True       |        -0.0112 |     0.524  | False        |
| patient_gcn            |      -0.0202 |   0.2576 | False      |        -0.0001 |     0.973  | False        |
| patient_gat            |      -0.0231 |   0.2088 | False      |        -0.0075 |     0.745  | False        |
| patient_sage           |      -0.0211 |   0.2633 | False      |        -0.0015 |     0.973  | False        |
| cox_lasso              |     nan      | nan      | False      |        -0.0278 |     0      | True         |
| cox_ridge              |     nan      | nan      | False      |        -0.0165 |     0.18   | False        |
| cox_clinical_only      |     nan      | nan      | False      |         0.0511 |     0      | True         |
| deepsurv               |     nan      | nan      | False      |        -0.0013 |     0.973  | False        |

### Ablation

_Not available: the run that produces this table did not complete._

### Leave-one-cancer-out transfer

_Not available: the run that produces this table did not complete._

### Sample-size curve

_Not available: the run that produces this table did not complete._

### Capacity

- trainable parameters: 273,796
- training patients: 5,841
- parameters per training patient: 46.9

### Top genes by integrated gradients

<details><summary>Top 15</summary>

| gene   | module     |   gate_mean |   gate_std |   ig_abs |   ig_cna |   ig_exp |   ig_mut |
|:-------|:-----------|------------:|-----------:|---------:|---------:|---------:|---------:|
| CDKN2B | MODULE_028 |      0.3454 |     0.0393 |   0.2922 |   0.1408 |   0.1203 |   0      |
| CDKN2A | MODULE_028 |      0.3277 |     0.0525 |   0.2615 |   0.1256 |   0.0924 |   0.0038 |
| TP53   | MODULE_055 |      0.1581 |     0.0183 |   0.2267 |   0.0251 |   0.0988 |   0.0863 |
| SMAD4  | MODULE_029 |      0.2722 |     0.0378 |   0.22   |   0.0578 |   0.1535 |   0.0067 |
| TP63   | MODULE_009 |      0.2465 |     0.0304 |   0.1573 |   0.0872 |  -0.0017 |  -0.0014 |
| CREBBP | MODULE_027 |      0.3479 |     0.0373 |   0.1493 |   0.0219 |   0.1032 |   0.0116 |
| MGA    | MODULE_063 |      0.3422 |     0.036  |   0.0804 |   0.0159 |   0.0491 |   0.0071 |
| EZH2   | MODULE_001 |      0.5013 |     0.0395 |   0.0762 |   0.0413 |  -0.028  |   0.0002 |
| RECQL4 | MODULE_053 |      0.3535 |     0.0504 |   0.076  |   0.0524 |  -0.0091 |  -0.0004 |
| AGO2   | MODULE_051 |      0.5015 |     0.0325 |   0.0753 |   0.0324 |  -0.0287 |  -0.0002 |
| RAC1   | MODULE_020 |      0.2298 |     0.0346 |   0.0644 |   0.0467 |  -0.0056 |   0      |
| ELOC   | MODULE_053 |      0.3242 |     0.0369 |   0.0615 |   0.046  |   0.0048 |   0      |
| NF1    | MODULE_060 |      0.3229 |     0.0306 |   0.0611 |   0.0108 |   0.0383 |   0.0063 |
| STAT3  | MODULE_060 |      0.3359 |     0.0389 |   0.0605 |   0.0144 |   0.0309 |   0.0001 |
| TEK    | MODULE_005 |      0.2105 |     0.0204 |   0.0555 |   0.0117 |   0.0325 |   0.0002 |

</details>
