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

### Where MOSAIC-GNN loses on tissue of origin, and why that was predictable

On the `cohort` target MOSAIC-GNN reaches balanced accuracy **0.751** against
logistic regression's **0.942**. That is not a narrow miss; it is a loss of
nineteen points to the simplest model in the panel, and no reading of the
table makes it anything else.

There is a mechanical reason, and it is stated here as an explanation under
test rather than as a defence. The architecture contains a cohort adversary
trained through gradient reversal, whose entire purpose is to remove
cohort-identifying information from the shared representation so that what
remains transfers across cancer types. Scoring that model on cohort
classification measures how well it recovers precisely the information it was
regularised to discard. Logistic regression operates under no such constraint
and is free to key on tissue-specific mutation patterns, which a 468-gene
panel supplies in abundance.

The honest position is therefore: `cohort` is the wrong benchmark for this
architecture, and it would have been the wrong benchmark whatever the result.
It is reported because it was run, and because a report that dropped the
target where the model does badly would not be worth reading. The
`no_disentangle` ablation is the direct test — if removing the adversary
recovers most of the nineteen points, the explanation holds; if it does not,
the explanation is wrong and the loss is unexplained. Until that ablation
runs, neither conclusion may be asserted, and this paragraph claims only that
the prediction is falsifiable.

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
3. **Leave-one-cancer-out on the `cohort` target is degenerate by
   construction.** Training on 31 cohorts and testing on the 32nd means the
   held-out class was never observed, so its balanced accuracy is near zero
   necessarily — that is arithmetic, not a finding. Only the survival transfer
   in that table carries information.
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

   | Metric | 120 genes, 6 epochs | 200 genes, 20 epochs | Delta |
   |---|---|---|---|
   | balanced accuracy | 0.5580 | 0.5938 | −0.0358 |
   | AUROC | 0.5980 | 0.6372 | −0.0392 |
   | C-index | 0.6841 | 0.7233 | −0.0392 |

   The reduction costs 0.036–0.039 on every discrimination metric — three to
   four times the ±0.01 noise floor, so a real effect rather than seed
   variation. Performance was still improving with genes and epochs when the
   budget ran out, so these results understate what the architecture would
   reach with more compute. Two points do not locate where the curve
   flattens, and nothing here suggests the gap to the baselines would close:
   they were not similarly constrained, and the gradient-boosted trees in
   particular were fitted to convergence.
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
