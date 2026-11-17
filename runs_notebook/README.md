# Notebook output: real data, reduced configuration

This directory holds what `notebooks/mosaic_gnn.ipynb` produces. It is **real
pan-cancer data** from `data_pan/` (32 studies, 10 953 patients), fitted at a
deliberately reduced configuration so the notebook finishes in about an hour
on a laptop-class machine.

Three directories, three provenances, and the distinction matters:

| Directory | Data | Configuration | Quotable as a result? |
|---|---|---|---|
| `runs/` | **synthetic** | smoke test | **No. Never.** |
| `runs_notebook/` | real | reduced (60 genes, 3 epochs) | No — demonstration only |
| `runs_pan_cohort/`, `runs_pan_stage/` | real | manuscript (200 genes, 20 epochs) | **Yes** |

Earlier the notebook wrote into `runs/`, putting real reduced output in the
same folder as synthetic fixtures. `scripts/build_notebook.py` now points at
this directory instead, so provenance is readable from the path alone.

## What the reduction costs, measured at three gene budgets

The shipped notebook runs at **60 genes, 3 epochs**. An earlier execution used
120 genes and 6 epochs, and the manuscript used 200 genes and 20 epochs. That
gives three points for MOSAIC-GNN and two for the strongest baseline:

| Genes / epochs | MOSAIC-GNN bacc | `lightgbm` bacc | Gap |
|---|---|---|---|
| 60 / 3 | 0.5687 | 0.6745 | 0.106 |
| 120 / 6 | 0.5717 | 0.6820 | 0.110 |
| 200 / 20 | 0.5938 | 0.6851 | 0.091 |

Full metric set for the shipped run, 2-fold CV:

| Metric | 60g / 3ep (2-fold CV) | 200g / 20ep (held-out fold) |
|---|---|---|
| balanced accuracy | 0.5687 ± 0.0186 | 0.5938 |
| AUROC | 0.6053 ± 0.0142 | 0.6372 |
| C-index | 0.6801 ± 0.0031 | 0.7233 |
| ECE | 0.0343 ± 0.0100 | 0.0352 |

**These columns are not strictly comparable.** The left is a mean over two
folds; the right is a single held-out fold. A fold mean and a point estimate
have different sampling properties, and the ±0.019 fold spread is of the same
order as the difference between them. Compare the two CV rows with each other
(60g 0.5687, 120g 0.5717) and treat the manuscript column as indicative.

Three things follow.

**Both method families improve with more genes, and the model improves
faster.** MOSAIC-GNN gains 0.025 from 60 to 200 genes against `lightgbm`'s
0.011. That is the evidence for the compute-ceiling limitation, and it is a
measurement rather than an assertion.

**The gap does not close.** It is 0.106, 0.110, 0.091 across the three
budgets — narrowing only at the last step, and by less than the fold spread.
Extrapolating the model past the baseline from this is not supported.

**The blunt version, which belongs in the manuscript.** `lightgbm` at 60
genes (0.6745) still beats MOSAIC-GNN at 200 genes (0.5938) by 0.081. The
baseline given a third of the features outperforms the proposed architecture
given all of them. No framing of the compute ceiling survives that sentence,
and it should be written rather than left for a reviewer to compute.

## The cross-validation file

`stage_binary_results.json` is `"complete": true`, both folds finished, and
the metric table above carries its spread. An SD from two folds has one
degree of freedom: quote it as a range, not as a standard error.
