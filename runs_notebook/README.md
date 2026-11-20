# Notebook output: real data, reduced configuration

This directory holds what the **retraining half** of
`notebooks/mosaic_gnn.ipynb` produces — Sections 3 onward, not Section 2. It
is **real pan-cancer data** from `data_pan/` (32 studies, 10 953 patients),
fitted at a deliberately reduced configuration so the notebook finishes in a
sitting.

`scripts/execute_notebook.py` ships it at **`cohort`, 80 genes, 4 epochs,
2 folds**, with one baseline fold and the ablation, transfer and permutation
phases set to the small counts its `--help` lists. Earlier revisions of this
file described a `stage_binary` run at 60 genes and 3 epochs; that was a
different invocation and the numbers below come from it. Read the settings
each cell prints, not this paragraph, if the two ever disagree — the notebook
prints the scope it actually used.

A `cohort_results.json` here is the demonstration's own cross-validation, not
the manuscript's. The manuscript targets have no such file at all, which is
the point Section 6.20 of the report makes about single-fold headline
numbers.

Three directories, three provenances, and the distinction matters:

| Directory | Data | Configuration | Quotable as a result? |
|---|---|---|---|
| `runs/` | **synthetic** | smoke test | **No. Never.** |
| `runs_notebook/` | real | reduced (80 genes, 4 epochs as shipped) | No — demonstration only |
| `runs_pan_cohort/`, `runs_pan_stage/` | real | manuscript (200 genes, 20 epochs) | **Yes** |

Earlier the notebook wrote into `runs/`, putting real reduced output in the
same folder as synthetic fixtures. `scripts/build_notebook.py` now points at
this directory instead, so provenance is readable from the path alone.

## This directory is not what the notebook reports any more

The notebook's **Section 2** reads `runs_pan_cohort/` and `runs_pan_stage/`
and prints the manuscript tables directly, checking every number against
`deliverables/manuscript_payload.json` as it goes. What lands here comes from
the retraining sections that follow, which exist so a reader can watch the
pipeline run and change something — not so their numbers can be quoted.

The two were once the same thing, and that was the problem: the notebook
printed only the reduced figures below while the report carried the full-run
figures, so the two documents shared no number. `scripts/verify_consistency.py`
now fails if that drift returns.

## What the reduction costs, measured at three gene budgets

These three points were measured on the **`stage_binary`** target at 60, 120
and 200 genes. They are kept because the trend they show is the evidence for
the compute-ceiling limitation, but note that none of them is the invocation
the notebook ships today (`cohort`, 80 genes, 4 epochs) — comparing them with
what the current notebook prints would be comparing two different targets.

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
