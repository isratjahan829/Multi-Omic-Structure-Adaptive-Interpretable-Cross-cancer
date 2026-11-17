# Notebook output: real data, reduced configuration

This directory holds what `notebooks/mosaic_gnn.ipynb` produces. It is **real
pan-cancer data** from `data_pan/` (32 studies, 10 953 patients), fitted at a
deliberately reduced configuration so the notebook finishes in about an hour
on a laptop-class machine.

Three directories, three provenances, and the distinction matters:

| Directory | Data | Configuration | Quotable as a result? |
|---|---|---|---|
| `runs/` | **synthetic** | smoke test | **No. Never.** |
| `runs_notebook/` | real | reduced (120 genes, 6 epochs) | No — demonstration only |
| `runs_pan_cohort/`, `runs_pan_stage/` | real | manuscript (200 genes, 20 epochs) | **Yes** |

Earlier the notebook wrote into `runs/`, putting real reduced output in the
same folder as synthetic fixtures. `scripts/build_notebook.py` now points at
this directory instead, so provenance is readable from the path alone.

## What the reduction costs, measured

The notebook fitted `stage_binary` on one held-out fold at 120 genes and 6
epochs. The manuscript run used 200 genes and 20 epochs on the same target:

| Metric | Notebook (120g, 6ep) | Manuscript (200g, 20ep) | Delta |
|---|---|---|---|
| balanced accuracy | 0.5580 | 0.5938 | −0.0358 |
| AUROC | 0.5980 | 0.6372 | −0.0392 |
| C-index | 0.6841 | 0.7233 | −0.0392 |
| ECE | 0.0459 | 0.0352 | +0.0107 |

Two things follow, and both belong in the manuscript rather than here alone.

**The reduction costs about 0.036–0.039 on every discrimination metric,
consistently.** That is three to four times the ±0.01 run-to-run noise floor,
so it is a real effect and not seed variation.

**The reported numbers are therefore not at the model's ceiling.** Performance
was still improving with genes and epochs when the compute budget ran out.
This converts the compute-ceiling limitation from an assertion into a
measurement: a reviewer asking "would more compute have helped?" can be told
yes, by this much, over this range — while noting that two points do not
establish where the curve flattens, and nothing here shows the gap would
close against the baselines, which were not similarly constrained.

## The cross-validation file

`stage_binary_results.json` is now `"complete": true`, both planned folds
finished:

| Metric | Mean ± SD (2 folds) |
|---|---|
| balanced accuracy | 0.5717 ± 0.0194 |
| AUROC | 0.6071 ± 0.0128 |
| C-index | 0.6839 ± 0.0003 |
| ECE | 0.0305 ± 0.0218 |

Two folds give a spread, not a reliable one — an SD from n=2 has one degree of
freedom and should be quoted as a range, not treated as a standard error. The
balanced-accuracy spread of ±0.019 is about twice the ±0.01 run-to-run noise
floor, so fold-to-fold variation here exceeds seed variation.

## The gene reduction hurts the model and not the baselines

`table3_comparison_full.csv` in this directory is the baseline panel refitted
at 120 genes. Comparing it with the manuscript panel at 200 genes:

| Model | 120 genes | 200 genes | Delta |
|---|---|---|---|
| `lightgbm` | 0.682 | 0.685 | −0.003 |
| MOSAIC-GNN | 0.5717 | 0.5938 | −0.0221 |

The gradient-boosted baseline is essentially unaffected by cutting the feature
space by 40%; MOSAIC-GNN loses roughly seven times as much. Read carefully,
this cuts both ways and the manuscript should say so:

* It supports the claim that the model is compute-constrained rather than
  converged — it is the method still climbing as resources increase.
* It equally shows the baselines were already extracting what this feature
  space holds at 120 genes, so the gap at 200 genes is not obviously
  something more genes would close. Extrapolating MOSAIC-GNN past
  `lightgbm` from two points would be wishful.

Neither reading is established by two configurations. What is established is
that the two method families respond very differently to the gene budget, and
any claim about convergence must account for that rather than assume the
model's curve keeps rising while the baselines' stays flat.
