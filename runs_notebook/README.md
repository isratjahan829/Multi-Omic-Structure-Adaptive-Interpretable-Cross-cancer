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

## A caveat on the cross-validation file

`stage_binary_results.json` records `"complete": false` with
`folds_completed: 1` of `folds_planned: 2`. The summary therefore averages one
fold and its standard deviations are all 0.0000 — that is an artefact of n=1,
not a claim of perfect stability. Do not report an SD from this file.
