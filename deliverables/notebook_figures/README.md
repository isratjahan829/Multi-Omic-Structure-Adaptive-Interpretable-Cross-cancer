# Figures from the real dataset, at reduced settings

These two figures were produced by `notebooks/mosaic_gnn.ipynb` executed
against the real pan-cancer dataset in `data_pan/` (32 cBioPortal studies,
10,953 patients). They are **not** synthetic. The synthetic smoke-test
outputs live in `runs/` and are marked as such in `runs/README.md`.

## The settings differ from the manuscript run, and that matters

| | Notebook (these figures) | Manuscript run |
|---|---|---|
| genes | 120 | 200 |
| epochs | 6 | 20 |
| target | `stage_binary` | `cohort` and `stage_binary` |

The notebook runs a reduced configuration so that it completes in roughly an
hour and a reader can execute it. The manuscript numbers come from
`scripts/run_pan.sh`. **Do not quote a number off these figures as a
manuscript result** — they will not match the tables, and they are not meant
to.

What they are good for is showing that the pipeline runs end to end on real
data and produces coherent output, which is the notebook's job.

## fig_km_risk.png

Model-defined risk tertiles on the held-out fold, n = 730 / 730 / 731.
Log-rank high vs low: **p = 1.12e-56**. The three curves separate
monotonically and do not cross, and the predicted survival curves from the
discrete-time hazard head reproduce the ordering. Risk stratification
survives the reduction to 120 genes and 6 epochs.

## fig_classification.png

Three panels for `stage_binary`:

- **Confusion matrix**, row-normalised: 0.56 on early, 0.61 on late.
- **One-vs-rest ROC**: AUC 0.62 for both classes. Consistent with the full
  run's 0.637 AUROC, which is the point — the reduction costs little here
  because the endpoint is hard for every model in the panel.
- **Selective prediction under MC dropout**: accuracy rises from 0.58 at full
  coverage to roughly **0.68 around 35–40% coverage**, then falls back toward
  the full-coverage line.

That third panel is the one worth attention. It says the model's dropout
uncertainty is informative rather than decorative: ranking predictions by
confidence and abstaining on the least confident 60% buys about ten points of
accuracy. A model whose uncertainty carried no signal would trace a flat line.

This is a property the head-to-head comparisons do not measure, since no
baseline in the panel emits calibrated predictive uncertainty. It is
therefore a genuine capability claim that survives the paired-test losses on
balanced accuracy — but it is a claim about *this* reduced fit, and it has
not yet been reproduced at manuscript settings or tested against an
uncertainty-equipped baseline such as a deep ensemble. State it that way or
not at all.
