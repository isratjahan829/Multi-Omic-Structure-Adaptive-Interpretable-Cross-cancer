# Sample-size run — reduced settings, NOT the manuscript numbers

This directory exists for one table: `tableS3_sample_size.csv`, the curve of
held-out performance against training-set size. Everything else here is a
by-product of producing it.

## Why it is separate

`scripts/export_tables.py` rewrites `table2`, `table2b`, `table2c` and the
generalisation gap on every run. Writing into `runs_pan_cohort/` would have
replaced the manuscript's held-out fold numbers — 200 genes, 20 epochs, 5
folds — with this run's cheaper settings, silently, in the directory the
report and the Blackbook read from. Same reasoning as
`runs_pan_cohort_perm/`.

## Settings

100 genes, 6 epochs, 2 folds, against the manuscript's 200 / 20 / 5. Its own
reference fold therefore scores below Table 4.2, and that gap is the
configuration rather than a discrepancy.

## What transfers and what does not

The **shape** of the sample-size curve is the point: whether held-out
performance is still climbing when the training data runs out. That shape is
a property of the model and data and is informative at this configuration.

The **absolute values** on the curve are not comparable with Table 4.2, for
the same reason every other reduced-configuration table in this project is
not. Report the trend, not the levels.

## Files

| File | Use |
|---|---|
| `tableS3_sample_size.csv` | the reason this directory exists |
| `table6_loco.csv` | CSV form of the 32-cohort LOCO JSON; copied to `runs_pan_cohort/` |
| `table5_ablation.csv` | CSV form of the 8-arm ablation JSON |
| everything else | by-products at reduced settings — do not quote |
