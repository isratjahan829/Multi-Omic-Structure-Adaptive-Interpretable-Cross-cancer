# Permutation-null run — reduced settings, NOT the manuscript numbers

This directory exists for one table: `tableS2_permutation_test.csv`, the
label-permutation leakage control. Everything else in here is a by-product of
producing it and must not be quoted.

## Why it is a separate directory

`scripts/export_tables.py` rewrites `table2`, `table2b`, `table2c`, `table4`
and the generalisation gap every time it runs. Running the permutations into
`runs_pan_cohort/` would therefore have overwritten the manuscript's held-out
fold numbers — computed at 200 genes, 20 epochs, 5 folds — with the
permutation run's cheaper settings, silently, in the directory the report and
the Blackbook read from.

That is the same class of mistake as the config overwrite earlier in this
work, where starting the ablation replaced `cohort_config.json` and made the
provenance record state the wrong settings. Caught in advance this time.

## Settings here

100 genes, 6 epochs, 2 folds, 4 permutations — against the manuscript's 200
genes, 20 epochs, 5 folds. Each permuted fit is a full retrain, so the
permutation count is what the compute budget allows rather than what a
reviewer would prefer.

Its own reference fold scores balanced accuracy 0.5937, AUROC 0.9693, C-index
0.7348. The manuscript fold scores 0.7512, 0.9846 and 0.7756. The gap is the
configuration, not a discrepancy.

## What the permutation test can and cannot establish here

With B permutations the smallest attainable p-value is 1/(B+1). At B = 4 that
is **0.2**, so this test cannot reject at the 0.05 level and no p-value from
it should be reported as significant.

What it can do is the thing it is actually needed for: show whether a model
retrained on labels shuffled within the training and validation folds scores
above chance on intact held-out labels. A pipeline that leaks does. A pipeline
that does not leak sits at chance — 0.031 for 32 classes. That is a diagnostic
about the pipeline, and four permutations answer it adequately even though
they cannot support a significance claim.

Report it that way: as a leakage diagnostic with its B stated and its p-value
floor stated, not as a hypothesis test that passed.

## Files

| File | Use |
|---|---|
| `tableS2_permutation_test.csv` | the reason this directory exists |
| `tableS2_permutation_null.csv` | per-permutation scores behind it |
| `table5_ablation.csv` | CSV form of the eight-arm ablation JSON; copied to `runs_pan_cohort/` |
| everything else | by-products at reduced settings — do not quote |
