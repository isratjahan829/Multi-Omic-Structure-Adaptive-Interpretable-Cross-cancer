# This directory is SYNTHETIC smoke-test output. Do not quote it.

Every CSV, JSON and PNG tracked in `runs/` was produced by the synthetic
data generator, not by cBioPortal data. The giveaway is in
`runs/table1_cohorts.csv`: four cohorts of **exactly 160 patients each**, with
**exactly 300 genes in each omic block**. Real cohorts are not round numbers.
The pan-cancer dataset this project actually reports on has 10,953 patients
across 32 studies and 469 genes.

These files exist because they are the fixture the smoke test asserts against,
and they are kept under version control for that reason alone.

## The hazard

`scripts/build_manuscript.py` takes `--runs <dir>` and reads
`fig1_cohorts.png`, `fig2_risk.png`, `fig3_classification.png`,
`figS1_learning_curves.png` and `figS2_sample_size.png` from it. Those five
filenames exist **here**, with synthetic content. Building the manuscript with

    python scripts/build_manuscript.py --runs runs ...      # WRONG

therefore produces a draft illustrated with figures drawn from invented data,
and nothing in the output says so. The numbers in the prose would come from
the real fact files while the figures came from here, which is the worst of
both worlds: a document that looks internally consistent and is not.

## What to use instead

    python scripts/build_manuscript.py --runs runs_pan_cohort \
        --facts cohort:runs_pan_cohort stage_binary:runs_pan_stage ...

`runs_pan_cohort/` and `runs_pan_stage/` hold the real pan-cancer results.
Neither currently contains figures, so the real draft reports every figure as
`[not run]`. That is correct and is preferable to the alternative: a missing
figure is visible, a wrong figure is not.

## Real figures

Figures generated from the real dataset are kept in
`deliverables/notebook_figures/`, with their settings recorded in the README
there. They are deliberately **not** stored here and deliberately **not**
named `figN_*.png`, so that they cannot be picked up by the builder as
manuscript figures without someone choosing to do so explicitly.
