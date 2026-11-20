# What is in this directory, and which file to submit

Three Word documents live here and they are **not** three versions of the same
thing. Two are superseded, and one of those two contains claims this project
has since retracted. Read this before sending anything to an examiner.

| File | Status | Submit? |
|---|---|---|
| `CSE400B-MOSAIC-GNN-Report.docx` / `.pdf` | **Current.** Built by `scripts/build_ewu_report.py`, 69 pages, in the course's printout format. | **Yes** |
| `MOSAIC_GNN_draft.docx` | Superseded. A journal-style manuscript from an earlier stage. Honest but less complete. | No |
| `MOSAIC_GNN_Blackbook_updated.docx` | **Superseded and unsafe as it stands** — see below. | **No** |

Every number in the current report is checked against the run exports, the
payload and the executed notebook by `scripts/verify_consistency.py`, which
exits non-zero on any disagreement. Neither superseded file is covered by that
check.

## Why `MOSAIC_GNN_Blackbook_updated.docx` must not be submitted as it stands

Two defects, both verifiable by searching the file.

**It reports leave-one-cancer-out classification metrics, which this project
committed to never reporting.** One figure caption reads:

> "Balanced accuracy (teal), AUROC (green), and C-index (grey) for each
> held-out cohort, with the chance/no-discrimination line at 0.5 marked.
> Balanced accuracy sits well below 0.5 for every cohort (0.25–0.3…)"

and another:

> "THCA showing the strongest AUROC (~0.55) and BLCA the weakest (~0.45, i.e.
> below chance)."

That is the claim the project **retracted**. In a leave-one-cancer-out fold the
held-out cohort contains one class, so balanced accuracy collapses to that
class's recall and Matthews correlation is zero by construction — the metric is
degenerate, not small. The measured values are nothing like 0.25–0.3: the
median single-class recall across the 32 cohorts is **0.986**, 26 cohorts
exceed 0.9 and 8 are exactly 1.000. Section 6.14.1 of the current report sets
out the measurements and the unresolved question they raise. A reader checking
the Blackbook's figure against the exported table would find they disagree.

**It carries dataset sizes from the synthetic-fixture era.** Section 3.1.3
reads:

> "All four cohorts are pooled into a single pan-cancer set of 640 patients."

The real assembled dataset is **32 studies and 10 953 patients**. The document
states the correct figure elsewhere, so it also contradicts itself.

Several table captions in both superseded files also say "across four
cross-validation folds (mean ± sd)". No such run exists for either target:
every headline number comes from one held-out fold, because
`scripts/export_tables.py` builds the tables from the saved primary fold and
`splits[0]` / `preds[0]`. The current report states this in its
threats-to-validity section, and the verifier fails if a multi-fold summary
ever appears while the documents still describe one fold.

## If you want the Blackbook rebuilt

`scripts/build_blackbook.py` edits a copy of the department template in place
and would pull the current numbers and figures through. It was not run for this
revision because it was not asked for. Say the word and it can be regenerated
against the current results, after which it would be covered by the verifier
like everything else.
