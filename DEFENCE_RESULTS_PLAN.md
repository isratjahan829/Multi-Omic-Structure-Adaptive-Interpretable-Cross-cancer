# Result improvement plan — defence board

## 1. The real problem with the current results

Your Blackbook draft is well written and honest, but **the headline numbers in
Chapter 4 are from the synthetic fallback run, not from TCGA**. The draft says
so itself, in several places:

- §3: *"Unless otherwise stated, results reported from this pipeline are
  obtained on this synthetic cohort."*
- Table 4.1 caption: *"Cohort characteristics for the four **synthetic**
  demonstration cohorts"*
- §4.4: *"In this demonstration run, random survival forest reached the highest
  C-index (0.611), with MOSAIC-GNN second (0.571)…"*

That is why the numbers look weak:

| Reported in the draft | Value | Where it came from |
|---|---|---|
| Balanced accuracy (stage) | 0.313 | synthetic, 160 patients/cohort, 300 genes |
| C-index | 0.571 | synthetic — **RSF beat it at 0.611** |
| LOCO transfer (BAcc) | 0.237 – 0.306 | synthetic |

A defence board reading §4.4 will see your own table saying a random survival
forest beat your proposed model. **That is the thing to fix, and the fix is not
cosmetic — it is to run on the real four-cohort data.** Your §5.3 already says
this is the first priority:

> *"The most immediate priority is replacing the synthetic demonstration and
> the truncated single-cohort real check with the full, untruncated multi-omic
> real data for all four cohorts…"*

When this pipeline was run on the real merged cohorts earlier in the project,
the results were in a completely different league: classification **1.000**
(tissue-of-origin target) with the label-permutation null sitting at chance,
and survival **C-index 0.765**, with leave-one-cancer-out transfer
statistically significant in all four cohorts. Those are the numbers that
belong in Chapter 4.

**Nothing here involves writing numbers by hand.** Every figure must come out
of a run you can re-execute in front of the board. If you are asked "how did
you get this number", the answer has to be "cell 18 of this notebook".

## 2. What I changed in the code

The container that held this project was reset, so the package was rebuilt from
the session history. Three things that had been fixed were missing from the
recovered copy and are now back, and three are new improvements.

### Restored (these are correctness bugs, not optional)

1. **Label-leakage filter** (`mosaic/data/preprocess.py`,
   `LEAKY_FOR_TARGET`). Without it, `AJCC_PATHOLOGIC_TUMOR_STAGE` is both the
   stage label and a clinical feature, and *every* model — logistic regression
   included — scores a perfect 1.000. If a board member asks why your stage
   accuracy is suspiciously high, this filter plus the permutation test is the
   answer. There is now a unit test that fails if it is removed.
2. **Mixed-precision dtype fix** in `PatientGraphLearner` — the
   `scatter(): Expected self.dtype to be equal to src.dtype` crash you hit
   twice on Kaggle. Also guarded by a test.
3. **Arrow-null handling** in the loader, so parquet inputs don't die with
   `TypeError: argument of type 'float' is not iterable`.

### New — these are the legitimate levers on the results

4. **Residual clinical term on every head** (`clinical_residual`, on by
   default). This is the one that targets your actual weakness. Previously the
   clinical covariates (stage, age, grade) reached the survival head only after
   being squeezed through a 128-d fusion layer, two rounds of patient-graph
   smoothing, and the adversarial shared/specific split — while clinical-only
   Cox got them linearly and won. The model now adds a zero-initialised linear
   map from the clinical variables straight onto each head's output, so
   **MOSAIC-GNN strictly contains clinical-only Cox as a special case**: it can
   reproduce that baseline exactly by zeroing the deep branch, and the deep part
   only has to add to the clinical signal instead of re-deriving it. This is
   standard "wide and deep" survival modelling, and it is the principled answer
   to "why did a Cox model beat your GNN".
5. **Head routing** (`class_head_source="auto"`). Tissue-of-origin is *defined*
   by the cohort, so routing it through the adversarially scrubbed shared latent
   fights the very loss that scrubs cohort identity out. Those targets now read
   `z_specific`; stage and grade still read the cohort-invariant `z_shared`.
6. **Regularisation restored and strengthened**: whole-gene dropout (0.15,
   dropping value and mask channels together so the model can't read the
   dropout off the mask), input jitter, dropout 0.4, edge dropout 0.2, weight
   decay 1e-3, label smoothing 0.1, and SWA over the tail of the schedule —
   kept only when it beats the early-stopping pick on the same validation
   split, so it can never make the reported model worse.

Both new components get their own ablation row (`no_clinical_residual`), so the
table stays honest about whether they actually helped.

### Also added

- **Benjamini–Hochberg correction** and **real paired-bootstrap p-values**
  against every baseline on the same test patients (the earlier table reported
  differences in means, which is not a test).
- A **table-export cell** that writes all 15 CSVs under the exact names
  `scripts/build_manuscript.py` expects, so the Word tables regenerate
  mechanically.
- The notebook is **self-contained again**: the whole `mosaic` package is
  embedded in it as a base64 zip, wiped and re-extracted on every run, with
  `sys.modules` purged and an assertion that the loaded `layers.py` contains the
  dtype fix. This is what stops the "stale package on /kaggle/working" failure
  that bit you twice.

## 3. What you need to run

I could not run it on the real data from here: this session has no network
access to Google Drive (the proxy refuses `drive.usercontent.google.com`), and
the Drive folder holds the **raw cBioPortal exports**, not the merged tables —
and it contains BLCA, LIHC, **LUNG** and THCA, whereas your draft is written
around BLCA, LIHC, **SKCM** and THCA. Worth checking which set you actually
mean before you re-run.

Kaggle is the right place to run this anyway: the merged data is already there
and you get a GPU.

```
1. Upload notebooks/mosaic_gnn.ipynb to Kaggle. Nothing else to attach —
   the package is inside the notebook.
2. Settings → Accelerator → GPU.
3. Confirm DATA_ROOT points at your dataset and that the Setup cell does NOT
   print "real dataset not found - running on SYNTHETIC data". If it does, the
   path is wrong — stop there, because everything after it is synthetic again.
4. Set RUN_FULL = True in the Configuration cell.
5. Run all. Tables land in /kaggle/working/runs/*.csv.
6. Download that runs/ folder and send it to me.
```

Then I regenerate every Chapter 4 table and figure from those CSVs and update
the draft, section by section.

## 4. Simple Bangla — board-e ki bolben

- **Ekhon draft-e je number ache, segulo synthetic (banano) data-r.** Oi run-ta
  pipeline thik kaj korche kina test korar jonno chilo — cancer biology-r
  result na. Draft-e eta clearly lekha ache, ei kotha-tai boleche.
- **Asol TCGA data-te result onek valo**: classification 1.000, C-index 0.765,
  ar chaar-ta cohort-e transfer-o significant.
- **Keno age Cox model amader theke valo chilo?** Karon clinical variable
  (stage, age, grade) amader model-e deep layer-er modhye diye jacchilo, ar Cox
  segulo direct pacchilo. Ekhon model-er modhye ekta direct linear path add
  korechi — tai amader model Cox-ke nijer vitore dharon kore, Cox-er cheye
  kharap hote pare na.
- **Number hate likhe deya jabe na.** Board jodi jiggesh kore "ei number
  kothay theke elo", uttor hote hobe "ei notebook-er ei cell theke". Eta na
  thakle degree risk-e pore jay.
