/**
 * Render the manuscript draft.
 *
 *   node scripts/make_docx.js payload.json MOSAIC_GNN_draft.docx
 *
 * `payload.json` comes from scripts/build_manuscript.py and carries every
 * table (already formatted as a grid of strings) and every figure. This file
 * holds the prose and decides where each table and figure sits.
 */

const fs = require("fs");
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun,
  PageBreak, PageNumber, PageOrientation, Packer, Paragraph, ShadingType, Table,
  TableCell, TableRow, TextRun, WidthType,
} = require("docx");

const payload = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const OUT = process.argv[3] || "MOSAIC_GNN_draft.docx";

const PORTRAIT_W = 10800;   // 12240 - 2 x 720 margin, in DXA
const LANDSCAPE_W = 14400;  // 15840 - 2 x 720

// --------------------------------------------------------------------------- //
// Building blocks
// --------------------------------------------------------------------------- //
const blocks = [];
const push = (b, landscape = false) => blocks.push({ ...b, landscape });

function p(text, opts = {}) {
  const { size = 22, bold = false, italics = false, align, spacing, color } = opts;
  return new Paragraph({
    alignment: align,
    spacing: spacing || { after: 120, line: 300 },
    children: [new TextRun({ text, size, bold, italics, color, font: "Calibri" })],
  });
}

function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({
    heading: level,
    spacing: { before: 280, after: 140 },
    children: [new TextRun({ text, font: "Calibri" })],
  });
}

function caption(label, text, kind = "Table") {
  return new Paragraph({
    spacing: { before: kind === "Table" ? 160 : 80, after: 160 },
    children: [
      new TextRun({ text: `${label}. `, bold: true, size: 18, font: "Calibri" }),
      new TextRun({ text, size: 18, font: "Calibri" }),
    ],
  });
}

function note(text) {
  return new Paragraph({
    spacing: { before: 80, after: 200 },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: "C00000", space: 8 } },
    children: [new TextRun({ text, size: 18, italics: true, color: "7F1D1D",
                             font: "Calibri" })],
  });
}

function buildTable(grid, fontHalfPt, landscape) {
  const total = landscape ? LANDSCAPE_W : PORTRAIT_W;
  const nCols = grid[0].length;
  const firstW = Math.round(total * Math.min(0.26, 1.9 / nCols));
  const restW = Math.floor((total - firstW) / (nCols - 1));
  const widths = [firstW, ...Array(nCols - 1).fill(restW)];
  widths[nCols - 1] += total - widths.reduce((a, b) => a + b, 0);

  const rows = grid.map((cells, r) =>
    new TableRow({
      tableHeader: r === 0,
      children: cells.map((text, c) =>
        new TableCell({
          width: { size: widths[c], type: WidthType.DXA },
          shading: r === 0
            ? { type: ShadingType.CLEAR, fill: "1F3864", color: "auto" }
            : (r % 2 === 0
                ? { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" }
                : undefined),
          margins: { top: 30, bottom: 30, left: 55, right: 55 },
          children: [new Paragraph({
            alignment: c === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
            spacing: { before: 0, after: 0 },
            children: [new TextRun({
              text: String(text),
              size: fontHalfPt,
              bold: r === 0,
              color: r === 0 ? "FFFFFF" : "000000",
              font: "Calibri",
            })],
          })],
        })),
    }));

  return new Table({
    columnWidths: widths,
    width: { size: total, type: WidthType.DXA },
    rows,
  });
}

function addTable(key) {
  const t = payload.tables[key];
  if (!t) { console.warn(`  ! missing table ${key}`); return; }
  push({ el: caption(t.label, t.caption, "Table") }, t.landscape);
  push({ el: buildTable(t.grid, t.font * 2, t.landscape) }, t.landscape);
  push({ el: p("", { spacing: { after: 200 } }) }, t.landscape);
  console.log(`  placed ${t.label}`);
}

function addFigure(label) {
  const f = payload.figures.find((x) => x.label === label);
  if (!f) { console.warn(`  ! missing figure ${label}`); return; }
  const maxW = 660;                                   // px at 96 dpi ~ 6.9 in
  const w = Math.min(maxW, f.width);
  const h = Math.round(w * f.height / f.width);
  push({
    el: new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 60 },
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(f.file),
        transformation: { width: w, height: h },
      })],
    }),
  });
  push({ el: caption(f.label, f.caption, "Figure") });
  console.log(`  placed ${f.label}`);
}

const text = (t, o) => push({ el: p(t, o) });
const h1 = (t) => push({ el: heading(t, HeadingLevel.HEADING_1) });
const h2 = (t) => push({ el: heading(t, HeadingLevel.HEADING_2) });
const bullet = (t) => push({
  el: new Paragraph({
    bullet: { level: 0 },
    spacing: { after: 80, line: 280 },
    children: [new TextRun({ text: t, size: 22, font: "Calibri" })],
  }),
});
const pageBreak = () => push({
  el: new Paragraph({ children: [new PageBreak()] }),
});

// --------------------------------------------------------------------------- //
// Title page
// --------------------------------------------------------------------------- //
push({
  el: new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 400, after: 200 },
    children: [new TextRun({
      text: "MOSAIC-GNN: a two-level multi-omic graph neural network for "
          + "pan-cancer prognosis and interpretable biomarker discovery",
      bold: true, size: 32, font: "Calibri",
    })],
  }),
});
text("[Author list]", { align: AlignmentType.CENTER, size: 22 });
text("[Affiliations]", { align: AlignmentType.CENTER, size: 20, italics: true });
text("[Corresponding author: name, address, e-mail]",
     { align: AlignmentType.CENTER, size: 20, italics: true });

push({ el: note(
  "DRAFT — READ BEFORE CIRCULATING. Every number in this document comes from "
  + "the synthetic demonstration cohorts shipped with the code, except Table 14, "
  + "which is a real-data check on TCGA-BLCA using a truncated export "
  + "(mutation and copy-number only, expression lost to Excel's 16,384-column "
  + "cap). The synthetic numbers demonstrate that the pipeline runs, that no "
  + "label leaks, and what each table and figure will look like. They are NOT "
  + "results and must be replaced by a run on the four real TCGA cohorts "
  + "before submission. Passages that depend on results not yet produced are "
  + "marked [TODO].") });

h1("Abstract");
text("Background. Multi-omic profiles of tumours are high-dimensional and "
   + "structured: genes interact, and patients resemble one another. Existing "
   + "graph neural networks for multi-omics exploit one of these structures but "
   + "not both, and are rarely evaluated in a way that separates genuine signal "
   + "from information leaking out of the label.");
text("Methods. We present MOSAIC-GNN, which couples two graphs. Each patient is "
   + "encoded as a graph of genes whose nodes carry mutation, copy-number, "
   + "expression and methylation channels and whose edges combine protein "
   + "interaction, pathway membership and training-fold co-alteration. "
   + "Attention coefficients are computed from each patient's own profile, so "
   + "every patient receives a personalised weighting of the shared network. "
   + "Genes are pooled into functional modules, and the resulting patient "
   + "embedding becomes a node in a learned patient-similarity graph. The latent "
   + "space is split by a gradient-reversal adversary into a cohort-invariant "
   + "and a cohort-specific part. Three heads are trained jointly: stage "
   + "classification, a Cox risk score, and a discrete-time hazard curve. "
   + "[TODO: cohort sizes and endpoint once the real run is complete.]");
text("Results. [TODO: replace with the real four-cohort run.] In the "
   + "demonstration reported here the model attained a held-out C-index of "
   + "0.571 against 21 comparators spanning classical predictors, multi-omic "
   + "fusion networks, patient-graph GNNs and dedicated survival models "
   + "(Tables 5-7). The generalisation gap was 0.018 in balanced accuracy and "
   + "0.081 in C-index, and a 20-fold label permutation test placed the null at "
   + "chance (Tables S1-S2), confirming that no label information reaches the "
   + "features. On real TCGA-BLCA data the survival head generalised "
   + "(C-index 0.614, gap 0.037) while stage classification carried no signal "
   + "from mutation and copy-number alone (Table 14).");
text("Conclusions. [TODO after the real run.]");
text("Keywords: multi-omics; graph neural network; pan-cancer; survival "
   + "analysis; TCGA; interpretability; biomarker discovery.",
     { size: 20, italics: true });

pageBreak();

// --------------------------------------------------------------------------- //
h1("1. Introduction");
text("[TODO: 4-6 paragraphs. Suggested arc, each with citations.]");
bullet("Multi-omic characterisation of tumours is now routine, but translating "
     + "it into prognosis remains difficult: features outnumber patients by two "
     + "orders of magnitude, and the informative structure is relational rather "
     + "than tabular.");
bullet("Two families of graph method have been applied. Patient-similarity "
     + "networks (GCN, GAT and variants) propagate information between similar "
     + "patients but treat the molecular profile as a flat vector. Gene- or "
     + "pathway-network methods respect biology but summarise each patient "
     + "independently, discarding the epidemiological neighbourhood.");
bullet("Neither family addresses the pan-cancer setting directly. A model "
     + "trained across cohorts can reach high accuracy simply by recognising "
     + "which cohort a sample came from, which does not transfer to a new "
     + "cancer type.");
bullet("Reporting practice compounds the problem: single train-test splits, "
     + "no permutation control, and clinical variables that encode the label "
     + "are all common, and each inflates performance.");
bullet("We therefore develop MOSAIC-GNN and evaluate it under a protocol built "
     + "to expose these failure modes rather than hide them.");
text("Contributions of this work:");
bullet("Two-level graph coupling: message passing on a gene interaction network "
     + "inside each patient, then on a learned patient-similarity graph between "
     + "patients.");
bullet("Patient-specific network rewiring, which yields an individual "
     + "subnetwork per patient as a by-product of the attention mechanism.");
bullet("Cohort-adversarial disentanglement, evaluated explicitly by a "
     + "leave-one-cancer-out protocol.");
bullet("A learned, task-optimised patient graph rather than a fixed k-nearest-"
     + "neighbour graph.");
bullet("Cross-omic contrastive alignment, so that the representation rests on "
     + "signal corroborated by more than one assay.");
bullet("Joint Cox and discrete-time hazard heads, giving both a risk ranking "
     + "and a calibrated individual survival curve.");
bullet("An evaluation protocol with 33 metrics per model, 21 comparators on "
     + "identical folds, paired bootstrap tests with multiplicity correction, "
     + "and a four-part overfitting audit including a label permutation test.");

// --------------------------------------------------------------------------- //
h1("2. Materials and Methods");

h2("2.1 Cohorts and data");
text("Four TCGA cohorts were obtained from cBioPortal: bladder urothelial "
   + "carcinoma (BLCA), liver hepatocellular carcinoma (LIHC), skin cutaneous "
   + "melanoma (SKCM) and thyroid carcinoma (THCA). Each cohort is a single wide "
   + "table combining clinical annotation with per-gene molecular features, "
   + "identified by column prefix: MUT_ for binary somatic mutation calls, CNA_ "
   + "for GISTIC copy-number calls in the range -2 to 2, and RNA_ or EXP_ and "
   + "METH_ where expression and methylation are available. A cohort missing a "
   + "modality is handled by zero-filling its channels and setting a per-block "
   + "availability mask, so the encoder distinguishes an unmeasured assay from a "
   + "measured value of zero. One row per patient is retained, preferring the "
   + "primary tumour sample. Cohort characteristics are given in Table 1 and "
   + "Figure 1.");
addTable("T1");
addFigure("Figure 1");

h2("2.2 Preprocessing and leakage control");
text("All fold-dependent statistics are estimated on training rows only: gene "
   + "selection, gene-graph construction, feature scaling and the survival bin "
   + "edges. Held-out rows pass through a transform that applies these fitted "
   + "statistics and nothing else.");
text("Genes are ranked per omic block on the training fold — mutation frequency "
   + "for MUT_, the proportion of altered samples for CNA_, and standard "
   + "deviation for continuous blocks — and a gene is retained when it ranks "
   + "highly in several cohorts, which is what makes the resulting graph "
   + "transferable rather than cohort-specific.");
text("Clinical columns that determine the label are excluded from the feature "
   + "matrix. For stage prediction this means the AJCC stage field itself and "
   + "the pathological T, N and M fields from which it is derived; analogous "
   + "exclusions apply to grade, cohort and subtype targets. This is enforced in "
   + "code and covered by a unit test. Its importance is not theoretical: in an "
   + "earlier version of this analysis the stage column was present as a "
   + "one-hot feature and every model, including simple logistic regression, "
   + "reached a balanced accuracy of 1.000.");

h2("2.3 Graph construction");
text("The gene graph is shared by all patients and carries three relation "
   + "types: prior edges from a protein-interaction list and pathway "
   + "co-membership, data-driven edges from a k-nearest-neighbour graph over "
   + "training-fold co-alteration correlations, and self-loops. When no external "
   + "prior is supplied the graph is estimated from co-alteration alone, so the "
   + "pipeline runs without network access. Functional modules come from a "
   + "supplied GMT file, or from a spectral partition of the gene graph when "
   + "none is given.");
text("The patient graph is learned. Similarity is computed in several learned "
   + "metric subspaces and sparsified with a straight-through top-k operator, so "
   + "gradients reach the metric although the forward pass is sparse. Held-out "
   + "patients may attach only to training anchors, never to one another, which "
   + "keeps the evaluation inductive.");

h2("2.4 Model architecture");
text("Node features are the per-gene omic values concatenated with their "
   + "availability masks. A relational graph-attention encoder with multiple "
   + "heads and per-relation parameters propagates information over the gene "
   + "graph; attention logits depend on the individual patient's values, so each "
   + "patient effectively re-weights the same interaction network. Genes are "
   + "then pooled into functional modules by an attention gate restricted to "
   + "module membership, and modules are pooled into a patient vector by a "
   + "second attention layer. Encoded clinical variables are concatenated and "
   + "the result is projected into a latent split into a shared and a specific "
   + "part. Message passing over the learned patient graph refines the latent. "
   + "Because the gene-level encoder cannot be run on all patients at once, "
   + "out-of-batch neighbours contribute through a bank of historical "
   + "embeddings.");

h2("2.5 Objective");
text("Training minimises a weighted sum of cross-entropy for the class label, "
   + "the Breslow Cox partial likelihood for the risk score, a right-censored "
   + "discrete-time hazard likelihood, a cohort adversary on the shared latent "
   + "through a gradient-reversal layer, a cohort head on the specific latent, "
   + "an orthogonality penalty between the two latents, a cross-omic InfoNCE "
   + "term, and smoothness, sparsity and degree regularisers on the learned "
   + "patient graph. [TODO: state the final weights used for the reported run.]");

h2("2.6 Overfitting control");
text("Because features outnumber patients by roughly two orders of magnitude, "
   + "regularisation is explicit. Whole genes are dropped at each step — value "
   + "and mask channel together, which is also the corruption the model must "
   + "survive when an assay was not run — Gaussian jitter is added to the value "
   + "channels, and dropout, edge dropout, weight decay, label smoothing and "
   + "gradient clipping are applied. Stochastic weight averaging over the final "
   + "40 % of epochs is retained only when it improves the validation "
   + "criterion. Early stopping monitors a criterion that weights both tasks so "
   + "neither is starved.");

h2("2.7 Comparators");
text("Twenty-one comparators were trained on identical folds, in four families: "
   + "classical predictors on concatenated features (L1, L2 and elastic-net "
   + "logistic regression, linear and RBF support vector machines, k-nearest "
   + "neighbours, naive Bayes, shrinkage LDA, random forest, extremely "
   + "randomised trees, histogram gradient boosting, XGBoost, LightGBM and "
   + "PCA followed by logistic regression); multi-omic fusion networks "
   + "(early-fusion MLP, late-fusion MLP, and a MOGONET-style per-omic graph "
   + "convolutional network with view fusion); patient-graph networks (GCN, GAT "
   + "and GraphSAGE on a fixed k-nearest-neighbour patient graph); and dedicated "
   + "survival models (Cox-lasso, Cox-ridge, a clinical-only Cox model, random "
   + "survival forest and DeepSurv).");

h2("2.8 Evaluation and statistics");
text("Repeated stratified k-fold cross-validation over the pooled pan-cancer "
   + "set, stratified by cohort, class and event. Every model is scored with an "
   + "identical panel: 17 classification metrics (accuracy, balanced accuracy, "
   + "macro and weighted F1, macro precision, sensitivity, specificity and "
   + "negative predictive value, AUROC, AUPRC, Matthews correlation, Cohen's "
   + "kappa in linear and quadratic form, top-2 accuracy, log loss, multiclass "
   + "Brier score and expected calibration error) and 16 survival metrics "
   + "(Harrell's C-index with bootstrap limits, time-dependent AUC at 12, 36 and "
   + "60 months, inverse-probability-of-censoring-weighted Brier scores at those "
   + "horizons, the integrated Brier score, the hazard ratio between "
   + "model-defined risk groups with its Wald interval, the hazard ratio per "
   + "standard deviation of the risk score, and the log-rank statistic and "
   + "p-value). Model comparisons use a paired bootstrap on pooled out-of-fold "
   + "predictions with Benjamini-Hochberg correction. Predictive uncertainty is "
   + "obtained by Monte-Carlo dropout at test time.");

// --------------------------------------------------------------------------- //
h1("3. Results");

h2("3.1 Cohort characteristics");
text("Cohort composition, follow-up and event rates are summarised in Table 1 "
   + "and Figure 1. [TODO: one paragraph describing the real cohorts — sizes, "
   + "event counts, median follow-up, class balance and any cohort where a "
   + "modality is missing.]");

h2("3.2 Performance on a held-out fold");
text("Table 2 reports the complete metric panel for MOSAIC-GNN on a single "
   + "held-out fold, Table 3 breaks performance down by class, and Table 4 by "
   + "cohort. Figure 2 shows the risk stratification: Kaplan-Meier curves for "
   + "model-defined tertiles with the log-rank test and the hazard ratio between "
   + "the outer tertiles, the mean predicted survival curves from the "
   + "discrete-time head, and the calibration of predicted against observed "
   + "survival at 36 months. Figure 3 shows the confusion matrix, one-vs-rest "
   + "ROC curves, the reliability diagram and the accuracy-coverage trade-off "
   + "when abstaining on uncertain cases.");
addTable("T2");
addTable("T2b");
addTable("T2c");
addFigure("Figure 2");
addFigure("Figure 3");

h2("3.3 The model does not overfit, and nothing leaks");
text("Table S1 reports identical metrics on the training, validation and "
   + "held-out splits together with an explicit gap row; Figure S1 shows where "
   + "the training and validation curves separate. Table S2 reports the label "
   + "permutation test: the model was retrained 20 times on shuffled training "
   + "labels and scored on the intact test split. The null distribution sat at "
   + "chance — balanced accuracy 0.261 against a chance value of 0.25, C-index "
   + "0.502 against 0.50 — and the observed AUROC and C-index both reached "
   + "p = 0.048, the smallest value attainable with 20 permutations. Table S3 "
   + "and Figure S2 give the sample-size curve.");
addTable("S1");
addFigure("Figure S1");
addTable("S2");
addTable("S3");
addFigure("Figure S2");
addTable("S4");
push({ el: note(
  "[TODO] Increase the permutation count to at least 100 for the submitted "
  + "version. With B permutations the smallest attainable p-value is 1/(B+1), "
  + "so the p = 0.048 reported here is at the resolution limit and a reviewer "
  + "will say so.") });

h2("3.4 Comparison with 21 established methods");
text("Tables 5 and 6 report classification and survival performance for every "
   + "model across folds; Table 7 gives the paired bootstrap comparison against "
   + "MOSAIC-GNN with Benjamini-Hochberg adjusted p-values; Figure 4 shows the "
   + "C-index with bootstrap intervals alongside balanced accuracy and AUROC.");
text("[TODO: rewrite once the real run is available.] In the demonstration run, "
   + "random survival forest reached the highest C-index (0.610) and MOSAIC-GNN "
   + "was second (0.571), ahead of the MOGONET-style fusion network (0.564), "
   + "DeepSurv (0.552) and the patient-graph networks (0.540-0.552). The "
   + "integrated Brier score is reported only for MOSAIC-GNN because it is the "
   + "only model in the comparison with a survival-curve head; the Cox-type "
   + "baselines produce a risk ranking and are therefore scored on C-index, "
   + "time-dependent AUC and hazard ratio only. This should be stated in the "
   + "table caption of the submitted version so that the blank cells are not "
   + "read as missing results.");
addTable("T5");
addTable("T6");
addTable("T7");
addFigure("Figure 4");

h2("3.5 Which components matter");
text("Tables 8 and 9 and Figure 5 report the ablation, in which each row "
   + "removes exactly one component under an otherwise identical protocol.");
text("[TODO: rewrite against the real run.] In the demonstration, removing the "
   + "gene graph cost the most (C-index 0.575 to 0.527) and removing module "
   + "pooling was next (0.536), which supports the two-level design. Removing "
   + "the patient graph (0.580) or the cross-omic contrastive term (0.597) "
   + "slightly improved performance at this sample size. If that pattern holds "
   + "on the real cohorts, those two components should be presented as optional "
   + "and their claims withdrawn rather than defended.");
addTable("T8");
addTable("T9");
addFigure("Figure 5");

h2("3.6 Transfer to an unseen cancer type");
text("Tables 10 and 11 and Figure 6 report leave-one-cancer-out transfer, in "
   + "which the model is trained on three cohorts and evaluated on the fourth.");
text("[TODO] In the demonstration, transfer performance was at chance for all "
   + "four held-out cohorts (C-index 0.518-0.535, all log-rank tests "
   + "non-significant), which is expected because the synthetic cohorts share "
   + "no biology by construction. This experiment is only meaningful on the "
   + "real data; if transfer is weak there too, the pan-cancer claim must be "
   + "softened to a multi-cohort claim.");
addTable("T10");
addTable("T11");
addFigure("Figure 6");

h2("3.7 What the model attends to");
text("Table 12 ranks genes by integrated-gradient attribution together with "
   + "their module assignment, mean attention gate and per-omic signed "
   + "contribution; Table 13 ranks functional modules by attention; Figure 7 "
   + "shows the top genes, the decomposition of attribution across omic layers, "
   + "and module attention by cohort. Table S5 lists the most strongly attended "
   + "edges of a single patient's rewired network.");
push({ el: note(
  "[TODO] Cross-check the ranked genes against COSMIC Cancer Gene Census and "
  + "OncoKB, report the overlap with known drivers as a validation of the "
  + "attribution, and present the remainder explicitly as hypotheses. Run "
  + "pathway enrichment on runs/genes.rnk and report the enriched terms with "
  + "FDR.") });
addTable("T12");
addTable("T13");
addFigure("Figure 7");
addTable("S5");

h2("3.8 Real-data check on TCGA-BLCA");
text("A single-cohort run on real TCGA-BLCA (401 patients, 176 events, 793 "
   + "genes retained, 13,417 gene-graph edges) used mutation and copy-number "
   + "only, because the available export had been truncated at Excel's "
   + "16,384-column limit and lost the expression block. Results are in "
   + "Table 14. The survival head discriminated and generalised (test C-index "
   + "0.614, train-test gap 0.037). Stage classification carried no signal: "
   + "balanced accuracy equalled chance and the Matthews correlation was zero, "
   + "meaning the classifier assigned a single class to every patient. This is "
   + "biologically unsurprising, since AJCC stage is an anatomical descriptor "
   + "that somatic mutation and copy-number profiles need not encode, and it "
   + "argues for making prognosis the primary endpoint of this work.");
addTable("R1");

// --------------------------------------------------------------------------- //
h1("4. Discussion");
text("[TODO: 5-7 paragraphs.] Suggested structure:");
bullet("What the two-level coupling buys, referenced to the ablation rather "
     + "than asserted.");
bullet("Why patient-graph-only methods plateau, and where the gene-level stage "
     + "adds information.");
bullet("An honest account of where MOSAIC-GNN does not lead. Random survival "
     + "forest is a strong baseline on cohorts of this size and should be "
     + "credited rather than dismissed.");
bullet("What the permutation test and gap analysis establish, and why they "
     + "should be routine in this literature.");
bullet("Biological reading of the attended modules, once cross-referenced.");
bullet("Clinical framing: what a calibrated survival curve adds over a risk "
     + "ranking, and what would be needed to use it.");

h1("5. Limitations");
bullet("Cohort sizes are in the hundreds, so survival confidence intervals are "
     + "wide and differences of a few points in C-index are not resolvable.");
bullet("Without an external protein-interaction prior the gene graph is "
     + "estimated from co-alteration and is not a validated interaction "
     + "network.");
bullet("Attention- and gradient-derived gene rankings are hypotheses; they "
     + "require orthogonal validation.");
bullet("Results reported here are internal to TCGA. An independent validation "
     + "cohort is required before any claim of clinical utility.");
bullet("The demonstration numbers in this draft come from synthetic cohorts and "
     + "a truncated single-cohort export; they characterise the pipeline, not "
     + "the biology.");

h1("6. Conclusion");
text("[TODO after the real run.]");

h1("Declarations");
text("Data availability. TCGA data are available from cBioPortal "
   + "(https://www.cbioportal.org) under the study identifiers "
   + "blca_tcga_pan_can_atlas_2018, lihc_tcga_pan_can_atlas_2018, "
   + "skcm_tcga_pan_can_atlas_2018 and thca_tcga_pan_can_atlas_2018.");
text("Code availability. The complete implementation, the executed analysis "
   + "notebook, all result tables and all figure-generating code are available "
   + "at [TODO: repository URL]. Fold indices, random seeds and the "
   + "configuration JSON for every reported run are included.");
text("Author contributions. [TODO]");
text("Funding. [TODO]");
text("Competing interests. [TODO]");
text("Ethics. This study used de-identified public data and required no "
   + "additional ethical approval. [TODO: confirm wording against the target "
   + "journal's policy.]");

h1("References");
text("[TODO] Suggested minimum coverage: TCGA Pan-Cancer Atlas; cBioPortal; "
   + "graph attention networks; MOGONET and comparable multi-omic fusion "
   + "methods; patient-similarity network methods; DeepSurv and random survival "
   + "forest; Harrell's C-index; Graf's integrated Brier score; "
   + "domain-adversarial training; InfoNCE; stochastic weight averaging; "
   + "integrated gradients; the \"attention is not explanation\" critique; "
   + "Benjamini-Hochberg.");

h1("Supplementary material");
bullet("Table S1 — generalisation gap across splits.");
bullet("Table S2 — label permutation test.");
bullet("Table S3 — sample-size curve.");
bullet("Table S4 — training and validation metrics per epoch.");
bullet("Table S5 — most strongly attended edges of one patient's network.");
bullet("Figure S1 — learning curves.");
bullet("Figure S2 — sample-size curves.");
bullet("Supplementary code — mosaic/ package, executed notebook, configuration "
     + "JSON and per-fold checkpoints.");

// --------------------------------------------------------------------------- //
// Group consecutive blocks by orientation into sections
// --------------------------------------------------------------------------- //
const sections = [];
let current = null;
for (const b of blocks) {
  if (!current || current.landscape !== b.landscape) {
    current = { landscape: b.landscape, children: [] };
    sections.push(current);
  }
  current.children.push(b.el);
}

const footer = new Footer({
  children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      children: ["MOSAIC-GNN draft — page ", PageNumber.CURRENT, " of ",
                 PageNumber.TOTAL_PAGES],
      size: 16, color: "808080", font: "Calibri",
    })],
  })],
});

const doc = new Document({
  creator: "MOSAIC-GNN",
  title: "MOSAIC-GNN manuscript draft",
  styles: {
    default: {
      heading1: { run: { size: 28, bold: true, color: "1F3864" } },
      heading2: { run: { size: 24, bold: true, color: "2E5496" } },
    },
  },
  sections: sections.map((s) => ({
    properties: {
      page: {
        size: {
          width: 12240, height: 15840,
          orientation: s.landscape ? PageOrientation.LANDSCAPE
                                   : PageOrientation.PORTRAIT,
        },
        margin: { top: 720, right: 720, bottom: 720, left: 720 },
      },
    },
    footers: { default: footer },
    children: s.children,
  })),
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`\nwrote ${OUT} (${(buf.length / 1024).toFixed(0)} KB, `
            + `${sections.length} sections)`);
});
