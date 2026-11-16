"""Generate notebooks/mosaic_gnn.ipynb.

The notebook is kept in a generator so it stays diffable and can be regenerated
after the library changes; run `python scripts/build_notebook.py`.
"""

from __future__ import annotations

import json
import os

CELLS: list[tuple[str, str]] = []

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _package_blob_literal(width: int = 96) -> str:
    """Zip the mosaic package and return it as a wrapped base64 literal.

    Embedding the source in the notebook is what makes it runnable on Kaggle
    with nothing attached: the alternative (a companion dataset zip) failed
    repeatedly because the notebook and the zip drifted apart.
    """
    import base64
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, _, files in os.walk(os.path.join(REPO, "mosaic")):
            if "__pycache__" in folder:
                continue
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                full = os.path.join(folder, name)
                zf.write(full, os.path.relpath(full, REPO))
        # The synthetic-data generator ships too: the notebook calls it when
        # the real dataset is not mounted, so without it the fallback path
        # dies with a missing-file error instead of running.
        gen = os.path.join(REPO, "scripts", "make_synthetic.py")
        if os.path.isfile(gen):
            zf.write(gen, "scripts/make_synthetic.py")
    blob = base64.b64encode(buf.getvalue()).decode()
    lines = [blob[i: i + width] for i in range(0, len(blob), width)]
    return "\n".join(f'    "{line}"' for line in lines)


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# --------------------------------------------------------------------------- #
md(r"""
# MOSAIC-GNN — pan-cancer multi-omics graph neural network

**M**ulti-**O**mic **S**tructure-**A**daptive **I**nterpretable **C**ross-cancer GNN
on TCGA (BLCA, LIHC, SKCM, THCA).

Each patient is encoded as a *graph of genes* (mutation / CNA / expression /
methylation channels on nodes, PPI + pathway + co-alteration edges), pooled into
functional modules, and the resulting patient vector becomes a node in a
*learned patient-similarity graph*. A cohort-adversarial split of the latent
(`z_shared` vs `z_specific`) is what makes the representation transfer to a
held-out cancer type rather than memorising the cohort.

Heads: stage/grade/subtype classification, Cox risk, and a discrete-time hazard
curve — trained jointly.

### How to use this notebook
1. Run **Setup** and **Configuration**.
2. **Explore** the cohorts.
3. **Single fold** trains one model quickly and produces the figures.
4. **Full experiments** (CV, baselines, ablations, leave-one-cancer-out) are
   guarded by `RUN_FULL` — flip it on when you have GPU time.
5. **Interpretation** produces the gene / module tables and the paper figures.
""")

# --------------------------------------------------------------------------- #
md("## 1. Setup")

code("# The whole `mosaic` package, zipped and base64-encoded, so this notebook\n"
     "# is self-contained: no companion dataset, no upload, nothing to attach.\n"
     "MOSAIC_PKG_B64 = (\n" + _package_blob_literal() + "\n)")

code(r"""
import os, sys, subprocess, glob, warnings, json, base64, io, shutil, zipfile
warnings.filterwarnings("ignore")

# Set to True only when you are editing mosaic/ next to the notebook and want
# your local copy to win. Default False on purpose: /kaggle/working persists
# between sessions, so an older extraction left there would silently shadow the
# version embedded above and you would debug bugs that are already fixed.
USE_LOCAL_PACKAGE = False

DEST = "/kaggle/working/mosaic-pkg" if os.path.isdir("/kaggle/working") \
       else os.path.abspath("./mosaic-pkg")

PKG_ROOT = None
if USE_LOCAL_PACKAGE:
    PKG_ROOT = next((os.path.abspath(p) for p in (".", "..")
                     if os.path.isfile(os.path.join(p, "mosaic", "config.py"))),
                    None)

if PKG_ROOT is None:
    # Always wipe and re-extract: never trust what a previous run left behind.
    shutil.rmtree(DEST, ignore_errors=True)
    os.makedirs(DEST, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(base64.b64decode(MOSAIC_PKG_B64))).extractall(DEST)
    PKG_ROOT = DEST
    print("extracted embedded package ->", DEST)

# Drop anything already imported under the `mosaic` name, or a stale module
# object would survive the path change.
for name in [m for m in list(sys.modules) if m == "mosaic" or m.startswith("mosaic.")]:
    del sys.modules[name]
while PKG_ROOT in sys.path:
    sys.path.remove(PKG_ROOT)
sys.path.insert(0, PKG_ROOT)

# Fail loudly now rather than with a confusing traceback 40 minutes in.
assert os.path.isfile(os.path.join(PKG_ROOT, "mosaic", "config.py")), \
    "extraction produced no mosaic/config.py"
_layers = open(os.path.join(PKG_ROOT, "mosaic", "models", "layers.py")).read()
assert "autocast(device_type=z.device.type, enabled=False)" in _layers, \
    "loaded layers.py predates the mixed-precision dtype fix -- stale package"
print("package root:", PKG_ROOT)

for pkg in ("torch", "sklearn", "scipy", "pandas", "matplotlib"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        {"sklearn": "scikit-learn"}.get(pkg, pkg)], check=True)

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
""")

# --------------------------------------------------------------------------- #
md("""
## 2. Configuration

Everything that defines a run lives in one `Config` object, which is saved
alongside every checkpoint — so a result is always reproducible from its JSON.
""")

code(r"""
from mosaic.config import Config
from mosaic.pipeline import Experiment

DATA_ROOT = "/kaggle/input/datasets/jannatuljerin/tcga-capstone"
RUN_FULL  = False        # flip on for the full CV / baseline / ablation suite
OUT_DIR   = "/kaggle/working/runs" if os.path.isdir("/kaggle/working") else "./runs"

# If the real dataset is absent, fall back to schema-identical synthetic data
# so the notebook still runs end to end (CI, laptop, reviewer).
if not glob.glob(os.path.join(DATA_ROOT, "*_merged_full.csv")):
    DATA_ROOT = os.path.abspath("./synthetic")
    if not glob.glob(os.path.join(DATA_ROOT, "*_merged_full.csv")):
        subprocess.run([sys.executable,
                        os.path.join(PKG_ROOT, "scripts", "make_synthetic.py"),
                        "--out", DATA_ROOT, "--patients", "160",
                        "--genes", "300"], check=True)
    print("!! real dataset not found - running on SYNTHETIC data:", DATA_ROOT)

cfg = Config()
cfg.data.root             = DATA_ROOT
cfg.data.cohorts          = ["BLCA", "LIHC", "SKCM", "THCA"]
cfg.data.class_target     = "stage"      # stage | grade | cohort | subtype
cfg.data.survival_endpoint = "OS"        # OS | DSS | PFS | DFS
cfg.data.max_genes        = 2000 if torch.cuda.is_available() else 300
cfg.data.n_genes_per_block = cfg.data.max_genes // 2

# Optional biological priors -- supply these for sharper interpretation.
# cfg.data.ppi_edgelist   = "/kaggle/input/string-ppi/string_human_links.csv"
# cfg.data.pathway_gmt    = "/kaggle/input/msigdb/c2.cp.reactome.v2023.1.symbols.gmt"

cfg.train.device   = "cuda" if torch.cuda.is_available() else "cpu"
cfg.train.epochs   = 200 if torch.cuda.is_available() else 40
cfg.train.out_dir  = OUT_DIR
cfg.train.batch_size, cfg.train.full_batch = 32, False
cfg.tag = "main"

os.makedirs(OUT_DIR, exist_ok=True)
print(json.dumps({"root": cfg.data.root, "target": cfg.data.class_target,
                  "endpoint": cfg.data.survival_endpoint,
                  "max_genes": cfg.data.max_genes, "device": cfg.train.device,
                  "epochs": cfg.train.epochs}, indent=2))
""")

# --------------------------------------------------------------------------- #
md("""
## 3. Explore the cohorts

The loader scans headers first and only materialises the columns it needs, so a
100 000-column merged table does not have to fit in memory twice. Omic blocks
are detected by prefix (`MUT_`, `CNA_`, `RNA_`/`EXP_`, `METH_`); a cohort that
is missing a modality gets a zero-filled channel plus an availability mask, so
the encoder can tell "absent" from "zero".
""")

code(r"""
exp = Experiment(cfg)
exp.prepare()
sk = exp.skeleton

rows = []
for name, (a, b) in exp.offsets.items():
    coh = exp.cohorts[name]
    ev = np.nan_to_num(coh.event, nan=0)
    rows.append({
        "cohort": name,
        "patients": len(coh),
        "blocks": ", ".join(sorted(coh.blocks)),
        "genes_per_block": ", ".join(str(len(v.genes)) for _, v in
                                     sorted(coh.blocks.items())),
        "events": int(ev.sum()),
        "event_rate": round(float(ev.mean()), 3),
        "median_OS_months": round(float(np.nanmedian(coh.time)), 1),
        "labelled": int((coh.label >= 0).sum()),
    })
cohort_table = pd.DataFrame(rows)
display(cohort_table)

print("class distribution (pooled):",
      dict(zip(*np.unique(sk.label[sk.label >= 0], return_counts=True))))
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))

# class balance per cohort
lab = pd.DataFrame({"cohort": [sk.cohort_names[c] for c in sk.cohort_id],
                    "label": sk.label})
lab = lab[lab.label >= 0]
pd.crosstab(lab.cohort, lab.label).plot(kind="bar", stacked=True, ax=ax[0],
                                        colormap="viridis")
ax[0].set_title(f"{cfg.data.class_target} distribution"); ax[0].set_xlabel("")

# follow-up
ax[1].hist(sk.time[np.isfinite(sk.time)], bins=40, color="#4C72B0")
ax[1].set_title(f"{cfg.data.survival_endpoint} follow-up (months)")

# Kaplan-Meier per cohort
from mosaic.metrics import km_curve
for ci, name in enumerate(sk.cohort_names):
    m = sk.cohort_id == ci
    t, s = km_curve(sk.time[m], np.nan_to_num(sk.event[m], nan=0))
    ax[2].step(t, s, where="post", label=name)
ax[2].set_title("Kaplan-Meier by cohort"); ax[2].set_xlabel("months")
ax[2].set_ylim(0, 1.02); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------- #
md("""
## 4. One fold, end to end

`build_fold` re-fits **everything** that could leak — gene selection, the gene
graph, scalers, survival bin edges — on the training rows only. Held-out
patients are passed through `transform` and attach to training anchors in the
patient graph, never to each other.
""")

code(r"""
from sklearn.model_selection import StratifiedKFold
from mosaic.data.preprocess import stratification_key

strat = stratification_key(sk)
idx = np.arange(len(sk))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.train.seed)
train_idx, test_idx = next(iter(skf.split(idx, strat)))

inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.train.seed)
fit_i, val_i = next(inner.split(train_idx, strat[train_idx]))
fit_idx, val_idx = train_idx[fit_i], train_idx[val_i]
print(f"train={len(fit_idx)}  val={len(val_idx)}  test={len(test_idx)}")

tensors, prep, data = exp.build_fold(train_idx)
n_classes = max(int(sk.label.max()) + 1, 2)
print(tensors.graph.summary())
print("node feature tensor:", tuple(tensors.x.shape),
      "| clinical:", tuple(tensors.clinical.shape))
""")

code(r"""
%time model = exp.train_fold(tensors, fit_idx, val_idx, n_classes)
result = exp.evaluate_fold(model, tensors, test_idx, train_idx)

show = {k: v for k, v in result.items()
        if isinstance(v, (int, float)) and not k.startswith("_")}
display(pd.Series(show).to_frame("value").round(4))
print("\nper-cohort:")
display(pd.DataFrame(result["per_cohort"]).T.round(3))
""")

# --------------------------------------------------------------------------- #
md("### 4.1 Figure — risk stratification")

code(r"""
from mosaic.metrics import km_curve, logrank_test

pred = exp.predict(model, tensors, test_idx, anchor_idx=train_idx,
                   mc_samples=cfg.train.mc_dropout_samples)
t_te = tensors.time.numpy()[test_idx]
e_te = tensors.event.numpy()[test_idx]
y_te = tensors.label.numpy()[test_idx]

cuts = np.quantile(pred["risk"], [1/3, 2/3])
group = np.digitize(pred["risk"], cuts)
names = ["low risk", "intermediate", "high risk"]
colors = ["#2E7D32", "#F9A825", "#C62828"]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for g in range(3):
    m = group == g
    if m.sum() < 5:
        continue
    t, s = km_curve(t_te[m], e_te[m])
    ax[0].step(t, s, where="post", color=colors[g],
               label=f"{names[g]} (n={int(m.sum())})")
hi_lo = np.isin(group, [0, 2])
chi2, p = logrank_test(t_te[hi_lo], e_te[hi_lo], (group[hi_lo] == 2).astype(int))
ax[0].set_title(f"Model-defined risk tertiles\nlog-rank (high vs low) p = {p:.2e}")
ax[0].set_xlabel("months"); ax[0].set_ylabel("survival probability")
ax[0].set_ylim(0, 1.02); ax[0].legend(fontsize=8)

# predicted individual survival curves from the discrete-time hazard head
edges = np.concatenate([[0], prep.surv_bins])
for g, c in zip(range(3), colors):
    m = group == g
    if m.any():
        ax[1].step(edges[:pred["survival"].shape[1]],
                   pred["survival"][m].mean(0), where="post", color=c,
                   label=names[g])
ax[1].set_title("Predicted survival curves (hazard head)")
ax[1].set_xlabel("months"); ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig_km_risk.png", dpi=200); plt.show()
""")

md("### 4.2 Figure — classification quality and uncertainty")

code(r"""
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
m = y_te >= 0

ConfusionMatrixDisplay.from_predictions(
    y_te[m], pred["prob"][m].argmax(1), normalize="true", cmap="Blues",
    colorbar=False, ax=ax[0])
ax[0].set_title(f"{cfg.data.class_target} (row-normalised)")

for c in np.unique(y_te[m]):
    fpr, tpr, _ = roc_curve((y_te[m] == c).astype(int), pred["prob"][m][:, c])
    ax[1].plot(fpr, tpr, label=f"class {c} (AUC={auc(fpr, tpr):.2f})")
ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
ax[1].set_title("One-vs-rest ROC"); ax[1].legend(fontsize=8)

# selective prediction: MC-dropout entropy as an abstention signal
order = np.argsort(pred["entropy"][m])
correct = (pred["prob"][m].argmax(1) == y_te[m])[order]
cov = np.arange(1, len(order) + 1) / len(order)
ax[2].plot(cov, np.cumsum(correct) / np.arange(1, len(order) + 1))
ax[2].axhline(correct.mean(), ls="--", c="grey", lw=0.8, label="full coverage")
ax[2].set_xlabel("coverage (most confident first)"); ax[2].set_ylabel("accuracy")
ax[2].set_title("Selective prediction (MC dropout)"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig_classification.png", dpi=200)
plt.show()
""")

# --------------------------------------------------------------------------- #
md("""
## 5. Full experiments

These are the four tables the manuscript needs. They are slow — set
`RUN_FULL = True` in the configuration cell (GPU strongly recommended).
""")

code(r"""
cv = None
if RUN_FULL:
    cv = exp.run_cv()
    summary = pd.DataFrame(cv["summary"]).T[["mean", "std", "n"]]
    display(summary.round(4))
else:
    print("RUN_FULL is False - skipping repeated cross-validation.")
""")

code(r"""
baseline_table = None
if RUN_FULL:
    from mosaic.baselines import run_baseline_suite
    per_fold, per_fold_preds, per_fold_split = [], [], []
    for f, (tr, te) in enumerate(skf.split(idx, strat)):
        tn, _, dt = exp.build_fold(tr)
        print(f"baseline fold {f + 1}/5")
        m, pr = run_baseline_suite(
            tn.x.numpy(), tn.clinical.numpy(), dt.label, dt.time, dt.event,
            tr, te, n_classes, block_names=sorted(sk.blocks),
            seed=cfg.train.seed, device=str(exp.device))
        per_fold.append(m)
        per_fold_preds.append(pr)
        per_fold_split.append((tr, te))
    # Keep every metric each baseline reported, not just three: the comparison
    # table in the paper is the full panel, and averaging is done per metric so
    # a model that could not produce one (e.g. a survival-only model has no
    # accuracy) leaves a blank rather than a zero.
    all_metrics = sorted({m for f in per_fold for d in f.values() for m in d})
    baseline_table = pd.DataFrame({
        name: {metric: np.nanmean([f[name].get(metric, np.nan) for f in per_fold])
               for metric in all_metrics}
        for name in sorted(per_fold[0])}).T
    if cv is not None:
        baseline_table.loc["MOSAIC-GNN"] = {
            m: cv["summary"].get(m, {}).get("mean", np.nan) for m in all_metrics}
    baseline_table.to_csv(f"{OUT_DIR}/table3_comparison_full.csv")
    display(baseline_table[[c for c in ("balanced_accuracy", "auroc", "c_index")
                            if c in baseline_table]].round(3))
else:
    print("RUN_FULL is False - skipping baselines.")
""")

code(r"""
ablation_table = None
if RUN_FULL:
    from mosaic.cli import ABLATIONS
    from mosaic.config import Config as C
    rows = {}
    for name, override in ABLATIONS.items():
        mcfg = type(cfg.model)(**{**cfg.model.__dict__, **override})
        tcfg = type(cfg.train)(**{**cfg.train.__dict__, "n_repeats": 1})
        sub = C(data=cfg.data, graph=cfg.graph, model=mcfg, loss=cfg.loss,
                train=tcfg, tag=f"{cfg.tag}-{name}")
        print("ablation:", name)
        e = Experiment(sub); e.prepare()
        s = e.run_cv()["summary"]
        rows[name] = {k: s.get(k, {}).get("mean", np.nan)
                      for k in ("balanced_accuracy", "auroc", "c_index")}
    ablation_table = pd.DataFrame(rows).T
    display(ablation_table.round(3))
    (ablation_table - ablation_table.loc["full"]).drop("full").plot(
        kind="barh", figsize=(7, 4), title="Change vs full model (negative = component helps)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig_ablation.png", dpi=200); plt.show()
else:
    print("RUN_FULL is False - skipping ablations.")
""")

code(r"""
loco_table = None
if RUN_FULL:
    rows = {}
    for held in cfg.data.cohorts:
        a, b = exp.offsets[held]
        te = np.arange(a, b)
        tr = np.setdiff1d(idx, te)
        rng = np.random.default_rng(cfg.train.seed)
        va = rng.choice(tr, size=max(int(0.15 * len(tr)), 20), replace=False)
        ft = np.setdiff1d(tr, va)
        print("leave out:", held)
        tn, _, _ = exp.build_fold(tr)
        mdl = exp.train_fold(tn, ft, va, n_classes)
        r = exp.evaluate_fold(mdl, tn, te, tr)
        rows[held] = {k: r.get(k, np.nan) for k in
                      ("balanced_accuracy", "auroc", "c_index", "logrank_p")}
    loco_table = pd.DataFrame(rows).T
    display(loco_table.round(3))
else:
    print("RUN_FULL is False - skipping leave-one-cancer-out.")
""")

# --------------------------------------------------------------------------- #
md("""
## 6. Interpretation

Three complementary attributions, because attention on its own is not evidence:
module attention (which pathways drive the embedding), the within-module gene
gate (which genes dominate), and integrated gradients on the node features
(signed, per-omic, baseline-referenced).
""")

code(r"""
from mosaic.explain import (collect_attention, export_ranked_gene_list,
                            gene_importance_table, integrated_gradients,
                            module_importance_table, personalised_subnetwork)

att = collect_attention(model, tensors, test_idx, exp.device)
ig = integrated_gradients(model, tensors, test_idx[:min(len(test_idx), 128)],
                          exp.device, target="risk")
blocks = sorted(sk.blocks)

genes_df = gene_importance_table(tensors.graph.genes, tensors.graph.modules,
                                 tensors.graph.module_names, att["gene_gate"],
                                 ig, blocks)
modules_df = module_importance_table(tensors.graph.module_names,
                                     att["module_attention"],
                                     tensors.graph.modules, tensors.graph.genes,
                                     gate=att["gene_gate"])
genes_df.to_csv(f"{OUT_DIR}/gene_importance.csv", index=False)
modules_df.to_csv(f"{OUT_DIR}/module_importance.csv", index=False)
export_ranked_gene_list(genes_df, f"{OUT_DIR}/genes.rnk")   # feed to GSEA / g:Profiler

display(genes_df.head(25))
display(modules_df.head(10))
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

top = genes_df.head(25).iloc[::-1]
score_col = "ig_abs" if "ig_abs" in top else "gate_mean"
ax[0].barh(top["gene"], top[score_col], color="#4C72B0")
ax[0].set_title(f"Top genes by {score_col}"); ax[0].tick_params(labelsize=8)

# module attention per cohort -> which pathways matter for which cancer
coh_te = tensors.cohort.numpy()[test_idx]
mat = np.vstack([att["module_attention"][coh_te == c].mean(0)
                 for c in range(len(sk.cohort_names))
                 if (coh_te == c).sum() > 0])
keep = np.argsort(-mat.mean(0))[:30]
im = ax[1].imshow(mat[:, keep], aspect="auto", cmap="magma")
ax[1].set_yticks(range(mat.shape[0]))
ax[1].set_yticklabels([n for c, n in enumerate(sk.cohort_names)
                       if (coh_te == c).sum() > 0])
ax[1].set_xticks(range(len(keep)))
ax[1].set_xticklabels([tensors.graph.module_names[i] for i in keep],
                      rotation=90, fontsize=6)
ax[1].set_title("Module attention by cohort")
plt.colorbar(im, ax=ax[1], fraction=0.03)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig_interpretation.png", dpi=200)
plt.show()
""")

code(r"""
# One patient's rewired interaction network -- edges are that patient's own
# attention weights over the shared topology.
sub = personalised_subnetwork(model, tensors, int(test_idx[0]), exp.device,
                              top_edges=150)
sub.to_csv(f"{OUT_DIR}/subnetwork_patient0.csv", index=False)
display(sub.head(15))
print("export this to Cytoscape / networkx for the figure")
""")

# --------------------------------------------------------------------------- #
md("""
## 6.5 Overfitting, leakage and sample-size diagnostics

Four checks, because "the test score is high" is not evidence that a model
generalises:

* **Generalisation gap** — train vs validation vs test on the same fold. A gap
  that widens monotonically is memorisation.
* **Learning curves** — the training trace, to see *when* the gap opened.
* **Label permutation test** — retrain on shuffled labels. This is the only
  check that can show the absence of leakage: with the labels destroyed, a
  clean pipeline must fall to chance. If it does not, information about the
  label is reaching the model through a feature.
* **Sample-size curve** — performance against the number of training patients,
  which says whether the result is data-limited or model-limited.
""")

code(r"""
from mosaic.diagnostics import (generalisation_gap, learning_curves,
                                parameter_report, permutation_p_value,
                                permutation_test, sample_size_curve)

gap = generalisation_gap(exp, tensors, model,
                         {"train": fit_idx, "validation": val_idx,
                          "test": test_idx}, train_idx)
gap_df = pd.DataFrame(gap).T
gap_df.to_csv(f"{OUT_DIR}/tableS1_generalisation_gap.csv")
display(gap_df.round(4))

params = parameter_report(model, n_train=len(fit_idx))
print("parameters:", {k: round(v, 2) if isinstance(v, float) else v
                      for k, v in params.items()})

lc = learning_curves(model)
if lc is not None and len(lc):
    lc.to_csv(f"{OUT_DIR}/tableS1b_learning_curve.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for col in [c for c in lc.columns if "loss" in c]:
        ax[0].plot(lc.index, lc[col], label=col)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[0].set_title("Training trace")
    for col in [c for c in lc.columns if "loss" not in c]:
        ax[1].plot(lc.index, lc[col], label=col)
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("Validation score")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig_learning_curve.png", dpi=200)
    plt.show()
""")

code(r"""
# Label permutation: expensive, so it is guarded. Run it at least once for the
# write-up -- it is the single most convincing panel in the supplement.
perm_table = None
if RUN_FULL:
    perm_df = permutation_test(exp, tensors, fit_idx, val_idx, test_idx,
                               n_classes, n_permutations=10,
                               seed=cfg.train.seed)
    display(perm_df.round(4))

    # Compare the real fold against the shuffled-label null.
    observed = {k: v for k, v in result.items()
                if isinstance(v, (int, float)) and not k.startswith("_")}
    rows = {}
    for metric in ("balanced_accuracy", "auroc", "c_index"):
        if metric not in perm_df.columns or metric not in observed:
            continue
        null = perm_df[metric].dropna().to_numpy(dtype=float)
        if null.size == 0:
            continue
        rows[metric] = {
            "observed": observed[metric],
            "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)) if null.size > 1 else 0.0,
            "null_max": float(null.max()),
            "n_permutations": int(null.size),
            "p_value": permutation_p_value(observed[metric], null),
        }
    perm_table = pd.DataFrame(rows).T
    perm_table.to_csv(f"{OUT_DIR}/tableS2_permutation_test.csv")
    display(perm_table.round(4))
else:
    print("RUN_FULL is False - skipping the permutation test.")
""")

code(r"""
ss_df = None
if RUN_FULL:
    ss_df = sample_size_curve(exp, tensors, fit_idx, val_idx, test_idx,
                              n_classes, fractions=(0.25, 0.5, 0.75, 1.0),
                              seed=cfg.train.seed)
    ss_df.to_csv(f"{OUT_DIR}/tableS3_sample_size.csv")
    display(ss_df.round(4))
    # sample_size_curve indexes by n_train, so bring it back as a column.
    plot_df = ss_df.reset_index()
    cols = [c for c in plot_df.columns if c not in ("n_train", "fraction")]
    plot_df.plot(x="n_train", y=cols, marker="o", figsize=(7, 4),
                 title="Performance vs training-set size")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig_sample_size.png", dpi=200)
    plt.show()
else:
    print("RUN_FULL is False - skipping the sample-size curve.")
""")

# --------------------------------------------------------------------------- #
md("""
## 6.6 Export every table

Writes the CSVs that `scripts/build_manuscript.py` turns into the Word tables,
under the exact names it expects. Anything that was skipped is simply absent,
and the manuscript builder omits that table rather than inventing numbers.
""")

code(r"""
import os
os.makedirs(OUT_DIR, exist_ok=True)

# --- Table 1: cohort characteristics ---------------------------------------
cohort_table.set_index("cohort").to_csv(f"{OUT_DIR}/table1_cohorts.csv")

# --- Table 2: the full metric panel on the held-out fold -------------------
panel = {k: v for k, v in result.items()
         if isinstance(v, (int, float)) and not k.startswith("_")}
pd.Series(panel).to_frame("value").to_csv(f"{OUT_DIR}/table2_single_fold.csv")

# --- Table 2b / 2c: per class and per cohort ------------------------------
if "per_class" in result:
    pd.DataFrame(result["per_class"]).T.to_csv(f"{OUT_DIR}/table2b_per_class.csv")
if "per_cohort" in result:
    pd.DataFrame(result["per_cohort"]).T.to_csv(f"{OUT_DIR}/table2c_per_cohort.csv")

# --- Table 4: paired tests, MOSAIC-GNN against every baseline -------------
# A difference in means is not a result; the paired bootstrap on the same test
# patients is, and the p-values are corrected for the number of comparisons.
if RUN_FULL and baseline_table is not None:
    from sklearn.metrics import balanced_accuracy_score

    from mosaic.metrics import (benjamini_hochberg, concordance_index,
                                paired_bootstrap_test)

    # Fold 0 of the same StratifiedKFold is the primary fold, so `model`,
    # `test_idx` and the fold-0 baseline predictions all describe the same
    # patients -- which is what makes the test paired.
    tr0, te0 = per_fold_split[0]
    assert np.array_equal(te0, test_idx), "fold 0 must match the primary fold"
    ours_pred = exp.predict(model, tensors, test_idx, anchor_idx=train_idx)

    y_te = sk.label[test_idx]
    t_te = sk.time[test_idx]
    e_te = np.nan_to_num(sk.event[test_idx], nan=0.0)

    lab_ok = y_te >= 0
    surv_ok = np.isfinite(t_te) & (t_te > 0)

    def bacc(y, p):
        return balanced_accuracy_score(y, p.argmax(1))

    def cidx(packed, risk):
        return concordance_index(packed[:, 0], risk, packed[:, 1])

    rows = []
    for name, pr in per_fold_preds[0].items():
        row = {"model": name}
        if pr.get("prob") is not None and lab_ok.sum() > 1:
            r = paired_bootstrap_test(bacc, y_te[lab_ok],
                                      ours_pred["prob"][lab_ok],
                                      np.asarray(pr["prob"])[lab_ok],
                                      seed=cfg.train.seed)
            row.update({"bacc_delta": r["delta"], "bacc_p": r["p_value"],
                        "bacc_ci_low": r.get("ci_low"),
                        "bacc_ci_high": r.get("ci_high")})
        if pr.get("risk") is not None and surv_ok.sum() > 1:
            packed = np.column_stack([t_te[surv_ok], e_te[surv_ok]])
            r = paired_bootstrap_test(cidx, packed,
                                      ours_pred["risk"][surv_ok],
                                      np.asarray(pr["risk"])[surv_ok],
                                      seed=cfg.train.seed)
            row.update({"cindex_delta": r["delta"], "cindex_p": r["p_value"],
                        "cindex_ci_low": r.get("ci_low"),
                        "cindex_ci_high": r.get("ci_high")})
        rows.append(row)

    paired = pd.DataFrame(rows).set_index("model")
    # One correction over every test in the table, not per column.
    for col in [c for c in paired.columns if c.endswith("_p")]:
        q, rej = benjamini_hochberg(paired[col].tolist())
        paired[col.replace("_p", "_q")] = q
        paired[col.replace("_p", "_sig")] = rej
    paired.to_csv(f"{OUT_DIR}/table4_paired_tests.csv")
    display(paired.round(4))

# --- Tables 5 / 6: ablation and leave-one-cancer-out ----------------------
if ablation_table is not None:
    ablation_table.to_csv(f"{OUT_DIR}/table5_ablation.csv")
if loco_table is not None:
    loco_table.to_csv(f"{OUT_DIR}/table6_loco.csv")

# --- Tables 7 / 8 / 9: interpretation ------------------------------------
genes_df.head(40).set_index(genes_df.columns[0]).to_csv(
    f"{OUT_DIR}/table7_gene_importance.csv")
modules_df.head(20).set_index(modules_df.columns[0]).to_csv(
    f"{OUT_DIR}/table8_module_importance.csv")
sub.head(25).to_csv(f"{OUT_DIR}/table9_patient_subnetwork.csv")

print("tables written to", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".csv"):
        print("  ", f)
""")

# --------------------------------------------------------------------------- #
md("""
## 7. Before you write it up

- Report **mean ± sd over ≥ 3 repeats × 5 folds**, never a single split.
- Tune baselines on the *same* inner splits as the model.
- Give a **paired bootstrap p-value** for every headline comparison
  (`mosaic.metrics.paired_bootstrap_test`).
- Show **per-cohort** numbers next to the pooled pan-cancer number.
- Report the **leave-one-cancer-out** result even where it is weak — it is the
  honest measure of "pan-cancer".
- Cross-check the top-ranked genes against COSMIC CGC / OncoKB, and present the
  novel ones as **hypotheses**, not findings.
- State the limitations explicitly: cohort sizes in the hundreds → wide survival
  CIs; the co-alteration fallback graph is not a validated interaction network;
  attention-derived rankings need orthogonal validation.
""")


# --------------------------------------------------------------------------- #
def build() -> dict:
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "notebooks", "mosaic_gnn.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(build(), fh, indent=1)
    print(f"wrote {out}  ({len(CELLS)} cells)")
