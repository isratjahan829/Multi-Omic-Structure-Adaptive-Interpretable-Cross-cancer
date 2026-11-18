"""Generate the Blackbook's Chapter 4 figures from the real run outputs.

    python scripts/make_figures.py --runs runs_pan_cohort --out figures/

Every figure is drawn from a CSV or JSON a run actually wrote. A panel whose
input did not run is not drawn and not faked: the figure is skipped and the
caller is told, so the document can mark it as absent rather than display a
plausible picture of nothing.

Design rules followed here (see the dataviz skill):
  * categorical hues assigned in fixed order from a validated palette, never
    cycled, never chosen per chart;
  * one axis per chart -- no dual-axis anywhere;
  * thin marks, recessive grid and axes, direct labels rather than a number on
    every point;
  * a legend whenever two or more series share a panel.

The palette's light-mode aqua/yellow/magenta sit below 3:1 contrast on a white
surface, so every series that uses them carries a visible direct label. That is
the relief rule, not an optional nicety: without it those marks are unreadable
for some readers and in greyscale print, which is how a thesis is examined.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Validated categorical slots, in order. Slots 1-3 clear the all-pairs gate in
# both modes, which is why no figure here puts more than three series on one
# set of axes without faceting.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
C4, C8 = "#eda100", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8982"
GRID = "#e3e2de"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def tidy(ax, bottom=True, left=True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_visible(left)


def read_csv(p: str, index_col=0) -> Optional[pd.DataFrame]:
    if not os.path.isfile(p):
        return None
    try:
        d = pd.read_csv(p, index_col=index_col)
    except Exception:
        return None
    return None if d.empty else d


def read_json(p: str):
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
def fig_cohorts(t1: pd.DataFrame, out: str) -> str:
    """4.1 Cohort composition: size, events, follow-up."""
    d = t1.sort_values("patients", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    ax = axes[0]
    ax.barh(range(len(d)), d["patients"], color=C1, height=0.72)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.index, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("patients")
    ax.set_title("(a) Cohort size")
    ax.grid(axis="y", visible=False)
    tidy(ax)

    ax = axes[1]
    er = d["event_rate"].astype(float)
    ax.barh(range(len(d)), er, color=C2, height=0.72)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.index, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("deaths / patients")
    ax.set_title("(b) Event rate")
    ax.grid(axis="y", visible=False)
    # Direct-label the extremes; a number on every bar would be noise.
    for i, v in enumerate(er):
        if v == er.max() or v == er.min():
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=6.5,
                    color=INK2)
    tidy(ax)

    ax = axes[2]
    if "median_followup_m" in d.columns:
        fu = d["median_followup_m"].astype(float)
        ax.scatter(fu, d["events"].astype(float), s=26, color=C3,
                   edgecolor="white", linewidth=0.8, zorder=3)
        # Relief rule: aqua is below 3:1 on white, so label the points.
        for name, x, y in zip(d.index, fu, d["events"].astype(float)):
            if y > d["events"].astype(float).quantile(0.88) or x > fu.quantile(0.93):
                ax.annotate(name, (x, y), textcoords="offset points",
                            xytext=(4, 3), fontsize=6.5, color=INK2)
        ax.set_xlabel("median follow-up (months)")
        ax.set_ylabel("deaths")
        ax.set_title("(c) Follow-up vs events")
    tidy(ax)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_gap(gap: pd.DataFrame, out: str) -> str:
    """4.2 Generalisation: train / validation / test side by side."""
    rows = [r for r in ("train", "validation", "test") if r in gap.index]
    mets = [m for m in ("balanced_accuracy", "auroc", "c_index")
            if m in gap.columns]
    fig, axes = plt.subplots(1, len(mets), figsize=(3.6 * len(mets), 4.0))
    if len(mets) == 1:
        axes = [axes]
    colors = [C1, C2, C3]
    for ax, m in zip(axes, mets):
        vals = [float(gap.at[r, m]) for r in rows]
        bars = ax.bar(range(len(rows)), vals, color=colors[:len(rows)],
                      width=0.62)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[:5] for r in rows])
        ax.set_ylim(0, max(1.0, max(vals) * 1.15))
        ax.set_title(m.replace("_", " "))
        ax.grid(axis="x", visible=False)
        for b, v in zip(bars, vals):           # direct labels, relief rule
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", fontsize=7.5, color=INK)
        tidy(ax)
    fig.suptitle("Identical metrics on training, validation and held-out splits",
                 fontsize=10, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_per_class(t2b: pd.DataFrame, out: str) -> str:
    """4.3 Per-class behaviour on the held-out fold."""
    cols = [c for c in ("f1", "precision", "recall") if c in t2b.columns]
    if not cols:
        return ""
    d = t2b.sort_values(cols[0], ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [2.0, 1.0]})

    ax = axes[0]
    n = len(d)
    w = 0.8 / len(cols)
    for i, (c, col) in enumerate(zip(cols, [C1, C2, C3])):
        ax.bar([x + i * w for x in range(n)], d[c].astype(float), width=w,
               color=col, label=c)
    ax.set_xticks([x + w for x in range(n)])
    ax.set_xticklabels(d.index, rotation=90, fontsize=6.5)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) Per-class scores on the held-out fold")
    ax.legend(ncol=len(cols), loc="upper right")
    ax.grid(axis="x", visible=False)
    tidy(ax)

    ax = axes[1]
    if "support" in d.columns:
        sup = d["support"].astype(float)
        f1 = d[cols[0]].astype(float)
        ax.scatter(sup, f1, s=26, color=C1, edgecolor="white", linewidth=0.8,
                   zorder=3)
        for name, x, y in zip(d.index, sup, f1):
            if y < f1.quantile(0.12) or x > sup.quantile(0.92):
                ax.annotate(name, (x, y), textcoords="offset points",
                            xytext=(4, 3), fontsize=6.5, color=INK2)
        ax.set_xlabel("support (patients in class)")
        ax.set_ylabel(cols[0])
        ax.set_title("(b) Score against class size")
    tidy(ax)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_learning(lc: pd.DataFrame, out: str) -> str:
    """4.4 Learning curves."""
    x = lc.index.astype(float)
    pairs = [(a, b) for a, b in (("train_balanced_accuracy",
                                  "val_balanced_accuracy"),
                                 ("train_c_index", "val_c_index"))
             if a in lc.columns and b in lc.columns]
    if not pairs:
        cand = [c for c in lc.columns if lc[c].dtype.kind in "fc"][:2]
        if not cand:
            return ""
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for c, col in zip(cand, [C1, C2]):
            ax.plot(x, lc[c].astype(float), color=col, linewidth=2, label=c)
        ax.set_xlabel("epoch")
        ax.legend()
        tidy(ax)
        fig.tight_layout(); fig.savefig(out); plt.close(fig)
        return out

    fig, axes = plt.subplots(1, len(pairs), figsize=(6.4 * len(pairs), 4.2))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (a, b) in zip(axes, pairs):
        ax.plot(x, lc[a].astype(float), color=C1, linewidth=2, label="train")
        ax.plot(x, lc[b].astype(float), color=C2, linewidth=2,
                label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(a.replace("train_", "").replace("_", " "))
        ax.set_title(a.replace("train_", "").replace("_", " "))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        tidy(ax)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_comparison(t3: pd.DataFrame, mosaic: Dict[str, float], out: str) -> str:
    """4.6 Model comparison: the proposed model against every baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    for ax, metric, label in ((axes[0], "balanced_accuracy",
                               "balanced accuracy"),
                              (axes[1], "c_index", "C-index")):
        if metric not in t3.columns:
            continue
        s = t3[metric].dropna().sort_values()
        mv = mosaic.get(metric)
        ax.barh(range(len(s)), s.values, color=C1, height=0.68)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=7)
        ax.set_xlabel(label)
        # MOSAIC-GNN is a reference line, not a bar: it is not one of the
        # baselines and drawing it as one invites reading it as ranked. It
        # takes categorical slot 2 because it is an entity; the status red is
        # reserved for good/warning/serious/critical and means none of those
        # here.
        hi = float(s.max())
        if mv is not None:
            hi = max(hi, mv)
            ax.axvline(mv, color=C2, linewidth=2, zorder=4)
        # Don't pad out to 1.0: on C-index the whole panel would be half
        # empty and the differences that matter would be squeezed.
        ax.set_xlim(0, hi * 1.16)
        if mv is not None:
            ax.text(mv + hi * 0.012, len(s) - 0.6,
                    f"MOSAIC-GNN {mv:.3f}", color=C2, fontsize=8,
                    fontweight="bold", va="top", ha="left")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"({'a' if metric.startswith('bal') else 'b'}) {label}")
        tidy(ax)

    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_ablation(abl: dict, out: str) -> str:
    """4.7 Ablation: change in each metric when one component is removed."""
    if "full" not in abl or len(abl) < 2:
        return ""
    base = abl["full"]
    mets = [m for m in ("balanced_accuracy", "auroc", "c_index")
            if m in base]
    names = [k for k in abl if k != "full"]
    fig, ax = plt.subplots(figsize=(max(6.4, 1.6 * len(names) + 3.4), 4.8))
    # Keep bars narrow when few variants are present, so two arms do not
    # render as two slabs filling the panel.
    group = min(0.8, 0.26 * len(mets))
    w = group / len(mets)
    off = (1 - group) / 2
    for i, (m, col) in enumerate(zip(mets, [C1, C2, C3])):
        deltas = [abl[n][m]["mean"] - base[m]["mean"] for n in names]
        ax.bar([x + off + i * w for x in range(len(names))], deltas, width=w,
               color=col, label=m.replace("_", " "))
    ax.axhline(0, color=INK2, linewidth=1)
    # The noise floor, drawn, because a bar inside it is not an effect.
    ax.axhspan(-0.01, 0.01, color=MUTED, alpha=0.16, zorder=0)
    ax.text(0.995, 0.985, "shaded band = ±0.01 noise floor",
            transform=ax.transAxes, fontsize=7.5, color=INK2, ha="right",
            va="top")
    ax.set_xticks([x + off + group / 2 for x in range(len(names))])
    ax.set_xticklabels([n.replace("no_", "−") for n in names], fontsize=8.5)
    ax.set_xlim(-0.1, len(names) - 0.1)
    ax.set_ylabel("change vs full model")
    ax.set_title("Effect of removing one component "
                 "(negative = component helps)")
    # Legend above the plot: inside it collides with whichever bar runs
    # deepest, and that bar is the one the reader most needs to see.
    ax.legend(ncol=len(mets), loc="lower center",
              bbox_to_anchor=(0.5, 1.045), borderaxespad=0)
    ax.grid(axis="x", visible=False)
    tidy(ax)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_interpretation(genes: pd.DataFrame, mods: Optional[pd.DataFrame],
                       out: str) -> str:
    """4.9 Interpretation: attribution by gene, by module, and by omic block."""
    n = 2 if mods is None else 3
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.0))

    gcol = "ig_abs" if "ig_abs" in genes.columns else genes.columns[-1]
    g = genes.sort_values(gcol, ascending=False).head(15)[::-1]
    name = g.index if g.index.name else g.iloc[:, 0]
    ax = axes[0]
    labels = list(g.index.astype(str)) if g.index.name else list(
        g.iloc[:, 0].astype(str))
    ax.barh(range(len(g)), g[gcol].astype(float), color=C1, height=0.7)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("|integrated gradient|")
    ax.set_title("(a) Top genes")
    ax.grid(axis="y", visible=False)
    tidy(ax)

    ax = axes[1]
    blocks = [c for c in ("ig_mut", "ig_cna", "ig_exp") if c in genes.columns]
    if blocks:
        vals = [genes[c].abs().mean() for c in blocks]
        tot = sum(vals) or 1.0
        bars = ax.bar(range(len(blocks)), vals, color=[C1, C2, C3][:len(blocks)],
                      width=0.62)
        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels([b.replace("ig_", "") for b in blocks])
        ax.set_ylabel("mean |attribution|")
        ax.set_title("(b) Attribution by omic block")
        for b, v in zip(bars, vals):           # relief rule + the headline
            ax.text(b.get_x() + b.get_width() / 2, v, f"{100*v/tot:.1f}%",
                    ha="center", va="bottom", fontsize=8, color=INK)
        ax.grid(axis="x", visible=False)
        tidy(ax)

    if mods is not None and n == 3:
        mcol = [c for c in mods.columns if mods[c].dtype.kind in "fc"]
        ax = axes[2]
        if mcol:
            m = mods.sort_values(mcol[0], ascending=False).head(12)[::-1]
            ax.barh(range(len(m)), m[mcol[0]].astype(float), color=C2,
                    height=0.7)
            ax.set_yticks(range(len(m)))
            ax.set_yticklabels(m.index.astype(str), fontsize=7)
            ax.set_xlabel(mcol[0].replace("_", " "))
            ax.set_title("(c) Top modules")
            ax.grid(axis="y", visible=False)
        tidy(ax)

    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_pan_cohort")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--tag", default="cohort")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R, O = args.runs, args.out
    made, missing = [], []

    def record(name: str, path: str) -> None:
        (made if path else missing).append(name)
        if path:
            print(f"  wrote {path}")

    t1 = read_csv(f"{R}/table1_cohorts.csv")
    if t1 is not None:
        record("4.1 cohorts", fig_cohorts(t1, f"{O}/fig1_cohorts.png"))
    else:
        missing.append("4.1 cohorts")

    gap = read_csv(f"{R}/tableS1_generalisation_gap.csv")
    if gap is not None:
        record("4.2 generalisation", fig_gap(gap, f"{O}/fig2_risk.png"))
    else:
        missing.append("4.2 generalisation")

    t2b = read_csv(f"{R}/table2b_per_class.csv")
    if t2b is not None:
        record("4.3 per class",
               fig_per_class(t2b, f"{O}/fig3_classification.png"))
    else:
        missing.append("4.3 per class")

    lc = read_csv(f"{R}/tableS1b_learning_curve.csv")
    if lc is not None:
        record("4.4 learning curves",
               fig_learning(lc, f"{O}/figS1_learning_curves.png"))
    else:
        missing.append("4.4 learning curves")

    t3 = read_csv(f"{R}/table3_comparison_full.csv")
    t2 = read_csv(f"{R}/table2_single_fold.csv")
    mosaic = {}
    if t2 is not None:
        col = t2.columns[0]
        for k in ("balanced_accuracy", "c_index"):
            if k in t2.index:
                mosaic[k] = float(t2.at[k, col])
    if t3 is not None:
        record("4.6 comparison",
               fig_comparison(t3, mosaic, f"{O}/fig4_comparison.png"))
    else:
        missing.append("4.6 comparison")

    abl = read_json(f"{R}/{args.tag}_ablation.json")
    if abl:
        record("4.7 ablation", fig_ablation(abl, f"{O}/fig5_ablation.png"))
    else:
        missing.append("4.7 ablation")

    genes = read_csv(f"{R}/table7_gene_importance.csv", index_col=0)
    mods = read_csv(f"{R}/table8_module_importance.csv", index_col=0)
    if genes is not None:
        record("4.9 interpretation",
               fig_interpretation(genes, mods, f"{O}/fig7_interpretation.png"))
    else:
        missing.append("4.9 interpretation")

    print(f"\nmade {len(made)}: {', '.join(made)}")
    if missing:
        print(f"NOT DRAWN {len(missing)}: {', '.join(missing)}")
        print("Those phases did not run. The figure is absent rather than "
              "invented; mark it as not run in the document.")


if __name__ == "__main__":
    main()
