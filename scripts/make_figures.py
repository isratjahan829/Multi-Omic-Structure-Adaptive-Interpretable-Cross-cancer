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

import numpy as np
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

# The measured run-to-run spread of the pipeline under a fixed seed. Drawn on
# the ablation panels because an arm inside it is not an effect.
FLOOR = 0.01

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
        # Bars encode magnitude from zero, which is right for balanced
        # accuracy -- chance is 1/32 there, so zero is close to the floor of
        # what the metric can mean. It is wrong for the C-index, whose floor
        # is 0.5: half that axis is unreachable, so bars from zero make
        # 0.734 and 0.779 look identical. Dots carry no area, so the axis can
        # start where the metric does, with the random value marked.
        dots = metric == "c_index"
        if dots:
            ax.scatter(s.values, range(len(s)), s=46, color=C1,
                       edgecolor="white", linewidth=0.9, zorder=3)
            ax.axvline(0.5, color=INK2, linewidth=1.2, zorder=2)
        else:
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
        if dots:
            lo = min(float(s.min()), mv if mv is not None else 1.0)
            span = max(hi - 0.5, 0.02)
            ax.set_xlim(min(0.5 - 0.04 * span, lo - 0.06 * span),
                        hi + 0.22 * span)
            ax.text(0.5, -0.9, "0.5 = random ranking", fontsize=7.5,
                    color=INK2, ha="center", va="top")
        else:
            # Don't pad out to 1.0: the panel would be half empty.
            ax.set_xlim(0, hi * 1.16)
        if mv is not None:
            pad = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
            # Put the label where the panel is empty. The bars are sorted
            # ascending, so in the bar panel the free space is beside the
            # shortest bar at the bottom; in the dot panel it is at the top.
            yy, va = ((len(s) - 0.6, "top") if dots else (-0.2, "bottom"))
            ax.text(mv + pad, yy, f"MOSAIC-GNN {mv:.3f}", color=C2,
                    fontsize=8, fontweight="bold", va=va, ha="left")
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

    # Two panels. On one scale the clinical arms are so large that the five
    # structural arms become invisible, and whether those clear the noise
    # floor is the question the section actually asks. Panel (b) therefore
    # repeats the arms whose largest effect is within ten noise floors --
    # a rule applied to the data, not a hand-picked subset.
    def worst(n):
        return max(abs(abl[n][m]["mean"] - base[m]["mean"]) for m in mets)

    small = [n for n in names if worst(n) <= 10 * FLOOR]
    two = len(small) >= 2 and len(small) < len(names)
    fig, axes = plt.subplots(
        1, 2 if two else 1,
        figsize=((max(6.4, 1.1 * len(names) + 2.0)
                  + max(4.2, 1.0 * len(small) + 1.6)) if two
                 else max(6.4, 1.6 * len(names) + 3.4), 4.8),
        gridspec_kw={"width_ratios": [len(names), max(2, len(small))]}
        if two else None)
    panels = list(axes) if two else [axes]
    ax = panels[0]
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
    ax.axhspan(-FLOOR, FLOOR, color=MUTED, alpha=0.16, zorder=0)
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
    if two:
        ax.set_title("(a) Every arm, on one scale")
    tidy(ax)

    if two:
        ax2 = panels[1]
        for i, (m, col) in enumerate(zip(mets, [C1, C2, C3])):
            deltas = [abl[n][m]["mean"] - base[m]["mean"] for n in small]
            bars = ax2.bar([x + off + i * w for x in range(len(small))],
                           deltas, width=w, color=col)
            if m == mets[0]:
                # Direct labels on the headline metric only: with fifteen
                # bars a number on each would be noise, and this is the
                # series the section's argument is made on.
                for bar, d in zip(bars, deltas):
                    ax2.text(bar.get_x() + bar.get_width() / 2,
                             d - 0.0016, f"{d:.4f}", ha="center", va="top",
                             fontsize=6.8, color=INK)
        ax2.axhline(0, color=INK2, linewidth=1)
        ax2.axhspan(-FLOOR, FLOOR, color=MUTED, alpha=0.16, zorder=0)
        lo = min(abl[n][m]["mean"] - base[m]["mean"]
                 for n in small for m in mets)
        ax2.set_ylim(min(lo * 1.45, -2.2 * FLOOR), max(1.6 * FLOOR, 0.004))
        ax2.set_xticks([x + off + group / 2 for x in range(len(small))])
        ax2.set_xticklabels([n.replace("no_", "−") for n in small],
                            fontsize=8.5)
        ax2.set_xlim(-0.1, len(small) - 0.1)
        ax2.set_title("(b) The same arms within ten noise floors, enlarged")
        ax2.grid(axis="x", visible=False)
        ax2.text(0.985, 0.03, "a bar inside the band is not an effect",
                 transform=ax2.transAxes, fontsize=7.5, color=INK2,
                 ha="right", va="bottom")
        tidy(ax2)

    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_samplesize(ss: pd.DataFrame, out: str) -> str:
    """4.5 Held-out performance against training-set size.

    Train and test are drawn on each panel, because the interesting feature
    is how the gap closes, not the test curve alone.
    """
    x = ss.index.astype(float)
    pairs = [("balanced accuracy", "train_balanced_accuracy",
              "test_balanced_accuracy"),
             ("C-index", "train_c_index", "test_c_index")]
    pairs = [p for p in pairs if p[1] in ss.columns and p[2] in ss.columns]
    if not pairs:
        return ""
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.6 * len(pairs), 4.6))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (label, tr, te) in zip(axes, pairs):
        ax.plot(x, ss[tr].astype(float), color=C2, linewidth=2,
                marker="o", markersize=5, label="training")
        ax.plot(x, ss[te].astype(float), color=C1, linewidth=2,
                marker="o", markersize=5, label="held-out")
        # Direct-label the endpoints rather than every point.
        ax.annotate(f"{float(ss[te].iloc[-1]):.3f}", (x[-1], ss[te].iloc[-1]),
                    textcoords="offset points", xytext=(-6, -14),
                    fontsize=8, color=C1, fontweight="bold")
        ax.annotate(f"{float(ss[te].iloc[0]):.3f}", (x[0], ss[te].iloc[0]),
                    textcoords="offset points", xytext=(2, -14),
                    fontsize=8, color=C1)
        ax.set_xlabel("training patients")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        tidy(ax)
    fig.suptitle("Held-out performance against training-set size",
                 fontsize=10, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_loco(loco: dict, t1: pd.DataFrame, out: str) -> str:
    """4.8 Leave-one-cancer-out survival transfer.

    Only the survival column is drawn. The classification column is a
    single-class metric and is not a transfer measurement -- see Section 4.6.
    """
    rows = []
    for name, r in loco.items():
        if name not in t1.index:
            continue
        rows.append((name, float(r["c_index"]), float(r["logrank_p"]),
                     int(t1.at[name, "events"]),
                     float(t1.at[name, "event_rate"])))
    if not rows:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) every cohort, ordered, with the evidence behind it visible.
    rows.sort(key=lambda r: r[1])
    ax = axes[0]
    # Two colours by whether the row is powered enough to read, which is the
    # distinction the text makes and the one a reader needs at a glance.
    colors = [C1 if r[3] >= 60 else MUTED for r in rows]
    # Dots, not bars. The C-index floor is 0.5, so a bar drawn from zero
    # gives UVM at 0.42 -- below random -- a substantial-looking length, and
    # squeezes the whole range that matters into the right-hand third.
    vals = [r[1] for r in rows]
    ax.hlines(range(len(rows)), 0.5, vals, color=colors, linewidth=1.6,
              alpha=0.55, zorder=2)
    ax.scatter(vals, range(len(rows)), s=42, c=colors, edgecolor="white",
               linewidth=0.8, zorder=3)
    ax.axvline(0.5, color=INK2, linewidth=1.4, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]}  ({r[3]})" for r in rows], fontsize=7)
    ax.set_xlabel("transfer C-index on the held-out cohort")
    lo, hi = min(vals), max(vals)
    pad = max(0.03, (hi - lo) * 0.12)
    ax.set_xlim(min(0.5, lo) - pad, hi + pad)
    ax.text(0.5, -0.9, "0.5 = random", fontsize=7.5, color=INK2,
            ha="center", va="top")
    ax.set_title("(a) Transfer by cohort, with event count")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C1, label="≥ 60 events"),
                       Patch(facecolor=MUTED, label="< 60 events")],
              loc="lower right")
    tidy(ax)

    # (b) the relationship the text reports.
    ax = axes[1]
    big = [r for r in rows if r[3] >= 60]
    small = [r for r in rows if r[3] < 60]
    ax.scatter([r[4] for r in small], [r[1] for r in small], s=26,
               color=MUTED, edgecolor="white", linewidth=0.8, zorder=3,
               label="< 60 events")
    ax.scatter([r[4] for r in big], [r[1] for r in big], s=42, color=C1,
               edgecolor="white", linewidth=0.9, zorder=4,
               label="≥ 60 events")
    if len(big) >= 3:
        xs = np.array([r[4] for r in big]); ys = np.array([r[1] for r in big])
        b, a = np.polyfit(xs, ys, 1)
        gx = np.linspace(xs.min(), xs.max(), 20)
        ax.plot(gx, a + b * gx, color=C2, linewidth=2, zorder=5)
    for r in big:            # relief rule: label the extremes, not every point
        if r[4] > 0.6 or r[1] > 0.70:
            ax.annotate(r[0], (r[4], r[1]), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, color=INK2)
    ax.axhline(0.5, color=INK2, linewidth=1)
    ax.set_xlabel("event rate of the held-out cohort")
    ax.set_ylabel("transfer C-index")
    ax.set_title("(b) Transfer falls as the cohort's event rate rises")
    ax.legend(loc="upper right")
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

    ss = read_csv(f"{R}/tableS3_sample_size.csv")
    if ss is not None:
        record("4.5 sample size",
               fig_samplesize(ss, f"{O}/figS2_sample_size.png"))
    else:
        missing.append("4.5 sample size")

    loco = read_json(f"{R}/{args.tag}_loco.json")
    if loco and t1 is not None:
        record("4.8 LOCO transfer", fig_loco(loco, t1, f"{O}/fig6_loco.png"))
    else:
        missing.append("4.8 LOCO transfer")

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
