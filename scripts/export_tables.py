"""Diagnostics and the manuscript table set, from a real run.

`python -m mosaic.cli {train,baseline,ablate,loco}` produces the JSON summaries
for cross-validation, the baseline panel, the ablation and the
leave-one-cancer-out transfer. This script consumes those, adds the four
overfitting / leakage diagnostics that only make sense on one concrete fold,
and writes the CSV table set the manuscript is built from.

The primary fold is fold 0 of ``StratifiedKFold(5, shuffle=True,
random_state=seed)`` on the pan-cancer patient set -- the same split
`mosaic.cli baseline` uses -- so the paired bootstrap in table 4 compares
MOSAIC-GNN against every baseline on exactly the same held-out patients.

    python scripts/export_tables.py --root ./data_real --target cohort \
        --runs runs_real_cohort --out runs_real_cohort --tag cohort \
        --max-genes 1500 --epochs 120 --batch-size 32 --permutations 10
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from mosaic.config import Config
from mosaic.data.preprocess import stratification_key
from mosaic.diagnostics import (generalisation_gap, learning_curves,
                                parameter_report, permutation_p_value,
                                permutation_test, sample_size_curve)
from mosaic.pipeline import Experiment

COHORTS = ["BLCA", "LIHC", "THCA", "LUNG"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json(path: str):
    if not os.path.isfile(path):
        log(f"  !! absent, table will be omitted: {path}")
        return None
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--runs", required=True, help="dir holding the CLI JSONs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-genes", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--permutations", type=int, default=10)
    ap.add_argument("--endpoint", default="OS")
    ap.add_argument("--skip-sample-size", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = Config()
    cfg.data.root = args.root
    cfg.data.cohorts = list(COHORTS)
    cfg.data.class_target = args.target
    cfg.data.survival_endpoint = args.endpoint
    cfg.data.max_genes = args.max_genes
    cfg.data.n_genes_per_block = max(args.max_genes // 2, 100)
    cfg.train.device = "cpu"
    cfg.train.epochs = args.epochs
    cfg.train.outer_folds = args.folds
    cfg.train.n_repeats = 1
    cfg.train.batch_size, cfg.train.full_batch = args.batch_size, False
    cfg.train.out_dir = args.out
    cfg.tag = args.tag

    exp = Experiment(cfg)
    exp.prepare()
    sk = exp.skeleton

    # ---- table 1: cohort characteristics --------------------------------- #
    rows = []
    for name in exp.offsets:
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
    cohort_table.set_index("cohort").to_csv(f"{args.out}/table1_cohorts.csv")
    log("table1_cohorts.csv")
    print(cohort_table.to_string(index=False), flush=True)
    labs, counts = np.unique(sk.label[sk.label >= 0], return_counts=True)
    log(f"pooled class distribution: {dict(zip(labs.tolist(), counts.tolist()))}")

    # ---- the primary fold ------------------------------------------------- #
    strat = stratification_key(sk)
    idx = np.arange(len(sk))
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                          random_state=cfg.train.seed)
    train_idx, test_idx = next(iter(skf.split(idx, strat)))
    inner = StratifiedKFold(n_splits=3, shuffle=True,
                            random_state=cfg.train.seed)
    fit_i, val_i = next(inner.split(train_idx, strat[train_idx]))
    fit_idx, val_idx = train_idx[fit_i], train_idx[val_i]
    log(f"primary fold: fit={len(fit_idx)} val={len(val_idx)} test={len(test_idx)}")

    tensors, _, data = exp.build_fold(train_idx)
    n_classes = max(int(sk.label.max()) + 1, 2)
    log(f"graph: {tensors.graph.summary()}  x={tuple(tensors.x.shape)}")

    log("training the primary-fold model")
    model = exp.train_fold(tensors, fit_idx, val_idx, n_classes)
    result = exp.evaluate_fold(model, tensors, test_idx, train_idx)
    log(f"primary fold: bacc={result.get('balanced_accuracy', float('nan')):.4f} "
        f"auroc={result.get('auroc', float('nan')):.4f} "
        f"c_index={result['c_index']:.4f} logrank_p={result['logrank_p']:.3e}")

    # ---- tables 2 / 2b / 2c ---------------------------------------------- #
    panel = {k: v for k, v in result.items()
             if isinstance(v, (int, float)) and not k.startswith("_")}
    pd.Series(panel).to_frame("value").to_csv(f"{args.out}/table2_single_fold.csv")
    if result.get("per_class"):
        pd.DataFrame(result["per_class"]).T.to_csv(
            f"{args.out}/table2b_per_class.csv")
    if result.get("per_cohort"):
        pd.DataFrame(result["per_cohort"]).T.to_csv(
            f"{args.out}/table2c_per_cohort.csv")
    log("table2 / table2b / table2c written")

    # ---- table S1: generalisation gap, parameters, learning curve -------- #
    gap_df = generalisation_gap(exp, tensors, model,
                                {"train": fit_idx, "validation": val_idx,
                                 "test": test_idx}, train_idx)
    gap_df.to_csv(f"{args.out}/tableS1_generalisation_gap.csv")
    log("tableS1_generalisation_gap.csv")
    print(gap_df.round(4).to_string(), flush=True)

    params = parameter_report(model, n_train=len(fit_idx))
    with open(f"{args.out}/{args.tag}_parameter_report.json", "w") as fh:
        json.dump(params, fh, indent=2, default=float)
    log(f"parameters: trainable={params['trainable_parameters']} "
        f"per_patient={params['parameters_per_patient']:.1f}")

    lc = learning_curves(model)
    if lc is not None and len(lc):
        lc.to_csv(f"{args.out}/tableS1b_learning_curve.csv")
        log(f"tableS1b_learning_curve.csv ({len(lc)} evaluations)")

    # ---- table 3: baseline comparison panel ------------------------------ #
    cv = load_json(f"{args.runs}/{args.tag}_results.json")
    bl = load_json(f"{args.runs}/{args.tag}_baselines.json")
    baseline_table = None
    if bl is not None:
        all_metrics = sorted({m for d in bl.values() for m in d})
        baseline_table = pd.DataFrame(
            {name: {m: d.get(m, {}).get("mean", np.nan)
                    for m in all_metrics} for name, d in bl.items()}).T
        if cv is not None:
            baseline_table.loc["MOSAIC-GNN"] = {
                m: cv["summary"].get(m, {}).get("mean", np.nan)
                for m in all_metrics}
        baseline_table.to_csv(f"{args.out}/table3_comparison_full.csv")
        log("table3_comparison_full.csv")
        show = [c for c in ("balanced_accuracy", "auroc", "c_index")
                if c in baseline_table]
        print(baseline_table[show].round(4).to_string(), flush=True)

    # ---- table 4: paired bootstrap, ours vs each baseline ---------------- #
    pkl = f"{args.runs}/{args.tag}_baseline_preds.pkl"
    if os.path.isfile(pkl):
        from sklearn.metrics import balanced_accuracy_score

        from mosaic.metrics import (benjamini_hochberg, concordance_index,
                                    paired_bootstrap_test)
        with open(pkl, "rb") as fh:
            saved = pickle.load(fh)
        tr0, te0 = saved["splits"][0]
        if not np.array_equal(te0, test_idx):
            log("  !! fold 0 of the baseline run does not match the primary "
                "fold; skipping the paired test rather than pairing the wrong "
                "patients")
        else:
            ours = exp.predict(model, tensors, test_idx, anchor_idx=train_idx)
            y_te = sk.label[test_idx]
            t_te = sk.time[test_idx]
            e_te = np.nan_to_num(sk.event[test_idx], nan=0.0)
            lab_ok = y_te >= 0
            surv_ok = np.isfinite(t_te) & (t_te > 0)

            def bacc(y, p):
                return balanced_accuracy_score(y, p.argmax(1))

            def cidx(packed, risk):
                return concordance_index(packed[:, 0], risk, packed[:, 1])

            prows = []
            for name, pr in saved["preds"][0].items():
                row = {"model": name}
                if pr.get("prob") is not None and lab_ok.sum() > 1:
                    r = paired_bootstrap_test(
                        bacc, y_te[lab_ok], ours["prob"][lab_ok],
                        np.asarray(pr["prob"])[lab_ok], seed=cfg.train.seed)
                    row.update({"bacc_delta": r["delta"],
                                "bacc_p": r["p_value"],
                                "bacc_ci_low": r.get("ci_low"),
                                "bacc_ci_high": r.get("ci_high")})
                if pr.get("risk") is not None and surv_ok.sum() > 1:
                    packed = np.column_stack([t_te[surv_ok], e_te[surv_ok]])
                    r = paired_bootstrap_test(
                        cidx, packed, ours["risk"][surv_ok],
                        np.asarray(pr["risk"])[surv_ok], seed=cfg.train.seed)
                    row.update({"cindex_delta": r["delta"],
                                "cindex_p": r["p_value"],
                                "cindex_ci_low": r.get("ci_low"),
                                "cindex_ci_high": r.get("ci_high")})
                prows.append(row)
            paired = pd.DataFrame(prows).set_index("model")
            for col in [c for c in paired.columns if c.endswith("_p")]:
                q, rej = benjamini_hochberg(paired[col].tolist())
                paired[col.replace("_p", "_q")] = q
                paired[col.replace("_p", "_sig")] = rej
            paired.to_csv(f"{args.out}/table4_paired_tests.csv")
            log("table4_paired_tests.csv")
            print(paired.round(4).to_string(), flush=True)
    else:
        log(f"  !! no baseline predictions at {pkl}; table 4 omitted")

    # ---- tables 5 / 6: ablation and leave-one-cancer-out ----------------- #
    abl = load_json(f"{args.runs}/{args.tag}_ablation.json")
    if abl is not None:
        ablation_table = pd.DataFrame(
            {name: {k: s.get(k, {}).get("mean", np.nan)
                    for k in ("balanced_accuracy", "auroc", "c_index")}
             for name, s in abl.items()}).T
        ablation_table.to_csv(f"{args.out}/table5_ablation.csv")
        log("table5_ablation.csv")
        print(ablation_table.round(4).to_string(), flush=True)

    loco = load_json(f"{args.runs}/{args.tag}_loco.json")
    if loco is not None:
        loco_table = pd.DataFrame(
            {held: {k: r.get(k, np.nan) for k in
                    ("balanced_accuracy", "auroc", "c_index", "logrank_p")}
             for held, r in loco.items()}).T
        loco_table.to_csv(f"{args.out}/table6_loco.csv")
        log("table6_loco.csv")
        print(loco_table.round(4).to_string(), flush=True)

    # ---- tables 7 / 8 / 9: interpretation -------------------------------- #
    from mosaic.explain import (collect_attention, export_ranked_gene_list,
                                gene_importance_table, integrated_gradients,
                                module_importance_table,
                                personalised_subnetwork)
    log("interpretation: attention + integrated gradients")
    att = collect_attention(model, tensors, idx, exp.device)
    ig = integrated_gradients(model, tensors, idx[: min(len(idx), 200)],
                              exp.device)
    blocks = sorted(sk.blocks)
    genes_df = gene_importance_table(tensors.graph.genes, tensors.graph.modules,
                                     tensors.graph.module_names,
                                     att["gene_gate"], ig, blocks)
    modules_df = module_importance_table(tensors.graph.module_names,
                                         att["module_attention"],
                                         tensors.graph.modules,
                                         tensors.graph.genes,
                                         gate=att["gene_gate"])
    genes_df.head(40).set_index(genes_df.columns[0]).to_csv(
        f"{args.out}/table7_gene_importance.csv")
    modules_df.head(20).set_index(modules_df.columns[0]).to_csv(
        f"{args.out}/table8_module_importance.csv")
    export_ranked_gene_list(genes_df, f"{args.out}/{args.tag}_genes.rnk")
    sub = personalised_subnetwork(model, tensors, 0, exp.device)
    sub.head(25).to_csv(f"{args.out}/table9_patient_subnetwork.csv", index=False)
    log("table7 / table8 / table9 written")

    # ---- table S2: label permutation null -------------------------------- #
    if args.permutations > 0:
        log(f"permutation test ({args.permutations} retrains on shuffled labels)")
        perm_df = permutation_test(exp, tensors, fit_idx, val_idx, test_idx,
                                   n_classes, n_permutations=args.permutations,
                                   seed=cfg.train.seed)
        perm_df.to_csv(f"{args.out}/tableS2_permutation_null.csv")
        prows = {}
        for metric in ("balanced_accuracy", "auroc", "c_index"):
            if metric not in perm_df.columns or metric not in panel:
                continue
            null = perm_df[metric].dropna().to_numpy(dtype=float)
            if null.size == 0:
                continue
            prows[metric] = {
                "observed": panel[metric],
                "null_mean": float(null.mean()),
                "null_sd": float(null.std(ddof=1)) if null.size > 1 else 0.0,
                "null_max": float(null.max()),
                "n_permutations": int(null.size),
                "p_value": permutation_p_value(panel[metric], null),
            }
        perm_table = pd.DataFrame(prows).T
        perm_table.to_csv(f"{args.out}/tableS2_permutation_test.csv")
        log("tableS2_permutation_test.csv")
        print(perm_table.round(4).to_string(), flush=True)

    # ---- table S3: sample-size curve ------------------------------------- #
    if not args.skip_sample_size:
        log("sample-size curve")
        ss_df = sample_size_curve(exp, tensors, fit_idx, val_idx, test_idx,
                                  n_classes, fractions=(0.25, 0.5, 0.75, 1.0),
                                  seed=cfg.train.seed)
        ss_df.to_csv(f"{args.out}/tableS3_sample_size.csv")
        log("tableS3_sample_size.csv")
        print(ss_df.round(4).to_string(), flush=True)

    torch.save({"state_dict": model.state_dict(),
                "genes": tensors.graph.genes,
                "modules": tensors.graph.modules.tolist()},
               f"{args.out}/{args.tag}_primary_fold.pt")
    log("done; CSV tables in " + args.out)
    for f in sorted(os.listdir(args.out)):
        if f.endswith(".csv"):
            print("   ", f, flush=True)


if __name__ == "__main__":
    main()
