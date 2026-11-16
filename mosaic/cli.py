"""Command line entry point.

    python -m mosaic.cli train    --root /kaggle/input/.../tcga-capstone
    python -m mosaic.cli ablate   --root ...
    python -m mosaic.cli baseline --root ...
    python -m mosaic.cli explain  --root ... --checkpoint runs/mosaic-gnn_r0_f0.pt
    python -m mosaic.cli loco     --root ...     # leave-one-cancer-out transfer
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from .config import Config
from .pipeline import Experiment


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.root:
        cfg.data.root = args.root
    if args.cohorts:
        cfg.data.cohorts = args.cohorts.split(",")
    if args.target:
        cfg.data.class_target = args.target
    if args.endpoint:
        cfg.data.survival_endpoint = args.endpoint
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.folds:
        cfg.train.outer_folds = args.folds
    if args.repeats:
        cfg.train.n_repeats = args.repeats
    if args.max_genes:
        cfg.data.max_genes = args.max_genes
        cfg.data.n_genes_per_block = max(args.max_genes // 2, 100)
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
        cfg.train.full_batch = False
    if args.ppi:
        cfg.data.ppi_edgelist = args.ppi
    if args.gmt:
        cfg.data.pathway_gmt = args.gmt
    if args.out:
        cfg.train.out_dir = args.out
    if args.device:
        cfg.train.device = args.device
    if args.tag:
        cfg.tag = args.tag
    return cfg


def cmd_train(cfg: Config) -> None:
    exp = Experiment(cfg)
    exp.prepare()
    res = exp.run_cv()
    print("\n=== summary ===")
    for k, v in res["summary"].items():
        print(f"{k:>24}: {v['mean']:.4f} +/- {v['std']:.4f}  (n={v['n']})")


ABLATIONS: Dict[str, Dict[str, bool]] = {
    "full": {},
    "no_gene_graph": {"use_gene_graph": False},
    "no_module_pool": {"use_module_pool": False},
    "no_patient_graph": {"use_patient_graph": False},
    "no_disentangle": {"use_disentangle": False},
    "no_contrast": {"use_cross_omic_contrast": False},
    "no_clinical": {"use_clinical": False},
    "no_clinical_residual": {"clinical_residual": False},
}


def cmd_ablate(cfg: Config) -> None:
    table = {}
    for name, overrides in ABLATIONS.items():
        model_cfg = type(cfg.model)(**{**cfg.model.__dict__, **overrides})
        # Ablations are cheaper: one repeat is enough for the table.
        train_cfg = type(cfg.train)(**{**cfg.train.__dict__, "n_repeats": 1})
        sub = Config(data=cfg.data, graph=cfg.graph, model=model_cfg,
                     loss=cfg.loss, train=train_cfg, tag=f"{cfg.tag}-{name}")
        print(f"\n########## ablation: {name} ##########")
        exp = Experiment(sub)
        exp.prepare()
        table[name] = exp.run_cv()["summary"]
    path = os.path.join(cfg.train.out_dir, f"{cfg.tag}_ablation.json")
    with open(path, "w") as fh:
        json.dump(table, fh, indent=2, default=float)
    print(f"\nwrote {path}")
    print(f"{'variant':>18} {'bacc':>8} {'auroc':>8} {'c-index':>8}")
    for name, s in table.items():
        print(f"{name:>18} {s.get('balanced_accuracy', {}).get('mean', float('nan')):8.3f} "
              f"{s.get('auroc', {}).get('mean', float('nan')):8.3f} "
              f"{s.get('c_index', {}).get('mean', float('nan')):8.3f}")


def cmd_baseline(cfg: Config) -> None:
    import pickle

    from sklearn.model_selection import StratifiedKFold

    from .baselines import run_baseline_suite
    from .data.preprocess import stratification_key

    exp = Experiment(cfg)
    exp.prepare()
    strat = stratification_key(exp.skeleton)
    idx = np.arange(len(exp.skeleton))
    skf = StratifiedKFold(cfg.train.outer_folds, shuffle=True,
                          random_state=cfg.train.seed)

    per_fold: List[Dict[str, Dict[str, float]]] = []
    per_fold_preds: List[Dict[str, Dict[str, np.ndarray]]] = []
    per_fold_split: List[tuple] = []

    # Per-fold checkpointing, for the same reason run_cv has it: this machine is
    # reclaimed without warning, and a panel of twenty-odd models over five
    # folds does not fit in the window between reclaims. Without this, every
    # restart threw away every finished fold and the panel could never
    # complete. The finished folds are reloaded and skipped instead.
    ppath = os.path.join(cfg.train.out_dir, f"{cfg.tag}_baseline_preds.pkl")
    if os.path.isfile(ppath):
        try:
            with open(ppath, "rb") as fh:
                prev = pickle.load(fh)
            # The gene budget and target are part of the identity of these
            # folds, not just the fold count and seed: reusing folds fitted at
            # a different gene budget would silently average two different
            # experiments into one panel.
            if prev.get("folds") == cfg.train.outer_folds and \
                    prev.get("seed") == cfg.train.seed and \
                    prev.get("max_genes") == cfg.data.max_genes and \
                    prev.get("target") == cfg.data.class_target:
                per_fold = prev["per_fold"]
                per_fold_preds = prev["preds"]
                per_fold_split = prev["splits"]
                print(f"[baseline] resuming: {len(per_fold)} of "
                      f"{cfg.train.outer_folds} folds already done")
            else:
                print("[baseline] existing partial results describe a "
                      "different split; starting over")
        except Exception as err:
            print(f"[baseline] could not read {ppath} ({err}); starting over")

    def _save_partial() -> None:
        tmp = ppath + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"per_fold": per_fold, "preds": per_fold_preds,
                         "splits": per_fold_split,
                         "folds": cfg.train.outer_folds,
                         "seed": cfg.train.seed,
                         "max_genes": cfg.data.max_genes,
                         "target": cfg.data.class_target}, fh)
        os.replace(tmp, ppath)

    for fold, (tr, te) in enumerate(skf.split(idx, strat)):
        if fold < len(per_fold):
            continue                      # already finished in an earlier run
        tensors, _, data = exp.build_fold(tr)
        print(f"[baseline] fold {fold + 1}/{cfg.train.outer_folds}")
        # run_baseline_suite needs the omic block order so that late-fusion and
        # MOGONET-style baselines can split the flat feature axis back into
        # modalities, and it returns (metrics, predictions) -- the predictions
        # are what the paired bootstrap test in the manuscript is computed on.
        metrics, preds = run_baseline_suite(
            tensors.x.numpy(), tensors.clinical.numpy(), data.label, data.time,
            data.event, tr, te, max(int(data.label.max()) + 1, 2),
            block_names=sorted(exp.skeleton.blocks),
            seed=cfg.train.seed, device=str(exp.device),
            # Per-model cache, so an interrupted fold resumes at the next
            # model instead of refitting the whole panel.
            cache_path=os.path.join(
                cfg.train.out_dir, f"{cfg.tag}_baseline_fold{fold}.pkl"))
        per_fold.append(metrics)
        per_fold_preds.append(preds)
        per_fold_split.append((tr, te))
        _save_partial()
        print(f"[baseline] checkpoint: {len(per_fold)}/"
              f"{cfg.train.outer_folds} folds saved to {ppath}")

    # Keep every metric any baseline reported rather than three: a survival-only
    # model has no accuracy and must leave a blank, not a zero.
    all_metrics = sorted({m for f in per_fold for d in f.values() for m in d})
    names = sorted(per_fold[0])
    print(f"\n{'model':>24} {'bacc':>8} {'auroc':>8} {'c-index':>8}")
    summary = {}
    for name in names:
        agg = {}
        for metric in all_metrics:
            vals = [f[name].get(metric, float("nan")) for f in per_fold]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            agg[metric] = {"mean": float(np.mean(vals)) if vals else float("nan"),
                           "std": float(np.std(vals)) if vals else float("nan"),
                           "n": len(vals)}
        summary[name] = agg
        print(f"{name:>24} {agg['balanced_accuracy']['mean']:8.3f} "
              f"{agg['auroc']['mean']:8.3f} {agg['c_index']['mean']:8.3f}")
    path = os.path.join(cfg.train.out_dir, f"{cfg.tag}_baselines.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"wrote {path}")

    # Fold-0 predictions are kept so the paired comparison against MOSAIC-GNN
    # can be run later without refitting every baseline. Already written by the
    # per-fold checkpoint above; rewritten here so the file is complete even if
    # the loop was skipped entirely on a resume.
    _save_partial()
    print(f"wrote {ppath}")


def cmd_loco(cfg: Config) -> None:
    """Leave-one-cancer-out: train on three cohorts, test on the fourth.

    This is the experiment that separates a pan-cancer representation from a
    cohort-detector, and it is the headline transfer result.
    """
    exp = Experiment(cfg)
    exp.prepare()
    results = {}
    for held in cfg.data.cohorts:
        a, b = exp.offsets[held]
        test_idx = np.arange(a, b)
        train_idx = np.setdiff1d(np.arange(len(exp.skeleton)), test_idx)
        rng = np.random.default_rng(cfg.train.seed)
        val = rng.choice(train_idx, size=max(int(0.15 * len(train_idx)), 20),
                         replace=False)
        fit = np.setdiff1d(train_idx, val)

        print(f"\n########## leave-out cohort: {held} "
              f"(train={len(fit)} test={len(test_idx)}) ##########")
        tensors, _, _ = exp.build_fold(train_idx)
        n_classes = max(int(exp.skeleton.label.max()) + 1, 2)
        model = exp.train_fold(tensors, fit, val, n_classes)
        res = exp.evaluate_fold(model, tensors, test_idx, train_idx)
        res.pop("_pred", None)
        results[held] = res
        print(f"[loco:{held}] bacc={res.get('balanced_accuracy', float('nan')):.3f} "
              f"c_index={res['c_index']:.3f} logrank_p={res['logrank_p']:.2e}")
    path = os.path.join(cfg.train.out_dir, f"{cfg.tag}_loco.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"wrote {path}")


def cmd_explain(cfg: Config, checkpoint: str) -> None:
    from .explain import (collect_attention, export_ranked_gene_list,
                          gene_importance_table, integrated_gradients,
                          module_importance_table, personalised_subnetwork)

    exp = Experiment(cfg)
    exp.prepare()
    idx = np.arange(len(exp.skeleton))
    tensors, _, _ = exp.build_fold(idx)
    n_classes = max(int(exp.skeleton.label.max()) + 1, 2)
    model = exp.build_model(tensors, n_classes)
    state = torch.load(checkpoint, map_location=exp.device)
    model.load_state_dict(state["state_dict"], strict=False)

    att = collect_attention(model, tensors, idx, exp.device)
    ig = integrated_gradients(model, tensors, idx[: min(len(idx), 200)], exp.device)
    blocks = sorted(exp.skeleton.blocks)

    genes_df = gene_importance_table(tensors.graph.genes, tensors.graph.modules,
                                     tensors.graph.module_names,
                                     att["gene_gate"], ig, blocks)
    mods_df = module_importance_table(tensors.graph.module_names,
                                      att["module_attention"],
                                      tensors.graph.modules, tensors.graph.genes,
                                      gate=att["gene_gate"])
    out = cfg.train.out_dir
    genes_df.to_csv(os.path.join(out, f"{cfg.tag}_gene_importance.csv"), index=False)
    mods_df.to_csv(os.path.join(out, f"{cfg.tag}_module_importance.csv"), index=False)
    export_ranked_gene_list(genes_df, os.path.join(out, f"{cfg.tag}_genes.rnk"))
    personalised_subnetwork(model, tensors, 0, exp.device).to_csv(
        os.path.join(out, f"{cfg.tag}_subnetwork_patient0.csv"), index=False)

    print(genes_df.head(30).to_string(index=False))
    print()
    print(mods_df.head(15).to_string(index=False))
    print(f"\nwrote interpretation tables to {out}/")


def main(argv=None) -> None:
    p = argparse.ArgumentParser("mosaic")
    p.add_argument("command", choices=["train", "ablate", "baseline", "explain",
                                       "loco"])
    p.add_argument("--config")
    p.add_argument("--root")
    p.add_argument("--cohorts")
    p.add_argument("--target", choices=["stage", "stage_binary", "grade",
                                       "cohort", "subtype"])
    p.add_argument("--endpoint", choices=["OS", "DSS", "PFS", "DFS"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--folds", type=int)
    p.add_argument("--repeats", type=int)
    p.add_argument("--max-genes", dest="max_genes", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--ppi")
    p.add_argument("--gmt")
    p.add_argument("--out")
    p.add_argument("--device")
    p.add_argument("--tag")
    p.add_argument("--checkpoint")
    args = p.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    cfg = _apply_overrides(cfg, args)
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.train.out_dir, f"{cfg.tag}_config.json"))

    if args.command == "train":
        cmd_train(cfg)
    elif args.command == "ablate":
        cmd_ablate(cfg)
    elif args.command == "baseline":
        cmd_baseline(cfg)
    elif args.command == "loco":
        cmd_loco(cfg)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for `explain`")
        cmd_explain(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
