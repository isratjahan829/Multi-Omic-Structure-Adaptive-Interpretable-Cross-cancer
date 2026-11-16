"""Comparators.

A credible Q1 comparison table needs four families, not one:

1. **classical predictors on concatenated features** — penalised regression,
   kernel machines, tree ensembles, gradient boosting;
2. **multi-omics fusion networks** — early fusion, late fusion, and a
   MOGONET-style per-omic GCN with view fusion;
3. **patient-graph GNNs** — GCN / GAT / GraphSAGE on a fixed kNN patient graph,
   which is what most published multi-omics GNNs actually are;
4. **dedicated survival models** — penalised Cox, clinical-only Cox, random
   survival forest, DeepSurv.

Every model is scored with the *same* metric function, so rows are comparable,
and every model returns its raw predictions so paired bootstrap tests can be
run against MOSAIC-GNN.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .losses import cox_ph_loss
from .metrics import evaluate_predictions


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #
def flatten_features(x: np.ndarray, clinical: np.ndarray) -> np.ndarray:
    """(N, G, F) node features -> (N, G*F) concatenated with clinical columns."""
    n = x.shape[0]
    return np.concatenate([x.reshape(n, -1), clinical], axis=1).astype(np.float32)


def omic_blocks(x: np.ndarray, clinical: np.ndarray,
                block_names: List[str]) -> Dict[str, np.ndarray]:
    """Split the node tensor back into one matrix per modality (+ clinical)."""
    nb = len(block_names)
    out = {name: x[:, :, i].astype(np.float32) for i, name in enumerate(block_names)}
    out["clinical"] = clinical.astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# 1. Classical classifiers
# --------------------------------------------------------------------------- #
def _classical_models(seed: int, n_features: int) -> Dict[str, object]:
    sc = lambda: StandardScaler(with_mean=False)
    models: Dict[str, object] = {
        "logistic_l2": Pipeline([("sc", sc()), ("clf", LogisticRegression(
            max_iter=3000, C=0.1, class_weight="balanced"))]),
        # liblinear cannot fit a multiclass problem (n_classes >= 3), so with
        # a four-class target this baseline raised and was silently dropped
        # from the comparison panel entirely. saga handles L1 multinomially.
        "logistic_l1": Pipeline([("sc", sc()), ("clf", LogisticRegression(
            max_iter=3000, C=0.1, penalty="l1", solver="saga",
            class_weight="balanced"))]),
        "logistic_elasticnet": Pipeline([("sc", sc()), ("clf", LogisticRegression(
            max_iter=3000, C=0.1, penalty="elasticnet", l1_ratio=0.5,
            solver="saga", class_weight="balanced"))]),
        "linear_svm": Pipeline([("sc", sc()), ("clf", LinearSVC(
            C=0.05, class_weight="balanced", max_iter=5000))]),
        "svm_rbf": Pipeline([("sc", sc()), ("clf", SVC(
            C=1.0, gamma="scale", probability=True, class_weight="balanced",
            random_state=seed))]),
        "knn": Pipeline([("sc", sc()), ("clf", KNeighborsClassifier(
            n_neighbors=15, weights="distance"))]),
        "naive_bayes": Pipeline([("sc", sc()), ("clf", GaussianNB())]),
        "lda_shrinkage": Pipeline([("sc", sc()), ("clf",
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced_subsample", random_state=seed),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced", random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=4, random_state=seed),
        "pca_logistic": Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=min(64, n_features), random_state=seed)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.3, reg_lambda=1.0, tree_method="hist",
            random_state=seed, eval_metric="mlogloss")
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            colsample_bytree=0.3, class_weight="balanced", random_state=seed,
            verbose=-1)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.06, verbose=0,
            random_seed=seed)
    except ImportError:
        pass
    return models


def _proba(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba, with a softmax fallback for margin-only classifiers."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
    else:
        d = model.decision_function(x)
        d = d.reshape(len(x), -1) if d.ndim > 1 else np.stack([-d, d], 1)
        p = np.exp(d - d.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
    if p.shape[1] < n_classes:          # a class was absent from the train fold
        full = np.full((len(x), n_classes), 1e-6, dtype=float)
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "steps"):
            classes = model.steps[-1][1].classes_
        full[:, np.asarray(classes, dtype=int)] = p
        p = full / full.sum(1, keepdims=True)
    return p


# --------------------------------------------------------------------------- #
# 2. Neural fusion baselines
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout=0.4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout))
        self.cls = nn.Linear(hidden // 2, n_classes)
        self.risk = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.body(x)
        return self.cls(h), self.risk(h).squeeze(-1)


def _train_torch(model, tensors: Dict[str, torch.Tensor], train_idx, test_idx,
                 forward: Callable, epochs: int, lr: float = 1e-3,
                 device="cpu", seed: int = 0, survival_only: bool = False):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr = torch.as_tensor(train_idx, device=device)
    y, t, e = tensors["y"], tensors["t"], tensors["e"]
    lab_ok = y[tr] >= 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits, risk = forward(model)
        loss = cox_ph_loss(risk[tr], t[tr], e[tr])
        if not survival_only and lab_ok.any():
            loss = loss + F.cross_entropy(logits[tr][lab_ok], y[tr][lab_ok])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits, risk = forward(model)
    return (F.softmax(logits, -1).cpu().numpy()[test_idx],
            risk.cpu().numpy()[test_idx])


class LateFusion(nn.Module):
    """One encoder per modality, logits averaged -- the classic late-fusion
    multi-omics network."""

    def __init__(self, dims: Dict[str, int], hidden: int, n_classes: int,
                 dropout=0.4):
        super().__init__()
        self.nets = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                             nn.LayerNorm(hidden))
            for k, d in dims.items()})
        self.cls = nn.ModuleDict({k: nn.Linear(hidden, n_classes) for k in dims})
        self.risk = nn.ModuleDict({k: nn.Linear(hidden, 1) for k in dims})
        self.weight = nn.Parameter(torch.zeros(len(dims)))

    def forward(self, blocks: Dict[str, torch.Tensor]):
        w = torch.softmax(self.weight, 0)
        logits, risks = 0.0, 0.0
        for i, k in enumerate(self.nets):
            h = self.nets[k](blocks[k])
            logits = logits + w[i] * self.cls[k](h)
            risks = risks + w[i] * self.risk[k](h).squeeze(-1)
        return logits, risks


class MogonetStyle(nn.Module):
    """Per-omic patient-graph GCN with a cross-view fusion head.

    Structurally close to MOGONET: each modality gets its own patient kNN graph
    and GCN, and the views are combined by a learned bilinear fusion.
    """

    def __init__(self, dims: Dict[str, int], hidden: int, n_classes: int,
                 dropout=0.4):
        super().__init__()
        self.enc = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
            for k, d in dims.items()})
        self.gcn = nn.ModuleDict({k: nn.Linear(hidden, hidden) for k in dims})
        self.norm = nn.ModuleDict({k: nn.LayerNorm(hidden) for k in dims})
        self.fuse = nn.Sequential(
            nn.Linear(hidden * len(dims), hidden), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(hidden, n_classes)
        self.risk = nn.Linear(hidden, 1)

    def forward(self, blocks: Dict[str, torch.Tensor],
                adjs: Dict[str, torch.Tensor]):
        hs = []
        for k in self.enc:
            h = self.enc[k](blocks[k])
            h = self.norm[k](h + F.gelu(self.gcn[k](adjs[k] @ h)))
            hs.append(h)
        z = self.fuse(torch.cat(hs, dim=-1))
        return self.cls(z), self.risk(z).squeeze(-1)


class PatientGraphBaseline(nn.Module):
    """GCN / GAT / GraphSAGE on a fixed kNN patient graph over concatenated omics.

    This is the architecture most published multi-omics GNNs use, so it is the
    reference point MOSAIC-GNN's gene-level stage has to beat.
    """

    def __init__(self, in_dim: int, hidden: int, n_classes: int,
                 kind: str = "gcn", layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.kind = kind
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout))
        self.lins = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.self_lins = nn.ModuleList([nn.Linear(hidden, hidden)
                                        for _ in range(layers)]) \
            if kind == "sage" else None
        self.att = nn.ModuleList([nn.Linear(2 * hidden, 1) for _ in range(layers)]) \
            if kind == "gat" else None
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.cls = nn.Linear(hidden, n_classes)
        self.risk = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        h = self.inp(x)
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            if self.kind == "gat":
                # The attention score is a linear map of the concatenated pair
                # [h_i ; h_j], so it splits into a source term and a
                # destination term:
                #     att([h_i ; h_j]) = W[:, :H] h_i + W[:, H:] h_j + b
                # Scoring that as a dense (N, N, 2H) tensor costs 3.5 GB at
                # N = 1848, H = 128 -- and autograd keeps it for the backward
                # pass -- which is enough to be OOM-killed on its own. The two
                # per-node terms broadcast to the same (N, N) logits for the
                # price of two (N, 1) vectors.
                hid = h.shape[1]
                w = self.att[i].weight                      # (1, 2H)
                a_src = h @ w[:, :hid].t()                   # (N, 1)
                a_dst = h @ w[:, hid:].t()                   # (N, 1)
                logits = a_src + a_dst.t()                   # (N, N)
                if self.att[i].bias is not None:
                    logits = logits + self.att[i].bias
                logits = F.leaky_relu(logits, 0.2)
                logits = logits.masked_fill(adj <= 0, float("-inf"))
                agg = torch.softmax(logits, dim=1) @ h
            else:
                agg = adj @ h
            msg = lin(agg)
            if self.kind == "sage":
                msg = msg + self.self_lins[i](h)
            h = norm(h + self.drop(F.gelu(msg)))
        return self.cls(h), self.risk(h).squeeze(-1)


def knn_adjacency(z: np.ndarray, k: int = 10) -> torch.Tensor:
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    sim = zn @ zn.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, max(1, len(z) - 1))
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    adj = np.zeros_like(sim)
    np.put_along_axis(adj, idx, 1.0, axis=1)
    adj = np.maximum(adj, adj.T) + np.eye(len(z))
    adj = adj / adj.sum(1, keepdims=True)
    return torch.tensor(adj, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# 3. Survival-specific models
# --------------------------------------------------------------------------- #
def _torch_cox(xtr, ttr, etr, xte, seed: int = 0, epochs: int = 400,
               l1: float = 1e-3, weight_decay: float = 1e-2) -> np.ndarray:
    torch.manual_seed(seed)
    sc = StandardScaler().fit(xtr)
    a = torch.tensor(sc.transform(xtr), dtype=torch.float32)
    b = torch.tensor(sc.transform(xte), dtype=torch.float32)
    t = torch.tensor(ttr, dtype=torch.float32)
    e = torch.tensor(etr, dtype=torch.float32)
    lin = nn.Linear(a.shape[1], 1, bias=False)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        loss = cox_ph_loss(lin(a).squeeze(-1), t, e)
        if l1:
            loss = loss + l1 * lin.weight.abs().sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return lin(b).squeeze(-1).numpy()


def cox_penalised(xtr, ttr, etr, xte, seed: int = 0, l1_ratio: float = 0.9
                  ) -> np.ndarray:
    """Penalised Cox: scikit-survival when installed, else an equivalent torch fit."""
    try:
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
        sc = StandardScaler().fit(xtr)
        y = Surv.from_arrays(event=etr.astype(bool), time=np.maximum(ttr, 1e-3))
        model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.05)
        model.fit(sc.transform(xtr), y)
        return model.predict(sc.transform(xte))
    except Exception:
        return _torch_cox(xtr, ttr, etr, xte, seed,
                          l1=1e-3 if l1_ratio > 0.5 else 0.0)


def random_survival_forest(xtr, ttr, etr, xte, seed: int = 0) -> Optional[np.ndarray]:
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
    except ImportError:
        return None
    y = Surv.from_arrays(event=etr.astype(bool), time=np.maximum(ttr, 1e-3))
    rsf = RandomSurvivalForest(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                               random_state=seed)
    rsf.fit(xtr, y)
    return rsf.predict(xte)


def deepsurv(xtr, ttr, etr, xte, seed: int = 0, epochs: int = 300,
             hidden: int = 64, device: str = "cpu") -> np.ndarray:
    """DeepSurv: an MLP trained on the Cox partial likelihood."""
    torch.manual_seed(seed)
    sc = StandardScaler().fit(xtr)
    a = torch.tensor(sc.transform(xtr), dtype=torch.float32, device=device)
    b = torch.tensor(sc.transform(xte), dtype=torch.float32, device=device)
    t = torch.tensor(ttr, dtype=torch.float32, device=device)
    e = torch.tensor(etr, dtype=torch.float32, device=device)
    net = nn.Sequential(nn.Linear(a.shape[1], hidden), nn.GELU(), nn.Dropout(0.4),
                        nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.4),
                        nn.Linear(hidden // 2, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        cox_ph_loss(net(a).squeeze(-1), t, e).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(b).squeeze(-1).cpu().numpy()


# --------------------------------------------------------------------------- #
# The suite
# --------------------------------------------------------------------------- #
def run_baseline_suite(x, clinical, y, time, event, train_idx, test_idx,
                       n_classes: int, block_names: List[str], seed: int = 0,
                       device: str = "cpu", epochs: int = 300,
                       verbose: bool = True
                       ) -> Tuple[Dict[str, Dict[str, float]],
                                  Dict[str, Dict[str, np.ndarray]]]:
    """Fit every baseline on one fold.

    Returns
    -------
    metrics : {model: metric dict}   -- identical keys for every model
    preds   : {model: {"prob":..., "risk":...}}  -- for paired bootstrap tests
    """
    feats = flatten_features(x, clinical)
    blocks = omic_blocks(x, clinical, block_names)
    event = np.nan_to_num(event, nan=0.0)

    lab_tr = y[train_idx] >= 0
    surv_tr = np.isfinite(time[train_idx]) & (time[train_idx] > 0)
    metrics: Dict[str, Dict[str, float]] = {}
    preds: Dict[str, Dict[str, np.ndarray]] = {}

    def record(name, prob=None, risk=None):
        preds[name] = {"prob": prob, "risk": risk}
        metrics[name] = evaluate_predictions(y[test_idx], prob, time[test_idx],
                                             risk, event[test_idx], seed=seed)
        if verbose:
            m = metrics[name]
            print(f"  {name:<24} bacc={m.get('balanced_accuracy', float('nan')):.3f} "
                  f"auroc={m.get('auroc', float('nan')):.3f} "
                  f"c-index={m.get('c_index', float('nan')):.3f}")

    # --- family 1: classical classifiers ---------------------------------- #
    for name, model in _classical_models(seed, feats.shape[1]).items():
        try:
            model.fit(feats[train_idx][lab_tr], y[train_idx][lab_tr])
            record(name, prob=_proba(model, feats[test_idx], n_classes))
        except Exception as err:                       # keep the suite running
            if verbose:
                print(f"  {name:<24} skipped ({type(err).__name__}: {err})")

    # --- family 2: neural fusion ------------------------------------------ #
    sc = StandardScaler().fit(feats[train_idx])
    fs = sc.transform(feats).astype(np.float32)
    tt = {"y": torch.as_tensor(y, device=device),
          "t": torch.as_tensor(time, dtype=torch.float32, device=device),
          "e": torch.as_tensor(event, dtype=torch.float32, device=device)}

    xt = torch.tensor(fs, device=device)
    mlp = _MLP(xt.shape[1], 128, n_classes).to(device)
    prob, risk = _train_torch(mlp, tt, train_idx, test_idx,
                              lambda m: m(xt), epochs, device=device, seed=seed)
    record("mlp_early_fusion", prob, risk)

    bt, dims = {}, {}
    for k, v in blocks.items():
        s = StandardScaler().fit(v[train_idx])
        bt[k] = torch.tensor(s.transform(v).astype(np.float32), device=device)
        dims[k] = v.shape[1]
    late = LateFusion(dims, 128, n_classes).to(device)
    prob, risk = _train_torch(late, tt, train_idx, test_idx,
                              lambda m: m(bt), epochs, device=device, seed=seed)
    record("late_fusion_mlp", prob, risk)

    adjs = {}
    for k, v in bt.items():
        z = v.cpu().numpy()
        pca = PCA(n_components=min(32, z.shape[1], len(train_idx) - 1),
                  random_state=seed).fit(z[train_idx])
        adjs[k] = knn_adjacency(pca.transform(z), k=10).to(device)
    mog = MogonetStyle(dims, 128, n_classes).to(device)
    prob, risk = _train_torch(mog, tt, train_idx, test_idx,
                              lambda m: m(bt, adjs), epochs, device=device, seed=seed)
    record("mogonet_style", prob, risk)

    # --- family 3: patient-graph GNNs ------------------------------------- #
    pca = PCA(n_components=min(64, fs.shape[1], len(train_idx) - 1),
              random_state=seed).fit(fs[train_idx])
    adj = knn_adjacency(pca.transform(fs), k=10).to(device)
    for kind in ("gcn", "gat", "sage"):
        net = PatientGraphBaseline(xt.shape[1], 128, n_classes, kind).to(device)
        prob, risk = _train_torch(net, tt, train_idx, test_idx,
                                  lambda m: m(xt, adj), epochs, device=device,
                                  seed=seed)
        record(f"patient_{kind}", prob, risk)

    # --- family 4: survival models ---------------------------------------- #
    tr_s = train_idx[surv_tr]
    xtr, ttr, etr = feats[tr_s], time[tr_s], event[tr_s]
    record("cox_lasso", risk=cox_penalised(xtr, ttr, etr, feats[test_idx], seed, 0.9))
    record("cox_ridge", risk=cox_penalised(xtr, ttr, etr, feats[test_idx], seed, 0.0))
    record("cox_clinical_only",
           risk=cox_penalised(clinical[tr_s], ttr, etr, clinical[test_idx], seed, 0.0))
    rsf = random_survival_forest(xtr, ttr, etr, feats[test_idx], seed)
    if rsf is not None:
        record("random_survival_forest", risk=rsf)
    elif verbose:
        print("  random_survival_forest   skipped (scikit-survival not installed)")
    record("deepsurv", risk=deepsurv(xtr, ttr, etr, feats[test_idx], seed,
                                     epochs=epochs, device=device))
    return metrics, preds
