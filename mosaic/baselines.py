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

import contextlib
import os
import signal
import time
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
class _BaselineTimeout(Exception):
    pass


@contextlib.contextmanager
def _time_budget(name: str, verbose: bool):
    """Give one baseline a wall-clock budget, then move on.

    At pan-cancer scale a single solver can stall the whole panel: fitting
    multinomial L1 logistic regression with saga over 32 classes, ~8 700
    training patients and 600 features ran for over twenty minutes without
    finishing, and nothing is reported until every model in the panel has
    run. One slow model must not cost the Cox and random-survival-forest
    comparisons, which are the point of the table.

    A model that exceeds the budget is recorded as skipped with the reason,
    which is honest -- the comparison then says the model did not fit in the
    time allowed rather than quietly reporting a number from a truncated fit.
    Budget comes from MOSAIC_BASELINE_BUDGET_S (default 20 minutes); 0
    disables it.
    """
    budget = int(os.environ.get("MOSAIC_BASELINE_BUDGET_S", "1200"))
    if budget <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _fire(signum, frame):
        raise _BaselineTimeout(f"exceeded {budget}s budget")

    old = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(budget)
    t0 = time.time()
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        if verbose and time.time() - t0 > 60:
            print(f"  {name:<24} fit took {time.time() - t0:.0f}s", flush=True)


def _classical_models(seed: int, n_features: int) -> Dict[str, object]:
    sc = lambda: StandardScaler(with_mean=False)
    models: Dict[str, object] = {
        "logistic_l2": Pipeline([("sc", sc()), ("clf", LogisticRegression(
            max_iter=3000, C=0.1, class_weight="balanced"))]),
        # liblinear cannot fit a multiclass problem (n_classes >= 3), so with
        # a four-class target this baseline raised and was silently dropped
        # from the comparison panel entirely. saga handles L1 multinomially.
        # saga is the only solver here that handles multinomial L1 and
        # elastic net, and it is single-threaded and slow: at 32 classes,
        # ~8 700 training patients and 600 features, max_iter=3000 ran past
        # twenty minutes without returning and stalled the whole panel. 300
        # iterations on standardised features is enough to converge in
        # practice, and a baseline that fits is worth more to the comparison
        # than one that times out and leaves a blank row. sklearn emits a
        # ConvergenceWarning if it is not enough, which is visible in the log.
        "logistic_l1": Pipeline([("sc", sc()), ("clf", LogisticRegression(
            max_iter=300, C=0.1, penalty="l1", solver="saga",
            class_weight="balanced"))]),
        "logistic_elasticnet": Pipeline([("sc", sc()), ("clf", LogisticRegression(
            max_iter=300, C=0.1, penalty="elasticnet", l1_ratio=0.5,
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
            # adjs[k] is sparse (see knn_adjacency), so the propagation uses a
            # sparse matmul; `@` on a sparse operand would densify it.
            h = self.norm[k](h + F.gelu(self.gcn[k](torch.sparse.mm(adjs[k], h))))
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
                # Attention is scored only on the edges the graph actually
                # holds (~k*N of them), never on the N^2 pairs it does not: a
                # dense (N, N) logit matrix is 480 MB at N = 10 953 and
                # autograd keeps it for the backward pass. The softmax is
                # taken per destination row with scatter reductions, which is
                # the same normalisation the dense masked_fill produced.
                hid = h.shape[1]
                w = self.att[i].weight                      # (1, 2H)
                a_src = (h @ w[:, :hid].t()).squeeze(-1)     # (N,)
                a_dst = (h @ w[:, hid:].t()).squeeze(-1)     # (N,)
                r, c = adj.indices()[0], adj.indices()[1]
                e = a_src[r] + a_dst[c]
                if self.att[i].bias is not None:
                    e = e + self.att[i].bias
                e = F.leaky_relu(e, 0.2)
                # Row-wise softmax, numerically stabilised by the per-row max.
                n_nodes = h.shape[0]
                row_max = torch.full((n_nodes,), float("-inf"), device=e.device,
                                     dtype=e.dtype).index_reduce_(
                    0, r, e.detach(), "amax", include_self=True)
                ex = torch.exp(e - row_max[r])
                denom = torch.zeros(n_nodes, device=e.device, dtype=e.dtype
                                    ).index_add_(0, r, ex)
                alpha = ex / denom[r].clamp_min(1e-16)
                agg = torch.zeros_like(h).index_add_(0, r, alpha.unsqueeze(-1) * h[c])
            else:
                agg = torch.sparse.mm(adj, h)
            msg = lin(agg)
            if self.kind == "sage":
                msg = msg + self.self_lins[i](h)
            h = norm(h + self.drop(F.gelu(msg)))
        return self.cls(h), self.risk(h).squeeze(-1)


def knn_adjacency(z: np.ndarray, k: int = 10,
                  block: int = 2048) -> torch.Tensor:
    """Row-normalised symmetric kNN patient graph, as a SPARSE tensor.

    A k-nearest-neighbour graph over N patients holds about k*N edges -- some
    110 000 at N = 10 953, k = 10. Materialising it as a dense N x N matrix
    instead costs 960 MB in float64, and this function used to build four such
    matrices in a row (`sim`, `adj`, its transpose-maximum, and the torch
    copy). With one graph per omic block for the MOGONET-style baseline plus
    one for the patient-graph family, the panel was certain to be OOM-killed
    at pan-cancer scale -- taking the Cox and random-survival-forest results
    down with it, since nothing is returned until every model has run.

    The similarity matrix is also never needed whole: it is reduced to k
    indices per row, so it is computed in row blocks and discarded as it goes.
    Peak memory is then O(block * N) rather than O(N^2).
    """
    n = len(z)
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    k = min(k, max(1, n - 1))

    rows, cols = [], []
    for start in range(0, n, block):
        stop = min(start + block, n)
        sim = zn[start:stop] @ zn.T                      # (b, N)
        # Never let a patient be its own neighbour; the self-loop is added
        # explicitly below so every row has one regardless of k.
        for local, globl in enumerate(range(start, stop)):
            sim[local, globl] = -np.inf
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        rows.append(np.repeat(np.arange(start, stop), k))
        cols.append(idx.reshape(-1))
        del sim
    r = np.concatenate(rows)
    c = np.concatenate(cols)

    # Symmetrise (union of the two directions) and add self-loops, then
    # row-normalise -- the same graph the dense version produced.
    r_sym = np.concatenate([r, c, np.arange(n)])
    c_sym = np.concatenate([c, r, np.arange(n)])
    pairs = np.unique(np.stack([r_sym, c_sym], axis=1), axis=0)
    r_sym, c_sym = pairs[:, 0], pairs[:, 1]
    deg = np.bincount(r_sym, minlength=n).astype(np.float32)
    vals = (1.0 / np.maximum(deg, 1.0))[r_sym]

    idx_t = torch.tensor(np.stack([r_sym, c_sym]), dtype=torch.long)
    return torch.sparse_coo_tensor(
        idx_t, torch.tensor(vals, dtype=torch.float32), (n, n)).coalesce()


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
                       verbose: bool = True, cache_path: Optional[str] = None
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

    # Per-MODEL checkpointing. Saving only once a whole fold has finished is
    # not enough here: a fold is twenty-odd models over 10 953 patients, which
    # takes longer than this container stays alive, so every restart threw the
    # fold away and began again at the first model. The panel could never
    # finish, however many times it ran. With a cache, each finished model is
    # on disk immediately and a restart picks up at the next one.
    if cache_path and os.path.isfile(cache_path):
        try:
            import pickle
            with open(cache_path, "rb") as fh:
                cached = pickle.load(fh)
            if cached.get("test_hash") == int(np.asarray(test_idx).sum()):
                metrics.update(cached["metrics"])
                preds.update(cached["preds"])
                if verbose and metrics:
                    print(f"  resuming fold: {len(metrics)} model(s) already "
                          f"done ({', '.join(sorted(metrics))})", flush=True)
            elif verbose:
                print("  cache describes a different fold; ignoring it")
        except Exception as err:
            if verbose:
                print(f"  could not read {cache_path} ({err}); ignoring it")

    def _save_cache():
        if not cache_path:
            return
        import pickle
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"metrics": metrics, "preds": preds,
                         "test_hash": int(np.asarray(test_idx).sum())}, fh)
        os.replace(tmp, cache_path)

    def record(name, prob=None, risk=None):
        preds[name] = {"prob": prob, "risk": risk}
        metrics[name] = evaluate_predictions(y[test_idx], prob, time[test_idx],
                                             risk, event[test_idx], seed=seed)
        _save_cache()
        if verbose:
            m = metrics[name]
            print(f"  {name:<24} bacc={m.get('balanced_accuracy', float('nan')):.3f} "
                  f"auroc={m.get('auroc', float('nan')):.3f} "
                  f"c-index={m.get('c_index', float('nan')):.3f}", flush=True)

    # --- family 1: classical classifiers ---------------------------------- #
    for name, model in _classical_models(seed, feats.shape[1]).items():
        if name in metrics:
            continue                              # already in the cache
        try:
            with _time_budget(name, verbose):
                model.fit(feats[train_idx][lab_tr], y[train_idx][lab_tr])
            record(name, prob=_proba(model, feats[test_idx], n_classes))
        except Exception as err:                       # keep the suite running
            if verbose:
                print(f"  {name:<24} skipped ({type(err).__name__}: {err})",
                      flush=True)

    # --- family 2: neural fusion ------------------------------------------ #
    sc = StandardScaler().fit(feats[train_idx])
    fs = sc.transform(feats).astype(np.float32)
    tt = {"y": torch.as_tensor(y, device=device),
          "t": torch.as_tensor(time, dtype=torch.float32, device=device),
          "e": torch.as_tensor(event, dtype=torch.float32, device=device)}

    xt = torch.tensor(fs, device=device)
    if "mlp_early_fusion" not in metrics:
        mlp = _MLP(xt.shape[1], 128, n_classes).to(device)
        prob, risk = _train_torch(mlp, tt, train_idx, test_idx,
                                  lambda m: m(xt), epochs, device=device, seed=seed)
        record("mlp_early_fusion", prob, risk)

    bt, dims = {}, {}
    for k, v in blocks.items():
        s = StandardScaler().fit(v[train_idx])
        bt[k] = torch.tensor(s.transform(v).astype(np.float32), device=device)
        dims[k] = v.shape[1]
    if "late_fusion_mlp" not in metrics:
        late = LateFusion(dims, 128, n_classes).to(device)
        prob, risk = _train_torch(late, tt, train_idx, test_idx,
                                  lambda m: m(bt), epochs, device=device, seed=seed)
        record("late_fusion_mlp", prob, risk)

    if "mogonet_style" not in metrics:
        adjs = {}
        for k, v in bt.items():
            z = v.cpu().numpy()
            pca = PCA(n_components=min(32, z.shape[1], len(train_idx) - 1),
                      random_state=seed).fit(z[train_idx])
            adjs[k] = knn_adjacency(pca.transform(z), k=10).to(device)
        mog = MogonetStyle(dims, 128, n_classes).to(device)
        prob, risk = _train_torch(mog, tt, train_idx, test_idx,
                                  lambda m: m(bt, adjs), epochs, device=device,
                                  seed=seed)
        record("mogonet_style", prob, risk)

    # --- family 3: patient-graph GNNs ------------------------------------- #
    if any(f"patient_{k}" not in metrics for k in ("gcn", "gat", "sage")):
        pca = PCA(n_components=min(64, fs.shape[1], len(train_idx) - 1),
                  random_state=seed).fit(fs[train_idx])
        adj = knn_adjacency(pca.transform(fs), k=10).to(device)
        for kind in ("gcn", "gat", "sage"):
            if f"patient_{kind}" in metrics:
                continue
            net = PatientGraphBaseline(xt.shape[1], 128, n_classes, kind).to(device)
            prob, risk = _train_torch(net, tt, train_idx, test_idx,
                                      lambda m: m(xt, adj), epochs, device=device,
                                      seed=seed)
            record(f"patient_{kind}", prob, risk)

    # --- family 4: survival models ---------------------------------------- #
    tr_s = train_idx[surv_tr]
    xtr, ttr, etr = feats[tr_s], time[tr_s], event[tr_s]

    # These run last and had no guard at all, which is the worst place for
    # one: an exception or a stall in any of them lost the entire panel,
    # including every classifier already fitted, because nothing is returned
    # until the function ends. They are also the most important rows in the
    # comparison -- whether a Cox model or a random survival forest beats
    # MOSAIC-GNN on c-index is the question the table exists to answer -- so
    # each is attempted under its own budget and recorded as skipped, with
    # the reason, if it does not finish.
    def _survival(name: str, fn):
        if name in metrics:
            return                                # already in the cache
        try:
            with _time_budget(name, verbose):
                risk = fn()
        except Exception as err:
            if verbose:
                print(f"  {name:<24} skipped ({type(err).__name__}: {err})",
                      flush=True)
            return
        if risk is not None:
            record(name, risk=risk)
        elif verbose:
            print(f"  {name:<24} skipped (unavailable)", flush=True)

    _survival("cox_lasso",
              lambda: cox_penalised(xtr, ttr, etr, feats[test_idx], seed, 0.9))
    _survival("cox_ridge",
              lambda: cox_penalised(xtr, ttr, etr, feats[test_idx], seed, 0.0))
    _survival("cox_clinical_only",
              lambda: cox_penalised(clinical[tr_s], ttr, etr,
                                    clinical[test_idx], seed, 0.0))
    _survival("random_survival_forest",
              lambda: random_survival_forest(xtr, ttr, etr, feats[test_idx], seed))
    _survival("deepsurv",
              lambda: deepsurv(xtr, ttr, etr, feats[test_idx], seed,
                               epochs=epochs, device=device))
    return metrics, preds
