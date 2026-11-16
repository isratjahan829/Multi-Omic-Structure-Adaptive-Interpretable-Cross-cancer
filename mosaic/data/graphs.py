"""Graph construction.

Two graphs carry the inductive bias of MOSAIC-GNN:

1. the *gene interaction graph*, shared by every patient, whose topology comes
   from a biological prior (PPI / pathway edges) fused with a co-alteration
   graph estimated on the training fold;
2. the *functional module assignment*, which turns thousands of gene nodes into
   a few dozen interpretable pathway-level super-nodes.

Both are fitted from training rows only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import GraphConfig

REL_PRIOR, REL_DATA, REL_SELF = 0, 1, 2


@dataclass
class GeneGraph:
    genes: List[str]
    edge_index: np.ndarray   # (2, E) int64, directed (both directions stored)
    edge_type: np.ndarray    # (E,) int64 in {REL_PRIOR, REL_DATA, REL_SELF}
    edge_weight: np.ndarray  # (E,) float32
    modules: np.ndarray      # (G,) int64 module id per gene
    module_names: List[str]

    @property
    def n_nodes(self) -> int:
        return len(self.genes)

    @property
    def n_modules(self) -> int:
        return int(self.modules.max()) + 1 if len(self.modules) else 0

    def summary(self) -> str:
        per_rel = {int(r): int((self.edge_type == r).sum())
                   for r in np.unique(self.edge_type)}
        return (f"GeneGraph(genes={self.n_nodes}, edges={self.edge_index.shape[1]}, "
                f"per_relation={per_rel}, modules={self.n_modules})")


# --------------------------------------------------------------------------- #
# Prior edges
# --------------------------------------------------------------------------- #
def load_ppi(path: Optional[str], genes: List[str]) -> List[Tuple[int, int, float]]:
    if not path or not os.path.exists(path):
        return []
    import pandas as pd

    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    a_col = next((c for c in cols if c in ("gene_a", "protein1", "source", "a")), cols[0])
    b_col = next((c for c in cols if c in ("gene_b", "protein2", "target", "b")), cols[1])
    w_col = next((c for c in cols if c in ("weight", "score", "combined_score")), None)

    pos = {g: i for i, g in enumerate(genes)}
    edges: List[Tuple[int, int, float]] = []
    for a, b, w in zip(df[a_col].astype(str), df[b_col].astype(str),
                       df[w_col] if w_col else [1.0] * len(df)):
        i, j = pos.get(a), pos.get(b)
        if i is None or j is None or i == j:
            continue
        wf = float(w)
        if wf > 1.0:  # STRING-style 0..1000
            wf /= 1000.0
        edges.append((i, j, wf))
    return edges


def load_gmt(path: Optional[str], genes: List[str]
             ) -> Tuple[Optional[np.ndarray], List[str], List[Tuple[int, int, float]]]:
    """Return module assignment + within-pathway edges from a .gmt file."""
    if not path or not os.path.exists(path):
        return None, [], []

    pos = {g: i for i, g in enumerate(genes)}
    names: List[str] = []
    members: List[List[int]] = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            idx = [pos[g] for g in parts[2:] if g in pos]
            if len(idx) >= 3:
                names.append(parts[0])
                members.append(idx)

    if not names:
        return None, [], []

    modules = np.full(len(genes), -1, dtype=np.int64)
    edges: List[Tuple[int, int, float]] = []
    for m, idx in enumerate(members):
        for k, i in enumerate(idx):
            if modules[i] < 0:
                modules[i] = m
            for j in idx[k + 1:]:
                edges.append((i, j, 1.0))
    unassigned = modules < 0
    if unassigned.any():  # leftovers share one "unannotated" module
        modules[unassigned] = len(names)
        names = names + ["UNANNOTATED"]
    return modules, names, edges


# --------------------------------------------------------------------------- #
# Data-driven edges
# --------------------------------------------------------------------------- #
def coalteration_edges(x_train: np.ndarray, cfg: GraphConfig
                       ) -> List[Tuple[int, int, float]]:
    """kNN graph over genes from cross-omic co-alteration correlation.

    `x_train` is (n_train, G, F); gene profiles are the concatenation of their
    value channels across patients, so two genes are linked when they are
    altered in the same patients across any modality.
    """
    n, g, f = x_train.shape
    half = f // 2
    prof = x_train[:, :, :half].transpose(1, 0, 2).reshape(g, -1)  # (G, n*half)
    prof = prof - prof.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(prof, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    prof = (prof / norm).astype(np.float32)

    edges: List[Tuple[int, int, float]] = []
    k = min(cfg.gene_knn, g - 1)
    chunk = max(1, int(2e7 // max(g, 1)))
    for start in range(0, g, chunk):
        stop = min(start + chunk, g)
        sim = prof[start:stop] @ prof.T                      # (c, G)
        for local, row in enumerate(sim):
            i = start + local
            row[i] = 0.0  # never let a gene be its own neighbour
            top = np.argpartition(-np.abs(row), k)[:k]
            for j in top:
                j = int(j)
                w = abs(float(row[j]))
                if j != i and np.isfinite(w) and w >= cfg.gene_corr_floor:
                    edges.append((i, j, min(w, 1.0)))
    return edges


def spectral_modules(edge_index: np.ndarray, edge_weight: np.ndarray,
                     n_nodes: int, n_modules: int, seed: int = 0
                     ) -> Tuple[np.ndarray, List[str]]:
    """Partition the gene graph into functional modules when no .gmt is given."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh
    from sklearn.cluster import KMeans

    a = coo_matrix((edge_weight, (edge_index[0], edge_index[1])),
                   shape=(n_nodes, n_nodes)).tocsr()
    a = ((a + a.T) / 2.0).tocsr()
    deg = np.asarray(a.sum(axis=1)).ravel()
    deg[deg < 1e-8] = 1e-8
    dinv = 1.0 / np.sqrt(deg)

    from scipy.sparse import diags
    lap_norm = diags(dinv) @ a @ diags(dinv)

    k = min(n_modules, max(2, n_nodes - 2))
    try:
        _, vecs = eigsh(lap_norm, k=k, which="LA")
    except Exception:  # tiny or degenerate graphs
        rng = np.random.default_rng(seed)
        vecs = rng.normal(size=(n_nodes, k))
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    modules = km.fit_predict(vecs).astype(np.int64)
    return modules, [f"MODULE_{i:03d}" for i in range(k)]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _to_arrays(edges: List[Tuple[int, int, float]], rel: int
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not edges:
        empty_i = np.zeros((2, 0), dtype=np.int64)
        return empty_i, np.zeros(0, np.int64), np.zeros(0, np.float32)
    arr = np.asarray(edges, dtype=np.float64)
    src = arr[:, 0].astype(np.int64)
    dst = arr[:, 1].astype(np.int64)
    w = arr[:, 2].astype(np.float32)
    # Symmetrise and drop duplicates.
    ei = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    w = np.concatenate([w, w])
    key = ei[0] * (ei.max() + 1) + ei[1]
    _, keep = np.unique(key, return_index=True)
    ei, w = ei[:, keep], w[keep]
    return ei, np.full(ei.shape[1], rel, dtype=np.int64), w


def build_gene_graph(genes: List[str], x_train: np.ndarray, cfg: GraphConfig,
                     ppi_path: Optional[str] = None,
                     gmt_path: Optional[str] = None,
                     seed: int = 0) -> GeneGraph:
    prior_edges = load_ppi(ppi_path, genes)
    modules, module_names, pathway_edges = load_gmt(gmt_path, genes)
    prior_edges = prior_edges + pathway_edges

    data_edges: List[Tuple[int, int, float]] = []
    if cfg.gene_graph_mode in ("data", "hybrid") or not prior_edges:
        data_edges = coalteration_edges(x_train, cfg)

    parts = []
    if prior_edges and cfg.gene_graph_mode in ("prior", "hybrid"):
        parts.append(_to_arrays(prior_edges, REL_PRIOR))
    if data_edges:
        parts.append(_to_arrays(data_edges, REL_DATA))

    g = len(genes)
    self_i = np.stack([np.arange(g), np.arange(g)]).astype(np.int64)
    parts.append((self_i, np.full(g, REL_SELF, np.int64), np.ones(g, np.float32)))

    edge_index = np.concatenate([p[0] for p in parts], axis=1)
    edge_type = np.concatenate([p[1] for p in parts])
    edge_weight = np.concatenate([p[2] for p in parts])

    if modules is None:
        mask = edge_type != REL_SELF
        if mask.sum() == 0:
            modules = np.zeros(g, np.int64)
            module_names = ["MODULE_000"]
        else:
            modules, module_names = spectral_modules(
                edge_index[:, mask], edge_weight[mask], g,
                cfg.n_modules, seed=seed)

    return GeneGraph(genes, edge_index, edge_type, edge_weight, modules, module_names)


def module_assignment_matrix(graph: GeneGraph) -> np.ndarray:
    """(G, M) row-normalised hard assignment, used by the pooling layer."""
    m = graph.n_modules
    s = np.zeros((graph.n_nodes, m), dtype=np.float32)
    s[np.arange(graph.n_nodes), graph.modules] = 1.0
    s /= np.maximum(s.sum(axis=0, keepdims=True), 1.0)
    return s
