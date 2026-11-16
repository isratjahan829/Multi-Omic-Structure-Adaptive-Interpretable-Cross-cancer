"""Interpretation.

Three complementary attributions, because attention alone is not evidence:

* module attention  -- which functional modules drive the patient embedding;
* gene gate         -- which genes dominate inside those modules;
* integrated gradients on the node features -- signed, per-omic attribution
  that survives the "attention is not explanation" critique.

Outputs are plain tables ready for pathway enrichment or a figure.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def collect_attention(model, tensors, idx: np.ndarray, device,
                      batch_size: int = 32) -> Dict[str, np.ndarray]:
    model.eval()
    mods, gates = [], []
    for s in range(0, len(idx), batch_size):
        sel = torch.from_numpy(idx[s: s + batch_size]).long()
        out = model(tensors.x[sel].to(device), tensors.clinical[sel].to(device),
                    tensors.edge_index, tensors.edge_type, tensors.edge_weight,
                    bank_idx=None, anchor_idx=None, grl_lambda=0.0, explain=True)
        if out.module_attention is not None:
            mods.append(out.module_attention.float().cpu().numpy())
        if out.gene_gate is not None:
            gates.append(out.gene_gate.float().cpu().numpy())
    return {
        "module_attention": np.concatenate(mods) if mods else np.zeros((0, 0)),
        "gene_gate": np.concatenate(gates) if gates else np.zeros((0, 0)),
    }


def integrated_gradients(model, tensors, idx: np.ndarray, device,
                         target: str = "risk", steps: int = 32,
                         batch_size: int = 8) -> np.ndarray:
    """(n, G, F) attributions against an all-zero (population) baseline."""
    model.eval()
    attrs = []
    for s in range(0, len(idx), batch_size):
        sel = torch.from_numpy(idx[s: s + batch_size]).long()
        x = tensors.x[sel].to(device)
        clin = tensors.clinical[sel].to(device)
        base = torch.zeros_like(x)
        total = torch.zeros_like(x)
        for a in torch.linspace(1.0 / steps, 1.0, steps, device=device):
            xi = (base + a * (x - base)).requires_grad_(True)
            out = model(xi, clin, tensors.edge_index, tensors.edge_type,
                        tensors.edge_weight, bank_idx=None, anchor_idx=None,
                        grl_lambda=0.0)
            score = (out.risk.sum() if target == "risk"
                     else out.logits.max(dim=1).values.sum())
            grad = torch.autograd.grad(score, xi)[0]
            total = total + grad.detach()
        attrs.append(((x - base) * total / steps).cpu().numpy())
    return np.concatenate(attrs, axis=0)


def gene_importance_table(genes: List[str], modules: np.ndarray,
                          module_names: List[str],
                          gate: np.ndarray, ig: Optional[np.ndarray] = None,
                          block_names: Optional[List[str]] = None
                          ) -> pd.DataFrame:
    rows = {
        "gene": genes,
        "module": [module_names[m] if m < len(module_names) else "NA"
                   for m in modules],
    }
    if gate.size:
        rows["gate_mean"] = gate.mean(0)
        rows["gate_std"] = gate.std(0)
    if ig is not None and ig.size:
        nb = ig.shape[2] // 2
        rows["ig_abs"] = np.abs(ig[:, :, :nb]).mean(0).sum(1)
        for i, name in enumerate(block_names or []):
            if i < nb:
                rows[f"ig_{name}"] = ig[:, :, i].mean(0)
    df = pd.DataFrame(rows)
    sort_key = "ig_abs" if "ig_abs" in df else "gate_mean"
    return df.sort_values(sort_key, ascending=False).reset_index(drop=True)


def module_importance_table(module_names: List[str], attention: np.ndarray,
                            modules: np.ndarray, genes: List[str],
                            top_genes: int = 8,
                            gate: Optional[np.ndarray] = None) -> pd.DataFrame:
    rows = []
    mean_att = attention.mean(0) if attention.size else np.zeros(len(module_names))
    mean_gate = gate.mean(0) if gate is not None and gate.size else None
    for m, name in enumerate(module_names):
        members = np.where(modules == m)[0]
        if members.size == 0:
            continue
        if mean_gate is not None:
            order = members[np.argsort(-mean_gate[members])][:top_genes]
        else:
            order = members[:top_genes]
        rows.append({
            "module": name,
            "size": int(members.size),
            "attention": float(mean_att[m]) if m < len(mean_att) else 0.0,
            "top_genes": ", ".join(genes[i] for i in order),
        })
    return pd.DataFrame(rows).sort_values("attention", ascending=False
                                          ).reset_index(drop=True)


def personalised_subnetwork(model, tensors, patient_row: int, device,
                            top_edges: int = 200) -> pd.DataFrame:
    """Edges of one patient's rewired interaction network, for a figure."""
    model.eval()
    x = tensors.x[patient_row: patient_row + 1].to(device)
    clin = tensors.clinical[patient_row: patient_row + 1].to(device)
    with torch.no_grad():
        h = model.encoder.input(x) + model.encoder.gene_embed
        _, alpha = model.encoder.layers[-1](
            h, tensors.edge_index, tensors.edge_type, tensors.edge_weight,
            return_attention=True)
    if alpha is None:
        return pd.DataFrame()
    weight = alpha[0].mean(-1).cpu().numpy()          # (E,)
    ei = tensors.edge_index.cpu().numpy()
    keep = np.argsort(-weight)[:top_edges]
    genes = tensors.graph.genes
    return pd.DataFrame({
        "source": [genes[i] for i in ei[0, keep]],
        "target": [genes[i] for i in ei[1, keep]],
        "attention": weight[keep],
        "relation": tensors.edge_type.cpu().numpy()[keep],
    })


def export_ranked_gene_list(df: pd.DataFrame, path: str,
                            score_col: str = "ig_abs") -> None:
    """Write a .rnk file that GSEA / g:Profiler can consume directly."""
    col = score_col if score_col in df.columns else df.columns[-1]
    df[["gene", col]].to_csv(path, sep="\t", index=False, header=False)
