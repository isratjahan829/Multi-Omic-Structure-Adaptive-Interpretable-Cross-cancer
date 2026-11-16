"""Building blocks.

All message passing is written directly against `index_add_` on a *shared*
topology with per-patient node features, i.e. one edge list of shape (2, E) and
a feature tensor of shape (B, G, F). That keeps the gene graph in memory once
instead of B times and removes the dependency on torch-geometric, which is the
usual blocker for reproducing multi-omics GNNs on a fresh machine.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Gradient reversal (domain-adversarial disentanglement)
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        return -ctx.lambd * grad_out, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


# --------------------------------------------------------------------------- #
# Relational graph attention over a shared topology
# --------------------------------------------------------------------------- #
class SharedTopologyRGAT(nn.Module):
    """Multi-head, multi-relation attention with patient-specific weights.

    The topology is fixed (protein interactions do not depend on the patient)
    but the attention coefficients do, so every patient effectively gets their
    own re-weighted interaction network -- that per-patient rewiring is what we
    read out later as a personalised subnetwork.
    """

    def __init__(self, in_dim: int, out_dim: int, n_relations: int,
                 heads: int = 4, dropout: float = 0.1,
                 edge_dropout: float = 0.0, residual: bool = True):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.heads = heads
        self.d = out_dim // heads
        self.out_dim = out_dim
        self.n_relations = n_relations
        self.edge_dropout = edge_dropout

        self.lin = nn.Linear(in_dim, out_dim * n_relations, bias=False)
        self.att_src = nn.Parameter(torch.empty(n_relations, heads, self.d))
        self.att_dst = nn.Parameter(torch.empty(n_relations, heads, self.d))
        self.rel_bias = nn.Parameter(torch.zeros(n_relations, heads))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.res_lin = (nn.Identity() if in_dim == out_dim
                        else nn.Linear(in_dim, out_dim, bias=False))
        self.norm = nn.LayerNorm(out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor, edge_weight: torch.Tensor,
                return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, g, _ = x.shape
        src, dst = edge_index[0], edge_index[1]
        e = src.numel()

        if self.training and self.edge_dropout > 0:
            keep = torch.rand(e, device=x.device) >= self.edge_dropout
            keep |= edge_type == (self.n_relations - 1)  # never drop self-loops
            src, dst = src[keep], dst[keep]
            edge_type, edge_weight = edge_type[keep], edge_weight[keep]
            e = src.numel()

        h = self.lin(x).view(b, g, self.n_relations, self.heads, self.d)
        h = h.permute(2, 0, 1, 3, 4)  # (R, B, G, H, d)

        # Per-relation attention logits, gathered onto edges.
        a_src = (h * self.att_src[:, None, None]).sum(-1)  # (R, B, G, H)
        a_dst = (h * self.att_dst[:, None, None]).sum(-1)
        rel_idx = edge_type
        logit = (a_src[rel_idx, :, src] + a_dst[rel_idx, :, dst])  # (E, B, H)
        logit = logit.permute(1, 0, 2)                              # (B, E, H)
        logit = F.leaky_relu(logit + self.rel_bias[rel_idx][None], 0.2)
        logit = logit + torch.log(edge_weight.clamp_min(1e-6))[None, :, None]

        # Segment softmax. Softmax is shift invariant, so subtracting the global
        # maximum is exact and avoids a scatter-max.
        logit = logit - logit.amax()
        expo = logit.exp()
        denom = torch.zeros(b, g, self.heads, device=x.device, dtype=expo.dtype)
        denom.index_add_(1, dst, expo)
        alpha = expo / (denom[:, dst] + 1e-16)
        alpha = self.dropout(alpha)

        msg = h[rel_idx, :, src] .permute(1, 0, 2, 3)  # (B, E, H, d)
        msg = msg * alpha.unsqueeze(-1)
        out = torch.zeros(b, g, self.heads, self.d, device=x.device, dtype=msg.dtype)
        out.index_add_(1, dst, msg)
        out = out.reshape(b, g, self.out_dim) + self.bias

        if self.residual:
            out = out + self.res_lin(x)
        out = self.norm(out)
        return out, (alpha.detach() if return_attention else None)


# --------------------------------------------------------------------------- #
# Functional-module pooling
# --------------------------------------------------------------------------- #
class ModulePool(nn.Module):
    """Gene -> pathway pooling with a learned within-module gate.

    `assign` is a fixed (G, M) matrix from pathway annotation or spectral
    partitioning. The gate scores each gene inside its module, so the resulting
    module embedding is an attention-weighted summary and the gate values are
    directly reportable as per-patient gene importances.
    """

    def __init__(self, in_dim: int, out_dim: int, assign: torch.Tensor,
                 dropout: float = 0.1):
        super().__init__()
        self.register_buffer("assign", assign)          # (G, M)
        self.register_buffer("member", (assign > 0).float())
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.GELU(),
            nn.Linear(in_dim // 2, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, h: torch.Tensor, return_gate: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b = h.shape[0]
        score = self.gate(h).squeeze(-1)                 # (B, G)
        # Softmax restricted to each gene's own module.
        score = score - score.amax(dim=1, keepdim=True)
        expo = score.exp().unsqueeze(-1) * self.member   # (B, G, M)
        weight = expo / (expo.sum(dim=1, keepdim=True) + 1e-12)
        pooled = torch.einsum("bgm,bgf->bmf", weight, h)  # (B, M, F)
        pooled = self.proj(pooled)
        gate = weight.sum(-1) if return_gate else None
        return pooled, gate


class ModuleAttentionReadout(nn.Module):
    """Attention over functional modules -> one patient vector."""

    def __init__(self, dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Linear(dim, out_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.LayerNorm(out_dim))

    def forward(self, m: torch.Tensor, return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        k = self.key(m)
        logits = (k * self.query).sum(-1) / math.sqrt(k.shape[-1])
        attn = torch.softmax(logits, dim=1)              # (B, M)
        z = torch.einsum("bm,bmf->bf", attn, self.value(m))
        return self.out(z), (attn if return_attention else None)


# --------------------------------------------------------------------------- #
# Learned patient graph
# --------------------------------------------------------------------------- #
class PatientGraphLearner(nn.Module):
    """Differentiable kNN graph over patient embeddings.

    Similarity is computed in several learned metric subspaces (heads) and the
    resulting dense affinity is sparsified with a straight-through top-k, so
    gradients reach the metric even though the forward pass is sparse.
    """

    def __init__(self, dim: int, heads: int = 4, k: int = 10, temp: float = 0.2):
        super().__init__()
        self.heads, self.k, self.temp = heads, k, temp
        self.metric = nn.Parameter(torch.ones(heads, dim))

    def forward(self, z: torch.Tensor, anchor_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """Return a row-stochastic (N, N) adjacency with a self-loop.

        The whole block runs with autocast disabled and in fp32. Under mixed
        precision the scatter targets and the scattered values are produced by
        different ops, so autocast would hand them different dtypes and
        ``scatter_`` would raise; forcing fp32 here also keeps the top-k
        straight-through estimator numerically stable.
        """
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            zh = z.unsqueeze(0) * self.metric.float().unsqueeze(1)  # (H, N, D)
            zh = F.normalize(zh, dim=-1)
            sim = torch.einsum("hnd,hmd->nm", zh, zh) / self.heads
            sim = sim / self.temp

            n = z.shape[0]
            eye = torch.eye(n, device=z.device, dtype=torch.bool)
            sim = sim.masked_fill(eye, float("-inf"))
            if anchor_mask is not None:
                # Only training anchors may act as sources of information.
                sim = sim.masked_fill(~anchor_mask.view(1, -1), float("-inf"))

            k = int(min(self.k, max(1, int(anchor_mask.sum())
                                    if anchor_mask is not None else n - 1)))
            topv, topi = sim.topk(k, dim=1)
            hard = torch.zeros_like(sim).scatter_(1, topi, 1.0)
            soft = torch.softmax(sim.nan_to_num(neginf=-1e9), dim=1)
            mask = hard + soft - soft.detach()                  # straight-through

            adj = torch.softmax(topv, dim=1).to(sim.dtype)
            dense = torch.zeros_like(sim).scatter_(1, topi, adj) * mask
            dense = dense + torch.eye(n, device=z.device, dtype=sim.dtype)
            return dense / dense.sum(dim=1, keepdim=True).clamp_min(1e-8)


class PatientGCN(nn.Module):
    def __init__(self, dim: int, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])

    def forward(self, z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = z
        for block, norm in zip(self.blocks, self.norms):
            h = norm(h + block(adj @ h))
        return h


# --------------------------------------------------------------------------- #
# Historical embedding bank (mini-batch training with a global patient graph)
# --------------------------------------------------------------------------- #
class EmbeddingBank(nn.Module):
    """Cached embeddings of patients outside the current mini-batch.

    The patient graph is global but the gene-level encoder is far too heavy to
    run on every patient at once, so out-of-batch neighbours contribute through
    their most recent embedding (detached). This is the standard historical
    embedding approximation and it keeps memory linear in the batch size.
    """

    def __init__(self, n_patients: int, dim: int, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("bank", torch.zeros(n_patients, dim))
        self.register_buffer("filled", torch.zeros(n_patients, dtype=torch.bool))

    @torch.no_grad()
    def update(self, idx: torch.Tensor, z: torch.Tensor) -> None:
        z = z.detach().float()
        fresh = ~self.filled[idx]
        prev = self.bank[idx]
        new = torch.where(fresh.unsqueeze(-1), z,
                          self.momentum * prev + (1 - self.momentum) * z)
        self.bank[idx] = new
        self.filled[idx] = True

    def gather(self, idx: torch.Tensor,
               dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        out = self.bank[idx]
        return out if dtype is None else out.to(dtype)
