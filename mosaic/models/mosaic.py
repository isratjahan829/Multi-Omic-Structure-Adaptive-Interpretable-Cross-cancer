"""MOSAIC-GNN.

Multi-Omic Structure-Adaptive Interpretable Cross-cancer Graph Neural Network.

The forward pass has four stages:

    genes ->  relational graph attention on a biological interaction network
           ->  attention pooling into functional modules
           ->  a disentangled patient latent (shared vs cohort-specific)
           ->  message passing on a *learned* patient-similarity graph

and three heads: class label, Cox risk, and a discrete-time hazard curve. The
cohort-adversarial branch on the shared latent is what makes the pan-cancer
representation transfer to a held-out cohort instead of memorising which
cohort a patient came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .layers import (EmbeddingBank, ModuleAttentionReadout, ModulePool,
                     PatientGCN, PatientGraphLearner, SharedTopologyRGAT,
                     grad_reverse)


@dataclass
class MosaicOutput:
    logits: torch.Tensor                 # (N, n_classes)
    risk: torch.Tensor                   # (N,) Cox log-hazard ratio
    hazard: torch.Tensor                 # (N, n_bins) discrete-time logits
    z_shared: torch.Tensor
    z_specific: torch.Tensor
    cohort_logits_adv: torch.Tensor      # from z_shared through gradient reversal
    cohort_logits_spec: torch.Tensor     # from z_specific, trained normally
    omic_views: Dict[str, torch.Tensor]  # per-block projections for contrast
    adjacency: Optional[torch.Tensor]    # learned patient graph
    module_attention: Optional[torch.Tensor]
    gene_gate: Optional[torch.Tensor]


class GeneEncoder(nn.Module):
    """Per-patient molecular interaction network -> module embeddings."""

    def __init__(self, in_dim: int, cfg: ModelConfig, n_relations: int,
                 assign: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Sequential(
            nn.Linear(in_dim, cfg.gene_hidden), nn.GELU(),
            nn.LayerNorm(cfg.gene_hidden),
        )
        # A learnable per-gene embedding lets the model carry gene identity
        # that is not expressible from the omic values alone.
        self.gene_embed = nn.Parameter(
            torch.randn(assign.shape[0], cfg.gene_hidden) * 0.02)

        self.layers = nn.ModuleList([
            SharedTopologyRGAT(cfg.gene_hidden, cfg.gene_hidden, n_relations,
                               heads=cfg.gene_heads, dropout=cfg.dropout,
                               edge_dropout=cfg.edge_dropout)
            for _ in range(cfg.gene_layers)
        ])
        self.pool = ModulePool(cfg.gene_hidden, cfg.module_hidden, assign,
                               dropout=cfg.dropout)
        self.readout = ModuleAttentionReadout(cfg.module_hidden,
                                              cfg.patient_hidden,
                                              dropout=cfg.dropout)
        # Fallback path when the gene graph is ablated away.
        self.flat = nn.Sequential(
            nn.Linear(cfg.gene_hidden, cfg.module_hidden), nn.GELU(),
            nn.LayerNorm(cfg.module_hidden),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor, edge_weight: torch.Tensor,
                explain: bool = False):
        h = self.input(x) + self.gene_embed
        attentions = []
        if self.cfg.use_gene_graph:
            for layer in self.layers:
                h, att = layer(h, edge_index, edge_type, edge_weight,
                               return_attention=explain)
                if explain:
                    attentions.append(att)

        if self.cfg.use_module_pool:
            m, gate = self.pool(h, return_gate=explain)
            z, mod_att = self.readout(m, return_attention=explain)
        else:
            m = self.flat(h)
            gate = None
            z, mod_att = self.readout(m.mean(dim=1, keepdim=True),
                                      return_attention=explain)
        return z, mod_att, gate, attentions


class MosaicGNN(nn.Module):
    def __init__(self, n_node_features: int, n_clinical: int, n_classes: int,
                 n_cohorts: int, n_relations: int, assign: torch.Tensor,
                 n_patients: int, block_names, cfg: ModelConfig,
                 class_head_source: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.block_names = list(block_names)
        self.n_classes = n_classes
        # "auto" must already be resolved by the caller, which is the only
        # place that knows the classification target.
        self.class_head_source = class_head_source or cfg.class_head_source
        if self.class_head_source == "auto":
            self.class_head_source = "shared"
        self.survival_head_source = cfg.survival_head_source

        self.encoder = GeneEncoder(n_node_features, cfg, n_relations, assign)

        self.clinical = nn.Sequential(
            nn.Linear(max(n_clinical, 1), cfg.clinical_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.LayerNorm(cfg.clinical_hidden),
        ) if cfg.use_clinical and n_clinical > 0 else None

        fuse_in = cfg.patient_hidden + (cfg.clinical_hidden if self.clinical else 0)
        z_dim = cfg.z_shared + cfg.z_specific
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, cfg.patient_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.patient_hidden, z_dim),
        )

        self.graph_learner = PatientGraphLearner(
            z_dim, heads=4, k=10, temp=0.2) if cfg.use_patient_graph else None
        self.patient_gnn = PatientGCN(z_dim, cfg.patient_layers,
                                      cfg.dropout) if cfg.use_patient_graph else None
        self.bank = EmbeddingBank(n_patients, z_dim)

        def src_dim(src: str) -> int:
            if not cfg.use_disentangle:
                # Without disentangling there is one latent; "specific" still
                # means its leading slice so the cohort head keeps its shape.
                return cfg.z_specific if src == "specific" else z_dim
            return {"shared": cfg.z_shared,
                    "specific": cfg.z_specific,
                    "full": z_dim}[src]

        cls_in = src_dim(self.class_head_source)
        surv_in = src_dim(self.survival_head_source)
        self.cls_head = nn.Sequential(
            nn.Linear(cls_in, cfg.patient_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.patient_hidden, max(n_classes, 2)),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(surv_in, cfg.patient_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.patient_hidden, 1),
        )
        self.hazard_head = nn.Sequential(
            nn.Linear(surv_in, cfg.patient_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.patient_hidden, cfg.n_survival_bins),
        )

        # Residual linear clinical terms, zero-initialised so training starts
        # from the purely deep model and grows the linear part only where it
        # helps. Kept out of the graph path on purpose.
        use_res = cfg.clinical_residual and cfg.use_clinical and n_clinical > 0
        if use_res:
            self.res_cls = nn.Linear(n_clinical, max(n_classes, 2), bias=False)
            self.res_risk = nn.Linear(n_clinical, 1, bias=False)
            self.res_hazard = nn.Linear(n_clinical, cfg.n_survival_bins, bias=False)
            for layer in (self.res_cls, self.res_risk, self.res_hazard):
                nn.init.zeros_(layer.weight)
        else:
            self.res_cls = self.res_risk = self.res_hazard = None
        self.cohort_adv = nn.Linear(cfg.z_shared, n_cohorts)
        self.cohort_spec = nn.Linear(cfg.z_specific, n_cohorts)

        # Per-omic projection heads for the cross-omic contrastive term.
        self.omic_proj = nn.ModuleDict({
            b: nn.Sequential(nn.Linear(cfg.patient_hidden, 64), nn.GELU(),
                             nn.Linear(64, 64))
            for b in self.block_names
        }) if cfg.use_cross_omic_contrast else None

    # ------------------------------------------------------------------ #
    def _omic_views(self, x: torch.Tensor, edge_index, edge_type, edge_weight
                    ) -> Dict[str, torch.Tensor]:
        """Encode each modality alone; used only for the contrastive loss.

        Masking one block at a time and asking the encoder to land in the same
        neighbourhood forces the shared latent to encode biology that is
        corroborated across modalities rather than a single noisy assay.
        """
        views: Dict[str, torch.Tensor] = {}
        nb = len(self.block_names)
        for i, name in enumerate(self.block_names):
            xi = torch.zeros_like(x)
            xi[:, :, i] = x[:, :, i]
            xi[:, :, nb + i] = x[:, :, nb + i]
            z, _, _, _ = self.encoder(xi, edge_index, edge_type, edge_weight)
            views[name] = F.normalize(self.omic_proj[name](z), dim=-1)
        return views

    def forward(self, x: torch.Tensor, clinical: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor,
                edge_weight: torch.Tensor,
                bank_idx: Optional[torch.Tensor] = None,
                anchor_idx: Optional[torch.Tensor] = None,
                grl_lambda: float = 1.0,
                explain: bool = False,
                contrast: bool = False) -> MosaicOutput:
        z_gene, mod_att, gene_gate, _ = self.encoder(
            x, edge_index, edge_type, edge_weight, explain=explain)

        parts = [z_gene]
        if self.clinical is not None:
            parts.append(self.clinical(clinical))
        z = self.fuse(torch.cat(parts, dim=-1))

        adjacency = None
        if self.graph_learner is not None:
            if anchor_idx is not None and anchor_idx.numel() > 0:
                anchors = self.bank.gather(anchor_idx)
                nodes = torch.cat([z, anchors], dim=0)
                mask = torch.zeros(nodes.shape[0], dtype=torch.bool, device=z.device)
                mask[z.shape[0]:] = True
                mask[:z.shape[0]] = True  # in-batch training rows are anchors too
            else:
                nodes, mask = z, None
            adjacency = self.graph_learner(nodes, mask)
            nodes = self.patient_gnn(nodes, adjacency)
            z = nodes[: z.shape[0]]

        if bank_idx is not None:
            self.bank.update(bank_idx, z)

        if self.cfg.use_disentangle:
            z_shared = z[:, : self.cfg.z_shared]
            z_specific = z[:, self.cfg.z_shared:]
        else:
            z_shared, z_specific = z, z[:, : self.cfg.z_specific]

        def pick(src: str) -> torch.Tensor:
            if src == "specific":
                return z_specific
            if src == "full":
                return z
            return z_shared

        cls_in = pick(self.class_head_source)
        surv_in = pick(self.survival_head_source)

        logits = self.cls_head(cls_in)
        risk = self.risk_head(surv_in).squeeze(-1)
        hazard = self.hazard_head(surv_in)
        if self.res_cls is not None:
            c = clinical.to(logits.dtype)
            logits = logits + self.res_cls(c)
            risk = risk + self.res_risk(c).squeeze(-1)
            hazard = hazard + self.res_hazard(c)

        return MosaicOutput(
            logits=logits,
            risk=risk,
            hazard=hazard,
            z_shared=z_shared,
            z_specific=z_specific,
            cohort_logits_adv=self.cohort_adv(
                grad_reverse(z[:, : self.cfg.z_shared], grl_lambda)),
            cohort_logits_spec=self.cohort_spec(z[:, self.cfg.z_shared:]),
            omic_views=(self._omic_views(x, edge_index, edge_type, edge_weight)
                        if (contrast and self.omic_proj is not None) else {}),
            adjacency=adjacency,
            module_attention=mod_att,
            gene_gate=gene_gate,
        )
