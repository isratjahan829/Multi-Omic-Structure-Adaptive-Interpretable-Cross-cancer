"""Objective terms."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor,
                eps: float = 1e-7) -> torch.Tensor:
    """Breslow negative partial log-likelihood, averaged over events.

    Risk sets are formed inside the mini-batch, which is the standard
    approximation; with full-batch training it is exact.
    """
    valid = torch.isfinite(time) & torch.isfinite(event)
    if valid.sum() < 2 or event[valid].sum() < 1:
        return risk.sum() * 0.0
    risk, time, event = risk[valid], time[valid], event[valid]

    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    risk = risk - risk.max().detach()          # numerical stability
    log_cum = torch.logcumsumexp(risk, dim=0)  # risk set = all with T >= t_i
    ll = (risk - log_cum) * event
    return -ll.sum() / event.sum().clamp_min(1.0)


def discrete_hazard_loss(logits: torch.Tensor, bin_idx: torch.Tensor,
                         event: torch.Tensor) -> torch.Tensor:
    """Likelihood of a discrete-time hazard model with right censoring.

    An observed event contributes log h(k) + sum_{j<k} log(1 - h(j)); a censored
    patient only contributes the survival part. This head gives a calibrated
    survival curve, which the Cox risk score on its own cannot.
    """
    valid = (bin_idx >= 0) & torch.isfinite(event)
    if valid.sum() == 0:
        return logits.sum() * 0.0
    logits, bin_idx, event = logits[valid], bin_idx[valid], event[valid]
    n, k = logits.shape
    bin_idx = bin_idx.clamp(max=k - 1)

    log_h = F.logsigmoid(logits)
    log_1mh = F.logsigmoid(-logits)
    ar = torch.arange(k, device=logits.device).unsqueeze(0)
    before = (ar < bin_idx.unsqueeze(1)).float()
    at = F.one_hot(bin_idx, k).float()

    surv = (log_1mh * before).sum(1)
    ll = surv + event * (log_h * at).sum(1) + (1 - event) * (log_1mh * at).sum(1)
    return -ll.mean()


def orthogonality_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius norm of the cross-covariance of two latents."""
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    a = a / (a.norm(dim=0, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=0, keepdim=True) + 1e-8)
    return (a.t() @ b).pow(2).mean()


def cross_omic_contrast(views: Dict[str, torch.Tensor],
                        temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE across modalities: the same patient seen through two assays is
    a positive pair, different patients are negatives."""
    names = list(views)
    if len(names) < 2:
        return torch.zeros((), device=next(iter(views.values())).device) \
            if views else torch.zeros(())
    total, count = 0.0, 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = views[names[i]], views[names[j]]
            logits = a @ b.t() / temperature
            target = torch.arange(a.shape[0], device=a.device)
            total = total + 0.5 * (F.cross_entropy(logits, target)
                                   + F.cross_entropy(logits.t(), target))
            count += 1
    return total / max(count, 1)


def graph_regulariser(adj: torch.Tensor, z: torch.Tensor,
                      w_smooth: float = 1.0, w_sparse: float = 0.1,
                      w_degree: float = 0.1) -> torch.Tensor:
    """Keep the learned patient graph smooth, sparse and non-degenerate."""
    n = adj.shape[0]
    dist = torch.cdist(z, z).pow(2)
    smooth = (adj * dist).sum() / (n * n)
    sparse = adj.pow(2).sum() / (n * n)
    degree = -torch.log(adj.sum(1) + 1e-8).mean() / n
    return w_smooth * smooth + w_sparse * sparse + w_degree * degree


class MosaicLoss(nn.Module):
    def __init__(self, cfg, class_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "class_weight",
            class_weight if class_weight is not None else torch.tensor([]))

    def forward(self, out, batch: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        c = self.cfg
        terms: Dict[str, torch.Tensor] = {}
        dev = out.logits.device
        zero = torch.zeros((), device=dev)

        y = batch["label"]
        lab_valid = y >= 0
        terms["class"] = (F.cross_entropy(
            out.logits[lab_valid], y[lab_valid],
            weight=self.class_weight if self.class_weight.numel() else None,
            label_smoothing=c.label_smoothing) if lab_valid.any() else zero)

        terms["cox"] = cox_ph_loss(out.risk, batch["time"], batch["event"])
        terms["hazard"] = discrete_hazard_loss(out.hazard, batch["bin"],
                                               batch["event"])
        terms["adversarial"] = F.cross_entropy(out.cohort_logits_adv,
                                               batch["cohort"])
        terms["cohort"] = F.cross_entropy(out.cohort_logits_spec, batch["cohort"])
        terms["orthogonal"] = orthogonality_loss(out.z_shared, out.z_specific)
        terms["contrast"] = (cross_omic_contrast(out.omic_views, c.contrast_temp)
                             if out.omic_views else zero)
        terms["graph_reg"] = (graph_regulariser(
            out.adjacency, torch.cat([out.z_shared, out.z_specific], -1)
            if out.adjacency.shape[0] == out.z_shared.shape[0] else
            out.adjacency.new_zeros(out.adjacency.shape[0], 1))
            if out.adjacency is not None else zero)

        total = (c.w_class * terms["class"]
                 + c.w_cox * terms["cox"]
                 + c.w_hazard * terms["hazard"]
                 + batch.get("grl_weight", 1.0) * c.w_adversarial * terms["adversarial"]
                 + c.w_cohort * terms["cohort"]
                 + c.w_orthogonal * terms["orthogonal"]
                 + c.w_contrast * terms["contrast"]
                 + c.w_graph_reg * terms["graph_reg"])
        terms["total"] = total
        return terms
