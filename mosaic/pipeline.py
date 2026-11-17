"""End-to-end experiment: nested cross-validation, training, evaluation.

Fold protocol
-------------
Repeated stratified k-fold on the *pan-cancer* patient set, stratified by
cohort x class x event so every fold sees the same case mix. Gene selection,
graph construction, scalers and survival bins are re-fitted inside each
training fold; the held-out fold is only ever passed through `transform`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time as _time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from .config import Config
from .data import graphs as G
from .data.loader import Cohort, load_all
from .data.preprocess import (PanCohortData, assemble, fit_preprocessor,
                              select_genes, stratification_key, transform,
                              valid_mask)
from .losses import MosaicLoss
from .metrics import (bootstrap_ci, classification_metrics, concordance_index,
                      expected_calibration_error, km_curve, logrank_test,
                      summarise, time_dependent_auc)
from .models.mosaic import MosaicGNN


# --------------------------------------------------------------------------- #
@dataclass
class FoldTensors:
    x: torch.Tensor           # (N, G, F) node features for every patient
    clinical: torch.Tensor    # (N, C)
    label: torch.Tensor
    time: torch.Tensor
    event: torch.Tensor
    bin: torch.Tensor
    cohort: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: torch.Tensor
    assign: torch.Tensor
    graph: G.GeneGraph


def _fit_fingerprint(t: FoldTensors, train_idx: np.ndarray,
                     val_idx: np.ndarray, n_classes: int, epochs: int,
                     seed: int) -> str:
    """Identify a fit, so a resume checkpoint is never applied to another one.

    Everything that would change the optimisation goes in: the gene axis, the
    exact training and validation rows, the class count, the epoch budget and
    the seed. Two fits that agree on all of these are the same fit.
    """
    h = hashlib.sha256()
    for part in (",".join(map(str, t.graph.genes)),
                 ",".join(map(str, np.sort(np.asarray(train_idx)))),
                 ",".join(map(str, np.sort(np.asarray(val_idx)))),
                 f"{n_classes}|{epochs}|{seed}|{tuple(t.x.shape)}"):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


class Experiment:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(
            cfg.train.device if torch.cuda.is_available() else "cpu")
        self.cohorts: Dict[str, Cohort] = {}
        self.skeleton: Optional[PanCohortData] = None
        self.offsets: Dict[str, Tuple[int, int]] = {}
        os.makedirs(cfg.train.out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def prepare(self) -> None:
        self.cohorts = load_all(self.cfg.data)
        start = 0
        for name, cohort in self.cohorts.items():
            self.offsets[name] = (start, start + len(cohort))
            start += len(cohort)
        self.skeleton = assemble(self.cohorts, [], self.cfg.data)

        n_cls = int(self.skeleton.label.max()) + 1 if (self.skeleton.label >= 0).any() else 2
        print(f"[data] patients={len(self.skeleton)} cohorts={self.skeleton.cohort_names} "
              f"classes={n_cls} blocks={sorted(self.skeleton.blocks)}")
        ev = np.nan_to_num(self.skeleton.event, nan=0).sum()
        print(f"[data] survival events={int(ev)} "
              f"median_followup={np.nanmedian(self.skeleton.time):.1f} months")

    # ------------------------------------------------------------------ #
    def _cohort_train_idx(self, train_idx: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        tr = np.zeros(len(self.skeleton), dtype=bool)
        tr[train_idx] = True
        for name, (a, b) in self.offsets.items():
            out[name] = np.where(tr[a:b])[0]
        return out

    def build_fold(self, train_idx: np.ndarray) -> Tuple[FoldTensors, object, PanCohortData]:
        cfg = self.cfg
        genes = select_genes(self.cohorts, self._cohort_train_idx(train_idx), cfg.data)
        data = assemble(self.cohorts, genes, cfg.data)
        pp = fit_preprocessor(data, train_idx, cfg.data, cfg.model.n_survival_bins)
        x, clin, ybin = transform(data, pp)

        graph = G.build_gene_graph(genes, x[train_idx], cfg.graph,
                                   cfg.data.ppi_edgelist, cfg.data.pathway_gmt,
                                   seed=cfg.train.seed)
        print(f"[graph] {graph.summary()}")
        assign = G.module_assignment_matrix(graph)

        t = FoldTensors(
            x=torch.from_numpy(x),
            clinical=torch.from_numpy(clin),
            label=torch.from_numpy(data.label),
            time=torch.from_numpy(data.time),
            event=torch.from_numpy(np.nan_to_num(data.event, nan=0.0)),
            bin=torch.from_numpy(ybin),
            cohort=torch.from_numpy(data.cohort_id),
            edge_index=torch.from_numpy(graph.edge_index).to(self.device),
            edge_type=torch.from_numpy(graph.edge_type).to(self.device),
            edge_weight=torch.from_numpy(graph.edge_weight).to(self.device),
            assign=torch.from_numpy(assign),
            graph=graph,
        )
        return t, pp, data

    # ------------------------------------------------------------------ #
    def _resolve_class_head(self) -> str:
        """Turn cfg.model.class_head_source == "auto" into a concrete slice.

        Tissue of origin and molecular subtype are defined by the cohort, so
        they are read from z_specific; everything else (stage, grade) is read
        from the cohort-invariant z_shared.
        """
        src = self.cfg.model.class_head_source
        if src != "auto":
            return src
        return ("specific" if self.cfg.data.class_target in ("cohort", "subtype")
                else "shared")

    def build_model(self, t: FoldTensors, n_classes: int) -> MosaicGNN:
        cfg = self.cfg
        model = MosaicGNN(
            n_node_features=t.x.shape[2],
            n_clinical=t.clinical.shape[1],
            n_classes=n_classes,
            n_cohorts=len(self.skeleton.cohort_names),
            n_relations=cfg.graph.n_relations,
            assign=t.assign,
            n_patients=t.x.shape[0],
            block_names=sorted(self.skeleton.blocks),
            cfg=cfg.model,
            class_head_source=self._resolve_class_head(),
        ).to(self.device)
        if model.graph_learner is not None:
            model.graph_learner.k = cfg.graph.patient_knn
            model.graph_learner.temp = cfg.graph.graph_learner_temp
        return model

    def _batch(self, t: FoldTensors, idx: np.ndarray) -> Dict[str, torch.Tensor]:
        i = torch.from_numpy(idx).long()
        d = self.device
        return {
            "x": t.x[i].to(d, non_blocking=True),
            "clinical": t.clinical[i].to(d),
            "label": t.label[i].to(d),
            "time": t.time[i].to(d),
            "event": t.event[i].to(d),
            "bin": t.bin[i].to(d),
            "cohort": t.cohort[i].to(d),
            "index": i.to(d),
        }

    # ------------------------------------------------------------------ #
    def _augment(self, x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Gene dropout + input jitter on a training batch.

        ``x`` is (batch, genes, 2 * n_blocks): the first half of the channel
        axis holds omic values, the second half the per-block availability
        mask. A dropped gene loses both halves, so the model cannot read
        "this gene was dropped" off a mask channel that disagrees with a
        zeroed value channel -- it simply sees a gene that was never measured,
        which is how omic data really goes missing.
        """
        cfg = self.cfg.train
        if cfg.gene_dropout <= 0 and cfg.input_noise <= 0:
            return x
        x = x.clone()
        nb = x.shape[2] // 2
        if cfg.gene_dropout > 0:
            keep = torch.from_numpy(
                (rng.random((x.shape[0], x.shape[1])) >= cfg.gene_dropout)
            ).to(x.device, x.dtype).unsqueeze(-1)
            x = x * keep
        if cfg.input_noise > 0:
            noise = torch.randn(x[:, :, :nb].shape, device=x.device, dtype=x.dtype)
            # Jitter only where the value is actually observed.
            x[:, :, :nb] = x[:, :, :nb] + cfg.input_noise * noise * x[:, :, nb:]
        return x

    def train_fold(self, t: FoldTensors, train_idx: np.ndarray,
                   val_idx: np.ndarray, n_classes: int,
                   verbose: bool = True,
                   resume_path: Optional[str] = None) -> MosaicGNN:
        """Train one fold.

        `resume_path` makes a long fit survive being killed. The full
        optimisation state is written there every few epochs and picked up
        again on the next call, so a run that is interrupted -- by a wall
        clock, a container being reclaimed, an OOM -- restarts from the last
        saved epoch instead of from scratch. At ~240 s/epoch a 120-epoch fit
        takes eight hours, which is longer than this machine stays alive
        unattended, so without this the fit can never finish at all.

        The checkpoint records a fingerprint of everything that defines the
        fit (gene axis, split, class count, epoch budget). A checkpoint that
        does not match is ignored rather than loaded, so a resume can never
        silently mix two different folds' weights.
        """
        cfg = self.cfg
        model = self.build_model(t, n_classes)

        counts = np.bincount(t.label.numpy()[train_idx][t.label.numpy()[train_idx] >= 0],
                             minlength=n_classes).astype(np.float32)
        weight = torch.from_numpy(
            (counts.sum() / np.maximum(counts, 1)) / n_classes).to(self.device)
        criterion = MosaicLoss(cfg.loss, weight).to(self.device)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                weight_decay=cfg.train.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and
                                           self.device.type == "cuda")

        best_score, best_state, stale = -np.inf, None, 0
        # Per-evaluation trace, read back by diagnostics.learning_curves. It is
        # recorded here rather than reconstructed afterwards because the only
        # place the train/validation split is seen epoch by epoch is this loop.
        history: List[Dict[str, float]] = []
        swa_state: Optional[Dict[str, torch.Tensor]] = None
        swa_n = 0
        swa_start = int(cfg.train.swa_start_frac * cfg.train.epochs)
        every = max(int(cfg.train.eval_every), 1)
        rng = np.random.default_rng(cfg.train.seed)
        bs = len(train_idx) if cfg.train.full_batch else cfg.train.batch_size

        fingerprint = _fit_fingerprint(t, train_idx, val_idx, n_classes,
                                       cfg.train.epochs, cfg.train.seed)
        start_epoch = 0
        # Checkpoint cadence is set by how long this container lives, not by a
        # round number. It is rebuilt roughly every forty-five minutes, and at
        # pan-cancer scale an epoch costs minutes, so a ten-epoch interval put
        # the first save at or beyond the container's lifetime -- which is how
        # three hours of training were lost once already, with nothing on disk
        # to resume from. Save every third epoch, and never less often than the
        # evaluation itself. MOSAIC_SAVE_EVERY overrides it.
        save_every = max(int(os.environ.get("MOSAIC_SAVE_EVERY", "3")), 1)
        if resume_path is not None and os.path.isfile(resume_path):
            try:
                ck = torch.load(resume_path, map_location=self.device,
                                weights_only=False)
            except Exception as err:
                print(f"  !! could not read {resume_path} ({err}); "
                      "starting this fit from scratch")
                ck = None
            if ck is not None and ck.get("fingerprint") != fingerprint:
                print("  !! checkpoint describes a different fit; "
                      "starting this fit from scratch")
            elif ck is not None:
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                scaler.load_state_dict(ck["scaler"])
                rng.bit_generator.state = ck["rng"]
                # Dropout in every layer, and the edge dropout in the gene
                # attention, draw from torch's global generator; MC-dropout at
                # validation does too. Restoring only the local NumPy
                # generator leaves those streams starting afresh, which makes
                # a resumed fit a measurably different fit -- the first
                # version of this code did exactly that and the resumed run
                # came out at c-index 0.742 against 0.723 for the
                # uninterrupted one.
                torch.set_rng_state(ck["torch_rng"].to(torch.uint8).cpu()
                                    if torch.is_tensor(ck["torch_rng"])
                                    else ck["torch_rng"])
                if ck.get("np_legacy") is not None:
                    np.random.set_state(ck["np_legacy"])
                if ck.get("cuda_rng") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(ck["cuda_rng"])
                best_score = ck["best_score"]
                best_state = ck["best_state"]
                stale = ck["stale"]
                history = ck["history"]
                swa_state = ck["swa_state"]
                swa_n = ck["swa_n"]
                start_epoch = ck["next_epoch"]
                print(f"  resumed from {resume_path} at epoch "
                      f"{start_epoch + 1}/{cfg.train.epochs} "
                      f"(best so far {best_score:.4f})")
                if start_epoch >= cfg.train.epochs:
                    print("  the saved fit is already complete")

        def _save(next_epoch: int) -> None:
            if resume_path is None:
                return
            payload = {
                "fingerprint": fingerprint,
                "next_epoch": next_epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "np_legacy": np.random.get_state(),
                "cuda_rng": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None),
                "best_score": best_score,
                "best_state": best_state,
                "stale": stale,
                "history": history,
                "swa_state": swa_state,
                "swa_n": swa_n,
            }
            tmp = resume_path + ".tmp"
            torch.save(payload, tmp)
            os.replace(tmp, resume_path)   # atomic: never a half-written file

        for epoch in range(start_epoch, cfg.train.epochs):
            model.train()
            perm = rng.permutation(train_idx)
            p = epoch / max(cfg.train.grl_warmup, 1)
            grl = float(2.0 / (1.0 + np.exp(-5 * min(p, 1.0))) - 1.0)
            epoch_terms: Dict[str, float] = {}

            for s in range(0, len(perm), bs):
                chunk = perm[s: s + bs]
                if len(chunk) < 4:
                    continue
                batch = self._batch(t, chunk)
                anchors = torch.from_numpy(
                    rng.choice(train_idx, size=min(256, len(train_idx)),
                               replace=False)).long().to(self.device)

                batch["x"] = self._augment(batch["x"], rng)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    out = model(batch["x"], batch["clinical"], t.edge_index,
                                t.edge_type, t.edge_weight,
                                bank_idx=batch["index"],
                                anchor_idx=anchors if cfg.model.use_patient_graph
                                else None,
                                grl_lambda=grl,
                                contrast=cfg.model.use_cross_omic_contrast)
                    batch["grl_weight"] = grl
                    terms = criterion(out, batch)
                scaler.scale(terms["total"]).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                for k, v in terms.items():
                    if torch.is_tensor(v):
                        epoch_terms[k] = epoch_terms.get(k, 0.0) + float(v.detach())
            sched.step()

            if cfg.train.swa and epoch >= swa_start:
                # Running mean of the weights over the tail of the schedule.
                sd = {k: v.detach().float().cpu()
                      for k, v in model.state_dict().items()}
                if swa_state is None:
                    swa_state, swa_n = sd, 1
                else:
                    swa_n += 1
                    for k, v in sd.items():
                        if swa_state[k].is_floating_point():
                            swa_state[k] += (v - swa_state[k]) / swa_n
                        else:
                            swa_state[k] = v

            if (epoch + 1) % every == 0 or epoch == cfg.train.epochs - 1:
                score, detail = self._validate(model, t, val_idx, train_idx)
                n_steps = max(sum(1 for _ in range(0, len(perm), bs)), 1)
                history.append({
                    "epoch": epoch + 1,
                    "train_loss": epoch_terms.get("total", float("nan")) / n_steps,
                    "val_score": score,
                    "val_balanced_accuracy": detail.get("balanced_accuracy",
                                                        float("nan")),
                    "val_c_index": detail.get("c_index", float("nan")),
                })
                # A fixed cadence of 20 epochs means a 20-epoch fit prints one
                # line, at the very end: for over an hour there is no way to
                # tell a live fit from a wedged one. Throttle in units of
                # evaluations, not epochs -- validation only runs every
                # `every` epochs, so an epoch-based cadence can miss every
                # eval and print nothing at all. Aim for at most ~12 lines.
                n_evals = max(cfg.train.epochs // every, 1)
                log_every = max(n_evals // 12, 1)
                if verbose and (len(history) % log_every == 0
                                or epoch == cfg.train.epochs - 1):
                    loss_str = " ".join(f"{k}={v:.3f}" for k, v in
                                        sorted(epoch_terms.items()) if k == "total")
                    print(f"  epoch {epoch + 1:3d} {loss_str} "
                          f"val_score={score:.4f} "
                          f"bacc={detail.get('balanced_accuracy', float('nan')):.3f} "
                          f"cidx={detail.get('c_index', float('nan')):.3f}")
                if score > best_score + 1e-4:
                    best_score, stale = score, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += every
                if (epoch + 1) % save_every == 0:
                    _save(epoch + 1)
                if stale >= cfg.train.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # The averaged weights are kept only if they actually beat the
        # early-stopping pick on the same validation split, so SWA can never
        # make the reported model worse.
        if swa_state is not None and swa_n > 1:
            keep = copy.deepcopy(model.state_dict())
            try:
                model.load_state_dict({k: v.to(self.device)
                                       for k, v in swa_state.items()})
                swa_score, _ = self._validate(model, t, val_idx, train_idx)
            except Exception:
                swa_score = -np.inf
            if swa_score > best_score + 1e-4:
                if verbose:
                    print(f"  SWA kept ({swa_score:.4f} > {best_score:.4f}, "
                          f"{swa_n} snapshots)")
            else:
                model.load_state_dict(keep)
        model._history = history
        # The fit is finished, so the resume state is no longer wanted: leaving
        # it behind would make the next fold think it had work to pick up.
        if resume_path is not None and os.path.isfile(resume_path):
            try:
                os.remove(resume_path)
            except OSError:
                pass
        return model

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, model: MosaicGNN, t: FoldTensors, idx: np.ndarray,
                anchor_idx: Optional[np.ndarray] = None,
                mc_samples: int = 0) -> Dict[str, np.ndarray]:
        model.eval()
        if mc_samples > 0:  # MC dropout for predictive uncertainty
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()

        anchors = (torch.from_numpy(anchor_idx).long().to(self.device)
                   if anchor_idx is not None and self.cfg.model.use_patient_graph
                   else None)
        bs = 64
        runs = max(mc_samples, 1)
        probs, risks, hazards = [], [], []
        for _ in range(runs):
            p_chunks, r_chunks, h_chunks = [], [], []
            for s in range(0, len(idx), bs):
                batch = self._batch(t, idx[s: s + bs])
                out = model(batch["x"], batch["clinical"], t.edge_index,
                            t.edge_type, t.edge_weight,
                            bank_idx=None, anchor_idx=anchors, grl_lambda=0.0)
                p_chunks.append(F.softmax(out.logits.float(), -1).cpu().numpy())
                r_chunks.append(out.risk.float().cpu().numpy())
                h_chunks.append(torch.sigmoid(out.hazard.float()).cpu().numpy())
            probs.append(np.concatenate(p_chunks))
            risks.append(np.concatenate(r_chunks))
            hazards.append(np.concatenate(h_chunks))

        prob = np.mean(probs, 0)
        risk = np.mean(risks, 0)
        hazard = np.mean(hazards, 0)
        return {
            "prob": prob,
            "risk": risk,
            "survival": np.cumprod(1.0 - hazard, axis=1),
            "prob_std": np.std(probs, 0).mean(1) if runs > 1 else np.zeros(len(idx)),
            "risk_std": np.std(risks, 0) if runs > 1 else np.zeros(len(idx)),
            "entropy": -(prob * np.log(prob + 1e-12)).sum(1),
        }

    def _validate(self, model: MosaicGNN, t: FoldTensors, val_idx: np.ndarray,
                  train_idx: np.ndarray) -> Tuple[float, Dict[str, float]]:
        pred = self.predict(model, t, val_idx, anchor_idx=train_idx)
        y = t.label.numpy()[val_idx]
        cm = classification_metrics(y, pred["prob"])
        ci = concordance_index(t.time.numpy()[val_idx], pred["risk"],
                               t.event.numpy()[val_idx])
        detail = dict(cm)
        detail["c_index"] = ci
        # Selection criterion weights both tasks; NaNs fall back to the other.
        parts = [v for v in (cm.get("balanced_accuracy"), ci) if v is not None
                 and np.isfinite(v)]
        return (float(np.mean(parts)) if parts else -np.inf), detail

    # ------------------------------------------------------------------ #
    def evaluate_fold(self, model: MosaicGNN, t: FoldTensors, test_idx: np.ndarray,
                      train_idx: np.ndarray) -> Dict[str, object]:
        pred = self.predict(model, t, test_idx, anchor_idx=train_idx,
                            mc_samples=self.cfg.train.mc_dropout_samples)
        y = t.label.numpy()[test_idx]
        tt = t.time.numpy()[test_idx]
        ee = t.event.numpy()[test_idx]

        res: Dict[str, object] = classification_metrics(y, pred["prob"])
        m = y >= 0
        if m.any():
            res["ece"] = expected_calibration_error(pred["prob"][m], y[m])

        ci, lo, hi = bootstrap_ci(
            lambda a, b, c: concordance_index(a, b, c), tt, pred["risk"], ee,
            n_boot=500, seed=self.cfg.train.seed)
        res["c_index"] = ci
        res["c_index_ci"] = [lo, hi]
        res.update(time_dependent_auc(tt, pred["risk"], ee))

        median = np.median(pred["risk"])
        group = (pred["risk"] > median).astype(int)
        chi2, p = logrank_test(tt, ee, group)
        res["logrank_chi2"] = chi2
        res["logrank_p"] = p
        res["mean_predictive_entropy"] = float(pred["entropy"].mean())

        # Per class, one row per label value. A pooled macro average hides the
        # case that actually matters for a four-class stage task: a model can
        # look respectable while never once predicting stage III.
        per_class: Dict[str, Dict[str, float]] = {}
        if m.any():
            n_cls = pred["prob"].shape[1]
            y_lab, p_lab = y[m], pred["prob"][m]
            yhat = p_lab.argmax(1)
            for c in range(n_cls):
                support = int((y_lab == c).sum())
                if support == 0:
                    continue
                tp = float(((y_lab == c) & (yhat == c)).sum())
                fp = float(((y_lab != c) & (yhat == c)).sum())
                fn = float(((y_lab == c) & (yhat != c)).sum())
                tn = float(((y_lab != c) & (yhat != c)).sum())
                prec = tp / (tp + fp) if tp + fp else float("nan")
                rec = tp / (tp + fn) if tp + fn else float("nan")
                f1 = (2 * prec * rec / (prec + rec)
                      if np.isfinite(prec) and np.isfinite(rec) and prec + rec
                      else float("nan"))
                per_class[str(c)] = {
                    "support": support,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "specificity": tn / (tn + fp) if tn + fp else float("nan"),
                    "mean_predicted_prob": float(p_lab[:, c].mean()),
                }
        res["per_class"] = per_class

        per_cohort = {}
        coh = t.cohort.numpy()[test_idx]
        for ci_idx, name in enumerate(self.skeleton.cohort_names):
            sel = coh == ci_idx
            if sel.sum() < 10:
                continue
            sub = classification_metrics(y[sel], pred["prob"][sel])
            sub["c_index"] = concordance_index(tt[sel], pred["risk"][sel], ee[sel])
            sub["n"] = int(sel.sum())
            per_cohort[name] = sub
        res["per_cohort"] = per_cohort
        res["_pred"] = {k: v.tolist() for k, v in pred.items()
                        if k in ("risk", "entropy")}
        return res

    # ------------------------------------------------------------------ #
    def run_cv(self) -> Dict[str, object]:
        cfg = self.cfg
        assert self.skeleton is not None, "call prepare() first"
        need_lab = cfg.data.class_target != "none"
        keep = valid_mask(self.skeleton, need_label=False, need_surv=False)
        eligible = np.where(keep)[0]
        strat = stratification_key(self.skeleton)[eligible]

        fold_results: List[Dict[str, object]] = []
        for rep in range(cfg.train.n_repeats):
            skf = StratifiedKFold(n_splits=cfg.train.outer_folds, shuffle=True,
                                  random_state=cfg.train.seed + rep)
            for fold, (tr, te) in enumerate(skf.split(eligible, strat)):
                torch.manual_seed(cfg.train.seed + 100 * rep + fold)
                np.random.seed(cfg.train.seed + 100 * rep + fold)
                train_idx, test_idx = eligible[tr], eligible[te]

                inner = StratifiedKFold(n_splits=2, shuffle=True,
                                        random_state=cfg.train.seed + rep)
                sub_tr, sub_val = next(inner.split(train_idx, strat[tr]))
                fit_idx, val_idx = train_idx[sub_tr], train_idx[sub_val]

                t0 = _time.time()
                print(f"\n=== repeat {rep + 1}/{cfg.train.n_repeats} "
                      f"fold {fold + 1}/{cfg.train.outer_folds} "
                      f"(train={len(fit_idx)} val={len(val_idx)} test={len(test_idx)})")
                tensors, _, _ = self.build_fold(train_idx)
                n_classes = max(int(self.skeleton.label.max()) + 1, 2)
                model = self.train_fold(
                    tensors, fit_idx, val_idx, n_classes,
                    resume_path=os.path.join(
                        cfg.train.out_dir,
                        f"{cfg.tag}_resume_r{rep}f{fold}.pt"))
                res = self.evaluate_fold(model, tensors, test_idx, train_idx)
                res["repeat"], res["fold"] = rep, fold
                res["seconds"] = round(_time.time() - t0, 1)
                fold_results.append(res)
                print(f"[fold] bacc={res.get('balanced_accuracy', float('nan')):.3f} "
                      f"auroc={res.get('auroc', float('nan')):.3f} "
                      f"c_index={res['c_index']:.3f} "
                      f"logrank_p={res['logrank_p']:.2e} ({res['seconds']}s)")

                torch.save({"state_dict": model.state_dict(),
                            "genes": tensors.graph.genes,
                            "modules": tensors.graph.modules.tolist()},
                           os.path.join(cfg.train.out_dir,
                                        f"{cfg.tag}_r{rep}_f{fold}.pt"))

                # Write the summary after every fold, not only at the end. A
                # fold here can cost hours, so a run that is interrupted --
                # or that runs out of wall clock -- must still leave the folds
                # it did finish in a readable form, with `folds_completed` and
                # `complete` saying exactly how much of the schedule ran.
                self._write_results(fold_results, rep, fold)

        summary = self._write_results(fold_results)
        return {"folds": fold_results, "summary": summary}

    def _write_results(self, fold_results: List[Dict[str, object]],
                       rep: Optional[int] = None, fold: Optional[int] = None
                       ) -> Dict[str, Dict[str, float]]:
        """Serialise the folds finished so far; return their summary."""
        cfg = self.cfg
        summary = self._summarise(fold_results)
        planned = cfg.train.n_repeats * cfg.train.outer_folds
        done = len(fold_results)
        payload = {
            "config": cfg.to_dict(),
            "folds": fold_results,
            "summary": summary,
            "folds_completed": done,
            "folds_planned": planned,
            "complete": done >= planned,
        }
        path = os.path.join(cfg.train.out_dir, f"{cfg.tag}_results.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2, default=float)
        os.replace(tmp, path)       # atomic: never leave a half-written file
        if rep is not None:
            print(f"[checkpoint] wrote {done}/{planned} folds to {path}")
        return summary

    @staticmethod
    def _summarise(folds: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
        keys = [k for k in folds[0] if isinstance(folds[0][k], (int, float))
                and not k.startswith("_")]
        return {k: summarise([float(f[k]) for f in folds if k in f]) for k in keys}
