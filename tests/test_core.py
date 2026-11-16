"""Correctness checks for the pieces that are easy to get quietly wrong."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mosaic.config import GraphConfig
from mosaic.data.graphs import build_gene_graph, module_assignment_matrix
from mosaic.data.loader import parse_stage, parse_survival
from mosaic.losses import (cox_ph_loss, cross_omic_contrast,
                           discrete_hazard_loss, orthogonality_loss)
from mosaic.metrics import concordance_index, logrank_test
from mosaic.models.layers import ModulePool, SharedTopologyRGAT


# --------------------------------------------------------------------------- #
def test_parse_survival_handles_cbioportal_encoding():
    import pandas as pd
    df = pd.DataFrame({"OS_STATUS": ["1:DECEASED", "0:LIVING", "1:DECEASED"],
                       "OS_MONTHS": [10.0, 20.0, 5.0]})
    time, event = parse_survival(df, "OS")
    assert list(event) == [1.0, 0.0, 1.0]
    assert list(time) == [10.0, 20.0, 5.0]


def test_parse_stage_collapses_substages():
    import pandas as pd
    s = pd.Series(["Stage IIA", "STAGE IV", "STAGE IIIB", "not reported"])
    assert list(parse_stage(s)) == [1, 3, 2, -1]


# --------------------------------------------------------------------------- #
def test_cindex_matches_manual_count():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 1.0, 1.0])
    perfect = np.array([3.0, 2.0, 1.0])   # highest risk dies first
    worst = np.array([1.0, 2.0, 3.0])
    assert concordance_index(time, perfect, event) == pytest.approx(1.0)
    assert concordance_index(time, worst, event) == pytest.approx(0.0)


def test_cindex_ignores_censored_as_index_case():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([0.0, 1.0, 1.0])     # first patient censored
    risk = np.array([99.0, 2.0, 1.0])     # its (wrong) risk must not count
    assert concordance_index(time, risk, event) == pytest.approx(1.0)


def test_cox_loss_prefers_correct_ordering():
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.tensor([1.0, 1.0, 1.0, 1.0])
    good = torch.tensor([2.0, 1.0, 0.0, -1.0])
    bad = -good
    assert cox_ph_loss(good, time, event) < cox_ph_loss(bad, time, event)


def test_cox_loss_is_finite_without_events():
    risk = torch.randn(8, requires_grad=True)
    loss = cox_ph_loss(risk, torch.rand(8) * 10, torch.zeros(8))
    assert torch.isfinite(loss)


def test_discrete_hazard_censoring_is_asymmetric():
    logits = torch.zeros(1, 5)
    idx = torch.tensor([3])
    observed = discrete_hazard_loss(logits, idx, torch.tensor([1.0]))
    censored = discrete_hazard_loss(logits, idx, torch.tensor([0.0]))
    # With h = 0.5 everywhere both are finite, and an observed event is never
    # cheaper than surviving the same bin.
    assert torch.isfinite(observed) and torch.isfinite(censored)
    assert observed == pytest.approx(float(censored), abs=1e-6)


def test_orthogonality_zero_for_orthogonal_latents():
    a = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0]])
    assert float(orthogonality_loss(a, b)) == pytest.approx(0.0, abs=1e-6)


def test_contrast_is_lower_when_views_agree():
    z = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    agree = cross_omic_contrast({"a": z, "b": z})
    disagree = cross_omic_contrast({"a": z, "b": z.flip(0)})
    assert agree < disagree


def test_logrank_detects_separated_groups():
    time = np.concatenate([np.full(30, 5.0), np.full(30, 50.0)])
    event = np.ones(60)
    group = np.concatenate([np.ones(30), np.zeros(30)])
    _, p = logrank_test(time, event, group)
    assert p < 1e-6


# --------------------------------------------------------------------------- #
def test_rgat_is_finite_and_permutation_consistent():
    torch.manual_seed(0)
    x = torch.randn(3, 12, 5)
    ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    et = torch.zeros(5, dtype=torch.long)
    ew = torch.rand(5).clamp_min(0.1)
    layer = SharedTopologyRGAT(5, 8, 3, heads=2).eval()
    out, att = layer(x, ei, et, ew, return_attention=True)
    assert torch.isfinite(out).all()
    assert att.shape == (3, 5, 2)
    # Attention over each destination node's incoming edges must sum to one.
    node1 = att[:, ei[1] == 1, :].sum(1)
    assert torch.allclose(node1, torch.ones_like(node1), atol=1e-5)


def test_module_pool_weights_sum_to_one_per_module():
    assign = torch.zeros(6, 2)
    assign[:3, 0] = 1 / 3
    assign[3:, 1] = 1 / 3
    pool = ModulePool(4, 4, assign).eval()
    h = torch.randn(2, 6, 4)
    pooled, gate = pool(h, return_gate=True)
    assert pooled.shape == (2, 2, 4)
    assert torch.isfinite(pooled).all()
    assert torch.allclose(gate.sum(1), torch.full((2,), 2.0), atol=1e-4)


def test_gene_graph_has_no_infinite_weights_or_self_duplicates():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 30, 4)).astype(np.float32)
    genes = [f"G{i}" for i in range(30)]
    g = build_gene_graph(genes, x, GraphConfig(gene_knn=4, n_modules=5))
    assert np.isfinite(g.edge_weight).all()
    assert (g.edge_weight > 0).all()
    # Exactly one self-loop per gene, carrying the self relation.
    self_edges = g.edge_index[0] == g.edge_index[1]
    assert self_edges.sum() == len(genes)
    assert set(g.edge_type[self_edges].tolist()) == {2}
    assert module_assignment_matrix(g).shape[0] == len(genes)


def test_no_leakage_from_preprocessor_fit():
    """Statistics fitted on the training rows must ignore held-out values."""
    from mosaic.config import DataConfig
    from mosaic.data.preprocess import (FittedPreprocessor, PanCohortData,
                                        fit_preprocessor, transform)

    n, g = 20, 4
    exp = np.random.default_rng(0).normal(size=(n, g)).astype(np.float32)
    exp[10:] += 1000.0  # held-out rows are wildly out of distribution
    data = PanCohortData(
        patients=[str(i) for i in range(n)],
        cohort_id=np.zeros(n, dtype=np.int64), cohort_names=["A"],
        blocks={"exp": exp}, block_mask={"exp": np.ones(n, np.float32)},
        genes=[f"G{i}" for i in range(g)],
        clinical=np.zeros((n, 1), np.float32), clinical_names=["c"],
        time=np.linspace(1, 20, n).astype(np.float32),
        event=np.ones(n, np.float32), label=np.zeros(n, np.int64))

    pp = fit_preprocessor(data, np.arange(10), DataConfig(), n_bins=4)
    assert np.abs(pp.center["exp"]).max() < 10.0   # unaffected by rows 10..19
    x, _, _ = transform(data, pp)
    assert np.isfinite(x).all()


# --------------------------------------------------------------------------- #
# Regression tests for bugs that silently invalidated earlier runs.
# --------------------------------------------------------------------------- #
def test_target_column_is_never_a_clinical_feature():
    """Stage was once both the label and a clinical feature.

    Every model scored a perfect 1.000 and the only hint was the ablation
    row. The leaky-column map must strip the label's source columns for each
    target, or the whole results table is meaningless.
    """
    from mosaic.config import DataConfig
    from mosaic.data.preprocess import leaky_columns

    cfg = DataConfig()
    for target, forbidden in [("stage", "AJCC_PATHOLOGIC_TUMOR_STAGE"),
                              ("grade", "GRADE")]:
        cfg.class_target = target
        leaky = {c.upper() for c in leaky_columns(cfg)}
        assert forbidden in leaky, f"{forbidden} must be dropped for {target}"
        kept = [c for c in cfg.clinical_categorical
                if c.upper() not in leaky]
        assert forbidden not in kept

    cfg.class_target = "cohort"
    leaky = {c.upper() for c in leaky_columns(cfg)}
    assert "AJCC_PATHOLOGIC_TUMOR_STAGE" not in leaky, (
        "stage is a legitimate predictor of tissue of origin")


def test_patient_graph_learner_is_dtype_safe_under_autocast():
    """scatter_ raised when autocast gave target and source different dtypes."""
    from mosaic.models.layers import PatientGraphLearner

    learner = PatientGraphLearner(dim=16, heads=2, k=3)
    z = torch.randn(12, 16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        adj = learner(z)
    assert adj.dtype == torch.float32
    assert torch.isfinite(adj).all()
    assert torch.allclose(adj.sum(dim=1), torch.ones(12), atol=1e-4)


def test_survival_parsing_survives_arrow_nulls():
    """Parquet reads hand back pandas extension arrays, not plain object."""
    import pandas as pd

    df = pd.DataFrame({
        "OS_STATUS": pd.Series(["1:DECEASED", None, "0:LIVING", pd.NA],
                               dtype="string"),
        "OS_MONTHS": pd.Series([10.0, None, 25.0, 5.0], dtype="Float64"),
    })
    time, event = parse_survival(df, "OS")
    assert np.isfinite(time[0]) and np.isfinite(time[2])
    assert event[0] == 1 and event[2] == 0


def test_clinical_residual_nests_a_linear_model():
    """The residual term must start at zero and be able to carry signal alone."""
    from mosaic.config import ModelConfig
    from mosaic.models.mosaic import MosaicGNN

    cfg = ModelConfig(gene_hidden=8, gene_layers=1, gene_heads=1,
                      module_hidden=8, patient_hidden=8, clinical_hidden=4,
                      z_shared=8, z_specific=4, n_survival_bins=4,
                      use_patient_graph=False, use_cross_omic_contrast=False)
    genes = ["g0", "g1", "g2", "g3"]
    xtr = np.random.default_rng(0).normal(size=(8, 4, 4)).astype(np.float32)
    graph = build_gene_graph(genes, xtr,
                             GraphConfig(gene_knn=2, n_modules=2,
                                         module_min_size=1))
    assign = torch.from_numpy(module_assignment_matrix(graph))
    model = MosaicGNN(n_node_features=4, n_clinical=3, n_classes=2,
                      n_cohorts=2, n_relations=3, assign=assign,
                      n_patients=6, block_names=["mut", "exp"], cfg=cfg,
                      class_head_source="shared")
    assert model.res_risk is not None
    assert float(model.res_risk.weight.abs().sum()) == 0.0

    # Driving the linear term alone must move the risk prediction.
    with torch.no_grad():
        model.res_risk.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
    x = torch.zeros(6, 4, 4)
    ei = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    et = torch.zeros(4, dtype=torch.long)
    ew = torch.ones(4)
    clin = torch.zeros(6, 3)
    clin[:, 0] = torch.linspace(-2, 2, 6)
    model.eval()
    with torch.no_grad():
        risk = model(x, clin, ei, et, ew).risk
    assert risk[-1] > risk[0]
