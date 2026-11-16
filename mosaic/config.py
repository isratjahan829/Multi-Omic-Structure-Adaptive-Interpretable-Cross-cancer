"""Configuration objects for MOSAIC-GNN.

Everything that changes between experiments lives here so that a run is fully
described by a single serialisable object (dumped next to every checkpoint).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    root: str = "/kaggle/input/datasets/jannatuljerin/tcga-capstone"
    cohorts: List[str] = field(
        default_factory=lambda: ["BLCA", "LIHC", "SKCM", "THCA"]
    )
    file_template: str = "{cohort}_merged_full.csv"

    # Omic blocks are detected by column prefix. Order defines node-feature
    # channel order; blocks missing from a cohort are zero-filled and flagged
    # through their availability mask channel.
    omic_prefixes: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "mut": ["MUT_"],
            "cna": ["CNA_"],
            "exp": ["RNA_", "EXP_", "MRNA_", "GEXP_"],
            "met": ["METH_", "MET_"],
        }
    )

    # Number of genes retained per omic block (train-fold statistics only).
    n_genes_per_block: int = 900
    # Hard cap on the size of the shared gene graph.
    max_genes: int = 2000
    # Minimum mutation frequency for a gene to be eligible (kills singletons).
    min_mut_freq: float = 0.02
    # Minimum variance quantile for continuous blocks.
    var_quantile: float = 0.50

    clinical_numeric: List[str] = field(
        default_factory=lambda: [
            "AGE",
            "ANEUPLOIDY_SCORE",
            "TMB_NONSYNONYMOUS",
            "MSI_SCORE_MANTIS",
            "MSI_SENSOR_SCORE",
            "TBL_SCORE",
            "WEIGHT",
        ]
    )
    clinical_categorical: List[str] = field(
        default_factory=lambda: [
            "SEX",
            "RACE",
            "GRADE",
            "AJCC_PATHOLOGIC_TUMOR_STAGE",
            "PATH_T_STAGE",
            "PATH_N_STAGE",
            "PATH_M_STAGE",
            "HISTORY_NEOADJUVANT_TRTYN",
            "RADIATION_THERAPY",
        ]
    )

    id_col: str = "PATIENT_ID"
    sample_col: str = "SAMPLE_ID"
    cohort_col: str = "CANCER_TYPE_ACRONYM"

    # Survival endpoint. Any of OS / DSS / PFS / DFS.
    survival_endpoint: str = "OS"
    # Classification target. "cohort" | "stage" | "grade" | "subtype"
    class_target: str = "stage"

    # Optional external priors. When absent, the gene graph is estimated from
    # training-fold co-alteration statistics only (no leakage, no internet).
    ppi_edgelist: Optional[str] = None  # csv with columns: gene_a,gene_b[,weight]
    pathway_gmt: Optional[str] = None  # MSigDB-style .gmt

    cache_dir: str = "./cache"


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #
@dataclass
class GraphConfig:
    # ---- gene-level (intra-patient) graph -------------------------------- #
    # "prior"  : only the supplied PPI edge list
    # "data"   : co-alteration graph estimated on the training fold
    # "hybrid" : union of both, with a relation id per source
    gene_graph_mode: str = "hybrid"
    gene_knn: int = 12            # neighbours per gene in the data-driven graph
    gene_corr_floor: float = 0.15  # |rho| below this is never an edge
    n_relations: int = 3           # prior, data-driven, self-loop

    # ---- functional modules (pathway pooling) ---------------------------- #
    n_modules: int = 64            # used when no .gmt is supplied
    module_min_size: int = 5

    # ---- patient-level graph --------------------------------------------- #
    patient_knn: int = 10
    graph_learner_heads: int = 4
    graph_learner_temp: float = 0.2
    # Anchors are training patients only; validation/test patients attach to
    # them but never to each other -> strictly inductive, no transduction leak.
    inductive_anchors: bool = True


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    gene_hidden: int = 64
    gene_layers: int = 3
    gene_heads: int = 4
    module_hidden: int = 128
    patient_hidden: int = 128
    patient_layers: int = 2
    clinical_hidden: int = 32

    # Disentangled latent: z = [z_shared ; z_specific]
    z_shared: int = 96
    z_specific: int = 32

    dropout: float = 0.4
    edge_dropout: float = 0.2
    n_survival_bins: int = 12       # discrete-time hazard head

    # ---- head routing ---------------------------------------------------- #
    # Which slice of the patient latent each head reads.
    #   "shared"   : z_shared (cohort-invariant, what transfers)
    #   "specific" : z_specific (cohort-identifying)
    #   "full"     : the whole latent
    #   "auto"     : "specific" for tissue-of-origin style targets, else
    #                "shared" -- resolved by the pipeline, never by the model.
    # Tissue of origin is *defined* by the cohort, so routing it through the
    # adversarially scrubbed shared latent fights the loss that scrubs it.
    class_head_source: str = "auto"
    survival_head_source: str = "shared"

    # ---- residual clinical term ------------------------------------------ #
    # A zero-initialised linear map from the clinical covariates straight onto
    # every head's output, bypassing the graph bottleneck. This makes the model
    # a strict generalisation of a linear clinical model: it can recover
    # clinical-only Cox exactly by zeroing the deep branch, so the deep part
    # only has to add to clinical signal rather than re-derive it through a
    # 128-d fusion layer and two rounds of patient-graph smoothing.
    clinical_residual: bool = True

    # Ablation switches (each maps to a row of the ablation table).
    use_gene_graph: bool = True
    use_module_pool: bool = True
    use_patient_graph: bool = True
    use_disentangle: bool = True
    use_cross_omic_contrast: bool = True
    use_clinical: bool = True


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
@dataclass
class LossConfig:
    w_class: float = 1.0
    w_cox: float = 1.0
    w_hazard: float = 0.5
    w_adversarial: float = 0.3     # gradient-reversal on z_shared
    w_cohort: float = 0.3          # cohort head on z_specific
    w_orthogonal: float = 0.1
    w_contrast: float = 0.2        # cross-omic InfoNCE
    w_graph_reg: float = 0.05      # smoothness + sparsity on learned adjacency
    contrast_temp: float = 0.1
    label_smoothing: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-3

    # ---- regularisation / augmentation ----------------------------------- #
    # Whole genes are dropped (value channel and availability mask together),
    # which is the augmentation that matches how omic data actually goes
    # missing; per-element dropout would let the model infer the dropped value
    # from its own mask channel.
    gene_dropout: float = 0.15
    input_noise: float = 0.05      # gaussian jitter on continuous channels
    swa: bool = True               # stochastic weight averaging over the tail
    swa_start_frac: float = 0.6    # start averaging after this fraction of run
    eval_every: int = 5            # epochs between validation checks
    batch_size: int = 64           # 0 -> full batch (needed for patient graph)
    full_batch: bool = True
    patience: int = 40
    grl_warmup: int = 30           # epochs before adversary reaches full weight
    seed: int = 42
    device: str = "cuda"
    amp: bool = True

    outer_folds: int = 5
    inner_folds: int = 3
    n_repeats: int = 3             # repeated CV -> variance estimates
    mc_dropout_samples: int = 30   # predictive uncertainty at test time

    out_dir: str = "./runs"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tag: str = "mosaic-gnn"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as fh:
            raw = json.load(fh)
        return Config(
            data=DataConfig(**raw["data"]),
            graph=GraphConfig(**raw["graph"]),
            model=ModelConfig(**raw["model"]),
            loss=LossConfig(**raw["loss"]),
            train=TrainConfig(**raw["train"]),
            tag=raw.get("tag", "mosaic-gnn"),
        )
