"""Generate small synthetic cohorts with the same schema as TCGA-CAPSTONE.

Used to smoke-test the pipeline without the real data (CI, laptops, reviewers).
Signal is planted deliberately: a handful of "driver" genes drive both the stage
label and the survival hazard, so a working model must recover them.

    python scripts/make_synthetic.py --out ./synthetic --patients 160 --genes 300
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

COHORTS = ["BLCA", "LIHC", "SKCM", "THCA"]


def make_cohort(name: str, n: int, n_genes: int, rng: np.random.Generator,
                n_drivers: int = 12) -> pd.DataFrame:
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    drivers = rng.choice(n_genes, n_drivers, replace=False)

    mut = (rng.random((n, n_genes)) < rng.uniform(0.02, 0.15, n_genes)).astype(int)
    cna = rng.choice([-2, -1, 0, 1, 2], size=(n, n_genes),
                     p=[0.05, 0.15, 0.6, 0.15, 0.05])
    expr = rng.normal(0, 1, (n, n_genes))

    # Latent burden on the driver set drives label and hazard.
    burden = (mut[:, drivers].sum(1) / n_drivers
              + 0.5 * np.abs(cna[:, drivers]).mean(1)
              + 0.5 * expr[:, drivers].mean(1))
    burden = (burden - burden.mean()) / (burden.std() + 1e-8)
    expr[:, drivers] += 1.5 * burden[:, None]

    stage_idx = np.clip(np.digitize(burden + rng.normal(0, 0.5, n),
                                    [-0.7, 0.0, 0.7]), 0, 3)
    stage = np.asarray(["STAGE I", "STAGE II", "STAGE III", "STAGE IV"])[stage_idx]

    hazard = np.exp(0.9 * burden)
    time = rng.exponential(60.0 / hazard)
    censor = rng.exponential(90.0, n)
    obs = np.minimum(time, censor)
    event = (time <= censor).astype(int)

    df = pd.DataFrame({
        "PATIENT_ID": [f"TCGA-{name}-{i:04d}" for i in range(n)],
        "SAMPLE_ID": [f"TCGA-{name}-{i:04d}-01" for i in range(n)],
        "CANCER_TYPE_ACRONYM": name,
        "SUBTYPE": name,
        "SAMPLE_TYPE": "Primary",
        "AGE": rng.integers(35, 85, n),
        "SEX": rng.choice(["Male", "Female"], n),
        "RACE": rng.choice(["White", "Asian", "Black Or African American"], n),
        "GRADE": rng.choice(["Low Grade", "High Grade"], n),
        "ANEUPLOIDY_SCORE": rng.integers(0, 30, n),
        "TMB_NONSYNONYMOUS": rng.gamma(2, 3, n).round(3),
        "MSI_SCORE_MANTIS": rng.random(n).round(4),
        "MSI_SENSOR_SCORE": rng.random(n).round(4),
        "TBL_SCORE": rng.integers(0, 400, n),
        "WEIGHT": rng.normal(75, 15, n).round(1),
        "AJCC_PATHOLOGIC_TUMOR_STAGE": stage,
        "PATH_T_STAGE": rng.choice(["T1", "T2", "T3", "T4"], n),
        "PATH_N_STAGE": rng.choice(["N0", "N1", "N2"], n),
        "PATH_M_STAGE": rng.choice(["M0", "M1"], n),
        "HISTORY_NEOADJUVANT_TRTYN": rng.choice(["Yes", "No"], n),
        "RADIATION_THERAPY": rng.choice(["Yes", "No"], n),
        "OS_MONTHS": obs.round(4),
        "OS_STATUS": np.where(event == 1, "1:DECEASED", "0:LIVING"),
        "PFS_MONTHS": (obs * rng.uniform(0.5, 1.0, n)).round(4),
        "PFS_STATUS": np.where(event == 1, "1:PROGRESSION", "0:CENSORED"),
    })

    for prefix, block in (("MUT_", mut), ("CNA_", cna), ("RNA_", expr.round(4))):
        df = pd.concat([df, pd.DataFrame(
            block, columns=[prefix + g for g in genes])], axis=1)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./synthetic")
    ap.add_argument("--patients", type=int, default=160)
    ap.add_argument("--genes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for cohort in COHORTS:
        df = make_cohort(cohort, args.patients, args.genes, rng)
        path = os.path.join(args.out, f"{cohort}_merged_full.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path}  shape={df.shape}")


if __name__ == "__main__":
    main()
