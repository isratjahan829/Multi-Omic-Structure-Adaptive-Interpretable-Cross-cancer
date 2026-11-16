"""Build a pan-cancer multi-omic table from the cBioPortal public API.

Why this exists alongside scripts/merge_cbioportal.py
-----------------------------------------------------
merge_cbioportal.py consumes study *file exports* that were downloaded by
hand, and it selects the highest-variance genes separately inside each
cohort. That has two consequences a reviewer will press on. The feature
space then differs from cohort to cohort, which undercuts both the
pan-cancer claim and leave-one-cancer-out transfer (the held-out cohort's
features are not the ones the model was trained on). And "the 3000
highest-variance genes" is a weak answer to "why these genes?".

This script instead pulls data for a fixed, curated, clinically deployed
gene panel -- MSK-IMPACT 468 by default -- straight from the API, for as
many studies as asked for. Every cohort then has an identical feature
space, chosen before any data was seen, for a reason that can be cited.

Three omic blocks are written, matching the existing schema so the loader
needs no change:

    MUT_<gene>   1 if the gene carries a non-silent mutation, else 0
    CNA_<gene>   GISTIC discrete call, -2 .. +2
    RNA_<gene>   RNA-seq v2 RSEM, continuous

Output is one parquet per study, named <ACRONYM>_merged_full.parquet, with
PATIENT_ID and CANCER_TYPE_ACRONYM first, then the clinical attributes the
study publishes, then the three blocks.

    python scripts/fetch_cbioportal_api.py --out ./data_pan --panel IMPACT468
    python scripts/fetch_cbioportal_api.py --out ./data_pan --studies blca,lihc

Data fetched from a public API is untrusted input: nothing here is executed,
every value is coerced to a number or a plain string, and gene symbols are
used only after being matched against the panel the API itself returned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

API = "https://www.cbioportal.org/api"
SILENT = {"Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "IGR",
          "RNA", "Splice_Region"}


def _request(path: str, body=None, retries: int = 4, timeout: int = 300):
    """One API call, with backoff. Returns parsed JSON."""
    url = API + path
    last = None
    for attempt in range(retries):
        try:
            if body is None:
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
            else:
                req = urllib.request.Request(
                    url, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json",
                             "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError) as err:
            last = err
            if attempt < retries - 1:
                wait = 3 * (attempt + 1)
                print(f"    retry {attempt + 1}/{retries - 1} after {wait}s ({err})",
                      flush=True)
                time.sleep(wait)
    raise RuntimeError(f"{path} failed after {retries} attempts: {last}")


def panel_genes(panel_id: str) -> List[dict]:
    p = _request(f"/gene-panels/{panel_id}")
    genes = p.get("genes", [])
    out, seen = [], set()
    for g in genes:
        eid, sym = g.get("entrezGeneId"), g.get("hugoGeneSymbol")
        # A panel can list the same gene twice, and a symbol with a slash or a
        # space would make an unusable column name.
        if eid and sym and sym not in seen and sym.replace("-", "").isalnum():
            seen.add(sym)
            out.append({"entrezGeneId": int(eid), "hugoGeneSymbol": str(sym)})
    return out


def list_pancan_studies() -> List[dict]:
    studies = _request("/studies?projection=SUMMARY")
    return sorted((s for s in studies
                   if str(s.get("studyId", "")).endswith("_tcga_pan_can_atlas_2018")),
                  key=lambda s: -int(s.get("allSampleCount") or 0))


def profile_ids(study: str) -> Dict[str, Optional[str]]:
    profs = _request(f"/studies/{study}/molecular-profiles")
    want = {"mut": None, "cna": None, "rna": None}
    for p in profs:
        pid, kind, dtype = (p.get("molecularProfileId"),
                            p.get("molecularAlterationType"),
                            p.get("datatype"))
        if kind == "MUTATION_EXTENDED" and want["mut"] is None:
            want["mut"] = pid
        elif kind == "COPY_NUMBER_ALTERATION" and dtype == "DISCRETE":
            want["cna"] = pid
        elif (kind == "MRNA_EXPRESSION" and dtype == "CONTINUOUS"
              and pid and pid.endswith("rna_seq_v2_mrna")):
            want["rna"] = pid
    return want


def clinical_frame(study: str) -> pd.DataFrame:
    """One row per patient, columns = the study's own clinical attributes."""
    rows = _request(f"/studies/{study}/clinical-data"
                    f"?clinicalDataType=PATIENT&pageSize=10000000")
    if not rows:
        return pd.DataFrame()
    recs: Dict[str, Dict[str, object]] = {}
    for r in rows:
        pid = str(r.get("patientId", ""))
        attr = str(r.get("clinicalAttributeId", ""))
        if not pid or not attr:
            continue
        recs.setdefault(pid, {})[attr] = r.get("value")
    df = pd.DataFrame.from_dict(recs, orient="index")
    df.index.name = "PATIENT_ID"
    return df.reset_index()


def sample_to_patient(study: str) -> Dict[str, str]:
    rows = _request(f"/studies/{study}/samples?pageSize=10000000")
    return {str(s["sampleId"]): str(s["patientId"]) for s in rows
            if s.get("sampleId") and s.get("patientId")}


def _block(values, s2p: Dict[str, str], genes_by_eid: Dict[int, str],
           prefix: str, reduce_max: bool) -> pd.DataFrame:
    """Pivot API molecular records to one row per patient, one column per gene."""
    cells: Dict[str, Dict[str, float]] = {}
    for v in values:
        sid = str(v.get("sampleId", ""))
        pid = s2p.get(sid)
        eid = v.get("entrezGeneId")
        if pid is None or eid not in genes_by_eid:
            continue
        val = v.get("value")
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(num):
            continue
        col = prefix + genes_by_eid[eid]
        row = cells.setdefault(pid, {})
        # A patient can have more than one sample; keep the stronger signal
        # rather than letting dict order decide.
        if col in row and reduce_max:
            row[col] = max(row[col], num)
        elif col in row:
            row[col] = row[col] if abs(row[col]) >= abs(num) else num
        else:
            row[col] = num
    df = pd.DataFrame.from_dict(cells, orient="index")
    df.index.name = "PATIENT_ID"
    return df


def fetch_study(study: str, acronym: str, genes: List[dict],
                out_dir: str) -> Optional[str]:
    t0 = time.time()
    eids = [g["entrezGeneId"] for g in genes]
    by_eid = {g["entrezGeneId"]: g["hugoGeneSymbol"] for g in genes}
    syms = [g["hugoGeneSymbol"] for g in genes]
    prof = profile_ids(study)
    s2p = sample_to_patient(study)
    sample_list = f"{study}_all"

    clin = clinical_frame(study)
    if clin.empty:
        print(f"  !! {acronym}: no clinical data; skipped")
        return None

    # ---- mutations: gene-level non-silent indicator ---------------------- #
    mut = pd.DataFrame()
    if prof["mut"]:
        recs = _request(f"/molecular-profiles/{prof['mut']}/mutations/fetch"
                        f"?projection=DETAILED",
                        {"entrezGeneIds": eids, "sampleListId": sample_list})
        hit: Dict[str, Dict[str, float]] = {}
        for m in recs:
            if str(m.get("mutationType", "")) in SILENT:
                continue
            pid = s2p.get(str(m.get("sampleId", "")))
            eid = m.get("entrezGeneId")
            if pid is None or eid not in by_eid:
                continue
            hit.setdefault(pid, {})["MUT_" + by_eid[eid]] = 1.0
        mut = pd.DataFrame.from_dict(hit, orient="index")
        mut.index.name = "PATIENT_ID"
        # A patient present in the mutation profile with no listed hit is a
        # true zero; a patient absent from the profile is unknown. Only the
        # former is filled in, below, via the profiled-sample list.
        profiled = {s2p[s] for s in s2p if s in s2p}
        cols = ["MUT_" + s for s in syms]
        mut = mut.reindex(index=sorted(profiled), columns=cols).fillna(0.0)

    # ---- copy number and expression -------------------------------------- #
    cna = pd.DataFrame()
    if prof["cna"]:
        vals = _request(f"/molecular-profiles/{prof['cna']}/molecular-data/fetch"
                        f"?projection=SUMMARY",
                        {"entrezGeneIds": eids, "sampleListId": sample_list})
        cna = _block(vals, s2p, by_eid, "CNA_", reduce_max=False)

    rna = pd.DataFrame()
    if prof["rna"]:
        vals = _request(f"/molecular-profiles/{prof['rna']}/molecular-data/fetch"
                        f"?projection=SUMMARY",
                        {"entrezGeneIds": eids, "sampleListId": sample_list})
        rna = _block(vals, s2p, by_eid, "RNA_", reduce_max=True)

    # ---- join ------------------------------------------------------------- #
    out = clin.set_index("PATIENT_ID")
    for blk in (mut, cna, rna):
        if not blk.empty:
            out = out.join(blk, how="left")

    # Keep only patients with at least one omic measurement; a clinical-only
    # row contributes nothing to a multi-omics model and would be imputed
    # from end to end.
    omic_cols = [c for c in out.columns
                 if c.startswith(("MUT_", "CNA_", "RNA_"))]
    if not omic_cols:
        print(f"  !! {acronym}: no omic data returned; skipped")
        return None
    keep = out[omic_cols].notna().any(axis=1)
    out = out.loc[keep]

    # CANCER_TYPE_ACRONYM must name the cohort as loaded, because the loader
    # builds the tissue-of-origin vocabulary from it.
    for col in ("PATIENT_ID", "CANCER_TYPE_ACRONYM"):
        if col in out.columns:
            out = out.rename(columns={col: f"TCGA_{col}"})
    out.insert(0, "PATIENT_ID", out.index)
    out.insert(1, "CANCER_TYPE_ACRONYM", acronym)
    out = out.reset_index(drop=True)

    path = os.path.join(out_dir, f"{acronym}_merged_full.parquet")
    out.to_parquet(path, index=False)
    n_mut = sum(c.startswith("MUT_") for c in out.columns)
    n_cna = sum(c.startswith("CNA_") for c in out.columns)
    n_rna = sum(c.startswith("RNA_") for c in out.columns)
    print(f"  {acronym:9s} patients={len(out):5d} "
          f"mut={n_mut:4d} cna={n_cna:4d} rna={n_rna:4d} "
          f"({time.time() - t0:5.1f}s) -> {os.path.basename(path)}", flush=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data_pan")
    ap.add_argument("--panel", default="IMPACT468",
                    help="cBioPortal gene-panel id fixing the feature space")
    ap.add_argument("--studies", default="",
                    help="comma-separated acronyms (default: all TCGA PanCanAtlas)")
    ap.add_argument("--min-samples", type=int, default=0,
                    help="skip studies smaller than this")
    ap.add_argument("--limit", type=int, default=0, help="first N studies only")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    genes = panel_genes(args.panel)
    print(f"panel {args.panel}: {len(genes)} genes")
    with open(os.path.join(args.out, "panel_genes.json"), "w") as fh:
        json.dump({"panel": args.panel, "genes": genes}, fh, indent=1)

    studies = list_pancan_studies()
    if args.studies:
        want = {s.strip().lower() for s in args.studies.split(",") if s.strip()}
        studies = [s for s in studies
                   if s["studyId"].split("_")[0].lower() in want]
    if args.min_samples:
        studies = [s for s in studies
                   if int(s.get("allSampleCount") or 0) >= args.min_samples]
    if args.limit:
        studies = studies[:args.limit]
    print(f"{len(studies)} studies to fetch\n")

    done, failed = [], []
    for s in studies:
        sid = s["studyId"]
        acr = sid.split("_")[0].upper()
        path = os.path.join(args.out, f"{acr}_merged_full.parquet")
        if os.path.isfile(path):
            print(f"  {acr:9s} already present; skipped")
            done.append(acr)
            continue
        try:
            if fetch_study(sid, acr, genes, args.out):
                done.append(acr)
            else:
                failed.append(acr)
        except Exception as err:               # one study must not sink the run
            print(f"  !! {acr}: {err}")
            failed.append(acr)

    print(f"\nfetched {len(done)}: {' '.join(sorted(done))}")
    if failed:
        print(f"failed/skipped {len(failed)}: {' '.join(sorted(failed))}")
    sys.exit(0 if done else 1)


if __name__ == "__main__":
    main()
