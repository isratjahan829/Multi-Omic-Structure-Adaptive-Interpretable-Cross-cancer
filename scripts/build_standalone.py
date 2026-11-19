"""Build one self-contained Python file that runs the whole study.

    python scripts/build_standalone.py --out deliverables/MOSAIC_GNN_complete.py

The repository is ~9,400 lines across a package and a dozen scripts, which is
the right shape for development and the wrong shape for handing to an examiner
or attaching to a submission. This emits a single file that carries the entire
`mosaic` package inside it, base64-encoded, unpacks it to a temporary
directory at import time, and exposes the full schedule behind one CLI.

It is the same mechanism the notebook already uses, so the two deliverables
cannot drift apart: both embed the package from the same source tree at build
time.

What the generated file can run:

    python MOSAIC_GNN_complete.py fetch      # download from cBioPortal
    python MOSAIC_GNN_complete.py train      # primary fold + diagnostics
    python MOSAIC_GNN_complete.py baseline   # the 24-model panel
    python MOSAIC_GNN_complete.py ablate     # 8-arm component ablation
    python MOSAIC_GNN_complete.py loco       # leave-one-cancer-out, 32 folds
    python MOSAIC_GNN_complete.py tables     # every CSV in the report
    python MOSAIC_GNN_complete.py all        # the whole schedule in order

Defaults reproduce the manuscript configuration (200 genes, 20 epochs, 5
folds). The reduced settings used for the ablation, permutation, LOCO and
sample-size phases are passed explicitly, exactly as they were in the runs
this report describes, so nothing about the provenance is hidden in a default.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import textwrap
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def package_blob(pkg_dir: str, extra: list | None = None) -> str:
    """Zip a package directory and return it as a wrapped base64 literal.

    `extra` names loose scripts to include beside the package. export_tables
    lives in scripts/ rather than inside the package, and the `tables`
    subcommand runs it by path, so a build without it ships a CLI with one
    command that cannot work.
    """
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in (extra or []):
            if os.path.isfile(f):
                zf.write(f, os.path.basename(f))
                n += 1
        for root, _dirs, files in os.walk(pkg_dir):
            if "__pycache__" in root:
                continue
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                full = os.path.join(root, name)
                rel = os.path.relpath(full, os.path.dirname(pkg_dir))
                zf.write(full, rel)
                n += 1
    blob = base64.b64encode(buf.getvalue()).decode()
    lines = textwrap.wrap(blob, 96)
    print(f"  embedded {n} modules from {pkg_dir} "
          f"({len(blob) / 1024:.0f} KB base64)")
    return "\n".join(f'    "{ln}"' for ln in lines)


HEADER = '''"""MOSAIC-GNN — complete, self-contained implementation.

A two-level multi-omic graph neural network for pan-cancer prognosis and
interpretable biomarker discovery.

This single file contains the entire implementation. The `mosaic` package is
embedded below as base64 and unpacked at import time, so there is nothing to
install beyond the dependencies listed under REQUIREMENTS and nothing to
download except the data itself.

QUICK START
-----------
    pip install torch numpy pandas scikit-learn scipy lifelines pyarrow
    python MOSAIC_GNN_complete.py fetch --out ./data_pan
    python MOSAIC_GNN_complete.py all --root ./data_pan --target cohort

WHAT THE SUBCOMMANDS DO
-----------------------
    fetch     retrieve every TCGA PanCanAtlas study from the cBioPortal
              public REST API and write Parquet
    train     fit the primary fold and write the diagnostic tables
    baseline  train the 24-model comparator panel on identical folds
    ablate    remove one architectural component at a time (8 arms)
    loco      leave-one-cancer-out transfer, one fit per held-out study
    tables    export every CSV the report is built from
    all       the whole schedule, in order

REPRODUCING THE REPORTED NUMBERS
--------------------------------
The defaults here are the manuscript configuration: 200 genes, 20 epochs, 5
folds. The ablation, permutation, leave-one-cancer-out and sample-size phases
in the report were run at a REDUCED configuration -- 100 genes, 6 epochs, 2
folds -- because one ablation fold costs about half an hour at pan-cancer
scale. To reproduce those tables, pass those values explicitly:

    python MOSAIC_GNN_complete.py ablate --max-genes 100 --epochs 6 --folds 2

Absolute values from the reduced phases are NOT comparable with the primary
fold; only differences within a reduced table are. Every table in the report
that comes from a reduced run says so.

A NOTE ON WHAT THIS MODEL DOES AND DOES NOT DO
----------------------------------------------
On the evaluation in the accompanying report, this architecture does not win
on classification: 18 of 20 comparators are significantly better on tissue of
origin and 8 of 20 on tumour stage, under a paired bootstrap on identical
held-out patients with Benjamini-Hochberg correction. It is competitive on
survival, and it significantly exceeds a clinical-only Cox model on both
endpoints. The defensible contribution is the quantified
molecular-over-clinical prognostic gain and the audited comparison protocol,
not a performance claim for the architecture. Anyone building on this should
start from that, not from the abstract of a paper that says otherwise.

REQUIREMENTS
------------
    torch >= 2.0, numpy, pandas, scikit-learn, scipy, lifelines, pyarrow
    optional: xgboost, lightgbm (two of the 24 baselines are skipped without
    them, and the panel reports them as not run rather than silently shrinking)
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import zipfile

__all__ = ["main"]

# --------------------------------------------------------------------------- #
# The mosaic package, zipped and base64-encoded.
# --------------------------------------------------------------------------- #
MOSAIC_PKG_B64 = (
{blob}
)


def _install_package() -> str:
    """Unpack the embedded package and put it on sys.path.

    Written to a temporary directory rather than the working directory so that
    running this file never leaves a `mosaic/` folder beside the user's data,
    and so that two copies of this file cannot fight over the same unpack
    location.
    """
    import hashlib
    # A content digest, not hash(): Python salts string hashing per process,
    # so hash() would pick a new directory on every run and re-extract the
    # package each time. The digest also means a rebuilt file with different
    # contents unpacks somewhere else instead of reusing a stale tree.
    digest = hashlib.sha256(MOSAIC_PKG_B64.encode()).hexdigest()[:16]
    dest = os.path.join(tempfile.gettempdir(), "mosaic_pkg_" + digest)
    marker = os.path.join(dest, "mosaic", "__init__.py")
    if not os.path.isfile(marker):
        os.makedirs(dest, exist_ok=True)
        zipfile.ZipFile(
            io.BytesIO(base64.b64decode(MOSAIC_PKG_B64))).extractall(dest)
    if dest not in sys.path:
        sys.path.insert(0, dest)
    return dest


_PKG_DIR = _install_package()
'''

BODY = '''

# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="MOSAIC_GNN_complete.py",
        description="MOSAIC-GNN: pan-cancer multi-omic graph neural network.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("QUICK START")[1].split("REQUIREMENTS")[0])
    ap.add_argument("command",
                    choices=["fetch", "train", "baseline", "ablate", "loco",
                             "tables", "all", "where"],
                    help="which stage to run; 'where' prints the unpacked "
                         "package path and exits")
    ap.add_argument("--root", default="./data_pan",
                    help="directory holding the merged Parquet (default "
                         "./data_pan)")
    ap.add_argument("--out", default="./runs",
                    help="where results are written (default ./runs)")
    ap.add_argument("--target", default="cohort",
                    choices=["cohort", "stage", "stage_binary", "grade",
                             "subtype"])
    ap.add_argument("--cohorts", default="",
                    help="comma-separated study acronyms; default is every "
                         "study found under --root")
    ap.add_argument("--max-genes", type=int, default=200,
                    help="manuscript configuration is 200")
    ap.add_argument("--epochs", type=int, default=20,
                    help="manuscript configuration is 20")
    ap.add_argument("--folds", type=int, default=5,
                    help="manuscript configuration is 5; the reduced phases "
                         "used 2")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--permutations", type=int, default=0,
                    help="label-permutation retrains for the leakage control; "
                         "the smallest attainable p is 1/(B+1)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None,
                    help="prefix for output files; defaults to --target")
    args = ap.parse_args(argv)

    if args.command == "where":
        print(_PKG_DIR)
        return 0

    tag = args.tag or args.target
    cohorts = args.cohorts
    if not cohorts and os.path.isdir(args.root):
        import glob
        found = sorted(
            os.path.basename(p).split("_merged")[0]
            for p in glob.glob(os.path.join(args.root, "*_merged*.parquet")))
        cohorts = ",".join(found)
        if found:
            print(f"[auto] {len(found)} cohorts found under {args.root}")

    if args.command == "fetch":
        return _run_embedded("fetch_cbioportal_api.py",
                             ["--out", args.root])

    common = ["--root", args.root, "--target", args.target,
              "--max-genes", str(args.max_genes),
              "--epochs", str(args.epochs), "--folds", str(args.folds),
              "--batch-size", str(args.batch_size),
              "--device", args.device, "--out", args.out, "--tag", tag]
    if cohorts:
        common += ["--cohorts", cohorts]

    from mosaic.cli import main as cli_main

    def stage(name):
        print(f"\\n{'=' * 70}\\n{name}\\n{'=' * 70}", flush=True)
        cli_main([name] + common)

    if args.command in ("train", "baseline", "ablate", "loco"):
        stage(args.command)
        return 0

    if args.command == "tables":
        return _tables(args, cohorts, tag)

    if args.command == "all":
        for name in ("train", "baseline", "ablate", "loco"):
            stage(name)
        return _tables(args, cohorts, tag)
    return 0


def _run_embedded(script: str, extra_argv: list) -> int:
    """Run one of the embedded loose scripts by path."""
    import runpy
    path = os.path.join(_PKG_DIR, script)
    if not os.path.isfile(path):
        print(f"{script} was not embedded in this build.", file=sys.stderr)
        return 2
    sys.argv = [script] + extra_argv
    runpy.run_path(path, run_name="__main__")
    return 0


def _tables(args, cohorts: str, tag: str) -> int:
    """Export the CSV table set, which is a separate entry point upstream."""
    import runpy
    argv = ["export_tables.py", "--root", args.root, "--target", args.target,
            "--runs", args.out, "--out", args.out, "--tag", tag,
            "--max-genes", str(args.max_genes), "--epochs", str(args.epochs),
            "--folds", str(args.folds), "--batch-size", str(args.batch_size),
            "--permutations", str(args.permutations)]
    if cohorts:
        argv += ["--cohorts", cohorts]
    sys.argv = argv
    path = os.path.join(_PKG_DIR, "export_tables.py")
    if not os.path.isfile(path):
        print("export_tables.py was not embedded in this build; run the "
              "repository copy instead.", file=sys.stderr)
        return 2
    runpy.run_path(path, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="deliverables/MOSAIC_GNN_complete.py")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    blob = package_blob(os.path.join(REPO, "mosaic"),
                        extra=[os.path.join(REPO, "scripts", f) for f in
                               ("export_tables.py", "merge_cbioportal.py",
                                "fetch_cbioportal_api.py")])
    text = HEADER.replace("{blob}", blob) + BODY
    with open(args.out, "w") as fh:
        fh.write(text)
    kb = os.path.getsize(args.out) / 1024
    lines = text.count("\n") + 1
    print(f"wrote {args.out} ({kb:.0f} KB, {lines:,} lines)")


if __name__ == "__main__":
    main()
