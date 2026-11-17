"""Execute the analysis notebook on the real data and save it with its outputs.

    python scripts/execute_notebook.py                 # defaults below
    python scripts/execute_notebook.py --epochs 8 --max-genes 150

A notebook shipped without outputs is a script with extra punctuation. One
shipped with outputs from synthetic data is worse than useless. This runs
notebooks/mosaic_gnn.ipynb end to end against ./data_pan and writes
notebooks/mosaic_gnn_executed.ipynb.

Three things it does that a bare nbclient call does not, each because of a
failure that actually happened here:

1.  It ALWAYS writes the notebook, in a finally block. nbclient's
    `allow_errors` covers exceptions raised inside a cell; it does NOT cover a
    cell timeout, which raises CellTimeoutError out of execute() instead. A
    first attempt therefore ran for an hour, hit the per-cell timeout on the
    training cell, and died before writing anything at all -- an hour of real
    computation discarded because the result was held in memory until the end.
    Whatever happens now, the outputs produced up to that point are on disk.

2.  It refuses to present a synthetic run as a real one. The notebook falls
    back to a small synthetic sample when it cannot find the data, and prints a
    banner saying so. This scans the executed outputs for that banner and exits
    non-zero if it is there, because the failure mode being guarded against is
    precisely a notebook full of plausible numbers that mean nothing. An
    earlier version of the configuration cell looked for a CSV extension while
    the pipeline writes Parquet, which is where a balanced accuracy of 0.313
    came from in early drafts.

3.  It reports per cell what happened -- output, error, or nothing -- so the
    result is checkable without opening the notebook.

The phase budgets passed through the environment are deliberately small. The
manuscript numbers come from scripts/run_pan.sh; this notebook's job is to
show the pipeline running end to end on the real data and to be honest about
the difference. See README section 8.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The banner the notebook prints when it cannot find the real data. If this
# appears anywhere in the executed outputs, the run is not a result.
SYNTHETIC_MARKER = "REAL DATASET NOT FOUND"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default=os.path.join(
        REPO, "notebooks", "mosaic_gnn.ipynb"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "notebooks", "mosaic_gnn_executed.ipynb"))
    ap.add_argument("--target", default="cohort")
    ap.add_argument("--max-genes", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--baseline-folds", type=int, default=1)
    ap.add_argument("--ablations", type=int, default=1)
    ap.add_argument("--loco", type=int, default=1)
    ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--sample-size", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cell-timeout", type=int, default=10800,
                    help="seconds per cell; a cell that exceeds it is marked "
                         "timed out and the notebook is still written")
    args = ap.parse_args()

    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(args.notebook, as_version=4)
    print(f"{len(nb.cells)} cells from {args.notebook}", flush=True)

    os.environ.update({
        "MOSAIC_TARGET": args.target,
        "MOSAIC_MAX_GENES": str(args.max_genes),
        "MOSAIC_EPOCHS": str(args.epochs),
        "MOSAIC_NB_FOLDS": str(args.folds),
        "MOSAIC_NB_REPEATS": "1",
        "MOSAIC_NB_BASELINE_FOLDS": str(args.baseline_folds),
        "MOSAIC_NB_ABLATIONS": str(args.ablations),
        "MOSAIC_NB_LOCO": str(args.loco),
        "MOSAIC_NB_PERMUTATIONS": str(args.permutations),
        "MOSAIC_NB_SAMPLE_SIZE": str(args.sample_size),
        # random_survival_forest is OOM-killed at pan-cancer scale, and that
        # signal takes the kernel with it -- ending the whole execution rather
        # than one model. Skipped by name, as the CLI schedule skips it.
        "MOSAIC_SKIP_BASELINES": "random_survival_forest",
        "MOSAIC_BASELINE_BUDGET_S": "900",
        "MOSAIC_SAVE_EVERY": "2",
        "PYTHONPATH": REPO,
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
    })

    client = NotebookClient(
        nb, timeout=args.cell_timeout, kernel_name="python3",
        # The kernel's working directory must be the repository root, not
        # notebooks/: the notebook looks for ./data_pan relative to where it
        # runs, and starting it in notebooks/ is what sent a previous execution
        # to the synthetic fallback.
        resources={"metadata": {"path": REPO}},
        allow_errors=True,          # a failing cell is recorded, not hidden
    )

    t0 = time.time()
    failure = None
    try:
        client.execute()
    except Exception as err:                       # timeout, dead kernel, ...
        failure = err
    finally:
        # Unconditional. The outputs computed so far are worth more than a
        # clean exit, and this is the whole reason this script exists.
        nbformat.write(nb, args.out)
        print(f"wrote {args.out} after {time.time() - t0:.0f}s", flush=True)

    errored, empty = [], []
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        outs = cell.get("outputs") or []
        if any(o.get("output_type") == "error" for o in outs):
            errored.append((i, outs))
        elif not outs:
            empty.append(i)

    code_cells = [c for c in nb.cells if c.get("cell_type") == "code"]
    print(f"{len(code_cells) - len(empty)}/{len(code_cells)} code cells "
          f"produced output; {len(errored)} errored; "
          f"{len(empty)} produced nothing", flush=True)
    for i, outs in errored:
        err = next(o for o in outs if o.get("output_type") == "error")
        print(f"  cell {i} ERROR {err.get('ename')}: "
              f"{str(err.get('evalue'))[:200]}", flush=True)
    if empty:
        print(f"  cells with no output: {empty}", flush=True)
    if failure is not None:
        print(f"\nexecution stopped early: "
              f"{type(failure).__name__}: {str(failure)[:300]}", flush=True)

    # Did it actually see the real data?
    text = "\n".join(
        o.get("text", "") if isinstance(o.get("text"), str)
        else "".join(o.get("text", []))
        for c in nb.cells for o in (c.get("outputs") or []))
    if SYNTHETIC_MARKER in text:
        print("\nFAILED: the notebook fell back to SYNTHETIC data. Every "
              "number in it is meaningless and none may be quoted. Check that "
              "./data_pan exists and holds *_merged_full.parquet, and that the "
              "kernel started in the repository root.", flush=True)
        return 2
    print("\nno synthetic-fallback banner in the outputs: the run saw the "
          "real dataset", flush=True)
    return 1 if failure is not None else 0


if __name__ == "__main__":
    sys.exit(main())
