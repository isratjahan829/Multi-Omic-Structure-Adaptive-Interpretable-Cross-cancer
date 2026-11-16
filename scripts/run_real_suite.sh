#!/usr/bin/env bash
# Run the full MOSAIC-GNN suite on the real merged cohorts for one target.
#
#   scripts/run_real_suite.sh <target> <out_dir> [ablate_folds]
#
# Order matters: the headline cross-validation, the baseline panel and the
# leave-one-cancer-out transfer come first, then the diagnostics and the
# manuscript tables, and the ablation -- eight model variants, which is by far
# the largest block of compute -- comes last, so an interrupted run still
# leaves every other table complete.
set -u

TARGET="${1:?target: cohort|stage}"
OUT="${2:?output directory}"
ABL_FOLDS="${3:-5}"

ROOT=./data_real
COHORTS=BLCA,LIHC,THCA,LUNG
GENES=1500
EPOCHS=120
FOLDS=5
# The default configuration trains full batch. On this machine a full-batch
# step over ~800 training patients x 1500 genes x 3 relations does not fit in
# RAM, so training is mini-batched at 32 patients (the same batch size the
# notebook uses on CPU).
BATCH=32

COMMON=(--root "$ROOT" --cohorts "$COHORTS" --target "$TARGET"
        --max-genes "$GENES" --epochs "$EPOCHS" --folds "$FOLDS"
        --repeats 1 --batch-size "$BATCH" --device cpu
        --out "$OUT" --tag "$TARGET")

mkdir -p "$OUT"

step () {   # step <name> <command...>
    local name="$1"; shift
    echo "################ START $name  $(date -u +%H:%M:%S) ################"
    local t0=$SECONDS
    if "$@"; then
        echo "################ DONE $name after $((SECONDS - t0))s ################"
    else
        echo "################ FAILED $name after $((SECONDS - t0))s (rc=$?) ################"
    fi
}

step train    python -m mosaic.cli train    "${COMMON[@]}"
step baseline python -m mosaic.cli baseline "${COMMON[@]}"
step loco     python -m mosaic.cli loco     "${COMMON[@]}"

step tables   python scripts/export_tables.py \
    --root "$ROOT" --target "$TARGET" --runs "$OUT" --out "$OUT" \
    --tag "$TARGET" --max-genes "$GENES" --epochs "$EPOCHS" \
    --folds "$FOLDS" --batch-size "$BATCH" --permutations 10

step ablate   python -m mosaic.cli ablate   "${COMMON[@]}" --folds "$ABL_FOLDS"

# The ablation writes the last table, so the export is re-run to pick it up.
step tables2  python scripts/export_tables.py \
    --root "$ROOT" --target "$TARGET" --runs "$OUT" --out "$OUT" \
    --tag "$TARGET" --max-genes "$GENES" --epochs "$EPOCHS" \
    --folds "$FOLDS" --batch-size "$BATCH" --no-model

echo "================ SUITE COMPLETE: $TARGET -> $OUT  $(date -u) ================"
