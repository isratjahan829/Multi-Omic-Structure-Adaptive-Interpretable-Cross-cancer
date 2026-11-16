#!/usr/bin/env bash
# The schedule that actually completes on this machine, for BOTH targets.
#
# WHY THESE SETTINGS, AND WHAT THEY COST
# --------------------------------------
# The originally requested settings were 1500 genes and 120 epochs. Measured
# on this four-core CPU container, on the real merged data:
#
#   genes  hidden  layers  batch  knn    s/epoch
#    1500      64       3     32   12      240.0
#     300      64       3     32   12       30.2
#     300      64       3    128   12       42.8
#     300      64       2    128    8       23.0
#
# At 240 s/epoch a 120-epoch fit is eight hours. The complete schedule below
# is 42 fits across the two targets, i.e. roughly two weeks of compute, on a
# container that is reclaimed and rebuilt roughly every forty-five minutes.
# That is not slow, it is unreachable: the first attempt lost three hours and
# twenty minutes mid-fit and produced nothing.
#
# At 300 genes and 30 epochs one fit is about fifteen minutes and the whole
# schedule is about ten hours, which does complete. What is given up is gene
# budget, not method: 300 of the 3000 unsupervised-prefiltered genes instead
# of 1500. Two things make that an acceptable trade rather than a fudge.
# First, the architecture is unchanged -- three RGAT layers, knn 12, hidden
# 64, exactly as designed -- because changing the model would mean the
# evaluated model is not the proposed one. Only the data width moves.
# Second, the gene budget cannot change the headline findings: plain logistic
# regression already reaches balanced accuracy 1.000 on tissue of origin at
# 200 genes AND at 1500, so that target is saturated at every budget, and
# four-class stage is hard at every budget tested.
#
# Epochs 30 rather than 120 because the validation trace at 1500 genes was
# already flat: epoch 20 -> 40 moved balanced accuracy 0.991 -> 0.994 and
# c-index 0.737 -> 0.740. Early stopping can cut it shorter still.
#
# ORDERING
# --------
# Phases run in descending value-per-minute, so that an interruption leaves
# the most defensible set of tables rather than an arbitrary one:
#
#   1. baseline panels      no GNN fit    answers the Cox / RSF question
#   2. primary fold         2 fits        the model's own headline metrics
#   3. paired tests         0 fits        reuses the step-2 checkpoints
#   4. 5-fold CV           10 fits        mean +/- SD, the generalisation gap
#   5. ablation            16 fits        which components earn their place
#   6. leave-one-cancer-out 8 fits        transfer
#   7. permutation null     6 fits        leakage check
#
# Everything is restart-safe: finished phases are skipped by marker file, the
# baseline panel resumes per fold, and each GNN fit resumes from its own
# epoch-level checkpoint. Run this script as often as the container dies.
set -u

ROOT=./data_real
COHORTS=BLCA,LIHC,THCA,LUNG
GENES="${GENES:-300}"
EPOCHS="${EPOCHS:-30}"
FOLDS="${FOLDS:-5}"
BATCH="${BATCH:-32}"
ABL_FOLDS="${ABL_FOLDS:-1}"
PERMUTATIONS="${PERMUTATIONS:-3}"
MARKERS="${MARKERS:-runs_markers}"
mkdir -p "$MARKERS" runs_real_cohort runs_real_stage

common () {   # common <target> <out>
    echo --root "$ROOT" --cohorts "$COHORTS" --target "$1" \
         --max-genes "$GENES" --epochs "$EPOCHS" --folds "$FOLDS" \
         --repeats 1 --batch-size "$BATCH" --device cpu \
         --out "$2" --tag "$1"
}

step () {
    local name="$1"; shift
    local marker="$MARKERS/$name.done"
    if [ -f "$marker" ]; then
        echo "################ SKIP $name (finished $(cat "$marker")) ################"
        return 0
    fi
    echo "################ START $name  $(date -u +%H:%M:%S) ################"
    local t0=$SECONDS
    if "$@"; then
        echo "################ DONE $name after $((SECONDS - t0))s ################"
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$marker"
    else
        echo "################ FAILED $name after $((SECONDS - t0))s ################"
    fi
}

tables () {   # tables <target> <out> <extra args...>
    local target="$1" out="$2"; shift 2
    python -u scripts/export_tables.py \
        --root "$ROOT" --target "$target" --runs "$out" --out "$out" \
        --tag "$target" --max-genes "$GENES" --epochs "$EPOCHS" \
        --folds "$FOLDS" --batch-size "$BATCH" "$@"
}

# ---- 1. baseline panels (no MOSAIC-GNN fit) ----------------------------- #
step cohort_baseline python -u -m mosaic.cli baseline $(common cohort runs_real_cohort)
step stage_baseline  python -u -m mosaic.cli baseline $(common stage  runs_real_stage)

# ---- 2. primary fold + diagnostics ------------------------------------- #
step cohort_primary tables cohort runs_real_cohort --permutations 0 --skip-sample-size
step stage_primary  tables stage  runs_real_stage  --permutations 0 --skip-sample-size

# ---- 3. paired bootstrap tests, reusing the step-2 checkpoints ---------- #
step cohort_paired tables cohort runs_real_cohort \
    --permutations 0 --skip-sample-size --reuse-model
step stage_paired  tables stage  runs_real_stage \
    --permutations 0 --skip-sample-size --reuse-model

# ---- 4. five-fold cross-validation ------------------------------------- #
step cohort_cv python -u -m mosaic.cli train $(common cohort runs_real_cohort)
step stage_cv  python -u -m mosaic.cli train $(common stage  runs_real_stage)
step cohort_tables_cv tables cohort runs_real_cohort --no-model
step stage_tables_cv  tables stage  runs_real_stage  --no-model

# ---- 5. ablation ------------------------------------------------------- #
step cohort_ablate python -u -m mosaic.cli ablate \
    $(common cohort runs_real_cohort) --folds "$ABL_FOLDS"
step stage_ablate  python -u -m mosaic.cli ablate \
    $(common stage  runs_real_stage)  --folds "$ABL_FOLDS"
step cohort_tables_abl tables cohort runs_real_cohort --no-model
step stage_tables_abl  tables stage  runs_real_stage  --no-model

# ---- 6. leave-one-cancer-out ------------------------------------------- #
step cohort_loco python -u -m mosaic.cli loco $(common cohort runs_real_cohort)
step stage_loco  python -u -m mosaic.cli loco $(common stage  runs_real_stage)
step cohort_tables_loco tables cohort runs_real_cohort --no-model
step stage_tables_loco  tables stage  runs_real_stage  --no-model

# ---- 7. permutation null ----------------------------------------------- #
step cohort_perm tables cohort runs_real_cohort \
    --permutations "$PERMUTATIONS" --skip-sample-size --reuse-model
step stage_perm  tables stage  runs_real_stage \
    --permutations "$PERMUTATIONS" --skip-sample-size --reuse-model

echo "================ ALL PHASES ATTEMPTED  $(date -u) ================"
