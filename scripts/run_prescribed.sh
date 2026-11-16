#!/usr/bin/env bash
# Run at the prescribed settings (--max-genes 1500 --epochs 120 --folds 5
# --repeats 1 --device cpu) and deliver whatever finishes, in priority order.
#
# At 1500 genes one model fit measured out at >180 s/epoch on this 4-core box,
# i.e. ~4.5 h, so the full schedule (~64 fits per target) cannot run. The
# order below is therefore chosen so that the metrics named in the brief --
# balanced accuracy, AUROC, C-index, the generalisation gap -- exist for BOTH
# targets after roughly ten hours rather than after one target's 22-hour
# cross-validation:
#
#   1. primary-fold model + diagnostics, per target   (1 fit each)
#   2. baseline panel, per target                     (no MOSAIC-GNN fit)
#   3. paired tests, reusing the saved checkpoint      (no new fit)
#   4. 5-fold cross-validation, per target             (5 fits each)
#   5. leave-one-cancer-out                            (4 fits each)
#   6. permutation null                                (5 fits each)
#   7. ablation                                        (8 x folds fits each)
#
# Every step is independent and writes its own files, so stopping at any point
# leaves a coherent, if partial, result set. run_cv additionally checkpoints
# after each fold, and records folds_completed / complete in its JSON.
set -u

ROOT=./data_real
COHORTS=BLCA,LIHC,THCA,LUNG
GENES=1500
EPOCHS=120
FOLDS=5
BATCH=32          # full batch does not fit in 16 GB at this gene count
ABL_FOLDS="${ABL_FOLDS:-2}"
PERMUTATIONS="${PERMUTATIONS:-5}"

common () {   # common <target> <out>
    echo --root "$ROOT" --cohorts "$COHORTS" --target "$1" \
         --max-genes "$GENES" --epochs "$EPOCHS" --folds "$FOLDS" \
         --repeats 1 --batch-size "$BATCH" --device cpu \
         --out "$2" --tag "$1"
}

MARKERS="${MARKERS:-runs_markers}"
mkdir -p "$MARKERS"

# This machine is reclaimed after a couple of hours of session inactivity, and
# that kills whatever is running detached: the first attempt at this schedule
# lost 3h20m of training that way, mid-fit, with nothing to show. So the script
# has to be safe to start again from the top, as often as needed:
#
#   * a step that has already finished is skipped, by its marker file;
#   * a step that was interrupted part-way starts again, but the training loop
#     itself resumes from its own epoch-level checkpoint (see
#     Experiment.train_fold), so at most the last few epochs are repeated.
#
# Together those mean the total work is bounded no matter how many times the
# container is taken away.
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

mkdir -p runs_real_cohort runs_real_stage

# ---- 1. baseline panels ------------------------------------------------- #
# These run FIRST, ahead of the MOSAIC-GNN fits, for two reasons. They are the
# only step that answers the question the brief asks directly -- whether a Cox
# model or a random survival forest beats MOSAIC-GNN on c-index -- and they
# cost far less than the eight hours one GNN fit needs. They are also the step
# that cannot resume: a baseline panel killed half way starts again from
# nothing, so it wants the freshest, longest window available. The GNN fits
# that follow accumulate progress across interruptions and so lose nothing by
# waiting.
step cohort_baseline python -u -m mosaic.cli baseline $(common cohort runs_real_cohort)
step stage_baseline  python -u -m mosaic.cli baseline $(common stage  runs_real_stage)

# ---- 2. primary fold + diagnostics for both targets --------------------- #
step cohort_primary tables cohort runs_real_cohort \
    --permutations 0 --skip-sample-size
step stage_primary  tables stage  runs_real_stage \
    --permutations 0 --skip-sample-size

# ---- 3. paired tests, reusing the checkpoints from step 1 --------------- #
step cohort_paired tables cohort runs_real_cohort \
    --permutations 0 --skip-sample-size --reuse-model
step stage_paired  tables stage  runs_real_stage \
    --permutations 0 --skip-sample-size --reuse-model

# ---- 4. five-fold cross-validation -------------------------------------- #
step cohort_cv python -u -m mosaic.cli train $(common cohort runs_real_cohort)
step stage_cv  python -u -m mosaic.cli train $(common stage  runs_real_stage)

# Fold in the CV means now that they exist.
step cohort_tables3 tables cohort runs_real_cohort --no-model
step stage_tables3  tables stage  runs_real_stage  --no-model

# ---- 5. leave-one-cancer-out -------------------------------------------- #
step cohort_loco python -u -m mosaic.cli loco $(common cohort runs_real_cohort)
step stage_loco  python -u -m mosaic.cli loco $(common stage  runs_real_stage)
step cohort_tables4 tables cohort runs_real_cohort --no-model
step stage_tables4  tables stage  runs_real_stage  --no-model

# ---- 6. permutation null ------------------------------------------------ #
step cohort_perm tables cohort runs_real_cohort \
    --permutations "$PERMUTATIONS" --skip-sample-size --reuse-model
step stage_perm  tables stage  runs_real_stage \
    --permutations "$PERMUTATIONS" --skip-sample-size --reuse-model

# ---- 7. ablation -------------------------------------------------------- #
step cohort_ablate python -u -m mosaic.cli ablate \
    $(common cohort runs_real_cohort) --folds "$ABL_FOLDS"
step stage_ablate  python -u -m mosaic.cli ablate \
    $(common stage  runs_real_stage)  --folds "$ABL_FOLDS"
step cohort_tables5 tables cohort runs_real_cohort --no-model
step stage_tables5  tables stage  runs_real_stage  --no-model

echo "================ ALL STEPS ATTEMPTED  $(date -u) ================"
