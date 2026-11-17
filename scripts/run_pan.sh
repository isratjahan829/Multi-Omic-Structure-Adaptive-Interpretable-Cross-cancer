#!/usr/bin/env bash
# Pan-cancer schedule: 32 TCGA studies, 10 953 patients, one fixed gene panel.
#
# DATA
# ----
# ./data_pan, built by scripts/fetch_cbioportal_api.py from the cBioPortal
# public API: every TCGA PanCanAtlas study, restricted to the MSK-IMPACT 468
# panel so that all 32 cohorts share one feature space fixed before any data
# was seen. 10 953 patients, 3 543 deaths, 6 707 staged.
#
# TARGETS
# -------
#   cohort        tissue of origin, 32 classes. Chance is ~3% here rather than
#                 25%, so unlike the four-cohort version this is NOT saturated
#                 and methods can actually be separated on it.
#   stage_binary  early (I-II) vs late (III-IV), 6 707 labelled patients.
#                 Four-class AJCC is a worse question across cohorts: the
#                 sub-stage vocabularies differ by study, the classes are badly
#                 unbalanced, and the prognostic boundary is I/II vs III/IV.
#
# COST, MEASURED
# --------------
#   200 genes, fit = 5 841 patients, batch 32   ->  137.9 s/epoch
# so one 20-epoch fit is roughly 35-46 min depending on the split. The phases
# below are ordered by value per minute, because the container running this is
# rebuilt roughly every forty-five minutes and the schedule has to survive
# being interrupted at any point:
#
#   1. baseline panels            no GNN fit   the Cox / RSF / XGBoost panel
#   2. five-fold CV, both targets  10 fits     headline mean +/- SD
#   3. diagnostics + paired tests   2 fits     per-class, per-cohort, the gap
#   4. permutation null              3 fits    leakage control
#   5. ablation                      8 fits    which components earn their place
#   6. leave-one-cancer-out         32 fits    runs last; may not finish, and
#                                              is reported as not run if so
#
# Restart-safe throughout: finished phases are skipped by marker file, the
# baseline panel resumes per fold, and every GNN fit resumes from its own
# epoch-level checkpoint. Re-run as often as the container dies.
set -u

ROOT=./data_pan
GENES="${GENES:-200}"
EPOCHS="${EPOCHS:-20}"
FOLDS="${FOLDS:-5}"
BATCH="${BATCH:-32}"
ABL_FOLDS="${ABL_FOLDS:-1}"
PERMUTATIONS="${PERMUTATIONS:-3}"
MARKERS="${MARKERS:-runs_markers_pan}"
# The baseline panel is the slowest phase at this scale: twenty-odd models
# over 10 953 patients. Evaluating it on one fold instead of five leaves
# compute for MOSAIC-GNN's own cross-validation, without which the work has
# no headline result. The report states the fold count; raise this once the
# GNN numbers exist and the panel will fill in the remaining folds from its
# per-model cache.
export MOSAIC_BASELINE_FOLDS="${MOSAIC_BASELINE_FOLDS:-1}"
# random_survival_forest cannot be fitted on this machine at pan-cancer scale.
# Bounded to two threads, max_depth=12 and max_features="sqrt" it was still
# OOM-killed, and a SIGKILL cannot be caught, so each attempt destroyed the
# whole panel rather than leaving one blank row. It is reported as not run,
# with that reason, rather than silently omitted.
export MOSAIC_SKIP_BASELINES="${MOSAIC_SKIP_BASELINES:-random_survival_forest}"

# Every cohort present on disk, so adding a study needs no edit here.
COHORTS="$(ls "$ROOT" | sed -n 's/_merged_full\.parquet$//p' | paste -sd, -)"
mkdir -p "$MARKERS" runs_pan_cohort runs_pan_stage

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
        --root "$ROOT" --cohorts "$COHORTS" --target "$target" \
        --runs "$out" --out "$out" --tag "$target" --max-genes "$GENES" \
        --epochs "$EPOCHS" --folds "$FOLDS" --batch-size "$BATCH" "$@"
}

echo "cohorts: $COHORTS"
echo "genes=$GENES epochs=$EPOCHS folds=$FOLDS batch=$BATCH"

# ---- 1. baseline panels ------------------------------------------------- #
step pan_cohort_baseline python -u -m mosaic.cli baseline $(common cohort runs_pan_cohort)
step pan_stage_baseline  python -u -m mosaic.cli baseline $(common stage_binary runs_pan_stage)

# ---- 2. primary-fold diagnostics: MOSAIC-GNN's own headline numbers ----- #
# Before cross-validation, not after. This is ONE fit per target rather than
# five, and it produces the model's own metrics, the per-class and per-cohort
# breakdowns, the generalisation gap and the paired tests against every
# baseline. Until it has run the comparison table has baselines and nothing
# to compare them with, so it buys more of the paper per hour than any other
# phase on the list.
step pan_cohort_primary tables cohort       runs_pan_cohort --permutations 0 --skip-sample-size
step pan_stage_primary  tables stage_binary runs_pan_stage  --permutations 0 --skip-sample-size

# ---- 3. five-fold cross-validation (mean +/- SD) ------------------------ #
step pan_cohort_cv python -u -m mosaic.cli train $(common cohort runs_pan_cohort)
step pan_stage_cv  python -u -m mosaic.cli train $(common stage_binary runs_pan_stage)

# ---- 4. permutation null ------------------------------------------------ #
step pan_stage_perm  tables stage_binary runs_pan_stage \
    --permutations "$PERMUTATIONS" --skip-sample-size --reuse-model
step pan_cohort_perm tables cohort runs_pan_cohort \
    --permutations "$PERMUTATIONS" --skip-sample-size --reuse-model

# ---- 5. ablation -------------------------------------------------------- #
step pan_stage_ablate python -u -m mosaic.cli ablate \
    $(common stage_binary runs_pan_stage) --folds "$ABL_FOLDS"
step pan_stage_tables_abl tables stage_binary runs_pan_stage --no-model
step pan_cohort_ablate python -u -m mosaic.cli ablate \
    $(common cohort runs_pan_cohort) --folds "$ABL_FOLDS"
step pan_cohort_tables_abl tables cohort runs_pan_cohort --no-model

# ---- 6. leave-one-cancer-out (32 fits; last on purpose) ----------------- #
step pan_stage_loco  python -u -m mosaic.cli loco $(common stage_binary runs_pan_stage)
step pan_cohort_loco python -u -m mosaic.cli loco $(common cohort runs_pan_cohort)
step pan_stage_tables_loco  tables stage_binary runs_pan_stage --no-model
step pan_cohort_tables_loco tables cohort runs_pan_cohort --no-model

echo "================ ALL PHASES ATTEMPTED  $(date -u) ================"
