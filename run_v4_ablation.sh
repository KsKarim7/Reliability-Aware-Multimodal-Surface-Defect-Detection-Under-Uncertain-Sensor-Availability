#!/bin/bash
# V4 corrected-pipeline ablation campaign.
# Usage: bash run_v4_ablation.sh <seed_segment>   (1=seed111, 2=seed222, 3=seed333)
#
# V4 = Path B fixes applied:
#   - gradients flow through the (frozen) encoder into the missing-aware prompts
#   - deep compound prompts injected into the attended path (innovation 1 functional)
#   - missing-aware prompts used in the cls test path
#   - sorted RGB<->depth pairing, BGR->RGB at test time
#   - final-epoch evaluation only (no best-epoch selection on the test set)
#   - runnable baseline config included

export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/pub_766/miniconda3/envs/ramsdd/bin:/home/pub_766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

PYTHON=/home/pub_766/miniconda3/envs/ramsdd/bin/python
ORIGINAL=/home/pub_766/MISDD-MM/MISDD_MM/model.py
BACKUP=/home/pub_766/MISDD-MM/MISDD_MM/model_full.py
RESULTS_DIR=/home/pub_766/MISDD-MM/ablation_results_v4
cd /home/pub_766/MISDD-MM

# always restore the full model, even if this script is killed mid-config —
# a leftover ablated model.py silently poisons later runs and regenerations
trap 'cp "$BACKUP" "$ORIGINAL"' EXIT

SEGMENT=${1:-1}
case "$SEGMENT" in
    1) SEED=111 ;;
    2) SEED=222 ;;
    3) SEED=333 ;;
    *) echo "Unknown segment $SEGMENT"; exit 1 ;;
esac

LOG=/home/pub_766/MISDD-MM/ablation_v4_seed${SEED}.log
echo "Starting V4 segment ${SEGMENT} (seed ${SEED}) at $(date)" | tee -a $LOG

CLASSES=(bagel cable_gland carrot cookie dowel foam peach potato rope tire)

is_complete() {
    local csv="${RESULTS_DIR}/seed${SEED}/$1.csv"
    if [ ! -f "$csv" ]; then return 1; fi
    local lines=$(wc -l < "$csv")
    if [ "$lines" -lt 11 ]; then return 1; fi
    local zeros=$(awk -F',' 'NR>1 && ($2==0||$3==0||$4==0)' "$csv")
    if [ -n "$zeros" ]; then return 1; fi
    return 0
}

run_config() {
    local name=$1
    local model_src=$2

    mkdir -p "${RESULTS_DIR}/seed${SEED}"

    if is_complete $name; then
        echo "SKIP (already complete): seed${SEED}/${name}" | tee -a $LOG
        return 0
    fi

    local result_csv="/home/pub_766/MISDD-MM/result/mvtec3d/both/0.7/csv/Seed_${SEED}-results.csv"
    local out_csv="${RESULTS_DIR}/seed${SEED}/${name}.csv"

    echo "========================================" | tee -a $LOG
    echo "START seed=${SEED}: ${name} at $(date)" | tee -a $LOG
    echo "========================================" | tee -a $LOG

    rm -f "$result_csv"
    cp "$model_src" "$ORIGINAL"

    for class in "${CLASSES[@]}"; do
        echo "--- seed${SEED} ${name}: ${class} ---" | tee -a $LOG
        WANDB_MODE=offline $PYTHON train_cls.py \
            --dataset mvtec3d --class_name $class \
            --missing_type both --missing_rate 0.7 \
            --seed $SEED --gpu-id 0 \
            --batch-size 32 --max_norm 1.0 --Epoch 50 2>&1 | tee -a $LOG
    done

    if [ -f "$result_csv" ]; then
        line_count=$(wc -l < "$result_csv")
        zeros=$(awk -F',' 'NR>1 && ($2==0||$3==0||$4==0) {print $0}' "$result_csv")
        if [ "$line_count" -ge 11 ] && [ -z "$zeros" ]; then
            cp "$result_csv" "$out_csv"
            echo "SAVED: seed${SEED}/${name}.csv (clean, $line_count lines)" | tee -a $LOG
        else
            echo "FAILED: seed${SEED}/${name} - lines=$line_count zeros=$zeros" | tee -a $LOG
            cp "$BACKUP" "$ORIGINAL"
            exit 1
        fi
    else
        echo "FAILED: no CSV for seed${SEED}/${name}" | tee -a $LOG
        cp "$BACKUP" "$ORIGINAL"
        exit 1
    fi

    cp "$BACKUP" "$ORIGINAL"
    echo "COMPLETE: seed${SEED}/${name} at $(date)" | tee -a $LOG
}

run_config baseline    /home/pub_766/MISDD-MM/ablation_models/model_baseline.py
run_config innov1_only /home/pub_766/MISDD-MM/ablation_models/model_innov1_only.py
run_config innov2_only /home/pub_766/MISDD-MM/ablation_models/model_innov2_only.py
run_config innov3_only /home/pub_766/MISDD-MM/ablation_models/model_innov3_only.py
run_config innov4_only /home/pub_766/MISDD-MM/ablation_models/model_innov4_only.py
run_config full_model  /home/pub_766/MISDD-MM/MISDD_MM/model_full.py

echo "V4 segment ${SEGMENT} (seed ${SEED}) COMPLETE at $(date)" | tee -a $LOG

echo "" | tee -a $LOG
echo "=== V4 SEED ${SEED} RESULTS ===" | tee -a $LOG
for name in baseline innov1_only innov2_only innov3_only innov4_only full_model; do
    csv="${RESULTS_DIR}/seed${SEED}/${name}.csv"
    if [ -f "$csv" ]; then
        mean=$(awk -F',' 'NR>1 {sum+=$2; count++} END {printf "%.2f", sum/count}' "$csv")
        echo "  seed${SEED}/${name}: mean I-AUROC = ${mean}%" | tee -a $LOG
    else
        echo "  seed${SEED}/${name}: NOT FOUND" | tee -a $LOG
    fi
done
