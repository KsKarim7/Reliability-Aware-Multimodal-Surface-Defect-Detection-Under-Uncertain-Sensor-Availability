#!/bin/bash
# Usage: bash run_v2_segment.sh <segment_number>
# Segment 1: seed111 innov1-4
# Segment 2: seed111 full + seed222 innov1-3
# Segment 3: seed222 innov4+full + seed333 innov1-2
# Segment 4: seed333 innov3-5+full

export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/pub_766/miniconda3/envs/ramsdd/bin:/home/pub_766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

PYTHON=/home/pub_766/miniconda3/envs/ramsdd/bin/python
ORIGINAL=/home/pub_766/MISDD-MM/MISDD_MM/model.py
BACKUP=/home/pub_766/MISDD-MM/MISDD_MM/model_full.py
RESULTS_DIR=/home/pub_766/MISDD-MM/ablation_results_v2
cd /home/pub_766/MISDD-MM

SEGMENT=${1:-1}
LOG=/home/pub_766/MISDD-MM/ablation_v2_segment${SEGMENT}.log
echo "Starting segment ${SEGMENT} at $(date)" | tee -a $LOG

CLASSES=(bagel cable_gland carrot cookie dowel foam peach potato rope tire)

# Check if a config is already complete (11 lines, no zeros)
is_complete() {
    local seed=$1
    local name=$2
    local csv="${RESULTS_DIR}/seed${seed}/${name}.csv"
    if [ ! -f "$csv" ]; then return 1; fi
    local lines=$(wc -l < "$csv")
    if [ "$lines" -lt 11 ]; then return 1; fi
    local zeros=$(awk -F',' 'NR>1 && ($2==0||$3==0||$4==0)' "$csv")
    if [ -n "$zeros" ]; then return 1; fi
    return 0
}

run_config() {
    local seed=$1
    local name=$2
    local model_src=$3

    mkdir -p "${RESULTS_DIR}/seed${seed}"

    # Skip if already complete
    if is_complete $seed $name; then
        echo "SKIP (already complete): seed${seed}/${name}" | tee -a $LOG
        return 0
    fi

    local result_csv="/home/pub_766/MISDD-MM/result/mvtec3d/both/0.7/csv/Seed_${seed}-results.csv"
    local out_csv="${RESULTS_DIR}/seed${seed}/${name}.csv"

    echo "========================================" | tee -a $LOG
    echo "START seed=${seed}: ${name} at $(date)" | tee -a $LOG
    echo "========================================" | tee -a $LOG

    rm -f "$result_csv"
    cp "$model_src" "$ORIGINAL"

    for class in "${CLASSES[@]}"; do
        echo "--- seed${seed} ${name}: ${class} ---" | tee -a $LOG
        WANDB_MODE=offline $PYTHON train_cls.py \
            --dataset mvtec3d --class_name $class \
            --missing_type both --missing_rate 0.7 \
            --seed $seed --gpu-id 0 2>&1 | tee -a $LOG
    done

    # Validate and save
    if [ -f "$result_csv" ]; then
        line_count=$(wc -l < "$result_csv")
        zeros=$(awk -F',' 'NR>1 && ($2==0||$3==0||$4==0) {print $0}' "$result_csv")
        if [ "$line_count" -ge 11 ] && [ -z "$zeros" ]; then
            cp "$result_csv" "$out_csv"
            echo "SAVED: seed${seed}/${name}.csv (clean, $line_count lines)" | tee -a $LOG
        else
            echo "FAILED: seed${seed}/${name} - lines=$line_count zeros=$zeros" | tee -a $LOG
            cp "$BACKUP" "$ORIGINAL"
            exit 1
        fi
    else
        echo "FAILED: no CSV for seed${seed}/${name}" | tee -a $LOG
        cp "$BACKUP" "$ORIGINAL"
        exit 1
    fi

    cp "$BACKUP" "$ORIGINAL"
    echo "COMPLETE: seed${seed}/${name} at $(date)" | tee -a $LOG
}

# Define all 15 configs in order
declare -a ALL_SEEDS=(111 111 111 111 111  222 222 222 222 222  333 333 333 333 333)
declare -a ALL_NAMES=(innov1_only innov2_only innov3_only innov4_only full_model \
                      innov1_only innov2_only innov3_only innov4_only full_model \
                      innov1_only innov2_only innov3_only innov4_only full_model)
declare -a ALL_MODELS=(
    /home/pub_766/MISDD-MM/ablation_models/model_innov1_only.py
    /home/pub_766/MISDD-MM/ablation_models/model_innov2_only.py
    /home/pub_766/MISDD-MM/ablation_models/model_innov3_only.py
    /home/pub_766/MISDD-MM/ablation_models/model_innov4_only.py
    /home/pub_766/MISDD-MM/MISDD_MM/model_full.py
    /home/pub_766/MISDD-MM/ablation_models/model_innov1_only.py
    /home/pub_766/MISDD-MM/ablation_models/model_innov2_only.py
    /home/pub_766/MISDD-MM/ablation_models/model_innov3_only.py
    /home/pub_766/MISDD-MM/ablation_models/model_innov4_only.py
    /home/pub_766/MISDD-MM/MISDD_MM/model_full.py
    /home/pub_766/MISDD-MM/ablation_models/model_innov1_only.py
    /home/pub_766/MISDD-MM/ablation_models/model_innov2_only.py
    /home/pub_766/MISDD-MM/ablation_models/model_innov3_only.py
    /home/pub_766/MISDD-MM/ablation_models/model_innov4_only.py
    /home/pub_766/MISDD-MM/MISDD_MM/model_full.py
)

# Each segment runs 4 configs (indices 0-3, 4-7, 8-11, 12-14)
START_IDX=$(( (SEGMENT - 1) * 4 ))
END_IDX=$(( START_IDX + 3 ))
if [ $END_IDX -ge 15 ]; then END_IDX=14; fi

echo "Segment ${SEGMENT}: running configs ${START_IDX} to ${END_IDX}" | tee -a $LOG

for idx in $(seq $START_IDX $END_IDX); do
    run_config "${ALL_SEEDS[$idx]}" "${ALL_NAMES[$idx]}" "${ALL_MODELS[$idx]}"
done

echo "Segment ${SEGMENT} COMPLETE at $(date)" | tee -a $LOG

# Print segment summary
echo "" | tee -a $LOG
echo "=== SEGMENT ${SEGMENT} RESULTS ===" | tee -a $LOG
for idx in $(seq $START_IDX $END_IDX); do
    seed="${ALL_SEEDS[$idx]}"
    name="${ALL_NAMES[$idx]}"
    csv="${RESULTS_DIR}/seed${seed}/${name}.csv"
    if [ -f "$csv" ]; then
        mean=$(awk -F',' 'NR>1 {sum+=$2; count++} END {printf "%.2f", sum/count}' "$csv")
        echo "  seed${seed}/${name}: mean I-AUROC = ${mean}%" | tee -a $LOG
    else
        echo "  seed${seed}/${name}: NOT FOUND" | tee -a $LOG
    fi
done
