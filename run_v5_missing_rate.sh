#!/bin/bash
# Gate 4: missing-rate curve under the final protocol (map + 3-NN, LayerScale).
# baseline + full_model x eta {0.3, 0.5, 0.9} x seeds {111,222,333}.
# eta=0.7 comes from the v5k campaign. Results: ablation_results_v5_missing_rate/

export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/pub_766/miniconda3/envs/ramsdd/bin:/home/pub_766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

PYTHON=/home/pub_766/miniconda3/envs/ramsdd/bin/python
ORIGINAL=/home/pub_766/MISDD-MM/MISDD_MM/model.py
BACKUP=/home/pub_766/MISDD-MM/MISDD_MM/model_full.py
RESULTS_DIR=/home/pub_766/MISDD-MM/ablation_results_v5_missing_rate
cd /home/pub_766/MISDD-MM
trap 'cp "$BACKUP" "$ORIGINAL"' EXIT

LOG=/home/pub_766/MISDD-MM/v5_missing_rate.log
echo "V5 missing-rate sweep starting at $(date)" | tee -a $LOG

CLASSES=(bagel cable_gland carrot cookie dowel foam peach potato rope tire)

is_complete() {
    local csv="${RESULTS_DIR}/eta$1/seed$2/$3.csv"
    if [ ! -f "$csv" ]; then return 1; fi
    local lines=$(wc -l < "$csv")
    if [ "$lines" -lt 11 ]; then return 1; fi
    local zeros=$(awk -F',' 'NR>1 && ($2==0||$3==0||$4==0)' "$csv")
    if [ -n "$zeros" ]; then return 1; fi
    return 0
}

run_config() {
    local eta=$1
    local seed=$2
    local name=$3
    local model_src=$4

    mkdir -p "${RESULTS_DIR}/eta${eta}/seed${seed}"

    if is_complete $eta $seed $name; then
        echo "SKIP (already complete): eta${eta}/seed${seed}/${name}" | tee -a $LOG
        return 0
    fi

    local result_csv="/home/pub_766/MISDD-MM/result/mvtec3d/both/${eta}/csv/Seed_${seed}-results.csv"
    local out_csv="${RESULTS_DIR}/eta${eta}/seed${seed}/${name}.csv"

    echo "========================================" | tee -a $LOG
    echo "START eta=${eta} seed=${seed}: ${name} at $(date)" | tee -a $LOG

    rm -f "$result_csv"
    cp "$model_src" "$ORIGINAL"
    rm -rf /home/pub_766/MISDD-MM/MISDD_MM/__pycache__

    for class in "${CLASSES[@]}"; do
        echo "--- eta${eta} seed${seed} ${name}: ${class} ---" | tee -a $LOG
        WANDB_MODE=offline $PYTHON train_cls.py \
            --dataset mvtec3d --class_name $class \
            --missing_type both --missing_rate $eta \
            --seed $seed --gpu-id 0 \
            --batch-size 32 --max_norm 1.0 --Epoch 25 \
            --img_score_mode map --map_knn 3 2>&1 | tee -a $LOG
    done

    if [ -f "$result_csv" ]; then
        line_count=$(wc -l < "$result_csv")
        zeros=$(awk -F',' 'NR>1 && ($2==0||$3==0||$4==0) {print $0}' "$result_csv")
        if [ "$line_count" -ge 11 ] && [ -z "$zeros" ]; then
            cp "$result_csv" "$out_csv"
            echo "SAVED: eta${eta}/seed${seed}/${name}.csv" | tee -a $LOG
        else
            echo "FAILED: eta${eta}/seed${seed}/${name} - lines=$line_count zeros=$zeros" | tee -a $LOG
            cp "$BACKUP" "$ORIGINAL"
            exit 1
        fi
    else
        echo "FAILED: no CSV for eta${eta}/seed${seed}/${name}" | tee -a $LOG
        cp "$BACKUP" "$ORIGINAL"
        exit 1
    fi

    cp "$BACKUP" "$ORIGINAL"
    echo "COMPLETE: eta${eta}/seed${seed}/${name} at $(date)" | tee -a $LOG
}

for eta in 0.3 0.5 0.9; do
    for seed in 111 222 333; do
        run_config $eta $seed baseline   /home/pub_766/MISDD-MM/ablation_models/model_baseline.py
        run_config $eta $seed full_model /home/pub_766/MISDD-MM/MISDD_MM/model_full.py
    done
done

echo "V5 MISSING-RATE SWEEP COMPLETE at $(date)" | tee -a $LOG
