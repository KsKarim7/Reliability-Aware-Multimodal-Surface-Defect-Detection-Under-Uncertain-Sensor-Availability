#!/bin/bash
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/pub_766/miniconda3/envs/ramsdd/bin:/home/pub_766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

PYTHON=/home/pub_766/miniconda3/envs/ramsdd/bin/python
ORIGINAL=/home/pub_766/MISDD-MM/MISDD_MM/model.py
BACKUP=/home/pub_766/MISDD-MM/MISDD_MM/model_full.py
RESULTS_DIR=/home/pub_766/MISDD-MM/ablation_results_v3
cd /home/pub_766/MISDD-MM

CLASSES=(bagel cable_gland carrot cookie dowel foam peach potato rope tire)
LOG=/home/pub_766/MISDD-MM/ablation_v3_fullmodel.log

mkdir -p $RESULTS_DIR

run_fullmodel() {
    local seed=$1
    local result_csv="/home/pub_766/MISDD-MM/result/mvtec3d/both/0.7/csv/Seed_${seed}-results.csv"
    local out_csv="${RESULTS_DIR}/seed${seed}_full_model.csv"

    if [ -f "$out_csv" ]; then
        lines=$(wc -l < "$out_csv")
        zeros=$(awk -F',' 'NR>1 && ($2==0||$3==0||$4==0)' "$out_csv")
        if [ "$lines" -ge 11 ] && [ -z "$zeros" ]; then
            echo "SKIP (complete): seed${seed}/full_model" | tee -a $LOG
            return 0
        fi
    fi

    echo "===== START full_model seed=${seed} at $(date) =====" | tee -a $LOG
    rm -f "$result_csv"
    cp "$BACKUP" "$ORIGINAL"

    for class in "${CLASSES[@]}"; do
        echo "--- seed${seed} full_model: ${class} ---" | tee -a $LOG
        WANDB_MODE=offline $PYTHON train_cls.py \
            --dataset mvtec3d --class_name $class \
            --missing_type both --missing_rate 0.7 \
            --seed $seed --gpu-id 0 2>&1 | tee -a $LOG
    done

    if [ -f "$result_csv" ]; then
        lines=$(wc -l < "$result_csv")
        zeros=$(awk -F',' 'NR>1 && ($2==0||$3==0||$4==0)' "$result_csv")
        if [ "$lines" -ge 11 ] && [ -z "$zeros" ]; then
            cp "$result_csv" "$out_csv"
            mean=$(awk -F',' 'NR>1 {sum+=$2; n++} END {printf "%.2f", sum/n}' "$out_csv")
            echo "SAVED: seed${seed}/full_model mean I-AUROC = ${mean}%" | tee -a $LOG
        else
            echo "FAILED: seed${seed}/full_model" | tee -a $LOG
            cp "$BACKUP" "$ORIGINAL"
            exit 1
        fi
    fi
    cp "$BACKUP" "$ORIGINAL"
}

for seed in 111 222 333; do
    run_fullmodel $seed
done

echo "" | tee -a $LOG
echo "===== V3 FULL MODEL COMPLETE =====" | tee -a $LOG
for seed in 111 222 333; do
    echo -n "seed${seed}: "
    awk -F',' 'NR>1 {sum+=$2; n++} END {printf "%.2f%%\n", sum/n}' \
        ${RESULTS_DIR}/seed${seed}_full_model.csv 2>/dev/null || echo "missing"
done | tee -a $LOG
