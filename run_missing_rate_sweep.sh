#!/bin/bash
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/pub_766/miniconda3/envs/ramsdd/bin:/home/pub_766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

PYTHON=/home/pub_766/miniconda3/envs/ramsdd/bin/python
BACKUP=/home/pub_766/MISDD-MM/MISDD_MM/model_full.py
ORIGINAL=/home/pub_766/MISDD-MM/MISDD_MM/model.py
RESULTS_DIR=/home/pub_766/MISDD-MM/ablation_results_missing_rate
cd /home/pub_766/MISDD-MM

CLASSES=(bagel cable_gland carrot cookie dowel foam peach potato rope tire)
LOG=/home/pub_766/MISDD-MM/missing_rate_sweep.log

mkdir -p $RESULTS_DIR

run_rate() {
    local rate=$1
    local seed=$2
    local result_csv="/home/pub_766/MISDD-MM/result/mvtec3d/both/${rate}/csv/Seed_${seed}-results.csv"
    local out_csv="${RESULTS_DIR}/rate${rate}_seed${seed}_full_model.csv"

    if [ -f "$out_csv" ]; then
        lines=$(wc -l < "$out_csv")
        zeros=$(awk -F',' 'NR>1 && ($2==0)' "$out_csv")
        if [ "$lines" -ge 11 ] && [ -z "$zeros" ]; then
            echo "SKIP (complete): η=${rate} seed${seed}" | tee -a $LOG
            return 0
        fi
    fi

    echo "===== START η=${rate} seed=${seed} at $(date) =====" | tee -a $LOG
    rm -f "$result_csv"
    cp "$BACKUP" "$ORIGINAL"

    for class in "${CLASSES[@]}"; do
        echo "--- η=${rate} seed${seed}: ${class} ---" | tee -a $LOG
        WANDB_MODE=offline $PYTHON train_cls.py \
            --dataset mvtec3d --class_name $class \
            --missing_type both --missing_rate $rate \
            --seed $seed --gpu-id 0 2>&1 | tee -a $LOG
    done

    if [ -f "$result_csv" ]; then
        lines=$(wc -l < "$result_csv")
        zeros=$(awk -F',' 'NR>1 && ($2==0)' "$result_csv")
        if [ "$lines" -ge 11 ] && [ -z "$zeros" ]; then
            cp "$result_csv" "$out_csv"
            mean=$(awk -F',' 'NR>1 {sum+=$2; n++} END {printf "%.2f", sum/n}' "$out_csv")
            echo "SAVED: η=${rate} seed${seed} mean I-AUROC = ${mean}%" | tee -a $LOG
        else
            echo "FAILED: η=${rate} seed${seed}" | tee -a $LOG
            cp "$BACKUP" "$ORIGINAL"
            exit 1
        fi
    fi
    cp "$BACKUP" "$ORIGINAL"
}

for rate in 0.3 0.5 0.9; do
    for seed in 111 222 333; do
        run_rate $rate $seed
    done
done

echo "" | tee -a $LOG
echo "===== MISSING RATE SWEEP COMPLETE =====" | tee -a $LOG
echo "" | tee -a $LOG
echo "Results summary:" | tee -a $LOG
for rate in 0.3 0.5 0.7 0.9; do
    echo -n "η=${rate}: " | tee -a $LOG
    if [ "$rate" = "0.7" ]; then
        echo "76.71% (V3, already done)" | tee -a $LOG
    else
        vals=()
        for seed in 111 222 333; do
            csv="${RESULTS_DIR}/rate${rate}_seed${seed}_full_model.csv"
            val=$(awk -F',' 'NR>1 {sum+=$2; n++} END {printf "%.2f", sum/n}' "$csv" 2>/dev/null)
            vals+=($val)
        done
        mean=$(python3 -c "v=[${vals[*]}]; print(f'{sum(v)/len(v):.2f}')" 2>/dev/null)
        echo "seed111=${vals[0]}% seed222=${vals[1]}% seed333=${vals[2]}% mean=${mean}%" | tee -a $LOG
    fi
done
