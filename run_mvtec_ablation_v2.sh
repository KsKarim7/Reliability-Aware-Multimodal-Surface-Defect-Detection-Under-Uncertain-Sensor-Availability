#!/bin/bash
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/pub_766/miniconda3/envs/ramsdd/bin:/home/pub_766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

PYTHON=/home/pub_766/miniconda3/envs/ramsdd/bin/python
ORIGINAL=/home/pub_766/MISDD-MM/MISDD_MM/model.py
BACKUP=/home/pub_766/MISDD-MM/MISDD_MM/model_full.py
cd /home/pub_766/MISDD-MM

CLASSES=(bagel cable_gland carrot cookie dowel foam peach potato rope tire)

CONFIGS=(
    "innov1_only:/home/pub_766/MISDD-MM/ablation_models/model_innov1_only.py"
    "innov2_only:/home/pub_766/MISDD-MM/ablation_models/model_innov2_only.py"
    "innov3_only:/home/pub_766/MISDD-MM/ablation_models/model_innov3_only.py"
    "innov4_only:/home/pub_766/MISDD-MM/ablation_models/model_innov4_only.py"
    "full_model:/home/pub_766/MISDD-MM/MISDD_MM/model_full.py"
)

run_config() {
    local seed=$1
    local name=$2
    local model_src=$3
    local log_file=$4
    local result_csv="/home/pub_766/MISDD-MM/result/mvtec3d/both/0.7/csv/Seed_${seed}-results.csv"
    local out_dir="/home/pub_766/MISDD-MM/ablation_results_v2/seed${seed}"
    local out_csv="${out_dir}/${name}.csv"

    mkdir -p "$out_dir"
    echo "========================================"
    echo "START v2 seed=${seed}: ${name}"
    echo "========================================"

    rm -f "$result_csv"
    cp "$model_src" "$ORIGINAL"

    for class in "${CLASSES[@]}"; do
        echo "--- v2 seed${seed} ${name}: ${class} ---" | tee -a "$log_file"
        WANDB_MODE=offline $PYTHON train_cls.py \
            --dataset mvtec3d --class_name $class \
            --missing_type both --missing_rate 0.7 \
            --seed $seed --gpu-id 0 2>&1 | tee -a "$log_file"

        if [ -f "$result_csv" ]; then
            zeros=$(awk -F',' -v cls="mvtec3d-${class}" '$1==cls && ($2==0||$3==0||$4==0)' "$result_csv")
            if [ -n "$zeros" ]; then
                echo "WARNING: Zero detected for ${class} in ${name} seed${seed}" | tee -a "$log_file"
            fi
        fi
    done

    if [ -f "$result_csv" ]; then
        line_count=$(wc -l < "$result_csv")
        if [ "$line_count" -ge 11 ]; then
            cp "$result_csv" "$out_csv"
            echo "SAVED: ablation_results_v2/seed${seed}/${name}.csv" | tee -a "$log_file"
            cat "$out_csv"
        else
            echo "FAILED: only $line_count lines" | tee -a "$log_file"
            cp "$BACKUP" "$ORIGINAL"
            exit 1
        fi
    else
        echo "FAILED: no CSV produced" | tee -a "$log_file"
        cp "$BACKUP" "$ORIGINAL"
        exit 1
    fi

    cp "$BACKUP" "$ORIGINAL"
    echo "COMPLETE: v2 seed${seed}/${name}"
    echo "========================================"
}

for seed in 111 222 333; do
    echo "### SEED $seed ###"
    for entry in "${CONFIGS[@]}"; do
        name="${entry%%:*}"
        model="${entry##*:}"
        run_config $seed "$name" "$model" \
            "/home/pub_766/MISDD-MM/ablation_v2_seed${seed}_${name}.log"
    done
done

echo ""
echo "########################################"
echo "ALL V2 ABLATIONS COMPLETE"
echo "########################################"
for seed in 111 222 333; do
    echo "=== SEED $seed ==="
    for entry in "${CONFIGS[@]}"; do
        name="${entry%%:*}"
        echo "--- seed${seed}/${name} ---"
        cat /home/pub_766/MISDD-MM/ablation_results_v2/seed${seed}/${name}.csv 2>/dev/null || echo "NOT FOUND"
    done
done
