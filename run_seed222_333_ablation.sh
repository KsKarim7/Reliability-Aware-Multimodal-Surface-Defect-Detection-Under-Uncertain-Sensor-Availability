#!/bin/bash
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/p3766/miniconda3/envs/misdd_mm/bin:/home/p3766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

PYTHON=/home/p3766/miniconda3/envs/misdd_mm/bin/python
ORIGINAL=/home/p3766/MISDD-MM/MISDD_MM/model.py
BACKUP=/home/p3766/MISDD-MM/MISDD_MM/model_full.py
mkdir -p /home/p3766/MISDD-MM/ablation_results/seed222
mkdir -p /home/p3766/MISDD-MM/ablation_results/seed333
cd /home/p3766/MISDD-MM

CLASSES=(bagel cable_gland carrot cookie dowel foam peach potato rope tire)

run_config() {
    local seed=$1
    local name=$2
    local model_src=$3
    local log_file=$4
    local result_csv="/home/p3766/MISDD-MM/result/mvtec3d/both/0.7/csv/Seed_${seed}-results.csv"
    local out_csv="/home/p3766/MISDD-MM/ablation_results/seed${seed}/${name}.csv"

    echo "========================================"
    echo "START seed=${seed}: ${name}"
    echo "========================================"

    rm -f $result_csv
    cp $model_src $ORIGINAL
    echo "Model: $model_src"

    for class in "${CLASSES[@]}"; do
        echo "--- seed${seed} ${name}: ${class} ---" | tee -a $log_file
        WANDB_MODE=offline $PYTHON train_cls.py \
            --dataset mvtec3d --class_name $class \
            --missing_type both --missing_rate 0.7 \
            --seed $seed 2>&1 | tee -a $log_file
    done

    if [ -f $result_csv ]; then
        line_count=$(wc -l < $result_csv)
        if [ "$line_count" -ge 11 ]; then
            cp $result_csv $out_csv
            echo "SAVED: seed${seed}/${name}.csv ($line_count lines)" | tee -a $log_file
            cat $out_csv
        else
            echo "FAILED: seed${seed}/${name} only $line_count lines" | tee -a $log_file
            cp $BACKUP $ORIGINAL
            exit 1
        fi
    else
        echo "FAILED: seed${seed}/${name} no CSV produced" | tee -a $log_file
        cp $BACKUP $ORIGINAL
        exit 1
    fi

    cp $BACKUP $ORIGINAL
    echo "COMPLETE: seed${seed}/${name}"
    echo "========================================"
}

CONFIGS=(
    "innov1_only:/home/p3766/MISDD-MM/ablation_models/model_innov1_only.py"
    "innov2_only:/home/p3766/MISDD-MM/ablation_models/model_innov2_only.py"
    "innov3_only:/home/p3766/MISDD-MM/ablation_models/model_innov3_only.py"
    "innov4_only:/home/p3766/MISDD-MM/ablation_models/model_innov4_only.py"
    "innov2_4:/home/p3766/MISDD-MM/ablation_models/model_innov2_4.py"
    "full_model:/home/p3766/MISDD-MM/MISDD_MM/model_full.py"
)

# ── SEED 222 ─────────────────────────────────────────────────────
echo ""
echo "###################################################"
echo "# STARTING SEED 222"
echo "###################################################"
for entry in "${CONFIGS[@]}"; do
    name="${entry%%:*}"
    model="${entry##*:}"
    run_config 222 "$name" "$model" "/home/p3766/MISDD-MM/seed222_${name}.log"
done

echo ""
echo "###################################################"
echo "# SEED 222 COMPLETE — STARTING SEED 333"
echo "###################################################"

# ── SEED 333 ─────────────────────────────────────────────────────
for entry in "${CONFIGS[@]}"; do
    name="${entry%%:*}"
    model="${entry##*:}"
    run_config 333 "$name" "$model" "/home/p3766/MISDD-MM/seed333_${name}.log"
done

echo ""
echo "###################################################"
echo "# ALL SEEDS COMPLETE"
echo "###################################################"

echo ""
echo "=== SEED 222 RESULTS ==="
for name in innov1_only innov2_only innov3_only innov4_only innov2_4 full_model; do
    echo "--- seed222/$name ---"
    cat /home/p3766/MISDD-MM/ablation_results/seed222/${name}.csv 2>/dev/null || echo "NOT FOUND"
done

echo ""
echo "=== SEED 333 RESULTS ==="
for name in innov1_only innov2_only innov3_only innov4_only innov2_4 full_model; do
    echo "--- seed333/$name ---"
    cat /home/p3766/MISDD-MM/ablation_results/seed333/${name}.csv 2>/dev/null || echo "NOT FOUND"
done
