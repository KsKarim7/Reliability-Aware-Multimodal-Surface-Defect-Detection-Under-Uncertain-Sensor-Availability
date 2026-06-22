#!/bin/bash
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/p3766/miniconda3/envs/misdd_mm/bin:/home/p3766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

PYTHON=/home/p3766/miniconda3/envs/misdd_mm/bin/python
ORIGINAL=/home/p3766/MISDD-MM/MISDD_MM/model.py
BACKUP=/home/p3766/MISDD-MM/MISDD_MM/model_full.py
RESULT_CSV=/home/p3766/MISDD-MM/result/mvtec3d/both/0.7/csv/Seed_222-results.csv
mkdir -p /home/p3766/MISDD-MM/ablation_results/seed222
cd /home/p3766/MISDD-MM

CLASSES=(bagel cable_gland carrot cookie dowel foam peach potato rope tire)

run_config() {
    local name=$1
    local model_src=$2
    local log_file=$3

    echo "========================================"
    echo "START seed=222: $name"
    echo "========================================"

    rm -f $RESULT_CSV
    cp $model_src $ORIGINAL
    echo "Model set: $model_src"

    for class in "${CLASSES[@]}"; do
        echo "--- seed222 $name: $class ---" | tee -a $log_file
        WANDB_MODE=offline $PYTHON train_cls.py \
            --dataset mvtec3d --class_name $class \
            --missing_type both --missing_rate 0.7 \
            --seed 222 2>&1 | tee -a $log_file
    done

    if [ -f $RESULT_CSV ]; then
        line_count=$(wc -l < $RESULT_CSV)
        if [ "$line_count" -ge 11 ]; then
            cp $RESULT_CSV /home/p3766/MISDD-MM/ablation_results/seed222/${name}.csv
            echo "SAVED: seed222/${name}.csv ($line_count lines)" | tee -a $log_file
            cat /home/p3766/MISDD-MM/ablation_results/seed222/${name}.csv
        else
            echo "FAILED: $name only $line_count lines" | tee -a $log_file
            cp $BACKUP $ORIGINAL
            exit 1
        fi
    else
        echo "FAILED: $name no CSV" | tee -a $log_file
        cp $BACKUP $ORIGINAL
        exit 1
    fi

    cp $BACKUP $ORIGINAL
    echo "COMPLETE: seed222/$name"
}

run_config "innov1_only" \
    "/home/p3766/MISDD-MM/ablation_models/model_innov1_only.py" \
    "/home/p3766/MISDD-MM/seed222_innov1.log"

run_config "innov2_only" \
    "/home/p3766/MISDD-MM/ablation_models/model_innov2_only.py" \
    "/home/p3766/MISDD-MM/seed222_innov2.log"

run_config "innov3_only" \
    "/home/p3766/MISDD-MM/ablation_models/model_innov3_only.py" \
    "/home/p3766/MISDD-MM/seed222_innov3.log"

run_config "innov4_only" \
    "/home/p3766/MISDD-MM/ablation_models/model_innov4_only.py" \
    "/home/p3766/MISDD-MM/seed222_innov4.log"

run_config "innov2_4" \
    "/home/p3766/MISDD-MM/ablation_models/model_innov2_4.py" \
    "/home/p3766/MISDD-MM/seed222_innov24.log"

run_config "full_model" \
    "/home/p3766/MISDD-MM/MISDD_MM/model_full.py" \
    "/home/p3766/MISDD-MM/seed222_full.log"

echo ""
echo "========================================"
echo "ALL SEED 222 CONFIGS COMPLETE"
echo "========================================"
for name in innov1_only innov2_only innov3_only innov4_only innov2_4 full_model; do
    echo ""
    echo "--- seed222/$name ---"
    cat /home/p3766/MISDD-MM/ablation_results/seed222/${name}.csv 2>/dev/null || echo "NOT FOUND"
done
