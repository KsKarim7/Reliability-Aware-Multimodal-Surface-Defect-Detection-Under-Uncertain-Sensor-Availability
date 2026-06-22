#!/bin/bash
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/p3766/miniconda3/envs/misdd_mm/bin:/home/p3766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

PYTHON=/home/p3766/miniconda3/envs/misdd_mm/bin/python
ORIGINAL=/home/p3766/MISDD-MM/MISDD_MM/model.py
BACKUP=/home/p3766/MISDD-MM/MISDD_MM/model_full.py
RESULT_CSV=/home/p3766/MISDD-MM/result/mvtec3d/both/0.7/csv/Seed_333-results.csv
mkdir -p /home/p3766/MISDD-MM/ablation_results/seed333
cd /home/p3766/MISDD-MM

CLASSES=(bagel cable_gland carrot cookie dowel foam peach potato rope tire)

run_config() {
    local name=$1
    local model_src=$2
    local log_file=$3
    local out_csv="/home/p3766/MISDD-MM/ablation_results/seed333/${name}.csv"

    echo "========================================"
    echo "START seed=333: ${name}"
    echo "========================================"

    rm -f $RESULT_CSV
    cp $model_src $ORIGINAL
    echo "Model: $model_src"

    for class in "${CLASSES[@]}"; do
        echo "--- seed333 ${name}: ${class} ---" | tee -a $log_file
        WANDB_MODE=offline $PYTHON train_cls.py \
            --dataset mvtec3d --class_name $class \
            --missing_type both --missing_rate 0.7 \
            --seed 333 2>&1 | tee -a $log_file
    done

    if [ -f $RESULT_CSV ]; then
        line_count=$(wc -l < $RESULT_CSV)
        if [ "$line_count" -ge 11 ]; then
            cp $RESULT_CSV $out_csv
            echo "SAVED: seed333/${name}.csv" | tee -a $log_file
            cat $out_csv
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
    echo "COMPLETE: seed333/$name"
}

# innov1_only and innov2_only are DONE — resume from innov3_only
run_config "innov3_only" \
    "/home/p3766/MISDD-MM/ablation_models/model_innov3_only.py" \
    "/home/p3766/MISDD-MM/seed333_innov3.log"

run_config "innov4_only" \
    "/home/p3766/MISDD-MM/ablation_models/model_innov4_only.py" \
    "/home/p3766/MISDD-MM/seed333_innov4.log"

run_config "innov2_4" \
    "/home/p3766/MISDD-MM/ablation_models/model_innov2_4.py" \
    "/home/p3766/MISDD-MM/seed333_innov24.log"

run_config "full_model" \
    "/home/p3766/MISDD-MM/MISDD_MM/model_full.py" \
    "/home/p3766/MISDD-MM/seed333_full.log"

echo ""
echo "========================================"
echo "SEED 333 COMPLETE"
echo "========================================"
for name in innov1_only innov2_only innov3_only innov4_only innov2_4 full_model; do
    echo "--- seed333/$name ---"
    cat /home/p3766/MISDD-MM/ablation_results/seed333/${name}.csv 2>/dev/null || echo "NOT FOUND"
done
