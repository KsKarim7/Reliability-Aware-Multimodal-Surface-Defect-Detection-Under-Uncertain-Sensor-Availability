#!/bin/bash
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/p3766/miniconda3/envs/misdd_mm/bin:/home/p3766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
PYTHON=/home/p3766/miniconda3/envs/misdd_mm/bin/python
ORIGINAL=/home/p3766/MISDD-MM/MISDD_MM/model.py
BACKUP=/home/p3766/MISDD-MM/MISDD_MM/model_full.py
RESULT_CSV=/home/p3766/MISDD-MM/result/mvtec3d/both/0.7/csv/Seed_111-results.csv
LOG=/home/p3766/MISDD-MM/ablation.log
mkdir -p /home/p3766/MISDD-MM/ablation_results
cd /home/p3766/MISDD-MM

run_config() {
    local name=$1
    echo "=== START: $name ===" >> $LOG

    # CRITICAL: delete shared CSV before this config runs
    rm -f $RESULT_CSV

    cp /home/p3766/MISDD-MM/ablation_models/model_${name}.py $ORIGINAL

    for class in bagel cable_gland carrot cookie dowel foam peach potato rope tire; do
        echo "--- $name: $class ---" >> $LOG
        WANDB_MODE=offline $PYTHON train_cls.py \
            --dataset mvtec3d --class_name $class \
            --missing_type both --missing_rate 0.7 \
            --seed 111 >> $LOG 2>&1
    done

    # Verify CSV has all 10 classes before copying
    if [ -f $RESULT_CSV ]; then
        line_count=$(wc -l < $RESULT_CSV)
        if [ "$line_count" -ge 11 ]; then
            cp $RESULT_CSV /home/p3766/MISDD-MM/ablation_results/${name}.csv
            echo "=== DONE: $name — $line_count lines saved ===" >> $LOG
        else
            echo "=== FAILED: $name — only $line_count lines in CSV ===" >> $LOG
        fi
    else
        echo "=== FAILED: $name — no CSV produced ===" >> $LOG
    fi

    cp $BACKUP $ORIGINAL
}

run_config "innov2_only"
run_config "innov3_only"
run_config "innov4_only"
run_config "innov1_3"
run_config "innov1_4"

cp $BACKUP $ORIGINAL
echo "ALL ABLATIONS COMPLETE" >> $LOG
