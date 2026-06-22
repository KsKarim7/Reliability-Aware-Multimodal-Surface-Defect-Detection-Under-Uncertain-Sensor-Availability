#!/bin/bash
LOG="/home/p3766/MISDD-MM/eyescandies_training.log"
echo "=== EYESCANDIES TRAINING STARTED: $(date) ===" >> $LOG

source /home/p3766/miniconda3/etc/profile.d/conda.sh
conda activate misdd_mm
cd /home/p3766/MISDD-MM

export WANDB_MODE=offline
export TORCH_CUDA_ARCH_LIST="8.6"

cp /home/p3766/MISDD-MM/MISDD_MM/model_full.py /home/p3766/MISDD-MM/MISDD_MM/model.py
rm -f /home/p3766/MISDD-MM/result/eyescandies/both/0.7/csv/Seed_111-results.csv

for class in CandyCane ChocolateCookie ChocolatePraline Confetto GummyBear HazelnutTruffle LicoriceSandwich Lollipop Marshmallow PeppermintCandy; do
    echo "--- Training: $class --- $(date)" >> $LOG
    /home/p3766/miniconda3/envs/misdd_mm/bin/python -u train_cls.py \
        --dataset eyescandies \
        --class_name $class \
        --missing_type both \
        --missing_rate 0.7 \
        --seed 111 >> $LOG 2>&1
    echo "--- DONE: $class --- $(date)" >> $LOG
done

cp /home/p3766/MISDD-MM/result/eyescandies/both/0.7/csv/Seed_111-results.csv \
   /home/p3766/MISDD-MM/ablation_results/eyescandies_full.csv

echo "=== EYESCANDIES TRAINING COMPLETE: $(date) ===" >> $LOG
