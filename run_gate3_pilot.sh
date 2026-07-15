#!/bin/bash
# Gate 3 pilot: synthetic-anomaly contrastive prompt objective.
# 4 classes x {baseline, full_model}, seed 111, final protocol (map + 3-NN) +
# --syn_anomaly. Results go to result_diag (campaign artifacts untouched).
# Reference numbers: ablation_results_v5_3nn/seed111/{baseline,full_model}.csv
# Success criterion (pre-registered in GATE3_PILOT_DESIGN.md): mean > +0.5pp over
# reference, no class regressing > 1pp.

export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/pub_766/miniconda3/envs/ramsdd/bin:/home/pub_766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

PYTHON=/home/pub_766/miniconda3/envs/ramsdd/bin/python
ORIGINAL=/home/pub_766/MISDD-MM/MISDD_MM/model.py
BACKUP=/home/pub_766/MISDD-MM/MISDD_MM/model_full.py
cd /home/pub_766/MISDD-MM
trap 'cp "$BACKUP" "$ORIGINAL"' EXIT

LOG=/home/pub_766/MISDD-MM/gate3_pilot.log
echo "Gate 3 pilot starting at $(date)" | tee -a $LOG

CLASSES=(peach cookie bagel dowel)

for cfg in baseline full_model; do
    if [ "$cfg" = "baseline" ]; then
        cp /home/pub_766/MISDD-MM/ablation_models/model_baseline.py "$ORIGINAL"
    else
        cp "$BACKUP" "$ORIGINAL"
    fi
    rm -rf /home/pub_766/MISDD-MM/MISDD_MM/__pycache__
    for class in "${CLASSES[@]}"; do
        echo "--- gate3 ${cfg}: ${class} ---" | tee -a $LOG
        WANDB_MODE=offline $PYTHON train_cls.py \
            --dataset mvtec3d --class_name $class \
            --missing_type both --missing_rate 0.7 \
            --seed 111 --gpu-id 0 \
            --batch-size 16 --max_norm 1.0 --Epoch 25 \
            --img_score_mode map --map_knn 3 \
            --syn_anomaly true --syn_weight 0.2 \
            --root-dir ./result_diag_gate3_${cfg} 2>&1 | tee -a $LOG
    done
    cp "$BACKUP" "$ORIGINAL"
done
echo "Gate 3 pilot COMPLETE at $(date)" | tee -a $LOG
