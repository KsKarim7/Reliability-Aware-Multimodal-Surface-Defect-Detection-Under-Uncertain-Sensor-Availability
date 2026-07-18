#!/bin/bash
# Reproduce the V4 full_model seed-333 checkpoints (all 10 classes) on the exact
# V4 code (worktree @ 43d6615) for the 10-class LayerNorm-inflation table.
# The originals were overwritten by the v5k campaign; the pipeline is
# deterministic (bit-exact replication previously demonstrated), and peach+cookie
# must reproduce the original Gate-1c receipts as the integrity check.
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/pub_766/miniconda3/envs/ramsdd/bin:/home/pub_766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
PYTHON=/home/pub_766/miniconda3/envs/ramsdd/bin/python
cd /home/pub_766/v4_repro
LOG=/home/pub_766/MISDD-MM/v4_repro.log
echo "V4 checkpoint reproduction starting at $(date)" | tee -a $LOG
# tagged code: v4-campaign-code (29251d7) = 43d6615 + accumulation loop,
# validated against the Gate-1c receipts (cookie+peach, <=2.4e-4 relative)
git -C /home/pub_766/v4_repro log --oneline -1 | tee -a $LOG
for class in bagel cable_gland carrot cookie dowel foam peach potato rope tire; do
    ck=/home/pub_766/v4_repro/result_run3/mvtec3d/both/0.7/checkpoint/CLS-Seed_333-${class}-check_point.pt
    if [ -f "$ck" ]; then
        echo "SKIP ${class} (checkpoint exists)" | tee -a $LOG
        continue
    fi
    echo "--- v4-repro: ${class} ---" | tee -a $LOG
    WANDB_MODE=offline $PYTHON train_cls.py \
        --dataset mvtec3d --class_name $class \
        --missing_type both --missing_rate 0.7 \
        --seed 333 --gpu-id 0 \
        --batch-size 32 --max_norm 1.0 --Epoch 25 \
        --root-dir ./result_run3 2>&1 | tee -a $LOG
done
echo "V4 checkpoint reproduction COMPLETE at $(date)" | tee -a $LOG
