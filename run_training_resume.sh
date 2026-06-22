#!/bin/bash
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/p3766/miniconda3/envs/misdd_mm/bin:/home/p3766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
source /home/p3766/miniconda3/etc/profile.d/conda.sh
source activate misdd_mm
PYTHON=/home/p3766/miniconda3/envs/misdd_mm/bin/python
cd /home/p3766/MISDD-MM

# seed=333 rate=0.7 — bagel done, start from cable_gland
for class in cable_gland carrot cookie dowel foam peach potato rope tire; do
    rm -rf /home/p3766/MISDD-MM/wandb/*
    echo "=== seed=333 rate=0.7 class=$class ===" >> /home/p3766/MISDD-MM/seed_sweep.log
    WANDB_MODE=offline $PYTHON train_cls.py \
        --dataset mvtec3d --class_name $class \
        --missing_type both --missing_rate 0.7 \
        --seed 333 >> /home/p3766/MISDD-MM/seed_sweep.log 2>&1
    echo "=== Done: $class ===" >> /home/p3766/MISDD-MM/seed_sweep.log
    rm -rf /home/p3766/MISDD-MM/wandb/*
done
echo "=== DONE seed=333 rate=0.7 ===" >> /home/p3766/MISDD-MM/seed_sweep.log

# seed=333 rate=0.9 — all classes
for class in bagel cable_gland carrot cookie dowel foam peach potato rope tire; do
    rm -rf /home/p3766/MISDD-MM/wandb/*
    echo "=== seed=333 rate=0.9 class=$class ===" >> /home/p3766/MISDD-MM/seed_sweep.log
    WANDB_MODE=offline $PYTHON train_cls.py \
        --dataset mvtec3d --class_name $class \
        --missing_type both --missing_rate 0.9 \
        --seed 333 >> /home/p3766/MISDD-MM/seed_sweep.log 2>&1
    echo "=== Done: $class ===" >> /home/p3766/MISDD-MM/seed_sweep.log
    rm -rf /home/p3766/MISDD-MM/wandb/*
done
echo "=== DONE seed=333 rate=0.9 ===" >> /home/p3766/MISDD-MM/seed_sweep.log
echo "ALL DONE" >> /home/p3766/MISDD-MM/seed_sweep.log
