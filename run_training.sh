#!/bin/bash
LOCKFILE=/home/p3766/MISDD-MM/ablation.lock

# If already running or lock exists, exit
if pgrep -f train_cls.py > /dev/null; then
    exit 0
fi
if [ -f $LOCKFILE ]; then
    exit 0
fi

# Create lock file
touch $LOCKFILE

export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/p3766/miniconda3/envs/misdd_mm/bin:/home/p3766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
cd /home/p3766/MISDD-MM

nohup bash /home/p3766/MISDD-MM/run_ablation.sh >> /home/p3766/MISDD-MM/ablation.log 2>&1

# Remove lock when done
rm -f $LOCKFILE
