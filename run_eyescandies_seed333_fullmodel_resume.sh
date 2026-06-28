#!/bin/bash
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:/home/pub_766/miniconda3/envs/misdd_mm/bin:/home/pub_766/miniconda3/bin
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

PYTHON=/home/pub_766/miniconda3/envs/misdd_mm/bin/python
ORIGINAL=/home/pub_766/MISDD-MM/MISDD_MM/model.py
BACKUP=/home/pub_766/MISDD-MM/MISDD_MM/model_full.py
cd /home/pub_766/MISDD-MM

CLASSES=(
    CandyCane ChocolateCookie ChocolatePraline Confetto GummyBear
    HazelnutTruffle LicoriceSandwich Lollipop Marshmallow PeppermintCandy
)

seed=333
name="full_model"
model_src="/home/pub_766/MISDD-MM/MISDD_MM/model_full.py"
log_file="/home/pub_766/MISDD-MM/eyescandies_seed333_full_resume2.log"
result_csv="/home/pub_766/MISDD-MM/result/eyescandies/both/0.7/csv/Seed_${seed}-results.csv"
out_dir="/home/pub_766/MISDD-MM/ablation_results/eyescandies_seed${seed}"
out_csv="${out_dir}/${name}.csv"

mkdir -p "$out_dir"
rm -f "$result_csv"
cp "$model_src" "$ORIGINAL"
echo "START eyescandies seed=${seed}: ${name}"

for class in "${CLASSES[@]}"; do
    echo "--- eyescandies seed${seed} ${name}: ${class} ---" | tee -a "$log_file"
    WANDB_MODE=offline $PYTHON train_cls.py \
        --dataset eyescandies --class_name $class \
        --missing_type both --missing_rate 0.7 \
        --seed $seed --gpu-id 0 2>&1 | tee -a "$log_file"
done

if [ -f "$result_csv" ]; then
    line_count=$(wc -l < "$result_csv")
    if [ "$line_count" -ge 11 ]; then
        cp "$result_csv" "$out_csv"
        echo "SAVED: eyescandies_seed${seed}/${name}.csv" | tee -a "$log_file"
        cat "$out_csv"
    else
        echo "FAILED: only $line_count lines" | tee -a "$log_file"
    fi
fi
cp "$BACKUP" "$ORIGINAL"
echo "COMPLETE: eyescandies seed${seed}/${name}"
