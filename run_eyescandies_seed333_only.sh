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

run_config() {
    local seed=$1
    local name=$2
    local model_src=$3
    local log_file=$4
    local result_csv="/home/pub_766/MISDD-MM/result/eyescandies/both/0.7/csv/Seed_${seed}-results.csv"
    local out_dir="/home/pub_766/MISDD-MM/ablation_results/eyescandies_seed${seed}"
    local out_csv="${out_dir}/${name}.csv"

    mkdir -p "$out_dir"

    echo "========================================"
    echo "START eyescandies seed=${seed}: ${name}"
    echo "========================================"

    rm -f "$result_csv"
    cp "$model_src" "$ORIGINAL"
    echo "Model: $model_src"

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
            echo "SAVED: eyescandies_seed${seed}/${name}.csv ($line_count lines)" | tee -a "$log_file"
            cat "$out_csv"
        else
            echo "FAILED: eyescandies seed${seed}/${name} only $line_count lines" | tee -a "$log_file"
            cp "$BACKUP" "$ORIGINAL"
            exit 1
        fi
    else
        echo "FAILED: eyescandies seed${seed}/${name} no CSV produced" | tee -a "$log_file"
        cp "$BACKUP" "$ORIGINAL"
        exit 1
    fi

    cp "$BACKUP" "$ORIGINAL"
    echo "COMPLETE: eyescandies seed${seed}/${name}"
    echo "========================================"
}

CONFIGS=(
    "innov1_only:/home/pub_766/MISDD-MM/ablation_models/model_innov1_only.py"
    "innov2_only:/home/pub_766/MISDD-MM/ablation_models/model_innov2_only.py"
    "innov3_only:/home/pub_766/MISDD-MM/ablation_models/model_innov3_only.py"
    "innov4_only:/home/pub_766/MISDD-MM/ablation_models/model_innov4_only.py"
    "full_model:/home/pub_766/MISDD-MM/MISDD_MM/model_full.py"
)

# Copy existing seed-111 results into seed111 folder for consistency
mkdir -p /home/pub_766/MISDD-MM/ablation_results/eyescandies_seed111
for name in innov1_only innov2_only innov3_only innov4_only full_model; do
    cp /home/pub_766/MISDD-MM/ablation_results/eyescandies_${name}.csv \
       /home/pub_766/MISDD-MM/ablation_results/eyescandies_seed111/${name}.csv 2>/dev/null
done


# ── SEED 333 — 5 configs ─────────────────────────────────────────
echo "### SEED 333 — 5 configs ###"
for entry in "${CONFIGS[@]}"; do
    name="${entry%%:*}"
    model="${entry##*:}"
    run_config 333 "$name" "$model" "/home/pub_766/MISDD-MM/eyescandies_seed333_${name}.log"
done

echo ""
echo "########################################"
echo "ALL EYESCANDIES MULTI-SEED RUNS COMPLETE"
echo "########################################"
for seed in 111 222 333; do
    echo ""
    echo "=== SEED $seed ==="
    for name in innov1_only innov2_only innov3_only innov4_only full_model; do
        echo "--- eyescandies_seed${seed}/$name ---"
        cat /home/pub_766/MISDD-MM/ablation_results/eyescandies_seed${seed}/${name}.csv 2>/dev/null || echo "NOT FOUND"
    done
done
