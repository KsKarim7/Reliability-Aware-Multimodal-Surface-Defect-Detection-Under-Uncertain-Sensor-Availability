# CLAUDE.md — MISDD-MM Extended Project Context

This file gives Claude full context to continue working on this research project without re-explanation. Read this entirely before responding to any request.

---

## Project Identity

**Project:** Reliability-Aware Multimodal Surface Defect Detection Under Uncertain Sensor Availability (MISDD-MM Extended)
**Student:** Aalavi Mahin Khan (KsKarim7 / Aalavikhan), BRAC University BSc
**Supervisor:** F. H. Orodi
**Target:** IEEE journal publication (extended from thesis)
**GitHub:** https://github.com/KsKarim7/Reliability-Aware-Multimodal-Surface-Defect-Detection-Under-Uncertain-Sensor-Availability (branch: main)

---

## Machine / Environment

- **OS:** Windows 11, WSL2 Ubuntu 22.04, user `pub_766`
- **GPU:** RTX 4090, 24GB VRAM
- **CUDA:** 11.8, `TORCH_CUDA_ARCH_LIST="8.9"`
- **Conda env:** `ramsdd` (Python 3.11, PyTorch 2.2.0+cu118)
- **Project root:** `~/MISDD-MM/`
- **Datasets:** `~/mvtec3d/` (10 categories), `~/eyescandies/Eyecandies/` (10 categories)
- **Critical constraints:** numpy<2, never reinstall torch/numpy/rebuild pointnet2_ops, always `conda activate ramsdd`, `WANDB_MODE=offline`
- **Key paths:**
  - Main model: `MISDD_MM/model.py` (gets swapped with ablation models during ablation runs)
  - Full model: `MISDD_MM/model_full.py` (the backup/source of truth)
  - Training script: `train_cls.py`
  - Ablation models: `ablation_models/model_innov{1,2,3,4}_only.py`
- **Systemd:** Enabled via `/etc/wsl.conf` `[boot] systemd=true`. Linger enabled for pub_766. Service: `~/.config/systemd/user/misdd-training.service`. Launch script: `~/MISDD-MM/launch_segment.sh` reads `~/MISDD-MM/.current_segment` to dispatch correct training script.
- **CRITICAL:** Windows sign-out kills WSL2 entirely regardless of linger. Lock screen (Win+L) instead. Never sign out during training.

---

## Architecture Overview

Frozen CLIP ViT-B-16-plus-240 backbone with prompt tuning. Four innovations inject learned residuals into a shared prompt parameter set inside `Missing_PromptLearner`.

**Four Innovations:**

1. **CorrelatedPromptMLP (Innovation 1):** Reads previous layer's prompt output and adds a correlation residual at each of layers 1–5. Creates a J-1 layer backprop chain. Gradient path: small magnitude (reads from near-zero prompt vectors).

2. **DynamicPromptGenerator (Innovation 2):** Reads raw image features and adds a dynamic residual at layer 0. Single direct hop. Gradient path: LARGER than Innovation 1 (reads from raw image pixels — large activations).

3. **GranularTextGuidance (Innovation 3):** Computes auxiliary loss `L_gran` touching ALL layers' prompts simultaneously. Broad gradient footprint.

4. **SensorSentinel (Innovation 4):** Computes quality weights `w_rgb`, `w_dep` via Laplacian variance and depth density — **fully deterministic, no learned parameters, fully detached via `.item()` calls**. Contributes no gradient of its own but scales which prompt parameters receive how much gradient.

**Shared parameters:** `image_prompt_complete`, `image_prompt_missing`, `depth_prompt_complete`, `depth_prompt_missing`, `common_prompt_complete` — all [12, 896] tensors in `Missing_PromptLearner`.

---

## The Core Problem (Gradient Interference)

When all four innovations are active simultaneously in training, their gradient signals all converge on the same shared prompt parameters. This caused seed-dependent instability where `full_model` occasionally underperformed individual innovations.

**Measured evidence:** Gradient diagnostic CSVs show Innovation 2 gradient norm is consistently 1.4–2× LARGER than Innovation 1 (ratio 0.48–0.70x innov1/innov2). Innovation 2 dominates because it reads from raw image features (large activations), while Innovation 1 reads from near-zero prompt vectors. Combined with Innovation 3's broad auxiliary loss active at full strength from epoch 0, this created seed-dependent compositional instability.

**Seed111 specificity:** Seed111 starts with a more balanced innov1/innov2 ratio (0.70x) vs seeds 222/333 (0.64–0.65x), which is why seed111 was best in V1 but most resistant to the V2/V3 fixes.

---

## Three Fixes Applied

**Fix 1 — L_gran Linear Warmup:** `gran_weight = args.gran_weight * (epoch+1) / args.Epoch`. Starts L_gran at nearly zero, ramps to full strength by epoch 49. Lets main SCL loss stabilize before Innovation 3 adds pressure. Added `--gran_weight` arg (default 0.1).

**Fix 2 — Config-Aware Gradient Clipping:**
```python
_clip_norm = args.max_norm * 2.0 if _is_full_model else args.max_norm
torch.nn.utils.clip_grad_norm_(model.missing_prompt_learner.parameters(), max_norm=_clip_norm)
```
Justification: with 3 active gradient sources (Innovation 4 is detached), combined norm scales as sqrt(3)≈1.73×, so 2.0 is a principled upper bound. Added `--max_norm` arg (default 1.0), giving full_model effective max_norm=2.0.

**Fix 3 — Dead Network Removal:** `rgb_corrector` and `depth_corrector` in SensorSentinel were instantiated but never called. Removed from all model files to clean optimizer state and gradient norm calculations.

**Gradient diagnostic:** `_compute_grad_diagnostics()` function defined before `fit()` in `train_cls.py`. Hooks registered on `all_prompts_image[0]` and `[1]` — NOTE: cosine similarity remains nan (hooks didn't fire on those tensors), but per-innovation private param norms ARE captured correctly. Logs to CSV at `check_path.replace('.pt', f'_grad_diag_seed{seed}.csv')`.

---

## Complete Results

### V1 — Original (no fixes), η=0.7, both-missing, MVTec 3D-AD

| Config | Seed111 | Seed222 | Seed333 | Mean |
|---|---:|---:|---:|---:|
| innov1_only | 77.32 | 77.66 | 76.44 | 77.14 |
| innov2_only | 77.50 | 77.40 | 76.65 | 77.18 |
| innov3_only | 77.26 | 77.78 | 76.61 | 77.22 |
| innov4_only | 77.97 | 76.97 | 76.38 | 77.11 |
| **full_model** | **78.44** | **77.49** | **76.47** | **77.47** |
| baseline | 73.83 | 73.83 | 73.83 | 73.83 |

Full vs indiv mean: seed111 +0.95pp ✅, seed222 -0.29pp ❌, seed333 -0.22pp ❌

### V2 — clip=1.0 all configs, η=0.7

| Config | Seed111 | Seed222 | Seed333 | Mean |
|---|---:|---:|---:|---:|
| innov1_only | 75.60 | 78.12 | 75.21 | 76.31 |
| innov2_only | 75.59 | 78.26 | 75.74 | 76.53 |
| innov3_only | 75.57 | 78.11 | 75.18 | 76.29 |
| innov4_only | 75.55 | 78.31 | 75.47 | 76.44 |
| **full_model** | **75.18** | **78.23** | **75.47** | **76.29** |

Full vs indiv mean: seed111 -0.40pp ❌, seed222 +0.03pp ✅, seed333 +0.07pp ✅
Note: individual std dropped from 0.28pp (V1) to 0.02pp (V2) — dramatically more stable.

### V3 — clip=2.0 for full_model only, clip=1.0 for individuals, η=0.7 ← PRIMARY CORRECTED ABLATION

| Config | Seed111 | Seed222 | Seed333 | Mean |
|---|---:|---:|---:|---:|
| individuals (from V2) | ~75.58 mean | ~78.20 mean | ~75.40 mean | 76.39 |
| **full_model** | **75.89** | **78.62** | **75.61** | **76.71** |

Full vs indiv mean: seed111 +0.31pp ✅, seed222 +0.42pp ✅, seed333 +0.21pp ✅
**First version where full_model beats individual mean in ALL 3 seeds.**
V3 mean (76.71%) is 0.76pp below V1 (77.47%) — performance/consistency tradeoff.

### Paper Strategy
- **V1 = primary reported results** (strongest absolute performance, +3.64pp over baseline)
- **V3 = corrected ablation** (proves consistent composition, cited in analysis/discussion)
- Story: "identified interference → principled fixes → demonstrated resolution with modest tradeoff"

### Baseline (published, both-missing)
| η | I-AUROC | P-AUROC | AUPRO |
|---|---:|---:|---:|
| 0.3 | 77.71 | 95.00 | 84.03 |
| 0.5 | 76.95 | 93.28 | 79.79 |
| 0.7 | 73.83 | 93.05 | 77.44 |

η=0.9 not published. Your reproduced baseline at η=0.7 matches exactly.

### Eyescandies (V1 only, full_model seed111)
Mean I-AUROC: 74.15%

---

## File / Directory Structure

```
~/MISDD-MM/
├── MISDD_MM/
│   ├── model.py                  # active model (swapped during ablation)
│   ├── model_full.py             # full model with all 4 innovations (BACKUP)
│   ├── model_full.py.bak_pregradfix
├── ablation_models/
│   ├── model_innov1_only.py
│   ├── model_innov2_only.py
│   ├── model_innov3_only.py
│   ├── model_innov4_only.py
├── train_cls.py                  # training loop with all fixes
├── train_cls.py.bak_pregradfix
├── ablation_results/             # V1 results (original)
├── ablation_results_v2/          # V2 results (clip=1.0)
│   ├── seed111/{innov1-4_only,full_model}.csv
│   ├── seed222/{innov1-4_only,full_model}.csv
│   └── seed333/{innov1-4_only,full_model}.csv
├── ablation_results_v3/          # V3 results (clip=2.0 full_model)
│   ├── seed111_full_model.csv
│   ├── seed222_full_model.csv
│   └── seed333_full_model.csv
├── ablation_results_missing_rate/ # missing-rate sweep results (IN PROGRESS)
├── result/mvtec3d/both/
│   ├── 0.3/csv/Seed_{222,333}-results.csv   # seed111 MISSING at 0.3
│   ├── 0.5/csv/Seed_{111,222,333}-results.csv
│   ├── 0.7/csv/Seed_{111,222,333}-results.csv  # V3 full_model results
│   └── 0.9/csv/Seed_{111,222,333}-results.csv
├── run_v2_segment.sh             # V2 ablation segment runner (segments 1-4)
├── run_v3_fullmodel.sh           # V3 full_model-only runner (3 seeds)
├── run_missing_rate_sweep.sh     # missing-rate sweep (η=0.3,0.5,0.9 × 3 seeds)
├── launch_segment.sh             # systemd dispatcher
├── .current_segment              # controls which script systemd runs
│                                 # values: "1","2","3","4","v3","rate_sweep"
├── ablation_v2_segment{1-4}.log
├── ablation_v3_fullmodel.log
├── missing_rate_sweep.log
└── systemd_training.log
```

---

## Training Commands

**Check training status:**
```bash
systemctl --user status misdd-training.service
tail -10 ~/MISDD-MM/missing_rate_sweep.log  # or relevant log
ls -la ~/MISDD-MM/ablation_results_missing_rate/
```

**Start/restart any training:**
```bash
echo "rate_sweep" > ~/MISDD-MM/.current_segment  # or v3, 1, 2, 3, 4
systemctl --user start misdd-training.service
sleep 20
tail -15 ~/MISDD-MM/missing_rate_sweep.log
```

**Quick test (2 epochs, innov1 config, won't interfere with background if different seed):**
```bash
WANDB_MODE=offline python train_cls.py \
    --dataset mvtec3d --class_name bagel \
    --missing_type both --missing_rate 0.7 \
    --seed 999 --gpu-id 0 --Epoch 2
```

**Check GPU:**
```bash
/usr/lib/wsl/lib/nvidia-smi
```

**Completion check across all configs:**
```bash
python3 << 'EOF'
import os, csv
results_dir = os.path.expanduser("~/MISDD-MM/ablation_results_v2")
seeds = [111, 222, 333]
configs = ["innov1_only","innov2_only","innov3_only","innov4_only","full_model"]
print(f"{'Config':<20} {'Seed111':>10} {'Seed222':>10} {'Seed333':>10}")
print("-" * 55)
for cfg in configs:
    row = [f"{cfg:<20}"]
    for seed in seeds:
        path = f"{results_dir}/seed{seed}/{cfg}.csv"
        if not os.path.exists(path):
            row.append(f"{'MISSING':>10}")
        else:
            with open(path) as f:
                lines = list(csv.reader(f))
            vals = [float(l[1]) for l in lines[1:] if len(l) > 1]
            mean = sum(vals)/len(vals) if vals else 0
            row.append(f"{mean:>9.2f}%")
    print("".join(row))
EOF
```

---

## Current Status & Remaining Work

### ⚠️ V4 / PATH B PIVOT (2026-07-12) — READ THIS FIRST

A full pipeline audit (`PIPELINE_AUDIT_FINDINGS.md`) found that in V1–V3 the missing-aware
prompts received **zero gradient from the main loss** (`encode_image_missing` was
`@torch.no_grad()`), were **ignored in the cls test path**, and the deep compound prompts
(innovation 1) were injected into a dead tensor path. Plus: RGB↔depth pairing was shuffled
for ~99% of MVTec3D good samples (unsorted globs), and test images were fed in BGR.
Khalid chose **Path B: fix everything and re-run**. All fixes are applied and verified
(see the addendum in `PIPELINE_AUDIT_FINDINGS.md` for the full fix table + evidence).

**Consequences:**
- **V1/V2/V3 results and the partial missing-rate sweep are SUPERSEDED.** Do not mix them
  with V4 numbers. The missing-rate sweep was stopped mid-run (η=0.5 seed222).
- The old "gradient interference" narrative (warmup, clip 2.0, grad diagnostics) described
  auxiliary-loss dynamics only and does not carry over to V4.
- Training now backprops through the frozen encoder: `--batch-size` default is 32
  (~8-10 steps/epoch, ~1.25 s/step), evaluation happens **once at the final epoch**
  (no best-epoch test-set selection), checkpoints persist all learned weights.
- New: `ablation_models/model_baseline.py` (all innovations off) — the baseline is now
  actually runnable. `create_ablations.py` treats `model_full.py` as read-only source.
- Invariants to preserve: `run_v4_ablation.sh` has a `trap EXIT` restoring `model.py`;
  never regenerate variants while `model.py` might be a swapped ablation; run
  `/tmp/variant_forward_test.py` (structural test) before launching any campaign.

### IN PROGRESS
- **V4 campaign:** baseline + innov1-4_only + full_model × seeds 111/222/333, η=0.7,
  MVTec 3D-AD. Runner: `run_v4_ablation.sh <1|2|3>` (segment = seed). Dispatcher values
  for `.current_segment`: `v4_1`, `v4_2`, `v4_3`. Results → `ablation_results_v4/seed{s}/{config}.csv`.
  Logs → `ablation_v4_seed{s}.log`. ~2h per config-seed, ~12h per segment.
  Resume after interruption: `echo v4_N > .current_segment && systemctl --user start misdd-training.service`
  (per-config resume is automatic via `is_complete`).

### PENDING (priority order)
1. **Finish V4 segments 1-3** (start segment 2 after 1 completes, then 3)
2. **V4 missing-rate sweep** (η=0.3/0.5/0.9, full_model, corrected pipeline) — needs a
   new runner or adapt `run_missing_rate_sweep.sh` with `--batch-size 32` + V4 model
3. **Statistical validation on V4 results only** (`02_STATISTICAL_VALIDATION_PROMPT.md`)
4. **Eyescandies V4** — full re-run on corrected pipeline
5. **Efficiency analysis** — parameter count, FLOPs, inference latency per config
6. **Paper rewrite around corrected architecture** — the V1–V3 story becomes a
   "flaws found and fixed" methodology note; V4 is the primary table

---

## Remaining Shortcomings (Honest)

1. **Seed111 margin thin:** full_model beats best individual by only 0.29pp — within variance
2. **V3 mean below V1:** 76.71% vs 77.47% — performance/consistency tradeoff
3. **Cosine similarity diagnostic failed:** hooks didn't capture directional gradient conflict; only norm imbalance measured
4. **Innovation 4 deterministic:** no learned parameters — needs justification in paper vs learned gate
5. **Missing-rate sweep incomplete**
6. **Single dataset for V2/V3:** Eyescandies only has V1 results
7. **No efficiency numbers** despite "lightweight" claims

---

## Key Literature to Cite

- **PCGrad** (Yu et al., 2020) — gradient surgery for multi-task learning
- **CAGrad** (Liu et al., 2021) — conflict-aware gradient descent
- **Ortho-LoRA** (Yang et al., 2026) — orthogonal gradient projection for shared PEFT
- **Recon** (Shi et al., 2023) — conflict resolution from the root
- **GradientStabilizer** (Huang et al., 2025) — adaptive clipping
- **PTP** (Chen et al., 2023) — prompt tuning stability
- **SoftCPT** (Ding et al., 2022) — soft context sharing for VLMs
- **AdaCLIP** (Cao et al., 2024) — CLIP adaptation for zero-shot anomaly detection

---

## Paper Positioning

Frame as: **"robustness paper proving reliability-aware multimodal anomaly detection under missing and degraded sensors"** — not "four innovations on MISDD-MM."

Core claim: reliability-aware prompting preserves anomaly detection when modalities are absent or degraded, while staying lightweight enough for practical inspection.

Strongest result: +3.64pp I-AUROC over published baseline at η=0.7 (V1 primary).

---

## Git State

Latest commit: `5423b75` — "Add V3 full_model results with config-aware gradient clipping"
All V1/V2/V3 results, training scripts, and model files are committed to main.
GitHub auth: PAT for KsKarim7 (Windows Credential Manager — if 403 errors, clear cached credentials).

---

## Code Style Preferences

- Comments should look human-written and minimal — not over-commented
- No AI-generated looking comment blocks
- Prefer surgical patches over full rewrites
- Always test with `--Epoch 2` on bagel before committing major changes
- Never run heavy builds from `/mnt/c/` paths
- Always `conda activate ramsdd` first
