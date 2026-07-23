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
- Training now backprops through the frozen encoder: microbatches of 32 accumulate into
  **one optimizer step per epoch** (the V1-V3 full-batch budget), evaluation happens
  **once at the final epoch** (no best-epoch test-set selection), checkpoints persist all
  learned weights, and every eval prints `[component-diag]` textual-only / map-only AUROCs.
- **Epoch 25, not 50**: the learned text anchors overfit and collapse by ep50 under the
  final-epoch protocol (bagel pilot: textual AUROC 51→55→72→52 at ep 5/10/25/50). V1-V3's
  best-epoch selection masked this. Verified at ep25 on bagel seed111: baseline 87.40,
  full_model **92.56** I-AUROC (vs V1 84.25 / V3 85.07 best-epoch-inflated).
- Diagnostic knobs (defaults = intended behavior): `--visual_prompt_lr` (separate lr for
  missing-prompt learner) and `MISDD_XORI_INJECT=0` (disables deep-prompt injection).
- New: `ablation_models/model_baseline.py` (all innovations off) — the baseline is now
  actually runnable. `create_ablations.py` treats `model_full.py` as read-only source.
- Invariants to preserve: `run_v4_ablation.sh` has a `trap EXIT` restoring `model.py`;
  never regenerate variants while `model.py` might be a swapped ablation; run
  `/tmp/variant_forward_test.py` (structural test) before launching any campaign.

### V4 COMPLETE — final table (mean I-AUROC, harmonic scoring, η=0.7, MVTec 3D-AD)

| Config | S111 | S222 | S333 | Mean | map-only mean* |
|---|---:|---:|---:|---:|---:|
| baseline | 76.00 | 74.23 | 77.62 | 75.95 | 76.26 |
| innov1_only | 73.98 | 70.57 | 74.33 | 72.96 | 74.99 |
| innov2_only | 76.38 | 74.78 | 75.89 | 75.68 | 76.09 |
| innov3_only | 75.98 | 74.23 | 77.63 | 75.95 | 76.27 |
| innov4_only | 75.51 | 73.78 | 76.39 | 75.23 | 76.38 |
| full_model | 72.30 | 72.75 | 74.04 | 73.03 | 74.77 |

*map-only mined from `[component-diag]` lines in the campaign logs (all 180 runs).

### V5 ROOT-CAUSE DIAGNOSIS (2026-07-14, from V4 checkpoint probes + log mining)

Three verified defects explain why nothing beat baseline in V4:

1. **Global textual branch carries no signal.** Textual AUROC ≈ 50 in all 180 runs
   incl. baseline. Not an anchor-training failure: anchors separate as designed
   (cos 0.96 → 0.53); the global CLS feature simply doesn't move for small defects
   (init/trained/manual anchors all chance; normal-vs-anomaly score means differ
   ≤ 7e-4). Patch-level textual has weak signal (56.7 mean); not competitive.
2. **Harmonic fusion is dominated by the smaller-scale branch.** Textual scores
   (~0.05) < map scores (~0.15), so `1/(1/map+1/txt)` injects the chance-level
   textual noise into the fused score: −0.3pp (baseline) to −2.0pp (innov1) on
   average, per-class up to ±10pp (rope s333 −10.3, carrot s333 +4.9).
   **Fix: `--img_score_mode map`** (image score = map branch only).
3. **Trailing LayerNorm in CorrelatedPromptMLP + DynamicPromptGenerator defeats
   the near-zero init** — LN makes residual magnitude invariant to weight
   magnitude, so the "residuals" ran at 0.8–2× the main pathway (measured on
   trained checkpoints). This is innov1's map damage (−1.28 mean, peach s333
   −15.3pp, seed-unstable). Causal test: `MISDD_XORI_INJECT=0` recovered peach
   full map 80.91→84.33. **Fix: LayerScale `gamma` (init 1e-2, learnable) after
   the LN in both modules.** `[gamma-diag]` prints learned scales per run.

**Validation (seed333, map-only AUROC, 25ep):** baseline/peach replicates 86.68
bit-exact; innov1_only/peach 71.34→86.65; full_model/peach 80.91→**87.66**
(+0.98 over baseline, AUPRO 59.31→68.10); full_model/cookie 76.77→82.07 (parity
with baseline 82.14). Learned gammas: layers 1-4 shrink below init (~0.005),
layer 5 grows to ~0.12 — the optimizer wants a modest final-layer residual only.

### V5 SEED 111 COMPLETE (2026-07-15) — mean I-AUROC, map scoring, η=0.7

| baseline | innov1 | innov2 | innov3 | innov4 | innov2_3_4 | full_model |
|---:|---:|---:|---:|---:|---:|---:|
| 76.74 | 76.51 | 76.74 | 76.75 | 76.88 | 76.88 | 76.61 |

All 7 configs within ±0.23pp of baseline; per-class deltas mostly <0.5pp.
Fixes verified at campaign scale: innov1 damage gone, no instability.
Gamma profile replicates in every innov1-bearing config: layers 1-4 suppressed
below init (~0.005-0.01), layer 5 grows to ~0.12-0.14 (publishable measurement).

### CEILING RESULT — narrowed + re-verified on key-exact checkpoints (2026-07-15)

First probe was INVALID (loaded V4 ckpts lacking gamma keys into the gamma model —
strict=False silently tamed their residuals). Rerun on V5 seed-111 full_model
checkpoints with verified key match (210 learner keys, 14 gamma keys, 0 missing,
0 unexpected, every class):

- Four-quadrant map AUROC ({trained,init} test × {trained,init} gallery), 3 classes:
  all combos within 1.6pp; trained-vs-trained is ≤ init-vs-init on 2 of 3 classes
  (peach −1.56). Spearman(PP,00) ≥ 0.99. Prompt training moves features
  substantially (~35% of spread) but rank-preservingly, and the shift is
  defect-blind (normal vs anomalous shift norms within 6%, inconsistent sign).
- **Correct claim: the current all-normal training objective cannot improve the
  map branch** (best case +0.1pp, worst −1.6pp). NOT proven for anomaly-aware
  objectives (e.g. synthetic-anomaly contrastive prompt training) — untested,
  and the one principled route left for making prompts matter.
- V4's innov1 per-class map swings were the untamed LayerNorm-inflated residual —
  unintended, seed-unstable, now fixed.

### FUSION + AGGREGATION AUDIT (10 classes, V5 seed111 ckpts, 2026-07-15)

- img-only map 66.9, depth-only 57.6, **max fusion 76.7** — the max is doing the
  reliability work already; streams are anti-correlated (Pearson −0.46) and
  per-image complementary. Best fixed-α blend only 72.9.
- **SensorSentinel map-fusion reroute is DOA**: sentinel-blend 74.2, sentinel-max
  69.5 — both worse than plain max. Laplacian-variance/depth-density weights do
  not predict which modality carries the defect.
- Patch aggregation: max pooling is right (top5/20/50-mean and all-mean all worse).
- **3-NN mean distance beats 1-NN min: 77.63 vs 76.74 (+0.89pp mean)** — eval-only
  change, no training; per-class consistency still to be verified before adoption.

### MEASUREMENT SPINE — all five load-bearing claims re-verified (2026-07-15)

Every claim re-verified on exact artifacts (git worktrees of the generating code,
key-exact checkpoint loads). Verified evidence, citable in the paper:

1. **no_grad block (V1-V3):** on code @ `49f01a9` (decorator at model.py:585),
   main loss → **0/171** missing-prompt params with grad; L_gran alone → 158/171.
   The V1-V3 visual prompt system was trained *only* by the auxiliary loss.
2. **Dead deep injection (V1-V3):** deep prompts ×1000 → exactly 0.000000 change
   in every encoder output on old code (layer-0 control moves everything).
   New code: deep prompts reach CLS + block-7 features (block-2 barely, 3e-6 —
   innov1 affects the deeper of the two kNN feature sets).
3. **LayerNorm inflation (sharpened):** V4 ckpts into V4 code (key-exact,
   196 keys, 0 gamma, 0 missing). Three distinct quantities — keep denominators
   explicit: (A) growth across training (trained/init output norm, same batch):
   corr **262-615×**, dyn 45-52×; (B) trained residual vs its host pathway
   (same forward): corr 0.58-2.20×, dyn 20.1-20.6×; (C) init residual vs
   pathway: corr 0.007-0.008, dyn 0.38-0.45 (dyn init was never near-zero).
   Mechanism: **trailing LN concentrates output scale in its affine gain,
   decoupled from weight magnitude — near-zero init has no persistence.**
   Measured on peach+cookie seed333, batch of 8; ratios stable across the two
   classes within ~10%.
4. **Gamma profile:** read from all 10 V5 seed-111 ckpts: layer5 γ 0.112-0.150
   (every class), layers 1-4 below init, dyn γ 0.005-0.010. Layer-0 corr module
   is never exercised (γ exactly at init) — say so in the paper.
5. **Ceiling:** quadrant test now on 6 classes (incl. weak: cable_gland, tire,
   potato), all key-exact: PP−00 ∈ [−1.63, +0.50], mean −0.55 (trained ≤ init on
   4/6), ρ ≥ 0.989, shifts defect-blind. Claim stands as narrowed: *the current
   all-normal objective cannot improve the map branch.*

**Provenance findings:** commit `43d6615`'s `model.py` is a swapped ablation
variant (corr call commented out) — `model_full.py` is the true V4 model.
V5 changes (gamma, --img_score_mode) are uncommitted working-tree state — commit
before the next campaign. `/tmp` was wiped by a WSL reboot 2026-07-15: worktrees
(`/tmp/old_repo` @ 49f01a9, `/tmp/v4_repo` @ 43d6615) and all probe scripts are
disposable; recreate via `git worktree add`.

### GATE PLAN (user-approved 2026-07-15): hardened-spine → stability → synthetic-anomaly → multi-dataset
Khalid approved a four-gate plan for the strongest-paper path: (1) re-verify the
measurement spine (DONE — see above), (2) verify+adopt 3-NN then the 3-seed
stability table, (3) synthetic-anomaly contrastive prompt objective as a proper
experiment (pilot gated before campaign; a negative result strengthens the
ceiling finding), (4) Eyescandies + efficiency + missing-rate curve. Audit every
target before fixing; key-match on every checkpoint load; flag baseline-hugging.

### GATE 2 COMPLETE (2026-07-16) — 3-seed stability table, final protocol

| config | s111 | s222 | s333 | mean | vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 77.57 | 75.49 | 77.22 | 76.76 | — |
| innov1_only | 77.52 | 75.48 | 77.13 | 76.71 | −0.05 |
| innov2_only | 77.60 | 75.47 | 77.22 | 76.76 | +0.00 |
| innov3_only | 77.58 | 75.50 | 77.22 | 76.77 | +0.00 |
| innov4_only | 77.58 | 75.37 | 77.23 | 76.73 | −0.03 |
| innov2_3_4 | 77.57 | 75.41 | 77.17 | 76.72 | −0.05 |
| full_model | 77.48 | 75.45 | 77.09 | 76.67 | −0.09 |

All configs within ±0.09pp of baseline (3-seed mean); largest single
class-seed deviation −1.21 (full_model s111 cable_gland). Seed std ≈1.1
identical across configs (seed-level, not config-level, variance). The k=3
choice held on both held-out seeds. **Flat line = the ceiling result at
3-seed rigor; the innovations are null on the corrected pipeline.**
Campaign details preserved below.

### GATE 2 execution notes — 3-NN adopted, final-protocol campaign
- Per-class kNN comparison (V5 s111 ckpts, key-exact): 3-NN beats 1-NN on 8/10
  classes (mean 76.74→77.63, +0.89; worst regression −0.28 carrot; peach +2.87).
  5-NN mean 77.81 but larger regressions — **k=3 adopted, frozen before seeds
  222/333** (they act as held-out confirmation of the k choice).
- `--map_knn` arg added (train_cls.py); `_gallery_distance` in model_full.py;
  variants regenerated; structural test recreated + all 10 PASS; 2-epoch smoke
  test passed (bagel 85.64).
- **Campaign `v5k_all` running since 2026-07-15 16:39**: 7 configs × seeds
  111/222/333 (seed 111 re-run for protocol uniformity), Epoch 25,
  `--img_score_mode map --map_knn 3`, results → `ablation_results_v5_3nn/seed{s}/`,
  logs → `ablation_v5_3nn_seed{s}.log`, ~42h total, seeds chain automatically.
- `ablation_results_v5/` (1-NN protocol, seed 111 only) is retained as the
  k-selection record; do not mix with v5_3nn numbers.

### GATE 3 COMPLETE (2026-07-16) — pilot NEGATIVE, ceiling extended

Synthetic-anomaly contrastive pilot (peach/cookie/bagel/dowel × baseline/full_model,
seed 111, final protocol + `--syn_anomaly --syn_weight 0.2`): baseline mean Δ −0.10,
full_model mean Δ +0.09 vs v5k seed-111 references — **FAIL on the pre-registered
criterion** (mean > +0.5, no class < −1.0). `[syn-diag]` 55–70: the patch-textual
pathway only partially learned even the synthetic task; where it learned best
(cookie ~70) the map delta was +0.14/+0.28. Global textual AUROC on bagel rose to
~72 (vs ~56) — the objective reshapes the textual pathway but cannot reach the map
branch. Scope limit for the paper: corruptions are crude proxies (CutPaste+blobs);
claim is "this route, as implemented, doesn't break the ceiling."

### PROVENANCE RULING (2026-07-18, Khalid): tolerance ACCEPTED, V4 code tagged

- Reproduction recipe accepted at disclosed ≤2.4e-4 tolerance; peach/cookie
  side-by-side goes in the paper appendix **as reproduction tolerance, not
  bit-exact match**. 10-class inflation table = SUPPORTING evidence only
  (growth-vs-init is the same phenomenon as trained/pathway ratio, which the
  10-class γ profile corroborates from original artifacts; drop growth range
  if contested, argument unchanged).
- **Tagged commit `v4-campaign-code` (29251d7)** on branch
  `v4-campaign-reconstruction`: 43d6615 + accumulation loop + model.py restored
  from model_full.py. This is the citable hash for all V4 numbers.
- **Provenance flags — RESOLVED 2026-07-23**: the whole V5-era working tree is
  committed on `main` as `3fa1d5a` ("Complete empirical phase: Eyescandies 3-seed,
  V4 inflation table, integrity audit"), preceded by `55909df` (V5 final-protocol
  results). All 60 result CSVs (21 v5_3nn + 18 missing-rate + 21 eyescandies)
  tracked; working tree clean. V4 numbers cite tag `v4-campaign-code` (29251d7).
  Every paper number now maps to a hash. (Historical note: these results were
  *produced* by the then-uncommitted tree; behaviorally identical to `3fa1d5a`
  with default flags.)
- 8-class retraining under the tagged code runs at the Eyescandies seed-111→222
  boundary (pause service, ~1.6h retrain + probes, resume — no lost work).

**10-CLASS INFLATION TABLE COMPLETE (2026-07-20, tagged code, all key-exact):**
corr (innov1): A growth-vs-init 239–616× (median 373; peach 593/cookie 615 are
MID-PACK — n=2 damage-selection worry resolved), B trained/pathway 0.49–2.76
(median 0.92), C init/pathway 0.0070–0.0077 (dead flat all 10). dyn (innov2):
A 36–69×, B 18.6–24.5, C 0.35–0.54 (init never near-zero, all 10). Supporting
evidence only; growth range labeled "measured under reproduction tolerance";
load-bearing claim = B (trained/pathway), corroborated by γ profile from original
artifacts. Probe: `/tmp/inflation_table.py` (recreate from git history if wiped).

### V4 CHECKPOINT REPRODUCTION (2026-07-16) — protocol identified, tolerance ruling PENDING

v5k overwrote all V4-era checkpoints (shared paths — campaign ckpts are ephemeral!).
Reproduction on worktree `~/v4_repro` @ 43d6615: **43d6615 steps the optimizer per
microbatch; the campaign ran the post-commit uncommitted accumulation loop** (one
step/epoch). With accumulation ported, cookie+peach reproduce the Gate-1c receipts:
all claim-bearing residual norms within ±0.001, pathways within ±0.012 (max 2.4e-4
relative; trainer is bit-stable within a context — identical back-to-back runs —
but not across process contexts). By Khalid's bit-exact criterion this is a FAIL;
his ruling pending: accept recipe with disclosed ≤2.4e-4 tolerance (→ retrain 8
classes ~1.6h → 10-class inflation table) vs keep inflation measurements 2-class
(preserved receipts only).

### GATE 4: missing-rate curve + efficiency COMPLETE (2026-07-18); Eyescandies RUNNING

**Missing-rate curve (mean I-AUROC over 3 seeds, final protocol):**

| η | baseline | full_model |
|---:|---:|---:|
| 0.3 | 80.00 ± 0.52 | 80.04 ± 0.58 |
| 0.5 | 77.84 ± 1.11 | 77.71 ± 0.95 |
| 0.7 | 76.76 ± 1.11 | 76.67 ± 1.07 |
| 0.9 | 75.40 ± 1.49 | 75.44 ± 1.38 |

Graceful degradation (−4.6pp over 0.3→0.9); ceiling holds at every η (|Δ|≤0.13);
seed variance grows with η. Results in `ablation_results_v5_missing_rate/`.
Note: the machine was OFF Fri ~01:30 → Sat 13:43 mid-sweep; auto-resume re-ran
the interrupted config; all 18 runs clean.

**Efficiency (RTX 4090, batch 8, fp32):** trainable 7.29M = 3.45% of the frozen
211.6M backbone (innov1 1.24M, innov2 0.37M, innov3 1.72M, innov4 0 — deterministic;
compound projections 3.62M, text learners 20K). Latency 11.1 ms/img full cls path
(~90 img/s); prompt system 1.87 ms/img = 16.9% of total.

**Eyescandies campaign COMPLETE (2026-07-23 03:53)** — 21/21 clean, final protocol.
3-seed mean I-AUROC:

| config | s111 | s222 | s333 | mean | vs base |
|---|---:|---:|---:|---:|---:|
| baseline | 65.24 | 67.26 | 78.36 | 70.29 | — |
| innov1_only | 64.81 | 67.11 | 78.27 | 70.06 | −0.22 |
| innov2_only | 65.31 | 67.21 | 78.41 | 70.31 | +0.02 |
| innov3_only | 65.23 | 67.27 | 78.35 | 70.28 | −0.00 |
| innov4_only | 65.00 | 67.19 | 78.44 | 70.21 | −0.08 |
| innov2_3_4 | 65.07 | 67.11 | 78.48 | 70.22 | −0.07 |
| full_model | 65.16 | 67.01 | 78.45 | 70.21 | −0.08 |

Second dataset independently reproduces the ceiling: all configs within ±0.22pp
of baseline (3-seed mean); per-seed spread ≤0.50pp; huge seed variance (65/67/78)
is dataset difficulty, not config. The flat line holds on BOTH datasets, 3 seeds
each. GATE 4 COMPLETE. All V5-era data collection done.
3. **Gate 4**: Eyescandies on final protocol + efficiency analysis + missing-rate curve
4. **Paper narrative**: measurement spine (verified) + ceiling + gamma profile +
   fusion/aggregation audits are the empirical core; git commit needed (V5 changes
   uncommitted; 43d6615's model.py is an ablated variant — fix in next commit)
3. **V5 missing-rate sweep** (η=0.3/0.5/0.9, full_model, fixed pipeline) — needs a
   new runner or adapt `run_missing_rate_sweep.sh` with `--batch-size 32 --img_score_mode map`
4. **Statistical validation on V5 results only** (`02_STATISTICAL_VALIDATION_PROMPT.md`)
5. **Eyescandies V5** — full re-run on fixed pipeline
6. **Efficiency analysis** — parameter count, FLOPs, inference latency per config
7. **Paper rewrite** — V1–V4 become the "flaws found and fixed" methodology arc
   (V4: dead gradients/pairing/BGR; V5: LayerNorm-defeated init + fusion noise);
   V5 is the primary table

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
