# Reliability-Aware Multimodal Surface Defect Detection Under Uncertain Sensor Availability

Research codebase extending **MISDD-MM** for industrial surface defect detection when sensor
data is partially missing. RGB and depth images are processed by a frozen CLIP backbone with
prompt tuning; the system is evaluated under simulated sensor loss at rates of 30–90%.

**Author:** Aalavi Mahin Khan, BSc, BRAC University
**Supervisor:** F. H. Orodi
**Datasets:** MVTec 3D-AD (10 categories), Eyecandies (10 categories)

---

## Read this first: what this repository actually shows

This project began as "four prompt-side innovations improve multimodal defect detection." A
full pipeline audit found that the early positive results were **measurement artifacts**. After
rebuilding the pipeline correctly, the honest finding is:

> **Prompt-side learning cannot improve gallery-based anomaly detection under an all-normal
> training objective.** All seven configurations — baseline, each innovation alone, and
> combinations — converge to within ~0.1 percentage points of one another.

This is not an inconclusive result. The mechanism is measured and reproducible: the anomaly
score is a k-NN distance between test-image patch features and a gallery of normal patch
features. Prompt training shifts **both** sides of that comparison in the same direction
(measured shift cosine = 1.000, rank correlation ≥ 0.99), so the shift cancels in the distance.
Every innovation in this work acts on the prompt pathway, and is therefore structurally locked
out of the branch carrying the detection signal.

The repository is organised to make that claim auditable: every reported number traces to a
committed hash, and the diagnostic probes that establish the mechanism are included.

---

## Repository layout

```
.
├── MISDD_MM/                    Model implementation
│   ├── model_full.py            ← CANONICAL architecture (all 4 innovations)
│   ├── model.py                 ← SWAP TARGET, see warning below
│   ├── ad_prompts.py            Text prompt templates (20 classes)
│   ├── CLIPAD/                  Frozen CLIP backbone
│   │   └── transformer.py       Modified: in-place ops made autograd-safe
│   └── model_backup_*.py        Dev snapshots (unused, historical)
│
├── ablation_models/             9 standalone model variants
│   ├── model_baseline.py        All innovations disabled
│   ├── model_innov{1,2,3,4}_only.py
│   ├── model_innov{1_3,1_4,2_4,2_3_4}.py
│   └── *.py.bak                 Cruft, ignore
│
├── datasets/
│   ├── mvtec3d.py               MVTec 3D-AD loader
│   ├── eyescandies.py           Eyecandies loader
│   ├── dataset.py               Depth derivation from point cloud Z-channel
│   └── seeds_mvtec3d/           Fixed missing-modality masks per seed
│
├── utils/
│   ├── metrics.py               Scoring: k-NN gallery distance, score fusion
│   ├── syn_anomaly.py           CutPaste / Perlin corruptions (Gate-3 experiment)
│   ├── eval_utils.py            AUROC / AUPRO computation
│   └── csv_utils.py             Result writing
│
├── train_cls.py                 ← THE entry point; produced every reported number
├── create_ablations.py          Regenerates ablation_models/ from model_full.py
├── launch_segment.sh            systemd dispatcher (reads .current_segment)
│
├── run_v5_3nn.sh                ← Current main campaign
├── run_v5_missing_rate.sh       ← Robustness curve
├── run_v5_eyescandies.sh        ← Second dataset
├── run_gate3_pilot.sh           ← Synthetic-anomaly experiment
├── run_v4_repro.sh              V4 checkpoint reconstruction
├── run_*.sh (~25 others)        Superseded, kept for history
│
├── ablation_results*/           Results by generation, see table below
├── result/                      Raw per-run output CSVs
├── result_diag*/                Diagnostic probe outputs
│
├── CLAUDE.md                    Working context file (detailed internal notes)
├── PIPELINE_AUDIT_FINDINGS.md   Formal audit report
├── GATE3_PILOT_DESIGN.md        Pre-registered experiment design
├── RESULTS.md                   ⚠ STALE — V1–V3 era, superseded
│
├── Pointnet2_PyTorch/           Vendored dependency (point-cloud path is DEAD, see below)
├── static/, index.html          Project-page template, unused
└── test_cls.py, test_seg.py,
    train_seg.py                 ⚠ DEAD CODE — stale signatures, would crash
```

### ⚠ `model.py` is a swap target, not the architecture

`run_*.sh` scripts copy a file from `ablation_models/` over `MISDD_MM/model.py` before each
training run, and a `trap EXIT` restores it afterwards. At any moment `model.py` may be an
ablation variant. **Read `model_full.py` for the reference architecture.** Never commit
`model.py` while a campaign is running — verify first:

```bash
diff MISDD_MM/model.py MISDD_MM/model_full.py   # must be empty
```

### ⚠ The point-cloud path is dead

`Pointnet2_PyTorch/` is vendored and `MISDD_MM/model.py` still instantiates a
`PointTransformer`, but **no `encode_pc` method exists in any model variant**, and every
point-cloud line in the training and evaluation scripts is commented out. Depth is derived
from the point cloud's Z-channel and processed as a 2D image through frozen CLIP. The system
is **RGB + depth**, not three-modality.

---

## Results directories, by generation

Each directory is a distinct pipeline generation, kept rather than overwritten so the audit
trail survives. **Later supersedes earlier.**

| Directory | Generation | Status |
|---|---|---|
| `ablation_results/` | V1 — original, pre-audit | ❌ Artifacts |
| `ablation_results_v2/` | V2 — uniform gradient clipping | ❌ Artifacts |
| `ablation_results_v3/` | V3 — config-aware clipping | ❌ Artifacts |
| `ablation_results_v4/` | V4 — first architecturally correct run | ⚠ Superseded |
| `ablation_results_v5/` | V5 — LayerScale fix, 1-NN scoring | ⚠ k-selection record |
| `ablation_results_v5_3nn/` | **V5 final protocol** | ✅ **Current** |
| `ablation_results_v5_missing_rate/` | Robustness curve, η = 0.3 / 0.5 / 0.9 | ✅ Current |
| `ablation_results_missing_rate/` | Pre-V4 sweep, incomplete | ❌ Superseded |

CSV format: one row per object category, columns `class, I-AUROC, P-AUROC, AUPRO`.

---

## Headline results

### Final protocol (V5 + LayerScale + map scoring + 3-NN), η = 0.7, MVTec 3D-AD

Seed 111, mean I-AUROC over 10 categories:

| Config | I-AUROC | vs baseline |
|---|---:|---:|
| baseline | 77.57 | — |
| innov1_only | 77.52 | −0.05 |
| innov2_only | 77.60 | +0.03 |
| innov3_only | 77.58 | +0.01 |
| innov4_only | 77.58 | +0.01 |
| innov2_3_4 | 77.57 | 0.00 |
| full_model | 77.48 | −0.09 |

Total spread: **0.12 pp**, against seed-to-seed variation on a single config of ~1.5 pp. The
configurations are statistically indistinguishable. Seeds 222 and 333 are in
`ablation_results_v5_3nn/` (see repository for per-class values).

### Scoring improvement (evaluation-only, no retraining)

Replacing 1-NN minimum gallery distance with 3-NN mean distance: **+0.89 pp** mean,
8/10 categories improve. `k = 3` was selected on seed 111 alone and frozen before seeds
222/333 ran, so those seeds are held-out confirmation.

| Class | 1-NN | 3-NN | Δ |
|---|---:|---:|---:|
| bagel | 87.45 | 87.40 | −0.05 |
| cable_gland | 80.30 | 81.34 | +1.04 |
| carrot | 79.52 | 79.24 | −0.28 |
| cookie | 85.09 | 86.30 | +1.21 |
| dowel | 79.88 | 80.36 | +0.48 |
| foam | 70.88 | 71.75 | +0.87 |
| peach | 78.92 | 81.79 | +2.87 |
| potato | 54.55 | 56.23 | +1.68 |
| rope | 86.23 | 86.78 | +0.54 |
| tire | 64.55 | 65.10 | +0.55 |
| **mean** | **76.74** | **77.63** | **+0.89** |

### Robustness under sensor loss

Degradation from η = 0.3 to η = 0.9 (i.e. 90% of test samples missing one modality) is
**≈ 4.6 pp** I-AUROC. Full grid in `ablation_results_v5_missing_rate/`.

### Efficiency

≈ **3.5%** additional parameters over the frozen backbone; ≈ **90 images/second** inference.

### Published baseline reference (MISDD-MM, both-missing)

| η | I-AUROC | P-AUROC | AUPRO |
|---|---:|---:|---:|
| 0.3 | 77.71 | 95.00 | 84.03 |
| 0.5 | 76.95 | 93.28 | 79.79 |
| 0.7 | 73.83 | 93.05 | 77.44 |

Note the corrected baseline in this repository reaches **76.00** at η = 0.7 under a *stricter*
protocol (final-epoch evaluation, no best-epoch test-set selection) — above the published
73.83.

---

## The audit: what was wrong with V1–V3

Full detail in `PIPELINE_AUDIT_FINDINGS.md`. Summary of confirmed defects, each verified
against the exact code that produced the affected results:

| ID | Defect | Consequence |
|---|---|---|
| A3 | `@torch.no_grad()` on `encode_image_missing` | Main loss reached **0 of 171** prompt-learner parameters. Innovations 1, 2, 4 were frozen at random init throughout their own ablations. Only the auxiliary loss trained anything (158/171). |
| A4 | `cls` test branch called `encode_image()` without prompts | The headline I-AUROC never touched the missing-aware prompts at inference. |
| — | Innovation 1 injected into surgery path `x` | Attention never reads that tensor. A ×1000 perturbation changed the output by 0.000000 — a complete no-op. Now injects into the attended path `x_ori` (MaPLe-style). |
| A1 | MVTec3D `good`-split globs unsorted | RGB and depth paired from **different object instances** for ~99% of training samples. |
| A2 | Test images fed as BGR, gallery built as RGB | Channel mismatch at inference across all configs. |
| B2 | Best-epoch-on-test-set selection | Test-set model selection inflating every reported number. |
| — | Trailing LayerNorm in residual modules | Output magnitude invariant to weight magnitude, nullifying the documented near-zero initialisation. |

### The LayerNorm finding, stated precisely

The correlated-prompt residual begins at **0.7–0.8%** of its host pathway (as designed), but
the trailing LayerNorm decouples output scale from weight scale, so training inflates the
residual **260–615×** relative to its own initialisation — reaching **0.58–2.2×** the pathway
it was meant to gently perturb. The dynamic residual reaches **20×** its base prompt, having
started at 0.38–0.45× rather than near zero.

Growth figures measured on peach and cookie (damage-selected classes) under a reproduction
tolerance of ≤ 2.4e-4 relative; see provenance notes below.

**Fix:** LayerScale — a learnable per-layer scalar `γ` (init 1e-2) after the LayerNorm,
restoring the module's stated design rather than merely constraining it.

### What the optimiser does with the fix

Across all 10 categories, the learned γ suppresses layers 1–4 and grows only at layer 5:

```
per-layer mean γ: [0.0100, 0.0068, 0.0084, 0.0080, 0.0073, 0.1275]
layer-5 range across classes: 0.1124 – 0.1500  (10/10 classes)
```

The optimiser consistently wants a modest final-layer residual and actively suppresses the
deep chain. This reproduces on every class independently.

### The ceiling, measured

Four-quadrant probe: score {trained, init} prompts × {trained, init} gallery, on key-exact
checkpoints.

| Class | PP | 00 | P0 | 0P | Spearman(PP,00) |
|---|---:|---:|---:|---:|---:|
| bagel | 87.45 | 87.40 | 87.24 | 87.19 | 0.9966 |
| cookie | 85.09 | 85.68 | 84.85 | 86.17 | 0.9913 |
| peach | 78.92 | 80.48 | 78.01 | 79.97 | 0.9900 |
| cable_gland | 80.30 | 79.80 | 77.39 | 79.86 | 0.9903 |
| tire | 64.55 | 64.92 | 65.29 | 64.60 | 0.9941 |
| potato | 54.55 | 56.18 | 54.74 | 56.18 | 0.9888 |

All quadrants within ~1.5 pp; rank correlation ≥ 0.99. Prompt training moves mid-features by
~35% of within-class spread, but the movement is **rank-preserving and defect-blind** (normal
and anomalous shift norms within 6%, shift cosine = 1.000).

**Scope of the claim:** the *all-normal training objective* cannot improve the map branch
(best observed +0.1 pp, worst −1.6 pp). This does **not** claim that no prompt objective can —
an anomaly-aware objective was tested separately (below).

### Gate 3: the anomaly-aware objective

Because the ceiling result is specific to all-normal training, a synthetic-anomaly contrastive
prompt objective was designed and tested — CutPaste patch transplants and smooth-noise blob
corruptions with geometry-consistent depth offsets, driving a patch-level contrastive term.

A success criterion was **pre-registered before the experiment ran**: > +0.5 pp mean over the
campaign's own seed-111 reference, with no class regressing more than 1 pp. The pilot
(4 classes, seed 111) did not meet it. Results in `result_diag_gate3_*/`, design in
`GATE3_PILOT_DESIGN.md`.

---

## Provenance

| Ref | Hash | Covers |
|---|---|---|
| `main` | `55909df` | V5 final protocol results, missing-rate grid, Gate-3 pilot |
| tag `v4-campaign-code` | `29251d7` | Exact code that produced the V4 results |
| branch `v4-campaign-reconstruction` | `29251d7` | Same |
| — | `43d6615` | V4 architectural rebuild (see caveat) |
| — | `49f01a9` | Pre-audit code, used for V1–V3 gradient-census verification |

**Two disclosures a reviewer should have:**

1. **`43d6615` alone does not reproduce the V4 results.** That commit steps the optimizer per
   microbatch; the V4 campaign ran a gradient-accumulation loop (one step per epoch) that was
   uncommitted at the time. The tag `v4-campaign-code` reconstructs the exact campaign code.
   Reconstruction was validated against surviving probe receipts to ≤ 2.4e-4 relative
   deviation — not bit-exact, because the trainer is bit-stable within a process context but
   not across them (the campaign ran under systemd). All cited ratios are unaffected at three
   significant figures.

2. **The V5 3-NN campaign spans two tree states.** The synthetic-anomaly module was added
   mid-campaign. It is flag-gated and default-off, leaving the campaign's code path unchanged,
   but the hash differs between early and late runs.

**Checkpoints (`.pt`) are not tracked** — they are large binaries and are gitignored. All
reported numbers live in the CSVs. Reproducing a number means retraining from the pinned code.

---

## Environment

```
OS          Windows 11 + WSL2 Ubuntu 22.04
GPU         NVIDIA RTX 4090 (24 GB)
CUDA        11.8
Python      3.11 (conda env: ramsdd)
PyTorch     2.2.0+cu118
Backbone    CLIP ViT-B-16-plus-240 (frozen)
```

**Hard constraints:**
- `numpy < 2` — do not upgrade; do not rebuild `pointnet2_ops`
- `WANDB_MODE=offline`
- Never run heavy builds from `/mnt/c/` paths
- **Signing out of Windows kills WSL2 entirely**, regardless of systemd linger. Lock the
  screen (Win+L) instead. Long campaigns will die otherwise.

---

## Running experiments

Training runs under a systemd user service so it survives terminal disconnection.

```bash
conda activate ramsdd

# Launch a campaign
echo "v5_3nn" > ~/MISDD-MM/.current_segment
systemctl --user start misdd-training.service

# Monitor
systemctl --user status misdd-training.service
tail -f ~/MISDD-MM/systemd_training.log
```

Single run, directly:

```bash
WANDB_MODE=offline python train_cls.py \
    --dataset mvtec3d --class_name bagel \
    --missing_type both --missing_rate 0.7 \
    --seed 111 --gpu-id 0 \
    --Epoch 25 --batch-size 32 --max_norm 1.0 \
    --img_score_mode map --map_knn 3
```

### Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--missing_type` | — | `both` = either modality may be missing (never both on one sample) |
| `--missing_rate` | — | η; total fraction of samples with a missing modality |
| `--img_score_mode` | `harmonic` | `map` uses the gallery branch only; `harmonic` fuses with the textual branch |
| `--map_knn` | `1` | Gallery neighbours; **3** is the adopted setting |
| `--Epoch` | 50 | **25** is the protocol setting — text anchors overfit by 50 |
| `--syn_anomaly` | off | Enables the Gate-3 synthetic-anomaly objective |

### Protocol notes

- **Batch size 32 with gradient accumulation → exactly one optimizer step per epoch.**
  This reproduces the original full-batch budget (50 steps per class) while fitting in memory.
- **Evaluation runs once, at the final epoch.** No best-epoch selection.
- `--missing_type both` means *either* modality may be missing on a given sample, never both
  simultaneously. Total missing fraction is exactly η.
- Training runs with `model.eval()` throughout — intentional for frozen-CLIP prompt tuning.

---

## Known dead code and cruft

Kept for history, but not part of any result:

- `test_cls.py`, `test_seg.py`, `train_seg.py` — stale signatures, would crash on first batch
- `params.py` — imports a nonexistent `PromptAD` package
- `Pointnet2_PyTorch/`, `utils/pointnet2_utils.py` — point-cloud path is disabled everywhere
- `static/`, `index.html` — project-page template, unused
- `MISDD_MM/model_backup_*.py`, `ablation_models/*.py.bak` — development snapshots
- `RESULTS.md` — V1–V3 era numbers, superseded by this file
- Most `run_*.sh` scripts — superseded by the `run_v5_*` family
- `utils/__pycache__/` — currently tracked in error; should be untracked

---

## Citation context

This work extends:

> Resilient Multimodal Industrial Surface Defect Detection With Uncertain Sensors Availability
> (MISDD-MM)

Prompt architecture draws on **MaPLe** (multi-modal prompt learning) and **Deep Correlated
Prompting**. Synthetic anomaly generation follows **CutPaste** and **DRAEM** conventions.
LayerScale follows **CaiT** (Touvron et al.).
